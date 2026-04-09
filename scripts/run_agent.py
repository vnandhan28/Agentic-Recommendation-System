#!/usr/bin/env python3
"""
CLI script: Run the recommendation agent interactively (no Streamlit).

Useful for testing the agent loop from the terminal.

Usage:
    python scripts/run_agent.py --user user_001 --query "I need wireless headphones under $80"
    python scripts/run_agent.py --list-users
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rs.agent.loop import RecommendationAgent, StepType
from agentic_rs.config import config
from agentic_rs.vectorstore.build import VectorStore

app = typer.Typer(help="Run the recommendation agent from the terminal.")
console = Console()

STEP_STYLES = {
    StepType.THINKING:    ("🧠", "bold magenta"),
    StepType.TOOL_CALL:   ("🔧", "bold yellow"),
    StepType.TOOL_RESULT: ("📦", "bold green"),
    StepType.FINAL:       ("✅", "bold blue"),
    StepType.ERROR:       ("❌", "bold red"),
}


@app.command()
def run(
    user: str = typer.Option(..., "--user", "-u", help="User ID, e.g. 'user_001'"),
    query: str = typer.Option(..., "--query", "-q", help="Natural language request"),
    chroma_path: str = typer.Option(None, "--chroma", help="Override ChromaDB path"),
) -> None:
    """Run the agent and print the reasoning trace + recommendations."""
    store = VectorStore(persist_path=config.CHROMA_PATH if not chroma_path else __import__("pathlib").Path(chroma_path))
    agent = RecommendationAgent(store)

    console.print(Panel(
        f"[bold]User:[/] {user}\n[bold]Query:[/] {query}",
        title="🤖 Agentic Recommendation System",
        border_style="cyan",
    ))
    console.print()

    final_step = None
    for step in agent.run(user_id=user, query=query):
        icon, style = STEP_STYLES.get(step.step_type, ("•", "white"))
        console.print(
            f"  {icon} [[{style}]{step.step_type.value.upper()}[/{style}]] "
            f"(iter {step.iteration}) {step.content}"
        )
        if step.step_type == StepType.FINAL:
            final_step = step

    # Print final recommendations
    if final_step and final_step.recommendations:
        console.print()
        table = Table(title="🎯 Recommendations", border_style="blue", show_lines=True)
        table.add_column("Rank", style="bold", width=6)
        table.add_column("Title", min_width=30)
        table.add_column("Price", width=10)
        table.add_column("Rating", width=8)
        table.add_column("Why?", min_width=40)

        for rec in final_step.recommendations:
            table.add_row(
                f"#{rec.rank}",
                rec.title,
                f"${rec.price_usd:.2f}",
                f"⭐ {rec.avg_rating}",
                rec.explanation,
            )
        console.print(table)
    else:
        console.print("[red]No recommendations produced.[/]")


@app.command(name="list-users")
def list_users(
    chroma_path: str = typer.Option(None, "--chroma", help="Override ChromaDB path"),
) -> None:
    """List all available user IDs and names."""
    store = VectorStore(
        persist_path=config.CHROMA_PATH if not chroma_path
        else __import__("pathlib").Path(chroma_path)
    )
    users = store.list_users()
    if not users:
        console.print("[yellow]No users found. Run generate_data.py first.[/]")
        return

    table = Table(title="Available Users", border_style="cyan")
    table.add_column("User ID", style="bold")
    table.add_column("Name")
    for u in users:
        table.add_row(u["user_id"], u["name"])
    console.print(table)


if __name__ == "__main__":
    app()
