"""
Step 3b — Agent loop.

Iterates: call LLM → if tool call, execute → feed result back → repeat
until the LLM produces a final JSON answer with recommendations.

Heavy imports (openai / anthropic) are deferred to method bodies so
this module can be imported in test environments without those packages.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional

from agentic_rs.agent.tools import TOOL_SCHEMAS, ToolExecutor
from agentic_rs.config import config
from agentic_rs.models import AgentResponse, Recommendation


class StepType(str, Enum):
    THINKING    = "thinking"
    TOOL_CALL   = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL       = "final"
    ERROR       = "error"


@dataclass
class AgentStep:
    step_type: StepType
    iteration: int
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None
    recommendations: Optional[List[Any]] = None
    raw_message: Optional[dict] = None


SYSTEM_PROMPT = """You are an expert product recommendation agent.
Your job is to produce {num_recs} personalized product recommendations for a specific user.

You have four tools:
- fetch_user_history   — ALWAYS call this first to understand the user
- search_catalog       — semantic search for items matching a description
- filter_by_attributes — filter by price, rating, category, brand
- get_item_details     — get full details for a specific item_id

## Process
1. Call fetch_user_history to learn about the user
2. Run 1–3 search_catalog or filter_by_attributes calls based on their preferences
3. Call get_item_details on your top candidates
4. Return your final answer as JSON only (see format below)

## Final answer format — respond with ONLY this JSON, no markdown:
{{
  "recommendations": [
    {{
      "rank": 1,
      "item_id": "...",
      "title": "...",
      "price_usd": 0.0,
      "avg_rating": 0.0,
      "explanation": "2-3 sentences why this suits this specific user"
    }}
  ],
  "reasoning_summary": "1-2 sentences on your overall strategy"
}}

Return exactly {num_recs} recommendations, ranked best-first.
Do NOT include items the user has already purchased.
"""


class RecommendationAgent:
    """Tool-calling LLM agent that produces personalized recommendations."""

    def __init__(self, store: Any):
        self.store = store
        self.executor = ToolExecutor(store)

    def run(
        self,
        user_id: str,
        query: str,
    ) -> Generator[AgentStep, None, Optional[AgentResponse]]:
        """
        Run the agent loop, yielding AgentStep events for live UI rendering.
        Returns AgentResponse as the generator's return value (StopIteration.value).
        """
        messages: List[dict] = [
            {"role": "user", "content": f"User ID: {user_id}\nRequest: {query}"},
        ]

        for iteration in range(1, config.MAX_AGENT_ITERATIONS + 1):
            response_message = self._call_llm(messages)
            tool_calls = response_message.get("tool_calls") or []
            content = response_message.get("content") or ""

            if tool_calls:
                yield AgentStep(
                    step_type=StepType.THINKING,
                    iteration=iteration,
                    content=(
                        f"Decided to call {len(tool_calls)} tool(s): "
                        + ", ".join(tc["function"]["name"] for tc in tool_calls)
                    ),
                    raw_message=response_message,
                )

                messages.append({"role": "assistant", **response_message})

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    yield AgentStep(
                        step_type=StepType.TOOL_CALL,
                        iteration=iteration,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        content=f"Calling `{tool_name}` with: {json.dumps(tool_args)}",
                    )

                    result = self.executor.execute(tool_name, tool_args)

                    yield AgentStep(
                        step_type=StepType.TOOL_RESULT,
                        iteration=iteration,
                        tool_name=tool_name,
                        tool_result=result,
                        content=f"`{tool_name}` → {self._summarise(result)}",
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": json.dumps(result),
                    })
                continue

            if content:
                try:
                    parsed = self._parse_final(content)
                    user_meta = self.store.get_user_by_id(user_id) or {}
                    response = AgentResponse(
                        user_id=user_id,
                        user_name=user_meta.get("name", user_id),
                        query=query,
                        recommendations=parsed["recommendations"],
                        reasoning_summary=parsed.get("reasoning_summary", ""),
                    )

                    yield AgentStep(
                        step_type=StepType.FINAL,
                        iteration=iteration,
                        recommendations=parsed["recommendations"],
                        content="Final recommendations produced.",
                    )
                    return response

                except Exception as exc:
                    yield AgentStep(
                        step_type=StepType.ERROR,
                        iteration=iteration,
                        content=f"Failed to parse final answer: {exc}\nRaw:\n{content[:300]}",
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your response could not be parsed as JSON. Error: {exc}\n"
                            "Please respond with ONLY the JSON object as specified."
                        ),
                    })
                    continue

        yield AgentStep(
            step_type=StepType.ERROR,
            iteration=config.MAX_AGENT_ITERATIONS,
            content="Agent reached maximum iterations without producing a final answer.",
        )

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[dict]) -> dict:
        system = SYSTEM_PROMPT.format(num_recs=config.NUM_RECOMMENDATIONS)
        if config.LLM_PROVIDER == "anthropic":
            return self._call_anthropic(system, messages)
        if config.LLM_PROVIDER == "huggingface":
            return self._call_huggingface(system, messages)
        if config.LLM_PROVIDER == "groq":
            return self._call_groq(system, messages)
        return self._call_openai(system, messages)

    def _call_openai(self, system: str, messages: List[dict]) -> dict:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        full = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=full,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=2000,
        )
        msg = resp.choices[0].message
        result: dict = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return result

    def _call_huggingface(self, system: str, messages: List[dict]) -> dict:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key, base_url="https://router.huggingface.co/v1/")
        full = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=full,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=2000,
        )
        msg = resp.choices[0].message
        result: dict = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return result

    def _call_groq(self, system: str, messages: List[dict]) -> dict:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        full = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=full,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=1500,
        )
        msg = resp.choices[0].message
        result: dict = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return result

    def _call_anthropic(self, system: str, messages: List[dict]) -> dict:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in TOOL_SCHEMAS
        ]
        converted = self._to_anthropic_messages(messages)
        resp = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=2000,
            system=system,
            tools=anthropic_tools,
            messages=converted,
        )
        result: dict = {"content": ""}
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                })
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    @staticmethod
    def _to_anthropic_messages(messages: List[dict]) -> List[dict]:
        out = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                out.append({"role": "assistant", "content": content})
            elif role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_final(text: str) -> dict:
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break
        data = json.loads(text)
        recs_raw = data.get("recommendations", [])
        recs = [Recommendation.model_validate(r) for r in recs_raw]
        return {"recommendations": recs, "reasoning_summary": data.get("reasoning_summary", "")}

    @staticmethod
    def _summarise(result: dict) -> str:
        if "error" in result:
            return f"ERROR: {result['error']}"
        if "items" in result:
            return f"{result.get('count', len(result['items']))} items"
        if "user" in result:
            u = result["user"]
            return f"user '{u.get('name')}' ({u.get('price_sensitivity')})"
        if "item" in result:
            return f"item '{result['item'].get('title')}'"
        return str(result)[:100]


def _make_response(user_id, user_name, query, recommendations, reasoning_summary):
    """Construct AgentResponse without relying on Pydantic."""
    r = AgentResponse()
    r.user_id = user_id
    r.user_name = user_name
    r.query = query
    r.recommendations = recommendations
    r.reasoning_summary = reasoning_summary
    return r
