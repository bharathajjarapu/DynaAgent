"""Minimal client for any OpenAI-compatible Responses API. Stdlib only."""

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


def load(path):
    """Load KEY=value lines from a .env file without overriding the real environment."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
KEY = os.getenv("LLM_KEY") or os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
SMART = os.getenv("LLM_SMART_MODEL", MODEL)
RETRY = {408, 409, 429, 500, 502, 503, 504}
CLIP = 6000  # chars of tool output fed back to the model


class LLMError(Exception):
    """Model or network failure."""


class BudgetExceeded(Exception):
    """A budget limit was hit."""


@dataclass
class Budget:
    """Per-query limits B = (cost, time, agents, calls) plus tokens, checked before every call."""

    calls: int = 60
    tokens: int = 500_000
    seconds: float = 300
    cost: float = 0.5  # USD, enforced only when the provider reports usage.cost
    agents: int = 8  # max tasks per query
    used_calls: int = 0
    used_tokens: int = 0
    used_cost: float = 0.0
    start: float = field(default_factory=time.monotonic)

    def elapsed(self):
        """Seconds since the query started."""
        return time.monotonic() - self.start

    def check(self):
        """Raise if any limit is spent."""
        for used, limit, what in (
            (self.used_calls, self.calls, "calls"),
            (self.used_tokens, self.tokens, "tokens"),
            (self.elapsed(), self.seconds, "seconds"),
            (self.used_cost, self.cost, "cost"),
        ):
            if used >= limit:
                raise BudgetExceeded(f"{what} limit {limit} reached")

    def add(self, usage):
        """Record one call's usage."""
        usage = usage or {}
        self.used_calls += 1
        self.used_tokens += usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        self.used_cost += usage.get("cost") or 0

    def left(self):
        """Seconds remaining; raises once time is spent, so retries stop too."""
        left = self.seconds - self.elapsed()
        if left <= 0:
            raise BudgetExceeded(f"seconds limit {self.seconds} reached")
        return left

    def summary(self):
        """One-line usage report."""
        cost = f" · ${self.used_cost:.4f}" if self.used_cost else ""
        return f"{self.elapsed():.1f}s · {self.used_calls} calls · {self.used_tokens:,} tokens{cost}"


def post(body, budget):
    """POST to /responses with retries and backoff, capped by the time budget."""
    request = urllib.request.Request(
        f"{BASE}/responses",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=min(120, budget.left())) as response:
                data = json.load(response)
            if data.get("error"):
                raise LLMError(str(data["error"])[:300])
            return data
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            if error.code not in RETRY or attempt == 3:
                raise LLMError(f"HTTP {error.code}: {detail}") from None
            try:
                wait = float(error.headers.get("retry-after") or 0)
            except ValueError:
                wait = 0
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise LLMError(f"Network error: {error}") from None
            wait = 0
        time.sleep(min(wait or 2**attempt + random.random(), 20, budget.left()))


def reply(output):
    """Join the assistant message text from response output items."""
    return "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def parse(text):
    """Parse JSON, falling back to the outermost {...} span."""
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}") + 1
        try:
            return json.loads(text[start:end])
        except ValueError:
            raise LLMError(f"Expected JSON, got: {text[:200]}") from None


async def invoke(tool, call, on_tool):
    """Run one function call in a thread and wrap its output for the model."""
    try:
        args = json.loads(call.get("arguments") or "{}")
        result = await asyncio.to_thread(tool.run, **args) if tool else f"Error: unknown tool {call['name']}"
    except Exception as error:
        args, result = call.get("arguments"), f"Error: {error}"
    result = str(result)[:CLIP]
    on_tool(call["name"], json.dumps(args, indent=2), result)
    return {"type": "function_call_output", "call_id": call["call_id"], "output": result}


async def respond(prompt, *, budget, model=None, system=None, tools=(), schema=None, on_tool=None, steps=6):
    """One agent turn: call the model and run its tool calls, at most `steps` times; the last step forbids tools."""
    on_tool = on_tool or (lambda name, args, result: None)
    lookup = {tool.name: tool for tool in tools}
    items = [{"role": "user", "content": prompt}]
    for step in range(steps):
        budget.check()
        body = {"model": model or MODEL, "input": items}
        if system:
            body["instructions"] = system
        if tools:
            body["tools"] = [tool.spec for tool in tools]
            if step == steps - 1:
                body["tool_choice"] = "none"
        if schema:
            body["text"] = {"format": {"type": "json_schema", "name": "out", "strict": True, "schema": schema}}
        data = await asyncio.to_thread(post, body, budget)
        budget.add(data.get("usage"))
        output = data.get("output") or []
        for item in output:
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                on_tool("search", "\n".join(action.get("queries") or [action.get("query") or ""]), "")
        calls = [item for item in output if item.get("type") == "function_call"]
        if not calls or step == steps - 1:
            text = reply(output)
            return parse(text) if schema else text
        items += [{key: value for key, value in item.items() if key != "status"} for item in output]  # some providers reject echoed status
        items += await asyncio.gather(*(invoke(lookup.get(call["name"]), call, on_tool) for call in calls))
