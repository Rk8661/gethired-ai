"""Single entry point for all LLM calls.

The project talks to LLMs through Fireworks AI's OpenAI-compatible endpoint
(not the OpenAI API directly). Agents must route every model call through here
rather than instantiating ``OpenAI(...)`` ad hoc, so the provider/model/JSON
handling lives in one place.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from backend.config import BASE_URL, MODEL

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Cached Fireworks client (OpenAI-compatible)."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FIREWORKS_API_KEY is not set. Add it to .env before using llm mode."
        )
    # Bound worst-case latency: a hung Fireworks call should fail with a clear
    # error rather than leave the UI waiting indefinitely.
    return OpenAI(base_url=BASE_URL, api_key=api_key, timeout=90.0)


def chat(messages: list[dict], reasoning_effort: str | None = None) -> str:
    """Plain text completion."""
    kwargs: dict = {"model": MODEL, "messages": messages}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _extract_json(text: str) -> str:
    """Strip ``` fences / surrounding prose and return the JSON substring."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    # Fall back to the outermost {...} span.
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t.strip()


def complete_json(system: str, user: str, reasoning_effort: str | None = "low") -> dict:
    """Structured-output entry point for llm-mode agents.

    Instructs JSON-only output, tolerates code fences, and retries once with a
    stricter nudge if the first response does not parse. Defaults to "low"
    reasoning effort: these are short extraction/scoring tasks, not deep
    reasoning, and "low" measured consistently faster with no loss in JSON
    validity during testing.
    """
    messages = [
        {"role": "system", "content": system + "\n\nRespond with ONLY valid JSON. No prose, no markdown fences."},
        {"role": "user", "content": user},
    ]
    raw = chat(messages, reasoning_effort=reasoning_effort)
    try:
        return json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."})
        raw = chat(messages, reasoning_effort=reasoning_effort)
        return json.loads(_extract_json(raw))


# --- Coercion helpers for the LLM's freeform JSON output ---------------------
# The model doesn't always follow the requested schema exactly (e.g. returns a
# list where a string was asked for). These sit at the external-API boundary so
# downstream pydantic models always get well-typed input instead of raising on
# a malformed field.
def safe_str(value, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if value is None:
        return default
    return str(value)


def safe_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def safe_float(value, default: float = 0.5, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def safe_int(value, default: int = 5, lo: int = 0, hi: int = 10) -> int:
    try:
        i = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, i))


__all__ = [
    "get_client",
    "chat",
    "complete_json",
    "safe_str",
    "safe_str_list",
    "safe_float",
    "safe_int",
]
