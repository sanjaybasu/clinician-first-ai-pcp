"""Thin, vendor-neutral LLM clients for the evaluation arms.

The deterministic engine never imports this module. Every call records vendor,
exact model id and token usage, so a run's cost and its SOUP versions are known.
PHI may go only to vendors under a Waymark BAA; ``phi_allowed`` is set by the
caller from the BAA register, not inferred here.
"""

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


_capture = threading.local()


@contextmanager
def capture_usage():
    """Collect the tokens spent by LLM calls made on this thread inside the block, keyed by vendor:model."""
    _capture.log = {}
    try:
        yield _capture.log
    finally:
        _capture.log = None


def _record(llm, input_tokens: int, output_tokens: int) -> None:
    log = getattr(_capture, "log", None)
    if log is not None:
        u = log.setdefault(f"{llm.vendor}:{llm.model}", {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        u["calls"] += 1
        u["input_tokens"] += input_tokens or 0
        u["output_tokens"] += output_tokens or 0


class LLM(Protocol):
    vendor: str
    model: str
    phi_allowed: bool
    usage: Usage

    def complete(self, system: str, messages: list[dict[str, str]], *, max_tokens: int = 800, temperature: float = 0.0) -> str: ...


def parse_json(text: str) -> dict:
    """First JSON object in a model reply; raises ValueError if none."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2**i)
    raise RuntimeError("unreachable")


@dataclass
class AnthropicLLM:
    """Opus and Sonnet 5.x emit a thinking block before text; headroom keeps the text from being truncated."""

    model: str
    thinking_headroom: int = 2048
    phi_allowed: bool = False
    vendor: str = "anthropic"
    usage: Usage = field(default_factory=Usage)
    # None sends no thinking parameter (the model default, as in v3). {"type": "between_tools"} turns thinking off on
    # Sonnet 5.5 (v4 structured screen); {"type": "disabled"} is rejected there.
    thinking: dict | None = None

    def __post_init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(timeout=600.0, max_retries=4)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        msgs = [m for m in messages if m["role"] in ("user", "assistant")] or [{"role": "user", "content": "(start)"}]
        if msgs[0]["role"] != "user":
            msgs = [{"role": "user", "content": "(conversation continues)"}, *msgs]
        extra = {"thinking": self.thinking} if self.thinking is not None else {}
        resp = _retry(
            lambda: self._client.messages.create(
                model=self.model, system=system, messages=msgs, max_tokens=max_tokens + self.thinking_headroom, **extra
            )
        )
        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens
        _record(self, resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


@dataclass
class GeminiLLM:
    """Thinking models count thinking tokens against the output cap, as in Compass (EN-3854)."""

    model: str
    thinking_level: str = "LOW"  # matches Compass settings.gemini_thinking_level
    thinking_headroom: int = 2048
    phi_allowed: bool = False
    vendor: str = "google"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                                    http_options=genai_types.HttpOptions(timeout=600_000))

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        from google.genai import types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens + self.thinking_headroom,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )
        resp = _retry(lambda: self._client.models.generate_content(model=self.model, contents=contents, config=cfg))
        self.usage.calls += 1
        um = resp.usage_metadata
        self.usage.input_tokens += (um.prompt_token_count or 0) if um else 0
        self.usage.output_tokens += (um.candidates_token_count or 0) if um else 0
        # thinking tokens are billed as output, so the per-call record includes them
        _record(self, (um.prompt_token_count or 0) if um else 0,
                ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0)
        return resp.text or ""


@dataclass
class OpenAILLM:
    model: str
    phi_allowed: bool = False
    vendor: str = "openai"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        import openai

        self._client = openai.OpenAI(timeout=600.0, max_retries=4)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        msgs = [{"role": "system", "content": system}, *[m for m in messages if m["role"] in ("user", "assistant")]]
        resp = _retry(
            lambda: self._client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=max_tokens, temperature=temperature
            )
        )
        self.usage.calls += 1
        if resp.usage:
            self.usage.input_tokens += resp.usage.prompt_tokens
            self.usage.output_tokens += resp.usage.completion_tokens
        _record(self, resp.usage.prompt_tokens if resp.usage else 0, resp.usage.completion_tokens if resp.usage else 0)
        return resp.choices[0].message.content or ""


@dataclass
class KevJudge:
    """Local System-1 typed judgments (jaredpalmer/kev, Jev-compatible API). Runs on this machine, so PHI stays local."""

    url: str = "http://127.0.0.1:8009/v1/systemone"
    vendor: str = "kev-local"
    model: str = "kev-4b@v1.0"
    phi_allowed: bool = True
    usage: Usage = field(default_factory=Usage)

    def choice(self, state: str, instructions: str, options: dict[str, str]) -> dict[str, float]:
        import urllib.request

        body = json.dumps({"model": "jev-latest", "state": state[-30000:],
                           "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": options}}})
        req = urllib.request.Request(self.url, data=body.encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            out = json.loads(fh.read())
        self.usage.calls += 1
        self.usage.input_tokens += out.get("usage", {}).get("input_tokens", 0)
        _record(self, out.get("usage", {}).get("input_tokens", 0), 0)
        return out["answers"]["q"]["probabilities"]
