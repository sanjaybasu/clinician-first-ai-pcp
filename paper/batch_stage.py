"""Staged batch execution for multi-call pipelines, at the vendors' batch prices.

A pipeline (one COMPASS reply, or one rubric grade) is re-executed from the start on every pass. Each model call is
keyed by (slot, request content, occurrence of that content within the slot) and answered from the store when the key
is there. A missing call is queued for the next batch and answered with a placeholder or by stopping the pipeline:
  tier 0 and 1 calls (router; hazard screen and extractors) have inputs that depend only on the conversation and on
  tier 0 results, so they are queued together and the pipeline continues on a placeholder;
  a call of a higher tier than one already missing in the slot is not queued, because its input may be a placeholder;
  tier 3 calls (wording, check, revision) depend on everything before them, so the first miss stops the pipeline.
Passes alternate with batch submissions until nothing is missing. Requests are built exactly as the realtime clients
in compass_pcp/arms/llm.py build them, so a batch reply is a draw from the same model with the same settings.
The local Kev judge is called directly and cached under the same keys.
"""

import hashlib
import io
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from compass_pcp.arms.llm import _record

BATCH_PRICE = {  # USD per million tokens (input, output), half the list price; used only for the cost report
    "anthropic:claude-opus-5-5": (2.0, 10.0),
    "anthropic:claude-sonnet-5-5": (1.0, 5.0),
    "openai:gpt-4.1-2025-04-14": (1.0, 4.0),
    "google:gemini-3.1-pro-preview": (1.0, 6.0),
    "google:gemini-3.8-flash": (0.25, 1.5),
}


class Pending(Exception):
    """The pipeline needs a model reply that is not in the store yet."""


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Store:
    """Append-only results file plus the queue of requests for the next batch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.results_path = root / "results.jsonl"
        self.results: dict[str, dict] = {}
        if self.results_path.exists():
            for line in self.results_path.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.results[r["key"]] = r
        self.queue: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        return self.results.get(key)

    def enqueue(self, key: str, request: dict) -> None:
        with self._lock:
            self.queue.setdefault(key, request)

    def put(self, key: str, text: str, usage: dict, spec: str) -> None:
        rec = {"key": key, "spec": spec, "text": text, "usage": usage}
        with self._lock:
            self.results[key] = rec
            with self.results_path.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def cost(self) -> dict:
        out: dict = {}
        for r in self.results.values():
            u = out.setdefault(r["spec"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            u["calls"] += 1
            u["input_tokens"] += r["usage"].get("input_tokens", 0)
            u["output_tokens"] += r["usage"].get("output_tokens", 0)
        for spec, u in out.items():
            p = BATCH_PRICE.get(spec)
            u["usd_at_batch_price"] = None if p is None else round(u["input_tokens"] / 1e6 * p[0] + u["output_tokens"] / 1e6 * p[1], 2)
        return out


class SlotState:
    """Occurrence counts and the lowest missing tier for one pipeline execution."""

    def __init__(self, slot_id) -> None:
        self.id, self.counts, self.min_miss, self._lock = slot_id, {}, 99, threading.Lock()

    def key(self, content) -> str:
        h = _h(content)
        with self._lock:
            n = self.counts.get(h, 0)
            self.counts[h] = n + 1
        return _h([self.id, h, n])

    @property
    def missed(self) -> bool:
        return self.min_miss < 99


_slot = threading.local()


@contextmanager
def slot(slot_id):
    """One pipeline execution on this thread. Occurrence counts restart, so a re-execution, or a variant that repeats
    the same calls in the same order, maps each call to the same key and reuses its result."""
    _slot.state = SlotState(slot_id)
    try:
        yield _slot.state
    finally:
        _slot.state = None


@dataclass
class StagedLLM:
    """Drop-in for an AnthropicLLM, GeminiLLM or OpenAILLM. ``tier_of(system)`` gives a call's tier; ``placeholder`` is
    returned for a queued call that does not stop the pipeline. ``realtime`` answers misses at once (smoke tests)."""

    inner: object
    store: Store
    tier_of: object = lambda system: 3
    placeholder: str = ""
    realtime: bool = False
    shared: SlotState | None = None  # used instead of the thread's slot when the caller runs its own threads (grading)
    vendor: str = field(init=False)
    model: str = field(init=False)
    phi_allowed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.vendor, self.model, self.phi_allowed = self.inner.vendor, self.inner.model, self.inner.phi_allowed
        self.usage = self.inner.usage

    @property
    def spec(self) -> str:
        return f"{self.vendor}:{self.model}"

    def request(self, system, messages, max_tokens, temperature) -> dict:
        i = self.inner
        if self.vendor == "anthropic":
            msgs = [m for m in messages if m["role"] in ("user", "assistant")] or [{"role": "user", "content": "(start)"}]
            if msgs[0]["role"] != "user":
                msgs = [{"role": "user", "content": "(conversation continues)"}, *msgs]
            return {"model": self.model, "system": system, "messages": msgs, "max_tokens": max_tokens + i.thinking_headroom}
        if self.vendor == "google":
            return {
                "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                             for m in messages if m["role"] in ("user", "assistant")],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {"maxOutputTokens": max_tokens + i.thinking_headroom, "temperature": temperature,
                                     "thinkingConfig": {"thinkingLevel": i.thinking_level}},
            }
        if self.vendor == "openai":
            return {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                    "messages": [{"role": "system", "content": system},
                                 *[m for m in messages if m["role"] in ("user", "assistant")]]}
        raise ValueError(self.vendor)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        st = self.shared or _slot.state
        req = self.request(system, messages, max_tokens, temperature)
        key = st.key([self.spec, req])
        hit = self.store.get(key)
        if hit is not None:
            _record(self, hit["usage"].get("input_tokens", 0), hit["usage"].get("output_tokens", 0))
            return hit["text"]
        if self.realtime:
            before = (self.inner.usage.input_tokens, self.inner.usage.output_tokens)
            text = self.inner.complete(system, messages, max_tokens=max_tokens, temperature=temperature)
            usage = {"input_tokens": self.inner.usage.input_tokens - before[0],
                     "output_tokens": self.inner.usage.output_tokens - before[1]}
            self.store.put(key, text, usage, self.spec)
            return text
        tier = self.tier_of(system)
        if tier > st.min_miss:
            if tier >= 3:
                raise Pending
            return self.placeholder
        self.store.enqueue(key, {"spec": self.spec, "vendor": self.vendor, "model": self.model, "body": req})
        st.min_miss = min(st.min_miss, tier)
        if tier >= 3:
            raise Pending
        return self.placeholder


@dataclass
class CachedKev:
    """The local Kev judge, called at once and stored, so later passes do not call it again."""

    inner: object
    store: Store

    def __post_init__(self) -> None:
        self.vendor, self.model, self.phi_allowed, self.usage = (self.inner.vendor, self.inner.model,
                                                                 self.inner.phi_allowed, self.inner.usage)

    def choice(self, state, instructions, options):
        key = _slot.state.key(["kev", self.model, state[-30000:], instructions, options])
        hit = self.store.get(key)
        if hit is not None:
            return json.loads(hit["text"])
        probs = self.inner.choice(state, instructions, options)
        self.store.put(key, json.dumps(probs), {}, f"{self.vendor}:{self.model}")
        return probs


# ---- batch submission and collection -------------------------------------------------------------------------------

MAX_PER_BATCH = {"anthropic": 50_000, "openai": 40_000, "google": 20_000}
MAX_BYTES = 150_000_000


def _chunks(items: list[tuple[str, dict]], vendor: str):
    chunk, size = [], 0
    for k, r in items:
        b = len(json.dumps(r["body"]))
        if chunk and (len(chunk) >= MAX_PER_BATCH[vendor] or size + b > MAX_BYTES):
            yield chunk
            chunk, size = [], 0
        chunk.append((k, r))
        size += b
    if chunk:
        yield chunk


def submit(store: Store) -> list[dict]:
    """Submit every queued request, grouped by vendor and model, and record the batch ids in jobs.jsonl."""
    jobs = []
    groups: dict[tuple[str, str], list] = {}
    for k, r in store.queue.items():
        groups.setdefault((r["vendor"], r["model"]), []).append((k, r))
    for (vendor, model), items in groups.items():
        for chunk in _chunks(items, vendor):
            keys = {_h(k)[:60]: k for k, _ in chunk}  # custom ids must be short; map them back to keys
            if vendor == "anthropic":
                import anthropic

                b = anthropic.Anthropic().messages.batches.create(
                    requests=[{"custom_id": _h(k)[:60], "params": r["body"]} for k, r in chunk])
                jid = b.id
            elif vendor == "openai":
                import openai

                c = openai.OpenAI()
                lines = "".join(json.dumps({"custom_id": _h(k)[:60], "method": "POST", "url": "/v1/chat/completions",
                                            "body": r["body"]}) + "\n" for k, r in chunk)
                f = c.files.create(file=("batch.jsonl", io.BytesIO(lines.encode())), purpose="batch")
                jid = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h").id
            else:
                from google import genai

                c = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
                lines = "".join(json.dumps({"key": _h(k)[:60], "request": r["body"]}) + "\n" for k, r in chunk)
                path = store.root / f"gemini_src_{int(time.time() * 1000)}.jsonl"
                path.write_text(lines)
                f = c.files.upload(file=str(path), config={"mime_type": "jsonl"})
                jid = c.batches.create(model=model, src=f.name).name
            job = {"vendor": vendor, "model": model, "id": jid, "n": len(chunk), "keys": keys, "done": False}
            jobs.append(job)
            with (store.root / "jobs.jsonl").open("a") as fh:
                fh.write(json.dumps(job) + "\n")
    store.queue.clear()
    return jobs


def _anthropic_collect(job, store) -> bool:
    import anthropic

    c = anthropic.Anthropic()
    if c.messages.batches.retrieve(job["id"]).processing_status != "ended":
        return False
    for r in c.messages.batches.results(job["id"]):
        if r.result.type != "succeeded":
            continue  # errored or expired: the key stays missing and is queued again on the next pass
        m = r.result.message
        text = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
        store.put(job["keys"][r.custom_id], text, {"input_tokens": m.usage.input_tokens,
                                                   "output_tokens": m.usage.output_tokens}, f"anthropic:{job['model']}")
    return True


def _openai_collect(job, store) -> bool:
    import openai

    c = openai.OpenAI()
    b = c.batches.retrieve(job["id"])
    if b.status not in ("completed", "failed", "expired", "cancelled"):
        return False
    if b.output_file_id:
        for line in c.files.content(b.output_file_id).text.splitlines():
            r = json.loads(line)
            body = (r.get("response") or {}).get("body") or {}
            if not body.get("choices"):
                continue
            u = body.get("usage") or {}
            store.put(job["keys"][r["custom_id"]], body["choices"][0]["message"].get("content") or "",
                      {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)},
                      f"openai:{job['model']}")
    return True


def _google_collect(job, store) -> bool:
    from google import genai

    c = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    b = c.batches.get(name=job["id"])
    state = str(b.state)
    if not any(s in state for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
        return False
    if b.dest and b.dest.file_name:
        for line in c.files.download(file=b.dest.file_name).decode().splitlines():
            r = json.loads(line)
            resp = r.get("response")
            if not resp:
                continue
            parts = ((resp.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            um = resp.get("usageMetadata") or {}
            # thinking tokens are billed as output, as in GeminiLLM
            store.put(job["keys"][r["key"]], text, {"input_tokens": um.get("promptTokenCount", 0),
                      "output_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)},
                      f"google:{job['model']}")
    return True


def collect(store: Store) -> int:
    """Fetch results of finished batches; returns the number still running."""
    path = store.root / "jobs.jsonl"
    if not path.exists():
        return 0
    jobs = {}
    for line in path.read_text().splitlines():
        j = json.loads(line)
        jobs[j["id"]] = j  # a later line for the same id (done=True) supersedes the earlier one
    running = 0
    for j in jobs.values():
        if j["done"]:
            continue
        fn = {"anthropic": _anthropic_collect, "openai": _openai_collect, "google": _google_collect}[j["vendor"]]
        if fn(j, store):
            with path.open("a") as fh:
                fh.write(json.dumps({**j, "done": True}) + "\n")
        else:
            running += 1
    return running
