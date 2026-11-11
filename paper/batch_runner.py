"""Run HealthBench Professional arms through the Anthropic and OpenAI Batch APIs (50% of list price), with the chains,
prompts, models and parameters unchanged.

Execution-mode deviation from https://osf.io/tnezd, recorded in docs/regulatory/freeze/paper2_addendum/deviations.md:
each model call that a chain makes (ProPipeline._opus via messages.stream(...).get_final_message(), and
OpenAIReasoningLLM.complete via chat.completions.create(...)) is queued; when the queue has had no new request for
QUIET seconds, the queued requests are sent as one batch, and each waiting chain gets its own result and continues.
Request parameters are the ones the chain would have sent in real time (no streaming). Replies, usage (at list price)
and search logs are written exactly as by paper.hbpro sample, so grading, selection and analysis are unchanged.
Usage: python -m paper.batch_runner --out <dir> --split test --arms <arm> ...
"""

import argparse
import io
import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

QUIET, POLL = 30.0, 30.0


def _plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True)
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


class Batcher:
    def __init__(self, kind: str):
        self.kind, self.q, self.last, self.lock = kind, [], 0.0, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def call(self, params: dict):
        ev, box = threading.Event(), {}
        with self.lock:
            self.q.append((f"r{uuid.uuid4().hex[:24]}", params, ev, box))
            self.last = time.time()
        ev.wait()
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["result"]

    def _loop(self):
        while True:
            time.sleep(5)
            with self.lock:
                if not self.q or time.time() - self.last < QUIET:
                    continue
                todo, self.q = self.q, []
            try:
                res = self._anthropic(todo) if self.kind == "anthropic" else self._openai(todo)
            except Exception as e:  # noqa: BLE001
                res = {cid: ("error", f"batch failed: {type(e).__name__} {e}") for cid, *_ in todo}
            for cid, _, ev, box in todo:
                kind, val = res.get(cid, ("error", "missing result"))
                box["result" if kind == "ok" else "error"] = val
                ev.set()

    def _anthropic(self, todo):
        import anthropic
        c = anthropic.Anthropic(base_url="https://api.anthropic.com", timeout=600.0, max_retries=4)
        b = c.messages.batches.create(requests=[{"custom_id": cid, "params": _plain(p)} for cid, p, *_ in todo])
        print(f"anthropic batch {b.id}: {len(todo)} requests", flush=True)
        while c.messages.batches.retrieve(b.id).processing_status != "ended":
            time.sleep(POLL)
        out = {}
        for e in c.messages.batches.results(b.id):
            out[e.custom_id] = ("ok", e.result.message) if e.result.type == "succeeded" else ("error", str(e.result.type))
        print(f"anthropic batch {b.id}: ended, {sum(v[0] == 'ok' for v in out.values())} succeeded", flush=True)
        return out

    def _openai(self, todo):
        import openai
        c = openai.OpenAI(timeout=600.0, max_retries=4)
        lines = "".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": p}) + "\n"
                        for cid, p, *_ in todo)
        f = c.files.create(file=("batch.jsonl", io.BytesIO(lines.encode())), purpose="batch")
        b = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
        print(f"openai batch {b.id}: {len(todo)} requests", flush=True)
        while (b := c.batches.retrieve(b.id)).status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(POLL)
        out = {}
        if b.output_file_id:
            for ln in c.files.content(b.output_file_id).text.splitlines():
                r = json.loads(ln)
                body = (r.get("response") or {}).get("body") or {}
                if body.get("choices"):
                    u = body.get("usage") or {}
                    out[r["custom_id"]] = ("ok", SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=body["choices"][0]["message"].get("content")))],
                        usage=SimpleNamespace(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0))))
                else:
                    out[r["custom_id"]] = ("error", json.dumps(r.get("error") or body)[:200])
        print(f"openai batch {b.id}: {b.status}, {sum(v[0] == 'ok' for v in out.values())} succeeded", flush=True)
        return out


ANTH, OAI = None, None


class _Stream:
    def __init__(self, kw):
        self.kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return ANTH.call(self.kw)


def _patch():
    from compass_pcp.arms import pro
    from compass_pcp.arms.openai_reasoning import OpenAIReasoningLLM

    def post_init(self):
        self._a = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream(kw)))
        self._g = OpenAIReasoningLLM("gpt-6.1-sol")
        self._g._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: OAI.call(kw))))

    pro.ProPipeline.__post_init__ = post_init


def main() -> None:
    global ANTH, OAI
    from paper import hbpro

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["development", "test"])
    ap.add_argument("--arms", nargs="+", required=True, choices=hbpro.BLK_ARMS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ANTH, OAI = Batcher("anthropic"), Batcher("openai")
    _patch()
    rows = hbpro.load()
    ids = json.loads((a.out / "split.json").read_text())[a.split][:a.limit]
    done = {(r["id"], r["arm"]) for r in hbpro._rows(a.out / "replies.jsonl")}
    todo = [(i, arm) for arm in a.arms for i in ids if (i, arm) not in done]
    print(f"{len(todo)} (task, arm) chains", flush=True)
    res = []

    def one(t):
        i, arm = t
        try:
            text = hbpro.answer(arm, rows[i]["conversation"]["messages"], a.out)
        except Exception as e:  # noqa: BLE001
            print("error", arm, type(e).__name__, str(e)[:160], flush=True)
            res.append("error")
            return
        hbpro._append(a.out / "replies.jsonl", {"id": i, "arm": arm, "text": text})
        res.append("ok")

    ts = [threading.Thread(target=one, args=(t,)) for t in todo]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(json.dumps({"done": res.count("ok"), "errors": res.count("error"), "usd_list": round(hbpro.spent(a.out), 2)}))


if __name__ == "__main__":
    main()
