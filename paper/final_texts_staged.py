"""Final test on the fresh real-text set at batch prices (Paper 2 preregistration, sections 3.1 and 5).

Each arm replies to each labeled text R times through the staged batch runner of paper/v4_dev_staged.py (the reply
pass only; HealthBench rubric grading does not apply), then a blinded classifier (the Analysis 5 urgency prompt,
Claude Sonnet 5.5) reads each reply alone and labels the urgency it advises. Runs after the labels are complete and the
preregistration is registered. Safe to stop and restart.

Inputs come from custodian storage (compass-pcp-data/fresh_set): pool.jsonl (texts) and sample.jsonl (the 949 sampled
ids). Outputs stay there: replies.jsonl and urgency.jsonl (ids, arm, repeat, urgency). Only counts are printed.
Message text goes only to the vendors the arms call (Anthropic, Google) under BAA, and to the local Kev judge.

Usage: python -m paper.final_texts_staged replies  --fresh <fresh_set> --out <dir> [--repeats 3] [--budget 900]
       python -m paper.final_texts_staged classify --out <dir> [--threads 8]
"""

import argparse
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from paper import v4_dev_staged as vs
from paper.batch_stage import Store
from paper.sealed_comparators import CLASSIFIER_PROMPT, parse_urgency

ARMS = ["v22", "raw_llm", "amie_style", "mai_dxo_style"]
CLASSIFIER_SPEC = "anthropic:claude-sonnet-5-5"  # the development urgency classifier (paper/dev_realmsg.py)


def rows_from(fresh: Path, ids: list[str] | None = None) -> list[dict]:
    pool = {json.loads(x)["id"]: json.loads(x)["message"] for x in (fresh / "pool.jsonl").read_text().splitlines()}
    want = ids or [json.loads(x)["id"] for x in (fresh / "sample.jsonl").read_text().splitlines()]
    return [{"prompt_id": i, "prompt": [{"role": "user", "content": pool[i]}]} for i in sorted(want)]


def replies(fresh: Path, out: Path, repeats: int, threads: int, budget: float, ids_file: Path | None,
            realtime: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ids = json.loads(ids_file.read_text()) if ids_file else None
    rows = rows_from(fresh, ids)
    store = Store(out / "store")
    vs.run_loop([vs.sample_pass], rows, repeats, store, out, realtime, threads, ARMS, poll=60, budget=budget)
    done = vs._done(out)
    print(json.dumps({"texts": len(rows), "repeats": repeats, "replies": len(done),
                      "expected": len(rows) * repeats * len(ARMS)}))


def classify(out: Path, threads: int) -> None:
    from scripts.run_healthbench import make_llm

    reps = [json.loads(x) for x in (out / "replies.jsonl").read_text().splitlines()]
    path = out / "urgency.jsonl"
    done = {(r["prompt_id"], r["repeat"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} \
        if path.exists() else set()
    todo = [r for r in reps if (r["prompt_id"], r["repeat"], r["arm"]) not in done]
    lock, local = threading.Lock(), threading.local()

    def one(r):
        if not hasattr(local, "c"):
            local.c = make_llm(CLASSIFIER_SPEC)
        urgency = None
        for _ in range(2):
            urgency = parse_urgency(local.c.complete(CLASSIFIER_PROMPT, [{"role": "user", "content": "REPLY:\n" + r["text"]}],
                                                     max_tokens=200))
            if urgency:
                break
        rec = {"prompt_id": r["prompt_id"], "repeat": r["repeat"], "arm": r["arm"], "urgency": urgency,
               "review_flag": ((r.get("info") or {}).get("review") or {}).get("flag"),
               "route": (r.get("path") or {}).get("route"), "decision": (r.get("path") or {}).get("decision")}
        with lock, path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return urgency is None

    with cf.ThreadPoolExecutor(threads) as ex:
        unparsed = sum(ex.map(one, todo))
    print(json.dumps({"classified": len(done) + len(todo), "unparsed_this_run": unparsed}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["replies", "classify"])
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--ids", type=Path, help="JSON list of message ids (smoke tests on development data)")
    ap.add_argument("--realtime", action="store_true", help="smoke tests only; the final run uses batch")
    a = ap.parse_args()
    if a.cmd == "replies":
        replies(a.fresh, a.out, a.repeats, a.threads, a.budget, a.ids, a.realtime)
    else:
        classify(a.out, a.threads)


if __name__ == "__main__":
    main()
