"""Capture the v4 crisis screen's per-criterion statuses on the real-message tuning half (Paper 2 development).

Each message is screened once with the given crisis policy. The output holds message ids, the Cohort A labels, and the
status of each criterion; never quotes or message text. Routing variants are then compared offline (``evaluate``):
catches among safety positives and call-911 messages against flags among non-hazard messages.

Message text goes only to Anthropic (BAA). Usage:
  python -m paper.screen_capture run --cases <realworld_cases_n2000_clean.json> --ids <cohort_a_tune_ids.json>
                                     --policy crisis_v15_dev.yaml --out <file.jsonl> [--threads 8]
  python -m paper.screen_capture evaluate --out <file.jsonl>
"""

import argparse
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from compass_pcp.arms.llm import AnthropicLLM
from compass_pcp.arms.screen_v4 import run_screen
from compass_pcp.harness.cohort_a import label
from compass_pcp.policy_v4 import POLICY_DIR, load_crisis_policy
from paper.dev_realmsg import message_fields

SCREEN_MODEL = "claude-sonnet-5-5"  # paper/dev_multiturn.py SCREEN_SPEC


def run(cases_path: Path, ids_path: Path, policy_file: str, out: Path, threads: int) -> None:
    keep = set(json.loads(ids_path.read_text()))
    cases = {message_fields(c)[0]: c for c in json.loads(cases_path.read_text()) if message_fields(c)[0] in keep}
    done = {json.loads(x)["id"] for x in out.read_text().splitlines()} if out.exists() else set()
    policy = load_crisis_policy(POLICY_DIR / policy_file)
    lock, local = threading.Lock(), threading.local()

    def one(cid):
        if not hasattr(local, "llm"):
            local.llm = AnthropicLLM(model=SCREEN_MODEL, thinking={"type": "between_tools"})
        try:
            form = run_screen(local.llm, [{"role": "user", "content": message_fields(cases[cid])[1]}], policy)
        except Exception as err:  # noqa: BLE001 - the type only; the message never leaves this function
            print("error", type(err).__name__, flush=True)
            return 1
        items = {k: v.model_dump().get("status") for k, v in form.items.items()}
        rec = {"id": cid, **label(cases[cid]), "policy": policy.version, "completed": form.completed, "items": items}
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        failed = sum(ex.map(one, sorted(keep - done)))
    print(json.dumps({"screened": len(keep - done), "failed": failed}))


def evaluate(out: Path) -> None:
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    pos = [r for r in rows if r["safety_positive"]]
    neg = [r for r in rows if not r["safety_positive"]]
    c911 = [r for r in rows if r["call_911"]]
    marked = lambda r, s, pre="": any(v == s and k.startswith(pre) for k, v in r["items"].items())
    variants = {
        "any criterion present": lambda r: marked(r, "present"),
        "any criterion present or uncertain": lambda r: marked(r, "present") or marked(r, "uncertain"),
        "suicide/self-harm (SH) present or uncertain": lambda r: marked(r, "present", "SH") or marked(r, "uncertain", "SH"),
        "substance use (SU) or overdose (OD) present or uncertain": lambda r: any(
            marked(r, s, p) for s in ("present", "uncertain") for p in ("SU", "OD")),
    }
    print(f"n={len(rows)}; safety positives {len(pos)}; call-911 {len(c911)}; non-hazard {len(neg)}")
    for name, f in variants.items():
        print(f"{name}: positives {sum(map(f, pos))}/{len(pos)}, call-911 {sum(map(f, c911))}/{len(c911)}, "
              f"non-hazard {sum(map(f, neg))}/{len(neg)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "evaluate"])
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--policy", default="crisis_v15_dev.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cases, a.ids, a.policy, a.out, a.threads)
    evaluate(a.out)


if __name__ == "__main__":
    main()
