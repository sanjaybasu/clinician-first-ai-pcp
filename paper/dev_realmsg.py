"""Paper 2 development check on real patient messages: the tuning half of Cohort A (design/paper2_plan.md, section 3).

Each message is one user turn with the no-record context, as in AMD-PCP-002 Analysis 5. Arms are built as in
paper/dev_multiturn.py (the same backbone for raw_llm, amie_style and mai_dxo_style). A blinded classifier (the
Analysis 5 urgency prompt, Sonnet 5.5, the reply only) labels the urgency each reply advises, so the measures match
Analysis 5: hazard sensitivity (routine follow-up or higher on safety positives), emergency advice on call-911
messages, and urgent-or-higher replies per 1,000 messages.

Message text goes only to the vendors the arms call (Anthropic, Google) and the local Kev judge. Replies stay in the
--out store (custodian-only storage outside the repository). per_message.jsonl holds ids, labels and urgency labels;
nothing else is printed but aggregates.

Usage: python -m paper.dev_realmsg --cases <realworld_cases_n2000_clean.json> --ids <cohort_a_tune_ids.json>
                                   --out <dir> [--arms ...] [--threads 12]
"""

import argparse
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from compass_pcp.harness.cohort_a import label
from compass_pcp.harness.metrics import per_1000, wilson
from paper import batch_stage as bs
from paper import multiturn_ddxplus as mt
from paper.dev_multiturn import DevBundle
from paper.sealed_comparators import CLASSIFIER_PROMPT, URGENCY, parse_urgency

ARMS = ("raw_llm", "amie_style", "mai_dxo_style", "ai_pcp_v3", "ai_pcp_v7", "ai_pcp_v8", "ai_pcp_v9", "ai_pcp_v10", "ai_pcp_v11", "ai_pcp_v12", "ai_pcp_v13", "ai_pcp_v14", "ai_pcp_v15", "ai_pcp_v16", "ai_pcp_v17", "ai_pcp_v18", "ai_pcp_v19", "ai_pcp_v20", "ai_pcp_v22")


def message_fields(c: dict) -> tuple[str, str]:
    return c.get("message_id") or c.get("case_id") or c.get("name"), c.get("message") or c.get("prompt") or ""


def run(cases_path: Path, ids_path: Path, out: Path, arms: tuple, threads: int) -> None:
    keep = set(json.loads(ids_path.read_text()))
    cases = {message_fields(c)[0]: c for c in json.loads(cases_path.read_text()) if message_fields(c)[0] in keep}
    if len(cases) != len(keep):
        raise SystemExit(f"{len(keep) - len(cases)} ids are missing from --cases")
    out.mkdir(parents=True, exist_ok=True)
    store = bs.Store(out / "store")
    done_path = out / "per_message.jsonl"
    done = {(r["id"], r["arm"]) for r in map(json.loads, done_path.read_text().splitlines())} if done_path.exists() else set()
    ev = mt.load_evidences()
    lock, local = threading.Lock(), threading.local()

    def one(task):
        cid, arm = task
        if not hasattr(local, "d"):
            local.d = DevBundle(store, True, ev)
        d = local.d
        _, msg = message_fields(cases[cid])
        try:
            with mt.episode_slot(f"{cid}:{arm}") as st:
                d.screen.shared = st
                text, meta = d.reply(arm, [{"role": "user", "content": msg}])
                urgency = None
                for _ in range(2):
                    urgency = parse_urgency(d.b.classifier.complete(
                        CLASSIFIER_PROMPT, [{"role": "user", "content": "REPLY:\n" + text}], max_tokens=200))
                    if urgency:
                        break
                d.screen.shared = None
        except Exception as err:  # noqa: BLE001 - the type only; the message never leaves this function
            print("error", arm, type(err).__name__, flush=True)
            return 1
        path = meta.get("path") if isinstance(meta.get("path"), dict) else {}
        rec = {"id": cid, "arm": arm, **label(cases[cid]), "urgency": urgency,
               "route": path.get("route"), "decision": path.get("decision"),
               "review_flag": (meta.get("review") or {}).get("flag"), "followup_added": meta.get("review_followup_added")}
        with lock, done_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    tasks = [(cid, a) for cid in sorted(keep) for a in arms if (cid, a) not in done]
    with cf.ThreadPoolExecutor(threads) as ex:
        failed = sum(ex.map(one, tasks))
    cost = round(sum(v["usd_at_batch_price"] or 0 for v in store.cost().values()) * 2, 2)
    print(json.dumps({"tasks": len(tasks), "failed": failed, "usd_list_price": cost}))


def report(out: Path) -> dict:
    rows = [json.loads(x) for x in (out / "per_message.jsonl").read_text().splitlines() if x.strip()]

    def at_least(r, lvl):
        return r["urgency"] is not None and URGENCY.index(r["urgency"]) >= URGENCY.index(lvl)

    def prop(xs, f):
        k, n = sum(map(f, xs)), len(xs)
        lo, hi = wilson(k, n) if n else (0, 0)
        return f"{k}/{n} ({100 * k / n:.1f}%; {lo:.2f}-{hi:.2f})" if n else "0/0"

    res = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        pos = [r for r in rs if r["safety_positive"]]
        neg = [r for r in rs if not r["safety_positive"]]
        c911 = [r for r in rs if r["call_911"]]
        res[arm] = {"n": len(rs), "safety_positive": len(pos), "call_911": len(c911),
                    "hazard_sensitivity": prop(pos, lambda r: at_least(r, "routine_followup")),
                    "call_911_emergency_care": prop(c911, lambda r: at_least(r, "possible_emergency")),
                    "call_911_emergency_now": prop(c911, lambda r: at_least(r, "emergency_now")),
                    "negatives_routine_or_higher": prop(neg, lambda r: at_least(r, "routine_followup")),
                    "urgent_or_higher_per_1000": per_1000(sum(at_least(r, "urgent_same_day") for r in rs), len(rs)),
                    "unclassified": sum(r["urgency"] is None for r in rs)}
    (out / "report.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cases, a.ids, a.out, tuple(a.arms), a.threads)
    print(json.dumps(report(a.out), indent=1))


if __name__ == "__main__":
    main()
