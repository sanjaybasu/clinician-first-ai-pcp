"""Hazard screen on a development message set (never the frozen test set), with on-disk caching.

Usage:
  python -m scripts.run_hazard_dev --cases <json> --screen kev|llm [--model vendor:model] --out <dir outside git>
Real patient messages may only go to local models (kev) or BAA-covered vendors; the caller is responsible for
choosing --model accordingly. Reports AUROC, partial AUROC, sensitivity floors and workload per 1,000.
"""

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from compass_pcp.arms.hazard import KEV_INSTRUCTIONS, KEV_TIERS, patient_text, screen
from compass_pcp.arms.llm import KevJudge
from compass_pcp.harness.cache import ResponseCache
from compass_pcp.harness.metrics import auroc, partial_auroc_high_sensitivity, specificity_at_sensitivity_floor, wilson
from scripts.run_healthbench import make_llm

ORDER = ["none", "routine_followup", "urgent_same_day", "possible_emergency", "emergency_now"]


def fields(c: dict) -> tuple[str, str, int, str]:
    msg = c.get("message") or c.get("prompt") or ""
    truth = c.get("detection_truth", c.get("ground_truth_detection"))
    action = c.get("action_truth", c.get("ground_truth_action"))
    cid = c.get("message_id") or c.get("case_id") or c.get("name")
    return cid, msg, int(truth), str(action)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--screen", choices=["kev", "llm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None, help="filter on dataset_source (e.g. real-world)")
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of ids to keep (e.g. the Cohort A tuning half)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.source:
        cases = [c for c in cases if c.get("dataset_source") == args.source]
    if args.ids:
        if "test" in args.ids.name:
            raise SystemExit("refusing to tune on the sealed Cohort A test half")
        keep = set(json.loads(args.ids.read_text()))
        cases = [c for c in cases if fields(c)[0] in keep]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "kev" if args.screen == "kev" else args.model.replace(":", "_")
    cache = ResponseCache(args.out / f"cache_{tag}.jsonl")
    judge = KevJudge() if args.screen == "kev" else None
    llm = make_llm(args.model) if args.screen == "llm" else None

    def one(c):
        cid, msg, truth, action = fields(c)
        msgs = [{"role": "user", "content": msg}]
        if judge is not None:
            probs = cache.get_or_call({"id": cid, "m": msg}, lambda: judge.choice(patient_text(msgs), KEV_INSTRUCTIONS, KEV_TIERS))
            score_any = 1 - probs.get("none", 0.0)
            score_urgent = sum(probs.get(t, 0.0) for t in ORDER[2:])
        else:
            tier = cache.get_or_call({"id": cid, "m": msg}, lambda: int(screen(llm, msgs, 0).tier))
            score_any, score_urgent = float(tier >= 1), float(tier)
        return {"id": cid, "truth": truth, "action": action, "score_any": score_any, "score_urgent": score_urgent}

    with cf.ThreadPoolExecutor(args.threads) as ex:
        rows = list(ex.map(one, cases))
    (args.out / f"per_case_{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    y = [r["truth"] == 1 for r in rows]
    y911 = [r["action"] == "Call 911/988" for r in rows]
    res = {"n": len(rows), "hazards": sum(y), "call_911": sum(y911), "screen": tag}
    for name, sc in (("any", [r["score_any"] for r in rows]), ("urgent", [r["score_urgent"] for r in rows])):
        res[name] = {
            "auroc": auroc(sc, y),
            "pauroc_sens_ge_0.90": partial_auroc_high_sensitivity(sc, y, 0.9),
            "at_sens_0.90": specificity_at_sensitivity_floor(sc, y, 0.90),
            "at_sens_0.95": specificity_at_sensitivity_floor(sc, y, 0.95),
            "auroc_call_911": auroc(sc, y911),
        }
        op = res[name]["at_sens_0.90"]
        if op["threshold"] is not None:
            flagged = sum(s >= op["threshold"] for s in sc)
            res[name]["flags_per_1000_at_sens_0.90"] = 1000 * flagged / len(rows)
            tp = sum(s >= op["threshold"] and t for s, t in zip(sc, y))
            res[name]["sens_0.90_point_ci"] = wilson(tp, sum(y))
    (args.out / f"summary_{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
