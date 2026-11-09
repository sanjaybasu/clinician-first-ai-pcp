"""Compare hazard scorers on the 2,000 physician-labeled development messages (development data only).

Each source is a JSONL of {id, kev_tier: {tier: p}} (Kev, off the shelf or fine-tuned) or the deployed classifier CSV.
For each outcome it reports AUROC, Brier score of the matching cumulative probability, and conformal risk control (CRC)
of the miss rate over random 50/50 calibration/test splits at alpha 0.01, 0.05 and 0.10 (undefined when the calibration
half has too few positives for that alpha). Prints numbers only.

Outcomes (physician labels):
  safety     Cohort A safety positive (hazard present and not other_hazard, or action Call 911/988)
  doctor     action at least "Contact Doctor"
  any        any hazard (detection_truth == 1)
  emergency  action "Call 911/988"
Usage: python -m paper.system1_eval --cases <json> --deployed <csv> --src name=path [--src ...]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ORDER = ["none", "routine_followup", "urgent_same_day", "possible_emergency", "emergency_now"]
SCORE_FOR = {"safety": 2, "doctor": 2, "any": 1, "emergency": 3}  # cumulative P(tier >= ORDER[k])


def outcomes(r: dict) -> dict[str, int]:
    act = str(r["action_truth"])
    return {"safety": int((r["detection_truth"] == 1 and r["hazard_category"] != "other_hazard") or "911" in act),
            "doctor": int(act in ("Contact Doctor", "Call 911/988")), "any": int(r["detection_truth"] == 1),
            "emergency": int("911" in act)}


def crc(score, y, alpha, splits, rng):
    rev, miss, undefined = [], [], 0
    n = len(y)
    for _ in range(splits):
        idx = rng.permutation(n)
        cal, te = idx[: n // 2], idx[n // 2:]
        pos = np.sort(score[cal][y[cal] == 1])
        k = int(np.floor(alpha * (len(pos) + 1) - 1))
        if k < 0:
            undefined += 1
            continue
        t = pos[k] if k < len(pos) else np.inf
        r = score[te] >= t
        rev.append(r.mean())
        if (y[te] == 1).any():
            miss.append(1 - r[y[te] == 1].mean())
    if not rev:
        return {"undefined": True}
    return {"review": round(float(np.mean(rev)), 3), "miss": round(float(np.mean(miss)), 3),
            "undefined_splits": undefined}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--deployed", type=Path)
    ap.add_argument("--src", action="append", default=[], help="name=path to {id, kev_tier} JSONL")
    ap.add_argument("--splits", type=int, default=500)
    a = ap.parse_args()
    lab = {str(r["message_id"]): outcomes(r) for r in json.loads(a.cases.read_text())}
    src: dict[str, dict[str, object]] = {}
    for s in a.src:
        name, path = s.split("=", 1)
        src[name] = {}
        for x in Path(path).read_text().splitlines():
            d = json.loads(x)
            if d["id"] in lab:
                src[name][d["id"]] = d["kev_tier"]
    if a.deployed:
        with a.deployed.open() as fh:
            src["deployed"] = {r["message_id"]: float(r["score"]) for r in csv.DictReader(fh) if r["message_id"] in lab}
    common = sorted(set(lab).intersection(*[set(v) for v in src.values()]))
    out = {"n": len(common), "positives": {o: sum(lab[i][o] for i in common) for o in SCORE_FOR}, "results": {}}
    for o, k in SCORE_FOR.items():
        y = np.array([lab[i][o] for i in common])
        res = {}
        for name, d in src.items():
            if name == "deployed":
                s = np.array([d[i] for i in common])
            else:
                s = np.array([sum(d[i].get(t, 0.0) for t in ORDER[k:]) for i in common])
            rng = np.random.default_rng(20261107)
            res[name] = {"auroc": round(roc_auc_score(y, s), 3),
                         "brier": None if name == "deployed" else round(float(np.mean((s - y) ** 2)), 4),
                         **{f"crc_{al}": crc(s, y, al, a.splits, rng) for al in (0.01, 0.05, 0.10)}}
        out["results"][o] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
