"""Score fine-tuned Kev runs on the 2,000 physician-labeled development messages (development data only).

Reads rows.json (ids, question, option keys, probabilities; no text) pulled from each run's development folder: the
five physician folds give one out-of-fold prediction per message, and a run scored on the whole set (teacher model)
gives one each directly. For each outcome and state variant (chart, message only) it reports AUROC and conformal risk
control at alpha 0.01/0.05/0.10 (paper/system1_eval.py), and writes per-message scores to <out>.jsonl for stacking.

Scores: physician-question runs use P(hazard) for "any" and P(action >= Contact Doctor) for "safety" and "doctor",
P(Call 911/988) for "emergency". Teacher-question runs use the cumulative tier probabilities (as system1_eval).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from paper.system1_eval import ORDER, crc, outcomes

PHYS = {"any": ("hazard", ["true"]), "safety": ("action", ["Contact Doctor", "Call 911/988"]),
        "doctor": ("action", ["Contact Doctor", "Call 911/988"]), "emergency": ("action", ["Call 911/988"])}
TEACH = {"any": 1, "safety": 2, "doctor": 2, "emergency": 3}


def load_rows(paths):
    by = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            if r.get("variant", "clean") != "clean":
                continue
            by.setdefault(r["id"], {})[r["question"]] = dict(zip([str(k).lower() if isinstance(k, bool) else str(k)
                                                                  for k in r["keys"]], r["p"]))
    return by


def score(q: dict, outcome: str, kind: str) -> float:
    if kind == "phys":
        qid, keys = PHYS[outcome]
        d = q[qid]
        return sum(d.get(k, 0.0) for k in keys)
    d = q["tier"]
    return sum(d.get(t, 0.0) for t in ORDER[TEACH[outcome]:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--rows", nargs="+", required=True, help="rows.json files")
    ap.add_argument("--kind", choices=["phys", "teach"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="prefix for per-message score files")
    ap.add_argument("--splits", type=int, default=300)
    a = ap.parse_args()
    lab = {str(r["message_id"]): outcomes(r) for r in json.loads(a.cases.read_text())}
    rows = load_rows(a.rows)
    res = {}
    for tag in ("chart", "msg"):
        ids = sorted(i for i in lab if f"{i}|{tag}" in rows)
        sc = {o: np.array([score(rows[f"{i}|{tag}"], o, a.kind) for i in ids]) for o in PHYS}
        with Path(f"{a.out}_{tag}.jsonl").open("w") as fh:
            for j, i in enumerate(ids):
                fh.write(json.dumps({"id": i, **{o: float(sc[o][j]) for o in PHYS}}) + "\n")
        res[tag] = {"n": len(ids)}
        for o in PHYS:
            y = np.array([lab[i][o] for i in ids])
            rng = np.random.default_rng(20261107)
            res[tag][o] = {"auroc": round(roc_auc_score(y, sc[o]), 3), "positives": int(y.sum()),
                           **{f"crc_{al}": crc(sc[o], y, al, a.splits, rng) for al in (0.01, 0.05, 0.10)}}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
