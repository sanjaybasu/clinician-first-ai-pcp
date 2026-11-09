"""Development analysis of hazard scores for the conformal ask-or-answer rule (development data only).

Reads <scores>/scores.jsonl (paper/score_ensemble.py) and the BMC MIDM labels. Reports, for safety positives (Cohort A
definition): AUROC of each score and of a cross-fitted logistic stack; and conformal risk control (CRC) of the miss rate
at alpha 0.05 and 0.10 over 500 random 50/50 calibration/test splits: the share of messages sent to review and the
realized miss rate on the test half. The DS-2030 baseline (deployed classifier) was 73% review at a 4.9% miss rate.
Prints numbers only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ORDER = ["none", "routine_followup", "urgent_same_day", "possible_emergency", "emergency_now"]


def features(s: dict) -> dict[str, float]:
    t = s["kev_tier"]
    return {"kev_any": 1 - t.get("none", 0.0), "kev_urgent_plus": sum(t.get(k, 0.0) for k in ORDER[2:]),
            "kev_em": t.get("possible_emergency", 0.0) + t.get("emergency_now", 0.0),
            "gpt_see": s["gpt_see"] if s["gpt_see"] is not None else 0.5,
            "gpt_emergency": s["gpt_emergency"] if s["gpt_emergency"] is not None else 0.5}


def crc_threshold(score_pos: np.ndarray, alpha: float) -> float:
    """Largest threshold t such that the conformal bound on the miss rate, (misses(t) + 1) / (n_pos + 1), is <= alpha;
    a message is reviewed when its score >= t. Misses are positives with score < t."""
    s = np.sort(score_pos)
    n = len(s)
    k = int(np.floor(alpha * (n + 1) - 1))  # allowed misses
    if k < 0:
        return -np.inf
    return s[k] if k < n else np.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--outcome", choices=["safety", "any"], default="safety",
                    help="safety: Cohort A safety positive; any: any physician-labeled hazard (the DS-2030 outcome)")
    a = ap.parse_args()
    lab = {str(r["message_id"]): r for r in json.loads(a.cases.read_text())}
    sc = [json.loads(x) for x in (a.scores / "scores.jsonl").read_text().splitlines()]
    sc = [s for s in sc if s["id"] in lab]
    if a.outcome == "any":
        y = np.array([int(lab[s["id"]]["detection_truth"] == 1) for s in sc])
    else:
        y = np.array([int((lab[s["id"]]["detection_truth"] == 1 and lab[s["id"]]["hazard_category"] != "other_hazard")
                          or "911" in str(lab[s["id"]]["action_truth"])) for s in sc])
    F = [features(s) for s in sc]
    names = list(F[0])
    X = np.array([[f[k] for k in names] for f in F])
    out = {"n": len(y), "positives": int(y.sum()), "auroc": {k: round(roc_auc_score(y, X[:, i]), 3) for i, k in enumerate(names)}}
    stack = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
        stack[te] = m.predict_proba(X[te])[:, 1]
    out["auroc"]["stack_crossfit"] = round(roc_auc_score(y, stack), 3)
    rng = np.random.default_rng(20261107)
    for name, score in [("stack", stack), ("kev_any", X[:, names.index("kev_any")]), ("gpt_see", X[:, names.index("gpt_see")])]:
        for alpha in (0.05, 0.10):
            rev, miss = [], []
            for _ in range(a.splits):
                idx = rng.permutation(len(y))
                cal, te = idx[: len(y) // 2], idx[len(y) // 2:]
                t = crc_threshold(score[cal][y[cal] == 1], alpha)
                r = score[te] >= t
                rev.append(r.mean())
                miss.append(1 - r[y[te] == 1].mean())
            out[f"crc_{name}_a{alpha}"] = {"review": round(float(np.mean(rev)), 3), "miss": round(float(np.mean(miss)), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
