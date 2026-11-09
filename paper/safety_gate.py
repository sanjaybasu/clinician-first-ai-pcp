"""Tiered safety gate around a free-writing reply model (development data only).

The writer drafts every reply. The gate reads only the message and picks one route:
  emergency: the signed 911/988 line goes first and a clinician is paged (score_emergency >= t_e)
  review:    a clinician sees the message before or right after the reply (score_safety >= t_s)
  send:      the reply goes out; a clinician reviews it later as in usual practice
Scores are out-of-fold stacks (logistic regression, cross-fitted on folds.json) of message-only scorers: the deployed
classifier and the fine-tuned Kev runs (paper/kev_crossfit_eval.py output). Thresholds come from conformal risk control
on a random half of the messages and are tested on the other half, repeated over many splits:
  t_e: miss rate of the emergency route among call-911/988 messages <= alpha_e
  t_s: miss rate of (emergency or review) among safety-positive messages <= alpha_s
Reports the share of messages on each route and the realized miss rates. Prints numbers only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper.system1_eval import outcomes


def crossfit(X, y, fold):
    p = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def threshold(score, y, alpha):
    pos = np.sort(score[y == 1])
    k = int(np.floor(alpha * (len(pos) + 1) - 1))
    if k < 0:
        return None  # too few positives for this alpha
    return pos[k] if k < len(pos) else np.inf


def gate(se, ss, ye, ys, alpha_e, alpha_s, splits, seed):
    rng = np.random.default_rng(seed)
    n, res, undefined = len(ye), [], 0
    for _ in range(splits):
        idx = rng.permutation(n)
        cal, te = idx[: n // 2], idx[n // 2:]
        te_ = threshold(se[cal], ye[cal], alpha_e)
        # the safety threshold applies to messages not already on the emergency route
        ts_ = threshold(np.maximum(ss[cal], np.where(se[cal] >= (te_ if te_ is not None else np.inf), np.inf, -np.inf)),
                        ys[cal], alpha_s)
        if te_ is None or ts_ is None:
            undefined += 1
            continue
        em = se[te] >= te_
        rv = ~em & (ss[te] >= ts_)
        r = {"emergency_route": em.mean(), "review_route": rv.mean(), "send": 1 - em.mean() - rv.mean()}
        if ye[te].any():
            r["emergency_miss"] = 1 - em[ye[te] == 1].mean()
        if ys[te].any():
            r["safety_miss"] = 1 - (em | rv)[ys[te] == 1].mean()
        res.append(r)
    if not res:
        return {"undefined_splits": undefined}
    keys = sorted({k for r in res for k in r})
    return {**{k: round(float(np.mean([r[k] for r in res if k in r])), 3) for k in keys}, "undefined_splits": undefined}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--deployed", type=Path, required=True)
    ap.add_argument("--src", action="append", default=[], help="name=path to per-message score JSONL")
    ap.add_argument("--alphas", default="0.10:0.10,0.10:0.05,0.20:0.10")
    ap.add_argument("--splits", type=int, default=500)
    a = ap.parse_args()
    lab = {str(r["message_id"]): outcomes(r) for r in json.loads(a.cases.read_text())}
    fold_of = json.loads(a.folds.read_text())
    with a.deployed.open() as fh:
        dep = {r["message_id"]: float(r["score"]) for r in csv.DictReader(fh)}
    srcs = [{json.loads(x)["id"]: json.loads(x) for x in Path(p.split("=", 1)[1]).read_text().splitlines()} for p in a.src]
    ids = sorted(i for i in lab if i in fold_of and i in dep and all(i in s for s in srcs))
    fold = np.array([fold_of[i] for i in ids])
    ye = np.array([lab[i]["emergency"] for i in ids])
    ys = np.array([lab[i]["safety"] for i in ids])
    Xe = np.column_stack([[dep[i] for i in ids]] + [[s[i]["emergency"] for i in ids] for s in srcs])
    Xs = np.column_stack([[dep[i] for i in ids]] + [[s[i]["safety"] for i in ids] for s in srcs])
    se, ss = crossfit(Xe, ye, fold), crossfit(Xs, ys, fold)
    out = {"n": len(ids), "emergency_positives": int(ye.sum()), "safety_positives": int(ys.sum())}
    for pair in a.alphas.split(","):
        ae, as_ = (float(x) for x in pair.split(":"))
        out[f"alpha_e={ae},alpha_s={as_}"] = gate(se, ss, ye, ys, ae, as_, a.splits, 20261108)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
