"""Two-axis triage on the 2,000 development messages: hazard now (from the message) and risk soon (from the chart).

Inputs: physician labels (cases), chart summaries (chart_context.jsonl), 30-day ADT outcomes (outcomes_30d.jsonl), and
message scores (the deployed classifier CSV and any per-message score JSONL with keys any/safety/doctor/emergency, such
as the out-of-fold fine-tuned Kev files from paper/kev_crossfit_eval.py). Chart features are counts parsed from the
chart text: age, ED visits and admissions in the past 12 months, days since the last discharge, number of claims
diagnoses, number of medicines, number of open tasks. No text leaves this script; it prints numbers only.

For each outcome (the four physician hazard outcomes and 30-day acute care) it fits a logistic regression on three
feature sets: message score only, chart only, and both; cross-fitted on the same five folds as the Kev fine-tunes
(folds.json), so every prediction is out of fold. Reports AUROC and conformal risk control (paper/system1_eval.crc).
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper.system1_eval import crc, outcomes

HAZ = ("safety", "doctor", "any", "emergency")


def chart_features(chart: str | None) -> list[float]:
    age = ed = ip = ndx = nmed = ntask = 0.0
    since = 365.0
    has = float(bool(chart))
    for line in (chart or "").splitlines():
        head, _, rest = line.partition(":")
        if head == "Patient":
            m = re.search(r"\b(\d{1,3})\b", rest)
            age = float(m.group(1)) if m else 0.0
        elif head == "Past 12 months":
            m = re.search(r"(\d+) emergency department visits, (\d+) hospital admissions", rest)
            if m:
                ed, ip = float(m.group(1)), float(m.group(2))
            m = re.search(r"last discharge (\d+) days ago", rest)
            if m:
                since = min(float(m.group(1)), 365.0)
        elif head.startswith("Diagnoses"):
            ndx = float(len([x for x in rest.split(";") if x.strip()]))
        elif head.startswith("Medicines"):
            nmed = float(len([x for x in rest.split(";") if x.strip()]))
        elif head.startswith("Open tasks"):
            m = re.search(r"(\d+)", rest)
            ntask = float(m.group(1)) if m else 0.0
    return [has, age, np.log1p(ed), np.log1p(ip), since / 365.0, np.log1p(ndx), np.log1p(nmed), np.log1p(ntask)]


def crossfit(X, y, fold):
    p = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--deployed", type=Path)
    ap.add_argument("--src", action="append", default=[], help="name=path to a per-message score JSONL")
    ap.add_argument("--splits", type=int, default=300)
    a = ap.parse_args()
    lab = {str(r["message_id"]): outcomes(r) for r in json.loads(a.cases.read_text())}
    for line in a.outcomes.read_text().splitlines():
        r = json.loads(line)
        if r["id"] in lab:
            lab[r["id"]]["acute30"] = r["acute30"]
    charts = {json.loads(x)["id"]: json.loads(x)["chart"] for x in a.chart.read_text().splitlines()}
    fold_of = json.loads(a.folds.read_text())
    msg = {}
    if a.deployed:
        with a.deployed.open() as fh:
            d = {r["message_id"]: float(r["score"]) for r in csv.DictReader(fh)}
        msg["deployed"] = {i: {o: d[i] for o in HAZ} for i in d}
    for s in a.src:
        name, path = s.split("=", 1)
        msg[name] = {json.loads(x)["id"]: json.loads(x) for x in Path(path).read_text().splitlines()}
    ids = sorted(i for i in lab if "acute30" in lab[i] and i in fold_of and all(i in m for m in msg.values()))
    fold = np.array([fold_of[i] for i in ids])
    C = np.array([chart_features(charts.get(i)) for i in ids])
    res = {"n": len(ids)}
    for o in HAZ + ("acute30",):
        y = np.array([lab[i][o] for i in ids])
        sets = {"chart": C}
        for name, m in msg.items():
            key = o if o in HAZ else "safety"
            S = np.array([[m[i][key]] for i in ids])
            sets[name] = S
            sets[f"{name}+chart"] = np.hstack([S, C])
        if len(msg) > 1:
            key = o if o in HAZ else "safety"
            allS = np.hstack([np.array([[m[i][key]] for i in ids]) for m in msg.values()])
            sets["all_messages"] = allS
            sets["all_messages+chart"] = np.hstack([allS, C])
        res[o] = {"positives": int(y.sum())}
        for name, X in sets.items():
            p = crossfit(X, y, fold)
            rng = np.random.default_rng(20261107)
            res[o][name] = {"auroc": round(roc_auc_score(y, p), 3),
                            **{f"crc_{al}": crc(p, y, al, a.splits, rng) for al in (0.01, 0.05, 0.10)}}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
