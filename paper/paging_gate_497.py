"""Paging-gate threshold on the 497-text emergency-enriched care-team SMS set (development data only).

The paging gate scores each inbound text from the message alone. This script scores the 497 physician-labeled texts in
compass-pcp-data/lighthouse_set (S. Basu labels) with the five Kev-9B fine-tunes on physician labels
(compass-phys9b-f0..f4, trained on the 2,000 development messages; message-only state) and sets conformal thresholds.

Steps (custodian storage only; nothing but counts and aggregates is printed or written to the report):
1. records: Kev records for the 497 texts, message-only state, the same three questions as paper/kev_physician_folds.py.
   Every record carries the same placeholder labels, so no physician label reaches Kev; the analysis reads labels only
   from labels_basu.jsonl.
     python -m paper.paging_gate_497 records --out <D>/paper2_dev/paging_gate_497/work
   Then fit the states to Kev's context and stage one folder per fold model with that fold's own calibration file (the
   temperature refit on it reproduces the temperature of the original run):
     <kev>/.venv/bin/python paper/fit_records.py --src <D>/paper2_dev/paging_gate_497/work/raw --out <D>/paper2_dev/paging_gate_497/work/fit
     for k in 0..4: <D>/paper2_dev/paging_gate_497/work/eval/fold$k/{development.jsonl <- fit/development.jsonl,
                                                    calibration.jsonl <- system1/kev_phys_fit/fold$k/calibration.jsonl}
     KEV_APP_NAME=kev-compass-train modal run kev_modal.py::evaluate --data eval/fold$k --name compass-lh497-phys9b-f$k
                                                                    --run compass-phys9b-f$k
2. analyze: per-text Kev scores (mean of the five fold models; each model is out of sample for every text), AUROC
   unweighted and weighted by 1/p_inclusion (crosswalk.jsonl), and conformal risk control (CRC) of the miss rate over
   random 50/50 calibration/test splits at alpha 0.05 and 0.10. Scores:
     kev_p911     Kev P(action = Call 911/988) (emergency) or P(action >= Contact Doctor) (safety); a one-feature
                  stack fit on the 2,000 development messages is a monotone map of this score, so it has the same
                  AUROC and CRC result
     stack_2000   logistic stack of the three Kev heads (P911, P>=Contact Doctor, P hazard) fit on the out-of-fold
                  Kev-9B scores of the 2,000 development messages (as paper/safety_gate.py), applied to the 497
     stack_cf497  the same stack cross-fitted on the 497 (5 folds, stratified on the outcome)
   Outcomes (physician labels, labels_basu.jsonl; missing labels are never filled):
     emergency    action = Call 911/988 (texts with hazard 0 and a blank action are not emergencies)
     safety       hazard = 1 and action in {Contact Doctor, Call 911/988}
     safety_s1    system1_eval definition: (hazard = 1 and category != other_hazard) or Call 911/988; texts with
                  hazard 1, a blank category and no 911/988 action are undetermined and dropped
   The thresholds use the enriched sample (each calibration-half positive counts once). Each test half reports the
   realized miss rate and the share of texts paged, unweighted and weighted to the care-team SMS pool. A weighted CRC
   rule (calibration positives weighted by 1/p_inclusion, test-point weight bounded by the largest positive weight) is
   also reported.
     python -m paper.paging_gate_497 analyze (defaults: --out <D>/paper2_dev/paging_gate_497/work --report <D>/paper2_dev/paging_gate_497)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper.kev_physician_folds import QUESTIONS
from paper.system1_eval import outcomes as dev_outcomes
from paper.system1_questions import state

DATA = Path.home() / "waymark-local/packaging/compass-pcp-data"
LH = DATA / "lighthouse_set"
CASES = Path.home() / "waymark-local/notebooks/compass-ai-pcp/golden_set/data/realworld_cases_n2000_clean.json"
DEV_KEV = DATA / "paper2_dev/score_ensemble/kev_ft_phys9b_msg.jsonl"
PLACEHOLDER = {"hazard": False, "action": "None", "category": "none"}
HEADS = ("emergency", "safety", "any")
SEED, SPLITS, ALPHAS = 20261108, 500, (0.05, 0.10)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def write_records(out: Path) -> None:
    lab = {r["sample_id"] for r in _jsonl(LH / "labels_basu.jsonl")}
    cw = {r["sample_id"]: r["id"] for r in _jsonl(LH / "crosswalk.jsonl")}
    need = {cw[s] for s in lab}
    msg = {}
    with (LH / "pool.jsonl").open() as fh:
        for x in fh:
            d = json.loads(x)
            if d["id"] in need:
                msg[d["id"]] = d["message"]
    recs = []
    for s in sorted(lab):
        qs = {q: {**QUESTIONS[q], "label": PLACEHOLDER[q]} for q in ("hazard", "action", "category")}
        recs.append({"state": state(None, msg[cw[s]]), "questions": qs, "_meta": {"id": f"{s}|msg"}})
    p = out / "raw" / "development.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    p.chmod(0o600)
    print(json.dumps({"records": len(recs)}))


# --- scores ----------------------------------------------------------------------------------------------------------

def kev_scores(rows_paths: list[Path]) -> tuple[dict, list[dict]]:
    """{sample_id: {head: mean over fold models}} and the per-model score dicts (clean variant rows only)."""
    per_model = []
    for p in rows_paths:
        by = {}
        for r in json.loads(p.read_text()):
            if r.get("variant", "clean") != "clean":
                continue
            keys = [str(k).lower() if isinstance(k, bool) else str(k) for k in r["keys"]]
            by.setdefault(r["id"].split("|")[0], {})[r["question"]] = dict(zip(keys, r["p"]))
        per_model.append({i: {"emergency": q["action"].get("Call 911/988", 0.0),
                              "safety": q["action"].get("Contact Doctor", 0.0) + q["action"].get("Call 911/988", 0.0),
                              "any": q["hazard"].get("true", 0.0)} for i, q in by.items()})
    ids = set.intersection(*[set(m) for m in per_model])
    mean = {i: {h: float(np.mean([m[i][h] for m in per_model])) for h in HEADS} for i in ids}
    return mean, per_model


def labels_497() -> tuple[list[str], dict, np.ndarray]:
    lab = {r["sample_id"]: r for r in _jsonl(LH / "labels_basu.jsonl")}
    cw = {r["sample_id"]: r for r in _jsonl(LH / "crosswalk.jsonl")}
    ids = sorted(lab)
    y = {"emergency": [], "safety": [], "safety_s1": []}
    for s in ids:
        r = lab[s]
        hz, act, cat = int(r["hazard"]), r["action"], r["category"]
        em = int(act == "Call 911/988")
        y["emergency"].append(em)
        y["safety"].append(int(hz == 1 and act in ("Contact Doctor", "Call 911/988")))
        if em or hz == 0:
            y["safety_s1"].append(em)
        elif cat in (None, ""):
            y["safety_s1"].append(-1)  # undetermined: hazard without a category
        else:
            y["safety_s1"].append(int(cat != "other_hazard"))
    w = np.array([1.0 / cw[s]["p_inclusion"] for s in ids])
    return ids, {k: np.array(v) for k, v in y.items()}, w


def head_for(outcome: str) -> str:
    return "emergency" if outcome == "emergency" else "safety"


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def stack_2000(X497: np.ndarray, outcome: str) -> np.ndarray:
    dev = {str(r["message_id"]): dev_outcomes(r) for r in json.loads(CASES.read_text())}
    kev = {d["id"]: d for d in _jsonl(DEV_KEV)}
    ids = sorted(i for i in dev if i in kev)
    X = np.array([[kev[i][h] for h in HEADS] for i in ids])
    y = np.array([dev[i]["emergency" if outcome == "emergency" else "safety"] for i in ids])
    return lr().fit(X, y).predict_proba(X497)[:, 1]


def stack_crossfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        p[te] = lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


# --- metrics ---------------------------------------------------------------------------------------------------------

def auroc_ci(s, y, w, rng, reps=1000):
    est = {"auroc": roc_auc_score(y, s), "auroc_weighted": roc_auc_score(y, s, sample_weight=w)}
    boot = {k: [] for k in est}
    n = len(y)
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        boot["auroc"].append(roc_auc_score(y[i], s[i]))
        boot["auroc_weighted"].append(roc_auc_score(y[i], s[i], sample_weight=w[i]))
    return {k: {"est": round(float(v), 3), "ci95": [round(float(np.percentile(boot[k], 2.5)), 3),
                                                   round(float(np.percentile(boot[k], 97.5)), 3)]} for k, v in est.items()}


def t_unweighted(s, y, alpha):
    pos = np.sort(s[y == 1])
    k = int(np.floor(alpha * (len(pos) + 1) - 1))
    if k < 0:
        return None
    return pos[k] if k < len(pos) else np.inf


def t_weighted(s, y, w, alpha):
    """Largest t with (sum of positive weights below t + w_max) / (total positive weight + w_max) <= alpha."""
    sp, wp = s[y == 1], w[y == 1]
    o = np.argsort(sp)
    sp, wp = sp[o], wp[o]
    wmax, tot = wp.max(), wp.sum()
    below = np.concatenate([[0.0], np.cumsum(wp)])  # weight strictly below sp[j] (ties broken by order)
    ok = [j for j in range(len(sp)) if (below[j] + wmax) / (tot + wmax) <= alpha]
    if not ok:
        return None  # no threshold meets alpha: page every text
    return sp[max(ok)]


def summarize(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 3), "p05": round(float(np.percentile(a, 5)), 3),
            "p95": round(float(np.percentile(a, 95)), 3)}


def crc(s, y, w, alpha, rng, rule):
    n = len(y)
    out = {k: [] for k in ("threshold", "miss", "paged", "miss_w", "paged_w", "miss_gt_alpha")}
    undefined = 0
    for _ in range(SPLITS):
        idx = rng.permutation(n)
        cal, te = idx[: n // 2], idx[n // 2:]
        t = t_unweighted(s[cal], y[cal], alpha) if rule == "enriched" else t_weighted(s[cal], y[cal], w[cal], alpha)
        if t is None:
            undefined += 1
            t = -np.inf  # page everything
        r = s[te] >= t
        yt, wt = y[te], w[te]
        out["threshold"].append(float(t) if np.isfinite(t) else (0.0 if t < 0 else 1.0))
        out["paged"].append(r.mean())
        out["paged_w"].append((wt * r).sum() / wt.sum())
        if (yt == 1).any():
            m = 1 - r[yt == 1].mean()
            out["miss"].append(m)
            out["miss_gt_alpha"].append(float(m > alpha))
            out["miss_w"].append(1 - (wt * r)[yt == 1].sum() / wt[yt == 1].sum())
    res = {k: summarize(v) for k, v in out.items() if k != "miss_gt_alpha"}
    res["share_splits_miss_above_alpha"] = round(float(np.mean(out["miss_gt_alpha"])), 3)
    res["splits"], res["undefined_splits_paged_all"] = SPLITS, undefined
    return res


def full_sample(s, y, w, alpha):
    """Threshold from all labeled texts (the value to deploy) and its in-sample paged share and miss."""
    t = t_unweighted(s, y, alpha)
    r = s >= t
    return {"threshold": round(float(t), 4), "calibration_positives": int(y.sum()),
            "paged": round(float(r.mean()), 3), "paged_w": round(float((w * r).sum() / w.sum()), 3),
            "miss": round(float(1 - r[y == 1].mean()), 3),
            "miss_w": round(float(1 - (w * r)[y == 1].sum() / w[y == 1].sum()), 3)}


def analyze(out: Path, report: Path) -> None:
    rows = sorted(out.glob("runs/compass-lh497-phys9b-f*/development/rows.json"))
    if len(rows) != 5:
        raise SystemExit(f"expected 5 fold-model rows.json files, found {len(rows)}")
    mean, per_model = kev_scores(rows)
    ids, Y, w = labels_497()
    missing = [i for i in ids if i not in mean]
    if missing:
        raise SystemExit(f"{len(missing)} labeled texts have no Kev score")
    X = np.array([[mean[i][h] for h in HEADS] for i in ids])
    rng = np.random.default_rng(SEED)
    res = {"n": len(ids), "pool_texts_weighted_total": round(float(w.sum()), 1), "kev_runs": [p.parts[-3] for p in rows],
           "outcomes": {}}
    for o in ("emergency", "safety", "safety_s1"):
        keep = Y[o] >= 0
        y, ww, Xo = Y[o][keep], w[keep], X[keep]
        h = HEADS.index(head_for(o))
        scores = {"kev_p911" if o == "emergency" else "kev_p_contact_doctor_or_911": Xo[:, h],
                  "stack_2000": stack_2000(Xo, o), "stack_cf497": stack_crossfit(Xo, y)}
        po = {"n": int(keep.sum()), "positives": int(y.sum()),
              "prevalence_weighted": round(float((ww * y).sum() / ww.sum()), 4),
              "largest_positive_weight_share": round(float(ww[y == 1].max() / ww[y == 1].sum()), 3),
              "per_model_auroc_kev": [round(roc_auc_score(y, np.array([m[i][head_for(o)] for i, k in zip(ids, keep) if k])), 3)
                                      for m in per_model],
              "scores": {}}
        for name, s in scores.items():
            d = auroc_ci(s, y, ww, rng)
            for a in ALPHAS:
                d[f"crc_{a}"] = crc(s, y, ww, a, rng, "enriched")
                d[f"crc_weighted_rule_{a}"] = crc(s, y, ww, a, rng, "weighted")
                d[f"full_sample_{a}"] = full_sample(s, y, ww, a)
            po["scores"][name] = d
        res["outcomes"][o] = po
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.json").write_text(json.dumps(res, indent=1))
    (report / "report.md").write_text(markdown(res))
    print(markdown(res))


def markdown(res: dict) -> str:
    L = [f"# Paging gate on the 497-text emergency-enriched set (development; {res['n']} texts; Kev runs "
         f"{', '.join(res['kev_runs'])})", "",
         "Miss = share of positive texts not paged on the test half; paged = share of all texts paged. Enriched = the "
         "labeled sample as drawn; weighted = 1/p_inclusion, to the care-team SMS pool "
         f"({res['pool_texts_weighted_total']:,.0f} texts). Mean over random 50/50 splits (5th-95th percentile).", ""]
    for o, po in res["outcomes"].items():
        L += [f"## {o}: {po['positives']} positives of {po['n']}; weighted prevalence {100 * po['prevalence_weighted']:.2f}%; "
              f"largest single positive = {100 * po['largest_positive_weight_share']:.0f}% of weighted positive mass", "",
              f"Kev-9B fold models alone, AUROC: {', '.join(str(x) for x in po['per_model_auroc_kev'])}", "",
              "| Score | AUROC (95% CI) | Weighted AUROC (95% CI) | alpha | Threshold (all 497) | Miss | Paged | "
              "Miss, weighted | Paged, weighted | Splits with miss > alpha |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in po["scores"].items():
            for a in ALPHAS:
                c, f = d[f"crc_{a}"], d[f"full_sample_{a}"]
                fmt = lambda x: f"{100 * x['mean']:.1f}% ({100 * x['p05']:.1f}-{100 * x['p95']:.1f})"
                L.append(f"| {name} | {d['auroc']['est']:.3f} ({d['auroc']['ci95'][0]:.3f}-{d['auroc']['ci95'][1]:.3f}) | "
                         f"{d['auroc_weighted']['est']:.3f} ({d['auroc_weighted']['ci95'][0]:.3f}-"
                         f"{d['auroc_weighted']['ci95'][1]:.3f}) | {a:.2f} | {f['threshold']:.4f} | {fmt(c['miss'])} | "
                         f"{fmt(c['paged'])} | {fmt(c['miss_w'])} | {fmt(c['paged_w'])} | "
                         f"{100 * c['share_splits_miss_above_alpha']:.0f}% |")
        L += ["", "Weighted CRC rule (positives weighted by 1/p_inclusion):", ""]
        for name, d in po["scores"].items():
            for a in ALPHAS:
                c = d[f"crc_weighted_rule_{a}"]
                L.append(f"- {name}, alpha {a:.2f}: no threshold met alpha on {c['undefined_splits_paged_all']} of "
                         f"{c['splits']} splits (page all); miss weighted {100 * c['miss_w']['mean']:.1f}%, paged weighted "
                         f"{100 * c['paged_w']['mean']:.1f}%, paged enriched {100 * c['paged']['mean']:.1f}%")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["records", "analyze"])
    ap.add_argument("--out", type=Path, default=DATA / "paper2_dev/paging_gate_497/work")
    ap.add_argument("--report", type=Path, default=DATA / "paper2_dev/paging_gate_497")
    a = ap.parse_args()
    if a.step == "records":
        write_records(a.out)
    else:
        analyze(a.out, a.report)


if __name__ == "__main__":
    main()
