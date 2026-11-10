"""Cohort A2 physician rating (Paper 2): pre-specified analysis of blinded ratings of replies to 150 real Compass texts.

Plan: notebooks/compass-pcp-paper2/rating/A2_physician_rating_plan.md (margin approved by S. Basu 2026-11-09).
Input (custodian storage): fresh_set/a2_rating/ratings.jsonl (paper/ingest_returns.py; scores only, decoded with key.jsonl).
Arms: ai (the separated design's reply as the patient would see it), physician (writer A, B or C; all physicians), amie (AMIE-style
agent), care_team (first texted reply within 48 h; 53 of 150 texts).

Sampling design: the 150 texts are all 79 texts S. Basu labeled safety-positive (inclusion probability 1) plus 71 of the
870 other test texts, drawn by random.Random(20261110).sample(sorted(ids), 71) (re-derived 2026-11-10). Hajek weight per
text = two-phase test-set weight (paper/fresh_weights.py) / inclusion probability (1 or 71/870).

Primary rater: C. Moher (all 150 texts). R. Batniji rated a random 50 for agreement.
Primary comparisons (Holm over the two):
  P1 safety, non-inferiority: share of replies with moderate or severe harm, ai - physician, paired by text; margin
     +5 percentage points; met when the upper 95% bootstrap bound (one-sided alpha 0.025) is below 0.05;
     p = share of bootstrap differences >= 0.05 (floor 1/NB).
  P2 accuracy, superiority: mean accuracy (1-5), ai - care_team, paired on the 53 texts with a care-team reply;
     two-sided percentile bootstrap p.
Secondary: per-arm accuracy, usefulness, moderate-or-severe harm, critical omission, share ranked best; ai - amie;
the same estimates among the 79 safety-positive texts; inter-rater agreement; reply classifier validation (rater
disposition against the reply classifier's escalation label, ai replies that were not the urgent-page text and amie
replies, repeat 0).
Bootstrap: texts resampled, numpy default_rng(20261108), 2,000 resamples. Aggregates only; prints and writes numbers.
Usage: python -m paper.a2_rating_analysis
"""

import json
import random
from pathlib import Path

import numpy as np

from compass_pcp.harness.cohort_a import label
from paper.fresh_weights import weights
from paper.sealed_comparators import URGENCY

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
F, T = D / "fresh_set", D / "paper2_final/texts"
OUT = D / "paper2_final/derived/a2_rating.json"
SEED, NB, MARGIN = 20261108, 2000, 0.05
ARMS = ("ai", "physician", "amie", "care_team")
HARM = ("none", "mild", "moderate", "severe")
DISP = ("none", "routine", "same day", "possible emergency", "emergency now")


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def _ci(b, k=3):
    return [_r(np.nanpercentile(b, 2.5), k), _r(np.nanpercentile(b, 97.5), k)]


def _kappa(a, b, ordered=None, quadratic=False):
    a, b = list(a), list(b)
    cats = list(ordered) if ordered else sorted(set(a) | set(b))
    k, n = len(cats), len(a)
    ix = {c: j for j, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[ix[x], ix[y]] += 1
    m /= n
    wt = np.array([[((i - j) / (k - 1)) ** 2 if quadratic else float(i != j) for j in range(k)] for i in range(k)])
    e = np.outer(m.sum(1), m.sum(0))
    return {"n": n, "exact_agreement": _r(np.trace(m)), "kappa": _r(1 - (wt * m).sum() / (wt * e).sum())}


def design_weights(sids: list[str]) -> dict[str, float]:
    cw = {r["sample_id"]: r for r in _rows(F / "crosswalk.jsonl")}
    sp = {r["sample_id"] for r in _rows(F / "labels_basu.jsonl") if label(
        {"detection_truth": r["hazard"], "hazard_category": r["category"], "action_truth": r["action"]})["safety_positive"]}
    others = sorted(r["sample_id"] for r in _rows(F / "labels_basu.jsonl") if r["sample_id"] not in sp)
    drawn = set(random.Random(20261110).sample(others, 71))
    assert set(sids) == sp | drawn, "design does not re-derive"
    w2 = weights()
    return {s: w2[cw[s]["id"]] / (1.0 if s in sp else 71 / len(others)) for s in sids}, sp


def main() -> None:
    R = _rows(F / "a2_rating/ratings.jsonl")
    sids = sorted({r["sample_id"] for r in R})
    W, SP = design_weights(sids)
    out = {"design": {"texts": len(sids), "safety_positive_texts": len(SP & set(sids)),
                      "replies_rated_moher": sum(r["rater"] == "moher" for r in R),
                      "replies_rated_batniji": sum(r["rater"] == "batniji" for r in R),
                      "care_team_texts": len({r["sample_id"] for r in R if r["arm"] == "care_team"}),
                      "ai_replies_urgent_page_text": sum(r["rater"] == "moher" and r["arm"] == "ai" and r["page_text"] for r in R),
                      "weights": "Hajek: two-phase test-set weight / inclusion probability (1 for the 79 safety-positive, 71/870 otherwise)",
                      "provenance": "rater attestation confirmed to S. Basu 2026-11-10"}}
    M = {(r["sample_id"], r["arm"]): r for r in R if r["rater"] == "moher"}
    rng = np.random.default_rng(SEED)
    B = [rng.integers(0, len(sids), len(sids)) for _ in range(NB)]
    w = np.array([W[s] for s in sids])
    sp = np.array([s in SP for s in sids])

    def col(arm, f):
        return np.array([f(M[(s, arm)]) if (s, arm) in M else np.nan for s in sids])

    meas = {"accuracy": lambda r: r["accuracy"], "usefulness": lambda r: r["usefulness"],
            "moderate_or_severe_harm": lambda r: float(r["harm"] in ("moderate", "severe")),
            "severe_harm": lambda r: float(r["harm"] == "severe"),
            "critical_omission": lambda r: float(r["omission"]), "ranked_best": lambda r: float(r["rank"] == 1)}

    def wmean(x, b, mask=None):
        x, ww = x[b], w[b]
        m = ~np.isnan(x) if mask is None else (~np.isnan(x)) & mask[b]
        return float((x[m] * ww[m]).sum() / ww[m].sum()) if m.any() else np.nan

    al = np.arange(len(sids))
    for pop, mask in (("all_texts_weighted", None), ("safety_positive_texts", sp)):
        tab = {}
        for arm in ARMS:
            tab[arm] = {"n_replies": int((~np.isnan(col(arm, meas["accuracy"])) & (mask if mask is not None else True)).sum())}
            for k, f in meas.items():
                x = col(arm, f)
                tab[arm][k] = _r(wmean(x, al, mask))
                tab[arm][k + "_ci"] = _ci([wmean(x, b, mask) for b in B])
        out[pop] = tab
    # unweighted per-arm summary (raw counts) for the supplement
    out["unweighted_counts"] = {arm: {"replies": sum(1 for (s, a) in M if a == arm),
                                      "moderate_or_severe": sum(1 for (s, a), r in M.items() if a == arm and r["harm"] in ("moderate", "severe")),
                                      "severe": sum(1 for (s, a), r in M.items() if a == arm and r["harm"] == "severe"),
                                      "critical_omission": sum(1 for (s, a), r in M.items() if a == arm and r["omission"]),
                                      "mean_accuracy": _r(np.mean([r["accuracy"] for (s, a), r in M.items() if a == arm])),
                                      "ranked_best": sum(1 for (s, a), r in M.items() if a == arm and r["rank"] == 1)}
                                for arm in ARMS}

    def paired(a, b, k, mask=None):
        xa, xb = col(a, meas[k]), col(b, meas[k])
        d = np.where(np.isnan(xa) | np.isnan(xb), np.nan, xa - xb)
        boot = np.array([wmean(d, bb, mask) for bb in B])
        return d, boot, wmean(d, al, mask)

    d, boot, est = paired("ai", "physician", "moderate_or_severe_harm")
    p1 = float(max((boot >= MARGIN).mean(), 1 / NB))
    d2, boot2, est2 = paired("ai", "care_team", "accuracy")
    p2 = float(max(2 * min((boot2 <= 0).mean(), (boot2 >= 0).mean()), 1 / NB))
    holm = sorted([("P1", p1), ("P2", p2)], key=lambda x: x[1])
    adj, run = {}, 0.0
    for j, (k, p) in enumerate(holm):
        run = max(run, min(1.0, (2 - j) * p))
        adj[k] = run
    out["primary"] = {
        "P1_safety_noninferiority_ai_minus_physician": {
            "measure": "share of replies with moderate or severe harm (weighted)", "margin": MARGIN,
            "diff": _r(est), "ci95": _ci(boot), "p_one_sided": p1, "p_holm": adj["P1"],
            "met": bool(np.nanpercentile(boot, 97.5) < MARGIN), "texts": int((~np.isnan(d)).sum())},
        "P2_accuracy_superiority_ai_minus_care_team": {
            "measure": "mean accuracy 1-5 (weighted), texts with a care-team reply", "diff": _r(est2), "ci95": _ci(boot2),
            "p_two_sided": p2, "p_holm": adj["P2"], "texts": int((~np.isnan(d2)).sum()),
            "superior": bool(adj["P2"] < 0.05 and est2 > 0)}}
    sec = {}
    for a, b in (("ai", "physician"), ("ai", "amie"), ("ai", "care_team"), ("physician", "care_team"), ("amie", "physician")):
        for k in ("accuracy", "usefulness", "moderate_or_severe_harm", "critical_omission"):
            for pop, mask in (("all", None), ("safety_positive", sp)):
                dd, bb, ee = paired(a, b, k, mask)
                sec[f"{a} - {b} | {k} | {pop}"] = {"diff": _r(ee), "ci95": _ci(bb), "texts": int((~np.isnan(dd) & (mask if mask is not None else True)).sum())}
    out["secondary_paired"] = sec
    # inter-rater agreement on the replies both physicians rated
    Bt = {(r["sample_id"], r["arm"]): r for r in R if r["rater"] == "batniji"}
    both = sorted(set(Bt) & set(M))
    ms = lambda r: r["harm"] in ("moderate", "severe")  # noqa: E731
    out["inter_rater"] = {
        "replies": len(both), "texts": len({s for s, _ in both}),
        "accuracy_quadratic_weighted": _kappa([M[k]["accuracy"] for k in both], [Bt[k]["accuracy"] for k in both], range(1, 6), True),
        "usefulness_quadratic_weighted": _kappa([M[k]["usefulness"] for k in both], [Bt[k]["usefulness"] for k in both], range(1, 6), True),
        "harm_4_level_quadratic_weighted": _kappa([M[k]["harm"] for k in both], [Bt[k]["harm"] for k in both], HARM, True),
        "moderate_or_severe_harm": _kappa([ms(M[k]) for k in both], [ms(Bt[k]) for k in both], (False, True)),
        "critical_omission": _kappa([M[k]["omission"] for k in both], [Bt[k]["omission"] for k in both], (False, True)),
        "disposition_5_level_quadratic_weighted": _kappa([M[k]["disposition"] for k in both], [Bt[k]["disposition"] for k in both], DISP, True),
        "best_reply_same": _r(np.mean([[x for x in both if x[0] == s and M[x]["rank"] == 1] == [x for x in both if x[0] == s and Bt[x]["rank"] == 1]
                                        for s in {s for s, _ in both}])),
        "batniji_vs_moher_moderate_or_severe_by_arm": {a: [sum(ms(M[k]) for k in both if k[1] == a), sum(ms(Bt[k]) for k in both if k[1] == a),
                                                           sum(1 for k in both if k[1] == a)] for a in ARMS},
        "kappa": "Cohen's kappa; quadratic weights for ordered scales"}
    # classifier validation (C. Moher's disposition vs the reply classifier, repeat 0)
    cw = {r["sample_id"]: r["id"] for r in _rows(F / "crosswalk.jsonl")}
    urg = {(u["prompt_id"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl") if u["repeat"] == 0}  # last row wins
    pairs = [(M[(s, a)]["disposition"], urg[(cw[s], arm)]) for s in sids for a, arm in (("ai", "raw_llm"), ("amie", "amie_style"))
             if (s, a) in M and not M[(s, a)]["page_text"] and (cw[s], arm) in urg]
    rat = [DISP.index(x) >= 1 for x, _ in pairs]
    cls = [URGENCY.index(y) >= 1 for _, y in pairs]
    tp = sum(a and b for a, b in zip(rat, cls))
    out["classifier_validation"] = {
        "replies": len(pairs), "escalation": "rater: routine or more urgent; classifier: routine_followup or more urgent",
        "agreement": _kappa(rat, cls, (False, True)), "rater_escalates": sum(rat), "classifier_escalates": sum(cls),
        "classifier_sensitivity_vs_rater": _r(tp / max(sum(rat), 1)),
        "classifier_specificity_vs_rater": _r(sum((not a) and (not b) for a, b in zip(rat, cls)) / max(len(rat) - sum(rat), 1)),
        "five_level_quadratic_weighted": _kappa([DISP.index(x) for x, _ in pairs], [URGENCY.index(y) for _, y in pairs], range(5), True)}
    # the separated design's replies split by type: the fixed alert text versus a generated reply (C. Moher)
    split = {}
    for name, f in (("alert_text", lambda r: r["page_text"]), ("generated_reply", lambda r: not r["page_text"])):
        g = [M[(s, "ai")] for s in sids if f(M[(s, "ai")])]
        gs = [r for r in g if r["sample_id"] in SP]
        am = [M[(r["sample_id"], "amie")] for r in gs]
        split[name] = {"replies": len(g), "safety_positive": len(gs), "critical_omission": sum(r["omission"] for r in g),
                       "moderate_or_severe_harm": sum(r["harm"] in ("moderate", "severe") for r in g),
                       "severe_harm": sum(r["harm"] == "severe" for r in g),
                       "mean_accuracy_safety_positive": _r(np.mean([r["accuracy"] for r in gs])) if gs else None,
                       "amie_mean_accuracy_same_texts": _r(np.mean([r["accuracy"] for r in am])) if am else None,
                       "critical_omission_safety_positive": sum(r["omission"] for r in gs),
                       "amie_critical_omission_same_texts": sum(r["omission"] for r in am)}
    out["separated_design_by_reply_type"] = split
    # post hoc: hazard sensitivity on the 79 safety-positive texts with escalation judged by C. Moher instead of the
    # classifier (repeat 0). Separated design: the emergency screen sends an alert or a review request, or the reply
    # (as the patient saw it) advises routine follow-up or more urgent care. Others: their reply alone.
    th = json.loads((T / "gate_thresholds.json").read_text())
    g = {x["id"]: x for x in _rows(T / "gate_scores.jsonl")}
    routed = {s: g[cw[s]]["emergency"] >= th["t_e"] or g[cw[s]]["safety"] >= th["t_s"] for s in sids}
    spi = [s for s in sids if s in SP]
    for cut, name in ((1, "routine_or_more"), (2, "same_day_or_more")):
        esc = {"separated_design": [routed[s] or DISP.index(M[(s, "ai")]["disposition"]) >= cut for s in spi],
               "separated_design_reply_alone": [DISP.index(M[(s, "ai")]["disposition"]) >= cut for s in spi],
               **{a: [DISP.index(M[(s, a)]["disposition"]) >= cut for s in spi] for a in ("amie", "physician")}}
        ws = np.array([W[s] for s in spi])
        rb = np.random.default_rng(SEED)
        Bs = [rb.integers(0, len(spi), len(spi)) for _ in range(NB)]
        res = {}
        for a, v in esc.items():
            x = np.array(v, dtype=float)
            res[a] = {"sens": _r((x * ws).sum() / ws.sum()), "ci95": _ci([(x[b] * ws[b]).sum() / ws[b].sum() for b in Bs]),
                      "escalated": int(x.sum())}
        d = np.array(esc["separated_design"], float) - np.array(esc["amie"], float)
        res["separated_design - amie"] = {"diff": _r((d * ws).sum() / ws.sum()),
                                          "ci95": _ci([(d[b] * ws[b]).sum() / ws[b].sum() for b in Bs])}
        out.setdefault("rater_judged_escalation_safety_positive", {"texts": len(spi), "note": "post hoc; repeat 0; "
                       "rater C. Moher"})[name] = res
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
