"""Registered addendum analyses of the independent physician's workbook (https://osf.io/tnezd, sections 4 and 5).

Written before the workbook returned. Input: compass-pcp-data/fresh_set/external_rater/returned/external.xlsx (sheets
Part1_text_labels, Part2_reply_care, Part3_reply_ratings) with key_part1.jsonl, key_part2.jsonl and the A2 key.
Part 1: Cohen's kappa (safety-positive) A.V. vs S.B. and A.V. vs J.M. on the 150 texts; majority-of-three labels (S.B.
  and J.M. agree on safety-positive and call 911/988: S.B.'s label; otherwise A.V.'s), written to
  fresh_set/labels_majority.jsonl; hazard sensitivity and paired differences recomputed with the texts() machinery,
  escalation burden on S.B.'s no-hazard set.
Part 2 (Study D): classifier sensitivity, specificity and kappa for escalation (routine_followup or higher, and
  urgent_same_day or higher) against A.V., per reply source, stratum-weighted (population / sampled); corrected hazard
  sensitivity and escalation burden for every system, replacing each reply's classifier escalation by the estimated
  probability that A.V. labels it escalating, given source, classifier stratum and text safety-positive status (S.B.);
  routed texts and held replies stay escalated. Bootstrap: texts and, within strata, validation replies (2,000, seed 20261108).
Part 3: agreement with C.M. and R.B.; A.V.'s results by source; blinding check (share guessed correctly; Bang's index).
Out: paper2_final/derived/addendum_rater.json (aggregates only). Usage: python -m paper.addendum_rater_analysis
"""

import json
from pathlib import Path

import numpy as np

from compass_pcp.harness.cohort_a import label
from paper import supplementary_analyses as S
from paper.a2_rating_analysis import DISP, HARM, _kappa
from paper.sealed_comparators import URGENCY

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
F = D / "fresh_set"
X = F / "external_rater"
OUT = D / "paper2_final/derived/addendum_rater.json"
NB = 2000


def _rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _sheet(ws):
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:] if r[0] is not None and not str(r[0]).startswith("I rated")]


def _lab(r):
    return label({"detection_truth": r["hazard"], "hazard_category": r["category"], "action_truth": r["action"]})


def _norm(v):
    return None if v is None or str(v).strip() == "" else str(v).strip()


def part1(ws) -> tuple[dict, dict]:
    av = {}
    for r in _sheet(ws):
        h = _norm(r["hazard (0/1)"])
        av[r["sample_id"]] = {"sample_id": r["sample_id"], "hazard": None if h is None else int(float(h)),
                              "category": _norm(r["hazard_category"]), "action": _norm(r["action"]), "labeler": "external"}
    b = {r["sample_id"]: r for r in _rows(F / "labels_basu.jsonl")}
    m = {r["sample_id"]: r for r in _rows(F / "labels_morgan.jsonl")}
    ids = sorted(av)
    sp = lambda d, s: _lab(d[s])["safety_positive"]  # noqa: E731
    out = {"texts": len(ids), "missing_hazard": sum(av[s]["hazard"] is None for s in ids),
           "kappa_external_basu": _kappa([sp(av, s) for s in ids], [sp(b, s) for s in ids], (False, True)),
           "kappa_external_morgan": _kappa([sp(av, s) for s in ids], [sp(m, s) for s in ids], (False, True)),
           "positives": {"external": sum(sp(av, s) for s in ids), "basu": sum(sp(b, s) for s in ids),
                         "morgan": sum(sp(m, s) for s in ids)}}
    maj, decided = [], 0
    for s in sorted(b):
        lb, lm = _lab(b[s]), _lab(m[s])
        if lb["safety_positive"] == lm["safety_positive"] and lb["call_911"] == lm["call_911"]:
            maj.append({**b[s], "labeler": "majority", "hazard_for_burden": b[s]["hazard"]})
        else:
            assert s in av, f"disagreement {s} not in Part 1"
            decided += 1
            maj.append({**av[s], "labeler": "majority", "hazard_for_burden": b[s]["hazard"]})
    (F / "labels_majority.jsonl").write_text("".join(json.dumps(r) + "\n" for r in maj))
    out["disagreements_decided_by_external"] = decided
    return out, {r["sample_id"]: r for r in maj}


def majority_metrics(maj: dict) -> dict:
    basu, _, rep, paged, _ = S._texts_data()
    cw = {r["sample_id"]: r for r in S._rows(F / "crosswalk.jsonl")}
    lab = {}
    for s, r in maj.items():
        i = cw[s]["id"]
        lab[i] = {**_lab(r), "w": basu[i]["w"], "hazard": r["hazard_for_burden"] == 1, "action": str(r["action"]),
                  "stratum": cw[s]["stratum"]}
    E = S._per_repeat(rep, paged)
    for arm in ("amie_style", "mai_dxo_style", "v22"):
        E[f"{arm}+layer"] = {i: [True] * len(v) if paged[i] else [v[k][0] or v[k][2] for k in sorted(v)]
                             for (x, i), v in rep.items() if x == arm}
    ids = sorted(i for i in E["ours"] if i in lab)
    B = S._B(len(ids), np.random.default_rng(S.SEED))
    arms = ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22", "amie_style+layer", "mai_dxo_style+layer", "v22+layer"]
    return {"metrics": S._metrics({a: E[a] for a in arms}, lab, ids, B),
            "paired_diffs": S._diffs({a: E[a] for a in arms}, lab, ids, B, "ours", arms[1:])}


def part2(ws) -> dict:
    key = {k["item"]: k for k in _rows(X / "key_part2.jsonl")}
    av = {}
    for r in _sheet(ws):
        v = _norm(r[[h for h in r if str(h).startswith("advised_care")][0]])
        v = None if v is None else v.lower().replace(" ", "_").replace("-", "_")
        av[int(r["item"])] = v if v in URGENCY else None
    items = [dict(key[n], av=av.get(n)) for n in key if av.get(n) is not None]
    out = {"replies_labeled": len(items), "replies_unusable": len(key) - len(items)}
    cells = {}  # (arm, stratum) -> list of (av_routine, av_sameday), with population / sampled
    for it in items:
        c = cells.setdefault((it["arm"], it["stratum"]), {"pop": it["stratum_population"], "n": it["stratum_sampled"], "y": []})
        c["y"].append((URGENCY.index(it["av"]) >= 1, URGENCY.index(it["av"]) >= 2))

    def acc(arm, thr, sp_only=False):
        tp = fn = tn = fp = 0.0
        for (a, st), c in cells.items():
            if a != arm or (sp_only and not st.endswith("|sp")):
                continue
            w = c["pop"] / len(c["y"])
            pos = sum(y[thr] for y in c["y"])
            if st.startswith("esc"):
                tp, fp = tp + w * pos, fp + w * (len(c["y"]) - pos)
            else:
                fn, tn = fn + w * pos, tn + w * (len(c["y"]) - pos)
        return {"sensitivity": S._r(tp / (tp + fn), 3) if tp + fn else None,
                "specificity": S._r(tn / (tn + fp), 3) if tn + fp else None}

    out["classifier_vs_external"] = {
        arm: {"routine_or_more": acc(arm, 0), "same_day_or_more": acc(arm, 1),
              "routine_or_more_safety_positive_texts": acc(arm, 0, True),
              "kappa_unweighted_sample": _kappa([it["stratum"].startswith("esc") for it in items if it["arm"] == arm],
                                                [URGENCY.index(it["av"]) >= 1 for it in items if it["arm"] == arm], (False, True))}
        for arm in sorted({a for a, _ in cells})}
    # corrected hazard sensitivity and escalation burden
    basu, _, rep, paged, _ = S._texts_data()
    ids = sorted(i for (a, i) in rep if a == "raw_llm" and i in basu)
    w = np.array([basu[i]["w"] for i in ids])
    sp = np.array([basu[i]["safety_positive"] for i in ids])
    nh = np.array([not basu[i]["hazard"] for i in ids])
    rng = np.random.default_rng(S.SEED)
    systems = {"ours": ("raw_llm", True), "amie_style": ("amie_style", False), "mai_dxo_style": ("mai_dxo_style", False),
               "raw_llm": ("raw_llm", False), "v22": ("v22", False), "amie_style+layer": ("amie_style", True),
               "mai_dxo_style+layer": ("mai_dxo_style", True), "v22+layer": ("v22", True)}

    def phat(cell_draw, thr):
        return {k: (np.mean([y[thr] for y in ys]) if ys else np.nan) for k, ys in cell_draw.items()}

    def corrected(p, arm, layered, i):
        v = rep.get((arm, i), {})
        vals = []
        for k in sorted(v):
            e, _, unc = v[k]
            if layered and (paged[i] or unc):
                vals.append(1.0)
            else:
                vals.append(p.get((arm, ("esc" if e else "none") + ("|sp" if basu[i]["safety_positive"] else "|nonsp")), np.nan))
        return float(np.nanmean(vals)) if vals else np.nan

    def estimate(cell_draw, tix, thr):
        p = phat(cell_draw, thr)
        res = {}
        for name, (arm, layered) in systems.items():
            x = np.array([corrected(p, arm, layered, ids[j]) for j in tix])
            ww, ss, nn = w[tix], sp[tix], nh[tix]
            ok = ~np.isnan(x)
            res[name] = (float((x[ss & ok] * ww[ss & ok]).sum() / ww[ss & ok].sum()),
                         float(1000 * (x[nn & ok] * ww[nn & ok]).sum() / ww[nn & ok].sum()))
        return res

    base_cells = {k: c["y"] for k, c in cells.items()}
    full = np.arange(len(ids))
    corr = {}
    for thr, nm in ((0, "routine_or_more"), (1, "same_day_or_more")):
        point = estimate(base_cells, full, thr)
        boots = []
        for _ in range(NB):
            tix = rng.integers(0, len(ids), len(ids))
            cd = {k: [ys[j] for j in rng.integers(0, len(ys), len(ys))] for k, ys in base_cells.items()}
            boots.append(estimate(cd, tix, thr))
        res = {}
        for name in systems:
            sens = [b[name][0] for b in boots]
            bur = [b[name][1] for b in boots]
            res[name] = {"hazard_sensitivity": S._r(point[name][0], 3), "hazard_sensitivity_ci": S._pct_ci(sens, 1, 3),
                         "escalation_burden_per_1000": S._r(point[name][1]), "escalation_burden_ci": S._pct_ci(bur, 1)}
            if name != "ours":
                d = [b["ours"][0] - b[name][0] for b in boots]
                res[name]["ours_minus_this_sensitivity"] = S._r(point["ours"][0] - point[name][0], 3)
                res[name]["ours_minus_this_sensitivity_ci"] = S._pct_ci(d, 1, 3)
        corr[nm] = res
    out["corrected"] = corr
    return out


def part3(ws) -> dict:
    key = {(k["text_no"], k["reply_label"]): k for k in _rows(F / "a2_rating/key.jsonl")}
    av = {}
    for r in _sheet(ws):
        k = key[(int(r["text_no"]), r["reply_label"])]
        disp = [h for h in r if str(h).startswith("advised_disposition")][0]
        av[(k["sample_id"], k["arm"])] = {
            "accuracy": int(r["accuracy_1to5"]), "harm": _norm(r["harm_none_mild_moderate_severe"]),
            "omission": (_norm(r["critical_omission_yes_no"]) or "").lower() == "yes", "usefulness": int(r["usefulness_1to5"]),
            "disposition": _norm(r[disp]), "rank": int(r["rank_1_best"]),
            "guess": (_norm(r["source_guess_physician_ai_careteam"]) or "").lower()}
    prior = {(r["rater"], r["sample_id"], r["arm"]): r for r in _rows(F / "a2_rating/ratings.jsonl")}
    ms = lambda r: r["harm"] in ("moderate", "severe")  # noqa: E731
    out = {"replies": len(av)}
    for other in ("moher", "batniji"):
        both = [k for k in av if (other, *k) in prior]
        o = {k: prior[(other, *k)] for k in both}
        out[f"agreement_with_{other}"] = {
            "replies": len(both),
            "accuracy_quadratic": _kappa([av[k]["accuracy"] for k in both], [o[k]["accuracy"] for k in both], range(1, 6), True),
            "usefulness_quadratic": _kappa([av[k]["usefulness"] for k in both], [o[k]["usefulness"] for k in both], range(1, 6), True),
            "harm_4_quadratic": _kappa([av[k]["harm"] for k in both], [o[k]["harm"] for k in both], HARM, True),
            "moderate_or_severe": _kappa([ms(av[k]) for k in both], [ms(o[k]) for k in both], (False, True)),
            "critical_omission": _kappa([av[k]["omission"] for k in both], [o[k]["omission"] for k in both], (False, True)),
            "advised_care_quadratic": _kappa([av[k]["disposition"] for k in both], [o[k]["disposition"] for k in both], DISP, True)}
    page = {(r["sample_id"], r["arm"]): r["page_text"] for r in prior.values() if r["rater"] == "moher"}
    by = {}
    for grp, f in (("separated_alert_text", lambda k: k[1] == "ai" and page.get(k)),
                   ("separated_generated", lambda k: k[1] == "ai" and not page.get(k)),
                   ("physician", lambda k: k[1] == "physician"), ("amie", lambda k: k[1] == "amie"),
                   ("care_team", lambda k: k[1] == "care_team")):
        ks = [k for k in av if f(k)]
        by[grp] = {"replies": len(ks), "moderate_or_severe": sum(ms(av[k]) for k in ks),
                   "critical_omission": sum(av[k]["omission"] for k in ks),
                   "mean_accuracy": S._r(np.mean([av[k]["accuracy"] for k in ks]), 2) if ks else None}
    out["external_by_source"] = by
    truth = {"ai": "ai", "amie": "ai", "physician": "physician", "care_team": "care team"}
    blind = {}
    for src in ("physician", "ai", "care team"):
        ks = [k for k in av if truth[k[1]] == src]
        g = [av[k]["guess"].replace("careteam", "care team").replace("_", " ") for k in ks]
        g = ["ai" if x.startswith("ai") else x for x in g]
        right = sum(x == src for x in g)
        known = sum(x in ("physician", "ai", "care team") for x in g)
        blind[src] = {"replies": len(ks), "guessed_correctly": right, "guessed": known,
                      "bang_index": S._r((right - (known - right)) / len(ks), 3) if ks else None}
    out["blinding"] = blind
    return out


def main() -> None:
    import openpyxl

    wb = openpyxl.load_workbook(X / "returned/external.xlsx", read_only=True)
    p1, maj = part1(wb["Part1_text_labels"])
    res = {"registration": "https://osf.io/tnezd", "part1": p1, "majority_labels": majority_metrics(maj),
           "part2_classifier": part2(wb["Part2_reply_care"]), "part3_ratings": part3(wb["Part3_reply_ratings"])}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k != "majority_labels" else "written") for k, v in res.items()}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
