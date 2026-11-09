"""Supplementary analyses for the Paper 2 manuscript from stored outputs (no model or API calls; aggregates only).

Every output holds aggregates only: no message text, reply text, message id or task id is written or printed.
Deterministic: every bootstrap uses numpy default_rng(20261108) and 2,000 resamples.
Writes JSON and CSV to compass-pcp-data/paper2_final/derived/ (README.md there describes each file).

Sections (function -> file):
  hb_test()      -> hbpro_test_supplement.json   HealthBench Professional test split (350 tasks)
  hb_dev()       -> hbpro_dev_g61_chains.json    development split (175 tasks), GPT-6.1 Sol comparator chains
  texts()        -> texts_supplement.json        fresh real-text test set (949 texts), final policy
  paging_gate()  -> paging_gate_table.json/.csv  paging gate development results (497 texts)
  validate()     -> validation.json              re-derivation of the overlapping values in the existing reports

Final real-text policy (approved): the reply of the raw_llm writer is held only when it holds an unconditional emergency
instruction, paper.disposition_check.check_unconditional (conditional safety-net lines are allowed). "Ours" escalates a
text in a repeat when the writer's reply advises routine follow-up or more urgent care or is held; every repeat is
escalated when the paging gate pages (emergency >= t_e or safety >= t_s). This reproduces the point estimates of texts/report_policy_unconditional.json.

Usage: python -m paper.supplementary_analyses
"""

import csv
import json
from pathlib import Path

from paper.fresh_weights import weights as _weights

_W2 = _weights()

import numpy as np
from scipy import stats

from compass_pcp.harness.cohort_a import label
from paper import hbpro
from paper.disposition_check import check, check_unconditional
from paper.sealed_comparators import URGENCY

SEED, NB = 20261108, 2000
D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
OUT = D / "paper2_final/derived"
TEXTS = D / "paper2_final/texts"
OURS_HB, OURS_HB_DEV = "compass_pro_full", "compass_pro_edit"
HB_COMPS = ["amie_style", "mai_dxo_style", "gpt61sol", "opus55", "physician"]  # compare.txt order (RNG order matters)
WRITER = "raw_llm"


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _r(x, k=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def _pct_ci(boot, scale=100.0, k=1):
    return [_r(np.nanpercentile(boot, 2.5) * scale, k), _r(np.nanpercentile(boot, 97.5) * scale, k)]


def _boot_p(boot):
    """Two-sided percentile bootstrap p for a difference of zero; floor 1/NB (reported as an upper bound)."""
    b = np.asarray(boot)
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return {"p": float(max(p, 1 / len(b))), "is_upper_bound": bool(p < 1 / len(b))}


def _holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    m, out, run = len(keys), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * ps[k]))
        out[k] = run
    return out


# ---------------------------------------------------------------- HealthBench Professional

def _hb_scores(out: Path) -> dict:
    """(id, arm) -> (raw, length_adjusted, chars), exactly as hbpro.report/compare compute them."""
    rows = hbpro.load()
    met = {(g["id"], g["arm"], g["k"]): g["met"] for g in hbpro._rows(out / "grades.jsonl")}
    sc = {}
    for r in hbpro._rows(out / "replies.jsonl"):
        items = rows[r["id"]]["rubric_items"]
        if not all((r["id"], r["arm"], k) in met for k in range(len(items))):
            continue
        pos = sum(it["points"] for it in items if it["points"] > 0)
        s = sum(it["points"] for k, it in enumerate(items) if met[(r["id"], r["arm"], k)]) / pos
        sc[(r["id"], r["arm"])] = (s, s - 2.94e-5 * (len(r["text"]) - 2000), len(r["text"]))
    return sc


def _arm_summary(sc, arm):
    v = np.array([x for (i, a), x in sc.items() if a == arm])
    return {"n": len(v), "raw": _r(np.clip(v[:, 0].mean(), 0, 1) * 100), "raw_se": _r(v[:, 0].std(ddof=1) / np.sqrt(len(v)) * 100),
            "length_adjusted": _r(np.clip(v[:, 1].mean(), 0, 1) * 100),
            "length_adjusted_se": _r(v[:, 1].std(ddof=1) / np.sqrt(len(v)) * 100),
            "mean_chars": int(round(v[:, 2].mean())), "median_chars": int(np.median(v[:, 2])),
            "share_le_2000_chars": _r((v[:, 2] <= 2000).mean(), 3)}


def _paired(sc, a, bs, col, rng, ids_filter=None):
    """Paired a - b on common tasks; bootstrap over tasks in the given rng order (hbpro.compare method)."""
    res = {}
    for b in bs:
        ids = sorted(i for (i, x) in sc if x == a and (i, b) in sc and (ids_filter is None or ids_filter(i, b)))
        d = np.array([sc[(i, a)][col] - sc[(i, b)][col] for i in ids])
        if len(d) == 0:
            res[b] = {"n": 0}
            continue
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)]
        t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
        res[b] = {"n": len(ids), "diff": _r(d.mean() * 100), "ci95": _pct_ci(boot),
                  "paired_t_p": float(t.pvalue) if t is not None else None, "bootstrap_p": _boot_p(boot)}
    return res


def _wtl(sc, a, b, col, tol=1e-9):
    ids = [i for (i, x) in sc if x == a and (i, b) in sc]
    d = np.array([sc[(i, a)][col] - sc[(i, b)][col] for i in ids])
    return {"n": len(d), "win": int((d > tol).sum()), "tie": int((np.abs(d) <= tol).sum()), "loss": int((d < -tol).sum())}


def _ols_boot(x, y, rng):
    X = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bb = []
    for _ in range(NB):
        k = rng.integers(0, len(x), len(x))
        bb.append(np.linalg.lstsq(X[k], y[k], rcond=None)[0])
    bb = np.array(bb)
    return beta, bb


def _cost(out: Path, n_tasks: int, bare_counts: dict) -> dict:
    """List-price cost per task. Pipeline usage rows carry no arm label, so two attribution rules are reported.

    Each bare frontier arm (opus55, gpt61sol) writes one usage row per task; each pipeline task writes one opus55 row,
    one gpt61sol row and (when it searched) one web_search row, each the sum over that task's calls.
    Rule A (file order): the first n rows of each model in usage.jsonl are the bare arm (the sample step runs the arms
    in order, so the bare arms start first). Rule B (size): the n rows with the smallest input-token counts are the bare
    arm (the bare arm sends only the conversation; the pipeline adds system prompts, drafts and search results).
    """
    P = hbpro.PRICE
    u = [x for x in hbpro._rows(out / "usage.jsonl") if not x["who"].startswith("grader")]
    cost = lambda x: x["in"] / 1e6 * P[x["who"]][0] + x["out"] / 1e6 * P[x["who"]][1]  # noqa: E731
    per = {}
    for who in ("amie_style", "mai_dxo_style"):
        rs = [x for x in u if x["who"] == who]
        per[who] = {"usage_rows": len(rs), "usd_per_task": _r(sum(map(cost, rs)) / n_tasks, 4),
                    "input_tokens_per_task": int(sum(x["in"] for x in rs) / n_tasks),
                    "output_tokens_per_task": int(sum(x["out"] for x in rs) / n_tasks)}
    rules = {}
    for rule in ("A_file_order", "B_smallest_input"):
        bare, pipe = {}, []
        for who in ("opus55", "gpt61sol"):
            rs = [(j, x) for j, x in enumerate(u) if x["who"] == who]
            order = rs if rule == "A_file_order" else sorted(rs, key=lambda t: (t[1]["in"], t[0]))
            bare[who] = [x for _, x in order[:bare_counts[who]]]
            pipe += [x for _, x in order[bare_counts[who]:]]
        pipe += [x for x in u if x["who"] == "web_search"]
        r = {f"{w}_bare": {"rows": len(bare[w]), "usd_per_task": _r(sum(map(cost, bare[w])) / n_tasks, 4),
                           "input_tokens_per_task": int(sum(x["in"] for x in bare[w]) / n_tasks),
                           "output_tokens_per_task": int(sum(x["out"] for x in bare[w]) / n_tasks)} for w in bare}
        r["compass_pro_full"] = {
            "rows": len(pipe), "usd_per_task": _r(sum(map(cost, pipe)) / n_tasks, 4),
            "by_component_usd_per_task": {w: _r(sum(cost(x) for x in pipe if x["who"] == w) / n_tasks, 4)
                                          for w in ("opus55", "gpt61sol", "web_search")},
            "web_search_requests_per_task": _r(sum(x["in"] for x in pipe if x["who"] == "web_search") / n_tasks, 2)}
        rules[rule] = r
    tot = sum(map(cost, u))
    return {"n_tasks": n_tasks, "price_usd_per_million_tokens": {k: P[k] for k in ("opus55", "gpt61sol", "amie_style",
                                                                                   "mai_dxo_style")},
            "web_search_usd_per_request": P["web_search"][0] / 1e6, "comparator_chains": per, "attribution_rules": rules,
            "total_reply_usd_all_arms": _r(tot, 2),
            "attribution_uncertainty_usd_per_task_compass": _r(abs(rules["A_file_order"]["compass_pro_full"]["usd_per_task"]
                                                               - rules["B_smallest_input"]["compass_pro_full"]["usd_per_task"]), 4),
            "note": "Rule A and rule B bound the misattribution at the boundary between the bare arms and the pipeline; "
                    "the bare opus55 and gpt61sol arms each made one call per task. Totals are exact; only the split is "
                    "attributed. Grader usage is excluded."}


def hb_test() -> dict:
    out = D / "hbpro_test"
    sc = _hb_scores(out)
    rows = hbpro.load()
    arms = sorted({a for _, a in sc})
    res = {"split": "test", "n_tasks": len(json.loads((out / "split.json").read_text())["test"]), "ours": OURS_HB,
           "scoring": "score = sum of met rubric points / sum of positive points; length adjusted = score - 2.94e-5 * "
                      "(characters - 2000); arm means x 100 (clipped to [0, 100]); grader GPT-5.4",
           "arms": {a: _arm_summary(sc, a) for a in arms}}
    rng = np.random.default_rng(SEED)
    res["paired_length_adjusted"] = _paired(sc, OURS_HB, HB_COMPS, 1, rng)  # reproduces compare.txt
    rng = np.random.default_rng(SEED)
    res["paired_raw"] = _paired(sc, OURS_HB, HB_COMPS, 0, rng)
    fam = ("amie_style", "mai_dxo_style")
    res["holm_primary_family"] = {
        "family": f"{OURS_HB} vs amie_style and vs mai_dxo_style (2 tests per score type)",
        "test": "two-sided paired t test on task-level differences (Holm adjusted); two-sided percentile paired "
                "bootstrap p also given (floor 1/2000)",
        **{col: {b: {"p": res[f"paired_{col}"][b]["paired_t_p"], "p_holm": h}
                 for b, h in _holm({b: res[f"paired_{col}"][b]["paired_t_p"] for b in fam}).items()}
           for col in ("length_adjusted", "raw")}}
    res["win_tie_loss"] = {b: {"raw": _wtl(sc, OURS_HB, b, 0), "length_adjusted": _wtl(sc, OURS_HB, b, 1)} for b in HB_COMPS}
    # subgroups vs amie_style
    sub = {}
    rng = np.random.default_rng(SEED)
    for key in ("use_case", "type"):
        for val in sorted({rows[i][key] for (i, a) in sc if a == OURS_HB}):
            f = lambda i, b, key=key, val=val: rows[i][key] == val  # noqa: E731
            sub[f"{key}={val}"] = {
                "ours": _r(np.mean([sc[(i, OURS_HB)][1] for (i, a) in sc if a == OURS_HB and rows[i][key] == val]) * 100),
                "amie_style": _r(np.mean([sc[(i, "amie_style")][1] for (i, a) in sc if a == "amie_style" and rows[i][key] == val]) * 100),
                "paired_length_adjusted": _paired(sc, OURS_HB, ["amie_style"], 1, rng, f)["amie_style"],
                "paired_raw": _paired(sc, OURS_HB, ["amie_style"], 0, rng, f)["amie_style"]}
    res["subgroups_vs_amie_style"] = sub
    # length
    rng = np.random.default_rng(SEED)
    both_short = lambda i, b: sc[(i, OURS_HB)][2] <= 2000 and sc[(i, b)][2] <= 2000  # noqa: E731
    length = {"paired_both_le_2000_chars": {"raw": _paired(sc, OURS_HB, HB_COMPS, 0, rng, both_short),
                                            "length_adjusted": _paired(sc, OURS_HB, HB_COMPS, 1, rng, both_short)},
              "regression": {}}
    rng = np.random.default_rng(SEED)
    for b in HB_COMPS:
        ids = sorted(i for (i, a) in sc if a == OURS_HB and (i, b) in sc)
        x = np.array([(sc[(i, OURS_HB)][2] - sc[(i, b)][2]) / 1000 for i in ids])
        y = np.array([sc[(i, OURS_HB)][0] - sc[(i, b)][0] for i in ids]) * 100
        beta, bb = _ols_boot(x, y, rng)
        length["regression"][b] = {
            "model": "raw score difference (points) = intercept + slope x character difference (thousands); OLS, "
                     "bootstrap 95% CI over tasks",
            "n": len(ids), "intercept_diff_at_equal_length": _r(beta[0], 2), "intercept_ci95": _pct_ci(bb[:, 0], 1, 2),
            "slope_per_1000_chars": _r(beta[1], 2), "slope_ci95": _pct_ci(bb[:, 1], 1, 2),
            "mean_char_diff": int(round(x.mean() * 1000)),
            "spearman_rho": _r(stats.spearmanr(x, y).statistic, 3),
            "note": "the length-adjusted difference equals the raw difference minus 2.94 points per 1,000 characters "
                    "of the character difference, by construction"}
    res["length"] = length
    res["cost"] = _cost(out, res["n_tasks"], {"opus55": res["n_tasks"], "gpt61sol": res["n_tasks"]})
    return res


def hb_dev() -> dict:
    out = D / "hbpro_dev"
    sc = _hb_scores(out)
    comps = ["amie_style_g61", "mai_dxo_style_g61", "amie_style", "mai_dxo_style", "gpt61sol", "opus55", "physician"]
    arms = [OURS_HB_DEV, "compass_pro"] + comps
    res = {"split": "development", "ours": OURS_HB_DEV,
           "note": "compass_pro_edit is the development-split equivalent of compass_pro_full (step 5 applied to the "
                   "stored compass_pro answer); the _g61 arms run the same comparator chains on a GPT-6.1 Sol backbone",
           "arms": {a: _arm_summary(sc, a) for a in arms if any(x == a for _, x in sc)}}
    rng = np.random.default_rng(SEED)
    res["paired_length_adjusted"] = _paired(sc, OURS_HB_DEV, comps, 1, rng)
    rng = np.random.default_rng(SEED)
    res["paired_raw"] = _paired(sc, OURS_HB_DEV, comps, 0, rng)
    rng = np.random.default_rng(SEED)
    res["g61_chain_minus_opus_chain"] = {
        f"{c}_g61 - {c}": {col: _paired(sc, f"{c}_g61", [c], k, rng)[c] for col, k in (("raw", 0), ("length_adjusted", 1))}
        for c in ("amie_style", "mai_dxo_style")}
    return res


# ---------------------------------------------------------------- real texts

def _texts_data():
    cw = {r["sample_id"]: r for r in _rows(D / "fresh_set/crosswalk.jsonl")}

    def labs(path):
        out = {}
        for r in _rows(path):
            c = cw[r["sample_id"]]
            out[c["id"]] = {**label({"detection_truth": r["hazard"], "hazard_category": r["category"],
                                     "action_truth": r["action"]}), "w": _W2[c["id"]], "hazard": r["hazard"] == 1,
                            "action": str(r["action"]), "stratum": c["stratum"]}
        return out

    basu, morgan = labs(D / "fresh_set/labels_basu.jsonl"), labs(D / "fresh_set/labels_morgan.jsonl")
    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(TEXTS / "urgency.jsonl")}  # last row wins, as texts_report
    rep = {}  # (arm, id) -> {repeat: (escalated, any_hit, unconditional_hit)}
    for r in _rows(TEXTS / "replies.jsonl"):
        i, k, arm = r["prompt_id"], r["repeat"], r["arm"]
        u = urg.get((i, k, arm))
        e = u is not None and URGENCY.index(u) >= URGENCY.index("routine_followup")
        t = r["text"] or ""
        rep.setdefault((arm, i), {})[k] = (e, bool(check(t)), bool(check_unconditional(t)))
    th = json.loads((TEXTS / "gate_thresholds.json").read_text())
    g = {x["id"]: x for x in _rows(TEXTS / "gate_scores.jsonl")}
    paged = {i: g[i]["emergency"] >= th["t_e"] or g[i]["safety"] >= th["t_s"] for i in g}
    page_e = {i: g[i]["emergency"] >= th["t_e"] for i in g}
    return basu, morgan, rep, paged, page_e


def _per_repeat(rep, paged):
    """arm -> id -> list of 0/1 escalation over repeats (sorted repeat order)."""
    arms = sorted({a for a, _ in rep})
    E = {a: {i: [v[k][0] for k in sorted(v)] for (x, i), v in rep.items() if x == a} for a in arms}
    w = {i: v for (x, i), v in rep.items() if x == WRITER}
    E["ours"] = {i: [True] * len(v) if paged[i] else [v[k][0] or v[k][2] for k in sorted(v)] for i, v in w.items()}
    E["writer_alone"] = {i: [v[k][0] or v[k][2] for k in sorted(v)] for i, v in w.items()}
    E["gate_alone"] = {i: [paged[i]] * len(v) for i, v in w.items()}
    E["clinician_first_any_hit_held"] = {i: [True] * len(v) if paged[i] else [v[k][0] or v[k][1] for k in sorted(v)]
                                         for i, v in w.items()}
    return E


def _wm(x, m, w):
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")


def _metrics(E, lab, ids, B):
    """Per-arm sensitivity and escalation without hazard; CIs over the shared bootstrap index set B."""
    w = np.array([lab[i]["w"] for i in ids])
    sp = np.array([lab[i]["safety_positive"] for i in ids])
    nh = np.array([not lab[i]["hazard"] for i in ids])
    c9 = np.array([lab[i]["call_911"] for i in ids])
    out = {}
    for arm, d in E.items():
        e = np.array([np.mean(d[i]) for i in ids])
        out[arm] = {
            "texts": len(ids), "safety_positive": int(sp.sum()), "no_hazard": int(nh.sum()), "call_911": int(c9.sum()),
            "hazard_sensitivity_weighted": _r(_wm(e, sp, w), 3),
            "hazard_sensitivity_weighted_ci": _pct_ci([_wm(e[b], sp[b], w[b]) for b in B], 1, 3),
            "hazard_sensitivity_unweighted": _r(e[sp].mean(), 3) if sp.any() else None,
            "hazard_sensitivity_unweighted_ci": _pct_ci([e[b][sp[b]].mean() if sp[b].any() else np.nan for b in B], 1, 3),
            "escalation_no_hazard_per_1000_weighted": _r(1000 * _wm(e, nh, w)),
            "escalation_no_hazard_weighted_ci": _pct_ci([_wm(e[b], nh[b], w[b]) for b in B], 1000),
            "escalation_no_hazard_per_1000_unweighted": _r(1000 * e[nh].mean()) if nh.any() else None,
            "escalation_no_hazard_unweighted_ci": _pct_ci([e[b][nh[b]].mean() if nh[b].any() else np.nan for b in B], 1000),
            "call_911_escalated_mean": _r(e[c9].sum(), 2)}
    return out


def _diffs(E, lab, ids, B, a, bs):
    w = np.array([lab[i]["w"] for i in ids])
    sp = np.array([lab[i]["safety_positive"] for i in ids])
    nh = np.array([not lab[i]["hazard"] for i in ids])
    ea = np.array([np.mean(E[a][i]) for i in ids])
    out = {}
    for b in bs:
        eb = np.array([np.mean(E[b][i]) for i in ids])
        fs = lambda k: _wm(ea[k], sp[k], w[k]) - _wm(eb[k], sp[k], w[k])  # noqa: E731
        fn = lambda k: _wm(ea[k], nh[k], w[k]) - _wm(eb[k], nh[k], w[k])  # noqa: E731
        us = lambda k: ea[k][sp[k]].mean() - eb[k][sp[k]].mean()  # noqa: E731
        un = lambda k: ea[k][nh[k]].mean() - eb[k][nh[k]].mean()  # noqa: E731
        al = np.arange(len(ids))
        out[f"{a} - {b}"] = {
            "sens_diff_weighted": _r(fs(al), 3), "sens_diff_weighted_ci": _pct_ci([fs(k) for k in B], 1, 3),
            "nohaz_diff_per1000_weighted": _r(1000 * fn(al)), "nohaz_diff_weighted_ci": _pct_ci([fn(k) for k in B], 1000),
            "sens_diff_unweighted": _r(us(al), 3), "sens_diff_unweighted_ci": _pct_ci([us(k) for k in B], 1, 3),
            "nohaz_diff_per1000_unweighted": _r(1000 * un(al)), "nohaz_diff_unweighted_ci": _pct_ci([un(k) for k in B], 1000)}
    return out


def _B(n, rng):
    return [rng.integers(0, n, n) for _ in range(NB)]


def _kappa(a, b):
    binary = all(isinstance(x, bool) for x in list(a) + list(b))
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return {"n": len(a), "observed_agreement": _r(po, 3), "kappa": _r((po - pe) / (1 - pe), 3) if pe < 1 else None,
            "positives_basu": int(a.sum()) if binary else None, "positives_morgan": int(b.sum()) if binary else None,
            "both_positive": int((a & b).sum()) if binary else None}


def _registered_v22(E, lab, ids, B) -> dict:
    """OSF hypotheses H1a-H1c (design/paper2_preregistration_draft.md section 10) for the registered v22 arm as run.

    v22 escalation = its own reply's urgency routine_followup or higher (no gate). Sensitivity among safety-positive
    texts. Registered binary outcome per message = majority escalation (>= 2 of 3 repeats); exact McNemar two-sided on
    the safety-positive messages, unweighted. H1c (non-inferiority, margin 0.10): the lower 95% bootstrap bound of
    v22 - amie_style vs -0.10, plus a one-sided bootstrap p = share of resampled differences <= -0.10 (majority outcome,
    unweighted, same index vectors), used as the H1c p in the Holm step.
    """
    sp = np.array([lab[i]["safety_positive"] for i in ids])
    al = np.arange(len(ids))
    maj = {a: np.array([sum(E[a][i]) >= 2 for i in ids]) for a in ("v22", "raw_llm", "mai_dxo_style", "amie_style")}
    out = {"population": f"{int(sp.sum())} safety-positive texts (S. Basu labels) of {len(ids)}",
           "v22_escalation": "own reply urgency routine_followup or higher; no gate, no check",
           "paired_diffs_mean_over_repeats": _diffs(E, lab, ids, B, "v22", ["raw_llm", "mai_dxo_style", "amie_style"])}
    hyp = {"H1a": "raw_llm", "H1b": "mai_dxo_style", "H1c": "amie_style"}
    ps = {}
    for h, b in hyp.items():
        a1, b1 = maj["v22"][sp], maj[b][sp]
        n10, n01 = int((a1 & ~b1).sum()), int((~a1 & b1).sum())
        p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        d = lambda k: maj["v22"][k][sp[k]].mean() - maj[b][k][sp[k]].mean()  # noqa: E731
        boot = np.array([d(k) for k in B])
        r = {"comparator": b, "majority_sens_v22": _r(a1.mean(), 3), "majority_sens_comparator": _r(b1.mean(), 3),
             "majority_diff": _r(d(al), 3), "majority_diff_ci95": _pct_ci(boot, 1, 3),
             "discordant_v22_only": n10, "discordant_comparator_only": n01, "mcnemar_exact_two_sided_p": p}
        if h == "H1c":
            pw = out["paired_diffs_mean_over_repeats"]["v22 - amie_style"]
            r["non_inferiority"] = {
                "margin": -0.10, "majority_unweighted_lower_bound": r["majority_diff_ci95"][0],
                "mean_over_repeats_weighted_lower_bound": pw["sens_diff_weighted_ci"][0],
                "mean_over_repeats_unweighted_lower_bound": pw["sens_diff_unweighted_ci"][0],
                "one_sided_bootstrap_p": float(max((boot <= -0.10).mean(), 1 / NB)),
                "met_majority_unweighted": r["majority_diff_ci95"][0] > -0.10,
                "met_weighted": pw["sens_diff_weighted_ci"][0] > -0.10,
                "met_unweighted": pw["sens_diff_unweighted_ci"][0] > -0.10}
            ps[h] = r["non_inferiority"]["one_sided_bootstrap_p"]
        else:
            ps[h] = p
        out[h] = r
    h3 = _holm(ps)
    h6 = _holm({**ps, "H2a": 1.0, "H2b": 1.0, "H2c": 1.0})
    for h in hyp:
        out[h]["p_holm_3_tested"] = h3[h]
        out[h]["p_holm_6_with_H2_as_p1"] = h6[h]
        out[h]["reject_holm_3"] = h3[h] < 0.05
        out[h]["reject_holm_6"] = h6[h] < 0.05
    out.update({"H2a": "not run", "H2b": "not run", "H2c": "not run (simulation outcomes not completed)",
                "holm_note": "Primary family: the 3 tested hypotheses (H1a, H1b McNemar p; H1c one-sided "
                             "non-inferiority bootstrap p). Sensitivity: Holm over the registered 6 with the untested "
                             "H2a-c entered as p = 1, which is the most conservative reading."})
    return out


def texts() -> dict:
    basu, morgan, rep, paged, page_e = _texts_data()
    E = _per_repeat(rep, paged)
    ids = sorted(i for i in E["ours"] if i in basu)
    res = {"labels": "S. Basu labels on all 949 texts; J. Morgan labels on 200; Hajek two-phase weights (paper/fresh_weights.py)",
           "policy": "ours = paging gate + raw_llm writer + check_unconditional (final policy; conditional safety-net "
                     "lines allowed). clinician_first_any_hit_held = the registered report.json rule (any check() hit held).",
           "arms": "amie_style, mai_dxo_style, raw_llm, v22 as run; writer_alone = raw_llm escalation or unconditional "
                   "hit, no gate; gate_alone = gate pages, no writer"}
    B = _B(len(ids), np.random.default_rng(SEED))
    res["main"] = _metrics(E, basu, ids, B)
    res["paired_diffs"] = _diffs(E, basu, ids, B, "ours", ["amie_style", "mai_dxo_style", "raw_llm", "v22"])
    # each writer + the same paging gate + the same final-policy check (built exactly as ours)
    L = {"ours": E["ours"]}
    for arm in ("amie_style", "mai_dxo_style", "v22", WRITER):
        L[f"{arm}+layer"] = {i: [True] * len(v) if paged[i] else [v[k][0] or v[k][2] for k in sorted(v)]
                             for (x, i), v in rep.items() if x == arm}
    assert L[f"{WRITER}+layer"] == L["ours"]
    res["same_safety_layer"] = {
        "definition": "writer + the same paging gate + check_unconditional: escalated in a repeat when the reply advises "
                      "routine_followup or higher or check_unconditional hits; all repeats escalated when the gate pages. "
                      "raw_llm+layer is ours (identical by construction). Same bootstrap index vectors as main.",
        "unconditional_instructions_reaching_patient_per_1000": {
            k: 0.0 for k in L if k != "ours"},
        "unconditional_instructions_note": "0 by construction: every reply with an unconditional hit is held and the "
                                           "text is escalated to clinician review",
        "replies_held_per_1000": {f"{a}+layer": ins_u for a, ins_u in (
            (a, _r(1000 * np.mean([np.mean([v[k][2] for k in v]) for (x, i), v in rep.items() if x == a and i in basu])))
            for a in ("amie_style", "mai_dxo_style", "v22", WRITER))},
        "metrics": _metrics({k: v for k, v in L.items() if k != "ours"}, basu, ids, B),
        "paired_diffs": _diffs(L, basu, ids, B, "ours", [f"{a}+layer" for a in ("amie_style", "mai_dxo_style", "v22")])}
    res["registered_hypotheses_v22"] = _registered_v22(E, basu, ids, B)
    strata = {}
    for s in ("random", "enriched"):
        sid = [i for i in ids if basu[i]["stratum"] == s]
        Bs = _B(len(sid), np.random.default_rng(SEED))
        strata[s] = {"metrics": _metrics(E, basu, sid, Bs),
                     "paired_diffs": _diffs(E, basu, sid, Bs, "ours", ["amie_style", "mai_dxo_style"])}
    res["by_stratum"] = strata
    # second physician
    mid = sorted(i for i in morgan if i in E["ours"])
    Bm = _B(len(mid), np.random.default_rng(SEED))
    agree_ids = [i for i in mid if basu[i]["safety_positive"] == morgan[i]["safety_positive"]
                 and basu[i]["hazard"] == morgan[i]["hazard"]]
    agree_lab = {i: basu[i] for i in agree_ids}
    Ba = _B(len(agree_ids), np.random.default_rng(SEED))
    res["second_physician"] = {
        "n_double_labeled": len(mid), "strata": {s: sum(basu[i]["stratum"] == s for i in mid) for s in ("random", "enriched")},
        "basu_labels_on_200": _metrics(E, basu, mid, Bm),
        "morgan_labels_on_200": _metrics(E, morgan, mid, Bm),
        "both_agree": {"definition": "texts where both physicians give the same safety_positive and the same any-hazard label",
                       "n": len(agree_ids), "metrics": _metrics(E, agree_lab, agree_ids, Ba)},
        "agreement": {"safety_positive": _kappa([basu[i]["safety_positive"] for i in mid], [morgan[i]["safety_positive"] for i in mid]),
                      "call_911": _kappa([basu[i]["call_911"] for i in mid], [morgan[i]["call_911"] for i in mid]),
                      "any_hazard": _kappa([basu[i]["hazard"] for i in mid], [morgan[i]["hazard"] for i in mid]),
                      "action_4_class": _kappa([basu[i]["action"] for i in mid], [morgan[i]["action"] for i in mid]),
                      "kappa": "Cohen's kappa, unweighted"}}
    # repeat consistency
    cons = {}
    for arm, d in E.items():
        v = [d[i] for i in ids]
        full = [x for x in v if len(x) == 3]
        allsame = np.array([len(set(x)) == 1 for x in full])
        w = np.array([basu[i]["w"] for i in ids if len(d[i]) == 3])
        sp = np.array([basu[i]["safety_positive"] for i in ids if len(d[i]) == 3])
        cons[arm] = {"texts_with_3_repeats": len(full), "share_all_3_agree": _r(allsame.mean(), 3),
                     "share_all_3_agree_weighted": _r((allsame * w).sum() / w.sum(), 3),
                     "share_all_3_agree_safety_positive": _r(allsame[sp].mean(), 3)}
    res["repeat_consistency"] = cons
    # gate / writer contributions to ours (Basu labels, all texts)
    w = np.array([basu[i]["w"] for i in ids])
    sp = np.array([basu[i]["safety_positive"] for i in ids])
    nh = np.array([not basu[i]["hazard"] for i in ids])
    gt = np.array([paged[i] for i in ids], dtype=float)
    wr = np.array([np.mean(E["writer_alone"][i]) for i in ids])
    ours = np.array([np.mean(E["ours"][i]) for i in ids])
    parts = {"gate_and_writer": gt * wr, "gate_only": gt * (1 - wr), "writer_only": (1 - gt) * wr,
             "neither": (1 - gt) * (1 - wr)}
    res["gate_writer_contributions"] = {
        "definition": "per text, expected share over repeats; gate is constant across repeats",
        "texts_paged": int(gt.sum()), "texts_paged_emergency_threshold": int(sum(page_e[i] for i in ids)),
        "texts_paged_weighted_share": _r((gt * w).sum() / w.sum(), 4),
        "among_safety_positive_weighted": {k: _r(_wm(v, sp, w), 3) for k, v in parts.items()},
        "among_safety_positive_unweighted": {k: _r(v[sp].mean(), 3) for k, v in parts.items()},
        "among_no_hazard_per_1000_weighted": {k: _r(1000 * _wm(v, nh, w)) for k, v in parts.items()},
        "ours_sensitivity_weighted": _r(_wm(ours, sp, w), 3),
        "check": "gate_and_writer + gate_only + writer_only equals ours"}
    # instruction rates
    ins = {}
    for arm in sorted({a for a, _ in rep}):
        hits = [[v[k][1] for k in v] for (x, i), v in rep.items() if x == arm and i in basu]
        unc = [[v[k][2] for k in v] for (x, i), v in rep.items() if x == arm and i in basu]
        ins[arm] = {"any_instruction_per_1000_replies": _r(1000 * np.mean([np.mean(h) for h in hits])),
                    "unconditional_instruction_per_1000_replies": _r(1000 * np.mean([np.mean(h) for h in unc]))}
    res["ai_emergency_instructions"] = {"definition": "check() any disallowed instruction; check_unconditional() "
                                                      "sentence without a conditional word; per-text mean over repeats",
                                        **ins, "ours_writer_replies_held_per_1000": ins[WRITER]["unconditional_instruction_per_1000_replies"]}
    return res


# ---------------------------------------------------------------- paging gate

def paging_gate() -> tuple[dict, list[dict]]:
    rep = json.loads((D / "paper2_dev/paging_gate_497/report.json").read_text())
    th = json.loads((TEXTS / "gate_thresholds.json").read_text())
    rows = []
    for outcome, o in rep["outcomes"].items():
        for score, s in o["scores"].items():
            for key, v in s.items():
                if key.startswith("auroc"):
                    continue
                rule, alpha = key.rsplit("_", 1)
                row = {"outcome": outcome, "n": o["n"], "positives": o["positives"], "score": score, "rule": rule,
                       "alpha": float(alpha), "auroc": s["auroc"]["est"], "auroc_ci95": s["auroc"]["ci95"],
                       "auroc_weighted": s["auroc_weighted"]["est"], "auroc_weighted_ci95": s["auroc_weighted"]["ci95"]}
                if rule == "full_sample":
                    row.update({"threshold": v["threshold"], "miss": v["miss"], "miss_w": v["miss_w"], "paged": v["paged"],
                                "paged_w": v["paged_w"]})
                else:
                    for m in ("threshold", "miss", "paged", "miss_w", "paged_w"):
                        row.update({m: v[m]["mean"], f"{m}_p05": v[m]["p05"], f"{m}_p95": v[m]["p95"]})
                    row.update({"splits": v["splits"], "share_splits_miss_above_alpha": v["share_splits_miss_above_alpha"],
                                "undefined_splits_paged_all": v["undefined_splits_paged_all"]})
                rows.append(row)
    deployed = {"t_e": th["t_e"], "alpha_e": th["alpha_e"], "t_s": th["t_s"], "alpha_s": th["alpha_s"],
                "matches": [{k: r[k] for k in ("outcome", "score", "rule", "alpha", "threshold")} for r in rows
                            if r["rule"] == "full_sample" and r["score"].startswith("kev")
                            and ((r["outcome"] == "emergency" and r["alpha"] == th["alpha_e"] and abs(r["threshold"] - th["t_e"]) < 5e-5)
                                 or (r["outcome"] == "safety" and r["alpha"] == th["alpha_s"] and abs(r["threshold"] - th["t_s"]) < 5e-5))]}
    meta = {"source": "compass-pcp-data/paper2_dev/paging_gate_497/report.json",
            "columns": "miss = share of positives not paged; paged = share of texts paged; _w = Hajek-weighted to the "
                       "pool; crc = conformal risk control over random calibration/evaluation splits (mean, p05, p95 "
                       "over splits); crc_weighted_rule = the weighted CRC rule; full_sample = threshold fit on all texts",
            "per_model_auroc_kev": {k: o["per_model_auroc_kev"] for k, o in rep["outcomes"].items()},
            "deployed_thresholds": deployed, "rows": rows}
    return meta, rows


# ---------------------------------------------------------------- validation

def validate(hbt: dict, tx_main: dict, tx_diffs: dict) -> dict:
    checks = []

    def eq(name, got, want):
        checks.append({"item": name, "derived": got, "existing": want, "match": got == want})

    rep = json.loads((D / "hbpro_test/report.json").read_text())
    for a, v in hbt["arms"].items():
        eq(f"hbpro_test {a} length_adjusted", v["length_adjusted"], rep[a]["length_adjusted"]["score"])
        eq(f"hbpro_test {a} raw", v["raw"], rep[a]["raw"]["score"])
        eq(f"hbpro_test {a} mean_chars", v["mean_chars"], rep[a]["mean_chars"])
    cmp_ = json.loads((D / "hbpro_test/compare.txt").read_text())
    for b in HB_COMPS:
        p = hbt["paired_length_adjusted"][b]
        eq(f"hbpro_test compare {OURS_HB} - {b}", [p["n"], p["diff"], p["ci95"]],
           [cmp_[f"{OURS_HB} - {b}"][k] for k in ("n", "diff", "ci95")])
    drep = json.loads((D / "hbpro_dev/report.json").read_text())
    dsc = _hb_scores(D / "hbpro_dev")
    for a in (OURS_HB_DEV, "amie_style_g61", "mai_dxo_style_g61", "amie_style", "mai_dxo_style", "gpt61sol", "opus55"):
        s = _arm_summary(dsc, a)
        eq(f"hbpro_dev {a} length_adjusted/raw", [s["length_adjusted"], s["raw"]],
           [drep[a]["length_adjusted"]["score"], drep[a]["raw"]["score"]])
    # texts: report.json exactly as texts_report.py computes it (sequential RNG over sorted arms)
    basu, _, rep_t, paged, _ = _texts_data()
    E = _per_repeat(rep_t, paged)
    E2 = {k: E[k] for k in ("amie_style", "mai_dxo_style", "raw_llm", "v22")}
    E2["clinician_first"] = E["clinician_first_any_hit_held"]
    rng = np.random.default_rng(SEED)
    tr = json.loads((TEXTS / "report.json").read_text())
    for arm in sorted(E2):
        ids = sorted(i for i in E2[arm] if i in basu)
        e = np.array([np.mean(E2[arm][i]) for i in ids])
        w = np.array([basu[i]["w"] for i in ids])
        sp = np.array([basu[i]["safety_positive"] for i in ids])
        nh = np.array([not basu[i]["hazard"] for i in ids])

        def ci(f):
            bs = [f(rng.integers(0, len(ids), len(ids))) for _ in range(NB)]
            return [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
        got = {"hazard_sensitivity_weighted": round(_wm(e, sp, w), 3), "hazard_sensitivity_ci": ci(lambda b: _wm(e[b], sp[b], w[b])),
               "hazard_sensitivity_unweighted": round(float(e[sp].mean()), 3),
               "escalation_no_hazard_per_1000_weighted": round(1000 * _wm(e, nh, w), 1),
               "escalation_no_hazard_ci": [round(1000 * x, 1) for x in ci(lambda b: _wm(e[b], nh[b], w[b]))]}
        for k, v in got.items():
            eq(f"texts report.json {arm} {k}", v, tr[arm][k])
    # report_policy_unconditional.json (texts_report.py --policy unconditional): point estimates of the deployable policy
    ph = json.loads((TEXTS / "report_policy_unconditional.json").read_text())["clinician_first"]
    m = tx_main["ours"]
    eq("texts deployable sens", m["hazard_sensitivity_weighted"], ph["hazard_sensitivity_weighted"])
    eq("texts deployable sens unweighted", m["hazard_sensitivity_unweighted"], ph["hazard_sensitivity_unweighted"])
    eq("texts deployable nohaz_per1000", m["escalation_no_hazard_per_1000_weighted"], ph["escalation_no_hazard_per_1000_weighted"])
    return {"all_match": all(c["match"] for c in checks), "n_checks": len(checks),
            "mismatches": [c for c in checks if not c["match"]], "checks": checks}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hbt = hb_test()
    (OUT / "hbpro_test_supplement.json").write_text(json.dumps(hbt, indent=1))
    hbd = hb_dev()
    (OUT / "hbpro_dev_g61_chains.json").write_text(json.dumps(hbd, indent=1))
    tx = texts()
    (OUT / "texts_supplement.json").write_text(json.dumps(tx, indent=1))
    meta, rows = paging_gate()
    (OUT / "paging_gate_table.json").write_text(json.dumps(meta, indent=1))
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / "paging_gate_table.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    val = validate(hbt, tx["main"], tx["paired_diffs"])
    (OUT / "validation.json").write_text(json.dumps(val, indent=1))
    print(json.dumps({"written": sorted(p.name for p in OUT.iterdir()), "validation_all_match": val["all_match"],
                      "n_checks": val["n_checks"], "mismatches": [c["item"] for c in val["mismatches"]]}, indent=1))


if __name__ == "__main__":
    main()
