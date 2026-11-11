"""Registered addendum analyses on HealthBench Professional (https://osf.io/tnezd, sections 2, 3 and 6).

Study A: the pipeline with blocked search (compass_pro_full_blk) minus each tuned comparator (selection in
hbpro_dev/addendum_selection.json), paired on the 350 test tasks; bootstrap 95% CI (2,000 resamples, default_rng(20261108))
and paired t-test P, Holm over A1 and A2; secondary: the two-step pipeline with blocked search minus each tuned agent.
Study B: blocked minus original for compass_pro_full, opus_search_edit and amie_style_search (95% and 90% CIs; rule:
each 90% CI within +/-4 points); the pipeline's lead over the AMIE-style agent with search, both blocked; search audit
(results on the blocked list; results whose URL or title contains "healthbench"), with A and B repeated without flagged tasks.
Study E: list-price cost per test task of each addendum arm from arm-labeled usage; two-step versus tuned AMIE-style.
Out: paper2_final/derived/addendum_hb.json. Usage: python -m paper.addendum_hb_analysis
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from compass_pcp.arms.pro import SEARCH_BLOCKED_DOMAINS
from paper import hbpro
from paper.supplementary_analyses import _hb_scores

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
T, DEV = D / "hbpro_test", D / "hbpro_dev"
OUT = D / "paper2_final/derived/addendum_hb.json"


def _diff(sc, ids, a, b):
    ids = [i for i in ids if (i, a) in sc and (i, b) in sc]
    x = np.array([sc[(i, a)][1] - sc[(i, b)][1] for i in ids]) * 100
    rng = np.random.default_rng(20261108)
    boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)])
    return {"a": a, "b": b, "tasks": len(ids), "diff": round(float(x.mean()), 2),
            "ci95": [round(float(np.percentile(boot, q)), 2) for q in (2.5, 97.5)],
            "ci90": [round(float(np.percentile(boot, q)), 2) for q in (5, 95)],
            "paired_t_p": float(stats.ttest_1samp(x, 0).pvalue)}


def _audit(arms):
    flagged, onlist, total, hb = set(), 0, 0, 0
    for r in hbpro._rows(T / "search_log.jsonl"):
        if r["arm"] not in arms:
            continue
        for e in r["log"]:
            for res in e.get("results", []):
                total += 1
                u, t = (res.get("url") or "").lower(), (res.get("title") or "").lower()
                if any(d in u for d in SEARCH_BLOCKED_DOMAINS):
                    onlist += 1
                    flagged.add(r["id"])
                if "healthbench" in u or "healthbench" in t:
                    hb += 1
                    flagged.add(r["id"])
    return {"results_logged": total, "results_on_blocked_list": onlist, "results_mentioning_healthbench": hb,
            "tasks_flagged": len(flagged)}, flagged


def main() -> None:
    ids = json.loads((T / "split.json").read_text())["test"]
    sel = json.loads((DEV / "addendum_selection.json").read_text())
    am, ma = sel["amie_style"]["selected"], sel["mai_dxo_style"]["selected"]
    sc = _hb_scores(T)
    blk = ["compass_pro_full_blk", "opus_search_edit_blk", "amie_style_search_blk"] + [a for a in (am, ma) if a.endswith("_blk")]
    audit, flagged = _audit(set(blk))

    def run(task_ids):
        a1, a2 = _diff(sc, task_ids, "compass_pro_full_blk", am), _diff(sc, task_ids, "compass_pro_full_blk", ma)
        order = sorted([("A1", a1["paired_t_p"]), ("A2", a2["paired_t_p"])], key=lambda x: x[1])
        holm, run_ = {}, 0.0
        for j, (k, p) in enumerate(order):
            run_ = max(run_, min(1.0, (2 - j) * p))
            holm[k] = run_
        for k, d in (("A1", a1), ("A2", a2)):
            d["p_holm"] = holm[k]
            d["lead_shown_after_tuning"] = bool(holm[k] < 0.05 and d["ci95"][0] > 0)
        b = {f"{x}_blk - {x}": _diff(sc, task_ids, f"{x}_blk", x) for x in ("compass_pro_full", "opus_search_edit", "amie_style_search")}
        for d in b.values():
            d["within_4_points_90ci"] = bool(-4 <= d["ci90"][0] and d["ci90"][1] <= 4)
        return {"A1_pipeline_minus_tuned_amie": a1, "A2_pipeline_minus_tuned_mai": a2,
                "A_secondary": {"two_step_minus_tuned_amie": _diff(sc, task_ids, "opus_search_edit_blk", am),
                                "two_step_minus_tuned_mai": _diff(sc, task_ids, "opus_search_edit_blk", ma)},
                "B_blocked_minus_original": b,
                "B_contamination_unlikely": all(d["within_4_points_90ci"] for d in b.values()),
                "B_pipeline_minus_amie_search_both_blocked": _diff(sc, task_ids, "compass_pro_full_blk", "amie_style_search_blk")}

    res = {"registration": "https://osf.io/tnezd", "selected": {"amie_style": am, "mai_dxo_style": ma},
           "selection": sel, "audit": audit, "all_tasks": run(ids)}
    # descriptive: mean length-adjusted score (SE) on the 350 test tasks, per arm (added after registration; no test)
    res["pipeline_blk_minus"] = {x: _diff(sc, ids, "compass_pro_full_blk", x) for x in ("opus_search_edit_blk", am, ma)}
    res["scores"] = {}
    for a in ["compass_pro_full", "compass_pro_full_blk", "opus_search_edit", "opus_search_edit_blk", "amie_style",
              "amie_style_search", "amie_style_search_blk", "mai_dxo_style", ma]:
        v = np.array([sc[(i, a)][1] for i in ids if (i, a) in sc]) * 100
        if len(v):
            res["scores"][a] = {"tasks": len(v), "mean": round(float(v.mean()), 4),
                                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 4)}
    if flagged:
        res["without_flagged_tasks"] = run([i for i in ids if i not in flagged])
    v1 = json.loads((D / "paper2_final/derived/hbpro_test_supplement.json").read_text())["cost"]["comparator_chains"]
    use = [u for u in hbpro._rows(T / "usage.jsonl") if u.get("arm")]
    cost = {a: round(sum(u["in"] / 1e6 * hbpro.PRICE[u["who"]][0] + u["out"] / 1e6 * hbpro.PRICE[u["who"]][1]
                         for u in use if u["arm"] == a) / len(ids), 4) for a in sorted({u["arm"] for u in use})}
    for a in (am, ma):
        if a in v1:
            cost[a] = v1[a]["usd_per_task"]
    ratio = cost["opus_search_edit_blk"] / cost[am]
    res["E_cost_per_task_usd"] = cost
    res["E_two_step_vs_tuned_amie"] = {"cost_ratio": round(ratio, 3), "cost_matched": bool(abs(ratio - 1) <= 0.25),
                                      **res["all_tasks"]["A_secondary"]["two_step_minus_tuned_amie"]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
