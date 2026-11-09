"""Real-text error analysis for Paper 2 (aggregates only; no API calls).

(1) Missed safety hazards by physician-labeled category: per system, the expected number of safety-positive texts not
    escalated (mean over the 3 replies), by category. (2) STARD cross-tabulation: expected escalated / not escalated
    counts by label group (safety-positive, other hazard, no hazard), unweighted and weighted to the sampling pool.
(3) Label groups of all 949 texts. Ours = persona writer + paging gate + check_unconditional (deployable policy),
    the rule of paper2_final/texts/report_posthoc.json.
Out: paper2_final/derived/texts_errors.json.  Usage: python -m paper.texts_errors_paper2
"""

import json
from collections import defaultdict
from pathlib import Path

from paper.fresh_weights import weights as _weights

_W2 = _weights()

from compass_pcp.harness.cohort_a import label
from paper.disposition_check import check_unconditional
from paper.sealed_comparators import URGENCY

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
T = D / "paper2_final/texts"


def _rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    cw = {r["sample_id"]: r for r in _rows(D / "fresh_set/crosswalk.jsonl")}
    lab = {}
    for r in _rows(D / "fresh_set/labels_basu.jsonl"):
        c = cw[r["sample_id"]]
        lb = label({"detection_truth": r["hazard"], "hazard_category": r["category"], "action_truth": r["action"]})
        grp = "safety_positive" if lb["safety_positive"] else ("other_hazard" if r["hazard"] == 1 else "no_hazard")
        lab[c["id"]] = {"grp": grp, "cat": (r["category"] or "none") if lb["safety_positive"] else None,
                        "w": _W2[c["id"]]}
    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl")}
    th = json.loads((T / "gate_thresholds.json").read_text())
    g = {x["id"]: x["emergency"] >= th["t_e"] or x["safety"] >= th["t_s"] for x in _rows(T / "gate_scores.jsonl")}
    esc = defaultdict(list)
    for r in _rows(T / "replies.jsonl"):
        u = urg.get((r["prompt_id"], r["repeat"], r["arm"]))
        e = u is not None and URGENCY.index(u) >= 1
        esc[(r["arm"], r["prompt_id"])].append(e)
        if r["arm"] == "raw_llm":
            esc[("ours", r["prompt_id"])].append(True if g[r["prompt_id"]] else (e or bool(check_unconditional(r["text"] or ""))))
    arms = ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22"]
    out = {"label_groups": {}, "missed_by_category": {}, "crosstab": {}}
    for grp in ["safety_positive", "other_hazard", "no_hazard"]:
        out["label_groups"][grp] = sum(v["grp"] == grp for v in lab.values())
    cats = sorted({v["cat"] for v in lab.values() if v["cat"]})
    out["label_groups"]["safety_positive_by_category"] = {c: sum(v["cat"] == c for v in lab.values()) for c in cats}
    for a in arms:
        m = {}
        for c in cats:
            ids = [i for i, v in lab.items() if v["cat"] == c]
            m[c] = round(sum(1 - sum(esc[(a, i)]) / len(esc[(a, i)]) for i in ids), 2)
        out["missed_by_category"][a] = m
        ct = {}
        for grp in ["safety_positive", "other_hazard", "no_hazard"]:
            ids = [i for i, v in lab.items() if v["grp"] == grp]
            e = sum(sum(esc[(a, i)]) / len(esc[(a, i)]) for i in ids)
            ew = sum(lab[i]["w"] * sum(esc[(a, i)]) / len(esc[(a, i)]) for i in ids)
            W = sum(lab[i]["w"] for i in ids)
            ct[grp] = {"n": len(ids), "escalated_expected": round(e, 1), "not_escalated_expected": round(len(ids) - e, 1),
                       "weighted_pool_total": round(W, 1), "weighted_escalated": round(ew, 1)}
        out["crosstab"][a] = ct
    out["note"] = ("Expected counts are means over 3 replies per text; the gate's decision is the same for every reply. "
                   "Weighted values use 1/p_inclusion (random stratum p = 0.238; enriched p = 1).")
    (D / "paper2_final/derived/texts_errors.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["label_groups"]), json.dumps(out["missed_by_category"]["ours"]),
          json.dumps(out["missed_by_category"]["amie_style"]), json.dumps(out["crosstab"]["ours"]))


if __name__ == "__main__":
    main()
