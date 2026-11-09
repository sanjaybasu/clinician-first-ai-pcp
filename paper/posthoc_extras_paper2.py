"""Post hoc Paper 2 analyses not covered by paper/supplementary_analyses.py (aggregates only; no API calls).

Writes to compass-pcp-data/paper2_final/derived/:
  hbpro_retest_summary.json   primary vs second run of compass_pro_full and amie_style on the 100 retest tasks
  intake_instructions.json    emergency instructions in agent messages of the history-taking episodes (final check)
  texts_gate_test_and_leak.json  gate decisions on the 949 test texts by physician label; classifier "emergency_now"
                              replies per arm and among the architecture's released replies
Usage: python -m paper.posthoc_extras_paper2
"""

import collections
import json
from pathlib import Path

import numpy as np

import paper.hbpro as h
from paper.disposition_check import check, check_unconditional
from paper.fresh_weights import weights

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
OUT = D / "paper2_final/derived"
T = D / "paper2_final/texts"


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _scores(out: Path) -> dict:
    rows = h.load()
    met = {(g["id"], g["arm"], g["k"]): g["met"] for g in h._rows(out / "grades.jsonl")}
    sc = {}
    for r in h._rows(out / "replies.jsonl"):
        it = rows[r["id"]]["rubric_items"]
        if not all((r["id"], r["arm"], k) in met for k in range(len(it))):
            continue
        pos = sum(i["points"] for i in it if i["points"] > 0)
        s = sum(i["points"] for k, i in enumerate(it) if met[(r["id"], r["arm"], k)]) / pos
        sc[(r["id"], r["arm"])] = s - 2.94e-5 * (len(r["text"]) - 2000)
    return sc


def retest() -> dict:
    P, R = _scores(D / "hbpro_test"), _scores(D / "hbpro_retest")
    ids = sorted({i for i, a in R if a == "compass_pro_full"} & {i for i, a in R if a == "amie_style"})
    dp = np.array([P[(i, "compass_pro_full")] - P[(i, "amie_style")] for i in ids])
    dr = np.array([R[(i, "compass_pro_full")] - R[(i, "amie_style")] for i in ids])
    rng = np.random.default_rng(20261108)

    def ci(x):
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [round(100 * float(np.percentile(b, 2.5)), 1), round(100 * float(np.percentile(b, 97.5)), 1)]

    out = {"n": len(ids), "primary_run_diff_same_tasks": round(100 * float(dp.mean()), 1), "primary_ci": ci(dp),
           "second_run_diff": round(100 * float(dr.mean()), 1), "second_ci": ci(dr),
           "mean_of_two_runs_diff": round(100 * float(((dp + dr) / 2).mean()), 1), "mean_ci": ci((dp + dr) / 2)}
    for a in ("compass_pro_full", "amie_style"):
        x, y = np.array([P[(i, a)] for i in ids]), np.array([R[(i, a)] for i in ids])
        out[f"{a}_run1"], out[f"{a}_run2"] = round(100 * float(x.mean()), 1), round(100 * float(y.mean()), 1)
        out[f"{a}_task_corr"] = round(float(np.corrcoef(x, y)[0, 1]), 3)
    return out


def intake() -> dict:
    c = collections.defaultdict(lambda: {"messages": 0, "any_instruction": 0, "unconditional_instruction": 0})
    for r in _rows(D / "paper2_dev/intake_eval/episodes.jsonl"):
        for m in r["messages"]:
            if m.get("role") in ("assistant", "agent"):
                t, d = m.get("content") or "", c[r["arm"]]
                d["messages"] += 1
                d["any_instruction"] += bool(check(t))
                d["unconditional_instruction"] += bool(check_unconditional(t))
    return {"source": "paper2_dev/intake_eval/episodes.jsonl; paper.disposition_check (final policy)", "arms": dict(c)}


def gate_and_leak() -> dict:
    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl")}
    th = json.loads((T / "gate_thresholds.json").read_text())
    g = {x["id"]: x for x in _rows(T / "gate_scores.jsonl")}
    page = {i: v["emergency"] >= th["t_e"] for i, v in g.items()}
    review = {i: (not page[i]) and v["safety"] >= th["t_s"] for i, v in g.items()}
    leak = {"n_released": 0, "emergency_now": 0, "possible_emergency": 0}
    per, n = collections.Counter(), collections.Counter()
    for r in _rows(T / "replies.jsonl"):
        u = urg[(r["prompt_id"], r["repeat"], r["arm"])]
        n[r["arm"]] += 1
        per[r["arm"]] += u == "emergency_now"
        if r["arm"] == "raw_llm" and not page[r["prompt_id"]] and not check_unconditional(r["text"] or ""):
            leak["n_released"] += 1
            if u in leak:
                leak[u] += 1
    leak["emergency_now_per_1000_released"] = round(1000 * leak["emergency_now"] / leak["n_released"], 1)
    cw = {x["sample_id"]: x for x in _rows(D / "fresh_set/crosswalk.jsonl")}
    lab = [(cw[x["sample_id"]]["id"], x) for x in _rows(D / "fresh_set/labels_basu.jsonl")]
    w = weights()

    def gated(sel, emergency_only):
        ids = [i for i, x in lab if sel(x)]
        return [len(ids), sum(page[i] if emergency_only else (page[i] or review[i]) for i in ids)]

    tot = sum(w.values())
    return {"released_raw_llm_replies": leak,
            "gate_test": {"call911_paged": gated(lambda x: x["action"] == "Call 911/988", True),
                          "hazard_contact_doctor_or_911_gated": gated(
                              lambda x: x["hazard"] == 1 and x["action"] in ("Call 911/988", "Contact Doctor"), False)},
            "weighted_page_share": round(sum(w[i] for i, _ in lab if page[i]) / tot, 4),
            "weighted_review_share": round(sum(w[i] for i, _ in lab if review[i]) / tot, 4),
            "classifier_emergency_now_all_replies": {a: {"replies": n[a], "emergency_now": per[a],
                                                         "per_1000": round(1000 * per[a] / n[a], 1)} for a in n}}


def main() -> None:
    for name, fn in [("hbpro_retest_summary.json", retest), ("intake_instructions.json", intake),
                     ("texts_gate_test_and_leak.json", gate_and_leak)]:
        new = fn()
        old = json.loads((OUT / name).read_text()) if (OUT / name).exists() else None
        (OUT / name).write_text(json.dumps(new, indent=1))
        print(name, "unchanged" if old == new else "written (differs from previous file)")


if __name__ == "__main__":
    main()
