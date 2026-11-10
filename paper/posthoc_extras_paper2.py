"""Post hoc Paper 2 analyses not covered by paper/supplementary_analyses.py (aggregates only; no API calls).

Writes to compass-pcp-data/paper2_final/derived/:
  hbpro_retest_summary.json   primary vs second run of compass_pro_full and amie_style on the 100 retest tasks
  intake_instructions.json    emergency instructions in agent messages of the history-taking episodes (final check)
  texts_gate_test_and_leak.json  gate decisions on the 949 test texts by physician label; classifier "emergency_now"
                              replies per arm and among the architecture's released replies
  texts_routing_endpoints.json   the architecture's escalations split into routing to a clinician (page, review, held
                              reply) and reply advice; withheld-reply breakdown; the alternative endpoint "physician action
                              label contact a doctor or call 911/988" (any hazard category) for every arm, with CIs
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


def routing_endpoints(labels: str = "labels_basu.jsonl") -> dict:
    from compass_pcp.harness.cohort_a import label
    from paper.sealed_comparators import URGENCY

    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl")}
    th = json.loads((T / "gate_thresholds.json").read_text())
    g = {x["id"]: x for x in _rows(T / "gate_scores.jsonl")}
    page = {i: v["emergency"] >= th["t_e"] for i, v in g.items()}
    review = {i: (not page[i]) and v["safety"] >= th["t_s"] for i, v in g.items()}
    text = {(r["prompt_id"], r["repeat"], r["arm"]): r["text"] or "" for r in _rows(T / "replies.jsonl")}
    cw = {x["sample_id"]: x for x in _rows(D / "fresh_set/crosswalk.jsonl")}
    w8 = weights()
    lab = {}
    for r in _rows(D / "fresh_set" / labels):
        i = cw[r["sample_id"]]["id"]
        lb = label({"detection_truth": r["hazard"], "hazard_category": r["category"], "action_truth": r["action"]})
        lab[i] = {"sp": lb["safety_positive"], "nohaz": r["hazard"] != 1,
                  "doc": r["action"] in ("Contact Doctor", "Call 911/988")}
    ids = sorted(lab)
    w = np.array([w8[i] for i in ids])
    sp, nh, doc = (np.array([lab[i][k] for i in ids]) for k in ("sp", "nohaz", "doc"))

    def adv(arm, i, k):
        u = urg.get((i, k, arm))
        return u is not None and URGENCY.index(u) >= 1

    def ours(i, routed_only):
        v = []
        for k in range(3):
            routed = page[i] or review[i] or bool(check_unconditional(text.get((i, k, "raw_llm"), "")))
            v.append(routed if routed_only else (routed or adv("raw_llm", i, k)))
        return float(np.mean(v))

    est = {"ours": np.array([ours(i, False) for i in ids]), "ours_routed_only": np.array([ours(i, True) for i in ids])}
    for arm in ("amie_style", "mai_dxo_style", "raw_llm", "v22"):
        est[arm] = np.array([np.mean([adv(arm, i, k) for k in range(3)]) for i in ids])
        if arm != "raw_llm":  # the same gate and deployable check added to each comparator writer (post hoc)
            est[arm + "+layer"] = np.array([1.0 if (page[i] or review[i]) else np.mean(
                [adv(arm, i, k) or bool(check_unconditional(text.get((i, k, arm), ""))) for k in range(3)]) for i in ids])
    from paper.disposition_check import check as check_any
    est["ours_registered_check"] = np.array([1.0 if (page[i] or review[i]) else np.mean(
        [adv("raw_llm", i, k) or bool(check_any(text.get((i, k, "raw_llm"), ""))) for k in range(3)]) for i in ids])
    est["gate_alone"] = np.array([1.0 if (page[i] or review[i]) else 0.0 for i in ids])
    rng = np.random.default_rng(20261108)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    wm = lambda x, m, b: float((x[b][m[b]] * w[b][m[b]]).sum() / w[b][m[b]].sum())  # noqa: E731
    full = np.arange(len(ids))

    def ci(x, m):
        v = [wm(x, m, b) for b in B]
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

    out = {"definition": {"routed": "paging gate page or review, or the reply held by check_unconditional",
                          "doctor_call_endpoint": "physician action label Contact Doctor or Call 911/988, any hazard "
                                                  "category (incl. 'other'); weighted (all such texts are screen-positive)"}}
    for k, x in est.items():
        out[k] = {"sens_safety_positive": round(wm(x, sp, full), 3), "sens_safety_positive_ci": ci(x, sp),
                  "sens_doctor_call": round(wm(x, doc, full), 3), "sens_doctor_call_ci": ci(x, doc),
                  "per1000_no_hazard": round(1000 * wm(x, nh, full), 1)}
    for b_arm in ("amie_style", "mai_dxo_style", "v22"):
        d = est["ours"] - est[b_arm]
        v = [wm(d, doc, b) for b in B]
        out[f"ours - {b_arm} (doctor_call)"] = {"diff": round(wm(d, doc, full), 3),
                                                "ci": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
    out["n"] = {"safety_positive": int(sp.sum()), "doctor_call": int(doc.sum()), "no_hazard": int(nh.sum())}
    out["withheld"] = {"texts_paged": int(sum(page.values())), "texts_review": int(sum(review.values())),
                       "replies_withheld_by_page": 3 * int(sum(page.values())),
                       "replies_held_by_check_not_paged": sum(1 for i in ids for k in range(3) if not page[i] and
                                                              check_unconditional(text.get((i, k, "raw_llm"), "")))}
    out["withheld"]["total"] = out["withheld"]["replies_withheld_by_page"] + out["withheld"]["replies_held_by_check_not_paged"]
    lk = gate_and_leak()["classifier_emergency_now_all_replies"]
    rel = gate_and_leak()["released_raw_llm_replies"]
    out["emergency_now_reaching_patient_per_1000_all_replies"] = {
        "ours": round(1000 * rel["emergency_now"] / lk["raw_llm"]["replies"], 1),
        **{a: lk[a]["per_1000"] for a in ("amie_style", "mai_dxo_style", "raw_llm", "v22")}}
    for arm in ("amie_style", "mai_dxo_style", "v22"):  # released by the same gate and deployable check
        n_em = sum(1 for i in ids for k in range(3) if not page[i] and urg.get((i, k, arm)) == "emergency_now"
                   and not check_unconditional(text.get((i, k, arm), "")))
        out["emergency_now_reaching_patient_per_1000_all_replies"][arm + "+layer"] = round(1000 * n_em / (3 * len(ids)), 1)
    n_em = sum(1 for i in ids for k in range(3) if not page[i] and urg.get((i, k, "raw_llm")) == "emergency_now"
               and not check_any(text.get((i, k, "raw_llm"), "")))
    out["emergency_now_reaching_patient_per_1000_all_replies"]["ours_registered_check"] = round(1000 * n_em / (3 * len(ids)), 1)
    return out


def main() -> None:
    for name, fn in [("texts_routing_endpoints.json", routing_endpoints),
                     ("texts_routing_endpoints_morgan.json", lambda: routing_endpoints("labels_morgan.jsonl")),
                     ("hbpro_retest_summary.json", retest), ("intake_instructions.json", intake),
                     ("texts_gate_test_and_leak.json", gate_and_leak)]:
        new = fn()
        old = json.loads((OUT / name).read_text()) if (OUT / name).exists() else None
        (OUT / name).write_text(json.dumps(new, indent=1))
        print(name, "unchanged" if old == new else "written (differs from previous file)")


if __name__ == "__main__":
    main()
