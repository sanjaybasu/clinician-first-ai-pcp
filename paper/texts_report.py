"""Real-text test report for the clinician-first architecture (Paper 2; fresh 949 texts; aggregates only).

Inputs (custodian storage, compass-pcp-data):
  fresh_set/labels_basu.jsonl, fresh_set/crosswalk.jsonl (sample_id -> id, stratum, p_inclusion)
  <run>/replies.jsonl (prompt_id, repeat, arm, text) and <run>/urgency.jsonl (prompt_id, repeat, arm, urgency)
  --gate: JSONL {id, emergency, safety} stacked paging-gate scores; --thresholds: JSON {t_e, t_s}
Arms reported: each replying arm as run, and "clinician_first" = paging gate + the raw_llm writer + the clinician-first
check. For clinician_first a text is escalated when the gate pages (score_emergency >= t_e), the gate sends it to
clinician review (score_safety >= t_s), the writer's reply advises routine follow-up or more urgent care, or the reply
holds an AI-originated emergency instruction (held by paper/disposition_check.py and sent to the page route).

Measures (per text, averaged over repeats; Hájek two-phase weights, paper/fresh_weights.py; bootstrap 95% CI over texts, 2,000 resamples):
  hazard sensitivity: escalated among safety-positive texts (cohort_a.label definition)
  call-911 capture: escalated among texts labeled "Call 911/988" (unweighted count; too few for a weighted rate)
  escalation without hazard: escalated per 1,000 texts with no hazard
  AI emergency instructions: replies per 1,000 with an emergency-room or 911/988 instruction outside a signed route
Prints numbers only.
Usage: python -m paper.texts_report --run <dir> --gate <jsonl> --thresholds <json> [--out report.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from compass_pcp.harness.cohort_a import label
from paper.disposition_check import check, check_unconditional
from paper.fresh_weights import weights
from paper.sealed_comparators import URGENCY

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
WRITER = "raw_llm"


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gate", type=Path)
    ap.add_argument("--thresholds", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--policy", choices=["any", "unconditional"], default="any",
                    help="clinician-first check: hold any hit (registered) or unconditional hits only (deployable)")
    a = ap.parse_args()
    chk = check if a.policy == "any" else check_unconditional
    w2 = weights()
    cw = {r["sample_id"]: r for r in _rows(D / "fresh_set/crosswalk.jsonl")}
    lab = {}
    for r in _rows(D / "fresh_set/labels_basu.jsonl"):
        c = cw[r["sample_id"]]
        lab[c["id"]] = {**label({"detection_truth": r["hazard"], "hazard_category": r["category"],
                                 "action_truth": r["action"]}), "w": w2[c["id"]], "hazard": r["hazard"] == 1}
    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(a.run / "urgency.jsonl")}
    esc, instr = {}, {}  # (arm, id) -> list of 0/1 over repeats
    for r in _rows(a.run / "replies.jsonl"):
        i, k, arm = r["prompt_id"], r["repeat"], r["arm"]
        u = urg.get((i, k, arm))
        e = u is not None and URGENCY.index(u) >= URGENCY.index("routine_followup")
        hit = bool(chk(r["text"] or ""))
        esc.setdefault((arm, i), []).append(e)
        instr.setdefault((arm, i), []).append(hit)
        if arm == WRITER and a.gate:
            esc.setdefault(("clinician_first", i), []).append(e or hit)
            instr.setdefault(("clinician_first", i), []).append(False)  # held replies never reach the patient
    held = {i: np.mean(v) for (arm, i), v in instr.items() if arm == WRITER}
    if a.gate:
        th = json.loads(a.thresholds.read_text())
        g = {x["id"]: x for x in _rows(a.gate)}
        for (arm, i), v in list(esc.items()):
            if arm == "clinician_first":
                paged = g[i]["emergency"] >= th["t_e"] or g[i]["safety"] >= th["t_s"]
                esc[(arm, i)] = [True] * len(v) if paged else v
    rng = np.random.default_rng(20261108)
    res = {}
    for arm in sorted({k[0] for k in esc}):
        ids = sorted(i for (x, i) in esc if x == arm and i in lab)
        e = np.array([np.mean(esc[(arm, i)]) for i in ids])
        h = np.array([np.mean(instr[(arm, i)]) for i in ids])
        w = np.array([lab[i]["w"] for i in ids])
        sp = np.array([lab[i]["safety_positive"] for i in ids])
        nh = np.array([not lab[i]["hazard"] for i in ids])
        c9 = np.array([lab[i]["call_911"] for i in ids])

        def wmean(x, m, ww):
            return float((x[m] * ww[m]).sum() / ww[m].sum()) if m.any() else float("nan")

        def ci(f):
            bs = []
            for _ in range(2000):
                b = rng.integers(0, len(ids), len(ids))
                bs.append(f(b))
            return [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]

        res[arm] = {
            "texts": len(ids), "safety_positive": int(sp.sum()), "call_911": int(c9.sum()), "no_hazard": int(nh.sum()),
            "hazard_sensitivity_weighted": round(wmean(e, sp, w), 3),
            "hazard_sensitivity_ci": ci(lambda b: wmean(e[b], sp[b], w[b])),
            "hazard_sensitivity_unweighted": round(float(e[sp].mean()), 3),
            "call_911_escalated_mean": round(float(e[c9].sum()), 2),
            "escalation_no_hazard_per_1000_weighted": round(1000 * wmean(e, nh, w), 1),
            "escalation_no_hazard_ci": [round(1000 * x, 1) for x in ci(lambda b: wmean(e[b], nh[b], w[b]))],
            "ai_emergency_instructions_per_1000_replies": round(1000 * float(h.mean()), 1),
        }
    if held:
        res["_writer_replies_held_per_1000"] = round(1000 * float(np.mean(list(held.values()))), 1)
    out = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(out)
    print(out)


if __name__ == "__main__":
    main()
