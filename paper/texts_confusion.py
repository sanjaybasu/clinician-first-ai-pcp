"""Confusion matrices and diagnostic-accuracy measures for escalation on the 949 test texts (aggregates only).

For each system (as tested, and each agent with the safety layer), a text's escalation e in [0, 1] is the share of its
three replies that escalated (paper.supplementary_analyses._per_repeat). Two reference standards from S.B.'s labels:
  safety hazard: positives = texts with a safety hazard (79); negatives = texts without a hazard (571); texts with a
    hazard in the "other" category (299) enter neither group, as in the main analysis.
  doctor's call: positives = texts labeled "Call 911/988" or "Contact Doctor" (53); negatives = all other texts (896).
Expected counts (unweighted, summed over texts) and two-phase weighted sensitivity, specificity, positive and negative
predictive value and overall escalation share, with bootstrap 95% CIs (2,000 resamples of texts, seed 20261108).
Appropriateness: replies that reached the patient and that the reply classifier labeled "emergency_now", by the
text's labeled action. Transfer: the emergency screen's doctor's-call capture without the test texts whose possible
sender also sent a calibration text (fresh_set/calib_overlap_ids.json).
Out: paper2_final/derived/texts_confusion.json. Usage: python -m paper.texts_confusion
"""

import collections
import json
from pathlib import Path

import numpy as np

from paper.disposition_check import check_unconditional
from paper.supplementary_analyses import WRITER, _per_repeat, _texts_data

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
T = D / "paper2_final/texts"
OUT = D / "paper2_final/derived/texts_confusion.json"
ACTIONS = ("Call 911/988", "Contact Doctor", "Routine Follow-up", "None")


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _measures(e, pos, neg, w, B):
    def f(ix):
        ee, pp, nn, ww = e[ix], pos[ix], neg[ix], w[ix]
        tp, fn = (ww * ee)[pp].sum(), (ww * (1 - ee))[pp].sum()
        fp, tn = (ww * ee)[nn].sum(), (ww * (1 - ee))[nn].sum()
        allw = ww[pp | nn].sum()
        return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "ppv": tp / (tp + fp),
                "npv": tn / (tn + fn), "escalated_share": (tp + fp) / allw}
    est = f(np.arange(len(e)))
    boots = [f(b) for b in B]
    out = {k: {"est": round(float(v), 3),
               "ci": [round(float(np.nanpercentile([x[k] for x in boots], q)), 3) for q in (2.5, 97.5)]}
           for k, v in est.items()}
    out["expected_counts"] = {"tp": round(float(e[pos].sum()), 1), "fn": round(float((1 - e[pos]).sum()), 1),
                              "fp": round(float(e[neg].sum()), 1), "tn": round(float((1 - e[neg]).sum()), 1),
                              "positives": int(pos.sum()), "negatives": int(neg.sum())}
    return out


def main() -> None:
    basu, _, rep, paged, page_e = _texts_data()
    E = _per_repeat(rep, paged)
    for arm in ("amie_style", "mai_dxo_style", "v22"):
        E[f"{arm}+layer"] = {i: [True] * len(v) if paged[i] else [v[k][0] or v[k][2] for k in sorted(v)]
                             for (x, i), v in rep.items() if x == arm}
    ids = sorted(i for i in E["ours"] if i in basu)
    w = np.array([basu[i]["w"] for i in ids])
    sp = np.array([basu[i]["safety_positive"] for i in ids])
    nh = np.array([not basu[i]["hazard"] for i in ids])
    doc = np.array([basu[i]["action"] in ("Call 911/988", "Contact Doctor") for i in ids])
    rng = np.random.default_rng(20261108)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    arms = ["ours", "gate_alone", "amie_style", "mai_dxo_style", WRITER, "v22",
            "amie_style+layer", "mai_dxo_style+layer", "v22+layer"]
    res = {"definition": __doc__.split("Out:")[0].strip(), "systems": {}}
    for arm in arms:
        e = np.array([float(np.mean(E[arm][i])) for i in ids])
        res["systems"][arm] = {"safety_hazard": _measures(e, sp, nh, w, B),
                               "doctor_call": _measures(e, doc, ~doc, w, B)}
    # appropriateness of emergency advice that reached the patient
    urg = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl")}
    text = {(r["prompt_id"], r["repeat"], r["arm"]): r["text"] or "" for r in _rows(T / "replies.jsonl")}
    act = {i: basu[i]["action"] for i in ids}
    em = {}
    for arm, layer in (("raw_llm", True), ("amie_style", False), ("mai_dxo_style", False), ("raw_llm", False),
                       ("v22", False), ("amie_style", True), ("mai_dxo_style", True), ("v22", True)):
        name = "ours" if (arm == WRITER and layer) else (f"{arm}+layer" if layer else arm)
        c = collections.Counter()
        for i in ids:
            for k in range(3):
                if urg.get((i, k, arm)) != "emergency_now":
                    continue
                if layer and (page_e[i] or check_unconditional(text.get((i, k, arm), ""))):
                    continue  # withheld: the patient gets the fixed alert text instead
                c[act[i]] += 1
        em[name] = {a: c[a] for a in ACTIONS}
        em[name]["total"] = sum(c.values())
    res["emergency_now_reaching_patient_by_label"] = em
    # reply-level advice on the texts labeled "call 911 or 988" (before any filter), and emergency screen alerts
    e911 = [i for i in ids if act[i] == "Call 911/988"]
    res["call_911_texts"] = {"texts": len(e911), "replies_per_system": 3 * len(e911),
                             "emergency_screen_alerts": int(sum(page_e[i] for i in e911)),
                             "reply_urgency": {a: dict(collections.Counter(urg[(i, k, a)] for i in e911 for k in range(3)))
                                               for a in ("amie_style", "mai_dxo_style", "raw_llm", "v22")}}
    res["emergency_now_note"] = ("counts of replies over 2,847 per system; a withheld reply (alert or held by the "
                                 "filter) does not reach the patient")
    # transfer without overlapping senders
    ov = set(json.loads((D / "fresh_set/calib_overlap_ids.json").read_text()))
    th = json.loads((T / "gate_thresholds.json").read_text())
    g = {x["id"]: x for x in _rows(T / "gate_scores.jsonl")}
    sent = {i: g[i]["emergency"] >= th["t_e"] or g[i]["safety"] >= th["t_s"] for i in g}
    dc = [i for i in ids if basu[i]["hazard"] and basu[i]["action"] in ("Call 911/988", "Contact Doctor")]
    res["transfer"] = {"doctor_call_hazard_texts": len(dc), "sent_to_physician": sum(sent[i] for i in dc),
                       "without_overlap": {"texts": sum(i not in ov for i in dc),
                                           "sent_to_physician": sum(sent[i] for i in dc if i not in ov)},
                       "overlapping_test_texts": len(ov)}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main()
