"""Frozen paging-gate thresholds and Kev-9B gate scores for the 949 fresh test texts (final test; nothing is tuned here).

thresholds: t_e and t_s from the 497-text emergency-enriched development set (paper/paging_gate_497.py; Kev-9B
  compass-phys9b-f0..f4 fold-model mean, message only; conformal risk control threshold fit on all 497 texts):
  t_e on P(Call 911/988) at alpha 0.05 (emergency = action Call 911/988); t_s on P(Contact Doctor or higher) at
  alpha 0.10 (safety = hazard and action Contact Doctor or Call 911/988). -> paper2_final/texts/gate_thresholds.json
records: Kev records for the 949 fresh texts (fresh_set/pool.jsonl restricted to fresh_set/sample.jsonl), message only,
  placeholder labels, record id = crosswalk sample_id. Fresh-set labels are never read.
  Then fit_records.py and the same five Modal evaluations as for the 497 (names compass-fresh949-phys9b-f0..f4).
scores: fold-model mean -> paper2_final/texts/gate_scores.jsonl {id (fresh pool id), emergency, safety}; prints counts and
  the share of texts at or above each frozen threshold (no labels).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.kev_physician_folds import QUESTIONS
from paper.paging_gate_497 import PLACEHOLDER, _jsonl, kev_scores, labels_497, t_unweighted
from paper.system1_questions import state

DATA = Path.home() / "waymark-local/packaging/compass-pcp-data"
FRESH = DATA / "fresh_set"
FINAL = DATA / "paper2_final/texts"
W497 = DATA / "paper2_dev/paging_gate_497/work"
WORK = DATA / "paper2_final/texts/gate_work"


def thresholds() -> None:
    mean, _ = kev_scores(sorted(W497.glob("runs/compass-lh497-phys9b-f*/development/rows.json")))
    ids, Y, _ = labels_497()
    se = np.array([mean[i]["emergency"] for i in ids])
    ss = np.array([mean[i]["safety"] for i in ids])
    out = {"t_e": float(t_unweighted(se, Y["emergency"], 0.05)), "t_s": float(t_unweighted(ss, Y["safety"], 0.10)),
           "alpha_e": 0.05, "alpha_s": 0.10,
           "source": "paper/paging_gate_497.py: Kev-9B physician fine-tunes compass-phys9b-f0..f4 (mean of fold models, "
                     "message only); conformal risk control threshold fit on all 497 texts of the emergency-enriched "
                     "care-team SMS development set (S. Basu labels). t_e on P(Call 911/988), emergency = action Call "
                     "911/988 (104 positives). t_s on P(Contact Doctor or Call 911/988), safety = hazard and action "
                     "Contact Doctor or Call 911/988 (263 positives). Frozen 2026-11-08 before any fresh-set scoring.",
           "score_file": "gate_scores.jsonl (emergency vs t_e, safety vs t_s; page when score >= threshold)"}
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "gate_thresholds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("t_e", "t_s", "alpha_e", "alpha_s")}))


def records() -> None:
    sample = {r["id"] for r in _jsonl(FRESH / "sample.jsonl")}
    sid = {r["id"]: r["sample_id"] for r in _jsonl(FRESH / "crosswalk.jsonl")}
    msg = {}
    with (FRESH / "pool.jsonl").open() as fh:
        for x in fh:
            d = json.loads(x)
            if d["id"] in sample:
                msg[d["id"]] = d["message"]
    recs = [{"state": state(None, msg[i]), "_meta": {"id": f"{sid[i]}|msg"},
             "questions": {q: {**QUESTIONS[q], "label": PLACEHOLDER[q]} for q in ("hazard", "action", "category")}}
            for i in sorted(sample)]
    p = WORK / "raw" / "development.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    p.chmod(0o600)
    print(json.dumps({"records": len(recs), "sample": len(sample)}))


def scores() -> None:
    rows = sorted(WORK.glob("runs/compass-fresh949-phys9b-f*/development/rows.json"))
    if len(rows) != 5:
        raise SystemExit(f"expected 5 rows.json files, found {len(rows)}")
    mean, _ = kev_scores(rows)
    back = {r["sample_id"]: r["id"] for r in _jsonl(FRESH / "crosswalk.jsonl")}
    sample = {r["id"] for r in _jsonl(FRESH / "sample.jsonl")}
    out = sorted(({"id": back[s], "emergency": v["emergency"], "safety": v["safety"]} for s, v in mean.items()),
                 key=lambda r: r["id"])
    if {r["id"] for r in out} != sample:
        raise SystemExit("scored ids do not match fresh_set/sample.jsonl")
    p = FINAL / "gate_scores.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in out))
    p.chmod(0o600)
    th = json.loads((FINAL / "gate_thresholds.json").read_text())
    e = np.array([r["emergency"] for r in out]) >= th["t_e"]
    s = np.array([r["safety"] for r in out]) >= th["t_s"]
    print(json.dumps({"scored": len(out), "emergency_paged": int(e.sum()), "safety_flagged": int(s.sum()),
                      "either": int((e | s).sum()), "share_emergency": round(float(e.mean()), 3),
                      "share_safety": round(float(s.mean()), 3), "share_either": round(float((e | s).mean()), 3)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["thresholds", "records", "scores"])
    {"thresholds": thresholds, "records": records, "scores": scores}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
