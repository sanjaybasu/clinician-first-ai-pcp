"""Cross-fitted Kev fine-tune data from the physician labels on the 2,000 development messages (development data only).

Questions use the physicians' own constructs, so the fine-tune learns their convention:
  hazard    noul    detection_truth
  action    choice  action_truth (None, Routine Follow-up, Contact Doctor, Call 911/988)
  category  choice  hazard_category ("" -> none)
Five folds, stratified on the safety outcome (seeded). For fold k: development = fold k; the other four folds are split
90/10 into train and calibration (temperature only). Every message gives a record with the chart in the state and one
without (chart dropout); the record id is "<message_id>|chart" or "<message_id>|msg". Writes <out>/fold{k}/... and
<out>/all/{development.jsonl} (every message, for scoring a model trained elsewhere). Prints counts only.
"""

import argparse
import json
import random
from pathlib import Path

from paper.system1_questions import state

ACTION = {"None": "No clinical action is needed", "Routine Follow-up": "Routine follow-up by the care team",
          "Contact Doctor": "A doctor or clinician should contact the person", "Call 911/988": "Call 911 or 988 now"}
CATS = ["none", "other_hazard", "environmental_housing_safety", "behavioral_other", "med_safety", "falls_mobility",
        "behavioral_suicidality", "substance_use", "cardiac_chest_pain", "maternity_ob", "respiratory", "neuro_stroke",
        "bleeding", "allergy_anaphylaxis", "endocrine_diabetes", "pediatric", "violence_abuse"]
QUESTIONS = {
    "hazard": {"type": "noul", "instructions": "Does this message contain a health or safety hazard that a clinician "
                                               "should know about?"},
    "action": {"type": "choice", "criteria": ACTION, "instructions": "What should the care team do about this message?"},
    "category": {"type": "choice", "criteria": {c: None for c in CATS},
                 "instructions": "Which kind of hazard does the message raise, if any?"},
}
MAX_STATE_CHARS = 4500


def safety(r) -> int:
    return int((r["detection_truth"] == 1 and r["hazard_category"] != "other_hazard") or "911" in str(r["action_truth"]))


def records(r, chart):
    text = r["prompt"] if isinstance(r["prompt"], str) else r["prompt"][-1]["content"]
    cat = r["hazard_category"] or "none"
    out = []
    for tag, ch in (("chart", chart), ("msg", None)):
        if tag == "chart" and not ch:
            continue
        st = state(ch, text)
        if len(st) > MAX_STATE_CHARS:
            st = state(ch[: max(MAX_STATE_CHARS - len(st) + len(ch), 0)], text)
        qs = {"hazard": {**QUESTIONS["hazard"], "label": bool(r["detection_truth"] == 1)},
              "action": {**QUESTIONS["action"], "label": str(r["action_truth"])},
              "category": {**QUESTIONS["category"], "label": cat if cat in CATS else "other_hazard"}}
        out.append({"state": st, "questions": qs, "_meta": {"id": f"{r['message_id']}|{tag}"}})
    return out


def write(p: Path, recs) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in recs))
    p.chmod(0o600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    rows = json.loads(a.cases.read_text())
    charts = {json.loads(x)["id"]: json.loads(x)["chart"] for x in a.chart.read_text().splitlines()}
    rng = random.Random(a.seed)
    pos = [r for r in rows if safety(r)]
    neg = [r for r in rows if not safety(r)]
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold = {}
    for lst in (pos, neg):
        for i, r in enumerate(lst):
            fold[str(r["message_id"])] = i % a.folds
    recs = {str(r["message_id"]): records(r, charts.get(str(r["message_id"]))) for r in rows}
    write(a.out / "all" / "development.jsonl", [x for v in recs.values() for x in v])
    counts = {}
    for k in range(a.folds):
        dev = [x for i, v in recs.items() if fold[i] == k for x in v]
        rest = [i for i in recs if fold[i] != k]
        rng.shuffle(rest)
        ncal = len(rest) // 10
        cal = [x for i in rest[:ncal] for x in recs[i]]
        tr = [x for i in rest[ncal:] for x in recs[i]]
        rng.shuffle(tr)
        for name, lst in (("train", tr), ("calibration", cal), ("development", dev)):
            write(a.out / f"fold{k}" / f"{name}.jsonl", lst)
        counts[f"fold{k}"] = {"train": len(tr), "calibration": len(cal), "development": len(dev)}
    (a.out / "folds.json").write_text(json.dumps(fold))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
