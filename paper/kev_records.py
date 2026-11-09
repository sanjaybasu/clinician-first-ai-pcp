"""Kev fine-tune records from teacher labels (development data only).

Reads <teacher>/items.jsonl and <teacher>/labels.jsonl (paper/teacher_labels.py). Uses only the unlabeled Compass texts
(source "sms"); the 2,000 physician-labeled development messages stay out of training and are the evaluation set.
Each text gives two records: one with the chart in the state and one without (chart dropout), so the same model can
be scored both ways. Labels: the teachers' shared answer, or a 50/50 soft target when they disagree (the label is then
the Opus answer). Split by contact-free text id into train / calibration / development (80 / 10 / 10, seeded); both
copies of a text fall in the same split. Writes <out>/{train,calibration,development}.jsonl. Prints counts only.
"""

import argparse
import json
import random
from pathlib import Path

from paper.system1_questions import QUESTIONS, state

MAX_STATE_CHARS = 4500


def record(chart, text, labels, target):
    st = state(chart, text)
    if len(st) > MAX_STATE_CHARS:  # trim the chart, never the message
        keep = MAX_STATE_CHARS - len(st) + len(chart or "")
        st = state((chart or "")[: max(keep, 0)], text)
    qs = {}
    for q, spec in QUESTIONS.items():
        d = {"type": spec["type"], "instructions": spec["instructions"]}
        if spec["type"] == "choice":
            d["criteria"] = spec["criteria"]
            d["label"] = labels[q]
        else:
            d["label"] = bool(labels[q])
        if len(target[q]) > 1:
            d["target"] = {str(k).lower(): v for k, v in target[q].items()}
        qs[q] = d
    return {"state": st, "questions": qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    items = {json.loads(x)["id"]: json.loads(x) for x in (a.teacher / "items.jsonl").read_text().splitlines()}
    labs = [json.loads(x) for x in (a.teacher / "labels.jsonl").read_text().splitlines()]
    labs = [r for r in labs if r["source"] == "sms"]
    rng = random.Random(a.seed)
    rng.shuffle(labs)
    n = len(labs)
    cut = {"train": (0, int(0.8 * n)), "calibration": (int(0.8 * n), int(0.9 * n)), "development": (int(0.9 * n), n)}
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, (lo, hi) in cut.items():
        recs = []
        for r in labs[lo:hi]:
            it = items[r["id"]]
            labels = r["teachers"]["opus"]
            if it.get("chart"):
                recs.append(record(it["chart"], it["text"], labels, r["consensus"]))
            recs.append(record(None, it["text"], labels, r["consensus"]))
        rng.shuffle(recs)
        p = a.out / f"{split}.jsonl"
        p.write_text("".join(json.dumps(x) + "\n" for x in recs))
        p.chmod(0o600)
        counts[split] = len(recs)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
