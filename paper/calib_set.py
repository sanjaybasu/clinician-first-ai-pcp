"""Physician-labeled calibration set for the conformal ask-or-answer rule (development data, never test data).

Same two-phase design and frozen screen as paper/fresh_set.py, on incoming Compass texts from before the fresh test
window (before 2025-12-01). Contacts who have any text in the fresh test set's pool are excluded, so no person appears in
both. 100 random texts plus 300 screen-positive texts, one per contact. Sample ids start with C. Only counts are printed;
text stays in custodian storage. Usage:
  python -m paper.calib_set all --out <compass-pcp-data>/calib_set --fresh <compass-pcp-data>/fresh_set
"""

import argparse
import csv
import json
import random
from pathlib import Path

from paper import fresh_set as fs

fs.WINDOW = ("2024-01-01", "2025-12-01")
fs.N_RANDOM, fs.N_ENRICHED, fs.SEED = 100, 300, 20261108


def exclude(out: Path, fresh: Path) -> None:
    gone = {r["contact_id"] for r in fs._load(fresh / "pool.jsonl")}
    rows = fs._load(out / "pool.jsonl")
    keep = [r for r in rows if r["contact_id"] not in gone]
    with (out / "pool.jsonl").open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"pool_messages": len(rows), "kept": len(keep), "excluded_test_contacts": len(rows) - len(keep)}))


def export(out: Path) -> None:
    msgs = {r["id"]: r for r in fs._load(out / "pool.jsonl")}
    rows = fs._load(out / "sample.jsonl")
    random.Random(fs.SEED + 2).shuffle(rows)
    with (out / "label_sheet.csv").open("w", newline="") as fh, (out / "crosswalk.jsonl").open("w") as cw:
        w = csv.writer(fh)
        w.writerow(["sample_id", "message", "hazard (0/1)", "hazard_category", "action", "notes"])
        for k, r in enumerate(rows, 1):
            sid = f"C{k:04d}"
            w.writerow([sid, msgs[r["id"]]["message"], "", "", "", ""])
            cw.write(json.dumps({"sample_id": sid, **r}) + "\n")
    print(json.dumps({"label_sheet_rows": len(rows)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pull", "exclude", "sample", "screen", "screen_random", "enrich", "export", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = ["pull", "exclude", "sample", "screen", "screen_random", "enrich", "export"] if a.step == "all" else [a.step]
    for s in steps:
        if s == "pull":
            fs.pull(a.out)
        elif s == "exclude":
            exclude(a.out, a.fresh)
        elif s == "sample":
            fs.sample(a.out)
        elif s == "screen":
            fs.screen(a.out, a.threads)
        elif s == "screen_random":
            fs.screen(a.out, a.threads, "random_ids", "screen_random.jsonl")
        elif s == "enrich":
            fs.enrich(a.out)
        else:
            export(a.out)


if __name__ == "__main__":
    main()
