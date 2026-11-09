"""Load the physician-returned sheets into canonical id-keyed files (no message text is copied).

Provenance (S. Basu, 2026-11-08): the authors affirm every returned label, note and reply is human-written.
Missing cells stay missing; nothing is filled by a model or by rule.
Writes, under compass-pcp-data:
  lighthouse_set/labels_basu.jsonl         sample_id, hazard, category, action, labeler
  fresh_set/clinician_replies/replies.jsonl sample_id, writer, reply, minutes
  calib_set/replies.jsonl                   sample_id, writer, reply, minutes
  calib_set/labels_morgan.jsonl             sample_id, hazard, category, action, labeler
  fresh_set/labels_basu.jsonl, fresh_set/labels_morgan.jsonl (949 and 200 test labels; same fields)
"""

import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
PROV = "author-affirmed human, S. Basu 2026-11-08"


def _v(x):
    if pd.isna(x):
        return None
    return x.strip() if isinstance(x, str) else x


def _write(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(p.relative_to(D), len(rows))


def labels(xlsx: Path, labeler: str) -> list[dict]:
    df = pd.read_excel(xlsx)
    return [{"sample_id": r["sample_id"], "hazard": None if pd.isna(r["hazard (0/1)"]) else int(r["hazard (0/1)"]),
             "category": _v(r["hazard_category"]), "action": _v(r["action"]), "labeler": labeler, "provenance": PROV}
            for _, r in df.iterrows()]


def replies(xlsx: Path, writer: str, col: str, mcol: str | None) -> list[dict]:
    df = pd.read_excel(xlsx)
    return [{"sample_id": r["sample_id"], "writer": writer, "reply": _v(r[col]),
             "minutes": None if mcol is None or pd.isna(r[mcol]) else float(r[mcol]), "provenance": PROV}
            for _, r in df.iterrows()]


def main() -> None:
    _write(D / "lighthouse_set/labels_basu.jsonl", labels(D / "lighthouse_set/returns/basu_labels.xlsx", "basu"))
    cr = D / "fresh_set/clinician_replies"
    rows = replies(cr / "basu_completed.xlsx", "basu", "your reply (one text message)", None)
    for w in "ABC":
        rows += replies(cr / f"returned/writer_{w}.xlsx", f"writer_{w}", "your reply (one text message)", "minutes")
    _write(cr / "replies.jsonl", rows)
    c = D / "calib_set"
    _write(c / "replies.jsonl", replies(c / "returns/basu_calibration_replies_no_labels.xlsx", "basu", "reply", "minutes")
           + replies(c / "returns/morgan_returned_replies_no_labels.xlsx", "morgan", "clinician_reply", None))
    _write(c / "labels_morgan.jsonl", labels(c / "labels/morgan_corrected.xlsx", "morgan"))
    f = D / "fresh_set"
    _write(f / "labels_basu.jsonl", labels(f / "labels/basu_completed.xlsx", "basu"))
    _write(f / "labels_morgan.jsonl", labels(f / "labels/morgan_completed.xlsx", "morgan"))


if __name__ == "__main__":
    main()
