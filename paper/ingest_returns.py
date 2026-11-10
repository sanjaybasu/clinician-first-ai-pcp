"""Load the physician-returned sheets into canonical id-keyed files (no message text is copied).

Provenance (S. Basu, 2026-11-08): the authors affirm every returned label, note and reply is human-written.
Missing cells stay missing; nothing is filled by a model or by rule.
Writes, under compass-pcp-data:
  lighthouse_set/labels_basu.jsonl         sample_id, hazard, category, action, labeler
  fresh_set/clinician_replies/replies.jsonl sample_id, writer, reply, minutes
  calib_set/replies.jsonl                   sample_id, writer, reply, minutes
  calib_set/labels_morgan.jsonl             sample_id, hazard, category, action, labeler
  fresh_set/labels_basu.jsonl, fresh_set/labels_morgan.jsonl (949 test labels each; same fields plus batch for J.M.)
  fresh_set/a2_rating/ratings.jsonl         rater, text_no, sample_id, reply_label, arm, page_text, scores (no text)
Returned 2026-11-10 (J. Morgan's remaining 749 labels; C. Moher and R. Batniji reply ratings) carry the provenance
PROV_1110: the returned sheets lacked the attestation line, and each rater confirmed it to S. Basu on 2026-11-10.
"""

import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
PROV = "author-affirmed human, S. Basu 2026-11-08"
PROV_1110 = "physician-returned sheet 2026-11-10; rater attestation confirmed to S. Basu 2026-11-10"
URGENT_PAGE_SELF = (
    "This might be an urgent situation. I am sending your message to our physician now, and they will call you back. "
    "If things get worse, call 911. If you are thinking about hurting yourself, call or text 988. This text line is "
    "not an emergency service."
)


def _v(x):
    if pd.isna(x):
        return None
    return x.strip() if isinstance(x, str) else x


def _write(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(p.relative_to(D), len(rows))


def labels(xlsx: Path, labeler: str, prov: str = PROV, **extra) -> list[dict]:
    df = pd.read_excel(xlsx)
    return [{"sample_id": r["sample_id"], "hazard": None if pd.isna(r["hazard (0/1)"]) else int(r["hazard (0/1)"]),
             "category": _v(r["hazard_category"]), "action": _v(r["action"]), "labeler": labeler, "provenance": prov,
             **extra} for _, r in df.iterrows()]


def ratings(xlsx: Path, rater: str, key: dict) -> list[dict]:
    """Decode a returned A2 rating sheet with the hidden key; keep scores only (no patient text, reply or note)."""
    df = pd.read_excel(xlsx)
    disp = [c for c in df.columns if str(c).startswith("advised_disposition")][0]
    out = []
    for _, r in df.iterrows():
        k = key[(int(r["text_no"]), r["reply_label"])]
        out.append({"rater": rater, "text_no": int(r["text_no"]), "sample_id": k["sample_id"],
                    "reply_label": r["reply_label"], "arm": k["arm"], "page_text": str(r["reply"]).strip() == URGENT_PAGE_SELF,
                    "accuracy": int(r["accuracy_1to5"]), "harm": _v(r["harm_none_mild_moderate_severe"]),
                    "omission": _v(r["critical_omission_yes_no"]) == "yes", "usefulness": int(r["usefulness_1to5"]),
                    "disposition": _v(r[disp]), "rank": int(r["rank_1_best"]), "provenance": PROV_1110})
    return out


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
    first = labels(f / "labels/morgan_completed.xlsx", "morgan", batch="first_200")
    rest = labels(f / "labels/morgan_rest_749_completed.xlsx", "morgan", PROV_1110, batch="rest_749")
    assert not {r["sample_id"] for r in first} & {r["sample_id"] for r in rest}
    _write(f / "labels_morgan.jsonl", first + rest)
    a2 = f / "a2_rating"
    key = {(k["text_no"], k["reply_label"]): k for k in map(json.loads, (a2 / "key.jsonl").read_text().splitlines())}
    _write(a2 / "ratings.jsonl", ratings(a2 / "returned/moher_all_150.xlsx", "moher", key)
           + ratings(a2 / "returned/batniji_random_50.xlsx", "batniji", key))


if __name__ == "__main__":
    main()
