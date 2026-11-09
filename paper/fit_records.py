"""Fit Kev fine-tune records to Kev's training context (384 state tokens), so no record is dropped (development only).

Run with the kev environment's Python (it needs kev and the Qwen3.5 tokenizer, both local):
  <kev>/.venv/bin/python paper/fit_records.py --src <dir> --out <dir>
For every {train,calibration,development}.jsonl under --src (recursively), each record that does not fit is shortened:
  1. Chart records: drop chart lines from the end, then cut the last kept line, until the state fits. The message is kept
     whole. If the message alone does not fit, the chart is removed and step 2 applies.
  2. Message-only state that does not fit: keep the start and the end of the message with " [...] " between them.
Records keep their ids and labels. Writes the same tree under --out (chmod 600). Prints counts only, never text.
"""

import argparse
import copy
import json
from pathlib import Path

from kev.data import materialize
from kev.model import MAX_STATE, fits, load_tokenizer

SEP = "\n\nText message to the care team:\n"
HEAD = "Chart before this message:\n"
MSG = "Text message to the care team:\n"


def make_fit(tok):
    def ok(rec, st):
        r = copy.deepcopy(rec)
        r["state"] = st
        for q in r["questions"].values():
            q.setdefault("src", "compass")
        return fits(materialize(r), tok, max_state=MAX_STATE - 8)  # margin: the base revision can tokenize a few tokens longer
    return ok


def shorten_message(rec, text, ok):
    lo, hi = 0, len(text)  # largest kept length that fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        h = mid * 2 // 3
        cut = text[:h] + " [...] " + text[len(text) - (mid - h):]
        if ok(rec, MSG + cut):
            lo = mid
        else:
            hi = mid - 1
    h = lo * 2 // 3
    return MSG + text[:h] + " [...] " + text[len(text) - (lo - h):]


def fit_one(rec, ok):
    st = rec["state"]
    if ok(rec, st):
        return st, "kept"
    if st.startswith(HEAD) and SEP in st:
        chart, text = st[len(HEAD):].split(SEP, 1)
        if ok(rec, MSG + text):
            lines = chart.splitlines()
            while lines and not ok(rec, HEAD + "\n".join(lines) + SEP + text):
                lines.pop()
            if not lines:
                return MSG + text, "chart_removed"
            # put back as much of the next line as fits
            extra = chart.splitlines()[len(lines)] if len(lines) < len(chart.splitlines()) else ""
            lo, hi = 0, len(extra)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if ok(rec, HEAD + "\n".join(lines + [extra[:mid]]) + SEP + text):
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 20:
                lines.append(extra[:lo])
            return HEAD + "\n".join(lines) + SEP + text, "chart_trimmed"
        text = st[len(HEAD):].split(SEP, 1)[1]
        return shorten_message(rec, text, ok), "message_shortened"
    text = st[len(MSG):] if st.startswith(MSG) else st
    return shorten_message(rec, text, ok), "message_shortened"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B-Base")
    a = ap.parse_args()
    ok = make_fit(load_tokenizer(a.tokenizer))
    counts = {}
    for p in sorted(a.src.rglob("*.jsonl")):
        if p.stem not in ("train", "calibration", "development"):
            continue
        c = {}
        out = []
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            rec["state"], how = fit_one(rec, ok)
            c[how] = c.get(how, 0) + 1
            out.append(json.dumps(rec))
        q = a.out / p.relative_to(a.src)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text("\n".join(out) + "\n")
        q.chmod(0o600)
        counts[str(p.relative_to(a.src))] = c
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
