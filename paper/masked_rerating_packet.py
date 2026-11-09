"""AMD-PCP-002 Analysis 7: masked physician re-rating packet (HealthBench Consensus test; public synthetic data).

make:   python -m paper.masked_rerating_packet --run ../compass-pcp-data/test_healthbench \\
            --registered-key <notebooks>/compass-ai-pcp/physician_scoring/physician_key.json \\
            --out <notebooks>/compass-ai-pcp/physician_scoring/amd002_masked
Draws 50 new conversations not in the registered physician sample, stratified by theme exactly as
scripts/physician_packet.make (proportional allocation, largest remainders), seed 20261104. Takes the first-repeat reply
of ai_pcp_v3 and raw_llm (the registered repeat rule), masks every mention of a care team, care coordinator, nurse,
Waymark, or "your team"/"our team" with "a clinician" (MASK_RULES, applied in order to both arms), and writes the rater
sheet in the registered format with blinded A/B labels in random order. The unblinding key and the masking report go
next to the sheet; only send/ goes to raters. Scoring uses scripts.physician_packet score with this key.
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from scripts.physician_packet import ARMS, _examples, _text, _themes

SEED, N = 20261104, 50
REPLACEMENT = "a clinician"
_DET = r"(?:(?:a|an|the|your|our|my|his|her|their)\s+(?:(?:child|son|daughter|baby|mother|father|parent)'s\s+)?)?"
_CARE_TEAM = r"(?:(?:primary|health)\s+)?(?:health)?care[\s-]+teams?"
_PERSON = r"(?:nurse\s+or\s+doctor|doctor\s+or\s+nurse|nurse|doctor|physician|clinician|someone|member|staff\s+member)s?"
# Applied in this order; each pattern absorbs a leading determiner so "your care team" -> "a clinician", and the
# first rule takes the whole "<person> on/from <care team>" phrase so that no "a clinician on a clinician" results.
MASK_RULES: list[tuple[str, str]] = [
    ("person_on_care_team", _DET + _PERSON + r"\s+(?:on|from|in|of)\s+" + _DET + r"(?:" + _CARE_TEAM + r"|Waymark|team)"),
    ("waymark", _DET + r"Waymark(?:'s)?(?:\s+(?:care\s+team|team|staff|care\s+coordinators?|nurses?))?"),
    ("care_coordinator", _DET + r"care\s+coordinat(?:ors?|ion\s+team)"),
    ("care_team", _DET + _CARE_TEAM),
    ("nurse", _DET + r"(?:(?:registered|practice|triage|on-call|clinic|school|[a-z]+-[a-z]+)\s+)?nurses?"
              r"(?:\s+(?:practitioners?|advice\s+lines?|lines?|hotlines?))?"),
    ("your_our_team", r"(?:your|our)\s+team"),
]
_COMPILED = [(name, re.compile(r"\b" + pat + r"\b", re.I)) for name, pat in MASK_RULES]


def _sub(m: re.Match) -> str:
    return REPLACEMENT[0].upper() + REPLACEMENT[1:] if m.group(0)[0].isupper() else REPLACEMENT


def mask(text: str) -> tuple[str, dict[str, int]]:
    """Return masked text and the number of replacements per rule."""
    counts = {}
    for name, rx in _COMPILED:
        text, n = rx.subn(_sub, text)
        counts[name] = n
    return text, counts


def select(ids: list[str], themes: dict[str, str], exclude: set[str], n: int = N, seed: int = SEED) -> tuple[list[str], random.Random]:
    """Registered stratification (scripts/physician_packet.make) on the pool not in the registered sample."""
    pool = sorted(set(ids) - exclude)
    by_theme: dict[str, list[str]] = defaultdict(list)
    for pid in pool:
        by_theme[themes.get(pid, "unknown")].append(pid)
    rng = random.Random(seed)
    for theme in sorted(by_theme):
        rng.shuffle(by_theme[theme])
    quotas = {t: n * len(v) / len(pool) for t, v in by_theme.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    for t in sorted(quotas, key=lambda t: quotas[t] - alloc[t], reverse=True)[: n - sum(alloc.values())]:
        alloc[t] += 1
    picked: list[str] = []
    for t in sorted(by_theme):
        picked += by_theme[t][: alloc[t]]
    return picked, rng


def make(run: Path, registered_key: Path, out: Path) -> dict:
    arms = {a: _examples(run / f"{a}.json") for a in ARMS}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    themes = _themes(run)
    registered = set(json.loads(registered_key.read_text())["items"])
    picked, rng = select(ids, themes, registered)
    assert len(picked) == N and not set(picked) & registered
    key, rows = {}, []
    changed = {a: 0 for a in ARMS}
    subs = {a: {name: 0 for name, _ in MASK_RULES} for a in ARMS}
    for i, pid in enumerate(picked, start=1):
        order = list(ARMS)
        rng.shuffle(order)
        masked_counts = {}
        convo = "\n\n".join(f'{m["role"].upper()}: {m["content"]}' for m in arms[ARMS[0]][pid]["prompt"])
        for letter, arm in zip("AB", order):
            e = arms[arm][pid]
            text, counts = mask(_text(e["completion"]))
            masked_counts[arm] = sum(counts.values())
            changed[arm] += masked_counts[arm] > 0
            for k, v in counts.items():
                subs[arm][k] += v
            for j, item in enumerate(e["rubric_items"]):
                rows.append({"item": i, "prompt_id": pid, "response": letter, "criterion_index": j,
                             "conversation": convo if (letter == "A" and j == 0) else "",
                             "response_text": text if j == 0 else "",
                             "criterion": item["criterion"], "points": item["points"], "met": "",
                             "preferred": "" if (letter == "A" and j == 0) else "n/a"})
        key[pid] = {"A": order[0], "B": order[1], "item": i, "theme": themes.get(pid, "unknown"),
                    "replacements": masked_counts}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "physician_rating_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = {
        "analysis": "AMD-PCP-002 Analysis 7", "seed": SEED, "n_items": N, "arms": list(ARMS),
        "repeat_rule": "first repeat of each prompt id (as scripts/physician_packet.py)",
        "pool": {"conversations_with_both_arms": len(ids), "registered_sample_excluded": len(registered & set(ids)),
                 "eligible": len(set(ids) - registered)},
        "themes_picked": {t: sum(themes.get(p, "unknown") == t for p in picked) for t in sorted({themes.get(p, "unknown") for p in picked})},
        "replacement": REPLACEMENT, "mask_rules": [{"name": n, "regex": p, "flags": "IGNORECASE, word-bounded"} for n, p in MASK_RULES],
        "replies_changed": {a: {"n": changed[a], "of": N} for a in ARMS},
        "replacements_by_rule": subs,
        "rating_rows": len(rows), "criterion_rows_per_response": len(rows) // 2,
        "departures": [
            "Term list: the amendment names 'a care team, nurse, or Waymark staff'. The rule also masks 'care coordinator' "
            "and 'your team'/'our team', and absorbs a leading determiner and the modifiers listed in the regex "
            "(e.g. 'your care team', 'triage nurse', 'nurse advice line') so that the replacement reads as one noun phrase.",
            "Plural mentions ('nurses', 'care teams') become the singular 'a clinician'.",
        ],
    }
    (out / "physician_key.json").write_text(json.dumps({"seed": SEED, "arms": ARMS, "registered_key_excluded": str(registered_key),
                                                        "items": key}, indent=2))
    (out / "masking_report.json").write_text(json.dumps(report, indent=2))
    return report


def instructions(registered: str, n_items: int, rows_per_response: int) -> str:
    """Registered rater instructions with only the sample size, counts, time and file name changed."""
    lo, hi = round(n_items * 3 / 60, 1), round(n_items * 4 / 60, 1)
    edits = [
        ("The sample has 100 conversations from HealthBench",
         f"This is a second sample, added to the registration by amendment AMD-PCP-002. It has {n_items} new conversations from HealthBench"),
        ('numbered 1 to 100', f"numbered 1 to {n_items}"),
        ("The sample has 438 marks to record: 219 criterion rows for Response A and 219 for Response B, plus one preference "
         "for each of the 100 items. The task takes about 5 to 7 hours in total",
         f"The sample has {2 * rows_per_response} marks to record: {rows_per_response} criterion rows for Response A and "
         f"{rows_per_response} for Response B, plus one preference for each of the {n_items} items. The task takes about "
         f"{lo:g} to {hi:g} hours in total"),
        ("`physician_rating_sheet_SB.csv`", "`physician_rating_sheet_amd002_SB.csv`"),
    ]
    for old, new in edits:
        if old not in registered:
            raise SystemExit(f"registered instructions changed; cannot find: {old[:60]}")
        registered = registered.replace(old, new)
    return registered


def build_send(out: Path, registered_dir: Path, n_items: int, rows_per_response: int) -> None:
    """Rater files: sheet, reading booklet (registered build_booklet.py) and instructions PDF, in out/send."""
    import importlib.util
    import shutil
    import subprocess

    spec = importlib.util.spec_from_file_location("build_booklet", registered_dir / "build_booklet.py")
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)
    bb.HERE = out
    bb.main()
    md = out / "PHYSICIAN_SCORING_INSTRUCTIONS.md"
    md.write_text(instructions((registered_dir / "PHYSICIAN_SCORING_INSTRUCTIONS.md").read_text(), n_items, rows_per_response))
    html = out / "_instructions.html"
    subprocess.run(["pandoc", str(md), "-s", "--metadata", "title= ", "-o", str(html)], check=True, cwd=out)
    subprocess.run([bb.CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out / 'PHYSICIAN_SCORING_INSTRUCTIONS.pdf'}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    send = out / "send"
    send.mkdir(exist_ok=True)
    for f in ("physician_rating_sheet.csv", "physician_reading_booklet.pdf", "PHYSICIAN_SCORING_INSTRUCTIONS.pdf"):
        shutil.move(str(out / f), send / f) if f.endswith(".pdf") else shutil.copy(out / f, send / f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--registered-key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-send", action="store_true", help="skip the booklet and instructions PDFs")
    args = ap.parse_args()
    r = make(args.run, args.registered_key, args.out)
    if not args.no_send:
        build_send(args.out, args.registered_key.parent, r["n_items"], r["criterion_rows_per_response"])
    print(json.dumps({k: r[k] for k in ("pool", "replies_changed", "replacements_by_rule", "rating_rows")}, indent=2))


if __name__ == "__main__":
    main()
