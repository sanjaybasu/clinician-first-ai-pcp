"""Cohort A, red-flag floor: deterministic screens vs physician labels.

Scores four deterministic floors on the frozen BMC message set (2,000 real
Medicaid SMS, three blinded physicians):
  compass_keyword      Compass critical-checks keyword screen (step 1, before the LLM confirm)
  compass_befast       Compass ANCHOR BE-FAST floor (app/core/anchor_rules.py)
  anchor_undertriage   full ANCHOR corpus, under-triage rules, over patient text
  union                any of the three
No model is called. The LLM confirmation step that follows a Compass screen hit
is not part of the floor and is scored separately when a backbone is available.

Usage:
  python scripts/run_cohort_a_floor.py --corpus <json> --compass <compass checkout> --out <dir outside git>
Requires ANCHOR_REPO (or the default checkout) and the Compass checkout's
dependencies (or stubs for wm_py_common) on PYTHONPATH.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from compass_pcp.anchor_bridge import anchor_commit, red_flag_floor
from compass_pcp.harness.metrics import wilson

ENDPOINTS = {
    "any_hazard": lambda c: c["detection_truth"] == 1,
    "contact_doctor_or_911": lambda c: c["action_truth"] in ("Contact Doctor", "Call 911/988"),
    "call_911_988": lambda c: c["action_truth"] == "Call 911/988",
}


def compass_screens(compass_dir: Path):
    sys.path.insert(0, str(compass_dir))
    from app.services.anchor_service import AnchorService  # type: ignore[import-not-found]
    from app.services.critical_checks_service import CriticalChecksService  # type: ignore[import-not-found]

    ccs, anchor = CriticalChecksService(), AnchorService()

    def keyword(msg: str) -> bool:
        return bool(ccs.check_keyword_emergency(msg))

    def befast(msg: str) -> bool:
        return anchor.assess_floor(msg).floor_level == "EMERGENCY"

    return keyword, befast


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--compass", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = json.loads(args.corpus.read_text())
    keyword, befast = compass_screens(args.compass)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in cases:
        msg = c["message"] or ""
        kw, bf = keyword(msg), befast(msg)
        at = red_flag_floor(msg)
        rows.append(
            {
                "message_id": c["message_id"],
                "compass_keyword": kw,
                "compass_befast": bf,
                "anchor_undertriage": bool(at),
                "anchor_rules": at,
                "union": kw or bf or bool(at),
                **{f"truth_{k}": f(c) for k, f in ENDPOINTS.items()},
            }
        )
    with (args.out / "cohort_a_floor_per_message.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    summary = {
        "run_utc": datetime.now(UTC).isoformat(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "n": len(rows),
        "compass_commit": git_head(args.compass),
        "anchor_commit": anchor_commit(),
        "results": {},
    }
    for screen in ("compass_keyword", "compass_befast", "anchor_undertriage", "union"):
        res = {}
        for ep in ENDPOINTS:
            pos = [r for r in rows if r[f"truth_{ep}"]]
            neg = [r for r in rows if not r[f"truth_{ep}"]]
            tp = sum(r[screen] for r in pos)
            fp = sum(r[screen] for r in neg)
            res[ep] = {
                "positives": len(pos),
                "tp": tp,
                "sensitivity": tp / len(pos) if pos else None,
                "sensitivity_95ci": wilson(tp, len(pos)) if pos else None,
                "fp": fp,
                "specificity": (len(neg) - fp) / len(neg),
                "specificity_95ci": wilson(len(neg) - fp, len(neg)),
                "flags_per_1000": 1000 * (tp + fp) / len(rows),
            }
        summary["results"][screen] = res
    (args.out / "cohort_a_floor_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["results"], indent=1))


if __name__ == "__main__":
    main()
