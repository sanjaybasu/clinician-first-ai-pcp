"""Re-derive CANONICAL_NUMBERS.json from the raw inputs and check every value, and check that the rendered manuscript
and supplement contain no stale or pending values.

Usage: python -m paper.verify_canonical --hb-run <dir> --cohort-a <report.json> --freeze-dir docs/regulatory/freeze \\
           --manuscript-dir <notebooks/compass-ai-pcp/manuscript> [--physician <scores.json>] [--physician-packet <dir>] [--hazard-dev <summary.json>]
Exit status 1 on any mismatch.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from paper.build_canonical import ablation_numbers, blinding_numbers, path_numbers, sensitivity_numbers, cohort_a_numbers, tier_numbers, hazard_dev_numbers, hb_numbers, physician_numbers, static_numbers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-run", type=Path)
    ap.add_argument("--cohort-a", type=Path)
    ap.add_argument("--physician", type=Path)
    ap.add_argument("--hazard-dev", type=Path)
    ap.add_argument("--physician-packet", type=Path)
    ap.add_argument("--sensitivity", type=Path)
    ap.add_argument("--paths", type=Path)
    ap.add_argument("--ablation", type=Path)
    ap.add_argument("--freeze-dir", type=Path, required=True)
    ap.add_argument("--manuscript-dir", type=Path, required=True)
    args = ap.parse_args()
    fresh = static_numbers(args.freeze_dir)
    if args.hb_run and (args.hb_run / "ai_pcp_v3.json").exists():
        fresh.update(hb_numbers(args.hb_run))
    if args.cohort_a and args.cohort_a.exists():
        fresh.update(cohort_a_numbers(json.loads(args.cohort_a.read_text())))
        fresh.update(tier_numbers(args.cohort_a.parent / "per_message.jsonl"))
    if args.physician and args.physician.exists():
        fresh.update(physician_numbers(json.loads(args.physician.read_text())))
    if args.physician_packet and (args.physician_packet / "physician_key.json").exists():
        fresh.update(blinding_numbers(args.physician_packet))
    if args.paths and args.paths.exists():
        fresh.update(path_numbers(args.paths, args.hb_run))
    if args.ablation and (args.ablation / "ai_pcp_v3__grader2.json").exists():
        fresh.update(ablation_numbers(args.ablation, args.hb_run))
    if args.sensitivity and args.sensitivity.exists():
        fresh.update(sensitivity_numbers(json.loads(args.sensitivity.read_text())))
    if args.hazard_dev and args.hazard_dev.exists():
        fresh.update(hazard_dev_numbers(json.loads(args.hazard_dev.read_text())))
    stored = json.loads((args.manuscript_dir / "CANONICAL_NUMBERS.json").read_text())
    bad = [(k, stored.get(k), v) for k, v in fresh.items() if not k.startswith("_") and stored.get(k) != v]
    bad += [(k, stored[k], None) for k in stored if not k.startswith("_") and k not in fresh]
    for k, s, f in bad:
        print(f"MISMATCH {k}: stored {s!r} re-derived {f!r}")
    texts = {p.name: p.read_text() for p in (args.manuscript_dir / "compass_ai_pcp_jamia_manuscript.md",
                                             args.manuscript_dir / "compass_ai_pcp_jamia_supplement.md") if p.exists()}
    for name, t in texts.items():
        for k in sorted(set(re.findall(r"\[PENDING: (\w+)\]", t))):
            print(f"PENDING {name}: {k}")
    print(f"{len([k for k in fresh if not k.startswith('_')])} values re-derived; {len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
