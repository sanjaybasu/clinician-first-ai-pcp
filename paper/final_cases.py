"""Final-test simulated cases for Paper 2 (design/paper2_preregistration_draft.md, section 3, item 1).

Draws the DDXPlus test split (never the validate split used in development) in the development strata, 30 cases per
stratum, excluding the 720 test-split cases of Paper 1 (amd002_a4/test_run/cases.jsonl). Writes cases.jsonl into two
run directories that paper/dev_multiturn.py then uses as they are:
  <out>/ddxplus   the six cannot-miss and three benign strata of Analysis 4 (paper/multiturn_ddxplus.STRATA)
  <out>/outpt     the sore-throat, sinus and ear strata (paper/dev_orders.STRATA)
DDXPlus is public data; case files hold no patient data. Usage:
  python -m paper.final_cases --data <compass-pcp-data> --out <compass-pcp-data>/paper2_final --per-stratum 30
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import paper.dev_multiturn  # noqa: F401 - registers the outpatient openers on multiturn_ddxplus
from paper import dev_orders as do
from paper import multiturn_ddxplus as mt

SEED = 20261107  # differs from every development seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=30)
    a = ap.parse_args()
    paper1 = {json.loads(x)["case_id"] for x in (a.data / "amd002_a4/test_run/cases.jsonl").read_text().splitlines()}
    rows = [r for r in mt.read_split("test") if r["case_id"] not in paper1]
    keep = {s for s, _, _ in mt.STRATA}
    ddx, outpt = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["stratum"] in keep:
            ddx[r["stratum"]].append(r)
        key = (r["complaint"], r["pathology"])
        if key in do.STRATA:
            slug, cm = do.STRATA[key]
            s = f"{r['complaint']}/{slug}"
            outpt[s].append({**r, "stratum": s, "cannot_miss": cm, "benign": cm is None})
    summary = {}
    for name, by in (("ddxplus", ddx), ("outpt", outpt)):
        rng = random.Random(SEED)
        cases = [c for s in sorted(by) for c in rng.sample(by[s], min(a.per_stratum, len(by[s])))]
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "cases.jsonl").exists():
            raise SystemExit(f"{d / 'cases.jsonl'} exists; the final sample is drawn once")
        (d / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
        summary[name] = {s: min(a.per_stratum, len(by[s])) for s in sorted(by)}
    print(json.dumps({"excluded_paper1": len(paper1), **summary}, indent=1))


if __name__ == "__main__":
    main()
