"""Split the BMC real-world message set 50/50 into Cohort A tuning and sealed test halves.

Stratified by (detection_truth, action_truth); within each stratum, messages are ordered by a salted SHA-256 of
message_id and alternated between halves, so the split is reproducible and balanced by construction. Writes the two
id lists (outside git) and a manifest of id hashes (safe for git and for the OSF freeze).

Usage: python -m scripts.split_cohort_a --cases <realworld json> --out <dir outside git> --manifest docs/regulatory/cohort_a_split_manifest.json
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.episode import hash_ids

SALT = "compass-pcp-cohort-a-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text())
    strata: dict[tuple, list[str]] = defaultdict(list)
    for c in cases:
        strata[(int(c["detection_truth"]), str(c["action_truth"]))].append(c["message_id"])
    tune, test = [], []
    for key in sorted(strata):
        ids = sorted(strata[key], key=lambda i: hashlib.sha256(f"{SALT}:{i}".encode()).hexdigest())
        for j, i in enumerate(ids):
            (tune if j % 2 == 0 else test).append(i)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cohort_a_tune_ids.json").write_text(json.dumps(sorted(tune)))
    (args.out / "cohort_a_test_ids.json").write_text(json.dumps(sorted(test)))
    by = {c["message_id"]: c for c in cases}
    summary = {
        k: {"tune": sum(1 for i in tune if (int(by[i]["detection_truth"]), str(by[i]["action_truth"])) == k),
            "test": sum(1 for i in test if (int(by[i]["detection_truth"]), str(by[i]["action_truth"])) == k)}
        for k in sorted(strata)
    }
    manifest = {
        "salt": SALT,
        "source_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "strata": {f"{k[0]}|{k[1]}": v for k, v in summary.items()},
        "tune": hash_ids(tune),
        "test": hash_ids(test),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["strata"], indent=1), len(tune), len(test))


if __name__ == "__main__":
    main()
