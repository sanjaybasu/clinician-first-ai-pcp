"""Freeze the analysis before the test set is unsealed.

The freeze file records the architecture commit, every threshold and pass
criterion, and the SHA-256 of the test-set episode ids. It is what goes on OSF.
``verify`` refuses to score if any of it has changed.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from compass_pcp.harness.episode import hash_ids


class Freeze(BaseModel):
    created_utc: str
    architecture_commit: str
    compass_commit: str
    thresholds_sha256: str
    thresholds: dict
    test_set: dict


def _sha(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def freeze(path: Path, *, architecture_commit: str, compass_commit: str, thresholds: dict, test_ids: list[str]) -> Freeze:
    if path.exists():
        raise FileExistsError(f"{path} already exists; a freeze is written once")
    f = Freeze(
        created_utc=datetime.now(UTC).isoformat(),
        architecture_commit=architecture_commit,
        compass_commit=compass_commit,
        thresholds_sha256=_sha(thresholds),
        thresholds=thresholds,
        test_set=hash_ids(test_ids),
    )
    path.write_text(f.model_dump_json(indent=2))
    return f


def verify(path: Path, *, architecture_commit: str, compass_commit: str, thresholds: dict, test_ids: list[str]) -> None:
    f = Freeze.model_validate_json(path.read_text())
    problems = []
    if f.architecture_commit != architecture_commit:
        problems.append("architecture commit changed")
    if f.compass_commit != compass_commit:
        problems.append("compass commit changed")
    if f.thresholds_sha256 != _sha(thresholds):
        problems.append("thresholds changed")
    if f.test_set["set_sha256"] != hash_ids(test_ids)["set_sha256"]:
        problems.append("test set changed")
    if problems:
        raise RuntimeError("freeze violated: " + "; ".join(problems))
