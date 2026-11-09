"""Run a system under test over episodes, three times each, with a component manifest.

Two lessons from CITE are enforced here, not left to discipline:
1. Every run records which components executed. A run with any required
   component missing is FAILED_COMPONENT and counts as a failure, never a skip.
2. Generator, verifier, extractor B and gatekeeper paraphraser must come from
   different vendors where the protocol says so; the runner refuses otherwise.

Output is one JSON line per (episode, run). Re-running resumes from the file.
"""

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from compass_pcp.harness.episode import Episode
from compass_pcp.harness.gatekeeper import Gatekeeper, GatekeeperAnswer, Paraphraser

N_RUNS = 3


class RunStatus(StrEnum):
    OK = "ok"
    FAILED_COMPONENT = "failed_component"
    ERROR = "error"


class SystemOutput(BaseModel):
    decision: str = Field(description="ask/defer/escalate/complete at the end of the encounter")
    disposition: str | None = None
    orders: list[str] = Field(default_factory=list)
    prescriptions: list[str] = Field(default_factory=list)
    patient_messages: list[str] = Field(default_factory=list)
    verifier_findings: list[str] = Field(default_factory=list)
    deferral_reasons: list[str] = Field(default_factory=list)
    top3_differential: list[str] = Field(default_factory=list)
    components_executed: dict[str, bool]
    versions: dict[str, str] = Field(description="commit, prompt labels, model ids, script and rule versions")
    certificates: list[dict] = Field(default_factory=list)


class RunRecord(BaseModel):
    episode_id: str
    run_index: int
    status: RunStatus
    missing_components: list[str] = Field(default_factory=list)
    output: SystemOutput | None = None
    gatekeeper_log: list[GatekeeperAnswer] = Field(default_factory=list)
    error: str | None = None


class SystemUnderTest(Protocol):
    name: str
    vendors: dict[str, str]  # role -> vendor, e.g. {"generator": "google", "verifier": "anthropic"}

    def run(self, episode: Episode, gatekeeper: Gatekeeper, run_index: int) -> SystemOutput: ...


class VendorPolicy(BaseModel):
    must_differ: list[tuple[str, str]] = [
        ("generator", "verifier"),
        ("extractor_a", "extractor_b"),
        ("generator", "gatekeeper_paraphraser"),
    ]

    def check(self, vendors: dict[str, str]) -> None:
        for a, b in self.must_differ:
            if a in vendors and b in vendors and vendors[a] == vendors[b]:
                raise ValueError(f"vendor separation violated: {a} and {b} are both {vendors[a]}")


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out.add((rec["episode_id"], rec["run_index"]))
    return out


def _append(path: Path, record: RunRecord) -> None:
    with path.open("a") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_all(
    system: SystemUnderTest,
    episodes: list[Episode],
    out_path: Path,
    required_components: list[str],
    *,
    paraphraser: Paraphraser | None = None,
    vendor_policy: VendorPolicy | None = None,
    n_runs: int = N_RUNS,
) -> list[RunRecord]:
    vendors = dict(system.vendors)
    if paraphraser is not None:
        vendors["gatekeeper_paraphraser"] = paraphraser.vendor
    (vendor_policy or VendorPolicy()).check(vendors)

    done = _done(out_path)
    written: list[RunRecord] = []
    for ep in episodes:
        for k in range(n_runs):
            if (ep.episode_id, k) in done:
                continue
            gk = Gatekeeper(ep, paraphraser)
            try:
                output = system.run(ep, gk, k)
                missing = [c for c in required_components if not output.components_executed.get(c, False)]
                status = RunStatus.FAILED_COMPONENT if missing else RunStatus.OK
                record = RunRecord(
                    episode_id=ep.episode_id, run_index=k, status=status, missing_components=missing,
                    output=output, gatekeeper_log=gk.log,
                )
            except Exception as exc:  # noqa: BLE001  (recorded as a failure, never dropped)
                record = RunRecord(
                    episode_id=ep.episode_id, run_index=k, status=RunStatus.ERROR, error=repr(exc), gatekeeper_log=gk.log
                )
            _append(out_path, record)
            written.append(record)
    return written


DECISION_FIELDS = ("decision", "disposition", "orders", "prescriptions")


def run_to_run_disagreement(records: list[RunRecord]) -> dict[str, list[str]]:
    """episode_id -> decision-relevant fields that differ across that episode's runs."""
    by_ep: dict[str, list[SystemOutput]] = {}
    for r in records:
        if r.output is not None:
            by_ep.setdefault(r.episode_id, []).append(r.output)
    out: dict[str, list[str]] = {}
    for ep, outputs in by_ep.items():
        fields = []
        for f in DECISION_FIELDS:
            values = {json.dumps(sorted(v) if isinstance(v, list) else v) for v in (getattr(o, f) for o in outputs)}
            if len(values) > 1:
                fields.append(f)
        out[ep] = fields
    return out
