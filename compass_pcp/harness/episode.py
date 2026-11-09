"""Episodes, cohorts and the patient-disjoint, forward-in-time split.

An episode is one real index contact reconstructed from the opening message,
the Clinical Context as it stood that day, the full record the gatekeeper may
reveal on request, and the 30-day outcome from claims and ADT. Episode files
hold PHI and live only on Waymark infrastructure; this repo holds code only.
"""

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from compass_pcp.findings import Finding, Polarity


class Cohort(StrEnum):
    A_PARSER = "A"  # parser and interlock: frozen BMC messages + forward Compass openers
    B_LANE = "B"  # lane episodes, random within lane (headline)
    C_CANNOT_MISS = "C"  # outcome-confirmed cannot-miss look-back (SPADE in reverse) + matched negatives
    D_STRESS = "D"  # authored, adversarial and known-miss regression cases; never pooled
    HEALTHBENCH = "HB"  # public comparability layer; never gates a lane


class RecordFact(BaseModel):
    polarity: Polarity
    value: float | None = None
    patient_phrasing: str | None = Field(default=None, description="The patient's own words, if they ever said it")


class LabResult(BaseModel):
    loinc: str
    value: float
    unit: str | None = None
    resulted_at: datetime


class Outcome30d(BaseModel):
    diagnosis_codes: list[str] = Field(default_factory=list)
    cannot_miss_diagnoses: list[str] = Field(default_factory=list)
    ed_visit: bool = False
    ed_avoidable: bool | None = None
    admission: bool = False
    filled_rx: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    episode_id: str
    patient_key: str = Field(description="Salted hash of the member id, never the raw id")
    cohort: Cohort
    lane: str
    complaint: str
    index_time: datetime
    opening_message: str
    context_snapshot: list[Finding] = Field(description="Structured record as of index_time")
    record_facts: dict[str, RecordFact] = Field(description="feature_id -> fact the gatekeeper may reveal")
    labs: list[LabResult] = Field(default_factory=list)
    outcome: Outcome30d | None = None
    historical_decision: str | None = Field(default=None, description="What the care team actually did")
    matched_to: str | None = Field(default=None, description="Cohort C negative: the positive it was matched to")
    # Pre-specified subgroups (design doc 3; FDA AI draft guidance, subgroup performance).
    state: str | None = None
    age_band: str | None = None  # e.g. "18-34", "35-49", "50-64", "65+"
    sex: str | None = None
    race_ethnicity: str | None = None
    language: str | None = None
    payer: str | None = None


class Split(BaseModel):
    cutoff: datetime
    dev: list[str]
    test: list[str]
    dropped_for_patient_overlap: list[str]


def temporal_patient_disjoint_split(episodes: list[Episode], cutoff: datetime) -> Split:
    """Dev is before the cutoff; test is on or after it; a dev patient never appears in test."""
    dev = [e for e in episodes if e.index_time < cutoff]
    dev_patients = {e.patient_key for e in dev}
    test, dropped = [], []
    for e in episodes:
        if e.index_time < cutoff:
            continue
        (dropped if e.patient_key in dev_patients else test).append(e.episode_id)
    return Split(cutoff=cutoff, dev=sorted(e.episode_id for e in dev), test=sorted(test), dropped_for_patient_overlap=sorted(dropped))


def hash_ids(ids: list[str]) -> dict[str, str | list[str]]:
    per_id = sorted(hashlib.sha256(i.encode()).hexdigest() for i in ids)
    whole = hashlib.sha256("\n".join(per_id).encode()).hexdigest()
    return {"n": str(len(ids)), "set_sha256": whole, "id_sha256": per_id}
