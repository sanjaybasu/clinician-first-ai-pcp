"""Findings schema, dual-extraction reconciliation and provenance merging.

Layer 1 of the architecture. Two independent extractors (different vendors and
prompts) parse the patient's messages into ``Finding`` objects. A finding is
accepted only when both extractors agree on code, polarity and value; anything
else is ``disputed`` and becomes a confirmation question, then [0, 1] (unknown).
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Polarity(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"


class Provenance(IntEnum):
    """Higher outranks lower when sources conflict (design doc, step 1)."""

    INFERENCE = 0
    PATIENT_REPORT = 1  # new patient report
    PATIENT_CONFIRMED_RECORD = 2  # patient confirmed a record item
    STRUCTURED_RECORD = 3  # claims, labs, pharmacy fills


class CodeSystem(StrEnum):
    SNOMED = "SNOMED"
    LOINC = "LOINC"
    RXNORM = "RXNORM"
    LOCAL = "LOCAL"


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_id: str = Field(description="Key into the illness script's feature list")
    code_system: CodeSystem
    code: str
    polarity: Polarity
    value: float | None = None
    unit: str | None = None
    onset_days: float | None = Field(default=None, description="Days since onset, if stated")
    provenance: Provenance
    extractor_id: str = Field(description="Extractor (model+prompt version) or record source")
    source_ref: str | None = Field(default=None, description="Message id / turn / record row")
    observed_on: str | None = Field(default=None, description="ISO date of the record item, for recency windows")
    confirmed: bool = Field(default=False, description="Patient answered an explicit confirmation question")


class ReconcileResult(BaseModel):
    accepted: list[Finding]
    disputed: list[str] = Field(description="feature_ids the extractors disagree on or only one found")


def _agree(a: Finding, b: Finding, value_tol: float) -> bool:
    if (a.code_system, a.code, a.polarity) != (b.code_system, b.code, b.polarity):
        return False
    if a.value is None and b.value is None:
        return True
    if a.value is None or b.value is None:
        return False
    return abs(a.value - b.value) <= value_tol


def reconcile(
    extractor_a: list[Finding], extractor_b: list[Finding], *, value_tol: float = 1e-6
) -> ReconcileResult:
    """Accept a feature only when both extractors produce one agreeing finding for it."""
    by_a = _index(extractor_a)
    by_b = _index(extractor_b)
    accepted: list[Finding] = []
    disputed: list[str] = []
    for feature_id in sorted(set(by_a) | set(by_b)):
        fa, fb = by_a.get(feature_id), by_b.get(feature_id)
        if fa is not None and fb is not None and len(fa) == 1 and len(fb) == 1 and _agree(fa[0], fb[0], value_tol):
            accepted.append(fa[0])
        else:
            disputed.append(feature_id)
    return ReconcileResult(accepted=accepted, disputed=disputed)


def _index(findings: list[Finding]) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for f in findings:
        out.setdefault(f.feature_id, []).append(f)
    return out


def merge_with_record(record: list[Finding], reported: list[Finding]) -> list[Finding]:
    """One finding per feature; the highest provenance wins.

    Structured lighthouse/coredb data outranks patient report, which outranks
    inference. Ties keep the record-side finding.
    """
    merged: dict[str, Finding] = {}
    for f in [*record, *reported]:
        current = merged.get(f.feature_id)
        if current is None or f.provenance > current.provenance:
            merged[f.feature_id] = f
    return list(merged.values())
