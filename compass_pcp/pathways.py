"""Guideline pathways over the workup engine (design/pathway_schema.md).

A pathway maps each engine state (decision kind, cannot-miss statuses, red flags reported present) to a disposition
with a time frame, a standing disposition for a person who stops replying, and tests and referrals proposed for
clinician approval. Every node cites sources defined in the file. Deterministic and LLM-free; clinical values live in
``data/pathways/*.yaml``.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from compass_pcp.engine.workup import Decision, DecisionKind
from compass_pcp.findings import Polarity

PATHWAY_DIR = Path(__file__).resolve().parent / "data" / "pathways"
Level = Literal["self_care", "routine_visit", "clinician_review", "urgent_same_day", "emergency_now"]
LEVEL_ORDER = {"self_care": 0, "routine_visit": 1, "clinician_review": 2, "urgent_same_day": 3, "emergency_now": 4}


class Source(BaseModel):
    citation: str
    pmid: str | None = None
    url: str | None = None
    verified: bool = False


class Disposition(BaseModel):
    level: Level
    within_hours: int | None = None


class Proposal(BaseModel):
    kind: Literal["lab", "imaging", "test", "referral", "medication_review"]
    name: str
    source: list[str] = Field(min_length=1)


class When(BaseModel):
    kinds: list[DecisionKind] = Field(default_factory=list, description="Any of these engine decisions; empty = any")
    cannot_miss: dict[str, list[str]] = Field(default_factory=dict, description="Diagnosis -> any of these statuses")
    red_flags_any: list[str] = Field(default_factory=list, description="Any of these features reported present")
    unanswered_any: list[str] = Field(default_factory=list,
                                      description="Any of these features asked in an earlier reply and still unanswered")


class Node(BaseModel):
    id: str
    when: When
    disposition: Disposition | None = None  # None: still asking; the standing disposition applies
    standing: Disposition | None = None
    proposals: list[Proposal] = Field(default_factory=list)
    source: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _has_output(self) -> "Node":
        if self.disposition is None and self.standing is None:
            raise ValueError(f"node {self.id} has neither a disposition nor a standing disposition")
        return self


class Pathway(BaseModel):
    pathway: str
    version: str
    script: str
    status: Literal["draft_unsigned", "signed"]
    signed_by: str | None = None
    sources: dict[str, Source]
    nodes: list[Node] = Field(min_length=1)

    @model_validator(mode="after")
    def _sources_defined(self) -> "Pathway":
        for n in self.nodes:
            for sid in [*n.source, *(s for p in n.proposals for s in p.source)]:
                if sid.split(":", 1)[0] not in self.sources:
                    raise ValueError(f"node {n.id} cites undefined source {sid}")
        return self


def load_pathway(path: Path) -> Pathway:
    return Pathway.model_validate(yaml.safe_load(path.read_text()))


def _matches(w: When, decision: Decision, present: set[str], unanswered: set[str]) -> bool:
    if w.kinds and decision.kind not in w.kinds:
        return False
    status = {c.diagnosis_id: c.status.value for c in decision.certificate.cannot_miss}
    for dx, allowed in w.cannot_miss.items():
        if status.get(dx) not in allowed:
            return False
    if w.unanswered_any and not unanswered & set(w.unanswered_any):
        return False
    return not w.red_flags_any or bool(present & set(w.red_flags_any))


def evaluate(pathway: Pathway, decision: Decision, findings, unanswered=()) -> Node | None:
    """The first node, in file order, that matches the engine's decision; None if no node matches (a gap).
    ``unanswered``: features asked in an earlier reply and still unanswered (for ``When.unanswered_any``)."""
    present = {f.feature_id for f in findings if f.polarity == Polarity.PRESENT}
    return next((n for n in pathway.nodes if _matches(n.when, decision, present, set(unanswered))), None)
