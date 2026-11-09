"""Event-sourced findings ledger (panel v1, item 2).

Findings are appended as events per patient turn; nothing is overwritten. The
current state of a feature is derived from its events:
- record events (claims, labs, fills) and patient events are both kept;
- a later patient statement about the same feature supersedes an earlier one
  only if it is an explicit correction, otherwise a polarity change is a
  conflict that must be confirmed;
- source reconciliation is fact-specific: a record value wins for objective
  facts (labs, fills, diagnoses on file), a confirmed patient report wins for
  current symptoms and whether a medication is actually being taken.
A hazard tier, once raised, never goes down within an encounter.
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

from compass_pcp.findings import Finding, Provenance


class FactKind(StrEnum):
    OBJECTIVE = "objective"  # lab value, fill, diagnosis on file: record wins
    SUBJECTIVE = "subjective"  # current symptom, adherence, side effect: confirmed patient report wins


class HazardTier(IntEnum):
    NONE = 0
    ROUTINE_FOLLOWUP = 1
    URGENT_SAME_DAY = 2
    POSSIBLE_EMERGENCY = 3  # could be an emergency depending on one or two facts: give conditional 911 guidance and ask now
    EMERGENCY_NOW = 4


class FindingEvent(BaseModel):
    turn: int
    finding: Finding
    correction: bool = Field(default=False, description="Patient explicitly corrected an earlier statement")


class HazardEvent(BaseModel):
    turn: int
    tier: HazardTier
    category: str
    quote: str
    quote_verified: bool
    rationale: str = ""


class Ledger(BaseModel):
    findings: list[FindingEvent] = Field(default_factory=list)
    hazards: list[HazardEvent] = Field(default_factory=list)
    active_complaints: list[str] = Field(default_factory=list, description="Script ids, priority order, union over turns")
    kinds: dict[str, FactKind] = Field(default_factory=dict)

    def add_complaints(self, ids: list[str]) -> None:
        for i in ids:
            if i not in self.active_complaints:
                self.active_complaints.append(i)

    def add_finding(self, turn: int, f: Finding, *, correction: bool = False) -> None:
        self.findings.append(FindingEvent(turn=turn, finding=f, correction=correction))

    def add_hazard(self, e: HazardEvent) -> None:
        self.hazards.append(e)

    @property
    def hazard_tier(self) -> HazardTier:
        return max((h.tier for h in self.hazards), default=HazardTier.NONE)

    def state(self) -> tuple[list[Finding], list[str]]:
        """Current findings (one or two per feature) and features in conflict needing confirmation."""
        by_feature: dict[str, list[FindingEvent]] = {}
        for e in self.findings:
            by_feature.setdefault(e.finding.feature_id, []).append(e)
        current: list[Finding] = []
        conflicts: list[str] = []
        for fid, events in by_feature.items():
            record = [e for e in events if e.finding.provenance >= Provenance.STRUCTURED_RECORD]
            patient = [e for e in events if e.finding.provenance < Provenance.STRUCTURED_RECORD]
            latest_patient = self._latest_patient(patient)
            if latest_patient is None and not record:
                continue
            if latest_patient == "conflict":
                conflicts.append(fid)
                continue
            kind = self.kinds.get(fid, FactKind.SUBJECTIVE)
            rec = record[-1].finding if record else None
            if rec is not None and latest_patient is not None and rec.polarity != latest_patient.polarity:
                if kind == FactKind.OBJECTIVE:
                    current.append(rec)
                elif latest_patient.confirmed:
                    current.append(latest_patient)
                else:
                    conflicts.append(fid)
                continue
            chosen = latest_patient if (kind == FactKind.SUBJECTIVE and latest_patient is not None) else (rec or latest_patient)
            if chosen is not None:
                current.append(chosen)
        return current, sorted(conflicts)

    @staticmethod
    def _latest_patient(events: list[FindingEvent]):
        if not events:
            return None
        last = events[-1]
        earlier = [e for e in events[:-1] if e.finding.polarity != last.finding.polarity]
        if earlier and not (last.correction or last.finding.confirmed):
            return "conflict"
        return last.finding

