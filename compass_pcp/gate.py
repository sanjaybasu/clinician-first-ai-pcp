"""Verifier as a gate. Every finding blocks an action or creates a task for a named person.

No finding is advisory. With a physician in the loop (rung 0 to 1), a finding
blocks sign-off until it is accepted or overridden with a coded reason. In an
autonomous lane (rung 2 to 3), a finding halts the action and escalates.

Rules are data (``compass_pcp/data/rules/*.yaml``). Each rule's antecedent is a
list of predicates evaluated with Łukasiewicz bounds; a rule fires when its
antecedent is definitely true, and an unknown antecedent becomes a question or
a deferral rather than a pass.
"""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from compass_pcp.engine.truth import FALSE, TRUE, UNKNOWN, Bounds, all_of, or_
from compass_pcp.lanes import LaneConfig, autonomous


class FindingKind(StrEnum):
    COMMISSION = "commission"  # e.g. contraindicated drug, wrong dose
    ALLERGY = "allergy"
    CANNOT_MISS_OMISSION = "cannot_miss_omission"
    UNSUPPORTED_STATEMENT = "unsupported_statement"
    LAB_RECENCY = "lab_recency"
    OUT_OF_LANE = "out_of_lane"  # request the lane may never handle (controlled substance, REMS drug)


class Owner(StrEnum):
    PHARMACIST = "pharmacist"
    ORDERING_CLINICIAN = "ordering_clinician"
    RECORD_CURATOR = "record_curator"


ROUTING: dict[FindingKind, Owner] = {
    FindingKind.COMMISSION: Owner.PHARMACIST,
    FindingKind.ALLERGY: Owner.PHARMACIST,
    FindingKind.LAB_RECENCY: Owner.PHARMACIST,
    FindingKind.CANNOT_MISS_OMISSION: Owner.ORDERING_CLINICIAN,
    FindingKind.UNSUPPORTED_STATEMENT: Owner.RECORD_CURATOR,
    FindingKind.OUT_OF_LANE: Owner.ORDERING_CLINICIAN,
}


class GateRule(BaseModel):
    id: str
    kind: FindingKind
    when_all: list[str | list[str]] = Field(
        description="Conjunction; a nested list is a disjunction, e.g. ['rx:ace', ['k_gt_5_5', 'egfr_lt_30', 'k_unknown']]"
    )
    message: str
    source: str = Field(min_length=3)


class LabRecencyRule(BaseModel):
    id: str
    medication_class: str
    loinc: str | list[str] = Field(description="Any one of these codes satisfies the rule")
    max_age_days: int = Field(gt=0)
    source: str = Field(min_length=3)


class VerifierFinding(BaseModel):
    rule_id: str
    kind: FindingKind
    owner: Owner
    message: str
    source: str
    unresolved_predicates: list[str] = Field(default_factory=list)

    @property
    def is_unknown(self) -> bool:
        return bool(self.unresolved_predicates)


class ResolutionAction(StrEnum):
    ACCEPTED = "accepted"  # the change the finding asks for was made
    OVERRIDDEN = "overridden"


class Resolution(BaseModel):
    rule_id: str
    action: ResolutionAction
    actor: str
    reason_code: str | None = None

    def valid(self) -> bool:
        return self.action == ResolutionAction.ACCEPTED or bool(self.reason_code)


class GateDecision(StrEnum):
    PROCEED = "proceed"
    BLOCK_UNTIL_RESOLVED = "block_until_resolved"  # L_cond = 1, or held for a missing fact (rung 0-1)
    HALT_AND_ESCALATE = "halt_and_escalate"  # any finding in an autonomous lane (rung 2-3)
    DEFER = "defer"  # a predicate has L > U: the facts conflict


def _term_bounds(term: str | list[str], predicates: dict[str, Bounds]) -> tuple[Bounds, list[str]]:
    names = [term] if isinstance(term, str) else term
    bounds = [predicates.get(n, UNKNOWN) for n in names]
    out = FALSE
    for b in bounds:
        out = or_(out, b)
    return out, [n for n, b in zip(names, bounds) if b.is_unknown or b.is_contradiction]


def evaluate_rules(rules: list[GateRule], predicates: dict[str, Bounds]) -> list[VerifierFinding]:
    """L_cond = 1 blocks; U_cond = 1 with L_cond < 1 holds and asks for the missing fact."""
    out: list[VerifierFinding] = []
    for rule in rules:
        terms = [_term_bounds(t, predicates) for t in rule.when_all]
        antecedent = all_of([b for b, _ in terms])
        if antecedent.upper < 1 - 1e-9 and not any(b.is_contradiction for b, _ in terms):
            continue
        unresolved = [n for _, names in terms for n in names]
        if antecedent.is_true or unresolved:
            out.append(
                VerifierFinding(
                    rule_id=rule.id,
                    kind=rule.kind,
                    owner=ROUTING[rule.kind],
                    message=rule.message,
                    source=rule.source,
                    unresolved_predicates=[] if antecedent.is_true else unresolved,
                )
            )
    return out


def evaluate_lab_recency(
    rules: list[LabRecencyRule], medication_classes: set[str], last_result: dict[str, date], today: date
) -> list[VerifierFinding]:
    out: list[VerifierFinding] = []
    for rule in rules:
        if rule.medication_class not in medication_classes:
            continue
        codes = [rule.loinc] if isinstance(rule.loinc, str) else rule.loinc
        seen = max((last_result[c] for c in codes if c in last_result), default=None)
        if seen is None or (today - seen).days > rule.max_age_days:
            out.append(
                VerifierFinding(
                    rule_id=rule.id,
                    kind=FindingKind.LAB_RECENCY,
                    owner=ROUTING[FindingKind.LAB_RECENCY],
                    message=f"{rule.medication_class}: LOINC {'/'.join(codes)} older than {rule.max_age_days} days or absent",
                    source=rule.source,
                )
            )
    return out


def decide(
    lane: LaneConfig,
    findings: list[VerifierFinding],
    resolutions: list[Resolution],
    predicates: dict[str, Bounds] | None = None,
) -> GateDecision:
    if predicates and any(b.is_contradiction for b in predicates.values()):
        return GateDecision.DEFER
    if not findings:
        return GateDecision.PROCEED
    if autonomous(lane):
        return GateDecision.HALT_AND_ESCALATE
    resolved = {r.rule_id for r in resolutions if r.valid()}
    return GateDecision.PROCEED if all(f.rule_id in resolved for f in findings) else GateDecision.BLOCK_UNTIL_RESOLVED


def lab_bounds(
    value: float | None,
    observed: date | None,
    *,
    op: Literal["gt", "ge", "lt", "le"],
    threshold: float,
    max_age_days: int,
    today: date,
) -> Bounds:
    """A lab predicate such as 'potassium > 5.5'. Missing or older than the lane window is unknown [0, 1]."""
    if value is None or observed is None or (today - observed).days > max_age_days:
        return UNKNOWN
    hit = {"gt": value > threshold, "ge": value >= threshold, "lt": value < threshold, "le": value <= threshold}[op]
    return TRUE if hit else FALSE


class RuleReview(BaseModel):
    """Rule files need both leads: the physician and the pharmacist sign before a lane deploys."""
    physician_approved_by: str | None = None
    physician_approved_on: str | None = None
    pharmacist_approved_by: str | None = None
    pharmacist_approved_on: str | None = None

    @property
    def complete(self) -> bool:
        return bool(self.physician_approved_by and self.pharmacist_approved_by)


class RuleSet(BaseModel):
    version: str
    clinical_review: RuleReview = Field(default_factory=RuleReview)
    rules: list[GateRule] = Field(default_factory=list)
    lab_recency: list[LabRecencyRule] = Field(default_factory=list)


def load_rules(path: Path) -> RuleSet:
    return RuleSet.model_validate(yaml.safe_load(path.read_text()))


LANE_RULE_FILES: dict[str, list[str]] = {
    "refill": ["refill.yaml"],
    "htn_titration": ["refill.yaml", "titration.yaml"],
    "dm_titration": ["refill.yaml", "titration.yaml"],
    "contraception": ["contraception.yaml"],
}


def load_lane_rules(lane: str, rules_dir: Path) -> RuleSet:
    """Compose the rule files a lane runs; rule ids must be unique across the composed set."""
    sets = [load_rules(rules_dir / name) for name in LANE_RULE_FILES.get(lane, [])]
    rules = [r for rs in sets for r in rs.rules]
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate rule ids across composed rule sets: {dupes}")
    return RuleSet(
        version="+".join(f"{n}@{rs.version}" for n, rs in zip(LANE_RULE_FILES.get(lane, []), sets)),
        rules=rules,
        lab_recency=[r for rs in sets for r in rs.lab_recency],
    )
