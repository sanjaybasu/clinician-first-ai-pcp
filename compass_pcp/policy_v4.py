"""COMPASS v4 policy files (v4_spec.md A1, A2, B1).

Crisis criteria, required instructions, fallbacks and contract wording are data in ``compass_pcp/data/policies/``,
each with a ``source``. This module only validates and loads them, and matches the family lexicons. A lexicon hit is
an applicability trigger for the router; it never sets a route or a floor by itself.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

POLICY_DIR = Path(__file__).resolve().parent / "data" / "policies"

Route = Literal["emergency_911", "crisis_988_page", "er_now", "urgent_review", "poison_control_addition", "none"]
ROUTES: tuple[str, ...] = ("emergency_911", "crisis_988_page", "er_now", "urgent_review", "poison_control_addition", "none")
INSTRUCTION_ROUTES: tuple[str, ...] = ("emergency_911", "crisis_988_page", "er_now", "poison_control_addition")
PolicyStatus = Literal["placeholder_unsigned", "signed"]


class Sourced(BaseModel):
    source: str = Field(min_length=3)


class Instruction(Sourced):
    id: str
    text: str = Field(min_length=3, description="Required clause addressed to the person at risk")
    text_other: str = Field(min_length=3, description="Required clause for someone writing about another person")
    fallback: str = Field(min_length=3)
    fallback_other: str = Field(min_length=3)
    page_care_team: bool = False
    required_clause: str | None = Field(default=None, description="Short clause the reply must contain (any case)")
    clause_max_sentences: int | None = Field(default=None, ge=1, description="The required clause starts within this "
                                             "many sentences, before any question")


class CriterionInstruction(Sourced):
    """Signed per-criterion wording. It replaces the route's text and fallback for that criterion."""

    id: str
    text: str = Field(min_length=3)
    text_other: str | None = None
    fallback: str | None = None
    fallback_other: str | None = None


class OrderException(Sourced):
    """Signed exception to the route's required clause and its position, for one criterion."""

    required_clause: str = Field(min_length=3)
    max_sentences: int = Field(ge=1)


class Criterion(Sourced):
    id: str
    description: str = Field(min_length=10)
    route: Route
    routing_relevant: bool = True
    requires_current: bool = True
    source_ids: list[str] = Field(default_factory=list)
    instruction: CriterionInstruction | None = None
    order_exception: OrderException | None = None


class Family(BaseModel):
    id: str
    name: str
    lexicon: list[str] = Field(default_factory=list)
    classifier_cut: float | None = Field(default=None, ge=0, le=1)
    classifier_cut_source: str | None = None
    # Paper 2 development (v18): where an uncertain, unquoted or currentness-unknown item of this family routes; None
    # keeps the policy-wide uncertainty.uncertain_route. Incomplete items and screen failures are unaffected.
    uncertain_route: Route | None = None
    criteria: list[Criterion] = Field(min_length=1)


class UncertaintyPolicy(Sourced):
    uncertain_route: Route = "urgent_review"
    incomplete_route: Route = "urgent_review"
    screen_failure_route: Route = "urgent_review"
    present_without_valid_quote: Literal["uncertain_applicable", "uncertain"] = "uncertain_applicable"
    unknown_currentness: Literal["uncertain"] = "uncertain"


class ScreenSettings(Sourced):
    timeout_seconds: float = Field(gt=0)


class CrisisPolicy(BaseModel):
    version: str
    status: PolicyStatus
    signed_by: str | None = None
    signed_on: str | None = None
    screen: ScreenSettings
    route_precedence: list[Route]
    additions: list[Route] = Field(default_factory=list)
    additions_suppressed_by: list[Route] = Field(default_factory=list)
    instructions: dict[str, Instruction]
    uncertainty: UncertaintyPolicy
    families: list[Family] = Field(min_length=1)
    sources: dict[str, str] = Field(default_factory=dict, description="Source id -> citation")

    @model_validator(mode="after")
    def _consistent(self) -> "CrisisPolicy":
        if len(self.route_precedence) != len(set(self.route_precedence)) or "none" not in self.route_precedence:
            raise ValueError("route_precedence must list each route once, including none")
        unc = self.uncertainty
        used = {c.route for f in self.families for c in f.criteria} | set(self.additions) | set(
            self.additions_suppressed_by) | {unc.uncertain_route, unc.incomplete_route, unc.screen_failure_route} | {
            f.uncertain_route for f in self.families if f.uncertain_route}
        if used - set(self.route_precedence):
            raise ValueError(f"route_precedence is missing {sorted(used - set(self.route_precedence))}")
        ids = [c.id for f in self.families for c in f.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        for f in self.families:
            for c in f.criteria:
                if c.route in INSTRUCTION_ROUTES and c.route not in self.instructions:
                    raise ValueError(f"criterion {c.id} routes to {c.route}, which has no instruction")
                if self.sources and (missing := [s for s in c.source_ids if s not in self.sources]):
                    raise ValueError(f"criterion {c.id} cites unknown sources {missing}")
        if self.status == "signed" and not (self.signed_by and self.signed_on):
            raise ValueError("a signed policy names its signer and date")
        return self

    def criterion(self, cid: str) -> tuple[Family, Criterion]:
        for f in self.families:
            for c in f.criteria:
                if c.id == cid:
                    return f, c
        raise KeyError(cid)

    def rank(self, route: str) -> int:
        """0 is the most urgent route."""
        return self.route_precedence.index(route)  # type: ignore[arg-type]


class Referral(Sourced):
    text: str | None = None
    text_other: str | None = None
    position: Literal["first", "any", "none"] = "any"
    fallback: str = Field(min_length=3)
    fallback_other: str | None = None

    @model_validator(mode="after")
    def _position_needs_text(self) -> "Referral":
        if self.position != "none" and not self.text:
            raise ValueError("a referral with a position needs text")
        return self


class ContractChecks(BaseModel):
    referral_max_sentences: int = Field(default=2, ge=1)
    question_overlap_min: float = Field(default=0.6, gt=0, le=1)


DISPOSITIONS = ("escalate", "possible_emergency", "urgent_same_day", "ask", "defer", "complete", "out_of_engine")


class ContractPolicy(Sourced):
    version: str
    status: PolicyStatus
    signed_by: str | None = None
    signed_on: str | None = None
    referrals: dict[str, Referral]
    prohibited: list[str] = Field(min_length=1)
    page_claim_patterns: list[str] = Field(default_factory=list)
    checks: ContractChecks = Field(default_factory=ContractChecks)

    @model_validator(mode="after")
    def _complete(self) -> "ContractPolicy":
        missing = [d for d in DISPOSITIONS if d not in self.referrals]
        if missing:
            raise ValueError(f"contract policy has no referral entry for {missing}")
        for p in self.page_claim_patterns:
            re.compile(p)
        return self


def load_crisis_policy(path: Path | None = None, *, require_signed: bool = False) -> CrisisPolicy:
    """Load and validate the crisis policy. With ``require_signed`` (production), refuse any unsigned policy."""
    policy = _load_crisis(str(path or POLICY_DIR / "crisis_v4.yaml"))
    if require_signed and policy.status != "signed":
        raise ValueError(f"crisis policy {policy.version} has status {policy.status}; a signed policy is required")
    return policy


def load_contract_policy(path: Path | None = None) -> ContractPolicy:
    return _load_contract(str(path or POLICY_DIR / "contract_v4.yaml"))


@lru_cache(maxsize=8)
def _load_crisis(path: str) -> CrisisPolicy:
    return CrisisPolicy.model_validate(yaml.safe_load(Path(path).read_text()))


@lru_cache(maxsize=8)
def _load_contract(path: str) -> ContractPolicy:
    return ContractPolicy.model_validate(yaml.safe_load(Path(path).read_text()))


def lexicon_hits(text: str, policy: CrisisPolicy) -> dict[str, list[str]]:
    """Family id -> lexicon terms found in the text (case-insensitive, whole words). Applicability trigger only."""
    out: dict[str, list[str]] = {}
    low = text.lower()
    for f in policy.families:
        hits = [t for t in f.lexicon if re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", low)]
        if hits:
            out[f.id] = hits
    return out


def load_first_aid(path: Path) -> dict[str, list[tuple[str, bool]]]:
    """Crisis criterion id -> (passage text, adult_only) pairs, in order (compass_pcp/data/policies/first_aid_*.yaml)."""
    data = yaml.safe_load(path.read_text())
    for k, v in data["passages"].items():
        if not v.get("sources"):
            raise ValueError(f"first-aid passage {k} has no source")
    passages = {k: (v["text"], bool(v.get("adult_only", False))) for k, v in data["passages"].items()}
    return {crit: [passages[p] for p in ids] for crit, ids in data.get("criteria", {}).items()}
