"""Deferral (design doc, step 8). The case defers if any reason holds."""

from enum import StrEnum

from pydantic import BaseModel

from compass_pcp.engine.stopping import CannotMissCheck, CannotMissStatus


class DeferralReason(StrEnum):
    OUTSIDE_LANE = "outside_lane"
    CONTRADICTION = "conflicting_facts"
    UNCONFIRMED_DECISION_FINDING = "unconfirmed_decision_relevant_finding"
    CANNOT_MISS_NO_REMOTE_TEST = "cannot_miss_middle_zone_no_remote_test"
    CANNOT_MISS_OPEN_AFTER_BUDGET = "cannot_miss_middle_zone_after_budget"
    RUN_DISAGREEMENT = "repeated_runs_disagree"
    RESIDUAL_UNCERTAINTY = "residual_uncertainty_after_budget"
    SCHEMA_GAP = "llm_proposed_diagnosis_outside_schema"
    RED_FLAGS_UNASKED = "required_red_flags_unasked_after_budget"
    RED_FLAG_PRESENT = "red_flag_present_defer"
    LANE_NEVER_CLOSES = "lane_history_and_escalation_only"
    CANNOT_MISS_NO_RED_FLAGS = "cannot_miss_without_red_flag_list_and_no_posterior_authority"
    CANNOT_MISS_FUTILE_QUESTIONS = "cannot_miss_middle_zone_no_answer_can_change_it"


class DeferralInputs(BaseModel):
    in_lane: bool
    contradictions: list[str]
    unconfirmed_after_question: list[str]
    decision_features: set[str]
    cannot_miss: list[CannotMissCheck]
    runs_disagree_on: list[str]
    max_entropy: float
    eta: float
    budget_spent: bool
    schema_gaps: list[str]


def deferral_reasons(x: DeferralInputs) -> list[DeferralReason]:
    r: list[DeferralReason] = []
    if not x.in_lane:
        r.append(DeferralReason.OUTSIDE_LANE)
    if x.contradictions:
        r.append(DeferralReason.CONTRADICTION)
    if set(x.unconfirmed_after_question) & x.decision_features:
        r.append(DeferralReason.UNCONFIRMED_DECISION_FINDING)
    if any(c.status == CannotMissStatus.NEEDS_WORKUP and not c.remote_rule_out_possible for c in x.cannot_miss):
        r.append(DeferralReason.CANNOT_MISS_NO_REMOTE_TEST)
    if x.budget_spent and any(c.status == CannotMissStatus.NEEDS_WORKUP for c in x.cannot_miss):
        r.append(DeferralReason.CANNOT_MISS_OPEN_AFTER_BUDGET)
    if x.runs_disagree_on:
        r.append(DeferralReason.RUN_DISAGREEMENT)
    if x.budget_spent and x.max_entropy > x.eta:
        r.append(DeferralReason.RESIDUAL_UNCERTAINTY)
    if x.schema_gaps:
        r.append(DeferralReason.SCHEMA_GAP)
    return r
