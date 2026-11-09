"""One deterministic workup step: ask, defer, escalate or complete.

Compass calls ``step`` once per patient turn, after the extractors and the
reconciler. The returned ``Certificate`` records every input, LR source,
threshold and candidate action, so a reviewer can replay the decision exactly.
"""

import os
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from compass_pcp import __version__
from compass_pcp.engine.bayes import Posterior, differential, h
from compass_pcp.engine.deferral import DeferralInputs, DeferralReason, deferral_reasons
from compass_pcp.engine.next_action import ScoredAction, score_actions
from compass_pcp.engine.scripts import IllnessScript
from compass_pcp.engine.stopping import CannotMissCheck, CannotMissStatus, check_cannot_miss
from compass_pcp.engine.truth import predicate_bounds
from compass_pcp.findings import Finding, Polarity
from compass_pcp.lanes import may_close


class DecisionKind(StrEnum):
    ASK = "ask"
    DEFER = "defer"
    ESCALATE = "escalate"
    COMPLETE = "complete"


class WorkupConfig(BaseModel):
    question_budget: int = Field(default=12, gt=0, description="Patient turns")
    questions_per_message: int = Field(default=2, gt=0)
    eta: float = Field(default=0.5, gt=0, description="Residual-uncertainty limit per diagnosis, nats")
    lambda_cost: float = 0.0
    lambda_risk: float = 0.0
    # Ablation switches (harness/configurations.py); production keeps the defaults.
    enforce_stopping_rule: bool = True
    # Panel v1 item 3 (adopted 2026-10-28): a cannot-miss diagnosis may not be closed on the posterior alone;
    # closure needs every must_ask red flag observed and no diagnosis without a red-flag list. True is the ablation.
    bayes_rule_out_authority: bool = False
    # Panel v1 item 5 (adopted 2026-10-28, together with item 3): "obligation" asks only questions whose answer could
    # change a cannot-miss status, then closes; "eig" (ask while residual entropy > eta) is the ablation.
    question_policy: str = Field(default="obligation", pattern="^(eig|obligation)$")
    fixed_question_order: bool = False
    # Paper 2 development: with a cannot-miss diagnosis open and every red flag answered, defer at once when no
    # combination of answers to the remaining remote questions could change any open cannot-miss status.
    futility_stop: bool = False
    # Paper 2 development: ask every must_ask red flag that acts by rule (no likelihood ratio in any diagnosis) before
    # any likelihood question, so an airway red flag is not queued behind the workup of another cannot-miss diagnosis.
    red_flags_first: bool = False
    # Paper 2 development: a feature asked in an earlier reply that still has no answer is not asked again. A must_ask
    # red flag left unanswered defers at once (it can never close the encounter unanswered); a "record" must_ask
    # feature left unanswered is treated as not reported, so the pathway acts only on what the patient reported.
    skip_unanswered: bool = False

    @model_validator(mode="after")
    def _no_ablation_in_production(self) -> "WorkupConfig":
        ablated = not self.enforce_stopping_rule or self.fixed_question_order
        if ablated and os.environ.get("COMPASS_PCP_ENV") == "production":
            raise ValueError("ablation switches are evaluation-only and cannot run with COMPASS_PCP_ENV=production")
        return self


class WorkupState(BaseModel):
    findings: list[Finding] = Field(description="Accepted findings; may repeat a feature from different sources")
    unconfirmed_after_question: list[str] = Field(default_factory=list)
    asked_unanswered: list[str] = Field(default_factory=list, description="Asked in an earlier reply, still unanswered")
    turns_used: int = 0
    runs_disagree_on: list[str] = Field(default_factory=list)
    in_lane: bool = True
    llm_proposed_diagnoses: list[str] = Field(default_factory=list)


class Certificate(BaseModel):
    engine_version: str = __version__
    script: str
    script_version: str
    posteriors: dict[str, Posterior]
    cannot_miss: list[CannotMissCheck]
    contradictions: list[str]
    max_entropy: float
    deferral_reasons: list[DeferralReason]
    candidate_actions: list[ScoredAction]


class Decision(BaseModel):
    kind: DecisionKind
    ask_features: list[str] = Field(default_factory=list)
    certificate: Certificate


_IMMEDIATE = {
    DeferralReason.OUTSIDE_LANE,
    DeferralReason.CONTRADICTION,
    DeferralReason.UNCONFIRMED_DECISION_FINDING,
    DeferralReason.CANNOT_MISS_NO_REMOTE_TEST,
    DeferralReason.RUN_DISAGREEMENT,
    DeferralReason.SCHEMA_GAP,
}


def step(script: IllnessScript, state: WorkupState, config: WorkupConfig | None = None) -> Decision:
    config = config or WorkupConfig()
    bounds = predicate_bounds(state.findings)
    contradictions = sorted(k for k, b in bounds.items() if b.is_contradiction)
    usable = {
        f.feature_id: f
        for f in sorted(state.findings, key=lambda f: (f.provenance, f.confirmed))
        if f.feature_id not in contradictions
    }

    posteriors = differential(script, usable)
    checks = check_cannot_miss(script, posteriors, observed=set(usable) | set(contradictions))
    max_entropy = max((h(p.p) for p in posteriors.values()), default=0.0)
    budget_spent = state.turns_used >= config.question_budget
    known_dx = {d.id for d in script.diagnoses}
    reasons = deferral_reasons(
        DeferralInputs(
            in_lane=state.in_lane,
            contradictions=contradictions,
            unconfirmed_after_question=state.unconfirmed_after_question,
            decision_features={fid for d in script.diagnoses for fid in d.lrs},
            cannot_miss=checks,
            runs_disagree_on=state.runs_disagree_on,
            max_entropy=max_entropy,
            eta=config.eta,
            budget_spent=budget_spent,
            schema_gaps=[d for d in state.llm_proposed_diagnoses if d not in known_dx],
        )
    )
    actions = score_actions(
        script,
        usable,
        {k: v.p for k, v in posteriors.items()},
        lambda_cost=config.lambda_cost,
        lambda_risk=config.lambda_risk,
    )
    cert = Certificate(
        script=script.complaint,
        script_version=script.version,
        posteriors=posteriors,
        cannot_miss=checks,
        contradictions=contradictions,
        max_entropy=max_entropy,
        deferral_reasons=reasons,
        candidate_actions=actions,
    )

    if config.fixed_question_order:
        order = {f.id: i for i, f in enumerate(script.features)}
        actions = sorted(actions, key=lambda a: order[a.feature_id])
        cert.candidate_actions = actions
    if not config.enforce_stopping_rule:
        checks = []
        reasons = [r for r in reasons if r not in (
            DeferralReason.CANNOT_MISS_NO_REMOTE_TEST, DeferralReason.CANNOT_MISS_OPEN_AFTER_BUDGET)]
        cert.deferral_reasons = reasons

    # A red flag the patient reports acts by rule, whether or not it has a likelihood ratio.
    present_flags = [
        f for f in script.must_ask
        if f in usable and usable[f].polarity == Polarity.PRESENT and script.red_flag_action.get(f) != "record"
    ]
    if any(script.red_flag_action.get(f, "escalate") == "escalate" for f in present_flags):
        return Decision(kind=DecisionKind.ESCALATE, certificate=cert)
    if present_flags:
        cert.deferral_reasons.append(DeferralReason.RED_FLAG_PRESENT)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)
    if any(c.status == CannotMissStatus.ESCALATE for c in checks):
        return Decision(kind=DecisionKind.ESCALATE, certificate=cert)
    if _IMMEDIATE & set(reasons):
        return Decision(kind=DecisionKind.DEFER, certificate=cert)

    skip = set(state.asked_unanswered) - set(usable) if config.skip_unanswered else set()
    if any(script.red_flag_action.get(f) != "record" for f in skip & set(script.must_ask)):
        cert.deferral_reasons.append(DeferralReason.RED_FLAGS_UNASKED)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)
    if skip:
        actions = [a for a in actions if a.feature_id not in skip]

    if config.red_flags_first and not budget_spent:
        lr_features = {fid for d in script.diagnoses for fid in d.lrs}
        rule_flags = [f for f in script.must_ask
                      if f not in usable and f not in contradictions and f not in lr_features and f not in skip]
        if rule_flags:
            return Decision(kind=DecisionKind.ASK, ask_features=rule_flags[: config.questions_per_message], certificate=cert)

    open_cannot_miss = {c.diagnosis_id for c in checks if c.status == CannotMissStatus.NEEDS_WORKUP}
    positive = [a for a in actions if a.score > 0 or config.fixed_question_order]
    if open_cannot_miss:
        targeted = {fid for d in script.cannot_miss if d.id in open_cannot_miss for fid in d.lrs}
        ask = [a.feature_id for a in positive if a.feature_id in targeted] or [
            a.feature_id for a in actions if a.feature_id in targeted
        ]
        flags_done = all(f in usable or f in contradictions for f in script.must_ask)
        if ask and config.futility_stop and flags_done and not _status_can_change(script, usable, checks, open_cannot_miss):
            cert.deferral_reasons.append(DeferralReason.CANNOT_MISS_FUTILE_QUESTIONS)
            return Decision(kind=DecisionKind.DEFER, certificate=cert)
        if ask and not budget_spent:
            return Decision(kind=DecisionKind.ASK, ask_features=ask[: config.questions_per_message], certificate=cert)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)

    unasked = [f for f in script.must_ask if f not in usable and f not in contradictions and f not in skip]
    if unasked and config.enforce_stopping_rule:
        if not budget_spent:
            return Decision(kind=DecisionKind.ASK, ask_features=unasked[: config.questions_per_message], certificate=cert)
        cert.deferral_reasons.append(DeferralReason.RED_FLAGS_UNASKED)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)

    if not config.bayes_rule_out_authority and any(not d.must_ask_covered for d in _coverage(script)):
        cert.deferral_reasons.append(DeferralReason.CANNOT_MISS_NO_RED_FLAGS)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)

    if config.question_policy == "obligation":
        pivotal = _pivotal_features(script, usable, checks)
        ask = [a.feature_id for a in actions if a.feature_id in pivotal]
        if ask and not budget_spent:
            return Decision(kind=DecisionKind.ASK, ask_features=ask[: config.questions_per_message], certificate=cert)
    else:
        if positive and not budget_spent and max_entropy > config.eta:
            return Decision(
                kind=DecisionKind.ASK,
                ask_features=[a.feature_id for a in positive[: config.questions_per_message]],
                certificate=cert,
            )
        if DeferralReason.RESIDUAL_UNCERTAINTY in reasons:
            return Decision(kind=DecisionKind.DEFER, certificate=cert)
    if not may_close(script.lane):
        cert.deferral_reasons.append(DeferralReason.LANE_NEVER_CLOSES)
        return Decision(kind=DecisionKind.DEFER, certificate=cert)
    return Decision(kind=DecisionKind.COMPLETE, certificate=cert)


class _Coverage(BaseModel):
    diagnosis_id: str
    must_ask_covered: bool


def _coverage(script: IllnessScript) -> list[_Coverage]:
    """Each cannot-miss diagnosis needs at least one must_ask feature that screens for it (an LR feature or a listed
    red flag) to be closable without the posterior."""
    flags = set(script.must_ask)
    return [_Coverage(diagnosis_id=d.id, must_ask_covered=bool((set(d.lrs) | set(d.red_flags)) & flags))
            for d in script.cannot_miss]


def _pivotal_features(script: IllnessScript, usable: dict, checks: list[CannotMissCheck]) -> set[str]:
    """Unasked remote features for which either answer would change some cannot-miss status."""
    from compass_pcp.engine.bayes import differential as _diff
    from compass_pcp.findings import CodeSystem, Provenance

    base = {c.diagnosis_id: c.status for c in checks}
    pivotal: set[str] = set()
    for f in script.features:
        if f.id in usable or not f.remote_askable:
            continue
        for pol in (Polarity.PRESENT, Polarity.ABSENT):
            hyp = Finding(feature_id=f.id, code_system=CodeSystem.LOCAL, code=f.id, polarity=pol,
                          provenance=Provenance.PATIENT_REPORT, extractor_id="pivot", confirmed=True)
            after = check_cannot_miss(script, _diff(script, {**usable, f.id: hyp}), set(usable) | {f.id})
            if any(base.get(c.diagnosis_id) != c.status for c in after):
                pivotal.add(f.id)
                break
    return pivotal


def _status_can_change(script: IllnessScript, usable: dict, checks: list[CannotMissCheck], open_ids: set[str]) -> bool:
    """True if some joint answer to the unasked remote features of the open cannot-miss diagnoses changes the status of
    one of them. For each diagnosis, every feature is set to the answer that lowers (then raises) that diagnosis's
    posterior most; with independent likelihood ratios these joint answers bound every other combination."""
    from compass_pcp.engine.bayes import differential as _diff
    from compass_pcp.findings import CodeSystem, Provenance

    def hyp(fid: str, pol: Polarity) -> Finding:
        return Finding(feature_id=fid, code_system=CodeSystem.LOCAL, code=fid, polarity=pol,
                       provenance=Provenance.PATIENT_REPORT, extractor_id="futility", confirmed=True)

    base = {c.diagnosis_id: c.status for c in checks}
    for dx in script.cannot_miss:
        if dx.id not in open_ids:
            continue
        remaining = [fid for fid in dx.lrs if fid not in usable and script.feature(fid).remote_askable]
        if not remaining:
            continue
        for direction in (min, max):
            chosen = {}
            for fid in remaining:
                ps = {pol: _diff(script, {**usable, fid: hyp(fid, pol)})[dx.id].p for pol in (Polarity.PRESENT, Polarity.ABSENT)}
                chosen[fid] = hyp(fid, direction(ps, key=ps.get))
            after = check_cannot_miss(script, _diff(script, {**usable, **chosen}), set(usable) | set(chosen))
            if any(c.diagnosis_id == dx.id and c.status != base[dx.id] for c in after):
                return True
    return False
