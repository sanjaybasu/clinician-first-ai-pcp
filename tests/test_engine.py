import math

import pytest

from compass_pcp.engine.bayes import differential, posterior
from compass_pcp.engine.deferral import DeferralReason
from compass_pcp.engine.next_action import score_actions
from compass_pcp.engine.scripts import IllnessScript
from compass_pcp.engine.stopping import CannotMissStatus, check_cannot_miss
from compass_pcp.engine.workup import DecisionKind, WorkupConfig, WorkupState, step
from compass_pcp.findings import Polarity, Provenance
from tests.conftest import make_finding


def by_id(script, i):
    return next(d for d in script.diagnoses if d.id == i)


# The pre-adoption engine (posterior rule-out authority, EIG questions), for tests of those mechanisms in isolation.
POSTERIOR_EIG = {"bayes_rule_out_authority": True, "question_policy": "eig"}

def test_log_odds_update_matches_hand_calculation(script):
    f = {"recent_lifting": make_finding("recent_lifting")}
    p = posterior(by_id(script, "mechanical"), script, f).p
    odds = 0.6 / 0.4 * 3.0
    assert math.isclose(p, odds / (1 + odds))


def test_correlated_group_applies_strongest_only(script):
    f = {"saddle_anesthesia": make_finding("saddle_anesthesia"), "urinary_retention": make_finding("urinary_retention")}
    post = posterior(by_id(script, "cauda_equina"), script, f)
    assert [a.feature_id for a in post.applied] == ["saddle_anesthesia"]
    assert any("urinary_retention:weaker_in_group" in s for s in post.skipped)


def test_unconfirmed_patient_negative_cannot_rule_out(script):
    neg = make_finding("saddle_anesthesia", Polarity.ABSENT, Provenance.PATIENT_REPORT)
    post = posterior(by_id(script, "cauda_equina"), script, {"saddle_anesthesia": neg})
    assert post.applied == [] and math.isclose(post.p, 0.01)
    rec = make_finding("saddle_anesthesia", Polarity.ABSENT, Provenance.STRUCTURED_RECORD)
    assert posterior(by_id(script, "cauda_equina"), script, {"saddle_anesthesia": rec}).p < 0.01
    confirmed = neg.model_copy(update={"confirmed": True})
    assert posterior(by_id(script, "cauda_equina"), script, {"saddle_anesthesia": confirmed}).p < 0.01


def test_draw_interval_brackets_point_and_replays(script):
    f = {"saddle_anesthesia": make_finding("saddle_anesthesia")}
    a = posterior(by_id(script, "cauda_equina"), script, f)
    b = posterior(by_id(script, "cauda_equina"), script, f)
    assert a.p05 < a.p < a.p95 and a == b


def test_stopping_rule_escalates_on_positive_red_flag(script):
    f = {"saddle_anesthesia": make_finding("saddle_anesthesia")}
    checks = check_cannot_miss(script, differential(script, f), set(f))
    assert checks[0].status == CannotMissStatus.ESCALATE


def test_eig_never_asks_known_or_non_remote(script):
    f = {"recent_lifting": make_finding("recent_lifting")}
    current = {k: v.p for k, v in differential(script, f).items()}
    ids = [a.feature_id for a in score_actions(script, f, current)]
    assert "recent_lifting" not in ids and "rectal_exam" not in ids


def test_eig_weights_cannot_miss(script):
    current = {k: v.p for k, v in differential(script, {}).items()}
    top = score_actions(script, {}, current)[0]
    assert top.feature_id in {"saddle_anesthesia", "urinary_retention"} and 0 < top.p_positive < 1


def test_pauker_kassirer():
    from compass_pcp.engine.scripts import LikelihoodRatio, pauker_kassirer, sens_spec

    t_test, t_treat = pauker_kassirer(50, 1, 0.0, 1.0, 0.0)
    assert math.isclose(t_treat, 1 / 51) and t_test == t_treat
    t_test, t_treat = pauker_kassirer(50, 1, 0.95, 0.90, 0.0)
    assert 0 < t_test < t_treat < 1
    se, sp = sens_spec(LikelihoodRatio(lr_pos=0.9 / 0.2, lr_neg=0.1 / 0.8, source="test"))
    assert math.isclose(se, 0.9) and math.isclose(sp, 0.8)


def test_workup_asks_cannot_miss_question_first(script):
    d = step(script, WorkupState(findings=[]))
    assert d.kind == DecisionKind.ASK
    assert d.ask_features and set(d.ask_features) <= {"saddle_anesthesia", "urinary_retention"}


def test_workup_escalates(script):
    d = step(script, WorkupState(findings=[make_finding("urinary_retention")]))
    assert d.kind == DecisionKind.ESCALATE


def test_workup_defers_on_contradiction(script):
    d = step(script, WorkupState(findings=[make_finding("radiating_leg"), make_finding("radiating_leg", Polarity.ABSENT)]))
    assert d.kind == DecisionKind.DEFER and DeferralReason.CONTRADICTION in d.certificate.deferral_reasons


def test_workup_defers_when_only_patient_negatives_and_budget_spent(script):
    negs = [make_finding(x, Polarity.ABSENT) for x in ("saddle_anesthesia", "urinary_retention")]
    d = step(script, WorkupState(findings=negs, turns_used=12))
    assert d.kind == DecisionKind.DEFER
    assert DeferralReason.CANNOT_MISS_OPEN_AFTER_BUDGET in d.certificate.deferral_reasons


def test_workup_completes_when_record_rules_out_and_entropy_low(script):
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("saddle_anesthesia", "urinary_retention")]
    rec.append(make_finding("rectal_exam", Polarity.ABSENT, Provenance.STRUCTURED_RECORD))
    d = step(script.model_copy(update={"lane": "uri"}), WorkupState(findings=rec), WorkupConfig(eta=10.0, **POSTERIOR_EIG))
    assert d.kind == DecisionKind.COMPLETE, d.certificate.cannot_miss


def test_workup_defers_outside_lane(script):
    d = step(script, WorkupState(findings=[], in_lane=False))
    assert d.kind == DecisionKind.DEFER


def test_workup_defers_on_schema_gap_and_run_disagreement(script):
    d = step(script, WorkupState(findings=[], llm_proposed_diagnoses=["spinal_epidural_abscess"]))
    assert d.kind == DecisionKind.DEFER and DeferralReason.SCHEMA_GAP in d.certificate.deferral_reasons
    d = step(script, WorkupState(findings=[], runs_disagree_on=["disposition"]))
    assert d.kind == DecisionKind.DEFER


def test_residual_uncertainty_defers_after_budget(script):
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("urinary_retention", "rectal_exam")]
    d = step(script, WorkupState(findings=rec, turns_used=12))
    assert d.kind == DecisionKind.DEFER and DeferralReason.RESIDUAL_UNCERTAINTY in d.certificate.deferral_reasons


def test_script_rejects_unsourced_or_bad_ci():
    with pytest.raises(ValueError):
        IllnessScript.model_validate({
            "complaint": "x", "version": "0", "lane": "uri",
            "features": [{"id": "f", "question": "q"}],
            "diagnoses": [{"id": "d", "label": "d", "prior": {"p": 0.1, "source": "s"},
                           "lrs": {"f": {"lr_pos": 2.0, "lr_neg": 0.5, "lr_pos_ci": [3.0, 4.0], "source": "src"}}}],
        })
    with pytest.raises(ValueError):
        IllnessScript.model_validate({
            "complaint": "x", "version": "0", "lane": "uri", "features": [],
            "diagnoses": [{"id": "d", "label": "d", "cannot_miss": True, "prior": {"p": 0.1, "source": "s"},
                           "thresholds": {"source": "s"}}],
        })


def test_must_ask_blocks_rule_out_by_prior(script):
    gated = script.model_copy(update={"must_ask": ["radiating_leg"]})
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("urinary_retention", "rectal_exam")]
    d = step(gated, WorkupState(findings=rec), WorkupConfig(eta=10.0))
    assert d.kind == DecisionKind.ASK and d.ask_features == ["radiating_leg"]
    d = step(gated, WorkupState(findings=rec, turns_used=12), WorkupConfig(eta=10.0))
    assert d.kind == DecisionKind.DEFER and DeferralReason.RED_FLAGS_UNASKED in d.certificate.deferral_reasons
    d = step(gated.model_copy(update={"lane": "uri"}),
             WorkupState(findings=[*rec, make_finding("radiating_leg", Polarity.ABSENT)]), WorkupConfig(eta=10.0, **POSTERIOR_EIG))
    assert d.kind == DecisionKind.COMPLETE


def test_present_red_flag_without_lr_escalates_or_defers(script):
    gated = script.model_copy(update={"must_ask": ["radiating_leg", "recent_lifting"]})
    d = step(gated, WorkupState(findings=[make_finding("recent_lifting")]))
    assert d.kind == DecisionKind.ESCALATE
    deferring = gated.model_copy(update={"red_flag_action": {"recent_lifting": "defer"}})
    d = step(deferring, WorkupState(findings=[make_finding("recent_lifting")]))
    assert d.kind == DecisionKind.DEFER and DeferralReason.RED_FLAG_PRESENT in d.certificate.deferral_reasons


def test_high_risk_lane_never_closes(script):
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("urinary_retention", "rectal_exam")]
    assert script.lane == "undifferentiated_high_risk"
    d = step(script, WorkupState(findings=rec), WorkupConfig(eta=10.0, **POSTERIOR_EIG))
    assert d.kind == DecisionKind.DEFER and DeferralReason.LANE_NEVER_CLOSES in d.certificate.deferral_reasons


def test_obligation_policy_skips_non_pivotal_questions(script):
    uri = script.model_copy(update={"lane": "uri"})
    partial = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("urinary_retention", "rectal_exam")]
    d = step(uri, WorkupState(findings=partial), WorkupConfig(question_policy="obligation", bayes_rule_out_authority=True))
    assert d.kind == DecisionKind.ASK and d.ask_features == ["saddle_anesthesia"]  # a yes would reopen cauda equina
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD)
           for x in ("urinary_retention", "rectal_exam", "saddle_anesthesia")]
    eig = step(uri, WorkupState(findings=rec), WorkupConfig(**POSTERIOR_EIG))
    obl = step(uri, WorkupState(findings=rec), WorkupConfig(question_policy="obligation", bayes_rule_out_authority=True))
    assert eig.kind == DecisionKind.ASK  # entropy rule keeps asking about non-dangerous diagnoses
    assert obl.kind == DecisionKind.COMPLETE  # nothing left can change a cannot-miss status


def test_no_posterior_authority_requires_red_flag_list(script):
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("urinary_retention", "rectal_exam")]
    uri = script.model_copy(update={"lane": "uri"})
    d = step(uri, WorkupState(findings=rec), WorkupConfig(eta=10.0, bayes_rule_out_authority=False))
    assert d.kind == DecisionKind.DEFER and DeferralReason.CANNOT_MISS_NO_RED_FLAGS in d.certificate.deferral_reasons
    flagged = uri.model_copy(update={"must_ask": ["saddle_anesthesia"]})
    d = step(flagged, WorkupState(findings=[*rec, make_finding("saddle_anesthesia", Polarity.ABSENT, Provenance.STRUCTURED_RECORD)]),
             WorkupConfig(eta=10.0, bayes_rule_out_authority=False))
    assert d.kind == DecisionKind.COMPLETE


def test_default_defers_cannot_miss_without_red_flag_screen(script):
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for x in ("saddle_anesthesia", "urinary_retention")]
    d = step(script.model_copy(update={"lane": "uri"}), WorkupState(findings=rec), WorkupConfig(eta=10.0))
    assert d.kind == DecisionKind.DEFER and DeferralReason.CANNOT_MISS_NO_RED_FLAGS in d.certificate.deferral_reasons


def test_listed_red_flag_covers_cannot_miss_without_lr(script):
    dx = [d.model_copy(update={"red_flags": ["radiating_leg"]}) if d.cannot_miss else d for d in script.diagnoses]
    covered = script.model_copy(update={"lane": "uri", "must_ask": ["radiating_leg"], "diagnoses": dx})
    rec = [make_finding(x, Polarity.ABSENT, Provenance.STRUCTURED_RECORD)
           for x in ("saddle_anesthesia", "urinary_retention", "rectal_exam", "radiating_leg")]
    d = step(covered, WorkupState(findings=rec), WorkupConfig(eta=10.0))
    assert DeferralReason.CANNOT_MISS_NO_RED_FLAGS not in d.certificate.deferral_reasons


def test_red_flags_must_be_must_ask(script):
    dx = [d.model_copy(update={"red_flags": ["radiating_leg"]}) if d.cannot_miss else d for d in script.diagnoses]
    with pytest.raises(ValueError):
        type(script).model_validate({**script.model_dump(), "must_ask": [], "diagnoses": [d.model_dump() for d in dx]})
