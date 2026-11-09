from datetime import date

from compass_pcp.engine.truth import FALSE, TRUE, Bounds
from compass_pcp.gate import (
    FindingKind,
    GateDecision,
    GateRule,
    LabRecencyRule,
    Owner,
    Resolution,
    ResolutionAction,
    decide,
    evaluate_lab_recency,
    evaluate_rules,
    lab_bounds,
)
from compass_pcp.lanes import ActionKind, Lane, LaneConfig, Rung, permitted
from compass_pcp.monitoring import Check, ContractStatus, LabValue, MonitoringContract, evaluate_contract

RULE = GateRule(id="r1", kind=FindingKind.ALLERGY, when_all=["rx:x", "allergy:x"], message="m", source="test")


def test_rule_fires_when_true_and_on_unknown():
    assert evaluate_rules([RULE], {"rx:x": TRUE, "allergy:x": TRUE})[0].owner == Owner.PHARMACIST
    unknown = evaluate_rules([RULE], {"rx:x": TRUE})
    assert unknown[0].unresolved_predicates == ["allergy:x"]
    assert evaluate_rules([RULE], {"rx:x": TRUE, "allergy:x": FALSE}) == []


def test_findings_block_until_coded_resolution():
    lane = LaneConfig(lane=Lane.REFILL, rung=Rung.SIGN_EVERY_ITEM)
    f = evaluate_rules([RULE], {"rx:x": TRUE, "allergy:x": TRUE})
    assert decide(lane, f, []) == GateDecision.BLOCK_UNTIL_RESOLVED
    bare = Resolution(rule_id="r1", action=ResolutionAction.OVERRIDDEN, actor="md")
    assert decide(lane, f, [bare]) == GateDecision.BLOCK_UNTIL_RESOLVED
    coded = bare.model_copy(update={"reason_code": "allergy_is_intolerance"})
    assert decide(lane, f, [coded]) == GateDecision.PROCEED


def test_autonomous_lane_halts():
    lane = LaneConfig(lane=Lane.REFILL, rung=Rung.BATCH_REVIEW)
    f = evaluate_rules([RULE], {"rx:x": TRUE, "allergy:x": TRUE})
    assert decide(lane, f, []) == GateDecision.HALT_AND_ESCALATE


def test_lab_recency():
    rule = LabRecencyRule(id="l1", medication_class="ace", loinc="2823-3", max_age_days=365, source="test")
    today = date(2026, 10, 27)
    assert evaluate_lab_recency([rule], {"ace"}, {"2823-3": date(2026, 1, 1)}, today) == []
    assert evaluate_lab_recency([rule], {"ace"}, {}, today)[0].kind == FindingKind.LAB_RECENCY


def test_lane_permissions():
    assert not permitted(LaneConfig(lane=Lane.URI, rung=Rung.SIGN_EVERY_ITEM), ActionKind.PRESCRIBE_NEW)
    assert permitted(LaneConfig(lane=Lane.REFILL, rung=Rung.SIGN_EVERY_ITEM), ActionKind.PRESCRIBE_RENEWAL)


def test_monitoring_contract():
    c = MonitoringContract(id="ace", trigger="ace", window_days=(7, 14),
                           checks=[Check(loinc="K", max_abs_rise=0.5), Check(loinc="Cr", max_rel_rise=0.30)], source="test")
    start, base = date(2026, 10, 1), {"K": 4.0, "Cr": 1.0}
    ok = [LabValue(loinc="K", value=4.2, on=date(2026, 10, 9)), LabValue(loinc="Cr", value=1.1, on=date(2026, 10, 9))]
    assert evaluate_contract(c, start, base, ok, date(2026, 10, 10)) == ContractStatus.OK
    bad = [LabValue(loinc="K", value=4.6, on=date(2026, 10, 9)), ok[1]]
    assert evaluate_contract(c, start, base, bad, date(2026, 10, 10)) == ContractStatus.BREACH
    assert evaluate_contract(c, start, base, [], date(2026, 10, 10)) == ContractStatus.PENDING
    assert evaluate_contract(c, start, base, [], date(2026, 10, 20)) == ContractStatus.OVERDUE


ACE = GateRule(id="ace", kind=FindingKind.COMMISSION, when_all=["rx:ace", ["k_gt_5_5", "egfr_lt_30"]], message="m", source="test")


def test_or_terms_and_missing_lab_hold():
    today = date(2026, 10, 27)
    stale = lab_bounds(4.1, date(2025, 1, 1), op="gt", threshold=5.5, max_age_days=365, today=today)
    assert stale.is_unknown
    fresh_ok = lab_bounds(4.1, date(2026, 9, 1), op="gt", threshold=5.5, max_age_days=365, today=today)
    assert fresh_ok.is_false
    assert evaluate_rules([ACE], {"rx:ace": TRUE, "k_gt_5_5": fresh_ok, "egfr_lt_30": FALSE}) == []
    held = evaluate_rules([ACE], {"rx:ace": TRUE, "k_gt_5_5": stale, "egfr_lt_30": FALSE})
    assert held[0].unresolved_predicates == ["k_gt_5_5"]
    assert evaluate_rules([ACE], {"rx:ace": TRUE, "k_gt_5_5": TRUE})[0].unresolved_predicates == []


def test_conflicting_facts_defer():
    lane = LaneConfig(lane=Lane.REFILL, rung=Rung.SIGN_EVERY_ITEM)
    preds = {"allergy:x": Bounds(1.0, 0.0), "rx:x": TRUE}
    assert decide(lane, evaluate_rules([RULE], preds), [], preds) == GateDecision.DEFER


def test_inclusive_comparisons_and_multi_loinc():
    today = date(2026, 10, 27)
    assert lab_bounds(1.5, today, op="ge", threshold=1.5, max_age_days=30, today=today).is_true
    assert lab_bounds(1.5, today, op="gt", threshold=1.5, max_age_days=30, today=today).is_false
    rule = LabRecencyRule(id="l2", medication_class="metformin", loinc=["2160-0", "62238-1"], max_age_days=365, source="test")
    assert evaluate_lab_recency([rule], {"metformin"}, {"62238-1": date(2026, 6, 1)}, today) == []


def test_contract_band_catches_fall_and_ceiling():
    c = MonitoringContract(id="d", trigger="thiazide", window_days=(7, 28),
                           checks=[Check(loinc="K", low=3.5, high=5.5)], source="test")
    start = date(2026, 10, 1)
    low = [LabValue(loinc="K", value=3.1, on=date(2026, 10, 9))]
    assert evaluate_contract(c, start, {"K": 4.0}, low, date(2026, 10, 10)) == ContractStatus.BREACH


def test_titration_lane_composes_refill_rules():
    from pathlib import Path

    from compass_pcp.gate import load_lane_rules

    d = Path(__file__).resolve().parents[1] / "compass_pcp" / "data" / "rules"
    rs = load_lane_rules("htn_titration", d)
    assert "refill.yaml@" in rs.version and "titration.yaml@" in rs.version and len(rs.rules) > 100
