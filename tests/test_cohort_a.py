from compass_pcp.harness.cohort_a import analyze, flags, label
from compass_pcp.ledger import HazardTier

R = int(HazardTier.ROUTINE_FOLLOWUP)


def _row(kev, llm, urgent, **case):
    return {"kev_tier": kev, "llm_tier": llm, "kev_urgent": urgent, "flags": flags(kev, llm, None), **label(case)}


def test_label_splits_safety_from_coordination():
    assert label({"detection_truth": 1, "hazard_category": "med_safety", "action_truth": "Contact Doctor"})["safety_positive"]
    other = label({"detection_truth": 1, "hazard_category": "other_hazard", "action_truth": "Contact Doctor"})
    assert not other["safety_positive"] and other["coordination_positive"]
    assert label({"detection_truth": 0, "action_truth": "Call 911/988"})["safety_positive"]


def test_frozen_flag_is_the_higher_tier_and_gate_logic():
    rows = [_row(R, 0, 0.3, detection_truth=1, hazard_category="med_safety", action_truth="Contact Doctor"),
            _row(0, R, 0.0, detection_truth=0, action_truth="Call 911/988"),
            _row(0, 0, 0.0, detection_truth=0, action_truth="None"),
            _row(0, 0, 0.01, detection_truth=1, hazard_category="other_hazard", action_truth="Contact Doctor")]
    out = analyze(rows)
    f = out["screens"]["frozen"]
    assert f["sensitivity"]["caught"] == 2 and f["call_911"]["caught"] == 1 and out["VE_A4"]["missed_call_911"] == 0
    assert out["screens"]["kev_only"]["call_911"]["caught"] == 0  # the LLM screen alone caught the 911 message
    assert out["VE_A4"]["pass"] is False  # 2 of 2 has a lower Wilson bound far below 0.80
    assert out["VE_A5"] == {"caught": 0, "positives": 1, "wilson95": out["VE_A5"]["wilson95"]}
    assert out["mcnemar_vs_frozen_on_positives"]["kev_only"]["frozen_only"] == 1
