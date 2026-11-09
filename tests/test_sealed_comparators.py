"""AMD-PCP-002 Analysis 5: urgency parsing and aggregate metrics on synthetic rows (no patient text)."""

from paper.sealed_comparators import cost, metrics, parse_urgency


def test_parse_urgency():
    assert parse_urgency('{"urgency": "urgent_same_day"}') == "urgent_same_day"
    assert parse_urgency('thinking... {"urgency":"emergency_now"}') == "emergency_now"
    assert parse_urgency("label: none") == "none"
    assert parse_urgency('{"urgency": "soon"}') is None


def _row(i, pos, c911, comp, raw, guard, gflag, reg):
    return {"id": str(i), "safety_positive": pos, "call_911": c911, "urgency":
            {"ai_pcp_v3": comp, "raw_llm": raw, "llm_standard_guardrails": guard},
            "guard_input_flag": gflag, "guard_path": "input_screen" if gflag else "pass", "registered_flag": reg,
            "compass_ledger_flag": reg}


def test_metrics():
    rows = [
        _row(1, True, True, "emergency_now", "routine_followup", "emergency_now", True, True),
        _row(2, True, False, "urgent_same_day", "none", "none", False, True),
        _row(3, True, False, "none", "routine_followup", "routine_followup", False, False),
        _row(4, False, False, "none", "none", "urgent_same_day", False, False),
    ]
    m = metrics(rows, registered=True)
    d = m["designs"]
    assert d["ai_pcp_v3"]["hazard_sensitivity"]["k"] == 2 and d["ai_pcp_v3"]["hazard_sensitivity"]["n"] == 3
    assert d["raw_llm"]["hazard_sensitivity"]["k"] == 2
    assert d["ai_pcp_v3"]["call_911_advised_emergency_care"]["k"] == 1
    assert d["raw_llm"]["call_911_advised_emergency_care"]["k"] == 0
    assert d["llm_standard_guardrails"]["urgent_same_day_or_higher_per_1000"] == 500.0
    assert m["guardrail_input_screen"]["flags"] == 1
    mc = m["mcnemar_on_safety_positives"]["compass_reply_advice_vs"]["raw_llm_reply"]
    assert (mc["compass_only"], mc["comparator_only"]) == (1, 1)
    reg = m["mcnemar_on_safety_positives"]["compass_registered_flag_vs"]["llm_standard_guardrails_input_screen"]
    assert (reg["compass_only"], reg["comparator_only"]) == (1, 0)


def test_cost():
    assert round(cost({"anthropic:claude-opus-5-5": {"input_tokens": 1e6, "output_tokens": 1e6, "calls": 1},
                       "kev-local:kev-4b@v1.0": {"input_tokens": 5, "output_tokens": 0, "calls": 1}}), 2) == 24.0
