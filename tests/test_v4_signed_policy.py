"""COMPASS v4 signed crisis policy: the er_now route and its precedence, per-criterion instruction text, per-criterion
order exceptions honored by the deterministic reply check, and the require_signed loader. Synthetic strings only."""

import importlib.util
from pathlib import Path

import pytest

from compass_pcp.contract import build_contract, deterministic_check
from compass_pcp.crisis_router import parse_form, route_form
from compass_pcp.policy_v4 import CrisisPolicy, load_contract_policy, load_crisis_policy
from tests.conftest import FIXTURES

ROOT = Path(__file__).resolve().parents[1]
CP = load_contract_policy()


def _ins(rid, clause, n, page=True):
    return {"id": f"i_{rid}", "text": f"Route default {clause}.", "text_other": f"Route default other {clause}.",
            "fallback": f"Fallback {clause}.", "fallback_other": f"Fallback other {clause}.", "page_care_team": page,
            "required_clause": clause, "clause_max_sentences": n, "source": "synthetic"}


def _crit(cid, route, instruction=None, exception=None):
    c = {"id": cid, "description": f"Synthetic criterion {cid} for tests.", "route": route, "source": "src_a",
         "source_ids": ["src_a"]}
    if instruction:
        c["instruction"] = {"id": f"instr_{cid}", "source": f"signed:{cid}", **instruction}
    if exception:
        c["order_exception"] = {"source": "synthetic approval", **exception}
    return c


SYNTH = CrisisPolicy.model_validate({
    "version": "t1", "status": "signed", "signed_by": "Test Signer", "signed_on": "2026-01-01",
    "screen": {"timeout_seconds": 5, "source": "synthetic"},
    "route_precedence": ["emergency_911", "crisis_988_page", "er_now", "urgent_review", "poison_control_addition",
                         "none"],
    "additions": ["poison_control_addition"], "additions_suppressed_by": ["emergency_911"],
    "instructions": {"emergency_911": _ins("911", "call 911 now", 1), "crisis_988_page": _ins("988", "text 988 now", 2),
                     "er_now": _ins("er", "nearest emergency room now", 1),
                     "poison_control_addition": _ins("pc", "call poison control", 2, page=False)},
    "uncertainty": {"source": "synthetic"},
    "sources": {"src_a": "Synthetic citation"},
    "families": [{"id": "fam", "name": "Synthetic", "lexicon": [], "criteria": [
        _crit("E1", "emergency_911", {"text": "Call 911 now. Use the blue kit.", "text_other": "Call 911 now for them."}),
        _crit("E2", "emergency_911", {"text": "If you are in danger right now, call 911 if it is safe."},
              {"required_clause": "call 911", "max_sentences": 1}),
        _crit("E3", "emergency_911", {"text": "Go outside into fresh air now. Then call 911 from outside."},
              {"required_clause": "call 911", "max_sentences": 2}),
        _crit("C1", "crisis_988_page"),
        _crit("R1", "er_now", {"text": "Go to the nearest emergency room now. Bring your pill bottles."}),
        _crit("U1", "urgent_review", {"text": "A clinician will look at this soon. If it gets worse, call 911."}),
        _crit("P1", "poison_control_addition"),
    ]}],
})

ALL = [c.id for f in SYNTH.families for c in f.criteria]
TEXT = "synthetic message alpha beta gamma"


def _route(*present, person="patient"):
    rows = [{"criterion_id": cid, "status": "present" if cid in present else "not_mentioned", "current": True,
             "person_at_risk": person, "quote": TEXT if cid in present else ""} for cid in ALL]
    return route_form(parse_form({"items": rows}, SYNTH, [TEXT]), SYNTH)


# ---- (i) er_now and its precedence ---------------------------------------------------------------------------------
def test_er_now_ranks_below_911_and_988_and_above_urgent_review():
    assert _route("R1").route == "er_now"
    assert _route("R1", "U1").route == "er_now"
    assert _route("R1", "C1").route == "crisis_988_page"
    assert _route("R1", "E1").route == "emergency_911"
    r = _route("R1", "P1")
    assert r.route == "er_now" and r.additions == ["poison_control_addition"]


def test_er_now_is_an_emergency_contract_that_pages():
    r = _route("R1")
    assert r.page_care_team and r.required_clause == "nearest emergency room now" and r.clause_max_sentences == 1
    c = build_contract(disposition="ask", route=r, policy=CP, question="Q?")
    assert c.state == "emergency" and c.referral_position == "first" and c.selected_question is None
    assert c.page_requested


# ---- (ii) per-criterion instruction text --------------------------------------------------------------------------
def test_criterion_instruction_overrides_the_route_default():
    r = _route("E1")
    assert r.instruction_text == "Call 911 now. Use the blue kit." and r.fallback_text == r.instruction_text
    assert r.criterion_instruction_id == "instr_E1" and "instr_E1" in r.source_ids and "i_911" in r.source_ids
    other = _route("E1", person="other")
    assert other.addressed_to_other and other.instruction_text == "Call 911 now for them."
    plain = _route("C1")  # no criterion wording: the route default stays
    assert plain.instruction_text == "Route default text 988 now." and plain.criterion_instruction_id is None


def test_first_fired_criterion_wording_wins_within_a_route():
    r = _route("E1", "E2")
    assert r.criterion_instruction_id == "instr_E1" and r.order_exception_for is None


def test_urgent_review_criterion_wording_becomes_a_required_clause():
    r = _route("U1")
    assert r.route == "urgent_review" and not r.page_care_team
    c = build_contract(disposition="complete", route=r, policy=CP)
    assert "A clinician will look at this soon. If it gets worse, call 911." in c.required_clauses()
    assert deterministic_check(f"{CP.referrals['complete'].text} Thanks for writing.", c, CP)  # clause missing
    ok = f"A clinician will look at this soon. If it gets worse, call 911. {CP.referrals['complete'].text}"
    assert deterministic_check(ok, c, CP) == []


def test_no_route_leaves_the_v3_contract_unchanged():
    r = _route()
    assert r.route == "none" and r.instruction_text is None
    assert build_contract(disposition="general_out_of_lane", route=r, policy=CP) is None


# ---- (iii) per-criterion order exceptions -------------------------------------------------------------------------
def _contract(*present):
    return build_contract(disposition="ask", route=_route(*present), policy=CP, question="Q?")


def test_route_clause_must_open_a_911_reply():
    c = _contract("E1")
    assert c.route_clause == "call 911 now" and c.route_clause_max_sentences == 1 and c.order_exception_for is None
    assert deterministic_check("Call 911 now. Use the blue kit. Stay with someone.", c, CP) == []
    late = deterministic_check("I am sorry this is happening. Call 911 now. Use the blue kit.", c, CP)
    assert "the route clause is not within the first 1 sentence(s)" in late


def test_order_exception_replaces_the_route_clause_for_that_criterion():
    c = _contract("E2")
    assert c.order_exception_for == "E2" and c.route_clause == "call 911"
    reply = "If you are in danger right now, call 911 if it is safe. I am here."
    assert deterministic_check(reply, c, CP) == []
    # the same reply under the route's own rule fails: "call 911 now" is absent
    strict = c.model_copy(update={"route_clause": "call 911 now", "route_clause_max_sentences": 1,
                                  "order_exception_for": None})
    assert 'the route clause is missing: "call 911 now"' in deterministic_check(reply, strict, CP)


def test_order_exception_allows_the_clause_in_the_second_sentence():
    c = _contract("E3")
    reply = "Go outside into fresh air now. Then call 911 from outside."
    assert c.route_clause_max_sentences == 2 and deterministic_check(reply, c, CP) == []
    no_exception = c.model_copy(update={"route_clause_max_sentences": 1})
    assert "the route clause is not within the first 1 sentence(s)" in deterministic_check(reply, no_exception, CP)
    asked = "Is anyone with you? Go outside into fresh air now. Then call 911 from outside."
    assert "a question comes before the route clause" in deterministic_check(asked, c, CP)


# ---- loader, signature and the generated file ---------------------------------------------------------------------
def test_require_signed_refuses_the_placeholder():
    placeholder = FIXTURES / "crisis_v4_placeholder.yaml"
    assert load_crisis_policy(placeholder).status == "placeholder_unsigned"
    with pytest.raises(ValueError, match="signed policy is required"):
        load_crisis_policy(placeholder, require_signed=True)


def test_shipped_policy_is_signed_and_fully_sourced():
    p = load_crisis_policy(require_signed=True)
    assert p.status == "signed" and p.signed_by and p.signed_on
    crits = [c for f in p.families for c in f.criteria]
    assert crits and all(c.source in p.sources and c.source_ids and set(c.source_ids) <= set(p.sources) for c in crits)
    assert all(c.instruction is not None for c in crits)
    assert {c.id for c in crits if c.order_exception} == {"EH-01", "EH-03"}
    assert p.rank("er_now") == p.rank("crisis_988_page") + 1 < p.rank("urgent_review")


def test_shipped_policy_matches_a_fresh_conversion_of_the_signed_record():
    spec = importlib.util.spec_from_file_location("convert_signed_criteria", ROOT / "design" / "convert_signed_criteria.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import yaml

    fresh = mod.render(yaml.safe_load(mod.SIGNED.read_text()))
    assert fresh == mod.OUT.read_text()


def test_v4_arm_requires_a_signed_policy_outside_test_mode():
    from compass_pcp.arms.ai_pcp_v4 import AiPcpV4

    placeholder = load_crisis_policy(FIXTURES / "crisis_v4_placeholder.yaml")
    with pytest.raises(ValueError, match="signed policy is required"):
        AiPcpV4(engine=None, screen_llm=None, crisis_policy=placeholder, test_mode=False)  # type: ignore[arg-type]
    assert AiPcpV4(engine=None, screen_llm=None, crisis_policy=placeholder, test_mode=True).crisis_policy is placeholder  # type: ignore[arg-type]
    assert AiPcpV4(engine=None, screen_llm=None, test_mode=False).crisis_policy.status == "signed"  # type: ignore[arg-type,union-attr]
