"""COMPASS v4 A1/A2/A3: contract builder per state, deterministic checks, writer with one revision and fallback."""

import json

import pytest

from compass_pcp.arms.writer_v4 import compose_v3_audited, write_under_contract
from compass_pcp.contract import ResponseContract, build_contract, deterministic_check, v3_disposition
from compass_pcp.crisis_router import RouteResult
from compass_pcp.engine.workup import DecisionKind
from compass_pcp.ledger import HazardTier
from compass_pcp.policy_v4 import load_contract_policy, load_crisis_policy
from tests.conftest import FIXTURES
from tests.test_arms import FakeLLM

CP = load_contract_policy()
CRISIS = load_crisis_policy(FIXTURES / "crisis_v4_placeholder.yaml")
NONE = RouteResult(route="none")


def crisis_route(route="crisis_988_page", other=False, additions=()) -> RouteResult:
    ins = CRISIS.instructions[route]
    return RouteResult(route=route, rules_fired=["ssh_current_ideation"], source_ids=[ins.id],
                       instruction_text=ins.text_other if other else ins.text,
                       fallback_text=ins.fallback_other if other else ins.fallback,
                       addition_texts=[CRISIS.instructions[a].text for a in additions], additions=list(additions),
                       page_care_team=ins.page_care_team, addressed_to_other=other)


def test_v3_disposition_reproduces_v3_order():
    E, A, D = DecisionKind.ESCALATE, DecisionKind.ASK, DecisionKind.DEFER
    assert v3_disposition("health_professional", HazardTier.EMERGENCY_NOW, [E], ["q"]) == "clinician_support"
    assert v3_disposition("patient", HazardTier.NONE, [A, E], ["q"]) == "escalate"
    assert v3_disposition("patient", HazardTier.POSSIBLE_EMERGENCY, [A], ["q"]) == "possible_emergency"
    assert v3_disposition("patient", HazardTier.URGENT_SAME_DAY, [], []) == "urgent_same_day"
    assert v3_disposition("patient", HazardTier.ROUTINE_FOLLOWUP, [], []) == "general_out_of_lane"
    assert v3_disposition("patient", HazardTier.NONE, [A, D], ["q"]) == "ask"
    assert v3_disposition("patient", HazardTier.NONE, [D], []) == "defer"
    assert v3_disposition("patient", HazardTier.NONE, [DecisionKind.COMPLETE], []) == "complete"


def test_contract_for_each_state():
    em = build_contract(disposition="ask", route=crisis_route(), policy=CP, question="Q?")
    assert em.state == "emergency" and em.referral_position == "first" and em.selected_question is None
    assert em.page_requested and not em.page_executed and em.disposition == "crisis_988_page"

    esc = build_contract(disposition="escalate", route=NONE, policy=CP)
    assert esc.state == "emergency" and esc.referral_text == CP.referrals["escalate"].text

    ask = build_contract(disposition="ask", route=NONE, policy=CP, question="Do you have a fever?")
    assert ask.state == "ask" and ask.selected_question == "Do you have a fever?" and ask.referral_text is None

    pe = build_contract(disposition="possible_emergency", route=NONE, policy=CP)
    assert pe.state == "answer" and pe.selected_question is None and pe.referral_position == "first"
    pe_q = build_contract(disposition="possible_emergency", route=NONE, policy=CP, question="Q?")
    assert pe_q.state == "ask" and pe_q.selected_question == "Q?"

    urg = build_contract(disposition="urgent_same_day", route=NONE, policy=CP)
    assert urg.state == "answer" and urg.referral_position == "any"

    de = build_contract(disposition="defer", route=NONE, policy=CP)
    assert de.state == "defer" and de.assessment_open

    done = build_contract(disposition="complete", route=NONE, policy=CP)
    assert done.state == "answer" and not done.assessment_open

    assert build_contract(disposition="general_out_of_lane", route=NONE, policy=CP) is None
    assert build_contract(disposition="clinician_support", route=NONE, policy=CP) is None
    poison = RouteResult(route="poison_control_addition", addition_texts=[], source_ids=["instr_poison_v0"])
    poison.addition_texts = [CRISIS.instructions["poison_control_addition"].text]
    add = build_contract(disposition="general_out_of_lane", route=poison, policy=CP)
    assert add.state == "answer" and add.additional_clauses == poison.addition_texts
    other = build_contract(disposition="escalate", route=NONE, policy=CP, reporter_about_other=True)
    assert other.referral_text == CP.referrals["escalate"].text_other


def test_open_cannot_miss_question_carries_the_standing_clause():
    from compass_pcp.policy_v4 import POLICY_DIR
    v6 = load_contract_policy(POLICY_DIR / "contract_v6_dev.yaml")
    standing = v6.referrals["ask_open_cannot_miss"]
    ask = build_contract(disposition="ask", route=NONE, policy=v6, question="Q?", open_cannot_miss=True)
    assert ask.state == "ask" and ask.selected_question == "Q?" and ask.referral_text == standing.text
    assert "a plan of home care alone while the assessment is open" in ask.prohibited
    assert standing.text in ask.fallback_text
    closed = build_contract(disposition="ask", route=NONE, policy=v6, question="Q?")
    assert closed.referral_text is None
    no_entry = build_contract(disposition="ask", route=NONE, policy=CP, question="Q?", open_cannot_miss=True)
    assert no_entry.referral_text is None  # policies without the entry (v4, v5) are unchanged


def _em(**kw) -> ResponseContract:
    return build_contract(disposition="ask", route=crisis_route(**kw), policy=CP)


def test_referral_order_checker():
    c = _em()
    clause = c.referral_text
    assert deterministic_check(f"{clause} I am glad you told me. Stay with someone you trust.", c, CP) == []
    assert deterministic_check(f"I'm sorry. {clause}", c, CP) == []  # second sentence is allowed
    late = deterministic_check(f"I'm sorry. That sounds hard. {clause}", c, CP)
    assert any("first 2 sentences" in p for p in late)
    q_first = deterministic_check(f"Are you safe right now? {clause}", c, CP)
    assert any("question comes before" in p for p in q_first)
    altered = deterministic_check("Please call or text 988 soon to reach the Lifeline.", c, CP)
    assert any("missing or altered" in p for p in altered)
    assert deterministic_check(clause.upper() + " Stay safe.", c, CP) == []  # case-insensitive verbatim


def test_page_claim_is_rejected_unless_the_page_executed():
    c = _em()
    claim = f"{c.referral_text} Your care team has been alerted."
    assert any("paged" in p for p in deterministic_check(claim, c, CP))
    c2 = c.model_copy(update={"page_executed": True})
    assert deterministic_check(claim, c2, CP) == []


def test_selected_question_check():
    c = build_contract(disposition="ask", route=NONE, policy=CP, question="Does the pain spread to your left arm?")
    assert deterministic_check("That can have many causes. Does the pain spread to your left arm?", c, CP) == []
    assert deterministic_check("Many causes. Is the pain spreading to your left arm at all?", c, CP) == []
    assert deterministic_check("Many causes. How long has it lasted?", c, CP) != []
    no_q = build_contract(disposition="possible_emergency", route=NONE, policy=CP)  # no engine question: answers
    assert not any("asks no question" in p for p in deterministic_check(no_q.referral_text + " Rest.", no_q, CP))


def test_additional_clause_required():
    c = build_contract(disposition="ask", route=crisis_route(additions=("poison_control_addition",)), policy=CP)
    assert any("Poison Control" in p for p in deterministic_check(c.referral_text + " Stay with someone.", c, CP))


@pytest.mark.parametrize("route,other", [("emergency_911", False), ("emergency_911", True), ("crisis_988_page", False),
                                         ("crisis_988_page", True)])
def test_signed_fallbacks_satisfy_the_deterministic_contract(route, other):
    c = build_contract(disposition="general_out_of_lane", route=crisis_route(route, other), policy=CP)
    assert deterministic_check(c.fallback_text, c, CP) == []


def test_engine_fallbacks_carry_their_referral():
    for d in ("escalate", "defer", "complete", "urgent_same_day", "possible_emergency"):
        c = build_contract(disposition=d, route=NONE, policy=CP)
        assert deterministic_check(c.fallback_text, c, CP) == [] or c.state == "ask"


SEM_OK = json.dumps({"ok": True, "checks": {"within_contract": True, "useful_content": True,
                                            "no_contradictory_reassurance": True, "no_diagnosis": True,
                                            "no_prescription": True}, "problems": []})
SEM_BAD = json.dumps({"ok": False, "checks": {"no_diagnosis": False}, "problems": ["says you have angina"]})
MSG = [{"role": "user", "content": "i wish i could just disappear"}]


def test_writer_passes_first_time():
    c = _em()
    llm = FakeLLM({"against its response contract": SEM_OK, "under a response contract": f"{c.referral_text} Breathe."})
    text, log = write_under_contract(llm, MSG, c, CP)
    assert text.startswith(c.referral_text) and log["first_pass"]["ok"] and not log["revised"] and not log["fallback"]


def test_writer_revises_once_then_passes():
    c = _em()
    drafts = iter(["Are you safe? Call someone.", f"{c.referral_text} Stay with someone."])
    llm = FakeLLM({"against its response contract": SEM_OK, "under a response contract": lambda m: next(drafts)})
    text, log = write_under_contract(llm, MSG, c, CP)
    assert not log["first_pass"]["deterministic_ok"] and log["revised"] and log["post_revision"]["ok"]
    assert text.startswith(c.referral_text) and not log["fallback"]


def test_writer_falls_back_after_one_failed_revision():
    c = _em()
    llm = FakeLLM({"against its response contract": SEM_BAD, "under a response contract": f"{c.referral_text} Ok."})
    text, log = write_under_contract(llm, MSG, c, CP)
    assert log["first_pass"]["deterministic_ok"] and not log["first_pass"]["semantic_ok"]
    assert log["fallback"] and text == c.fallback_text
    unreadable = FakeLLM({"against its response contract": "not json", "under a response contract": c.referral_text})
    assert write_under_contract(unreadable, MSG, c, CP)[1]["fallback"]


def test_edit_variant_edits_the_no_engine_draft():
    c = _em()
    seen = {}

    def edit(m):
        seen["draft"] = m[-1]["content"]
        return f"{c.referral_text} Edited."

    llm = FakeLLM({"against its response contract": SEM_OK, "Edit it as little as possible": edit,
                   "REQUIREMENTS FROM THE CLINICAL ENGINE": "NO ENGINE DRAFT"})
    text, log = write_under_contract(llm, MSG, c, CP, variant="edit")
    assert seen["draft"] == "DRAFT TO EDIT:\nNO ENGINE DRAFT" and text.endswith("Edited.") and log["variant"] == "edit"
    with pytest.raises(ValueError):
        write_under_contract(llm, MSG, c, CP, variant="nope")


def test_v3_composer_audit_and_compact_policy():
    seen = []

    def draft(m):
        return "DRAFT"

    class Spy(FakeLLM):
        def complete(self, system, messages, **kw):
            seen.append(system)
            return super().complete(system, messages, **kw)

    llm = Spy({"before it is sent": '{"ok": true}', "REQUIREMENTS FROM THE CLINICAL ENGINE": draft})
    text, audit = compose_v3_audited(llm, "general", MSG)
    assert text == "DRAFT" and audit["verify_ok"] and not audit["revised"]
    assert "Response policy" not in seen[0]
    compose_v3_audited(llm, "general", MSG, compact=True)
    assert "Response policy" in seen[2]
    bad = FakeLLM({"before it is sent": '{"ok": false, "problems": ["x"]}', "Revise this draft": "REV",
                   "REQUIREMENTS FROM THE CLINICAL ENGINE": "DRAFT"})
    text, audit = compose_v3_audited(bad, "general", MSG)
    assert audit["revised"]


def test_full_writer_appends_a_dropped_question_instead_of_falling_back():
    q = "Do you have a fever?"
    c = build_contract(disposition="ask", route=NONE, policy=CP, question=q)
    llm = FakeLLM({"against its response contract": SEM_OK, "under a response contract": "Rest and drink fluids."})
    text, log = write_under_contract(llm, MSG, c, CP, variant="contract_full")
    assert text.endswith(q) and log["question_appended"] and log["post_repair"]["ok"] and not log["fallback"]
    text, log = write_under_contract(llm, MSG, c, CP)  # the registered variant still falls back
    assert log["fallback"] and text == c.fallback_text


def test_v9_possible_emergency_names_the_emergency_room_only():
    from compass_pcp.policy_v4 import POLICY_DIR

    cp9 = load_contract_policy(POLICY_DIR / "contract_v9_dev.yaml")
    c = build_contract(disposition="possible_emergency", route=NONE, policy=cp9)
    for t in (c.referral_text, c.fallback_text):
        assert "emergency" in t and "urgent care" not in t
    llm = FakeLLM({"against its response contract": SEM_OK, "under a response contract": c.referral_text})
    _, log = write_under_contract(llm, MSG, c, cp9, variant="contract_v9")
    assert not log["fallback"]


def test_v10_completed_pathway_gives_its_disposition_instead_of_physician_review():
    from compass_pcp.policy_v4 import POLICY_DIR

    cp10 = load_contract_policy(POLICY_DIR / "contract_v10_dev.yaml")
    home = build_contract(disposition="complete", route=NONE, policy=cp10, pathway_level="self_care")
    visit = build_contract(disposition="complete", route=NONE, policy=cp10, pathway_level="routine_visit")
    plain = build_contract(disposition="complete", route=NONE, policy=cp10)
    assert "cared for at home" in home.referral_text and "appointment" in visit.referral_text
    assert plain.referral_text == cp10.referrals["complete"].text
    assert not home.assessment_open and home.state == "answer"
    v9 = load_contract_policy(POLICY_DIR / "contract_v9_dev.yaml")  # no complete_<level> entry: physician review
    assert build_contract(disposition="complete", route=NONE, policy=v9, pathway_level="self_care").referral_text == v9.referrals["complete"].text


def test_v10_crisis_policy_excludes_sore_throat_pain_from_ca02():
    from compass_pcp.policy_v4 import POLICY_DIR

    ca02 = [c for f in load_crisis_policy(POLICY_DIR / "crisis_v10_dev.yaml").families for c in f.criteria if c.id == "CA-02"]
    assert len(ca02) == 1 and "sore throat" in ca02[0].description
    v4 = next(c for f in load_crisis_policy().families for c in f.criteria if c.id == "CA-02")
    assert ca02[0].description.replace(" Throat pain in a person with a sore throat, or pain on swallowing, is not neck "
                                       "pain for this item: count it as absent.", "") == v4.description


def test_first_aid_passages_are_required_clauses_on_crisis_contracts():
    from compass_pcp.policy_v4 import POLICY_DIR, load_first_aid

    fa = load_first_aid(POLICY_DIR / "first_aid_v1_dev.yaml")
    assert set(fa) >= {"CA-01", "NE-01", "BR-02", "OD-01"} and all(fa.values())
    route = crisis_route(route="emergency_911")
    heart = fa["CA-01"][0][0]
    c = build_contract(disposition="escalate", route=route, policy=CP, first_aid=(heart,))
    assert c.state == "emergency" and heart in c.required_clauses() and heart in c.fallback_text
    assert all(adult for t, adult in fa["BR-01"])  # CPR is adult-only
    assert c.required_clauses()[0] == route.instruction_text  # the instruction stays first


def test_first_aid_only_for_criteria_present_not_uncertain():
    from compass_pcp.arms.ai_pcp_v4 import first_aid_for

    table = {"CA-01": [("heart", False)], "OD-01": [("naloxone", False), ("cpr", True)], "BR-01": [("cpr", True)]}
    assert first_aid_for(["CA-01:uncertain"], table) == ()
    assert first_aid_for(["OD-01", "BR-01", "CA-01:uncertain"], table) == ("naloxone", "cpr")
    assert first_aid_for(["OD-01", "BR-01"], table, about_other=True) == ("naloxone",)  # adult CPR left out


def test_v17_infant_choking_gets_the_infant_passage_even_from_a_caregiver():
    from compass_pcp.arms.ai_pcp_v4 import first_aid_for
    from compass_pcp.policy_v4 import POLICY_DIR, load_first_aid

    fa = load_first_aid(POLICY_DIR / "first_aid_v2_dev.yaml")
    infant = first_aid_for(["BR-03"], fa, about_other=True)
    assert len(infant) == 1 and "chest pushes" in infant[0] and "belly" in infant[0]  # "not the belly"
    assert first_aid_for(["BR-01"], fa, about_other=True) == ()  # adult CPR never for someone else


def _v18_form(policy, marks: dict[str, str]):
    from compass_pcp.crisis_router import ScreenForm, ScreenItem

    items = {}
    for f in policy.families:
        for c in f.criteria:
            st = marks.get(c.id, "not_mentioned")
            items[c.id] = ScreenItem(criterion_id=c.id, status=st, current=True if st != "not_mentioned" else None,
                                     quote="q" if st != "not_mentioned" else "", quote_valid=st != "not_mentioned",
                                     completed=True)
    return ScreenForm(items=items, completed=True)


def test_v18_uncertain_items_route_by_family():
    from compass_pcp.crisis_router import route_form
    from compass_pcp.policy_v4 import POLICY_DIR

    v18 = load_crisis_policy(POLICY_DIR / "crisis_v18_dev.yaml")
    v17 = load_crisis_policy(POLICY_DIR / "crisis_v17_dev.yaml")
    assert route_form(_v18_form(v18, {"SH-05": "uncertain"}), v18).route == "crisis_988_page"
    assert route_form(_v18_form(v18, {"OD-01": "uncertain"}), v18).route == "er_now"
    assert route_form(_v18_form(v18, {"SU-01": "uncertain"}), v18).route == "er_now"
    assert route_form(_v18_form(v18, {"CA-01": "uncertain"}), v18).route == "urgent_review"  # other families unchanged
    assert route_form(_v18_form(v17, {"SH-05": "uncertain"}), v17).route == "urgent_review"  # v17 unchanged
    assert route_form(_v18_form(v18, {"SH-01": "present"}), v18).route == "emergency_911"  # present items unchanged
    strip = lambda p: [(c.id, c.route, c.description) for f in p.families for c in f.criteria]
    assert strip(v18) == strip(v17)  # criteria themselves are unchanged
