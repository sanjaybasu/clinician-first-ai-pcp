import json

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from compass_pcp.findings import Polarity, Provenance
from compass_pcp.ledger import FactKind, HazardEvent, HazardTier, Ledger
from tests.conftest import make_finding
from tests.test_arms import FakeLLM


def test_conflict_needs_confirmation_and_correction_supersedes():
    led = Ledger()
    led.add_finding(1, make_finding("fever"))
    led.add_finding(2, make_finding("fever", Polarity.ABSENT))
    assert led.state()[1] == ["fever"]
    led.add_finding(3, make_finding("fever", Polarity.ABSENT), correction=True)
    cur, conf = led.state()
    assert conf == [] and cur[0].polarity == Polarity.ABSENT


def test_fact_specific_reconciliation():
    led = Ledger(kinds={"on_metformin": FactKind.OBJECTIVE})
    led.add_finding(0, make_finding("on_metformin", Polarity.PRESENT, Provenance.STRUCTURED_RECORD))
    led.add_finding(1, make_finding("on_metformin", Polarity.ABSENT))
    assert led.state()[0][0].provenance == Provenance.STRUCTURED_RECORD
    led2 = Ledger()  # subjective default: unconfirmed patient denial vs record is a conflict
    led2.add_finding(0, make_finding("cough", Polarity.PRESENT, Provenance.STRUCTURED_RECORD))
    led2.add_finding(1, make_finding("cough", Polarity.ABSENT))
    assert led2.state()[1] == ["cough"]


def test_hazard_is_raise_only():
    led = Ledger()
    led.add_hazard(HazardEvent(turn=1, tier=HazardTier.URGENT_SAME_DAY, category="x", quote="q", quote_verified=True))
    led.add_hazard(HazardEvent(turn=2, tier=HazardTier.NONE, category="y", quote="", quote_verified=False))
    assert led.hazard_tier == HazardTier.URGENT_SAME_DAY


def _v2(script, hazard="none", cats=("back",), extract="{}"):
    common = {"First decide who is writing": json.dumps({"categories": list(cats), "suspected": []}),
              "Does this draft": '{"leak": false}'}
    return AiPcpV2(
        scripts={"back": script},
        router=FakeLLM(common),
        hazard_llm=FakeLLM({"You screen patient messages": json.dumps({"tier": hazard, "category": "c", "quote": "back hurts"})}),
        extractor_a=FakeLLM({"You code the patient": extract}, vendor="a"),
        extractor_b=FakeLLM({"You code the patient": extract}, vendor="b"),
        dialogue=FakeLLM({**common, "What you may do": lambda m: "MSG"}),
    )


MSG = [{"role": "user", "content": "my back hurts"}]


def test_v2_emergency_from_semantic_screen(script):
    text, meta = _v2(script, hazard="emergency_now").next_message(MSG)
    assert meta["decision"] == "escalate" and text == "MSG"


def test_v2_out_of_lane_gets_general_help(script):
    _, meta = _v2(script, cats=()).next_message(MSG)
    assert meta["decision"] == "general_out_of_lane"


def test_v2_asks_red_flags_in_lane(script):
    _, meta = _v2(script).next_message(MSG)
    assert meta["decision"] == "ask" and set(meta["questions"]) <= {"saddle_anesthesia", "urinary_retention"}


def test_v2_scope_leak_triggers_revision(script):
    arm = _v2(script, cats=())
    arm.dialogue = FakeLLM({"Does this draft": '{"leak": true, "span": "take 500 mg"}',
                            "Rewrite this draft": "REVISED", "What you may do": "DRAFT take 500 mg"})
    text, _ = arm.next_message(MSG)
    assert text == "REVISED"


def test_v2_clinician_mode_and_age_eligibility(script):
    arm = _v2(script)
    arm.router = FakeLLM({"First decide who is writing": json.dumps(
        {"user_type": "health_professional", "categories": ["back"], "suspected": []})})
    _, meta = arm.next_message(MSG)
    assert meta["decision"] == "clinician_support"
    arm.router = FakeLLM({"First decide who is writing": json.dumps(
        {"user_type": "caregiver", "patient_age_years": 4, "categories": ["back"], "suspected": []})})
    _, meta = arm.next_message(MSG)
    assert meta["decision"] == "general_out_of_lane"


def test_v2_possible_emergency_tier(script):
    _, meta = _v2(script, hazard="possible_emergency").next_message(MSG)
    assert meta["decision"] == "possible_emergency"


def test_dual_process_escalates_to_deliberate_only_on_trigger(script):
    from compass_pcp.arms.dual_process import DualProcess

    calm = DualProcess(fast=_v2(script, hazard="none", cats=()), deliberate=_v2(script, hazard="none", cats=()))
    _, meta = calm.next_message(MSG)
    assert meta["tier"] == "fast" and meta["triggers"] == []
    hot = DualProcess(fast=_v2(script, hazard="urgent_same_day"), deliberate=_v2(script, hazard="urgent_same_day"))
    _, meta = hot.next_message(MSG)
    assert meta["tier"] == "deliberate" and "hazard_urgent_same_day" in meta["triggers"]


class _Judge:
    def __init__(self, probs):
        self.probs = probs

    def choice(self, state, instructions, options):
        return self.probs


def test_v2_two_hazard_screens_keep_the_higher_tier(script):
    quiet = {"none": 1.0}
    arm = _v2(script, hazard="emergency_now")
    arm.hazard_judge = _Judge(quiet)  # judge alone replaces the LLM screen
    led, _, _ = arm.build_ledger(MSG, [])
    assert led.hazard_tier == HazardTier.NONE
    arm.hazard_both = True  # both run; the LLM screen's emergency survives the judge's none
    led, _, _ = arm.build_ledger(MSG, [])
    assert led.hazard_tier == HazardTier.EMERGENCY_NOW
    arm = _v2(script, hazard="none")
    arm.hazard_judge, arm.hazard_both = _Judge({"none": 0.2, "urgent_same_day": 0.8}), True
    led, _, _ = arm.build_ledger(MSG, [])
    assert led.hazard_tier == HazardTier.URGENT_SAME_DAY


def test_v3_patient_urgency_ignores_the_care_team_judge_and_verifies(script):
    arm = _v2(script, hazard="none", cats=())
    arm.hazard_judge, arm.hazard_both, arm.compose_version = _Judge({"none": 0.2, "urgent_same_day": 0.8}), True, 3
    arm.dialogue = FakeLLM({"You check a draft reply": '{"ok": true, "problems": []}', "You may: answer": lambda m: "ANSWER",
                            "First decide who is writing": json.dumps({"categories": [], "suspected": []})})
    text, meta = arm.next_message(MSG)
    assert meta["care_team_flag_tier"] == "URGENT_SAME_DAY" and meta["hazard_tier"] == "NONE"
    assert meta["decision"] == "general_out_of_lane" and text == "ANSWER"


def test_v3_never_sends_an_empty_reply(script):
    arm = _v2(script, hazard="none", cats=())
    arm.compose_version = 3
    arm.dialogue = FakeLLM({"You check a draft reply": '{"ok": false, "problems": ["x"]}', "You may: answer": lambda m: "",
                            "First decide who is writing": json.dumps({"categories": [], "suspected": []})})
    text, _ = arm.next_message(MSG)
    assert text.strip()
