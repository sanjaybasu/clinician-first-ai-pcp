"""COMPASS v4 arm wiring with fake LLMs: crisis routes before routing, contract replies, v3 paths, engine_off,
paging, timeouts, and that concurrent conversations on one arm do not share state. Synthetic text only."""

import json
import re
import sys
import threading
import time
from pathlib import Path

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from compass_pcp.arms.ai_pcp_v4 import AiPcpV4
from compass_pcp.policy_v4 import load_crisis_policy
from tests.conftest import FIXTURES
from tests.test_arms import FakeLLM

POLICY = load_crisis_policy(FIXTURES / "crisis_v4_placeholder.yaml")
SEM_OK = json.dumps({"ok": True, "checks": {}, "problems": []})


class Dialogue(FakeLLM):
    """Writes replies that follow whatever contract it is given; records every system prompt."""

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        self.usage.calls += 1
        if "against its response contract" in system:
            return SEM_OK
        if "under a response contract" in system:
            req = re.search(r'REQUIRED CLAUSE \(as the first sentence[^)]*\): "([^"]+)"', system)
            q = re.search(r'SELECTED QUESTION: "([^"]+)"', system)
            return " ".join(x for x in [req.group(1) if req else "", "Here is some general help.",
                                        q.group(1) if q else ""] if x)
        if "The person writing is a health professional" in system:
            return "CLINICIAN REPLY"
        if "before it is sent" in system:
            return '{"ok": true}'
        if "REQUIREMENTS FROM THE CLINICAL ENGINE" in system:
            return "GENERAL REPLY"
        return "ok"


def router_reply(messages):
    last = messages[-1]["content"].lower()
    pro = "my patient" in last or "pt " in last
    cats = ["back"] if "back" in last else []
    return json.dumps({"user_type": "health_professional" if pro else "patient", "categories": cats, "suspected": []})


def screen_reply(messages):
    text = messages[-1]["content"]
    items = []
    m = re.search(r"i plan to take them tonight", text)
    if m:
        items.append({"criterion_id": "ssh_immediate_danger", "status": "present", "current": True,
                      "person_at_risk": "patient", "quote": m.group(0)})
    m = re.search(r"wishes he were dead", text)
    if m:
        items.append({"criterion_id": "ssh_current_ideation", "status": "present", "current": True,
                      "person_at_risk": "other", "quote": m.group(0)})
    ids = {i["criterion_id"] for i in items}
    items += [{"criterion_id": c.id, "status": "not_mentioned", "current": None, "person_at_risk": "unknown", "quote": ""}
              for f in POLICY.families for c in f.criteria if c.id not in ids]
    return json.dumps({"items": items})


def make_arm(script, *, router=None, screen=None, extract="{}", **kw) -> AiPcpV4:
    engine = AiPcpV2(
        scripts={"back": script},
        router=router or FakeLLM({"First decide who is writing": router_reply}),
        hazard_llm=FakeLLM({"You screen patient messages": json.dumps({"tier": "none", "category": "none", "quote": ""})}),
        extractor_a=FakeLLM({"You code the patient": extract}, vendor="a"),
        extractor_b=FakeLLM({"You code the patient": extract}, vendor="b"),
        dialogue=Dialogue({}),
        compose_version=3,
    )
    kw.setdefault("crisis_policy", POLICY)
    kw.setdefault("test_mode", True)
    return AiPcpV4(engine=engine, screen_llm=screen or FakeLLM({"You fill a safety form": screen_reply}), **kw)


def msg(t):
    return [{"role": "user", "content": t}]


def test_crisis_route_comes_first_with_instruction_and_no_page_claim(script):
    arm = make_arm(script)
    text, info = arm.next_message(msg("my back hurts and i saved my pills, i plan to take them tonight"))
    clause = POLICY.instructions["emergency_911"].text
    assert text.startswith(clause)
    assert info["path"]["route"] == "emergency_911" and info["path"]["contract_state"] == "emergency"
    assert info["path"]["rules_fired"] == ["ssh_immediate_danger"]
    assert info["page"] == {"requested": True, "executed": False, "status": "no_pager_configured"}
    assert info["review"]["flag"] and "route:emergency_911" in info["review"]["floors"]
    assert info["compliance"]["first_pass"]["ok"] and not info["fallback"]
    assert info["policy_status"]["crisis"] == "placeholder_unsigned"


def test_pager_result_is_reported():
    class Pager:
        def __init__(self, ok):
            self.ok = ok

        def page(self, *, route, rules, messages):
            if self.ok is None:
                raise ConnectionError("down")
            return self.ok

    from compass_pcp.engine.scripts import load_script
    from tests.conftest import FIXTURES

    script = load_script(FIXTURES / "synthetic_script.yaml")
    m = msg("i plan to take them tonight")
    assert make_arm(script, pager=Pager(True)).next_message(m)[1]["page"]["status"] == "executed"
    assert make_arm(script, pager=Pager(False)).next_message(m)[1]["page"]["status"] == "failed"
    assert make_arm(script, pager=Pager(None)).next_message(m)[1]["page"]["status"] == "failed: ConnectionError"


def test_professional_message_is_screened_and_gets_patient_directed_action(script):
    arm = make_arm(script)
    text, info = arm.next_message(msg("My patient told me today he wishes he were dead. How should I document it?"))
    assert info["path"]["user_type"] == "health_professional"
    assert info["path"]["route"] == "crisis_988_page" and info["route"]["addressed_to_other"]
    assert text.startswith(POLICY.instructions["crisis_988_page"].text_other)


def test_professional_message_without_crisis_keeps_the_v3_clinician_path(script):
    text, info = make_arm(script).next_message(msg("my patient needs a prior auth letter for a back brace"))
    assert text == "CLINICIAN REPLY" and info["path"]["reply_path"] == "v3_clinician" and info["contract"] is None


def test_out_of_lane_message_keeps_the_v3_general_path(script):
    text, info = make_arm(script).next_message(msg("how do i get a ride to my appointment"))
    assert text == "GENERAL REPLY" and info["path"]["decision"] == "general_out_of_lane"
    assert info["path"]["route"] == "none" and not info["review"]["flag"]
    assert info["compose_audit"]["kind"] == "general" and not info["compose_audit"]["compact_policy"]
    _, info2 = make_arm(script, compact_policy=True).next_message(msg("how do i get a ride to my appointment"))
    assert info2["path"]["reply_path"] == "a3_compact"


def test_engine_ask_becomes_an_ask_contract_with_the_selected_question(script):
    text, info = make_arm(script).next_message(msg("my back hurts"))
    c = info["contract"]
    assert info["path"]["decision"] == "ask" and c["state"] == "ask" and c["selected_question"]
    assert text.endswith(c["selected_question"])
    assert any(s.startswith("script:synthetic_back_pain@") for s in c["source_ids"])


def test_engine_off_skips_the_engine_and_its_extractors(script):
    arm = make_arm(script, engine_off=True)
    text, info = arm.next_message(msg("my back hurts"))
    assert info["path"]["decision"] == "general_out_of_lane" and text == "GENERAL REPLY"
    assert arm.engine.extractor_a.usage.calls == 0 and arm.engine.extractor_b.usage.calls == 0


def test_screen_timeout_sets_the_review_floor_and_still_replies(script):
    class Slow(FakeLLM):
        def complete(self, system, messages, **kw):
            time.sleep(0.5)
            return "{}"

    arm = make_arm(script, screen=Slow({}))
    arm.crisis_policy = arm.crisis_policy.model_copy(update={"screen": arm.crisis_policy.screen.model_copy(
        update={"timeout_seconds": 0.05})})
    text, info = arm.next_message(msg("how do i get a ride"))
    assert text == "GENERAL REPLY" and info["screen"]["error"] == "timeout"
    assert info["path"]["route"] == "urgent_review" and info["review"]["flag"]


def test_kev_probability_feeds_the_review_score(script):
    class Kev:
        def choice(self, state, instructions, options):
            return {"none": 0.1, "routine_followup": 0.1, "urgent_same_day": 0.5, "possible_emergency": 0.2,
                    "emergency_now": 0.1}

    arm = make_arm(script)
    arm.engine.hazard_judge, arm.engine.hazard_both = Kev(), True
    _, info = arm.next_message(msg("how do i get a ride"))
    assert abs(info["review"]["features"]["kev_urgent"] - 0.8) < 1e-9


def test_kev_review_only_caps_kev_at_clinician_review(script):
    class Kev:
        def choice(self, state, instructions, options):
            return {"none": 0.1, "routine_followup": 0.1, "urgent_same_day": 0.5, "possible_emergency": 0.2,
                    "emergency_now": 0.1}

    tiers = {}
    for cap in (False, True):
        arm = make_arm(script)
        arm.engine.hazard_judge, arm.engine.hazard_both = Kev(), True
        arm.kev_review_only = cap
        _, info = arm.next_message(msg("how do i get a ride"))
        tiers[cap] = info["path"]["care_team_flag_tier"]
    assert tiers[False] != "ROUTINE_FOLLOWUP" and tiers[True] == "ROUTINE_FOLLOWUP"

def test_concurrent_conversations_do_not_share_state(script):
    """One arm, two threads, interleaved inside the router: a professional and a patient conversation."""
    barrier = threading.Barrier(2, timeout=5)

    def router(messages):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return router_reply(messages)

    arm = make_arm(script, router=FakeLLM({"First decide who is writing": router}))
    out = {}

    def run(key, text):
        out[key] = arm.next_message(msg(text))

    threads = [threading.Thread(target=run, args=("pro", "my patient needs a letter")),
               threading.Thread(target=run, args=("pt", "how do i get a ride"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out["pro"][1]["path"]["user_type"] == "health_professional" and out["pro"][0] == "CLINICIAN REPLY"
    assert out["pt"][1]["path"]["user_type"] == "patient" and out["pt"][0] == "GENERAL REPLY"
    assert not hasattr(arm.engine, "user_type")  # the v3 arm was never asked to store per-call state


def test_v4_code_hash_is_separate_from_v3():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_healthbench as rh

    from compass_pcp.harness.configurations import DESCRIPTIONS, Configuration

    assert not set(rh.V4_CODE) & set(rh.AI_PCP_CODE)
    assert rh.code_sha(Configuration.AI_PCP_V4) != rh.code_sha(Configuration.AI_PCP_V3)
    assert Configuration.AI_PCP_V4 in DESCRIPTIONS


def test_review_followup_sentence_only_on_flagged_general_replies(script, monkeypatch):
    import compass_pcp.arms.ai_pcp_v4 as v4
    from compass_pcp.policy_v4 import POLICY_DIR, load_contract_policy
    from compass_pcp.review_score import ReviewDecision

    cp = load_contract_policy(POLICY_DIR / "contract_v18_dev.yaml")
    sentence = cp.referrals["review_followup"].text
    out = {}
    for flag in (False, True):
        monkeypatch.setattr(v4, "review_decision", lambda floors, features, params, f=flag: ReviewDecision(
            flag=f, floors=[], score=0.0, threshold=0.5, features={}))
        for on in (False, True):
            arm = make_arm(script, contract_policy=cp, review_followup=on)
            text, info = arm.next_message(msg("how do i get a ride"))
            out[(flag, on)] = (text, info["review_followup_added"])
    assert out[(True, True)] == ("GENERAL REPLY " + sentence, True)
    assert all(t == "GENERAL REPLY" and not added for k, (t, added) in out.items() if k != (True, True))


def test_screen_check_asks_then_follows_the_latest_screen():
    from compass_pcp.arms.ai_pcp_v4 import screen_check
    from compass_pcp.ledger import HazardEvent, HazardTier

    q = "Do you have any chest pressure or tightness, shortness of breath, or sweating, or does the pain come on when you walk or climb stairs?"
    ev = lambda turn, tier, cat: HazardEvent(turn=turn, tier=tier, category=cat, quote="", quote_verified=False)
    pe, rt = HazardTier.POSSIBLE_EMERGENCY, HazardTier.ROUTINE_FOLLOWUP
    first = [{"role": "user", "content": "x"}]
    assert screen_check([ev(1, pe, "throat_pain_possible_cardiac")], first, q, 0.6) == ("ask", None)
    assert screen_check([ev(1, pe, "new_shortness_of_breath")], first, q, 0.6) == (None, None)  # not heart-related
    assert screen_check([ev(1, pe, "cardiac"), ev(1, HazardTier.EMERGENCY_NOW, "cardiac")], first, q, 0.6) == (None, None)
    asked = first + [{"role": "assistant", "content": "If you have chest pressure, call 911 now. " + q},
                     {"role": "user", "content": "no to all"}]
    assert screen_check([ev(1, pe, "possible_cardiac"), ev(2, rt, "sore_throat")], asked, q, 0.6) == ("resolved", rt)
    assert screen_check([ev(1, pe, "possible_cardiac"), ev(2, pe, "exertional_angina")], asked, q, 0.6) == ("resolved", pe)


def test_screen_check_contract_asks_with_the_911_line():
    from compass_pcp.contract import build_contract
    from compass_pcp.policy_v4 import POLICY_DIR, load_contract_policy

    cp = load_contract_policy(POLICY_DIR / "contract_v19_dev.yaml")
    q = cp.referrals["screen_check_question"].text
    c = build_contract(disposition="screen_check", route=None, policy=cp, question=q)
    assert c.state == "ask" and c.selected_question == q and c.referral_text.endswith("call 911 now.")


def test_resolved_tier_drops_only_earlier_heart_ratings():
    from compass_pcp.arms.ai_pcp_v4 import resolved_tier
    from compass_pcp.ledger import HazardEvent, HazardTier

    ev = lambda turn, tier, cat: HazardEvent(turn=turn, tier=tier, category=cat, quote="", quote_verified=False)
    pe, ug, rt = HazardTier.POSSIBLE_EMERGENCY, HazardTier.URGENT_SAME_DAY, HazardTier.ROUTINE_FOLLOWUP
    assert resolved_tier([ev(1, pe, "possible_cardiac"), ev(2, rt, "sore_throat")]) == rt
    assert resolved_tier([ev(1, pe, "possible_cardiac"), ev(2, ug, "immunosuppressed_fever"), ev(3, rt, "x")]) == ug
    assert resolved_tier([ev(1, pe, "possible_cardiac"), ev(2, pe, "exertional_angina")]) == pe  # latest kept


def test_screen_check_never_replaces_emergency_reply_for_chest_pain(script, monkeypatch):
    import compass_pcp.arms.ai_pcp_v4 as v4
    from compass_pcp.ledger import HazardTier
    from compass_pcp.policy_v4 import POLICY_DIR, load_contract_policy

    cp = load_contract_policy(POLICY_DIR / "contract_v19_dev.yaml")
    monkeypatch.setattr(v4, "screen_check", lambda *a, **k: ("ask", None))
    monkeypatch.setattr(v4.AiPcpV4, "_tiers", lambda self, st: (HazardTier.POSSIBLE_EMERGENCY,) * 2 + (0.0, "cardiac"))
    out = {}
    for excl in ((), ("back",)):  # the fixture script's complaint is "back"
        arm = make_arm(script, contract_policy=cp, resolve_screen_pe=True, screen_check_exclude=excl)
        _, info = arm.next_message(msg("my back hurts"))
        out[excl] = info["path"]["decision"]
    assert out[()] == "screen_check" and out[("back",)] == "possible_emergency"
