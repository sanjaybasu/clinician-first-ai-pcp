"""paper/multiturn_ddxplus.py with fake clients: the stop rules (engine 'ask' for ai_pcp_v3, classifier class for the
other arms, 12-turn cap with no disposition), the patient prompt roles, ledger-call reuse across turns, batch staging
(a miss queues the call and the next pass resumes), and the rule-based complaint mapping. Synthetic text only."""

import json
from dataclasses import dataclass, field

import pytest

from compass_pcp.arms.baselines import RawLLM
from compass_pcp.arms.llm import Usage
from paper import batch_stage as bs
from paper import multiturn_ddxplus as mt

EV = {
    "E_53": {"data_type": "B", "question_en": "Do you have pain somewhere?"},
    "E_55": {"data_type": "M", "question_en": "Where is the pain?", "value_meaning": {"V_29": {"en": "lower chest"}, "V_123": {"en": "nowhere"}}},
    "E_91": {"data_type": "B", "question_en": "Do you have a fever?"},
    "E_45": {"data_type": "B", "question_en": "Have you been coughing up blood?"},
    "E_66": {"data_type": "B", "question_en": "Are you short of breath?"},
    "E_59": {"data_type": "C", "question_en": "How fast did the pain appear?"},
}
CASE = {"case_id": "validate:1", "age": 50, "sex": "F", "complaint": "cough_acute", "initial_evidence": "E_201",
        "evidences": ["E_201", "E_91", "E_53", "E_55_@_V_29", "E_59_@_7"], "stratum": "cough_acute/community_acquired_pneumonia"}


@dataclass
class Scripted:
    """A fake client that answers by a function of (system, messages) and counts calls."""

    fn: object
    vendor: str = "anthropic"
    model: str = "fake"
    phi_allowed: bool = False
    thinking_headroom: int = 0
    usage: Usage = field(default_factory=Usage)
    calls: list = field(default_factory=list)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        self.usage.calls += 1
        self.calls.append((system, messages))
        return self.fn(system, messages)


def classifier_fn(system, messages):
    text = messages[-1]["content"]
    cls = "asking_for_more_information" if text.endswith("?") else "emergency_now"
    return json.dumps({"class": cls, "disposition": "urgent_same_day"})


def patient_fn(system, messages):
    return json.dumps({"reply": "yes", "asked_codes": ["E_91", "E_999"]})


def bundle(store, raw_fn, realtime=True, arms=None):
    raw = RawLLM(mt.EpisodeLLM(Scripted(raw_fn), store, lambda s: 3, realtime=realtime))
    pat = mt.EpisodeLLM(Scripted(patient_fn, vendor="openai"), store, lambda s: 2, realtime=realtime)
    cls = mt.EpisodeLLM(Scripted(classifier_fn), store, lambda s: 2, realtime=realtime)
    return mt.Bundle(arms or {}, raw, pat, cls, EV)


def test_opener_and_patient_prompt_hold_the_inventory_answers():
    assert mt.opener_text(CASE) == "I'm a 50-year-old woman and I have a cough."
    lines = mt.fact_lines(CASE, EV)
    assert "E_91 | Do you have a fever? | yes" in lines
    assert "E_45 | Have you been coughing up blood? | no" in lines
    assert "E_55 | Where is the pain? | lower chest" in lines
    assert "E_59 | How fast did the pain appear? | 7 on a scale from 0 to 10" in lines
    assert mt.feature_truth("cough_acute", "chest_pain", CASE) and mt.feature_truth("cough_acute", "fever_history", CASE)
    assert not mt.feature_truth("cough_acute", "coughing_blood", CASE)


def test_raw_arm_stops_at_first_reply_that_is_not_a_question(tmp_path):
    store = bs.Store(tmp_path / "s")
    replies = iter(["Do you have a fever?", "Is there blood?", "Call 911 now."])
    b = bundle(store, lambda s, m: next(replies))
    rec = mt.run_episode(CASE, "raw_llm", b)
    assert rec["n_turns"] == 3 and rec["final_disposition"] == "emergency_now" and not rec["no_disposition"]
    assert [t["asking"] for t in rec["turns"]] == [True, True, False]
    # unknown codes are dropped from the closed list
    assert rec["turns"][0]["patient"]["asked_codes"] == ["E_91"]
    # the patient sees itself as assistant, the doctor as user, and its own opener first
    system, msgs = b.patient.inner.calls[0]
    assert msgs[0] == {"role": "assistant", "content": mt.opener_text(CASE)}
    assert msgs[-1] == {"role": "user", "content": "Do you have a fever?"} and "50-year-old woman" in system


def test_cap_at_twelve_turns_gives_no_disposition(tmp_path):
    store = bs.Store(tmp_path / "s")
    n = iter(range(100))
    b = bundle(store, lambda s, m: f"Question {next(n)}?")
    rec = mt.run_episode(CASE, "raw_llm", b)
    assert rec["n_turns"] == mt.MAX_TURNS and rec["no_disposition"] and rec["final_disposition"] is None
    assert "patient" not in rec["turns"][-1]


@dataclass
class FakeCompass:
    """Engine decision 'ask' twice, then 'complete'; the ledger router call goes through the staged client."""

    llm: object
    script: list = field(default_factory=lambda: ["ask", "ask", "complete"])

    def next_message(self, messages, *, context=None):
        turn = sum(m["role"] == "user" for m in messages)
        self.build_ledger(messages, [])
        d = self.script[turn - 1]
        return f"reply {turn}?", {"decision": d, "questions": ["fever_history"] if d == "ask" else []}

    def build_ledger(self, messages, ctx):
        from compass_pcp.ledger import Ledger

        for i in range(1, sum(m["role"] == "user" for m in messages) + 1):
            prefix = [m for m in messages][: 2 * i - 1]
            self.llm.complete(mt.ROUTER.split("{")[0] + "x", prefix, max_tokens=10)
        return Ledger(), [], []


def test_compass_stops_on_engine_decision_and_ledger_calls_are_reused(tmp_path):
    store = bs.Store(tmp_path / "s")
    router = Scripted(lambda s, m: "{}")
    staged = mt.EpisodeLLM(router, store, mt.tier_of, realtime=True)
    b = bundle(store, lambda s, m: "x", arms={"ai_pcp_v3": FakeCompass(staged)})
    rec = mt.run_episode(CASE, "ai_pcp_v3", b)
    # the reply ends with "?" but the engine said complete, so the episode stops at turn 3
    assert rec["n_turns"] == 3 and rec["final_decision"] == "complete"
    # the classifier read only the final reply; the patient answered the two ask turns
    assert b.classifier.inner.usage.calls == 1 and b.patient.inner.usage.calls == 2
    # the final class is "asking" (reply ends with "?"), so the disposition field is used
    assert rec["final_class"] == "asking_for_more_information" and rec["final_disposition"] == "urgent_same_day"
    # three distinct prefixes -> three router calls, although the ledger was rebuilt at every turn and at the end
    assert router.usage.calls == 3


def test_batch_miss_queues_and_next_pass_resumes(tmp_path):
    store = bs.Store(tmp_path / "s")
    replies = {"n": 0}

    def raw_fn(s, m):
        replies["n"] += 1
        return "Is there blood?" if sum(x["role"] == "user" for x in m) == 1 else "Rest at home."

    b = bundle(store, raw_fn, realtime=False)
    with pytest.raises(bs.Pending):
        mt.run_episode(CASE, "raw_llm", b)
    assert len(store.queue) == 1  # the raw reply; nothing after it is queued
    # answer every queued call as a batch would, then rerun the pass
    for _ in range(10):
        if not store.queue:
            break
        for k, r in list(store.queue.items()):
            body = r["body"]
            msgs = body["messages"]
            if r["vendor"] == "openai":
                text = patient_fn(None, msgs)
            elif body["system"] == mt.CLASSIFY:
                text = classifier_fn(None, msgs)
            else:
                text = raw_fn(None, msgs)
            store.put(k, text, {"input_tokens": 10, "output_tokens": 5}, r["spec"])
        store.queue.clear()
        try:
            rec = mt.run_episode(CASE, "raw_llm", b)
            break
        except bs.Pending:
            continue
    assert rec["n_turns"] == 2 and rec["final_disposition"] == "emergency_now"


def test_parsers_reject_unknown_classes():
    assert mt.parse_class('{"class": "go home", "disposition": "self_care"}') == {
        "class": None, "disposition": "self_care", "parsed": False}
    assert mt.parse_patient("not json", {"E_1"}) == {"reply": "not json", "asked_codes": [], "parsed": False}


def test_complaint_mapping_rules():
    mt.LEG_SIDES_CACHE["sides"] = {"V_119": "R"}
    assert mt.map_complaint("E_53", {"E_55": ["V_29", "V_89"]}) == "chest_pain"  # chest has precedence over head
    assert mt.map_complaint("E_53", {"E_55": ["V_89"]}) == "headache"
    assert mt.map_complaint("E_201", {}) == "cough_acute"
    assert mt.map_complaint("E_66", {}) == "dyspnea"
    assert mt.map_complaint("E_151", {"E_152": ["V_119"]}) == "leg_swelling"
    assert mt.map_complaint("E_151", {"E_152": ["V_89"]}) is None
    assert mt.map_complaint("E_50", {}) is None
