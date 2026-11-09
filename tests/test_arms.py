import json
from dataclasses import dataclass, field

from compass_pcp.arms.ai_pcp import AiPcpArm
from compass_pcp.arms.baselines import AmieStyle, MaiDxoStyle, RawLLM, StandardGuardrails
from compass_pcp.arms.llm import Usage


@dataclass
class FakeLLM:
    """Answers by matching a keyword in the system prompt."""

    replies: dict
    vendor: str = "fake"
    model: str = "fake-1"
    phi_allowed: bool = False
    usage: Usage = field(default_factory=Usage)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        self.usage.calls += 1
        for key, reply in self.replies.items():
            if key in system:
                return reply(messages) if callable(reply) else reply
        return "ok"


ROUTE_BACK = json.dumps({"category": "back", "suspected": ["cauda_equina"]})
ROUTE_NONE = json.dumps({"category": "none", "suspected": []})


def arm(script, extract_a, extract_b, route=ROUTE_BACK):
    common = {"Pick the one complaint": route, "Write the next text": "ASK", "physician on the care team": "HANDOFF",
              "need to be checked": "ESCALATE"}
    return AiPcpArm(
        scripts={"back": script},
        router=FakeLLM(common),
        extractor_a=FakeLLM({"Code what the PATIENT": extract_a}, vendor="a"),
        extractor_b=FakeLLM({"Code what the PATIENT": extract_b}, vendor="b"),
        dialogue=FakeLLM(common),
    )


MSG = [{"role": "user", "content": "my back hurts"}]


def test_ai_pcp_asks_cannot_miss_first(script):
    text, meta = arm(script, "{}", "{}").next_message(MSG)
    assert text == "ASK" and meta["decision"] == "ask" and meta["kappa_unanimous"]


def test_ai_pcp_escalates_on_agreed_red_flag(script):
    flag = json.dumps({"urinary_retention": {"value": "present", "confirmed": False}})
    text, meta = arm(script, flag, flag).next_message(MSG)
    assert meta["decision"] == "escalate" and text == "ESCALATE"


def test_extractor_disagreement_is_not_accepted(script):
    flag = json.dumps({"urinary_retention": {"value": "present", "confirmed": False}})
    _, meta = arm(script, flag, "{}").next_message(MSG)
    assert meta["decision"] == "ask" and "urinary_retention" in meta["disputed"]


def test_out_of_lane_hands_off(script):
    text, meta = arm(script, "{}", "{}", route=ROUTE_NONE).next_message(MSG)
    assert meta["decision"] == "defer_out_of_lane" and text == "HANDOFF"


def test_schema_gap_defers(script):
    route = json.dumps({"category": "back", "suspected": ["other:spinal_epidural_abscess"]})
    _, meta = arm(script, "{}", "{}", route=route).next_message(MSG)
    assert meta["decision"] == "defer"


def test_run_disagreement_defers(script):
    calls = iter([ROUTE_BACK, ROUTE_NONE, ROUTE_BACK])
    a = arm(script, "{}", "{}")
    a.router = FakeLLM({"Pick the one complaint": lambda m: next(calls)})
    _, meta = a.next_message(MSG)
    assert meta["decision"] == "defer_run_disagreement"


def test_baselines_run():
    llm = FakeLLM({"Classify the patient": '{"emergency": true}'})
    assert "911" in StandardGuardrails(llm).next_message(MSG)[0]
    ok = FakeLLM({"Classify the patient": "{}", "You review a draft": '{"unsafe": false}'})
    assert StandardGuardrails(ok).next_message(MSG)[1]["path"] == "pass"
    for cls in (RawLLM, AmieStyle, MaiDxoStyle):
        text, meta = cls(FakeLLM({})).next_message(MSG)
        assert text == "ok" and meta["arm"]


def test_capture_usage_is_per_thread_and_scoped():
    import threading

    from compass_pcp.arms.llm import _record, capture_usage

    class Fake:
        vendor, model = "x", "m"

    seen = {}

    def other():
        _record(Fake, 99, 99)  # outside any capture block on this thread: not recorded anywhere
        seen["ok"] = True

    with capture_usage() as log:
        _record(Fake, 10, 3)
        t = threading.Thread(target=other)
        t.start()
        t.join()
        _record(Fake, 5, 2)
    _record(Fake, 1, 1)  # after the block: ignored
    assert seen["ok"] and log == {"x:m": {"calls": 2, "input_tokens": 15, "output_tokens": 5}}

