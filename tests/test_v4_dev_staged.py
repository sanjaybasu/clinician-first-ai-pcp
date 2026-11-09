"""paper/v4_dev_staged.py with fake clients: v4 arms reuse the v3 ledger calls of the same slot, the B1 screen on its
worker thread reads and writes the current slot, a screen miss is queued (never swallowed as a failed screen), and
the screen's thinking setting reaches the batch request. Synthetic text only."""

import json
from dataclasses import dataclass

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from paper.batch_stage import Store, slot
from paper.v4_dev_staged import ARMS, COMPARATORS, V4_ARMS, Bundle, StagedLLMv4, run_slot, tier_of
from tests.test_arms import FakeLLM
from tests.test_v4_arm import Dialogue, router_reply, screen_reply


@dataclass
class Fake(FakeLLM):
    vendor: str = "anthropic"
    thinking_headroom: int = 0


@dataclass
class FakeDialogue(Dialogue):
    vendor: str = "anthropic"
    thinking_headroom: int = 0


def make_bundle(script, store, realtime=True):
    engine = AiPcpV2(
        scripts={"back": script},
        router=Fake({"First decide who is writing": router_reply}, model="router"),
        hazard_llm=Fake({"You screen patient messages": json.dumps({"tier": "none", "category": "none", "quote": ""})},
                        model="hazard"),
        extractor_a=Fake({"You code the patient": "{}"}, model="xa"),
        extractor_b=Fake({"You code the patient": "{}"}, model="xb"),
        dialogue=FakeDialogue({}, model="dialogue"),
        compose_version=3,
    )
    screen = Fake({"You fill a safety form": screen_reply}, model="screen")
    screen.thinking = {"type": "between_tools"}
    return Bundle(engine, screen, store, realtime, FROZEN_SCRIPT_ARMS)


def row(pid, text):
    return {"prompt_id": pid, "prompt": [{"role": "user", "content": text}]}


# Arms on the frozen illness scripts share the v3 ledger calls; an arm with development scripts (v10) changes the
# extractor prompts for the complaints it overrides, so it makes its own calls for them.
FROZEN_SCRIPT_ARMS = [a for a in ARMS if a not in COMPARATORS and not V4_ARMS.get(a, {}).get("dev_scripts")]


def test_v4_arms_reuse_the_v3_ledger_calls_and_screen_on_the_slot(script, tmp_path):
    store = Store(tmp_path / "store")
    b = make_bundle(script, store)
    new, missing = run_slot(b, row("p1", "my back hurts"), 0, FROZEN_SCRIPT_ARMS, set(), realtime=True)
    assert missing == 0 and [r["arm"] for r in new] == FROZEN_SCRIPT_ARMS
    router, xa = b.base.router.inner, b.base.extractor_a.inner
    # one router call and one extractor A call in the prebuild; no arm calls them again
    assert router.usage.calls == 1 and xa.usage.calls == 1
    assert b.screen.inner.usage.calls == 1
    by = {r["arm"]: r for r in new}
    for name in ("v4_contract", "v4_edit", "v4_contract_compact", "v4_engine_off"):
        assert by[name]["info"]["screen"]["completed"] and by[name]["info"]["screen"]["error"] is None
    assert by["v4_contract"]["path"]["decision"] == by["v3"]["path"]["decision"] == "ask"
    assert by["v4_engine_off"]["path"]["decision"] == "general_out_of_lane"
    assert any(c["screen_v4"] for c in by["v4_contract"]["calls"])
    # v3's call log includes the ledger it reused, and never the v4 screen
    v3_specs = [c["spec"] for c in by["v3"]["calls"]]
    assert "anthropic:router" in v3_specs and not any(c["screen_v4"] for c in by["v3"]["calls"])


def test_screen_miss_is_queued_and_the_slot_waits(script, tmp_path):
    store = Store(tmp_path / "store")
    b = make_bundle(script, store, realtime=False)
    new, missing = run_slot(b, row("p2", "my back hurts"), 0, ARMS, set(), realtime=False)
    assert new == [] and missing == len(ARMS)
    systems = [r["body"]["system"] for r in store.queue.values()]
    assert any(s.startswith("You fill a safety form") for s in systems)
    assert any(s.startswith("First decide who is writing") for s in systems)
    screen_req = next(r["body"] for r in store.queue.values() if r["body"]["system"].startswith("You fill a safety form"))
    assert screen_req["thinking"] == {"type": "between_tools"}
    router_req = next(r["body"] for r in store.queue.values() if r["body"]["system"].startswith("First decide"))
    assert "thinking" not in router_req


def test_tiers():
    from compass_pcp.arms.screen_v4 import SCREEN_V4

    assert tier_of(SCREEN_V4) == 0 and tier_of("anything else") == 3


def test_v3_request_unchanged_by_the_v4_wrapper(tmp_path):
    from paper.batch_stage import StagedLLM

    inner = Fake({}, model="m")
    store = Store(tmp_path / "s")
    with slot(("x", 0)):
        a = StagedLLM(inner, store).request("sys", [{"role": "user", "content": "hi"}], 100, 0.0)
        b = StagedLLMv4(inner, store).request("sys", [{"role": "user", "content": "hi"}], 100, 0.0)
    assert a == b
