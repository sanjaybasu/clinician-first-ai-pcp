"""MedAgents port (paper/medagents_arm.py): verbatim prompts, staging, call structure, cap cut, bootstrap."""

import ast
from pathlib import Path

import pytest

from paper import medagents_arm as M
from paper.batch_stage import Pending, Store, slot

VENDOR = Path(__file__).resolve().parents[2] / "compass-pcp-data" / "vendor" / "MedicalAgentsBench" / "baselines" / "MedAgents"
VERBATIM = {
    "prompt_utils.py": ["get_question_domains_prompt", "get_question_analysis_prompt", "get_options_domains_prompt",
                        "get_options_analysis_prompt", "get_synthesized_report_prompt", "get_consensus_prompt",
                        "get_consensus_opinion_prompt", "get_revision_prompt"],
    "data_utils.py": ["remove_incomplete_sentence", "cleansing_analysis", "cleansing_syn_report", "cleansing_voting",
                      "transform_dict2text"],
}


def _funcs(path: Path) -> dict[str, str]:
    """Function bodies as normalized AST dumps (comments and formatting do not count; code does)."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = [n for n in node.body if not (isinstance(n, ast.Import) or isinstance(n, ast.ImportFrom))]
            out[node.name] = ast.dump(ast.Module(body=body, type_ignores=[]))
    return out


@pytest.mark.skipif(not VENDOR.exists(), reason="vendor copy of MedicalAgentsBench not present")
def test_prompts_and_cleansing_are_verbatim():
    mine = _funcs(Path(M.__file__))
    for fname, names in VERBATIM.items():
        theirs = _funcs(VENDOR / fname)
        for n in names:
            assert mine[n] == theirs[n], n


def test_render_question_and_punctuation_rule():
    q = M.render_question([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."},
                           {"role": "user", "content": "my head hurts"}])
    assert q == "User: hi\n\nAssistant: Hello.\n\nUser: my head hurts?"
    assert M.render_question([{"role": "user", "content": "Help."}]).endswith("Help.")


def test_request_body_and_cut():
    b = M.request_body("", "x", 30)
    assert "system" not in b and "temperature" not in b
    assert b["max_tokens"] == 30 + M.THINKING_HEADROOM and b["output_config"] == {"effort": "low"}
    assert M.request_body("sys", "x", 30)["system"] == "sys"
    assert M.cut_to_cap("a" * 1000, 30) == "a" * int(30 * M.CHARS_PER_TOKEN)
    assert M.cut_to_cap("abc", None) == "abc"


def test_final_prompt_carries_shared_policy_and_no_rubric():
    p = M.get_final_reply_prompt_wsyn("REPORT")
    assert M.OUTPUT_LIMITS in p and "REPORT" in p and "Option" not in p


class Fake:
    """Answers by role; votes 'NO' from the first domain in round 1 only, so one revision happens."""

    def __init__(self, no_rounds=1):
        self.calls, self.no_rounds, self.rounds = [], no_rounds, 0

    def __call__(self, system, user, cap):
        self.calls.append((system, cap))
        if "five different subfields" in user:
            return "Medical Field: A | B | C | D | E"
        if "two distinct subfields" in user:
            return "Medical Field: F | G"
        if "YES or NO" in user:
            return "NO" if ("domain A" in system or "A domain" in system) and self.rounds < self.no_rounds else "YES"
        if "propose revisions" in user:
            return "Revisions: be safer."
        if "revised analysis" in user:
            self.rounds += 1
            return "Total Analysis: revised."
        if "synthesized and synthesized" in user or "Key Knowledge: [extracted" in user:
            return "Key Knowledge: k \nTotal Analysis: t"
        if "write the reply" in user:
            return "Please call your doctor today."
        return "An analysis sentence. Another one."


class FakeOpus(M.Opus):
    def __init__(self, fake):
        super().__init__(store=None)
        self.fake = fake

    def lookup(self, system_role, user_input, max_tokens, headroom=M.THINKING_HEADROOM):
        self._record({"input_tokens": 1, "output_tokens": 1})
        return self.fake(system_role, user_input, max_tokens)


def test_pipeline_call_structure():
    pytest.importorskip("nltk")
    f = Fake(no_rounds=1)
    llm = FakeOpus(f)
    with slot("x"):
        text, meta = M.fully_decode([{"role": "user", "content": "I feel dizzy"}], llm)
    assert text == "Please call your doctor today."
    # 2 domains + 5 + 2 analyses + 1 synthesis + 7 votes + 1 advice + 1 revision + 7 votes + 1 final
    assert len(f.calls) == 27 and meta["vote_rounds"] == 2 and meta["n_revisions"] == 1
    caps = [c for _, c in f.calls]
    assert caps[:2] == [50, 50] and caps[2:9] == [300] * 7 and caps[9] == 2500 and caps[-1] == M.MAX_TOKENS


def test_minimum_18_calls_when_all_agree():
    pytest.importorskip("nltk")
    f = Fake(no_rounds=0)
    with slot("y"):
        M.fully_decode([{"role": "user", "content": "I feel dizzy"}], FakeOpus(f))
    assert len(f.calls) == 18


def test_stage_queues_every_independent_call(tmp_path):
    store = Store(tmp_path / "store")
    llm = M.Opus(store, realtime=False)
    with slot("z"), pytest.raises(Pending):
        M.fully_decode([{"role": "user", "content": "I feel dizzy"}], llm)
    assert len(store.queue) == 2  # both domain classifiers, nothing after them
    # answer them; the next pass queues the 5 question analyses together
    for k in list(store.queue):
        store.put(k, "Medical Field: A | B | C | D | E", {"input_tokens": 1, "output_tokens": 1}, M.SPEC)
    store.queue.clear()
    with slot("z"), pytest.raises(Pending):
        M.fully_decode([{"role": "user", "content": "I feel dizzy"}], llm)
    assert len(store.queue) == 5
    body = next(iter(store.queue.values()))["body"]
    assert body["model"] == M.MODEL and body["max_tokens"] == 300 + M.THINKING_HEADROOM


def test_paired_bootstrap_seeded():
    r1 = M.paired_bootstrap([0.1, 0.2, -0.1, 0.0, 0.3], n_boot=1000)
    r2 = M.paired_bootstrap([0.1, 0.2, -0.1, 0.0, 0.3], n_boot=1000)
    assert r1 == r2 and r1["ci95"][0] <= r1["diff"] <= r1["ci95"][1] and r1["seed"] == 20261028


def test_vendor_license_is_mit():
    lic = VENDOR.parents[1] / "LICENSE"
    if not lic.exists():
        pytest.skip("vendor copy not present")
    assert "MIT License" in lic.read_text() and "Gerstein Lab" in lic.read_text()
