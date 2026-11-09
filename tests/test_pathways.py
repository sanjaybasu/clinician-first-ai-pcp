"""Properties of the guideline pathways (design/pathway_schema.md, section 3), on every script that has a pathway.

States are enumerated as in test_engine_properties: every red-flag combination (present, absent, unknown), other
features at random with a fixed seed, plus low-risk states with the red flags confirmed absent in the record.
  Q1 every engine state matches a pathway node;
  Q2 a node's disposition is not below the engine's floor (escalate -> same day or higher; defer -> clinician review);
  Q3 every cited source is defined (checked when the file loads);
  Q4 a question asked while a cannot-miss diagnosis is open carries a standing disposition of clinician review or
     higher.
"""

import random

import pytest

from compass_pcp.engine.stopping import CannotMissStatus
from compass_pcp.engine.workup import DecisionKind, WorkupConfig, WorkupState, step
from compass_pcp.findings import Polarity, Provenance
from compass_pcp.pathways import LEVEL_ORDER, PATHWAY_DIR, evaluate, load_pathway
from tests.conftest import make_finding
from tests.test_engine_properties import SCRIPTS, SEED, _findings, _flag_states, _random_other

PATHWAYS = {p.stem: load_pathway(p) for p in sorted(PATHWAY_DIR.glob("*.yaml"))}
FLOOR = {DecisionKind.ESCALATE: "urgent_same_day", DecisionKind.DEFER: "clinician_review"}


def _states(script, rng):
    for flags in _flag_states(script):
        for _ in range(3):
            yield _findings({**_random_other(script, rng, set(flags)), **flags})
    for _ in range(200):
        other = {f.id: rng.choice((Polarity.ABSENT, Polarity.ABSENT, Polarity.PRESENT, None))
                 for f in script.features if f.id not in script.must_ask}
        yield _findings(other) + [make_finding(f, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for f in script.must_ask]
    for _ in range(200):  # record negatives for a random subset of every feature (the only route to completion in cough)
        yield [make_finding(f.id, Polarity.ABSENT, Provenance.STRUCTURED_RECORD) for f in script.features
               if rng.random() < 0.8]


@pytest.mark.parametrize("name", sorted(PATHWAYS))
def test_pathway_properties(name):
    pw = PATHWAYS[name]
    script = next(s for s in SCRIPTS if s.complaint == name)
    assert pw.script == f"{script.complaint}@{script.version}"
    rng = random.Random(SEED + 10)
    seen = set()
    states = [(f, t) for f in _states(script, rng) for t in (0, WorkupConfig().question_budget)]  # also budget spent
    for findings, turns in states:
        d = step(script, WorkupState(findings=findings, turns_used=turns))
        node = evaluate(pw, d, findings)
        assert node is not None, (name, d.kind, [(c.diagnosis_id, c.status) for c in d.certificate.cannot_miss])
        seen.add(node.id)
        out = node.disposition or node.standing
        if d.kind in FLOOR:
            assert node.disposition is not None and LEVEL_ORDER[node.disposition.level] >= LEVEL_ORDER[FLOOR[d.kind]], \
                (name, node.id, d.kind)
        if d.kind == DecisionKind.ASK and any(c.status == CannotMissStatus.NEEDS_WORKUP for c in d.certificate.cannot_miss):
            assert node.standing is not None and LEVEL_ORDER[node.standing.level] >= LEVEL_ORDER["clinician_review"], \
                (name, node.id)
        assert out is not None
    unreached = {n.id for n in pw.nodes} - seen
    assert not unreached, (name, "nodes never reached", unreached)


def test_pathway_escalation_keeps_emergency_without_a_pathway():
    from types import SimpleNamespace as NS

    from compass_pcp.arms.ai_pcp_v4 import pathway_escalation
    from compass_pcp.pathways import Disposition, Node, When

    urgent = Node(id="u", when=When(), disposition=Disposition(level="urgent_same_day", within_hours=24), source=["x"])
    asking = Node(id="a", when=When(), standing=Disposition(level="clinician_review"), source=["x"])
    esc = lambda c: (NS(complaint=c), NS(kind=DecisionKind.ESCALATE))
    ask = lambda c: (NS(complaint=c), NS(kind=DecisionKind.ASK))
    assert pathway_escalation([esc("cough_acute")], {"cough_acute": urgent}, "escalate") == "urgent_same_day"
    assert pathway_escalation([esc("cough_acute"), esc("chest_pain")], {"cough_acute": urgent}, "escalate") == "escalate"
    assert pathway_escalation([esc("cough_acute")], {"cough_acute": asking}, "escalate") == "escalate"
    assert pathway_escalation([esc("cough_acute"), ask("dyspnea")], {"cough_acute": urgent}, "escalate") == "urgent_same_day"


def test_dev_sore_throat_epiglottitis_red_flags_escalate():
    from pathlib import Path

    from compass_pcp.engine.scripts import load_script

    dev = load_script(Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts_dev/sore_throat.yaml")
    for flag in ("stridor", "trouble_breathing", "drooling"):
        assert step(dev, WorkupState(findings=[make_finding(flag)])).kind == DecisionKind.ESCALATE, flag
    assert step(dev, WorkupState(findings=[make_finding("fever_history")])).kind != DecisionKind.ESCALATE


def test_pathway_completion_needs_every_complaint_complete_with_a_home_or_routine_disposition():
    from types import SimpleNamespace as NS

    from compass_pcp.arms.ai_pcp_v4 import pathway_completion
    from compass_pcp.pathways import Disposition, Node, When

    home = Node(id="h", when=When(), disposition=Disposition(level="self_care"), source=["x"])
    visit = Node(id="v", when=When(), disposition=Disposition(level="routine_visit", within_hours=72), source=["x"])
    review = Node(id="r", when=When(), disposition=Disposition(level="clinician_review"), source=["x"])
    done = lambda c: (NS(complaint=c), NS(kind=DecisionKind.COMPLETE))
    assert pathway_completion([done("sore_throat")], {"sore_throat": home}) == "self_care"
    assert pathway_completion([done("sore_throat"), done("ear_pain")], {"sore_throat": home, "ear_pain": visit}) == "routine_visit"
    assert pathway_completion([done("sore_throat"), done("cough_acute")], {"sore_throat": home}) is None
    assert pathway_completion([done("sore_throat")], {"sore_throat": review}) is None
    assert pathway_completion([(NS(complaint="sore_throat"), NS(kind=DecisionKind.DEFER))], {"sore_throat": home}) is None


def test_red_flags_first_asks_rule_acting_flags_before_likelihood_questions():
    from pathlib import Path

    from compass_pcp.engine.scripts import load_script

    dev = load_script(Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts_dev/sore_throat.yaml")
    lr = {f for d in dev.diagnoses for f in d.lrs}
    rule = [f for f in dev.must_ask if f not in lr]
    assert rule == ["stridor", "trouble_breathing", "drooling"]
    d = step(dev, WorkupState(findings=[]), WorkupConfig(red_flags_first=True))
    assert d.kind == DecisionKind.ASK and d.ask_features[0] in rule
    assert step(dev, WorkupState(findings=[]), WorkupConfig()).ask_features[0] not in rule  # default unchanged
    answered = [make_finding(f, Polarity.ABSENT) for f in rule]
    d = step(dev, WorkupState(findings=answered), WorkupConfig(red_flags_first=True))
    assert d.kind != DecisionKind.ASK or d.ask_features[0] not in rule


def _v12(name):
    from pathlib import Path

    from compass_pcp.engine.scripts import load_script

    root = Path(__file__).resolve().parents[1] / "compass_pcp/data"
    return load_script(root / f"scripts_dev_v12/{name}.yaml"), load_pathway(root / f"pathways_dev_v12/{name}.yaml")


def test_v12_sinusitis_bacterial_criteria_are_recorded_and_route_to_a_routine_visit():
    from compass_pcp.arms.ai_pcp_v4 import pathway_completion

    script, pw = _v12("acute_sinusitis")
    known = [make_finding(f.id, Polarity.ABSENT) for f in script.features if f.remote_askable]
    base = step(script, WorkupState(findings=known), WorkupConfig(red_flags_first=True))
    assert base.kind == DecisionKind.COMPLETE
    assert evaluate(pw, base, known).disposition.level == "self_care"
    for crit in ("symptoms_10_days_plus", "worse_after_better", "severe_onset"):
        fs = [f for f in known if f.feature_id != crit] + [make_finding(crit)]
        d = step(script, WorkupState(findings=fs), WorkupConfig(red_flags_first=True))
        assert d.kind == DecisionKind.COMPLETE, crit  # recorded, not acted on by the engine
        node = evaluate(pw, d, fs)
        assert node.id == "complete_bacterial_criteria"
        assert pathway_completion([(script, d)], {"acute_sinusitis": node}) == "routine_visit"


def test_v12_ear_antibiotics_is_same_day_and_mastoid_flags_stay_emergency():
    from compass_pcp.arms.ai_pcp_v4 import pathway_escalation

    script, pw = _v12("ear_pain")
    absent = [make_finding(f, Polarity.ABSENT) for f in script.must_ask if f != "on_antibiotics_for_this"]
    fs = [*absent, make_finding("on_antibiotics_for_this")]
    d = step(script, WorkupState(findings=fs), WorkupConfig(red_flags_first=True))
    assert d.kind == DecisionKind.ESCALATE
    node = evaluate(pw, d, fs)
    assert node.id == "treatment_failure" and node.disposition.level == "urgent_same_day"
    assert pathway_escalation([(script, d)], {"ear_pain": node}, "escalate") == "urgent_same_day"
    both = [*[f for f in fs if f.feature_id != "swelling_behind_ear"], make_finding("swelling_behind_ear")]
    d2 = step(script, WorkupState(findings=both), WorkupConfig(red_flags_first=True))
    n2 = evaluate(pw, d2, both)
    assert n2.disposition.level == "emergency_now"
    assert pathway_escalation([(script, d2)], {"ear_pain": n2}, "escalate") == "escalate"


def test_skip_unanswered_does_not_repeat_a_question_and_defers_an_unanswerable_red_flag():
    from pathlib import Path

    from compass_pcp.engine.scripts import load_script

    script = load_script(Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts_dev_v12/sore_throat.yaml")
    cfg = WorkupConfig(red_flags_first=True, skip_unanswered=True)
    lr = {f for d in script.diagnoses for f in d.lrs}
    flags = [make_finding(f, Polarity.ABSENT) for f in script.must_ask if f not in lr]
    d = step(script, WorkupState(findings=flags), cfg)
    assert d.kind == DecisionKind.ASK
    first = d.ask_features[0]
    d2 = step(script, WorkupState(findings=flags, asked_unanswered=[first]), cfg)
    assert first not in (d2.ask_features or [])
    if first in script.must_ask:
        assert d2.kind == DecisionKind.DEFER
    assert step(script, WorkupState(findings=flags, asked_unanswered=[first]), WorkupConfig(red_flags_first=True)).ask_features[0] == first


def test_arm_finds_questions_asked_earlier_and_still_unanswered():
    from pathlib import Path
    from types import SimpleNamespace as NS

    from compass_pcp.arms.ai_pcp_v4 import asked_unanswered
    from compass_pcp.engine.scripts import load_script
    from compass_pcp.ledger import Ledger

    script = load_script(Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts_dev_v12/sore_throat.yaml")
    q = {f.id: f.question for f in script.features}
    ledger = Ledger()
    ledger.add_complaints(["sore_throat"])
    ledger.add_finding(1, make_finding("stridor", Polarity.ABSENT))
    msgs = [{"role": "user", "content": "my throat hurts"},
            {"role": "assistant", "content": f"Rest and fluids help. {q['trismus']}"},
            {"role": "user", "content": "I am not sure."},
            {"role": "assistant", "content": f"Thanks. {q['stridor']}"},
            {"role": "user", "content": "no"}]
    on = NS(config=WorkupConfig(skip_unanswered=True), scripts={"sore_throat": script})
    assert asked_unanswered(on, ledger, msgs, 0.6) == ("trismus",)  # stridor was answered; drooling never asked
    off = NS(config=WorkupConfig(), scripts={"sore_throat": script})
    assert asked_unanswered(off, ledger, msgs, 0.6) == ()


def test_remaining_red_flags_lists_unanswered_flags_of_escalating_complaints():
    from types import SimpleNamespace as NS

    from compass_pcp.arms.ai_pcp_v4 import remaining_red_flags
    from compass_pcp.ledger import Ledger

    ledger = Ledger()
    ledger.add_complaints(["cough_acute"])
    ledger.add_finding(1, make_finding("coughing_blood"))
    cough = NS(must_ask=["fever_history", "breathlessness", "chest_pain", "coughing_blood"])
    other = NS(must_ask=["sweating"])
    decisions = [(cough, NS(kind=DecisionKind.ESCALATE)), (other, NS(kind=DecisionKind.ASK))]
    assert remaining_red_flags(decisions, ledger) == ["fever_history", "breathlessness", "chest_pain"]


def test_unanswered_record_criterion_does_not_defer_or_repeat():
    script, pw = _v12("acute_sinusitis")
    cfg = WorkupConfig(red_flags_first=True, skip_unanswered=True)
    known = [make_finding(f.id, Polarity.ABSENT) for f in script.features
             if f.remote_askable and f.id != "symptoms_10_days_plus"]
    d = step(script, WorkupState(findings=known, asked_unanswered=["symptoms_10_days_plus"]), cfg)
    assert d.kind == DecisionKind.COMPLETE
    assert evaluate(pw, d, known).disposition.level == "self_care"
    flag = [f for f in known if f.feature_id != "vision_change"]
    d2 = step(script, WorkupState(findings=flag, asked_unanswered=["symptoms_10_days_plus", "vision_change"]), cfg)
    assert d2.kind == DecisionKind.DEFER  # an unanswered red flag still defers


def test_v14_unanswerable_sinusitis_criterion_routes_to_a_routine_visit():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "compass_pcp/data"
    script, _ = _v12("acute_sinusitis")
    pw = load_pathway(root / "pathways_dev_v14/acute_sinusitis.yaml")
    cfg = WorkupConfig(red_flags_first=True, skip_unanswered=True)
    known = [make_finding(f.id, Polarity.ABSENT) for f in script.features
             if f.remote_askable and f.id != "symptoms_10_days_plus"]
    d = step(script, WorkupState(findings=known, asked_unanswered=["symptoms_10_days_plus"]), cfg)
    assert d.kind == DecisionKind.COMPLETE
    assert evaluate(pw, d, known, ["symptoms_10_days_plus"]).id == "complete_criteria_unknown"
    assert evaluate(pw, d, known).disposition.level == "self_care"  # answered "no": home care
    present = [*known, make_finding("symptoms_10_days_plus")]
    assert evaluate(pw, step(script, WorkupState(findings=present), cfg), present).id == "complete_bacterial_criteria"


def test_v18_muffled_voice_acts_through_its_likelihood_ratio_only():
    from pathlib import Path

    from compass_pcp.engine.scripts import load_script

    v18 = load_script(Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts_dev_v18/sore_throat.yaml")
    q = next(f.question for f in v18.features if f.id == "muffled_voice")
    assert q.endswith("(Not just hoarse or raspy.)") and "muffled_voice" in v18.must_ask
    assert step(v18, WorkupState(findings=[make_finding("muffled_voice")])).kind != DecisionKind.ESCALATE
    for flag in ("stridor", "trouble_breathing", "drooling", "trismus"):
        assert step(v18, WorkupState(findings=[make_finding(flag)])).kind == DecisionKind.ESCALATE, flag


def test_v20_nice_high_risk_completes_to_same_day():
    from pathlib import Path
    from types import SimpleNamespace as NS

    from compass_pcp.arms.ai_pcp_v4 import pathway_completion
    from compass_pcp.engine.scripts import load_script
    from compass_pcp.pathways import evaluate, load_pathway

    root = Path(__file__).resolve().parents[1] / "compass_pcp/data"
    for name in ("sore_throat", "acute_sinusitis"):
        s = load_script(root / f"scripts_dev_v20/{name}.yaml")
        pw = load_pathway(root / f"pathways_dev_v20/{name}.yaml")
        for f in ("high_risk_condition", "older_high_risk", "very_unwell"):
            assert f in s.must_ask and s.red_flag_action[f] == "record"
            assert step(s, WorkupState(findings=[make_finding(f)])).kind != DecisionKind.ESCALATE  # recorded, not escalated
        done = NS(kind=DecisionKind.COMPLETE, certificate=NS(cannot_miss=[]))
        node = evaluate(pw, done, [make_finding("very_unwell")])
        assert node.id == "complete_high_risk" and node.disposition.level == "urgent_same_day"
        dec = [(NS(complaint=name), NS(kind=DecisionKind.COMPLETE))]
        assert pathway_completion(dec, {name: node}) is None  # older arms: physician review
        assert pathway_completion(dec, {name: node}, allow_urgent=True) == "urgent_same_day"
