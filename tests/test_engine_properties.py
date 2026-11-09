"""Safety properties of the workup engine, checked on every signed illness script (design/paper2_plan.md, workstream 3).

Red-flag states are enumerated exhaustively (present, absent or unknown for every must_ask feature); the other features
are drawn at random with a fixed seed. Properties:
  P1 a reported red flag acts by rule: escalate (or defer, when the script says so), never ask or complete;
  P2 the engine completes only when every red flag is answered, no cannot-miss diagnosis is open or escalating, there
     is no contradiction, and the lane may close;
  P3 a workup ends: the engine never asks a feature it already has, and stops asking within the question budget;
  P4 once a patient reports an escalating red flag, no later answers lead to completion;
  P5 the same inputs give the same decision.
"""

import itertools
import random
from pathlib import Path

import pytest

from compass_pcp.engine.scripts import load_script
from compass_pcp.engine.stopping import CannotMissStatus
from compass_pcp.engine.workup import DecisionKind, WorkupConfig, WorkupState, step
from compass_pcp.findings import Polarity, Provenance
from compass_pcp.lanes import may_close
from tests.conftest import make_finding

SCRIPTS = [load_script(p) for p in sorted((Path(__file__).resolve().parents[1] / "compass_pcp/data/scripts").glob("*.yaml"))]
IDS = [s.complaint for s in SCRIPTS]
SEED = 20261103
STATES = (Polarity.PRESENT, Polarity.ABSENT, None)
COMPLETES: dict[str, int] = {}


def _findings(assign: dict) -> list:
    return [make_finding(f, pol) for f, pol in assign.items() if pol is not None]


def _random_other(script, rng: random.Random, exclude: set[str]) -> dict:
    return {f.id: rng.choice(STATES) for f in script.features if f.id not in exclude}


def _flag_states(script):
    for combo in itertools.product(STATES, repeat=len(script.must_ask)):
        yield dict(zip(script.must_ask, combo))


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p1_reported_red_flag_acts_by_rule(script):
    rng = random.Random(SEED)
    for flags in _flag_states(script):
        present = [f for f, pol in flags.items() if pol == Polarity.PRESENT]
        if not present:
            continue
        d = step(script, WorkupState(findings=_findings({**_random_other(script, rng, set(flags)), **flags})))
        expected = (DecisionKind.ESCALATE if any(script.red_flag_action.get(f, "escalate") == "escalate" for f in present)
                    else DecisionKind.DEFER)
        assert d.kind == expected, (script.complaint, flags, d.kind)


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p2_completion_only_with_obligations_met(script):
    rng = random.Random(SEED + 1)
    completes = 0
    for flags in _flag_states(script):
        for _ in range(3):
            assign = {**_random_other(script, rng, set(flags)), **flags}
            d = step(script, WorkupState(findings=_findings(assign)))
            if d.kind != DecisionKind.COMPLETE:
                continue
            completes += 1
            observed = {f for f, pol in assign.items() if pol is not None}
            assert set(script.must_ask) <= observed, (script.complaint, "unanswered red flag at completion")
            assert all(c.status not in (CannotMissStatus.NEEDS_WORKUP, CannotMissStatus.ESCALATE)
                       for c in d.certificate.cannot_miss), (script.complaint, "open cannot-miss at completion")
            assert not d.certificate.contradictions
            assert may_close(script.lane)
    # low-risk states: every red flag confirmed absent in the record, other features mostly absent
    for _ in range(300):
        assign = {f.id: rng.choice((Polarity.ABSENT, Polarity.ABSENT, Polarity.ABSENT, Polarity.PRESENT, None))
                  for f in script.features if f.id not in script.must_ask}
        findings = _findings(assign) + [make_finding(f, Polarity.ABSENT, Provenance.STRUCTURED_RECORD)
                                        for f in script.must_ask]
        d = step(script, WorkupState(findings=findings))
        if d.kind == DecisionKind.COMPLETE:
            completes += 1
            assert all(c.status not in (CannotMissStatus.NEEDS_WORKUP, CannotMissStatus.ESCALATE)
                       for c in d.certificate.cannot_miss), (script.complaint, "open cannot-miss at completion")
            assert may_close(script.lane)
    COMPLETES[script.complaint] = completes
    if not may_close(script.lane):
        assert completes == 0


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p3_workup_ends_without_repeat_questions(script):
    rng = random.Random(SEED + 2)
    cfg = WorkupConfig()
    for _ in range(200):
        truth = {f.id: rng.choice((Polarity.PRESENT, Polarity.ABSENT)) for f in script.features}
        known: dict = {}
        for turn in range(cfg.question_budget + 1):
            d = step(script, WorkupState(findings=_findings(known), turns_used=turn), cfg)
            if d.kind != DecisionKind.ASK:
                break
            assert d.ask_features and not set(d.ask_features) & set(known), (script.complaint, "repeat question")
            for f in d.ask_features:
                known[f] = truth[f]
        else:
            pytest.fail(f"{script.complaint}: still asking after the question budget")


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p4_no_completion_after_escalating_red_flag(script):
    rng = random.Random(SEED + 3)
    escalating = [f for f in script.must_ask if script.red_flag_action.get(f, "escalate") == "escalate"]
    for flag in escalating:
        for _ in range(100):
            later = _random_other(script, rng, {flag})
            d = step(script, WorkupState(findings=_findings({**later, flag: Polarity.PRESENT})))
            assert d.kind == DecisionKind.ESCALATE, (script.complaint, flag, d.kind)


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p5_deterministic(script):
    rng = random.Random(SEED + 4)
    for _ in range(50):
        f = _findings(_random_other(script, rng, set()))
        a, b = step(script, WorkupState(findings=f)), step(script, WorkupState(findings=list(f)))
        assert a.kind == b.kind and a.ask_features == b.ask_features
        assert a.certificate.model_dump() == b.certificate.model_dump()



# Scripts whose lane may close but whose cannot-miss diagnoses stay in the middle zone even when every feature is
# confirmed absent, so the engine always defers to a clinician (found 2026-11-03; kept by decision of S. Basu,
# 2026-11-03, because of the risk of these diagnoses).
ALWAYS_DEFER = {"dysuria", "red_eye"}


def test_p2_is_exercised():
    """Completion must actually occur for the closable scripts outside ALWAYS_DEFER, or P2 checks nothing there."""
    for s in SCRIPTS:
        if may_close(s.lane) and s.complaint not in ALWAYS_DEFER and s.complaint in COMPLETES:
            if s.complaint == "cough_acute":
                continue  # completes only on record negatives for the pneumonia features; see the next test
            assert COMPLETES[s.complaint] > 0, (s.complaint, COMPLETES)


@pytest.mark.parametrize("name", sorted(ALWAYS_DEFER | {"cough_acute"}))
def test_closure_reachability_is_as_documented(name):
    from compass_pcp.findings import Provenance as P
    s = next(x for x in SCRIPTS if x.complaint == name)
    kinds = {prov: step(s, WorkupState(findings=[make_finding(f.id, Polarity.ABSENT, prov) for f in s.features])).kind
             for prov in (P.PATIENT_REPORT, P.STRUCTURED_RECORD)}
    assert kinds[P.PATIENT_REPORT] == DecisionKind.DEFER
    assert kinds[P.STRUCTURED_RECORD] == (DecisionKind.DEFER if name in ALWAYS_DEFER else DecisionKind.COMPLETE)


@pytest.mark.parametrize("script", SCRIPTS, ids=IDS)
def test_p6_futility_stop_is_sound(script):
    """futility_stop (Paper 2 development) defers only with every red flag observed and a cannot-miss diagnosis open,
    and only when no single further answer to an open diagnosis's remote features would change its status."""
    from compass_pcp.engine.deferral import DeferralReason
    from compass_pcp.findings import CodeSystem, Finding

    rng = random.Random(SEED + 6)
    cfg = WorkupConfig(futility_stop=True)
    fired = 0
    for flags in _flag_states(script):
        for _ in range(3):
            assign = {**_random_other(script, rng, set(flags)), **flags}
            findings = _findings(assign)
            d = step(script, WorkupState(findings=findings), cfg)
            if DeferralReason.CANNOT_MISS_FUTILE_QUESTIONS not in d.certificate.deferral_reasons:
                continue
            fired += 1
            observed = {f for f, pol in assign.items() if pol is not None}
            assert set(script.must_ask) <= observed
            open_ids = {c.diagnosis_id for c in d.certificate.cannot_miss if c.status == CannotMissStatus.NEEDS_WORKUP}
            assert open_ids
            for dx in script.cannot_miss:
                if dx.id not in open_ids:
                    continue
                for fid in dx.lrs:
                    if fid in observed or not script.feature(fid).remote_askable:
                        continue
                    for pol in (Polarity.PRESENT, Polarity.ABSENT):
                        extra = Finding(feature_id=fid, code_system=CodeSystem.LOCAL, code=fid, polarity=pol,
                                        provenance=Provenance.PATIENT_REPORT, extractor_id="t", confirmed=True)
                        after = step(script, WorkupState(findings=[*findings, extra]))
                        st = {c.diagnosis_id: c.status for c in after.certificate.cannot_miss}
                        assert st[dx.id] == CannotMissStatus.NEEDS_WORKUP, (script.complaint, fid, pol)
    COMPLETES[f"futile:{script.complaint}"] = fired
