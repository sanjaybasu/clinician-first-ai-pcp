from datetime import datetime, timedelta

import pytest

from compass_pcp.findings import Polarity
from compass_pcp.harness.adjudication import blind_packet, fleiss_kappa, gwet_ac1, select_for_adjudication
from compass_pcp.harness.episode import (
    Cohort,
    Episode,
    LabResult,
    Outcome30d,
    RecordFact,
    temporal_patient_disjoint_split,
)
from compass_pcp.harness.gatekeeper import AnswerStatus, Gatekeeper
from compass_pcp.harness.healthbench import compare_consensus, dev_split
from compass_pcp.harness.metrics import (
    cases_needed_for_zero_event_upper,
    mcnemar_exact_p,
    paired_bootstrap_diff,
    weighted_proportion,
    wilson,
    zero_event_upper,
)
from compass_pcp.harness.prereg import freeze, verify
from compass_pcp.harness.runner import RunStatus, SystemOutput, VendorPolicy, run_all, run_to_run_disagreement

T0 = datetime(2026, 8, 1)


def ep(i: str, patient: str, t: datetime, cohort=Cohort.B_LANE, **kw) -> Episode:
    return Episode(episode_id=i, patient_key=patient, cohort=cohort, lane="dysuria", complaint="dysuria",
                   index_time=t, opening_message="it burns when i pee", context_snapshot=[],
                   record_facts={"fever": RecordFact(polarity=Polarity.ABSENT)},
                   labs=[LabResult(loinc="5799-2", value=1.0, resulted_at=t + timedelta(days=1))], **kw)


def test_split_is_temporal_and_patient_disjoint():
    eps = [ep("e1", "p1", T0 - timedelta(days=5)), ep("e2", "p1", T0 + timedelta(days=5)), ep("e3", "p2", T0 + timedelta(days=1))]
    s = temporal_patient_disjoint_split(eps, T0)
    assert s.dev == ["e1"] and s.test == ["e3"] and s.dropped_for_patient_overlap == ["e2"]


def test_gatekeeper_never_invents():
    gk = Gatekeeper(ep("e1", "p1", T0))
    assert gk.ask("fever", "any fever?").polarity == Polarity.ABSENT
    assert gk.ask("flank_pain", "any flank pain?").status == AnswerStatus.NOT_DOCUMENTED
    assert gk.order("5799-2", 3).status == AnswerStatus.ANSWERED
    assert gk.order("630-4", 3).status == AnswerStatus.NOT_AVAILABLE
    assert len(gk.log) == 4


class FakeSystem:
    name = "fake"
    vendors = {"generator": "google", "verifier": "anthropic", "extractor_a": "google", "extractor_b": "openai"}  # noqa: RUF012

    def __init__(self, drop_component_on=None):
        self.drop = drop_component_on

    def run(self, episode, gatekeeper, run_index):
        gatekeeper.ask("fever", "fever?")
        comps = {"extractor_a": True, "extractor_b": True, "workup": True, "gate": episode.episode_id != self.drop}
        return SystemOutput(decision="complete" if run_index < 2 else "defer", components_executed=comps, versions={"commit": "x"})


def test_runner_counts_missing_component_as_failure_and_resumes(tmp_path):
    out = tmp_path / "runs.jsonl"
    eps = [ep("e1", "p1", T0), ep("e2", "p2", T0)]
    recs = run_all(FakeSystem(drop_component_on="e2"), eps, out, ["extractor_a", "extractor_b", "workup", "gate"])
    assert len(recs) == 6
    assert {r.status for r in recs if r.episode_id == "e2"} == {RunStatus.FAILED_COMPONENT}
    assert run_all(FakeSystem(), eps, out, ["gate"]) == []  # resume: nothing left
    assert run_to_run_disagreement(recs)["e1"] == ["decision"]


def test_vendor_policy():
    with pytest.raises(ValueError):
        VendorPolicy().check({"generator": "google", "verifier": "google"})


def test_blind_packet_hides_model_and_outcome():
    e = ep("e1", "p1", T0, outcome=Outcome30d(ed_visit=True), historical_decision="home_care")
    dumped = blind_packet(e).model_dump()
    assert "outcome" not in dumped and "historical_decision" not in dumped


def test_selection_takes_all_c_positives():
    e = ep("c1", "p9", T0, cohort=Cohort.C_CANNOT_MISS, outcome=Outcome30d(cannot_miss_diagnoses=["cauda_equina"]))
    sel = select_for_adjudication([e], [])
    assert sel[0].reason == "cohort_c_positive" and sel[0].inclusion_prob == 1.0


def test_metrics():
    lo, hi = wilson(97, 100)
    assert 0.91 < lo < 0.92 and hi > 0.98
    assert math_close(zero_event_upper(300), 0.00994, 1e-4)
    assert cases_needed_for_zero_event_upper(0.01) == 299
    assert weighted_proportion([True, False], [1.0, 0.2]) == pytest.approx(1 / 6)
    d, lo, hi = paired_bootstrap_diff([1.0] * 50, [0.0] * 50, n_boot=200)
    assert d == 1.0 and lo == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0 and mcnemar_exact_p(10, 0) < 0.01


def math_close(a, b, tol):
    return abs(a - b) < tol


def test_ac1_stays_high_when_kappa_collapses():
    ratings = [["safe"] * 3 for _ in range(97)] + [["safe", "safe", "unsafe"]] * 3
    assert gwet_ac1(ratings) > 0.9
    assert fleiss_kappa(ratings) < 0.1


def test_freeze_detects_changes(tmp_path):
    p = tmp_path / "freeze.json"
    freeze(p, architecture_commit="a", compass_commit="c", thresholds={"t": 1}, test_ids=["e1", "e2"])
    verify(p, architecture_commit="a", compass_commit="c", thresholds={"t": 1}, test_ids=["e2", "e1"])
    with pytest.raises(RuntimeError):
        verify(p, architecture_commit="a", compass_commit="c", thresholds={"t": 2}, test_ids=["e1", "e2"])
    with pytest.raises(FileExistsError):
        freeze(p, architecture_commit="a", compass_commit="c", thresholds={}, test_ids=[])


def test_healthbench_split_and_consensus():
    ids = [f"p{i}" for i in range(1000)]
    dev, test = dev_split(ids, salt="s")
    assert set(dev).isdisjoint(test) and 150 < len(dev) < 250
    assert dev_split(ids, salt="s") == (dev, test)
    r = compare_consensus({"a": True, "b": True}, {"a": True, "b": False})
    assert r["a_only"] == 1 and r["b_only"] == 0


def test_hypotheses_and_holm():
    from compass_pcp.harness.configurations import DESCRIPTIONS, Configuration, holm, prespecified_hypotheses

    assert set(DESCRIPTIONS) == set(Configuration)
    hs = prespecified_hypotheses()
    assert len({h.id for h in hs}) == len(hs)
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "c": False, "b": False}
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}



def test_order_episodes_always_adjudicated():
    from compass_pcp.harness.runner import RunRecord

    e = ep("o1", "p5", T0)
    rec = RunRecord(episode_id="o1", run_index=0, status=RunStatus.OK,
                    output=SystemOutput(decision="complete", prescriptions=["nitrofurantoin"], components_executed={},
                                        versions={}))
    assert select_for_adjudication([e], [rec], random_frac=0.0)[0].reason == "has_order_or_prescription"


def test_ablation_blocked_in_production(monkeypatch):
    from compass_pcp.engine.workup import WorkupConfig

    monkeypatch.setenv("COMPASS_PCP_ENV", "production")
    with pytest.raises(ValueError):
        WorkupConfig(enforce_stopping_rule=False)
    WorkupConfig()


def test_discrimination_metrics():
    from compass_pcp.harness.metrics import (
        auroc,
        f1_youden,
        net_benefit,
        partial_auroc_high_sensitivity,
        specificity_at_sensitivity_floor,
    )

    labels = [True, True, True, False, False, False]
    assert auroc([3, 2, 1, 0, 0, 0], labels) == 1.0
    assert auroc([1, 1, 1, 1, 1, 1], labels) == 0.5
    assert partial_auroc_high_sensitivity([3, 2, 1, 0, 0, 0], labels, 0.9) == pytest.approx(1.0)
    assert partial_auroc_high_sensitivity([1, 1, 1, 1, 1, 1], labels, 0.9) == pytest.approx(0.05)
    op = specificity_at_sensitivity_floor([3, 2, 1, 2, 0, 0], labels, 1.0)
    assert op["sensitivity"] == 1.0 and op["specificity"] == pytest.approx(2 / 3)
    assert net_benefit(10, 0, 100, 0.02) == pytest.approx(0.1)
    m = f1_youden(8, 2, 2, 88)
    assert m["youden_j"] == pytest.approx(0.8 + 88 / 90 - 1)



def test_response_cache_resumes_and_keeps_repeats_distinct(tmp_path):
    from compass_pcp.harness.cache import ResponseCache

    calls = []
    c = ResponseCache(tmp_path / "c.jsonl")
    assert [c.get_or_call({"m": 1}, lambda: calls.append(1) or f"r{len(calls)}") for _ in range(3)] == ["r1", "r2", "r3"]
    with (tmp_path / "c.jsonl").open("a") as fh:
        fh.write('{"key": "torn')  # interrupted write
    c2 = ResponseCache(tmp_path / "c.jsonl")
    out = [c2.get_or_call({"m": 1}, lambda: calls.append(1) or f"r{len(calls)}") for _ in range(4)]
    assert out == ["r1", "r2", "r3", "r4"] and len(calls) == 4
    c3 = ResponseCache(tmp_path / "c.jsonl")  # the record written after the torn line survives
    assert [c3.get_or_call({"m": 1}, lambda: "new") for _ in range(4)] == ["r1", "r2", "r3", "r4"]
