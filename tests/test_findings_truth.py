from compass_pcp.engine.truth import FALSE, TRUE, UNKNOWN, Bounds, and_, implies, not_, or_, predicate_bounds
from compass_pcp.findings import Polarity, Provenance, merge_with_record, reconcile
from tests.conftest import make_finding


def test_reconcile_accepts_only_agreement():
    a = [make_finding("x"), make_finding("y"), make_finding("z", value=1.0)]
    b = [make_finding("x", extractor="ext-b"), make_finding("y", Polarity.ABSENT, extractor="ext-b"),
         make_finding("z", value=2.0, extractor="ext-b"), make_finding("w", extractor="ext-b")]
    r = reconcile(a, b)
    assert [f.feature_id for f in r.accepted] == ["x"]
    assert r.disputed == ["w", "y", "z"]


def test_record_outranks_patient_report():
    rec = [make_finding("eGFR_lt_30", Polarity.ABSENT, Provenance.STRUCTURED_RECORD)]
    rep = [make_finding("eGFR_lt_30", Polarity.PRESENT, Provenance.PATIENT_REPORT)]
    merged = merge_with_record(rec, rep)
    assert len(merged) == 1 and merged[0].provenance == Provenance.STRUCTURED_RECORD


def test_lukasiewicz_ops():
    assert and_(TRUE, TRUE) == TRUE and and_(TRUE, FALSE) == FALSE
    assert or_(FALSE, TRUE) == TRUE
    assert not_(UNKNOWN) == UNKNOWN
    assert implies(TRUE, FALSE) == FALSE and implies(FALSE, FALSE) == TRUE
    assert and_(TRUE, UNKNOWN).is_unknown


def test_conflicting_assertions_are_contradiction():
    b = predicate_bounds([make_finding("x", Polarity.PRESENT), make_finding("x", Polarity.ABSENT)])
    assert b["x"].is_contradiction
    assert Bounds(0.0, 1.0).is_unknown
