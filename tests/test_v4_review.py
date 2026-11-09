"""COMPASS v4 B3: protected floors, the monotone review score and the nested grouped cross-validation fitter."""

import pytest

from compass_pcp.ledger import HazardTier
from compass_pcp.review_score import (
    ReviewScoreParams,
    fit_review_score,
    load_review_params,
    protected_floors,
    review_decision,
    review_score,
)

P = load_review_params()


def floors(**kw):
    base = {"route": "none", "applicable_uncertainty": False, "engine_escalation": False,
            "screen_tier": HazardTier.NONE, "floor_tier": P.floor_screen_tier}
    return protected_floors(**{**base, **kw})


def test_placeholder_params_are_marked_and_monotone():
    assert P.status == "placeholder_unfitted" and len(P.features) <= 3
    with pytest.raises(ValueError):
        ReviewScoreParams(version="x", status="fitted", source="test", features=["a"], intercept=0,
                          coefficients={"a": -1.0}, threshold=0.5)
    with pytest.raises(ValueError):
        ReviewScoreParams(version="x", status="fitted", source="test", features=["a", "b", "c", "d"], intercept=0,
                          coefficients={"a": 1, "b": 1, "c": 1, "d": 1}, threshold=0.5)


def test_each_floor_flags_regardless_of_score():
    zero = {"anthropic_tier": 0.0, "kev_urgent": 0.0, "classifier_max": 0.0}
    assert not review_decision(floors(), zero, P).flag
    for kw, name in [({"route": "urgent_review"}, "route:urgent_review"),
                     ({"applicable_uncertainty": True}, "applicable_uncertainty"),
                     ({"engine_escalation": True}, "engine_escalation"),
                     ({"screen_tier": HazardTier.POSSIBLE_EMERGENCY}, "screen_tier:POSSIBLE_EMERGENCY")]:
        d = review_decision(floors(**kw), zero, P)
        assert d.flag and d.floors == [name] and d.score < d.threshold
    assert floors(screen_tier=HazardTier.URGENT_SAME_DAY) == []


def test_score_is_monotone_in_each_feature():
    lo = review_score({"anthropic_tier": 0.25, "kev_urgent": 0.1}, P)
    assert review_score({"anthropic_tier": 0.5, "kev_urgent": 0.1}, P) >= lo
    assert review_score({"anthropic_tier": 0.25, "kev_urgent": 0.9}, P) >= lo
    assert review_decision([], {"anthropic_tier": 1.0, "kev_urgent": 1.0}, P).flag


def _synthetic(n_groups=120, seed=0):
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    import numpy as np

    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for grp in range(n_groups):
        for _ in range(rng.integers(1, 4)):
            pos = rng.random() < 0.15
            a = np.clip(rng.normal(0.7 if pos else 0.25, 0.15), 0, 1)
            k = np.clip(rng.normal(0.6 if pos else 0.3, 0.2), 0, 1)
            noise = rng.random()
            X.append([a, k, noise])
            y.append(int(pos))
            g.append(grp)
    return np.array(X), np.array(y), np.array(g)


def test_nested_grouped_cv_selects_inside_and_evaluates_outside():
    import numpy as np

    X, y, g = _synthetic()
    fl = np.zeros(len(y), dtype=bool)
    fl[:5] = True
    rep = fit_review_score(X, y, g, feature_names=["anthropic", "kev", "noise"], floors=fl, cap_per_1000=480,
                           Cs=(0.1, 1.0, 10.0))
    assert len(rep["outer"]) == 5 and all(r["feasible"] for r in rep["outer"])
    assert rep["pooled"]["sensitivity"] >= 0.8 and rep["pooled"]["flags_per_1000"] <= 600
    assert sum(r["n"] for r in rep["outer"]) == len(y)
    fin = rep["final"]
    assert 1 <= len(fin["features"]) <= 3 and all(v >= 0 for v in fin["coefficients"].values())
    assert set(fin["features"]) & {"anthropic", "kev"}
    assert not rep["floors_alone_exceed_cap"]


def test_folds_never_split_a_group():
    import numpy as np

    from compass_pcp.review_score import _folds

    _, y, g = _synthetic(seed=1)
    for tr, te in _folds(5, y, g, 0):
        assert not set(g[tr]) & set(g[te])
    assert len(np.unique(g)) > 5


def test_exclusive_features_are_never_chosen_together():
    import numpy as np

    X, y, g = _synthetic(seed=2)
    X2 = np.column_stack([X[:, 0], X[:, 0] * 0.9 + 0.05, X[:, 1]])  # single call vs K = 3 vote fraction
    rep = fit_review_score(X2, y, g, feature_names=["single", "k3", "kev"], exclusive=[{"single", "k3"}],
                           Cs=(1.0,), outer_folds=3, inner_folds=3)
    for r in [*rep["outer"], rep["final"]]:
        assert not {"single", "k3"} <= set(r["features"])


def test_floors_above_the_cap_are_reported_infeasible():
    import numpy as np

    X, y, g = _synthetic(seed=3)
    fl = np.ones(len(y), dtype=bool)
    rep = fit_review_score(X, y, g, feature_names=["a", "k", "n"], floors=fl, Cs=(1.0,), outer_folds=3)
    assert rep["floors_alone_exceed_cap"] and rep["final"] is None
    assert all(not r["feasible"] for r in rep["outer"])


def test_inputs_must_line_up():
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError):
        fit_review_score(np.zeros((4, 2)), [0, 1, 0, 1], [1, 2, 3, 4], feature_names=["a"])
