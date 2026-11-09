"""COMPASS v4 review flag (v4_spec.md B3): protected floors plus a monotone logistic review score.

Flag = any protected floor, or score >= threshold. The floors cannot be downgraded: a B1 route other than none, an
applicable uncertainty or incomplete check, an engine escalation, or a screen tier at or above the policy's floor tier.
The score uses at most 3 predictors with nonnegative coefficients loaded from ``data/policies/review_score_v4.yaml``.
It sets the review flag only, never urgency or the reply.

``fit_review_score`` fits the score by nested grouped cross-validation (5 outer, 3 inner folds by default): the inner
folds choose the feature subset, the regularization C and the threshold (highest recall with total flags, floors
included, at or below the cap per 1,000; ties to fewer flags, then fewer features, then stronger regularization); the
outer folds only evaluate that whole procedure. It needs numpy, scipy and scikit-learn (the ``review`` extra).
"""

import itertools
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from compass_pcp.ledger import HazardTier
from compass_pcp.policy_v4 import POLICY_DIR

MAX_FEATURES = 3


class ReviewScoreParams(BaseModel):
    version: str
    status: Literal["placeholder_unfitted", "fitted"]
    source: str = Field(min_length=3)
    features: list[str] = Field(min_length=1, max_length=MAX_FEATURES)
    intercept: float
    coefficients: dict[str, float]
    threshold: float = Field(gt=0, lt=1)
    floor_screen_tier: str = "POSSIBLE_EMERGENCY"
    cap_per_1000: float = Field(default=480, gt=0)

    @model_validator(mode="after")
    def _monotone(self) -> "ReviewScoreParams":
        if set(self.coefficients) != set(self.features):
            raise ValueError("coefficients must name exactly the listed features")
        if any(v < 0 for v in self.coefficients.values()):
            raise ValueError("review-score coefficients must be nonnegative (monotone score)")
        HazardTier[self.floor_screen_tier]
        return self


def load_review_params(path: Path | None = None) -> ReviewScoreParams:
    return _load(str(path or POLICY_DIR / "review_score_v4.yaml"))


@lru_cache(maxsize=4)
def _load(path: str) -> ReviewScoreParams:
    return ReviewScoreParams.model_validate(yaml.safe_load(Path(path).read_text()))


class ReviewDecision(BaseModel):
    flag: bool
    floors: list[str]
    score: float
    threshold: float
    features: dict[str, float]


def protected_floors(*, route: str, applicable_uncertainty: bool, engine_escalation: bool, screen_tier: HazardTier,
                     floor_tier: str) -> list[str]:
    out = []
    if route != "none":
        out.append(f"route:{route}")
    if applicable_uncertainty:
        out.append("applicable_uncertainty")
    if engine_escalation:
        out.append("engine_escalation")
    if screen_tier >= HazardTier[floor_tier]:
        out.append(f"screen_tier:{HazardTier(screen_tier).name}")
    return out


def review_score(features: dict[str, float], params: ReviewScoreParams) -> float:
    z = params.intercept + sum(params.coefficients[f] * float(features.get(f, 0.0)) for f in params.features)
    return 1.0 / (1.0 + math.exp(-z))


def review_decision(floors: list[str], features: dict[str, float], params: ReviewScoreParams) -> ReviewDecision:
    s = review_score(features, params)
    return ReviewDecision(flag=bool(floors) or s >= params.threshold, floors=floors, score=s,
                          threshold=params.threshold, features={f: float(features.get(f, 0.0)) for f in params.features})


# ---- fitting ---------------------------------------------------------------------------------------------------------
def _fit_nonneg_logistic(X, y, C: float):
    """L2-regularized logistic regression with nonnegative slopes (free intercept), by bounded L-BFGS."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    _, p = X.shape

    def f(theta):
        b, w = theta[0], theta[1:]
        z = b + X @ w
        loss = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.dot(w, w) / C
        r = expit(z) - y
        return loss, np.concatenate([[r.sum()], X.T @ r + w / C])

    res = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0.0, None)] * p)
    return float(res.x[0]), np.asarray(res.x[1:], dtype=float)


def _predict(X, b, w):
    from scipy.special import expit

    return expit(b + X @ w)


def _select_threshold(scores, y, floors, cap_per_1000: float):
    """Highest-recall threshold whose total flags (floors included) are at or below the cap; ties to fewer flags.
    Returns (threshold, recall, flags) or None if no threshold is feasible."""
    import numpy as np

    n, pos = len(y), max(int(y.sum()), 1)
    best = None
    for t in [*np.unique(scores)[::-1], np.inf]:
        flags = floors | (scores >= t)
        nf = int(flags.sum())
        if 1000.0 * nf / n > cap_per_1000:
            continue
        rec = float((flags & (y == 1)).sum()) / pos
        key = (rec, -nf)
        if best is None or key > best[0]:
            best = (key, float(t), rec, nf)
    return None if best is None else best[1:]


def _folds(n_splits: int, y, groups, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        y.reshape(-1, 1), y, groups))


def _inner_select(X, y, groups, floors, *, subsets, Cs, inner_folds, cap_per_1000, seed):
    """Choose (feature subset, C, threshold) by inner cross-validated predictions. None if nothing is feasible."""
    import numpy as np

    folds = _folds(inner_folds, y, groups, seed)
    best = None
    for subset in subsets:
        cols = list(subset)
        for C in Cs:
            oof = np.zeros(len(y))
            for tr, va in folds:
                b, w = _fit_nonneg_logistic(X[tr][:, cols], y[tr], C)
                oof[va] = _predict(X[va][:, cols], b, w)
            sel = _select_threshold(oof, y, floors, cap_per_1000)
            if sel is None:
                continue
            t, rec, nf = sel
            key = (rec, -nf, -len(cols), -C)
            if best is None or key > best["key"]:
                best = {"key": key, "cols": cols, "C": C, "threshold": t, "inner_recall": rec,
                        "inner_flags_per_1000": 1000.0 * nf / len(y)}
    return best


def _subsets(names: list[str], max_features: int, exclusive) -> list[tuple[int, ...]]:
    excl = [set(e) for e in exclusive]
    out = []
    for k in range(1, min(max_features, MAX_FEATURES) + 1):
        for combo in itertools.combinations(range(len(names)), k):
            chosen = {names[i] for i in combo}
            if any(len(chosen & e) > 1 for e in excl):
                continue
            out.append(combo)
    return out


def fit_review_score(X, y, groups, *, feature_names: list[str], floors=None, cap_per_1000: float = 480.0,
                     Cs=(0.01, 0.1, 1.0, 10.0), outer_folds: int = 5, inner_folds: int = 3,
                     max_features: int = MAX_FEATURES, exclusive=(), seed: int = 0) -> dict:
    """Nested grouped cross-validation of the B3 review score.

    X: (n, k) predictors in [0, 1]; y: 0/1 safety-positive labels; groups: patient or thread ids (never split across
    folds); floors: boolean protected-floor flags per message (counted in every flag total and never removed);
    exclusive: feature-name sets of which at most one may be chosen (for example the single-call Anthropic score and
    the K = 3 vote fraction). Returns per-outer-fold evaluation, pooled outer results and the procedure refit on all
    data (``final``), whose coefficients and threshold are what a frozen ``review_score_v4.yaml`` records.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    floors = np.zeros(len(y), dtype=bool) if floors is None else np.asarray(floors, dtype=bool)
    if X.ndim != 2 or X.shape[1] != len(feature_names) or not (len(y) == len(groups) == len(floors) == X.shape[0]):
        raise ValueError("X, y, groups, floors and feature_names do not line up")
    subsets = _subsets(list(feature_names), max_features, exclusive)
    floors_per_1000 = 1000.0 * floors.sum() / len(y)
    report: dict = {"n": len(y), "positives": int(y.sum()), "cap_per_1000": cap_per_1000,
                    "floors_per_1000": floors_per_1000, "floors_alone_exceed_cap": floors_per_1000 > cap_per_1000,
                    "outer": []}
    caught = flagged = positives = 0
    for i, (tr, te) in enumerate(_folds(outer_folds, y, groups, seed)):
        sel = _inner_select(X[tr], y[tr], groups[tr], floors[tr], subsets=subsets, Cs=Cs, inner_folds=inner_folds,
                            cap_per_1000=cap_per_1000, seed=seed + 1 + i)
        if sel is None:
            report["outer"].append({"fold": i, "feasible": False, "n": len(te), "positives": int(y[te].sum())})
            continue
        cols = sel["cols"]
        b, w = _fit_nonneg_logistic(X[tr][:, cols], y[tr], sel["C"])
        flags = floors[te] | (_predict(X[te][:, cols], b, w) >= sel["threshold"])
        c, p, nf = int((flags & (y[te] == 1)).sum()), int(y[te].sum()), int(flags.sum())
        caught, positives, flagged = caught + c, positives + p, flagged + nf
        report["outer"].append({
            "fold": i, "feasible": True, "features": [feature_names[j] for j in cols], "C": sel["C"],
            "threshold": sel["threshold"], "n": len(te), "positives": p, "caught": c,
            "flags_per_1000": 1000.0 * nf / len(te),
        })
    evaluated = sum(r["n"] for r in report["outer"] if r["feasible"])
    report["pooled"] = {"caught": caught, "positives": positives,
                        "sensitivity": caught / positives if positives else None,
                        "flags_per_1000": 1000.0 * flagged / evaluated if evaluated else None,
                        "infeasible_folds": sum(not r["feasible"] for r in report["outer"])}
    sel = _inner_select(X, y, groups, floors, subsets=subsets, Cs=Cs, inner_folds=inner_folds,
                        cap_per_1000=cap_per_1000, seed=seed)
    if sel is None:
        report["final"] = None
    else:
        cols = sel["cols"]
        b, w = _fit_nonneg_logistic(X[:, cols], y, sel["C"])
        report["final"] = {"features": [feature_names[j] for j in cols], "intercept": b,
                           "coefficients": {feature_names[j]: float(v) for j, v in zip(cols, w)},
                           "C": sel["C"], "threshold": sel["threshold"]}
    return report
