"""Interval estimates for the lane gates.

Gates are stated on 95% bounds, never point estimates: Wilson bounds for
proportions, the exact one-sided upper bound for zero events (rule of three),
and inverse-probability weighting for the two-phase adjudication sample.
"""

import itertools
import math
import random
from collections.abc import Callable, Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound when 0 of n events are seen (about 3/n)."""
    if n == 0:
        return 1.0
    return 1 - alpha ** (1 / n)


def cases_needed_for_zero_event_upper(target: float, alpha: float = 0.05) -> int:
    return math.ceil(math.log(alpha) / math.log(1 - target))


def per_1000(k: int, n: int) -> float:
    return 1000 * k / n if n else float("nan")


def weighted_proportion(outcomes: Sequence[bool], inclusion_prob: Sequence[float]) -> float:
    """Hájek estimator for a proportion under two-phase sampling."""
    num = sum(o / p for o, p in zip(outcomes, inclusion_prob))
    den = sum(1 / p for p in inclusion_prob)
    return num / den if den else float("nan")


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, seed: int = 0, stat: Callable[[Sequence[float]], float] | None = None
) -> tuple[float, float, float]:
    """Mean difference a - b with a 95% percentile interval, resampling paired units."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be the same non-zero length")
    stat = stat or (lambda xs: sum(xs) / len(xs))
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(stat([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return stat(diffs), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def roc_points(scores: Sequence[float], labels: Sequence[bool]) -> list[tuple[float, float, float]]:
    """(threshold, sensitivity, 1 - specificity) at every distinct score, flagging score >= threshold."""
    pos = sum(labels)
    neg = len(labels) - pos
    pts = [(float("inf"), 0.0, 0.0)]
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if not y and s >= t)
        pts.append((t, tp / pos if pos else float("nan"), fp / neg if neg else float("nan")))
    return pts


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC; ties count one half, so ordinal tiers are handled correctly."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def partial_auroc_high_sensitivity(scores: Sequence[float], labels: Sequence[bool], min_sens: float = 0.9) -> float:
    """Normalized area of the ROC curve above sensitivity = min_sens: integral of max(TPR - min_sens, 0) dFPR / (1 - min_sens).

    1.0 for perfect discrimination; (1 - min_sens) / 2 for chance. Linear interpolation between ROC points.
    """
    pts = sorted({(fpr, sens) for _, sens, fpr in roc_points(scores, labels)} | {(0.0, 0.0), (1.0, 1.0)})
    area = 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if x1 == x0:
            continue
        a0, a1 = y0 - min_sens, y1 - min_sens
        if a0 >= 0 and a1 >= 0:
            area += (x1 - x0) * (a0 + a1) / 2
        elif a0 < 0 <= a1 or a1 < 0 <= a0:
            xc = x0 + (x1 - x0) * (-a0) / (a1 - a0)
            area += (x1 - xc) * a1 / 2 if a1 >= 0 else (xc - x0) * a0 / 2
    return area / (1 - min_sens)


def specificity_at_sensitivity_floor(scores: Sequence[float], labels: Sequence[bool], floor: float) -> dict:
    """Highest-specificity operating point whose sensitivity point estimate meets the floor."""
    best = None
    for t, sens, fpr in roc_points(scores, labels):
        if sens >= floor and (best is None or 1 - fpr > best["specificity"]):
            best = {"threshold": t, "sensitivity": sens, "specificity": 1 - fpr}
    return best or {"threshold": None, "sensitivity": None, "specificity": None}


def net_benefit(tp: int, fp: int, n: int, threshold_prob: float) -> float:
    """Decision-curve net benefit: TP/n - FP/n * pt/(1 - pt)."""
    return tp / n - fp / n * threshold_prob / (1 - threshold_prob)


def f1_youden(tp: int, fp: int, fn: int, tn: int) -> dict:
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) and not math.isnan(ppv) else float("nan")
    return {"f1": f1, "youden_j": sens + spec - 1, "ppv": ppv}
