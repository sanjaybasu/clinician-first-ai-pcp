"""Truth bounds with Łukasiewicz logic (LNN semantics, Riegel et al. 2020).

Each predicate carries [L, U] in [0, 1]. Unknown is [0, 1] and generates a
question. A contradiction (L > U + eps) means the parsed facts conflict and the
case defers. Weights are fixed at 1 in this version.
"""

from dataclasses import dataclass

from compass_pcp.findings import Finding, Polarity

EPS = 1e-9


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float

    @property
    def is_contradiction(self) -> bool:
        return self.lower > self.upper + EPS

    @property
    def is_true(self) -> bool:
        return self.lower >= 1 - EPS and not self.is_contradiction

    @property
    def is_false(self) -> bool:
        return self.upper <= EPS and not self.is_contradiction

    @property
    def is_unknown(self) -> bool:
        return not (self.is_true or self.is_false or self.is_contradiction)


TRUE = Bounds(1.0, 1.0)
FALSE = Bounds(0.0, 0.0)
UNKNOWN = Bounds(0.0, 1.0)


def and_(a: Bounds, b: Bounds) -> Bounds:
    return Bounds(max(0.0, a.lower + b.lower - 1), max(0.0, a.upper + b.upper - 1))


def or_(a: Bounds, b: Bounds) -> Bounds:
    return Bounds(min(1.0, a.lower + b.lower), min(1.0, a.upper + b.upper))


def not_(a: Bounds) -> Bounds:
    return Bounds(1 - a.upper, 1 - a.lower)


def implies(a: Bounds, b: Bounds) -> Bounds:
    return or_(not_(a), b)


def tighten(a: Bounds, b: Bounds) -> Bounds:
    """Combine two assertions about the same predicate. Conflict yields L > U."""
    return Bounds(max(a.lower, b.lower), min(a.upper, b.upper))


def all_of(items: list[Bounds]) -> Bounds:
    out = TRUE
    for item in items:
        out = and_(out, item)
    return out


def predicate_bounds(findings: list[Finding]) -> dict[str, Bounds]:
    """Turn findings into predicate bounds; repeated assertions are tightened."""
    out: dict[str, Bounds] = {}
    for f in findings:
        b = TRUE if f.polarity == Polarity.PRESENT else FALSE
        out[f.feature_id] = tighten(out[f.feature_id], b) if f.feature_id in out else b
    return out
