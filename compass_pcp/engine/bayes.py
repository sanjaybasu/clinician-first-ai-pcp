"""Sequential Bayesian update in log-odds (design doc, steps 3 and 4).

logit p_k = logit pi_k + s_k,   s_k = sum_j [1(x_j=1) ln LR+_kj + 1(x_j=0) ln LR-_kj]

Only features in the script for d_k count; unknown findings add nothing. Within
a correlated group only the finding with the largest |ln LR| counts.

Uncertainty: each ln LR is drawn 200 times from a normal distribution matching
its published 95% CI; the 5th and 95th percentiles of p_k are reported, and the
upper value p_k^95 is used for every safety decision. Draws use a seed derived
from the inputs, so a certificate replays exactly.

A patient-reported absence cannot by itself rule out a cannot-miss diagnosis:
for cannot-miss d_k, an ABSENT finding counts only if it came from the record or
the patient answered an explicit confirmation question.
"""

import hashlib
import math
import random

from pydantic import BaseModel

from compass_pcp.engine.scripts import Diagnosis, IllnessScript, LikelihoodRatio
from compass_pcp.findings import Finding, Polarity, Provenance

N_DRAWS = 200
Z95 = 1.959963984540054


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    if x < -700:
        return 0.0
    return 1 / (1 + math.exp(-x))


class AppliedLR(BaseModel):
    feature_id: str
    polarity: Polarity
    ln_lr: float
    ln_sd: float
    source: str


class Posterior(BaseModel):
    diagnosis_id: str
    p: float
    p05: float
    p95: float
    score: float  # s_k
    applied: list[AppliedLR]
    skipped: list[str]


def _ln_terms(lr: LikelihoodRatio, polarity: Polarity) -> tuple[float, float]:
    point = lr.lr_pos if polarity == Polarity.PRESENT else lr.lr_neg
    ci = lr.lr_pos_ci if polarity == Polarity.PRESENT else lr.lr_neg_ci
    sd = (math.log(ci[1]) - math.log(ci[0])) / (2 * Z95) if ci is not None else 0.0
    return math.log(point), sd


def counts_for_rule_out(f: Finding) -> bool:
    return f.provenance >= Provenance.PATIENT_CONFIRMED_RECORD or f.confirmed


def _seed(dx_id: str, applied: list[AppliedLR]) -> int:
    key = dx_id + "|" + "|".join(f"{a.feature_id}:{a.polarity}" for a in applied)
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def posterior(dx: Diagnosis, script: IllnessScript, findings: dict[str, Finding], n_draws: int = N_DRAWS) -> Posterior:
    candidates: dict[str, AppliedLR] = {}
    skipped: list[str] = []
    for feature_id, lr in dx.lrs.items():
        f = findings.get(feature_id)
        if f is None:
            continue
        if dx.cannot_miss and f.polarity == Polarity.ABSENT and not counts_for_rule_out(f):
            skipped.append(f"{feature_id}:unconfirmed_negative")
            continue
        ln, sd = _ln_terms(lr, f.polarity)
        applied = AppliedLR(feature_id=feature_id, polarity=f.polarity, ln_lr=ln, ln_sd=sd, source=lr.source)
        key = lr.group or feature_id
        current = candidates.get(key)
        if current is None or abs(ln) > abs(current.ln_lr):
            if current is not None:
                skipped.append(f"{current.feature_id}:weaker_in_group_{key}")
            candidates[key] = applied
        else:
            skipped.append(f"{feature_id}:weaker_in_group_{key}")

    applied_list = sorted(candidates.values(), key=lambda a: a.feature_id)
    base = logit(dx.prior.p)
    score = sum(a.ln_lr for a in applied_list)
    p = expit(base + score)
    if any(a.ln_sd > 0 for a in applied_list):
        rng = random.Random(_seed(dx.id, applied_list))
        draws = sorted(expit(base + sum(rng.gauss(a.ln_lr, a.ln_sd) for a in applied_list)) for _ in range(n_draws))
        p05, p95 = draws[int(0.05 * n_draws)], draws[int(0.95 * n_draws) - 1]
    else:
        p05 = p95 = p
    return Posterior(
        diagnosis_id=dx.id, p=p, p05=min(p05, p), p95=max(p95, p), score=score, applied=applied_list, skipped=sorted(skipped)
    )


def differential(script: IllnessScript, findings: dict[str, Finding]) -> dict[str, Posterior]:
    return {dx.id: posterior(dx, script, findings) for dx in script.diagnoses}


def h(p: float) -> float:
    """Binary entropy in nats."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))
