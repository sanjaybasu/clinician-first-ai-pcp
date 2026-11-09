"""Next question or test by miss-weighted value of information (design doc, step 5).

EIG(a) = sum_k u_k [ h(p_k) - E_{y ~ P(y|a)} h(p_k | y) ]
P(y=1 | a) = 1 - (1 - l_a) prod_k (1 - p_k q_ka)          (noisy-OR)
a* = argmax_a [ EIG(a) - lambda_c C(a) - lambda_r R(a) ]

q_ka is the sensitivity of finding a for d_k, recovered from its LR+ and LR-.
Posteriors after each hypothetical result use the same log-odds update as
step 4 (point estimates; the draws are only needed for safety decisions).
Asking stops when no action has positive value, subject to the stopping rule.
"""

import math

from pydantic import BaseModel

from compass_pcp.engine.bayes import expit, h, logit
from compass_pcp.engine.scripts import IllnessScript, sens_spec
from compass_pcp.findings import Finding


class ScoredAction(BaseModel):
    feature_id: str
    p_positive: float
    eig: float
    cost: float
    risk: float
    score: float


def _updated(p: float, lr: float) -> float:
    return expit(logit(min(max(p, 1e-12), 1 - 1e-12)) + math.log(lr))


def score_actions(
    script: IllnessScript,
    findings: dict[str, Finding],
    current: dict[str, float],
    *,
    lambda_cost: float = 0.0,
    lambda_risk: float = 0.0,
    remote_only: bool = True,
) -> list[ScoredAction]:
    """``current`` maps diagnosis id -> current point probability p_k."""
    scored: list[ScoredAction] = []
    for feature in script.features:
        if feature.id in findings or (remote_only and not feature.remote_askable):
            continue
        users = [d for d in script.diagnoses if feature.id in d.lrs]
        if not users:
            continue
        prod = 1.0
        for d in users:
            q = sens_spec(d.lrs[feature.id])[0]
            prod *= 1 - current[d.id] * q
        p_pos = 1 - (1 - script.leak(feature.id)) * prod
        eig = 0.0
        for d in users:
            lr = d.lrs[feature.id]
            pk = current[d.id]
            after_pos, after_neg = _updated(pk, lr.lr_pos), _updated(pk, lr.lr_neg)
            eig += d.u * (h(pk) - (p_pos * h(after_pos) + (1 - p_pos) * h(after_neg)))
        score = eig - lambda_cost * feature.cost - lambda_risk * feature.harm
        scored.append(
            ScoredAction(feature_id=feature.id, p_positive=p_pos, eig=eig, cost=feature.cost, risk=feature.harm, score=score)
        )
    return sorted(scored, key=lambda s: (-s.score, s.feature_id))

