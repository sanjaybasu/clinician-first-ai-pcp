"""Cannot-miss stopping rule on the upper probability p_k^95 (design doc, step 6).

p95 < T_test             -> ruled out
T_test <= p95 < T_treat  -> keep working up; defer if no remote test remains
p95 >= T_treat           -> escalate (or treat if the lane allows it)
The encounter cannot close while any cannot-miss diagnosis is in the middle zone.
"""

from enum import StrEnum

from pydantic import BaseModel

from compass_pcp.engine.bayes import Posterior
from compass_pcp.engine.scripts import IllnessScript


class CannotMissStatus(StrEnum):
    RULED_OUT = "ruled_out"
    NEEDS_WORKUP = "needs_workup"
    ESCALATE = "escalate"


class CannotMissCheck(BaseModel):
    diagnosis_id: str
    status: CannotMissStatus
    p: float
    p95: float
    theta_test: float
    theta_treat: float
    remote_rule_out_possible: bool


def check_cannot_miss(
    script: IllnessScript, posteriors: dict[str, Posterior], observed: set[str]
) -> list[CannotMissCheck]:
    checks: list[CannotMissCheck] = []
    for dx in script.cannot_miss:
        assert dx.thresholds is not None and dx.thresholds.test is not None and dx.thresholds.treat is not None
        post = posteriors[dx.id]
        t_test, t_treat = dx.thresholds.test, dx.thresholds.treat
        if post.p95 >= t_treat:
            status = CannotMissStatus.ESCALATE
        elif post.p95 >= t_test:
            status = CannotMissStatus.NEEDS_WORKUP
        else:
            status = CannotMissStatus.RULED_OUT
        unasked = [fid for fid in dx.lrs if fid not in observed]
        checks.append(
            CannotMissCheck(
                diagnosis_id=dx.id,
                status=status,
                p=post.p,
                p95=post.p95,
                theta_test=t_test,
                theta_treat=t_treat,
                remote_rule_out_possible=any(script.feature(fid).remote_askable for fid in unasked),
            )
        )
    return checks
