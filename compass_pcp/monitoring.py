"""Monitoring contracts with reopen-on-breach.

Every order or prescription carries a contract, for example "BMP 7 to 14 days
after starting an ACE inhibitor; breach if potassium rises by 0.5 mEq/L or more
or creatinine rises by 30% or more". A breached or overdue contract reopens the
case in the workup layer.
"""

from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field


class Check(BaseModel):
    """Expected response band for one measure: breach on a rise, a fall, or leaving [low, high]."""

    loinc: str
    max_abs_rise: float | None = None
    max_rel_rise: float | None = Field(default=None, description="0.30 means a 30% rise")
    max_abs_fall: float | None = None
    low: float | None = Field(default=None, description="Absolute floor, e.g. potassium 3.5")
    high: float | None = Field(default=None, description="Absolute ceiling, e.g. potassium 5.5")


class MonitoringContract(BaseModel):
    id: str
    trigger: str = Field(description="Medication class or order that starts the contract")
    window_days: tuple[int, int]
    checks: list[Check]
    source: str = Field(min_length=3)


class ContractStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    BREACH = "breach"
    OVERDUE = "overdue"


class LabValue(BaseModel):
    loinc: str
    value: float
    on: date


def evaluate_contract(
    contract: MonitoringContract,
    started: date,
    baseline: dict[str, float],
    results: list[LabValue],
    today: date,
) -> ContractStatus:
    hi = started + timedelta(days=contract.window_days[1])
    in_window = [r for r in results if r.on >= started]
    status = ContractStatus.OK
    for check in contract.checks:
        latest = max((r for r in in_window if r.loinc == check.loinc), key=lambda r: r.on, default=None)
        if latest is None:
            if today > hi:
                return ContractStatus.OVERDUE
            status = ContractStatus.PENDING
            continue
        if (check.low is not None and latest.value < check.low) or (check.high is not None and latest.value > check.high):
            return ContractStatus.BREACH
        base = baseline.get(check.loinc)
        if base is None:
            continue
        if check.max_abs_fall is not None and base - latest.value >= check.max_abs_fall:
            return ContractStatus.BREACH
        if check.max_abs_rise is not None and latest.value - base >= check.max_abs_rise:
            return ContractStatus.BREACH
        if check.max_rel_rise is not None and base > 0 and (latest.value - base) / base >= check.max_rel_rise:
            return ContractStatus.BREACH
    return status


def reopens_case(status: ContractStatus) -> bool:
    return status in (ContractStatus.BREACH, ContractStatus.OVERDUE)
