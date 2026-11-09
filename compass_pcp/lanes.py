"""Autonomy lanes and rungs.

Rung 0: AI drafts, physician writes the plan.
Rung 1: physician signs every item.
Rung 2: AI acts, physician reviews in batch within 24 hours.
Rung 3: AI acts, physician reviews an audit sample.

A lane moves up a rung only when every gate for that rung clears in the
retrospective validation. The current rung per lane is configuration, not code,
so that moving a lane is a reviewed config change.
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel


class Lane(StrEnum):
    REFILL = "refill"
    DYSURIA = "dysuria"
    URI = "uri"
    CONJUNCTIVITIS = "conjunctivitis"
    CONTRACEPTION = "contraception"
    HTN_TITRATION = "htn_titration"
    DM_TITRATION = "dm_titration"
    UNDIFFERENTIATED_HIGH_RISK = "undifferentiated_high_risk"


class Rung(IntEnum):
    DRAFT_ONLY = 0
    SIGN_EVERY_ITEM = 1
    BATCH_REVIEW = 2
    AUDIT_SAMPLE = 3


class ActionKind(StrEnum):
    ASK = "ask"
    PATIENT_MESSAGE = "patient_message"
    PRESCRIBE_RENEWAL = "prescribe_renewal"
    PRESCRIBE_NEW = "prescribe_new"
    ORDER_LAB = "order_lab"
    DISPOSITION = "disposition"


# Actions a lane may take at all, at any rung. Everything else defers.
LANE_ACTIONS: dict[Lane, frozenset[ActionKind]] = {
    Lane.REFILL: frozenset({ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_RENEWAL}),
    Lane.DYSURIA: frozenset({ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_NEW}),
    Lane.URI: frozenset({ActionKind.ASK, ActionKind.PATIENT_MESSAGE}),
    Lane.CONJUNCTIVITIS: frozenset({ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_NEW}),
    Lane.CONTRACEPTION: frozenset({ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_RENEWAL}),
    Lane.HTN_TITRATION: frozenset(
        {ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_RENEWAL, ActionKind.ORDER_LAB}
    ),
    Lane.DM_TITRATION: frozenset(
        {ActionKind.ASK, ActionKind.PATIENT_MESSAGE, ActionKind.PRESCRIBE_RENEWAL, ActionKind.ORDER_LAB}
    ),
    Lane.UNDIFFERENTIATED_HIGH_RISK: frozenset({ActionKind.ASK, ActionKind.DISPOSITION}),
}


class LaneConfig(BaseModel):
    lane: Lane
    rung: Rung


def permitted(config: LaneConfig, action: ActionKind) -> bool:
    return action in LANE_ACTIONS[config.lane]


def needs_physician_signature(config: LaneConfig) -> bool:
    return config.rung <= Rung.SIGN_EVERY_ITEM


def autonomous(config: LaneConfig) -> bool:
    return config.rung >= Rung.BATCH_REVIEW


# Lanes in which the AI may never close an encounter: it takes the history and escalates or hands off (design doc 2.5).
NO_REMOTE_CLOSE: frozenset[str] = frozenset({Lane.UNDIFFERENTIATED_HIGH_RISK.value})


def may_close(lane: str) -> bool:
    return lane not in NO_REMOTE_CLOSE
