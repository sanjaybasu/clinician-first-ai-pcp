"""Blind adjudication packets, two-phase sampling, and agreement statistics.

Physicians label first, blind to all model output and to the 30-day outcome
(process standard). No model drafts a reference label. Gwet AC1 is the
pre-specified agreement statistic, with Fleiss kappa alongside, because kappa
collapses under high prevalence (CITE: kappa 0.33 at 98.8% unanimity).
"""

import random
from collections import Counter

from pydantic import BaseModel

from compass_pcp.harness.episode import Cohort, Episode
from compass_pcp.harness.runner import RunRecord, RunStatus


class BlindPacket(BaseModel):
    episode_id: str
    complaint: str
    opening_message: str
    context_snapshot: list[dict]
    record_facts: dict[str, dict]
    historical_decision_hidden: bool = True


def blind_packet(ep: Episode) -> BlindPacket:
    """Everything knowable at the time, nothing from the model, nothing from the outcome."""
    return BlindPacket(
        episode_id=ep.episode_id,
        complaint=ep.complaint,
        opening_message=ep.opening_message,
        context_snapshot=[f.model_dump(mode="json") for f in ep.context_snapshot],
        record_facts={k: v.model_dump(mode="json") for k, v in ep.record_facts.items()},
    )


class Selection(BaseModel):
    episode_id: str
    inclusion_prob: float
    reason: str


def select_for_adjudication(
    episodes: list[Episode], records: list[RunRecord], *, random_frac: float = 0.20, seed: int = 0
) -> list[Selection]:
    """All Cohort C positives, all disagreements with the clinician, all hard stops/deferrals, plus a random 20%."""
    first_run = {r.episode_id: r for r in records if r.run_index == 0}
    rng = random.Random(seed)
    out: list[Selection] = []
    for ep in sorted(episodes, key=lambda e: e.episode_id):
        rec = first_run.get(ep.episode_id)
        out_ = rec.output if rec else None
        if ep.cohort == Cohort.C_CANNOT_MISS and ep.outcome and ep.outcome.cannot_miss_diagnoses:
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=1.0, reason="cohort_c_positive"))
        elif rec is None or rec.status != RunStatus.OK:
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=1.0, reason="failed_run"))
        elif out_ is not None and (out_.orders or out_.prescriptions):
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=1.0, reason="has_order_or_prescription"))
        elif out_ is not None and (out_.decision in ("defer", "escalate") or out_.verifier_findings):
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=1.0, reason="hard_stop_or_deferral"))
        elif out_ is not None and ep.historical_decision is not None and out_.disposition != ep.historical_decision:
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=1.0, reason="disagrees_with_clinician"))
        elif rng.random() < random_frac:
            out.append(Selection(episode_id=ep.episode_id, inclusion_prob=random_frac, reason="random_sample"))
    return out


def _category_props(ratings: list[list[str]]) -> tuple[list[str], list[Counter], int]:
    cats = sorted({c for item in ratings for c in item})
    counts = [Counter(item) for item in ratings]
    r = len(ratings[0])
    if any(len(item) != r for item in ratings):
        raise ValueError("every item needs the same number of raters")
    return cats, counts, r


def _observed_agreement(counts: list[Counter], r: int) -> float:
    return sum(sum(n * (n - 1) for n in c.values()) / (r * (r - 1)) for c in counts) / len(counts)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    cats, counts, r = _category_props(ratings)
    pa = _observed_agreement(counts, r)
    pi = [sum(c[k] for c in counts) / (len(counts) * r) for k in cats]
    pe = sum(p * p for p in pi)
    return (pa - pe) / (1 - pe) if pe < 1 else 1.0


def gwet_ac1(ratings: list[list[str]]) -> float:
    cats, counts, r = _category_props(ratings)
    if len(cats) == 1:
        return 1.0
    pa = _observed_agreement(counts, r)
    pi = [sum(c[k] for c in counts) / (len(counts) * r) for k in cats]
    pe = sum(p * (1 - p) for p in pi) / (len(cats) - 1)
    return (pa - pe) / (1 - pe)
