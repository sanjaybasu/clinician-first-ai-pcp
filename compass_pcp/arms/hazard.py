"""Semantic hazard screen (panel v1, item 1).

One LLM component reads the patient's own words across the whole conversation
and returns a tiered urgency with a verbatim quote. It is raise-only: it can make
the system more urgent, never less, and a deterministic mapping turns the tier
into the response. It runs alongside the lexical floors (keyword, BE-FAST,
ANCHOR), because on real Medicaid SMS those caught 0 of 10 emergencies.

The quote is checked against the patient's text. An unverifiable quote keeps the
tier (raise-only) and is flagged for review, so a paraphrase never silently
lowers urgency.
"""

import json
import re

from compass_pcp.arms.llm import LLM, parse_json
from compass_pcp.ledger import HazardEvent, HazardTier

SCREEN = """You screen patient messages to a primary care team for safety hazards that need a clinician.
Read ONLY what the patient wrote (lines starting PATIENT). Hazards are often implicit and buried inside other requests:
a past or recent ER visit, being turned away from detox, passive wishes to be dead, someone else's suicide weighing on them,
chest or belly pain mentioned in passing, running out of insulin, new confusion, falls, abuse or unsafe housing,
pregnancy warning signs, medication misuse or dangerous combinations, a child or older adult at risk.
Choose the single highest tier that applies:
- emergency_now: the patient describes clear emergency features now (for example chest pain with sweating or shortness of
  breath, face droop or one-sided weakness, severe trouble breathing, heavy bleeding, fainting with chest pain, active
  suicidal intent or plan, overdose)
- possible_emergency: the concern could be an emergency depending on one or two facts not yet known (for example new
  shortness of breath without other details, worst headache without details, abdominal pain in someone who may be pregnant)
- urgent_same_day: needs a clinician today (including passive suicidal thoughts, new concerning symptoms, out of a critical medication)
- routine_followup: a clinical issue a clinician should review in the next days
- none: no clinical hazard (purely logistics, benefits, scheduling, or a question from a health professional about someone else)
Between routine_followup and urgent_same_day, choose the higher when unsure. Use emergency_now only for clear features.
Your tier alerts the care team; it does not decide the reply, so do not lower a tier to keep the conversation smooth.
Reply JSON only: {"tier": "...", "category": "short_snake_case", "quote": "exact words copied from the patient", "rationale": "one sentence"}"""

_TIER = {
    "none": HazardTier.NONE,
    "routine_followup": HazardTier.ROUTINE_FOLLOWUP,
    "urgent_same_day": HazardTier.URGENT_SAME_DAY,
    "possible_emergency": HazardTier.POSSIBLE_EMERGENCY,
    "emergency_now": HazardTier.EMERGENCY_NOW,
}


def patient_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"PATIENT: {m['content']}" for m in messages if m["role"] == "user")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.lower())).strip()


def screen(llm: LLM, messages: list[dict[str, str]], turn: int) -> HazardEvent:
    text = patient_text(messages)
    try:
        out = parse_json(llm.complete(SCREEN, [{"role": "user", "content": text or "(no patient text)"}], max_tokens=300))
        tier = _TIER.get(str(out.get("tier", "")).strip(), HazardTier.URGENT_SAME_DAY)
        quote = str(out.get("quote", ""))
        category, rationale = str(out.get("category", "unspecified")), str(out.get("rationale", ""))
    except ValueError:
        # A screen that cannot be read is treated as a positive that needs a human, never as "none".
        return HazardEvent(turn=turn, tier=HazardTier.URGENT_SAME_DAY, category="screen_unparseable", quote="", quote_verified=False)
    verified = bool(quote) and _norm(quote) in _norm(text)
    return HazardEvent(turn=turn, tier=tier, category=category, quote=quote, quote_verified=verified, rationale=rationale)


KEV_TIERS = {
    "none": "No clinical hazard: logistics, benefits, scheduling, or general questions with no concerning symptom",
    "routine_followup": "A clinical issue a clinician should review in the next few days",
    "urgent_same_day": "Needs a clinician today: new concerning symptom, passive suicidal thoughts, out of a critical medication",
    "possible_emergency": "Could be an emergency depending on one or two facts not yet known",
    "emergency_now": "Clear emergency features now, or active suicidal intent or plan",
}
KEV_INSTRUCTIONS = (
    "Read only what the patient wrote. Choose the highest safety tier that applies. Hazards are often implicit and buried "
    "inside other requests (a recent ER visit, detox turned away, passive wishes to be dead, chest pain mentioned in passing)."
)


def kev_tier(probs: dict[str, float], *, urgent_threshold: float = 0.15) -> HazardTier:
    """The highest tier whose cumulative probability (that tier or above) reaches its threshold, so low but real risk
    still escalates."""
    order = ["none", "routine_followup", "urgent_same_day", "possible_emergency", "emergency_now"]
    cum = {t: sum(probs.get(u, 0.0) for u in order[i:]) for i, t in enumerate(order)}
    tier = HazardTier.NONE
    if cum["routine_followup"] >= 0.5:
        tier = HazardTier.ROUTINE_FOLLOWUP
    if cum["urgent_same_day"] >= urgent_threshold:
        tier = HazardTier.URGENT_SAME_DAY
    if cum["possible_emergency"] >= 0.5:
        tier = HazardTier.POSSIBLE_EMERGENCY
    if probs.get("emergency_now", 0.0) >= 0.5:
        tier = HazardTier.EMERGENCY_NOW
    return tier


def kev_screen(judge, messages: list[dict[str, str]], turn: int, *, urgent_threshold: float = 0.15) -> HazardEvent:
    """System-1 hazard tier from a local calibrated judge (raise-only; see ``kev_tier``)."""
    probs = judge.choice(patient_text(messages), KEV_INSTRUCTIONS, KEV_TIERS)
    tier = kev_tier(probs, urgent_threshold=urgent_threshold)
    return HazardEvent(turn=turn, tier=tier, category="kev", quote="", quote_verified=False,
                       rationale=json.dumps({k: round(v, 4) for k, v in probs.items()}))
