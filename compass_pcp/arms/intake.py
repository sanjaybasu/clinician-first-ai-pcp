"""Structured-state SMS intake agent (COMPASS AI PCP, Paper 2 development; 2026-11-08).

The agent keeps an explicit intake state. Past medical history and medicines are prefilled from the chart the care team
could see before the message, so the agent confirms them instead of asking from zero. The state tracks the required
fields below; the agent asks at most two questions per message and stops when every required field is filled (a value,
"denies", "declined" or "unknown" all count) or at the turn limit, then a separate note writer drafts the note.

Clinician-first policy (S. Basu, 2026-11-08): the agent never tells the patient to go to an emergency room or urgent
care, or to call 911/988, as its own decision. A possible emergency ends the intake with the urgent-page text
(compass_pcp/data/policies/urgent_page_v23_dev.yaml, draft) and an emergency disposition proposed to the physician.
Patient-facing messages never carry drug doses.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from compass_pcp.arms.llm import LLM, parse_json

HPI_ELEMENTS = ("onset", "location", "duration", "character", "aggravating_relieving", "timing", "severity",
                "associated_symptoms")
REQUIRED = (*(f"hpi.{e}" for e in HPI_ELEMENTS), "ros", "pmh_confirmed", "meds_confirmed", "med_adherence", "allergies",
            "family_history", "social_history", "social_needs")
MAX_TURNS = 8

URGENT_PAGE_SELF = (
    "This might be an urgent situation. I am sending your message to our physician now, and they will call you back. "
    "If things get worse, call 911. If you are thinking about hurting yourself, call or text 988. This text line is "
    "not an emergency service."
)
ALLOWED_ROUTE_TEXT = (URGENT_PAGE_SELF,)
CLOSING = ("Thank you, that is everything I need for now. I am sending your answers to your doctor, who will review "
           "them and contact you about next steps.")

DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|units?|iu|tabs?|tablets?|puffs?|drops?)\b", re.I)

SYSTEM = """You run a text-message (SMS) intake for a primary care team that serves Medicaid patients. A physician
reviews every intake and decides care; your job is to collect the history the physician needs, quickly and kindly.

You hold an INTAKE STATE (below). Each turn:
1. Update the state from everything the patient has said so far. Fill a field only with what the patient said or
   confirmed; write "denies", "declined" or "unknown" when the patient said so or cannot say. Chart-prefilled past
   medical history and medicines are not confirmed until the patient confirms them.
2. Decide if this could be an emergency now (for example chest pain with shortness of breath, stroke signs, thoughts of
   self-harm, severe bleeding, trouble breathing). If so set "emergency_concern": true; the system will page the
   physician. Never tell the patient yourself to go to an emergency room, urgent care or hospital, or to call 911 or
   988.
3. Otherwise write the next message: ask at most TWO questions, about the most important empty fields, in this order of
   priority: HPI elements and red-flag review of systems for the complaint; confirmation of the chart conditions and
   medicines (name them from the chart in plain words and ask if they are still right) and whether they take them as
   prescribed; allergies; family history; social history (tobacco, alcohol, drugs, living situation, work); social needs
   (food, housing, transport, money for medicines, safety). You may combine closely related items in one question.
   Do not ask about anything already filled. Never give medicine doses or prescribe. No diagnosis to the patient.
   SMS style: plain text, no markdown, at most 400 characters, about a 6th-grade reading level, in the patient's
   language.
4. Set "complete": true when every required field is filled.

Reply JSON only:
{"state": {"hpi": {"onset": ..., "location": ..., "duration": ..., "character": ..., "aggravating_relieving": ...,
 "timing": ..., "severity": ..., "associated_symptoms": ...}, "ros": ..., "pmh_confirmed": ..., "pmh_new": [...],
 "meds_confirmed": ..., "med_adherence": ..., "allergies": ..., "family_history": ..., "social_history": ...,
 "social_needs": ...},
 "emergency_concern": true|false, "complete": true|false, "message": "..."}
Use null for an empty field."""


def chart_lists(chart: str) -> tuple[list[str], list[str]]:
    """Conditions and medicines from the chart lines written by paper/chart_context.py."""
    cond, meds = [], []
    for line in chart.splitlines():
        if line.startswith("Diagnoses on claims:"):
            cond = [x.strip().rstrip(".") for x in line.split(":", 1)[1].split(";") if x.strip()]
        elif line.startswith("Medicines filled in the past 8 months:"):
            meds = [x.strip().rstrip(".") for x in line.split(":", 1)[1].split(";") if x.strip()]
    return cond, meds


def empty_state(chart: str) -> dict:
    cond, meds = chart_lists(chart)
    return {"hpi": {e: None for e in HPI_ELEMENTS}, "ros": None,
            "chart_pmh_unconfirmed": cond, "pmh_confirmed": None, "pmh_new": [],
            "chart_meds_unconfirmed": meds, "meds_confirmed": None, "med_adherence": None, "allergies": None,
            "family_history": None, "social_history": None, "social_needs": None}


def missing(state: dict) -> list[str]:
    out = []
    for k in REQUIRED:
        v = state.get("hpi", {}).get(k[4:]) if k.startswith("hpi.") else state.get(k)
        if v in (None, "", [], {}):
            out.append(k)
    return out


@dataclass
class StructuredIntake:
    backbone: LLM
    chart: str
    task: str = ""
    state: dict = field(default_factory=dict)
    turn: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.state:
            self.state = empty_state(self.chart)

    def _call(self, messages, extra: str = "") -> dict:
        system = (SYSTEM + (f"\n\nTASK: {self.task}" if self.task else "")
                  + f"\n\nCHART (available to the care team before this conversation):\n{self.chart}"
                  + f"\n\nINTAKE STATE:\n{json.dumps(self.state)}" + extra)
        for _ in range(2):
            try:
                return parse_json(self.backbone.complete(system, messages, max_tokens=1500))
            except (ValueError, json.JSONDecodeError):
                continue
        return {}

    def next_message(self, messages, *, configuration=None, context=None) -> tuple[str, dict[str, Any]]:
        from paper.disposition_check import check  # scoring helper shared with the evaluation

        self.turn += 1
        out = self._call(messages)
        meta: dict[str, Any] = {"arm": "intake", "turn": self.turn, "parsed": bool(out), "repairs": []}
        new = out.get("state")
        if isinstance(new, dict):
            keep = {k: self.state[k] for k in ("chart_pmh_unconfirmed", "chart_meds_unconfirmed")}
            self.state = {**self.state, **new, **keep}
            if not isinstance(self.state.get("hpi"), dict):
                self.state["hpi"] = {e: None for e in HPI_ELEMENTS}
        gaps = missing(self.state)
        meta["missing"] = len(gaps)
        if out.get("emergency_concern"):
            meta.update(done=True, emergency=True)
            return URGENT_PAGE_SELF, meta
        if not gaps or self.turn >= MAX_TURNS:
            meta.update(done=True, emergency=False, stop="complete" if not gaps else "turn_limit")
            return CLOSING, meta
        msg = str(out.get("message") or "").strip()
        if not msg:
            meta["repairs"].append("empty")
            msg = "Can you tell me more about what is going on, and when it started?"
        fix = []
        if msg.count("?") > 2:
            fix.append("Ask at most two questions.")
        if DOSE.search(msg):
            fix.append("Do not mention any medicine dose or amount.")
        if check(msg, ALLOWED_ROUTE_TEXT):
            fix.append("Do not tell the patient to go to an emergency room, urgent care or hospital, or to call 911 "
                       "or 988; if this could be an emergency, set emergency_concern true.")
        if fix:
            meta["repairs"].append(" ".join(fix))
            again = self._call(messages, "\n\nYOUR LAST DRAFT BROKE A RULE: " + " ".join(fix) + " Write it again.")
            if again.get("emergency_concern"):
                meta.update(done=True, emergency=True)
                return URGENT_PAGE_SELF, meta
            msg2 = str(again.get("message") or "").strip()
            if msg2:
                msg = msg2
            if check(msg, ALLOWED_ROUTE_TEXT):  # still an instruction: hold it and page the physician instead
                meta.update(done=True, emergency=True, held=True)
                return URGENT_PAGE_SELF, meta
        meta["done"] = False
        return msg, meta
