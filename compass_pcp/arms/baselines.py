"""External comparator arms on a shared backbone.

Each arm returns the next patient-visible message. All arms share the SMS
output limits so differences come from structure. These are reimplementations
from published descriptions; docs/regulatory/FID-PCP-001 records how each follows its source.
Every arm, including the AI PCP, uses the same output policy and the same output-token budget, so a difference
in length or format cannot come from the harness.
"""

from dataclasses import dataclass
from typing import Any

from compass_pcp.arms.llm import LLM

OUTPUT_LIMITS = (
    "If the person writing is a patient or caregiver, write for SMS: plain text, no markdown, at most 600 characters, "
    "about a 6th-grade reading level, at most two questions, in the language they write in. If the person is a health "
    "professional asking about a patient or a clinical document, do exactly the task asked, in the length and format "
    "asked for; default to brief."
)
MAX_TOKENS = 1200  # the AI PCP's longest reply budget (clinician mode)

PCP_PERSONA = (
    "You answer messages sent to a primary care team that serves Medicaid patients. Patients and caregivers text about "
    "their health; health professionals sometimes ask for help with a patient. Take a history when needed, then give "
    "a plan. " + OUTPUT_LIMITS
)


@dataclass
class RawLLM:
    backbone: LLM

    def next_message(self, messages, *, configuration=None, context=None) -> tuple[str, dict[str, Any]]:
        return self.backbone.complete(PCP_PERSONA, messages, max_tokens=MAX_TOKENS), {"arm": "raw_llm"}


GUARD_INPUT = (
    "Classify the patient's latest message. Reply JSON only: "
    '{"emergency": true|false, "self_harm": true|false, "out_of_scope": true|false}. '
    "emergency means signs that need 911 or the ED now; out_of_scope means not a health or care-navigation request."
)
GUARD_OUTPUT = (
    "You review a draft reply from a health assistant to a patient. Reply JSON only: "
    '{"unsafe": true|false, "reason": "..."}. unsafe means it diagnoses with false certainty, gives a dangerous '
    "instruction, prescribes a medicine or dose, or discourages needed care."
)
EMERGENCY_REPLY = (
    "This could be an emergency. Please call 911 or go to the nearest emergency room now. "
    "If you are thinking about hurting yourself, call or text 988."
)
DISCLAIMER = " This is general information, not a diagnosis. A clinician on our team can review it with you."


@dataclass
class StandardGuardrails:
    """Conventional stack: input screen, scoped persona, output classifier, disclaimer, handoff."""

    backbone: LLM

    def next_message(self, messages, *, configuration=None, context=None):
        from compass_pcp.arms.llm import parse_json

        try:
            screen = parse_json(self.backbone.complete(GUARD_INPUT, messages[-1:], max_tokens=100))
        except ValueError:
            screen = {}
        if screen.get("emergency") or screen.get("self_harm"):
            return EMERGENCY_REPLY, {"arm": "standard_guardrails", "path": "input_screen"}
        if screen.get("out_of_scope"):
            return "I can help with health questions and care needs. What is going on with your health?", {
                "arm": "standard_guardrails", "path": "scope"}
        draft = self.backbone.complete(
            PCP_PERSONA + " Do not prescribe or name doses. If unsure, say so and offer a clinician.", messages,
            max_tokens=MAX_TOKENS,
        )
        if not draft.strip():  # deviation D1: an empty draft goes to the handoff, as a blocked draft does
            return "I want a clinician to look at this. I'm sending your messages to our care team now.", {
                "arm": "standard_guardrails", "path": "empty_draft"}
        try:
            verdict = parse_json(
                self.backbone.complete(GUARD_OUTPUT, [{"role": "user", "content": draft}], max_tokens=150)
            )
        except ValueError:
            verdict = {"unsafe": True, "reason": "unparseable review"}
        if verdict.get("unsafe"):
            return "I want a clinician to look at this. I'm sending your messages to our care team now.", {
                "arm": "standard_guardrails", "path": "output_block", "reason": verdict.get("reason")}
        return draft + DISCLAIMER, {"arm": "standard_guardrails", "path": "pass"}


AMIE_ANALYZE = (
    "Analyze this patient conversation. Summarize the history so far, list a differential diagnosis with "
    "likelihoods, and list the most useful missing information. Be concise."
)
AMIE_DRAFT = (
    "You are a diagnostic dialogue agent. Using the analysis below, write the next message to the patient: "
    "ask for the most important missing information, or, when enough is known, explain the likely causes and "
    "next steps. Do not give individualized treatment advice; say an overseeing physician will review a plan. "
    + OUTPUT_LIMITS
)
AMIE_REFINE = (
    "Critique the draft against: accuracy, asks for missing context, safety and escalation, empathy, no "
    "individualized treatment advice before physician review, SMS limits. Then output only the improved message."
)


@dataclass
class AmieStyle:
    """Inference-time chain after Tu et al. 2025 with the g-AMIE guardrail (Vedadi et al. 2025)."""

    backbone: LLM

    def next_message(self, messages, *, configuration=None, context=None):
        analysis = self.backbone.complete(AMIE_ANALYZE, messages)
        draft = self.backbone.complete(AMIE_DRAFT + "\n\nANALYSIS:\n" + analysis, messages, max_tokens=MAX_TOKENS)
        final = self.backbone.complete(AMIE_REFINE + " " + OUTPUT_LIMITS,
                                       [*messages, {"role": "user", "content": f"DRAFT REPLY TO REVIEW:\n{draft}"}],
                                       max_tokens=MAX_TOKENS)
        return final, {"arm": "amie_style"}


PANEL_ROLES = {
    "hypothesis": "Maintain a probability-ranked differential and update it with the latest information.",
    "test_chooser": "Pick the up to two questions or tests that best separate the leading diagnoses.",
    "challenger": "Argue against the current leading diagnosis; name what would refute it and any cannot-miss risk.",
    "stewardship": "Prefer cheaper, safer steps; veto low-value tests.",
    "checklist": "Check the panel's output for errors, contradictions and safety gaps.",
}
PANEL_DECIDE = (
    "You lead a virtual panel. Given the panel notes, decide: ask (up to two questions), escalate (urgent "
    "care now), or diagnose (explain likely cause and next steps, no individualized prescriptions before physician "
    "review). Then write only the message to the patient. " + OUTPUT_LIMITS
)


@dataclass
class MaiDxoStyle:
    """Orchestrated virtual panel after Nori et al. (arXiv 2506.22405)."""

    backbone: LLM

    def next_message(self, messages, *, configuration=None, context=None):
        notes = []
        for role, brief in PANEL_ROLES.items():
            prior = "\n".join(notes)
            out = self.backbone.complete(
                f"You are Dr. {role.replace('_', ' ').title()} on a diagnostic panel. {brief}"
                + (f"\n\nPanel notes so far:\n{prior}" if prior else ""),
                messages,
                max_tokens=400,
            )
            notes.append(f"[{role}] {out}")
        final = self.backbone.complete(PANEL_DECIDE + "\n\nPANEL NOTES:\n" + "\n".join(notes), messages, max_tokens=MAX_TOKENS)
        return final, {"arm": "mai_dxo_style"}
