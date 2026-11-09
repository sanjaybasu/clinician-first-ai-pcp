"""Typed questions for the System-1 (Kev) triage layer: the teacher labels and the fine-tune use these exact strings.

Each question is a System One question (choice or noul). The state is the chart summary (paper/chart_context.py), when
there is one, followed by the text message. Development wording; not a clinical rule set (no doses, no thresholds).
"""

TIER = {
    "none": "No health or safety concern a clinician needs to see",
    "routine_followup": "A concern the care team should follow up on within days",
    "urgent_same_day": "A clinician should contact the person today",
    "possible_emergency": "Could be an emergency; needs a clinician right away to decide",
    "emergency_now": "Emergency now: 911, 988 or an emergency room",
}
ACTION = {
    "reply_only": "No clinical action; an ordinary reply is enough",
    "care_team_task": "A care-team task (scheduling, refill, paperwork, resources) within a few days",
    "clinician_today": "A clinician (nurse, pharmacist, therapist or physician) contacts the person today",
    "emergency_room": "Go to an emergency room or urgent care now",
    "call_911": "Call 911 now",
    "crisis_988": "Mental health or substance crisis: 988 or a crisis team now",
}
SENDER = {
    "patient": "The patient, writing about their own health",
    "caregiver": "A family member or caregiver, writing about someone else",
    "clinician": "A health professional, asking about a patient or a clinical question",
    "other": "Someone else, or a message with no health content",
}
SOCIAL = {
    "none": "No social need in the message",
    "food": "Food",
    "housing": "Housing, eviction or unsafe home",
    "transport": "Transport to care",
    "utilities_phone": "Utilities, heat or phone service",
    "medicine_cost": "Cannot afford or get medicines or supplies",
    "coverage_benefits": "Insurance, Medicaid renewal or other benefits",
    "safety": "Violence, abuse or personal safety",
    "other": "Another social need",
}

QUESTIONS = {
    "tier": {"type": "choice", "criteria": TIER,
             "instructions": "Read the text message, using the chart for context. Choose the highest safety tier that "
                             "applies to the person the message is about. Hazards are often implicit or buried inside "
                             "other requests."},
    "action": {"type": "choice", "criteria": ACTION,
               "instructions": "What should the care team do next about this message?"},
    "sender": {"type": "choice", "criteria": SENDER, "instructions": "Who wrote this text message?"},
    "current": {"type": "noul",
                "instructions": "Does the text message describe a health problem that someone has now (not only in the "
                                "past, hypothetical, or already resolved)?"},
    "missing": {"type": "noul",
                "instructions": "Would one or two answers from the person (how long, how bad, which other symptoms, "
                                "whether it is happening now) change how urgent this message is?"},
    "social": {"type": "choice", "criteria": SOCIAL,
               "instructions": "Which social need, if any, does the text message raise?"},
}


def state(chart: str | None, text: str) -> str:
    if chart:
        return ("Chart before this message:\n" + chart + "\n\nText message to the care team:\n" + text)
    return "Text message to the care team:\n" + text
