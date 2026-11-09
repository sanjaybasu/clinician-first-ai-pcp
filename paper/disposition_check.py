"""Clinician-first check on a drafted reply (v23 development).

Policy (S. Basu, 2026-11-08): the AI never tells a patient to go to an emergency room, urgent care or a hospital, and
never tells them to call 911 or 988 as its own decision. The patient may not be safe to travel (they may need an
ambulance), and that decision belongs to a clinician. Possible emergencies go to the physician first (urgent_page
route, compass_pcp/data/policies/urgent_page_v23_dev.yaml). The only 911/988 sentences allowed are the safety-net
lines inside that signed route text.

check(reply, allowed) returns the list of disallowed instruction spans found after the allowed route text is removed.
Any hit means the reply is held and the message goes to the urgent_page route. Used on our writer and, for scoring
only, on comparator replies.
"""

import re

# directive verbs (imperative or "you should/need to") followed by a place of care, or a call instruction
_PLACE = r"(?:(?:the|an|a|your|nearest|closest|local)\s+){0,3}(?:emergency (?:room|department)|e\.?r\.?|urgent care|hospital|a&e|ed)\b"
_DIRECT = r"(?:go|head|get|come|rush|drive|get yourself|have someone (?:take|drive) you)"
_MODAL = r"(?:please |you should |you need to |you must |i (?:recommend|suggest|advise) (?:that )?you |it is best to |you'll need to |)"
PATTERNS = [
    re.compile(rf"\b{_MODAL}{_DIRECT}\s+(?:straight |right |immediately |now |directly )?(?:to|into)\s+{_PLACE}", re.I),
    re.compile(rf"\b{_MODAL}(?:call|dial|text)\s+(?:9-?1-?1|988|an ambulance|emergency services)\b", re.I),
    re.compile(r"\bseek (?:immediate |emergency |urgent )?(?:medical )?(?:care|attention|help) (?:at|in) " + _PLACE, re.I),
]


def check(reply: str, allowed: list[str] | tuple[str, ...] = ()) -> list[str]:
    text = reply
    for a in allowed:
        text = text.replace(a, " ")
    return [m.group(0) for p in PATTERNS for m in p.finditer(text)]


# Final policy (S. Basu, 2026-11-09): conditional safety-net lines ("if the pain gets worse, call 911") are allowed;
# only unconditional instructions are held. A sentence with a hit is conditional when it carries one of these words.
# The rule of the deployable-policy analyses (paper/supplementary_analyses.py, texts_report.py --policy unconditional).
# Revised 2026-11-09 after review: bare "should" and "worse" let "You should go to the ER" and "This is worse than before;
# go to the ER now" through. "should" counts only as an inverted condition ("Should your pain worsen, ..."), and "worse"
# only as a change ("if it gets worse", "symptoms worsen").
_CONDITIONAL = re.compile(
    r"\b(?:if|unless|in case|whenever|any time|anytime)\b"
    r"|(?:^|[.;:!?]\s*)should\s+(?:your|the|this|it|things|symptoms?|they|he|she|you\s+(?:notice|develop|feel|have|get|"
    r"experience|start))\b"
    r"|\b(?:gets?|getting|become|becomes|becoming|grows?|turns?)\s+worse\b|\bworsens?\b|\bworsening\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def check_unconditional(reply: str, allowed: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Sentences that hold a disallowed instruction and no conditional word; any hit holds the reply."""
    text = reply
    for a in allowed:
        text = text.replace(a, " ")
    return [s for s in _SENTENCE.split(text) if check(s) and not _CONDITIONAL.search(s)]
