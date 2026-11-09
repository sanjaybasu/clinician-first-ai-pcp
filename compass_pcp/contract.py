"""COMPASS v4 response contract (v4_spec.md A1, A2 deterministic part).

The engine's decision and the crisis route become a versioned contract per turn. The LLM writes the reply under the
contract (``arms/writer_v4.py``); this module builds the contract and runs the deterministic checks on a reply. It
does not change the engine: the v3 decision is reproduced by ``v3_disposition`` from the same inputs v3 uses.

| State     | Reply                                                                    |
|-----------|--------------------------------------------------------------------------|
| emergency | Required action first; no question before it                             |
| ask       | Permitted conditional or partial answer, then the one selected question  |
| defer     | State the limitation, give permitted information, state the handoff      |
| answer    | Answer directly without implying the workup is complete                  |
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from compass_pcp.crisis_router import RouteResult
from compass_pcp.engine.workup import DecisionKind
from compass_pcp.ledger import HazardTier
from compass_pcp.policy_v4 import ContractPolicy

CONTRACT_VERSION = "v4.0"
State = Literal["emergency", "ask", "defer", "answer"]
CRISIS_ROUTES = ("emergency_911", "crisis_988_page", "er_now")


class ResponseContract(BaseModel):
    version: str = CONTRACT_VERSION
    state: State
    disposition: str
    source_ids: list[str] = Field(default_factory=list)
    assessment_open: bool = True
    selected_question: str | None = None
    conditional_statements: list[str] = Field(default_factory=list)
    referral_text: str | None = None
    referral_position: Literal["first", "any", "none"] = "none"
    additional_clauses: list[str] = Field(default_factory=list)
    prohibited: list[str] = Field(default_factory=list)
    page_requested: bool = False
    page_executed: bool = False
    fallback_text: str
    route_clause: str | None = Field(default=None, description="Signed short clause from the crisis route")
    route_clause_max_sentences: int | None = None
    order_exception_for: str | None = Field(default=None, description="Criterion whose signed order exception applies")

    def required_clauses(self) -> list[str]:
        return [c for c in [self.referral_text, *self.additional_clauses] if c]


def v3_disposition(user_type: str, tier: HazardTier, kinds: list[DecisionKind], questions: list[str]) -> str:
    """The v3 reply path (AiPcpV2.next_message), as a label, from the same inputs and in the same order."""
    if user_type == "health_professional":
        return "clinician_support"
    if tier >= HazardTier.EMERGENCY_NOW or DecisionKind.ESCALATE in kinds:
        return "escalate"
    if tier >= HazardTier.POSSIBLE_EMERGENCY:
        return "possible_emergency"
    if tier >= HazardTier.URGENT_SAME_DAY:
        return "urgent_same_day"
    if not kinds:
        return "general_out_of_lane"
    if questions:
        return "ask"
    if DecisionKind.DEFER in kinds:
        return "defer"
    return "complete"


def engine_source_ids(decisions) -> list[str]:
    """Signed-source ids behind the engine's decisions: each illness script version and the engine version."""
    out = []
    for script, d in decisions:
        c = d.certificate
        out += [f"script:{c.script}@{c.script_version}", f"engine:{c.engine_version}"]
    return list(dict.fromkeys(out))


def build_contract(*, disposition: str, route: RouteResult | None, policy: ContractPolicy, decisions=(),
                   question: str | None = None, screen_category: str | None = None,
                   reporter_about_other: bool = False, page_executed: bool = False,
                   open_cannot_miss: bool = False, pathway_level: str | None = None,
                   first_aid: tuple[str, ...] = ()) -> ResponseContract | None:
    """Contract for this turn, or None when the reply follows the v3 non-engine path (clinician support, out of lane)
    with nothing the contract must add. ``question`` is the text of the engine's first selected question.
    ``open_cannot_miss``: a cannot-miss diagnosis is still in the middle zone. When the policy has an
    ``ask_open_cannot_miss`` entry, a question asked then carries its standing clause, so that a person who stops
    replying still holds a safe disposition. ``pathway_level``: the disposition the guideline pathways give a completed
    workup. When the policy has a ``complete_<level>`` entry, the reply gives that disposition instead of the
    physician-review clause; the pathway proposals go to the care-team queue. ``first_aid``: signed first-aid passages
    added word for word to an emergency reply, after its instruction."""
    prohibited = list(policy.prohibited)
    if route is not None and route.route in CRISIS_ROUTES:
        prohibited.append("any question before the required instruction")
        return ResponseContract(
            state="emergency", disposition=route.route, source_ids=[*route.source_ids, *route.rules_fired],
            referral_text=route.instruction_text, referral_position="first",
            additional_clauses=[*route.addition_texts, *first_aid], prohibited=prohibited,
            page_requested=route.page_care_team, page_executed=page_executed,
            fallback_text=" ".join([route.fallback_text or "", *route.addition_texts, *first_aid]).strip(),
            route_clause=route.required_clause, route_clause_max_sentences=route.clause_max_sentences,
            order_exception_for=route.order_exception_for,
        )
    additions = list(route.addition_texts) if route is not None else []
    if route is not None and route.route != "none" and route.instruction_text:
        additions.insert(0, route.instruction_text)  # a signed criterion clause on a non-crisis route (urgent review)
    add_sources = list(route.source_ids) if route is not None and additions else []
    if disposition in ("clinician_support", "general_out_of_lane"):
        if not additions:
            return None
        ref = policy.referrals["out_of_engine"]
        return ResponseContract(state="answer", disposition=disposition, source_ids=add_sources,
                                assessment_open=True, additional_clauses=additions, prohibited=prohibited,
                                fallback_text=" ".join([ref.fallback, *additions]))
    ref = policy.referrals[disposition]
    other = reporter_about_other
    text = (ref.text_other or ref.text) if other else ref.text
    fallback = (ref.fallback_other or ref.fallback) if other else ref.fallback
    sources = engine_source_ids(decisions) + ([f"screen:{screen_category}"] if screen_category else []) + add_sources
    state: State
    selected = None
    if disposition == "escalate":
        state = "emergency"
        prohibited.append("any question before the required instruction")
    elif disposition == "ask":
        state, selected = "ask", question
        standing = policy.referrals.get("ask_open_cannot_miss")
        if open_cannot_miss and standing is not None and standing.text:
            text = (standing.text_other or standing.text) if other else standing.text
            fallback = (standing.fallback_other or standing.fallback) if other else standing.fallback
            prohibited.append("a plan of home care alone while the assessment is open")
            ref = standing
    elif disposition == "screen_check":  # Paper 2 v19: ask the facts that decide a screen-only possible heart emergency
        state, selected = "ask", question
    elif disposition == "possible_emergency":  # with no engine question left, the reply answers; it never asks blind
        state, selected = ("ask", question) if question else ("answer", None)
    elif disposition == "urgent_same_day":
        state, selected = ("ask", question) if question else ("answer", None)
    elif disposition == "defer":
        state = "defer"
    elif disposition == "complete":
        state = "answer"
        done = policy.referrals.get(f"complete_{pathway_level}") if pathway_level else None
        if done is not None and done.text:
            ref = done
            text = (done.text_other or done.text) if other else done.text
            fallback = (done.fallback_other or done.fallback) if other else done.fallback
    else:
        raise ValueError(f"unknown disposition {disposition}")
    return ResponseContract(
        state=state, disposition=disposition, source_ids=list(dict.fromkeys(sources + [f"contract_policy:{policy.version}"])),
        assessment_open=disposition != "complete", selected_question=selected, referral_text=text,
        referral_position=ref.position if text else "none", additional_clauses=additions, prohibited=prohibited,
        fallback_text=" ".join([fallback, *additions]),
    )


# ---- deterministic checks -------------------------------------------------------------------------------------------
_STOP = {"what", "when", "where", "which", "your", "have", "does", "this", "that", "with", "from", "there", "they",
         "them", "been", "were", "will", "would", "could", "about", "into", "than", "then", "also", "very", "just"}


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s.lower()).strip()


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", _norm(s)) if len(w) > 3 and w not in _STOP}


def clause_position(text: str, clause: str) -> int:
    """Index of the clause in the normalized reply, or -1."""
    return _norm(text).find(_norm(clause).rstrip(".!"))


def sentences_before(text: str, idx: int) -> int:
    return len(re.findall(r"[.!?]+(?=\s|$)", _norm(text)[:idx]))


def question_present(text: str, question: str, overlap_min: float) -> bool:
    """The selected question appears verbatim, or one question sentence in the reply shares at least ``overlap_min`` of
    its content words. A diagnostic aid for the writer's paraphrase, not proof of clinical compliance."""
    if _norm(question).rstrip("?") in _norm(text):
        return True
    want = _words(question)
    if not want:
        return "?" in text
    for sent in re.findall(r"[^.!?]*\?", text):
        if len(want & _words(sent)) / len(want) >= overlap_min:
            return True
    return False


def deterministic_check(text: str, contract: ResponseContract, policy: ContractPolicy) -> list[str]:
    """Problems found by rule (empty list = passes): required clauses verbatim, referral order on emergency, the
    selected question when asking, and no claim that the care team was paged unless a page executed."""
    problems: list[str] = []
    if not text.strip():
        return ["the reply is empty"]
    for clause in contract.required_clauses():
        if clause_position(text, clause) < 0:
            problems.append(f'required clause missing or altered: "{clause}"')
    if contract.referral_text and contract.referral_position == "first":
        idx = clause_position(text, contract.referral_text)
        if idx >= 0:
            if "?" in _norm(text)[:idx]:
                problems.append("a question comes before the required instruction")
            if sentences_before(text, idx) >= policy.checks.referral_max_sentences:
                problems.append(f"the required instruction is not within the first {policy.checks.referral_max_sentences} sentences")
    if contract.route_clause:
        idx = clause_position(text, contract.route_clause)
        limit = contract.route_clause_max_sentences
        if idx < 0:
            problems.append(f'the route clause is missing: "{contract.route_clause}"')
        else:
            if "?" in _norm(text)[:idx]:
                problems.append("a question comes before the route clause")
            if limit is not None and sentences_before(text, idx) >= limit:
                problems.append(f"the route clause is not within the first {limit} sentence(s)")
    if contract.state == "ask":
        if contract.selected_question:
            if not question_present(text, contract.selected_question, policy.checks.question_overlap_min):
                problems.append(f'the selected question is missing: "{contract.selected_question}"')
        elif "?" not in text:
            problems.append("the reply asks no question")
    if not contract.page_executed:
        for p in policy.page_claim_patterns:
            if re.search(p, _norm(text)):
                problems.append("the reply says the care team was alerted or paged, but no page was executed")
                break
    return problems
