"""Deterministic crisis router for COMPASS v4 (v4_spec.md B1, B2).

The structured screen (``arms/screen_v4.py``) fills one form item per criterion in the signed crisis policy. This
module validates the form against the user's own words and maps it to a route with the policy's rules:

* A criterion that is present, with a verbatim quote, current (when the criterion requires it), fires its route.
* "not mentioned" and "absent" never set a floor.
* A family is applicable when (a) any of its items is present or uncertain with a verbatim quote, (b) one of its
  lexicon terms occurs in the user's text, or (c) the B2 classifier's family probability reaches the family cut.
  A lexicon hit or a classifier score only makes the family applicable; neither sets a route.
* Within an applicable family, an uncertain routing-relevant item, unknown currentness, or an item the screen did not
  complete sets the policy's uncertainty route (urgent review).
* A screen that failed as a whole (no readable form, timeout) sets the policy's screen-failure route.
* The route is addressed to the person at risk: a clinician or relative reporting about someone else gets the
  third-person instruction for that person.
* When the first criterion that fired a route carries its own signed instruction, that wording replaces the route's
  text and fallback; its order exception, if any, replaces the route's required clause and position.

No clinical criterion or threshold lives here; they are in ``data/policies/crisis_v4.yaml``.
"""

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from compass_pcp.policy_v4 import CrisisPolicy

Status = Literal["present", "absent", "not_mentioned", "uncertain"]
Person = Literal["patient", "other", "clinician_self", "unknown"]
STATUSES = ("present", "absent", "not_mentioned", "uncertain")
PERSONS = ("patient", "other", "clinician_self", "unknown")


class ScreenItem(BaseModel):
    criterion_id: str
    status: Status = "uncertain"
    current: bool | None = None
    person_at_risk: Person = "unknown"
    quote: str = ""
    quote_valid: bool = False
    completed: bool = False


class ScreenForm(BaseModel):
    items: dict[str, ScreenItem]
    completed: bool = Field(description="The screen returned a readable form in time")
    error: str | None = None


class RouteResult(BaseModel):
    route: str
    additions: list[str] = Field(default_factory=list)
    rules_fired: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    instruction_id: str | None = None
    instruction_text: str | None = None
    fallback_text: str | None = None
    addition_texts: list[str] = Field(default_factory=list)
    addressed_to_other: bool = False
    page_care_team: bool = False
    criterion_instruction_id: str | None = Field(default=None, description="Criterion whose wording was used")
    required_clause: str | None = None
    clause_max_sentences: int | None = None
    order_exception_for: str | None = None
    applicable: dict[str, list[str]] = Field(default_factory=dict, description="Family id -> applicability triggers")
    applicable_uncertainty: bool = False
    screen_completed: bool = True
    policy_version: str = ""
    policy_status: str = ""


class CrisisClassifier(Protocol):
    """B2: local multi-label crisis classifier. Review feature and applicability trigger only; never sets a route."""

    def predict(self, text: str) -> dict[str, float]: ...


class NullClassifier:
    """Stand-in until the B2 model is trained and frozen; contributes no trigger and a score of 0."""

    def predict(self, text: str) -> dict[str, float]:
        return {}


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def quote_is_valid(quote: str, user_texts: list[str]) -> bool:
    """True if the quote is a non-empty exact substring of one user message (whitespace runs collapsed)."""
    q = _ws(quote or "")
    return bool(q) and any(q in _ws(t) for t in user_texts)


def incomplete_form(policy: CrisisPolicy, error: str) -> ScreenForm:
    """A missing, unreadable or late form: every item uncertain and not completed."""
    items = {c.id: ScreenItem(criterion_id=c.id) for f in policy.families for c in f.criteria}
    return ScreenForm(items=items, completed=False, error=error)


def parse_form(raw: dict, policy: CrisisPolicy, user_texts: list[str]) -> ScreenForm:
    """Validate the screen's JSON. Items with an unknown id are dropped; a missing or malformed item stays incomplete."""
    items = {c.id: ScreenItem(criterion_id=c.id) for f in policy.families for c in f.criteria}
    rows = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return ScreenForm(items=items, completed=False, error="form has no items list")
    for r in rows:
        if not isinstance(r, dict) or r.get("criterion_id") not in items or r.get("status") not in STATUSES:
            continue
        cur = r.get("current")
        person = r.get("person_at_risk")
        quote = r.get("quote") if isinstance(r.get("quote"), str) else ""
        items[r["criterion_id"]] = ScreenItem(
            criterion_id=r["criterion_id"], status=r["status"], current=cur if isinstance(cur, bool) else None,
            person_at_risk=person if person in PERSONS else "unknown", quote=quote,
            quote_valid=quote_is_valid(quote, user_texts), completed=True,
        )
    return ScreenForm(items=items, completed=True)


def route_form(form: ScreenForm, policy: CrisisPolicy, *, lexicon: dict[str, list[str]] | None = None,
               classifier: dict[str, float] | None = None) -> RouteResult:
    lexicon, classifier = lexicon or {}, classifier or {}
    unc = policy.uncertainty
    fired: dict[str, list[tuple[str, str, str]]] = {}  # route -> [(rule id, source, person)]
    applicable: dict[str, list[str]] = {}
    applicable_uncertainty = False

    def fire(route: str, rule: str, source: str, person: str = "unknown") -> None:
        fired.setdefault(route, []).append((rule, source, person))

    for fam in policy.families:
        triggers = []
        for c in fam.criteria:
            it = form.items[c.id]
            if it.completed and it.status in ("present", "uncertain") and it.quote_valid:
                triggers.append(f"item:{c.id}")
            elif it.completed and it.status == "present" and unc.present_without_valid_quote == "uncertain_applicable":
                triggers.append(f"unquoted_present:{c.id}")
        if fam.id in lexicon:
            triggers.append("lexicon")
        cut = fam.classifier_cut
        if cut is not None and classifier.get(fam.id, 0.0) >= cut:
            triggers.append("classifier")
        if triggers:
            applicable[fam.id] = triggers
        is_app = bool(triggers)
        fam_unc = fam.uncertain_route or unc.uncertain_route

        for c in fam.criteria:
            it = form.items[c.id]
            if not it.completed:
                if is_app and c.routing_relevant:
                    fire(unc.incomplete_route, f"{c.id}:incomplete", unc.source)
                    applicable_uncertainty = True
                continue
            if it.status == "present" and it.quote_valid:
                if c.requires_current and it.current is False:
                    continue  # historical, resolved or hypothetical: the normal path
                if c.requires_current and it.current is None:
                    if is_app and c.routing_relevant:
                        fire(fam_unc, f"{c.id}:currentness_unknown", unc.source, it.person_at_risk)
                        applicable_uncertainty = True
                    continue
                fire(c.route, c.id, c.source, it.person_at_risk)
            elif it.status in ("present", "uncertain"):
                # uncertain, or present without a verbatim quote: a floor only inside an applicable family
                if is_app and c.routing_relevant:
                    tag = "unverified_quote" if it.status == "present" else "uncertain"
                    fire(fam_unc, f"{c.id}:{tag}", unc.source, it.person_at_risk)
                    applicable_uncertainty = True
            # absent and not_mentioned never set a floor

    if not form.completed:
        fire(unc.screen_failure_route, "screen:incomplete", unc.source)
        applicable_uncertainty = True

    live = [r for r in policy.route_precedence if r in fired and r != "none"]
    route = live[0] if live else "none"
    additions = [] if route in policy.additions_suppressed_by else [
        r for r in live if r in policy.additions and r != route]
    contributing = [r for r in live if r == route or r in additions]
    rules = [rule for r in contributing for rule, _, _ in fired[r]]
    sources = list(dict.fromkeys(s for r in contributing for _, s, _ in fired[r]))
    out = RouteResult(route=route, additions=additions, rules_fired=rules, source_ids=sources,
                      applicable=applicable, applicable_uncertainty=applicable_uncertainty,
                      screen_completed=form.completed, policy_version=policy.version, policy_status=policy.status)
    criteria = {c.id: c for f in policy.families for c in f.criteria}

    def own(r: str):
        """The first criterion (policy order) that fired route r and carries its own signed instruction."""
        return next((criteria[rule] for rule, _, _ in fired[r] if rule in criteria and criteria[rule].instruction), None)

    if route != "none":
        persons = [p for _, _, p in fired[route]]
        other = bool(persons) and all(p == "other" for p in persons)
        ins = policy.instructions.get(route)
        crit = own(route)
        if ins is not None:
            out.instruction_id = ins.id
            out.instruction_text = ins.text_other if other else ins.text
            out.fallback_text = ins.fallback_other if other else ins.fallback
            out.page_care_team = ins.page_care_team
            out.required_clause, out.clause_max_sentences = ins.required_clause, ins.clause_max_sentences
            out.source_ids = list(dict.fromkeys([ins.id, *out.source_ids]))
        if crit is not None:
            ci = crit.instruction
            text = (ci.text_other or ci.text) if other else ci.text
            out.instruction_text = text
            out.fallback_text = ((ci.fallback_other or ci.fallback) if other else ci.fallback) or text
            out.criterion_instruction_id = ci.id
            out.source_ids = list(dict.fromkeys([ci.id, *out.source_ids]))
            if crit.order_exception is not None:
                ex = crit.order_exception
                out.required_clause, out.clause_max_sentences = ex.required_clause, ex.max_sentences
                out.order_exception_for = crit.id
                out.source_ids.append(ex.source)
        if ins is not None or crit is not None:
            out.addressed_to_other = other
    for a in additions:
        ins = policy.instructions[a]
        persons = [p for _, _, p in fired[a]]
        other = bool(persons) and all(p == "other" for p in persons)
        crit = own(a)
        if crit is not None:
            ci = crit.instruction
            out.addition_texts.append((ci.text_other or ci.text) if other else ci.text)
            out.source_ids.append(ci.id)
        else:
            out.addition_texts.append(ins.text_other if other else ins.text)
            out.source_ids.append(ins.id)
    return out
