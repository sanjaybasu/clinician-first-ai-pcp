"""AI PCP arm, v4 (design/v4_spec.md).

v4 wraps an unchanged v3 arm (``AiPcpV2`` built as ai_pcp_v3) and calls only its stateless helpers (router, extractors,
``_decide``, question text). It never calls ``AiPcpV2.build_ledger`` or ``next_message``, which keep the router's
user_type on the instance; all per-conversation state here is local or returned.

Per message:
1. B1 universal structured screen on every message (health professionals included), run in parallel with the v3
   ledger build, then the deterministic crisis router (lexicon and B2 classifier as applicability triggers only).
2. A B1 911 or 988 route makes the contract an emergency contract with the signed instruction first and no question
   before it. A care-team page is requested in the returned info; it is executed only through a pager that reports
   success, and the reply never claims a page.
3. Otherwise the v3 routing and engine decide as in v3; the decision becomes a response contract (A1), the writer
   writes under it ("contract" = A1-c, "edit" = A1-d), and the extended check (A2) passes it, revises it once, or sends
   the signed fallback. Clinician-support and out-of-lane replies keep the v3 composer, with the optional A3 compact
   policy for patients.
4. The B3 review flag: protected floors plus the monotone review score.
"""

import concurrent.futures as cf
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from compass_pcp.arms.hazard import kev_screen, screen
from compass_pcp.arms.llm import LLM
from compass_pcp.arms.screen_v4 import run_screen, user_texts
from compass_pcp.arms.writer_v4 import compose_v3_audited, write_under_contract
from compass_pcp.contract import CRISIS_ROUTES, build_contract, question_present, v3_disposition
from compass_pcp.crisis_router import CrisisClassifier, NullClassifier, route_form
from compass_pcp.engine.stopping import CannotMissStatus
from compass_pcp.engine.workup import DecisionKind
from compass_pcp.findings import Finding, reconcile
from compass_pcp.ledger import HazardEvent, HazardTier, Ledger
from compass_pcp.pathways import Node, Pathway, evaluate
from compass_pcp.policy_v4 import ContractPolicy, CrisisPolicy, lexicon_hits, load_contract_policy, load_crisis_policy
from compass_pcp.review_score import ReviewScoreParams, load_review_params, protected_floors, review_decision


def pathway_escalation(decisions, nodes: dict, disposition: str) -> str:
    """Disposition for an engine escalation: the most urgent pathway disposition among the escalating complaints, or
    ``disposition`` (emergency) when any escalating complaint has no pathway or its node gives no disposition."""
    levels = []
    for script, d in decisions:
        if d.kind != DecisionKind.ESCALATE:
            continue
        node = nodes.get(script.complaint)
        if node is None or node.disposition is None:
            return disposition
        levels.append(node.disposition.level)
    if not levels or "emergency_now" in levels:
        return disposition
    return "urgent_same_day" if "urgent_same_day" in levels else disposition


def pathway_completion(decisions, nodes: dict, allow_urgent: bool = False) -> str | None:
    """Disposition for a completed workup from the guideline pathways: the most urgent of self_care and routine_visit
    (and urgent_same_day when ``allow_urgent``, Paper 2 v20) when every complaint completed and its pathway node gives
    one of them; otherwise None (physician review)."""
    allowed = ("self_care", "routine_visit", "urgent_same_day") if allow_urgent else ("self_care", "routine_visit")
    levels = []
    for script, d in decisions:
        node = nodes.get(script.complaint)
        if d.kind != DecisionKind.COMPLETE or node is None or node.disposition is None:
            return None
        if node.disposition.level not in allowed:
            return None
        levels.append(node.disposition.level)
    if not levels:
        return None
    return next(lv for lv in reversed(allowed) if lv in levels)


def asked_unanswered(engine, ledger: Ledger, messages, overlap_min: float) -> tuple[str, ...]:
    """Features of the active scripts whose question appears in an earlier reply and that still have no finding. Empty
    unless the engine's WorkupConfig.skip_unanswered is on."""
    if not getattr(engine.config, "skip_unanswered", False):
        return ()
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    findings, _ = ledger.state()
    known = {f.feature_id for f in findings}
    out = []
    for sid in ledger.active_complaints:
        for f in engine.scripts[sid].features:
            if f.remote_askable and f.id not in known and any(question_present(r, f.question, overlap_min) for r in replies):
                out.append(f.id)
    return tuple(dict.fromkeys(out))


def remaining_red_flags(decisions, ledger: Ledger) -> list[str]:
    """Unanswered must_ask features of the escalating complaints, in script order."""
    findings, _ = ledger.state()
    known = {f.feature_id for f in findings}
    out = [f for script, d in decisions if d.kind == DecisionKind.ESCALATE for f in script.must_ask if f not in known]
    return list(dict.fromkeys(out))


CARDIAC_CATEGORY = re.compile(r"cardi|angina|ischem|heart|chest|coronary|myocard|\bacs\b|\bmi\b", re.IGNORECASE)


def screen_check(hazards, messages, question: str, overlap_min: float) -> tuple[str | None, HazardTier | None]:
    """("ask", None) when the possible_emergency ratings are all heart-related and the check question was never asked;
    ("resolved", latest tier) when it was asked and answered; (None, None) otherwise. ``hazards`` are the screen events
    of patient turns (Kev excluded)."""
    pe = [h for h in hazards if h.tier == HazardTier.POSSIBLE_EMERGENCY]
    if not pe or any(h.tier > HazardTier.POSSIBLE_EMERGENCY for h in hazards) or not all(
            CARDIAC_CATEGORY.search(f"{h.category} {h.rationale}") for h in pe):
        return None, None
    asked = [i for i, m in enumerate(messages) if m["role"] == "assistant" and question_present(m["content"], question, overlap_min)]
    if not asked:
        return "ask", None
    if not any(m["role"] == "user" for m in messages[asked[-1] + 1:]):
        return "ask", None
    last = max(h.turn for h in hazards)
    return "resolved", max(h.tier for h in hazards if h.turn == last)


def resolved_tier(hazards) -> HazardTier:
    """Highest screen tier once a heart check is answered: earlier heart-related possible_emergency ratings are dropped;
    the latest turn's rating and every other rating stay."""
    last = max(h.turn for h in hazards)
    kept = [h.tier for h in hazards if not (h.tier == HazardTier.POSSIBLE_EMERGENCY and h.turn < last
                                             and CARDIAC_CATEGORY.search(f"{h.category} {h.rationale}"))]
    return max(kept, default=HazardTier.NONE)


def first_aid_for(rules_fired, table: dict[str, list[tuple[str, bool]]], about_other: bool = False) -> tuple[str, ...]:
    """First-aid passages for the criteria that fired as present; an "uncertain" criterion ("CA-01:uncertain") adds none.
    Adult-only passages are left out when the message is about someone else, who may be a child."""
    return tuple(dict.fromkeys(t for r in rules_fired if ":" not in r for t, adult in table.get(r, [])
                               if not (adult and about_other)))


class CareTeamPager(Protocol):
    """Executes a care-team page and reports whether it went through. The reply never claims a page."""

    def page(self, *, route: str, rules: list[str], messages) -> bool: ...


@dataclass
class TurnState:
    ledger: Ledger
    suspected: list[str]
    disputed: list[str]
    user_type: str
    review_only_hazards: list[HazardEvent] = field(default_factory=list)  # screens of professional turns (B3 only)


@dataclass
class AiPcpV4:
    engine: AiPcpV2  # the v3 arm; only stateless helpers are called
    screen_llm: LLM  # B1: Anthropic Sonnet 5.5 with thinking off
    crisis_policy: CrisisPolicy | None = None  # None: load the default file (signed required unless test_mode)
    contract_policy: ContractPolicy = field(default_factory=load_contract_policy)
    review_params: ReviewScoreParams = field(default_factory=load_review_params)
    classifier: CrisisClassifier = field(default_factory=NullClassifier)
    pager: CareTeamPager | None = None
    writer_variant: str = "contract"  # "contract" (A1-c) or "edit" (A1-d)
    compact_policy: bool = False  # A3, off by default
    engine_off: bool = False  # ablation: the engine makes no decision (same as the v3 engine_off ablation)
    screen_professionals: bool = True  # also run the v3 hazard screens on professional turns, for B3 features only
    # Guideline pathways (design/pathway_schema.md): an engine escalation with no emergency tier takes its disposition
    # from the complaint's pathway node; complaints without a pathway keep the emergency disposition.
    pathways: dict[str, Pathway] = field(default_factory=dict)
    # Paper 2 development: when a pathway turns an engine escalation into same-day care, keep asking the escalating
    # complaints' unanswered red flags, one per reply, since one of them can raise the level to an emergency.
    ask_after_downgrade: bool = False
    # Paper 2 development: the Kev judge (near-flat tier probabilities on real messages) can raise a message to clinician
    # review but not to urgent advice; the LLM screen and the crisis screen still set urgent and emergency tiers.
    kev_review_only: bool = False
    # Paper 2 development: crisis criterion id -> signed first-aid passages added word for word to the emergency reply
    # when the criterion fires as present (policy_v4.load_first_aid).
    first_aid: dict[str, list[tuple[str, bool]]] = field(default_factory=dict)
    # Paper 2 development (v18): when the review flag is raised and the reply comes from the general (free-text) lane,
    # add the contract policy's fixed review_followup sentence at the end, so the reply tells the patient the care team
    # has the message. Off by default; needs a contract policy with a review_followup entry.
    review_followup: bool = False
    # Paper 2 development (v19): when only the general hazard screen rates the conversation possible_emergency, and every
    # such rating names a possible heart cause, the reply first asks the contract policy's screen_check_question (with
    # its 911 line) instead of sending the person to an emergency room. Once the person has answered, the reply level
    # follows the latest screen of the whole conversation; it can fall from possible_emergency, never from
    # emergency_now and never below a crisis route. The care-team flag keeps the highest tier.
    resolve_screen_pe: bool = False
    # Paper 2 development (v21): after the check question is answered, drop only the earlier heart-related
    # possible_emergency ratings; every other rating (for example a same-day rating from another turn) still counts.
    # False keeps the v19 rule (the reply follows the latest screen).
    resolve_cardiac_only: bool = False
    # Paper 2 development (v22): complaints for which a possible heart emergency is the presenting problem, not a hidden
    # cause; the heart check never replaces the emergency reply when one of them is active.
    screen_check_exclude: tuple[str, ...] = ()
    # Paper 2 development (v20): a completed workup whose pathway node gives urgent_same_day (the NICE high-risk node)
    # gets the urgent_same_day reply instead of the physician-review clause.
    complete_urgent: bool = False
    # Test mode allows an unsigned crisis policy; otherwise only a signed one loads (env COMPASS_PCP_TEST_MODE=1).
    test_mode: bool = field(default_factory=lambda: os.environ.get("COMPASS_PCP_TEST_MODE") == "1")

    def __post_init__(self) -> None:
        if self.writer_variant not in ("contract", "edit", "contract_full", "contract_v9"):
            raise ValueError(f"unknown writer variant {self.writer_variant}")
        if self.crisis_policy is None:
            self.crisis_policy = load_crisis_policy(require_signed=not self.test_mode)
        elif not self.test_mode and self.crisis_policy.status != "signed":
            raise ValueError(f"crisis policy status {self.crisis_policy.status}: a signed policy is required")

    # ---- v3 ledger, without instance state ------------------------------------------------------------------------
    def build_state(self, messages, context_findings: list[Finding]) -> TurnState:
        """AiPcpV2.build_ledger with user_type returned instead of stored, and the engine extractors skipped when the
        engine is off (they feed only the engine)."""
        e = self.engine
        ledger = Ledger()
        for f in context_findings:
            ledger.add_finding(0, f)
        suspected: list[str] = []
        disputed: list[str] = []
        user_type = "patient"
        review_only: list[HazardEvent] = []
        user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
        for turn, i in enumerate(user_idx, start=1):
            prefix = messages[: i + 1]
            cats, suspected, user_type, _ = e._route_full(prefix)
            professional = user_type == "health_professional"
            if professional and not self.screen_professionals:
                continue
            events = []
            if e.hazard_judge is not None:
                kev = kev_screen(e.hazard_judge, prefix, turn)
                if self.kev_review_only and kev.tier > HazardTier.ROUTINE_FOLLOWUP:
                    kev = kev.model_copy(update={"tier": HazardTier.ROUTINE_FOLLOWUP})
                events.append(kev)
            if e.hazard_judge is None or e.hazard_both:
                events.append(screen(e.hazard_llm, prefix, turn))
            if professional:
                review_only += events  # v3 does not screen these turns; they feed the review score only
                continue  # clinician-facing support, not a patient encounter
            for ev in events:
                ledger.add_hazard(ev)
            ledger.add_complaints(cats)
            active = [e.scripts[c] for c in ledger.active_complaints]
            if not active or self.engine_off:
                continue
            a = e._code_turn(e.extractor_a, active, prefix, turn)
            if e.extractor_b is None:
                accepted, disputed = a, []
            else:
                b = e._code_turn(e.extractor_b, active, prefix, turn)
                r = reconcile([f for f, _ in a], [f for f, _ in b])
                corr = {f.feature_id for f, c in a if c} & {f.feature_id for f, c in b if c}
                accepted = [(f, f.feature_id in corr) for f in r.accepted]
                disputed = r.disputed
            for f, corrected in accepted:
                ledger.add_finding(turn, f, correction=corrected)
        return TurnState(ledger=ledger, suspected=suspected, disputed=disputed, user_type=user_type,
                         review_only_hazards=review_only)

    def _tiers(self, st: TurnState) -> tuple[HazardTier, HazardTier, float, str]:
        """(patient-facing tier exactly as v3, Anthropic screen tier, Kev cumulative P(urgent or higher), screen
        category). The last three include professional-turn screens, which v3 does not run."""
        ledger, e = st.ledger, self.engine
        own = [h for h in ledger.hazards if h.category != "kev"]
        reply_tier = max((h.tier for h in own), default=HazardTier.NONE) if (
            e.compose_version >= 3 and e.hazard_both) else ledger.hazard_tier
        hazards = [*ledger.hazards, *st.review_only_hazards]
        llm_events = [h for h in hazards if h.category != "kev"]
        llm_tier = max((h.tier for h in llm_events), default=HazardTier.NONE)
        kev = 0.0
        for h in hazards:
            if h.category == "kev" and h.rationale:
                try:
                    p = json.loads(h.rationale)
                except ValueError:
                    continue
                kev = max(kev, sum(float(p.get(k, 0.0)) for k in ("urgent_same_day", "possible_emergency", "emergency_now")))
        top = max(own, key=lambda h: h.tier, default=None)
        return reply_tier, llm_tier, kev, (top.category if top and top.tier > HazardTier.NONE else "")

    def _pathway_nodes(self, decisions, st: TurnState, unanswered=()) -> dict[str, Node | None]:
        """Complaint -> matching pathway node, for the complaints that have a pathway."""
        findings, _ = st.ledger.state()
        return {s.complaint: evaluate(self.pathways[s.complaint], d, findings, unanswered)
                for s, d in decisions if s.complaint in self.pathways}

    # ---- reply -------------------------------------------------------------------------------------------------------
    def next_message(self, messages, *, configuration=None, context=None) -> tuple[str, dict[str, Any]]:
        policy = self.crisis_policy
        assert policy is not None
        context_findings: list[Finding] = list((context or {}).get("findings", []))
        text_all = "\n".join(user_texts(messages))
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_screen, self.screen_llm, messages, policy)
            st = self.build_state(messages, context_findings)
            form = fut.result()
        lex = lexicon_hits(text_all, policy)
        probs = self.classifier.predict(text_all)
        route = route_form(form, policy, lexicon=lex, classifier=probs)

        reply_tier, llm_tier, kev, category = self._tiers(st)
        turns = sum(1 for m in messages if m["role"] == "user")
        unanswered: tuple[str, ...] = ()
        if self.engine_off:
            decisions, questions = [], []
        else:
            unanswered = asked_unanswered(self.engine, st.ledger, messages, self.contract_policy.checks.question_overlap_min)
            decisions, questions = self.engine._decide(st.ledger, st.suspected, st.disputed, turns, unanswered)
        kinds = [d.kind for _, d in decisions]
        check_q, check = None, None
        if (self.resolve_screen_pe and route.route not in CRISIS_ROUTES and DecisionKind.ESCALATE not in kinds
                and not set(st.ledger.active_complaints) & set(self.screen_check_exclude)):
            check_q = self.contract_policy.referrals["screen_check_question"].text
            check, latest = screen_check([h for h in st.ledger.hazards if h.category != "kev"], messages, check_q,
                                         self.contract_policy.checks.question_overlap_min)
            if check == "resolved" and reply_tier == HazardTier.POSSIBLE_EMERGENCY and latest < reply_tier:
                if self.resolve_cardiac_only:
                    reply_tier = resolved_tier([h for h in st.ledger.hazards if h.category != "kev"])
                else:
                    reply_tier = latest
        disposition = v3_disposition(st.user_type, reply_tier, kinds, questions)
        if check == "ask" and disposition == "possible_emergency":
            disposition = "screen_check"
        nodes = self._pathway_nodes(decisions, st, unanswered)
        if disposition == "escalate" and reply_tier < HazardTier.EMERGENCY_NOW and route.route not in CRISIS_ROUTES:
            disposition = pathway_escalation(decisions, nodes, disposition)
            if disposition != "escalate" and self.ask_after_downgrade:
                questions = [f for f in remaining_red_flags(decisions, st.ledger) if f not in unanswered][:1] or questions

        level = pathway_completion(decisions, nodes, self.complete_urgent) if disposition == "complete" else None
        if level == "urgent_same_day":
            disposition, level = "urgent_same_day", None
        page = {"requested": False, "executed": False, "status": "not_requested"}
        if route.route in CRISIS_ROUTES and route.page_care_team:
            page["requested"] = True
            if self.pager is None:
                page["status"] = "no_pager_configured"
            else:
                try:
                    page["executed"] = bool(self.pager.page(route=route.route, rules=route.rules_fired, messages=messages))
                    page["status"] = "executed" if page["executed"] else "failed"
                except Exception as err:  # noqa: BLE001 - any pager failure is reported, never claimed as a page
                    page["status"] = f"failed: {type(err).__name__}"

        contract = build_contract(
            disposition=disposition, route=route, policy=self.contract_policy, decisions=decisions,
            question=check_q if disposition == "screen_check" else (
                self.engine._question_text(questions[0]) if questions else None),
            screen_category=category or None, page_executed=page["executed"],
            reporter_about_other=st.user_type in ("caregiver", "health_professional"),
            open_cannot_miss=any(c.status == CannotMissStatus.NEEDS_WORKUP
                                 for _, d in decisions for c in d.certificate.cannot_miss),
            pathway_level=level,
            first_aid=first_aid_for(route.rules_fired, self.first_aid,
                                    about_other=route.addressed_to_other or st.user_type in ("caregiver", "health_professional")),
        )
        compose: dict[str, Any] = {}
        if contract is not None:
            text, compliance = write_under_contract(self.engine.dialogue, messages, contract, self.contract_policy,
                                                    variant=self.writer_variant)
            path = "contract"
        elif disposition == "clinician_support":
            text, compose = compose_v3_audited(self.engine.dialogue, "clinician", messages)
            compliance, path = None, "v3_clinician"
        else:
            text, compose = compose_v3_audited(self.engine.dialogue, "general", messages, compact=self.compact_policy)
            compliance, path = None, "a3_compact" if self.compact_policy else "v3_general"

        floors = protected_floors(route=route.route, applicable_uncertainty=route.applicable_uncertainty,
                                  engine_escalation=DecisionKind.ESCALATE in kinds, screen_tier=llm_tier,
                                  floor_tier=self.review_params.floor_screen_tier)
        features = {"anthropic_tier": int(llm_tier) / int(HazardTier.EMERGENCY_NOW), "kev_urgent": kev,
                    "classifier_max": max(probs.values(), default=0.0)}
        review = review_decision(floors, features, self.review_params)
        followup_added = False
        if self.review_followup and review.flag and path in ("v3_general", "a3_compact"):
            text = text.rstrip() + " " + self.contract_policy.referrals["review_followup"].text
            followup_added = True

        info: dict[str, Any] = {
            "arm": "ai_pcp_v4",
            "path": {"user_type": st.user_type, "complaints": st.ledger.active_complaints, "decision": disposition,
                     "decisions": [k.value for k in kinds], "hazard_tier": reply_tier.name,
                     "care_team_flag_tier": st.ledger.hazard_tier.name, "route": route.route,
                     "rules_fired": route.rules_fired, "contract_state": contract.state if contract else None,
                     "reply_path": path},
            "questions": questions,
            "certificates": [d.certificate.model_dump(mode="json") for _, d in decisions],
            "pathway": {c: (n.model_dump(mode="json") if n else None) for c, n in nodes.items()},
            "screen": {"completed": form.completed, "error": form.error,
                       "items": {k: v.model_dump() for k, v in form.items.items()}},
            "lexicon_hits": lex, "classifier": probs,
            "route": route.model_dump(),
            "page": page,
            "contract": contract.model_dump() if contract else None,
            "compliance": compliance,
            "fallback": bool(compliance and compliance["fallback"]),
            "compose_audit": compose or None,
            "review": review.model_dump(),
            "review_followup_added": followup_added,
            "screen_check": check,
            "policy_status": {"crisis": policy.status, "contract": self.contract_policy.status,
                              "review_score": self.review_params.status},
            "writer_variant": self.writer_variant, "compact_policy": self.compact_policy, "engine_off": self.engine_off,
        }
        return text, info
