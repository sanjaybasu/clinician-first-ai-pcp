"""AI PCP arm, v2 (panel v1 items 1, 2 and 4).

Changes from v1:
1. Semantic hazard screen over all patient turns, raise-only, tiered, with a
   verbatim quote; the conversation continues at every tier.
2. Event-sourced ledger built turn by turn: each patient turn is coded once with
   the earlier conversation as context, corrections are explicit, and conflicts
   become confirmation questions. Complaint routing is multi-label; each script
   keeps its own posteriors and the red-flag obligations are unioned.
3. Out-of-lane response contract: general information, safety netting, focused
   history and a note that the care team will review, instead of a bare handoff.
   One scope check flags individualized diagnosis or treatment and triggers one
   revision (revise, not replace).
The deterministic engine (workup.step, gate) is unchanged and still decides.
"""

from dataclasses import dataclass, field
from typing import Any

from compass_pcp.arms.baselines import AMIE_ANALYZE, AMIE_REFINE, MAX_TOKENS, OUTPUT_LIMITS
from compass_pcp.arms.hazard import kev_screen, screen
from compass_pcp.arms.llm import LLM, parse_json
from compass_pcp.engine.scripts import IllnessScript
from compass_pcp.engine.workup import DecisionKind, WorkupConfig, WorkupState, step
from compass_pcp.findings import CodeSystem, Finding, Polarity, Provenance, reconcile
from compass_pcp.ledger import HazardTier, Ledger

PERMISSIONS = """What you may do before a physician signs off: take a history, give general health information that is true for
most people with this kind of concern, explain warning signs and when to get urgent or emergency care, help with logistics,
and say the care team will review. Always reply in the same language and script the person writes in. What you must not do yet: tell this person what condition they have, or tell them to
start, stop or change a specific medicine or dose. Write for SMS: plain text, no markdown, at most 600 characters, about a
6th-grade reading level, at most two questions."""

ROUTER = """First decide who is writing and about whom. user_type is health_professional only if they say they are a clinician or clearly
write as one about a patient or clinical document (for example "my patient", "pt", a note to revise), or ask about running
a clinic, a care program or health services (supplies, staffing, protocols, public health); a person describing their own or
a family member's health is patient or caregiver. If it is unclear whether the writer is a health professional, choose
patient. Then list every health concern raised so far that matches these
categories, most urgent first. Only list a category if it fits the concern itself, not just a shared word (for example,
brief dizziness on standing is not fainting).
{cats}
Reply JSON only: {{"user_type": "patient|caregiver|health_professional|other", "patient_age_years": <number or null>,
"categories": ["<id>", ...], "suspected": ["<diagnosis id from those categories or other:<name>>", ...]}}.
Use an empty list when nothing matches."""

EXTRACT = """You code the patient's LATEST message only (earlier messages are context). For each feature below, answer present,
absent or unknown based on what the patient says in the latest message. Use unknown if they did not address it, if it is
hypothetical, or if only the assistant said it. confirmed=true if the assistant's previous message asked about that feature
and the patient answered it directly. correction=true if the patient explicitly corrects something they said earlier.
Features:
{features}
Reply JSON only: {{"<feature_id>": {{"value": "present|absent|unknown", "confirmed": false, "correction": false}}, ...}}"""

COMPOSE = {
    "emergency": "The patient may need emergency care now. Reason (patient's words): \"{quote}\". Tell them clearly and calmly to "
    "call 911 (or the local emergency number) or go to the nearest emergency room now (or call or text 988 if this is about "
    "suicide or a mental health crisis), give the one to three most useful things to do while waiting, and say the care team "
    "has been alerted. Do not ask questions unless the answer changes what they should do in the next minutes.",
    "urgent": "First answer what they asked, helpfully and in general terms, as you would for any patient. Then say a clinician "
    "on the care team will also reach out today about \"{quote}\", and give the one or two warning signs that mean they should "
    "go to the ER sooner. Then {next}.",
    "ask": "First answer what they asked in general terms in one to three sentences (what this kind of symptom often means "
    "and what usually helps, without saying what they have). Then ask these questions, warmly and simply: {questions}. End "
    "with the one warning sign that means they should get urgent care.",
    "defer": "Say a physician on the care team will review their messages today and reply. Give brief general information "
    "about this kind of concern and the warning signs that mean they should get urgent care sooner.",
    "complete": "Summarize what you heard in one sentence, say a physician will review and reply with a plan within a few "
    "hours, and give the warning signs that mean they should get care sooner.",
    "possible_emergency": "What they describe could be an emergency depending on a few facts. In plain words, list the specific "
    "signs that mean they should call 911 or go to the ER right now. Then answer their question in general terms and ask the "
    "one or two questions that best tell whether this is an emergency{questions}.",
    "clinician": "The person writing is a health professional asking for help with a patient, a clinical document, or the "
    "delivery of care. Do exactly "
    "the task asked, in the length and format asked for; default to brief. Be accurate and guideline-based and name the "
    "guideline when you rely on one. If key clinical information is missing and would change the answer, ask for it (one "
    "or two questions) along with your best provisional answer. Never invent clinical facts that were not given (symptoms, "
    "exam findings, vital signs, results); in a note or document, leave a clearly marked blank for anything missing. When "
    "the question touches possible emergencies, say plainly "
    "which findings mean the patient needs emergency evaluation now. Reply in the language they write in. Do not tell them a "
    "clinician will contact them.",
    "general": "This concern is outside the care pathways you handle. Help the person as well as you can within the rules: "
    "answer their question with accurate general information, ask the one or two questions that matter most if key "
    "context is missing (but do not ask if you already have what you need), say when to get urgent or emergency care, "
    "and say the care team can review if they want.",
}

SCOPE_CHECK = """Does this draft message to a patient do either of these: (a) tell the person they have a specific condition,
or (b) tell them to start, stop or change a specific medicine or dose? General information ("these symptoms are often
caused by...", "many people use...") is allowed. Reply JSON only: {"leak": true|false, "span": "..."}"""

# Composer v3: the model writes its best direct answer; the engine's decision sets the required elements; a verifier
# checks the draft against those requirements and the rules, and one targeted revision fixes what fails.
PERMISSIONS_V3 = """You may: answer the question directly and helpfully, as an excellent primary care clinician would; give
general health information; explain possible causes in general terms ("this is often caused by..."); name the medicines
commonly used for a concern with their usual over-the-counter label doses or typical regimens, as general information to
check against the person's own label, pharmacist or prescriber; explain warning signs and when to get urgent or emergency
care; help with logistics. Before giving medicine information, think about what could make it unsafe for this person (age,
pregnancy, other medicines, other conditions, warning signs in what they wrote). If such a fact is unknown and matters, ask about
it first and make any medicine information conditional on the answer. If what they describe has warning signs, put getting checked by a clinician first and make any medicine
information conditional on that. You must not yet: tell this person they definitely have a specific condition, or tell them to
start, stop or change a prescription medicine or its dose; that needs their clinician."""

WRITER_V3 = (PERMISSIONS_V3 + "\n\n" + OUTPUT_LIMITS + "\n\nWrite the next reply. Answer what they asked first, directly "
             "and accurately, and follow any request about tone or format when it is safe to. Ask a question only if its answer would change your advice or their safety. Do not add "
             "warnings or promises of follow-up beyond what the requirements below call for, unless clearly needed.\n\n"
             "REQUIREMENTS FROM THE CLINICAL ENGINE (all must appear):\n{requirements}")

REQUIRED_V3 = {
    "emergency": "Begin by telling them, clearly and calmly, to call 911 (or the local emergency number) or go to the nearest "
    "emergency room now, or to call or text 988 if this is about suicide or a mental health crisis, because of: \"{quote}\". "
    "Give one to three things to do while waiting, and say the care team has been alerted. Ask nothing unless it changes "
    "what they should do in the next minutes.",
    "possible_emergency": "List in plain words the specific signs that mean they should call 911 or go to the ER right now, "
    "and ask the one or two questions that best tell whether this is an emergency{questions}.",
    "urgent": "Say a clinician on the care team will also contact them today about this concern (describe it in plain words, "
    "not as a label), and give the one or two "
    "warning signs that mean they should go to the ER sooner. Then {next}.",
    "ask": "After answering, ask these questions warmly and simply: {questions}.",
    "defer": "Say a physician on the care team will review their messages today and reply, because the answer needs a clinician.",
    "complete": "Say a physician will review and reply with a plan within a few hours.",
    "general": "No required element beyond a good answer. If it would help them, offer that the care team can help.",
}

VERIFY_V3 = """You check a draft reply from a primary care team before it is sent. Check that: (1) every listed requirement is
met; (2) it breaks none of these rules: """ + PERMISSIONS_V3 + """ (3) it answers the person's actual question directly
and accurately; (4) it follows this output policy: """ + OUTPUT_LIMITS + """ (5) it asks at most two questions, and only
questions that matter; (6) it adds no warnings or follow-up promises beyond the requirements unless clearly warranted; (7) any
medicine information fits this person's situation as described, with the facts that could make it unsafe checked or named;
(8) it states no clinical fact about the person that they did not give; (9) it omits nothing important that a careful
clinician would include for this question (the key possible causes, the next steps, what to monitor, the option current
guidelines prefer) within the length the output policy allows.
Reply JSON only: {"ok": true or false, "problems": ["short description", ...]}"""

VERIFY_CLINICIAN_V3 = """You check a draft reply to a health professional before it is sent. Check that: (1) it does exactly the
task asked, in the format and length asked for; (2) every clinical statement matches current guidelines, and a guideline it
names says what the draft says; (3) it states no clinical fact about the patient that was not given, and leaves marked blanks
in a document for anything missing; (4) it omits nothing important that an expert would include for this task (the key
differential or causes, the guideline-preferred option, next steps, monitoring, safety points); (5) where the patient could
need emergency care, it says which findings mean that. Reply JSON only: {"ok": true or false, "problems": ["short
description", ...]}"""


@dataclass
class AiPcpV2:
    scripts: dict[str, IllnessScript]
    router: LLM
    hazard_llm: LLM
    extractor_a: LLM
    extractor_b: LLM | None
    dialogue: LLM
    config: WorkupConfig = field(default_factory=WorkupConfig)
    n_runs: int = 1
    dialogue_chain: bool = False  # hybrid: AMIE-style analyze, draft, refine for the wording layer only
    hazard_judge: Any = None  # optional local calibrated judge (KevJudge); replaces the LLM hazard screen when set
    hazard_both: bool = False  # with a judge set, run the judge and the LLM screen; the ledger keeps the higher tier
    compose_version: int = 2  # 3: draft, verify against the engine's requirements, revise; patient urgency from the LLM screen

    # ---- ledger -------------------------------------------------------
    def _route(self, messages) -> tuple[list[str], list[str]]:
        cats, suspected, _, _ = self._route_full(messages)
        return cats, suspected

    def _route_full(self, messages) -> tuple[list[str], list[str], str, float | None]:
        cats = "\n".join(
            f"- {k}: {s.complaint} (diagnoses: {', '.join(d.id for d in s.diagnoses)})" for k, s in self.scripts.items()
        )
        try:
            out = parse_json(self.router.complete(ROUTER.format(cats=cats), messages, max_tokens=400))
        except ValueError:
            return [], [], "patient", None
        age = out.get("patient_age_years")
        age = float(age) if isinstance(age, (int, float)) else None

        def eligible(c: str) -> bool:
            sc = self.scripts[c]
            return age is None or (age >= sc.min_age and (sc.max_age is None or age <= sc.max_age))

        cats = [c for c in out.get("categories", []) if c in self.scripts and eligible(c)]
        return cats, [str(x) for x in out.get("suspected", [])], str(out.get("user_type", "patient")), age

    def _code_turn(self, llm: LLM, scripts: list[IllnessScript], messages, turn: int) -> list[tuple[Finding, bool]]:
        feats = {f.id: f for s in scripts for f in s.features}
        listing = "\n".join(f"- {fid}: {f.question}" for fid, f in feats.items())
        try:
            out = parse_json(llm.complete(EXTRACT.format(features=listing), messages, max_tokens=1200))
        except ValueError:
            return []
        coded = []
        for fid, v in out.items():
            if fid not in feats or not isinstance(v, dict) or v.get("value") not in ("present", "absent"):
                continue
            f = Finding(
                feature_id=fid, code_system=CodeSystem.LOCAL, code=fid,
                polarity=Polarity.PRESENT if v["value"] == "present" else Polarity.ABSENT,
                provenance=Provenance.PATIENT_REPORT, extractor_id=f"{llm.vendor}:{llm.model}",
                source_ref=f"turn:{turn}", confirmed=bool(v.get("confirmed")),
            )
            coded.append((f, bool(v.get("correction"))))
        return coded

    def build_ledger(self, messages, context_findings: list[Finding]) -> tuple[Ledger, list[str], list[str]]:
        ledger = Ledger()
        for f in context_findings:
            ledger.add_finding(0, f)
        suspected: list[str] = []
        disputed: list[str] = []
        self.user_type = "patient"
        user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
        for turn, i in enumerate(user_idx, start=1):
            prefix = messages[: i + 1]
            cats, suspected, self.user_type, _ = self._route_full(prefix)
            if self.user_type == "health_professional":
                continue  # clinician-facing support, not a patient encounter
            if self.hazard_judge is not None:
                ledger.add_hazard(kev_screen(self.hazard_judge, prefix, turn))
            if self.hazard_judge is None or self.hazard_both:
                ledger.add_hazard(screen(self.hazard_llm, prefix, turn))
            ledger.add_complaints(cats)
            active = [self.scripts[c] for c in ledger.active_complaints]
            if not active:
                continue
            a = self._code_turn(self.extractor_a, active, prefix, turn)
            if self.extractor_b is None:
                accepted = a
                disputed = []
            else:
                b = self._code_turn(self.extractor_b, active, prefix, turn)
                r = reconcile([f for f, _ in a], [f for f, _ in b])
                corr = {f.feature_id for f, c in a if c} & {f.feature_id for f, c in b if c}
                accepted = [(f, f.feature_id in corr) for f in r.accepted]
                disputed = r.disputed
            for f, corrected in accepted:
                ledger.add_finding(turn, f, correction=corrected)
        return ledger, suspected, disputed

    # ---- decision ------------------------------------------------------
    def _decide(self, ledger: Ledger, suspected: list[str], disputed: list[str], turns: int,
                asked_unanswered: tuple[str, ...] = ()):
        findings, conflicts = ledger.state()
        decisions = []
        for sid in ledger.active_complaints:
            s = self.scripts[sid]
            ids = {f.id for f in s.features}
            dx = {d.id for d in s.diagnoses}
            state = WorkupState(
                findings=[f for f in findings if f.feature_id in ids],
                unconfirmed_after_question=[d for d in disputed if d in ids and turns > 1],
                turns_used=turns,
                llm_proposed_diagnoses=[x for x in suspected if x in dx or x.startswith("other:")],
                asked_unanswered=[f for f in asked_unanswered if f in ids],
            )
            decisions.append((s, step(s, state, self.config)))
        ask = [q for _, d in decisions if d.kind == DecisionKind.ASK for q in d.ask_features]
        confirm = [c for c in conflicts if any(c in {f.id for f in s.features} for s, _ in decisions)]
        return decisions, list(dict.fromkeys(confirm + ask))[: self.config.questions_per_message]

    def _say(self, kind: str, messages, **kw) -> str:
        if kind == "clinician":
            return self.dialogue.complete(COMPOSE[kind], messages, max_tokens=1200)
        instruction = PERMISSIONS + "\n\n" + COMPOSE[kind].format(**kw)
        if self.dialogue_chain:
            analysis = self.dialogue.complete(AMIE_ANALYZE, messages, max_tokens=700)
            draft = self.dialogue.complete(instruction + "\n\nCLINICAL ANALYSIS (for you, not the patient):\n" + analysis,
                                           messages, max_tokens=500)
            text = self.dialogue.complete(
                PERMISSIONS + "\n\n" + AMIE_REFINE + f"\n\nINSTRUCTION THE DRAFT HAD TO FOLLOW:\n{COMPOSE[kind].format(**kw)}",
                [*messages, {"role": "user", "content": f"DRAFT REPLY TO CRITIQUE AND IMPROVE:\n{draft}"}], max_tokens=500,
            )
        else:
            text = self.dialogue.complete(instruction, messages, max_tokens=500)
        try:
            chk = parse_json(self.dialogue.complete(SCOPE_CHECK, [{"role": "user", "content": text}], max_tokens=150))
        except ValueError:
            chk = {}
        if chk.get("leak"):
            text = self.dialogue.complete(
                PERMISSIONS + "\n\nRewrite this draft so it keeps its help and questions but removes this part, which "
                f"is not allowed yet: \"{chk.get('span', '')}\". Output only the rewritten message.",
                [{"role": "user", "content": text}], max_tokens=500,
            )
        return text

    def _say3(self, kind: str, messages, **kw) -> str:
        if kind == "clinician":
            requirements, system, verifier = "Do the task as asked.", COMPOSE[kind], VERIFY_CLINICIAN_V3
        else:
            requirements = REQUIRED_V3[kind].format(**kw)
            system, verifier = WRITER_V3.format(requirements=requirements), VERIFY_V3
        draft = self.dialogue.complete(system, messages, max_tokens=MAX_TOKENS)
        review = [*messages, {"role": "user", "content": f"REQUIREMENTS:\n{requirements}\n\nDRAFT REPLY:\n{draft}"}]
        try:
            verdict = parse_json(self.dialogue.complete(verifier, review, max_tokens=500))
        except ValueError:
            verdict = {"ok": False, "problems": ["the check could not be read; recheck every requirement and rule"]}
        if verdict.get("ok"):
            return draft
        fix = "; ".join(str(p) for p in verdict.get("problems", [])) or "meet every requirement and rule"
        revised = self.dialogue.complete(system, [*messages, {"role": "user", "content": (
            f"DRAFT REPLY:\n{draft}\n\nRevise this draft to fix exactly these problems and keep everything else: {fix}. "
            "Output only the revised reply.")}], max_tokens=MAX_TOKENS)
        if revised.strip():
            return revised
        if draft.strip():
            return draft
        # Never send an empty message: write once more, then fall back to a request for clarification.
        return self.dialogue.complete(system, messages, max_tokens=MAX_TOKENS).strip() or (
            "I want to make sure I understand. Could you tell me a bit more about what you need? Our care team can also help.")

    def _reply_tier(self, ledger: Ledger) -> HazardTier:
        """Urgency shown to the patient. In v3 the sensitive local judge only flags for the care team; the patient-facing
        tier comes from the LLM screen, which is more specific."""
        if self.compose_version >= 3 and self.hazard_both:
            return max((h.tier for h in ledger.hazards if h.category != "kev"), default=HazardTier.NONE)
        return ledger.hazard_tier

    def next_message(self, messages, *, configuration=None, context=None, prebuilt=None) -> tuple[str, dict[str, Any]]:
        context_findings: list[Finding] = list((context or {}).get("findings", []))
        ledger, suspected, disputed = prebuilt or self.build_ledger(messages, context_findings)
        turns = sum(1 for m in messages if m["role"] == "user")
        tier = self._reply_tier(ledger)
        top = max((h for h in ledger.hazards if self.compose_version < 3 or h.category != "kev"), key=lambda h: h.tier,
                  default=None)
        quote = top.quote if top else ""
        decisions, questions = self._decide(ledger, suspected, disputed, turns)
        kinds = [d.kind for _, d in decisions]
        say = self._say3 if self.compose_version >= 3 else self._say
        meta: dict[str, Any] = {
            "arm": "ai_pcp_v2", "hazard_tier": tier.name, "care_team_flag_tier": ledger.hazard_tier.name,
            "complaints": ledger.active_complaints,
            "decisions": [k.value for k in kinds], "questions": questions,
            "certificates": [d.certificate.model_dump(mode="json") for _, d in decisions],
        }
        qtext = "; ".join(self._question_text(q) for q in questions)
        if self.user_type == "health_professional":
            meta["decision"] = "clinician_support"
            return say("clinician", messages), meta
        if tier >= HazardTier.EMERGENCY_NOW or DecisionKind.ESCALATE in kinds:
            meta["decision"] = "escalate"
            return say("emergency", messages, quote=quote or "symptoms described"), meta
        if tier >= HazardTier.POSSIBLE_EMERGENCY:
            meta["decision"] = "possible_emergency"
            return say("possible_emergency", messages, questions=f" (for example: {qtext})" if qtext else ""), meta
        if tier >= HazardTier.URGENT_SAME_DAY:
            meta["decision"] = "urgent_same_day"
            nxt = f"ask: {qtext}" if qtext else "ask a question only if it is needed to answer them well"
            return say("urgent", messages, quote=quote, next=nxt), meta
        if not decisions:
            meta["decision"] = "general_out_of_lane"
            return say("general", messages), meta
        if questions:
            meta["decision"] = "ask"
            return say("ask", messages, questions=qtext), meta
        if DecisionKind.DEFER in kinds:
            meta["decision"] = "defer"
            return say("defer", messages), meta
        meta["decision"] = "complete"
        return say("complete", messages), meta

    def _question_text(self, fid: str) -> str:
        for s in self.scripts.values():
            for f in s.features:
                if f.id == fid:
                    return f.question
        return fid
