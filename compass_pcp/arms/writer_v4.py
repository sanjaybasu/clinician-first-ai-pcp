"""COMPASS v4 writer and extended reply check (v4_spec.md A1, A2, A3).

A1-c ("contract"): the LLM writes the full reply from the contract. A1-d ("edit"): the LLM minimally edits the
no-engine draft so it satisfies the contract. Either way the reply then goes through the A2 check: the deterministic
rules in ``compass_pcp.contract`` plus one semantic LLM call. A failing reply gets at most one revision; if it still
fails, the signed fallback from the policy file is sent. A3 is a compact response policy for non-engine patient
replies, off by default.
"""

import json
from typing import Any

from compass_pcp.arms.ai_pcp_v2 import PERMISSIONS_V3, REQUIRED_V3, VERIFY_V3, WRITER_V3
from compass_pcp.arms.baselines import MAX_TOKENS, OUTPUT_LIMITS
from compass_pcp.arms.llm import LLM, parse_json
from compass_pcp.contract import ResponseContract, deterministic_check
from compass_pcp.policy_v4 import ContractPolicy

SHAPES = {
    "emergency": "Make the REQUIRED CLAUSE your first sentence, word for word. Ask no question before it. Then give one "
    "to three things to do right now. Ask nothing unless the answer changes what they should do in the next minutes.",
    "ask": "First give a short, useful answer in general terms, conditional where it depends on what you are about to "
    "ask. Then ask the SELECTED QUESTION and end the reply with it. Ask no other question.",
    "ask_open": "First give a short, useful answer in general terms. Then ask the one question whose answer most changes "
    "what they should do now, and end the reply with it.",
    "defer": "Say plainly that this needs a clinician and why, in one sentence. Give the general information you are "
    "permitted to give. State the handoff with the REQUIRED CLAUSE.",
    "answer": "Answer what they asked directly and helpfully in general terms. Do not imply that the workup is complete "
    "or that nothing could be wrong; say what would change the advice.",
}

# Paper 2 development ("contract_full"): fuller emergency, ask and defer replies; the required clause, its position and
# the selected question are unchanged. The v4 shapes above stay as registered.
SHAPES_FULL = {
    **SHAPES,
    "emergency": "Make the REQUIRED CLAUSE your first sentence, word for word. Ask no question before it. Then say "
    "briefly why this needs emergency care now, answer what they asked, and say what to do while waiting for help. "
    "Ask nothing that would delay the action.",
    "ask": "First give a complete, useful answer in general terms, conditional where it depends on what you are about "
    "to ask. Then ask the SELECTED QUESTION, word for word, and end the reply with it. Ask no other question.",
    "ask_open": "First give a complete, useful answer in general terms. Then ask the one question whose answer most "
    "changes what they should do now, and end the reply with it.",
    "defer": "Say plainly that this needs a clinician and why, in one sentence. Then give all the general information "
    "you are permitted to give that helps them now (what to watch for, what they can safely do meanwhile). State the "
    "handoff with the REQUIRED CLAUSE.",
}

# Paper 2 development ("contract_v9"): as contract_full, and emergency and defer replies explain the possible causes in
# general terms, as a clinician would, without telling the person they have one. Signed S. Basu 2026-11-05.
SHAPES_V9 = {
    **SHAPES_FULL,
    "emergency": "Make the REQUIRED CLAUSE your first sentence, word for word. Ask no question before it. Then explain "
    "why this needs emergency care now: name, in general terms, the serious causes that clinicians check for with these "
    "symptoms, without saying the person has one. Answer what they asked, and say what to do safely while getting help. "
    "Ask nothing that would delay the action.",
    "defer": "Say plainly that this needs a clinician and why, in one sentence. Then explain, in general terms, the common "
    "and the serious causes a clinician will consider, without saying the person has one; what to watch for; and what "
    "they can safely do meanwhile. State the handoff with the REQUIRED CLAUSE.",
}
SHAPE_SETS = {"contract": SHAPES, "edit": SHAPES, "contract_full": SHAPES_FULL, "contract_v9": SHAPES_V9}

WRITER_V4 = (PERMISSIONS_V3 + "\n\n" + OUTPUT_LIMITS + "\n\nYou write the next reply under a response contract from the "
             "clinical engine. The contract decides what the reply must do; you choose the words.\n\nREPLY SHAPE:\n{shape}"
             "\n\nCONTRACT:\n{contract}\n\nRules: copy every REQUIRED CLAUSE exactly, word for word. Add no clinical "
             "thresholds, numbers, diagnoses or medicine changes beyond the contract and general label information. "
             "Never say the care team was alerted, paged or notified unless the contract says a page was executed. "
             "Do not avoid the contract's prohibited content by hinting at it.")

EDIT_V4 = (WRITER_V4 + "\n\nBelow the conversation you get a DRAFT written without the clinical engine. Edit it as "
           "little as possible so it satisfies the contract and the reply shape: keep its wording where it already "
           "complies, put each required clause in its required place, remove what the contract prohibits. Output only "
           "the edited reply.")

VERIFY_V4 = ("You check a draft reply from a primary care team against its response contract before it is sent.\n"
             "CONTRACT:\n{contract}\n\nReply shape required:\n{shape}\n\nCheck: (1) within_contract: every clinical "
             "statement stays within the contract and general health information; (2) useful_content: when the "
             "contract permits an answer, the draft gives a useful one; (3) no_contradictory_reassurance: nothing "
             "softens or contradicts the required clause or the disposition; (4) no_diagnosis: it does not tell the "
             "person they have a specific condition; (5) no_prescription: it does not tell them to start, stop or "
             "change a prescription medicine or dose. Reply JSON only: {{\"ok\": true or false, \"checks\": "
             "{{\"within_contract\": bool, \"useful_content\": bool, \"no_contradictory_reassurance\": bool, "
             "\"no_diagnosis\": bool, \"no_prescription\": bool}}, \"problems\": [\"short description\", ...]}}")

SEMANTIC_CHECKS = ("within_contract", "useful_content", "no_contradictory_reassurance", "no_diagnosis", "no_prescription")

COMPACT_POLICY = ("Response policy: answer directly when you have enough context. Otherwise give a bounded, useful "
                  "answer and ask the one question you most need. If anything they describe could be an emergency, put "
                  "the advice to call 911 or go to the emergency room first, before any question. Add no clinical "
                  "thresholds or numbers that are not general label information. No filler, no repeated warnings.")


def contract_text(c: ResponseContract) -> str:
    lines = [f"state: {c.state}", f"assessment still open: {'yes' if c.assessment_open else 'no'}"]
    if c.referral_text:
        where = "as the first sentence, before any question" if c.referral_position == "first" else "anywhere"
        lines.append(f'REQUIRED CLAUSE ({where}): "{c.referral_text}"')
    for a in c.additional_clauses:
        lines.append(f'REQUIRED CLAUSE (anywhere): "{a}"')
    if c.selected_question:
        lines.append(f'SELECTED QUESTION: "{c.selected_question}"')
    for s in c.conditional_statements:
        lines.append(f'PERMITTED CONDITIONAL STATEMENT: "{s}"')
    lines.append("page executed: " + ("yes" if c.page_executed else "no"))
    lines.append("prohibited: " + "; ".join(c.prohibited))
    return "\n".join(lines)


def shape_for(c: ResponseContract, shapes: dict = SHAPES) -> str:
    return shapes["ask_open"] if c.state == "ask" and not c.selected_question else shapes[c.state]


def _only_question_missing(check: dict) -> bool:
    return (check["semantic_ok"] and bool(check["problems"])
            and all(str(p).startswith("the selected question is missing") for p in check["problems"]))


def semantic_check(llm: LLM, messages, contract: ResponseContract, draft: str,
                   shapes: dict = SHAPES) -> tuple[bool, list[str], dict]:
    review = [*messages, {"role": "user", "content": f"DRAFT REPLY:\n{draft}"}]
    try:
        out = parse_json(llm.complete(VERIFY_V4.format(contract=contract_text(contract), shape=shape_for(contract, shapes)),
                                      review, max_tokens=500))
    except (ValueError, json.JSONDecodeError):
        return False, ["the check could not be read; recheck every requirement and rule"], {}
    checks = out.get("checks") if isinstance(out.get("checks"), dict) else {}
    failed = [k for k in SEMANTIC_CHECKS if checks.get(k) is False]
    ok = bool(out.get("ok")) and not failed
    problems = [str(p) for p in out.get("problems", []) if p] or ([f"failed: {', '.join(failed)}"] if not ok else [])
    return ok, problems, checks


def _check(llm, messages, contract, policy, text, shapes: dict = SHAPES) -> dict:
    det = deterministic_check(text, contract, policy)
    sem_ok, sem_problems, checks = semantic_check(llm, messages, contract, text, shapes)
    return {"deterministic_ok": not det, "semantic_ok": sem_ok, "ok": not det and sem_ok,
            "problems": det + ([] if sem_ok else sem_problems), "semantic_checks": checks}


def no_engine_draft(llm: LLM, messages) -> str:
    """The v3 general-path first draft: the reply the pipeline would write with no engine decision."""
    return llm.complete(WRITER_V3.format(requirements=REQUIRED_V3["general"]), messages, max_tokens=MAX_TOKENS)


def write_under_contract(llm: LLM, messages, contract: ResponseContract, policy: ContractPolicy, *,
                         variant: str = "contract") -> tuple[str, dict[str, Any]]:
    """Write, check, revise once, then fall back. Returns the reply and a compliance log."""
    if variant not in SHAPE_SETS:
        raise ValueError(f"unknown writer variant {variant}")
    shapes = SHAPE_SETS[variant]
    system = WRITER_V4.format(shape=shape_for(contract, shapes), contract=contract_text(contract))
    log: dict[str, Any] = {"variant": variant, "first_pass": None, "post_revision": None, "revised": False,
                           "fallback": False}
    if variant == "edit":
        draft0 = no_engine_draft(llm, messages)
        log["no_engine_draft"] = draft0
        edit = EDIT_V4.format(shape=shape_for(contract), contract=contract_text(contract))
        first = llm.complete(edit, [*messages, {"role": "user", "content": f"DRAFT TO EDIT:\n{draft0}"}], max_tokens=MAX_TOKENS)
    else:
        first = llm.complete(system, messages, max_tokens=MAX_TOKENS)
    log["first_pass"] = _check(llm, messages, contract, policy, first, shapes)
    if log["first_pass"]["ok"]:
        return first, log
    fix = "; ".join(log["first_pass"]["problems"]) or "meet every requirement in the contract"
    revised = llm.complete(system, [*messages, {"role": "user", "content": (
        f"DRAFT REPLY:\n{first}\n\nRevise this draft to fix exactly these problems and keep everything else: {fix}. "
        "Output only the revised reply.")}], max_tokens=MAX_TOKENS)
    log["revised"] = True
    log["post_revision"] = _check(llm, messages, contract, policy, revised, shapes)
    if log["post_revision"]["ok"]:
        return revised, log
    if variant in ("contract_full", "contract_v9") and contract.selected_question and _only_question_missing(log["post_revision"]):
        # the engine's question is fixed text: append it rather than discard a reply that passed every other check
        repaired = revised.rstrip() + "\n\n" + contract.selected_question
        log["question_appended"] = True
        log["post_repair"] = _check(llm, messages, contract, policy, repaired, shapes)
        if log["post_repair"]["ok"]:
            return repaired, log
    log["fallback"] = True
    return contract.fallback_text, log


def compose_v3_audited(llm: LLM, kind: str, messages, *, compact: bool = False) -> tuple[str, dict[str, Any]]:
    """The v3 composer (AiPcpV2._say3) for the clinician and general paths, with the same prompts and calls, returning
    which stages changed the draft (A3 stage-level audit). ``compact`` swaps in the A3 response policy for patients."""
    from compass_pcp.arms.ai_pcp_v2 import COMPOSE, VERIFY_CLINICIAN_V3

    if kind == "clinician":
        requirements, system, verifier = "Do the task as asked.", COMPOSE[kind], VERIFY_CLINICIAN_V3
    else:
        requirements = REQUIRED_V3[kind]
        system, verifier = WRITER_V3.format(requirements=requirements), VERIFY_V3
        if compact:
            system = system + "\n\n" + COMPACT_POLICY
    audit: dict[str, Any] = {"kind": kind, "compact_policy": compact, "verify_ok": None, "revised": False,
                             "revision_changed_text": False}
    draft = llm.complete(system, messages, max_tokens=MAX_TOKENS)
    review = [*messages, {"role": "user", "content": f"REQUIREMENTS:\n{requirements}\n\nDRAFT REPLY:\n{draft}"}]
    try:
        verdict = parse_json(llm.complete(verifier, review, max_tokens=500))
    except ValueError:
        verdict = {"ok": False, "problems": ["the check could not be read; recheck every requirement and rule"]}
    audit["verify_ok"] = bool(verdict.get("ok"))
    if verdict.get("ok"):
        return draft, audit
    fix = "; ".join(str(p) for p in verdict.get("problems", [])) or "meet every requirement and rule"
    audit["problems"] = fix
    revised = llm.complete(system, [*messages, {"role": "user", "content": (
        f"DRAFT REPLY:\n{draft}\n\nRevise this draft to fix exactly these problems and keep everything else: {fix}. "
        "Output only the revised reply.")}], max_tokens=MAX_TOKENS)
    audit["revised"] = True
    if revised.strip():
        audit["revision_changed_text"] = revised.strip() != draft.strip()
        return revised, audit
    if draft.strip():
        return draft, audit
    return llm.complete(system, messages, max_tokens=MAX_TOKENS).strip() or (
        "I want to make sure I understand. Could you tell me a bit more about what you need? Our care team can also help."), audit
