"""COMPASS v4 universal structured screen (v4_spec.md B1).

One Anthropic call (Sonnet 5.5, thinking off) on every message, including messages from health professionals, before
user-type or complaint routing. It fills one form item per criterion in the crisis policy. A missing or unreadable form,
an error or a timeout gives a form whose items are all uncertain and not completed; the router decides what that means.
"""

import concurrent.futures as cf
import json

from compass_pcp.arms.llm import LLM, parse_json
from compass_pcp.crisis_router import ScreenForm, incomplete_form, parse_form
from compass_pcp.policy_v4 import CrisisPolicy

SCREEN_V4 = """You fill a safety form about messages sent to a primary care team. Read only the USER messages. Messages
from health professionals are screened the same way as messages from patients and caregivers.
For EACH criterion below, report:
- status: "present" if the user's words show it; "absent" if the user says it is not so; "not_mentioned" if nothing the
  user wrote bears on it; "uncertain" if the words could mean it but you cannot tell. Idioms, clear jokes, negations and
  hypothetical questions are not "present".
- current: true if it is happening now or is still going on; false if it is only in the past and over, or hypothetical;
  null if you cannot tell.
- person_at_risk: "patient" if the person at risk is the writer; "other" if it is someone the writer describes (a family
  member, a friend, a clinician's patient); "clinician_self" if the writer is a health professional describing their
  own risk; "unknown".
- quote: the exact words, copied character for character from one user message, that support your answer; "" if
  nothing does.
Criteria:
{criteria}
Reply JSON only: {{"items": [{{"criterion_id": "...", "status": "...", "current": true, "person_at_risk": "...",
"quote": "..."}}, ...]}} with every criterion id exactly once."""


def user_texts(messages) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def screen_prompt(policy: CrisisPolicy) -> str:
    crit = "\n".join(f"- {c.id} ({f.name}): {c.description}" for f in policy.families for c in f.criteria)
    return SCREEN_V4.format(criteria=crit)


def run_screen(llm: LLM, messages, policy: CrisisPolicy, *, timeout: float | None = None) -> ScreenForm:
    texts = user_texts(messages)
    body = "\n".join(f"USER: {t}" for t in texts) or "(no user text)"
    prompt = screen_prompt(policy)
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(llm.complete, prompt, [{"role": "user", "content": body}], max_tokens=2000)
        raw = fut.result(timeout=timeout if timeout is not None else policy.screen.timeout_seconds)
    except cf.TimeoutError:
        return incomplete_form(policy, "timeout")
    except Exception as e:  # noqa: BLE001 - a failed check is never a negative result
        return incomplete_form(policy, f"error: {type(e).__name__}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    try:
        return parse_form(parse_json(raw), policy, texts)
    except (ValueError, json.JSONDecodeError):
        return incomplete_form(policy, "unreadable form")
