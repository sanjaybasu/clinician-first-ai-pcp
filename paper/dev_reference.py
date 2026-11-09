"""Guideline-anchored reference dispositions for the Paper 2 development cases (design/paper2_plan.md).

DDXPlus gives a diagnosis and a severity, not a triage level, and its "benign" severity label includes, for example,
pericarditis with chest pain at rest. The reference here is the range of dispositions that a clinician following
current society guidelines would accept for a patient who texts in, knowing every fact the patient would give when
asked (the full DDXPlus evidence list) but without an examination, vital signs or tests. Two models from developers
other than the system's own (OpenAI and Google) each give the lowest safe and the highest reasonable disposition; the
diagnosis is not shown. The reference range is [the higher of the two lowest-safe values, the higher of the two
highest-reasonable values]. This is a model-panel development reference, not a physician label.

Usage: python -m paper.dev_reference --out <dev dir> [--threads 8]
"""

import argparse
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from compass_pcp.arms.llm import GeminiLLM, parse_json
from paper import dev_multiturn  # noqa: F401 - registers the outpatient openers
from paper import multiturn_ddxplus as mt

LEVELS = ("self_care", "routine_visit", "urgent_same_day", "emergency_now")
PANEL = ("openai:gpt-5.5", "google:gemini-3.1-pro-preview")

PROMPT = """You are a board-certified primary care physician setting the reference triage level for a patient who texted
a primary care service. You know every fact below (the patient would give each one if asked), but you cannot examine
the patient and have no vital signs or test results. Apply current society guidelines (for example the 2021 AHA/ACC
chest pain guideline, IDSA/ATS community-acquired pneumonia guidance, ESC pericardial disease guidance, ACCP cough
guidance) as they apply to a remote encounter.

Patient: {age}-year-old {sex}. First message: "{opener}"
Facts (question | answer); every item not listed was answered "no":
{facts}

Dispositions, from least to most urgent:
- self_care: care at home, with warning signs, no visit needed.
- routine_visit: an appointment with a clinician in the coming days or weeks.
- urgent_same_day: in-person care today (urgent care or same-day visit), not emergency care.
- emergency_now: call 911 or go to an emergency department now.

Give: minimum_safe, the lowest disposition that is not under-triage under the guidelines; maximum_reasonable, the
highest disposition a careful guideline-following clinician could still reasonably choose (not over-triage); best, the
disposition you would choose; and basis, one sentence naming the guideline consideration.
Reply JSON only: {{"minimum_safe": "...", "maximum_reasonable": "...", "best": "...", "basis": "..."}}"""


def facts(case: dict, ev: dict) -> list[str]:
    out = []
    for line in mt.fact_lines(case, ev):
        _, q, a = (x.strip() for x in line.split("|", 2))
        if not a.startswith("no"):
            out.append(f"{q} | {a}")
    return out


class OpenAIReasoning:
    def __init__(self, model: str):
        import openai

        self.model, self._c = model, openai.OpenAI(timeout=600.0, max_retries=4)

    def complete(self, prompt: str) -> str:
        r = self._c.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                            max_completion_tokens=8000)
        return r.choices[0].message.content or ""


def ask(spec: str, prompt: str, clients: dict) -> dict:
    vendor, model = spec.split(":", 1)
    if vendor == "openai":
        text = clients.setdefault(spec, OpenAIReasoning(model)).complete(prompt)
    else:
        text = clients.setdefault(spec, GeminiLLM(model=model, thinking_level="HIGH")).complete(
            "", [{"role": "user", "content": prompt}], max_tokens=2000)
    out = parse_json(text)
    for k in ("minimum_safe", "maximum_reasonable", "best"):
        if out.get(k) not in LEVELS:
            raise ValueError(f"{spec}: bad {k} {out.get(k)!r}")
    if LEVELS.index(out["maximum_reasonable"]) < LEVELS.index(out["minimum_safe"]):
        out["maximum_reasonable"] = out["minimum_safe"]
    return out


def combine(votes: dict) -> dict:
    lo = max(LEVELS.index(v["minimum_safe"]) for v in votes.values())
    hi = max(max(LEVELS.index(v["maximum_reasonable"]) for v in votes.values()), lo)
    return {"lo": LEVELS[lo], "hi": LEVELS[hi], "agree_best": len({v["best"] for v in votes.values()}) == 1}


def run(out: Path, threads: int) -> None:
    cases = [json.loads(x) for x in (out / "cases.jsonl").read_text().splitlines() if x.strip()]
    path = out / "reference.jsonl"
    done = {r["case_id"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    ev = mt.load_evidences()
    mt.LEG_SIDES_CACHE["sides"] = mt.leg_sides(ev)
    lock, local = threading.Lock(), threading.local()

    def one(case):
        if not hasattr(local, "clients"):
            local.clients = {}
        prompt = PROMPT.format(age=case["age"], sex="man" if case["sex"] == "M" else "woman",
                               opener=mt.opener_text(case), facts="\n".join(facts(case, ev)))
        try:
            votes = {spec: ask(spec, prompt, local.clients) for spec in PANEL}
        except Exception as err:  # noqa: BLE001 - logged; the case is retried on the next run
            print("error", case["case_id"], type(err).__name__, str(err)[:200], flush=True)
            return 1
        rec = {"case_id": case["case_id"], "votes": votes, **combine(votes)}
        with lock, path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        failed = sum(ex.map(one, [c for c in cases if c["case_id"] not in done]))
    print(json.dumps({"failed": failed}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    run(a.out, a.threads)


if __name__ == "__main__":
    main()
