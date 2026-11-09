"""Development scorer for the conformal ask-or-answer rule: calibrated hazard probabilities from local and BAA models.

Scores each development message with
  1. Kev (local System-1 judge, this machine only): the five-tier hazard distribution (compass_pcp.arms.hazard.KEV_TIERS),
     who is writing (patient, caregiver, clinician, other), and whether key facts are missing;
  2. GPT-4.1 (OpenAI, BAA): P(yes) from token log-probabilities for "a clinician should see this" and for "emergency now".
Writes <out>/scores.jsonl (ids and numbers only; no text). Resumable. Development data only: the BMC MIDM 2,000-message
set and the calibration set. Never run on the fresh test texts until the rule is frozen.

Usage: python -m paper.score_ensemble --cases <realworld_cases_n2000_clean.json> --out <dir> [--threads 8]
"""

import argparse
import concurrent.futures as cf
import json
import math
import threading
from pathlib import Path

from compass_pcp.arms.hazard import KEV_INSTRUCTIONS, KEV_TIERS
from compass_pcp.arms.llm import KevJudge

SENDER = {"patient": "The patient, writing about their own health",
          "caregiver": "A family member or caregiver, writing about someone else",
          "clinician": "A health professional, asking about a patient or a clinical question",
          "other": "Someone else, or a message with no health content"}
CONTEXT = {"enough": "The message gives enough facts to decide how urgent it is",
           "missing": "How urgent it is depends on facts the message does not give (how long, how bad, which symptoms, age)"}
GPT_SPEC = "gpt-4.1-2025-04-14"
GPT_Q = {"see": "Should a clinician look at this message (a symptom, medication problem, fall, mental health concern, "
                "substance use, pregnancy concern, violence or unsafe housing)? Answer yes or no.",
         "emergency": "Does the person who wrote this message need emergency care (911, 988 or an emergency room) right "
                      "now? Answer yes or no."}


def gpt_p_yes(client, question: str, text: str) -> float | None:
    r = client.chat.completions.create(model=GPT_SPEC, max_tokens=1, logprobs=True, top_logprobs=10, temperature=0,
                                       messages=[{"role": "system", "content": "You read one text message that a "
                                                  "person sent to a primary care team. " + question},
                                                 {"role": "user", "content": text}])
    tops = r.choices[0].logprobs.content[0].top_logprobs
    p = {"yes": 0.0, "no": 0.0}
    for t in tops:
        k = t.token.strip().lower()
        if k in p:
            p[k] += math.exp(t.logprob)
    s = p["yes"] + p["no"]
    return p["yes"] / s if s > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import openai

    rows = json.loads(a.cases.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "scores.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
    todo = [r for r in rows if str(r["message_id"]) not in done]
    lock, local = threading.Lock(), threading.local()

    def one(r):
        if not hasattr(local, "k"):
            local.k, local.o = KevJudge(), openai.OpenAI(timeout=120.0, max_retries=4)
        text = r["prompt"] if isinstance(r["prompt"], str) else r["prompt"][-1]["content"]
        try:
            rec = {"id": str(r["message_id"]),
                   "kev_tier": local.k.choice(text, KEV_INSTRUCTIONS, KEV_TIERS),
                   "kev_sender": local.k.choice(text, "Who wrote this message?", SENDER),
                   "kev_context": local.k.choice(text, "Can a clinician decide how urgent this is from the message alone?",
                                                 CONTEXT),
                   **{f"gpt_{k}": gpt_p_yes(local.o, q, text) for k, q in GPT_Q.items()}}
        except Exception as err:  # noqa: BLE001 - the type only; never the message text
            print("error", type(err).__name__, flush=True)
            return 1
        with lock, path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    with cf.ThreadPoolExecutor(a.threads) as ex:
        failed = sum(ex.map(one, todo))
    print(json.dumps({"scored": len(done) + len(todo) - failed, "failed": failed}))


if __name__ == "__main__":
    main()
