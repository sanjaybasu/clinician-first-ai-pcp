"""Second-classifier agreement for the reply-urgency classifier (Paper 2 real-text test; post hoc, 2026-11-09).

The real-text measures use Claude Sonnet 5.5 (paper.sealed_comparators.CLASSIFIER_PROMPT) to read each reply alone and
label the care it advises. No physician validation of that classifier exists yet. This script re-labels a stratified
random sample (200 replies per arm, seed 20261108) with GPT-6.1 Sol (low reasoning, same prompt; OpenAI under BAA) and
reports agreement on the binary escalation used by the paper (routine_followup or higher) and on the 5-level scale.
Aggregates only. Out: paper2_final/texts/urgency_g61_sample.jsonl and paper2_final/derived/urgency_agreement.json.
Usage: python -m paper.urgency_agreement [--per-arm 200] [--threads 16]
"""

import argparse
import concurrent.futures as cf
import json
import random
import threading
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
T = D / "paper2_final/texts"


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main() -> None:
    import numpy as np

    from compass_pcp.arms.openai_reasoning import OpenAIReasoningLLM
    from paper.sealed_comparators import CLASSIFIER_PROMPT, URGENCY, parse_urgency

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=200)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    reps = _rows(T / "replies.jsonl")
    rng = random.Random(20261108)
    sample = []
    for arm in sorted({r["arm"] for r in reps}):
        rs = [r for r in reps if r["arm"] == arm]
        sample += rng.sample(rs, min(a.per_arm, len(rs)))
    out = T / "urgency_g61_sample.jsonl"
    done = {(r["prompt_id"], r["repeat"], r["arm"]) for r in _rows(out)}
    lock, local = threading.Lock(), threading.local()

    def one(r):
        if (r["prompt_id"], r["repeat"], r["arm"]) in done:
            return
        if not hasattr(local, "c"):
            local.c = OpenAIReasoningLLM("gpt-6.1-sol", reasoning_effort="low", reasoning_headroom=4000, phi_allowed=True)
        u = None
        for _ in range(2):
            u = parse_urgency(local.c.complete(CLASSIFIER_PROMPT, [{"role": "user", "content": "REPLY:\n" + (r["text"] or "")}],
                                               max_tokens=200))
            if u:
                break
        with lock, out.open("a") as fh:
            fh.write(json.dumps({"prompt_id": r["prompt_id"], "repeat": r["repeat"], "arm": r["arm"], "urgency": u}) + "\n")

    with cf.ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, sample))
    son = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(T / "urgency.jsonl")}
    g = {(u["prompt_id"], u["repeat"], u["arm"]): u["urgency"] for u in _rows(out)}

    def kappa(x, y):
        x, y = np.array(x), np.array(y)
        po = (x == y).mean()
        cats = sorted(set(x) | set(y))
        pe = sum((x == c).mean() * (y == c).mean() for c in cats)
        return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

    res = {}
    for arm in sorted({k[2] for k in g} | {"_all"}):
        keys = [k for k in g if (arm == "_all" or k[2] == arm) and g[k] and son.get(k)]
        s = [URGENCY.index(son[k]) for k in keys]
        o = [URGENCY.index(g[k]) for k in keys]
        sb, ob = [v >= 1 for v in s], [v >= 1 for v in o]
        res[arm] = {"n": len(keys), "escalation_agreement": round(float(np.mean(np.array(sb) == np.array(ob))), 3),
                    "escalation_kappa": round(kappa(sb, ob), 3), "five_level_exact": round(float(np.mean(np.array(s) == np.array(o))), 3),
                    "five_level_kappa": round(kappa(s, o), 3),
                    "sonnet_escalation_rate": round(float(np.mean(sb)), 3), "g61_escalation_rate": round(float(np.mean(ob)), 3)}
    res["_note"] = "Sonnet 5.5 = classifier used in the paper; GPT-6.1 Sol low reasoning = second classifier; same prompt."
    (D / "paper2_final/derived/urgency_agreement.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
