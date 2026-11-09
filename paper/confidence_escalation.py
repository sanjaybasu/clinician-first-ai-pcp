"""Exploratory: escalation by stated confidence for the COMPASS hazard screens (suggested by B. Huynh).

The frozen screens (Kev and the Sonnet screen) set a hazard tier. Here the Sonnet screen also states the probability of
each tier after considering every tier (Tian et al., EMNLP 2023, found stated probabilities better calibrated than token
probabilities). When either screen gives emergency_now a probability of at least --theta (fixed in advance at 0.10)
and the frozen tier is below emergency_now, Opus reads the message with the same prompt. The final tier is the higher of
the frozen tier and the Opus tier, so a flag can be added or raised but never removed.

Reports, for the messages whose truth is Call 911/988, how many the frozen screens and the escalated screens place at
emergency_now, and the same for all other messages (the cost in false emergencies), with the share of messages sent to
Opus. Patient text goes only to the local Kev judge and to Anthropic (BAA). Nothing is written inside the repository.

Usage: python -m paper.confidence_escalation --cases <2,000-message file> --ids <cohort_a_tune_ids.json> --out <dir>
       [--theta 0.10] [--n 10] [--threads 4]
"""

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from compass_pcp.arms.hazard import _TIER, KEV_INSTRUCTIONS, KEV_TIERS, SCREEN, kev_tier, patient_text, screen
from compass_pcp.arms.llm import KevJudge, parse_json
from compass_pcp.harness import frozen
from compass_pcp.harness.cache import ResponseCache
from compass_pcp.harness.cohort_a import label
from compass_pcp.ledger import HazardTier
from scripts.custodian.run_cohort_a_sealed import message_fields
from scripts.run_healthbench import make_llm

ORDER = ["none", "routine_followup", "urgent_same_day", "possible_emergency", "emergency_now"]
CONFIDENCE_SCREEN = SCREEN.rsplit("Reply JSON only:", 1)[0] + (
    "Before you choose, consider each of the five tiers in turn and how likely it is to be the right one. Then give your "
    "probability, from 0 to 1, that each tier is the right one; the five probabilities sum to 1.\n"
    'Reply JSON only: {"probabilities": {"none": 0.0, "routine_followup": 0.0, "urgent_same_day": 0.0, '
    '"possible_emergency": 0.0, "emergency_now": 0.0}, "tier": "...", "category": "short_snake_case", '
    '"quote": "exact words copied from the patient", "rationale": "one sentence"}'
)
OPUS = "anthropic:claude-opus-5-5"


def stated(llm, text: str) -> tuple[dict[str, float], int]:
    """Stated tier probabilities and tier; an unreadable reply counts as urgent_same_day with no probabilities."""
    try:
        out = parse_json(llm.complete(CONFIDENCE_SCREEN, [{"role": "user", "content": text or "(no patient text)"}],
                                      max_tokens=400))
        probs = {t: float((out.get("probabilities") or {}).get(t, 0.0)) for t in ORDER}
        return probs, int(_TIER.get(str(out.get("tier")), HazardTier.URGENT_SAME_DAY))
    except (ValueError, TypeError):
        return {}, int(HazardTier.URGENT_SAME_DAY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if frozen.REPO in a.out.resolve().parents:
        raise SystemExit("--out must be outside the repository")
    keep = set(json.loads(a.ids.read_text()))
    cases = sorted((c for c in json.loads(a.cases.read_text()) if message_fields(c)[0] in keep),
                   key=lambda c: message_fields(c)[0])[: a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    judge, sonnet, opus = KevJudge(), make_llm(frozen.HAZARD_LLM), make_llm(OPUS)
    for llm in (sonnet, opus):
        llm.phi_allowed = True  # Anthropic is under the Waymark BAA
    cache = {n: ResponseCache(a.out / f"cache_{n}.jsonl") for n in ("kev", "frozen_llm", "stated", "opus")}

    def one(c: dict) -> dict:
        cid, msg = message_fields(c)
        msgs = [{"role": "user", "content": msg}]
        text = patient_text(msgs)
        kp = cache["kev"].get_or_call({"id": cid}, lambda: judge.choice(text, KEV_INSTRUCTIONS, KEV_TIERS))
        llm_tier = cache["frozen_llm"].get_or_call({"id": cid}, lambda: int(screen(sonnet, msgs, 1).tier))
        base = max(int(kev_tier(kp)), llm_tier)
        sp, _ = cache["stated"].get_or_call({"id": cid}, lambda: list(stated(sonnet, text)))
        p = max(kp.get("emergency_now", 0.0), sp.get("emergency_now", 0.0) if sp else 0.0)
        escalated = base < HazardTier.EMERGENCY_NOW and p >= a.theta
        final = base
        if escalated:
            _, ot = cache["opus"].get_or_call({"id": cid}, lambda: list(stated(opus, text)))
            final = max(base, ot)
        return {"id": cid, "base_tier": base, "final_tier": final, "p_emergency_kev": kp.get("emergency_now", 0.0),
                "p_emergency_stated": sp.get("emergency_now") if sp else None, "escalated": escalated, **label(c)}

    with cf.ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(one, cases))
    (a.out / "per_message.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    em = int(HazardTier.EMERGENCY_NOW)
    pos = [r for r in rows if r["call_911"]]
    neg = [r for r in rows if not r["call_911"]]
    report = {
        "n": len(rows), "theta": a.theta, "escalated_to_opus": sum(r["escalated"] for r in rows),
        "call_911": len(pos),
        "call_911_at_emergency_frozen": sum(r["base_tier"] >= em for r in pos),
        "call_911_at_emergency_escalated": sum(r["final_tier"] >= em for r in pos),
        "other_at_emergency_frozen": sum(r["base_tier"] >= em for r in neg),
        "other_at_emergency_escalated": sum(r["final_tier"] >= em for r in neg),
        "flagged_frozen": sum(r["base_tier"] >= HazardTier.ROUTINE_FOLLOWUP for r in rows),
        "flagged_escalated": sum(r["final_tier"] >= HazardTier.ROUTINE_FOLLOWUP for r in rows),
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
