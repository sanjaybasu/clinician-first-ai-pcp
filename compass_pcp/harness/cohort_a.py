"""Cohort A analysis: the hazard screens against physician labels (VVP-PCP-001 Section 8.5).

Primary (VE-A4): the frozen screen, which flags a message when the higher of the Kev and LLM tiers is routine_followup
or above. It passes with no missed Call 911/988 message and a lower 95% Wilson bound of safety-hazard sensitivity of at
least 0.80. Secondary, pre-specified: each screen alone, the deterministic floors, AUROC with bootstrap intervals,
positive predictive value, flags per 1,000 messages, net benefit, recall by hazard category, exact McNemar tests of
the frozen screen against the floors on positives, and coverage of care-coordination needs (VE-A5).
"""

import math
import random
from collections.abc import Sequence

from compass_pcp.harness.metrics import auroc, mcnemar_exact_p, net_benefit, per_1000, wilson
from compass_pcp.ledger import HazardTier

SCREENS = ("frozen", "kev_only", "llm_only", "floor_union", "compass_keyword", "compass_befast", "anchor_undertriage")
NET_BENEFIT_THRESHOLDS = (0.02, 0.05, 0.10)


def label(case: dict) -> dict:
    hazard = int(case.get("detection_truth", case.get("ground_truth_detection")) or 0) == 1
    cat = case.get("hazard_category")
    action = str(case.get("action_truth", case.get("ground_truth_action")))
    return {"safety_positive": (hazard and cat not in (None, "other_hazard", "none")) or action == "Call 911/988",
            "call_911": action == "Call 911/988", "coordination_positive": hazard and cat == "other_hazard",
            "category": cat if hazard else None}


def flags(kev_tier: int, llm_tier: int, floor: dict | None) -> dict:
    routine = HazardTier.ROUTINE_FOLLOWUP
    out = {"frozen": max(kev_tier, llm_tier) >= routine, "kev_only": kev_tier >= routine, "llm_only": llm_tier >= routine}
    if floor is not None:
        out.update({k: bool(floor[k]) for k in ("compass_keyword", "compass_befast", "anchor_undertriage")})
        out["floor_union"] = bool(floor["union"])
    return out


def _bootstrap_auroc(scores: Sequence[float], labels: Sequence[bool], n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng, n, vals = random.Random(seed), len(scores), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = auroc([scores[i] for i in idx], [labels[i] for i in idx])
        if not math.isnan(v):  # skip resamples with no positives or no negatives
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def screen_metrics(rows: list[dict], screen: str) -> dict:
    flagged = [r["flags"][screen] for r in rows]
    pos = [r["safety_positive"] for r in rows]
    tp = sum(f and p for f, p in zip(flagged, pos))
    fp = sum(f and not p for f, p in zip(flagged, pos))
    fn, tn = sum(pos) - tp, len(rows) - sum(pos) - fp
    c911 = [r for r in rows if r["call_911"]]
    caught_911 = sum(r["flags"][screen] for r in c911)
    return {
        "sensitivity": {"caught": tp, "positives": tp + fn, "estimate": tp / (tp + fn) if tp + fn else None,
                        "wilson95": list(wilson(tp, tp + fn))},
        "call_911": {"caught": caught_911, "positives": len(c911), "wilson95": list(wilson(caught_911, len(c911)))},
        "specificity": {"estimate": tn / (tn + fp) if tn + fp else None, "wilson95": list(wilson(tn, tn + fp))},
        "ppv": {"estimate": tp / (tp + fp) if tp + fp else None, "wilson95": list(wilson(tp, tp + fp))},
        "flags_per_1000": per_1000(tp + fp, len(rows)),
        "net_benefit": {str(t): net_benefit(tp, fp, len(rows), t) for t in NET_BENEFIT_THRESHOLDS},
    }


def analyze(rows: list[dict]) -> dict:
    """rows: one dict per message with label fields, 'flags', 'kev_urgent' (cumulative P >= urgent), 'llm_tier'."""
    screens = [s for s in SCREENS if all(s in r["flags"] for r in rows)]
    out: dict = {"n": len(rows), "screens": {s: screen_metrics(rows, s) for s in screens}}
    frozen = out["screens"]["frozen"]
    out["VE_A4"] = {"missed_call_911": frozen["call_911"]["positives"] - frozen["call_911"]["caught"],
                    "sensitivity_lower95": frozen["sensitivity"]["wilson95"][0]}
    out["VE_A4"]["pass"] = out["VE_A4"]["missed_call_911"] == 0 and out["VE_A4"]["sensitivity_lower95"] >= 0.80
    coord = [r for r in rows if r["coordination_positive"]]
    caught = sum(r["flags"]["frozen"] for r in coord)
    out["VE_A5"] = {"caught": caught, "positives": len(coord), "wilson95": list(wilson(caught, len(coord)))}
    pos = [r["safety_positive"] for r in rows]
    scores = {"kev_urgent_probability": [r["kev_urgent"] for r in rows], "llm_tier": [float(r["llm_tier"]) for r in rows],
              "frozen_max_tier": [max(r["kev_tier"], r["llm_tier"]) + r["kev_urgent"] / 10 for r in rows]}
    out["auroc_safety"] = {k: {"estimate": auroc(v, pos), "bootstrap95": _bootstrap_auroc(v, pos)} for k, v in scores.items()}
    cats: dict[str, list[bool]] = {}
    for r in rows:
        if r["safety_positive"] and r["category"]:
            cats.setdefault(r["category"], []).append(r["flags"]["frozen"])
    out["recall_by_category"] = {c: {"caught": sum(v), "positives": len(v)} for c, v in sorted(cats.items())}
    out["mcnemar_vs_frozen_on_positives"] = {}
    for s in screens:
        if s == "frozen":
            continue
        b = sum(1 for r in rows if r["safety_positive"] and r["flags"]["frozen"] and not r["flags"][s])
        c = sum(1 for r in rows if r["safety_positive"] and r["flags"][s] and not r["flags"]["frozen"])
        out["mcnemar_vs_frozen_on_positives"][s] = {"frozen_only": b, "other_only": c, "p_exact": mcnemar_exact_p(b, c)}
    return out
