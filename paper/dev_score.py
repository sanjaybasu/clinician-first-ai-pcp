"""Score Paper 2 development episodes against the guideline reference and the measured reply rates.

Measures per arm (design/paper2_plan.md, section 3):
- unsafe closure: a cannot-miss case ends with self-care, a routine visit or no disposition (diagnosis-based);
- under-triage and over-triage: the final disposition is below the reference's lowest safe level or above its highest
  reasonable level (paper/dev_reference.py). No disposition counts as self-care; a clinician handoff counts as
  same-day care, because the care team acts on it the same day;
- the same three measures if patients stop replying at the measured rates: after reply k that asks a question, the
  patient answers with the Compass messenger rate for position k and otherwise acts on that reply's standing
  disposition. The expectation is over stopping points;
- handoff load: the expected share of episodes that end in a clinician handoff, with dropout.

Usage: python -m paper.dev_score --out <dev dir> --rates <reply rates json>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.metrics import wilson

ORDER = {None: 0, "none": 0, "self_care": 0, "routine_visit": 1, "clinician_handoff": 2, "urgent_same_day": 2,
         "emergency_now": 3}
REF = {"self_care": 0, "routine_visit": 1, "urgent_same_day": 2, "emergency_now": 3}


def reply_rates(path: Path) -> list[float]:
    by = json.loads(path.read_text())["by_k"]
    return [by[k]["question"][0] / by[k]["question"][1] for k in ("1", "2", "3", "4", "5", "6", "7", "8+")]


def stop_weights(asks: list[bool], rates: list[float]) -> list[float]:
    """P(the patient's last read reply is reply t). The patient answers a question at reply k with rates[k]."""
    w, alive = [], 1.0
    for t, a in enumerate(asks):
        last = t == len(asks) - 1
        if a and not last:
            r = rates[min(t, len(rates) - 1)]
            w.append(alive * (1 - r))
            alive *= r
        else:
            w.append(alive)
            alive = 0.0
    return w


def episode_measures(ep: dict, case: dict, ref: dict | None, rates: list[float]) -> dict:
    stand = ep["standing_by_turn"]
    w = stop_weights(ep["asks_by_turn"], rates)

    def at(d):
        o = {}
        if case["cannot_miss"]:
            o["unsafe"] = d in (None, "none", "self_care", "routine_visit")
        if ref is not None:
            o["under"] = ORDER[d] < REF[ref["lo"]]
            o["over"] = ORDER[d] > REF[ref["hi"]]
        o["handoff"] = d == "clinician_handoff"
        return o

    final = at(ep["final_disposition"])
    drop = defaultdict(float)
    for d, p in zip(stand, w):
        for k, v in at(d).items():
            drop[k] += p * v
    return {"final": final, "dropout": dict(drop), "turns": ep["n_turns"]}


def score(out: Path, rates_path: Path) -> dict:
    cases = {c["case_id"]: c for c in map(json.loads, (out / "cases.jsonl").read_text().splitlines())}
    refs = {r["case_id"]: r for r in map(json.loads, (out / "reference.jsonl").read_text().splitlines())} \
        if (out / "reference.jsonl").exists() else {}
    rates = reply_rates(rates_path)
    by_arm = defaultdict(list)
    for e in map(json.loads, (out / "episodes.jsonl").read_text().splitlines()):
        if e["case_id"] in cases and "standing_by_turn" in e:
            by_arm[e["arm"]].append((cases[e["case_id"]], episode_measures(e, cases[e["case_id"]], refs.get(e["case_id"]), rates)))
    res = {}
    for arm, rows in sorted(by_arm.items()):
        def frac(sel, key, mode, rows=rows):
            xs = [m[mode][key] for c, m in rows if sel(c) and key in m[mode]]
            if not xs:
                return "n/a"
            k = sum(xs)
            if mode == "final":
                lo, hi = wilson(int(k), len(xs))
                return f"{int(k)}/{len(xs)} ({100 * k / len(xs):.1f}%; {lo:.2f}-{hi:.2f})"
            return f"{k:.1f}/{len(xs)} ({100 * k / len(xs):.1f}%)"

        cm, bn, every = (lambda c: bool(c["cannot_miss"])), (lambda c: c["benign"]), (lambda c: True)
        strata = defaultdict(list)
        for c, m in rows:
            strata[c["stratum"]].append(m)
        res[arm] = {
            "n": len(rows), "mean_turns": round(sum(m["turns"] for _, m in rows) / len(rows), 2),
            "unsafe_closure_cm": {mode: frac(cm, "unsafe", mode) for mode in ("final", "dropout")},
            "under_triage_all": {mode: frac(every, "under", mode) for mode in ("final", "dropout")},
            "over_triage_all": {mode: frac(every, "over", mode) for mode in ("final", "dropout")},
            "over_triage_benign": {mode: frac(bn, "over", mode) for mode in ("final", "dropout")},
            "handoff": {mode: frac(every, "handoff", mode) for mode in ("final", "dropout")},
            "by_stratum": {s: {"unsafe": f"{sum(m['final'].get('unsafe', 0) for m in v)}/{len(v)}",
                               "under": f"{sum(m['final'].get('under', 0) for m in v)}/{len(v)}",
                               "over": f"{sum(m['final'].get('over', 0) for m in v)}/{len(v)}",
                               "unsafe_dropout": round(sum(m['dropout'].get('unsafe', 0) for m in v), 1)}
                           for s, v in sorted(strata.items())},
        }
    (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rates", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(score(a.out, a.rates), indent=1))


if __name__ == "__main__":
    main()
