"""Analysis of the COMPASS v4 Problem A development comparison (paper/v4_dev_staged.py output).

Strata, from the v3 reply path of each conversation's graded (last) turn:
  engine_decided  a patient or caregiver turn on which the v3 engine made at least one decision
  non_engine      every other conversation (clinician support, out of lane)
Per arm and grader: mean per-conversation score (clipped to [0, 1], compass_pcp.harness.hb_analysis.load) by stratum.
Paired differences (each v4 arm vs v3 and vs v4_engine_off; v3 vs v4_engine_off) with conversation-bootstrap 95% CIs
(paper.sensitivity._boot_diff, 10,000 resamples). Pool-weighted overall difference: the engine-routed conversations
are all run and the non-engine ones are a simple random sample, so the overall estimate weights the sample by
(non-engine pool size / sample size), with a bootstrap stratified by selection stratum.
Reply length in words; length adjustment: OLS of the paired score difference on the paired word-count difference
(the intercept is the difference at equal length; bootstrap CI), and the paired difference by tercile of the
word-count difference. Contract compliance of v4 contract replies, cost per reply at batch prices (all calls the
reply used, shared calls included), and p95 latency where realtime call timings exist.

Usage: python -m paper.v4_dev_analysis --out <v4_devA dir>   (writes report.json and report.md there)
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from compass_pcp.contract import ResponseContract, deterministic_check
from compass_pcp.harness.hb_analysis import N_BOOT, SEED, load
from compass_pcp.policy_v4 import load_contract_policy
from paper.batch_stage import BATCH_PRICE
from paper.sensitivity import _boot_diff

V4 = ["v4_contract", "v4_edit", "v4_contract_compact", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v22"]  # v7-v22: Paper 2 candidates
COMPARATORS = ["raw_llm", "amie_style", "mai_dxo_style"]
GRADERS = {"grader": "GPT-4.1", "grader2": "Gemini 3.1 Pro"}
REF_ORDER = ("a question comes before the required instruction", "the required instruction is not within the first")


def words(t: str) -> int:
    return len(t.split())


def ci(diffs: list[float]) -> dict | None:
    if not diffs:
        return None
    m, lo, hi = _boot_diff(diffs, 0.025, 0.975)
    return {"n": len(diffs), "diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx if sxx else 0.0
    return my - b * mx, b


def length_adjusted(dy: list[float], dx: list[float]) -> dict | None:
    if len(dy) < 10:
        return None
    a, b = ols(dx, dy)
    rng = random.Random(SEED)
    n, ints, slopes = len(dy), [], []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        ai, bi = ols([dx[i] for i in idx], [dy[i] for i in idx])
        ints.append(ai)
        slopes.append(bi)
    ints.sort()
    slopes.sort()
    q = lambda v: [round(v[int(0.025 * N_BOOT)], 4), round(v[int(0.975 * N_BOOT) - 1], 4)]  # noqa: E731
    return {"n": n, "intercept": round(a, 4), "intercept_ci95": q(ints), "slope_per_100_words": round(100 * b, 4),
            "slope_ci95_per_100_words": [round(100 * s, 4) for s in q(slopes)]}


def terciles(dy: list[float], dx: list[float]) -> list[dict]:
    if len(dy) < 9:
        return []
    order = sorted(range(len(dx)), key=lambda i: dx[i])
    k = len(order)
    out = []
    for t, part in enumerate((order[: k // 3], order[k // 3: 2 * k // 3], order[2 * k // 3:]), 1):
        out.append({"tercile": t, "length_diff_range": [dx[part[0]], dx[part[-1]]],
                    **(ci([dy[i] for i in part]) or {})})
    return out


def weighted_overall(a: dict, b: dict, engine_ids: set, sample_ids: set, w_sample: float) -> dict | None:
    """Pool-weighted mean difference: engine-routed conversations weight 1, sampled non-engine weight w_sample."""
    e = [a[p] - b[p] for p in engine_ids if p in a and p in b]
    s = [a[p] - b[p] for p in sample_ids if p in a and p in b]
    if not e or not s:
        return None

    def stat(e_, s_):
        return (sum(e_) + w_sample * sum(s_)) / (len(e_) + w_sample * len(s_))

    rng = random.Random(SEED)
    boots = sorted(stat([e[rng.randrange(len(e))] for _ in e], [s[rng.randrange(len(s))] for _ in s])
                   for _ in range(N_BOOT))
    return {"n": len(e) + len(s), "n_engine_routed": len(e), "n_sample": len(s), "w_sample": round(w_sample, 3), "diff": round(stat(e, s), 4),
            "ci95": [round(boots[int(0.025 * N_BOOT)], 4), round(boots[int(0.975 * N_BOOT) - 1], 4)]}


def reply_cost(calls: list[dict]) -> float:
    usd = 0.0
    for c in calls:
        p = BATCH_PRICE.get(c["spec"])
        if p:
            usd += (c.get("input_tokens") or 0) / 1e6 * p[0] + (c.get("output_tokens") or 0) / 1e6 * p[1]
    return usd


def reply_latency(calls: list[dict]) -> float | None:
    """Critical path from realtime call timings: the v4 screen runs in parallel with the ledger (tier 0 and 1 calls);
    everything else is sequential. None when any call lacks a timing (batch result)."""
    if not calls or any(c.get("latency_s") is None for c in calls):
        return None
    screen = sum(c["latency_s"] for c in calls if c.get("screen_v4"))
    ledger = sum(c["latency_s"] for c in calls if not c.get("screen_v4") and c["tier"] < 3)
    rest = sum(c["latency_s"] for c in calls if c["tier"] >= 3)
    return max(screen, ledger) + rest


def p95(v: list[float]) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(0.95 * len(v)))], 2)


def compliance(recs: list[dict], policy) -> dict:
    c = [r for r in recs if (r.get("info") or {}).get("contract")]
    if not c:
        return {"n_contract_replies": 0}
    first_ok = sum(bool(r["info"]["compliance"]["first_pass"]["ok"]) for r in c)
    det_first = sum(bool(r["info"]["compliance"]["first_pass"]["deterministic_ok"]) for r in c)
    sem_first = sum(bool(r["info"]["compliance"]["first_pass"]["semantic_ok"]) for r in c)
    revised = [r for r in c if r["info"]["compliance"]["revised"]]
    post_ok = sum(bool(r["info"]["compliance"]["post_revision"]["ok"]) for r in revised)
    fallback = sum(bool(r["info"]["compliance"]["fallback"]) for r in c)
    ref_first = sum(any(p.startswith(REF_ORDER) for p in r["info"]["compliance"]["first_pass"]["problems"]) for r in c)
    final_det, final_ref = 0, 0
    for r in c:
        probs = deterministic_check(r["text"], ResponseContract(**r["info"]["contract"]), policy)
        final_det += not probs
        final_ref += any(p.startswith(REF_ORDER) for p in probs)
    n = len(c)
    pct = lambda k, d: f"{k} of {d} ({100 * k / d:.1f}%)" if d else "0 of 0"  # noqa: E731
    return {
        "n_contract_replies": n, "states": dict(Counter(r["info"]["contract"]["state"] for r in c)),
        "first_pass_ok": pct(first_ok, n), "first_pass_deterministic_ok": pct(det_first, n),
        "first_pass_semantic_ok": pct(sem_first, n), "revised": pct(len(revised), n),
        "post_revision_ok_of_revised": pct(post_ok, len(revised)), "fallback": pct(fallback, n),
        "referral_order_failures_first_pass": pct(ref_first, n),
        "final_reply_deterministic_ok": pct(final_det, n), "referral_order_failures_final": pct(final_ref, n),
        "fallback_rate": round(fallback / n, 4),
    }


def analyze(out: Path) -> dict:
    replies: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in (out / "replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["repeat"] == 0:
            replies[r["arm"]][r["prompt_id"]] = r
    arms = [a for a in ["v3", *V4, "v4_engine_off", *COMPARATORS] if a in replies]
    sel = json.loads((out / "selection.json").read_text())
    engine_routed, sample = set(sel["engine_routed"]), set(sel["non_engine_sample"])
    n_non_engine_pool = sel["n_pool"] - len(engine_routed)
    w_sample = n_non_engine_pool / len(sample) if sample else 0.0

    v3 = replies["v3"]
    decided = {p for p, r in v3.items() if r["path"].get("decisions") and r["path"].get("user_type") != "health_professional"}
    strata = {"engine_decided": decided, "non_engine": set(v3) - decided}
    rep: dict = {"n_conversations": len(v3), "selection": {"engine_routed": len(engine_routed), "non_engine_sample": len(sample),
                 "router_counts": sel["counts"], "n_pool": sel["n_pool"]},
                 "strata": {k: len(v) for k, v in strata.items()},
                 "v3_decisions": dict(Counter(v3[p]["path"]["decision"] for p in v3)),
                 "engine_decided_by_selection": {"engine_routed": len(decided & engine_routed),
                                                 "non_engine_sample": len(decided & sample)},
                 "crisis_policy_status": "placeholder_unsigned (B1 uses the placeholder policy; no B2 classifier)"}

    scores = {}
    for a in arms:
        for g in GRADERS:
            f = out / f"{a}__{g}.json"
            if f.exists():
                scores[(a, g)] = load(f)["score"]
    rep["graded"] = sorted(f"{a}__{g}" for a, g in scores)

    means: dict = {}
    for (a, g), s in scores.items():
        for k, ids in strata.items():
            v = [s[p] for p in ids if p in s]
            means.setdefault(g, {}).setdefault(a, {})[k] = {"n": len(v), "mean": round(statistics.fmean(v), 4) if v else None}
    rep["mean_scores"] = means

    pairs = [(a, "v3") for a in V4] + [(a, "v4_engine_off") for a in V4] + [("v3", "v4_engine_off"),
                                                                           ("v4_contract_compact", "v4_contract")]
    pairs += [(a, c) for a in ("v3", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v22") for c in COMPARATORS]
    contrasts: dict = {}
    for g in GRADERS:
        for a, b in pairs:
            if (a, g) not in scores or (b, g) not in scores:
                continue
            sa, sb = scores[(a, g)], scores[(b, g)]
            key = f"{a} - {b}"
            row = {}
            for k, ids in strata.items():
                ps = sorted(p for p in ids if p in sa and p in sb)
                dy = [sa[p] - sb[p] for p in ps]
                dx = [words(replies[a][p]["text"]) - words(replies[b][p]["text"]) for p in ps]
                row[k] = {**(ci(dy) or {}), "mean_length_diff_words": round(statistics.fmean(dx), 1) if dx else None,
                          "length_adjusted": length_adjusted(dy, dx), "by_length_diff_tercile": terciles(dy, dx)}
            row["pool_weighted_overall"] = weighted_overall(sa, sb, engine_routed, sample, w_sample)
            contrasts.setdefault(g, {})[key] = row
    rep["contrasts"] = contrasts

    policy = load_contract_policy()
    per_arm: dict = {}
    for a in arms:
        recs = replies[a]
        lens = {k: [words(recs[p]["text"]) for p in ids if p in recs] for k, ids in strata.items()}
        lat = [x for x in (reply_latency(r.get("calls") or []) for r in recs.values()) if x is not None]
        per_arm[a] = {
            "n_replies": len(recs),
            "words": {k: {"mean": round(statistics.fmean(v), 1), "median": statistics.median(v)} for k, v in lens.items() if v},
            "cost_per_reply_usd_batch": round(statistics.fmean(reply_cost(r.get("calls") or []) for r in recs.values()), 4),
            "calls_per_reply": round(statistics.fmean(len(r.get("calls") or []) for r in recs.values()), 2),
            "latency_p95_s": p95(lat), "latency_n": len(lat),
            "reply_paths": dict(Counter((r["path"] or {}).get("reply_path") for r in recs.values())),
            "b1_routes": dict(Counter((r["path"] or {}).get("route") for r in recs.values() if a in V4)),
            "compliance": compliance([recs[p] for p in recs], policy) if a in V4 else None,
            "compliance_engine_decided": compliance([recs[p] for p in decided if p in recs], policy) if a in V4 else None,
        }
    rep["per_arm"] = per_arm
    store_cost = 0.0
    try:
        from paper.batch_stage import Store
        from paper.v4_dev_staged import spent

        store_cost = spent(Store(out / "store"))
    except Exception:  # noqa: BLE001
        pass
    rep["spent_usd_total"] = round(store_cost, 2)
    return rep


def fmt_c(c: dict | None) -> str:
    if not c or "diff" not in c:
        return "n/a"
    return f"{c['diff']:+.3f} ({c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f}), n={c['n']}"


def markdown(rep: dict) -> str:
    L = ["# COMPASS v4 Problem A development comparison (HealthBench non-Consensus pool)", "",
         f"Conversations: {rep['n_conversations']} ({rep['selection']['engine_routed']} engine-routed by the v3 router, "
         f"{rep['selection']['non_engine_sample']} sampled non-engine, seed 20261103; pool {rep['selection']['n_pool']}). "
         f"Router counts: {rep['selection']['router_counts']}. Strata from the v3 reply path: {rep['strata']}. "
         "One repeat per arm.", "",
         f"B1 crisis screen: {rep['crisis_policy_status']}.", "",
         f"Total spend in this store: USD {rep['spent_usd_total']} (batch price; realtime smoke calls at list price).", ""]
    for g, label in GRADERS.items():
        L += [f"## Mean scores, {label}", "", "| Arm | engine-decided | non-engine |", "|---|---|---|"]
        for a, m in rep["mean_scores"].get(g, {}).items():
            L.append(f"| {a} | {m['engine_decided']['mean']} (n={m['engine_decided']['n']}) | "
                     f"{m['non_engine']['mean']} (n={m['non_engine']['n']}) |")
        L += ["", f"## Paired differences, {label} (95% bootstrap CI)", "",
              "| Contrast | engine-decided | length-adjusted intercept (engine-decided) | non-engine | pool-weighted overall |",
              "|---|---|---|---|---|"]
        for k, row in rep["contrasts"].get(g, {}).items():
            la = row["engine_decided"].get("length_adjusted")
            las = f"{la['intercept']:+.3f} ({la['intercept_ci95'][0]:+.3f} to {la['intercept_ci95'][1]:+.3f})" if la else "n/a"
            L.append(f"| {k} | {fmt_c(row['engine_decided'])} | {las} | {fmt_c(row['non_engine'])} | "
                     f"{fmt_c(row['pool_weighted_overall'])} |")
        L += ["", "Engine-decided paired difference by tercile of the paired word-count difference:", ""]
        for k, row in rep["contrasts"].get(g, {}).items():
            ts = "; ".join(f"T{t['tercile']} [{t['length_diff_range'][0]}, {t['length_diff_range'][1]}] words: "
                           f"{fmt_c(t)}" for t in row["engine_decided"].get("by_length_diff_tercile", []))
            L.append(f"- {k}: {ts or 'n/a'}")
        L.append("")
    L += ["## Length, cost, latency", "", "| Arm | words engine-decided (mean) | words non-engine (mean) | "
          "USD per reply (batch) | calls per reply | p95 latency s (n) |", "|---|---|---|---|---|---|"]
    for a, p in rep["per_arm"].items():
        w = p["words"]
        L.append(f"| {a} | {w.get('engine_decided', {}).get('mean')} | {w.get('non_engine', {}).get('mean')} | "
                 f"{p['cost_per_reply_usd_batch']} | {p['calls_per_reply']} | {p['latency_p95_s']} ({p['latency_n']}) |")
    L += ["", "## Contract compliance (v4 contract replies, all conversations)", ""]
    for a, p in rep["per_arm"].items():
        if p["compliance"]:
            L.append(f"- {a}: {json.dumps(p['compliance'])}")
            L.append(f"  - B1 routes: {p['b1_routes']}; reply paths: {p['reply_paths']}")
    L += ["", "Latency is the critical path summed from realtime call timings (smoke and latency sample only); batch "
          "results carry no timings. Cost per reply counts every call the reply used, including calls shared with "
          "other arms through the cache.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rep = analyze(a.out)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    (a.out / "report.md").write_text(markdown(rep))
    print(markdown(rep))


if __name__ == "__main__":
    main()
