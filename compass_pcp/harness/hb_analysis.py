"""Pre-specified HealthBench analysis (VVP-PCP-001 Sections 8, 8.6 and 10; hypotheses in configurations.py).

HB1, non-inferiority: Consensus-criterion pass rate of ai_pcp_v3 minus raw_llm, where a criterion passes for a
conversation only when it passes in every repeat. The interval comes from a bootstrap over conversations (criteria are
nested in conversations). Non-inferior when the lower 95% bound exceeds -0.02.
HB2-HB5, superiority: mean per-conversation score (mean over repeats, clipped to [0, 1]) of ai_pcp_v3 minus each
comparator; paired bootstrap over conversations; two-sided p from the bootstrap distribution; Holm within the family.
Descriptive: scores by theme, Consensus pass rate by behavior cluster with exact McNemar p and Holm across clusters,
calls and tokens per conversation, and agreement between the two graders.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.adjudication import gwet_ac1
from compass_pcp.harness.configurations import holm
from compass_pcp.harness.metrics import mcnemar_exact_p

N_BOOT, SEED = 10_000, 20261028


def _met(item: dict) -> bool:
    """A criterion is satisfied when a positive criterion is met or a negative (penalty) criterion is not met."""
    return bool(item["criteria_met"]) == (item["points"] > 0)


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    reps: dict[str, list[float]] = defaultdict(list)
    crit: dict[str, dict[int, bool]] = defaultdict(dict)
    cluster_of: dict[tuple[str, int], str] = {}
    graded: dict[tuple[str, str, int], bool] = {}
    for e in d["example_level"] or []:
        pid = e["prompt_id"]
        reps[pid].append(max(0.0, min(1.0, e["score"])))
        completion = e["completion"][0]["content"] if isinstance(e["completion"], list) else str(e["completion"])
        for j, item in enumerate(e["rubric_items"]):
            crit[pid][j] = crit[pid].get(j, True) and _met(item)
            graded[(pid, completion, j)] = bool(item["criteria_met"])
            for t in item.get("tags", []):
                if t.startswith("cluster:"):
                    cluster_of[(pid, j)] = t.split(":", 1)[1]
    return {"score": {p: sum(v) / len(v) for p, v in reps.items()}, "criteria": dict(crit), "cluster_of": cluster_of,
            "graded": graded, "usage": d.get("usage_per_arm")}


def _boot(pids: list[str], stat, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    rng = random.Random(seed)
    return sorted(stat([pids[rng.randrange(len(pids))] for _ in pids]) for _ in range(n_boot))


def superiority(a: dict, b: dict) -> dict:
    pids = sorted(set(a["score"]) & set(b["score"]))

    def stat(ps):
        return sum(a["score"][p] - b["score"][p] for p in ps) / len(ps)

    boots = _boot(pids, stat)
    p = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / len(boots)
    return {"n": len(pids), "diff": stat(pids), "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
            "p_bootstrap": min(1.0, p)}


def mean_ci(r: dict) -> dict:
    pids = sorted(r["score"])
    boots = _boot(pids, lambda ps: sum(r["score"][p] for p in ps) / len(ps))
    return {"mean": sum(r["score"].values()) / len(pids), "n": len(pids),
            "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]]}


def consensus_rate(r: dict, pids: list[str]) -> float:
    vals = [v for p in pids for v in r["criteria"][p].values()]
    return sum(vals) / len(vals)


def noninferiority(a: dict, b: dict, margin: float) -> dict:
    pids = sorted(set(a["criteria"]) & set(b["criteria"]))

    def stat(ps):
        return consensus_rate(a, ps) - consensus_rate(b, ps)

    boots = _boot(pids, stat)
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    return {"n_conversations": len(pids), "rate_a": consensus_rate(a, pids), "rate_b": consensus_rate(b, pids),
            "diff": stat(pids), "ci95": [lo, hi], "margin": margin, "noninferior": lo > -margin}


def by_cluster(a: dict, b: dict) -> dict:
    keys = sorted(set(a["cluster_of"]) & {(p, j) for p in b["criteria"] for j in b["criteria"][p]})
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k in keys:
        groups[a["cluster_of"][k]].append(k)
    out, ps = {}, {}
    for c, ks in sorted(groups.items()):
        av = [a["criteria"][p][j] for p, j in ks]
        bv = [b["criteria"][p][j] for p, j in ks]
        only_a = sum(x and not y for x, y in zip(av, bv))
        only_b = sum(y and not x for x, y in zip(av, bv))
        ps[c] = mcnemar_exact_p(only_a, only_b)
        out[c] = {"n": len(ks), "rate_a": sum(av) / len(av), "rate_b": sum(bv) / len(bv), "a_only": only_a,
                  "b_only": only_b, "p_exact": ps[c]}
    for c, rejected in holm(ps).items():
        out[c]["holm_significant"] = rejected
    return out


def by_theme(r: dict, themes: dict[str, str]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for p, s in r["score"].items():
        groups[themes.get(p, "unknown")].append(s)
    return {t: {"n": len(v), "mean_score": sum(v) / len(v)} for t, v in sorted(groups.items())}


def grader_agreement(g1: dict, g2: dict) -> dict:
    keys = sorted(set(g1["graded"]) & set(g2["graded"]))
    pairs = [[str(int(g1["graded"][k])), str(int(g2["graded"][k]))] for k in keys]
    return {"n_criteria": len(keys), "ac1": gwet_ac1(pairs) if pairs else None,
            "raw_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None}


def per_conversation_cost(r: dict) -> dict | None:
    u = r.get("usage")
    if not u or not u.get("responses_with_usage"):
        return None
    n = u["responses_with_usage"]
    return {k: {f: v[f] / n for f in ("calls", "input_tokens", "output_tokens")} for k, v in u["tokens"].items()}


def themes_from_split(run: Path) -> dict[str, str]:
    out = {}
    for path in run.glob("*consensus_*.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["prompt_id"]] = next((t.split(":", 1)[1] for t in r["example_tags"] if t.startswith("theme:")), "unknown")
    return out


def analyze(run: Path, final: str, raw: str, comparators: list[str], margin: float = 0.02) -> dict:
    """Primary grader results, then the same contrasts under the second grader when its files exist."""
    themes = themes_from_split(run)
    out: dict = {}
    for grader, suffix in (("grader1", ""), ("grader2", "__grader2")):
        files = {c: run / f"{c}{suffix}.json" for c in [final, *comparators]}
        if not all(f.exists() for f in files.values()):
            continue
        runs = {c: load(f) for c, f in files.items()}
        res = {"scores": {c: mean_ci(r) for c, r in runs.items()},
               "HB1": noninferiority(runs[final], runs[raw], margin),
               "superiority": {c: superiority(runs[final], runs[c]) for c in comparators},
               "by_cluster_vs_raw": by_cluster(runs[final], runs[raw]),
               "by_theme": {c: by_theme(r, themes) for c, r in runs.items()},
               "cost_per_conversation": {c: per_conversation_cost(r) for c, r in runs.items()}}
        for c, rejected in holm({c: v["p_bootstrap"] for c, v in res["superiority"].items()}).items():
            res["superiority"][c]["holm_significant"] = rejected
        out[grader] = res
        if grader == "grader2":
            g1 = {c: load(run / f"{c}.json") for c in files}
            out["grader_agreement"] = {c: grader_agreement(g1[c], runs[c]) for c in files}
    return out
