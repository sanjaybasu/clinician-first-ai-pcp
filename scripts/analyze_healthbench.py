"""Summarize HealthBench runs: score with CI, paired differences, Consensus pass rates, tokens.

Usage: python scripts/analyze_healthbench.py <run dir> [--reference raw_llm]
Writes summary.json next to the per-config files. Paired comparisons use only
prompt_ids present in both configurations.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.healthbench import compare_consensus, compare_scores


def bootstrap_mean_ci(xs: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(xs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    per_prompt: dict[str, list[float]] = defaultdict(list)
    criteria: dict[str, bool] = {}
    by_cluster: dict[str, list[bool]] = defaultdict(list)
    for e in d["example_level"] or []:
        per_prompt[e["prompt_id"]].append(max(0.0, min(1.0, e["score"])))
        for i, item in enumerate(e["rubric_items"]):
            key = f'{e["prompt_id"]}:{i}'
            criteria[key] = criteria.get(key, True) and bool(item["criteria_met"]) == (item["points"] > 0)
            for t in item["tags"]:
                if t.startswith("cluster:"):
                    by_cluster[t].append(bool(item["criteria_met"]) == (item["points"] > 0))
    d["per_prompt"] = {k: sum(v) / len(v) for k, v in per_prompt.items()}
    d["criteria_pass"] = criteria
    d["by_cluster"] = {k: sum(v) / len(v) for k, v in by_cluster.items()}
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", default="raw_llm")
    args = ap.parse_args()
    runs = {p.stem: load(p) for p in sorted(args.run_dir.glob("*.json")) if p.stem not in ("split_manifest", "summary")}
    ref = runs.get(args.reference)
    out = {}
    for name, d in runs.items():
        scores = list(d["per_prompt"].values())
        mean, lo, hi = bootstrap_mean_ci(scores)
        delta_tokens = {}
        for k, u in (d.get("usage_after") or {}).items():
            b = (d.get("usage_before") or {}).get(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            delta_tokens[k] = {f: u[f] - b[f] for f in ("calls", "input_tokens", "output_tokens")}
        row = {
            "n_prompts": len(scores),
            "mean_score": mean,
            "usage_per_arm": d.get("usage_per_arm"),
            "ci95": [lo, hi],
            "simple_evals_overall": d["score"],
            "consensus_clusters": d["by_cluster"],
            "tokens": delta_tokens,
        }
        if ref is not None and name != args.reference:
            row["vs_reference"] = compare_scores(d["per_prompt"], ref["per_prompt"], n_boot=5000)
            row["consensus_vs_reference"] = compare_consensus(d["criteria_pass"], ref["criteria_pass"])
            worse = [c for c, v in d["by_cluster"].items() if v < ref["by_cluster"].get(c, 0)]
            row["clusters_below_reference"] = sorted(worse)
        out[name] = row
    (args.run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        line = f"{name:28s} n={r['n_prompts']:3d} score {r['mean_score']:.3f} ({r['ci95'][0]:.3f}-{r['ci95'][1]:.3f})"
        if "vs_reference" in r:
            v = r["vs_reference"]
            line += f"  diff vs {args.reference} {v['diff']:+.3f} ({v['ci_low']:+.3f} to {v['ci_high']:+.3f})"
            c = r["consensus_vs_reference"]
            line += f"  consensus criteria: only-this {c['a_only']}, only-ref {c['b_only']}, p={c['p_exact']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
