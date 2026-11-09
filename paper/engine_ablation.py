"""Post hoc ablations of the frozen ai_pcp_v3 on the HealthBench test conversations that reach the decision engine.

Each variant builds the frozen ai_pcp_v3 arm exactly as the test run did and changes one thing:
  v3                    no change; the in-run control, so every contrast is paired within one run
  engine_off            the engine makes no decision; every patient reply uses the composer's general path, with the
                        same router, hazard screens and reply check (the extractors, which feed only the engine, are skipped)
  minus_stopping_rule   WorkupConfig.enforce_stopping_rule = False
  single_extractor      extractor B removed, so findings are not reconciled across two vendors
  fixed_question_order  WorkupConfig.fixed_question_order = True
  rule_out_authority    WorkupConfig.bayes_rule_out_authority = True (panel item 3 switched off)
  eig_questions         WorkupConfig.question_policy = "eig" (panel item 5 switched off)
The frozen code is not edited. Each worker thread builds its own arm, because AiPcpV2 keeps the router's user_type
on the instance between build_ledger and next_message, and a shared arm lets one conversation read another's value.
Conversations are those that paper/route_paths.py estimated to reach the engine.
Settings (models, graders, repeats) come from the freeze. Outputs and caches go to --out, outside git.

Usage: python -m paper.engine_ablation --paths <route_paths.jsonl> --test-dir <test_healthbench> --out <dir>
       --variants engine_off ... [--n 20] [--repeats 3] [--threads 8]
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import build_arm, capture_usage, make_llm  # noqa: E402

from compass_pcp.harness import frozen  # noqa: E402
from compass_pcp.harness.cache import ResponseCache  # noqa: E402
from compass_pcp.harness.configurations import Configuration  # noqa: E402
from compass_pcp.harness.healthbench import NO_RECORD_CONTEXT  # noqa: E402

WORKUP = {
    "minus_stopping_rule": {"enforce_stopping_rule": False},
    "fixed_question_order": {"fixed_question_order": True},
    "rule_out_authority": {"bayes_rule_out_authority": True},
    "eig_questions": {"question_policy": "eig"},
}
VARIANTS = ["v3", "engine_off", "single_extractor", *WORKUP]


def _not_cached(msgs):
    raise RuntimeError("second-grader pass found a reply that is not in the sampler cache")


def build_variant(name: str):
    hb = frozen.frozen_thresholds()["healthbench"]
    arm = build_arm(Configuration.AI_PCP_V3, hb["comparator_backbone"], frozen.SECOND, require_review=True,
                    deliberate_spec=frozen.DELIBERATE)
    if name == "engine_off":
        arm._decide = lambda *a, **k: ([], [])
        arm._code_turn = lambda *a, **k: []
    elif name == "v3":
        pass
    elif name == "single_extractor":
        arm.extractor_b = None
    else:
        arm.config = arm.config.model_copy(update=WORKUP[name])
    return arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    ap.add_argument("--n", type=int, default=None, help="limit conversations (pilot)")
    ap.add_argument("--repeats", type=int, default=frozen.frozen_thresholds()["healthbench"]["repeats"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    hb = frozen.frozen_thresholds()["healthbench"]

    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    from simple_evals.healthbench_eval import HealthBenchEval  # type: ignore[import-not-found]
    from simple_evals.types import SamplerBase, SamplerResponse  # type: ignore[import-not-found]

    engine_ids = {r["prompt_id"] for r in map(json.loads, a.paths.open()) if r["path"] == "engine"}
    rows = [r for r in map(json.loads, (a.test_dir / "consensus_test.jsonl").open()) if r["prompt_id"] in engine_ids]
    rows.sort(key=lambda r: r["prompt_id"])
    if a.n:
        rows = rows[: a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    split_path = a.out / "engine_path_test.jsonl"
    split_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    class Cached(SamplerBase):
        def __init__(self, fn, cache: ResponseCache):
            self.fn, self.cache, self.usage = fn, cache, {}

        def _call(self, msgs):
            with capture_usage() as log:
                text = self.fn(msgs)
            return {"text": text, "usage": dict(log)}

        def __call__(self, msgs):
            v = self.cache.get_or_call(msgs, lambda: self._call(msgs))
            for k, u in v.get("usage", {}).items():
                t = self.usage.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                for f in t:
                    t[f] += u[f]
            return SamplerResponse(response_text=v["text"], actual_queried_message_list=msgs, response_metadata={})

    def grader(spec: str, name: str):
        llm = make_llm(spec)

        def grade(msgs):
            for _ in range(3):
                text = llm.complete("You grade health responses against a rubric.", msgs, max_tokens=1200)
                if text.strip():
                    return text
            return json.dumps({"explanation": "grader returned no text (prompt blocked); scored as not met",
                               "criteria_met": False})

        tag = spec.replace(":", "_").replace("/", "_")
        return Cached(grade, ResponseCache(a.out / "cache" / f"grader__{tag}__{name}.jsonl"))

    for name in a.variants:
        local = threading.local()

        def reply(msgs, name=name, local=local):
            if not hasattr(local, "arm"):
                local.arm = build_variant(name)
            return local.arm.next_message(msgs, context=NO_RECORD_CONTEXT)[0]

        cache_path = a.out / "cache" / f"sampler__{name}.jsonl"
        for key, spec in (("grader", hb["grader"]), ("grader2", hb["grader2"])):
            # The cache maps the k-th identical call to the k-th stored reply, so the second grader gets a fresh reader
            # of the same file and re-scores the identical replies; a miss there is an error, not a new sample.
            fn = reply if key == "grader" else _not_cached
            sampler = Cached(fn, ResponseCache(cache_path))
            g = grader(spec, name)
            r = HealthBenchEval(grader_model=g, n_repeats=a.repeats, n_threads=a.threads, input_path=str(split_path))(sampler)
            (a.out / f"{name}__{key}.json").write_text(json.dumps({
                "variant": name, "grader": spec, "n": len(rows), "repeats": a.repeats, "score": r.score,
                "metrics": r.metrics, "example_level": (r.metadata or {}).get("example_level_metadata"),
                "usage_sampler": sampler.usage, "usage_grader": g.usage,
            }, default=str))
            print(name, key, r.score, flush=True)


if __name__ == "__main__":
    main()
