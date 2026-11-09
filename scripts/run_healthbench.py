"""HealthBench public layer: run configurations through OpenAI's own grading code.

Uses openai/simple-evals ``HealthBenchEval`` unchanged, so scores compare with
published ones. Each configuration is wrapped as a simple-evals sampler that
returns the next patient-visible message, with no patient record.

Split discipline: the subset is split by a salted hash of prompt_id into a 20%
dev split and an 80% test split. ``--split test`` refuses to run unless the
freeze file verifies (harness/prereg.py), so the test split is scored once.

Usage:
  SIMPLE_EVALS_PARENT=<dir containing a simple_evals package> python scripts/run_healthbench.py \
    --subset consensus --split dev --configs raw_llm ai_pcp --backbone anthropic:<model> \
    --grader openai:gpt-4.1-2025-04-14 --n 50 --repeats 3 --out <dir outside git>
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from compass_pcp.arms.baselines import AmieStyle, MaiDxoStyle, RawLLM, StandardGuardrails
from compass_pcp.arms.llm import AnthropicLLM, GeminiLLM, OpenAILLM, capture_usage
from compass_pcp.engine.scripts import load_script
from compass_pcp.engine.workup import WorkupConfig
from compass_pcp.harness.configurations import Configuration
from compass_pcp.harness.healthbench import DATA_URLS, NO_RECORD_CONTEXT, dev_split

SALT = "compass-pcp-healthbench-v1"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "compass_pcp" / "data" / "scripts"


CREATED: list = []


def make_llm(spec: str):
    """vendor:model, optionally @THINKING for Gemini (e.g. google:gemini-3.1-pro-preview@HIGH)."""
    vendor, model = spec.split(":", 1)
    model, _, thinking = model.partition("@")
    if vendor == "google" and thinking:
        llm = GeminiLLM(model=model, thinking_level=thinking, thinking_headroom=8192)
    else:
        llm = {"anthropic": AnthropicLLM, "google": GeminiLLM, "openai": OpenAILLM}[vendor](model=model)
    CREATED.append(llm)
    return llm


def usage_snapshot() -> dict:
    out: dict = {}
    for llm in CREATED:
        k = f"{llm.vendor}:{llm.model}"
        u = out.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        u["calls"] += llm.usage.calls
        u["input_tokens"] += llm.usage.input_tokens
        u["output_tokens"] += llm.usage.output_tokens
    return out


# Hazard screen settings fixed on the Cohort A tuning half (VVP-PCP-001 Section 8.5): a message is flagged when the
# ledger tier is routine_followup or higher, with Kev at its default tier cut-offs and this model as the LLM screen.
FINAL_HAZARD_LLM = "anthropic:claude-sonnet-5-5"


PKG = Path(__file__).resolve().parents[1] / "compass_pcp"
COMPARATOR_CODE = ["arms/baselines.py"]
AI_PCP_CODE = ["arms/ai_pcp_v2.py", "arms/hazard.py", "arms/dual_process.py", "ledger.py", "findings.py", "gate.py",
               "lanes.py", "engine/*.py", "data/scripts/*.yaml", "data/rules/*.yaml"]
V4_CODE = ["arms/ai_pcp_v4.py", "arms/screen_v4.py", "arms/writer_v4.py", "contract.py", "crisis_router.py",
           "policy_v4.py", "review_score.py", "data/policies/*.yaml"]  # added to the v4 hash only; v3 hashes unchanged


def code_sha(config: Configuration) -> str:
    """Short hash of the code that shapes a configuration's replies. It is part of the response-cache name, so a cached
    reply is never replayed for an arm whose prompts or logic changed."""
    import hashlib

    h = hashlib.sha256()
    patterns = AI_PCP_CODE if config.value.startswith("ai_pcp") else COMPARATOR_CODE
    if config == Configuration.AI_PCP_V4:
        patterns = [*patterns, *V4_CODE]
    for pattern in patterns:
        for f in sorted(PKG.glob(pattern)):
            h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:10]


def _not_cached(msgs):
    raise RuntimeError("second-grader pass found a response that is not in the sampler cache")


def build_arm(config: Configuration, backbone_spec: str, second_spec: str, require_review: bool, deliberate_spec: str | None = None):
    system2 = (Configuration.RAW_DELIBERATE, Configuration.AI_PCP_DELIBERATE, Configuration.AI_PCP_DUAL_PROCESS,
               Configuration.AI_PCP_FINAL, Configuration.AI_PCP_V3)
    if config in system2 and not deliberate_spec:
        raise SystemExit("--deliberate is required for dual-process configurations")
    if config == Configuration.RAW_DELIBERATE:
        return RawLLM(make_llm(deliberate_spec))
    def second_for(primary: str) -> str:
        # extractor B must come from a different vendor than the tier's primary model
        return backbone_spec if second_spec.split(":")[0] == primary.split(":")[0] else second_spec

    if config == Configuration.AI_PCP_DELIBERATE:
        return build_arm(Configuration.AI_PCP, deliberate_spec, second_for(deliberate_spec), require_review)
    if config == Configuration.AI_PCP_V4:
        return build_v4(backbone_spec, second_spec, require_review, deliberate_spec)
    if config in (Configuration.AI_PCP_FINAL, Configuration.AI_PCP_V3):
        from compass_pcp.arms.llm import KevJudge

        arm = build_arm(Configuration.AI_PCP, deliberate_spec, second_for(deliberate_spec), require_review)
        arm.hazard_judge, arm.hazard_both = KevJudge(), True
        arm.compose_version = 3 if config == Configuration.AI_PCP_V3 else 2
        arm.hazard_llm = make_llm(FINAL_HAZARD_LLM)  # the LLM screen whose operating point was fixed on Cohort A tuning
        return arm
    if config == Configuration.AI_PCP_DUAL_PROCESS:
        from compass_pcp.arms.dual_process import DualProcess
        from compass_pcp.arms.llm import KevJudge

        fast = build_arm(Configuration.AI_PCP, backbone_spec, second_for(backbone_spec), require_review)
        fast.hazard_judge = KevJudge()  # System 1: local calibrated typed judgments
        return DualProcess(
            fast=fast,
            deliberate=build_arm(Configuration.AI_PCP, deliberate_spec, second_for(deliberate_spec), require_review),
        )
    backbone = make_llm(backbone_spec)
    if config == Configuration.RAW_LLM:
        return RawLLM(backbone)
    if config == Configuration.LLM_STANDARD_GUARDRAILS:
        return StandardGuardrails(backbone)
    if config == Configuration.AMIE_STYLE:
        return AmieStyle(backbone)
    if config == Configuration.MAI_DXO_STYLE:
        return MaiDxoStyle(backbone)
    scripts = {p.stem: load_script(p, require_review=require_review) for p in sorted(SCRIPTS_DIR.glob("*.yaml"))}
    if not scripts:
        raise RuntimeError("no illness scripts in compass_pcp/data/scripts; the AI PCP arm would defer every case")
    cfg = WorkupConfig(
        enforce_stopping_rule=config != Configuration.AI_PCP_NO_STOPPING_RULE,
        fixed_question_order=config == Configuration.AI_PCP_FIXED_QUESTIONS,
        # Items 3 and 5 are the default; the single-item configurations keep the other item switched off.
        question_policy="eig" if config == Configuration.AI_PCP_NO_POSTERIOR_RULE_OUT else "obligation",
        bayes_rule_out_authority=config == Configuration.AI_PCP_OBLIGATION,
    )
    single = config == Configuration.AI_PCP_SINGLE_EXTRACTOR
    if not single and second_spec.split(":")[0] == backbone_spec.split(":")[0]:
        raise SystemExit(f"extractor B ({second_spec}) must be a different vendor from the backbone ({backbone_spec})")
    return AiPcpV2(
        scripts=scripts, router=backbone, hazard_llm=backbone, extractor_a=backbone,
        extractor_b=None if single else make_llm(second_spec), dialogue=backbone, config=cfg,
        dialogue_chain=config == Configuration.AI_PCP_AMIE_DIALOGUE,
    )


# B1 structured screen: Sonnet 5.5 with thinking off ("between_tools" is the off switch on Sonnet 5.5).
V4_SCREEN_LLM = "anthropic:claude-sonnet-5-5"


def build_v4(backbone_spec: str, second_spec: str, require_review: bool, deliberate_spec: str | None, *,
             writer_variant: str = "contract", compact_policy: bool = False, engine_off: bool = False):
    """ai_pcp_v4: the ai_pcp_v3 arm, built exactly as for v3, wrapped by AiPcpV4. v3's builder is not changed."""
    from compass_pcp.arms.ai_pcp_v4 import AiPcpV4
    from compass_pcp.arms.llm import AnthropicLLM

    engine = build_arm(Configuration.AI_PCP_V3, backbone_spec, second_spec, require_review, deliberate_spec)
    screen_llm = AnthropicLLM(model=V4_SCREEN_LLM.split(":", 1)[1], thinking={"type": "between_tools"})
    CREATED.append(screen_llm)
    return AiPcpV4(engine=engine, screen_llm=screen_llm, writer_variant=writer_variant, compact_policy=compact_policy,
                   engine_off=engine_off)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "hard", "consensus", "nonconsensus"], default="consensus",
                    help="nonconsensus: HealthBench conversations outside the Consensus subset, for optimization only")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--backbone", required=True, help="vendor:model, shared by every arm")
    ap.add_argument("--deliberate", default=None, help="vendor:model[@THINKING] for System-2 configurations")
    ap.add_argument("--second", default=None, help="vendor:model for extractor B (must differ from backbone vendor)")
    ap.add_argument("--grader", required=True, help="vendor:model; openai:gpt-4.1-2025-04-14 for comparability")
    ap.add_argument("--grader2", default=None, help="second-vendor grader; regrades the same cached responses")
    ap.add_argument("--n", type=int, default=None, help="limit conversations (pilot)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--allow-unsigned-scripts", action="store_true", help="dev split only")
    ap.add_argument("--exclude-from", type=Path, nargs="*", default=None, help="run files whose prompt_ids to exclude")
    args = ap.parse_args()

    if args.split == "test":
        if args.allow_unsigned_scripts or args.freeze is None:
            sys.exit("test split needs --freeze and signed scripts")
        from compass_pcp.harness.frozen import verify_current

        verify_current(args.freeze, json.loads((args.freeze.parent / "hb_test_ids.json").read_text()))

    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    import urllib.request

    from simple_evals.healthbench_eval import HealthBenchEval  # type: ignore[import-not-found]
    from simple_evals.types import SamplerBase, SamplerResponse  # type: ignore[import-not-found]

    args.out.mkdir(parents=True, exist_ok=True)
    def fetch(subset: str) -> list[dict]:
        with urllib.request.urlopen(DATA_URLS[subset], timeout=120) as fh:
            return [json.loads(line) for line in fh.read().decode().splitlines() if line.strip()]

    if args.subset == "nonconsensus":
        # Optimization pool: no conversation here appears in the Consensus subset, so no test item can be touched.
        if args.split == "test":
            sys.exit("the nonconsensus pool is for optimization only; it has no test split")
        consensus = {r["prompt_id"] for r in fetch("consensus")}
        rows = [r for r in fetch("full") if r["prompt_id"] not in consensus]
        dev_ids, test_ids = [r["prompt_id"] for r in rows], []
    else:
        rows = fetch(args.subset)
        dev_ids, test_ids = dev_split([r["prompt_id"] for r in rows], salt=SALT)
    keep = set(dev_ids if args.split == "dev" else test_ids)
    if args.split == "test":
        from compass_pcp.harness import frozen

        hb = frozen.frozen_thresholds()["healthbench"]
        if set(test_ids) != set(json.loads((args.freeze.parent / "hb_test_ids.json").read_text())):
            sys.exit("downloaded HealthBench test split differs from the frozen test ids")
        wanted = (args.subset, args.backbone, args.deliberate, args.second, args.grader, args.grader2, args.repeats)
        if wanted != (hb["subset"], hb["comparator_backbone"], frozen.DELIBERATE, frozen.SECOND, hb["grader"], hb["grader2"],
                      hb["repeats"]):
            sys.exit(f"test run settings {wanted} differ from the frozen settings")
        if set(args.configs) - set(hb["configs"]):
            sys.exit("test run includes a configuration that was not frozen")
    if args.exclude_from is not None:
        if args.split == "test":
            sys.exit("--exclude-from is for development sub-splits only")
        used = {e["prompt_id"] for f in args.exclude_from for e in json.loads(f.read_text())["example_level"]}
        keep -= used
    split_path = args.out / f"{args.subset}_{args.split}.jsonl"
    split_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["prompt_id"] in keep))
    (args.out / "split_manifest.json").write_text(json.dumps({"salt": SALT, "subset": args.subset, "n_dev": len(dev_ids), "n_test": len(test_ids)}))

    from compass_pcp.harness.cache import ResponseCache

    class Wrapped(SamplerBase):
        """Sampler whose every response is cached to disk, so an interrupted run resumes where it stopped."""

        def __init__(self, fn, cache: ResponseCache):
            self.fn, self.cache = fn, cache
            self.usage: dict = {}  # tokens per vendor:model, summed over responses whose cache entry records usage
            self.with_usage = self.without_usage = 0
            self._lock = threading.Lock()

        def _call(self, message_list):
            with capture_usage() as log:
                text = self.fn(message_list)[0]
            return {"text": text, "usage": dict(log)}

        def __call__(self, message_list):
            value = self.cache.get_or_call(message_list, lambda: self._call(message_list))
            text = value["text"] if isinstance(value, dict) else value  # entries cached before usage was recorded are bare text
            with self._lock:
                if isinstance(value, dict):
                    self.with_usage += 1
                    for k, u in value["usage"].items():
                        t = self.usage.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                        for f in t:
                            t[f] += u[f]
                else:
                    self.without_usage += 1
            return SamplerResponse(response_text=text, actual_queried_message_list=message_list, response_metadata={})

    grader_llm = make_llm(args.grader)
    cache_dir = args.out / "cache"
    def grader_for(spec: str, llm, name: str):
        """One grader cache per configuration, so parallel runs of different configurations never share a file. The
        older shared cache, when present, is read first so development runs keep their graded results."""
        tag = spec.replace(":", "_").replace("/", "_")
        cache = ResponseCache(cache_dir / f"grader__{tag}__{name}.jsonl")
        shared = cache_dir / f"grader__{tag}.jsonl"
        legacy = ResponseCache(shared) if shared.exists() else None

        def grade(msgs):
            # A grader that blocks the prompt (Gemini PROHIBITED_CONTENT) replies with no text on every attempt, and the
            # HealthBench loop would retry forever. After three empty replies the rubric item is scored as not met.
            for _ in range(3):
                text = llm.complete("You grade health responses against a rubric.", msgs, max_tokens=1200)
                if text.strip():
                    return text
            return json.dumps({"explanation": "grader returned no text (prompt blocked); scored as not met",
                               "criteria_met": False})

        def call(msgs):
            if legacy is not None:
                return legacy.get_or_call(msgs, lambda: grade(msgs)), {}
            return grade(msgs), {}

        return Wrapped(call, cache)

    for name in args.configs:
        before = usage_snapshot()
        config = Configuration(name)
        grader = grader_for(args.grader, grader_llm, name)
        arm = build_arm(config, args.backbone, args.second or args.backbone, require_review=not args.allow_unsigned_scripts,
                        deliberate_spec=args.deliberate)
        tag = f"{name}__{args.backbone}__{args.deliberate or ''}__{args.second or ''}__{code_sha(config)}".replace(":", "_").replace("/", "_")
        sampler = Wrapped(lambda msgs, a=arm, c=config: a.next_message(msgs, configuration=c, context=NO_RECORD_CONTEXT),
                          ResponseCache(cache_dir / f"sampler__{tag}.jsonl"))
        ev = HealthBenchEval(grader_model=grader, num_examples=args.n, n_repeats=args.repeats, n_threads=args.threads,
                             input_path=str(split_path))
        result = ev(sampler)
        (args.out / f"{name}.json").write_text(json.dumps({
            "config": name, "subset": args.subset, "split": args.split, "backbone": args.backbone,
            "second": args.second, "grader": args.grader, "n": args.n, "repeats": args.repeats,
            "score": result.score, "metrics": result.metrics,
            "example_level": (result.metadata or {}).get("example_level_metadata"),
            "usage_after": usage_snapshot(), "usage_before": before,
            "usage_per_arm": {"tokens": sampler.usage, "responses_with_usage": sampler.with_usage,
                              "responses_without_usage": sampler.without_usage},
        }, default=str))
        print(name, result.score)
        if args.grader2:
            g2 = grader_for(args.grader2, make_llm(args.grader2), name)
            # A fresh reader of the same sampler cache replays the identical responses, so no model is called again.
            replay = Wrapped(_not_cached, ResponseCache(cache_dir / f"sampler__{tag}.jsonl"))
            ev2 = HealthBenchEval(grader_model=g2, num_examples=args.n, n_repeats=args.repeats, n_threads=args.threads,
                                  input_path=str(split_path))
            r2 = ev2(replay)
            (args.out / f"{name}__grader2.json").write_text(json.dumps({
                "config": name, "grader": args.grader2, "score": r2.score, "metrics": r2.metrics,
                "example_level": (r2.metadata or {}).get("example_level_metadata"),
            }, default=str))
            print(name, "grader2", r2.score)


if __name__ == "__main__":
    main()
