"""COMPASS v4 Problem A development comparison on the HealthBench non-Consensus pool, at batch prices.

design/v4_spec.md, Development and test, step 2. The pool is the 1,329 HealthBench conversations outside the Consensus
subset (opt_pool/nonconsensus_dev.jsonl); no Consensus conversation is read. Arms, chosen with --arms:
  v3                   frozen ai_pcp_v3 (AiPcpV2 built as for the registered run)
  v4_contract          AiPcpV4, A1-c (write from the contract) + A2 extended check
  v4_edit              AiPcpV4, A1-d (minimally edit the no-engine draft) + A2
  v4_contract_compact  AiPcpV4, A1-c + A2 + A3 compact policy for non-engine patient replies
  v4_engine_off        AiPcpV4 with engine_off=True (the engine makes no decision; A1-c, no A3)
  v7 ... v10           Paper 2 candidates (design/paper2_plan.md)
  raw_llm, amie_style, mai_dxo_style   the multi-turn comparators of paper/dev_multiturn.py on the frozen comparator
                       backbone, so every arm is graded on the same conversations
For each conversation and repeat (a slot) every arm shares one set of model clients, and each model call is keyed by
(slot, request, occurrence) in one store (paper/batch_stage.py). AiPcpV4 rebuilds the v3 ledger with the same router,
hazard-screen and extractor calls in the same order, so it reads v3's results for that slot and the contrasts are
paired on the same ledger. The v4 B1 structured screen runs in a worker thread; every staged client of a thread's arm
is pointed at the current slot, so the worker reads and writes the same slot. B1 uses the placeholder crisis policy
(compass_pcp/data/policies/crisis_v4.yaml, status placeholder_unsigned) and no B2 classifier.

Commands (after `source scripts/env.sh`; Kev must be up, scripts/ensure_kev.sh):
  python -m paper.v4_dev_staged route  --pool <nonconsensus_dev.jsonl> --out <dir>
      runs only the v3 router on each full conversation, writes route_paths.jsonl and selection.json
      (all engine-routed conversations plus a seeded random sample of non-engine ones)
  python -m paper.v4_dev_staged run    --pool <...> --out <dir> [--arms ...] [--ids f.json] [--realtime] [--budget 600]
  python -m paper.v4_dev_staged status --out <dir>
`run` alternates passes over the slots with batch submissions until every reply exists, then grades every arm with
both registered graders (GPT-4.1, and Gemini 3.1 Pro at the registered setting) through simple-evals with the full
rubric of each conversation. It is safe to stop and restart. It stops before a submission whose projected spend would
take the store past --budget (USD; realtime calls are counted at list price, twice the batch price).
"""

import argparse
import json
import random
import sys
import threading
import time
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import make_llm  # noqa: E402

from compass_pcp.arms.ai_pcp_v2 import EXTRACT, ROUTER  # noqa: E402
from compass_pcp.arms.ai_pcp_v4 import AiPcpV4  # noqa: E402
from compass_pcp.arms.baselines import AmieStyle, MaiDxoStyle, RawLLM  # noqa: E402
from compass_pcp.arms.hazard import SCREEN  # noqa: E402
from compass_pcp.arms.llm import AnthropicLLM, _record  # noqa: E402
from compass_pcp.arms.screen_v4 import SCREEN_V4, run_screen  # noqa: E402
from compass_pcp.harness import frozen  # noqa: E402
from compass_pcp.harness.healthbench import NO_RECORD_CONTEXT  # noqa: E402
from compass_pcp.pathways import PATHWAY_DIR, load_pathway  # noqa: E402
from compass_pcp.policy_v4 import POLICY_DIR, load_contract_policy, load_crisis_policy, load_first_aid  # noqa: E402
from compass_pcp.review_score import load_review_params  # noqa: E402
from paper.batch_stage import BATCH_PRICE, CachedKev, Pending, SlotState, StagedLLM, Store, collect, slot, submit  # noqa: E402
from paper.dev_multiturn import dev_engine, dev_pathways  # noqa: E402
from paper.engine_ablation import build_variant  # noqa: E402

V4_ARMS = {
    "v4_contract": {"writer_variant": "contract"},
    "v4_edit": {"writer_variant": "edit"},
    "v4_contract_compact": {"writer_variant": "contract", "compact_policy": True},
    "v4_engine_off": {"engine_off": True},
    # Paper 2 candidate (design/paper2_plan.md): v6 development contract (standing clause) plus guideline pathways.
    "v7": {"contract_file": "contract_v6_dev.yaml", "pathways": "all"},
    "v8": {"contract_file": "contract_v6_dev.yaml", "pathways": "all", "writer_variant": "contract_full"},
    "v9": {"contract_file": "contract_v9_dev.yaml", "pathways": "all", "writer_variant": "contract_v9"},
    "v10": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v10_dev.yaml", "dev_scripts": True},
    "v11": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v10_dev.yaml", "dev_scripts": True, "engine_flags": {"red_flags_first": True}},
    "v12": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v10_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v12",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}},
    "v13": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v10_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v12",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True},
    "v14": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v10_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True},
    "v15": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v15_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True},
    "v16": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v15_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v1_dev.yaml"},
    "v17": {"contract_file": "contract_v10_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v17_dev.yaml", "dev_scripts": "scripts_dev_v12", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v2_dev.yaml"},
    "v18": {"contract_file": "contract_v18_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v18_dev.yaml", "dev_scripts": "scripts_dev_v18", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v2_dev.yaml", "review_followup": True},
    "v19": {"contract_file": "contract_v19_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v18_dev.yaml", "dev_scripts": "scripts_dev_v18", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v2_dev.yaml", "review_followup": True,
            "resolve_screen_pe": True},
    "v20": {"contract_file": "contract_v19_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v18_dev.yaml", "dev_scripts": "scripts_dev_v20", "dev_pathways": "pathways_dev_v20",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v2_dev.yaml", "review_followup": True,
            "resolve_screen_pe": True, "complete_urgent": True},
    # Paper 2 freeze (docs/regulatory/freeze/paper2_v22/freeze.json): v19 with the narrower heart-check reset and no heart check while
    # chest pain, shortness of breath or fainting is an active complaint.
    "v22": {"contract_file": "contract_v19_dev.yaml", "pathways": "all", "writer_variant": "contract_v9",
            "crisis_file": "crisis_v18_dev.yaml", "dev_scripts": "scripts_dev_v18", "dev_pathways": "pathways_dev_v14",
            "engine_flags": {"red_flags_first": True, "skip_unanswered": True}, "ask_after_downgrade": True,
            "kev_review_only": True, "first_aid_file": "first_aid_v2_dev.yaml", "review_followup": True,
            "resolve_screen_pe": True, "resolve_cardiac_only": True,
            "screen_check_exclude": ("chest_pain", "dyspnea", "syncope")},
}
COMPARATORS = ("raw_llm", "amie_style", "mai_dxo_style")
ARMS = ["v3", *V4_ARMS, *COMPARATORS]
DATA_DIR = Path(__file__).resolve().parents[1] / "compass_pcp" / "data"
DEV_SCRIPTS_DIR = DATA_DIR / "scripts_dev"
V4_SCREEN_SPEC = "anthropic:claude-sonnet-5-5"  # scripts/run_healthbench.py V4_SCREEN_LLM
SAMPLE_SEED, N_NON_ENGINE = 20261103, 200
GRADE_PENDING = json.dumps({"explanation": "pending batch result", "criteria_met": False})
_TIER_PREFIXES = ((ROUTER.split("{")[0], 0), (SCREEN_V4.split("{")[0], 0), (SCREEN.split("{")[0], 1),
                  (EXTRACT.split("{")[0], 1))


def tier_of(system: str) -> int:
    """Router and the v4 structured screen depend only on the conversation (tier 0); the v3 hazard screen and the
    extractors depend on the router (tier 1); everything else (wording, checks, revisions) is tier 3."""
    for prefix, tier in _TIER_PREFIXES:
        if system.startswith(prefix):
            return tier
    return 3


class StagedLLMv4(StagedLLM):
    """StagedLLM that (1) sends the client's ``thinking`` setting in the batch request, as AnthropicLLM sends it in
    realtime (the v4 screen turns thinking off; v3 clients send none, so their requests and keys are unchanged),
    (2) logs every call on the slot (``st.calls``) for per-reply cost and latency, and (3) records the wall time of
    realtime calls in the stored usage (``latency_s``)."""

    def request(self, system, messages, max_tokens, temperature) -> dict:
        req = super().request(system, messages, max_tokens, temperature)
        thinking = getattr(self.inner, "thinking", None)
        if self.vendor == "anthropic" and thinking is not None:
            req["thinking"] = thinking
        return req

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        from paper.batch_stage import _slot

        st = self.shared or _slot.state
        req = self.request(system, messages, max_tokens, temperature)
        key = st.key([self.spec, req])
        tier = self.tier_of(system)
        hit = self.store.get(key)
        if hit is None and self.realtime:
            before = (self.inner.usage.input_tokens, self.inner.usage.output_tokens)
            t0 = time.monotonic()
            text = self.inner.complete(system, messages, max_tokens=max_tokens, temperature=temperature)
            usage = {"input_tokens": self.inner.usage.input_tokens - before[0],
                     "output_tokens": self.inner.usage.output_tokens - before[1],
                     "latency_s": round(time.monotonic() - t0, 3)}
            self.store.put(key, text, usage, self.spec)
            hit = self.store.get(key)
        if hit is not None:
            _record(self, hit["usage"].get("input_tokens", 0), hit["usage"].get("output_tokens", 0))
            log = getattr(st, "calls", None)
            if log is not None:
                log.append({"spec": self.spec, "tier": tier, "screen_v4": system.startswith(_TIER_PREFIXES[1][0]),
                            **{k: hit["usage"].get(k) for k in ("input_tokens", "output_tokens", "latency_s")}})
            return hit["text"]
        if tier > st.min_miss:
            if tier >= 3:
                raise Pending
            return self.placeholder
        self.store.enqueue(key, {"spec": self.spec, "vendor": self.vendor, "model": self.model, "body": req})
        st.min_miss = min(st.min_miss, tier)
        if tier >= 3:
            raise Pending
        return self.placeholder


class Bundle:
    """One thread's arms: a staged v3 arm and the v4 wrappers around it, sharing the same staged clients."""

    def __init__(self, base, screen_inner, store: Store, realtime: bool, arms: list[str]):
        wrapped: dict[int, StagedLLMv4] = {}

        def staged(llm):
            if llm is None:
                return None
            if id(llm) not in wrapped:  # router, extractor A and dialogue are one client in v3; keep them one
                wrapped[id(llm)] = StagedLLMv4(llm, store, tier_of, realtime=realtime)
            return wrapped[id(llm)]

        for attr in ("router", "extractor_a", "dialogue", "hazard_llm", "extractor_b"):
            setattr(base, attr, staged(getattr(base, attr)))
        if base.hazard_judge is not None:
            base.hazard_judge = CachedKev(base.hazard_judge, store)
        screen = staged(screen_inner)
        self.base, self.screen = base, screen
        self.staged = list(wrapped.values())
        crisis, contract, review = load_crisis_policy(), load_contract_policy(), load_review_params()
        self.crisis = crisis
        def kwargs(name: str) -> dict:
            kw = dict(V4_ARMS[name])
            kw["contract_policy"] = load_contract_policy(POLICY_DIR / kw.pop("contract_file")) if "contract_file" in kw else contract
            if kw.get("pathways") == "all":
                kw["pathways"] = {p.stem: load_pathway(p) for p in sorted(PATHWAY_DIR.glob("*.yaml"))}
            kw["crisis_policy"] = load_crisis_policy(POLICY_DIR / kw.pop("crisis_file")) if "crisis_file" in kw else crisis
            ds = kw.pop("dev_scripts", False)  # True: scripts_dev/; a string: that directory under compass_pcp/data
            kw["engine"] = dev_engine(base, (DATA_DIR / ds if isinstance(ds, str) else DEV_SCRIPTS_DIR) if ds else None,
                                      kw.pop("engine_flags", None))
            if "first_aid_file" in kw:
                kw["first_aid"] = load_first_aid(POLICY_DIR / kw.pop("first_aid_file"))
            if "dev_pathways" in kw:
                kw["pathways"] = dev_pathways(kw["pathways"], DATA_DIR / kw.pop("dev_pathways"))
            return kw

        self.v4 = {name: AiPcpV4(screen_llm=screen, review_params=review, **kwargs(name))
                   for name in arms if name in V4_ARMS}
        backbone = frozen.frozen_thresholds()["healthbench"]["comparator_backbone"]
        kinds = {"raw_llm": RawLLM, "amie_style": AmieStyle, "mai_dxo_style": MaiDxoStyle}
        self.comparators = {name: kinds[name](staged(make_llm(backbone))) for name in arms if name in kinds}
        self.staged = list(wrapped.values())

    @classmethod
    def real(cls, store: Store, realtime: bool, arms: list[str]) -> "Bundle":
        """The frozen ai_pcp_v3 arm (built as for the registered run) and the v4 screen client of build_v4."""
        base = build_variant("v3")
        assert base.router is base.dialogue and base.extractor_a is base.dialogue
        screen = AnthropicLLM(model=V4_SCREEN_SPEC.split(":", 1)[1], thinking={"type": "between_tools"})
        return cls(base, screen, store, realtime, arms)

    def bind(self, st: SlotState) -> None:
        """Point every staged client at this slot (the B1 screen runs on a worker thread without the thread-local)."""
        st.calls = []
        for s in self.staged:
            s.shared = st


def v3_path(arm, meta: dict) -> dict:
    return {"user_type": arm.user_type, "complaints": meta.get("complaints"), "decision": meta.get("decision"),
            "decisions": meta.get("decisions"), "hazard_tier": meta.get("hazard_tier"),
            "care_team_flag_tier": meta.get("care_team_flag_tier"), "reply_path": "v3"}


def _done(out: Path) -> set:
    p = out / "replies.jsonl"
    if not p.exists():
        return set()
    return {(r["prompt_id"], r["repeat"], r["arm"]) for r in map(json.loads, p.read_text().splitlines())}


def run_slot(b: Bundle, row: dict, rep: int, arms: list[str], done: set, realtime: bool) -> tuple[list[dict], int]:
    """Every arm's reply for one slot: the finished replies and the number still waiting for batch results."""
    pid, msgs = row["prompt_id"], row["prompt"]
    new: list[dict] = []
    with slot((pid, rep)) as st:
        b.bind(st)
        # Tier 0 and 1 calls of every arm, queued together: the v3 ledger and the v4 structured screen.
        prebuilt = b.base.build_ledger(msgs, list(NO_RECORD_CONTEXT.get("findings", [])))
        run_screen(b.screen, msgs, b.crisis, timeout=None if realtime else 3600)
        if st.missed:
            return [], len([a for a in arms if (pid, rep, a) not in done])
        ledger_calls = [c for c in st.calls if not c["screen_v4"]]  # v3 reuses this ledger; its log needs these calls
    missing = 0
    for name in arms:
        if (pid, rep, name) in done:
            continue
        with slot((pid, rep)) as st:
            b.bind(st)
            t0 = time.monotonic()
            try:
                if name == "v3":
                    text, meta = b.base.next_message(msgs, context=NO_RECORD_CONTEXT, prebuilt=prebuilt)
                    path, info = v3_path(b.base, meta), {"questions": meta.get("questions")}
                    st.calls[:0] = ledger_calls
                elif name in b.comparators:
                    text, _ = b.comparators[name].next_message(msgs, context=NO_RECORD_CONTEXT)
                    path, info = {"reply_path": name}, {}
                else:
                    text, info = b.v4[name].next_message(msgs, context=NO_RECORD_CONTEXT)
                    path = info.pop("path")
                if st.missed:  # a lower-tier call was answered with a placeholder; the reply is not final
                    raise Pending
                new.append({"prompt_id": pid, "repeat": rep, "arm": name, "text": text, "path": path, "info": info,
                            "calls": st.calls, "wall_s": round(time.monotonic() - t0, 3) if realtime else None})
            except Pending:
                missing += 1
    return new, missing


def sample_pass(rows, repeats, store, out: Path, realtime: bool, threads: int, arms: list[str]) -> int:
    """One pass over every slot; returns the number of replies still missing."""
    done = _done(out)
    lock, local = threading.Lock(), threading.local()

    def one(task):
        row, rep = task
        if all((row["prompt_id"], rep, a) in done for a in arms):
            return 0
        if not hasattr(local, "b"):
            local.b = Bundle.real(store, realtime, arms)
        new, missing = run_slot(local.b, row, rep, arms, done, realtime)
        with lock, (out / "replies.jsonl").open("a") as fh:
            for r in new:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return missing

    tasks = [(row, rep) for row in rows for rep in range(repeats)]
    with cf.ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, tasks))


def grade_pass(rows, repeats, store, out: Path, realtime: bool, threads: int, arms: list[str]) -> int:
    """Grade every arm with both registered graders through simple-evals (paper/ablation_staged.py grade_pass, with the
    arm list as a parameter). Returns the number of grades queued."""
    import os

    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    from simple_evals.healthbench_eval import HealthBenchEval  # type: ignore[import-not-found]
    from simple_evals.types import SamplerBase, SamplerResponse  # type: ignore[import-not-found]

    hb = frozen.frozen_thresholds()["healthbench"]
    replies: dict[tuple, str] = {}
    for line in (out / "replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        replies[(r["arm"], r["prompt_id"], r["repeat"])] = r["text"]
    by_prompt = {json.dumps(r["prompt"], sort_keys=True): r["prompt_id"] for r in rows}
    queued_before = len(store.queue)
    for name in arms:
        for key, spec in (("grader", hb["grader"]), ("grader2", hb["grader2"])):
            dest = out / f"{name}__{key}.json"
            if dest.exists():
                continue
            # Identical replies in two arms map to the same grade keys, so they are graded once.
            shared = SlotState(("grade", spec))
            g = StagedLLM(make_llm(spec), store, lambda s: 1, placeholder=GRADE_PENDING, realtime=realtime, shared=shared)

            def grade(msgs, g=g):
                for _ in range(3):
                    text = g.complete("You grade health responses against a rubric.", msgs, max_tokens=1200)
                    if text.strip():
                        return text
                return json.dumps({"explanation": "grader returned no text (prompt blocked); scored as not met",
                                   "criteria_met": False})

            class Grader(SamplerBase):
                def __call__(self, msgs):
                    return SamplerResponse(response_text=grade(msgs), actual_queried_message_list=msgs, response_metadata={})

            seen: dict[str, int] = {}
            lock = threading.Lock()

            class Replay(SamplerBase):
                def __call__(self, msgs, name=name):
                    pid = by_prompt[json.dumps(msgs, sort_keys=True)]
                    with lock:
                        k = seen.get(pid, 0)
                        seen[pid] = k + 1
                    return SamplerResponse(response_text=replies[(name, pid, k)], actual_queried_message_list=msgs,
                                           response_metadata={})

            res = HealthBenchEval(grader_model=Grader(), n_repeats=repeats, n_threads=threads,
                                  input_path=str(out / "split.jsonl"))(Replay())
            if not shared.missed:
                dest.write_text(json.dumps({
                    "config": name, "grader": spec, "pool": "healthbench_nonconsensus", "repeats": repeats,
                    "n": len(rows), "score": res.score, "metrics": res.metrics,
                    "example_level": res.metadata.get("example_level_metadata") or [],
                }, default=str))
                print(name, key, round(res.score, 4), flush=True)
    return len(store.queue) - queued_before


# ---- cost ----------------------------------------------------------------------------------------------------------
def spent(store: Store) -> float:
    """USD so far: batch price for batch results, list price (twice batch) for realtime calls."""
    usd = 0.0
    for r in store.results.values():
        p = BATCH_PRICE.get(r["spec"])
        if p is None:
            continue
        u = r["usage"]
        c = u.get("input_tokens", 0) / 1e6 * p[0] + u.get("output_tokens", 0) / 1e6 * p[1]
        usd += c * (2 if u.get("latency_s") is not None else 1)
    return usd


def projected(store: Store) -> float:
    """Upper-side estimate of the queued requests at batch price: input from request size (3.5 characters per token),
    output from the mean output of stored results of the same model and system-prompt family, else the token cap."""
    means: dict[str, float] = {}
    by_spec: dict[str, list[int]] = {}
    for r in store.results.values():
        by_spec.setdefault(r["spec"], []).append(r["usage"].get("output_tokens", 0))
    for s, v in by_spec.items():
        means[s] = sum(v) / len(v)
    usd = 0.0
    for r in store.queue.values():
        p = BATCH_PRICE.get(r["spec"])
        if p is None:
            continue
        body = r["body"]
        cap = body.get("max_tokens") or (body.get("generationConfig") or {}).get("maxOutputTokens") or 2000
        out_tok = means.get(r["spec"], cap)
        usd += len(json.dumps(body)) / 3.5 / 1e6 * p[0] + min(out_tok * 1.5, cap) / 1e6 * p[1]
    return usd


def run_loop(phases, rows, repeats, store, out, realtime, threads, arms, poll, budget) -> None:
    for phase in phases:
        while True:
            while collect(store):
                time.sleep(poll)
            missing = phase(rows, repeats, store, out, realtime, threads, arms)
            print(phase.__name__, "missing", missing, "queued", len(store.queue), "spent", round(spent(store), 2),
                  flush=True)
            if not store.queue:
                if missing and not realtime:
                    sys.exit(f"{missing} still missing with nothing to submit; inspect {out}")
                break
            proj = projected(store)
            if spent(store) + proj > budget:
                sys.exit(f"stopping: spent {spent(store):.2f} + projected {proj:.2f} would pass the budget {budget}")
            jobs = submit(store)
            print("submitted", [(j["vendor"], j["model"], j["n"]) for j in jobs], "projected", round(proj, 2), flush=True)
    print(json.dumps({"cost": store.cost(), "spent_usd": round(spent(store), 2)}, indent=1))


# ---- router stage --------------------------------------------------------------------------------------------------
def route_pass(rows, repeats, store, out, realtime, threads, arms) -> int:
    """v3 router on each full conversation (paper/route_paths.py), keyed as the full run's last router call of the
    slot (repeat 0), so the full run reuses it when the conversation ends with a user message."""
    local = threading.local()
    res: dict[str, dict] = {}

    def one(row):
        if not hasattr(local, "b"):
            local.b = Bundle.real(store, realtime, [])
        with slot((row["prompt_id"], 0)) as st:
            local.b.bind(st)
            cats, _, user_type, _ = local.b.base._route_full(row["prompt"])
            if st.missed:
                return 1
        path = "clinician" if user_type == "health_professional" else ("engine" if cats else "general")
        res[row["prompt_id"]] = {"prompt_id": row["prompt_id"], "user_type": user_type, "n_complaints": len(cats),
                                 "path": path}
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        missing = sum(ex.map(one, rows))
    if not missing:
        with (out / "route_paths.jsonl").open("w") as fh:
            for r in sorted(res.values(), key=lambda r: r["prompt_id"]):
                fh.write(json.dumps(r) + "\n")
    return missing


def select(out: Path) -> dict:
    paths = [json.loads(x) for x in (out / "route_paths.jsonl").open()]
    engine = sorted(r["prompt_id"] for r in paths if r["path"] == "engine")
    rest = sorted(r["prompt_id"] for r in paths if r["path"] != "engine")
    sample = sorted(random.Random(SAMPLE_SEED).sample(rest, min(N_NON_ENGINE, len(rest))))
    sel = {"engine_routed": engine, "non_engine_sample": sample, "seed": SAMPLE_SEED, "n_pool": len(paths),
           "counts": {p: sum(r["path"] == p for r in paths) for p in ("engine", "clinician", "general")}}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["route", "run", "status"])
    ap.add_argument("--pool", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of prompt ids (smoke test); default selection.json")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--realtime", action="store_true", help="answer misses at once at list price (smoke test)")
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--budget", type=float, default=600.0, help="USD ceiling for everything in this store")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    store = Store(a.out / "store")
    if a.cmd == "status":
        print(json.dumps({"running_batches": collect(store), "cost": store.cost(), "spent_usd": round(spent(store), 2),
                          "replies": len(_done(a.out))}, indent=1))
        return
    if "test_healthbench" in str(a.pool) or "743" in str(a.pool) or "dev_hb_input" in str(a.pool):
        sys.exit("this driver runs only on the non-Consensus pool")
    rows = sorted((json.loads(x) for x in a.pool.open()), key=lambda r: r["prompt_id"])
    if a.cmd == "route":
        run_loop([route_pass], rows, 1, store, a.out, a.realtime, a.threads, [], a.poll, a.budget)
        print(json.dumps({k: v for k, v in select(a.out).items() if k in ("counts", "n_pool")}))
        return
    if a.ids:
        keep = set(json.loads(a.ids.read_text()))
    else:
        sel = json.loads((a.out / "selection.json").read_text())
        keep = set(sel["engine_routed"]) | set(sel["non_engine_sample"])
    rows = [r for r in rows if r["prompt_id"] in keep]
    (a.out / "split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    phases = [sample_pass] if a.no_grade else [sample_pass, grade_pass]
    run_loop(phases, rows, a.repeats, store, a.out, a.realtime, a.threads, a.arms, a.poll, a.budget)


if __name__ == "__main__":
    main()
