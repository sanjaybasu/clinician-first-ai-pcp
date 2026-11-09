"""Paired ablations of the frozen ai_pcp_v3 on the HealthBench Consensus test split, run at batch prices.

For each test conversation and repeat (a slot), the frozen ai_pcp_v3 builds its ledger (router, hazard screens,
extractors) and writes its reply, exactly as in the registered run. Each ablation then reuses that ledger and changes
one component:
  engine_off            the engine makes no decision; the reply comes from the composer's general path
  minus_stopping_rule   WorkupConfig.enforce_stopping_rule = False
  fixed_question_order  WorkupConfig.fixed_question_order = True
  rule_out_authority    WorkupConfig.bayes_rule_out_authority = True (panel item 3 switched off)
  eig_questions         WorkupConfig.question_policy = "eig" (panel item 5 switched off)
  single_extractor      the ledger is rebuilt from extractor A's findings alone (same router and screen results)
When an ablation leads to the same model calls as ai_pcp_v3, it reuses ai_pcp_v3's replies, so a difference between
arms comes only from the component removed. Every reply's path (user type, complaints, decision, hazard tier) is saved.
Each thread builds its own arm (AiPcpV2 keeps user_type on the instance). The frozen code is not edited.

Usage (after `source scripts/env.sh`):
  python -m paper.ablation_staged run   --test-dir <test_healthbench> --out <dir> [--n 5] [--repeats 3] [--realtime]
  python -m paper.ablation_staged status --out <dir>
`run` loops: one pass over all slots, submit the missing calls as batches, wait, collect, repeat; then grades every
arm with both frozen graders the same way. It is safe to stop and restart at any point.
"""

import argparse
import copy
import json
import sys
import threading
import time
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import make_llm  # noqa: E402

from compass_pcp.arms.ai_pcp_v2 import EXTRACT, ROUTER  # noqa: E402
from compass_pcp.arms.hazard import SCREEN  # noqa: E402
from compass_pcp.harness import frozen  # noqa: E402
from compass_pcp.harness.healthbench import NO_RECORD_CONTEXT  # noqa: E402
from paper.batch_stage import CachedKev, Pending, SlotState, StagedLLM, Store, collect, slot, submit  # noqa: E402
from paper.engine_ablation import WORKUP, build_variant  # noqa: E402

ARMS = ["ai_pcp_v3", "engine_off", *WORKUP, "single_extractor"]
GRADE_PENDING = json.dumps({"explanation": "pending batch result", "criteria_met": False})


def tier_of(system: str) -> int:
    for prefix, tier in ((ROUTER.split("{")[0], 0), (SCREEN.split("{")[0], 1), (EXTRACT.split("{")[0], 1)):
        if system.startswith(prefix):
            return tier
    return 3


def staged_arm(store: Store, realtime: bool):
    arm = build_variant("v3")
    opus = StagedLLM(arm.dialogue, store, tier_of, realtime=realtime)
    assert arm.router is arm.dialogue and arm.extractor_a is arm.dialogue
    arm.router = arm.extractor_a = arm.dialogue = opus
    arm.hazard_llm = StagedLLM(arm.hazard_llm, store, tier_of, realtime=realtime)
    arm.extractor_b = StagedLLM(arm.extractor_b, store, tier_of, realtime=realtime)
    arm.hazard_judge = CachedKev(arm.hazard_judge, store)
    return arm


def variant_of(base, name: str):
    a = copy.copy(base)  # shares the clients and keeps base.user_type from the ledger build
    if name == "engine_off":
        a._decide = lambda *x, **k: ([], [])
    elif name in WORKUP:
        a.config = base.config.model_copy(update=WORKUP[name])
    return a


def path_of(arm, meta: dict) -> dict:
    return {"user_type": arm.user_type, "complaints": meta.get("complaints"), "decision": meta.get("decision"),
            "decisions": meta.get("decisions"), "hazard_tier": meta.get("hazard_tier"),
            "care_team_flag_tier": meta.get("care_team_flag_tier")}


def sample_pass(rows, repeats, store, out: Path, realtime: bool, threads: int) -> int:
    """One pass over every slot; returns the number of replies still missing."""
    done_path = out / "replies.jsonl"
    done = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt_id"], r["repeat"], r["arm"]))
    lock, local = threading.Lock(), threading.local()

    def one(task):
        row, rep = task
        pid, msgs = row["prompt_id"], row["prompt"]
        if all((pid, rep, a) in done for a in ARMS):
            return 0
        if not hasattr(local, "arm"):
            local.arm = staged_arm(store, realtime)
        base, new, missing = local.arm, [], 0
        with slot((pid, rep)) as st:
            prebuilt = base.build_ledger(msgs, list(NO_RECORD_CONTEXT.get("findings", [])))
            if st.missed:
                return len(ARMS)
        for name in ARMS:
            if (pid, rep, name) in done:
                continue
            with slot((pid, rep)) as st:
                try:
                    if name == "single_extractor":
                        a = copy.copy(base)
                        a.extractor_b = None
                        built = a.build_ledger(msgs, [])
                        if st.missed:
                            raise Pending
                    else:
                        a, built = variant_of(base, name), prebuilt
                    text, meta = a.next_message(msgs, context=NO_RECORD_CONTEXT, prebuilt=built)
                    new.append({"prompt_id": pid, "repeat": rep, "arm": name, "text": text, "path": path_of(a, meta)})
                except Pending:
                    missing += 1
        with lock, done_path.open("a") as fh:
            for r in new:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return missing

    tasks = [(row, rep) for row in rows for rep in range(repeats)]
    with cf.ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, tasks))


def grade_pass(rows, repeats, store, out: Path, realtime: bool, threads: int) -> int:
    """Grade every arm with both frozen graders through simple-evals; returns the number of grades still missing."""
    import os

    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    from simple_evals.healthbench_eval import HealthBenchEval  # type: ignore[import-not-found]
    from simple_evals.types import SamplerBase, SamplerResponse  # type: ignore[import-not-found]

    hb = frozen.frozen_thresholds()["healthbench"]
    replies: dict[tuple, str] = {}
    paths: dict[tuple, dict] = {}
    for line in (out / "replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        replies[(r["arm"], r["prompt_id"], r["repeat"])] = r["text"]
        paths[(r["arm"], r["prompt_id"], r["repeat"])] = r["path"]
    split = out / "split.jsonl"
    queued_before = len(store.queue)
    for name in ARMS:
        for key, spec in (("grader", hb["grader"]), ("grader2", hb["grader2"])):
            dest = out / f"{name}__{key}.json"
            if dest.exists():
                continue
            # Occurrence counts are shared across simple-evals' threads and restart for each arm, so a reply that is
            # identical in two arms is graded once and both arms get that grade.
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
                    pid = next(r["prompt_id"] for r in rows if r["prompt"] == msgs)
                    with lock:
                        k = seen.get(pid, 0)
                        seen[pid] = k + 1
                    return SamplerResponse(response_text=replies[(name, pid, k)], actual_queried_message_list=msgs,
                                           response_metadata={"path": paths[(name, pid, k)]})

            res = HealthBenchEval(grader_model=Grader(), n_repeats=repeats, n_threads=threads, input_path=str(split))(Replay())
            if not shared.missed:  # every grade came from the store; a queued or still-running grade makes this False
                ex = res.metadata.get("example_level_metadata") or []
                dest.write_text(json.dumps({
                    "config": name, "grader": spec, "split": "test", "repeats": repeats, "n": len(rows),
                    "score": res.score, "metrics": res.metrics, "example_level": ex,
                }, default=str))
                print(name, key, round(res.score, 4), flush=True)
    return len(store.queue) - queued_before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--test-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="first n conversations by prompt id (smoke test)")
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of prompt ids to run (smoke test)")
    ap.add_argument("--repeats", type=int, default=frozen.frozen_thresholds()["healthbench"]["repeats"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--realtime", action="store_true", help="answer misses at once at list price (smoke test)")
    ap.add_argument("--poll", type=int, default=300, help="seconds between batch status checks")
    a = ap.parse_args()
    store = Store(a.out / "store")
    if a.cmd == "status":
        print(json.dumps({"running_batches": collect(store), "cost": store.cost()}, indent=1))
        return

    rows = sorted((json.loads(x) for x in (a.test_dir / "consensus_test.jsonl").open()), key=lambda r: r["prompt_id"])
    if a.ids:
        keep = set(json.loads(a.ids.read_text()))
        rows = [r for r in rows if r["prompt_id"] in keep]
    if a.n:
        rows = rows[: a.n]
    (a.out / "split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for phase in (sample_pass, grade_pass):
        while True:
            while collect(store):
                time.sleep(a.poll)
            missing = phase(rows, a.repeats, store, a.out, a.realtime, a.threads)
            print(phase.__name__, "missing", missing, "queued", len(store.queue), flush=True)
            if not store.queue:
                if missing and not a.realtime:
                    sys.exit(f"{missing} still missing with nothing to submit; inspect {a.out}")
                break
            jobs = submit(store)
            print("submitted", [(j["vendor"], j["model"], j["n"]) for j in jobs], flush=True)
    print(json.dumps(store.cost(), indent=1))


if __name__ == "__main__":
    main()
