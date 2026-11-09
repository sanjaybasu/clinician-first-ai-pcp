"""Choose the HealthBench grader by agreement with physicians (design/paper2_plan.md, section 3).

Runs the HealthBench meta-evaluation (simple-evals healthbench_meta_eval.py) for each candidate grader: the grader
labels each rubric item met or not met for a physician-labeled response, and the score is the balanced F1 of its
labels against the physicians' labels, pairwise, next to the same metric for each physician against the other
physicians (the ceiling). Every candidate grades the same seeded sample (num_examples, seed 0 in simple-evals) with
the registered grading system prompt. The meta-evaluation data (openai/healthbench on Hugging Face) are public HealthBench data; no patient data are used.

A candidate that returns no valid label after four attempts is scored "not met" for that item, and the count of such
items is reported. Each item's label is saved, keyed by the SHA-256 of its grader prompt, for paired comparisons.

Usage (after `source scripts/env.sh`): python -m paper.grader_physician_agreement --out <dir> [--n 1000]
       [--graders openai:gpt-4.1-2025-04-14 ...] [--threads 16]
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import make_llm

SYSTEM = "You grade health responses against a rubric."  # paper/v4_dev_staged.py grade_pass
CANDIDATES = (
    "openai:gpt-4.1-2025-04-14",          # registered grader (HealthBench reference grader)
    "google:gemini-3.1-pro-preview@LOW",  # registered second grader
    "google:gemini-3.1-pro-preview@HIGH",
    "google:gemini-3.8-flash",
    "openai:gpt-5.5",
    "openai:gpt-6.1-sol",
)
NO_LABEL = json.dumps({"explanation": "grader returned no valid label after four attempts", "criteria_met": False})


class OpenAIReasoningChat:
    """OpenAI reasoning models (GPT-5 and later) reject max_tokens and temperature."""

    def __init__(self, model: str):
        import openai

        self.model, self._c = model, openai.OpenAI(timeout=600.0, max_retries=4)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        r = self._c.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system}, *messages],
                                            max_completion_tokens=8000)
        return r.choices[0].message.content or ""


def meta_data() -> Path:
    """The meta-evaluation file from OpenAI's openai/healthbench dataset on Hugging Face, from the local cache."""
    name = "2025-05-07-06-14-12_oss_meta_eval.jsonl"
    hits = sorted((Path.home() / ".cache/huggingface/hub/datasets--openai--healthbench/snapshots").glob(f"*/{name}"))
    if not hits:
        raise SystemExit(f"{name} is not in the Hugging Face cache; download it from the openai/healthbench dataset")
    return hits[-1]


def client(spec: str):
    vendor, model = spec.split(":", 1)
    if vendor == "openai" and not model.startswith("gpt-4"):
        return OpenAIReasoningChat(model)
    return make_llm(spec)


def run(spec: str, out: Path, n: int, threads: int) -> dict:
    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    from simple_evals import healthbench_meta_eval as hme  # type: ignore[import-not-found]
    from simple_evals.healthbench_eval import parse_json_to_dict  # type: ignore[import-not-found]
    from simple_evals.healthbench_meta_eval import HealthBenchMetaEval  # type: ignore[import-not-found]
    from simple_evals.types import SamplerBase, SamplerResponse  # type: ignore[import-not-found]

    llm, lock, failed, labels = client(spec), threading.Lock(), [0], {}

    class Grader(SamplerBase):
        def __call__(self, msgs):
            for _ in range(4):
                try:
                    text = llm.complete(SYSTEM, msgs, max_tokens=1200)
                except Exception as err:  # noqa: BLE001 - retried; the error type is logged
                    print("error", spec, type(err).__name__, flush=True)
                    continue
                label = parse_json_to_dict(text).get("criteria_met")
                if label in (True, False):
                    with lock:
                        labels[hashlib.sha256(msgs[-1]["content"].encode()).hexdigest()] = label
                    return SamplerResponse(response_text=text, actual_queried_message_list=msgs, response_metadata={})
            with lock:
                failed[0] += 1
            return SamplerResponse(response_text=NO_LABEL, actual_queried_message_list=msgs, response_metadata={})

    hme.INPUT_PATH = str(meta_data())  # the public blob URL in simple-evals no longer resolves
    g = Grader()
    res = HealthBenchMetaEval(grader_model=g, num_examples=n, n_threads=threads)(g)  # the sampler under test is the grader
    phys = res.metadata["physician_agreement_metric_lists"].get("pairwise_physician_f1_balanced", {})
    phys = {pid: v["value"] for pid, v in phys.items() if v.get("value") is not None}
    phys_vals = list(phys.values())
    rec = {"grader": spec, "n_items": n, "no_valid_label": failed[0],
           "model_f1_balanced": res.metrics.get("pairwise_model_f1_balanced"),
           "physician_f1_balanced_mean": round(statistics.fmean(phys_vals), 4) if phys_vals else None,
           "physician_f1_balanced_median": round(statistics.median(phys_vals), 4) if phys_vals else None,
           "n_physicians": len(phys_vals),
           "physician_f1_balanced_by_physician": phys,
           "metrics": {k: v for k, v in res.metrics.items() if k.startswith("pairwise_")},
           "labels_by_prompt_sha256": labels}  # for paired bootstrap between graders
    (out / f"{spec.replace(':', '_').replace('@', '_')}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--graders", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.graders:
        dest = a.out / f"{spec.replace(':', '_').replace('@', '_')}.json"
        if dest.exists():
            continue
        rec = run(spec, a.out, a.n, a.threads)
        print(json.dumps({k: rec[k] for k in ("grader", "model_f1_balanced", "physician_f1_balanced_mean",
                                              "no_valid_label")}), flush=True)


if __name__ == "__main__":
    main()
