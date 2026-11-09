"""HealthBench adapter: a public, comparable layer. It never gates a lane.

HealthBench (Arora et al., arXiv 2505.08775) grades multi-turn conversations
against physician-written rubrics; code and data are in openai/simple-evals
(``healthbench_eval.py``). This module wraps a Compass configuration as a
simple-evals sampler: it takes the conversation so far and returns the next
message the patient would see (a question, a recommendation, or a handoff).

Rules carried from the protocol:
* No patient record. Clinical Context is "no source available".
* Same backbone model across configurations (see configurations.py).
* A frozen 20% dev split for prompt and threshold work; the other 80% is scored
  once, after the architecture is locked at a commit hash.
* Three runs per conversation; disagreement is reported.
* Default grader for comparability, plus a second-vendor grader as sensitivity.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from compass_pcp.harness.configurations import Configuration
from compass_pcp.harness.metrics import mcnemar_exact_p, paired_bootstrap_diff

# Pinned to the files healthbench_eval.py loads; re-check if simple-evals moves them.
DATA_URLS = {
    "full": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}
REFERENCE_GRADER = "gpt-4.1-2025-04-14"
NO_RECORD_CONTEXT = {"clinical_context": "no source available"}


class NextMessageEndpoint(Protocol):
    """Compass under one configuration: conversation -> next patient-visible message."""

    def next_message(
        self, messages: list[dict[str, str]], *, configuration: Configuration, context: dict[str, str]
    ) -> tuple[str, dict[str, Any]]: ...


@dataclass
class SamplerResponse:
    """Matches the attributes healthbench_eval.py reads from a sampler response."""

    response_text: str
    response_metadata: dict[str, Any]
    actual_queried_message_list: list[dict[str, str]]


@dataclass
class CompassSampler:
    endpoint: NextMessageEndpoint
    configuration: Configuration
    calls: int = field(default=0)

    def __call__(self, message_list: list[dict[str, str]]) -> SamplerResponse:
        self.calls += 1
        text, meta = self.endpoint.next_message(message_list, configuration=self.configuration, context=NO_RECORD_CONTEXT)
        return SamplerResponse(
            response_text=text,
            response_metadata={**meta, "configuration": self.configuration.value},
            actual_queried_message_list=message_list,
        )


def dev_split(prompt_ids: list[str], *, salt: str, dev_frac: float = 0.20) -> tuple[list[str], list[str]]:
    """Deterministic hash split, so the dev set cannot drift between runs."""
    dev, test = [], []
    for pid in sorted(prompt_ids):
        h = int(hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev if h < dev_frac else test).append(pid)
    return dev, test


def compare_scores(a: dict[str, float], b: dict[str, float], *, n_boot: int = 10_000) -> dict[str, float]:
    """Paired bootstrap over shared prompt ids of per-conversation scores."""
    ids = sorted(set(a) & set(b))
    diff, lo, hi = paired_bootstrap_diff([a[i] for i in ids], [b[i] for i in ids], n_boot=n_boot)
    return {"n": len(ids), "diff": diff, "ci_low": lo, "ci_high": hi}


def compare_consensus(a: dict[str, bool], b: dict[str, bool]) -> dict[str, float]:
    """Paired pass/fail on Consensus criteria (keys are prompt_id:criterion)."""
    keys = sorted(set(a) & set(b))
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    return {"n": len(keys), "a_only": only_a, "b_only": only_b, "p_exact": mcnemar_exact_p(only_a, only_b)}
