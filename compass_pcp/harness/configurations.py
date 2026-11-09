"""Pre-specified configurations and hypotheses for the comparative evaluation.

One configuration list drives both the Waymark retrospective cohorts and the
HealthBench public layer, so results line up arm for arm. Every arm uses the
same backbone model and the same patient-facing output constraints (SMS length,
reading level), so differences come from structure, not from the model.

Comparator arms are reimplementations from the published method descriptions;
each one needs a fidelity memo (what the paper specifies, what we built, what we
had to choose) checked by someone outside the build team before the freeze.
"""

from enum import StrEnum

from pydantic import BaseModel


class Configuration(StrEnum):
    # External comparators
    RAW_LLM = "raw_llm"
    LLM_STANDARD_GUARDRAILS = "llm_standard_guardrails"
    AMIE_STYLE = "amie_style"
    MAI_DXO_STYLE = "mai_dxo_style"
    # Proposed system and its leave-one-out ablations
    AI_PCP = "ai_pcp"
    AI_PCP_AMIE_DIALOGUE = "ai_pcp_amie_dialogue"
    AI_PCP_OBLIGATION = "ai_pcp_obligation_questions"  # panel item 5
    AI_PCP_NO_POSTERIOR_RULE_OUT = "ai_pcp_no_posterior_rule_out"  # panel item 3
    AI_PCP_ITEMS_3_5 = "ai_pcp_items_3_and_5"
    # Dual-process tests (System 1 / System 2)
    RAW_DELIBERATE = "raw_llm_deliberate"  # unstructured deliberation: reasoning model, high thinking, no scaffold
    AI_PCP_DELIBERATE = "ai_pcp_deliberate"  # structured deliberation: same reasoning model inside the AI PCP engine
    AI_PCP_DUAL_PROCESS = "ai_pcp_dual_process"  # fast tier every turn, deliberate tier on triggers
    # The configuration frozen for the test sets
    AI_PCP_FINAL = "ai_pcp_final"
    AI_PCP_V3 = "ai_pcp_v3"  # final plus composer v3 (draft, verify, revise) and care-team-only local judge
    AI_PCP_V4 = "ai_pcp_v4"  # v3 engine plus universal structured crisis screen, response contract and review score
    AI_PCP_NO_STOPPING_RULE = "ai_pcp_minus_stopping_rule"
    AI_PCP_SINGLE_EXTRACTOR = "ai_pcp_minus_second_extractor"
    AI_PCP_NO_GATE = "ai_pcp_minus_rule_gate"
    AI_PCP_ADVISORY_GATE = "ai_pcp_gate_advisory_not_blocking"
    AI_PCP_FIXED_QUESTIONS = "ai_pcp_minus_next_question_selection"


DESCRIPTIONS: dict[Configuration, str] = {
    Configuration.RAW_LLM: "Backbone model with a PCP persona prompt; no tools, no checks.",
    Configuration.LLM_STANDARD_GUARDRAILS: (
        "Backbone plus a conventional guardrail stack: input crisis/emergency screen, scope limits, "
        "output safety classifier, uncertainty disclaimer and human-handoff rule. Operationalizes "
        "CHAI-style assurance practice; CHAI publishes assurance standards, not a guardrail implementation, "
        "so the stack is specified in the fidelity memo."
    ),
    Configuration.AMIE_STYLE: (
        "Inference-time chain per Tu et al. (Nature 2025): summarize dialogue and differential, draft "
        "the reply, self-critique and refine; with the g-AMIE guardrail (Vedadi et al. 2025) that withholds "
        "individualized advice and hands a SOAP note to an overseeing physician."
    ),
    Configuration.MAI_DXO_STYLE: (
        "Orchestrated virtual panel per Nori et al. (arXiv 2506.22405): hypothesis, test-chooser, "
        "challenger, stewardship and checklist roles debating before each action, with a cost budget."
    ),
    Configuration.AI_PCP: "Full architecture: dual extraction, illness scripts, Bayesian workup, "
    "cannot-miss stopping rule, blocking rule gate, lane permissions, deferral.",
    Configuration.AI_PCP_AMIE_DIALOGUE: "Hybrid: AI PCP engine and gate unchanged; the wording layer uses the AMIE-style "
    "analyze, draft and self-critique chain.",
    Configuration.AI_PCP_OBLIGATION: "AI PCP asking only questions that can change a cannot-miss status (panel item 5).",
    Configuration.AI_PCP_NO_POSTERIOR_RULE_OUT: "AI PCP without posterior-only rule-out authority (panel item 3).",
    Configuration.AI_PCP_ITEMS_3_5: "AI PCP with panel items 3 and 5 together.",
    Configuration.RAW_DELIBERATE: "Reasoning model at high thinking with the PCP persona and no scaffold (unstructured System 2).",
    Configuration.AI_PCP_DELIBERATE: "AI PCP pipeline with the reasoning model in every LLM role (structured System 2).",
    Configuration.AI_PCP_DUAL_PROCESS: "Fast-model AI PCP on every turn; the reasoning-model AI PCP answers when a raise-only "
    "trigger fires (urgent or higher hazard, cannot-miss middle zone, extractor disagreement, schema gap, clinician request).",
    Configuration.AI_PCP_FINAL: "AI PCP with the reasoning model in every LLM role, panel items 3 and 5, and two raise-only hazard "
    "screens (local Kev judge and an LLM screen) whose higher tier is kept; the operating point was fixed on the Cohort A tuning half.",
    Configuration.AI_PCP_V3: "As ai_pcp_final, but the reply is drafted first, checked against the engine's required elements "
    "and the rules, and revised once; the local judge flags for the care team only, and patient-facing urgency comes from "
    "the engine and the LLM screen.",
    Configuration.AI_PCP_V4: "The unchanged v3 engine and hazard screens, plus (design/v4_spec.md): a structured crisis screen "
    "on every message with a deterministic router over signed criteria (B1); the engine decision written as a response "
    "contract that the model writes the full reply under (A1), checked by rule and by one semantic call, revised once, "
    "else the signed fallback (A2); protected review floors plus a monotone review score (B3).",
    Configuration.AI_PCP_NO_STOPPING_RULE: "Cannot-miss handled as a weighted term in the objective instead of a stopping rule.",
    Configuration.AI_PCP_SINGLE_EXTRACTOR: "Extractor A only; no reconciliation.",
    Configuration.AI_PCP_NO_GATE: "No rule gate.",
    Configuration.AI_PCP_ADVISORY_GATE: "Gate findings shown as advisory flags; a simulated reviewer accepts drafts.",
    Configuration.AI_PCP_FIXED_QUESTIONS: "Fixed question list per complaint instead of EIG selection.",
}

EXTERNAL = [
    Configuration.RAW_LLM,
    Configuration.LLM_STANDARD_GUARDRAILS,
    Configuration.AMIE_STYLE,
    Configuration.MAI_DXO_STYLE,
]
ABLATIONS = [
    Configuration.AI_PCP_NO_STOPPING_RULE,
    Configuration.AI_PCP_SINGLE_EXTRACTOR,
    Configuration.AI_PCP_NO_GATE,
    Configuration.AI_PCP_ADVISORY_GATE,
    Configuration.AI_PCP_FIXED_QUESTIONS,
]


class Hypothesis(BaseModel):
    id: str
    family: str
    contrast: tuple[Configuration, Configuration]
    endpoint: str
    cohort: str
    kind: str  # "superiority" or "noninferiority"
    margin: float | None = None


def prespecified_hypotheses() -> list[Hypothesis]:
    """Primary family: the frozen AI PCP vs each external comparator on the safety primary.

    Holm correction within each family. Ablations form a separate, secondary
    family. The list is hashed into the freeze file.
    """
    primary = [
        Hypothesis(
            id=f"P{i + 1}", family="primary", contrast=(Configuration.AI_PCP_V3, c),
            endpoint="cannot_miss_sensitivity", cohort="C", kind="superiority",
        )
        for i, c in enumerate(EXTERNAL)
    ]
    public = [
        Hypothesis(
            id="HB1", family="healthbench", contrast=(Configuration.AI_PCP_V3, Configuration.RAW_LLM),
            # Paired Consensus-criterion pass rate (a criterion passes when all three repeats pass); non-inferior when
            # the lower 95% bound of (ai_pcp_v3 - raw_llm) exceeds -margin. Per-cluster results are descriptive.
            endpoint="healthbench_consensus_criterion_pass_rate", cohort="HB", kind="noninferiority",
            margin=0.02,
        ),
        *[
            Hypothesis(
                id=f"HB{i + 2}", family="healthbench_secondary", contrast=(Configuration.AI_PCP_V3, c),
                # Mean per-conversation HealthBench score; paired bootstrap over conversations, two-sided p from the
                # bootstrap distribution; Holm within the family.
                endpoint="healthbench_mean_score", cohort="HB", kind="superiority",
            )
            for i, c in enumerate(EXTERNAL)
        ],
    ]
    ablation = [
        Hypothesis(
            id=f"A{i + 1}", family="ablation", contrast=(Configuration.AI_PCP, c),
            endpoint="cannot_miss_sensitivity", cohort="C", kind="superiority",
        )
        for i, c in enumerate(ABLATIONS)
    ]
    return [*primary, *public, *ablation]


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: hypothesis id -> rejected."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, bool] = {}
    still = True
    for rank, (hid, p) in enumerate(ordered):
        still = still and p <= alpha / (m - rank)
        out[hid] = still
    return out
