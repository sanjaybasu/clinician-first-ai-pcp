# Clinician-first AI primary care: code for the npj Digital Medicine submission

Code, prompts and aggregate results for "A large language model pipeline with a clinician-first safety layer for
primary care, evaluated on HealthBench Professional and real Medicaid patient texts" (Basu et al., submitted).

This repository holds code only. It holds no patient messages, labels, model outputs or identifiers. The real-text
analyses run on data that stay in Waymark's custodian storage under a HIPAA business associate agreement; the
HealthBench Professional analyses run on the public benchmark.

## Where things are

| Paper element | Code |
|---|---|
| Physician-facing pipeline (two drafts, review, final writer, length editor; ablation flags) | `compass_pcp/arms/pro.py` |
| GPT-6.1 Sol client for reasoning models | `compass_pcp/arms/openai_reasoning.py` |
| AMIE-style and MAI-DxO-style re-implementations and the persona-only writer | `compass_pcp/arms/baselines.py` |
| HealthBench Professional split, sampling, grading (primary and alternate graders), reports, paired comparisons | `paper/hbpro.py` |
| Clinician-first check (any-instruction and deployable unconditional policies) | `paper/disposition_check.py` |
| Paging gate: Kev-9B fold scores, conformal risk control on the 497 enriched texts; scoring of the test texts | `paper/paging_gate_497.py`, `paper/gate_score_fresh.py`, `paper/kev_physician_folds.py` |
| Real-text test: replies, reply classifier, report (registered and deployable policies) | `paper/final_texts_staged.py`, `paper/texts_report.py` |
| Two-phase sampling weights | `paper/fresh_weights.py` |
| Supplementary analyses (weights, strata, second physician, same safety layer, registered hypotheses) | `paper/supplementary_analyses.py` |
| Post hoc extras (second run, intake instruction counts, gate test period, classifier emergency advice) | `paper/posthoc_extras_paper2.py` |
| Error analysis by hazard category and cross-tabulation | `paper/texts_errors_paper2.py` |
| Second reply classifier agreement | `paper/urgency_agreement.py` |
| History-taking intake agent and evaluation | `compass_pcp/arms/intake.py`, `paper/intake_eval.py` |
| Figures, supplementary tables, supplement assembly | `paper/make_figures_paper2.py`, `paper/make_supplement_tables_paper2.py`, `paper/build_supplement_paper2.py` |
| Number registry that re-derives every manuscript number from the result files | `results/audit/verify_canonical_paper2.py`, `results/audit/CANONICAL_NUMBERS.md` |

Prompts as run are printed in Supplementary Note 2 and live as constants in the files above.

## Aggregate results

`results/` holds the aggregate outputs behind every table and figure: HealthBench Professional reports and paired
comparisons (test, development, second run; `split.json` lists the public task ids of the development and test
splits), real-text estimates, the paging-gate calibration report, and the history-taking report. They contain counts,
rates and confidence intervals only.

## Running

Python 3.11+. `pip install -e .` installs the package. The HealthBench Professional scripts read the public dataset
from the Hugging Face cache (`openai/healthbench-professional`) and need `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`;
`SIMPLE_EVALS_PARENT` must point to a checkout of OpenAI's simple-evals for the grader template. The real-text scripts
expect the custodian data directory in `COMPASS_PCP_DATA` and cannot run without it. `pytest` runs the unit tests.

The research harness from the earlier evaluation of this program (Paper 1) is also included; its README is
`README_research_harness.md`.

## Contact

Sanjay Basu, sanjay.basu@ucsf.edu
