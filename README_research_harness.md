# compass-pcp

COMPASS (Clinically-Optimized Medical Patient Assistance & Support System), a neuro-symbolic primary care messaging system for Medicaid patients, and the pre-registered harness that validates it. In the code, COMPASS is the `ai_pcp` arm; its frozen configuration is `ai_pcp_v3`. A large language model talks with the patient, extracts findings and writes the reply; a deterministic engine makes every clinical decision from physician-signed, sourced clinical content, and defers to a clinician when it cannot decide.

## Layout

| Path | What it is |
|---|---|
| `compass_pcp/findings.py`, `ledger.py` | Findings with provenance; two-extractor reconciliation; event-sourced ledger; raise-only hazard tiers |
| `compass_pcp/engine/` | Illness-script schema (every number sourced, physician sign-off required), log-odds Bayesian updating with likelihood-ratio intervals, expected-information-gain questions, cannot-miss stopping rule, deferral reasons, one replayable decision step per turn |
| `compass_pcp/gate.py`, `lanes.py`, `monitoring.py` | Blocking medication rules with three-valued logic, lanes and autonomy rungs, monitoring contracts |
| `compass_pcp/arms/` | COMPASS (`ai_pcp_v2.py`; the frozen configuration is `ai_pcp_v3`), hazard screens (`hazard.py`), and the comparators: model alone, guardrail stack, AMIE-style chain, MAI-DxO-style panel (`baselines.py`) |
| `compass_pcp/data/` | 18 illness scripts and 148 medication rules, signed by a physician; sources in `docs/clinical/` |
| `compass_pcp/harness/` | Frozen settings, freeze and verification, HealthBench and patient-message analyses, metrics, response cache |
| `scripts/` | HealthBench runner, hazard development runs, OSF registration, public export; `scripts/custodian/` holds the test-set runners |
| `paper/` | Builds the manuscript numbers, figures and tables from the frozen test outputs |
| `docs/regulatory/` | Validation protocol (VVP-PCP-001), comparator fidelity memo, risk analysis, traceability, model card, custodian runbook, freeze files |

## Validation

The evaluation was registered on the Open Science Framework before any test item was scored. Two test sets were sealed until the freeze: the HealthBench Consensus test split and half of 2,000 real Medicaid patient messages labeled by three physicians. `docs/regulatory/freeze/` holds the freeze files (hashes only), and every test runner refuses to start if the code, settings, clinical content or test ids differ from them. The custodian procedure is in `docs/regulatory/CUSTODIAN_RUNBOOK.md`.

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e .
.venv/bin/pytest -q
export COMPASS_PCP_DATA=<data folder outside the repository>
scripts/setup_vendor.sh            # simple-evals and the Kev judge at pinned commits
source scripts/env.sh              # API keys from the environment
```

HealthBench development run, for example:

```bash
.venv/bin/python scripts/run_healthbench.py --subset consensus --split dev --repeats 1 --n 50 \
  --grader openai:gpt-4.1-2025-04-14 --configs ai_pcp_v3 raw_llm \
  --backbone anthropic:claude-opus-5-5 --deliberate anthropic:claude-opus-5-5 --second google:gemini-3.8-flash \
  --out $COMPASS_PCP_DATA/hb_dev
```

Every model and grader call is cached on disk with the code version in the cache name, so an interrupted run resumes where it stopped.

## Data

This repository holds code, clinical content and protocol documents only. Patient messages and run outputs stay on Waymark infrastructure. Real patient text is sent only to the local Kev judge and to a vendor under a business associate agreement.

## Reproducing the paper

The frozen configuration is `ai_pcp_v3` at commit `32b907d` of the development repository; `docs/regulatory/freeze/` records it, and `docs/regulatory/DEVIATIONS.json` lists the one comparator change made during the test run. This public copy has a new git history, so the freeze check, which compares commits, runs only in the development repository; the file hashes in the freeze files can be checked against this copy.

```bash
scripts/custodian/run_healthbench_test.sh $COMPASS_PCP_DATA/test_healthbench      # HealthBench Consensus test split
.venv/bin/python -m paper.build_canonical --hb-run $COMPASS_PCP_DATA/test_healthbench \
  --cohort-a <cohort_a_sealed_report.json> --freeze-dir docs/regulatory/freeze --manuscript-dir <manuscript folder>
.venv/bin/python -m paper.verify_canonical ...   # same arguments; re-derives every number and lists any pending one
.venv/bin/python -m paper.make_figures --manuscript-dir <manuscript folder>
```

Every number in the manuscript is a key in `CANONICAL_NUMBERS.json`, which `paper/build_canonical.py` writes from the test outputs; the manuscript text is rendered from a template, so no number is typed by hand.
