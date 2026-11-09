# compass-pcp agent directives

- Code only. No episode data, run outputs, transcripts or member identifiers in this repo. `data_local/` and `runs/` are git-ignored; keep them on Waymark infrastructure.
- Everything in `compass_pcp/` is deterministic and LLM-free. LLM calls live in Compass and reach this package through the protocols in `harness/runner.py` and `harness/healthbench.py`.
- Clinical numbers (priors, likelihood ratios, thresholds, gate rules, monitoring contracts) are data in `compass_pcp/data/`, each with a `source`. Do not add a clinical number in Python code.
- Branch per ticket (`ds-xxxx-…` or `en-xxxx-…`), commit message starts with the ticket id, `ruff check` and `pytest` pass before a PR, matching the `compass` repo conventions.
- Do not edit `waymark-care/compass` from here. Proposed Compass changes go in `patches/compass/` as `git format-patch` files with tests.
