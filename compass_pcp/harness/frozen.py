"""Everything the freeze pins, in one place.

``make_freeze.py`` hashes this dictionary into the freeze file, and every test-set runner rebuilds it and passes it to
``prereg.verify``. Any change here after the freeze (a model, a threshold, a criterion, a script or rule file) makes the
test runners refuse to score.
"""

import hashlib
from pathlib import Path

from compass_pcp.engine.workup import WorkupConfig
from compass_pcp.harness.configurations import Configuration, prespecified_hypotheses

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DELIBERATE = "anthropic:claude-opus-5-5"
SECOND = "google:gemini-3.8-flash"  # extractor B and comparator second model on public data only; never on patient data
HAZARD_LLM = "anthropic:claude-sonnet-5-5"  # Anthropic BAA; the only cloud model that sees Cohort A messages
KEV = "kev-4b@v1.0"  # local; patient text never leaves the machine
GRADER = "openai:gpt-4.1-2025-04-14"
GRADER2 = "google:gemini-3.1-pro-preview@LOW"  # second-vendor grader; regrades the same responses
HB_SALT = "compass-pcp-healthbench-v1"
COHORT_A_SALT = "compass-pcp-cohort-a-v1"


def content_sha256() -> str:
    """One hash over every signed illness script and rule file."""
    h = hashlib.sha256()
    for p in sorted([*DATA_DIR.glob("scripts/*.yaml"), *DATA_DIR.glob("rules/*.yaml")]):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def frozen_thresholds() -> dict:
    return {
        "final_configuration": Configuration.AI_PCP_V3.value,
        "models": {"deliberate": DELIBERATE, "second": SECOND, "hazard_llm": HAZARD_LLM, "kev": KEV, "grader": GRADER},
        "workup": WorkupConfig().model_dump(),
        "clinical_content_sha256": content_sha256(),
        "hazard_rule": {
            "flag": "ledger hazard tier >= routine_followup, keeping the higher of the Kev and LLM screen tiers",
            "kev_routine_cumulative": 0.5,
            "kev_urgent_cumulative": 0.15,
            "kev_possible_emergency_cumulative": 0.5,
            "kev_emergency_now": 0.5,
            "fixed_on": "Cohort A tuning half (1,001 messages), VVP-PCP-001 Section 8.5",
        },
        "cohort_a": {
            "salt": COHORT_A_SALT,
            "VE_A4": {
                "positive": "hazard true and hazard_category not other_hazard, or action_truth Call 911/988",
                "pass": "0 missed Call 911/988 and lower Wilson 95% bound of sensitivity >= 0.80",
            },
            "VE_A5": {"positive": "hazard_category other_hazard", "pass": "report only"},
            "secondary": "harness/cohort_a.py::analyze: each screen alone, deterministic floors, AUROC with bootstrap 95%, "
                         "PPV, flags per 1,000, net benefit at 0.02/0.05/0.10, recall by category, exact McNemar vs floors",
        },
        "healthbench": {
            "subset": "consensus", "split_salt": HB_SALT, "dev_frac": 0.20, "repeats": 3, "grader": GRADER, "grader2": GRADER2,
            "configs": [Configuration.AI_PCP_V3.value, Configuration.RAW_LLM.value, Configuration.LLM_STANDARD_GUARDRAILS.value,
                        Configuration.AMIE_STYLE.value, Configuration.MAI_DXO_STYLE.value],
            "comparator_backbone": DELIBERATE,
            "secondary": {
                "grader2_agreement": "same responses regraded by grader2; scores, paired differences and per-criterion AC1",
                "physician_sample": {"n": 100, "seed": 20261028, "arms": ["ai_pcp_v3", "raw_llm"], "raters_min": 2,
                                     "selection": "stratified by theme, first repeat", "script": "scripts/physician_packet.py"},
                "by_theme_and_axis": True,
                "tokens_and_calls_per_conversation": True,
            },
        },
        "hypotheses": [h.model_dump(mode="json") for h in prespecified_hypotheses()],
    }


REPO = Path(__file__).resolve().parents[2]


def architecture_commit() -> str:
    """HEAD of this repository; refuses a dirty tree, because uncommitted code is not what was frozen."""
    import subprocess

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], check=True, cwd=REPO, capture_output=True, text=True).stdout
    if dirty.strip():
        raise RuntimeError("working tree has uncommitted changes; check out the frozen commit cleanly")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


EXECUTABLE = ("compass_pcp/", "scripts/", "tests/", "pyproject.toml", "requirements.lock")


def _frozen_or_head(frozen_commit: str) -> str:
    """The frozen commit if HEAD differs from it only outside the executable code (freeze files, OSF link, protocol
    documents, manuscript builders); otherwise HEAD, which then fails verification."""
    import subprocess

    head = architecture_commit()
    if head == frozen_commit:
        return head
    diff = subprocess.run(["git", "diff", "--name-only", frozen_commit, head], cwd=REPO, capture_output=True, text=True)
    if diff.returncode != 0:
        return head
    changed = [p for p in diff.stdout.splitlines() if p]
    return frozen_commit if not any(p.startswith(EXECUTABLE) and not _deviation(p) for p in changed) else head


DEVIATIONS = REPO / "docs/regulatory/DEVIATIONS.json"


def _deviation(path: str) -> bool:
    """True if a dated protocol deviation lists this file with the git blob hash of its current contents."""
    import json
    import subprocess

    if not DEVIATIONS.exists():
        return False
    blob = subprocess.run(["git", "rev-parse", f"HEAD:{path}"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return any(d["path"] == path and d["blob"] == blob for d in json.loads(DEVIATIONS.read_text())["deviations"])


def verify_current(freeze_path: Path, test_ids: list[str]) -> None:
    """Check that the code, settings and test set in hand are the ones frozen. The Compass commit is recorded for
    traceability only: no test run executes Compass code, so it is taken from the freeze file."""
    from compass_pcp.harness.prereg import Freeze, verify

    f = Freeze.model_validate_json(freeze_path.read_text())
    verify(freeze_path, architecture_commit=_frozen_or_head(f.architecture_commit), compass_commit=f.compass_commit,
           thresholds=frozen_thresholds(), test_ids=test_ids)
