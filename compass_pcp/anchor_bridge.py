"""Bridge to the full ANCHOR rule corpus (waymark-care/anchor, ``light/``).

Light ANCHOR is a negation-aware regex evaluator over about 3,128 rules in four
harm categories (omission, contradiction-style contraindication, under-triage,
anchoring). In this architecture it is not the small autonomous-lane gate
(design doc 2.4: keep about 300 rules there). It serves two roles:

1. The concordance check over the drafted plan in physician-signed lanes:
   each fired rule becomes a blocking ``VerifierFinding`` routed to an owner.
2. The red-flag floor over patient text (under-triage rules only), in addition
   to Compass's BE-FAST rules.

Set ``ANCHOR_REPO`` to a checkout of waymark-care/anchor. The commit is recorded
with every result so the rule corpus is pinned like any other SOUP item.
"""

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from compass_pcp.gate import ROUTING, FindingKind, VerifierFinding

CATEGORY_TO_KIND = {
    "omission": FindingKind.CANNOT_MISS_OMISSION,
    "anchoring": FindingKind.CANNOT_MISS_OMISSION,
    "undertriage": FindingKind.CANNOT_MISS_OMISSION,
    "contraindication": FindingKind.COMMISSION,
}
PATIENT_TEXT_CATEGORIES = frozenset({"undertriage"})


def anchor_repo() -> Path:
    path = Path(os.environ.get("ANCHOR_REPO", Path.home() / "waymark-local/packaging/anchor"))
    if not (path / "light" / "engine.py").exists():
        raise FileNotFoundError(f"Light ANCHOR not found under {path}; set ANCHOR_REPO")
    return path


@lru_cache(maxsize=1)
def light_anchor():
    repo = str(anchor_repo())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from light import LightAnchor  # type: ignore[import-not-found]

    return LightAnchor()


@lru_cache(maxsize=1)
def anchor_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(anchor_repo()), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "-C", str(anchor_repo()), "status", "--porcelain", "light", "model/lnn_rules"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def plan_findings(plan_text: str, *, min_severity: int = 7) -> list[VerifierFinding]:
    """Concordance check over a drafted plan; most severe first."""
    out = []
    for v in light_anchor().evaluate(plan_text, min_severity=min_severity):
        kind = CATEGORY_TO_KIND.get(v.harm_category, FindingKind.CANNOT_MISS_OMISSION)
        out.append(
            VerifierFinding(
                rule_id=f"anchor:{v.rule}",
                kind=kind,
                owner=ROUTING[kind],
                message=v.description,
                source=f"ANCHOR {v.harm_category} sev {v.severity}, grade {v.guideline_grade} ({v.guideline_year})",
            )
        )
    return out


def red_flag_floor(patient_text: str, *, min_severity: int = 7) -> list[str]:
    """Under-triage rules that fire on patient-reported text: ids of the rules."""
    return [
        v.rule
        for v in light_anchor().evaluate(patient_text, categories=set(PATIENT_TEXT_CATEGORIES), min_severity=min_severity)
    ]
