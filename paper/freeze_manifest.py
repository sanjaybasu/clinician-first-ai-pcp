"""Freeze manifest for the Paper 2 system (COMPASS v22) and its comparators.

Writes the SHA-256 of every file that defines the frozen arms (code, policies, illness scripts, guideline pathways,
review-score parameters, comparator prompts, and the development harness that builds the arms), the arm settings, and
the model identifiers, so the preregistration can name the exact system. Contains no data.

Usage: python -m paper.freeze_manifest --out docs/regulatory/freeze/paper2_v22/freeze.json
       python -m paper.freeze_manifest --check docs/regulatory/freeze/paper2_v22/freeze.json   (exit 1 if any file changed)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "compass_pcp" / "data"
ARM = "ai_pcp_v22"
FILES = [
    *sorted((ROOT / "compass_pcp").rglob("*.py")),
    DATA / "policies" / "contract_v19_dev.yaml",
    DATA / "policies" / "crisis_v18_dev.yaml",
    DATA / "policies" / "first_aid_v2_dev.yaml",
    DATA / "policies" / "review_score_v4.yaml",
    *sorted((DATA / "scripts").glob("*.yaml")),
    *sorted((DATA / "scripts_dev_v18").glob("*.yaml")),
    *sorted((DATA / "pathways").glob("*.yaml")),
    *sorted((DATA / "pathways_dev_v14").glob("*.yaml")),
    ROOT / "paper" / "dev_multiturn.py",
    ROOT / "paper" / "dev_realmsg.py",
    ROOT / "paper" / "v4_dev_staged.py",
    ROOT / "paper" / "multiturn_ddxplus.py",
    ROOT / "paper" / "sealed_comparators.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    from paper import dev_multiturn as dm

    flags = {k: (sorted(v) if isinstance(v, dict) else v) for k, v in dm.ARM_FLAGS[ARM].items() if k != "first_aid"}
    return {
        "system": "COMPASS v22 (Paper 2 freeze)",
        "arm": ARM,
        "settings": {"contract": dm.CONTRACTS[ARM].name, "crisis": dm.CRISIS[ARM].name, "writer": dm.WRITER[ARM],
                     "dev_scripts": dm.DEV_SCRIPTS[ARM].name, "dev_pathways": dm.DEV_PATHWAYS[ARM].name,
                     "engine_flags": dm.ENGINE_FLAGS[ARM], "arm_flags": flags, "first_aid": "first_aid_v2_dev.yaml",
                     "screen_model": dm.SCREEN_SPEC},
        "comparators": ["raw_llm", "amie_style", "mai_dxo_style"],
        "files": {str(p.relative_to(ROOT)): sha(p) for p in FILES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--check", type=Path)
    a = ap.parse_args()
    if a.out:
        a.out.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {a.out}")
        return
    want = json.loads(a.check.read_text())["files"]
    changed = [f for f, h in want.items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    print(json.dumps({"files": len(want), "changed": changed}))
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
