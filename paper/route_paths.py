"""Post hoc estimate of the reply path for each HealthBench test conversation (not logged during the frozen run).

Re-runs only the frozen router of ai_pcp_v3 on each full conversation and classifies the path:
clinician (health professional), engine (patient or caregiver with at least one defined complaint), or general
(no defined complaint). The router samples at the API default, so this estimates, and does not reproduce, the paths
of the scored replies. Writes per-conversation paths (prompt ids only) and a summary.

Usage: python -m paper.route_paths --run <HB test dir> --out <paths.jsonl> [--threads 8]
"""

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import build_arm  # noqa: E402

from compass_pcp.harness.configurations import Configuration  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    convs = [json.loads(line) for line in (a.run / "consensus_test.jsonl").open()]
    done = {json.loads(line)["prompt_id"] for line in a.out.open()} if a.out.exists() else set()
    todo = [c for c in convs if c["prompt_id"] not in done]
    arm = build_arm(Configuration.AI_PCP, "anthropic:claude-opus-5-5", "google:gemini-3.8-flash", require_review=True)

    def one(c):
        cats, _, user_type, _ = arm._route_full(c["prompt"])
        path = "clinician" if user_type == "health_professional" else ("engine" if cats else "general")
        return {"prompt_id": c["prompt_id"], "user_type": user_type, "n_complaints": len(cats), "path": path}

    with a.out.open("a") as fh, cf.ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(i, "of", len(todo), flush=True)
    rows = [json.loads(line) for line in a.out.open()]
    summary = {p: sum(r["path"] == p for r in rows) for p in ("engine", "clinician", "general")}
    print(json.dumps({"n": len(rows), **summary}))


if __name__ == "__main__":
    main()
