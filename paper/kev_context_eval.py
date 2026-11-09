"""Kev hazard tiers with the chart in the state (development data only).

Same five-tier question as paper/score_ensemble.py, but the state holds the chart summary from paper/chart_context.py
followed by the message. Writes <out>/kev_chart.jsonl ({id, kev_tier}); ids and numbers only. Resumable.
Usage: python -m paper.kev_context_eval --cases <json> --chart <chart_context.jsonl> --out <dir> [--url ...] [--tag 4b]
"""

import argparse
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from compass_pcp.arms.hazard import KEV_INSTRUCTIONS, KEV_TIERS
from compass_pcp.arms.llm import KevJudge


def state(chart: str, text: str) -> str:
    return ("What the care team's chart showed before this message:\n" + chart
            + "\n\nThe text message the person sent to the care team:\n" + text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--tag", default="4b")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nochart", action="store_true", help="message only (same question), for the paired comparison")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".config/compass-pcp/kev_serve_key")
    a = ap.parse_args()
    charts = {json.loads(x)["id"]: json.loads(x)["chart"] for x in a.chart.read_text().splitlines()}
    rows = json.loads(a.cases.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"kev_{'msg' if a.nochart else 'chart'}_{a.tag}.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
    todo = [r for r in rows if str(r["message_id"]) not in done and str(r["message_id"]) in charts]
    lock, local = threading.Lock(), threading.local()

    def remote_choice(st, instructions, options):
        import urllib.request

        body = json.dumps({"model": "kev-latest", "state": st,
                           "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": options}}})
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", data=body.encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + a.key_file.read_text().strip()})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as fh:
                    return json.loads(fh.read())["answers"]["q"]["probabilities"]
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise

    def one(r):
        if not hasattr(local, "k"):
            local.k = None if a.url else KevJudge()
        text = r["prompt"] if isinstance(r["prompt"], str) else r["prompt"][-1]["content"]
        try:
            st = text if a.nochart else state(charts[str(r["message_id"])], text)
            probs = remote_choice(st, KEV_INSTRUCTIONS, KEV_TIERS) if a.url else local.k.choice(st, KEV_INSTRUCTIONS, KEV_TIERS)
            rec = {"id": str(r["message_id"]), "kev_tier": probs}
        except Exception as err:  # noqa: BLE001 - type only, never text
            print("error", type(err).__name__, flush=True)
            return 1
        with lock, path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    with cf.ThreadPoolExecutor(a.threads) as ex:
        failed = sum(ex.map(one, todo))
    print(json.dumps({"scored": len(done) + len(todo) - failed, "failed": failed}))


if __name__ == "__main__":
    main()
