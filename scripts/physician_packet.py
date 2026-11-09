"""Physician scoring of a HealthBench test sample (pre-specified secondary analysis; VVP-PCP-001 Section 8.6).

make:   python -m scripts.physician_packet make --run <HB test output dir> --out <custodian dir>
        Picks 100 conversations (stratified by theme, fixed seed), takes the first-repeat response of ai_pcp_v3 and of
        raw_llm, shuffles them into blinded Response A and Response B, and writes one rating sheet (CSV, one row per
        response and rubric criterion) plus the unblinding key. The sheet holds no grader output.
score:  python -m scripts.physician_packet score --run <dir> --key <key.json> --ratings <rater1.csv> <rater2.csv> ...
        Unblinds each physician's marks, scores each arm with the HealthBench formula, and reports the paired
        difference, agreement between physicians, and agreement of each LLM grader with the physicians.
Physicians fill the 'met' column with 1 or 0 for every row, and 'preferred' (A, B or tie) on the first row of each item.
No model fills any rating.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.adjudication import gwet_ac1
from compass_pcp.harness.metrics import paired_bootstrap_diff

SEED, N = 20261028, 100
ARMS = ("ai_pcp_v3", "raw_llm")


def _examples(path: Path) -> dict[str, dict]:
    """First repeat of each prompt id."""
    out: dict[str, dict] = {}
    for e in json.loads(path.read_text())["example_level"]:
        out.setdefault(e["prompt_id"], e)
    return out


def _themes(run: Path) -> dict[str, str]:
    """prompt_id -> HealthBench theme, from the split file the run wrote (consensus_test.jsonl)."""
    out = {}
    for path in run.glob("consensus_*.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["prompt_id"]] = next((t.split(":", 1)[1] for t in r["example_tags"] if t.startswith("theme:")), "unknown")
    return out


def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else str(c)


def make(run: Path, out: Path) -> None:
    arms = {a: _examples(run / f"{a}.json") for a in ARMS}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    themes = _themes(run)
    by_theme: dict[str, list[str]] = defaultdict(list)
    for pid in ids:
        by_theme[themes.get(pid, "unknown")].append(pid)
    rng = random.Random(SEED)
    picked: list[str] = []
    for theme in sorted(by_theme):  # proportional allocation, largest remainders
        rng.shuffle(by_theme[theme])
    quotas = {t: N * len(v) / len(ids) for t, v in by_theme.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    for t in sorted(quotas, key=lambda t: quotas[t] - alloc[t], reverse=True)[: N - sum(alloc.values())]:
        alloc[t] += 1
    for t in sorted(by_theme):
        picked += by_theme[t][: alloc[t]]
    out.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for i, pid in enumerate(picked, start=1):
        order = list(ARMS)
        rng.shuffle(order)
        key[pid] = {"A": order[0], "B": order[1], "item": i}
        convo = "\n\n".join(f'{m["role"].upper()}: {m["content"]}' for m in arms[ARMS[0]][pid]["prompt"])
        for letter, arm in zip("AB", order):
            e = arms[arm][pid]
            for j, item in enumerate(e["rubric_items"]):
                rows.append({"item": i, "prompt_id": pid, "response": letter, "criterion_index": j,
                             "conversation": convo if (letter == "A" and j == 0) else "",
                             "response_text": _text(e["completion"]) if j == 0 else "",
                             "criterion": item["criterion"], "points": item["points"], "met": "",
                             "preferred": "" if (letter == "A" and j == 0) else "n/a"})
    with (out / "physician_rating_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "physician_key.json").write_text(json.dumps({"seed": SEED, "arms": ARMS, "items": key}, indent=2))
    print(f"{len(picked)} items, {len(rows)} rating rows -> {out}")


def _score(marks: dict[int, bool], items: list[dict]) -> float:
    """HealthBench example score: points of met criteria over total positive points, clipped to [0, 1]."""
    total = sum(it["points"] for it in items if it["points"] > 0)
    got = sum(it["points"] for j, it in enumerate(items) if marks.get(j))
    return max(0.0, min(1.0, got / total)) if total else 0.0


def score(run: Path, key_path: Path, ratings: list[Path]) -> dict:
    key = json.loads(key_path.read_text())["items"]
    arms = {a: _examples(run / f"{a}.json") for a in ARMS}
    graders = {"grader1": {a: _examples(run / f"{a}.json") for a in ARMS}}
    if all((run / f"{a}__grader2.json").exists() for a in ARMS):
        graders["grader2"] = {a: _examples(run / f"{a}__grader2.json") for a in ARMS}
    marks: list[dict] = []  # per rater: (pid, arm, criterion) -> bool
    for path in ratings:
        m = {}
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r["met"].strip() not in ("0", "1"):
                    raise SystemExit(f"{path.name}: item {r['item']} criterion {r['criterion_index']} is not rated 0 or 1")
                m[(r["prompt_id"], key[r["prompt_id"]][r["response"]], int(r["criterion_index"]))] = r["met"].strip() == "1"
        marks.append(m)
    pids = sorted(key)
    per_rater = []
    for m in marks:
        per_rater.append({a: [_score({j: m[(p, a, j)] for j in range(len(arms[a][p]["rubric_items"]))}, arms[a][p]["rubric_items"])
                              for p in pids] for a in ARMS})
    mean = {a: [sum(r[a][i] for r in per_rater) / len(per_rater) for i in range(len(pids))] for a in ARMS}
    keys = [(p, a, j) for p in pids for a in ARMS for j in range(len(arms[a][p]["rubric_items"]))]
    out = {"n_items": len(pids), "n_raters": len(marks),
           "physician_score": {a: sum(mean[a]) / len(pids) for a in ARMS},
           "paired_diff_ai_pcp_minus_raw": paired_bootstrap_diff(mean["ai_pcp_v3"], mean["raw_llm"]),
           "agreement_between_physicians_ac1": gwet_ac1([[str(int(m[k])) for m in marks] for k in keys]) if len(marks) > 1 else None}
    for g, ex in graders.items():
        gm = {(p, a, j): bool(ex[a][p]["rubric_items"][j]["criteria_met"]) for (p, a, j) in keys}
        majority = {k: sum(m[k] for m in marks) * 2 > len(marks) for k in keys}
        out[f"{g}_vs_physician_majority_ac1"] = gwet_ac1([[str(int(majority[k])), str(int(gm[k]))] for k in keys])
        out[f"{g}_vs_physician_majority_raw_agreement"] = sum(majority[k] == gm[k] for k in keys) / len(keys)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--ratings", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.run, args.out)
    else:
        print(json.dumps(score(args.run, args.key, args.ratings), indent=2))


if __name__ == "__main__":
    main()
