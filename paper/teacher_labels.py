"""Teacher labels for the System-1 fine-tune: Opus 5.5 and Gemini 3.1 Pro (both under BAA) answer the typed questions in
paper/system1_questions.py for each text, at batch prices (development data only; never the fresh test texts).

Commands (all resumable; ids and counts are printed, never text):
  items   build <out>/items.jsonl: the 2,000 development messages (physician-labeled; teacher quality is measured on
          them), and the unlabeled Compass texts from contacts outside the fresh test set and outside the labeled
          calibration sample. Each item: {id, source, text, chart}.
  submit  queue every missing (item, teacher) request and submit the batches; stops before the USD cap.
  collect fetch finished batches.
  export  <out>/labels.jsonl: {id, source, teacher answers, consensus}; consensus is the shared answer when both
          teachers agree and a 50/50 soft target when they do not.
"""

import argparse
import json
import re
from pathlib import Path

from paper.batch_stage import Store, StagedLLM, collect, slot, submit, Pending  # noqa: F401
from paper.system1_questions import QUESTIONS, state

TEACHERS = {"opus": ("anthropic", "claude-opus-5-5"), "gemini": ("google", "gemini-3.1-pro-preview")}
SYSTEM = """You are an experienced primary care physician reviewing text messages that people send to a Medicaid care
team (community health workers, care coordinators, pharmacists, therapists, with physicians on call). For each question,
pick exactly one option. Use the chart only as context; judge the message. Be safety-minded but do not over-escalate:
choose emergency options only when the message itself supports them. Answer with one JSON object and nothing else:
{"tier": ..., "action": ..., "sender": ..., "current": true|false, "missing": true|false, "social": ...}"""


def user_prompt(chart, text) -> str:
    lines = []
    for q, spec in QUESTIONS.items():
        if spec["type"] == "choice":
            opts = "; ".join(f'"{k}" = {v}' for k, v in spec["criteria"].items())
            lines.append(f'- {q}: {spec["instructions"]} Options: {opts}.')
        else:
            lines.append(f'- {q}: {spec["instructions"]} Answer true or false.')
    return state(chart, text) + "\n\nQuestions:\n" + "\n".join(lines)


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for q, spec in QUESTIONS.items():
        v = d.get(q)
        if spec["type"] == "choice" and v in spec["criteria"]:
            out[q] = v
        elif spec["type"] == "noul" and isinstance(v, bool):
            out[q] = v
    return out if len(out) == len(QUESTIONS) else None


def _jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def items(a) -> None:
    D = a.data
    dev = json.loads(a.cases.read_text())
    chart_dev = {r["id"]: r["chart"] for r in _jl(D / "paper2_dev/chart_context/chart_context.jsonl")}
    chart_sms = {r["id"]: r["chart"] for r in _jl(D / "paper2_dev/system1/chart_sms.jsonl")}
    fresh = _jl(D / "fresh_set/pool.jsonl")
    calib = _jl(D / "calib_set/pool.jsonl")
    contact = {r["id"]: r["contact_id"] for r in fresh + calib}
    excluded = {contact[r["id"]] for r in _jl(D / "fresh_set/crosswalk.jsonl")}
    excluded |= {contact[r["id"]] for r in _jl(D / "calib_set/crosswalk.jsonl")}
    n = {"dev": 0, "sms": 0, "sms_with_chart": 0}
    with (a.out / "items.jsonl").open("w") as fh:
        for r in dev:
            t = r["prompt"] if isinstance(r["prompt"], str) else r["prompt"][-1]["content"]
            fh.write(json.dumps({"id": str(r["message_id"]), "source": "dev", "text": t,
                                 "chart": chart_dev.get(str(r["message_id"]))}) + "\n")
            n["dev"] += 1
        seen = set()
        for r in fresh + calib:
            if r["contact_id"] in excluded or r["id"] in seen:
                continue
            seen.add(r["id"])
            fh.write(json.dumps({"id": r["id"], "source": "sms", "text": r["message"], "chart": chart_sms.get(r["id"])}) + "\n")
            n["sms"] += 1
            n["sms_with_chart"] += r["id"] in chart_sms
    (a.out / "items.jsonl").chmod(0o600)
    print(json.dumps(n))


def _clients(store, realtime=False):
    from compass_pcp.arms.llm import AnthropicLLM, GeminiLLM

    return {"opus": StagedLLM(AnthropicLLM(model=TEACHERS["opus"][1], phi_allowed=True), store, realtime=realtime),
            "gemini": StagedLLM(GeminiLLM(model=TEACHERS["gemini"][1], phi_allowed=True), store, realtime=realtime)}


def _pass(a, store, realtime=False) -> dict:
    cl = _clients(store, realtime)
    got, miss = {}, 0
    rows = _jl(a.out / "items.jsonl")[: a.limit or None]
    for r in rows:
        for t, c in cl.items():
            with slot((r["id"], t)):
                try:
                    txt = c.complete(SYSTEM, [{"role": "user", "content": user_prompt(r["chart"], r["text"])}],
                                     max_tokens=300, temperature=0.0)
                    got[(r["id"], t)] = txt
                except Pending:
                    miss += 1
                else:
                    if txt == "":
                        miss += 1
    return {"got": got, "missing": miss, "rows": rows}


def usd(store) -> float:
    return round(sum(v["usd_at_batch_price"] or 0 for v in store.cost().values()), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "submit", "collect", "export", "pilot"])
    ap.add_argument("--data", type=Path, required=True, help="compass-pcp-data root")
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=150.0, help="USD at batch price for this labeling")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "items":
        return items(a)
    store = Store(a.out / "store")
    if a.cmd == "pilot":  # realtime on a few items, to measure tokens per item
        p = _pass(a, store, realtime=True)
        ok = sum(parse(v) is not None for v in p["got"].values())
        print(json.dumps({"answers": len(p["got"]), "parsed": ok, "cost": store.cost()}))
        return
    if a.cmd == "collect":
        print(json.dumps({"running": collect(store), "usd": usd(store)}))
        return
    if a.cmd == "submit":
        running = collect(store)
        p = _pass(a, store)
        # projected cost: queued requests at the mean cost per answer so far (USD 0.008 when none: the 2026-11-07 pilot measured 0.006 Opus, 0.002 Gemini)
        c = store.cost()
        per = {k: v["usd_at_batch_price"] / max(v["calls"], 1) for k, v in c.items() if v["usd_at_batch_price"]}
        proj = usd(store) + sum(per.get(r["spec"], 0.008) for r in store.queue.values())
        if proj > a.budget:
            print(json.dumps({"stopped": "budget", "projected_usd": round(proj, 2), "queued": len(store.queue)}))
            return
        jobs = submit(store) if store.queue else []
        print(json.dumps({"running_before": running, "submitted_jobs": len(jobs), "missing": p["missing"],
                          "projected_usd": round(proj, 2)}))
        return
    p = _pass(a, store)
    n = {"items": 0, "both": 0, "agree_tier": 0, "agree_action": 0, "parse_fail": 0}
    with (a.out / "labels.jsonl").open("w") as fh:
        for r in p["rows"]:
            ans = {t: parse(p["got"].get((r["id"], t), "")) for t in TEACHERS}
            n["parse_fail"] += sum(v is None for v in ans.values())
            if not all(ans.values()):
                continue
            cons = {}
            for q in QUESTIONS:
                x, y = ans["opus"][q], ans["gemini"][q]
                cons[q] = {str(x): 1.0} if x == y else {str(x): 0.5, str(y): 0.5}
            fh.write(json.dumps({"id": r["id"], "source": r["source"], "teachers": ans, "consensus": cons}) + "\n")
            n["items"] += 1
            n["both"] += 1
            n["agree_tier"] += ans["opus"]["tier"] == ans["gemini"]["tier"]
            n["agree_action"] += ans["opus"]["action"] == ans["gemini"]["action"]
    print(json.dumps({**n, "usd": usd(store)}))


if __name__ == "__main__":
    main()
