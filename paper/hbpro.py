"""HealthBench Professional (OpenAI, arXiv 2604.27470) on a development split (development data only).

Data: openai/healthbench-professional (525 tasks; local HF cache). The split is fixed once (seed 20261108, stratified by
use case): 175 development, 350 test. Only the development split is run here; the test split runs once at the end.

Scoring follows the paper: a GPT-5.4 grader (low reasoning) marks each rubric criterion met or not met with the
HealthBench grader template (vendored simple-evals); example score s = sum(points of met criteria) / sum(positive
points); length adjustment s - 2.94e-5 * (characters - 2000); the mean is clipped to [0, 1] and times 100; analytic
standard error. The physician responses in the dataset are graded the same way, as a check that our grader reproduces
the published physician score (43.7 overall).

Arms: frontier models with no system prompt at high reasoning (opus55, gpt61sol = GPT-6.1 Sol, current OpenAI
flagship; gpt55 and gemini31pro were run once on the development split and are not carried forward); the frozen comparator
reconstructions (mai_dxo_style, amie_style; compass_pcp.arms.baselines, Opus 5.5 backbone; the _g61 arms run the same
chains on a GPT-6.1 Sol backbone at high reasoning via compass_pcp.arms.openai_reasoning); our pipeline and its
ablations (compass_pro*, compass_pcp.arms.pro); physician.
Usage: python -m paper.hbpro {split,sample,grade,report} --out <dir> [--arms ...] [--threads 8] [--budget 150]
       python -m paper.hbpro compare --out <dir> --a <arm> --b <arm> [<arm> ...]
Writes replies.jsonl, grades.jsonl, usage.jsonl (resumable).
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

DATA = Path.home() / (".cache/huggingface/hub/datasets--openai--healthbench-professional/snapshots/"
                      "349962fd46dd02343a0d8a606491baf59154ea1a/healthbench_professional_eval.jsonl")
SEED, N_DEV = 20261108, 175
GRADER = "gpt-5.4"
# Sensitivity graders (test split, post hoc): same template and scoring; grades go to grades_<tag>.jsonl.
ALT_GRADERS = {"g61": "gpt-6.1-sol", "opus": "claude-opus-5-5"}
PRICE = {"opus55": (4.0, 20.0), "gpt55": (2.5, 15.0), "gpt61sol": (5.0, 30.0), "gemini31pro": (2.0, 12.0), "grader": (2.5, 15.0),
         "mai_dxo_style": (4.0, 20.0), "amie_style": (4.0, 20.0), "mai_dxo_style_g61": (5.0, 30.0),
         "amie_style_g61": (5.0, 30.0), "web_search": (10000.0, 0.0), "grader_g61": (5.0, 30.0), "grader_opus": (4.0, 20.0),
         "amie_style_uncapped": (4.0, 20.0), "mai_dxo_style_uncapped": (4.0, 20.0)}  # web_search: "in" counts requests (USD 10/1,000)  # USD per million tokens (list), cost report only
ARMS = ["opus55", "gpt61sol", "gpt55", "gemini31pro", "mai_dxo_style", "amie_style", "mai_dxo_style_g61", "amie_style_g61",
        "compass_pro", "compass_pro_nosearch", "compass_pro_single", "compass_pro_noreview", "compass_pro_edit", "compass_pro_full", "compass_pro_opusonly", "opus_search_edit",
        "amie_style_uncapped", "mai_dxo_style_uncapped", "opus_edit", "physician"]
PRO_FLAGS = {"compass_pro": {}, "compass_pro_nosearch": {"search": False}, "compass_pro_single": {"dual": False},
             "compass_pro_noreview": {"review": False}, "compass_pro_full": {"edit": True},
             "compass_pro_opusonly": {"edit": True, "second": "opus"},  # single-developer pipeline (post hoc, dev)
             "opus_search_edit": {"dual": False, "review": False, "final": False, "edit": True},
             "opus_edit": {"search": False, "dual": False, "review": False, "final": False, "edit": True}}  # production default without search  # searched draft + editor
_lock = threading.Lock()


def load() -> dict:
    return {r["id"]: r for r in map(json.loads, filter(None, DATA.read_text().split("\n")))}


def split(out: Path) -> None:
    p = out / "split.json"
    if p.exists():
        print(p.read_text()[:200])
        return
    rows = load()
    rng = random.Random(SEED)
    dev, test = [], []
    for uc in sorted({r["use_case"] for r in rows.values()}):
        ids = sorted(i for i, r in rows.items() if r["use_case"] == uc)
        rng.shuffle(ids)
        k = round(len(ids) * N_DEV / len(rows))
        dev += ids[:k]
        test += ids[k:]
    p.write_text(json.dumps({"seed": SEED, "development": sorted(dev), "test": sorted(test)}))
    print(json.dumps({"development": len(dev), "test": len(test)}))


def _append(path: Path, row: dict) -> None:
    with _lock, path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x] if path.exists() else []


def spent(out: Path) -> float:
    return sum(u["in"] / 1e6 * PRICE[u["who"]][0] + u["out"] / 1e6 * PRICE[u["who"]][1] for u in _rows(out / "usage.jsonl"))


def _text_of(conv: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in conv]


def answer(arm: str, conv: list[dict], out: Path) -> str:
    if arm == "opus55":
        import anthropic
        c = anthropic.Anthropic(base_url="https://api.anthropic.com", max_retries=4)
        with c.messages.stream(model="claude-opus-5-5", max_tokens=32000, thinking={"type": "adaptive"},
                               messages=_text_of(conv)) as s:
            m = s.get_final_message()
        _append(out / "usage.jsonl", {"who": arm, "in": m.usage.input_tokens, "out": m.usage.output_tokens})
        return "".join(b.text for b in m.content if b.type == "text")
    if arm in ("gpt55", "gpt61sol"):
        import openai
        r = openai.OpenAI(timeout=900, max_retries=4).chat.completions.create(
            model={"gpt55": "gpt-5.5", "gpt61sol": "gpt-6.1-sol"}[arm], reasoning_effort="high", messages=_text_of(conv))
        _append(out / "usage.jsonl", {"who": arm, "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens})
        return r.choices[0].message.content or ""
    if arm == "gemini31pro":
        from google import genai
        from google.genai import types
        c = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
                    for m in conv]
        r = c.models.generate_content(model="gemini-3.1-pro-preview", contents=contents,
                                      config=types.GenerateContentConfig(
                                          thinking_config=types.ThinkingConfig(thinking_level="HIGH")))
        u = r.usage_metadata
        _append(out / "usage.jsonl", {"who": arm, "in": u.prompt_token_count or 0,
                                      "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)})
        return r.text or ""
    if arm in ("mai_dxo_style", "amie_style", "mai_dxo_style_g61", "amie_style_g61", "amie_style_uncapped",
               "mai_dxo_style_uncapped"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_healthbench import make_llm
        from compass_pcp.arms.baselines import AmieStyle, MaiDxoStyle
        from compass_pcp.arms.openai_reasoning import OpenAIReasoningLLM
        from compass_pcp.harness.healthbench import NO_RECORD_CONTEXT
        if arm.endswith("_uncapped"):  # post hoc (test): same chains with the reply cap raised from 1,200 to 12,000 tokens
            import compass_pcp.arms.baselines as bl  # run these arms in their own process: the cap is a module global
            bl.MAX_TOKENS = 12000
        llm = OpenAIReasoningLLM("gpt-6.1-sol") if arm.endswith("_g61") else make_llm("anthropic:claude-opus-5-5")
        sysm = {"mai_dxo_style": MaiDxoStyle, "amie_style": AmieStyle}[arm.removesuffix("_g61").removesuffix("_uncapped")](llm)
        text, _ = sysm.next_message(_text_of(conv), context=NO_RECORD_CONTEXT)
        _append(out / "usage.jsonl", {"who": arm, "in": llm.usage.input_tokens, "out": llm.usage.output_tokens})
        return text
    if arm == "compass_pro_edit":  # development: step 5 applied to the stored step-4 answer of compass_pro
        from compass_pcp.arms.pro import ProPipeline
        base = {(r["id"]): r["text"] for r in _rows(out / "replies.jsonl") if r["arm"] == "compass_pro"}
        key = next((i for i, r in load().items() if r["conversation"]["messages"] == conv), None)
        if key not in base:
            raise ValueError("no stored compass_pro answer")
        p = ProPipeline()
        text = p.edit_only(_text_of(conv), base[key])
        for who, (i, o) in p.usage.items():
            _append(out / "usage.jsonl", {"who": who, "in": i, "out": o})
        return text
    if arm in PRO_FLAGS:
        from compass_pcp.arms.pro import ProPipeline
        p = ProPipeline(**PRO_FLAGS[arm])
        text, _ = p.answer(_text_of(conv))
        for who, (i, o) in p.usage.items():
            _append(out / "usage.jsonl", {"who": who, "in": i, "out": o})
        return text
    raise ValueError(arm)


def sample(out: Path, arms: list[str], threads: int, budget: float, limit: int | None = None,
           part: str = "development") -> None:
    rows, dev = load(), json.loads((out / "split.json").read_text())[part][:limit]
    done = {(r["id"], r["arm"]) for r in _rows(out / "replies.jsonl")}
    todo = [(i, a) for a in arms for i in dev if (i, a) not in done]

    def one(t):
        i, a = t
        if spent(out) > budget:
            return "budget"
        try:
            text = rows[i]["physician_response"] if a == "physician" else answer(a, rows[i]["conversation"]["messages"], out)
        except Exception as e:  # noqa: BLE001
            print("error", a, type(e).__name__, str(e)[:120], flush=True)
            return "error"
        _append(out / "replies.jsonl", {"id": i, "arm": a, "text": text})
        return "ok"

    with cf.ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, todo))
    print(json.dumps({"done": res.count("ok"), "errors": res.count("error"), "budget_stops": res.count("budget"),
                      "usd_list": round(spent(out), 2)}))


def _gfile(out: Path, grader: str | None) -> Path:
    return out / (f"grades_{grader}.jsonl" if grader else "grades.jsonl")


def grade(out: Path, threads: int, budget: float, grader: str | None = None) -> None:
    import openai
    sys.path.insert(0, os.environ["SIMPLE_EVALS_PARENT"])
    from simple_evals.healthbench_eval import GRADER_TEMPLATE, parse_json_to_dict
    rows = load()
    gfile = _gfile(out, grader)
    done = {(g["id"], g["arm"], g["k"]) for g in _rows(gfile)}
    todo = [(r, k) for r in _rows(out / "replies.jsonl") for k in range(len(rows[r["id"]]["rubric_items"]))
            if (r["id"], r["arm"], k) not in done]
    local = threading.local()

    def one(t):
        r, k = t
        if spent(out) > budget:
            return "budget"
        if not hasattr(local, "c"):
            if grader == "opus":
                import anthropic
                local.c = anthropic.Anthropic(base_url="https://api.anthropic.com", timeout=600, max_retries=4)
            else:
                local.c = openai.OpenAI(timeout=600, max_retries=4)
        conv = rows[r["id"]]["conversation"]["messages"] + [{"role": "assistant", "content": r["text"]}]
        conv_s = "\n\n".join(f"{m['role']}: {m['content']}" for m in conv)
        item = rows[r["id"]]["rubric_items"][k]
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", conv_s).replace(
            "<<rubric_item>>", f"[{item['points']}] {item['criterion_text']}")
        for _ in range(3):
            try:
                if grader == "opus":
                    m = local.c.messages.create(model=ALT_GRADERS["opus"], max_tokens=4000,
                                                messages=[{"role": "user", "content": prompt}])
                    usage, txt = (m.usage.input_tokens, m.usage.output_tokens), "".join(
                        b.text for b in m.content if b.type == "text")
                else:
                    g = local.c.chat.completions.create(model=ALT_GRADERS.get(grader, GRADER), reasoning_effort="low",
                                                        messages=[{"role": "user", "content": prompt}])
                    usage, txt = (g.usage.prompt_tokens, g.usage.completion_tokens), g.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                print("error", type(e).__name__, flush=True)
                time.sleep(5)
                continue
            _append(out / "usage.jsonl", {"who": f"grader_{grader}" if grader else "grader", "in": usage[0],
                                          "out": usage[1]})
            d = parse_json_to_dict(txt)
            if not isinstance(d.get("criteria_met"), bool):  # malformed JSON (e.g. unescaped quotes in the explanation)
                import re
                m = re.search(r'"criteria_met"\s*:\s*(true|false)', txt)
                d = {"criteria_met": m.group(1) == "true"} if m else {}
            if isinstance(d.get("criteria_met"), bool):
                _append(gfile, {"id": r["id"], "arm": r["arm"], "k": k, "met": d["criteria_met"]})
                return "ok"
        return "error"

    with cf.ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, todo))
    print(json.dumps({"graded": res.count("ok"), "errors": res.count("error"), "budget_stops": res.count("budget"),
                      "usd_list": round(spent(out), 2)}))


def report(out: Path, grader: str | None = None) -> None:
    import numpy as np
    rows = load()
    met = {(g["id"], g["arm"], g["k"]): g["met"] for g in _rows(_gfile(out, grader))}
    res = {}
    for r in _rows(out / "replies.jsonl"):
        items = rows[r["id"]]["rubric_items"]
        if not all((r["id"], r["arm"], k) in met for k in range(len(items))):
            continue
        pos = sum(it["points"] for it in items if it["points"] > 0)
        s = sum(it["points"] for k, it in enumerate(items) if met[(r["id"], r["arm"], k)]) / pos
        s_len = s - 2.94e-5 * (len(r["text"]) - 2000)
        res.setdefault(r["arm"], []).append((r["id"], s, s_len, len(r["text"])))

    def summ(v):
        a = np.array(v)
        return {"n": len(a), "score": round(float(np.clip(a.mean(), 0, 1)) * 100, 1),
                "se": round(float(a.std(ddof=1) / np.sqrt(len(a))) * 100, 1) if len(a) > 1 else None}

    table = {}
    for arm, v in sorted(res.items()):
        t = {"length_adjusted": summ([x[2] for x in v]), "raw": summ([x[1] for x in v]),
             "mean_chars": round(float(np.mean([x[3] for x in v])))}
        for key in ("use_case", "type"):
            for val in sorted({rows[x[0]][key] for x in v}):
                t[f"{key}={val}"] = summ([x[2] for x in v if rows[x[0]][key] == val])
        table[arm] = t
    common = set.intersection(*[{x[0] for x in v} for v in res.values()]) if res else set()
    table["_paired_vs_physician_on_common_ids"] = {
        arm: round(float(np.mean([x[2] for x in v if x[0] in common]) -
                         np.mean([x[2] for x in res["physician"] if x[0] in common])) * 100, 1)
        for arm, v in res.items() if "physician" in res and arm != "physician"}
    table["_usd_list"] = round(spent(out), 2)
    (out / (f"report_{grader}.json" if grader else "report.json")).write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


def compare(out: Path, a: str, bs: list[str], grader: str | None = None) -> None:
    """Paired length-adjusted difference a - b on tasks graded for both, bootstrap 95% CI (2,000 resamples of tasks)."""
    import numpy as np
    rows = load()
    met = {(g["id"], g["arm"], g["k"]): g["met"] for g in _rows(_gfile(out, grader))}
    sc = {}
    for r in _rows(out / "replies.jsonl"):
        items = rows[r["id"]]["rubric_items"]
        if not all((r["id"], r["arm"], k) in met for k in range(len(items))):
            continue
        pos = sum(it["points"] for it in items if it["points"] > 0)
        s = sum(it["points"] for k, it in enumerate(items) if met[(r["id"], r["arm"], k)]) / pos
        sc[(r["id"], r["arm"])] = s - 2.94e-5 * (len(r["text"]) - 2000)
    rng = np.random.default_rng(20261108)
    res = {}
    for b in bs:
        ids = sorted(i for (i, x) in sc if x == a and (i, b) in sc)
        d = np.array([sc[(i, a)] - sc[(i, b)] for i in ids])
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)] if len(d) else [np.nan]
        res[f"{a} - {b}"] = {"n": len(ids), "diff": round(float(d.mean()) * 100, 1) if len(d) else None,
                             "ci95": [round(float(np.percentile(boot, 2.5)) * 100, 1),
                                      round(float(np.percentile(boot, 97.5)) * 100, 1)]}
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "sample", "grade", "report", "compare"])
    ap.add_argument("--a", help="compare: the arm of interest")
    ap.add_argument("--b", nargs="+", help="compare: the reference arms")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first n development ids (smoke test)")
    ap.add_argument("--split", default="development", choices=["development", "test"],
                    help="test: the single final run of the frozen system (compass_pro_full) and comparators")
    ap.add_argument("--budget", type=float, default=150.0, help="USD list-price ceiling for this output folder")
    ap.add_argument("--grader", default=None, choices=sorted(ALT_GRADERS), help="sensitivity grader (default GPT-5.4)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"split": lambda: split(a.out), "sample": lambda: sample(a.out, a.arms, a.threads, a.budget, a.limit, a.split),
     "grade": lambda: grade(a.out, a.threads, a.budget, a.grader), "report": lambda: report(a.out, a.grader),
     "compare": lambda: compare(a.out, a.a, a.b, a.grader)}[a.cmd]()


if __name__ == "__main__":
    main()
