"""Emergency-enriched development labeling set from care-team SMS in lighthouse (development data only).

Purpose: the fresh Compass test set has 5 call-911/988 texts, too few to set a strict miss-rate threshold for the
emergency tier. Care-team SMS in lighthouse public.messages (inbound, patient scope) has about 47,000 distinct texts
longer than 20 characters before 2025-12-01, the start of the fresh test window. This script draws a set enriched for
emergencies for S. Basu to label with the same columns as the fresh set.

Steps (all output in custodian storage under --out; nothing but counts is printed):
1. pull: inbound texts before 2025-12-01, longer than 20 characters, exact duplicate texts removed; for each text, the
   first outgoing care-team message in the same thread within 48 hours (kept for a later care-team arm; never shown
   to labelers) -> pool.jsonl.
2. screen: every keyword hit (fresh_set.KEYWORDS) plus a random N_NONKW of the other texts, through GPT-5.5 (OpenAI
   BAA) with a three-level prompt (emergency / today / no) -> screen.jsonl. Resumable.
3. export: all screen-emergency texts (up to N_EMERG), N_TODAY random screen-today texts, and N_NEG random
   screen-negative texts (half keyword, half non-keyword); shuffled; ids L0001..; no stratum or screen result shown ->
   label_sheet.csv; crosswalk.jsonl holds the stratum and inclusion probability for weighting.
Usage: python3 -m paper.lighthouse_set all --out <compass-pcp-data>/lighthouse_set [--threads 12]
"""

import argparse
import concurrent.futures as cf
import csv
import json
import random
import sys
import threading
from pathlib import Path

from paper.fresh_set import KEYWORDS, SCREEN_MODEL

BEFORE, MIN_CHARS, SEED = "2025-12-01", 20, 20261108
N_NONKW, N_EMERG, N_TODAY, N_NEG = 4000, 400, 100, 100
PROMPT = """You read one text message that a patient sent to their primary care team. Answer with one word.
EMERGENCY: the message could describe something that needs 911, 988, or an emergency room now, for example chest
pain, trouble breathing, signs of stroke, suicidal thoughts or self-harm, overdose, severe bleeding, a seizure, a
person who will not wake up, or violence that is happening now.
TODAY: a clinician should look at it today, but it is not an emergency, for example a new or worse symptom, a
medication problem, a fall, a pregnancy concern, or unsafe housing.
NO: only logistics, scheduling, thanks, or no health or safety content."""


def _load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def pull(out: Path) -> None:
    sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
    from wm_conn import lighthouse, query

    df = query(lighthouse("prod"), """
        with inb as (
          select id::text id, patient_id, thread_id, created_at, message from public.messages
          where direction::text = 'INCOMING' and scope::text = 'USER' and patient_id is not null
            and created_at < :before and length(btrim(message)) > :n)
        select inb.*, r.message care_reply, r.waymarker_id care_author, r.created_at care_reply_at
        from inb left join lateral (
          select o.message, o.waymarker_id, o.created_at from public.messages o
          where o.direction::text = 'OUTGOING' and o.thread_id = inb.thread_id and o.created_at > inb.created_at
            and o.created_at < inb.created_at + interval '48 hours'
          order by o.created_at limit 1) r on true
        order by inb.id""", before=BEFORE, n=MIN_CHARS)
    df = df.drop_duplicates("message")
    with (out / "pool.jsonl").open("w") as fh:
        for r in df.itertuples():
            fh.write(json.dumps({"id": r.id, "patient_id": r.patient_id, "created_at": str(r.created_at),
                                 "message": r.message, "care_reply": r.care_reply, "care_author": r.care_author,
                                 "care_reply_at": None if r.care_reply_at is None else str(r.care_reply_at)}) + "\n")
    (out / "pool.jsonl").chmod(0o600)
    print(json.dumps({"pool": len(df), "patients": int(df.patient_id.nunique()),
                      "with_care_reply_48h": int(df.care_reply.notna().sum())}))


def screen(out: Path, threads: int) -> None:
    import openai

    pool = _load(out / "pool.jsonl")
    kw = [r["id"] for r in pool if KEYWORDS.search(r["message"])]
    rest = sorted(r["id"] for r in pool if not KEYWORDS.search(r["message"]))
    plan = out / "screen_plan.json"
    if not plan.exists():
        nonkw = random.Random(SEED).sample(rest, min(N_NONKW, len(rest)))
        plan.write_text(json.dumps({"keyword": kw, "nonkeyword_sample": nonkw, "n_nonkeyword_all": len(rest)}))
    p = json.loads(plan.read_text())
    msgs = {r["id"]: r["message"] for r in pool}
    done = {r["id"] for r in _load(out / "screen.jsonl")}
    todo = [i for i in p["keyword"] + p["nonkeyword_sample"] if i not in done]
    kwset = set(p["keyword"])
    lock, local = threading.Lock(), threading.local()

    def one(i):
        if not hasattr(local, "c"):
            local.c = openai.OpenAI(timeout=300.0, max_retries=4)
        try:
            r = local.c.chat.completions.create(model=SCREEN_MODEL, max_completion_tokens=2000, reasoning_effort="low",
                                                messages=[{"role": "system", "content": PROMPT},
                                                          {"role": "user", "content": msgs[i]}])
            t = (r.choices[0].message.content or "").strip().upper()
            lev = next((x for x in ("EMERGENCY", "TODAY", "NO") if t.startswith(x)), None)
        except Exception as err:  # noqa: BLE001 - the type only; the message never leaves this function
            print("error", type(err).__name__, flush=True)
            return 1
        with lock, (out / "screen.jsonl").open("a") as fh:
            fh.write(json.dumps({"id": i, "keyword": i in kwset, "level": lev}) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        failed = sum(ex.map(one, todo))
    rows = _load(out / "screen.jsonl")
    count = {}
    for r in rows:
        k = f"{'kw' if r['keyword'] else 'nokw'}_{r['level']}"
        count[k] = count.get(k, 0) + 1
    print(json.dumps({"screened": len(rows), "failed_this_run": failed, **count}))


def export(out: Path) -> None:
    p = json.loads((out / "screen_plan.json").read_text())
    scr = {r["id"]: r for r in _load(out / "screen.jsonl")}
    if set(p["keyword"] + p["nonkeyword_sample"]) - set(scr):
        raise SystemExit("screen is incomplete; rerun screen first")
    msgs = {r["id"]: r["message"] for r in _load(out / "pool.jsonl")}
    rng = random.Random(SEED + 1)
    frac_nonkw = len(p["nonkeyword_sample"]) / max(p["n_nonkeyword_all"], 1)  # sampling fraction of non-keyword texts

    def grp(level, kw):
        return sorted(i for i, r in scr.items() if r["level"] == level and r["keyword"] == kw)

    rows = []
    for level, n in (("EMERGENCY", N_EMERG), ("TODAY", N_TODAY)):
        for kw in (True, False):
            g = grp(level, kw)
            cap = n if level == "EMERGENCY" else n // 2  # every screen emergency up to N_EMERG; TODAY split by group
            take = g if len(g) <= cap else rng.sample(g, cap)
            base = 1.0 if kw else frac_nonkw
            rows += [{"id": i, "stratum": f"{level.lower()}_{'kw' if kw else 'nokw'}",
                      "p_inclusion": base * (len(take) / len(g))} for i in take]
    for kw in (True, False):
        g = grp("NO", kw)
        take = rng.sample(g, min(N_NEG // 2, len(g)))
        base = 1.0 if kw else frac_nonkw
        rows += [{"id": i, "stratum": f"no_{'kw' if kw else 'nokw'}", "p_inclusion": base * (len(take) / len(g))}
                 for i in take]
    rng.shuffle(rows)
    with (out / "label_sheet.csv").open("w", newline="") as fh, (out / "crosswalk.jsonl").open("w") as cw:
        w = csv.writer(fh)
        w.writerow(["sample_id", "message", "hazard (0/1)", "hazard_category", "action", "notes"])
        for k, r in enumerate(rows, 1):
            sid = f"L{k:04d}"
            w.writerow([sid, msgs[r["id"]], "", "", "", ""])
            cw.write(json.dumps({"sample_id": sid, **r}) + "\n")
    for f in ("label_sheet.csv", "crosswalk.jsonl"):
        (out / f).chmod(0o600)
    by = {}
    for r in rows:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    print(json.dumps({"label_sheet_rows": len(rows), **by}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pull", "screen", "export", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = ["pull", "screen", "export"] if a.step == "all" else [a.step]
    for s in steps:
        if s == "pull" and a.step == "all" and (a.out / "pool.jsonl").exists():
            continue
        {"pull": lambda: pull(a.out), "screen": lambda: screen(a.out, a.threads), "export": lambda: export(a.out)}[s]()


if __name__ == "__main__":
    main()
