"""Build the fresh real-message test set for Paper 2 (design/paper2_preregistration_draft.md, section 3.1).

Two-phase sample of incoming Compass texts, labeled by S. Basu before any system output exists.

Eligible: incoming messages in messenger public.conversations from 2025-12-01 to 2026-10-31 (after the development
notes, which end 2025-11-24), longer than 20 characters after trimming (shorter texts are one-word replies such as "Yes"
or "STOP"). One message per contact, chosen at random (seed), so contacts are independent.

1. pull: eligible messages -> <out>/pool.jsonl (custodian storage, outside the repository).
2. sample: one per contact; random stratum of N_RANDOM; the rest form the enrichment pool.
3. screen: the enrichment pool through a fixed keyword list and one model that no tested arm uses (GPT-5.5, OpenAI BAA)
   -> <out>/screen.jsonl (ids and yes/no only). Resumable.
4. enrich: N_ENRICHED drawn at random from screen-positive messages of the enrichment pool; inclusion probabilities
   recorded for weighting -> <out>/sample.jsonl.
5. export: blinded labeling sheet (shuffled, synthetic ids, no stratum, no system output) -> <out>/label_sheet.csv and
   a separate crosswalk <out>/crosswalk.jsonl.

Message text goes only to OpenAI (BAA). Nothing but counts is printed. Usage:
  python -m paper.fresh_set all --out <compass-pcp-data>/fresh_set [--threads 8]
"""

import argparse
import concurrent.futures as cf
import csv
import json
import random
import re
import sys
import threading
from pathlib import Path

WINDOW = ("2025-12-01", "2026-11-01")
MIN_CHARS, N_RANDOM, N_ENRICHED, SEED = 20, 600, 400, 20261106
SCREEN_MODEL = "gpt-5.5"
# Fixed before sampling; written from the label taxonomy (cardiac, breathing, neurologic, bleeding, suicidality,
# substance use, falls, medication safety, maternity, housing and violence), not from any tested system's lexicon.
KEYWORDS = re.compile(r"\b(" + r"chest|heart|breath\w*|can'?t breathe|wheez\w*|faint\w*|pass(ed)? out|dizz\w*|seizure\w*|stroke|numb\w*|slurr\w*|bleed\w*|blood|vomit\w*|throw\w* up|suicid\w*|kill (my|him|her)self|want to die|end (it|my life)|hurt (my|him|her)self|overdos\w*|pills|drunk|alcohol|drinking|withdraw\w*|relapse\w*|fell|fall(en)?|fallen|insulin|sugar|glucose|ran out|out of (my )?(meds|medication|medicine)|pregnan\w*|contraction\w*|abus\w*|hit me|beat\w*|unsafe|homeless|evict\w*|no heat|emergency|\ber\b|911|ambulance|hospital|severe|worst|pain|swell\w*|fever|infect\w*|confus\w*" + r")\b", re.IGNORECASE)
SCREEN_PROMPT = """You read one text message that a patient sent to a primary care team. Answer yes if the message could
describe a health or safety concern that a clinician should look at (a symptom, a medication problem, a fall, a mental
health crisis, substance use, a pregnancy concern, violence or unsafe housing, or anything that could be urgent).
Answer no if it is only logistics, scheduling, thanks, or a reply with no health content. Reply with one word: yes or
no."""


def pull(out: Path) -> None:
    sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
    from wm_conn import messenger, query

    df = query(messenger("prod"), """select id::text id, contact_id::text contact_id, inserted_at, message
        from public.conversations where direction = 'incoming' and inserted_at >= :lo and inserted_at < :hi
        and length(trim(message)) > :n and contact_id is not null order by id""", lo=WINDOW[0], hi=WINDOW[1], n=MIN_CHARS)
    with (out / "pool.jsonl").open("w") as fh:
        for r in df.itertuples():
            fh.write(json.dumps({"id": r.id, "contact_id": r.contact_id, "inserted_at": str(r.inserted_at),
                                 "message": r.message}) + "\n")
    print(json.dumps({"pool_messages": len(df), "contacts": df.contact_id.nunique()}))


def _load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def sample(out: Path) -> None:
    rng = random.Random(SEED)
    by_contact: dict[str, list[dict]] = {}
    for r in _load(out / "pool.jsonl"):
        by_contact.setdefault(r["contact_id"], []).append(r)
    one = [rng.choice(sorted(v, key=lambda r: r["id"])) for _, v in sorted(by_contact.items())]
    rng.shuffle(one)
    rand = one[:N_RANDOM]
    rest = one[N_RANDOM:]
    (out / "phase1.json").write_text(json.dumps({"n_contacts": len(one), "random_ids": [r["id"] for r in rand],
                                                 "enrichment_pool_ids": [r["id"] for r in rest]}))
    print(json.dumps({"contacts": len(one), "random": len(rand), "enrichment_pool": len(rest)}))


def screen(out: Path, threads: int, which: str = "enrichment_pool_ids", fname: str = "screen.jsonl") -> None:
    """Screen one phase-1 group. The random stratum is screened too (screen_random.jsonl), so the weighted analysis can
    split it into screen-positive and screen-negative messages; the labeler never sees screen results."""
    import openai

    phase = json.loads((out / "phase1.json").read_text())
    msgs = {r["id"]: r["message"] for r in _load(out / "pool.jsonl")}
    done = {r["id"] for r in _load(out / fname)}
    todo = [i for i in phase[which] if i not in done]
    lock, local = threading.Lock(), threading.local()

    def one(i):
        if not hasattr(local, "c"):
            local.c = openai.OpenAI(timeout=300.0, max_retries=4)
        kw = bool(KEYWORDS.search(msgs[i]))
        try:
            r = local.c.chat.completions.create(model=SCREEN_MODEL, max_completion_tokens=2000, messages=[
                {"role": "system", "content": SCREEN_PROMPT}, {"role": "user", "content": msgs[i]}])
            txt = (r.choices[0].message.content or "").strip().lower()
            llm = txt.startswith("yes") if txt[:3] in ("yes", "no", "no.") or txt.startswith(("yes", "no")) else None
        except Exception as err:  # noqa: BLE001 - the type only; the message never leaves this function
            print("error", type(err).__name__, flush=True)
            return 1
        with lock, (out / fname).open("a") as fh:
            fh.write(json.dumps({"id": i, "keyword": kw, "llm": llm}) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        failed = sum(ex.map(one, todo))
    rows = _load(out / fname)
    print(json.dumps({"file": fname, "screened": len(rows), "failed_this_run": failed, "keyword": sum(r["keyword"] for r in rows),
                      "llm_yes": sum(bool(r["llm"]) for r in rows), "llm_unparsed": sum(r["llm"] is None for r in rows),
                      "positive": sum(r["keyword"] or bool(r["llm"]) for r in rows)}))


def enrich(out: Path) -> None:
    phase = json.loads((out / "phase1.json").read_text())
    scr = {r["id"]: r for r in _load(out / "screen.jsonl")}
    if set(phase["enrichment_pool_ids"]) - set(scr):
        raise SystemExit("screen is incomplete; rerun screen first")
    pos = sorted(i for i in phase["enrichment_pool_ids"] if scr[i]["keyword"] or scr[i]["llm"])
    rng = random.Random(SEED + 1)
    take = pos if len(pos) <= N_ENRICHED else rng.sample(pos, N_ENRICHED)
    n_all, n_rand = phase["n_contacts"], len(phase["random_ids"])
    p_rand = n_rand / n_all
    p_enr = len(take) / len(pos) if pos else 0.0
    rows = [{"id": i, "stratum": "random", "p_inclusion": p_rand} for i in phase["random_ids"]]
    rows += [{"id": i, "stratum": "enriched", "p_inclusion": p_enr, "screen_positive": True} for i in take]
    with (out / "sample.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "design.json").write_text(json.dumps({
        "window": WINDOW, "min_chars": MIN_CHARS, "seed": SEED, "contacts": n_all, "random": n_rand,
        "enrichment_pool": len(phase["enrichment_pool_ids"]), "enrichment_pool_screen_positive": len(pos),
        "enriched": len(take), "p_random": p_rand, "p_enriched_given_screen_positive": p_enr,
        "screen_model": SCREEN_MODEL, "keyword_pattern_sha": __import__("hashlib").sha256(
            KEYWORDS.pattern.encode()).hexdigest()}, indent=1))
    print(json.dumps({"random": n_rand, "enrichment_pool_positive": len(pos), "enriched": len(take)}))


def export(out: Path) -> None:
    msgs = {r["id"]: r for r in _load(out / "pool.jsonl")}
    rows = _load(out / "sample.jsonl")
    rng = random.Random(SEED + 2)
    rng.shuffle(rows)
    with (out / "label_sheet.csv").open("w", newline="") as fh, (out / "crosswalk.jsonl").open("w") as cw:
        w = csv.writer(fh)
        w.writerow(["sample_id", "message", "hazard (0/1)", "hazard_category", "action", "notes"])
        for k, r in enumerate(rows, 1):
            sid = f"F{k:04d}"
            w.writerow([sid, msgs[r["id"]]["message"], "", "", "", ""])
            cw.write(json.dumps({"sample_id": sid, **r}) + "\n")
    print(json.dumps({"label_sheet_rows": len(rows)}))


def export_second(out: Path, n: int = 200) -> None:
    """Agreement subsample for the second physician: n rows of the label sheet chosen at random, in a new random order,
    with the same sample ids and empty label columns. The second physician never sees the first physician's labels."""
    with (out / "label_sheet.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    rng = random.Random(SEED + 3)
    pick = rng.sample(body, n)
    with (out / "label_sheet_second_physician.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in pick:
            w.writerow([r[0], r[1], "", "", "", ""])
    print(json.dumps({"second_physician_rows": len(pick)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pull", "sample", "screen", "screen_random", "enrich", "export",
                                        "export_second", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = ["pull", "sample", "screen", "screen_random", "enrich", "export"] if a.step == "all" else [a.step]
    for s in steps:
        if s == "pull" and a.step == "all" and (a.out / "pool.jsonl").exists():
            continue  # resume: the pool is fixed once pulled
        if s == "sample" and a.step == "all" and (a.out / "phase1.json").exists():
            continue
        {"pull": lambda: pull(a.out), "sample": lambda: sample(a.out), "screen": lambda: screen(a.out, a.threads),
         "screen_random": lambda: screen(a.out, a.threads, "random_ids", "screen_random.jsonl"),
         "enrich": lambda: enrich(a.out), "export": lambda: export(a.out), "export_second": lambda: export_second(a.out)}[s]()


if __name__ == "__main__":
    main()
