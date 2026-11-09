"""Paper 2 development: real-message SMS intake and note, COMPASS structured intake vs AMIE-style and MAI-DxO-style.

Design (design/paper2_refocus_plan.md, "Real messages" 3a; development data only):
  Cases. Development Compass SMS texts (system1/teacher/items.jsonl source "sms": contacts outside the fresh test set and
    the calibration sample) whose send-time chart lists claims diagnoses and filled medicines; one text per patient;
    shuffled with random.Random(20261108); Sonnet 5.5 screen (text only) keeps texts that describe a symptom or clinical
    problem. A hidden case card (GPT-6.1 Sol) extends each text into an HPI (8 elements), family history, social
    history and social needs, and marks which chart entries are conditions and medicines; it is frozen before any arm
    runs. The first 60 screened texts whose card keeps at least one condition and one medicine are the sample.
  Simulated patient. GPT-6.1 Sol (low reasoning): the real text, the chart conditions and medicines, and the card;
    answers only what is asked, in short SMS style.
  Arms (Opus 5.5, model-default thinking). amie_style and mai_dxo_style: the frozen chains of compass_pcp/arms/baselines.py
    with the chart and one TASK line appended to every system prompt. intake: compass_pcp/arms/intake.py with the same
    chart and TASK line. Up to 8 agent messages; the episode stops at the first agent message without a "?" (the
    reply-rate definition of a question). The same note prompt (NOTE) then writes a SOAP note from chart + transcript.
  Scoring. Judge GPT-6.1 Sol (low), blinded to arm (patient texts and note only, shuffled order). Chart-condition and
    chart-medicine recall (over card-kept entries) and precision (note list items matching any chart entry); HPI
    coverage out of 8 and FH/SH/social-need coverage against the card; turns; drop-off-weighted completeness;
    AI-originated ER/911 instructions per patient-facing message (paper/disposition_check.check; the urgent-page route
    text is the only allowed span, and only our arm sends it).
  Drop-off weighting. After agent message k with a question the patient replies with the measured rate r_k
    (compass_reply_rates_2026-11-03.json, "question", k>=8 pooled). Completeness after j replies = share of card items
    (8 HPI + FH/SH/social) that the note documents and that the patient had stated by reply j (judge "elicited"). The
    expectation over the stopping point approximates a note written from the truncated transcript by the final note
    restricted to facts already stated; it does not re-write notes per truncation.
  Judge check. 10 constructed notes (card and chart items with known deletions, an unsupported condition and medicine in
    5, patient texts with known elicitation turns) scored by the same judge prompt.

PHI: message, chart, card, transcripts and notes go only to Anthropic and OpenAI (BAAs) and to files under
compass-pcp-data (chmod 600). This script prints aggregates only.

Usage (after `source scripts/env.sh`):
  python -m paper.intake_eval sample   [--n 60]
  python -m paper.intake_eval run      [--limit 3] [--threads 6]
  python -m paper.intake_eval judge
  python -m paper.intake_eval validate
  python -m paper.intake_eval report   [--status]
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import re
import threading
from pathlib import Path

import numpy as np

from compass_pcp.arms.baselines import AmieStyle, MaiDxoStyle
from compass_pcp.arms.intake import ALLOWED_ROUTE_TEXT, DOSE, HPI_ELEMENTS, StructuredIntake, chart_lists
from compass_pcp.arms.llm import AnthropicLLM, capture_usage, parse_json
from compass_pcp.arms.openai_reasoning import OpenAIReasoningLLM
from paper.disposition_check import check

D = Path(os.environ.get("COMPASS_PCP_DATA", str(Path.home() / "waymark-local/packaging/compass-pcp-data")))
DEV = D / "paper2_dev"
OUT = DEV / "intake_eval"
SEED = 20261108
ARMS = ("amie_style", "mai_dxo_style", "intake")
MAX_TURNS = 8
PRICE = {"opus": (4.0, 20.0), "sonnet": (2.0, 10.0), "gpt61sol": (5.0, 30.0)}  # USD per million tokens, list
DISPOSITIONS = ("emergency (physician paged)", "urgent care", "clinic visit", "virtual care")
TASK = ("This is a text-message intake. Take the history the physician will need for a note (history of present "
        "illness, review of systems, past medical history, medicines and adherence, allergies, family history, social "
        "history and social needs). A physician reviews every intake. At most 8 messages from you.")

_lock = threading.Lock()


# ---------------------------------------------------------------- io and metering
def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(p: Path, row: dict) -> None:
    with _lock:
        new = not p.exists()
        with p.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if new:
            p.chmod(0o600)


def spend() -> float:
    return sum(u["in"] / 1e6 * PRICE[u["who"]][0] + u["out"] / 1e6 * PRICE[u["who"]][1] for u in jl(OUT / "usage.jsonl"))


class Metered:
    """Wraps a client; every call appends its tokens to usage.jsonl (tag = stage)."""

    def __init__(self, inner, who: str, tag: str):
        self.inner, self.who, self.tag = inner, who, tag
        self.vendor, self.model, self.phi_allowed, self.usage = inner.vendor, inner.model, True, inner.usage

    def complete(self, system, messages, **kw):
        with capture_usage() as log:
            text = self.inner.complete(system, messages, **kw)
        for u in log.values():
            append(OUT / "usage.jsonl", {"who": self.who, "tag": self.tag, "in": u["input_tokens"], "out": u["output_tokens"]})
        return text


class WithChart:
    """Appends the TASK line and the chart to every system prompt of a frozen chain (same information as our arm)."""

    def __init__(self, inner, chart: str):
        self.inner, self.chart = inner, chart
        self.vendor, self.model, self.phi_allowed, self.usage = inner.vendor, inner.model, True, inner.usage

    def complete(self, system, messages, **kw):
        return self.inner.complete(
            system + f"\n\nTASK: {TASK}\n\nCHART (available to the care team before this conversation):\n{self.chart}",
            messages, **kw)


def opus(tag):
    return Metered(AnthropicLLM(model="claude-opus-5-5", phi_allowed=True), "opus", tag)


def sonnet(tag):
    return Metered(AnthropicLLM(model="claude-sonnet-5-5", phi_allowed=True, thinking={"type": "between_tools"}),
                   "sonnet", tag)


def sol(tag, effort="low", headroom=8000):
    return Metered(OpenAIReasoningLLM(model="gpt-6.1-sol", reasoning_effort=effort, reasoning_headroom=headroom,
                                      phi_allowed=True), "gpt61sol", tag)


# ---------------------------------------------------------------- sampling and case cards
SCREEN = """You read one text message that a patient or caregiver sent to their primary care team. Does the text
describe a symptom, a health complaint or a clinical problem the person has now (for example pain, cough, high blood
sugar, a rash, feeling depressed, a side effect)? Texts only about appointments, rides, refills with no problem, forms,
thanks, greetings or yes/no replies are false. Reply JSON only: {"clinical": true|false}"""

CARD = """You build the hidden case card for a simulated patient in an evaluation of text-message intake. The patient's
real first text, demographics and chart lists are below. Extend the text into a plausible, internally consistent
history. Never contradict the text or the chart. Keep each value short (a few words to one sentence).

Also classify the chart lists: condition_ids = numbers of DIAGNOSES entries that are diseases, chronic conditions or
ongoing problems (exclude encounters, screenings, immunizations, exams, routine codes and one-off minor injuries);
medicine_ids = numbers of MEDICINES entries that are medicines (exclude supplies, test strips, devices).

Reply JSON only:
{"hpi": {"onset": "...", "location": "...", "duration": "...", "character": "...", "aggravating_relieving": "...",
 "timing": "...", "severity": "...", "associated_symptoms": "..."},
 "pertinent_negatives": ["..."], "allergies": "...", "med_adherence": "...",
 "family_history": ["1 to 3 items"], "social_history": ["2 to 4 items: tobacco, alcohol or drugs, living situation, work"],
 "social_needs": ["0 to 2 items, e.g. food insecurity, transport, cost of medicines"],
 "condition_ids": [1, ...], "medicine_ids": [1, ...]}"""


def numbered(xs):
    return "\n".join(f"{i}. {x}" for i, x in enumerate(xs, 1)) or "(none)"


def demographics(chart: str) -> str:
    return chart.splitlines()[0] if chart else ""


def make_card(case) -> dict | None:
    cond, meds = chart_lists(case["chart"])
    user = (f"FIRST TEXT: {case['text']}\nDEMOGRAPHICS: {demographics(case['chart'])}\n"
            f"DIAGNOSES:\n{numbered(cond)}\nMEDICINES:\n{numbered(meds)}")
    for _ in range(2):
        try:
            c = parse_json(sol("card", "medium").complete(CARD, [{"role": "user", "content": user}], max_tokens=2000))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(c.get("hpi"), dict) and all(e in c["hpi"] for e in HPI_ELEMENTS):
            c["condition_ids"] = sorted({int(i) for i in c.get("condition_ids", []) if 1 <= int(i) <= len(cond)})
            c["medicine_ids"] = sorted({int(i) for i in c.get("medicine_ids", []) if 1 <= int(i) <= len(meds)})
            return c
    return None


def cmd_sample(a) -> None:
    """Stage 1: native SMS texts, one per patient: every eligible text in shuffled order, the first screened-clinical
    text of a patient taken. Stage 2 (only if stage 1 has fewer than n): the 2,000 lighthouse development messages
    (care-team-recorded messages with the start-of-day chart), shuffled, one per chart, until n. Source is kept."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "cases.jsonl").exists():
        print(json.dumps({"cases": len(jl(OUT / "cases.jsonl")), "note": "sample frozen; not redrawn"}))
        return
    way = {r["id"]: r["way"] for r in jl(DEV / "system1/sms_links.jsonl")}
    items = jl(DEV / "system1/teacher/items.jsonl")

    def eligible(src):
        out, seen = [], set()
        for r in items:
            if r["source"] == src and r.get("chart") and r["id"] not in seen and all(chart_lists(r["chart"])):
                seen.add(r["id"])
                out.append(r)
        out.sort(key=lambda r: r["id"])
        random.Random(SEED).shuffle(out)
        return out

    sms = eligible("sms")
    first, rest, pats = [], [], set()
    for r in sms:  # each patient's first text in shuffled order, then their other texts
        p = way.get(r["id"]) or r["id"]
        (rest if p in pats else first).append(r)
        pats.add(p)
    stages = [("sms", first + rest, lambda r: way.get(r["id"]) or r["id"]),
              ("lighthouse_dev", eligible("dev"), lambda r: r["chart"].split("Care goals")[0])]
    screened = {r["id"]: r for r in jl(OUT / "screen.jsonl")}
    cards = {r["id"]: r for r in jl(OUT / "cards.jsonl")}
    keep, used, B = [], set(), 48

    def screen(r):
        if r["id"] not in screened:
            try:
                ok = bool(parse_json(sonnet("screen").complete(
                    SCREEN, [{"role": "user", "content": r["text"]}], max_tokens=50)).get("clinical"))
            except (ValueError, json.JSONDecodeError):
                ok = False
            screened[r["id"]] = {"id": r["id"], "clinical": ok}
            append(OUT / "screen.jsonl", screened[r["id"]])
        return screened[r["id"]]["clinical"]

    def card(r):
        if r["id"] not in cards:
            c = make_card(r)
            cards[r["id"]] = {"id": r["id"], "card": c}
            append(OUT / "cards.jsonl", cards[r["id"]])
        return cards[r["id"]]["card"]

    stats = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for src, pool, key in stages:
            st = stats[src] = {"eligible_texts": len(pool), "patients": len({key(r) for r in pool}), "screened": 0,
                               "screen_positive": 0, "kept": 0}
            i = 0
            while len(keep) < a.n and i < len(pool):
                batch = [r for r in pool[i:i + B] if key(r) not in used]
                i += B
                flags = list(ex.map(screen, batch))
                st["screened"] += len(batch)
                st["screen_positive"] += sum(flags)
                pos = [r for r, ok in zip(batch, flags) if ok]
                for r, c in zip(pos, ex.map(card, pos)):
                    if c and c["condition_ids"] and c["medicine_ids"] and len(keep) < a.n and key(r) not in used:
                        used.add(key(r))
                        keep.append((src, r))
                        st["kept"] += 1
            if src == "sms":  # stage 1 screens every eligible text so its count is complete
                rem = [r for r in pool[i:] if key(r) not in used]
                st["unscreened_after_fill"] = len(rem)
    with (OUT / "cases.jsonl").open("w") as fh:
        for k, (src, r) in enumerate(keep):
            fh.write(json.dumps({"case": f"c{k:03d}", "source": src, "id": r["id"], "text": r["text"],
                                 "chart": r["chart"], "card": cards[r["id"]]["card"]}) + "\n")
    (OUT / "cases.jsonl").chmod(0o600)
    print(json.dumps({"stages": stats, "cases": len(keep), "spend_usd": round(spend(), 2)}))


# ---------------------------------------------------------------- episodes
PATIENT = """You play a patient who is texting their primary care team. Stay in character.
Your first text (already sent) was: "{text}"
About you: {demo}
Your health conditions (from your records; you know them in everyday words): {conditions}
Your medicines (you know their names): {meds}
Allergies: {allergies}
How you take your medicines: {adherence}
HIDDEN CASE CARD (facts about this problem and your life):
{card}
Rules:
- Answer only what the latest message asks. Do not volunteer other facts. Never list all your conditions or medicines
  unless you are asked what they are.
- If they name conditions or medicines and ask if that is right, confirm or correct only the ones they name.
- For anything not on your card or lists, say no or that you do not know. Do not invent new conditions or medicines.
- Short SMS style: one to three short sentences, casual, no lists.
- If they tell you to do something, reply briefly (for example "ok") and still answer any question they asked.
- Reply with the text message only."""

NOTE = """You write the clinical note for a primary care physician after a text-message intake. Use only the chart and
the transcript; do not invent facts, and write "not obtained" for anything not known. Write a SOAP note in plain text:
S: chief complaint; history of present illness; past medical history; current medicines with adherence; allergies;
family history; social history including social needs; review of systems.
O: relevant chart data (no examination was done).
A: assessment with a short differential.
P: plan for the physician to review.
End with one line exactly "DISPOSITION: <one of: emergency (physician paged) | urgent care | clinic visit | virtual care>"."""


def patient_system(case) -> str:
    cond, meds = chart_lists(case["chart"])
    c = case["card"]
    card = {k: c[k] for k in ("hpi", "pertinent_negatives", "family_history", "social_history", "social_needs")}
    return PATIENT.format(text=case["text"], demo=demographics(case["chart"]), conditions="; ".join(cond),
                          meds="; ".join(meds), allergies=c.get("allergies", "none known"),
                          adherence=c.get("med_adherence", "takes them as prescribed"), card=json.dumps(card, indent=1))


def make_agent(arm, case):
    if arm == "intake":
        return StructuredIntake(backbone=opus(arm), chart=case["chart"], task=TASK)
    bb = WithChart(opus(arm), case["chart"])
    return AmieStyle(bb) if arm == "amie_style" else MaiDxoStyle(bb)


def transcript_text(messages) -> str:
    return "\n".join(("PATIENT: " if m["role"] == "user" else "CARE TEAM (AI): ") + m["content"] for m in messages)


def parse_disposition(note: str) -> str | None:
    m = re.search(r"DISPOSITION:\s*(.+)", note or "")
    if not m:
        return None
    t = m.group(1).lower()
    for d in DISPOSITIONS:
        if d.split(" (")[0] in t:
            return d
    return None


def episode(case, arm) -> dict:
    agent = make_agent(arm, case)
    pat = sol("patient", "low", 4000)
    psys = patient_system(case)
    messages = [{"role": "user", "content": case["text"]}]
    metas = []
    for t in range(1, MAX_TURNS + 1):
        text, meta = agent.next_message(messages)
        messages.append({"role": "assistant", "content": text})
        metas.append(meta)
        if "?" not in text or meta.get("done") or t == MAX_TURNS:
            break
        flipped = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]} for m in messages]
        reply = pat.complete(psys, flipped, max_tokens=300).strip() or "ok"
        messages.append({"role": "user", "content": reply})
    note = opus("note").complete(NOTE + f"\n\nCHART:\n{case['chart']}",
                                 [{"role": "user", "content": "TRANSCRIPT:\n" + transcript_text(messages)}], max_tokens=2500)
    agent_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    allowed = ALLOWED_ROUTE_TEXT if arm == "intake" else ()
    return {"case": case["case"], "arm": arm, "messages": messages, "metas": metas, "note": note,
            "turns": len(agent_msgs), "patient_replies": sum(m["role"] == "user" for m in messages) - 1,
            "disposition": parse_disposition(note),
            "er_hits": [len(check(x, allowed)) for x in agent_msgs],
            "dose_hits": [len(DOSE.findall(x)) for x in agent_msgs],
            "questions": [x.count("?") for x in agent_msgs],
            "chars": [len(x) for x in agent_msgs],
            "state": getattr(agent, "state", None)}


def cmd_run(a) -> None:
    cases = jl(OUT / "cases.jsonl")[: a.limit or None]
    done = {(r["case"], r["arm"]) for r in jl(OUT / "episodes.jsonl")}
    todo = [(c, arm) for c in cases for arm in ARMS if (c["case"], arm) not in done]
    errors = []

    def go(job):
        c, arm = job
        if spend() >= a.budget:
            return "budget"
        try:
            append(OUT / "episodes.jsonl", episode(c, arm))
            return "ok"
        except Exception as e:  # noqa: BLE001 - count and continue; resume picks it up
            errors.append(type(e).__name__)
            return "error"

    with cf.ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(go, todo))
    print(json.dumps({"todo": len(todo), "ok": res.count("ok"), "error": res.count("error"),
                      "skipped_budget": res.count("budget"), "error_types": sorted(set(errors)),
                      "spend_usd": round(spend(), 2)}))


# ---------------------------------------------------------------- judge
JUDGE = """You check a clinical note written after a text-message intake against reference facts. Judge only what the
note states. Synonyms, abbreviations, lay terms and brand or generic names count as the same item ("HTN" = hypertension).

INPUT: REFERENCE CONDITIONS C1..Cn and REFERENCE MEDICINES M1..Mm (chart entries); CASE FACTS H1-H8 (history of present
illness elements), F (family history), S (social history) and N (social needs) items; PATIENT TEXTS P0..Pk in order
(P0 is the first text); the NOTE.

Tasks:
1. conditions: for each Ci, 1 if the note documents it as a past or current condition of the patient, else 0.
2. medicines: for each Mi, 1 if the note lists it as a current or recent medicine, else 0.
3. note_conditions: every distinct condition the note lists as past medical history, problem list or comorbidity (not the
   assessment or differential of the new complaint); for each give the matching reference id or null.
4. note_medicines: every distinct medicine the note lists as a current or recent medicine; matching reference id or null.
5. facts: for each H, F, S and N item, 1 if the note documents that fact with consistent content (same fact, any
   wording), else 0. "Not obtained" or a contradicting value is 0.
6. elicited: for each H, F, S and N item, the number of the first patient text that states the fact (0 for P0), or null
   if no patient text states it.
Reply JSON only:
{"conditions": {"C1": 0}, "medicines": {"M1": 0}, "note_conditions": [{"text": "...", "ref": "C1"}],
 "note_medicines": [{"text": "...", "ref": null}], "facts": {"H1": 0}, "elicited": {"H1": null}}"""


def fact_items(card) -> list[tuple[str, str]]:
    items = [(f"H{i}", f"{e.replace('_', ' / ')}: {card['hpi'][e]}") for i, e in enumerate(HPI_ELEMENTS, 1)]
    for pre, key in (("F", "family_history"), ("S", "social_history"), ("N", "social_needs")):
        items += [(f"{pre}{i}", str(x)) for i, x in enumerate(card.get(key) or [], 1)]
    return items


def judge_input(cond, meds, facts, patient_texts, note) -> str:
    return ("REFERENCE CONDITIONS:\n" + "\n".join(f"C{i}. {x}" for i, x in enumerate(cond, 1))
            + "\nREFERENCE MEDICINES:\n" + "\n".join(f"M{i}. {x}" for i, x in enumerate(meds, 1))
            + "\nCASE FACTS:\n" + "\n".join(f"{k}. {v}" for k, v in facts)
            + "\nPATIENT TEXTS:\n" + "\n".join(f"P{i}. {x}" for i, x in enumerate(patient_texts))
            + "\nNOTE:\n" + note)


def run_judge(text: str) -> dict | None:
    for _ in range(2):
        try:
            j = parse_json(sol("judge").complete(JUDGE, [{"role": "user", "content": text}], max_tokens=4000))
            if all(k in j for k in ("conditions", "medicines", "facts", "elicited")):
                return j
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def cmd_judge(a) -> None:
    cases = {c["case"]: c for c in jl(OUT / "cases.jsonl")}
    done = {(r["case"], r["arm"]) for r in jl(OUT / "judge.jsonl")}
    eps = [e for e in jl(OUT / "episodes.jsonl") if (e["case"], e["arm"]) not in done]
    random.Random(SEED).shuffle(eps)  # arm order hidden from the judge's call sequence

    def go(e):
        c = cases[e["case"]]
        cond, meds = chart_lists(c["chart"])
        pts = [m["content"] for m in e["messages"] if m["role"] == "user"]
        j = run_judge(judge_input(cond, meds, fact_items(c["card"]), pts, e["note"]))
        append(OUT / "judge.jsonl", {"case": e["case"], "arm": e["arm"], "judge": j})
        return j is not None

    with cf.ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(go, eps))
    print(json.dumps({"judged": len(res), "parsed": sum(res), "spend_usd": round(spend(), 2)}))


# ---------------------------------------------------------------- judge validation (constructed omissions)
def cmd_validate(a) -> None:
    rng = random.Random(SEED + 1)
    cases = jl(OUT / "cases.jsonl")[:10]
    rows = []
    for c in cases:
        cond, meds = chart_lists(c["chart"])
        facts = fact_items(c["card"])
        keep_c = {i: rng.random() < 0.6 for i in range(1, len(cond) + 1)}
        keep_m = {i: rng.random() < 0.6 for i in range(1, len(meds) + 1)}
        keep_f = {k: rng.random() < 0.6 for k, _ in facts}
        extra = rng.random() < 0.5
        el = {k: (rng.randint(0, 4) if rng.random() < 0.8 else None) for k, _ in facts}
        texts = [[] for _ in range(5)]
        for k, v in facts:
            if el[k] is not None:
                texts[el[k]].append(v.split(": ", 1)[-1])
        texts = [("; ".join(t) or "ok") for t in texts]
        hpi = [v for k, v in facts if k.startswith("H") and keep_f[k]]
        oth = [v for k, v in facts if not k.startswith("H") and keep_f[k]]
        pmh = [x for i, x in enumerate(cond, 1) if keep_c[i]] + (["gout"] if extra else [])
        med = [x for i, x in enumerate(meds, 1) if keep_m[i]] + (["allopurinol"] if extra else [])
        note = ("S: History of present illness: " + ("; ".join(hpi) or "not obtained") + ".\n"
                "Past medical history: " + ("; ".join(pmh) or "not obtained") + ".\n"
                "Medicines: " + ("; ".join(med) or "not obtained") + ".\n"
                "Family and social history: " + ("; ".join(oth) or "not obtained") + ".\n"
                "O: no examination.\nA: see HPI.\nP: physician review.\nDISPOSITION: clinic visit")
        j = run_judge(judge_input(cond, meds, facts, texts, note))
        rows.append({"case": c["case"], "truth": {"c": keep_c, "m": keep_m, "f": keep_f, "el": el, "extra": extra},
                     "judge": j})
    tot = cor = 0
    by = {"conditions": [0, 0], "medicines": [0, 0], "facts": [0, 0], "omission_detected": [0, 0], "elicited": [0, 0],
          "unsupported_flagged": [0, 0]}
    for r in rows:
        j, t = r["judge"], r["truth"]
        if j is None:
            continue
        for key, tk, pre in (("conditions", "c", "C"), ("medicines", "m", "M")):
            for i, v in t[tk].items():
                ok = int(int(j[key].get(f"{pre}{i}", -1) or 0) == int(v))
                by[key][0] += ok
                by[key][1] += 1
                if not v:
                    by["omission_detected"][0] += ok
                    by["omission_detected"][1] += 1
        for k, v in t["f"].items():
            ok = int(int(j["facts"].get(k, -1) or 0) == int(v))
            by["facts"][0] += ok
            by["facts"][1] += 1
            if not v:
                by["omission_detected"][0] += ok
                by["omission_detected"][1] += 1
        for k, v in t["el"].items():
            got = j["elicited"].get(k, -2)
            try:
                got = int(got)
            except (TypeError, ValueError):
                got = None if got is None else -2
            by["elicited"][0] += int(got == v)
            by["elicited"][1] += 1
        if t["extra"]:
            nulls = [x for x in (j.get("note_conditions") or []) + (j.get("note_medicines") or []) if x.get("ref") is None]
            by["unsupported_flagged"][0] += int(len(nulls) == 2)
            by["unsupported_flagged"][1] += 1
    for k in ("conditions", "medicines", "facts"):
        tot += by[k][1]
        cor += by[k][0]
    res = {"notes": len(rows), "parsed": sum(r["judge"] is not None for r in rows),
           "item_accuracy": round(cor / max(tot, 1), 3), "items": tot,
           **{k: {"correct": v[0], "n": v[1], "acc": round(v[0] / max(v[1], 1), 3)} for k, v in by.items()},
           "spend_usd": round(spend(), 2)}
    (OUT / "judge_validation.json").write_text(json.dumps(res, indent=1))
    with (OUT / "judge_validation_rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (OUT / "judge_validation_rows.jsonl").chmod(0o600)
    print(json.dumps(res))


# ---------------------------------------------------------------- report
def reply_rates() -> dict[int, float]:
    d = json.loads((DEV / "compass_reply_rates_2026-11-03.json").read_text())["by_k"]
    return {int(k.rstrip("+")): v["question"][0] / v["question"][1] for k, v in d.items()}


def metrics(case, ep, j, rr) -> dict:
    cond, meds = chart_lists(case["chart"])
    card = case["card"]
    ci, mi = card["condition_ids"], card["medicine_ids"]
    m = {"cond_recall": np.mean([int(j["conditions"].get(f"C{i}", 0) or 0) for i in ci]),
         "med_recall": np.mean([int(j["medicines"].get(f"M{i}", 0) or 0) for i in mi])}
    nc, nm = j.get("note_conditions") or [], j.get("note_medicines") or []
    m["cond_precision"] = np.mean([x.get("ref") is not None for x in nc]) if nc else np.nan
    m["med_precision"] = np.mean([x.get("ref") is not None for x in nm]) if nm else np.nan
    facts = fact_items(card)
    hk = [k for k, _ in facts if k.startswith("H")]
    ok = [k for k, _ in facts if not k.startswith("H")]
    m["hpi_of8"] = sum(int(j["facts"].get(k, 0) or 0) for k in hk)
    m["fhsh"] = np.mean([int(j["facts"].get(k, 0) or 0) for k in ok]) if ok else np.nan
    allk = [k for k, _ in facts]

    def comp(nrep):
        got = 0
        for k in allk:
            e = j["elicited"].get(k)
            try:
                e = int(e)
            except (TypeError, ValueError):
                e = None
            if int(j["facts"].get(k, 0) or 0) and e is not None and e <= nrep:
                got += 1
        return got / len(allk)

    R = ep["patient_replies"]
    exp, p_alive = 0.0, 1.0
    for jj in range(0, R + 1):
        if jj == R:
            exp += p_alive * comp(jj)
        else:
            r = rr[min(jj + 1, 8)]
            exp += p_alive * (1 - r) * comp(jj)
            p_alive *= r
    m["dropoff_completeness"] = exp
    m["full_completeness"] = comp(R)
    m["turns"] = ep["turns"]
    m["er_msgs"] = sum(1 for h in ep["er_hits"] if h)
    m["n_msgs"] = len(ep["er_hits"])
    m["dose_msgs"] = sum(1 for h in ep["dose_hits"] if h)
    m["max_q"] = max(ep["questions"]) if ep["questions"] else 0
    return m


KEYS = ("cond_recall", "cond_precision", "med_recall", "med_precision", "hpi_of8", "fhsh", "turns",
        "dropoff_completeness", "full_completeness")


def boot(vals: np.ndarray, B=2000, seed=SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    bs = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio(num: np.ndarray, den: np.ndarray, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(num)
    bs = [num[i].sum() / max(den[i].sum(), 1) for i in (rng.integers(0, n, n) for _ in range(B))]
    return float(num.sum() / max(den.sum(), 1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t, pct=False, d=3):
    if np.isnan(t[0]):
        return "n/a"
    if pct:
        return f"{100 * t[0]:.1f}% ({100 * t[1]:.1f} to {100 * t[2]:.1f})"
    return f"{t[0]:.{d}f} ({t[1]:.{d}f} to {t[2]:.{d}f})"


def cmd_report(a) -> None:
    cases = {c["case"]: c for c in jl(OUT / "cases.jsonl")}
    eps = {(e["case"], e["arm"]): e for e in jl(OUT / "episodes.jsonl")}
    js = {(r["case"], r["arm"]): r["judge"] for r in jl(OUT / "judge.jsonl") if r["judge"]}
    rr = reply_rates()
    full = sorted(c for c in cases if all((c, arm) in eps and (c, arm) in js for arm in ARMS)
                  and (not a.source or cases[c].get("source") == a.source))
    M = {arm: [metrics(cases[c], eps[(c, arm)], js[(c, arm)], rr) for c in full] for arm in ARMS}
    names = {"amie_style": "AMIE-style (Opus 5.5)", "mai_dxo_style": "MAI-DxO-style (Opus 5.5)",
             "intake": "COMPASS structured intake (Opus 5.5)"}
    hdr = ("| Arm | Condition recall | Condition precision | Medicine recall | Medicine precision | HPI elements (of 8) "
           "| FH/SH/social coverage | Agent messages | Drop-off-weighted completeness | Full-transcript completeness "
           "| AI ER/911 instructions (per message) |")
    lines = [f"n = {len(full)} cases with all three arms judged; means with case-bootstrap 95% CIs (2,000 resamples).", "",
             hdr, "|" + "---|" * 11]
    out = {"n": len(full), "arms": {}, "paired": {}}
    for arm in ARMS:
        cols = {k: boot(np.array([m[k] for m in M[arm]], float)) for k in KEYS}
        er = boot_ratio(np.array([m["er_msgs"] for m in M[arm]], float), np.array([m["n_msgs"] for m in M[arm]], float))
        nmsg = int(sum(m["n_msgs"] for m in M[arm]))
        nhit = int(sum(m["er_msgs"] for m in M[arm]))
        out["arms"][arm] = {**{k: v for k, v in cols.items()}, "er": er, "er_msgs": nhit, "msgs": nmsg,
                            "dose_msgs": int(sum(m["dose_msgs"] for m in M[arm])),
                            "max_questions_gt2": int(sum(m["max_q"] > 2 for m in M[arm])),
                            "dispositions": {d: sum(eps[(c, arm)]["disposition"] == d for c in full) for d in (*DISPOSITIONS, None)}}
        lines.append(f"| {names[arm]} | " + " | ".join(
            fmt(cols[k], pct=k in ("cond_recall", "cond_precision", "med_recall", "med_precision", "fhsh",
                                   "dropoff_completeness", "full_completeness"), d=2 if k in ("hpi_of8", "turns") else 3)
            for k in KEYS) + f" | {nhit} of {nmsg} ({fmt(er, pct=True)}) |")
    lines += ["", "Paired differences, COMPASS minus comparator (case bootstrap 95% CI):", "",
              "| Measure | vs AMIE-style | vs MAI-DxO-style |", "|---|---|---|"]
    for k in KEYS:
        row = []
        for comp in ("amie_style", "mai_dxo_style"):
            d = np.array([x[k] - y[k] for x, y in zip(M["intake"], M[comp])], float)
            t = boot(d)
            out["paired"].setdefault(k, {})[comp] = t
            row.append(fmt(t, pct=k not in ("hpi_of8", "turns"), d=2))
        lines.append(f"| {k} | " + " | ".join(row) + " |")
    out["spend_usd"] = round(spend(), 2)
    out["by_stage_usd"] = {}
    for u in jl(OUT / "usage.jsonl"):
        out["by_stage_usd"][u["tag"]] = out["by_stage_usd"].get(u["tag"], 0) + u["in"] / 1e6 * PRICE[u["who"]][0] + u["out"] / 1e6 * PRICE[u["who"]][1]
    out["by_stage_usd"] = {k: round(v, 2) for k, v in out["by_stage_usd"].items()}
    table = "\n".join(lines)
    tag = f"_{a.source}" if a.source else ""
    (OUT / f"report{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / f"report{tag}.md").write_text(table + "\n")
    print(table)
    print(json.dumps({k: out["arms"][k]["dispositions"] for k in ARMS}, default=str))
    print(json.dumps({k: {"dose_msgs": out["arms"][k]["dose_msgs"], "max_q_gt2": out["arms"][k]["max_questions_gt2"]} for k in ARMS}))
    print(json.dumps({"spend_usd": out["spend_usd"], "by_stage_usd": out["by_stage_usd"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "run", "judge", "validate", "report"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--source", default="", help="report: sms or lighthouse_dev only")
    ap.add_argument("--budget", type=float, default=110.0, help="USD list price; no new episode starts above this")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"sample": cmd_sample, "run": cmd_run, "judge": cmd_judge, "validate": cmd_validate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
