"""Paper 2 development: outpatient cases and order appropriateness (design/paper2_plan.md, section 5).

The Analysis 4 strata are emergency presentations, where the right order is the emergency department. Order
appropriateness needs presentations that are worked up in primary care, so this module adds DDXPlus validate-split
strata for sore throat, sinus symptoms and ear pain, and scores the orders each design recommends.

1. cases:    --per-stratum patients per (complaint, pathology) stratum below; epiglottitis is the cannot-miss stratum.
2. reference: two models from developers other than the system's own (OpenAI GPT-5.5, Google Gemini 3.1 Pro) read every
             fact (no diagnosis) and rate each item of the complaint's order list as indicated, acceptable or
             not_indicated under current guidelines. Both indicated -> indicated; both not_indicated -> low_value;
             otherwise acceptable. A model-panel development reference, not a physician label.
3. extract:  a model from another developer (OpenAI GPT-5.5) reads every reply of an episode, plus the clinician-facing
             pathway proposals when the design has them, and marks which list items the design recommends, proposes
             or says it will arrange.
4. score:    per arm, the share of indicated orders recommended, the share of episodes with any low-value order, and
             antibiotics recommended where the reference rates them low value.

Usage: python -m paper.dev_orders {cases|reference|extract|score} --out <dir> [--per-stratum 12] [--threads 8]
Episodes come from: python -m paper.dev_multiturn run --out <dir> --arms ...
"""

import argparse
import concurrent.futures as cf
import json
import random
import threading
from collections import defaultdict
from pathlib import Path

from compass_pcp.arms.llm import GeminiLLM, parse_json
from compass_pcp.harness.metrics import wilson
from paper import dev_multiturn  # noqa: F401 - registers the outpatient openers
from paper import multiturn_ddxplus as mt
from paper.dev_reference import OpenAIReasoning, facts

SEED = 20261104
STRATA = {  # (complaint, DDXPlus pathology) -> (stratum slug, cannot-miss id or None)
    ("sore_throat", "Viral pharyngitis"): ("viral_pharyngitis", None),
    ("sore_throat", "URTI"): ("urti", None),
    ("sore_throat", "Acute laryngitis"): ("laryngitis", None),
    ("sore_throat", "Epiglottitis"): ("epiglottitis", "epiglottitis"),
    ("acute_sinusitis", "Acute rhinosinusitis"): ("acute_rhinosinusitis", None),
    ("acute_sinusitis", "Chronic rhinosinusitis"): ("chronic_rhinosinusitis", None),
    ("acute_sinusitis", "Allergic sinusitis"): ("allergic_sinusitis", None),
    ("ear_pain", "Acute otitis media"): ("acute_otitis_media", None),
}
ORDERS = {
    "sore_throat": ["rapid strep antigen test or throat culture", "oral antibiotic prescription",
                    "mononucleosis (EBV) test", "HIV test", "neck imaging (lateral neck X-ray or CT)",
                    "emergency airway evaluation", "complete blood count", "corticosteroid prescription"],
    "acute_sinusitis": ["oral antibiotic prescription", "sinus CT", "sinus X-ray", "ENT referral",
                        "intranasal corticosteroid", "allergy testing or allergist referral", "MRI of the head",
                        "emergency evaluation for orbital or intracranial complication"],
    "ear_pain": ["in-person ear examination (otoscopy)", "oral antibiotic prescription", "antibiotic ear drops",
                 "hearing test", "ENT referral", "CT of the temporal bone", "emergency evaluation for mastoiditis"],
}
ANTIBIOTIC = "oral antibiotic prescription"
PANEL = ("openai:gpt-5.5", "google:gemini-3.1-pro-preview")
EXTRACTOR = "gpt-5.5"

REF_PROMPT = """You are a board-certified primary care physician. A patient texted a primary care service. You know every
fact below (the patient would give each one if asked), but you cannot examine them and have no test results. Apply
current society guidelines (for example IDSA group A strep pharyngitis 2012, IDSA acute bacterial rhinosinusitis 2012,
AAO-HNS adult sinusitis 2015 and acute otitis externa 2014, ACP/CDC high-value care for respiratory infections 2016).

Patient: {age}-year-old {sex}. First message: "{opener}"
Facts (question | answer); every item not listed was answered "no":
{facts}

For each item below, rate whether a clinician should order or recommend it for this patient now:
indicated (guidelines call for it), acceptable (reasonable either way), or not_indicated (low value or not
recommended). Items:
{items}
Reply JSON only: {{"ratings": {{"<item>": "indicated|acceptable|not_indicated", ...}}}}"""

EXTRACT_PROMPT = """Below are all the replies that a health service sent to a patient in one text conversation, and the
proposals it sent to a clinician, if any. For each item in the list, answer true if the service recommended it,
proposed it, said it would arrange it, or told the patient to get it; answer false if it only mentioned it as
something that might be needed later under a condition, or did not mention it.
Items:
{items}

REPLIES TO THE PATIENT:
{replies}

PROPOSALS TO A CLINICIAN:
{proposals}

Reply JSON only: {{"recommended": {{"<item>": true or false, ...}}}}"""


def build_cases(out: Path, per_stratum: int) -> list[dict]:
    path = out / "cases.jsonl"
    if path.exists():
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in mt.read_split("validate"):
        key = (r["complaint"], r["pathology"])
        if key in STRATA:
            slug, cm = STRATA[key]
            by[f"{r['complaint']}/{slug}"].append({**r, "stratum": f"{r['complaint']}/{slug}", "cannot_miss": cm,
                                                    "benign": cm is None})
    rng = random.Random(SEED)
    cases = [c for s in sorted(by) for c in rng.sample(by[s], min(per_stratum, len(by[s])))]
    out.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    return cases


def _items(complaint: str) -> str:
    return "\n".join(f"- {o}" for o in ORDERS[complaint])


def _complete(spec: str, prompt: str, clients: dict) -> dict:
    vendor, model = spec.split(":", 1)
    if vendor == "openai":
        text = clients.setdefault(spec, OpenAIReasoning(model)).complete(prompt)
    else:
        text = clients.setdefault(spec, GeminiLLM(model=model, thinking_level="HIGH")).complete(
            "", [{"role": "user", "content": prompt}], max_tokens=2000)
    return parse_json(text)


def _pool(items, fn, threads: int, path: Path) -> int:
    lock, local = threading.Lock(), threading.local()

    def one(x):
        if not hasattr(local, "clients"):
            local.clients = {}
        try:
            rec = fn(x, local.clients)
        except Exception as err:  # noqa: BLE001 - logged; retried on the next run
            print("error", type(err).__name__, str(err)[:160], flush=True)
            return 1
        with lock, path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, items))


def reference(out: Path, threads: int) -> None:
    cases = build_cases(out, 0)
    path = out / "order_reference.jsonl"
    done = {r["case_id"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    ev = mt.load_evidences()
    mt.LEG_SIDES_CACHE["sides"] = mt.leg_sides(ev)

    def fn(case, clients):
        prompt = REF_PROMPT.format(age=case["age"], sex="man" if case["sex"] == "M" else "woman",
                                   opener=mt.opener_text(case), facts="\n".join(facts(case, ev)),
                                   items=_items(case["complaint"]))
        votes = {}
        for spec in PANEL:
            r = _complete(spec, prompt, clients)["ratings"]
            votes[spec] = {o: r.get(o) for o in ORDERS[case["complaint"]]}
        label = {}
        for o in ORDERS[case["complaint"]]:
            vs = [v[o] for v in votes.values()]
            label[o] = ("indicated" if all(x == "indicated" for x in vs) else
                        "low_value" if all(x == "not_indicated" for x in vs) else "acceptable")
        return {"case_id": case["case_id"], "votes": votes, "label": label}

    print(json.dumps({"failed": _pool([c for c in cases if c["case_id"] not in done], fn, threads, path)}))


def extract(out: Path, threads: int) -> None:
    cases = {c["case_id"]: c for c in build_cases(out, 0)}
    path = out / "orders_extracted.jsonl"
    done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    eps = [e for e in map(json.loads, (out / "episodes.jsonl").read_text().splitlines())
           if e["case_id"] in cases and (e["case_id"], e["arm"]) not in done]

    def fn(ep, clients):
        replies = "\n\n".join(f"[{t['turn']}] {t['reply']}" for t in ep["turns"])
        proposals = sorted({p["name"] for t in ep["turns"] for n in (t.get("pathway") or {}).values() if n
                            for p in n.get("proposals", [])})
        prompt = EXTRACT_PROMPT.format(items=_items(ep["complaint"]), replies=replies,
                                       proposals="\n".join(f"- {p}" for p in proposals) or "(none)")
        rec = clients.setdefault("x", OpenAIReasoning(EXTRACTOR)).complete(prompt)
        got = parse_json(rec)["recommended"]
        return {"case_id": ep["case_id"], "arm": ep["arm"],
                "recommended": {o: bool(got.get(o)) for o in ORDERS[ep["complaint"]]}}

    print(json.dumps({"failed": _pool(eps, fn, threads, path)}))


def score(out: Path) -> dict:
    cases = {c["case_id"]: c for c in build_cases(out, 0)}
    ref = {r["case_id"]: r["label"] for r in map(json.loads, (out / "order_reference.jsonl").read_text().splitlines())}
    rows = [r for r in map(json.loads, (out / "orders_extracted.jsonl").read_text().splitlines()) if r["case_id"] in ref]

    def prop(k, n):
        lo, hi = wilson(k, n) if n else (0, 0)
        return f"{k}/{n} ({100 * k / n:.1f}%; {lo:.2f}-{hi:.2f})" if n else "0/0"

    res = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        ind = [(r, o) for r in rs for o, lab in ref[r["case_id"]].items() if lab == "indicated"]
        low = [r for r in rs if any(r["recommended"][o] and ref[r["case_id"]][o] == "low_value" for o in r["recommended"])]
        abx_low = [r for r in rs if ref[r["case_id"]].get(ANTIBIOTIC) == "low_value"]
        res[arm] = {"episodes": len(rs),
                    "indicated_orders_recommended": prop(sum(r["recommended"][o] for r, o in ind), len(ind)),
                    "episodes_with_low_value_order": prop(len(low), len(rs)),
                    "antibiotic_when_low_value": prop(sum(r["recommended"][ANTIBIOTIC] for r in abx_low), len(abx_low)),
                    "by_complaint": {c: prop(sum(1 for r in low if cases[r["case_id"]]["complaint"] == c),
                                             sum(1 for r in rs if cases[r["case_id"]]["complaint"] == c))
                                     for c in ORDERS}}
    (out / "order_score.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "reference", "extract", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "cases":
        print(len(build_cases(a.out, a.per_stratum)))
    elif a.cmd == "reference":
        reference(a.out, a.threads)
    elif a.cmd == "extract":
        extract(a.out, a.threads)
    else:
        print(json.dumps(score(a.out), indent=1))


if __name__ == "__main__":
    main()
