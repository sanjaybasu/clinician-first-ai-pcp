"""Post hoc sensitivity analyses for the paper (not pre-registered; reported as such in Supplementary Section S14).

1. Reply-to-reply variability: spread of the HealthBench score across the three replies, and the COMPASS minus
   LLM-alone difference computed from each single reply.
2. Grader disagreement by theme: COMPASS minus LLM-alone difference within each HealthBench theme, under each grader.
3. Equivalence with the MAI-DxO-style panel: two one-sided tests at the registered HB1 margin (0.02), from the 90%
   paired-bootstrap interval of the score difference.
4. Physician scoring without the care-team cue: the physician comparison restricted to items in which neither reply
   mentions a care team or nurse.
5. Patient overlap: hazard sensitivity of the frozen screens on sealed messages from patients with no message in the
   tuning half.
6. Patient groups: flag rate, specificity and hazards flagged on the sealed half by sex, race and ethnicity, and age group.
7. Patient-level bootstrap: hazard sensitivity and specificity with a bootstrap that resamples patients.
8. Reply length: COMPASS minus each comparator at equal reply length, from a paired per-conversation regression of the
   score difference on the difference in words per reply (intercept; bootstrap over conversations).
Also written, as descriptive inputs to the tables rather than sensitivity analyses: words per reply by system, agreement
between the two physicians (Cohen kappa and Gwet AC1), physician preferences, and the tuning-half operating point of the
frozen hazard rule (Supplementary Section S8).

Usage: python -m paper.sensitivity --hb-run <dir> --physician-packet <dir> --ratings <csv> <csv> \\
           --per-message <per_message.jsonl> --tune-ids <json> --test-ids <json> --notes <encounter notes.csv> --members <member_attributes.parquet> \\
           [--tune-scores <hazard_cohortA_tune dir> --tune-labels <tune_diag_1001/per_message.jsonl>] > out.json
Writes aggregate numbers only.
"""

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from compass_pcp.harness.hb_analysis import N_BOOT, SEED, load, themes_from_split
from compass_pcp.harness.metrics import paired_bootstrap_diff

FINAL, RAW, MAI = "ai_pcp_v3", "raw_llm", "mai_dxo_style"
COMPARATORS = ("raw_llm", "llm_standard_guardrails", "amie_style", "mai_dxo_style")
POST_HOC = ("repeats", "by_theme", "equivalence", "physician_blinding", "patient_overlap", "patient_groups",
            "patient_clustered", "length_adjusted")
CARE_TEAM = re.compile(r"care team|nurse", re.IGNORECASE)
MARGIN = 0.02


def _repeat_scores(path: Path) -> dict[str, list[float]]:
    reps: dict[str, list[float]] = defaultdict(list)
    for e in json.loads(path.read_text())["example_level"]:
        reps[e["prompt_id"]].append(max(0.0, min(1.0, e["score"])))
    return reps


def _boot_diff(diffs: list[float], lo_q: float, hi_q: float) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(diffs) / n, boots[int(lo_q * N_BOOT)], boots[int(hi_q * N_BOOT) - 1]


def repeats(run: Path) -> dict:
    out = {}
    for grader, suffix in (("grader1", ""), ("grader2", "__grader2")):
        a, b = _repeat_scores(run / f"{FINAL}{suffix}.json"), _repeat_scores(run / f"{RAW}{suffix}.json")
        pids = sorted(p for p in set(a) & set(b) if len(a[p]) == len(b[p]) == 3)
        sd = {arm: statistics.mean(statistics.pstdev(r[p]) for p in pids) for arm, r in ((FINAL, a), (RAW, b))}
        single = [_boot_diff([a[p][k] - b[p][k] for p in pids], 0.025, 0.975) for k in range(3)]
        out[grader] = {"n": len(pids), "mean_within_sd": sd, "single_reply_diffs": single}
    return out


def by_theme(run: Path) -> dict:
    themes = themes_from_split(run)
    out = {}
    for grader, suffix in (("grader1", ""), ("grader2", "__grader2")):
        a, b = load(run / f"{FINAL}{suffix}.json")["score"], load(run / f"{RAW}{suffix}.json")["score"]
        groups: dict[str, list[float]] = defaultdict(list)
        for p in set(a) & set(b):
            groups[themes.get(p, "unknown")].append(a[p] - b[p])
        out[grader] = {t: {"n": len(v), "diff": sum(v) / len(v)} for t, v in sorted(groups.items())}
    return out


def equivalence(run: Path) -> dict:
    out = {}
    for grader, suffix in (("grader1", ""), ("grader2", "__grader2")):
        a, b = load(run / f"{FINAL}{suffix}.json")["score"], load(run / f"{MAI}{suffix}.json")["score"]
        pids = sorted(set(a) & set(b))
        d, lo, hi = _boot_diff([a[p] - b[p] for p in pids], 0.05, 0.95)
        out[grader] = {"diff": d, "ci90": [lo, hi], "equivalent": lo > -MARGIN and hi < MARGIN}
    return out


def _physician_marks(key: dict, ratings: list[Path]) -> list[dict]:
    """Per physician: (prompt_id, arm, criterion index) -> criterion marked met."""
    marks = []
    for path in ratings:
        m = {}
        with path.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                m[(r["prompt_id"], key[r["prompt_id"]][r["response"]], int(r["criterion_index"]))] = r["met"].strip() == "1"
        marks.append(m)
    return marks


def physician_agreement(packet: Path, ratings: list[Path]) -> dict:
    """Agreement between the two physicians on criterion judgments (the same judgments as the Gwet AC1 in
    scripts/physician_packet.py): Cohen kappa, Gwet AC1 and raw agreement."""
    from compass_pcp.harness.adjudication import gwet_ac1

    key = json.loads((packet / "physician_key.json").read_text())["items"]
    m1, m2 = _physician_marks(key, ratings)
    assert set(m1) == set(m2)
    keys = sorted(m1)
    n = len(keys)
    po = sum(m1[k] == m2[k] for k in keys) / n
    p1, p2 = sum(m1[k] for k in keys) / n, sum(m2[k] for k in keys) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return {"n_judgments": n, "raw_agreement": po, "cohen_kappa": (po - pe) / (1 - pe),
            "gwet_ac1": gwet_ac1([[str(int(m1[k])), str(int(m2[k]))] for k in keys]), "met_rate": [p1, p2]}


def _words(e: dict) -> int:
    text = e["completion"][0]["content"] if isinstance(e["completion"], list) else str(e["completion"])
    return len(text.split())


def reply_length(run: Path) -> dict:
    """Words per reply (whitespace tokens of the final reply) for each system, over all registered replies."""
    out = {}
    for arm in (FINAL, *COMPARATORS):
        w = [_words(e) for e in json.loads((run / f"{arm}.json").read_text())["example_level"]]
        out[arm] = {"n_replies": len(w), "mean": statistics.mean(w), "median": statistics.median(w)}
    return out


def _per_conversation(path: Path) -> tuple[dict, dict]:
    """Mean clipped score and mean words per reply for each conversation."""
    sc, wd = defaultdict(list), defaultdict(list)
    for e in json.loads(path.read_text())["example_level"]:
        sc[e["prompt_id"]].append(max(0.0, min(1.0, e["score"])))
        wd[e["prompt_id"]].append(_words(e))
    return ({p: sum(v) / len(v) for p, v in sc.items()}, {p: sum(v) / len(v) for p, v in wd.items()})


def length_adjusted(run: Path) -> dict:
    """COMPASS minus each comparator at equal reply length: per conversation, regress the score difference on the
    difference in mean words per reply (per 100 words); the intercept is the difference at equal length. 95% CI from a
    bootstrap over conversations (N_BOOT resamples, seed SEED)."""
    import numpy as np

    out = {}
    for grader, suffix in (("grader1", ""), ("grader2", "__grader2")):
        sa, wa = _per_conversation(run / f"{FINAL}{suffix}.json")
        res = {}
        for arm in COMPARATORS:
            sb, wb = _per_conversation(run / f"{arm}{suffix}.json")
            pids = sorted(set(sa) & set(sb))
            y = np.array([sa[p] - sb[p] for p in pids])
            x = np.array([(wa[p] - wb[p]) / 100 for p in pids])

            def fit(xx, yy):
                mx, my = xx.mean(-1, keepdims=True), yy.mean(-1, keepdims=True)
                b = ((xx - mx) * (yy - my)).sum(-1) / ((xx - mx) ** 2).sum(-1)
                return my[..., 0] - b * mx[..., 0], b

            a0, b0 = fit(x, y)
            rng = np.random.default_rng(SEED)
            ints, slopes = [], []
            for _ in range(N_BOOT // 1000):
                idx = rng.integers(0, len(pids), size=(1000, len(pids)))
                a, b = fit(x[idx], y[idx])
                ints.append(a)
                slopes.append(b)
            ints, slopes = np.sort(np.concatenate(ints)), np.sort(np.concatenate(slopes))
            lo, hi = int(0.025 * N_BOOT), int(0.975 * N_BOOT) - 1
            res[arm] = {"n": len(pids), "unadjusted_diff": float(y.mean()), "mean_word_diff": float(100 * x.mean()),
                        "intercept": float(a0), "intercept_ci95": [float(ints[lo]), float(ints[hi])],
                        "slope_per_100_words": float(b0), "slope_ci95": [float(slopes[lo]), float(slopes[hi])]}
        out[grader] = res
    return out


def physician_blinding(run: Path, packet: Path, ratings: list[Path]) -> dict:
    key = json.loads((packet / "physician_key.json").read_text())["items"]
    first = {}
    for arm in (FINAL, RAW):
        for e in json.loads((run / f"{arm}.json").read_text())["example_level"]:
            first.setdefault((arm, e["prompt_id"]), e)
    cue = defaultdict(bool)
    with (packet / "physician_rating_sheet.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["response_text"].strip() and CARE_TEAM.search(r["response_text"]):
                cue[r["prompt_id"]] = True
    marks = _physician_marks(key, ratings)

    def score(pid: str, arm: str) -> float:
        items = first[(arm, pid)]["rubric_items"]
        total = sum(it["points"] for it in items if it["points"] > 0)
        per_rater = [max(0.0, min(1.0, sum(it["points"] for j, it in enumerate(items) if m[(pid, arm, j)]) / total))
                     for m in marks]
        return sum(per_rater) / len(per_rater)

    keep = sorted(p for p in key if not cue[p])
    d, lo, hi = paired_bootstrap_diff([score(p, FINAL) for p in keep], [score(p, RAW) for p in keep], seed=SEED)
    return {"n_items": len(keep), "n_excluded": len(key) - len(keep), "diff": [d, lo, hi]}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return [c - h, c + h]


def patient_overlap(per_message: Path, tune_ids: Path, test_ids: Path, notes: Path) -> dict:
    def ids(p: Path) -> list[str]:
        d = json.loads(p.read_text())
        return d if isinstance(d, list) else d["ids"]

    patient = {}
    with notes.open(newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): (v or "").strip() for k, v in r.items()}
            patient[r["Encounter ID"]] = r["WaymarkId"]
    tune_patients = {patient.get(i) for i in ids(tune_ids)} - {None}
    test_set = set(ids(test_ids))
    rows = [json.loads(line) for line in per_message.open()]
    kept = [r for r in rows if r["id"] in test_set and patient.get(r["id"]) not in tune_patients]
    pos = [r for r in kept if r["safety_positive"]]
    caught = sum(r["flags"]["frozen"] for r in pos)
    e = [r for r in kept if r["call_911"]]
    tune_list, test_list = ids(tune_ids), ids(test_ids)
    all_patients = {patient.get(i) for i in tune_list + test_list} - {None}
    test_patients = {patient.get(i) for i in test_list} - {None}
    return {"n_messages_total": len(set(tune_list) | set(test_list)), "n_messages_tune": len(tune_list),
            "n_patients_total": len(all_patients), "n_patients_unlinked_messages": sum(patient.get(i) is None for i in tune_list + test_list),
            "n_patients_both_halves": len(tune_patients & test_patients),
            "n_messages": len(kept), "n_excluded": len(rows) - len(kept), "positives": len(pos), "caught": caught,
            "sensitivity": caught / len(pos), "wilson95": wilson(caught, len(pos)),
            "call_911": len(e), "call_911_caught": sum(r["flags"]["frozen"] for r in e)}


def _race(v: str) -> str:
    v = " ".join(str(v).split())
    return {"Caucasian": "White", "African American": "Black", "Hispanic": "Hispanic"}.get(
        v, "Unknown" if v in ("None", "Unknown", "nan", "") else "Other")


def _age(birth, ref: str = "2026-01-01") -> str:
    import pandas as pd

    if pd.isna(birth):
        return "Unknown"
    a = (pd.Timestamp(ref) - pd.Timestamp(birth)).days / 365.25
    return "under 35" if a < 35 else ("35 to 49" if a < 50 else ("50 to 64" if a < 65 else "65 or older"))


def patient_groups(per_message: Path, test_ids: Path, notes: Path, members: Path) -> dict:
    """Aggregate screen performance by patient group on the sealed half; no row-level output."""
    import pandas as pd

    patient = {}
    with notes.open(newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): (v or "").strip() for k, v in r.items()}
            patient[r["Encounter ID"]] = r["WaymarkId"]
    ma = pd.read_parquet(members, columns=["waymark_patient_number", "gender", "ethnicity", "birth_date"])
    ma["waymark_patient_number"] = ma["waymark_patient_number"].astype(str).str.strip()
    ma = ma.drop_duplicates("waymark_patient_number").set_index("waymark_patient_number")
    test = json.loads(test_ids.read_text())
    test = set(test if isinstance(test, list) else test["ids"])
    rows = [json.loads(line) for line in per_message.open()]
    rows = [r for r in rows if r["id"] in test]
    out: dict = {}
    for dim in ("sex", "race_ethnicity", "age_group"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            m = ma.loc[patient[r["id"]]] if patient.get(r["id"]) in ma.index else None
            if m is None:
                g = "Unknown"
            elif dim == "sex":
                g = {"F": "Female", "M": "Male"}.get(str(m["gender"]).upper(), "Unknown")
            elif dim == "race_ethnicity":
                g = _race(m["ethnicity"])
            else:
                g = _age(m["birth_date"])
            groups[g].append(r)
        res = {}
        for g, rs in sorted(groups.items()):
            neg = [r for r in rs if not r["safety_positive"]]
            pos = [r for r in rs if r["safety_positive"]]
            tn = sum(not r["flags"]["frozen"] for r in neg)
            res[g] = {"n": len(rs), "flags_per_1000": 1000 * sum(r["flags"]["frozen"] for r in rs) / len(rs),
                      "specificity": tn / len(neg) if neg else None, "specificity_wilson95": wilson(tn, len(neg)) if neg else None,
                      "positives": len(pos), "caught": sum(r["flags"]["frozen"] for r in pos)}
        out[dim] = res
    return out


def patient_clustered(per_message: Path, test_ids: Path, notes: Path) -> dict:
    """Hazard sensitivity and specificity with a bootstrap that resamples patients, not messages."""
    patient = {}
    with notes.open(newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): (v or "").strip() for k, v in r.items()}
            patient[r["Encounter ID"]] = r["WaymarkId"]
    test = json.loads(test_ids.read_text())
    test = set(test if isinstance(test, list) else test["ids"])
    rows = [json.loads(line) for line in per_message.open()]
    by_pt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["id"] in test:
            by_pt[patient.get(r["id"], r["id"])].append(r)
    pts = sorted(by_pt)
    pos_pts = {p for p in pts if any(r["safety_positive"] for r in by_pt[p])}

    def stats(sample):
        rs = [r for p in sample for r in by_pt[p]]
        pos = [r for r in rs if r["safety_positive"]]
        neg = [r for r in rs if not r["safety_positive"]]
        return (sum(r["flags"]["frozen"] for r in pos) / len(pos) if pos else float("nan"),
                sum(not r["flags"]["frozen"] for r in neg) / len(neg))

    rng = random.Random(SEED)
    boots = [stats([pts[rng.randrange(len(pts))] for _ in pts]) for _ in range(N_BOOT)]
    sens = sorted(b[0] for b in boots if b[0] == b[0])
    spec = sorted(b[1] for b in boots)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
    s0, sp0 = stats(pts)
    return {"n_patients": len(pts), "patients_with_hazard": len(pos_pts), "sensitivity": s0,
            "sensitivity_ci95": [q(sens, 0.025), q(sens, 0.975)], "specificity": sp0, "specificity_ci95": [q(spec, 0.025), q(spec, 0.975)]}


def physician_preferences(packet: Path, ratings: list[Path]) -> dict:
    """Overall preference of each physician between the COMPASS reply and the LLM-alone reply."""
    key = json.loads((packet / "physician_key.json").read_text())["items"]
    out = []
    for path in ratings:
        c = {"compass": 0, "llm_alone": 0, "tie": 0}
        with path.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["response"] == "A" and r["criterion_index"] == "0":
                    v = r["preferred"].strip()
                    c["tie" if v == "tie" else ("compass" if key[r["prompt_id"]][v] == FINAL else "llm_alone")] += 1
        out.append(c)
    return {"raters": out, "total": {k: sum(o[k] for o in out) for k in ("compass", "llm_alone", "tie")}}


def tuning_operating_point(scores: Path, labels: Path) -> dict:
    """The frozen hazard rule on the tuning half, from the screen scores recorded when the rule was chosen
    (scripts/run_hazard_dev.py: Kev cumulative probabilities and the LLM screen tier) and the physician labels.
    A message is flagged when the Kev cumulative probability of routine follow-up or higher, or of urgent or higher,
    reaches its frozen threshold, or the LLM screen assigns routine follow-up or higher."""
    from compass_pcp.harness.frozen import frozen_thresholds

    rule = frozen_thresholds()["hazard_rule"]
    read = lambda f: {(r := json.loads(line))["id"]: r for line in f.open()}
    kev, llm = read(scores / "per_case_kev.jsonl"), read(scores / "per_case_anthropic_claude-sonnet-5-5.jsonl")
    lab = read(labels)
    assert set(kev) == set(llm) == set(lab)
    flag = {i: kev[i]["score_any"] >= rule["kev_routine_cumulative"] or kev[i]["score_urgent"] >= rule["kev_urgent_cumulative"]
            or llm[i]["score_any"] >= 1 for i in lab}
    pos = [i for i in lab if lab[i]["safety_positive"]]
    neg = [i for i in lab if not lab[i]["safety_positive"]]
    e = [i for i in lab if lab[i]["call_911"]]
    caught = sum(flag[i] for i in pos)
    return {"n": len(lab), "positives": len(pos), "caught": caught, "wilson95": wilson(caught, len(pos)),
            "call_911": len(e), "call_911_caught": sum(flag[i] for i in e),
            "specificity": sum(not flag[i] for i in neg) / len(neg), "flags_per_1000": 1000 * sum(flag.values()) / len(lab)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-run", type=Path, required=True)
    ap.add_argument("--physician-packet", type=Path, required=True)
    ap.add_argument("--ratings", type=Path, nargs="+", required=True)
    ap.add_argument("--per-message", type=Path, required=True)
    ap.add_argument("--tune-ids", type=Path, required=True)
    ap.add_argument("--test-ids", type=Path, required=True)
    ap.add_argument("--notes", type=Path, required=True)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--tune-scores", type=Path)
    ap.add_argument("--tune-labels", type=Path)
    a = ap.parse_args()
    extra = {}
    if a.tune_scores and a.tune_labels:
        extra["tuning_operating_point"] = tuning_operating_point(a.tune_scores, a.tune_labels)
    print(json.dumps({"post_hoc_analyses": list(POST_HOC), "repeats": repeats(a.hb_run), "by_theme": by_theme(a.hb_run),
                      "equivalence": equivalence(a.hb_run),
                      "reply_length": reply_length(a.hb_run), "length_adjusted": length_adjusted(a.hb_run),
                      "physician_agreement": physician_agreement(a.physician_packet, a.ratings),
                      "physician_blinding": physician_blinding(a.hb_run, a.physician_packet, a.ratings),
                      "patient_overlap": patient_overlap(a.per_message, a.tune_ids, a.test_ids, a.notes),
                      "patient_groups": patient_groups(a.per_message, a.test_ids, a.notes, a.members),
                      "patient_clustered": patient_clustered(a.per_message, a.test_ids, a.notes),
                      "physician_preferences": physician_preferences(a.physician_packet, a.ratings), **extra}, indent=1))


if __name__ == "__main__":
    main()
