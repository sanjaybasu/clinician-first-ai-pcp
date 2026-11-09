"""Build CANONICAL_NUMBERS from the frozen test outputs and render the manuscript from its template.

Usage:
  python -m paper.build_canonical --hb-run <HB test output dir> --cohort-a <cohort_a_sealed_report.json> \\
      --freeze-dir docs/regulatory/freeze --manuscript-dir <notebooks/compass-ai-pcp/manuscript> \\
      [--physician <physician_scores.json>] [--physician-packet <dir>] [--prose <prose.md>]

Every number in the manuscript comes from CANONICAL_NUMBERS.json, which this script writes; paper/verify_canonical.py
re-derives it from the same inputs. A placeholder with no value renders as [PENDING: key], so a missing result is
visible, never silently filled.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from compass_pcp.harness import frozen
from compass_pcp.harness.hb_analysis import analyze

ARMS = {"final": "ai_pcp_v3", "raw": "raw_llm", "guard": "llm_standard_guardrails", "amie": "amie_style",
        "maidxo": "mai_dxo_style"}
NAMES = {"raw": "the LLM alone", "guard": "the guardrail stack", "amie": "the AMIE-style chain", "maidxo": "the MAI-DxO-style panel"}


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    """Three decimals with sign; a nonzero value that rounds to zero keeps enough decimals to show its sign."""
    for k in (3, 4, 5, 6):
        s = f"{x:+.{k}f}"
        if s.strip("+-0.") or x == 0:
            break
    return "0.000" if x == 0 or not s.strip("+-0.") else s


def diff(d: float, lo: float, hi: float) -> str:
    return f"{signed(d)} (95% CI, {signed(lo)} to {signed(hi)})"


def prop(k: int, n: int) -> str:
    return f"{k:,} of {n:,} ({100 * k / n:.1f}%)" if n else f"{k} of {n}"


def wilson_lo(k: int, n: int, z: float = 1.959964) -> float:
    p = k / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)


def ci_prop(est: float, ci: list[float]) -> str:
    return f"{est:.3f} (95% CI, {ci[0]:.3f} to {ci[1]:.3f})"


def hb_numbers(run: Path) -> dict:
    comps = [ARMS[k] for k in ("raw", "guard", "amie", "maidxo")]
    res = analyze(run, ARMS["final"], ARMS["raw"], comps)
    out: dict = {"_hb_analysis": res}
    g1 = res["grader1"]
    out["hb_n_test"] = f"{g1['scores'][ARMS['final']]['n']:,}"
    for k, arm in ARMS.items():
        out[f"hb_score_{k}"] = f3(g1["scores"][arm]["mean"])
        cost = g1["cost_per_conversation"].get(arm)
        if cost:
            out[f"hb_calls_{k}"] = f"{sum(v['calls'] for v in cost.values()):.1f}"
            out[f"hb_tokens_{k}"] = f"{sum(v['input_tokens'] + v['output_tokens'] for v in cost.values()):,.0f}"
    for k in ("raw", "guard", "amie", "maidxo"):
        s = g1["superiority"][ARMS[k]]
        out[f"hb_diff_{k}"] = diff(s["diff"], *s["ci95"])
        out[f"hb_p_{k}"] = f"{s['p_bootstrap']:.3f}" if s["p_bootstrap"] >= 0.001 else "<0.001"
    best = max(("guard", "amie", "maidxo"), key=lambda k: g1["scores"][ARMS[k]]["mean"])
    out["hb_best_comparator_name"] = NAMES[best]
    out["hb_diff_best"] = out[f"hb_diff_{best}"]
    sig = [k for k in ("raw", "guard", "amie", "maidxo") if g1["superiority"][ARMS[k]]["holm_significant"]]
    def join(names):
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    rest = [NAMES[k] for k in ("raw", "guard", "amie", "maidxo") if k not in sig]
    out["hb_holm_sentence"] = (
        "After Holm correction, COMPASS scored higher than all four comparators." if not rest else
        "After Holm correction, COMPASS scored higher than no comparator." if not sig else
        f"After Holm correction, COMPASS scored higher than {join([NAMES[k] for k in sig])}, but not {join(rest)}.")
    hb1 = g1["HB1"]
    out["hb1_rate_final"], out["hb1_rate_raw"] = f3(hb1["rate_a"]), f3(hb1["rate_b"])
    out["hb1_diff"] = diff(hb1["diff"], *hb1["ci95"])
    out["hb1_verb"] = "was met" if hb1["noninferior"] else "was not met"
    cl = g1["by_cluster_vs_raw"]
    out["hb_n_clusters"] = str(len(cl))
    out["hb_clusters_worse"] = str(sum(v["holm_significant"] and v["b_only"] > v["a_only"] for v in cl.values()))
    out["hb_clusters_better"] = str(sum(v["holm_significant"] and v["a_only"] > v["b_only"] for v in cl.values()))
    if "grader2" in res:
        g2 = res["grader2"]
        h2 = g2["HB1"]
        out["hb1g2_diff"] = diff(h2["diff"], *h2["ci95"])
        out["hb1g2_verb"] = "was met" if h2["noninferior"] else "was not met"
        out["hb2_score_final"], out["hb2_score_raw"] = f3(g2["scores"][ARMS["final"]]["mean"]), f3(g2["scores"][ARMS["raw"]]["mean"])
        for k in ("raw", "guard", "amie", "maidxo"):
            s = g2["superiority"][ARMS[k]]
            out[f"hb2_diff_{k}"] = diff(s["diff"], *s["ci95"])
        ag = [v for v in res["grader_agreement"].values() if v["ac1"] is not None]
        out["hb_grader_ac1"] = f"{min(a['ac1'] for a in ag):.3f} to {max(a['ac1'] for a in ag):.3f} across systems"
        raw_ag = sum(a["raw_agreement"] * a["n_criteria"] for a in ag) / sum(a["n_criteria"] for a in ag)
        out["hb_grader_raw_agreement"] = f"{100 * raw_ag:.1f}%"
    return out


def cohort_a_numbers(report: dict) -> dict:
    out: dict = {"_cohort_a": report}
    s = report["screens"]
    fz = s["frozen"]
    out["ca_n"] = f"{report['n']:,}"
    out["ca_pos"], out["ca_caught"] = str(fz["sensitivity"]["positives"]), str(fz["sensitivity"]["caught"])
    out["ca_sens"] = ci_prop(fz["sensitivity"]["estimate"], fz["sensitivity"]["wilson95"])
    out["ca_911"], out["ca_911_caught"] = str(fz["call_911"]["positives"]), str(fz["call_911"]["caught"])
    out["ca_spec"] = ci_prop(fz["specificity"]["estimate"], fz["specificity"]["wilson95"])
    out["ca_ppv"] = ci_prop(fz["ppv"]["estimate"], fz["ppv"]["wilson95"])
    out["ca_flags"] = f"{fz['flags_per_1000']:.0f}"
    out["ca_missed"] = str(fz["sensitivity"]["positives"] - fz["sensitivity"]["caught"])
    nb = fz["net_benefit"]
    out["ca_nb_02"], out["ca_nb_05"], out["ca_nb_10"] = (f"{nb[k]:.4f}" for k in ("0.02", "0.05", "0.1"))
    pos = fz["sensitivity"]["positives"]
    out["ca_gate_min"] = str(next(k for k in range(pos + 1) if wilson_lo(k, pos) >= 0.80))
    e = fz["call_911"]
    out["ca_911_ci"] = f"{e['caught']} of {e['positives']} (95% CI, {wilson_lo(e['caught'], e['positives']):.3f} to 1.000)" if e["caught"] == e["positives"] else ""
    out["ca_gate_verb"] = "passed" if report["VE_A4"]["pass"] else "did not pass"
    for k, name in (("kev", "kev_only"), ("llm", "llm_only"), ("floor", "floor_union")):
        if name in s:
            out[f"ca_{k}_caught"] = str(s[name]["sensitivity"]["caught"])
            out[f"ca_{k}_flags"] = f"{s[name]['flags_per_1000']:.0f}"
    a = report["auroc_safety"]["frozen_max_tier"]
    out["ca_auroc_frozen"] = f"{a['estimate']:.3f} (95% CI, {a['bootstrap95'][0]:.3f} to {a['bootstrap95'][1]:.3f})"
    out["ca_coord_caught"], out["ca_coord_pos"] = str(report["VE_A5"]["caught"]), str(report["VE_A5"]["positives"])
    return out


def hazard_dev_numbers(summary: dict) -> dict:
    """Exploratory screen refit after the sealed test: cross-validated on all labeled messages (dev_hazard_v2.py)."""
    st = summary["stacked"]
    out: dict = {"_hazard_dev": summary, "hz_n": f"{summary['n']:,}", "hz_pos": str(summary["positives"])}
    for k, name in (("kev", "kev"), ("stack", "kev+tax")):
        r = st[name]["0.9"]
        out[f"hz_auroc_{k}"] = f"{r['auroc_oof']:.3f}"
        out[f"hz_sens_{k}"], out[f"hz_spec_{k}"] = f"{r['sensitivity']:.3f}", f"{r['specificity']:.3f}"
        out[f"hz_flags_{k}"] = f"{r['flags_per_1000']:.0f}"
    return out


def physician_numbers(p: dict) -> dict:
    d = p["paired_diff_ai_pcp_minus_raw"]
    return {"_physician": p, "phys_diff": f"{d[0]:+.3f} (95% CI, {d[1]:+.3f} to {d[2]:+.3f})",
            "physician_results_sentence": (
                f"In the physician-scored sample of {p['n_items']} conversations, scored by {p['n_raters']} physicians, "
                f"COMPASS scored {p['physician_score']['ai_pcp_v3']:.3f} and the LLM alone "
                f"{p['physician_score']['raw_llm']:.3f} (difference {diff(*d)}). The two physicians agreed with each other "
                f"with Gwet AC1 {p['agreement_between_physicians_ac1']:.3f}. Counting a criterion as met when both physicians "
                f"marked it met, the reference grader agreed with the physicians on "
                f"{100 * p['grader1_vs_physician_majority_raw_agreement']:.1f}% of criterion judgments (Gwet AC1 "
                f"{p['grader1_vs_physician_majority_ac1']:.3f}) and the second grader on "
                f"{100 * p['grader2_vs_physician_majority_raw_agreement']:.1f}% (Gwet AC1 {p['grader2_vs_physician_majority_ac1']:.3f}).")}


def static_numbers(freeze_dir: Path) -> dict:
    data = frozen.DATA_DIR
    import yaml

    n_rules = n_lab = 0
    for f in ("refill.yaml", "titration.yaml", "contraception.yaml"):
        d = yaml.safe_load((data / "rules" / f).read_text())
        n_rules += len(d.get("rules", []))
        n_lab += len(d.get("lab_recency", []))
    out = {"n_scripts": str(len([p for p in (data / "scripts").glob("*.yaml") if p.stem != "TEMPLATE"])),
           "n_rules": str(n_rules), "n_lab_rules": str(n_lab), "hb_n_dev": "743", "hb_n_total": "3,671", "hb_n_nonconsensus": "1,329", "code_url": "https://github.com/sanjaybasu/compass-ai-pcp", "irb_number": "tracking ID 20253751"}
    osf = freeze_dir / "osf.json"
    if osf.exists():
        o = json.loads(osf.read_text())
        if o.get("url"):
            out["osf_doi"] = o.get("doi") or o["url"]
    return out


CARE_TEAM = re.compile(r"care team|nurse", re.IGNORECASE)


def blinding_numbers(packet: Path) -> dict:
    """Replies in the physician sample that mention a care team or nurse, by system (a cue that can break blinding)."""
    import csv

    key = json.loads((packet / "physician_key.json").read_text())["items"]
    hits = {"ai_pcp_v3": [0, 0], "raw_llm": [0, 0]}
    with (packet / "physician_rating_sheet.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["response_text"].strip():
                h = hits[key[r["prompt_id"]][r["response"]]]
                h[1] += 1
                h[0] += bool(CARE_TEAM.search(r["response_text"]))
    return {"phys_careteam_final": f"{hits['ai_pcp_v3'][0]} of {hits['ai_pcp_v3'][1]}",
            "phys_careteam_raw": f"{hits['raw_llm'][0]} of {hits['raw_llm'][1]}"}


CATEGORY_WORDS = {"behavioral_suicidality": "suicidality", "substance_use": "substance use"}


def tier_numbers(per_message: Path) -> dict:
    """Highest hazard tier (0 none, 1 routine, 2 urgent same day, 3 possible emergency, 4 emergency now) for emergency
    messages and safety hazards in the sealed half."""
    if not per_message.exists():
        return {}
    rows = [json.loads(line) for line in per_message.open()]
    top = lambda r: max(r["kev_tier"], r["llm_tier"])
    e = [top(r) for r in rows if r["call_911"]]
    s = [top(r) for r in rows if r["safety_positive"]]
    cats: dict[str, int] = {}
    for r in rows:
        if r["call_911"]:
            c = CATEGORY_WORDS.get(r["category"], str(r["category"]).replace("_", " "))
            cats[c] = cats.get(c, 0) + 1
    parts = [f"{k} ({v})" for k, v in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"ca_911_urgent": str(sum(x >= 2 for x in e)), "ca_911_possible": str(sum(x >= 3 for x in e)),
            "ca_911_now": str(sum(x >= 4 for x in e)), "ca_safety_urgent": str(sum(x >= 2 for x in s)),
            "ca_911_categories": parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]}


NUMBER_WORDS = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def sensitivity_numbers(s: dict) -> dict:
    """Post hoc sensitivity analyses (paper/sensitivity.py; Supplementary Section S14)."""
    out: dict = {"_sensitivity": s}
    r1, r2 = s["repeats"]["grader1"], s["repeats"]["grader2"]
    rng = lambda r: f"{min(x[0] for x in r['single_reply_diffs']):+.3f} to {max(x[0] for x in r['single_reply_diffs']):+.3f}"
    out["sens_rep_g1"], out["sens_rep_g2"] = rng(r1), rng(r2)
    out["sens_rep_sd_final"] = f"{r1['mean_within_sd']['ai_pcp_v3']:.3f}"
    out["sens_rep_sd_raw"] = f"{r1['mean_within_sd']['raw_llm']:.3f}"
    t1, t2 = s["by_theme"]["grader1"], s["by_theme"]["grader2"]
    for th in ("context_seeking", "complex_responses", "hedging"):
        out[f"sens_theme_{th}_g1"], out[f"sens_theme_{th}_g2"] = f"{t1[th]['diff']:+.3f}", f"{t2[th]['diff']:+.3f}"
    for g in ("grader1", "grader2"):
        e = s["equivalence"][g]
        out[f"sens_eq_{g}"] = f"{e['diff']:+.3f} (90% CI, {e['ci90'][0]:+.3f} to {e['ci90'][1]:+.3f})".replace("+0.000", "0.000").replace("-0.000", "0.000")
    both = all(s["equivalence"][g]["equivalent"] for g in ("grader1", "grader2"))
    out["sens_eq_verb"] = "fell within the margin under both graders" if both else "did not fall within the margin under both graders"
    pb = s["physician_blinding"]
    out["sens_phys_n"], out["sens_phys_excl"] = str(pb["n_items"]), str(pb["n_excluded"])
    out["sens_phys_diff"] = diff(*pb["diff"])
    po = s["patient_overlap"]
    out["sens_ovl_excl"], out["sens_ovl_n"] = str(po["n_excluded"]), str(po["n_messages"])
    out["sens_ovl_sens"] = f"{po['caught']} of {po['positives']} ({po['sensitivity']:.3f}; 95% CI, {po['wilson95'][0]:.3f} to {po['wilson95'][1]:.3f})"
    out["sens_ovl_911"] = f"{po['call_911_caught']} of {po['call_911']}"
    pc = s.get("patient_clustered")
    if pc:
        out["clust_hazard_patients"] = str(pc["patients_with_hazard"])
        out["clust_sens"] = f"{pc['sensitivity']:.3f} (95% CI, {pc['sensitivity_ci95'][0]:.3f} to {pc['sensitivity_ci95'][1]:.3f})"
    pp = s.get("physician_preferences")
    if pp:
        tot = pp["total"]; n = sum(tot.values())
        out["phys_pref"] = (f"{tot['compass']} of {n} judgments preferred the COMPASS reply, {tot['llm_alone']} preferred the LLM "
                            f"alone, and {tot['tie']} were ties")
    out["ovl_patients_both"] = f"{po['n_patients_both_halves']:,}"
    if "n_patients_total" in po:
        out["ca_n_patients"], out["ca_n_total"] = f"{po['n_patients_total']:,}", f"{po['n_messages_total']:,}"
        out["ca_n_tune"] = f"{po['n_messages_tune']:,}"
    if "post_hoc_analyses" in s:
        out["n_post_hoc"] = NUMBER_WORDS[len(s["post_hoc_analyses"])]
    rl = s.get("reply_length")
    if rl:
        for k, arm in ARMS.items():
            out[f"hb_words_{k}"], out[f"hb_words_median_{k}"] = f"{rl[arm]['mean']:.0f}", f"{rl[arm]['median']:.0f}"
    la = s.get("length_adjusted")
    if la:
        for g, gk in (("grader1", "g1"), ("grader2", "g2")):
            for k in ("raw", "guard", "amie", "maidxo"):
                v = la[g][ARMS[k]]
                out[f"sens_len_{k}_{gk}"] = diff(v["intercept"], *v["intercept_ci95"])
    pa = s.get("physician_agreement")
    if pa:
        out["phys_kappa"] = f"{pa['cohen_kappa']:.3f}"
        out["phys_ac1_judgments"] = f"{pa['n_judgments']:,}"
    tp = s.get("tuning_operating_point")
    if tp:
        out["tune_n"], out["tune_pos"], out["tune_caught"] = f"{tp['n']:,}", str(tp["positives"]), str(tp["caught"])
        out["tune_911"], out["tune_911_caught"] = str(tp["call_911"]), str(tp["call_911_caught"])
        out["tune_sens_lo"] = f"{tp['wilson95'][0]:.3f}"
        out["tune_spec"], out["tune_flags"] = f"{tp['specificity']:.3f}", f"{tp['flags_per_1000']:.0f}"
    pg = s.get("patient_groups")
    if pg:
        known = [v for dim in pg.values() for g, v in dim.items() if v["n"] >= 30]
        out["sens_grp_flag_range"] = f"{min(v['flags_per_1000'] for v in known):.0f} to {max(v['flags_per_1000'] for v in known):.0f}"
        out["sens_grp_spec_range"] = f"{min(v['specificity'] for v in known):.3f} to {max(v['specificity'] for v in known):.3f}"
        sx = pg["sex"]
        out["sens_grp_female"] = f"{sx['Female']['caught']} of {sx['Female']['positives']}"
        out["sens_grp_male"] = f"{sx['Male']['caught']} of {sx['Male']['positives']}"
    return out


def path_numbers(paths: Path, run: Path | None = None) -> dict:
    """Post hoc router re-run (paper/route_paths.py): share of test conversations on each reply path, and the COMPASS
    minus LLM-alone score difference within each path under each grader."""
    rows = [json.loads(line) for line in paths.open()]
    n = len(rows)
    out = {"path_n": f"{n:,}"}
    for k in ("engine", "clinician", "general"):
        c = sum(r["path"] == k for r in rows)
        out[f"path_{k}"] = f"{c:,} of {n:,} ({100 * c / n:.1f}%)"
    if run is not None and (run / "ai_pcp_v3.json").exists():
        from compass_pcp.harness.hb_analysis import load
        from paper.sensitivity import _boot_diff

        path = {r["prompt_id"]: r["path"] for r in rows}
        for g, suffix in (("g1", ""), ("g2", "__grader2")):
            a, b = load(run / f"ai_pcp_v3{suffix}.json")["score"], load(run / f"raw_llm{suffix}.json")["score"]
            for k in ("engine", "clinician", "general"):
                ids = [p for p in a if p in b and path.get(p) == k]
                out[f"path_{k}_{g}"] = diff(*_boot_diff([a[p] - b[p] for p in ids], 0.025, 0.975))
    return out

ABLATIONS = {"engine_off": "eoff", "minus_stopping_rule": "stop", "fixed_question_order": "order",
             "rule_out_authority": "ruleout", "eig_questions": "eig", "single_extractor": "single"}
ENGINE_DECISIONS = ("ask", "defer", "complete")
COMPONENT_NAMES = {"stop": "the stopping rule", "order": "the question order", "ruleout": "the rule-out rule",
                   "eig": "the question policy", "single": "the second extractor"}
ABLATION_LABELS = {"eoff": "No engine decision", "stop": "No stopping rule", "order": "Fixed question order",
                   "ruleout": "Rule-out on probability alone", "eig": "Questions by information gain",
                   "single": "One extractor"}


def _slot_scores(abl: Path, arm: str, suffix: str, replies: dict) -> dict:
    """Score of each (prompt_id, repeat) slot: simple-evals reports per-example scores without the repeat, so each
    example is matched to a slot of that conversation with the same reply text."""
    free = defaultdict(list)
    for (a, pid, rep), text in replies.items():
        if a == arm:
            free[(pid, text)].append(rep)
    out = {}
    for e in json.loads((abl / f"{arm}__{suffix}.json").read_text())["example_level"]:
        text = e["completion"][0]["content"] if isinstance(e["completion"], list) else str(e["completion"])
        rep = free[(e["prompt_id"], text)].pop(0)
        out[(e["prompt_id"], rep)] = max(0.0, min(1.0, e["score"]))
    return out


def _clustered_diff(a: dict, b: dict, slots: list, raw: bool = False):
    """Mean slot difference a - b, with a bootstrap over conversations (slots averaged within each conversation)."""
    from paper.sensitivity import _boot_diff

    by = defaultdict(list)
    for s in slots:
        by[s[0]].append(a[s] - b[s])
    r = _boot_diff([sum(v) / len(v) for v in by.values()], 0.025, 0.975)
    return r if raw else diff(*r)


def ablation_numbers(abl: Path, run: Path | None = None) -> dict:
    """Paired ablations (paper/ablation_staged.py, amendment AMD-PCP-001): recorded reply paths of the re-run frozen
    system, its score next to the registered run, and the score change when one component is removed."""
    replies, paths = {}, {}
    for line in (abl / "replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        replies[(r["arm"], r["prompt_id"], r["repeat"])] = r["text"]
        paths[(r["arm"], r["prompt_id"], r["repeat"])] = r["path"]
    base = {(p, rep): v for (a, p, rep), v in paths.items() if a == "ai_pcp_v3"}
    n = len(base)
    engine = [s for s, v in base.items() if v["decision"] in ENGINE_DECISIONS
              or (v["decision"] == "escalate" and "escalate" in (v["decisions"] or []) and v["hazard_tier"] != "EMERGENCY_NOW")]
    clinician = [s for s, v in base.items() if v["user_type"] == "health_professional"]
    out = {"abl_n_replies": f"{n:,}", "abl_n_conversations": f"{len({p for p, _ in base}):,}",
           "abl_engine_decided": prop(len(engine), n), "abl_clinician": prop(len(clinician), n),
           "abl_engine_contributed": prop(sum(bool(v["decisions"]) for v in base.values()), n)}
    kinds = Counter(base[s]["decision"] for s in engine)
    decided = set(engine)
    out["abl_eoff_other_changed"] = f"{sum(replies[('engine_off', p, rep)] != replies[('ai_pcp_v3', p, rep)] for p, rep in base if (p, rep) not in decided):,}"
    out["abl_engine_conversations"] = f"{len({p for p, _ in engine}):,}"
    for kind in ("ask", "defer", "complete", "escalate"):
        out[f"abl_engine_{kind}"] = f"{kinds[kind]:,}"
    for g, suffix in (("g1", "grader"), ("g2", "grader2")):
        v3 = _slot_scores(abl, "ai_pcp_v3", suffix, replies)
        out[f"abl_rerun_{g}"] = f3(sum(v3.values()) / len(v3))
        if run is not None:
            from compass_pcp.harness.hb_analysis import load

            reg = load(run / f"ai_pcp_v3{'' if g == 'g1' else '__grader2'}.json")["score"]
            pids = sorted({p for p, _ in v3} & set(reg))
            conv = defaultdict(list)
            for (p, _), x in v3.items():
                conv[p].append(x)
            re_mean = sum(sum(conv[p]) / len(conv[p]) for p in pids) / len(pids)
            reg_mean = sum(reg[p] for p in pids) / len(pids)
            out[f"abl_registered_{g}"] = f3(reg_mean)
            out[f"abl_rerun_minus_registered_{g}"] = f"{re_mean - reg_mean:+.3f}"
        for arm, k in ABLATIONS.items():
            other = _slot_scores(abl, arm, suffix, replies)
            r = _clustered_diff(v3, other, engine, raw=True)
            out[f"abl_{k}_engine_{g}"] = diff(*r)
            if k == "eoff":
                out[f"abl_eoff_engine_{g}_hi"] = f"{r[2]:+.4f}"
            out[f"abl_{k}_all_{g}"] = _clustered_diff(v3, other, list(base))
    n_changed = {}
    for arm, k in ABLATIONS.items():
        n_changed[k] = sum(replies[(arm, p, rep)] != replies[("ai_pcp_v3", p, rep)] for p, rep in base)
        out[f"abl_{k}_changed"] = prop(n_changed[k], n)
    comps = [k for k in ABLATIONS.values() if k != "eoff"]
    lo, hi = min(n_changed[k] for k in comps), max(n_changed[k] for k in comps)

    def excludes_zero(k):  # the 95% CI on engine-decided replies excludes zero under both graders, with one sign
        signs = set()
        for g in ("g1", "g2"):
            d, a, b = (float(x) for x in re.findall(r"[+-]?\d+\.\d+", out[f"abl_{k}_engine_{g}"]))
            if a <= 0 <= b:
                return False
            signs.add(d > 0)
        return len(signs) == 1

    shifted = [COMPONENT_NAMES[k] for k in comps if excludes_zero(k)]
    out["abl_component_sentence"] = (
        f"Changing one engine component changed between {lo:,} and {hi:,} of {n:,} replies, and "
        + ("no single change shifted the score on engine-decided replies under both graders." if not shifted else
           f"changing {', '.join(shifted[:-1]) + ' and ' + shifted[-1] if len(shifted) > 1 else shifted[0]} shifted the "
           "score on engine-decided replies under both graders."))
    out["abl_table_rows"] = "\n".join(
        f"| {ABLATION_LABELS[k]} | {out[f'abl_{k}_changed']} | {out[f'abl_{k}_engine_g1']} | {out[f'abl_{k}_engine_g2']} "
        f"| {out[f'abl_{k}_all_g1']} | {out[f'abl_{k}_all_g2']} |" for k in ABLATIONS.values())
    return out


def render(template: str, numbers: dict, prose: dict, base: Path | None = None) -> str:
    """Fill {{key}} from the canonical numbers and prose, and {{include:relative/path.md}} from generated files."""
    def include(m):
        f = (base or Path(".")) / m.group(1)
        return f.read_text().strip() if f.exists() else f"[PENDING: {m.group(1)}]"

    template = re.sub(r"\{\{include:([^}]+)\}\}", include, template)
    vals = {**{k: v for k, v in numbers.items() if not k.startswith("_")}, **prose}
    fill = lambda s: re.sub(r"\{\{(\w+)\}\}", lambda m: vals.get(m.group(1), f"[PENDING: {m.group(1)}]"), s)
    return fill(fill(template))  # second pass fills numbers that sit inside prose blocks


def read_prose(path: Path | None) -> dict:
    """Prose blocks written after the results, as '## key' sections in a markdown file."""
    if not path or not path.exists():
        return {}
    out, key, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            if key:
                out[key] = "\n".join(buf).strip()
            key, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if key:
        out[key] = "\n".join(buf).strip()
    return out


def word_count(md: str, start: str, end: str) -> int:
    body = md.split(start, 1)[1].split(end, 1)[0]
    body = re.sub(r"\[@[^\]]+\]", "", body)
    return len(re.findall(r"\b[\w'.,%+-]+\b", re.sub(r"^#.*$", "", body, flags=re.MULTILINE)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-run", type=Path)
    ap.add_argument("--cohort-a", type=Path)
    ap.add_argument("--physician", type=Path)
    ap.add_argument("--hazard-dev", type=Path)
    ap.add_argument("--physician-packet", type=Path)
    ap.add_argument("--sensitivity", type=Path)
    ap.add_argument("--paths", type=Path)
    ap.add_argument("--ablation", type=Path)
    ap.add_argument("--freeze-dir", type=Path, required=True)
    ap.add_argument("--manuscript-dir", type=Path, required=True)
    ap.add_argument("--prose", type=Path)
    args = ap.parse_args()
    numbers = static_numbers(args.freeze_dir)
    if args.hb_run and (args.hb_run / "ai_pcp_v3.json").exists():
        numbers.update(hb_numbers(args.hb_run))
    if args.cohort_a and args.cohort_a.exists():
        numbers.update(cohort_a_numbers(json.loads(args.cohort_a.read_text())))
        numbers.update(tier_numbers(args.cohort_a.parent / "per_message.jsonl"))
    if args.physician and args.physician.exists():
        numbers.update(physician_numbers(json.loads(args.physician.read_text())))
    if args.physician_packet and (args.physician_packet / "physician_key.json").exists():
        numbers.update(blinding_numbers(args.physician_packet))
    if args.paths and args.paths.exists():
        numbers.update(path_numbers(args.paths, args.hb_run))
    if args.ablation and (args.ablation / "ai_pcp_v3__grader2.json").exists():
        numbers.update(ablation_numbers(args.ablation, args.hb_run))
    if args.sensitivity and args.sensitivity.exists():
        numbers.update(sensitivity_numbers(json.loads(args.sensitivity.read_text())))
    if args.hazard_dev and args.hazard_dev.exists():
        numbers.update(hazard_dev_numbers(json.loads(args.hazard_dev.read_text())))
    md_dir = args.manuscript_dir
    (md_dir / "CANONICAL_NUMBERS.json").write_text(json.dumps(numbers, indent=2, default=str))
    flat = {k: v for k, v in numbers.items() if not k.startswith("_")}
    (md_dir / "CANONICAL_NUMBERS.md").write_text(
        "# Canonical numbers\n\nGenerated by paper/build_canonical.py; do not edit by hand.\n\n| Key | Value |\n| :-- | :-- |\n"
        + "".join(f"| {k} | {v} |\n" for k, v in sorted(flat.items())))
    prose = read_prose(args.prose or md_dir / "compass_ai_pcp_jamia_prose.md")
    text = render((md_dir / "compass_ai_pcp_jamia_template.md").read_text(), numbers, prose, md_dir)
    wc_main = word_count(text, "# Background and significance", "# Acknowledgments")
    wc_abs = word_count(text, "# Abstract", "# Background and significance")
    text = text.replace("[PENDING: word_count_main]", f"{wc_main:,}").replace("[PENDING: word_count_abstract]", str(wc_abs))
    (md_dir / "compass_ai_pcp_jamia_manuscript.md").write_text(text)
    supp_t = md_dir / "compass_ai_pcp_jamia_supplement_template.md"
    if supp_t.exists():
        supp = render(supp_t.read_text(), numbers, prose, md_dir)
        (md_dir / "compass_ai_pcp_jamia_supplement.md").write_text(supp)
        text += supp
    pending = sorted(set(re.findall(r"\[PENDING: (\w+)\]", text)))
    print(json.dumps({"words_main": wc_main, "words_abstract": wc_abs, "pending": pending}, indent=1))


if __name__ == "__main__":
    main()
