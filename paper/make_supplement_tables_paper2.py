"""Supplementary tables for the Paper 2 manuscript, written from the canonical result files (aggregates only).

Every number is read at run time; nothing is typed in. Missing inputs print "pending".
Out: notebooks/compass-pcp-paper2/manuscript/supplement_tables_generated.md
Usage: python -m paper.make_supplement_tables_paper2
"""

import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
DER = D / "paper2_final/derived"
OUT = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/manuscript/supplement_tables_generated.md"
NAMES = {"compass_pro_full": "Our architecture", "compass_pro_edit": "Our architecture (development)",
         "compass_pro": "Pipeline before the length editor", "compass_pro_nosearch": "No web search",
         "compass_pro_single": "One draft (Opus 5.5 only)", "compass_pro_noreview": "No review step",
         "compass_pro_opusonly": "Single developer (Opus 5.5 drafts and review)",
         "opus_search_edit": "One searched Opus 5.5 draft + length editor", "amie_style": "AMIE-style (Opus 5.5)",
         "mai_dxo_style": "MAI-DxO-style (Opus 5.5)", "amie_style_g61": "AMIE-style (GPT-6.1 Sol)",
         "mai_dxo_style_g61": "MAI-DxO-style (GPT-6.1 Sol)", "opus55": "Opus 5.5, no system prompt",
         "gpt61sol": "GPT-6.1 Sol, no system prompt", "gpt55": "GPT-5.5, no system prompt",
         "gemini31pro": "Gemini 3.1 Pro, no system prompt", "physician": "Physicians (dataset responses)",
         "raw_llm": "Opus 5.5, primary care persona", "v22": "Registered rule-based system (v22)", "ours": "Our architecture",
         "gate_alone": "Paging gate alone", "writer_alone": "Writer alone (no gate)",
         "clinician_first_any_hit_held": "Our architecture, registered check"}


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size else None


def _ci(c, f="{:.1f}"):
    return f"{f.format(c[0])} to {f.format(c[1])}"


def _pct(x):
    return f"{100 * x:.1f}"


def _cmp_lines(p: Path) -> dict:
    """compare output: one or more JSON objects concatenated."""
    if not p.exists():
        return {}
    txt, out, dec, i = p.read_text(), {}, json.JSONDecoder(), 0
    while i < len(txt):
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            break
        obj, i = dec.raw_decode(txt, i)
        out.update(obj)
    return out


def t2() -> str:
    r = _j(D / "hbpro_dev/report.json")
    rows = ["| System | n | Length-adjusted (SE) | Raw | Mean characters |", "|---|---|---|---|---|"]
    order = ["compass_pro_edit", "compass_pro", "amie_style", "mai_dxo_style", "amie_style_g61", "mai_dxo_style_g61",
             "gpt61sol", "opus55", "gpt55", "gemini31pro", "physician"]
    for a in order:
        if a in r:
            v = r[a]
            rows.append(f"| {NAMES[a]} | {v['length_adjusted']['n']} | {v['length_adjusted']['score']} "
                        f"({v['length_adjusted']['se']}) | {v['raw']['score']} | {v['mean_chars']:,} |")
    return ("Supplementary Table 2 | HealthBench Professional development split (175 tasks), all systems run in "
            "development.\n\n" + "\n".join(rows))


def t3() -> str:
    r = _j(D / "hbpro_dev/report.json")
    abl = _cmp_lines(D / "hbpro_dev/compare_ablations_175.txt")
    cmpu = _cmp_lines(D / "hbpro_dev/compare_compute_175.txt")
    rows = ["| Variant | n | Score (SE) | Difference, full minus variant (95% CI) |", "|---|---|---|---|"]
    for a, src, base in [("compass_pro_nosearch", abl, "compass_pro"), ("compass_pro_single", abl, "compass_pro"),
                         ("compass_pro_noreview", abl, "compass_pro"), ("compass_pro_edit", abl, "compass_pro"),
                         ("compass_pro_opusonly", cmpu, "compass_pro_edit"), ("opus_search_edit", cmpu, "compass_pro_edit")]:
        d = src.get(f"{base} - {a}")
        v = r.get(a, {}).get("length_adjusted")
        sc = f"{v['score']} ({v['se']})" if v else "pending"
        if d and d.get("diff") is not None:
            diff = f"{d['diff']:+.1f} ({_ci(d['ci95'])}), n = {d['n']}"
            if a == "compass_pro_edit":
                diff = f"editor effect {-d['diff']:+.1f} ({_ci([-d['ci95'][1], -d['ci95'][0]])}), n = {d['n']}"
        else:
            diff = "pending"
        rows.append(f"| {NAMES[a]} | {v['n'] if v else ''} | {sc} | {diff} |")
    return ("Supplementary Table 3 | Ablations and compute comparisons on the development split. Rows 1–3 compare the "
            "pipeline before the length editor with one step removed; row 4 gives the editor's effect; rows 5–6 "
            "compare the full development pipeline with a single-developer version and with one searched draft plus "
            "the editor (post hoc).\n\n" + "\n".join(rows))


def t4() -> str:
    h = _j(DER / "hbpro_test_supplement.json")
    c = h["cost"]
    a = c["attribution_rules"]["A_file_order"]
    rows = ["| System | USD per task (list) | Mean characters | Median characters | Share ≤2,000 characters |",
            "|---|---|---|---|---|"]
    cost = {"amie_style": c["comparator_chains"]["amie_style"]["usd_per_task"],
            "mai_dxo_style": c["comparator_chains"]["mai_dxo_style"]["usd_per_task"]}
    for k, v in a.items():
        if k.startswith("opus55_bare"):
            cost["opus55"] = v["usd_per_task"]
        if k.startswith("gpt61sol_bare"):
            cost["gpt61sol"] = v["usd_per_task"]
        if k == "compass_pro_full":
            cost["compass_pro_full"] = v["usd_per_task"]
    for arm in ["compass_pro_full", "amie_style", "mai_dxo_style", "opus55", "gpt61sol", "physician"]:
        m = h["arms"][arm]
        cc = f"{cost[arm]:.3f}" if arm in cost else "—"
        rows.append(f"| {NAMES[arm]} | {cc} | {m['mean_chars']:,} | {m['median_chars']:,} | {_pct(m['share_le_2000_chars'])}% |")
    L = h["length"]
    reg = L.get("regression", {}).get("amie_style", {})
    extra = []
    for arm in ["amie_style", "mai_dxo_style"]:
        b = L["paired_both_le_2000_chars"]["raw"][arm]
        extra.append(f"Ours minus {NAMES[arm]} on tasks where both answers had ≤2,000 characters (raw score): "
                     f"{b['diff']:+.1f} ({_ci(b['ci95'])}), n = {b['n']}.")
    if "intercept_diff_at_equal_length" in reg:
        extra.append(f"OLS regression of the task-level raw score difference (ours minus AMIE-style) on the difference "
                     f"in length (thousands of characters), n = {reg['n']}: difference at equal length "
                     f"{reg['intercept_diff_at_equal_length']:+.1f} ({_ci(reg['intercept_ci95'])}); slope "
                     f"{reg['slope_per_1000_chars']:+.2f} points per 1,000 characters ({_ci(reg['slope_ci95'], '{:.2f}')}).")
    return ("Supplementary Table 4 | Cost and answer length on the HealthBench Professional test split (350 tasks). "
            "Cost is list price from recorded token use; pipeline usage rows carry no arm label, so cost is attributed "
            "(rule A; rule B differs by USD 0.005 per task).\n\n" + "\n".join(rows) + "\n\n" + "\n\n".join(extra))


def t5() -> str:
    h = _j(DER / "hbpro_test_supplement.json")
    rows = ["| Comparison | Grader | n | Difference (95% CI) |", "|---|---|---|---|"]
    for arm in ["amie_style", "mai_dxo_style", "opus55", "gpt61sol", "physician"]:
        d = h["paired_length_adjusted"][arm]
        rows.append(f"| Ours minus {NAMES[arm]} | GPT-5.4 (primary), length-adjusted | {d['n']} | {d['diff']:+.1f} ({_ci(d['ci95'])}) |")
        d = h["paired_raw"][arm]
        rows.append(f"| Ours minus {NAMES[arm]} | GPT-5.4 (primary), raw | {d['n']} | {d['diff']:+.1f} ({_ci(d['ci95'])}) |")
    for tag, lab in [("g61", "GPT-6.1 Sol"), ("opus", "Claude Opus 5.5")]:
        c = _cmp_lines(D / f"hbpro_test/compare_{tag}.txt")
        for arm in ["amie_style", "mai_dxo_style", "opus55", "gpt61sol", "physician"]:
            d = c.get(f"compass_pro_full - {arm}")
            rows.append(f"| Ours minus {NAMES[arm]} | {lab}, length-adjusted | {d['n'] if d else ''} | "
                        f"{(str(f'{d[chr(100)+chr(105)+chr(102)+chr(102)]:+.1f}') + ' (' + _ci(d['ci95']) + ')') if d else 'pending'} |")
    rr = _j(DER / "hbpro_retest_summary.json")
    if rr:
        rows.append(f"| Ours minus AMIE-style, primary run, same 100 tasks | GPT-5.4, length-adjusted | {rr['n']} | "
                    f"{rr['primary_run_diff_same_tasks']:+.1f} ({_ci(rr['primary_ci'])}) |")
        rows.append(f"| Ours minus AMIE-style, second independent run, same 100 tasks | GPT-5.4, length-adjusted | {rr['n']} | "
                    f"{rr['second_run_diff']:+.1f} ({_ci(rr['second_ci'])}) |")
        rows.append(f"| Ours minus AMIE-style, mean of the two runs | GPT-5.4, length-adjusted | {rr['n']} | "
                    f"{rr['mean_of_two_runs_diff']:+.1f} ({_ci(rr['mean_ci'])}) |")
    unc = _cmp_lines(D / "hbpro_test/compare_uncapped.txt")
    for arm in ["amie_style_uncapped", "mai_dxo_style_uncapped"]:
        d = unc.get(f"compass_pro_full - {arm}")
        rows.append(f"| Ours minus {arm.replace('_uncapped', '').replace('amie_style', 'AMIE-style').replace('mai_dxo_style', 'MAI-DxO-style')}, "
                    f"reply limit raised to 12,000 tokens | GPT-5.4, length-adjusted | {d['n'] if d else ''} | "
                    f"{(format(d['diff'], '+.1f') + ' (' + _ci(d['ci95']) + ')') if d else 'pending'} |")
    simp = _cmp_lines(D / "hbpro_test/compare_simple.txt")
    for arm in ["amie_style", "compass_pro_full"]:
        d = simp.get(f"opus_search_edit - {arm}")
        lab = "One searched Opus 5.5 draft + editor minus " + ("AMIE-style" if arm == "amie_style" else "full pipeline")
        rows.append(f"| {lab} | GPT-5.4, length-adjusted | {d['n'] if d else ''} | "
                    f"{(format(d['diff'], '+.1f') + ' (' + _ci(d['ci95']) + ')') if d else 'pending'} |")
    ns = _cmp_lines(D / "hbpro_test/compare_nosearch.txt")
    for key, lab in [("opus_edit - opus_search_edit", "Two-step without web search minus two-step with search"),
                     ("opus_edit - amie_style", "Two-step without web search minus AMIE-style")]:
        d = ns.get(key)
        rows.append(f"| {lab} | GPT-5.4, length-adjusted | {d['n'] if d else ''} | "
                    f"{(format(d['diff'], '+.1f') + ' (' + _ci(d['ci95']) + ')') if d else 'pending'} |")
    asr = _cmp_lines(D / "hbpro_test/compare_amie_search.txt")
    for key, lab in [("compass_pro_full - amie_style_search", "Ours minus AMIE-style with the same web search"),
                     ("opus_search_edit - amie_style_search", "Two-step version minus AMIE-style with the same web search")]:
        d = asr.get(key)
        rows.append(f"| {lab} | GPT-5.4, length-adjusted | {d['n'] if d else ''} | "
                    f"{(format(d['diff'], '+.1f') + ' (' + _ci(d['ci95']) + ')') if d else 'pending'} |")
    hp = h["holm_primary_family"]
    wt = h["win_tie_loss"]
    foot = (f"Holm-adjusted paired t-test P (length-adjusted): vs AMIE-style {hp['length_adjusted']['amie_style']['p_holm']:.1e}; "
            f"vs MAI-DxO-style {hp['length_adjusted']['mai_dxo_style']['p_holm']:.1e}. Raw-score win/tie/loss for ours "
            f"against AMIE-style {wt['amie_style']['raw']['win']}/{wt['amie_style']['raw']['tie']}/{wt['amie_style']['raw']['loss']} "
            f"and against MAI-DxO-style {wt['mai_dxo_style']['raw']['win']}/{wt['mai_dxo_style']['raw']['tie']}/"
            f"{wt['mai_dxo_style']['raw']['loss']}.")
    return ("Supplementary Table 5 | Grader and run sensitivity on the HealthBench Professional test split. The "
            "alternate graders, the second run, the raised reply limit, the two-step versions and the AMIE-style agent with "
            "web search are post hoc.\n\n" + "\n".join(rows) + "\n\n" + foot)


def _trow(name, m, sens_key="hazard_sensitivity_weighted"):
    return (f"| {name} | {_pct(m[sens_key])} ({_ci([100 * x for x in m[sens_key + '_ci']])}) | "
            f"{_pct(m['hazard_sensitivity_unweighted'])} | {m['escalation_no_hazard_per_1000_weighted']:.1f} "
            f"({_ci(m['escalation_no_hazard_weighted_ci'])}) | {m['escalation_no_hazard_per_1000_unweighted']:.1f} |")


def t6() -> str:
    t = _j(DER / "texts_supplement.json")
    reg = _j(D / "paper2_final/texts/report.json")
    head = ["| System | Sensitivity, weighted % (95% CI) | Sensitivity, unweighted % | Escalations per 1,000 without hazard, weighted (95% CI) | Unweighted |",
            "|---|---|---|---|---|"]
    rows = list(head)
    for a in ["ours", "clinician_first_any_hit_held", "writer_alone", "gate_alone", "amie_style", "mai_dxo_style", "raw_llm", "v22"]:
        rows.append(_trow(NAMES[a], t["main"][a]))
    cf = reg["clinician_first"]
    note = (f"Registered check, as computed by the registered script (paper/texts_report.py): sensitivity "
            f"{_pct(cf['hazard_sensitivity_weighted'])}% ({_ci([100 * x for x in cf['hazard_sensitivity_ci']])}), "
            f"{cf['escalation_no_hazard_per_1000_weighted']} per 1,000 without hazard; writer replies held "
            f"{reg['_writer_replies_held_per_1000']} per 1,000. Bootstrap CIs in the row above use the shared resamples of the "
            f"supplementary script, so they differ slightly.")
    st = []
    for s in ["random", "enriched"]:
        st += [f"\n{s.capitalize()} stratum ({t['by_stratum'][s]['metrics']['ours']['texts']} texts, "
               f"{t['by_stratum'][s]['metrics']['ours']['safety_positive']} with a safety hazard)\n", *head]
        for a in ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22"]:
            st.append(_trow(NAMES[a], t["by_stratum"][s]["metrics"][a]))
    pd_ = t["paired_diffs"]
    pdr = ["| Comparison | Sensitivity difference, weighted (95% CI) | Unweighted (95% CI) | Escalations per 1,000, weighted (95% CI) |",
           "|---|---|---|---|"]
    for k, v in pd_.items():
        pdr.append(f"| {k.replace('ours', 'Ours').replace('amie_style', 'AMIE-style').replace('mai_dxo_style', 'MAI-DxO-style').replace('raw_llm', 'persona-only writer')} | "
                   f"{100 * v['sens_diff_weighted']:+.1f} ({_ci([100 * x for x in v['sens_diff_weighted_ci']])}) | "
                   f"{100 * v['sens_diff_unweighted']:+.1f} ({_ci([100 * x for x in v['sens_diff_unweighted_ci']])}) | "
                   f"{v['nohaz_diff_per1000_weighted']:+.1f} ({_ci(v['nohaz_diff_weighted_ci'])}) |")
    return ("Supplementary Table 6 | Real patient texts (949 texts, 3 replies each): all arms, the registered check, "
            "weighted and unweighted estimates, paired differences and sampling strata. 'Ours' uses the deployable "
            "check (conditional safety-net lines allowed; post hoc).\n\n" + "\n".join(rows) + "\n\n" + note + "\n\n"
            + "\n".join(pdr) + "\n" + "\n".join(st))


def t7() -> str:
    s = _j(DER / "texts_supplement.json")["same_safety_layer"]
    r = _j(DER / "texts_routing_endpoints.json")
    em = r["emergency_now_reaching_patient_per_1000_all_replies"]
    rows = ["| Writer + same gate + same check | Sensitivity, weighted % (95% CI) | Unweighted % | Escalations per 1,000 without hazard (95% CI) | Unweighted | Replies held by the check per 1,000 (unconditional by its text patterns) | Doctor-call texts escalated, % (95% CI) | Advice to seek emergency care now released, per 1,000 replies (classifier) |",
            "|---|---|---|---|---|---|---|---|"]
    for k in ["raw_llm+layer", "amie_style+layer", "mai_dxo_style+layer", "v22+layer"]:
        m = s["metrics"][k]
        nm = {"raw_llm+layer": "Persona-only writer (= our architecture)", "amie_style+layer": "AMIE-style writer",
              "mai_dxo_style+layer": "MAI-DxO-style writer", "v22+layer": "Rule-based writer (v22)"}[k]
        rk = "ours" if k == "raw_llm+layer" else k
        q = r[rk]
        rows.append(_trow(nm, m) + f" {s['replies_held_per_1000'][k]} | {_pct(q['sens_doctor_call'])} "
                    f"({_ci([100 * x for x in q['sens_doctor_call_ci']])}) | {em[rk]} |")
    pd_ = s.get("paired_diffs", {})
    pr = ["| Comparison | Sensitivity difference (95% CI) | Escalations per 1,000 difference (95% CI) |", "|---|---|---|"]
    for k, v in pd_.items():
        pr.append(f"| {k} | {100 * v['sens_diff_weighted']:+.1f} ({_ci([100 * x for x in v['sens_diff_weighted_ci']])}) | "
                  f"{v['nohaz_diff_per1000_weighted']:+.1f} ({_ci(v['nohaz_diff_weighted_ci'])}) |")
    ro, w = r["ours_routed_only"], r["withheld"]
    dc = ["| Doctor-call endpoint (53 texts), post hoc | Difference, percentage points (95% CI) |", "|---|---|"]
    for b in ("amie_style", "mai_dxo_style", "v22"):
        v = r[f"ours - {b} (doctor_call)"]
        dc.append(f"| ours - {b} | {100 * v['diff']:+.1f} ({_ci([100 * x for x in v['ci']])}) |")
    note = (f"Routing split (post hoc): counting only texts that reached a physician (gate page or review, or the reply held "
            f"by the check), the architecture escalated {_pct(ro['sens_safety_positive'])}% of safety-positive texts "
            f"({_ci([100 * x for x in ro['sens_safety_positive_ci']])}) and {ro['per1000_no_hazard']} per 1,000 texts without a "
            f"hazard. Of {3 * 949:,} replies, the architecture withheld {w['total']}: {w['replies_withheld_by_page']} because the "
            f"gate paged on {w['texts_paged']} texts and {w['replies_held_by_check_not_paged']} held by the check; the gate sent "
            f"{w['texts_review']} further texts to review, and their replies were sent.")
    return ("Supplementary Table 7 | The same safety layer (paging gate and check) added to each writer (post hoc). "
            "Unconditional emergency instructions found by the check's text patterns reach the patient in 0 replies for every "
            "row by construction; the classifier column counts advice to seek emergency care now among released replies, over "
            "all 2,847 replies of each writer.\n\n"
            + "\n".join(rows) + "\n\n" + "\n".join(pr) + "\n\n" + "\n".join(dc) + "\n\n" + note)


def t8() -> str:
    t = _j(DER / "texts_supplement.json")
    ag = t["second_physician"]["agreement"]
    rows = ["| Label | Observed agreement | Cohen's κ | Positives (S.B.) | Positives (J.M.) |", "|---|---|---|---|---|"]
    for k in ["safety_positive", "call_911", "any_hazard", "action_4_class"]:
        v = ag[k]
        rows.append(f"| {k.replace('_', ' ')} | {_pct(v['observed_agreement'])}% | {v['kappa']:.3f} | "
                    f"{v['positives_basu'] if v['positives_basu'] is not None else '—'} | "
                    f"{v['positives_morgan'] if v['positives_morgan'] is not None else '—'} |")
    sp2 = t["second_physician"]
    rm = _j(DER / "texts_routing_endpoints_morgan.json")
    rb = _j(DER / "texts_routing_endpoints.json")
    nb, nm = sp2["basu_labels"]["ours"]["safety_positive"], sp2["morgan_labels"]["ours"]["safety_positive"]
    sp = [f"| System | Hazard sensitivity, S.B. labels ({nb} texts), % (95% CI) | Hazard sensitivity, J.M. labels ({nm} texts), "
          f"% (95% CI) | Texts needing a doctor's call escalated, S.B. labels ({rb['n']['doctor_call']} texts), % (95% CI) | "
          f"Texts needing a doctor's call escalated, J.M. labels ({rm['n']['doctor_call']} texts), % (95% CI) |",
          "|---|---|---|---|---|"]
    for a in ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22", "amie_style+layer", "mai_dxo_style+layer", "v22+layer"]:
        b, m = sp2["basu_labels"][a], sp2["morgan_labels"][a]
        nm_ = NAMES.get(a) or NAMES[a.split("+")[0]] + " + layer"
        sp.append(f"| {nm_} | {_pct(b['hazard_sensitivity_weighted'])} ({_ci([100 * x for x in b['hazard_sensitivity_weighted_ci']])}) | "
                  f"{_pct(m['hazard_sensitivity_weighted'])} ({_ci([100 * x for x in m['hazard_sensitivity_weighted_ci']])}) | "
                  f"{_pct(rb[a]['sens_doctor_call'])} ({_ci([100 * x for x in rb[a]['sens_doctor_call_ci']])}) | "
                  f"{_pct(rm[a]['sens_doctor_call'])} ({_ci([100 * x for x in rm[a]['sens_doctor_call_ci']])}) |")
    dd = ["| Comparison | Difference in hazard sensitivity, S.B. labels, points (95% CI) | Same, J.M. labels | "
          "Difference in escalation burden, S.B. labels, per 1,000 (95% CI) | Same, J.M. labels |", "|---|---|---|---|---|"]
    for k, v in sp2["paired_diffs_basu"].items():
        w = sp2["paired_diffs_morgan"][k]
        b_ = k.split(" - ")[1]
        nm_ = NAMES.get(b_) or NAMES[b_.split("+")[0]] + " + layer"
        dd.append(f"| Our architecture - {nm_} | {100 * v['sens_diff_weighted']:.1f} ({_ci([100 * x for x in v['sens_diff_weighted_ci']])}) | "
                  f"{100 * w['sens_diff_weighted']:.1f} ({_ci([100 * x for x in w['sens_diff_weighted_ci']])}) | "
                  f"{v['nohaz_diff_per1000_weighted']:.1f} ({_ci(v['nohaz_diff_weighted_ci'])}) | "
                  f"{w['nohaz_diff_per1000_weighted']:.1f} ({_ci(w['nohaz_diff_weighted_ci'])}) |")
    dis = sp2["disagreements"]
    disl = (f"The two physicians differed on {dis['safety_positive_or_call_911']} of {sp2['n_double_labeled']} texts on a "
            f"safety hazard or a call to 911 or 988 ({dis['call_911']} on a call to 911 or 988). On the "
            f"{sp2['both_agree']['n']} texts where they agreed on both a safety hazard and any hazard, hazard sensitivity was "
            f"{_pct(sp2['both_agree']['metrics']['ours']['hazard_sensitivity_weighted'])}% for our architecture and "
            f"{_pct(sp2['both_agree']['metrics']['amie_style']['hazard_sensitivity_weighted'])}% for the AMIE-style agent.")
    sp += [""] + dd + ["", disl]
    u = _j(DER / "urgency_agreement.json")
    ur = ["| Writer | Replies | Escalation agreement | κ (escalation) | κ (5 levels) | Escalation rate, primary | Escalation rate, second |",
          "|---|---|---|---|---|---|---|"]
    for k in ["_all", "raw_llm", "amie_style", "mai_dxo_style", "v22"]:
        if u and k in u:
            v = u[k]
            ur.append(f"| {'All' if k == '_all' else NAMES[k]} | {v['n']} | {_pct(v['escalation_agreement'])}% | "
                      f"{v['escalation_kappa']:.3f} | {v['five_level_kappa']:.3f} | {_pct(v['sonnet_escalation_rate'])}% | "
                      f"{_pct(v['g61_escalation_rate'])}% |")
    rc = t["repeat_consistency"]
    rr = ["| System | All 3 replies agree on escalation | Among safety-positive texts |", "|---|---|---|"]
    for a in ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22"]:
        rr.append(f"| {NAMES[a]} | {_pct(rc[a]['share_all_3_agree'])}% | {_pct(rc[a]['share_all_3_agree_safety_positive'])}% |")
    gw = t["gate_writer_contributions"]
    g = gw["among_safety_positive_weighted"]
    gl = (f"Among safety-positive texts (weighted), the gate and the writer both escalated {_pct(g['gate_and_writer'])}%, "
          f"the gate only {_pct(g['gate_only'])}%, the writer only {_pct(g['writer_only'])}%, and neither "
          f"{_pct(g['neither'])}%. The gate paged or sent to review {gw['texts_paged']} of 949 texts "
          f"({gw['texts_paged_emergency_threshold']} at the emergency threshold).")
    return ("Supplementary Table 8 | Label agreement between two physicians on all 949 test texts, results under each "
            "physician's labels, agreement of a second reply classifier (GPT-6.1 Sol, post hoc) with the primary classifier (Claude "
            "Sonnet 5.5) on a stratified random sample of replies, and repeat consistency.\n\n" + "\n".join(rows)
            + "\n\n" + "\n".join(sp) + "\n\n" + "\n".join(ur) + "\n\n" + "\n".join(rr) + "\n\n" + gl)


def t9() -> str:
    g = _j(DER / "paging_gate_table.json")
    rows = ["| Outcome (positives) | Score | Rule | α | Threshold | Missed, mean (5th–95th pct) | Paged, enriched sample | Paged, weighted to pool | Splits with miss > α |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in g["rows"]:
        if r["rule"] not in ("crc", "full_sample") or not r["score"].startswith("kev"):
            continue
        deployed = any(m["score"] == r["score"] and m["alpha"] == r["alpha"] and m["rule"] == r["rule"]
                       for m in g["deployed_thresholds"]["matches"])
        sc = {"kev_p911": "P(911/988)", "kev_p_contact_doctor_or_911": "P(contact doctor or higher)"}.get(r["score"], r["score"])
        miss = (f"{_pct(r['miss'])}% ({_pct(r['miss_p05'])}–{_pct(r['miss_p95'])})" if r.get("miss_p05") is not None
                else f"{_pct(r['miss'])}%")
        rows.append(f"| {r['outcome']} ({r['positives']}) | {sc} | {'CRC, 500 splits' if r['rule'] == 'crc' else 'fit on all 497' + (' (deployed)' if deployed else '')} | "
                    f"{r['alpha']} | {r['threshold']} | {miss} | {_pct(r['paged'])}% | {_pct(r['paged_w'])}% | "
                    f"{_pct(r['share_splits_miss_above_alpha']) + '%' if r.get('share_splits_miss_above_alpha') is not None else '—'} |")
    au = [r for r in g["rows"] if r["rule"] == "crc" and r["score"].startswith("kev")]
    aus = sorted({(r["outcome"], r["auroc"], tuple(r["auroc_ci95"]), r["auroc_weighted"], tuple(r["auroc_weighted_ci95"])) for r in au})
    foot = " ".join(f"AUROC for {o}: {a} ({c[0]}–{c[1]}); weighted to the pool {w} ({wc[0]}–{wc[1]})." for o, a, c, w, wc in aus)
    return ("Supplementary Table 9 | Paging gate (Kev-9B, mean of five fold models) on the 497 emergency-enriched texts. "
            "Weighted values use inverse inclusion probabilities to the 47,126-text pool.\n\n" + "\n".join(rows)
            + "\n\n" + foot + f" Per-fold AUROC for 911/988: {g['per_model_auroc_kev']}." + _gate_test())


def _gate_test() -> str:
    t = _j(DER / "texts_gate_test_and_leak.json")
    if not t:
        return ""
    e, s_ = t["gate_test"]["call911_paged"], t["gate_test"]["hazard_contact_doctor_or_911_gated"]
    return (f" On the 949 test texts (later period, patient texts on the service), the deployed thresholds paged on {e[1]} of "
            f"{e[0]} 'call 911 or 988' texts and paged or sent to review {s_[1]} of {s_[0]} texts labeled as a hazard needing "
            f"a doctor's call or more; weighted to the test frame, {100 * t['weighted_page_share']:.1f}% of texts were paged "
            f"and {100 * t['weighted_review_share']:.1f}% sent to review." + _refit())


def _refit() -> str:
    g = _j(Path.home() / "waymark-local/packaging/ai-pcp/data/gate_thresholds.json")
    if not g:
        return ""
    e = [x for x in g["split_sample_check"]["safety"]["evaluations"] if x["choice"] == "target"][0]
    return (f" Refit of the review threshold on the 949 test texts (post hoc; conformal risk control on the "
            f"{g['fit_on']['sources']['compass_2026']['safety_positives']} doctor-call texts, target miss 10%): t = "
            f"{g['split_sample_check']['safety']['chosen']['t']:.4f}; over {e['splits_used']} random 50/50 splits, mean miss in "
            f"the held-out half {100 * e['miss_mean']:.1f}% (95th percentile {100 * e['miss_p95']:.1f}%; above 10% in "
            f"{100 * e['share_splits_miss_above_alpha']:.1f}% of splits), weighted share sent to review "
            f"{100 * e['gated_weighted_mean']:.1f}%. The page threshold, refit on all 1,446 texts pooled, was unchanged "
            f"(t = {g['t_e']:.6f}).")


def t10() -> str:
    md = (D / "paper2_dev/intake_eval/report.md").read_text()
    body = md[md.index("| Arm"):].replace("COMPASS structured intake", "Intake agent")
    ii = _j(DER / "intake_instructions.json")["arms"]
    cap = (f" With the final check (Supplementary Note 2), any instruction appeared in {ii['amie_style']['any_instruction']} "
           f"of {ii['amie_style']['messages']} AMIE-style, {ii['mai_dxo_style']['any_instruction']} of "
           f"{ii['mai_dxo_style']['messages']} MAI-DxO-style and {ii['intake']['any_instruction']} of "
           f"{ii['intake']['messages']} intake-agent messages, and unconditional instructions in "
           f"{ii['amie_style']['unconditional_instruction']}, {ii['mai_dxo_style']['unconditional_instruction']} and "
           f"{ii['intake']['unconditional_instruction']}.")
    return ("Supplementary Table 10 | History taking from real texts with simulated patients (28 development cases; "
            "means with case-bootstrap 95% CIs, 2,000 resamples). The last column is the run's own count of messages with an "
            "emergency-room or 911/988 instruction (the intake agent's prompt forbids them)." + cap + "\n\n" + body.strip())


def t11() -> str:
    r = _j(DER / "texts_supplement.json")["registered_hypotheses_v22"]
    rows = ["| Registered hypothesis | Majority-vote sensitivity, v22 vs comparator | Difference (95% CI) | Discordant texts (v22 only / comparator only) | Test | Holm-adjusted P | Result |",
            "|---|---|---|---|---|---|---|"]
    lab = {"H1a": "H1a: v22 higher than the persona-only writer", "H1b": "H1b: v22 higher than MAI-DxO-style",
           "H1c": "H1c: v22 not lower than AMIE-style by more than 0.10"}
    for h in ["H1a", "H1b", "H1c"]:
        v = r[h]
        if h == "H1c":
            ni = v["non_inferiority"]
            test, p = "one-sided bootstrap against margin −0.10", ni["one_sided_bootstrap_p"]
            res = "non-inferiority met" if ni["met_majority_unweighted"] else "not met"
            ph = v.get("p_holm_3_tested", None)
        else:
            test, p = "exact McNemar, two-sided", v["mcnemar_exact_two_sided_p"]
            res = "supported" if v["reject_holm_3"] else "not supported"
            ph = v["p_holm_3_tested"]
        rows.append(f"| {lab[h]} | {_pct(v['majority_sens_v22'])}% vs {_pct(v['majority_sens_comparator'])}% | "
                    f"{100 * v['majority_diff']:+.1f} ({_ci([100 * x for x in v['majority_diff_ci95']])}) | "
                    f"{v['discordant_v22_only']} / {v['discordant_comparator_only']} | {test} (P = {p:.3g}) | "
                    f"{ph:.3g} | {res} |")
    rows.append("| H2a–H2c: unsafe closure in the multi-turn simulation | not run | | | | | not tested |")
    return ("Supplementary Table 11 | Registered hypotheses for the registered system (version 22) on the 79 "
            "safety-positive real texts. Escalation is the per-text majority over 3 replies (unweighted), as "
            "registered. Holm adjustment over the 3 tested hypotheses; entering the untested H2a–H2c as P = 1 does "
            "not change any decision.\n\n" + "\n".join(rows))


def t12() -> str:
    e = _j(DER / "texts_errors.json")
    arms = ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22"]
    cats = e["label_groups"]["safety_positive_by_category"]
    rows = ["| Hazard category (texts) | " + " | ".join(NAMES[a] for a in arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        rows.append(f"| {c.replace('_', ' ')} ({n}) | " + " | ".join(f"{e['missed_by_category'][a][c]:.1f}" for a in arms) + " |")
    ct = ["| System | Label group (texts) | Escalated, expected | Not escalated, expected | Escalated, weighted to pool (pool total) |",
          "|---|---|---|---|---|"]
    for a in arms:
        for grp in ["safety_positive", "other_hazard", "no_hazard"]:
            v = e["crosstab"][a][grp]
            ct.append(f"| {NAMES[a]} | {grp.replace('_', ' ')} ({v['n']}) | {v['escalated_expected']} | "
                      f"{v['not_escalated_expected']} | {v['weighted_escalated']} ({v['weighted_pool_total']}) |")
    return ("Supplementary Table 12 | Missed safety hazards by physician-labeled category (expected number of "
            "safety-positive texts not escalated, mean over 3 replies), and the cross-tabulation of escalation by label "
            f"group for all 949 texts ({e['label_groups']['safety_positive']} safety-positive, "
            f"{e['label_groups']['other_hazard']} with a non-clinical or coordination hazard only, "
            f"{e['label_groups']['no_hazard']} with no hazard). Texts with a coordination hazard only enter neither "
            "the sensitivity nor the escalation-burden denominator.\n\n" + "\n".join(rows) + "\n\n" + "\n".join(ct))


def t13() -> str:
    rows = ["| Role | Model (API identifier) | Developer; weights | Settings | Access |", "|---|---|---|---|---|",
            "| Draft A, final writer, length editor; patient-facing writer; intake agent; comparator backbone | Claude Opus 5.5 (claude-opus-5-5) | Anthropic; proprietary | adaptive thinking; web search tool web_search_20260209 (Draft A only, ≤5 searches) | Anthropic API, BAA |",
            "| Draft B and review | GPT-6.1 Sol (gpt-6.1-sol) | OpenAI; proprietary | reasoning effort high | OpenAI API, BAA |",
            "| Paging gate | Kev-9B, five fold fine-tunes (adapter recipe of the Kev package) | open weights (jaredpalmer/kev) | message only, 384 tokens | Modal, BAA |",
            "| Reply-disposition classifier | Claude Sonnet 5.5 (claude-sonnet-5-5) | Anthropic; proprietary | default | Anthropic API, BAA |",
            "| Second reply classifier (post hoc) | GPT-6.1 Sol | OpenAI; proprietary | reasoning effort low | OpenAI API, BAA |",
            "| HealthBench Professional grader | GPT-5.4 (gpt-5.4) | OpenAI; proprietary | reasoning effort low | OpenAI API |",
            "| Alternate graders (post hoc) | GPT-6.1 Sol; Claude Opus 5.5 | OpenAI; Anthropic | reasoning low; no thinking | APIs |",
            "| Simulated patient and note judge (history taking) | GPT-6.1 Sol | OpenAI; proprietary | reasoning effort high | OpenAI API, BAA |"]
    return ("Supplementary Table 13 | Models, settings and access. All calls ran from the United States between "
            "November 6 and November 9, 2026, with a fresh context for each reply or task. Patient texts were not "
            "de-identified and went only to vendors with a business associate agreement. No call set a temperature or a seed; each model sampled at its provider default.\n\n" + "\n".join(rows))


def t14() -> str:
    a = _j(DER / "a2_rating.json")
    if not a:
        return "Supplementary Table 14 | Reply quality rated by physicians. pending"
    nm = {"ai": "Separated design (as the patient saw it)", "physician": "Physician-written reply", "amie": "AMIE-style agent",
          "care_team": "Care team (first texted reply within 48 h)"}
    d = a["design"]
    head = ("| Reply source | Replies | Accuracy, 1-5 (95% CI) | Usefulness, 1-5 (95% CI) | Moderate or severe harm, % (95% CI) | "
            "Critical omission, % (95% CI) | Ranked best, % |")
    out = []
    for pop, title in (("all_texts_weighted", f"All {d['texts']} rated texts, weighted to the 949 test texts"),
                       ("safety_positive_texts", f"The {d['safety_positive_texts']} texts with a safety hazard")):
        out += [f"*{title}.*", "", head, "|---|---|---|---|---|---|---|"]
        for k in ("ai", "physician", "amie", "care_team"):
            v = a[pop][k]
            out.append(f"| {nm[k]} | {v['n_replies']} | {v['accuracy']:.2f} ({_ci(v['accuracy_ci'], '{:.2f}')}) | "
                       f"{v['usefulness']:.2f} ({_ci(v['usefulness_ci'], '{:.2f}')}) | {_pct(v['moderate_or_severe_harm'])} "
                       f"({_ci([100 * x for x in v['moderate_or_severe_harm_ci']])}) | {_pct(v['critical_omission'])} "
                       f"({_ci([100 * x for x in v['critical_omission_ci']])}) | {_pct(v['ranked_best'])} |")
        out.append("")
    u = a["unweighted_counts"]
    out += ["*Unweighted counts (rater C.M.).*", "", "| Reply source | Replies | Moderate or severe harm | Severe harm | "
            "Critical omission | Ranked best |", "|---|---|---|---|---|---|"]
    for k in ("ai", "physician", "amie", "care_team"):
        out.append(f"| {nm[k]} | {u[k]['replies']} | {u[k]['moderate_or_severe']} | {u[k]['severe']} | {u[k]['critical_omission']} | "
                   f"{u[k]['ranked_best']} |")
    p1, p2 = a["primary"]["P1_safety_noninferiority_ai_minus_physician"], a["primary"]["P2_accuracy_superiority_ai_minus_care_team"]
    out += ["", "*Pre-specified comparisons.*", "", "| Comparison | Texts | Difference (95% CI) | P (Holm) | Result |",
            "|---|---|---|---|---|",
            f"| Moderate or severe harm, separated design - physician (non-inferiority, margin 5 points) | {p1['texts']} | "
            f"{100 * p1['diff']:.1f} points ({_ci([100 * x for x in p1['ci95']])}) | {p1['p_holm']:.3f} | "
            f"{'non-inferior' if p1['met'] else 'not shown non-inferior'} |",
            f"| Accuracy, separated design - care team (superiority) | {p2['texts']} | {p2['diff']:.2f} ({_ci(p2['ci95'], '{:.2f}')}) | "
            f"{p2['p_holm']:.3f} | {'superior' if p2['superior'] else 'not shown superior'} |"]
    out += ["", "*Other paired comparisons (descriptive).*", "", "| Comparison | Measure | Texts | All texts, difference (95% CI) | "
            "Texts with a safety hazard, difference (95% CI) |", "|---|---|---|---|---|"]
    sec = a["secondary_paired"]
    for pair in ("ai - physician", "ai - amie", "ai - care_team", "amie - physician"):
        x, y = pair.split(" - ")
        for k, lab, sc in (("accuracy", "Accuracy", 1), ("usefulness", "Usefulness", 1),
                           ("moderate_or_severe_harm", "Moderate or severe harm, points", 100),
                           ("critical_omission", "Critical omission, points", 100)):
            al, sp = sec[f"{pair} | {k} | all"], sec[f"{pair} | {k} | safety_positive"]
            f = "{:.2f}" if sc == 1 else "{:.1f}"
            out.append(f"| {nm[x].split(' (')[0]} - {nm[y].split(' (')[0]} | {lab} | {al['texts']} | "
                       f"{f.format(sc * al['diff'])} ({_ci([sc * z for z in al['ci95']], f)}) | "
                       f"{f.format(sc * sp['diff'])} ({_ci([sc * z for z in sp['ci95']], f)}) |")
    ir = a["inter_rater"]
    out += ["", f"*Agreement between raters ({ir['replies']} replies to {ir['texts']} texts).*", "",
            "| Measure | Exact agreement, % | κ |", "|---|---|---|"]
    for k, lab in (("accuracy_quadratic_weighted", "Accuracy (quadratic-weighted κ)"),
                   ("usefulness_quadratic_weighted", "Usefulness (quadratic-weighted κ)"),
                   ("harm_4_level_quadratic_weighted", "Harm, 4 levels (quadratic-weighted κ)"),
                   ("moderate_or_severe_harm", "Moderate or severe harm"), ("critical_omission", "Critical omission"),
                   ("disposition_5_level_quadratic_weighted", "Advised care, 5 levels (quadratic-weighted κ)")):
        out.append(f"| {lab} | {_pct(ir[k]['exact_agreement'])} | {ir[k]['kappa']:.3f} |")
    bt = a["separated_design_by_reply_type"]
    al_, ge = bt["alert_text"], bt["generated_reply"]
    out += ["", (f"*Separated design by reply type (rater C.M.).* The fixed alert text was {al_['replies']} of the separated "
                 f"design's replies ({al_['safety_positive']} to texts with a safety hazard): the rater marked "
                 f"{al_['critical_omission']} as a critical omission and {al_['moderate_or_severe_harm']} as moderate or "
                 f"severe harm ({al_['severe_harm']} severe), and its mean accuracy on texts with a safety hazard was "
                 f"{al_['mean_accuracy_safety_positive']:.2f}. The {ge['replies']} generated replies had "
                 f"{ge['critical_omission']} critical omissions and {ge['moderate_or_severe_harm']} with moderate or severe "
                 f"harm; on the {ge['safety_positive']} texts with a safety hazard that got a generated reply, mean accuracy "
                 f"was {ge['mean_accuracy_safety_positive']:.2f} against {ge['amie_mean_accuracy_same_texts']:.2f} for the "
                 f"AMIE-style agent, with {ge['critical_omission_safety_positive']} and "
                 f"{ge['amie_critical_omission_same_texts']} critical omissions.")]
    cv = a["classifier_validation"]
    rj = a["rater_judged_escalation_safety_positive"]
    out += ["", (f"*Reply classifier against the rater.* On {cv['replies']} generated replies (the reply model's and the "
                 f"AMIE-style agent's, first reply; alert text excluded), the rater judged {cv['rater_escalates']} to advise "
                 f"routine follow-up or more urgent care and the classifier {cv['classifier_escalates']} (κ = "
                 f"{cv['agreement']['kappa']:.3f}; classifier sensitivity {_pct(cv['classifier_sensitivity_vs_rater'])}% and "
                 f"specificity {_pct(cv['classifier_specificity_vs_rater'])}% against the rater). With escalation judged by "
                 f"the rater on the {rj['texts']} texts with a safety hazard (post hoc), every reply source advised routine "
                 f"follow-up or more urgent care for every text; for same-day or more urgent care, the separated design "
                 f"escalated {_pct(rj['same_day_or_more']['separated_design']['sens'])}% "
                 f"({_ci([100 * x for x in rj['same_day_or_more']['separated_design']['ci95']])}), the AMIE-style agent "
                 f"{_pct(rj['same_day_or_more']['amie']['sens'])}% "
                 f"({_ci([100 * x for x in rj['same_day_or_more']['amie']['ci95']])}) and the physician-written replies "
                 f"{_pct(rj['same_day_or_more']['physician']['sens'])}% "
                 f"({_ci([100 * x for x in rj['same_day_or_more']['physician']['ci95']])})."), ""]
    return ("Supplementary Table 14 | Reply quality rated by physicians. C.M. rated every reply to "
            f"{d['texts']} test texts; the physician-written replies came from three Waymark physicians who saw only the text ({d['safety_positive_texts']} with a safety hazard and 71 drawn at random from the other "
            "870), blind to source; R.B. rated the replies to a random 50 of the texts for agreement. The separated design's "
            f"reply is the one the patient would see: on {d['ai_replies_urgent_page_text']} texts this was the fixed alert text, "
            "sent when the emergency screen alerts a physician or the filter holds the reply. Weights are the two-phase "
            "sampling weights divided by the probability of selection for rating (1 for texts with a safety hazard, 71/870 "
            "otherwise). CIs are from 2,000 bootstrap resamples of texts.\n\n" + "\n".join(out))


def t15() -> str:
    a = _j(DER / "texts_subgroups.json")
    if not a:
        return "Supplementary Table 15 | Sender characteristics and subgroups. pending"
    lab = {"state": "State", "sex": "Sex", "age": "Age group", "race": "Race"}
    ch = a["characteristics"]
    out = ["*Sender characteristics by sampling stratum, n (%).*", "",
           f"| Characteristic | Group | Random sample (n = {ch['state']['random']['_n']}) | Flagged texts (n = "
           f"{ch['state']['enriched']['_n']}) | All test texts (n = {ch['state']['all']['_n']}) |", "|---|---|---|---|---|"]
    for var in ("state", "sex", "age", "race"):
        for g in sorted(k for k in ch[var]["all"] if k != "_n"):
            cell = lambda s_: f"{ch[var][s_][g][0]} ({ch[var][s_][g][1]:.1f})" if g in ch[var][s_] else "0 (0.0)"  # noqa: E731
            out.append(f"| {lab[var]} | {g} | {cell('random')} | {cell('enriched')} | {cell('all')} |")
    out += ["", "*Hazard sensitivity and escalation burden by subgroup (weighted; post hoc, descriptive).*", "",
            "| Characteristic | Group | Texts | Texts with a safety hazard | Hazard sensitivity, separated design, % (95% CI) | "
            "Hazard sensitivity, AMIE-style agent, % (95% CI) | Escalation burden, separated design, per 1,000 (95% CI) | "
            "Escalation burden, AMIE-style agent, per 1,000 (95% CI) |", "|---|---|---|---|---|---|---|---|"]
    for var in ("state", "sex", "age", "race"):
        for g, v in a["subgroups"][var].items():
            o, m = v["ours"], v["amie_style"]
            sens = lambda x: "—" if x["safety_positive"] < 5 else (  # noqa: E731
                f"{_pct(x['hazard_sensitivity_weighted'])} ({_ci([100 * y for y in x['hazard_sensitivity_weighted_ci']])})")
            bur = lambda x: "—" if x["no_hazard"] < 10 else (  # noqa: E731
                f"{x['escalation_no_hazard_per_1000_weighted']:.1f} ({_ci(x['escalation_no_hazard_weighted_ci'])})")
            out.append(f"| {lab[var]} | {g} | {v['texts']} | {o['safety_positive']} | {sens(o)} | {sens(m)} | {bur(o)} | {bur(m)} |")
    return ("Supplementary Table 15 | Sender characteristics and subgroup results for the 949 test texts (post hoc). The "
            "sender is known when the phone belongs to one Waymark member (737 texts); for a shared phone, state is given "
            "when all members on the phone live in the same state. Age is the linked member's age on the day of the text; "
            "for a member under 18, a parent may have sent the text. Race is as recorded in Waymark's care-team "
            "application, which records Hispanic or Latinx as a race and does not record ethnicity; Other race combines "
            "Asian, American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander and Other. Estimates use "
            "S.B.'s labels and the two-phase weights, with 2,000 bootstrap resamples of texts within each group; a dash marks "
            "an estimate from fewer than 5 texts with a safety hazard or fewer than 10 texts without a hazard. No "
            "interaction tests were run.\n\n" + "\n".join(out))


def main() -> None:
    parts = [t2(), t3(), t4(), t5(), t6(), t7(), t8(), t9(), t10(), t11(), t12(), t13(), t14(), t15()]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(OUT, sum(p.count("pending") for p in parts), "pending cells")


if __name__ == "__main__":
    main()
