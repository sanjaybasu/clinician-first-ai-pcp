"""Re-derive every registered manuscript number from the canonical result files and check that the manuscript text
contains it as written. Aggregates only. Exit 1 on any mismatch.

Usage: python verify_canonical_paper2.py (needs the custodian result files; see README)
Writes CANONICAL_NUMBERS.md next to this file.
"""

import json
import re
import sys
from pathlib import Path

D = Path.home() / "waymark-local/packaging/compass-pcp-data"
DER = D / "paper2_final/derived"
MS = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/manuscript/compass_clinician_first_ai_pcp_npj_draft.md"


def j(p):
    return json.loads(Path(p).read_text())


def cmp_file(p):
    p = Path(p)
    if not p.exists():
        return {}
    txt, out, dec, i = p.read_text(), {}, json.JSONDecoder(), 0
    while i < len(txt):
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            break
        o, i = dec.raw_decode(txt, i)
        out.update(o)
    return out


f1 = lambda x: f"{x:.1f}"  # noqa: E731
pc = lambda x: f"{100 * x:.1f}"  # noqa: E731


def registry():
    R = []  # (key, value_string, source)
    rep = j(D / "hbpro_test/report.json")
    for arm in ["compass_pro_full", "amie_style", "mai_dxo_style", "opus55", "gpt61sol", "physician"]:
        la = rep[arm]["length_adjusted"]
        R.append((f"hb_test.{arm}.score", f1(la["score"]), "hbpro_test/report.json"))
        R.append((f"hb_test.{arm}.se", f"({f1(la['se'])})", "hbpro_test/report.json"))
        R.append((f"hb_test.{arm}.chars", f"{rep[arm]['mean_chars']:,}", "hbpro_test/report.json"))
        for k in ["use_case=consult", "use_case=research", "use_case=writing", "type=good_faith", "type=red_teaming"]:
            R.append((f"hb_test.{arm}.{k}", f1(rep[arm][k]["score"]), "hbpro_test/report.json"))
    c = cmp_file(D / "hbpro_test/compare.txt")
    for arm in ["amie_style", "mai_dxo_style"]:
        d = c[f"compass_pro_full - {arm}"]
        R.append((f"hb_test.diff.{arm}", f"{f1(d['diff'])} points", "hbpro_test/compare.txt"))
        R.append((f"hb_test.diff.{arm}.ci", f"{f1(d['ci95'][0])} to {f1(d['ci95'][1])}", "hbpro_test/compare.txt"))
    R.append(("hb_test.diff.amie.abstract", f"{f1(c['compass_pro_full - amie_style']['ci95'][0])}–{f1(c['compass_pro_full - amie_style']['ci95'][1])}", "compare.txt"))
    for g in ["g61", "opus"]:
        cg = cmp_file(D / f"hbpro_test/compare_{g}.txt")
        d = cg["compass_pro_full - amie_style"]
        R.append((f"hb_test.{g}.diff.amie", f"{f1(d['diff'])} points ({f1(d['ci95'][0])} to {f1(d['ci95'][1])})", f"compare_{g}.txt"))
    rep2 = j(D / "hbpro_test/report.json")
    for arm in ["amie_style_uncapped", "mai_dxo_style_uncapped", "opus_search_edit"]:
        R.append((f"hb_test.{arm}.score", f1(rep2[arm]["length_adjusted"]["score"]), "hbpro_test/report.json"))
    cu = cmp_file(D / "hbpro_test/compare_uncapped.txt")
    for arm in ["amie_style_uncapped", "mai_dxo_style_uncapped"]:
        d = cu[f"compass_pro_full - {arm}"]
        R.append((f"hb_test.uncapped.{arm}", f"{f1(d['diff'])} points ({f1(d['ci95'][0])} to {f1(d['ci95'][1])})", "compare_uncapped.txt"))
    cs = cmp_file(D / "hbpro_test/compare_simple.txt")
    d = cs["opus_search_edit - compass_pro_full"]
    R.append(("hb_test.simple_vs_full", f"{f1(-d['diff'])} points below the full pipeline ({f1(d['ci95'][0]).replace('-', '−')} to {f1(d['ci95'][1])})", "compare_simple.txt"))
    d = cs["opus_search_edit - amie_style"]
    R.append(("hb_test.simple_vs_amie", f"{f1(d['diff'])} points above the AMIE-style agent ({f1(d['ci95'][0])} to {f1(d['ci95'][1])})", "compare_simple.txt"))
    cc = cmp_file(D / "hbpro_dev/compare_compute_175.txt")
    d = cc["compass_pro_edit - compass_pro_opusonly"]
    R.append(("dev.opusonly", f"{f1(-d['diff'])} points higher than the full pipeline ({f1(-d['ci95'][1]).replace('-', '−')} to {f1(-d['ci95'][0])})", "compare_compute_175.txt"))
    d = cc["compass_pro_edit - opus_search_edit"]
    R.append(("dev.simple", f"{f1(d['diff'])} points lower ({f1(-d['ci95'][1]).replace('-', '−')} to {f1(-d['ci95'][0])})", "compare_compute_175.txt"))
    h = j(DER / "hbpro_test_supplement.json")
    d = h["paired_raw"]["amie_style"]
    R.append(("hb_test.raw_diff.amie", f"{f1(d['diff'])}\npoints ({f1(d['ci95'][0])} to {f1(d['ci95'][1])})", "hbpro_test_supplement.json"))
    reg = h["length"]["regression"]["amie_style"]
    R.append(("hb_test.len_reg", f"{f1(reg['intercept_diff_at_equal_length'])} points ({f1(reg['intercept_ci95'][0])} to {f1(reg['intercept_ci95'][1])})", "hbpro_test_supplement.json"))
    s = h["subgroups_vs_amie_style"]
    rt = s["type=red_teaming"]["paired_length_adjusted"] if "type=red_teaming" in s else None
    if rt:
        R.append(("hb_test.redteam_diff", f"{f1(rt['diff'])}; {f1(rt['ci95'][0]).replace('-', '−')} to {f1(rt['ci95'][1])}", "hbpro_test_supplement.json"))
    cost = h["cost"]
    R.append(("cost.ours", f"USD {cost['attribution_rules']['A_file_order']['compass_pro_full']['usd_per_task']:.2f}", "hbpro_test_supplement.json"))
    R.append(("cost.amie", f"USD {cost['comparator_chains']['amie_style']['usd_per_task']:.2f}", "hbpro_test_supplement.json"))
    R.append(("cost.mai", f"USD {cost['comparator_chains']['mai_dxo_style']['usd_per_task']:.2f}", "hbpro_test_supplement.json"))
    ratio = cost['attribution_rules']['A_file_order']['compass_pro_full']['usd_per_task'] / cost['comparator_chains']['amie_style']['usd_per_task']
    R.append(("cost.ratio", f"about {ratio:.1f} times", "derived"))
    rr = j(DER / "hbpro_retest_summary.json")
    R.append(("retest.primary", f"{f1(rr['primary_run_diff_same_tasks'])} points ({f1(rr['primary_ci'][0]).replace('-', '−')} to {f1(rr['primary_ci'][1])})", "hbpro_retest_summary.json"))
    R.append(("retest.second", f"{f1(rr['second_run_diff'])} points ({f1(rr['second_ci'][0]).replace('-', '−')} to {f1(rr['second_ci'][1])})", "hbpro_retest_summary.json"))
    R.append(("retest.ours", f"{f1(rr['compass_pro_full_run1'])} and {f1(rr['compass_pro_full_run2'])}", "hbpro_retest_summary.json"))
    R.append(("retest.amie", f"{f1(rr['amie_style_run1'])} and {f1(rr['amie_style_run2'])}", "hbpro_retest_summary.json"))
    dev = j(D / "hbpro_dev/report.json")
    for arm in ["amie_style", "amie_style_g61", "mai_dxo_style", "mai_dxo_style_g61"]:
        R.append((f"hb_dev.{arm}", f1(dev[arm]["length_adjusted"]["score"]), "hbpro_dev/report.json"))
    R.append(("hb_dev.chars", f"{dev['compass_pro']['mean_chars']:,} to {dev['compass_pro_edit']['mean_chars']:,}", "hbpro_dev/report.json"))
    ab = cmp_file(D / "hbpro_dev/compare_ablations_175.txt")
    for k, lab in [("compass_pro_nosearch", "search"), ("compass_pro_single", "single"), ("compass_pro_noreview", "review")]:
        d = ab[f"compass_pro - {k}"]
        R.append((f"abl.{lab}", f"{-d['diff']:+.1f}".replace("-", "−") + (" points" if lab == "search" else "") + f" ({f1(-d['ci95'][1]).replace('-', '−')} to {f1(-d['ci95'][0])}", "compare_ablations_175.txt"))
    d = ab["compass_pro - compass_pro_edit"]
    R.append(("abl.editor", f"{f1(-d['diff'])}\npoints ({f1(-d['ci95'][1]).replace('-', '−')} to {f1(-d['ci95'][0])})", "compare_ablations_175.txt"))
    # real texts
    t = j(DER / "texts_supplement.json")
    m = t["main"]
    for arm in ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22", "gate_alone"]:
        x = m[arm]
        R.append((f"txt.{arm}.sens.table", f"{pc(x['hazard_sensitivity_weighted'])} ({pc(x['hazard_sensitivity_weighted_ci'][0])}–{pc(x['hazard_sensitivity_weighted_ci'][1])})", "texts_supplement.json"))
        R.append((f"txt.{arm}.nohaz.table", f"{f1(x['escalation_no_hazard_per_1000_weighted'])} ({f1(x['escalation_no_hazard_weighted_ci'][0])}–{f1(x['escalation_no_hazard_weighted_ci'][1])})", "texts_supplement.json"))
    for arm, lab in [("amie_style", "AMIE-style"), ("mai_dxo_style", "MAI-DxO-style")]:
        d = t["paired_diffs"][f"ours - {arm}"]
        R.append((f"txt.diff.{arm}.sens", f"{pc(d['sens_diff_weighted'])}", "texts_supplement.json"))
        R.append((f"txt.diff.{arm}.sens_ci", f"{pc(d['sens_diff_weighted_ci'][0]).replace('-', '−')} to {pc(d['sens_diff_weighted_ci'][1])}", "texts_supplement.json"))
        R.append((f"txt.diff.{arm}.nohaz", f"{f1(d['nohaz_diff_per1000_weighted']).replace('-', '−')} per 1,000", "texts_supplement.json"))
        R.append((f"txt.diff.{arm}.nohaz_ci", f"{f1(d['nohaz_diff_weighted_ci'][0]).replace('-', '−')} to {f1(d['nohaz_diff_weighted_ci'][1]).replace('-', '−')}", "texts_supplement.json"))
    ins = t["ai_emergency_instructions"]
    for arm in ["amie_style", "mai_dxo_style", "raw_llm", "v22"]:
        R.append((f"txt.instr.{arm}", f1(ins[arm]["unconditional_instruction_per_1000_replies"]), "texts_supplement.json"))
    R.append(("txt.writer_alone", pc(m["writer_alone"]["hazard_sensitivity_weighted"]), "texts_supplement.json"))
    R.append(("txt.unweighted.ours", pc(m["ours"]["hazard_sensitivity_unweighted"]), "texts_supplement.json"))
    R.append(("txt.unweighted.amie", pc(m["amie_style"]["hazard_sensitivity_unweighted"]), "texts_supplement.json"))
    gw = t["gate_writer_contributions"]
    R.append(("txt.gate_paged", f"{gw['texts_paged']} of the 949", "texts_supplement.json"))
    R.append(("txt.gate_only", pc(gw["among_safety_positive_weighted"]["gate_only"]), "texts_supplement.json"))
    R.append(("txt.writer_only", pc(gw["among_safety_positive_weighted"]["writer_only"]), "texts_supplement.json"))
    sl = t["same_safety_layer"]
    a = sl["metrics"]["amie_style+layer"]
    R.append(("layer.amie", f"{pc(a['hazard_sensitivity_weighted'])}% of hazard texts ({pc(a['hazard_sensitivity_weighted_ci'][0])} to {pc(a['hazard_sensitivity_weighted_ci'][1])})", "same_safety_layer"))
    for k in ["amie_style+layer", "mai_dxo_style+layer", "v22+layer"]:
        R.append((f"layer.{k}.nohaz", f1(sl["metrics"][k]["escalation_no_hazard_per_1000_weighted"]), "same_safety_layer"))
        R.append((f"layer.{k}.sens", pc(sl["metrics"][k]["hazard_sensitivity_weighted"]), "same_safety_layer"))
    pdl = sl["paired_diffs"]["ours - amie_style+layer"]
    R.append(("layer.diff.amie", f"{pc(-pdl['sens_diff_weighted'])} points lower ({pc(pdl['sens_diff_weighted_ci'][0]).replace('-', '−')} to {pc(pdl['sens_diff_weighted_ci'][1]).replace('-', '−')})", "same_safety_layer"))
    R.append(("layer.diff.amie.nohaz", f"{f1(-pdl['nohaz_diff_per1000_weighted'])} per 1,000\nlower ({f1(pdl['nohaz_diff_weighted_ci'][0]).replace('-', '−')} to {f1(pdl['nohaz_diff_weighted_ci'][1]).replace('-', '−')})", "same_safety_layer"))
    ag = t["second_physician"]["agreement"]["safety_positive"]
    R.append(("kappa.sp", f"κ = {ag['kappa']:.3f}", "texts_supplement.json"))
    sp = t["second_physician"]
    R.append(("200.basu", f"{pc(sp['basu_labels_on_200']['ours']['hazard_sensitivity_weighted'])}% against {pc(sp['basu_labels_on_200']['amie_style']['hazard_sensitivity_weighted'])}%", "texts_supplement.json"))
    R.append(("200.morgan", f"{pc(sp['morgan_labels_on_200']['ours']['hazard_sensitivity_weighted'])}% against {pc(sp['morgan_labels_on_200']['amie_style']['hazard_sensitivity_weighted'])}%", "texts_supplement.json"))
    rc = t["repeat_consistency"]
    R.append(("repeat", f"{pc(rc['ours']['share_all_3_agree'])}% of texts with the architecture and {pc(rc['amie_style']['share_all_3_agree'])}%", "texts_supplement.json"))
    u = j(DER / "urgency_agreement.json")["_all"]
    R.append(("urg.agree", f"{pc(u['escalation_agreement'])}% (κ = {u['escalation_kappa']:.3f})", "urgency_agreement.json"))
    R.append(("urg.rates", f"{pc(u['g61_escalation_rate'])}% of\nreplies an escalation against {pc(u['sonnet_escalation_rate'])}%", "urgency_agreement.json"))
    reg = j(D / "paper2_final/texts/report.json")
    R.append(("strict.held", f"{reg['_writer_replies_held_per_1000']} of every 1,000 writer replies", "texts/report.json"))
    R.append(("strict.nohaz", f"{reg['clinician_first']['escalation_no_hazard_per_1000_weighted']} of every 1,000 texts without\na hazard", "texts/report.json"))
    rh = t["registered_hypotheses_v22"]
    R.append(("h1a", f"{pc(rh['H1a']['majority_diff'])} points; {pc(rh['H1a']['majority_diff_ci95'][0])} to {pc(rh['H1a']['majority_diff_ci95'][1])}", "registered_hypotheses_v22"))
    R.append(("h1b", f"{pc(rh['H1b']['majority_diff'])} points; exact McNemar P = {rh['H1b']['mcnemar_exact_two_sided_p']:.3f}", "registered_hypotheses_v22"))
    e = j(DER / "texts_errors.json")
    R.append(("err.med", f"{e['missed_by_category']['ours']['med_safety']:.1f} unescalated, against {e['missed_by_category']['amie_style']['med_safety']:.1f}", "texts_errors.json"))
    R.append(("err.house", f"it left {e['missed_by_category']['ours']['environmental_housing_safety']:.1f}, against {e['missed_by_category']['amie_style']['environmental_housing_safety']:.1f}", "texts_errors.json"))
    R.append(("groups", f"{e['label_groups']['other_hazard']} texts with a hazard", "texts_errors.json"))
    # gate
    g = j(D / "paper2_dev/paging_gate_497/report.json")["outcomes"]
    em = g["emergency"]["scores"]["kev_p911"]
    R.append(("gate.auroc", f"{em['auroc']['est']:.3f} (95% CI,\n{em['auroc']['ci95'][0]:.3f} to {em['auroc']['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    R.append(("gate.miss", f"{pc(em['crc_0.05']['miss']['mean'])}% of", "paging_gate_497/report.json"))
    R.append(("gate.paged", f"{pc(em['crc_0.05']['paged_w']['mean'])}% of care-team texts", "paging_gate_497/report.json"))
    R.append(("gate.above", f"{pc(em['crc_0.05']['share_splits_miss_above_alpha'])}% of the splits", "paging_gate_497/report.json"))
    R.append(("gate.full_paged", f"{pc(em['full_sample_0.05']['paged_w'])}% of care-team texts", "paging_gate_497/report.json"))
    sa = g["safety"]["scores"]["kev_p_contact_doctor_or_911"]["auroc"]
    R.append(("gate.safety_auroc", f"{sa['est']:.3f} ({sa['ci95'][0]:.3f} to {sa['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    aw = em["auroc_weighted"]
    R.append(("gate.auroc_w", f"{aw['est']:.3f} ({aw['ci95'][0]:.3f} to {aw['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    R.append(("gate.miss_p95", f"{pc(em['crc_0.05']['miss']['p95'])}%", "paging_gate_497/report.json"))
    ii = j(DER / "intake_instructions.json")["arms"]
    R.append(("intake.instr", f"{ii['amie_style']['unconditional_instruction']} of {ii['amie_style']['messages']} (AMIE-style) and "
              f"{ii['mai_dxo_style']['unconditional_instruction']} of {ii['mai_dxo_style']['messages']}", "intake_instructions.json"))
    R.append(("intake.none", f"none in {ii['intake']['messages']}", "intake_instructions.json"))
    lk = j(DER / "texts_gate_test_and_leak.json")
    rl = lk["released_raw_llm_replies"]
    R.append(("leak.released", f"Of the {rl['n_released']:,} replies that the\narchitecture released, the classifier placed {rl['emergency_now']} ({rl['emergency_now_per_1000_released']} per 1,000)", "texts_gate_test_and_leak.json"))
    cl = lk["classifier_emergency_now_all_replies"]
    R.append(("leak.cls", f"{cl['amie_style']['per_1000']} (AMIE-style), {cl['mai_dxo_style']['per_1000']} (MAI-DxO-style)\nand {cl['raw_llm']['per_1000']} (bare writer)", "texts_gate_test_and_leak.json"))
    R.append(("hb.simple_raw", f"two-step version scored {rep2['opus_search_edit']['raw']['score']} and the full pipeline {rep2['compass_pro_full']['raw']['score']}", "hbpro_test/report.json"))
    R.append(("hb.simple_chars", f"averaged {rep2['compass_pro_full']['mean_chars']:,} characters and the two-step version's {rep2['opus_search_edit']['mean_chars']:,}", "hbpro_test/report.json"))
    R.append(("layer.abstract", f"other writers reached {pc(t['same_safety_layer']['metrics']['mai_dxo_style+layer']['hazard_sensitivity_weighted'])}–{pc(t['same_safety_layer']['metrics']['amie_style+layer']['hazard_sensitivity_weighted'])}%", "same_safety_layer"))
    gt = lk["gate_test"]["hazard_contact_doctor_or_911_gated"]
    R.append(("gate.test", f"{gt[1]} of the {gt[0]} texts labeled", "texts_gate_test_and_leak.json"))
    R.append(("gate.test_share", f"paged on {100 * lk['weighted_page_share']:.1f}% of texts and sent {100 * lk['weighted_review_share']:.1f}% to review", "texts_gate_test_and_leak.json"))
    # intake
    ir = (D / "paper2_dev/intake_eval/report.md").read_text()
    for v in ["4.71", "3.46", "3.18", "71.7%", "40.9%", "55.8%", "6.43", "7.82", "7.71", "27.3%", "19.0%", "19.4%"]:
        R.append((f"intake.{v}", v.rstrip("%"), "intake_eval/report.md" if v in ir else "MISSING IN SOURCE"))
    return R


def main() -> None:
    ms = MS.read_text()
    flat = re.sub(r"\s+", " ", ms)
    rows, bad = [], 0
    for key, val, src in registry():
        ok = re.sub(r"\s+", " ", val) in flat and src != "MISSING IN SOURCE"
        bad += not ok
        rows.append(f"| {key} | {val.replace(chr(10), ' ')} | {src} | {'ok' if ok else 'NOT FOUND'} |")
    out = ["# CANONICAL_NUMBERS (Paper 2, generated by verify_canonical_paper2.py)", "",
           "| Key | Value as it must appear | Source | In manuscript |", "|---|---|---|---|", *rows]
    (Path(__file__).parent / "CANONICAL_NUMBERS.md").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} checks, {bad} not found")
    for r in rows:
        if "NOT FOUND" in r:
            print(r)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
