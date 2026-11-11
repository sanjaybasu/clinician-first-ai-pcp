"""Re-derive every registered manuscript number from the canonical result files and check that the manuscript text
contains it as written. Aggregates only. Exit 1 on any mismatch.

Usage: python verify_canonical_paper2.py
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
        R.append((f"hb_test.{g}.diff.amie", f"{f1(d['diff'])} points; {f1(d['ci95'][0])} to {f1(d['ci95'][1])}", f"compare_{g}.txt"))
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
    ratio = cost['attribution_rules']['A_file_order']['compass_pro_full']['usd_per_task'] / cost['comparator_chains']['amie_style']['usd_per_task']
    R.append(("cost.ratio", f"about {ratio:.1f} times", "derived"))
    # second registration (https://osf.io/tnezd): stricter comparisons, independent physician, classifier check
    m = lambda x: f1(x).replace("-", "−")  # noqa: E731
    ci = lambda c: f"{m(c[0])} to {m(c[1])}"  # noqa: E731
    ad = j(DER / "addendum_hb.json")
    t, sc_, co = ad["all_tasks"], ad["scores"], ad["E_cost_per_task_usd"]
    a1, a2 = t["A1_pipeline_minus_tuned_amie"], t["A2_pipeline_minus_tuned_mai"]
    R.append(("abs.ladder", f"scored {m(a1['diff'])} points (95% CI, {ci(a1['ci95'])}) and {m(a2['diff'])} points ({ci(a2['ci95'])}) higher than\nre-implemented AMIE-style and MAI-DxO-style agents", "addendum_hb.json"))
    R.append(("strict.scores", f"the multi-model pipeline scored {f1(sc_['compass_pro_full_blk']['mean'])} (SE, {f1(sc_['compass_pro_full_blk']['se'])}), the tuned AMIE-style agent {f1(sc_['amie_style_search_blk']['mean'])} ({f1(sc_['amie_style_search_blk']['se'])}) and the tuned MAI-DxO-style agent {f1(sc_['mai_dxo_style_search_edit_blk']['mean'])} ({f1(sc_['mai_dxo_style_search_edit_blk']['se'])})", "addendum_hb.json"))
    R.append(("strict.A1", f"{m(a1['diff'])} points higher than the tuned AMIE-style agent ({ci(a1['ci95'])}; Holm-adjusted P = {a1['p_holm']:.2f})", "addendum_hb.json"))
    R.append(("strict.A2", f"{m(a2['diff'])} points higher than the tuned MAI-DxO-style agent ({ci(a2['ci95'])}; P = {a2['p_holm']:.2f})", "addendum_hb.json"))
    e = ad["E_two_step_vs_tuned_amie"]
    R.append(("strict.E", f"(USD {co['opus_search_edit_blk']:.3f} per task), scored {f1(sc_['opus_search_edit_blk']['mean'])}, {m(e['diff'])} points higher than the tuned AMIE-style agent at USD {co['amie_style_search_blk']:.3f} per task ({ci(e['ci95'])})", "addendum_hb.json"))
    x = t["A_secondary"]["two_step_minus_tuned_mai"]
    R.append(("strict.twostep_mai", f"{m(x['diff'])} points higher than the tuned MAI-DxO-style agent ({ci(x['ci95'])}), which cost more (USD {co['mai_dxo_style_search_edit_blk']:.3f}", "addendum_hb.json"))
    R.append(("strict.cost_pipe", f"USD {co['compass_pro_full_blk']:.3f} per task", "addendum_hb.json"))
    R.append(("strict.ratio", f"cost {co['compass_pro_full_blk'] / co['amie_style_search_blk']:.1f} times as much as the tuned AMIE-style agent", "addendum_hb.json"))
    b = t["B_blocked_minus_original"]
    p_, o_, s_ = b["compass_pro_full_blk - compass_pro_full"], b["opus_search_edit_blk - opus_search_edit"], b["amie_style_search_blk - amie_style_search"]
    R.append(("B.diffs", f"With blocked search, the multi-model pipeline scored {m(-p_['diff'])} points lower (95% CI, {ci(p_['ci95'])}), the two-step pipeline {m(-o_['diff'])} points lower ({ci(o_['ci95'])}) and the AMIE-style agent with search {m(-s_['diff'])} points lower ({ci(s_['ci95'])})", "addendum_hb.json"))
    R.append(("B.amie90", "the\nAMIE-style agent's 90% CI lay within it" if b["amie_style_search_blk - amie_style_search"]["within_4_points_90ci"] else "MISSING", "addendum_hb.json"))
    R.append(("B.audit", f"The blocked runs logged {ad['audit']['results_logged']:,} search results", "addendum_hb.json"))
    R.append(("disc.ladder", f"and to {m(a1['diff'])} points over the AMIE-style agent and {m(a2['diff'])} points over the\nMAI-DxO-style agent", "addendum_hb.json"))
    for k, arm in (("t1.blk.pipe", "compass_pro_full_blk"), ("t1.blk.two", "opus_search_edit_blk"), ("t1.blk.amie", "amie_style_search_blk"), ("t1.blk.mai", "mai_dxo_style_search_edit_blk")):
        x = rep2[arm]
        R.append((k, f"{f1(x['length_adjusted']['score'])} ({f1(x['length_adjusted']['se'])}) | {f1(x['raw']['score'])} | " + " | ".join(f1(x[c]['score']) for c in ['use_case=consult','use_case=research','use_case=writing','type=good_faith','type=red_teaming']) + f" | {x['mean_chars']:,} |", "hbpro_test/report.json"))
    ar = j(DER / "addendum_rater.json")
    p1, mj = ar["part1"], ar["majority_labels"]
    R.append(("av.kappa", f"(κ with them, {p1['kappa_external_basu']['kappa']:.3f} and {p1['kappa_external_morgan']['kappa']:.3f})", "addendum_rater.json"))
    o, am_ = mj["metrics"]["ours"], mj["metrics"]["amie_style"]
    d = mj["paired_diffs"]["ours - amie_style"]
    R.append(("av.majority", f"Under majority-of-three labels, {o['safety_positive']} texts had a safety hazard. The separated design escalated {pc(o['hazard_sensitivity_weighted'])}% of them ({pc(o['hazard_sensitivity_weighted_ci'][0])} to {pc(o['hazard_sensitivity_weighted_ci'][1])}) and the AMIE-style agent {pc(am_['hazard_sensitivity_weighted'])}% ({pc(am_['hazard_sensitivity_weighted_ci'][0])} to {pc(am_['hazard_sensitivity_weighted_ci'][1])}), a difference of {m(100*d['sens_diff_weighted'])} points ({ci([100*v for v in d['sens_diff_weighted_ci']])})", "addendum_rater.json"))
    lay = [-100 * mj["paired_diffs"][f"ours - {k}+layer"]["sens_diff_weighted"] for k in ("amie_style", "mai_dxo_style", "v22")]
    R.append(("av.layer", f"escalated {f1(min(lay))} to {f1(max(lay))} points more than the separated", "addendum_rater.json"))
    cv = ar["part2_classifier"]["classifier_vs_external"]
    sp = [v["routine_or_more"]["specificity"] for v in cv.values()]; se_ = [v["routine_or_more"]["sensitivity"] for v in cv.values()]; kp = [v["kappa_unweighted_sample"]["kappa"] for v in cv.values()]
    R.append(("av.clf", f"specificity of {min(sp):.2f} to {max(sp):.2f}\nand a sensitivity of {min(se_):.2f} to {max(se_):.2f} for advice of routine follow-up or more urgent care", "addendum_rater.json"))
    cr = ar["part2_classifier"]["corrected"]["routine_or_more"]
    R.append(("av.corr", f"the separated design escalated\n{pc(cr['ours']['hazard_sensitivity'])}% of texts with a safety hazard ({pc(cr['ours']['hazard_sensitivity_ci'][0])} to {pc(cr['ours']['hazard_sensitivity_ci'][1])}) and the AMIE-style agent {pc(cr['amie_style']['hazard_sensitivity'])}% ({pc(cr['amie_style']['hazard_sensitivity_ci'][0])} to {pc(cr['amie_style']['hazard_sensitivity_ci'][1])}), a difference of\n{m(100*cr['amie_style']['ours_minus_this_sensitivity'])} points ({ci([100*v for v in cr['amie_style']['ours_minus_this_sensitivity_ci']])})", "addendum_rater.json"))
    bur = [v["escalation_burden_per_1000"] for v in cr.values()]
    R.append(("av.bur", f"escalation burdens were {min(bur)} to {max(bur)} per 1,000", "addendum_rater.json"))
    rl = cr["raw_llm"]
    R.append(("av.rawlayer", f"added {f1(100*rl['ours_minus_this_sensitivity'])} points ({f1(100*rl['ours_minus_this_sensitivity_ci'][0])} to {f1(100*rl['ours_minus_this_sensitivity_ci'][1])}) to the reply model", "addendum_rater.json"))
    r3 = ar["part3_ratings"]
    R.append(("av.agree", f"accuracy\n(quadratic-weighted κ, {r3['agreement_with_moher']['accuracy_quadratic']['kappa']:.3f} and {r3['agreement_with_batniji']['accuracy_quadratic']['kappa']:.3f})", "addendum_rater.json"))
    vs = r3["external_by_source"]
    R.append(("av.alert", f"harmful in {vs['separated_alert_text']['moderate_or_severe']} of {vs['separated_alert_text']['replies']} uses, and none of the {vs['separated_generated']['replies']} generated replies, {vs['physician']['replies']}\nphysician-written replies or {vs['amie']['replies']} AMIE-style replies; care-team replies had moderate or severe harm in {vs['care_team']['moderate_or_severe']} of {vs['care_team']['replies']}", "addendum_rater.json"))
    bl = r3["blinding"]
    R.append(("av.blind", f"source of all {bl['physician']['replies']} physician-written replies", "addendum_rater.json"))
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
        R.append((f"txt.diff.{arm}.nohaz", f"{f1(-d['nohaz_diff_per1000_weighted'])} per 1,000 lower", "texts_supplement.json"))
        R.append((f"txt.diff.{arm}.nohaz_ci", f"{f1(d['nohaz_diff_weighted_ci'][0]).replace('-', '−')} to {f1(d['nohaz_diff_weighted_ci'][1]).replace('-', '−')}", "texts_supplement.json"))
    ins = t["ai_emergency_instructions"]
    for arm in ["amie_style", "mai_dxo_style", "raw_llm"]:  # v22's count is in Supplementary Table 7 only
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
    R.append(("layer.amie", f"{pc(a['hazard_sensitivity_weighted'])}% of texts with a safety hazard ({pc(a['hazard_sensitivity_weighted_ci'][0])} to {pc(a['hazard_sensitivity_weighted_ci'][1])})", "same_safety_layer"))
    for k in ["amie_style+layer", "mai_dxo_style+layer", "v22+layer"]:
        R.append((f"layer.{k}.nohaz", f1(sl["metrics"][k]["escalation_no_hazard_per_1000_weighted"]), "same_safety_layer"))
        R.append((f"layer.{k}.sens", pc(sl["metrics"][k]["hazard_sensitivity_weighted"]), "same_safety_layer"))
    pdl = sl["paired_diffs"]["ours - amie_style+layer"]
    R.append(("layer.diff.amie", f"{pc(-pdl['sens_diff_weighted'])} points lower ({pc(pdl['sens_diff_weighted_ci'][0]).replace('-', '−')} to {pc(pdl['sens_diff_weighted_ci'][1]).replace('-', '−')})", "same_safety_layer"))
    R.append(("layer.diff.amie.nohaz", f"{f1(-pdl['nohaz_diff_per1000_weighted'])} per 1,000\nlower ({f1(pdl['nohaz_diff_weighted_ci'][0]).replace('-', '−')} to {f1(pdl['nohaz_diff_weighted_ci'][1]).replace('-', '−')})", "same_safety_layer"))
    ag = t["second_physician"]["agreement"]["safety_positive"]
    R.append(("kappa.sp", f"κ = {ag['kappa']:.3f}", "texts_supplement.json"))
    sp = t["second_physician"]
    m = lambda x: x.replace("-", "−")  # noqa: E731
    sm, sb = sp["morgan_labels"], sp["paired_diffs_morgan"]
    R.append(("jm.n", f"found {sm['ours']['safety_positive']} texts with a safety hazard (Cohen's κ = {ag['kappa']:.3f})", "texts_supplement.json"))
    R.append(("jm.sens", f"the separated design escalated {pc(sm['ours']['hazard_sensitivity_weighted'])}% of texts with a safety hazard ({pc(sm['ours']['hazard_sensitivity_weighted_ci'][0])} to {pc(sm['ours']['hazard_sensitivity_weighted_ci'][1])}) and the AMIE-style agent {pc(sm['amie_style']['hazard_sensitivity_weighted'])}% ({pc(sm['amie_style']['hazard_sensitivity_weighted_ci'][0])} to {pc(sm['amie_style']['hazard_sensitivity_weighted_ci'][1])})", "texts_supplement.json"))
    d = sb["ours - amie_style"]
    R.append(("jm.diff", f"a difference of {pc(d['sens_diff_weighted'])} points ({m(pc(d['sens_diff_weighted_ci'][0]))} to {pc(d['sens_diff_weighted_ci'][1])})", "texts_supplement.json"))
    d = sb["ours - amie_style+layer"]
    R.append(("jm.layer", f"escalated {pc(-d['sens_diff_weighted'])} points more than the separated design ({pc(-d['sens_diff_weighted_ci'][1])} to {pc(-d['sens_diff_weighted_ci'][0])})", "texts_supplement.json"))
    R.append(("jm.lim", f"agreed moderately (κ = {ag['kappa']:.3f}) and found fewer texts with a safety hazard ({sm['ours']['safety_positive']} against {sp['basu_labels']['ours']['safety_positive']})", "texts_supplement.json"))
    a2 = j(DER / "a2_rating.json")
    p1, p2 = a2["primary"]["P1_safety_noninferiority_ai_minus_physician"], a2["primary"]["P2_accuracy_superiority_ai_minus_care_team"]
    R.append(("a2.p1", f"{pc(p1['diff'])} percentage points more frequent in the separated design's replies than in physician-written replies (95% CI, {pc(p1['ci95'][0])} to {pc(p1['ci95'][1])})", "a2_rating.json"))
    R.append(("a2.p1.holm", f"margin of {round(100 * 0.05)} points (Holm-adjusted P = {p1['p_holm']:.3f})", "a2_rating.json"))
    R.append(("a2.p2", f"scored {p2['diff']:.2f} points higher than care-team replies, available for 53 texts, for accuracy on a 5-point scale ({p2['ci95'][0]:.2f} to {p2['ci95'][1]:.2f}; P = {p2['p_holm']:.3f})", "a2_rating.json"))
    d2 = a2["design"]
    R.append(("a2.design", f"all {d2['safety_positive_texts']} texts with a safety hazard and 71 random others", "a2_rating.json"))
    R.append(("a2.ct", f"for {d2['care_team_texts']} texts, the care team's reply", "a2_rating.json"))
    u2 = a2["unweighted_counts"]["ai"]
    sp2 = a2["safety_positive_texts"]["ai"]
    R.append(("a2.sp.harm", f"{u2['moderate_or_severe']} of the separated design's replies ({pc(sp2['moderate_or_severe_harm'])}%) had moderate or severe harm", "a2_rating.json"))
    bt = a2["separated_design_by_reply_type"]
    R.append(("a2.alert", f"which the separated design sent on {bt['alert_text']['replies']} texts", "a2_rating.json"))
    R.append(("a2.alert.disc", f"harmful in {bt['alert_text']['moderate_or_severe_harm']} of {bt['alert_text']['replies']} texts", "a2_rating.json"))
    R.append(("a2.gen", f"The {bt['generated_reply']['replies']} generated replies had no moderate or severe harm", "a2_rating.json"))
    assert bt["generated_reply"]["moderate_or_severe_harm"] == 0 and bt["alert_text"]["moderate_or_severe_harm"] == u2["moderate_or_severe"]
    R.append(("a2.gen.acc", f"{bt['generated_reply']['safety_positive']} texts with a safety hazard that got a generated reply, mean accuracy was {bt['generated_reply']['mean_accuracy_safety_positive']:.2f} for the separated design and {bt['generated_reply']['amie_mean_accuracy_same_texts']:.2f}", "a2_rating.json"))
    R.append(("a2.irr", f"(κ between raters for harm, {a2['inter_rater']['moderate_or_severe_harm']['kappa']:.3f})", "a2_rating.json"))
    cv = a2["classifier_validation"]
    R.append(("a2.cls", f"({cv['classifier_escalates']} against {cv['rater_escalates']} of {cv['replies']} generated replies; κ = {cv['agreement']['kappa']:.3f})", "a2_rating.json"))
    rj = a2["rater_judged_escalation_safety_positive"]["same_day_or_more"]
    R.append(("a2.rj", f"escalated {pc(rj['separated_design']['sens'])}% of the 79 texts with a safety hazard ({pc(rj['separated_design']['ci95'][0])} to {pc(rj['separated_design']['ci95'][1])}) and the AMIE-style agent {pc(rj['amie']['sens'])}% ({pc(rj['amie']['ci95'][0])} to {pc(rj['amie']['ci95'][1])})", "a2_rating.json"))
    sg = j(DER / "texts_subgroups.json")["subgroups"]
    st = {k: v["ours"]["escalation_no_hazard_per_1000_weighted"] for k, v in sg["state"].items() if k != "Unknown"}
    lo, hi = min(st, key=st.get), max(st, key=st.get)
    R.append(("sub.state", f"ranged from {f1(st[lo])} per 1,000 texts without a hazard in {lo} to {f1(st[hi])}\nin {hi}", "texts_subgroups.json"))
    sx = sg["sex"]
    R.append(("sub.sex", f"was {f1(sx['Female']['ours']['escalation_no_hazard_per_1000_weighted'])} for women and {f1(sx['Male']['ours']['escalation_no_hazard_per_1000_weighted'])} for men; the AMIE-style agent showed the same pattern ({f1(sx['Female']['amie_style']['escalation_no_hazard_per_1000_weighted'])} and\n{f1(sx['Male']['amie_style']['escalation_no_hazard_per_1000_weighted'])})", "texts_subgroups.json"))
    mx = max(v["ours"]["safety_positive"] for var in sg.values() for v in var.values())
    R.append(("sub.max", f"rested on {mx} or fewer texts with a safety hazard", "texts_subgroups.json"))
    ch = j(DER / "texts_subgroups.json")["characteristics"]["race"]["all"]
    R.append(("sub.race", f"race was unknown or\nnot recorded for {ch['Unknown or not recorded'][0]} of {949 - ch['Unknown sender'][0]} known senders", "texts_subgroups.json"))
    R.append(("a2.cls.lim", f"κ = {cv['agreement']['kappa']:.3f}", "a2_rating.json"))
    rc = t["repeat_consistency"]
    R.append(("repeat", f"{pc(rc['ours']['share_all_3_agree'])}% of texts with the separated design and {pc(rc['amie_style']['share_all_3_agree'])}%", "texts_supplement.json"))
    u = j(DER / "urgency_agreement.json")["_all"]
    R.append(("urg.agree", f"{pc(u['escalation_agreement'])}% of 800 sampled replies (κ = {u['escalation_kappa']:.3f})", "urgency_agreement.json"))
    R.append(("urg.rates", f"{pc(u['g61_escalation_rate'])}% of replies an escalation, and the primary\nclassifier {pc(u['sonnet_escalation_rate'])}%", "urgency_agreement.json"))
    reg = j(D / "paper2_final/texts/report.json")
    R.append(("strict.held", f"{reg['_writer_replies_held_per_1000']} of every 1,000 replies from the reply\nmodel", "texts/report.json"))
    R.append(("strict.nohaz", f"{reg['clinician_first']['escalation_no_hazard_per_1000_weighted']} of every 1,000 texts without\na hazard", "texts/report.json"))
    rh = t["registered_hypotheses_v22"]
    R.append(("h1a", f"{pc(rh['H1a']['majority_diff'])} points; {pc(rh['H1a']['majority_diff_ci95'][0])} to {pc(rh['H1a']['majority_diff_ci95'][1])}", "registered_hypotheses_v22"))
    R.append(("h1b", f"{pc(rh['H1b']['majority_diff'])} points; exact McNemar P = {rh['H1b']['mcnemar_exact_two_sided_p']:.3f}", "registered_hypotheses_v22"))
    e = j(DER / "texts_errors.json")
    R.append(("err.med", f"{e['missed_by_category']['ours']['med_safety']:.1f} unescalated on average over its three replies, and the AMIE-style agent {e['missed_by_category']['amie_style']['med_safety']:.1f}", "texts_errors.json"))
    R.append(("err.house", f"the separated design left {e['missed_by_category']['ours']['environmental_housing_safety']:.1f} unescalated on average over its three replies, and the AMIE-style agent {e['missed_by_category']['amie_style']['environmental_housing_safety']:.1f}", "texts_errors.json"))
    R.append(("groups", f"{e['label_groups']['other_hazard']} texts whose only hazard", "texts_errors.json"))
    # gate
    g = j(D / "paper2_dev/paging_gate_497/report.json")["outcomes"]
    em = g["emergency"]["scores"]["kev_p911"]
    R.append(("gate.auroc", f"{em['auroc']['est']:.3f} (95% CI,\n{em['auroc']['ci95'][0]:.3f} to {em['auroc']['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    R.append(("gate.miss", f"{pc(em['crc_0.05']['miss']['mean'])}% of", "paging_gate_497/report.json"))
    R.append(("gate.paged", f"would alert a physician for {pc(em['crc_0.05']['paged_w']['mean'])}% of texts", "paging_gate_497/report.json"))
    R.append(("gate.above", f"{pc(em['crc_0.05']['share_splits_miss_above_alpha'])}% of the splits", "paging_gate_497/report.json"))
    R.append(("gate.full_paged", f"{pc(em['full_sample_0.05']['paged_w'])}% of texts to care teams", "paging_gate_497/report.json"))
    sa = g["safety"]["scores"]["kev_p_contact_doctor_or_911"]["auroc"]
    R.append(("gate.safety_auroc", f"{sa['est']:.3f} ({sa['ci95'][0]:.3f} to {sa['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    aw = em["auroc_weighted"]
    R.append(("gate.auroc_w", f"{aw['est']:.3f} ({aw['ci95'][0]:.3f} to {aw['ci95'][1]:.3f})", "paging_gate_497/report.json"))
    R.append(("gate.miss_p95", f"{pc(em['crc_0.05']['miss']['p95'])}%", "paging_gate_497/report.json"))
    ii = j(DER / "intake_instructions.json")["arms"]
    R.append(("intake.instr", f"{ii['amie_style']['unconditional_instruction']} of {ii['amie_style']['messages']} messages, the MAI-DxO-style "
              f"agent in {ii['mai_dxo_style']['unconditional_instruction']} of {ii['mai_dxo_style']['messages']}", "intake_instructions.json"))
    R.append(("intake.none", f"none of {ii['intake']['messages']}", "intake_instructions.json"))
    lk = j(DER / "texts_gate_test_and_leak.json")
    rl = lk["released_raw_llm_replies"]
    re_ = j(DER / "texts_routing_endpoints.json")
    em = re_["emergency_now_reaching_patient_per_1000_all_replies"]
    R.append(("leak.released", f"released {rl['n_released']:,} of\nits 2,847 replies, including {rl['emergency_now']} with this advice ({em['ours']} per 1,000)", "texts_gate_test_and_leak.json + routing"))
    ro, w = re_["ours_routed_only"], re_["withheld"]
    R.append(("route.only", f"escalated {pc(ro['sens_safety_positive'])}% of texts with a safety hazard ({pc(ro['sens_safety_positive_ci'][0])} to {pc(ro['sens_safety_positive_ci'][1])}) and {ro['per1000_no_hazard']} per 1,000", "texts_routing_endpoints.json"))
    R.append(("route.withheld", f"the separated design withheld {w['total']}: {w['replies_withheld_by_page']}\nbecause the emergency screen sent an alert for {w['texts_paged']} texts, and {w['replies_held_by_check_not_paged']}", "texts_routing_endpoints.json"))
    R.append(("route.review", f"The emergency\nscreen sent another {w['texts_review']} texts to physician review", "texts_routing_endpoints.json"))
    R.append(("doc.n", f"the {re_['n']['doctor_call']} texts that S.B. labeled as needing at least\na call from a doctor", "texts_routing_endpoints.json"))
    R.append(("doc.vals", f"the separated design escalated {pc(re_['ours']['sens_doctor_call'])}% ({pc(re_['ours']['sens_doctor_call_ci'][0])} to {pc(re_['ours']['sens_doctor_call_ci'][1])}), the AMIE-style\nagent {pc(re_['amie_style']['sens_doctor_call'])}%, the MAI-DxO-style agent {pc(re_['mai_dxo_style']['sens_doctor_call'])}% and the rule-based system {pc(re_['v22']['sens_doctor_call'])}%", "texts_routing_endpoints.json"))
    dd = [re_[f"ours - {b} (doctor_call)"] for b in ("amie_style", "mai_dxo_style", "v22")]
    R.append(("doc.diffs", "The differences were " + f"{100*dd[0]['diff']:.1f}".replace('-', '−') + f"\npoints ({100*dd[0]['ci'][0]:.1f} to {100*dd[0]['ci'][1]:.1f})".replace('-', '−') + ", " + f"{100*dd[1]['diff']:.1f} points ({100*dd[1]['ci'][0]:.1f} to {100*dd[1]['ci'][1]:.1f})".replace('-', '−') + " and " + f"{100*dd[2]['diff']:.1f} points ({100*dd[2]['ci'][0]:.1f} to {100*dd[2]['ci'][1]:.1f})".replace('-', '−'), "texts_routing_endpoints.json"))
    rc = re_["ours_registered_check"]
    R.append(("reg.sens", f"escalated {pc(rc['sens_safety_positive'])}% of texts with a safety hazard ({pc(rc['sens_safety_positive_ci'][0])} to {pc(rc['sens_safety_positive_ci'][1])}) but {rc['per1000_no_hazard']} of every 1,000", "texts_routing_endpoints.json"))
    lay = [em[k] for k in ("amie_style+layer", "mai_dxo_style+layer", "v22+layer")]
    R.append(("leak.layer", f"released {min(lay)} to {max(lay)} such replies per 1,000", "texts_routing_endpoints.json"))
    g = j(Path.home() / "waymark-local/packaging/ai-pcp/data/gate_thresholds.json")
    e = [x for x in g["split_sample_check"]["safety"]["evaluations"] if x["choice"] == "target"][0]
    R.append(("refit", f"missed {100*e['miss_mean']:.1f}%\nof the texts needing a doctor's call in the other half, on average over 500 random splits (95th percentile {100*e['miss_p95']:.1f}%; above 10% in {100*e['share_splits_miss_above_alpha']:.1f}% of\nsplits), but it sent {100*e['gated_weighted_mean']:.1f}% of texts to review", "ai-pcp/data/gate_thresholds.json"))
    cost = j(DER / "hbpro_test_supplement.json")["cost"]
    R.append(("cost", f"USD {cost['attribution_rules']['A_file_order']['compass_pro_full']['usd_per_task']:.3f} per task at list prices, the AMIE-style agent USD {cost['comparator_chains']['amie_style']['usd_per_task']:.3f} and the MAI-DxO-style agent USD {cost['comparator_chains']['mai_dxo_style']['usd_per_task']:.3f}", "hbpro_test_supplement.json"))
    for arm in ["opus_search_edit", "opus_edit", "amie_style_uncapped", "mai_dxo_style_uncapped", "amie_style_search"]:
        if arm not in rep2:
            continue
        x = rep2[arm]
        R.append((f"t1.{arm}", f"{f1(x['length_adjusted']['score'])} ({f1(x['length_adjusted']['se'])}) | {f1(x['raw']['score'])} | " + " | ".join(f1(x[k]['score']) for k in ['use_case=consult','use_case=research','use_case=writing','type=good_faith','type=red_teaming']) + f" | {x['mean_chars']:,} |", "hbpro_test/report.json"))
    ca = cmp_file(D / "hbpro_test/compare_amie_search.txt")
    xa, xb = ca["compass_pro_full - amie_style_search"], ca["opus_search_edit - amie_style_search"]
    sc = rep2["amie_style_search"]["length_adjusted"]["score"]
    R.append(("search.abstract", f"cut the multi-model design's lead over AMIE from 9.1 points (95% CI, 4.5–13.8) to\n{f1(xa['diff'])} ({f1(xa['ci95'][0])}–{f1(xa['ci95'][1])})", "compare_amie_search.txt"))
    R.append(("search.results", f"the AMIE-style agent scored {f1(sc)}, and the pipeline's lead was {f1(xa['diff'])} points ({f1(xa['ci95'][0])} to {f1(xa['ci95'][1])})", "compare_amie_search.txt"))
    R.append(("search.twostep", f"the two-step\npipeline's lead was {f1(xb['diff'])} points ({f1(xb['ci95'][0])} to {f1(xb['ci95'][1])})".replace('-0', '−0'), "compare_amie_search.txt"))
    R.append(("search.discussion", f"the multi-model pipeline's lead fell to {f1(xa['diff'])} points ({f1(xa['ci95'][0])} to {f1(xa['ci95'][1])}), and the two-step pipeline's lead fell to {f1(xb['diff'])} points ({f1(xb['ci95'][0])} to {f1(xb['ci95'][1])})".replace('-0', '−0'), "compare_amie_search.txt"))
    R.append(("search.lift", f"by {f1(sc - rep2['amie_style']['length_adjusted']['score'])} points", "report.json"))
    sl = t["same_safety_layer"]["metrics"]
    lo, hi = pc(sl["raw_llm+layer"]["hazard_sensitivity_weighted"]), pc(max(sl[k]["hazard_sensitivity_weighted"] for k in sl))
    mn = t["main"]
    inc = {a: 100 * (sl[f"{a}+layer"]["hazard_sensitivity_weighted"] - mn[a]["hazard_sensitivity_weighted"]) for a in ("amie_style", "mai_dxo_style", "v22")}
    inc["raw_llm"] = 100 * (mn["ours"]["hazard_sensitivity_weighted"] - mn["raw_llm"]["hazard_sensitivity_weighted"])
    binc = {a: sl[f"{a}+layer"]["escalation_no_hazard_per_1000_weighted"] - mn[a]["escalation_no_hazard_per_1000_weighted"] for a in ("amie_style", "mai_dxo_style", "v22")}
    binc["raw_llm"] = mn["ours"]["escalation_no_hazard_per_1000_weighted"] - mn["raw_llm"]["escalation_no_hazard_per_1000_weighted"]
    wa = mn["writer_alone"]
    R.append(("writer.filter", f"escalated {pc(wa['hazard_sensitivity_weighted'])}% ({pc(wa['hazard_sensitivity_weighted_ci'][0])} to {pc(wa['hazard_sensitivity_weighted_ci'][1])}) and {f1(wa['escalation_no_hazard_per_1000_weighted'])} per 1,000 ({f1(wa['escalation_no_hazard_weighted_ci'][0])} to {f1(wa['escalation_no_hazard_weighted_ci'][1])})", "texts_supplement.json"))
    R.append(("layer.inc.abstract", f"hazard sensitivity by {min(inc.values()):.1f}\nto {max(inc.values()):.1f} percentage points", "same_safety_layer"))
    R.append(("layer.inc.each", f"by {inc['raw_llm']:.1f} percentage points for\nthe reply model alone (which with the safety layer is the separated design), {inc['amie_style']:.1f} points for the AMIE-style agent, {inc['mai_dxo_style']:.1f}\npoints for the MAI-DxO-style agent and {inc['v22']:.1f} points for the rule-based system", "same_safety_layer"))
    R.append(("layer.inc.burden", f"by {min(binc.values()):.1f} to {max(binc.values()):.1f} per 1,000 texts without a hazard", "same_safety_layer"))
    cf = j(DER / "texts_confusion.json")
    writers = ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22", "amie_style+layer", "mai_dxo_style+layer", "v22+layer"]
    pv = [cf["systems"][a]["safety_hazard"]["ppv"]["est"] for a in writers]
    nv = [cf["systems"][a]["safety_hazard"]["npv"]["est"] for a in writers]
    R.append(("conf.ppv", f"ranged from {pc(min(pv))}% to {pc(max(pv))}%, and negative predictive values from\n{pc(min(nv))}% to {pc(max(nv))}%", "texts_confusion.json"))
    emx = cf["emergency_now_reaching_patient_by_label"]
    ct = cf["call_911_texts"]; ru = ct["reply_urgency"]
    nu = lambda a: sum(v for k, v in ru[a].items() if k in ("urgent_same_day", "routine_followup", "none"))  # noqa: E731
    assert nu("amie_style") == nu("mai_dxo_style") == nu("raw_llm")
    R.append(("e911.reply", f"the AMIE-style agent advised emergency care now in {ru['amie_style']['emergency_now']} of {ct['replies_per_system']} replies, the MAI-DxO-style agent in {ru['mai_dxo_style']['emergency_now']},\nthe reply model in {ru['raw_llm']['emergency_now']} and the rule-based system in {ru['v22']['emergency_now']}; {nu('amie_style')} of {ct['replies_per_system']} replies from each of the first three advised only\nsame-day or routine care, and the emergency screen sent an alert for all {ct['emergency_screen_alerts']} texts", "texts_confusion.json"))
    lo_, hi_ = min(ru[a]["emergency_now"] for a in ("amie_style", "mai_dxo_style", "raw_llm")), max(ru[a]["emergency_now"] for a in ("amie_style", "mai_dxo_style", "raw_llm"))
    R.append(("e911.disc", f"advised emergency care\nnow in only {lo_} to {hi_} of {ct['replies_per_system']} replies to these texts", "texts_confusion.json"))
    from scipy.stats import beta
    R.append(("e911.ci", f"(95% CI, {100*beta.ppf(0.025, 5, 1):.1f}% to 100%;", "exact binomial, 5 of 5"))
    pass  # conf.em.amie retired
    assert emx["ours"]["None"] == emx["ours"]["total"] == rl["emergency_now"]
    R.append(("conf.em.ours", f"all to texts that needed no action", "texts_confusion.json"))
    ov, tr = j(DER / "calib_test_overlap.json"), cf["transfer"]
    R.append(("overlap", f"at most {ov['test_texts_whose_possible_sender_also_sent_a_calibration_text']} test texts came from a phone linked to a person who also sent a calibration text, and\nwithout them the thresholds sent {tr['without_overlap']['sent_to_physician']} of {tr['without_overlap']['texts']} such texts", "calib_test_overlap.json; texts_confusion.json"))
    dg, ta, tb = tr["doctor_call_texts_by_group"], tr["test_auroc_call_911"], tr["test_auroc_doctor_call"]
    R.append(("doc.split", f"({dg['safety_hazard']} with a safety hazard and {dg['other_hazard_only']} with only an \"other\" hazard)", "texts_confusion.json"))
    R.append(("test.auroc", f"AUROC was {ta['est']:.3f} ({ta['ci'][0]:.3f} to {ta['ci'][1]:.3f}) for \"call 911 or 988\" and {tb['est']:.3f} ({tb['ci'][0]:.3f} to {tb['ci'][1]:.3f}) for a doctor's call", "texts_confusion.json"))
    gg = j(D / "paper2_dev/paging_gate_497/report.json")["outcomes"]
    ge = gg["emergency"]; wr = ge["scores"]["kev_p911"]["crc_weighted_rule_0.05"]
    R.append(("gate.weighted", f"alerted on {pc(wr['paged']['mean'])}% of calibration texts on average", "paging_gate_497/report.json"))
    R.append(("gate.wshare", f"one emergency text held {pc(ge['largest_positive_weight_share'])}% of the weight of all {ge['positives']}", "paging_gate_497/report.json"))
    R.append(("gate.positives", f"{ge['positives']} texts needed a call to 911 or 988 and {gg['safety']['positives']} at least a call from a doctor", "paging_gate_497/report.json"))
    R.append(("layer.range.discussion", f"hazard sensitivity of {lo}% to {hi}%", "same_safety_layer"))
    cn = cmp_file(D / "hbpro_test/compare_nosearch.txt")
    for k, lab in [("opus_edit - opus_search_edit", "nosearch.vs_search"), ("opus_edit - amie_style", "nosearch.vs_amie")]:
        d = cn[k]
        R.append((lab, f"{f1(abs(d['diff']))} points ({f1(d['ci95'][0])} to {f1(d['ci95'][1])})".replace('-', '−') if lab.endswith("amie") else f"{f1(abs(d['diff']))} points ({f1(-d['ci95'][1])} to {f1(-d['ci95'][0])})", "compare_nosearch.txt"))
    cl = lk["classifier_emergency_now_all_replies"]
    R.append(("leak.cls", f"in {cl['amie_style']['per_1000']} of every 1,000 replies, the MAI-DxO-style agent in {cl['mai_dxo_style']['per_1000']}, the reply model in {cl['raw_llm']['per_1000']} and the rule-based system in {cl['v22']['per_1000']}", "texts_gate_test_and_leak.json"))
    R.append(("hb.simple_raw", f"two-step pipeline scored {rep2['opus_search_edit']['raw']['score']} and the full pipeline {rep2['compass_pro_full']['raw']['score']}", "hbpro_test/report.json"))
    R.append(("hb.simple_chars", f"averaged {rep2['compass_pro_full']['mean_chars']:,} characters\nand the two-step pipeline's {rep2['opus_search_edit']['mean_chars']:,}", "hbpro_test/report.json"))
    gt = lk["gate_test"]["hazard_contact_doctor_or_911_gated"]
    R.append(("gate.test", f"but the review threshold sent only {gt[1]} of the {gt[0]} texts needing a doctor's call to a physician", "texts_gate_test_and_leak.json"))
    R.append(("gate.test_share", f"the emergency screen sent an alert for {100 * lk['weighted_page_share']:.1f}% of texts and a review request for {100 * lk['weighted_review_share']:.1f}%", "texts_gate_test_and_leak.json"))
    # intake
    ir = (D / "paper2_dev/intake_eval/report.md").read_text()
    for v in ["4.71", "3.46", "3.18", "71.7%", "40.9%", "55.8%", "6.43", "7.82", "7.71", "27.3%", "19.0%", "19.4%"]:
        R.append((f"intake.{v}", v.rstrip("%"), "intake_eval/report.md" if v in ir else "MISSING IN SOURCE"))
    return R


SUPP = MS.parent / "compass_clinician_first_ai_pcp_npj_supplement.md"
SUPP_ONLY = {"gate.safety_auroc", "hb_test.uncapped.amie_style_uncapped", "sub.race", "intake.3.18", "strict.cost_pipe", "av.layer", "av.alert", "sub.state", "sub.sex", "gate.paged", "gate.test_share", "route.only", "hb_test.simple_vs_amie", "nosearch.vs_amie", "jm.layer", "a2.cls", "a2.cls.lim", "a2.rj", "intake.instr", "intake.none", "intake.4.71", "intake.3.46", "hb_test.uncapped.mai_dxo_style_uncapped", "retest.primary", "txt.gate_only", "txt.writer_only", "a2.ct",
             "a2.gen.acc", "repeat", "urg.agree", "route.withheld", "doc.diffs", "intake.40.9%", "intake.55.8%", "dev.opusonly", "dev.simple", "hb_test.len_reg", "hb_test.redteam_diff", "retest.ours", "retest.amie",
             "hb_dev.amie_style_g61", "hb_dev.chars", "abl.search", "abl.single", "abl.review", "abl.editor",
             "txt.instr.amie_style", "txt.instr.mai_dxo_style", "txt.instr.raw_llm", "txt.gate_paged", "urg.rates", "strict.held", "h1a", "h1b", "err.house", "gate.above", "gate.full_paged",
             "gate.auroc_w", "route.review", "hb.simple_raw", "hb.simple_chars", "intake.6.43", "intake.7.82",
             "intake.7.71", "intake.19.0%", "intake.19.4%"}  # reported in the Supplementary Information, not the main text


RETIRED = {"layer.range.abstract", "hb_test.diff.amie.abstract", "disc.ladder", "jm.lim", "a2.alert.disc", "layer.range.discussion", "sub.max", "cost.ratio", "search.abstract", "search.twostep", "search.discussion", "search.lift", "retest.ours", "retest.amie", "h1b", "gate.auroc_w", "hb.simple_raw", "hb.simple_chars"}  # no longer stated
# in the main text; the same values are in the supplement tables, which are generated from the same result files


def main() -> None:
    ms = MS.read_text()
    flat = re.sub(r"\s+", " ", ms)
    sup = re.sub(r"\s+", " ", SUPP.read_text())
    rows, bad = [], 0
    for key, val, src in registry():
        if key in RETIRED:
            continue
        if key in SUPP_ONLY:
            nums = re.findall(r"\d[\d,]*\.?\d*", val)
            ok = all(n.rstrip(".") in sup for n in nums) and src != "MISSING IN SOURCE"
            src += " [supplement]"
        else:
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
