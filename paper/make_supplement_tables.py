"""Supplement tables generated from the signed clinical content (scripts and rules), so the appendix cannot drift.

Usage: python -m paper.make_supplement_tables --manuscript-dir <notebooks/compass-ai-pcp/manuscript>
"""

import re
import argparse
from collections import Counter
from pathlib import Path

import yaml

from compass_pcp.engine.scripts import load_script
from compass_pcp.harness.frozen import DATA_DIR


def scripts_table() -> str:
    rows = ["| Complaint | Lane | Diagnoses | Must-not-miss diagnoses | Questions | Likelihood ratios | Red-flag questions |",
            "| :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
    for p in sorted((DATA_DIR / "scripts").glob("*.yaml")):
        if p.stem == "TEMPLATE":
            continue
        s = load_script(p)
        cm = ", ".join(d.label for d in s.diagnoses if d.cannot_miss)
        rows.append(f"| {s.complaint.replace('_', ' ')} | {s.lane.replace('_', ' ')} | {len(s.diagnoses)} | {cm} | "
                    f"{len(s.features)} | {sum(len(d.lrs) for d in s.diagnoses)} | {len(s.must_ask)} |")
    return "\n".join(rows) + "\n"


def rules_table() -> str:
    rows = ["| Rule file | Rules | Kinds | Lab recency rules | Physician sign-off | Pharmacist sign-off at freeze |",
            "| :-- | :-- | :-- | :-- | :-- | :-- |"]
    for name in ("refill.yaml", "titration.yaml", "contraception.yaml"):
        d = yaml.safe_load((DATA_DIR / "rules" / name).read_text())
        kinds = Counter(r.get("kind", "unspecified") for r in d.get("rules", []))
        rev = d.get("clinical_review", {})
        rows.append(f"| {name} | {len(d.get('rules', []))} | {', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))} | "
                    f"{len(d.get('lab_recency', []))} | {rev.get('physician_approved_on') or 'pending'} | "
                    f"{rev.get('pharmacist_approved_on') or 'pending'} |")
    return "\n".join(rows) + "\n"


def prompts_appendix() -> str:
    """Every prompt that shapes a reply, copied from the code at build time."""
    from compass_pcp.arms import ai_pcp_v2 as v2
    from compass_pcp.arms import baselines as bl
    from compass_pcp.arms import hazard as hz

    blocks = [("COMPASS: permissions", v2.PERMISSIONS), ("COMPASS: router", v2.ROUTER), ("COMPASS: extractor", v2.EXTRACT),
              *[(f"COMPASS: reply, {k}", t) for k, t in v2.COMPOSE.items()], ("COMPASS: scope check", v2.SCOPE_CHECK),
              ("Hazard screen: LLM", hz.SCREEN), ("Hazard screen: local judge instructions", hz.KEV_INSTRUCTIONS),
              ("Hazard screen: local judge tiers", "\n".join(f"{k}: {v}" for k, v in hz.KEV_TIERS.items())),
              ("Shared output policy", bl.OUTPUT_LIMITS), ("LLM alone: persona", bl.PCP_PERSONA),
              ("Guardrail stack: input screen", bl.GUARD_INPUT), ("Guardrail stack: output review", bl.GUARD_OUTPUT),
              ("AMIE-style: analysis", bl.AMIE_ANALYZE), ("AMIE-style: draft", bl.AMIE_DRAFT), ("AMIE-style: refinement", bl.AMIE_REFINE),
              *[(f"MAI-DxO-style: {k.replace('_', ' ')} role", t) for k, t in bl.PANEL_ROLES.items()],
              ("MAI-DxO-style: decision", bl.PANEL_DECIDE)]
    return "\n\n".join(f"*{name}.*\n\n```\n{text.strip()}\n```" for name, text in blocks) + "\n"


SCREEN_NAMES = {"frozen": "COMPASS screens (frozen)", "frozen max tier": "COMPASS screens (frozen), highest tier",
                "kev_only": "Local judge alone", "llm_only": "LLM screen alone", "floor_union": "Keyword and rule screens combined",
                "compass_keyword": "Emergency keyword screen", "compass_befast": "Stroke symptom rule (BE-FAST)",
                "anchor_undertriage": "Rule-based under-triage screen",
                "kev_urgent_probability": "Local judge, probability of urgent or higher", "llm_tier": "LLM screen, urgency tier",
                "frozen_max_tier": "COMPASS screens, highest urgency tier (ties broken by the local judge probability)"}


def _name(k: str) -> str:
    return SCREEN_NAMES.get(k, SCREEN_NAMES.get(k.replace(" ", "_"), k.replace("_", " ")))


ARM_NAMES = {"ai_pcp_v3": "COMPASS", "raw_llm": "LLM alone", "llm_standard_guardrails": "Guardrail stack",
             "amie_style": "AMIE-style chain", "mai_dxo_style": "MAI-DxO-style panel"}


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _ci(x: float, ci: list[float]) -> str:
    return f"{x:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})".replace("+0.000", "0.000").replace("-0.000", "0.000")


def result_tables(numbers: dict) -> dict[str, str]:
    out = {}
    hb = numbers.get("_hb_analysis")
    if hb:
        g1 = hb["grader1"]
        rows = ["| Behavior cluster | Criteria | COMPASS pass rate | LLM alone pass rate | COMPASS only | LLM alone only | Exact P | Significant after Holm |",
                "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
        for c, v in sorted(g1["by_cluster_vs_raw"].items()):
            rows.append(f"| {c.replace('_', ' ')} | {v['n']} | {v['rate_a']:.3f} | {v['rate_b']:.3f} | {v['a_only']} | {v['b_only']} | "
                        f"{_p(v['p_exact'])} | {'Yes' if v['holm_significant'] else 'No'} |")
        out["s16_clusters.md"] = "\n".join(rows) + "\n"
        arms = list(g1["by_theme"])
        themes = sorted({t for a in arms for t in g1["by_theme"][a]})
        rows = ["| Theme | " + " | ".join(ARM_NAMES.get(a, a) for a in arms) + " |", "| :-- |" + " :-- |" * len(arms)]
        for t in themes:
            cells = [f"{g1['by_theme'][a][t]['mean_score']:.3f} (n = {g1['by_theme'][a][t]['n']})" if t in g1["by_theme"][a] else "NA" for a in arms]
            rows.append(f"| {t.replace('_', ' ')} | " + " | ".join(cells) + " |")
        out["s15_themes.md"] = "\n".join(rows) + "\n"
    ca = numbers.get("_cohort_a")
    if ca:
        rows = ["| Score | AUROC for safety hazards (bootstrap 95% CI) |", "| :-- | :-- |"]
        for k, v in ca["auroc_safety"].items():
            rows.append(f"| {_name(k)} | {v['estimate']:.3f} ({v['bootstrap95'][0]:.3f} to {v['bootstrap95'][1]:.3f}) |")
        rows += ["", "| Screen | Net benefit at 0.02 | at 0.05 | at 0.10 |", "| :-- | :-- | :-- | :-- |"]
        for k, m in ca["screens"].items():
            nb = m["net_benefit"]
            rows.append(f"| {_name(k)} | {nb['0.02']:.4f} | {nb['0.05']:.4f} | {nb['0.1']:.4f} |".replace("-0.0000", "0.0000"))
        rows += ["", "| Hazard category | Detected by the frozen screen |", "| :-- | :-- |"]
        rows += [f"| {c.replace('_', ' ')} | {v['caught']} of {v['positives']} |" for c, v in ca["recall_by_category"].items()]
        rows += ["", "| Screen compared with the frozen screen, on safety hazards | Frozen only | Other only | Exact P |", "| :-- | :-- | :-- | :-- |"]
        rows += [f"| {_name(k)} | {v['frozen_only']} | {v['other_only']} | {_p(v['p_exact'])} |"
                 for k, v in ca["mcnemar_vs_frozen_on_positives"].items()]
        out["s12_patient_message_secondary.md"] = "\n".join(rows) + "\n"
    s = numbers.get("_sensitivity")
    if s:
        rows = ["| Analysis | Grader or set | Result |", "| :-- | :-- | :-- |"]
        for g, lab in (("grader1", "Reference grader"), ("grader2", "Second grader")):
            r = s["repeats"][g]
            for k, (d, lo, hi) in enumerate(r["single_reply_diffs"], 1):
                rows.append(f"| COMPASS minus LLM alone, reply {k} only (95% CI) | {lab} | {d:+.3f} ({lo:+.3f} to {hi:+.3f}) |")
            rows.append(f"| Mean within-conversation SD of score across 3 replies, COMPASS / LLM alone | {lab} | "
                        f"{r['mean_within_sd']['ai_pcp_v3']:.3f} / {r['mean_within_sd']['raw_llm']:.3f} |")
        for g, lab in (("grader1", "Reference grader"), ("grader2", "Second grader")):
            e = s["equivalence"][g]
            rows.append(f"| COMPASS minus MAI-DxO-style panel, 90% CI; equivalent within 0.02 | {lab} | {e['diff']:+.3f} "
                        f"({e['ci90'][0]:+.3f} to {e['ci90'][1]:+.3f}); {'yes' if e['equivalent'] else 'no'} |")
        la = s.get("length_adjusted")
        if la:
            for g, lab in (("grader1", "Reference grader"), ("grader2", "Second grader")):
                for arm, v in la[g].items():
                    rows.append(f"| COMPASS minus {ARM_NAMES[arm]} at equal reply length: intercept (95% CI); slope per 100 words "
                                f"(95% CI) | {lab} | {_ci(v['intercept'], v['intercept_ci95'])}; "
                                f"{_ci(v['slope_per_100_words'], v['slope_ci95'])} |")
        pb = s["physician_blinding"]
        rows.append(f"| Physician score, COMPASS minus LLM alone, items without care-team or nurse mention (95% CI) | "
                    f"{pb['n_items']} items ({pb['n_excluded']} excluded) | {pb['diff'][0]:+.3f} ({pb['diff'][1]:+.3f} to {pb['diff'][2]:+.3f}) |")
        po = s["patient_overlap"]
        rows.append(f"| Hazard sensitivity, sealed messages from patients with no tuning-half message (Wilson 95% CI) | "
                    f"{po['n_messages']} messages ({po['n_excluded']} excluded) | {po['caught']} of {po['positives']}, "
                    f"{po['sensitivity']:.3f} ({po['wilson95'][0]:.3f} to {po['wilson95'][1]:.3f}); emergency messages "
                    f"{po['call_911_caught']} of {po['call_911']} |")
        pc = s.get("patient_clustered")
        if pc:
            rows.append(f"| Hazard sensitivity with a bootstrap over patients (95% CI) | {pc['n_patients']} patients; "
                        f"{pc['patients_with_hazard']} with a hazard | {pc['sensitivity']:.3f} ({pc['sensitivity_ci95'][0]:.3f} to "
                        f"{pc['sensitivity_ci95'][1]:.3f}); specificity {pc['specificity']:.3f} ({pc['specificity_ci95'][0]:.3f} to "
                        f"{pc['specificity_ci95'][1]:.3f}) |")
        pp = s.get("physician_preferences")
        if pp:
            for i, r in enumerate(pp["raters"], 1):
                rows.append(f"| Physician {i} overall preference (100 conversations) | COMPASS / LLM alone / tie | "
                            f"{r['compass']} / {r['llm_alone']} / {r['tie']} |")
        pg = s.get("patient_groups")
        if pg:
            rows += ["", "| Patient group (sealed half) | Messages | Flags per 1,000 | Specificity (Wilson 95% CI) | Safety hazards flagged |",
                     "| :-- | :-- | :-- | :-- | :-- |"]
            for dim, lab in (("sex", "Sex"), ("race_ethnicity", "Race and ethnicity"), ("age_group", "Age group")):
                for g, v in pg[dim].items():
                    sp = (f"{v['specificity']:.3f} ({v['specificity_wilson95'][0]:.3f} to {v['specificity_wilson95'][1]:.3f})"
                          if v["specificity"] is not None else "NA")
                    rows.append(f"| {lab}: {g} | {v['n']} | {v['flags_per_1000']:.0f} | {sp} | {v['caught']} of {v['positives']} |")
        rows += ["", "| HealthBench theme | Conversations | COMPASS minus LLM alone, reference grader | COMPASS minus LLM alone, second grader |",
                 "| :-- | :-- | :-- | :-- |"]
        for th, v in s["by_theme"]["grader1"].items():
            rows.append(f"| {th.replace('_', ' ')} | {v['n']} | {v['diff']:+.3f} | {s['by_theme']['grader2'][th]['diff']:+.3f} |")
        out["s14_sensitivity.md"] = "\n".join(rows) + "\n"
    return out


def fidelity_section() -> str:
    """Sections 2 to 7 of FID-PCP-001, unchanged, so the appendix and the controlled memo cannot differ."""
    text = (DATA_DIR.parents[1] / "docs" / "regulatory" / "FID-PCP-001_comparator_fidelity.md").read_text()
    body = text.split("## 2. ", 1)[1].split("## 8. ", 1)[0]
    body = (body.replace("Tu et al., Nature 2025 (AMIE)", "AMIE[@tu2025amie]")
            .replace("Vedadi et al. 2025 (guardrailed AMIE)", "guardrailed AMIE[@vedadi2025gamie]")
            .replace("Nori et al., arXiv 2506.22405 (MAI Diagnostic Orchestrator)", "MAI Diagnostic Orchestrator[@nori2026sequential]"))
    body = body.replace(" (VVP-PCP-001 Section 8.6)", "")
    names = {"Rules shared by every arm": "Rules shared by every comparator", "Raw model (`raw_llm`)": "LLM alone",
             "Conventional guardrail stack (`llm_standard_guardrails`)": "Guardrail stack",
             "AMIE-style chain (`amie_style`)": "AMIE-style chain", "MAI-DxO-style panel (`mai_dxo_style`)": "MAI-DxO-style panel",
             "Known limits of the comparison": "Known limits of the comparison"}
    text = ("## 2. " + body)
    text = re.sub(r"(?m)^## [0-9]+\. (.+)$", lambda m: "### " + names.get(m.group(1).strip(), m.group(1).strip()), text)
    return text.strip() + "\n"


def freeze_section(freeze_dir: Path) -> str | None:
    import json

    rows = ["| Test set | Items | Test-set SHA-256 | Settings SHA-256 | Frozen commit | Frozen at (UTC) |", "| :-- | :-- | :-- | :-- | :-- | :-- |"]
    for name, label in (("healthbench", "HealthBench Consensus test split"), ("cohort_a", "Patient messages, sealed half")):
        f = freeze_dir / name / "freeze.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        rows.append(f"| {label} | {d['test_set']['n']} | `{d['test_set']['set_sha256']}` | `{d['thresholds_sha256']}` | "
                    f"`{d['architecture_commit'][:12]}` | {d['created_utc'][:19]} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript-dir", type=Path, required=True)
    ap.add_argument("--freeze-dir", type=Path, default=DATA_DIR.parents[1] / "docs" / "regulatory" / "freeze")
    args = ap.parse_args()
    supp = args.manuscript_dir / "supplement"
    supp.mkdir(parents=True, exist_ok=True)
    (supp / "s9_fidelity.md").write_text(fidelity_section())
    fz = freeze_section(args.freeze_dir)
    if fz:
        (supp / "s1_freeze.md").write_text(fz)
    out = args.manuscript_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s4_illness_scripts.md").write_text(scripts_table())
    (out / "s5_rules.md").write_text(rules_table())
    (out / "s11_prompts.md").write_text(prompts_appendix())
    canon = args.manuscript_dir / "CANONICAL_NUMBERS.json"
    if canon.exists():
        import json

        for name, text in result_tables(json.loads(canon.read_text())).items():
            (out / name).write_text(text)
    print("written")


if __name__ == "__main__":
    main()
