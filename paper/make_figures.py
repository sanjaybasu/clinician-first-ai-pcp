"""Figures and tables for the manuscript, read only from CANONICAL_NUMBERS.json (written by build_canonical.py).

Usage: python -m paper.make_figures --manuscript-dir <notebooks/compass-ai-pcp/manuscript>
Writes figures/fig1_architecture, fig2_healthbench, fig3_patient_messages (.png and .pdf) and tables/*.md.
A figure whose inputs are missing is skipped and listed, never drawn from placeholder values.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Validated categorical palette (dataviz reference instance, light mode); every mark is also direct-labeled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
ARM_ORDER = [("ai_pcp_v3", "COMPASS (frozen)"), ("raw_llm", "LLM alone"), ("llm_standard_guardrails", "Guardrail stack"),
             ("amie_style", "AMIE-style chain"), ("mai_dxo_style", "MAI-DxO-style panel")]
SCREEN_ORDER = [("frozen", "COMPASS screens (frozen)"), ("kev_only", "Local judge alone"), ("llm_only", "LLM screen alone"),
                ("floor_union", "Keyword and rule screens")]

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out / f"{name}.tiff", bbox_inches="tight", facecolor=SURFACE, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def fig1_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=SURFACE, ec=color, lw=1.6))
        ax.text(x + 0.12, y + h - 0.18, title, ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + 0.12, y + h - 0.52, body, ha="left", va="top", fontsize=7.2, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.1, color=INK2))

    ax.text(0.05, 5.85, "Language layer (LLMs)", fontsize=8, color=SERIES[0], fontweight="bold")
    ax.text(0.05, 0.02, "Decision layer (deterministic)", fontsize=8, color=SERIES[1], fontweight="bold")
    box(0.05, 4.1, 2.2, 1.6, "Patient message", "Text from a patient,\ncaregiver or clinician", INK2)
    box(2.55, 4.1, 2.4, 1.6, "Route and extract", "Complaint routing\nExtractor A and B (two\nvendors), with source\nand polarity", SERIES[0])
    box(5.25, 4.1, 2.2, 1.6, "Hazard screens", "Local judge and\nLLM screen; raise-\nonly, higher tier kept", SERIES[0])
    box(7.8, 4.1, 2.15, 1.6, "Reply", "Writes only what the\nengine permits; scope\ncheck before sending", SERIES[0])
    box(0.6, 0.4, 2.7, 2.2, "Findings ledger", "Reconciles the two\nextractions and the\nrecord; conflicts must\nbe confirmed", SERIES[1])
    box(3.65, 0.4, 2.9, 2.2, "Illness scripts", "Signed, sourced LRs\nLog-odds update, p95\nCannot-miss stopping rule\nRed flags act by rule", SERIES[1])
    box(6.9, 0.4, 3.05, 2.2, "Gate and lanes", "Blocking medication rules\nAsk, escalate, defer\nor complete\nDeferral reasons logged", SERIES[1])
    arrow(2.25, 4.9, 2.6, 4.9)
    arrow(4.9, 4.9, 5.25, 4.9)
    arrow(3.75, 4.1, 1.95, 2.6)
    arrow(6.35, 4.1, 8.4, 2.6)
    arrow(3.3, 1.45, 3.65, 1.45)
    arrow(6.55, 1.45, 6.9, 1.45)
    arrow(8.9, 2.6, 8.9, 4.1)
    _save(fig, out, "fig1_architecture")


THEMES = ("emergency_referrals", "context_seeking", "complex_responses", "global_health", "health_data_tasks", "communication", "hedging")


def cluster_label(name: str) -> str:
    """'hedging_no-uncertainty_seeks_context' -> 'Hedging, no uncertainty: seeks context'."""
    theme = next(th for th in THEMES if name.startswith(th + "_"))
    cond, crit = name[len(theme) + 1:].split("_", 1)
    return f"{theme.replace('_', ' ').capitalize()}, {cond.replace('-', ' ')}: {crit.replace('_', ' ')}"


def fig2_healthbench(hb: dict, out: Path) -> None:
    g1 = hb["grader1"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.4, 9.6), gridspec_kw={"height_ratios": [1, 3.3]})
    arms = [(k, lab) for k, lab in ARM_ORDER if k in g1["scores"]]
    for i, (k, lab) in enumerate(arms):
        y = len(arms) - 1 - i
        s = g1["scores"][k]
        a.plot(s["ci95"], [y, y], color=SERIES[i], lw=2, solid_capstyle="round")
        a.plot(s["mean"], y, "o", ms=8, color=SERIES[i], mec=SURFACE, mew=2)
        if "grader2" in hb and k in hb["grader2"]["scores"]:
            a.plot(hb["grader2"]["scores"][k]["mean"], y - 0.22, "D", ms=5, mfc=SURFACE, mec=SERIES[i], mew=1.4)
        a.text(s["ci95"][1] + 0.01, y, f"{s['mean']:.3f}", va="center", fontsize=7.5, color=INK)
    a.set_yticks(range(len(arms)), [lab for _, lab in reversed(arms)])
    a.set_xlabel("HealthBench score (mean, 95% CI)")
    a.set_title("A. Score by system (circle: reference grader; open diamond: second grader)", loc="left", fontsize=9, color=INK)
    a.grid(axis="x", color=GRID, lw=0.6)
    cl = sorted(g1["by_cluster_vs_raw"].items(), key=lambda kv: kv[1]["rate_a"] - kv[1]["rate_b"])
    for y, (name, v) in enumerate(cl):
        d = v["rate_a"] - v["rate_b"]
        color = SERIES[0] if d >= 0 else SERIES[1]
        b.plot([0, d], [y, y], color=GRID, lw=1)
        b.plot(d, y, "o", ms=5, color=color, mfc=color if v.get("holm_significant") else SURFACE, mew=1.3)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(range(len(cl)), [cluster_label(n) for n, _ in cl], fontsize=5.4)
    b.set_xlabel("Difference in consensus pass rate,\nCOMPASS minus LLM alone")
    b.set_title("B. By behavior cluster (filled: significant after Holm correction)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out, "fig2_healthbench")


def fig3_patient_messages(ca: dict, out: Path) -> None:
    s = ca["screens"]
    shown = [(k, lab) for k, lab in SCREEN_ORDER if k in s]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 0.45 * len(shown) + 1.3), sharey=True,
                               gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (k, lab) in enumerate(shown):
        y = len(shown) - 1 - i
        m = s[k]
        color = SERIES[0] if k == "frozen" else INK2
        lo, hi = m["sensitivity"]["wilson95"]
        a.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
        a.plot(m["sensitivity"]["estimate"], y, "o", ms=7, color=color, mec=SURFACE, mew=1.5)
        a.text(1.05, y, f"{m['sensitivity']['caught']}/{m['sensitivity']['positives']}", va="center", fontsize=7.2, color=INK,
               transform=a.get_yaxis_transform(), clip_on=False)
        b.barh(y, m["flags_per_1000"], height=0.55, color=color)
        b.text(m["flags_per_1000"] + 8, y, f"{m['flags_per_1000']:.0f}", va="center", fontsize=7.2, color=INK)
    a.axvline(0.80, color=INK2, lw=0.8, ls=(0, (3, 3)))
    a.set_yticks(range(len(shown)), [lab for _, lab in reversed(shown)])
    a.set_xlim(0, 1.03)
    a.set_xlabel("Sensitivity for safety hazards (95% CI)\ndashed line: gate on the lower bound, 0.80")
    a.set_title("A. Hazards detected", loc="left", fontsize=9, color=INK)
    b.set_xlabel("Messages flagged per 1,000")
    b.set_title("B. Flag rate", loc="left", fontsize=9, color=INK)
    b.set_xlim(0, max(1000, max(s[k]["flags_per_1000"] for k, _ in shown) * 1.15))
    for ax in (a, b):
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig3_patient_messages")


def tables(numbers: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    hb = numbers.get("_hb_analysis")
    if hb:
        g1, g2 = hb["grader1"], hb.get("grader2")
        rows = ["| System | Score, reference grader (95% CI) | COMPASS minus system, reference grader (95% CI) | "
                "Higher after Holm correction | Score, second grader (95% CI) | COMPASS minus system, second grader (95% CI) | "
                "Calls per reply | Tokens per reply | Words per reply, mean (median) |",
                "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
        words = (numbers.get("_sensitivity") or {}).get("reply_length", {})

        def dcell(sup):
            return (f"{sup['diff']:+.3f} ({sup['ci95'][0]:+.3f} to {sup['ci95'][1]:+.3f})".replace("+0.000", "0.000").replace("-0.000", "0.000")
                    if sup else "Reference")

        for k, lab in ARM_ORDER:
            if k not in g1["scores"]:
                continue
            s = g1["scores"][k]
            sup = g1["superiority"].get(k)
            h = ("Yes" if sup["holm_significant"] and sup["diff"] > 0 else "No") if sup else ""
            s2 = g2["scores"].get(k) if g2 else None
            sc2 = f"{s2['mean']:.3f} ({s2['ci95'][0]:.3f} to {s2['ci95'][1]:.3f})" if s2 else "Pending"
            d2 = dcell(g2["superiority"].get(k)) if g2 else "Pending"
            cost = g1["cost_per_conversation"].get(k)
            calls = f"{sum(v['calls'] for v in cost.values()):.1f}" if cost else "NA"
            toks = f"{sum(v['input_tokens'] + v['output_tokens'] for v in cost.values()):,.0f}" if cost else "NA"
            w = f"{words[k]['mean']:.0f} ({words[k]['median']:.0f})" if k in words else "NA"
            rows.append(f"| {lab} | {s['mean']:.3f} ({s['ci95'][0]:.3f} to {s['ci95'][1]:.3f}) | {dcell(sup)} | {h} | {sc2} | {d2} | {calls} | {toks} | {w} |")
        (out / "table2_healthbench.md").write_text("\n".join(rows) + "\n")
    ca = numbers.get("_cohort_a")
    if ca:
        rows = ["| Screen | Safety hazards detected | Sensitivity (95% CI) | Emergency messages detected | Specificity (95% CI) | PPV (95% CI) | Flags per 1,000 |",
                "| :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
        for k, lab in SCREEN_ORDER:
            if k not in ca["screens"]:
                continue
            m = ca["screens"][k]
            se, sp, pv = m["sensitivity"], m["specificity"], m["ppv"]

            def fmt(x):
                return f"{x['estimate']:.3f} ({x['wilson95'][0]:.3f} to {x['wilson95'][1]:.3f})" if x["estimate"] is not None else "NA"

            rows.append(f"| {lab} | {se['caught']} of {se['positives']} | {fmt(se)} | {m['call_911']['caught']} of {m['call_911']['positives']} | {fmt(sp)} | {fmt(pv)} | {m['flags_per_1000']:.0f} |")
        (out / "table3_patient_messages.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript-dir", type=Path, required=True)
    args = ap.parse_args()
    numbers = json.loads((args.manuscript_dir / "CANONICAL_NUMBERS.json").read_text())
    figs = args.manuscript_dir / "figures"
    fig1_architecture(figs)
    skipped = []
    if numbers.get("_hb_analysis"):
        fig2_healthbench(numbers["_hb_analysis"], figs)
    else:
        skipped.append("fig2_healthbench")
    if numbers.get("_cohort_a"):
        fig3_patient_messages(numbers["_cohort_a"], figs)
    else:
        skipped.append("fig3_patient_messages")
    tables(numbers, args.manuscript_dir / "tables")
    print(json.dumps({"skipped": skipped}))


if __name__ == "__main__":
    main()
