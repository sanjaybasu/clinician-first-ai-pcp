"""Figures for the Paper 2 manuscript (npj Digital Medicine draft). Every plotted number is read at run time from the
canonical result files in compass-pcp-data; nothing is typed in. Aggregates only.

Fig 1  architecture schematic (no data)
Fig 2  HealthBench Professional test: scores (95% CI) and paired differences  <- hbpro_test/report.json, compare.txt
Fig 3  real texts: hazard sensitivity vs escalations without hazard; unconditional emergency instructions
       <- paper2_final/derived/real_texts.json (paper/supplementary_analyses.py)
Usage: python -m paper.make_figures_paper2 [--out <dir>]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
OUT = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/figures"
ARMS = [("compass_pro_full", "Our architecture"), ("opus_search_edit", "Two-step version*"),
        ("opus_edit", "Two-step, no web search*"), ("amie_style", "AMIE-style"),
        ("amie_style_search", "AMIE-style with web search*"), ("amie_style_uncapped", "AMIE-style, 12,000 tokens*"),
        ("mai_dxo_style", "MAI-DxO-style"), ("mai_dxo_style_uncapped", "MAI-DxO-style, 12,000 tokens*"),
        ("opus55", "Opus 5.5, no system prompt"), ("gpt61sol", "GPT-6.1 Sol, no system prompt"), ("physician", "Physicians")]


def _diffs() -> dict:
    """compass_pro_full minus each arm, from every compare file (reversed pairs negated)."""
    out = {}
    for f in sorted((D / "hbpro_test").glob("compare*.txt")):
        if f.name in ("compare_g61.txt", "compare_opus.txt"):  # alternate graders
            continue
        txt = f.read_text()
        objs, dec, i = [], json.JSONDecoder(), 0
        while i < len(txt):
            j = txt.find("{", i)
            if j < 0:
                break
            o, i = dec.raw_decode(txt, j)
            objs.append(o)
        for o in objs:
            for k, v in o.items():
                a, _, b = k.partition(" - ")
                if a == "compass_pro_full":
                    out[b] = v
                elif b == "compass_pro_full":
                    out[a] = {"n": v["n"], "diff": -v["diff"], "ci95": [-v["ci95"][1], -v["ci95"][0]]}
    return out
C_OURS, C_OTHER = "#1b6ca8", "#8c8c8c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.75, "a  Patient-facing path (every message)", fontsize=8, weight="bold")
    det, llm, md = "#e8eef6", "#fdf1dc", "#e3f1e3"
    _box(ax, 0.1, 4.3, 1.5, 1.0, "Patient text", "white")
    _box(ax, 2.0, 4.3, 1.9, 1.0, "Paging gate\nKev-9B (fine-tuned)\nconformal threshold", det)
    _box(ax, 4.3, 4.3, 1.9, 1.0, "Reply / intake\nfrontier model\n(no added rules)", llm)
    _box(ax, 6.6, 4.3, 1.6, 1.0, "Clinician-first\ncheck (code)", det)
    _box(ax, 8.5, 4.3, 1.4, 1.0, "Reply sent", "white")
    _box(ax, 4.3, 2.95, 3.9, 0.8, "Physician paged;\nfixed urgent-page text to patient", md)
    for a, b in [((1.6, 4.8), (2.0, 4.8)), ((3.9, 4.8), (4.3, 4.8)), ((6.2, 4.8), (6.6, 4.8)), ((8.2, 4.8), (8.5, 4.8))]:
        _arrow(ax, *a, *b)
    _arrow(ax, 2.95, 4.3, 4.6, 3.75)
    _arrow(ax, 7.4, 4.3, 7.4, 3.75)
    ax.text(3.75, 3.95, "P(911/988) ≥ t_e: page", fontsize=6.0)
    _box(ax, 0.1, 2.85, 3.0, 0.95, "P(doctor or more) ≥ t_s:\nphysician review, no page;\nAI reply still sent", "#e3f1e3")
    _arrow(ax, 2.6, 4.3, 1.6, 3.75)
    ax.text(7.5, 3.95, "unconditional\nER/911/988", fontsize=6.5)
    ax.text(0.1, 2.35, "b  Physician-facing path (consults, notes, evidence questions)", fontsize=8, weight="bold")
    _box(ax, 0.1, 0.9, 1.5, 1.0, "Physician\nrequest", "white")
    _box(ax, 2.0, 1.5, 1.9, 0.65, "Draft A: Opus 5.5\n+ web search", llm)
    _box(ax, 2.0, 0.6, 1.9, 0.65, "Draft B: GPT-6.1 Sol", llm)
    _box(ax, 4.3, 0.9, 1.9, 1.0, "Review (GPT-6.1 Sol):\nerrors, omissions,\nconflicts", llm)
    _box(ax, 6.6, 0.9, 1.6, 1.0, "Final writer\n(Opus 5.5)", llm)
    _box(ax, 8.5, 0.9, 1.4, 1.0, "Length editor\n(target ≤2,000\nchars)", llm)
    _arrow(ax, 1.6, 1.4, 2.0, 1.8)
    _arrow(ax, 1.6, 1.4, 2.0, 0.95)
    _arrow(ax, 3.9, 1.8, 4.3, 1.5)
    _arrow(ax, 3.9, 0.95, 4.3, 1.3)
    _arrow(ax, 6.2, 1.4, 6.6, 1.4)
    _arrow(ax, 8.2, 1.4, 8.5, 1.4)
    for i, (fc, lab) in enumerate([(det, "code or fine-tuned small model"), (llm, "frontier language model"),
                                   (md, "physician")]):
        ax.add_patch(FancyBboxPatch((0.1 + 3.4 * i, 0.05), 0.25, 0.2, boxstyle="round,pad=0.01", fc=fc, ec="#333"))
        ax.text(0.45 + 3.4 * i, 0.15, lab, va="center", fontsize=6.5)
    fig.savefig(out / "Figure1_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2(out: Path) -> None:
    rep = json.loads((D / "hbpro_test/report.json").read_text())
    cmp_ = _diffs()
    arms = [x for x in ARMS if x[0] in rep]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    for k, (arm, lab) in enumerate(arms):
        s = rep[arm]["length_adjusted"]
        a.errorbar(s["score"], k, xerr=1.96 * s["se"], fmt="o", color=C_OURS if k == 0 else C_OTHER, ms=4, capsize=2)
    a.set_yticks(range(len(arms)), [x[1] for x in arms])
    a.invert_yaxis()
    a.set_xlabel("Length-adjusted score (95% CI)")
    a.set_title("a  HealthBench Professional, 350 test tasks", loc="left", fontsize=8)
    for k, (arm, lab) in enumerate(arms[1:]):
        d = cmp_[arm]
        b.errorbar(d["diff"], k, xerr=[[d["diff"] - d["ci95"][0]], [d["ci95"][1] - d["diff"]]], fmt="o", color=C_OURS,
                   ms=4, capsize=2)
    b.axvline(0, color="#333", lw=0.6)
    b.set_yticks(range(len(arms) - 1), [x[1] for x in arms[1:]])
    b.invert_yaxis()
    b.set_xlabel("Paired difference, ours minus system (95% CI)")
    b.set_title("b  Paired differences", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure2_healthbench_professional.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure2_healthbench_professional.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", default=["1", "2", "3"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if "1" in a.only:
        fig1(a.out)
    if "2" in a.only:
        fig2(a.out)
    if "3" in a.only:
        fig3(a.out)


TEXT_ARMS = [("ours", "Our architecture"), ("amie_style", "AMIE-style"), ("mai_dxo_style", "MAI-DxO-style"),
             ("raw_llm", "Opus 5.5, persona only"), ("v22", "Rule-based system (v22)"), ("gate_alone", "Paging gate alone")]


def fig3(out: Path) -> None:
    t = json.loads((D / "paper2_final/derived/texts_supplement.json").read_text())
    lk = json.loads((D / "paper2_final/derived/texts_gate_test_and_leak.json").read_text())
    m, ins, sl = t["main"], t["ai_emergency_instructions"], t["same_safety_layer"]["metrics"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    offs = {"ours": (-78, -16), "v22": (-122, 10), "amie_style": (10, -3), "mai_dxo_style": (10, -10), "raw_llm": (6, 4),
            "gate_alone": (6, 4)}
    bb = {"fc": "white", "ec": "none", "alpha": 0.85, "pad": 0.4}
    absl = {"ours": (14, 57.8), "v22": (183, 58.4)}  # label positions in data units, clear of the CI bars
    for arm, lab in TEXT_ARMS:
        r = m[arm]
        x, y = r["escalation_no_hazard_per_1000_weighted"], 100 * r["hazard_sensitivity_weighted"]
        xl, xh = r["escalation_no_hazard_weighted_ci"]
        yl, yh = (100 * v for v in r["hazard_sensitivity_weighted_ci"])
        a.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", ms=4, capsize=2,
                   color=C_OURS if arm == "ours" else C_OTHER, alpha=1 if arm == "ours" else 0.8)
        if arm in absl:
            a.annotate(lab, (x, y), xytext=absl[arm], textcoords="data", fontsize=6.3, va="center", bbox=bb)
        else:
            a.annotate(lab, (x, y), textcoords="offset points", xytext=offs.get(arm, (4, 4)), fontsize=6.3, bbox=bb)
    for k, lab in [("amie_style+layer", "AMIE-style + same gate and check"), ("mai_dxo_style+layer", "MAI-DxO-style + layer"),
                   ("v22+layer", "v22 + layer")]:
        r = sl[k]
        x, y = r["escalation_no_hazard_per_1000_weighted"], 100 * r["hazard_sensitivity_weighted"]
        xl, xh = r["escalation_no_hazard_weighted_ci"]
        yl, yh = (100 * v for v in r["hazard_sensitivity_weighted_ci"])
        a.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", ms=4, mfc="white", mec=C_OTHER,
                   ecolor="#cccccc", capsize=2, ls="none")
        a.annotate(lab, (r["escalation_no_hazard_per_1000_weighted"], 100 * r["hazard_sensitivity_weighted"]),
                   xytext=(98, 72.6) if k == "v22+layer" else (6, 3),
                   textcoords="data" if k == "v22+layer" else "offset points", fontsize=6.0, color="#555", bbox=bb)
    a.set_xlabel("Escalations per 1,000 texts without a hazard (95% CI)")
    a.set_ylabel("Hazard sensitivity, % (95% CI)")
    a.set_title("a  Real patient texts (949; open = same safety layer)", loc="left", fontsize=8)
    arms = [("ours", "Our architecture"), ("raw_llm", "Opus 5.5, persona only"), ("v22", "Rule-based (v22)"),
            ("amie_style", "AMIE-style"), ("mai_dxo_style", "MAI-DxO-style")]
    pat = [0.0 if k == "ours" else ins[k]["unconditional_instruction_per_1000_replies"] for k, _ in arms]
    em = json.loads((D / "paper2_final/derived/texts_routing_endpoints.json").read_text())[
        "emergency_now_reaching_patient_per_1000_all_replies"]
    cls = [em[k] for k, _ in arms]
    yy = range(len(arms))
    b.barh([y - 0.2 for y in yy], pat, height=0.38, color="#bbbbbb", label="text patterns (unconditional)")
    b.barh([y + 0.2 for y in yy], cls, height=0.38, color=C_OURS, label="classifier (emergency care now)")
    for y, v1, v2 in zip(yy, pat, cls):
        b.text(v1 + 2, y - 0.2, f"{v1:.1f}", va="center", fontsize=6)
        b.text(v2 + 2, y + 0.2, f"{v2:.1f}", va="center", fontsize=6)
    b.set_yticks(list(yy), [x[1] for x in arms])
    b.invert_yaxis()
    b.legend(fontsize=6, frameon=False, loc="upper right")
    b.set_xlabel("Replies reaching the patient that advise\nemergency care, per 1,000")
    b.set_title("b  Emergency advice reaching the patient", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure3_real_texts.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure3_real_texts.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
