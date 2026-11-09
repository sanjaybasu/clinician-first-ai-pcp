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

import numpy as np  # noqa: E402
import matplotlib.text  # noqa: E402

PROBLEMS: list[str] = []


def check_overlap(fig, name: str) -> None:
    """Flag text that touches a line, marker, bar or box edge, or other text. Draws the figure without text, then counts
    dark pixels inside each text's box (light box fills are ignored)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    for i, b1 in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b2 = boxes[j]
            ov = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) * (min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            if b1.overlaps(b2) and ov > 4:
                PROBLEMS.append(f"{name}: text on text: {texts[i].get_text()[:30]!r} / {texts[j].get_text()[:30]!r}")
    for t in texts:
        t.set_visible(False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    for t in texts:
        t.set_visible(True)
    h = img.shape[0]
    dark = img.min(axis=2) < 215
    for t, b in zip(texts, boxes):
        x0, x1, y0, y1 = int(b.x0) + 2, int(b.x1) - 2, int(h - b.y1) + 2, int(h - b.y0) - 2
        if x1 > x0 and y1 > y0 and dark[max(y0, 0):y1, max(x0, 0):x1].sum() > 3:
            PROBLEMS.append(f"{name}: text on a line or edge: {t.get_text()[:40]!r}")


D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
OUT = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/figures"
ARMS = [("compass_pro_full", "Our pipeline"), ("opus_search_edit", "Two-step pipeline*"),
        ("opus_edit", "Two-step pipeline, no web search*"), ("amie_style", "AMIE-style agent"),
        ("amie_style_search", "AMIE-style agent, web search*"), ("amie_style_uncapped", "AMIE-style agent, 12,000 tokens*"),
        ("mai_dxo_style", "MAI-DxO-style agent"), ("mai_dxo_style_uncapped", "MAI-DxO-style agent, 12,000 tokens*"),
        ("opus55", "Claude Opus 5.5, no system prompt"), ("gpt61sol", "GPT-6.1 Sol, no system prompt"),
        ("physician", "Physicians (dataset responses)")]


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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")
    det, llm, md = "#e8eef6", "#fdf1dc", "#e3f1e3"
    ax.text(0.1, 5.85, "a  Patient-facing path: our safety layer around the reply model", fontsize=8, weight="bold")
    _box(ax, 0.1, 4.55, 1.4, 0.95, "Patient text", "white")
    _box(ax, 1.9, 4.55, 2.1, 0.95, "Emergency screen\n(Kev-9B, fine-tuned;\nconformal thresholds)", det)
    _box(ax, 4.4, 4.55, 2.0, 0.95, "Reply model\n(frontier model,\nno clinical rules)", llm)
    _box(ax, 6.8, 4.55, 1.8, 0.95, "Emergency-\ninstruction filter\n(code)", det)
    _box(ax, 8.9, 4.55, 1.0, 0.95, "Reply\nsent", "white")
    _box(ax, 0.1, 3.0, 3.6, 0.95, "Review request (P[doctor's call] at or\nabove review threshold): physician\nreviews; AI reply still sent", md)
    _box(ax, 4.4, 3.0, 4.2, 0.95, "Alert (P[call 911 or 988] at or above alert\nthreshold, or reply held by the filter): physician\ncalls back; patient gets a fixed text", md)
    for x0, x1 in [(1.5, 1.9), (4.0, 4.4), (6.4, 6.8), (8.6, 8.9)]:
        _arrow(ax, x0, 5.02, x1, 5.02)
    _arrow(ax, 2.4, 4.55, 2.4, 3.95)
    _arrow(ax, 3.6, 4.55, 4.7, 3.95)
    _arrow(ax, 7.7, 4.55, 7.7, 3.95)
    ax.text(0.1, 2.5, "b  Physician-facing pipeline (consults, notes, evidence questions)", fontsize=8, weight="bold")
    _box(ax, 0.1, 1.0, 1.3, 0.95, "Physician\nrequest", "white")
    _box(ax, 1.8, 1.6, 2.0, 0.6, "Draft 1: Claude Opus 5.5\nwith web search", llm)
    _box(ax, 1.8, 0.75, 2.0, 0.6, "Draft 2: GPT-6.1 Sol", llm)
    _box(ax, 4.2, 1.0, 1.9, 0.95, "Review (GPT-6.1 Sol):\nerrors, omissions,\nconflicts", llm)
    _box(ax, 6.5, 1.0, 1.6, 0.95, "Final answer\n(Claude Opus 5.5)", llm)
    _box(ax, 8.5, 1.0, 1.4, 0.95, "Shortening step\n(about 2,000\ncharacters)", llm)
    _arrow(ax, 1.4, 1.6, 1.8, 1.9)
    _arrow(ax, 1.4, 1.35, 1.8, 1.05)
    _arrow(ax, 3.8, 1.9, 4.2, 1.6)
    _arrow(ax, 3.8, 1.05, 4.2, 1.35)
    _arrow(ax, 6.1, 1.47, 6.5, 1.47)
    _arrow(ax, 8.1, 1.47, 8.5, 1.47)
    for i, (fc, lab) in enumerate([(det, "code or fine-tuned small model"), (llm, "frontier language model"),
                                   (md, "physician")]):
        ax.add_patch(FancyBboxPatch((0.1 + 3.4 * i, 0.12), 0.25, 0.2, boxstyle="round,pad=0.01", fc=fc, ec="#333"))
        ax.text(0.45 + 3.4 * i, 0.22, lab, va="center", fontsize=6.5)
    check_overlap(fig, "Figure 1")
    fig.savefig(out / "Figure1_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2(out: Path) -> None:
    rep = json.loads((D / "hbpro_test/report.json").read_text())
    cmp_ = _diffs()
    arms = [x for x in ARMS if x[0] in rep]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.12})
    for k, (arm, lab) in enumerate(arms):
        s = rep[arm]["length_adjusted"]
        a.errorbar(s["score"], k, xerr=1.96 * s["se"], fmt="o", color=C_OURS if k == 0 else C_OTHER, ms=4, capsize=2)
        if k == 0:
            b.plot(0, k, marker="D", color=C_OURS, ms=4)
            continue
        d = cmp_[arm]
        b.errorbar(d["diff"], k, xerr=[[d["diff"] - d["ci95"][0]], [d["ci95"][1] - d["diff"]]], fmt="o", color=C_OURS,
                   ms=4, capsize=2)
    a.set_yticks(range(len(arms)), [x[1] for x in arms])
    a.invert_yaxis()
    a.set_xlabel("Length-adjusted score (95% CI)")
    a.set_xlim(34, 86)
    a.set_xticks([40, 50, 60, 70, 80])
    b.set_xlim(-6, 42)
    b.set_xticks([0, 10, 20, 30, 40])
    a.set_title("a  Score on 350 test tasks", loc="left", fontsize=8)
    b.axvline(0, color="#333", lw=0.6)
    b.tick_params(axis="y", left=False)
    b.set_xlabel("Our pipeline minus each system (95% CI)")
    b.set_title("b  Difference from our pipeline", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.14, top=0.92, wspace=0.12)
    check_overlap(fig, "Figure 2")
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
    if PROBLEMS:
        print("\n".join(PROBLEMS))
        raise SystemExit(1)


TEXT_ARMS = [("ours", "Our system"), ("amie_style", "AMIE-style agent"), ("mai_dxo_style", "MAI-DxO-style agent"),
             ("raw_llm", "Our reply model alone"), ("v22", "Rule-based system"), ("gate_alone", "Emergency screen alone")]


def fig3(out: Path) -> None:
    t = json.loads((D / "paper2_final/derived/texts_supplement.json").read_text())
    lk = json.loads((D / "paper2_final/derived/texts_gate_test_and_leak.json").read_text())
    m, ins, sl = t["main"], t["ai_emergency_instructions"], t["same_safety_layer"]["metrics"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    mk = {"ours": "o", "amie_style": "s", "mai_dxo_style": "^", "raw_llm": "D", "v22": "p", "gate_alone": "X"}
    for arm, lab in TEXT_ARMS:
        r = m[arm]
        x, y = r["escalation_no_hazard_per_1000_weighted"], 100 * r["hazard_sensitivity_weighted"]
        xl, xh = r["escalation_no_hazard_weighted_ci"]
        yl, yh = (100 * v for v in r["hazard_sensitivity_weighted_ci"])
        a.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt=mk[arm], ms=5, capsize=2, lw=0.8,
                   color=C_OURS if arm == "ours" else C_OTHER, label=lab)
    for k, lab in [("amie_style+layer", "AMIE-style agent + safety layer"),
                   ("mai_dxo_style+layer", "MAI-DxO-style agent + safety layer"), ("v22+layer", "Rule-based system + safety layer")]:
        r = sl[k]
        x, y = r["escalation_no_hazard_per_1000_weighted"], 100 * r["hazard_sensitivity_weighted"]
        xl, xh = r["escalation_no_hazard_weighted_ci"]
        yl, yh = (100 * v for v in r["hazard_sensitivity_weighted_ci"])
        a.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt=mk[k.split("+")[0]], ms=5,
                   mfc="white", mec="#555555", ecolor="#bbbbbb", capsize=2, lw=0.8, label=lab)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker=mk[k], ms=5, color=C_OURS if k == "ours" else C_OTHER, label=lab)
          for k, lab in TEXT_ARMS]
    hs += [Line2D([], [], ls="none", marker=mk[k], ms=5, mfc="white", mec="#555555", label=lab + " + safety layer")
           for k, lab in TEXT_ARMS if k in ("amie_style", "mai_dxo_style", "v22")]
    a.legend(handles=hs, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a.set_xlim(-10, 340)
    a.set_ylim(20, 82)
    a.set_xlabel("Escalations per 1,000 texts without a hazard (95% CI)")
    a.set_ylabel("Hazard sensitivity, % (95% CI)")
    a.set_title("a  Hazard sensitivity and escalation burden", loc="left", fontsize=8)
    arms = [("ours", "Our system"), ("raw_llm", "Our reply model alone"), ("v22", "Rule-based system"),
            ("amie_style", "AMIE-style agent"), ("mai_dxo_style", "MAI-DxO-style agent")]
    pat = [0.0 if k == "ours" else ins[k]["unconditional_instruction_per_1000_replies"] for k, _ in arms]
    em = json.loads((D / "paper2_final/derived/texts_routing_endpoints.json").read_text())[
        "emergency_now_reaching_patient_per_1000_all_replies"]
    cls = [em[k] for k, _ in arms]
    yy = range(len(arms))
    b.barh([y - 0.2 for y in yy], pat, height=0.38, color="#bbbbbb", label="filter text patterns (unconditional)")
    b.barh([y + 0.2 for y in yy], cls, height=0.38, color=C_OURS, label="classifier (emergency care now)")
    for y, v1, v2 in zip(yy, pat, cls):
        b.text(v1 + 2, y - 0.2, f"{v1:.1f}", va="center", fontsize=6)
        b.text(v2 + 2, y + 0.2, f"{v2:.1f}", va="center", fontsize=6)
    b.set_yticks(list(yy), [x[1] for x in arms])
    b.invert_yaxis()
    b.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    b.set_xlabel("Replies per 1,000 that reached the patient")
    b.set_xlim(0, 175)
    b.set_title("b  Emergency advice reaching the patient", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.36, top=0.93, wspace=0.55)
    check_overlap(fig, "Figure 3")
    fig.savefig(out / "Figure3_real_texts.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "Figure3_real_texts.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
