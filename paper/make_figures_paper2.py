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
ARMS = [("compass_pro_full", "Multi-model pipeline"), ("opus_search_edit", "Two-step pipeline"),
        ("opus_edit", "Two-step pipeline, no web search"),
        ("compass_pro_full_blk", "Multi-model pipeline, blocked search†"), ("opus_search_edit_blk", "Two-step pipeline, blocked search†"),
        ("amie_style", "AMIE-style agent"),
        ("amie_style_search", "AMIE-style agent, web search"), ("amie_style_uncapped", "AMIE-style agent, 12,000 tokens"),
        ("amie_style_search_blk", "Tuned AMIE-style agent, blocked search†"),
        ("mai_dxo_style", "MAI-DxO-style agent"), ("mai_dxo_style_uncapped", "MAI-DxO-style agent, 12,000 tokens"),
        ("mai_dxo_style_search_edit_blk", "Tuned MAI-DxO-style agent, blocked search†"),
        ("opus55", "Claude Opus 5.5, no system prompt"), ("gpt61sol", "GPT-6.1 Sol, no system prompt"),
        ("physician", "Physicians (dataset responses)")]


def _diffs() -> dict:
    """compass_pro_full minus each arm, from every compare file (reversed pairs negated); for arms with blocked search,
    compass_pro_full_blk minus the arm (and compass_pro_full minus compass_pro_full_blk), from addendum_hb.json."""
    out = {}
    ad = D / "paper2_final/derived/addendum_hb.json"
    if ad.exists():
        h = json.loads(ad.read_text())
        for k, v in h.get("pipeline_blk_minus", {}).items():
            out[k] = v
        v = h["all_tasks"]["B_blocked_minus_original"]["compass_pro_full_blk - compass_pro_full"]
        out["compass_pro_full_blk"] = {"diff": -v["diff"], "ci95": [-v["ci95"][1], -v["ci95"][0]]}
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
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.0)
    ax.axis("off")
    det, llm, md = "#e8eef6", "#fdf1dc", "#e3f1e3"
    # a. physician-facing test
    ax.text(0.05, 9.75, "a  Physician-facing test: does the agent design explain benchmark gains?", fontsize=8, weight="bold")
    _box(ax, 0.1, 8.2, 1.3, 1.1, "Physician\nrequest\n(HealthBench\nProfessional)", "white")
    _box(ax, 1.8, 8.75, 1.9, 0.55, "Draft 1: Claude Opus\n5.5, web search", llm)
    _box(ax, 1.8, 8.1, 1.9, 0.55, "Draft 2: GPT-6.1 Sol", llm)
    _box(ax, 4.0, 8.2, 1.7, 1.1, "Review (GPT-6.1 Sol):\nerrors, omissions,\nconflicts", llm)
    _box(ax, 6.0, 8.2, 1.6, 1.1, "Final answer and\nshortening step\n(Claude Opus 5.5)", llm)
    ax.text(7.85, 9.15, "Multi-model pipeline", fontsize=6.5, weight="bold")
    ax.text(7.85, 8.75, "versus", fontsize=6.5)
    ax.text(7.85, 8.35, "AMIE-style agent: analyze,\ndraft, refine own draft", fontsize=6.2, va="center")
    ax.text(7.85, 7.85, "MAI-DxO-style agent: panel\nof physician roles, decide", fontsize=6.2, va="center")
    for y0, y1 in [(8.75, 9.02), (8.75, 8.37)]:
        _arrow(ax, 1.4, y0, 1.8, y1)
    _arrow(ax, 3.7, 9.02, 4.0, 8.9)
    _arrow(ax, 3.7, 8.37, 4.0, 8.5)
    _arrow(ax, 5.7, 8.75, 6.0, 8.75)
    _box(ax, 0.1, 6.85, 9.8, 0.65, "Matched comparisons at four levels:  base agents  |  same web search  |\nsame tuning, with benchmark sources blocked from search  |  equal cost", "#f2f2f2")
    # b. patient-facing test
    ax.text(0.05, 6.55, "b  Patient-facing test: does a safety layer outside the reply model find more hazards?", fontsize=8,
            weight="bold")
    ax.text(0.1, 6.15, "Integrated design (AMIE-style, MAI-DxO-style, rule-based system)", fontsize=6.8, style="italic")
    _box(ax, 0.1, 5.25, 1.3, 0.65, "Patient text", "white")
    _box(ax, 1.8, 5.25, 4.3, 0.65, "One language model judges urgency\nand writes the reply", llm)
    _box(ax, 6.5, 5.25, 1.3, 0.65, "Reply sent", "white")
    _arrow(ax, 1.4, 5.57, 1.8, 5.57)
    _arrow(ax, 6.1, 5.57, 6.5, 5.57)
    ax.text(0.1, 4.85, "Separated design: reply model plus a safety layer (shaded blue) outside it", fontsize=6.8, style="italic")
    _box(ax, 0.1, 3.75, 1.3, 0.8, "Patient text", "white")
    _box(ax, 1.8, 3.75, 2.1, 0.8, "Emergency screen\n(fine-tuned classifier;\nconformal thresholds)", det)
    _box(ax, 4.3, 3.75, 1.8, 0.8, "Reply model\n(frontier model,\nno clinical rules)", llm)
    _box(ax, 6.5, 3.75, 1.9, 0.8, "Emergency-instruction\nfilter (text rules)", det)
    _box(ax, 8.8, 3.75, 1.1, 0.8, "Reply\nsent", "white")
    for x0, x1 in [(1.4, 1.8), (3.9, 4.3), (6.1, 6.5), (8.4, 8.8)]:
        _arrow(ax, x0, 4.15, x1, 4.15)
    _box(ax, 0.1, 2.55, 3.8, 0.8, "Review request: a physician reviews the\ntext; the AI reply is still sent", md)
    _box(ax, 4.3, 2.55, 5.6, 0.8, "Alert (emergency probability above the alert\nthreshold, or reply held by the filter): a physician\ncalls back; the patient gets a fixed text", md)
    _arrow(ax, 2.4, 3.75, 2.2, 3.35)
    _arrow(ax, 3.4, 3.75, 5.0, 3.35)
    _arrow(ax, 7.45, 3.75, 7.45, 3.35)
    ax.text(0.1, 2.25, "The same safety layer is also added to each integrated design.", fontsize=6.5)
    # c. transfer test
    ax.text(0.05, 1.8, "c  Transfer test: do the safety layer's thresholds hold in a new service?", fontsize=8, weight="bold")
    _box(ax, 0.1, 0.75, 3.0, 0.75, "497 texts to care teams\n(answered by staff),\nbefore December 2025", "white")
    _box(ax, 3.5, 0.75, 2.6, 0.75, "Thresholds set by conformal\nrisk control, then fixed", det)
    _box(ax, 6.5, 0.75, 3.4, 0.75, "949 later texts to Compass\n(Waymark's AI text-message chatbot),\nDecember 2025 to October 2026", "white")
    _arrow(ax, 3.1, 1.12, 3.5, 1.12)
    _arrow(ax, 6.1, 1.12, 6.5, 1.12)
    for i, (fc, lab) in enumerate([(det, "safety layer: classifier or text rules"), (llm, "frontier language model"),
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
    b.set_xlabel("Multi-model pipeline minus each system (95% CI)")
    b.set_title("b  Difference from the multi-model pipeline", loc="left", fontsize=8)
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


TEXT_ARMS = [("ours", "Separated design"), ("amie_style", "AMIE-style agent"), ("mai_dxo_style", "MAI-DxO-style agent"),
             ("raw_llm", "Reply model alone"), ("v22", "Rule-based system"), ("gate_alone", "Emergency screen alone")]


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
    arms = [("ours", "Separated design"), ("raw_llm", "Reply model alone"), ("v22", "Rule-based system"),
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
