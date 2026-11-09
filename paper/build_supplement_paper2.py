"""Assemble the Paper 2 Supplementary Information from its parts (no patient data).

Parts: manuscript/supplement_notes.md (hand-written notes and Supplementary Table 1), Supplementary Note 2 generated
here from the prompt constants in code (so the printed prompts are the run prompts), supplement_tables_generated.md
(paper/make_supplement_tables_paper2.py) and the checklists in manuscript/checklists/.
Out: manuscript/compass_clinician_first_ai_pcp_npj_supplement.md
Usage: python -m paper.build_supplement_paper2
"""

from pathlib import Path

M = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/manuscript"


def note2() -> str:
    from compass_pcp.arms import baselines as b
    from compass_pcp.arms import intake as i
    from compass_pcp.arms import pro as p
    from paper import disposition_check as d
    from paper.sealed_comparators import CLASSIFIER_PROMPT

    def block(title: str, text: str) -> str:
        return f"*{title}.*\n\n```text\n{text.strip()}\n```\n"

    parts = ["## Supplementary Note 2. Prompts and rules as run\n",
             "The text below is read from the source files at build time.\n",
             block("Physician-facing pipeline, draft prompt (both drafts)", p.DRAFT_SYS),
             block("Physician-facing pipeline, web search note appended to Draft A", p.SEARCH_NOTE),
             block("Physician-facing pipeline, review prompt (GPT-6.1 Sol)", p.REVIEW_SYS),
             block("Physician-facing pipeline, final writer prompt (Claude Opus 5.5)", p.FINAL_SYS),
             block("Physician-facing pipeline, length editor prompt (Claude Opus 5.5)", p.EDIT_SYS),
             block("Shared output limits (appended to the persona and the comparator prompts)", b.OUTPUT_LIMITS),
             block("Patient-facing writer persona (PCP_PERSONA)", b.PCP_PERSONA),
             block("AMIE-style re-implementation, analysis step", b.AMIE_ANALYZE),
             block("AMIE-style re-implementation, draft step", b.AMIE_DRAFT),
             block("AMIE-style re-implementation, refinement step", b.AMIE_REFINE),
             block("MAI-DxO-style re-implementation, panel decision step", b.PANEL_DECIDE),
             block("Intake agent, system prompt", i.SYSTEM),
             block("Urgent-page text (patient writes about themself; draft written by a physician author)", i.URGENT_PAGE_SELF),
             block("Reply-disposition classifier (Claude Sonnet 5.5; second classifier GPT-6.1 Sol)", CLASSIFIER_PROMPT),
             block("Clinician-first check, instruction patterns (regular expressions)",
                   "\n".join(x.pattern for x in d.PATTERNS)),
             block("Clinician-first check, conditional words (deployable policy)", d._CONDITIONAL.pattern)]
    roles = [k for k in dir(b) if k.startswith("PANEL_") and k != "PANEL_DECIDE" and isinstance(getattr(b, k), (str, tuple, dict))]
    for k in roles:
        v = getattr(b, k)
        parts.append(block(f"MAI-DxO-style re-implementation, {k}", v if isinstance(v, str) else repr(v)))
    return "\n".join(parts)


def main() -> None:
    head = (M / "supplement_notes.md").read_text()
    tables = (M / "supplement_tables_generated.md").read_text()
    ck = M / "checklists"
    checks = ""
    for name in ["TRIPOD-LLM_checklist.md", "CHART_checklist.md", "STARD2015_checklist.md"]:
        f = ck / name
        if f.exists():
            checks += "\n\n" + f.read_text()
    out = (head.replace("<<NOTE2>>", note2()).replace("<<TABLES>>", "## Supplementary Tables 2–13\n\n" + tables)
           .replace("<<CHECKLISTS>>", "## Supplementary Note 4. Reporting checklists" + (checks or "\n\n[checklists pending]")))
    (M / "compass_clinician_first_ai_pcp_npj_supplement.md").write_text(out)
    print("written", len(out.split()), "words")


if __name__ == "__main__":
    main()
