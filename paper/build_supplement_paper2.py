"""Assemble the Paper 2 Supplementary Information from its parts (no patient data).

Parts: manuscript/supplement_notes.md (hand-written notes and Supplementary Table 1), Supplementary Note 2 generated
here from the prompt constants in code (so the printed prompts are the run prompts), supplement_tables_generated.md
(paper/make_supplement_tables_paper2.py) and the checklists in manuscript/checklists/.
Out: manuscript/compass_clinician_first_ai_pcp_npj_supplement.md
Usage: python -m paper.build_supplement_paper2
"""

from pathlib import Path

M = Path.home() / "waymark-local/notebooks/compass-pcp-paper2/manuscript"


import re

# Supplementary Table numbers in order of first mention in the main text (generator numbering -> published numbering)
TABLE_MAP = {1: 1, 4: 2, 5: 3, 2: 4, 16: 5, 3: 6, 12: 7, 7: 8, 6: 9, 11: 10, 8: 11, 17: 12, 15: 13, 14: 14, 9: 15, 13: 19, 10: 20}
CHECKLISTS = [(16, "TRIPOD-LLM_checklist.md", "TRIPOD-LLM reporting checklist"),
              (17, "CHART_checklist.md", "CHART reporting checklist"),
              (18, "STARD2015_checklist.md", "STARD 2015 reporting checklist")]
HB_TABLES = {2, 3, 4, 5, 16}  # generator numbers of the HealthBench Professional tables
ARM_NAMES = {"amie_style+layer": "the AMIE-style agent with the safety layer",
             "mai_dxo_style+layer": "the MAI-DxO-style agent with the safety layer",
             "v22+layer": "the rule-based system with the safety layer", "amie_style": "the AMIE-style agent",
             "mai_dxo_style": "the MAI-DxO-style agent", "v22": "the rule-based system", "raw_llm": "the reply model alone"}
TERMS = [  # display terms for the reader; one term per concept, as in the main text
    (r" \((?:paper|scripts)/[\w./]+\)", ""), (r"\s*\((?:v22|version 22)\)", ""),
    ("Our architecture, registered check", "Separated design with the registered filter"),
    ("Persona-only writer (= our architecture)", "Reply model (= separated design)"),
    ("Writer + same gate + same check", "System with the safety layer"),
    ("Writer alone (no gate)", "Reply model with the filter, without the emergency screen"),
    ("Opus 5.5, primary care persona", "Reply model alone"), ("Registered rule-based system", "Rule-based system"),
    ("Rule-based writer", "Rule-based system"), ("registered system", "rule-based system"),
    ("Paging gate alone", "Emergency screen alone"), ("paging-gate", "emergency-screen"),
    ("Paging gate", "Emergency screen"), ("paging gate", "emergency screen"), ("paging thresholds", "emergency screen's thresholds"),
    ("The gate", "The emergency screen"), ("the gate", "the emergency screen"), (r"\bgate\b", "emergency screen"),
    ("Clinician-first check", "Emergency-instruction filter"), ("clinician-first check", "emergency-instruction filter"),
    ("Registered check", "Registered filter"), ("registered check", "registered filter"), ("deployable check", "filter"),
    ("The check", "The filter"), ("the check", "the filter"), ("deployable", "final"),
    ("Our architecture", "Separated design"), ("our architecture", "the separated design"), ("The architecture", "The separated design"),
    ("the architecture", "the separated design"), (r"\barchitecture\b", "design"),
    (r"(\d+) safety-positive (real |test |patient )?texts", r"\1 \2texts with a safety hazard"),
    ("safety-positive texts", "texts with a safety hazard"), ("Safety-positive", "With a safety hazard"),
    ("safety-positive", "with a safety hazard"), ("safety positive", "safety hazard"),
    ("Emergency-enriched texts", "Calibration texts"), ("emergency-enriched texts", "calibration texts"),
    ("emergency-enriched", "calibration"), ("Emergency-enriched", "Calibration"),
    ("persona-only", "reply-model-only"), ("Persona-only", "Reply-model-only"),
    ("writer replies", "replies from the reply model"), ("Writers", "Reply models"), ("writers", "reply models"),
    ("Writer", "Reply model"), ("writer", "reply model"), ("length editor", "shortening step"), ("Length editor", "Shortening step"),
    ("care-team texts", "texts to care teams"), ("Real patient texts", "Patient texts"), ("real patient texts", "patient texts"),
    ("real texts", "patient texts"), ("Real texts", "Patient texts"), ("real-text", "patient-text"), ("Real-text", "Patient-text"),
    ("Paged", "Alerted"), (r"\bpaged\b", "alerted"), ("pages the physician", "alerts the physician"),
    ('Results "Real patient texts"', 'Results "Patient texts"'), ('"Paging gate"', '"Transfer of the emergency screen"'),
    ('"The architecture"', '"Designs studied"'), ('"Setting and data"', '"Setting and data partitions"'),
    ('Methods "HealthBench Professional"', 'Methods "Physician-facing tasks"'),
    ('Results "HealthBench Professional"', 'Results "Physician-facing tasks"'),
    (r"Abstract sentences? [0-9–, and]+", "Abstract"),
    ("judge check", "judge validation"), ("Architecture prompts", "Our prompts"), ("Gate training", "Emergency screen training"),
    ("receives pages", "receives alerts"), ("page and review", "alert and review"), ("page or review", "alert or review"),
    ("page threshold", "alert threshold"), ("page rates", "alert rates"), ("miss and page", "miss and alert"),
    ("emergency screen page", "emergency screen alert"),
    ("Alerted, enriched sample", "Alerted, calibration sample"), ("enriched 349", "flagged 349"),
    ("enriched sampling of the patient texts", "sampling of flagged patient texts"), ("the enriched sampling", "the sampling of flagged texts"),
    (r"\bcheck patterns\b", "filter patterns"), (r"\bfinal check\b", "final filter"), (r"\band check\)", "and filter)"),
    (r"\bchecks?\b(?! the)", "filter"),
]


def _arm(m: re.Match) -> str:
    return ARM_NAMES.get(m.group(1), m.group(1))


def reader_terms(text: str, hb: bool = False) -> str:
    text = re.sub(r"\bours - (\w+\+?\w*)", lambda m: ("Multi-model pipeline" if hb else "Separated design") + " minus " + _arm(m), text)
    text = re.sub(r"\b(amie_style\+layer|mai_dxo_style\+layer|v22\+layer|amie_style|mai_dxo_style|raw_llm)\b", _arm, text)
    text = re.sub(r"\bOurs\b", "Multi-model pipeline" if hb else "Separated design", text)
    text = re.sub(r"\bours\b", "the multi-model pipeline" if hb else "the separated design", text)
    for a, b in TERMS:
        text = re.sub(a, b, text) if any(c in a for c in "\\()[]") or a.startswith("\\b") else text.replace(a, b)
    return re.sub(r"\bv22\b", "the rule-based system", text)


def renumber_refs(text: str) -> str:
    """Map 'Supplementary Table(s) N[, M][–K]' from generator numbering to published numbering."""
    def one(m: re.Match) -> str:
        nums = []
        for part in re.split(r",\s*|\s+and\s+", m.group(2)):
            if "–" in part:
                x, y = map(int, part.split("–"))
                nums += range(x, y + 1)
            elif part.strip():
                nums.append(int(part))
        new = sorted({TABLE_MAP.get(n, n) for n in nums})
        runs, i = [], 0
        while i < len(new):
            j = i
            while j + 1 < len(new) and new[j + 1] == new[j] + 1:
                j += 1
            runs += [str(n) for n in new[i:j + 1]] if j - i < 2 else [f"{new[i]}–{new[j]}"]
            i = j + 1
        word = "Supplementary Tables" if len(new) > 1 else "Supplementary Table"
        body = ", ".join(runs[:-1]) + (" and " if len(runs) > 1 else "") + runs[-1]
        return f"{word} {body}"
    text = re.sub(r"(Supplementary Tables? \d+)-(\d+)", r"\1–\2", text)
    return re.sub(r"Supplementary (Tables?) ((?:\d+(?:–\d+)?)(?:(?:,\s*|\s+and\s+)\d+(?:–\d+)?)*)", one, text)


def reflow(text: str) -> str:
    """Join hard-wrapped continuation lines of a prompt for display; keep blank lines, list items and JSON lines."""
    out = []
    for line in text.strip().split("\n"):
        cont = line.startswith("  ") or (line[:1].islower() and out and not out[-1].rstrip().endswith((":", ".", "{", "}")))
        if out and cont and out[-1].strip() and not line.strip().startswith(("{", "}", "-", "\"")):
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def note2() -> str:
    from compass_pcp.arms import baselines as b
    from compass_pcp.arms import intake as i
    from compass_pcp.arms import pro as p
    from paper import disposition_check as d
    from paper.sealed_comparators import CLASSIFIER_PROMPT

    def block(title: str, text: str) -> str:
        return f"*{title}.*\n\n```text\n{reflow(text)}\n```\n"

    parts = ["## Supplementary Note 2. Prompts and rules as run\n",
             "The text below is read from the source code at build time. Line breaks inside a sentence are joined for display.\n",
             block("Physician-facing pipeline, draft prompt (both drafts)", p.DRAFT_SYS),
             block("Physician-facing pipeline, web search note added to draft 1", p.SEARCH_NOTE),
             block("Physician-facing pipeline, review prompt (GPT-6.1 Sol)", p.REVIEW_SYS),
             block("Physician-facing pipeline, final answer prompt (Claude Opus 5.5)", p.FINAL_SYS),
             block("Physician-facing pipeline, shortening step prompt (Claude Opus 5.5)", p.EDIT_SYS),
             block("Shared output limits (added to the reply model and agent prompts)", b.OUTPUT_LIMITS),
             block("Reply model, primary care persona prompt", b.PCP_PERSONA),
             block("AMIE-style re-implementation, analysis step", b.AMIE_ANALYZE),
             block("AMIE-style re-implementation, draft step", b.AMIE_DRAFT),
             block("AMIE-style re-implementation, refinement step", b.AMIE_REFINE),
             block("MAI-DxO-style re-implementation, panel decision step", b.PANEL_DECIDE),
             block("Intake agent, system prompt", i.SYSTEM),
             block("Alert text sent to the patient (written by a physician author)", i.URGENT_PAGE_SELF),
             block("Reply-disposition classifier (Claude Sonnet 5.5; second classifier GPT-6.1 Sol)", CLASSIFIER_PROMPT),
             block("Emergency-instruction filter, instruction patterns (regular expressions)",
                   "\n".join(x.pattern for x in d.PATTERNS)),
             block("Emergency-instruction filter, condition words", d._CONDITIONAL.pattern)]
    roles = [k for k in dir(b) if k.startswith("PANEL_") and k != "PANEL_DECIDE" and isinstance(getattr(b, k), (str, tuple, dict))]
    for k in roles:
        v = getattr(b, k)
        parts.append(block("MAI-DxO-style re-implementation, " + k.replace("PANEL_", "panel ").lower().replace("_", " "),
                                v if isinstance(v, str) else repr(v)))
    return "\n".join(parts)


def main() -> None:
    head = (M / "supplement_notes.md").read_text()
    raw = (M / "supplement_tables_generated.md").read_text()
    blocks = {}
    for chunk in re.split(r"(?m)^(?=Supplementary Table \d+ \|)", raw):
        m = re.match(r"Supplementary Table (\d+) \|(.*)", chunk, re.S)
        if m:
            n = int(m.group(1))
            body = renumber_refs(reader_terms(m.group(2), hb=n in HB_TABLES))
            first, _, rest = body.strip().partition("\n")
            t, dot, more = first.partition(". ")
            blocks[TABLE_MAP[n]] = (f"## Supplementary Table {TABLE_MAP[n]}. {t.rstrip('.')}\n\n" + (more + "\n" if more else "")
                                    + rest.rstrip() + "\n")
    for n, fname, title in CHECKLISTS:
        t = (M / "checklists" / fname).read_text()
        t = re.sub(r"(?m)^# .*\n", "", t, count=1)
        t = re.sub(r"(?m)^## ", "### ", t)
        blocks[n] = f"## Supplementary Table {n}. {title}\n\n" + reader_terms(t).strip() + "\n"  # checklists cite published numbers
    tables = "\n".join(blocks[k] for k in sorted(blocks) if k != 1)
    out = head.replace("<<NOTE2>>", note2()).replace("<<TABLES>>", tables)
    (M / "compass_clinician_first_ai_pcp_npj_supplement.md").write_text(out)
    print("written", len(out.split()), "words; tables", sorted(blocks))


if __name__ == "__main__":
    main()
