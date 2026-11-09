"""AMD-PCP-002 Analysis 7: masking rule and sample draw."""

from collections import Counter
from pathlib import Path

from paper.masked_rerating_packet import N, SEED, instructions, mask, select


def test_mask_phrases():
    cases = {
        "A clinician on your care team will contact you today.": "A clinician will contact you today.",
        "A nurse or doctor from our care team will call.": "A clinician will call.",
        "Someone from his care team will call you.": "A clinician will call you.",
        "Our care team can help.": "A clinician can help.",
        "Ask your health care team or pharmacist.": "Ask a clinician or pharmacist.",
        "A doctor on your child's care team will review it.": "A clinician will review it.",
        "Call the nurse advice line tonight.": "Call a clinician tonight.",
        "Your care coordinator will follow up.": "A clinician will follow up.",
        "The Waymark team will reach out.": "A clinician will reach out.",
        "Our team can book it.": "A clinician can book it.",
        "a remote-area nurse, police officer": "a clinician, police officer",
    }
    for src, want in cases.items():
        got, counts = mask(src)
        assert got == want, (src, got)
        assert sum(counts.values()) >= 1


def test_mask_leaves_other_text():
    for s in ("See your doctor this week.", "Team sports are fine.", "Use team-based care.", "Ask a pharmacist."):
        assert mask(s) == (s, {k: 0 for k in mask("")[1]})


def test_select_excludes_registered_and_is_stratified():
    ids = [f"p{i:04d}" for i in range(1000)]
    themes = {p: ("t1" if i < 600 else "t2" if i < 900 else "t3") for i, p in enumerate(ids)}
    exclude = set(ids[:100])
    picked, _ = select(ids, themes, exclude)
    assert len(picked) == N == len(set(picked)) and not set(picked) & exclude
    c = Counter(themes[p] for p in picked)
    assert c == {"t1": 28, "t2": 17, "t3": 5}  # 500/900, 300/900, 100/900 of 50, largest remainders
    assert select(ids, themes, exclude, seed=SEED)[0] == picked


def test_instructions_counts():
    reg = Path.home() / "waymark-local/notebooks/compass-ai-pcp/physician_scoring/PHYSICIAN_SCORING_INSTRUCTIONS.md"
    if not reg.exists():
        return
    out = instructions(reg.read_text(), 50, 110)
    assert "numbered 1 to 50" in out and "220 marks" in out and "100" not in out.split("## Rules")[0]
