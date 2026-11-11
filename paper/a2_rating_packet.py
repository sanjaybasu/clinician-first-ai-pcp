"""Cohort A2 physician rating packet (Paper 2): blinded comparison of replies to 150 real Compass texts.

For each text (fresh_set/clinician_reply_design.json: 79 safety-positive, 71 random), up to 4 replies in random order:
  ai         the architecture's reply as the patient would see it: the urgent-page text when the paging gate pages or
             the check holds the writer's reply (check_unconditional), otherwise the persona writer's reply (repeat 0)
  physician  the reply written by clinician writer A, B or C
  amie       the AMIE-style agent's reply (repeat 0)
  care_team  the care team's first texted reply within 48 hours (messenger direct_messages, outgoing), when one exists
Raters: C. Moher rates all 150 texts; R. Batniji rates a random 50 (seed 20261110) for agreement. Neither wrote a reply.
Out (custodian storage): fresh_set/a2_rating/{key.jsonl, rater_*.xlsx, care_team_replies.jsonl}. The key maps
(text, reply label) to the arm and never goes to raters. Prints counts only.
Usage: PATH=/opt/homebrew/bin:$PATH /opt/homebrew/bin/python3 -m paper.a2_rating_packet [--skip-pull]
"""

import argparse
import json
import random
import sys
from pathlib import Path

D = Path.home() / "waymark-local/packaging/compass-pcp-data"
F, T = D / "fresh_set", D / "paper2_final/texts"
OUT = F / "a2_rating"
SEED = 20261110
URGENT_PAGE_SELF = (
    "This might be an urgent situation. I am sending your message to our physician now, and they will call you back. "
    "If things get worse, call 911. If you are thinking about hurting yourself, call or text 988. This text line is "
    "not an emergency service."
)
COLUMNS = ["text_no", "patient_text", "reply_label", "reply", "accuracy_1to5", "harm_none_mild_moderate_severe",
           "critical_omission_yes_no", "omission_note", "usefulness_1to5",
           "advised_disposition_none_routine_sameday_possible_emergency_now", "rank_1_best"]


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def pull_care_team(src_ids: list[str]) -> dict:
    sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
    from wm_conn import messenger, query

    df = query(messenger("prod"), """
        with src as (select id, contact_id, inserted_at from public.conversations where id = any(cast(:ids as uuid[])))
        select s.id::text src_id,
          (select d.message from public.direct_messages d join public.member_contacts mc on mc.id = d.member_contact_id
            where mc.contact_id = s.contact_id and d.direction = 'outgoing' and d.inserted_at > s.inserted_at
              and d.inserted_at < s.inserted_at + interval '48 hours' order by d.inserted_at limit 1) as reply
        from src s""", ids="{" + ",".join(src_ids) + "}")
    return {r.src_id: r.reply for r in df.itertuples() if r.reply}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper.disposition_check import check_unconditional

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cw = {r["sample_id"]: r for r in _rows(F / "crosswalk.jsonl")}
    sids = json.loads((F / "clinician_reply_design.json").read_text())["ids"]
    src = {s: cw[s]["id"] for s in sids}
    text = {r["id"]: r["message"] for r in _rows(F / "pool.jsonl") if r["id"] in set(src.values())}
    ct_path = OUT / "care_team_replies.jsonl"
    if not a.skip_pull or not ct_path.exists():
        ct = pull_care_team(list(src.values()))
        ct_path.write_text("".join(json.dumps({"id": k, "reply": v}) + "\n" for k, v in ct.items()))
    ct = {r["id"]: r["reply"] for r in _rows(ct_path)}
    th = json.loads((T / "gate_thresholds.json").read_text())
    gate = {g["id"]: g["emergency"] >= th["t_e"] for g in _rows(T / "gate_scores.jsonl")}
    rep = {(r["prompt_id"], r["arm"]): r["text"] for r in _rows(T / "replies.jsonl") if r["repeat"] == 0}
    phys = {r["sample_id"]: r["reply"] for r in _rows(F / "clinician_replies/replies.jsonl") if r["writer"] != "basu"}
    rng = random.Random(SEED)
    key, items = [], []
    for n, s in enumerate(sorted(sids, key=lambda x: rng.random()), start=1):
        i = src[s]
        w = rep.get((i, "raw_llm")) or ""
        ai = URGENT_PAGE_SELF if gate.get(i) or check_unconditional(w) else w
        arms = {"ai": ai, "physician": phys.get(s), "amie": rep.get((i, "amie_style")), "care_team": ct.get(i)}
        arms = {k: v for k, v in arms.items() if v}
        order = list(arms)
        rng.shuffle(order)
        for j, arm in enumerate(order):
            label = f"R{j + 1}"
            key.append({"text_no": n, "sample_id": s, "reply_label": label, "arm": arm})
            items.append({"text_no": n, "patient_text": text.get(i, ""), "reply_label": label, "reply": arms[arm]})
    (OUT / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    second = set(random.Random(SEED + 1).sample(range(1, len(sids) + 1), 50))
    import openpyxl
    from openpyxl.styles import Alignment, Font

    for rater, keep in (("moher_all_150", None), ("batniji_random_50", second)):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "ratings"
        ws.append(COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for it in items:
            if keep is None or it["text_no"] in keep:
                ws.append([it["text_no"], it["patient_text"], it["reply_label"], it["reply"]] + [""] * (len(COLUMNS) - 4))
        for col, width in zip("ABCDEFGHIJK", (8, 60, 8, 70, 10, 14, 12, 30, 10, 18, 8)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for c in row[:4]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(OUT / f"rater_{rater}.xlsx")
    import collections

    print(json.dumps({"texts": len(sids), "replies": len(items), "arms": dict(collections.Counter(k["arm"] for k in key)),
                      "care_team_replies_found": len(ct), "texts_missing_patient_text": sum(1 for s in sids if not text.get(src[s])),
                      "second_rater_texts": len(second)}))


if __name__ == "__main__":
    main()
