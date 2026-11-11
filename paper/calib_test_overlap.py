"""Patient overlap between the 497 calibration texts (care-team texts) and the 949 test texts (Compass texts).

Link: test text -> messenger contact -> member_contacts -> members.waymark_id -> lighthouse "Patient".id; calibration
texts carry lighthouse "Patient".id directly (lighthouse_set/pool.jsonl). Every member on a shared phone counts as a
possible sender, so the overlap is an upper bound. Out: paper2_final/derived/calib_test_overlap.json (counts only).
Needs WireGuard and Vault (waymark-data-access). Usage: python -m paper.calib_test_overlap
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, messenger, query  # noqa: E402

D = pathlib.Path(__file__).resolve().parents[2] / "compass-pcp-data"


def _rows(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    tid = {r["id"] for r in _rows(D / "fresh_set/crosswalk.jsonl")}
    tpool = {r["id"]: r for r in _rows(D / "fresh_set/pool.jsonl") if r["id"] in tid}
    cid = {r["id"] for r in _rows(D / "lighthouse_set/crosswalk.jsonl")}
    cpat = {str(r["patient_id"]) for r in _rows(D / "lighthouse_set/pool.jsonl") if r["id"] in cid}
    mc = query(messenger("prod"), """
        select mc.contact_id::text c, mm.waymark_id::text wid
        from public.member_contacts mc join public.members mm on mm.id = mc.member_id
        where mc.contact_id = any(cast(:ids as uuid[]))""",
               ids="{" + ",".join({r["contact_id"] for r in tpool.values()}) + "}")
    wids = sorted({r.wid for r in mc.itertuples() if r.wid})
    pt = query(lighthouse("prod"), 'select id::text pid, "waymarkPatientNumber"::text k from public."Patient" '
                                   'where "waymarkPatientNumber"::text = any(:ids)', ids=wids)
    w2p = {r.k: r.pid for r in pt.itertuples()}
    c2p: dict[str, set] = {}
    for r in mc.itertuples():
        if r.wid in w2p:
            c2p.setdefault(r.c, set()).add(w2p[r.wid])
    test_texts_linked = sum(1 for r in tpool.values() if r["contact_id"] in c2p)
    shared = [i for i, r in tpool.items() if c2p.get(r["contact_id"], set()) & cpat]
    out = {"calibration_texts": len(cid), "calibration_patients": len(cpat), "test_texts": len(tpool),
           "test_texts_linked_to_a_patient": test_texts_linked,
           "test_texts_whose_possible_sender_also_sent_a_calibration_text": len(shared),
           "note": "upper bound: every member on a shared phone counts as a possible sender"}
    (D / "fresh_set/calib_overlap_ids.json").write_text(json.dumps(sorted(shared)))  # custodian storage only
    (D / "paper2_final/derived/calib_test_overlap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
