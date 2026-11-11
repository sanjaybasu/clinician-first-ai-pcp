"""Sender characteristics for the 949 fresh test texts (Paper 2 subgroup analysis; custodian storage only).

Link: test text -> messenger contact -> member_contacts -> messenger members (state, birthday, waymark_id) ->
lighthouse "Patient" on "waymarkPatientNumber" (gender, race). A text's sender is known when its phone belongs to one
member; for a shared phone (several members) the sender is unknown, and state is kept only when all members share it.
Age is in whole years on the date the text was sent. Race is the first listed value; Hispanic or Latinx is recorded
as a race category in this source, and ethnicity is not recorded.
Out: compass-pcp-data/fresh_set/sender_demographics.jsonl {id, sender_known, state, sex, age, race} (no names, no text).
Prints counts only. Needs WireGuard and Vault (waymark-data-access). Usage: /opt/homebrew/bin/python3 -m paper.sender_demographics
"""

import collections
import json
import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, messenger, query  # noqa: E402

F = pathlib.Path(__file__).resolve().parents[2] / "compass-pcp-data/fresh_set"


def _rows(p):
    return [json.loads(x) for x in (F / p).read_text().splitlines() if x.strip()]


def main() -> None:
    ids = {r["id"] for r in _rows("crosswalk.jsonl")}
    pool = {r["id"]: r for r in _rows("pool.jsonl") if r["id"] in ids}
    mc = query(messenger("prod"), """
        select mc.contact_id::text c, mm.state, mm.birthday, mm.waymark_id::text wid
        from public.member_contacts mc join public.members mm on mm.id = mc.member_id
        where mc.contact_id = any(cast(:ids as uuid[]))""",
               ids="{" + ",".join({r["contact_id"] for r in pool.values()}) + "}")
    by = collections.defaultdict(list)
    for r in mc.itertuples():
        by[r.c].append(r)
    wids = [r.wid for g in by.values() if len(g) == 1 for r in g if r.wid]
    pt = query(lighthouse("prod"), 'select "waymarkPatientNumber"::text k, gender, race from public."Patient" '
                                   'where "waymarkPatientNumber"::text = any(:ids)', ids=wids)
    pmap = {r.k: r for r in pt.itertuples()}
    out = []
    for i, r in pool.items():
        g = by.get(r["contact_id"], [])
        states = {x.state for x in g}
        rec = {"id": i, "sender_known": len(g) == 1, "state": states.pop().title() if len(states) == 1 and None not in states else None,
               "sex": None, "age": None, "race": None}
        if len(g) == 1:
            m = g[0]
            sent = date.fromisoformat(str(r["inserted_at"])[:10])
            if m.birthday is not None:
                b = m.birthday
                rec["age"] = sent.year - b.year - ((sent.month, sent.day) < (b.month, b.day))
            p = pmap.get(m.wid)
            if p is not None:
                rec["sex"] = p.gender
                race = list(p.race) if p.race is not None else []
                rec["race"] = race[0] if race else None
        out.append(rec)
    (F / "sender_demographics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    c = collections.Counter
    print(json.dumps({"texts": len(out), "sender_known": sum(x["sender_known"] for x in out),
                      "state": dict(c(x["state"] for x in out)), "sex": dict(c(x["sex"] for x in out)),
                      "race": dict(c(x["race"] for x in out)),
                      "age_missing": sum(x["age"] is None for x in out),
                      "age_under_18": sum(x["age"] is not None and x["age"] < 18 for x in out)}, default=str))


if __name__ == "__main__":
    main()
