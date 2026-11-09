"""Link Compass texts (messenger public.conversations ids) to the patient they were about.

conversation -> workflow_assignment -> member -> waymark_id. The assignment names the member even when a phone number is
shared by a household. Writes <out> as JSONL {id, way, t}; prints counts only.
Usage: python -m paper.sms_links --pool <pool.jsonl> [--pool ...] --out <links.jsonl>
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))


def main() -> None:
    from wm_conn import messenger, query

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=pathlib.Path, action="append", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    ids = sorted({json.loads(x)["id"] for p in a.pool for x in p.read_text().splitlines()})
    df = query(messenger("prod"), """SELECT c.id::text id, m.waymark_id way, c.inserted_at t
        FROM public.conversations c JOIN public.workflow_assignments wa ON wa.id = c.workflow_assignment_id
        JOIN public.members m ON m.id = wa.member_id WHERE c.id::text = ANY(:ids)""", ids=ids)
    with a.out.open("w") as fh:
        for r in df.itertuples():
            fh.write(json.dumps({"id": r.id, "way": r.way, "t": str(r.t)}) + "\n")
    print(json.dumps({"messages": len(ids), "linked": int(df.way.notna().sum()), "patients": int(df.way.nunique())}))


if __name__ == "__main__":
    main()
