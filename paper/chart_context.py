"""Chart context for each development message: what the care team could see before the message (development data only).

For each message (a lighthouse EncounterNote id), builds a short chart summary from records dated strictly before the
calendar day of that note, and never the note itself:
  lighthouse  Patient (age at the message, sex, language), the last 3 care-team notes, goals and open tasks created
              before that day, AdmissionDischargeTransfer ED and inpatient episodes in the prior 365 days;
  coredb      dbt_tuva_core.condition (recorded 60 or more days before, to copy the claims lag that a live chart has)
              and dbt_tuva_core.pharmacy_claim (dispensed 30 to 240 days before).
Writes <out>/chart_context.jsonl ({id, chart, counts}). The chart text is PHI: custodian storage only. Prints counts only.

Usage: python -m paper.chart_context --cases <realworld_cases_n2000_clean.json> --out <dir>
"""

import argparse
import html
import json
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))

N_NOTES, NOTE_CHARS, N_COND, N_MEDS, N_GOALS, N_TASKS = 3, 500, 12, 12, 6, 5


def arr(ids) -> str:
    ids = [str(i) for i in ids]
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids)
    return "ARRAY[" + ",".join(f"'{i}'" for i in ids) + "]"


def strip_html(t) -> str:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    t = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", str(t))
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    from wm_conn import coredb, lighthouse, query

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=pathlib.Path, help="development messages (EncounterNote ids); cutoff = start of that day")
    ap.add_argument("--links", type=pathlib.Path, help="texts from paper/sms_links.py ({id, way, t}); cutoff = the send time")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--name", default="chart_context.jsonl")
    a = ap.parse_args()
    lh, cx = lighthouse("prod"), coredb("prod")

    if a.cases:
        ids = [str(r["message_id"]) for r in json.loads(a.cases.read_text())]
        idx = query(lh, f'''SELECT n.id, n."patientId" pid, COALESCE(n."startTime", n."dateOfEncounter", n."createdAt") t,
                                   p."waymarkPatientNumber" way, p.gender::text sex, p."birthDate" dob, p.languages::text lang
                            FROM public."EncounterNote" n JOIN public."Patient" p ON p.id = n."patientId"
                            WHERE n.id = ANY({arr(ids)})''')
        idx["cut"] = pd.to_datetime(idx.t).dt.tz_localize(None).dt.normalize()
    else:
        links = pd.DataFrame([json.loads(x) for x in a.links.read_text().splitlines()]).dropna(subset=["way"])
        pt = query(lh, f'''SELECT id pid, "waymarkPatientNumber" way, gender::text sex, "birthDate" dob, languages::text lang
                           FROM public."Patient" WHERE "waymarkPatientNumber" = ANY({arr(links.way.unique())})''')
        idx = links.merge(pt.drop_duplicates("way"), on="way")
        idx["cut"] = pd.to_datetime(idx.t, utc=True).dt.tz_localize(None)
    pids, ways = idx.pid.unique().tolist(), idx.way.dropna().unique().tolist()
    hi = idx.cut.max().to_pydatetime()

    notes = query(lh, f'''SELECT n.id, n."patientId" pid, COALESCE(n."startTime", n."dateOfEncounter", n."createdAt") t,
                                 n."contactType"::text ct, n."encounterType"::text et, w.title::text role,
                                 left(COALESCE(n."noteV2"::text, n.note::text, ''), 4000) body
                          FROM public."EncounterNote" n LEFT JOIN public."Waymarker" w ON w.id = n."createdByWaymarkerId"
                          WHERE n."patientId" = ANY({arr(pids)}) AND COALESCE(n.deleted, false) = false
                            AND COALESCE(n."startTime", n."dateOfEncounter", n."createdAt") < :hi''', hi=hi)
    notes["t"] = pd.to_datetime(notes.t, errors="coerce", utc=True).dt.tz_localize(None)
    goals = query(lh, f'''SELECT "patientId" pid, title, category::text category, "createdAt" t FROM public."Goal"
                          WHERE "patientId" = ANY({arr(pids)}) AND COALESCE(deleted, false) = false AND "createdAt" < :hi''',
                  hi=hi)
    goals["t"] = pd.to_datetime(goals.t, errors="coerce", utc=True).dt.tz_localize(None)
    tasks = query(lh, f'''SELECT "patientId" pid, "taskTitle" title, completed, "createdAt" t, "updatedAt" u FROM public."Task"
                          WHERE "patientId" = ANY({arr(pids)}) AND COALESCE(deleted, false) = false AND "createdAt" < :hi''',
                  hi=hi)
    for c in ("t", "u"):
        tasks[c] = pd.to_datetime(tasks[c], errors="coerce", utc=True).dt.tz_localize(None)
    adt = query(lh, f'''SELECT "patientId" pid, "visitNumber" v, "admitDate" t, "dischargedDate" d, "patientClassCode" cls
                        FROM public."AdmissionDischargeTransfer" WHERE "patientId" = ANY({arr(pids)}) AND "admitDate" < :hi''',
                hi=hi)
    adt["t"] = pd.to_datetime(adt.t, errors="coerce", utc=True).dt.tz_localize(None)
    adt["d"] = pd.to_datetime(adt.d, errors="coerce", utc=True).dt.tz_localize(None)
    adt["k"] = adt.v.where(adt.v.fillna("") != "", adt.t.astype(str).str[:10])
    adt = adt.sort_values("t").drop_duplicates(["pid", "k", "cls"])
    cond = query(cx, f"""SELECT person_id way, recorded_date t, normalized_description d FROM dbt_tuva_core.condition
                         WHERE person_id = ANY({arr(ways)}) AND recorded_date < :hi AND normalized_description IS NOT NULL""",
                 hi=hi.date())
    cond["t"] = pd.to_datetime(cond.t, errors="coerce")
    meds = query(cx, f"""SELECT person_id way, dispensing_date t, ndc_description d FROM dbt_tuva_core.pharmacy_claim
                         WHERE person_id = ANY({arr(ways)}) AND dispensing_date < :hi AND ndc_description IS NOT NULL""",
                 hi=hi.date())
    meds["t"] = pd.to_datetime(meds.t, errors="coerce")

    a.out.mkdir(parents=True, exist_ok=True)
    stats = {"messages": 0, "with_notes": 0, "with_adt": 0, "with_conditions": 0, "with_meds": 0}
    src = {"n": (notes, "pid"), "g": (goals, "pid"), "k": (tasks, "pid"), "a": (adt, "pid"), "c": (cond, "way"),
           "m": (meds, "way")}
    by = {k: dict(tuple(df.groupby(col))) for k, (df, col) in src.items()}
    blank = {k: df.iloc[0:0] for k, (df, _) in src.items()}
    with (a.out / a.name).open("w") as fh:
        for r in idx.itertuples():
            cut = r.cut
            age = int((cut - pd.Timestamp(r.dob)).days // 365.25) if pd.notna(r.dob) else None
            n = by["n"].get(r.pid, blank["n"])
            n = n[(n.t < cut) & (n.id != r.id)].sort_values("t").tail(N_NOTES)
            g = by["g"].get(r.pid, blank["g"])
            g = g[g.t < cut].sort_values("t").tail(N_GOALS)
            k = by["k"].get(r.pid, blank["k"])
            k = k[(k.t < cut) & ~((k.completed == True) & (k.u < cut))].sort_values("t").tail(N_TASKS)  # noqa: E712
            ad = by["a"].get(r.pid, blank["a"])
            ad = ad[(ad.t < cut) & (ad.t >= cut - pd.Timedelta(days=365))]
            c = by["c"].get(r.way, blank["c"])
            c = c[c.t < cut - pd.Timedelta(days=60)]
            m = by["m"].get(r.way, blank["m"])
            m = m[(m.t < cut - pd.Timedelta(days=30)) & (m.t >= cut - pd.Timedelta(days=240))]
            L = [f"Patient: age {age if age is not None else 'unknown'}, sex {r.sex or 'unknown'}, languages {r.lang or 'unknown'}."]
            ed = ad[ad.cls.astype(str).str.upper().str.startswith("E")]
            ip = ad[ad.cls.astype(str).str.upper().str.startswith("I")]
            last = ad.d.dropna().max() if len(ad) else None
            L.append(f"Past 12 months: {len(ed)} emergency department visits, {len(ip)} hospital admissions"
                     + (f"; last discharge {(cut - last).days} days ago." if last is not None and pd.notna(last) else "."))
            if len(c):
                top = c.d.value_counts().head(N_COND).index.tolist()
                L.append("Diagnoses on claims: " + "; ".join(top) + ".")
            if len(m):
                top = m.d.str.split().str[:2].str.join(" ").value_counts().head(N_MEDS).index.tolist()
                L.append("Medicines filled in the past 8 months: " + "; ".join(top) + ".")
            if len(g):
                L.append("Care goals: " + "; ".join(f"{x.title} ({x.category})" for x in g.itertuples()) + ".")
            if len(k):
                L.append("Open tasks: " + "; ".join(str(x.title) for x in k.itertuples()) + ".")
            for x in n.itertuples():
                body = strip_html(x.body)[:NOTE_CHARS]
                L.append(f"Note {(cut - x.t.normalize()).days} days before ({x.role or 'staff'}, {x.ct or ''} {x.et or ''}): {body}")
            fh.write(json.dumps({"id": r.id, "chart": "\n".join(L),
                                 "counts": {"notes": len(n), "adt_12m": len(ad), "conditions": int(c.d.nunique()) if len(c) else 0,
                                            "meds": len(m), "goals": len(g), "tasks": len(k)}}) + "\n")
            stats["messages"] += 1
            stats["with_notes"] += len(n) > 0
            stats["with_adt"] += len(ad) > 0
            stats["with_conditions"] += len(c) > 0
            stats["with_meds"] += len(m) > 0
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
