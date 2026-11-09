"""paper/multiturn_analysis.py on synthetic episodes: outcome definitions, paired statistics and the blinded packets."""

import csv
import json

from paper import multiturn_analysis as ma
from paper import multiturn_ddxplus as mt


def case(cid, stratum, complaint, cannot_miss, benign):
    return {"case_id": cid, "stratum": stratum, "complaint": complaint, "cannot_miss": cannot_miss, "benign": benign,
            "evidences": ["E_91"], "age": 40, "sex": "M"}


def ep(cid, arm, disp, turns, no_disp=False, questions=(), codes=()):
    ts = []
    for t in range(1, turns + 1):
        last = t == turns
        ts.append({"turn": t, "reply": f"{arm} {cid} reply {t}", "asking": not last,
                   "path": {"decision": "ask" if not last else "complete", "questions": list(questions)},
                   **({} if last else {"patient": {"reply": "yes", "asked_codes": list(codes), "parsed": True}})})
    return {"case_id": cid, "arm": arm, "n_turns": turns, "no_disposition": no_disp, "final_disposition": disp,
            "final_class": disp, "final_decision": "complete", "turns": ts, "ledger_events": []}


def test_outcomes_follow_the_registered_definitions():
    c = case("t:1", "cough_acute/community_acquired_pneumonia", "cough_acute", "community_acquired_pneumonia", False)
    o = ma.outcomes(c, ep("t:1", "raw_llm", "routine_visit", 2, codes=("E_91", "E_66", "E_14", "E_45")))
    assert o["unsafe_closure"] and not o["escalated"] and o["all_must_ask"]
    o = ma.outcomes(c, ep("t:1", "raw_llm", None, 12, no_disp=True))
    assert o["unsafe_closure"] and o["disposition"] == "no_disposition"
    # ai_pcp_v3 takes asked questions from the engine log, not from the patient codes
    o = ma.outcomes(c, ep("t:1", "ai_pcp_v3", "clinician_handoff", 2, questions=("fever_history",), codes=("E_45",)))
    assert o["escalated"] and o["asked"] == ["fever_history"] and o["asked_by_patient_codes"] == ["coughing_blood"]
    b = case("t:2", "cough_acute/benign", "cough_acute", None, True)
    o = ma.outcomes(b, ep("t:2", "raw_llm", "emergency_now", 1))
    assert o["benign_emergency"] and "unsafe_closure" not in o


def test_paired_statistics():
    d, lo, hi = ma.newcombe_paired(10, 8, 2, 30)
    assert lo < d < hi and -1 <= lo and hi <= 1 and abs(d - 0.12) < 1e-12
    d2, lo2, hi2 = ma.newcombe_paired(10, 2, 8, 30)
    assert abs(d2 + d) < 1e-12 and abs(lo2 + hi) < 1e-12 and abs(hi2 + lo) < 1e-12
    adj = ma.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
    p = ma.prop(0, 100)
    assert p["one_sided_upper95"] < 0.03


def test_analysis_and_blinded_packets(tmp_path, monkeypatch):
    cases = [case(f"t:{i}", "chest_pain/acs", "chest_pain", "acs", False) for i in range(70)]
    eps = []
    for c in cases:
        eps.append(ep(c["case_id"], "ai_pcp_v3", "emergency_now", 2, questions=("sweating",)))
        eps.append(ep(c["case_id"], "engine_off", "self_care", 1))
        eps.append(ep(c["case_id"], "raw_llm", "urgent_same_day", 3, codes=("E_50", "E_57")))
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    monkeypatch.setattr(mt, "load_evidences", lambda: {"E_152": {"value_meaning": {}}})
    res = ma.analyze(tmp_path)
    t = res["paired_tests"]["unsafe_closure:ai_pcp_v3-engine_off"]
    assert t["only_second"] == 70 and t["mcnemar_exact_p"] < 1e-10 and not t["inconclusive"] and t["family"] == "primary"
    assert res["proportions"]["unsafe_closure"]["ai_pcp_v3"]["one_sided_upper95"] < 0.05
    assert res["turns"]["ai_pcp_v3-raw_llm"]["mean_diff"] == -1
    out = ma.packets(tmp_path, tmp_path / "packets")
    assert out["disposition_items"] == 60 and out["audit_episodes"] == 50
    rows = list(csv.DictReader((tmp_path / "packets" / "send" / "disposition_sheet.csv").open()))
    assert set(rows[0]) == {"item", "reply_text", "physician_disposition", "comment"}
    sent = (tmp_path / "packets" / "send" / "question_audit_sheet.csv").read_text()
    assert "raw_llm" in sent  # synthetic reply text carries the arm here; real replies do not
    assert "patient_coded" not in sent and "classifier" not in (tmp_path / "packets" / "send" / "disposition_sheet.csv").read_text()
    key = json.loads((tmp_path / "packets" / "physician_key.json").read_text())
    for r in rows:
        r["physician_disposition"] = next(k for k in key["disposition"] if k["item"] == int(r["item"]))["classifier_disposition"]
    with (tmp_path / "ret.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert ma.agreement(tmp_path / "packets", tmp_path / "ret.csv")["agreement"] == 1.0
