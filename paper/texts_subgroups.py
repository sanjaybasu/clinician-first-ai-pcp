"""Post hoc, descriptive subgroup analysis of the fresh 949-text test set (Paper 2): sender characteristics by sampling
stratum, and hazard sensitivity and escalation burden by state, sex, age group and race, per system.

Inputs: fresh_set/sender_demographics.jsonl (paper/sender_demographics.py) and the same stored replies, labels, weights
and bootstrap method as paper/supplementary_analyses.texts() (S. Basu labels; two-phase Hajek weights; 2,000 resamples
of texts within each subgroup, numpy default_rng(20261108)). No interaction tests: the subgroups are descriptive.
Out: compass-pcp-data/paper2_final/derived/texts_subgroups.json (aggregates only). Usage: python -m paper.texts_subgroups
"""

import collections
import json

import numpy as np

from paper import supplementary_analyses as S

ARMS = ["ours", "amie_style", "mai_dxo_style", "raw_llm", "v22"]
RACE = {"Black or African American": "Black", "White": "White", "Hispanic or Latinx": "Hispanic or Latinx",
        "Asian": "Other race", "American Indian or Alaska Native": "Other race",
        "Native Hawaiian or Other Pacific Islander": "Other race", "Other": "Other race"}


def groups(d: dict) -> dict:
    age = d["age"]
    return {
        "state": d["state"] or "Unknown",
        "sex": "Unknown sender" if not d["sender_known"] else {"Female": "Female", "Male": "Male"}.get(d["sex"], "Other or unknown"),
        "age": "Unknown sender" if not d["sender_known"] else ("Unknown" if age is None else "Member under 18" if age < 18
                                                               else "18-39" if age < 40 else "40-64" if age < 65 else "65 or older"),
        "race": "Unknown sender" if not d["sender_known"] else RACE.get(d["race"], "Unknown or not recorded")}


def main() -> None:
    basu, _, rep, paged, _ = S._texts_data()
    E = S._per_repeat(rep, paged)
    ids = sorted(i for i in E["ours"] if i in basu)
    dem = {r["id"]: groups(r) for r in S._rows(S.D / "fresh_set/sender_demographics.jsonl")}
    assert set(ids) <= set(dem)
    out = {"definition": "post hoc, descriptive; S. Basu labels; weighted; sender known when the phone belongs to one "
                         "member; 'Member under 18' is the linked member's age (a parent may have sent the text)",
           "characteristics": {}, "subgroups": {}}
    for var in ("state", "sex", "age", "race"):
        tab = {}
        for s in ("random", "enriched", "all"):
            sel = [i for i in ids if s == "all" or basu[i]["stratum"] == s]
            c = collections.Counter(dem[i][var] for i in sel)
            tab[s] = {k: [v, S._r(100 * v / len(sel))] for k, v in sorted(c.items())}
            tab[s]["_n"] = len(sel)
        out["characteristics"][var] = tab
        sub = {}
        for g in sorted({dem[i][var] for i in ids}):
            gid = [i for i in ids if dem[i][var] == g]
            B = S._B(len(gid), np.random.default_rng(S.SEED))
            m = S._metrics({a: E[a] for a in ARMS}, basu, gid, B)
            sub[g] = {"texts": len(gid), **{a: {k: m[a][k] for k in ("safety_positive", "no_hazard",
                                                                      "hazard_sensitivity_weighted", "hazard_sensitivity_weighted_ci",
                                                                      "escalation_no_hazard_per_1000_weighted",
                                                                      "escalation_no_hazard_weighted_ci")} for a in ARMS}}
        out["subgroups"][var] = sub
    (S.OUT / "texts_subgroups.json").write_text(json.dumps(out, indent=1))
    for var, sub in out["subgroups"].items():
        for g, v in sub.items():
            o, a = v["ours"], v["amie_style"]
            print(f"{var:5} {g:24} n={v['texts']:4} sp={o['safety_positive']:3} nh={o['no_hazard']:4} | ours sens "
                  f"{o['hazard_sensitivity_weighted']} burden {o['escalation_no_hazard_per_1000_weighted']} "
                  f"{o['escalation_no_hazard_weighted_ci']} | amie sens {a['hazard_sensitivity_weighted']} burden "
                  f"{a['escalation_no_hazard_per_1000_weighted']}")


if __name__ == "__main__":
    main()
