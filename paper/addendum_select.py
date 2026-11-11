"""Study A variant selection (addendum registration https://osf.io/tnezd, section 2), applied mechanically.

For each comparator, the variant with the highest mean length-adjusted score on the first 80 development tasks (split
order) is selected; among variants within 0.5 points of the best, the one with the lowest list-price cost per task.
V1 cost: the comparator chain's test-split cost per task (paper2_final/derived/hbpro_test_supplement.json). V2/V3 cost:
arm-labeled usage rows in the development folder. Writes hbpro_dev/addendum_selection.json. Usage: python -m paper.addendum_select
"""

import json
from pathlib import Path

import numpy as np

from paper import hbpro
from paper.supplementary_analyses import _hb_scores

D = Path(__file__).resolve().parents[2] / "compass-pcp-data"
DEV = D / "hbpro_dev"
VARIANTS = {"amie_style": ["amie_style", "amie_style_search_blk", "amie_style_search_edit_blk"],
            "mai_dxo_style": ["mai_dxo_style", "mai_dxo_style_search_blk", "mai_dxo_style_search_edit_blk"]}


def main() -> None:
    ids = json.loads((DEV / "split.json").read_text())["development"][:80]
    sc = _hb_scores(DEV)
    v1cost = json.loads((D / "paper2_final/derived/hbpro_test_supplement.json").read_text())["cost"]["comparator_chains"]
    usage = [u for u in hbpro._rows(DEV / "usage.jsonl") if u.get("arm")]
    out = {"registration": "https://osf.io/tnezd", "tasks": len(ids)}
    for agent, arms in VARIANTS.items():
        res = {}
        for a in arms:
            vals = [sc[(i, a)][1] for i in ids if (i, a) in sc]
            if len(vals) != len(ids):
                raise SystemExit(f"{a}: {len(vals)} of {len(ids)} development tasks graded")
            if a in v1cost:
                cost = v1cost[a]["usd_per_task"]
            else:
                rows = [u for u in usage if u["arm"] == a]
                cost = sum(u["in"] / 1e6 * hbpro.PRICE[u["who"]][0] + u["out"] / 1e6 * hbpro.PRICE[u["who"]][1] for u in rows) / len(ids)
            res[a] = {"mean_length_adjusted": round(100 * float(np.mean(vals)), 2), "usd_per_task": round(cost, 4)}
        best = max(v["mean_length_adjusted"] for v in res.values())
        near = [a for a in arms if res[a]["mean_length_adjusted"] >= best - 0.5]
        pick = min(near, key=lambda a: res[a]["usd_per_task"])
        out[agent] = {"variants": res, "within_0.5_of_best": near, "selected": pick}
    (DEV / "addendum_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
