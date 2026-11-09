"""Two-phase sampling weights for the fresh real-text test set (Paper 2).

Design (compass-pcp-data/fresh_set/design.json): 2,525 people; a simple random sample of 600; the frozen screen
(keyword list or screening model) was run on all 2,525 (screen.jsonl for the other 1,925, screen_random.jsonl for the
600), and every screen-positive person outside the random sample was taken (349). Every screen-positive person is
therefore in the sample with probability 1 (450 = 101 random + 349 enriched), and the 2,075 screen-negative people are
represented by the 499 screen-negative random texts. Weights: screen-positive 1; screen-negative 2,075 / 499
(post-stratified to the known screen-negative count). This replaces 1/p_inclusion from crosswalk.jsonl, which gave the
101 screen-positive random texts weight 4.2 although they had inclusion probability 1.
"""

import json
from pathlib import Path

FRESH = Path(__file__).resolve().parents[2] / "compass-pcp-data/fresh_set"


def weights() -> dict[str, float]:
    rows = lambda p: [json.loads(x) for x in (FRESH / p).read_text().splitlines() if x.strip()]  # noqa: E731
    cw = rows("crosswalk.jsonl")
    pos_random = {r["id"] for r in rows("screen_random.jsonl") if r["keyword"] or r["llm"]}
    contacts = json.loads((FRESH / "design.json").read_text())["contacts"]
    enriched = [c for c in cw if c["stratum"] == "enriched"]
    neg = [c for c in cw if c["stratum"] == "random" and c["id"] not in pos_random]
    n_pos_pop = len(enriched) + sum(c["stratum"] == "random" and c["id"] in pos_random for c in cw)
    w_neg = (contacts - n_pos_pop) / len(neg)
    return {c["id"]: (w_neg if (c["stratum"] == "random" and c["id"] not in pos_random) else 1.0) for c in cw}


if __name__ == "__main__":
    w = weights()
    print(len(w), round(sum(w.values()), 1), sorted({round(v, 4) for v in w.values()}))
