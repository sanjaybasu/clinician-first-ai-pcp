"""Side-by-side development results for chosen arms on the four Paper 2 development sets (aggregates only).

Reads the scored outputs that the run chains write: paper2_dev/realmsg_tune/report.json, paper2_dev/{r3_v7,outpt}/
score_v<N>.txt (the latest one), paper2_dev/outpt/order_score_v<N>.txt, and v7_hb/report.md.

Usage: python -m paper.dev_compare --data <compass-pcp-data> --arms 17 18 19 20
"""

import argparse
import json
import re
from pathlib import Path


def _latest(d: Path, stem: str) -> dict:
    files = sorted(d.glob(f"{stem}_v*.txt"), key=lambda p: int(re.search(r"_v(\d+)", p.name).group(1)))
    t = files[-1].read_text()
    return json.loads(t[t.index("{"):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    dev = a.data / "paper2_dev"
    arms = [f"ai_pcp_v{n}" for n in a.arms] + ["amie_style", "mai_dxo_style"]
    rm = json.loads((dev / "realmsg_tune" / "report.json").read_text())
    r3, op, orders = _latest(dev / "r3_v7", "score"), _latest(dev / "outpt", "score"), _latest(dev / "outpt", "order_score")
    hb = (a.data / "v7_hb" / "report.md").read_text().splitlines()
    print("| arm | real: safety caught | real: 911 emergency now | real: non-hazard routine+ | DDXPlus unsafe / under / over |"
          " outpatient unsafe(dropout) / under / over | low-value orders |")
    print("|---|---|---|---|---|---|---|")
    for arm in arms:
        r = rm.get(arm, {})
        x, y = r3.get(arm, {}), op.get(arm, {})
        f = lambda d, k, m="final": d.get(k, {}).get(m, "-").split(" (")[0]
        print(f"| {arm} | {r.get('hazard_sensitivity', '-').split(' (')[0]} | {r.get('call_911_emergency_now', '-').split(' (')[0]} | "
              f"{r.get('negatives_routine_or_higher', '-').split(' (')[0]} | "
              f"{f(x, 'unsafe_closure_cm')} / {f(x, 'under_triage_all')} / {f(x, 'over_triage_all')} | "
              f"{f(y, 'unsafe_closure_cm')} ({f(y, 'unsafe_closure_cm', 'dropout')}) / {f(y, 'under_triage_all')} / "
              f"{f(y, 'over_triage_all')} | {orders.get(arm, {}).get('episodes_with_low_value_order', '-').split(' (')[0]} |")
    print("\nHealthBench (first grader block = GPT-4.1, second = Gemini):")
    for line in hb:
        if any(line.startswith(f"| v{n} - ") for n in a.arms):
            print(line)


if __name__ == "__main__":
    main()
