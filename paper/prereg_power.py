"""Sample-size check for the Paper 2 preregistration (design/paper2_preregistration_draft.md, section 10).

Paired binary outcomes per case (COMPASS and a comparator on the same message or case). Discordance rates come from the
development data (v18 against each comparator on the 28 safety positives of the real-message tuning half; unsafe
closure on the outpatient epiglottitis stratum). Power is simulated: exact McNemar test, two-sided alpha 0.05, for
superiority; a Wald paired-difference lower bound for non-inferiority. Seed fixed. Aggregates only.

Usage: python -m paper.prereg_power
"""

import numpy as np
from scipy.stats import binomtest

SEED, SIMS = 20261106, 4000
SENS = {"raw_llm": (8 / 28, 1 / 28), "mai_dxo_style": (14 / 28, 0.01), "amie_style": (3 / 28, 1 / 28), "v3": (3 / 28, 0.01)}
UNSAFE = {"amie_style": (2 / 12, 0.01), "mai_dxo_style": (6 / 12, 0.01), "comparator rate 0.10": (0.10, 0.01)}


def superiority(rng, n, p10, p01):
    hits = 0
    for _ in range(SIMS):
        b, c, _ = rng.multinomial(n, [p10, p01, 1 - p10 - p01])
        hits += b + c > 0 and b > c and binomtest(b, b + c, 0.5).pvalue < 0.05
    return hits / SIMS


def noninferiority(rng, n, p10, p01, margin):
    hits = 0
    for _ in range(SIMS):
        b, c, _ = rng.multinomial(n, [p10, p01, 1 - p10 - p01])
        d = (b - c) / n
        hits += d - 1.96 * np.sqrt(max((b + c) / n - d * d, 1e-9) / n) > -margin
    return hits / SIMS


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("Hazard sensitivity, superiority, power by number of safety positives")
    for k, (p10, p01) in SENS.items():
        print(f"  vs {k}: " + ", ".join(f"n={n} {superiority(rng, n, p10, p01):.2f}" for n in (50, 80, 100, 150)))
    print("Hazard sensitivity, non-inferiority vs amie_style, margin 0.10: " + ", ".join(
        f"n={n} {noninferiority(rng, n, *SENS['amie_style'], 0.10):.2f}" for n in (50, 80, 100)))
    print("Unsafe closure, superiority, power by number of cannot-miss cases")
    for k, (p10, p01) in UNSAFE.items():
        print(f"  vs {k}: " + ", ".join(f"n={n} {superiority(rng, n, p10, p01):.2f}" for n in (30, 60, 90, 210)))


if __name__ == "__main__":
    main()
