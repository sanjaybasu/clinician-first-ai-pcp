n = 28 cases with all three arms judged; means with case-bootstrap 95% CIs (2,000 resamples).

| Arm | Condition recall | Condition precision | Medicine recall | Medicine precision | HPI elements (of 8) | FH/SH/social coverage | Agent messages | Drop-off-weighted completeness | Full-transcript completeness | AI ER/911 instructions (per message) |
|---|---|---|---|---|---|---|---|---|---|---|
| AMIE-style (Opus 5.5) | 97.2% (93.6 to 100.0) | 96.7% (94.6 to 98.7) | 100.0% (100.0 to 100.0) | 98.3% (96.5 to 99.7) | 3.46 (2.86 to 4.04) | 40.9% (32.3 to 49.5) | 7.82 (7.46 to 8.00) | 19.0% (15.5 to 22.7) | 41.7% (35.4 to 48.6) | 141 of 219 (64.4% (54.3 to 74.4)) |
| MAI-DxO-style (Opus 5.5) | 98.4% (95.5 to 100.0) | 96.0% (93.4 to 98.1) | 99.7% (99.0 to 100.0) | 96.5% (92.1 to 99.4) | 3.18 (2.57 to 3.79) | 55.8% (45.2 to 65.8) | 7.71 (7.21 to 8.00) | 19.4% (15.7 to 23.2) | 46.9% (40.3 to 53.0) | 167 of 216 (77.3% (67.9 to 85.3)) |
| COMPASS structured intake (Opus 5.5) | 96.6% (92.8 to 99.3) | 96.8% (94.6 to 98.7) | 100.0% (100.0 to 100.0) | 98.4% (96.3 to 100.0) | 4.71 (3.96 to 5.43) | 71.7% (60.3 to 81.0) | 6.43 (5.71 to 7.04) | 27.3% (23.0 to 31.3) | 64.8% (55.0 to 73.1) | 0 of 180 (0.0% (0.0 to 0.0)) |

Paired differences, COMPASS minus comparator (case bootstrap 95% CI):

| Measure | vs AMIE-style | vs MAI-DxO-style |
|---|---|---|
| cond_recall | -0.6% (-2.0 to 0.9) | -1.8% (-4.9 to 0.5) |
| cond_precision | 0.1% (-1.4 to 1.6) | 0.8% (-1.6 to 3.3) |
| med_recall | 0.0% (0.0 to 0.0) | 0.3% (0.0 to 1.0) |
| med_precision | 0.0% (-1.3 to 1.1) | 1.9% (-1.0 to 6.3) |
| hpi_of8 | 1.25 (0.29 to 2.21) | 1.54 (0.71 to 2.36) |
| fhsh | 30.8% (17.3 to 42.7) | 15.9% (2.0 to 29.1) |
| turns | -1.39 (-2.00 to -0.89) | -1.29 (-1.86 to -0.82) |
| dropoff_completeness | 8.3% (4.0 to 12.3) | 7.9% (4.2 to 11.5) |
| full_completeness | 23.1% (11.1 to 33.5) | 17.8% (7.9 to 27.5) |
