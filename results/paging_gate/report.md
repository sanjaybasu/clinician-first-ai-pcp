# Paging gate on the 497-text emergency-enriched set (development; 497 texts; Kev runs compass-lh497-phys9b-f0, compass-lh497-phys9b-f1, compass-lh497-phys9b-f2, compass-lh497-phys9b-f3, compass-lh497-phys9b-f4)

Miss = share of positive texts not paged on the test half; paged = share of all texts paged. Enriched = the labeled sample as drawn; weighted = 1/p_inclusion, to the care-team SMS pool (47,126 texts). Mean over random 50/50 splits (5th-95th percentile).

## emergency: 104 positives of 497; weighted prevalence 0.52%; largest single positive = 42% of weighted positive mass

Kev-9B fold models alone, AUROC: 0.899, 0.91, 0.911, 0.9, 0.904

| Score | AUROC (95% CI) | Weighted AUROC (95% CI) | alpha | Threshold (all 497) | Miss | Paged | Miss, weighted | Paged, weighted | Splits with miss > alpha |
|---|---|---|---|---|---|---|---|---|---|
| kev_p911 | 0.909 (0.878-0.934) | 0.996 (0.990-0.999) | 0.05 | 0.0056 | 3.9% (0.0-10.9) | 50.2% (42.6-56.6) | 3.2% (0.0-16.9) | 3.0% (1.6-4.7) | 33% |
| kev_p911 | 0.909 (0.878-0.934) | 0.996 (0.990-0.999) | 0.10 | 0.0084 | 9.2% (1.8-19.0) | 42.4% (34.9-50.2) | 13.9% (0.6-39.4) | 1.9% (1.0-3.0) | 39% |
| stack_2000 | 0.908 (0.878-0.936) | 0.994 (0.984-0.998) | 0.05 | 0.5874 | 4.0% (0.0-10.7) | 53.3% (45.4-59.4) | 11.1% (0.0-31.7) | 3.6% (2.2-5.3) | 33% |
| stack_2000 | 0.908 (0.878-0.936) | 0.994 (0.984-0.998) | 0.10 | 0.8568 | 9.4% (1.9-19.3) | 42.7% (30.1-52.6) | 18.1% (1.6-40.0) | 2.2% (1.0-3.8) | 41% |
| stack_cf497 | 0.898 (0.864-0.928) | 0.989 (0.966-0.998) | 0.05 | 0.2080 | 3.7% (0.0-11.1) | 57.4% (44.2-72.7) | 9.7% (0.0-34.7) | 5.8% (2.2-14.5) | 29% |
| stack_cf497 | 0.898 (0.864-0.928) | 0.989 (0.966-0.998) | 0.10 | 0.3109 | 9.0% (1.8-18.6) | 42.5% (34.1-54.2) | 17.3% (0.7-40.9) | 2.2% (1.0-3.8) | 37% |

Weighted CRC rule (positives weighted by 1/p_inclusion):

- kev_p911, alpha 0.05: no threshold met alpha on 484 of 500 splits (page all); miss weighted 0.1%, paged weighted 96.9%, paged enriched 98.4%
- kev_p911, alpha 0.10: no threshold met alpha on 487 of 500 splits (page all); miss weighted 0.3%, paged weighted 97.5%, paged enriched 98.6%
- stack_2000, alpha 0.05: no threshold met alpha on 485 of 500 splits (page all); miss weighted 0.4%, paged weighted 97.1%, paged enriched 98.5%
- stack_2000, alpha 0.10: no threshold met alpha on 488 of 500 splits (page all); miss weighted 0.4%, paged weighted 97.6%, paged enriched 98.5%
- stack_cf497, alpha 0.05: no threshold met alpha on 489 of 500 splits (page all); miss weighted 0.3%, paged weighted 97.9%, paged enriched 98.9%
- stack_cf497, alpha 0.10: no threshold met alpha on 487 of 500 splits (page all); miss weighted 0.4%, paged weighted 97.5%, paged enriched 98.5%

## safety: 263 positives of 497; weighted prevalence 4.43%; largest single positive = 5% of weighted positive mass

Kev-9B fold models alone, AUROC: 0.901, 0.928, 0.926, 0.898, 0.925

| Score | AUROC (95% CI) | Weighted AUROC (95% CI) | alpha | Threshold (all 497) | Miss | Paged | Miss, weighted | Paged, weighted | Splits with miss > alpha |
|---|---|---|---|---|---|---|---|---|---|
| kev_p_contact_doctor_or_911 | 0.923 (0.896-0.945) | 0.921 (0.860-0.967) | 0.05 | 0.0289 | 4.6% (0.8-9.7) | 70.9% (62.2-78.7) | 21.7% (0.3-41.2) | 18.5% (8.8-30.3) | 40% |
| kev_p_contact_doctor_or_911 | 0.923 (0.896-0.945) | 0.921 (0.860-0.967) | 0.10 | 0.0453 | 9.9% (4.3-16.4) | 58.9% (52.2-67.1) | 37.0% (14.1-62.7) | 6.2% (3.3-11.1) | 46% |
| stack_2000 | 0.907 (0.881-0.931) | 0.940 (0.900-0.972) | 0.05 | 0.2093 | 4.5% (0.8-9.5) | 73.6% (67.1-79.9) | 12.3% (0.1-34.0) | 19.2% (12.1-27.3) | 39% |
| stack_2000 | 0.907 (0.881-0.931) | 0.940 (0.900-0.972) | 0.10 | 0.3076 | 9.4% (4.6-15.5) | 62.1% (54.6-70.3) | 33.1% (14.0-53.1) | 8.6% (4.8-15.7) | 40% |
| stack_cf497 | 0.926 (0.904-0.946) | 0.939 (0.902-0.973) | 0.05 | 0.2146 | 4.5% (0.8-9.4) | 69.3% (61.0-76.3) | 23.6% (0.3-42.3) | 13.4% (6.6-21.2) | 37% |
| stack_cf497 | 0.926 (0.904-0.946) | 0.939 (0.902-0.973) | 0.10 | 0.2837 | 9.6% (4.5-16.8) | 59.2% (51.8-65.9) | 41.7% (14.7-63.8) | 5.6% (3.3-8.6) | 42% |

Weighted CRC rule (positives weighted by 1/p_inclusion):

- kev_p_contact_doctor_or_911, alpha 0.05: no threshold met alpha on 500 of 500 splits (page all); miss weighted 0.0%, paged weighted 100.0%, paged enriched 100.0%
- kev_p_contact_doctor_or_911, alpha 0.10: no threshold met alpha on 137 of 500 splits (page all); miss weighted 3.9%, paged weighted 58.5%, paged enriched 89.5%
- stack_2000, alpha 0.05: no threshold met alpha on 500 of 500 splits (page all); miss weighted 0.0%, paged weighted 100.0%, paged enriched 100.0%
- stack_2000, alpha 0.10: no threshold met alpha on 129 of 500 splits (page all); miss weighted 3.5%, paged weighted 53.3%, paged enriched 87.1%
- stack_cf497, alpha 0.05: no threshold met alpha on 500 of 500 splits (page all); miss weighted 0.0%, paged weighted 100.0%, paged enriched 100.0%
- stack_cf497, alpha 0.10: no threshold met alpha on 123 of 500 splits (page all); miss weighted 3.8%, paged weighted 53.7%, paged enriched 87.7%

## safety_s1: 194 positives of 425; weighted prevalence 5.63%; largest single positive = 4% of weighted positive mass

Kev-9B fold models alone, AUROC: 0.898, 0.916, 0.912, 0.891, 0.905

| Score | AUROC (95% CI) | Weighted AUROC (95% CI) | alpha | Threshold (all 497) | Miss | Paged | Miss, weighted | Paged, weighted | Splits with miss > alpha |
|---|---|---|---|---|---|---|---|---|---|
| kev_p_contact_doctor_or_911 | 0.910 (0.880-0.936) | 0.898 (0.835-0.951) | 0.05 | 0.0218 | 4.4% (1.0-10.2) | 74.5% (62.0-84.5) | 18.4% (0.1-43.7) | 26.9% (14.1-42.3) | 36% |
| kev_p_contact_doctor_or_911 | 0.910 (0.880-0.936) | 0.898 (0.835-0.951) | 0.10 | 0.0375 | 9.5% (3.7-16.7) | 57.7% (47.4-70.9) | 41.8% (17.8-64.9) | 9.2% (3.4-20.5) | 41% |
| stack_2000 | 0.911 (0.884-0.937) | 0.896 (0.833-0.948) | 0.05 | 0.1929 | 4.2% (0.0-9.7) | 74.2% (62.4-85.0) | 19.2% (0.0-48.0) | 25.3% (12.3-46.2) | 37% |
| stack_2000 | 0.911 (0.884-0.937) | 0.896 (0.833-0.948) | 0.10 | 0.3076 | 9.6% (3.7-17.3) | 57.5% (47.9-68.5) | 43.1% (21.8-64.0) | 8.5% (4.0-17.1) | 39% |
| stack_cf497 | 0.918 (0.890-0.943) | 0.886 (0.818-0.947) | 0.05 | 0.1551 | 4.7% (1.0-10.6) | 74.1% (61.5-84.5) | 23.7% (0.1-53.5) | 22.7% (10.4-42.0) | 40% |
| stack_cf497 | 0.918 (0.890-0.943) | 0.886 (0.818-0.947) | 0.10 | 0.2535 | 9.6% (4.0-16.7) | 57.1% (46.9-70.4) | 48.5% (22.2-68.6) | 7.6% (3.3-16.9) | 41% |

Weighted CRC rule (positives weighted by 1/p_inclusion):

- kev_p_contact_doctor_or_911, alpha 0.05: no threshold met alpha on 499 of 500 splits (page all); miss weighted 0.0%, paged weighted 99.9%, paged enriched 100.0%
- kev_p_contact_doctor_or_911, alpha 0.10: no threshold met alpha on 15 of 500 splits (page all); miss weighted 6.0%, paged weighted 47.2%, paged enriched 85.0%
- stack_2000, alpha 0.05: no threshold met alpha on 498 of 500 splits (page all); miss weighted 0.0%, paged weighted 99.8%, paged enriched 99.9%
- stack_2000, alpha 0.10: no threshold met alpha on 14 of 500 splits (page all); miss weighted 6.4%, paged weighted 50.0%, paged enriched 86.7%
- stack_cf497, alpha 0.05: no threshold met alpha on 499 of 500 splits (page all); miss weighted 0.0%, paged weighted 99.9%, paged enriched 100.0%
- stack_cf497, alpha 0.10: no threshold met alpha on 10 of 500 splits (page all); miss weighted 6.3%, paged weighted 58.2%, paged enriched 88.1%

