# paper2_final/derived

The script `compass-pcp/paper/supplementary_analyses.py` writes every file in this folder. Run it with `.venv/bin/python -m paper.supplementary_analyses` from the repository root. The script makes no model or API calls. It reads only stored outputs. All files hold aggregates only, with no message text, reply text, message id or task id. Every bootstrap uses `numpy.random.default_rng(20261108)` and 2,000 resamples over tasks or texts.

## hbpro_test_supplement.json (function `hb_test`)

This file holds the HealthBench Professional test split (350 tasks, GPT-5.4 grader, `hbpro_test/`). "Ours" is `compass_pro_full`.

- `arms`: the raw and length-adjusted means (x 100) with standard errors, and the mean and median reply length in characters. Length adjusted = score − 2.94e-5 × (characters − 2,000).
- `paired_length_adjusted` and `paired_raw`: the paired difference of ours minus each arm, with a bootstrap 95% CI. The bootstrap uses one RNG in the order amie_style, mai_dxo_style, gpt61sol, opus55, physician, which is the `hbpro.compare` order, so the length-adjusted values reproduce `compare.txt`. Each row also gives the two-sided paired t-test p and a two-sided percentile bootstrap p (floor 1/2,000).
- `holm_primary_family`: Holm-adjusted paired t-test p values for ours vs amie_style and ours vs mai_dxo_style, computed separately for the raw and the length-adjusted score.
- `win_tie_loss`: per-task win, tie and loss counts for ours vs each comparator. A tie is an exact equal score. Raw rubric scores tie often. Length-adjusted scores almost never tie.
- `subgroups_vs_amie_style`: the results by `use_case` and by `type`, with paired raw and length-adjusted differences vs amie_style and bootstrap CIs.
- `length.paired_both_le_2000_chars`: the paired differences on the tasks where both replies have 2,000 characters or fewer.
- `length.regression`: an OLS of the task-level raw score difference on the character difference (thousands), with bootstrap CIs. The intercept is the difference at equal length.
- `cost`: the list-price USD per task, from `usage.jsonl` × `hbpro.PRICE` ÷ 350. The pipeline usage rows have no arm label. Each pipeline task writes one opus55 row, one gpt61sol row and, when it searched, one web_search row. Each bare frontier arm writes one row per task. Rule A gives the first 350 rows of each model in file order to the bare arm. Rule B gives the 350 rows with the smallest input-token counts to the bare arm. The two rules differ by USD 0.005 per task for the pipeline. The totals are exact; only the split between arms is attributed. Grader usage is excluded.

## hbpro_dev_g61_chains.json (function `hb_dev`)

This file holds the development split (175 tasks). "Ours" is `compass_pro_edit`, the development equivalent of `compass_pro_full`. It gives the raw and length-adjusted paired differences of ours vs the GPT-6.1 Sol comparator chains (`amie_style_g61`, `mai_dxo_style_g61`), the Opus 5.5 chains, and the bare models. `g61_chain_minus_opus_chain` compares each chain on its two backbones. Ablation arms were still running in `hbpro_dev/` when this file was made. The arms in this file were complete (175 of 175) and do not change.

## texts_supplement.json (function `texts`)

This file holds the fresh real-text test set (949 texts × 3 repeats). The labels are S. Basu (all 949 texts) and J. Morgan (200 texts). Hájek weights are 1/p_inclusion.

The final policy is the approved post hoc rule. The raw_llm writer's reply is held only when `paper.disposition_check.check_unconditional` finds an emergency instruction with no conditional word. "Ours" escalates a text in a repeat when the writer's reply advises routine follow-up or more urgent care, or when the reply is held. All three repeats are escalated when the paging gate pages (emergency ≥ t_e or safety ≥ t_s). This rule reproduces `texts/report_posthoc.json`. `clinician_first_any_hit_held` is the registered rule in `report.json`, in which any `check()` hit is held.

- `main`: the weighted and unweighted hazard sensitivity and escalation without hazard (per 1,000), with CIs, for each arm. The arms are amie_style, mai_dxo_style, raw_llm, v22, ours, writer_alone (writer with no gate), gate_alone (gate with no writer) and clinician_first_any_hit_held. All arms share one set of 2,000 bootstrap index vectors.
- `paired_diffs`: ours minus amie_style, mai_dxo_style, raw_llm and v22 on the same index vectors (the method of `report_posthoc.json`), weighted and unweighted.
- `same_safety_layer`: each comparator writer (amie_style, mai_dxo_style, v22) gets the same paging gate and the same `check_unconditional` hold, built exactly as ours is built. `raw_llm+layer` is identical to ours. The key gives the weighted and unweighted measures with CIs, the paired differences of ours minus each layered writer on the same index vectors, and the replies held per 1,000. Unconditional instructions that reach the patient are 0 by construction, because every reply with a hit is held. The comparison shows whether the result comes from the writer or from the layer.
- `registered_hypotheses_v22`: the OSF hypotheses H1a to H1c (preregistration section 10) for the registered v22 arm as run. v22 escalation means its own reply's urgency is routine_followup or higher, with no gate. The key gives the weighted and unweighted paired differences of v22 minus raw_llm, mai_dxo_style and amie_style, with the shared bootstrap CIs. It also gives the registered binary outcome, which is majority escalation (2 or more of 3 repeats) on the 79 safety-positive texts, with an exact two-sided McNemar test (unweighted). H1c is a non-inferiority test with a margin of −0.10. It is judged by the lower 95% bound, and its Holm p is the one-sided bootstrap share of differences ≤ −0.10. Holm is applied over the 3 tested hypotheses, and again over the registered 6 with H2a to H2c entered as p = 1. H2a to H2c are "not run".
- `by_stratum`: the same measures within the random stratum (600 texts) and the enriched stratum (349 texts).
- `second_physician`: the measures on the 200 double-labeled texts with each physician's labels. `both_agree` gives the measures on the texts where the two physicians agree on safety_positive and on any hazard, with the shared label as the reference. `agreement` gives the unweighted Cohen's kappa for safety_positive, call_911, any hazard and the 4-class action.
- `repeat_consistency`: the share of texts where the escalation decisions of all 3 repeats agree, for each arm.
- `gate_writer_contributions`: the breakdown of the safety-positive and no-hazard texts by which component escalated them (gate and writer, gate only, writer only, neither).
- `ai_emergency_instructions`: replies per 1,000 with any `check()` hit and with an unconditional hit, for each arm.

## paging_gate_table.json and paging_gate_table.csv (function `paging_gate`)

These files give the paging gate development results (497 texts, `paper2_dev/paging_gate_497/report.json`) as one row per outcome × score × rule × alpha. The rules are `crc` (conformal risk control over 500 random splits: mean, p05, p95), `crc_weighted_rule`, and `full_sample` (threshold fit on all 497 texts). The JSON also records the deployed thresholds and the rows they match. t_e matches kev_p911 full_sample at alpha 0.05, and t_s matches kev_p_contact_doctor_or_911 full_sample at alpha 0.10.

## validation.json (function `validate`)

This file re-derives every overlapping value in the existing reports and records each check. The checked reports are `hbpro_test/report.json`, `hbpro_test/compare.txt`, `hbpro_dev/report.json` (arms used here), `texts/report.json` (the `texts_report.py` method: sequential RNG over sorted arms) and `texts/report_posthoc.json`. `report_posthoc.json` stores the per-1,000 CI bounds after it rounds the rate to 3 decimals, so the check applies the same rounding.

## hbpro_retest_summary.json, intake_instructions.json, texts_gate_test_and_leak.json (paper/posthoc_extras_paper2.py)

Post hoc analyses: the second run of the architecture and the AMIE-style agent on 100 test tasks; emergency
instructions in the history-taking episodes under the final check; the paging gate's decisions on the 949 test texts by
physician label, and classifier "emergency_now" replies per arm and among the architecture's released replies.
