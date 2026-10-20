# Complete calibration analysis: ctx_cal_v1

Analysis date: 2 October 2026. HuGaDB development data only.


**1. Main conclusions**

The saved experiment is internally consistent. The main scientific limitation is sensitivity to the training and decoding procedure, rather than a demonstrated artifact or scoring error.

- Removing the unnecessary guard from raw inputs increases F1@50 from 36.08 to 50.00, but also changes which positions receive supervision and increases their count by 77.94%. This is an objective change, not merely removal of an input restriction.
- Native VQ benefits from the shorter-context package in seed 111 (38.81 to 43.42); raw does not show a clear aggregate F1@50 preference (50.00 versus 49.80). Context and mask span interact strongly.
- Longer masks reduce segment recovery under these settings. The native-VQ combined configuration loses F1@50 on every subject and predicts about half the ground-truth segment count. All ten states remain occupied; this is not single-state collapse.
- The preserved mask sampler creates substantial positional imbalance, especially at context 64/span 16. The sweep therefore does not isolate mask duration from the distribution of supervised positions.
- At 3,000 updates, native VQ scores 43.03 with a genuine 3k schedule but 19.15 under the still-running 5k schedule. A shorter schedule and an intermediate checkpoint are not equivalent.
- Validation selection helps the raw reference in both seeds but is not universally aligned with segmentation. Offline early-stopping replay selects exactly the same checkpoints as full-history validation in every evaluated replay comparison.
- More outer OT optimization consistently lowers its objective in the four sensitivity tests while reducing F1@50. More final inference is usually score-stable, with two important exceptions. Every outer fit remains capped.
- Aggregate improvements conceal major subject and action redistribution. Short transitions and elevator actions remain poorly recovered; no subject-transfer or tokenizer-superiority claim follows.


**2. Scope, provenance and verification**

Sources: [manifest](../manifest.json), [segmentation cells](../segmentation.csv), [training](../training.csv), [reproduction checks](../reproduction.json), [original report](../REPORT.md). The original files were preserved. This analysis fits no new model and uses labels only for post-hoc evaluation.

The population is **364 recordings, 18 subjects, 1,137,986 frames, 76,035 tokens and 3,821 ground-truth action runs**. There are ten observed action classes. Token duration is 15 frames / 60 Hz = 0.25 s and vocabulary size is 512. The contextualizer split contains 335 fitting recordings and 29 validation recordings. All 18 subjects occur in fitting; 16 also occur in validation. This is not subject-disjoint evaluation. OT and label mapping pool all 364 recordings.

All **14 jobs, 58 planned checkpoint/solver evaluations and 232 metric rows** are present. Deduplication leaves **44 distinct evaluations**, with four final-solver rungs each. Reused roles are not independent fits or replicates. The optional combined count-matching job was disabled. No large-model timing probe is saved in this run; the future search declaration is metadata, not executed experiments.

Checks completed:

- Recomputed hashes and checked finite tensors for **404 saved checkpoints**; verified selected/replay steps against histories.
- Validated all 44 evaluation artifact sets, array shapes, finite values, state ranges and exact recording coverage.
- Checked frame predictions against saved window predictions and token lengths.
- Independently rescored every final-rung prediction set using the original scorer; **all five metrics match for all 44 evaluations**.
- Checked all CSV/rung metrics against saved ladder JSON, status-count totals, checkpoint-to-evaluation identities, alias targets and exposure sums.
- Checked current implementation/source-manifest identities and tokenizer checkpoint hashes. Both full-data reproduction anchors match Task 2 exactly, including metrics, validation histories, checkpoint tensors and embeddings.
- Verified exact per-update target matching in the native-VQ count control.

**Audit discrepancies: 0.** [Machine-readable audit](audit.json). This is evidence of internal consistency, not proof that every methodological choice is optimal. Predictions are stored only for the last rung: intermediate-rung metrics were cross-checked against their saved ladders, not independently rescored from intermediate predictions.


**3. Main factorial comparison**

All rows below: seed 111, 5k training horizon, validation-selected checkpoint, outer cap 50, final rung 4,000. Differences are descriptive; only references have a second seed.

| pipeline | config_name | checkpoint_step | MoF | Edit | F1@10 | F1@25 | F1@50 | run_ratio |
|---|---|---|---|---|---|---|---|---|
| native_vq | reference | 4500 | 41.51 | 39.47 | 46.09 | 42.14 | 38.81 | 0.698 |
| native_vq | short_context | 4000 | 43.14 | 41.46 | 48.03 | 46.35 | 43.42 | 0.680 |
| native_vq | long_mask | 5000 | 42.44 | 30.63 | 38.55 | 36.41 | 33.64 | 0.589 |
| native_vq | combined | 4300 | 32.28 | 29.65 | 32.20 | 27.64 | 22.22 | 0.505 |
| raw | reference | 4100 | 49.98 | 45.31 | 53.47 | 52.33 | 50.00 | 0.884 |
| raw | short_context | 4200 | 48.82 | 46.40 | 55.52 | 52.53 | 49.80 | 0.931 |
| raw | long_mask | 3700 | 47.73 | 39.31 | 44.39 | 41.56 | 38.01 | 0.723 |
| raw | combined | 3500 | 47.07 | 39.91 | 47.14 | 45.53 | 42.76 | 0.793 |

Context 128/64 means 32/16 seconds; span 8/16 nominally means 2/4 seconds. Shorter context also changes stride (64 to 32), batch size (64 to 128) and chunk sampling. The intended full-batch token count stays 8,192, but realized valid-token exposure rises by 1.84% because padding and epoch tails differ. Its effect cannot be attributed solely to removing distant context.

At span 8, shortening context changes F1@50 by **+4.61 VQ / −0.20 raw**. At span 16 it changes F1@50 by **−11.42 VQ / +4.75 raw**. The corresponding difference-in-differences is **−16.03 VQ / +4.95 raw**. Thus context and mask span cannot be treated as independent knobs with universal winners. This is descriptive interaction, not a significance test.

Native VQ short context wins on all five aggregate metrics versus its reference and improves F1@50 on 12/18 subjects. Its segment count decreases slightly (ratio 0.698 to 0.680), so the gain is not explained simply by creating more boundaries. Segment precision rises from 47.21% to 53.62% and recall from 32.95% to 36.48%.

Raw short context produces more segments (ratio 0.884 to 0.931), with slightly higher segment recall but lower precision at IoU 0.5. It improves F1@10 and Edit while losing 1.16 MoF and 0.20 F1@50: a metric trade-off, not a contradiction.

![Main metrics and run counts](main_results.png)

The final-checkpoint comparisons preserve the qualitative preference for span 8 and the poor native-VQ combined result, so these findings do not rest only on validation-selected checkpoints.


**4. The raw guard effect and unequal supervision**

Guarded and unguarded raw references share masked input positions, architecture, schedule, seed and token cache. The guard changes the eligible positions entering the loss. Removing it raises eligible targets from **6,432,577 to 11,446,210 (+77.94%)** and adds targets nearer visible windows.

F1@50 increases **13.92 points**, MoF **6.11**, Edit **8.55**, F1@10 **10.76** and F1@25 **13.29**. Predicted/GT run ratio rises from **0.683 to 0.884**, consistent with recovering more action changes. This supports removing an unnecessary receptive-field guard for these raw tokens. It does not separate the effects of target count, target position and prediction difficulty; no target-count-matched guard-off control was run.

The benefit is heterogeneous: 12 subjects improve and six worsen under the same dataset-level mapping. Subject changes range from **−41.11 to +54.87 F1@50**. Subject 01 changes 65.08 to 23.97; subject 12 changes 17.25 to 72.12. Those are descriptive pooled-fit results, not independent tests of transfer.

| subject | F1@50_before | F1@50_after | delta_F1@50 |
|---|---|---|---|
| 1 | 65.08 | 23.97 | -41.11 |
| 2 | 52.03 | 33.33 | -18.70 |
| 3 | 53.80 | 66.67 | 12.87 |
| 4 | 16.72 | 59.16 | 42.44 |
| 5 | 19.71 | 72.23 | 52.52 |
| 6 | 63.21 | 28.39 | -34.82 |
| 7 | 33.17 | 65.52 | 32.35 |
| 8 | 38.58 | 12.09 | -26.49 |
| 9 | 30.86 | 37.97 | 7.11 |
| 10 | 55.95 | 20.85 | -35.10 |
| 11 | 48.76 | 29.67 | -19.09 |
| 12 | 17.25 | 72.12 | 54.87 |
| 13 | 23.06 | 38.31 | 15.25 |
| 14 | 31.44 | 54.62 | 23.18 |
| 15 | 20.66 | 70.29 | 49.63 |
| 16 | 21.85 | 63.94 | 42.09 |
| 17 | 30.62 | 71.88 | 41.26 |
| 18 | 22.29 | 73.17 | 50.88 |

The 6.11-point MoF improvement is concentrated in going_up (+3.40 dataset points), standing (+2.67) and going_down (+1.77). Walking contributes −1.00 points, and both elevator classes also decline. The improvement is substantial but not uniform across actions.


**5. What long masks change**

Raw span-8 and span-16 runs at the same context have exactly equal selected/eligible target counts, yet long masks lose **11.99 F1@50 at context 128** and **7.04 at context 64**. Unequal target count cannot explain these raw comparisons.

For native VQ, span 16 provides 33.21% more eligible targets than span 8 at context 128, but loses 5.17 F1@50. Matching the count exactly yields 28.20 rather than 33.64 (ordinary long mask) or 38.81 (span 8). Matching succeeded at all 5,000 updates with zero retries. Total frame weight differs from the reference by only 3,562 / 96,484,146 = 0.00369%. This does not make the target identities or prediction difficulty identical.

Run-count ratios for reference → long mask are **0.698 → 0.589 VQ** and **0.884 → 0.723 raw**. Native VQ combined falls to **0.505** and loses F1@50 on all 18 subjects. These are too few predicted segments overall; class errors and boundary placement can coexist with merging. Counts alone do not locate every error.

Every distinct evaluation uses all ten output states. The largest state's frame fraction is at most 34.85% across the audited evaluations. Therefore the observed deterioration is not complete state collapse.

Native VQ long mask illustrates why MoF and F1 can disagree: MoF increases **41.51 → 42.44**, while F1@50 decreases **38.81 → 33.64**. Some long action regions are labeled better, while fewer action segments are correctly recovered.


**6. Mask-position imbalance: a methodological confound**

The existing sampler sorts the union of randomly placed spans and truncates its highest indices to reach the desired mask count. This deterministically favors earlier masked positions. Span placement also constrains possible starts. Audited masking rates below are normalized by actual valid positions, not padded positions; masks are shared across pipelines for the same configuration.

| config_name | mean_realized_run | first_quarter_mask_pct | last_quarter_mask_pct | first_quarter_eligible_pct | last_quarter_eligible_pct |
|---|---|---|---|---|---|
| long_mask | 15.64 | 31.64 | 15.33 | 24.83 | 9.53 |
| reference | 8.90 | 30.41 | 22.60 | 17.83 | 10.90 |
| combined | 12.80 | 32.24 | 5.76 | 25.23 | 3.28 |
| short_context | 7.95 | 31.63 | 16.00 | 18.68 | 6.06 |

For context 64/span 16, first-quarter masking is **32.24%**, versus **5.76%** in the last quarter. Native-VQ eligible-target rates are **25.23% versus 3.28%**. Its mean realized masked run is **12.80 tokens**, not 16, and only 31.2% of realized runs have length exactly 16.

This is not a deviation from the calibration specification: preserving the historical sampler was explicit. It is an important limit on interpretation that the original report omits. “Long masks hurt” means these realized mask distributions hurt; the experiment does not establish that an unbiased long-span objective would hurt. Correcting the sampler would define a new protocol and must not silently replace these historical anchors.


**7. Training horizon and checkpoint policy**

| pipeline | genuine_3k_final | step3k_of_5k | step5k_final | selected_of_5k |
|---|---|---|---|---|
| native_vq | 43.03 | 19.15 | 38.57 | 38.81 |
| raw | 45.16 | 41.36 | 48.64 | 50.00 |

The native-VQ **23.88-point difference at the same update count** is the strongest schedule observation. Validation loss is actually slightly lower for step 3k of the long schedule (2.2127) than for the genuine 3k endpoint (2.2227), despite much worse segmentation. Raw also benefits from annealing by step 3k: 45.16 versus 41.36 (+3.80). These are matched-seed schedule comparisons, not evidence that fewer updates are inherently better.

Comparing each run's validation-selected checkpoint, genuine 3k versus 5k gives **43.03 vs 38.81 VQ**, but **45.70 vs 50.00 raw**. One common horizon is not empirically optimal for both, and neither 3k result has a second seed.

Validation selection versus the final checkpoint gives raw +1.36 in seed 111 and +8.27 in seed 222; native VQ +0.24 and 0.00. Across the eight main cells, selection wins in six, ties one, and loses one: native-VQ combined final is 26.14 versus selected 22.22.

All **11 evaluated replay roles** select the same checkpoint as full-history validation. Three main jobs simulate stopping before the endpoint or at its boundary; none changes the chosen weights. This run does not establish a segmentation advantage or disadvantage of patience replay over validation selection. It can indicate saved updates, not a different learned result in these cells.

Masked loss is not useless, but it is not a reliable segmentation surrogate. Within the same raw long-mask job, loss improves **1.9242 → 1.8851** from update 3,000 to 3,700 while F1@50 drops **44.26 → 38.01**. Conversely, VQ short context and raw reference have monotonic loss/score ranking among their few decoded checkpoints. Within-job Spearman values are therefore saved as descriptive only (3–4 distinct checkpoints for most main jobs); no pooled correlation across different target tasks is used.

Intermediate peaks such as raw short-context 51.16 at step 3,000 are diagnostics, not newly selected model results. Choosing whichever saved checkpoint has the highest action-label score would create an additional selection procedure.

![Checkpoint and validation trajectories](training_trajectories.png)

Lines between decoded checkpoints connect observations; they do not imply that intervening checkpoints were decoded.


**8. Seed variation and class recovery**

| pipeline | seed111 | seed222 | mean | population_SD |
|---|---|---|---|---|
| native_vq | 38.81 | 42.79 | 40.80 | 1.99 |
| raw | 50.00 | 52.96 | 51.48 | 1.48 |

These are **two-seed descriptive summaries**, not confidence intervals or a three-seed result. The VQ context gain (+4.61) is comparable in scale to its reference seed difference (+3.98); a reference seed does not replicate the context contrast. Raw's 0.20-point context difference should be treated as unresolved.

Frame recall by class follows. All subject/class breakdowns retain the evaluation's original dataset-level Hungarian mapping; no per-subject remapping was introduced.




| action | VQ reference | VQ short | raw guarded | raw reference | raw short |
|---|---|---|---|---|---|
| down_by_elevator | 14.45 | 13.44 | 5.93 | 0.029 | 0.00 |
| going_down | 21.60 | 48.58 | 39.08 | 55.89 | 47.18 |
| going_up | 52.16 | 43.41 | 30.81 | 59.42 | 58.18 |
| running | 61.14 | 76.61 | 59.99 | 62.42 | 54.20 |
| sitting | 88.09 | 77.80 | 96.43 | 99.29 | 73.31 |
| sitting_down | 10.08 | 4.06 | 13.61 | 0.083 | 4.74 |
| standing | 43.96 | 49.13 | 58.96 | 72.60 | 91.99 |
| standing_up | 10.08 | 0.379 | 0.00 | 0.00 | 14.61 |
| up_by_elevator | 17.62 | 13.69 | 24.41 | 17.97 | 0.00 |
| walking | 42.98 | 40.48 | 41.85 | 38.25 | 36.59 |

Raw reference seed 111 has almost no recall for sitting_down (0.08%), standing_up (0.00%) or down_by_elevator (0.03%). The highest headline score, raw seed 222 at 52.96 F1@50, still has recall **0.00% sitting_down**, **20.43% standing_up**, **12.84% up_by_elevator**, and **5.64% down_by_elevator**. A high aggregate score does not establish recovery of the complete action inventory.

Raw short context improves standing recall to 91.99% but loses both elevator classes almost entirely, while sitting recall decreases to 73.31%. Its near-tied F1@50 conceals a different semantic partition. VQ short context improves running and going_down but reduces sitting and the two short transition recalls. These are useful failure targets, not evidence that every action benefited.

![Class recall](class_recall.png)


**9. OT convergence and objective alignment**

There are **44/44 outer-capped fits**, including all four cap-150 fits. No stored outer objective increases were detected. Last relative objective changes range from about 4.4e−5 to 8.5e−3, above the 1e−6 stopping tolerance; these are budgeted fits, not convergence-certified optima.

The cap-150 checks use the exact same initial prototypes and reproduce the cap-50 objective traces for their first 50 iterations with zero difference. They then lower the objective further, while F1@50 falls:

| job_id | objective_at50 | objective_at150 | F1_at50 | F1_at150 |
|---|---|---|---|---|
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone | -115860.92 | -118834.76 | 38.57 | 38.43 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone | -164171.42 | -168921.11 | 26.14 | 25.74 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone | -59700.15 | -76810.52 | 48.64 | 45.73 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone | -94226.04 | -102562.13 | 40.22 | 39.08 |

This shows that more optimization of the declared objective does not guarantee better label agreement along these trajectories. It does not establish a globally optimal solution, nor that 50 is the best outer budget. Objective values should be compared along the same representation/checkpoint trajectory, not used to rank independently trained feature spaces.

For final inference, **42/44 distinct evaluations** change F1@50 by less than one point from rung 500 to 4,000. The exceptions are native-VQ matched-count selected (+6.28) and native-VQ combined at step 1,000 (+3.49). All eight main selected rankings are unchanged between those two rungs. This supports cheap screening with an explicit sensitivity check; it does not certify every candidate at 500.

At rung 500, 130–285 recordings per evaluation remain capped (median 192); at 4,000, 4–72 remain capped (median 16). Score stability is therefore not equivalent to numerical convergence. The original report counts 58 role-level comparisons; the appropriate distinct-evaluation denominator for these solver summaries is 44.

The 150-outer checks cover final checkpoints of reference/combined only. They do not test the selected checkpoints that give the main table, the short-context leader, or seed 222. The exact headline scores remain conditional on the declared outer budget.


**10. Mapping sensitivity and interpretation**

Independent Hungarian maps are the established scoring protocol, and maximize frame agreement rather than segmental F1. Independently trained partitions can change their mapping even when many frame assignments remain similar up to state permutation.

For raw seed 222 final versus selected, optimally state-aligned frame agreement is **84.05%**, but mapped-label agreement under the two independent maps is **65.72%**. For raw reference versus short context these agreements are **77.07% versus 57.20%**. Thus equal or changed aggregate scores cannot be interpreted as only boundary refinement.

The supplementary table aligns states using prediction-only maximum frame agreement and then holds either model's action mapping fixed. `own` is the official result; `A`/`B` are post-hoc sensitivities, not replacement scores. Alignment is itself diagnostic and can be ambiguous; this is not a unique causal decomposition of mapping and representation effects.

| comparison | mapping_policy | F1_A | F1_B | delta_F1 | MoF_A | MoF_B |
|---|---|---|---|---|---|---|
| raw_guard_off | own | 36.08 | 50.00 | 13.92 | 43.87 | 49.98 |
| raw_guard_off | A | 36.08 | 37.92 | 1.84 | 43.87 | 42.55 |
| raw_guard_off | B | 29.64 | 50.00 | 20.36 | 37.15 | 49.98 |
| native_vq_short_context | own | 38.81 | 43.42 | 4.61 | 41.51 | 43.14 |
| native_vq_short_context | A | 38.81 | 37.35 | -1.46 | 41.51 | 38.40 |
| native_vq_short_context | B | 38.53 | 43.42 | 4.89 | 38.93 | 43.14 |
| raw_seed222_final_to_selected | own | 44.69 | 52.96 | 8.27 | 51.04 | 51.37 |
| raw_seed222_final_to_selected | A | 44.69 | 38.34 | -6.35 | 51.04 | 45.16 |
| raw_seed222_final_to_selected | B | 48.38 | 52.96 | 4.58 | 47.41 | 51.37 |
| raw_short_context | own | 50.00 | 49.80 | -0.200 | 49.98 | 48.82 |
| raw_short_context | A | 50.00 | 48.14 | -1.86 | 49.98 | 47.58 |
| raw_short_context | B | 47.53 | 49.80 | 2.27 | 46.50 | 48.82 |
| raw_combined_outer50_to150 | own | 40.22 | 39.08 | -1.14 | 44.35 | 44.36 |
| raw_combined_outer50_to150 | A | 40.22 | 33.32 | -6.90 | 44.35 | 40.29 |
| raw_combined_outer50_to150 | B | 37.95 | 39.08 | 1.13 | 42.55 | 44.36 |

For example, VQ short context's official +4.61 becomes −1.46 when both aligned partitions use the reference mapping, but +4.89 when both use the short-context mapping. Raw seed 222's final-to-selected difference likewise changes sign under one fixed-map diagnostic. These alternatives are not estimates of how much of a gain is "caused by mapping": transferring a map between changed partitions imposes an additional assumption. They demonstrate why a gain under the official protocol cannot automatically be described as improved boundaries or stable semantic identities. Independent Hungarian matching remains the appropriate declared score for these independently discovered states.


**11. Runtime and missing evidence**

| kind | jobs | minutes | median_seconds | share_elapsed_pct |
|---|---|---|---|---|
| eval_cap150 | 4 | 57.47 | 849.07 | 15.14 |
| eval_cap50 | 40 | 277.94 | 407.83 | 73.24 |
| train | 14 | 43.44 | 180.66 | 11.44 |

Total elapsed time is **6 h 19 min 31 s**. Training accounts for about 43.4 minutes (11.4%); OT evaluation and extraction account for about 335.4 minutes (88.4%). Standard evaluation median is 6.80 minutes and cap-150 evaluation median is 14.15 minutes. Cost planning should prioritize the number of decoded checkpoints and solver budgets, while preserving agreed sensitivity checks.

One timing anomaly is native-VQ genuine 3k training taking 245.97 s versus 211.86 s for the 5k reference in training_info. Logged target exposures and update counts are correct; wall time is not proportional to update count. The artifacts do not establish the hardware/system reason, so do not infer it. The large-model probe was not run here and no capacity cost bound follows.

No default planned cell is missing. Evidence not obtained: seed replication of changed configurations; a factorial schedule × context study; a target-count/position-matched raw guard comparison; unbiased-mask comparisons; cap-150 results for the selected leaders; alternative vocabularies or patch sizes; new FSQ training; LARa or subject-disjoint transfer; the declared future architecture search.


**12. Decisions supported, and open questions**

The pilot supports keeping the raw guard off and retaining span 8 as a working setting for these caches. It supports retaining native-VQ short context as a candidate and using validation-selected checkpoints as a practical policy, while saving fixed endpoints and replaying patience. It does not establish a universal optimum for horizon, context or OT budget.

Before expanding the search, declare whether the historical asymmetric mask sampler remains the protocol. If it is changed, label the new protocol explicitly and retain matched historical references. Before freezing a VQ horizon or context choice, replicate the relevant contrast; a second reference seed is not a second short-context or 3k-schedule result. Cheap final-inference screening can be considered only with promotion to the full budget and explicit handling of score-sensitive cells.

Open questions:

- How much of the long-mask loss remains under a sampler without directional trimming bias?
- Does the large VQ 3k schedule effect replicate, and how does it interact with context length?
- Can short transition and elevator recall improve without sacrificing the aggregate segmentation gains?
- How much of the raw guard benefit is due to more targets versus adding easier edge targets?
- Which configurations preserve improvements across initialization, subject populations and solver budgets?

What these numbers cannot establish: superiority of raw versus VQ tokenization in general; failure of vector quantization; successful FSQ action recovery; a best architecture; a paper-level novelty or SOTA claim. The two pipelines differ in input path, supervision eligibility, tokenizer provenance and parameter count (924,800 raw versus 1,043,712 VQ); only within-pipeline interventions are directly calibrated. Action labels were used in pooled Hungarian mapping and all scoring/diagnostics. No analysis here adds label supervision to training or selects an unplanned checkpoint.


**13. Reproducible analysis artifacts**

- [Audit results](audit.json); [analysis script](analyze.py); [supplementary diagnostics script](additional_diagnostics.py); [plot regeneration](plot_results.py); [report generator](write_analysis.py).
- [All final role-level results with segment diagnostics](all_final_rows_with_diagnostics.csv); [independently rescored distinct evaluations](rescored_final.csv).
- [Matched contrasts and subject win/loss counts](paired_comparisons.csv); [per-class metrics](class_metrics.csv); [per-subject metrics](subject_metrics.csv); [per-recording diagnostics](recording_diagnostics.csv).
- [Checkpoint trajectories](checkpoint_trajectories.csv); [full validation histories](validation_histories.csv); [within-job loss associations](within_job_loss_association.csv).
- [Actual exposure](exposure_analysis.csv); [mask-position audit](mask_position_audit.csv); [outer-objective diagnostics](objective_diagnostics.csv); [outer-prefix verification](outer_continuation_audit.csv); [final-rung sensitivity](rung_sensitivity.csv); [mapping sensitivity](mapping_sensitivity.csv); [runtime breakdown](runtime_summary.csv).
- Figures are saved as PNG and PDF. Subject metrics use the original global map, and all contrasts are descriptive. No uncertainty intervals or significance tests are reported for temporally dependent pooled-fit observations.

From the repository root, use the smq Python interpreter to run `analysis/analyze.py`, then `analysis/additional_diagnostics.py`, then `analysis/write_analysis.py` (paths relative to this run directory). `plot_results.py` regenerates the figures without repeating the audit.

