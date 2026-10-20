# Contextualizer search calibration — ctx_cal_v1

Manifest: `manifest.json` (status `finished`, implementation `711c45a3146228da`). Sources: `C:\Users\gzaz976\Documents\repo\SMQ\results\quantized_input_comparison\qic_v3`, `C:\Users\gzaz976\Documents\repo\SMQ\results\quantized_input_comparison\task2\task2_v1`.

Completed 14/14 training jobs and 58/58 planned OT fits (before alias de-duplication). Unfinished units: 0 (see `not_run.csv`).

Reproduction gate against Task 2: **passed** (`reproduction.json`).

Pooling: tokenizer/OT fitting and scoring pool all HuGaDB recordings, including the masked-validation recordings, so decoding is transductive and development-only. Development F1@50 selects procedure only; nothing here is a test result or a tokenizer comparison.

## 1. Main configurations

Validation-selected checkpoint, outer cap 50; endpoints are the first and last final-solve rungs (500 / 4000 in a full run). Raw is the primary guard-radius-0 pipeline; native VQ uses guard radius 14.

**native_vq**

| config | ctx/span/stride/batch | selected step | val loss | F1@50 first rung | F1@50 last rung | MoF | Edit | F1@10 | F1@25 |
|---|---|---|---|---|---|---|---|---|---|
| reference | 128/8/64/64 | 4500 | 2.161 | 38.78 | 38.81 | 41.51 | 39.47 | 46.09 | 42.14 |
| short_context | 64/8/32/128 | 4000 | 2.285 | 43.32 | 43.42 | 43.14 | 41.46 | 48.03 | 46.35 |
| long_mask | 128/16/64/64 | 5000 | 2.315 | 33.61 | 33.64 | 42.44 | 30.63 | 38.55 | 36.41 |
| combined | 64/16/32/128 | 4300 | 2.449 | 22.33 | 22.22 | 32.28 | 29.65 | 32.20 | 27.64 |

**raw**

| config | ctx/span/stride/batch | selected step | val loss | F1@50 first rung | F1@50 last rung | MoF | Edit | F1@10 | F1@25 |
|---|---|---|---|---|---|---|---|---|---|
| reference | 128/8/64/64 | 4100 | 1.810 | 49.86 | 50.00 | 49.98 | 45.31 | 53.47 | 52.33 |
| short_context | 64/8/32/128 | 4200 | 1.894 | 49.61 | 49.80 | 48.82 | 46.40 | 55.52 | 52.53 |
| long_mask | 128/16/64/64 | 3700 | 1.885 | 38.05 | 38.01 | 47.73 | 39.31 | 44.39 | 41.56 |
| combined | 64/16/32/128 | 3500 | 2.017 | 42.82 | 42.76 | 47.07 | 39.91 | 47.14 | 45.53 |

Validation loss is not comparable across configurations that change the supervised targets (context, span, guard).

## 2. Target-count sensitivity (native VQ, guard 14)

| job | eligible targets (total) | real-frame weight (total) | F1@50 first rung (selected) | F1@50 last rung (selected) | F1@50 last rung (final) |
|---|---|---|---|---|---|
| span 8 reference | 6432577 | 96484146 | 38.78 | 38.81 | 38.57 |
| span 16 long mask | 8568753 | 128530037 | 33.61 | 33.64 | 33.64 |
| span 16, targets matched to span 8 | 6432577 | 96487708 | 21.92 | 28.20 | 29.01 |

Pairing check (matched job's reference counts equal the span-8 job's eligible counts at every update): **verified**. Matched job: 0 mask retries; achieved targets equal reference at 5000/5000 updates; real-frame weight 96487708 versus the span-8 job's 96484146. Retry-based resampling conditions the mask distribution on feasibility. Count equality does not equalize terminal-window frame weights.

This is a supervision-count sensitivity (long gaps versus more supervised targets); it does not isolate the guard and does not change the tokenizer.

## 3. Genuine 3,000-update schedule versus 5,000 (reference configuration)

| pipeline | schedule | selected step | F1@50 last rung (selected) | F1@50 last rung (final) | val loss |
|---|---|---|---|---|---|
| native_vq | 3k horizon | 3000 | 43.03 | 43.03 | 2.223 |
| native_vq | 5k horizon | 4500 | 38.81 | 38.57 | 2.161 |
| raw | 3k horizon | 2800 | 45.70 | 45.16 | 1.840 |
| raw | 5k horizon | 4100 | 50.00 | 48.64 | 1.810 |

Intermediate 1k/3k checkpoints of a 5k schedule measure progress under that schedule; only the 3k jobs have a cosine horizon of 3,000.

## 4. Checkpoint-policy replay

Fixed final, full-history validation-selected and patience-1000 replay (strict improvement, earliest ties). F1@50 at the last rung (4000 in a full run), outer cap 50.

| job | selected step | replay stop | replay step | final | selected | replay | 1k | 3k |
|---|---|---|---|---|---|---|---|---|
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone | 4500 | no stop | 4500 | 38.57 | 38.81 | 38.81 | 26.45 | 19.15 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone | 4400 | no stop | 4400 | n/a | 36.08 | 36.08 | n/a | n/a |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone | 4000 | 5000 | 4000 | 43.18 | 43.42 | 43.42 | 22.37 | 32.83 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone | 5000 | no stop | 5000 | 33.64 | 33.64 | 33.64 | 14.14 | 31.82 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone | 4300 | no stop | 4300 | 26.14 | 22.22 | 22.22 | 16.42 | 26.66 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone | 4100 | no stop | 4100 | 48.64 | 50.00 | 50.00 | 24.44 | 41.36 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone | 4200 | no stop | 4200 | 48.46 | 49.80 | 49.80 | 36.75 | 51.16 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone | 3700 | 4700 | 3700 | 36.61 | 38.01 | 38.01 | 11.51 | 44.26 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone | 3500 | 4500 | 3500 | 40.22 | 42.76 | 42.76 | 31.33 | 44.81 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone | 3000 | no stop | 3000 | 43.03 | 43.03 | n/a | n/a | n/a |
| raw__ctx128_span8_g0_h3000_seed111_tcnone | 2800 | no stop | 2800 | 45.16 | 45.70 | n/a | n/a | n/a |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone | 5000 | no stop | 5000 | 42.79 | 42.79 | 42.79 | n/a | n/a |
| raw__ctx128_span8_g0_h5000_seed222_tcnone | 4400 | no stop | 4400 | 44.69 | 52.96 | 52.96 | n/a | n/a |
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch | 4200 | no stop | 4200 | 29.01 | 28.20 | n/a | n/a | n/a |

Policies are compared on development segmentation only; an arbitrary checkpoint is never chosen by its highest evaluation score. Small or inconsistent differences are unresolved, not a winner.

## 5. Reference-seed replication (seed 111 versus 222)

| pipeline | policy | seed 111 | seed 222 | difference |
|---|---|---|---|---|
| native_vq | selected | 38.81 | 42.79 | 3.98 |
| native_vq | final | 38.57 | 42.79 | 4.22 |
| native_vq | replay | 38.81 | 42.79 | 3.98 |
| raw | selected | 50.00 | 52.96 | 2.96 |
| raw | final | 48.64 | 44.69 | -3.95 |
| raw | replay | 50.00 | 52.96 | 2.96 |

Unequal seed coverage: one extra seed per primary pipeline, reference configuration only.

## 6. Solver-budget effects

Final-rung continuation: F1@50 change between the first and last rung of each evaluation (flag at 1 point, a practical flag only). 2 of 58 evaluations flagged.

| evaluation | first rung | last rung | dF1@50 | flag | recording statuses (last rung) | outer status | outer iterations |
|---|---|---|---|---|---|---|---|
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch::final::cap50 | 500 | 4000 | -0.01 |  | {'converged': 348, 'capped': 16} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch::selected::cap50 | 500 | 4000 | 6.28 | FLAG | {'converged': 343, 'capped': 21} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | -0.01 |  | {'converged': 339, 'capped': 25} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | -0.03 |  | {'converged': 355, 'capped': 9} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.03 |  | {'converged': 359, 'capped': 5} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.03 |  | {'converged': 359, 'capped': 5} | capped | 50 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.03 |  | {'converged': 359, 'capped': 5} | capped | 50 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone::final::cap50 | 500 | 4000 | -0.04 |  | {'converged': 345, 'capped': 19} | capped | 50 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone::selected::cap50 | 500 | 4000 | -0.04 |  | {'converged': 345, 'capped': 19} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | -0.01 |  | {'converged': 292, 'capped': 72} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | -0.15 |  | {'converged': 340, 'capped': 24} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.09 |  | {'converged': 351, 'capped': 13} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::final::cap150 | 500 | 4000 | 0.06 |  | {'converged': 350, 'capped': 14} | capped | 150 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.03 |  | {'converged': 350, 'capped': 14} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.03 |  | {'converged': 350, 'capped': 14} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::final::cap50 | 500 | 4000 | 0.35 |  | {'converged': 334, 'capped': 30} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::replay::cap50 | 500 | 4000 | 0.35 |  | {'converged': 334, 'capped': 30} | capped | 50 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::selected::cap50 | 500 | 4000 | 0.35 |  | {'converged': 334, 'capped': 30} | capped | 50 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | 3.49 | FLAG | {'converged': 321, 'capped': 43} | capped | 50 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | 0.10 |  | {'converged': 345, 'capped': 19} | capped | 50 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.08 |  | {'converged': 337, 'capped': 27} | capped | 50 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::final::cap150 | 500 | 4000 | -0.07 |  | {'converged': 343, 'capped': 21} | capped | 150 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | -0.11 |  | {'converged': 330, 'capped': 34} | capped | 50 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | -0.11 |  | {'converged': 330, 'capped': 34} | capped | 50 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | 0.13 |  | {'capped': 59, 'converged': 305} | capped | 50 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | -0.05 |  | {'converged': 311, 'capped': 53} | capped | 50 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.00 |  | {'converged': 347, 'capped': 17} | capped | 50 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.10 |  | {'converged': 344, 'capped': 20} | capped | 50 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.10 |  | {'converged': 344, 'capped': 20} | capped | 50 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | -0.01 |  | {'converged': 347, 'capped': 17} | capped | 50 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | 0.02 |  | {'converged': 351, 'capped': 13} | capped | 50 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | -0.15 |  | {'converged': 345, 'capped': 19} | capped | 50 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | -0.04 |  | {'converged': 355, 'capped': 9} | capped | 50 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | -0.04 |  | {'converged': 355, 'capped': 9} | capped | 50 |
| raw__ctx128_span8_g0_h3000_seed111_tcnone::final::cap50 | 500 | 4000 | 0.10 |  | {'converged': 356, 'capped': 8} | capped | 50 |
| raw__ctx128_span8_g0_h3000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.06 |  | {'converged': 352, 'capped': 12} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | 0.07 |  | {'converged': 354, 'capped': 10} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | 0.29 |  | {'converged': 357, 'capped': 7} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.05 |  | {'converged': 343, 'capped': 21} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::final::cap150 | 500 | 4000 | 0.01 |  | {'converged': 355, 'capped': 9} | capped | 150 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.14 |  | {'converged': 357, 'capped': 7} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.14 |  | {'converged': 357, 'capped': 7} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::final::cap50 | 500 | 4000 | 0.00 |  | {'converged': 360, 'capped': 4} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::replay::cap50 | 500 | 4000 | -0.10 |  | {'converged': 358, 'capped': 6} | capped | 50 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::selected::cap50 | 500 | 4000 | -0.10 |  | {'converged': 358, 'capped': 6} | capped | 50 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.02 |  | {'converged': 354, 'capped': 10} | capped | 50 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.02 |  | {'converged': 354, 'capped': 10} | capped | 50 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | 0.07 |  | {'converged': 353, 'capped': 11} | capped | 50 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | -0.04 |  | {'converged': 348, 'capped': 16} | capped | 50 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.15 |  | {'converged': 332, 'capped': 32} | capped | 50 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::final::cap150 | 500 | 4000 | 0.01 |  | {'converged': 348, 'capped': 16} | capped | 150 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | -0.06 |  | {'converged': 347, 'capped': 17} | capped | 50 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | -0.06 |  | {'converged': 347, 'capped': 17} | capped | 50 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_1k::cap50 | 500 | 4000 | 0.05 |  | {'converged': 352, 'capped': 12} | capped | 50 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_3k::cap50 | 500 | 4000 | 0.20 |  | {'converged': 360, 'capped': 4} | capped | 50 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_5k::cap50 | 500 | 4000 | 0.15 |  | {'converged': 353, 'capped': 11} | capped | 50 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::replay::cap50 | 500 | 4000 | 0.19 |  | {'converged': 354, 'capped': 10} | capped | 50 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::selected::cap50 | 500 | 4000 | 0.19 |  | {'converged': 354, 'capped': 10} | capped | 50 |

Outer-iteration cap 50 versus 150 (final 5k checkpoint, same initialization, separate fits):

| job | F1@50 last rung, base cap | sensitivity cap | difference | first rung, base cap | first rung, sensitivity cap |
|---|---|---|---|---|---|
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone | 38.57 | 38.43 | -0.14 | 38.48 | 38.37 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone | 26.14 | 25.74 | -0.40 | 26.06 | 25.81 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone | 48.64 | 45.73 | -2.91 | 48.59 | 45.72 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone | 40.22 | 39.08 | -1.14 | 40.07 | 39.07 |

Ordering (native_vq): cap 50 reference > combined; cap 150 reference > combined.

Ordering (raw): cap 50 reference > combined; cap 150 reference > combined.

Configuration ordering by F1@50 (native_vq, selected checkpoint): first rung short_context > reference > long_mask > combined; last rung short_context > reference > long_mask > combined.

Configuration ordering by F1@50 (raw, selected checkpoint): first rung reference > short_context > combined > long_mask; last rung reference > short_context > combined > long_mask.

Outer-fit status: see `ot_status` in `segmentation.csv`; outer-capped evaluations are budgeted outcomes, not converged fits. Evaluations with any capped recording at the last rung: 58.

## 7. Missing cells and limits

All planned units completed in this invocation.

Limits: development data only; one tokenizer checkpoint family; seed 111 (seed 222 only for the reference configuration); GPU training is not guaranteed bit-reproducible across devices; 1 point is a flag, not a significance level. This calibration does not rank tokenizers.
