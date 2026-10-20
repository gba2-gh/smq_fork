# Contextualizer search calibration — ctx_cal_smoke_final

> **SMOKE RUN — shrunken budgets on a recording subset. Not a reproduction, not scientific evidence.**

Manifest: `manifest.json` (status `finished`, implementation `711c45a3146228da`). Sources: `C:\Users\gzaz976\Documents\repo\SMQ\results\quantized_input_comparison\qic_v3`, `C:\Users\gzaz976\Documents\repo\SMQ\results\quantized_input_comparison\task2\task2_v1`.

Completed 14/14 training jobs and 58/58 planned OT fits (before alias de-duplication). Unfinished units: 0 (see `not_run.csv`).

Reproduction gate against Task 2: **not_applicable_smoke** (`reproduction.json`).

Pooling: tokenizer/OT fitting and scoring pool all HuGaDB recordings, including the masked-validation recordings, so decoding is transductive and development-only. Development F1@50 selects procedure only; nothing here is a test result or a tokenizer comparison.

## 1. Main configurations

Validation-selected checkpoint, outer cap 3; endpoints are the first and last final-solve rungs (500 / 4000 in a full run). Raw is the primary guard-radius-0 pipeline; native VQ uses guard radius 14.

**native_vq**

| config | ctx/span/stride/batch | selected step | val loss | F1@50 first rung | F1@50 last rung | MoF | Edit | F1@10 | F1@25 |
|---|---|---|---|---|---|---|---|---|---|
| reference | 128/8/64/64 | 40 | 4.936 | 1.50 | 2.04 | 39.88 | 8.52 | 7.40 | 3.32 |
| short_context | 64/8/32/128 | 40 | 4.964 | 2.08 | 2.47 | 35.54 | 10.35 | 6.79 | 4.07 |
| long_mask | 128/16/64/64 | 40 | 4.961 | 1.36 | 1.80 | 40.76 | 7.61 | 7.19 | 3.40 |
| combined | 64/16/32/128 | 40 | 5.010 | 1.71 | 2.89 | 34.49 | 10.71 | 6.29 | 4.15 |

**raw**

| config | ctx/span/stride/batch | selected step | val loss | F1@50 first rung | F1@50 last rung | MoF | Edit | F1@10 | F1@25 |
|---|---|---|---|---|---|---|---|---|---|
| reference | 128/8/64/64 | 40 | 5.066 | 12.07 | 21.33 | 51.39 | 39.81 | 44.55 | 39.34 |
| short_context | 64/8/32/128 | 40 | 5.028 | 9.87 | 15.25 | 50.83 | 27.56 | 33.03 | 25.77 |
| long_mask | 128/16/64/64 | 40 | 5.017 | 15.44 | 32.80 | 55.91 | 54.99 | 56.04 | 47.38 |
| combined | 64/16/32/128 | 40 | 5.090 | 22.16 | 32.49 | 54.80 | 48.60 | 49.43 | 45.77 |

Validation loss is not comparable across configurations that change the supervised targets (context, span, guard).

## 2. Target-count sensitivity (native VQ, guard 14)

| job | eligible targets (total) | real-frame weight (total) | F1@50 first rung (selected) | F1@50 last rung (selected) | F1@50 last rung (final) |
|---|---|---|---|---|---|
| span 8 reference | 29053 | 435776 | 1.50 | 2.04 | 2.04 |
| span 16 long mask | 38729 | 580935 | 1.36 | 1.80 | 1.80 |
| span 16, targets matched to span 8 | 29053 | 435795 | 1.48 | 1.85 | 1.85 |

Pairing check (matched job's reference counts equal the span-8 job's eligible counts at every update): **verified**. Matched job: 0 mask retries; achieved targets equal reference at 40/40 updates; real-frame weight 435795 versus the span-8 job's 435776. Retry-based resampling conditions the mask distribution on feasibility. Count equality does not equalize terminal-window frame weights.

This is a supervision-count sensitivity (long gaps versus more supervised targets); it does not isolate the guard and does not change the tokenizer.

## 3. Genuine 3,000-update schedule versus 5,000 (reference configuration)

| pipeline | schedule | selected step | F1@50 last rung (selected) | F1@50 last rung (final) | val loss |
|---|---|---|---|---|---|
| native_vq | 3k horizon | 24 | 1.78 | 1.78 | 5.365 |
| native_vq | 5k horizon | 40 | 2.04 | 2.04 | 4.936 |
| raw | 3k horizon | 24 | 14.04 | 14.04 | 5.474 |
| raw | 5k horizon | 40 | 21.33 | 21.33 | 5.066 |

Intermediate 1k/3k checkpoints of a 5k schedule measure progress under that schedule; only the 3k jobs have a cosine horizon of 3,000.

## 4. Checkpoint-policy replay

Fixed final, full-history validation-selected and patience-1000 replay (strict improvement, earliest ties). F1@50 at the last rung (4000 in a full run), outer cap 3.

| job | selected step | replay stop | replay step | final | selected | replay | 1k | 3k |
|---|---|---|---|---|---|---|---|---|
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone | 40 | no stop | 40 | 2.04 | 2.04 | 2.04 | 12.30 | 1.99 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone | 40 | no stop | 40 | n/a | 18.92 | 18.92 | n/a | n/a |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone | 40 | no stop | 40 | 2.47 | 2.47 | 2.47 | 6.66 | 2.39 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone | 40 | no stop | 40 | 1.80 | 1.80 | 1.80 | 13.46 | 2.04 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone | 40 | no stop | 40 | 2.89 | 2.89 | 2.89 | 10.00 | 2.75 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone | 40 | no stop | 40 | 21.33 | 21.33 | 21.33 | 7.92 | 33.56 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone | 40 | no stop | 40 | 15.25 | 15.25 | 15.25 | 7.56 | 32.00 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone | 40 | no stop | 40 | 32.80 | 32.80 | 32.80 | 7.75 | 30.95 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone | 40 | no stop | 40 | 32.49 | 32.49 | 32.49 | 7.78 | 34.93 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone | 24 | no stop | 24 | 1.78 | 1.78 | n/a | n/a | n/a |
| raw__ctx128_span8_g0_h3000_seed111_tcnone | 24 | no stop | 24 | 14.04 | 14.04 | n/a | n/a | n/a |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone | 40 | no stop | 40 | 3.73 | 3.73 | 3.73 | n/a | n/a |
| raw__ctx128_span8_g0_h5000_seed222_tcnone | 40 | no stop | 40 | 5.57 | 5.57 | 5.57 | n/a | n/a |
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch | 40 | no stop | 40 | 1.85 | 1.85 | n/a | n/a | n/a |

Policies are compared on development segmentation only; an arbitrary checkpoint is never chosen by its highest evaluation score. Small or inconsistent differences are unresolved, not a winner.

## 5. Reference-seed replication (seed 111 versus 222)

| pipeline | policy | seed 111 | seed 222 | difference |
|---|---|---|---|---|
| native_vq | selected | 2.04 | 3.73 | 1.69 |
| native_vq | final | 2.04 | 3.73 | 1.69 |
| native_vq | replay | 2.04 | 3.73 | 1.69 |
| raw | selected | 21.33 | 5.57 | -15.76 |
| raw | final | 21.33 | 5.57 | -15.76 |
| raw | replay | 21.33 | 5.57 | -15.76 |

Unequal seed coverage: one extra seed per primary pipeline, reference configuration only.

## 6. Solver-budget effects

Final-rung continuation: F1@50 change between the first and last rung of each evaluation (flag at 1 point, a practical flag only). 40 of 58 evaluations flagged.

| evaluation | first rung | last rung | dF1@50 | flag | recording statuses (last rung) | outer status | outer iterations |
|---|---|---|---|---|---|---|---|
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch::final::cap3 | 20 | 40 | 0.37 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcmatch::selected::cap3 | 20 | 40 | 0.37 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 2.17 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 0.57 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 0.44 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 0.44 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span16_g14_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 0.44 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone::final::cap3 | 20 | 40 | 1.69 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h3000_seed111_tcnone::selected::cap3 | 20 | 40 | 1.69 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 3.36 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 0.38 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 0.54 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::final::cap6 | 20 | 40 | 0.65 |  | {'capped': 12} | capped | 6 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 0.54 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 0.54 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::final::cap3 | 20 | 40 | 1.36 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::replay::cap3 | 20 | 40 | 1.36 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx128_span8_g14_h5000_seed222_tcnone::selected::cap3 | 20 | 40 | 1.36 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 0.55 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 0.73 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 1.18 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::final::cap6 | 20 | 40 | 0.35 |  | {'capped': 12} | capped | 6 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 1.18 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 1.18 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 1.09 | FLAG | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 0.60 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 0.39 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 0.39 |  | {'capped': 12} | capped | 3 |
| native_vq__ctx64_span8_g14_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 0.39 |  | {'capped': 12} | capped | 3 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 2.46 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 13.09 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 17.36 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 17.36 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span16_g0_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 17.36 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h3000_seed111_tcnone::final::cap3 | 20 | 40 | 5.83 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h3000_seed111_tcnone::selected::cap3 | 20 | 40 | 5.83 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 2.56 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 11.34 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 9.26 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::final::cap6 | 20 | 40 | 8.62 | FLAG | {'capped': 12} | capped | 6 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 9.26 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 9.26 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::final::cap3 | 20 | 40 | 1.79 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::replay::cap3 | 20 | 40 | 1.79 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g0_h5000_seed222_tcnone::selected::cap3 | 20 | 40 | 1.79 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 7.65 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx128_span8_g14_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 7.65 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 2.60 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 8.44 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 10.33 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::final::cap6 | 20 | 40 | 10.65 | FLAG | {'capped': 12} | capped | 6 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 10.33 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 10.33 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_1k::cap3 | 20 | 40 | 2.16 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_3k::cap3 | 20 | 40 | 11.41 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::ckpt_5k::cap3 | 20 | 40 | 5.38 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::replay::cap3 | 20 | 40 | 5.38 | FLAG | {'capped': 12} | capped | 3 |
| raw__ctx64_span8_g0_h5000_seed111_tcnone::selected::cap3 | 20 | 40 | 5.38 | FLAG | {'capped': 12} | capped | 3 |

Outer-iteration cap 3 versus 6 (final 5k checkpoint, same initialization, separate fits):

| job | F1@50 last rung, base cap | sensitivity cap | difference | first rung, base cap | first rung, sensitivity cap |
|---|---|---|---|---|---|
| native_vq__ctx128_span8_g14_h5000_seed111_tcnone | 2.04 | 2.31 | 0.27 | 1.50 | 1.66 |
| native_vq__ctx64_span16_g14_h5000_seed111_tcnone | 2.89 | 2.31 | -0.58 | 1.71 | 1.96 |
| raw__ctx128_span8_g0_h5000_seed111_tcnone | 21.33 | 20.75 | -0.58 | 12.07 | 12.13 |
| raw__ctx64_span16_g0_h5000_seed111_tcnone | 32.49 | 36.95 | 4.46 | 22.16 | 26.30 |

Ordering (native_vq): cap 3 combined > reference; cap 6 combined > reference.

Ordering (raw): cap 3 combined > reference; cap 6 combined > reference.

Configuration ordering by F1@50 (native_vq, selected checkpoint): first rung short_context > combined > reference > long_mask; last rung combined > short_context > reference > long_mask — **ordering reversal**.

Configuration ordering by F1@50 (raw, selected checkpoint): first rung combined > long_mask > reference > short_context; last rung long_mask > combined > reference > short_context — **ordering reversal**.

Outer-fit status: see `ot_status` in `segmentation.csv`; outer-capped evaluations are budgeted outcomes, not converged fits. Evaluations with any capped recording at the last rung: 58.

## 7. Missing cells and limits

All planned units completed in this invocation.

Limits: development data only; one tokenizer checkpoint family; seed 111 (seed 222 only for the reference configuration); GPU training is not guaranteed bit-reproducible across devices; 1 point is a flag, not a significance level. This calibration does not rank tokenizers.
