# Time-scale inspection, phase 8a: gravity- and heading-aware relational tokens — ts5_v1

Status `finished`. Rules declared in PHASE8A_GRAVITY_HEADING_TOKENS.md before the run. Primary metric F1@50 at the last rung, paired with T2 (span 2, same seeds) from ts5_v1. The bar (mean gain >= 3, every seed positive) applies only to cells with three seeds; two-seed cells are screening.

## Gates

| gate | status | detail |
|---|---|---|
| g3 | passed |  |
| g8a | passed | {'same_recordings': True, 'codes': True, 'continuous': True, 'starts': True, 'lengths': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True, 'lookup': True} |
| g8b_babel2 | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True} {'head_minus_feet_median': [11.34, 513.0, 1175.44], 'order_heights': [362.7, 229.5, -200.5, -511.0, -856.3], 'hips_vertical_share_median': 0.09216251150629659, 'shoulders_vertical_share_median': 0.059260222776176076, 'hip_shoulder_axis_angle_median_deg': 3.939919127415764, 'frames_checked': 15919} |
| g8b_babel3 | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True} {'head_minus_feet_median': [21.63, 470.19, 1413.21], 'order_heights': [431.4, 283.5, -206.8, -560.8, -999.7], 'hips_vertical_share_median': 0.11843435848957122, 'shoulders_vertical_share_median': 0.09079052239278383, 'hip_shoulder_axis_angle_median_deg': 5.773083108808783, 'frames_checked': 16547} |
| g8b_lara | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True, 'collars_coincide': True} {'head_minus_feet_median': [4.25, 23.44, 1475.59], 'order_heights': [647.1, 538.4, 461.0, 110.3, 0.0, -429.7, -835.5], 'hips_vertical_share_median': 0.03598150081051062, 'shoulders_vertical_share_median': 0.07884607992786088, 'hip_shoulder_axis_angle_median_deg': 5.700381674983155, 'frames_checked': 526800} |
| g8c | passed | {'fixed_rotation_invariant': True, 'turn_adds_omega': True, 'turn_leaves_other_blocks': True, 'windows_tested': True} |


## T5 vs T2 (span 2, `context`, global Hungarian, last rung)

| dataset | tokenizer | T5 F1@50 per seed | T5 mean | T2 per seed (paired) | T2 mean (paired) | Δ F1@50 (per seed) | n | reading | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|---|---|---|---|
| babel2 | T5gh | 111:25.9 / 222:26.8 | 26.32 | 111:31.5 / 222:30.0 | 30.75 | -4.43 (-5.6 / -3.2) | 2 | screening only (2 seeds) | 2.9 | 3.7 |
| babel2 | T5h | 111:27.7 / 222:28.7 | 28.20 | 111:31.5 / 222:30.0 | 30.75 | -2.54 (-3.8 / -1.3) | 2 | screening only (2 seeds) | 3.2 | 3.7 |
| babel2 | T5g | 111:31.6 / 222:25.4 | 28.48 | 111:31.5 / 222:30.0 | 30.75 | -2.27 (+0.1 / -4.6) | 2 | screening only (2 seeds) | 2.8 | 3.7 |
| babel3 | T5gh | 111:13.0 / 222:14.8 | 13.88 | 111:20.1 / 222:22.8 | 21.45 | -7.57 (-7.1 / -8.0) | 2 | screening only (2 seeds) | 3.3 | 3.7 |
| babel3 | T5h | 111:20.9 / 222:27.5 / 1538574472:20.0 | 22.79 | 111:20.1 / 222:22.8 / 1538574472:26.8 | 23.23 | -0.44 (+0.8 / +4.6 / -6.8) | 3 | does not pass the bar | 3.3 | 3.7 |
| babel3 | T5g | 111:18.5 / 222:18.5 | 18.50 | 111:20.1 / 222:22.8 | 21.45 | -2.94 (-1.6 / -4.3) | 2 | screening only (2 seeds) | 3.3 | 3.7 |
| lara | T5gh | 111:25.2 / 222:22.4 | 23.80 | 111:23.4 / 222:22.6 | 23.04 | +0.75 (+1.8 / -0.3) | 2 | screening only (2 seeds) | 25.9 | 32.1 |

### Q1: is T5gh better than T2?

- **babel2** (primary; stand_up + kick = 19% of segments): -4.43 (-5.6 / -3.2) over 2 seeds -> screening only (2 seeds)
- **babel3** (non-inferiority; no gain predicted): F1@50 -7.57 (-7.1 / -8.0), MoF -0.38 (-1.1 / +0.4) over 2 seeds -> **LOSS >= 3**
- **lara** (non-inferiority; no gain predicted): F1@50 +0.75 (+1.8 / -0.3), MoF +3.96 (+5.4 / +2.6) over 2 seeds -> no loss >= 3 points

## Third-seed rule

A cell gets the third seed only if its two-seed mean F1@50 is >= 1 above T2's on the same seeds. Decided from the saved screening units before any third-seed job started.

| dataset / tokenizer | screening Δ F1@50 | per seed | decision | reason |
|---|---|---|---|---|
| babel2 / t5_both | -4.43 | -5.6 / -3.2 | no third seed | below threshold |
| babel2 / t5_gravity | -2.27 | +0.1 / -4.6 | no third seed | below threshold |
| babel2 / t5_heading | -2.54 | -3.8 / -1.3 | no third seed | below threshold |
| babel3 / t5_both | -7.57 | -7.1 / -8.0 | no third seed | below threshold |
| babel3 / t5_gravity | -2.94 | -1.6 / -4.3 | no third seed | below threshold |
| babel3 / t5_heading | +2.74 | +0.8 / +4.6 | third seed | promising |
| lara / t5_both | +0.75 | +1.8 / -0.3 | no third seed | below threshold |


## Every metric (paired with T2 on the common seeds)

| dataset | tokenizer | metric | Δ (per seed) | n | bar |
|---|---|---|---|---|---|
| babel2 | T5gh | MoF | -2.41 (-6.9 / +2.1) | 2 | screening |
| babel2 | T5gh | Edit | -3.06 (-8.8 / +2.7) | 2 | screening |
| babel2 | T5gh | F1@10 | -6.08 (-9.7 / -2.5) | 2 | screening |
| babel2 | T5gh | F1@25 | -8.01 (-10.7 / -5.3) | 2 | screening |
| babel2 | T5gh | F1@50 | -4.43 (-5.6 / -3.2) | 2 | screening |
| babel2 | T5h | MoF | -0.99 (-8.3 / +6.3) | 2 | screening |
| babel2 | T5h | Edit | -3.13 (-7.0 / +0.7) | 2 | screening |
| babel2 | T5h | F1@10 | -2.62 (-3.9 / -1.3) | 2 | screening |
| babel2 | T5h | F1@25 | -2.38 (-3.4 / -1.3) | 2 | screening |
| babel2 | T5h | F1@50 | -2.54 (-3.8 / -1.3) | 2 | screening |
| babel2 | T5g | MoF | -3.10 (-5.5 / -0.7) | 2 | screening |
| babel2 | T5g | Edit | -2.16 (-2.6 / -1.8) | 2 | screening |
| babel2 | T5g | F1@10 | -2.89 (-2.5 / -3.3) | 2 | screening |
| babel2 | T5g | F1@25 | -2.55 (-1.2 / -3.9) | 2 | screening |
| babel2 | T5g | F1@50 | -2.27 (+0.1 / -4.6) | 2 | screening |
| babel3 | T5gh | MoF | -0.38 (-1.1 / +0.4) | 2 | screening |
| babel3 | T5gh | Edit | -12.28 (-12.7 / -11.9) | 2 | screening |
| babel3 | T5gh | F1@10 | -13.64 (-14.0 / -13.3) | 2 | screening |
| babel3 | T5gh | F1@25 | -12.92 (-11.9 / -13.9) | 2 | screening |
| babel3 | T5gh | F1@50 | -7.57 (-7.1 / -8.0) | 2 | screening |
| babel3 | T5h | MoF | +1.38 (+0.3 / +7.1 / -3.3) | 3 | no |
| babel3 | T5h | Edit | -4.89 (-4.7 / +1.5 / -11.4) | 3 | no |
| babel3 | T5h | F1@10 | -3.21 (-3.4 / +2.9 / -9.0) | 3 | no |
| babel3 | T5h | F1@25 | -2.83 (-1.8 / +2.7 / -9.4) | 3 | no |
| babel3 | T5h | F1@50 | -0.44 (+0.8 / +4.6 / -6.8) | 3 | no |
| babel3 | T5g | MoF | -0.62 (+1.4 / -2.7) | 2 | screening |
| babel3 | T5g | Edit | -6.54 (-8.8 / -4.2) | 2 | screening |
| babel3 | T5g | F1@10 | -5.91 (-6.7 / -5.1) | 2 | screening |
| babel3 | T5g | F1@25 | -6.30 (-5.7 / -6.9) | 2 | screening |
| babel3 | T5g | F1@50 | -2.94 (-1.6 / -4.3) | 2 | screening |
| lara | T5gh | MoF | +3.96 (+5.4 / +2.6) | 2 | screening |
| lara | T5gh | Edit | -1.17 (-0.3 / -2.0) | 2 | screening |
| lara | T5gh | F1@10 | +0.65 (+2.3 / -1.0) | 2 | screening |
| lara | T5gh | F1@25 | +0.69 (+2.6 / -1.2) | 2 | screening |
| lara | T5gh | F1@50 | +0.75 (+1.8 / -0.3) | 2 | screening |

## Q3: which channel? (F1@50 minus T2, common seeds; descriptive)

| dataset | seeds | vertical only (T5g) | heading only (T5h) | both (T5gh) | interaction (gh − g − h) |
|---|---|---|---|---|---|
| babel2 | 111,222 | -2.27 (+0.1 / -4.6) | -2.54 (-3.8 / -1.3) | -4.43 (-5.6 / -3.2) | +0.38 (-1.9 / +2.7) |
| babel3 | 111,222 | -2.94 (-1.6 / -4.3) | +2.74 (+0.8 / +4.6) | -7.57 (-7.1 / -8.0) | -7.36 (-6.4 / -8.4) |


## Q2: per-class frame recall and precision, T2 vs T5 (uses labels; analysis only)

Mean over the common seeds; global Hungarian mapping as the evaluator (MoF is reproduced by the same mapping). * marks the classes the plan predicts to benefit. Small classes are noisy under a one-to-one mapping.

| dataset | tokenizer | class | frame share % | n seeds | T2 recall | T5 recall | Δ recall | T2 precision | T5 precision |
|---|---|---|---|---|---|---|---|---|---|
| babel2 | T5gh | sit | 50.7 | 2 | 45.0 | 38.7 | -6.3 | 80.4 | 87.8 |
| babel2 | T5gh | run | 8.5 | 2 | 48.2 | 51.1 | +2.9 | 21.8 | 23.3 |
| babel2 | T5gh | stand_up * | 9.5 | 2 | 15.1 | 41.6 | +26.5 | 8.8 | 16.6 |
| babel2 | T5gh | kick * | 3.9 | 2 | 21.6 | 40.7 | +19.1 | 5.0 | 10.8 |
| babel2 | T5gh | none | 27.4 | 2 | 48.2 | 38.2 | -10.1 | 55.7 | 55.4 |
| babel2 | T5h | sit | 50.7 | 2 | 45.0 | 49.1 | +4.1 | 80.4 | 79.3 |
| babel2 | T5h | run | 8.5 | 2 | 48.2 | 48.7 | +0.4 | 21.8 | 24.9 |
| babel2 | T5h | stand_up * | 9.5 | 2 | 15.1 | 11.2 | -3.9 | 8.8 | 7.0 |
| babel2 | T5h | kick * | 3.9 | 2 | 21.6 | 6.5 | -15.0 | 5.0 | 1.7 |
| babel2 | T5h | none | 27.4 | 2 | 48.2 | 40.4 | -7.8 | 55.7 | 51.5 |
| babel2 | T5g | sit | 50.7 | 2 | 45.0 | 35.6 | -9.4 | 80.4 | 75.0 |
| babel2 | T5g | run | 8.5 | 2 | 48.2 | 67.4 | +19.2 | 21.8 | 28.2 |
| babel2 | T5g | stand_up * | 9.5 | 2 | 15.1 | 20.3 | +5.2 | 8.8 | 13.4 |
| babel2 | T5g | kick * | 3.9 | 2 | 21.6 | 1.9 | -19.7 | 5.0 | 0.4 |
| babel2 | T5g | none | 27.4 | 2 | 48.2 | 49.3 | +1.1 | 55.7 | 56.4 |
| babel3 | T5gh | jog | 6.4 | 2 | 83.2 | 38.8 | -44.5 | 31.4 | 16.7 |
| babel3 | T5gh | wave * | 1.6 | 2 | 7.3 | 40.0 | +32.7 | 0.8 | 2.9 |
| babel3 | T5gh | dance | 43.1 | 2 | 31.4 | 31.2 | -0.1 | 58.0 | 59.2 |
| babel3 | T5gh | gesture | 25.8 | 2 | 25.4 | 39.6 | +14.2 | 28.4 | 42.2 |
| babel3 | T5gh | none | 23.1 | 2 | 38.1 | 30.9 | -7.2 | 41.1 | 40.7 |
| babel3 | T5h | jog | 6.4 | 3 | 83.2 | 70.7 | -12.6 | 29.9 | 26.2 |
| babel3 | T5h | wave * | 1.6 | 3 | 22.6 | 2.1 | -20.5 | 2.0 | 0.2 |
| babel3 | T5h | dance | 43.1 | 3 | 30.1 | 30.0 | -0.1 | 63.6 | 57.7 |
| babel3 | T5h | gesture | 25.8 | 3 | 32.9 | 25.7 | -7.2 | 38.3 | 35.6 |
| babel3 | T5h | none | 23.1 | 3 | 39.2 | 58.5 | +19.2 | 41.0 | 48.5 |
| babel3 | T5g | jog | 6.4 | 2 | 83.2 | 40.8 | -42.4 | 31.4 | 18.2 |
| babel3 | T5g | wave * | 1.6 | 2 | 7.3 | 12.6 | +5.3 | 0.8 | 1.1 |
| babel3 | T5g | dance | 43.1 | 2 | 31.4 | 31.9 | +0.5 | 58.0 | 53.2 |
| babel3 | T5g | gesture | 25.8 | 2 | 25.4 | 26.2 | +0.8 | 28.4 | 38.1 |
| babel3 | T5g | none | 23.1 | 2 | 38.1 | 45.1 | +7.0 | 41.1 | 42.4 |
| lara | T5gh | Standing | 11.2 | 2 | 27.3 | 15.8 | -11.5 | 18.3 | 14.5 |
| lara | T5gh | Walking | 12.4 | 2 | 69.0 | 11.7 | -57.3 | 42.8 | 17.1 |
| lara | T5gh | Cart | 11.9 | 2 | 19.8 | 74.9 | +55.1 | 23.0 | 45.1 |
| lara | T5gh | Handling(upwards) | 7.8 | 2 | 50.4 | 63.2 | +12.8 | 27.0 | 41.1 |
| lara | T5gh | Handling(centred) | 43.7 | 2 | 25.3 | 33.5 | +8.2 | 68.4 | 73.1 |
| lara | T5gh | Handling(downwards) | 7.0 | 2 | 48.7 | 66.0 | +17.3 | 48.0 | 51.5 |
| lara | T5gh | Synchronization | 1.8 | 2 | 98.6 | 98.3 | -0.3 | 26.5 | 23.7 |
| lara | T5gh | None | 4.2 | 2 | 11.4 | 10.9 | -0.5 | 6.1 | 4.6 |

## Tokenizer diagnostics (label-free)

| dataset | tokenizer | seed | share of PCA-64 variance by block | block weight vs dist | occupied codes | PCA-64 explained | heading undefined share |
|---|---|---|---|---|---|---|---|
| babel2 | T5gh | 111 | dist 0.374 / vertical 0.396 / heading 0.230 | {'vertical': 3.427, 'heading': 12.959} | 512 | 0.960 | 0.000 |
| babel2 | T5gh | 222 | dist 0.374 / vertical 0.396 / heading 0.230 | {'vertical': 3.427, 'heading': 12.959} | 512 | 0.960 | 0.000 |
| babel3 | T5gh | 111 | dist 0.392 / vertical 0.368 / heading 0.241 | {'vertical': 3.219, 'heading': 12.905} | 512 | 0.954 | 0.003 |
| babel3 | T5gh | 222 | dist 0.391 / vertical 0.369 / heading 0.240 | {'vertical': 3.223, 'heading': 12.897} | 512 | 0.954 | 0.003 |
| lara | T5gh | 111 | dist 0.469 / vertical 0.319 / heading 0.212 | {'vertical': 2.609, 'heading': 9.631} | 512 | 0.940 | 0.000 |
| lara | T5gh | 222 | dist 0.470 / vertical 0.318 / heading 0.212 | {'vertical': 2.601, 'heading': 9.637} | 512 | 0.941 | 0.000 |
| babel2 | T5g | 111 | dist 0.487 / vertical 0.513 | {'vertical': 3.427} | 512 | 0.951 | — |
| babel2 | T5g | 222 | dist 0.487 / vertical 0.513 | {'vertical': 3.427} | 512 | 0.951 | — |
| babel3 | T5g | 111 | dist 0.516 / vertical 0.484 | {'vertical': 3.219} | 512 | 0.944 | — |
| babel3 | T5g | 222 | dist 0.516 / vertical 0.484 | {'vertical': 3.223} | 512 | 0.944 | — |
| babel2 | T5h | 111 | dist 0.626 / heading 0.374 | {'heading': 12.959} | 512 | 0.960 | 0.000 |
| babel2 | T5h | 222 | dist 0.626 / heading 0.374 | {'heading': 12.959} | 512 | 0.960 | 0.000 |
| babel3 | T5h | 111 | dist 0.627 / heading 0.373 | {'heading': 12.905} | 512 | 0.956 | 0.003 |
| babel3 | T5h | 1538574472 | dist 0.626 / heading 0.374 | {'heading': 12.912} | 512 | 0.956 | 0.003 |
| babel3 | T5h | 222 | dist 0.627 / heading 0.373 | {'heading': 12.897} | 512 | 0.956 | 0.003 |

