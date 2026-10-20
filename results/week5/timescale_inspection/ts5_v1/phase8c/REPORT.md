# Time-scale inspection, phase 8c: which invariance? — ts5_v1

Status `finished`. Rules declared in PHASE8C_SYMMETRY_TEST.md (Revision 2) before the run. Primary metric F1@50 at the last rung, mask span 2, paired by seed; the bar is a mean gain >= 3 with every seed in the same direction and is read only with three seeds. 'Within the bar' means comparable performance, not equivalence.

## Gates

| gate | status | detail |
|---|---|---|
| g3 | passed |  |
| g8b_babel2 | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True} |
| g8b_babel3 | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True} |
| g8b_lara | passed | {'root_zero': True, 'axis2_vertical': True, 'anatomical_order': True, 'hips_horizontal': True, 'shoulders_horizontal': True, 'hip_shoulder_agree': True, 'collars_coincide': True} |
| g8c1 | passed | {'same_recordings': True, 'codes': True, 'continuous': True, 'starts': True, 'lengths': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True, 'lookup': True} |
| g8c2 | passed | {'babel2': {'stats_equal': True, 'flat_equal': True, 'positions_unchanged': True}, 'lara': {'stats_equal': True, 'flat_equal': True, 'positions_unchanged': True}} |
| g8c3 | passed | {'babel2': {'fixed_rotation_invariant': True, 'turn_changes_features': True, 'mirror_changes_features': True, 'windows_tested': True}, 'lara': {'fixed_rotation_invariant': True, 'turn_changes_features': True, 'mirror_changes_features': True, 'windows_tested': True}} |
| g8c4 | passed | {'same_recordings': True, 'codes': True, 'continuous': True, 'starts': True, 'lengths': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True, 'lookup': True} |


## Cells (F1@50, span 2, `context`)

| dataset | tokenizer | per seed | n | mean | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|
| babel2 | T0 | 111:15.1 / 222:20.7 / 1538574472:16.8 | 3 | 17.56 | 2.5 | 3.7 |
| babel2 | T2 | 111:31.5 / 222:30.0 / 1538574472:30.5 | 3 | 30.68 | 3.0 | 3.7 |
| babel2 | T6 | 111:15.0 / 222:16.9 / 1538574472:18.4 | 3 | 16.80 | 2.6 | 3.7 |
| babel2 | T6s | 111:20.4 / 222:21.2 / 1538574472:21.8 | 3 | 21.13 | 2.5 | 3.7 |
| babel2 | T5gh-std | 111:26.9 / 222:35.0 / 1538574472:22.9 | 3 | 28.28 | 3.0 | 3.7 |
| babel2 | T5gh (weighted) | 111:25.9 / 222:26.8 / 1538574472:24.3 | 3 | 25.63 | 2.9 | 3.7 |
| babel3 | T0 | 111:13.6 / 222:11.7 / 1538574472:14.7 | 3 | 13.32 | 3.2 | 3.7 |
| babel3 | T2 | 111:20.1 / 222:22.8 / 1538574472:26.8 | 3 | 23.23 | 3.3 | 3.7 |
| babel3 | T6 | 111:12.8 / 222:13.1 / 1538574472:12.3 | 3 | 12.76 | 3.5 | 3.7 |
| babel3 | T6s | 111:23.2 / 222:17.2 / 1538574472:15.3 | 3 | 18.58 | 3.0 | 3.7 |
| babel3 | T5gh-std | 111:19.0 / 222:15.4 / 1538574472:20.9 | 3 | 18.45 | 3.5 | 3.7 |
| babel3 | T5gh (weighted) | 111:13.0 / 222:14.8 / 1538574472:21.3 | 3 | 16.36 | 3.3 | 3.7 |
| lara | T2 | 111:23.4 / 222:22.6 / 1538574472:26.0 | 3 | 24.02 | 24.7 | 32.1 |
| lara | T6 | 111:17.4 / 222:16.7 / 1538574472:15.0 | 3 | 16.37 | 28.1 | 32.1 |
| lara | T6s | 111:21.2 / 222:20.0 / 1538574472:22.3 | 3 | 21.15 | 27.9 | 32.1 |

## Declared comparisons and descriptive ones (A minus B, paired by seed)

### Q1: T6 vs T0: does removing absolute heading (keeping gravity and chirality) help?

- **babel2**: -0.76 (-0.1 / -3.8 / +1.6) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: -0.56 (-0.8 / +1.5 / -2.3) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q2: T6s vs T2: at the same temporal summary, canonical coordinates vs pairwise distances

- **babel2**: -9.55 (-11.1 / -8.8 / -8.8) over 3 seeds -> **B better (reverse passes the bar)**
- **babel3**: -4.65 (+3.2 / -5.6 / -11.5) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: -2.87 (-2.3 / -2.6 / -3.7) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q3: T5gh-std vs weighted T5gh: did 8a's block weighting cause its loss?

- **babel2**: +2.65 (+1.0 / +8.2 / -1.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +2.09 (+6.1 / +0.6 / -0.4) over 3 seeds -> within the bar: comparable performance, not equivalence

### D1 (descriptive): T6 vs T2 (descriptive: invariance, temporal summary, layout and codebook all differ)

- **babel2**: -13.88 (-16.5 / -13.1 / -12.1) over 3 seeds -> **B better (reverse passes the bar)**
- **babel3**: -10.47 (-7.2 / -9.7 / -14.5) over 3 seeds -> **B better (reverse passes the bar)**
- **lara**: -7.66 (-6.1 / -5.9 / -11.0) over 3 seeds -> **B better (reverse passes the bar)**

### D2 (descriptive): T5gh-std vs T2 (descriptive)

- **babel2**: -2.40 (-4.6 / +5.0 / -7.6) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: -4.78 (-1.0 / -7.4 / -5.9) over 3 seeds -> **B better (reverse passes the bar)**

### LARa non-inferiority (no loss >= 3 F1@50 or MoF against T2)

- T6: F1@50 -7.66 (-6.1 / -5.9 / -11.0), MoF -7.74 (-8.1 / -5.3 / -9.8) over 3 seeds -> **LOSS >= 3**
- T6s: F1@50 -2.87 (-2.3 / -2.6 / -3.7), MoF -0.74 (+1.5 / -1.2 / -2.5) over 3 seeds -> no loss >= 3 points

## Every metric

| id | dataset | A − B | metric | Δ (per seed) | n | reading |
|---|---|---|---|---|---|---|
| Q1 | babel2 | T6 − T0 | MoF | -1.68 (+0.4 / -4.9 / -0.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel2 | T6 − T0 | Edit | -5.29 (-3.4 / -10.3 / -2.1) | 3 | **B better (reverse passes the bar)** |
| Q1 | babel2 | T6 − T0 | F1@10 | -1.96 (-2.9 / -3.8 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel2 | T6 − T0 | F1@25 | -0.95 (-2.3 / -3.4 / +2.8) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel2 | T6 − T0 | F1@50 | -0.76 (-0.1 / -3.8 / +1.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel3 | T6 − T0 | MoF | -2.21 (-5.4 / -0.2 / -0.9) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel3 | T6 − T0 | Edit | -2.52 (-4.4 / +1.8 / -5.0) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel3 | T6 − T0 | F1@10 | -0.49 (-4.1 / +4.0 / -1.5) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel3 | T6 − T0 | F1@25 | -0.07 (-2.5 / +2.8 / -0.4) | 3 | within the bar: comparable performance, not equivalence |
| Q1 | babel3 | T6 − T0 | F1@50 | -0.56 (-0.8 / +1.5 / -2.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | T6s − T2 | MoF | -4.01 (-9.4 / +3.3 / -5.9) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | T6s − T2 | Edit | -0.07 (-3.9 / -1.5 / +5.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | T6s − T2 | F1@10 | -8.13 (-10.7 / -10.7 / -3.0) | 3 | **B better (reverse passes the bar)** |
| Q2 | babel2 | T6s − T2 | F1@25 | -9.77 (-12.4 / -10.6 / -6.3) | 3 | **B better (reverse passes the bar)** |
| Q2 | babel2 | T6s − T2 | F1@50 | -9.55 (-11.1 / -8.8 / -8.8) | 3 | **B better (reverse passes the bar)** |
| Q2 | babel3 | T6s − T2 | MoF | -1.24 (+2.8 / -2.3 / -4.2) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | T6s − T2 | Edit | -10.20 (-3.7 / -10.9 / -16.0) | 3 | **B better (reverse passes the bar)** |
| Q2 | babel3 | T6s − T2 | F1@10 | -9.78 (-2.2 / -10.1 / -17.0) | 3 | **B better (reverse passes the bar)** |
| Q2 | babel3 | T6s − T2 | F1@25 | -9.04 (+0.4 / -10.5 / -17.0) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | T6s − T2 | F1@50 | -4.65 (+3.2 / -5.6 / -11.5) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | T6s − T2 | MoF | -0.74 (+1.5 / -1.2 / -2.5) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | T6s − T2 | Edit | -2.37 (-4.3 / -3.3 / +0.5) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | T6s − T2 | F1@10 | -4.32 (-4.8 / -5.3 / -2.9) | 3 | **B better (reverse passes the bar)** |
| Q2 | lara | T6s − T2 | F1@25 | -4.39 (-4.3 / -5.0 / -3.8) | 3 | **B better (reverse passes the bar)** |
| Q2 | lara | T6s − T2 | F1@50 | -2.87 (-2.3 / -2.6 / -3.7) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel2 | T5gh-std − T5gh (weighted) | MoF | +3.39 (+3.6 / +9.4 / -2.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel2 | T5gh-std − T5gh (weighted) | Edit | +0.02 (-1.3 / +4.2 / -2.8) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel2 | T5gh-std − T5gh (weighted) | F1@10 | +1.93 (+1.9 / +5.8 / -1.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel2 | T5gh-std − T5gh (weighted) | F1@25 | +2.80 (+3.3 / +8.8 / -3.8) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel2 | T5gh-std − T5gh (weighted) | F1@50 | +2.65 (+1.0 / +8.2 / -1.3) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel3 | T5gh-std − T5gh (weighted) | MoF | -2.39 (-1.2 / -4.2 / -1.7) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel3 | T5gh-std − T5gh (weighted) | Edit | +2.40 (+9.5 / -3.5 / +1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel3 | T5gh-std − T5gh (weighted) | F1@10 | +4.06 (+10.9 / -1.0 / +2.2) | 3 | within the bar: comparable performance, not equivalence |
| Q3 | babel3 | T5gh-std − T5gh (weighted) | F1@25 | +4.45 (+10.5 / +0.9 / +1.9) | 3 | **A better (passes the bar)** |
| Q3 | babel3 | T5gh-std − T5gh (weighted) | F1@50 | +2.09 (+6.1 / +0.6 / -0.4) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | T6 − T2 | MoF | -15.58 (-20.2 / -8.0 / -18.5) | 3 | **B better (reverse passes the bar)** |
| D1 | babel2 | T6 − T2 | Edit | -14.04 (-21.0 / -10.4 / -10.7) | 3 | **B better (reverse passes the bar)** |
| D1 | babel2 | T6 − T2 | F1@10 | -15.30 (-18.3 / -14.9 / -12.7) | 3 | **B better (reverse passes the bar)** |
| D1 | babel2 | T6 − T2 | F1@25 | -15.78 (-19.1 / -15.1 / -13.2) | 3 | **B better (reverse passes the bar)** |
| D1 | babel2 | T6 − T2 | F1@50 | -13.88 (-16.5 / -13.1 / -12.1) | 3 | **B better (reverse passes the bar)** |
| D1 | babel3 | T6 − T2 | MoF | -7.56 (-6.9 / -4.0 / -11.7) | 3 | **B better (reverse passes the bar)** |
| D1 | babel3 | T6 − T2 | Edit | -16.16 (-13.6 / -14.2 / -20.6) | 3 | **B better (reverse passes the bar)** |
| D1 | babel3 | T6 − T2 | F1@10 | -16.00 (-15.1 / -14.3 / -18.6) | 3 | **B better (reverse passes the bar)** |
| D1 | babel3 | T6 − T2 | F1@25 | -15.59 (-13.4 / -14.8 / -18.5) | 3 | **B better (reverse passes the bar)** |
| D1 | babel3 | T6 − T2 | F1@50 | -10.47 (-7.2 / -9.7 / -14.5) | 3 | **B better (reverse passes the bar)** |
| D1 | lara | T6 − T2 | MoF | -7.74 (-8.1 / -5.3 / -9.8) | 3 | **B better (reverse passes the bar)** |
| D1 | lara | T6 − T2 | Edit | -4.50 (-4.5 / -3.8 / -5.3) | 3 | **B better (reverse passes the bar)** |
| D1 | lara | T6 − T2 | F1@10 | -8.83 (-7.3 / -6.8 / -12.4) | 3 | **B better (reverse passes the bar)** |
| D1 | lara | T6 − T2 | F1@25 | -9.76 (-8.1 / -7.7 / -13.5) | 3 | **B better (reverse passes the bar)** |
| D1 | lara | T6 − T2 | F1@50 | -7.66 (-6.1 / -5.9 / -11.0) | 3 | **B better (reverse passes the bar)** |
| D2 | babel2 | T5gh-std − T2 | MoF | -0.67 (-3.3 / +11.5 / -10.2) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | T5gh-std − T2 | Edit | -0.78 (-10.1 / +6.8 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | T5gh-std − T2 | F1@10 | -2.92 (-7.8 / +3.3 / -4.3) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | T5gh-std − T2 | F1@25 | -3.67 (-7.4 / +3.5 / -7.1) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | T5gh-std − T2 | F1@50 | -2.40 (-4.6 / +5.0 / -7.6) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | T5gh-std − T2 | MoF | -4.65 (-2.4 / -3.9 / -7.8) | 3 | **B better (reverse passes the bar)** |
| D2 | babel3 | T5gh-std − T2 | Edit | -8.01 (-3.2 / -15.3 / -5.5) | 3 | **B better (reverse passes the bar)** |
| D2 | babel3 | T5gh-std − T2 | F1@10 | -7.49 (-3.1 / -14.3 / -5.1) | 3 | **B better (reverse passes the bar)** |
| D2 | babel3 | T5gh-std − T2 | F1@25 | -7.01 (-1.4 / -13.0 / -6.6) | 3 | **B better (reverse passes the bar)** |
| D2 | babel3 | T5gh-std − T2 | F1@50 | -4.78 (-1.0 / -7.4 / -5.9) | 3 | **B better (reverse passes the bar)** |

## Per-class frame recall and precision against T2 (uses labels; analysis only)

Mean over the common seeds, global Hungarian mapping as the evaluator. * marks 8a's targeted classes. T0 predictions at span 2 are not saved, so T0 is not in this table. Small classes are noisy under a one-to-one mapping.

| dataset | tokenizer | class | frame share % | n seeds | T2 recall | recall | Δ recall | T2 precision | precision |
|---|---|---|---|---|---|---|---|---|---|
| babel2 | T6 | sit | 50.7 | 3 | 48.6 | 29.3 | -19.3 | 82.9 | 62.8 |
| babel2 | T6 | run | 8.5 | 3 | 56.1 | 45.5 | -10.6 | 23.9 | 18.5 |
| babel2 | T6 | stand_up * | 9.5 | 3 | 24.0 | 17.9 | -6.1 | 13.0 | 8.6 |
| babel2 | T6 | kick * | 3.9 | 3 | 15.4 | 14.8 | -0.6 | 3.7 | 4.2 |
| babel2 | T6 | none | 27.4 | 3 | 43.4 | 27.7 | -15.8 | 56.6 | 38.6 |
| babel2 | T6s | sit | 50.7 | 3 | 48.6 | 44.3 | -4.3 | 82.9 | 78.0 |
| babel2 | T6s | run | 8.5 | 3 | 56.1 | 53.5 | -2.5 | 23.9 | 28.8 |
| babel2 | T6s | stand_up * | 9.5 | 3 | 24.0 | 33.3 | +9.4 | 13.0 | 16.6 |
| babel2 | T6s | kick * | 3.9 | 3 | 15.4 | 40.4 | +25.0 | 3.7 | 9.1 |
| babel2 | T6s | none | 27.4 | 3 | 43.4 | 30.8 | -12.7 | 56.6 | 48.8 |
| babel2 | T5gh-std | sit | 50.7 | 3 | 48.6 | 45.7 | -2.9 | 82.9 | 81.5 |
| babel2 | T5gh-std | run | 8.5 | 3 | 56.1 | 68.3 | +12.3 | 23.9 | 29.7 |
| babel2 | T5gh-std | stand_up * | 9.5 | 3 | 24.0 | 23.1 | -0.9 | 13.0 | 14.2 |
| babel2 | T5gh-std | kick * | 3.9 | 3 | 15.4 | 8.1 | -7.3 | 3.7 | 1.9 |
| babel2 | T5gh-std | none | 27.4 | 3 | 43.4 | 43.9 | +0.5 | 56.6 | 57.8 |
| babel2 | T5gh (weighted) | sit | 50.7 | 3 | 48.6 | 40.5 | -8.1 | 82.9 | 84.2 |
| babel2 | T5gh (weighted) | run | 8.5 | 3 | 56.1 | 47.7 | -8.3 | 23.9 | 23.9 |
| babel2 | T5gh (weighted) | stand_up * | 9.5 | 3 | 24.0 | 37.5 | +13.5 | 13.0 | 15.1 |
| babel2 | T5gh (weighted) | kick * | 3.9 | 3 | 15.4 | 32.3 | +16.9 | 3.7 | 8.7 |
| babel2 | T5gh (weighted) | none | 27.4 | 3 | 43.4 | 39.0 | -4.4 | 56.6 | 54.8 |
| babel3 | T6 | jog | 6.4 | 3 | 83.2 | 29.7 | -53.5 | 29.9 | 14.4 |
| babel3 | T6 | wave * | 1.6 | 3 | 22.6 | 34.6 | +12.0 | 2.0 | 2.9 |
| babel3 | T6 | dance | 43.1 | 3 | 30.1 | 25.7 | -4.4 | 63.6 | 44.9 |
| babel3 | T6 | gesture | 25.8 | 3 | 32.9 | 27.7 | -5.2 | 38.3 | 32.8 |
| babel3 | T6 | none | 23.1 | 3 | 39.2 | 34.5 | -4.7 | 41.0 | 39.6 |
| babel3 | T6s | jog | 6.4 | 3 | 83.2 | 26.3 | -57.0 | 29.9 | 11.9 |
| babel3 | T6s | wave * | 1.6 | 3 | 22.6 | 19.2 | -3.5 | 2.0 | 2.1 |
| babel3 | T6s | dance | 43.1 | 3 | 30.1 | 27.3 | -2.8 | 63.6 | 53.6 |
| babel3 | T6s | gesture | 25.8 | 3 | 32.9 | 36.5 | +3.6 | 38.3 | 36.6 |
| babel3 | T6s | none | 23.1 | 3 | 39.2 | 51.1 | +11.9 | 41.0 | 48.2 |
| babel3 | T5gh-std | jog | 6.4 | 3 | 83.2 | 62.3 | -20.9 | 29.9 | 21.6 |
| babel3 | T5gh-std | wave * | 1.6 | 3 | 22.6 | 27.6 | +5.0 | 2.0 | 2.4 |
| babel3 | T5gh-std | dance | 43.1 | 3 | 30.1 | 23.7 | -6.4 | 63.6 | 54.0 |
| babel3 | T5gh-std | gesture | 25.8 | 3 | 32.9 | 31.5 | -1.4 | 38.3 | 38.0 |
| babel3 | T5gh-std | none | 23.1 | 3 | 39.2 | 38.1 | -1.2 | 41.0 | 38.6 |
| babel3 | T5gh (weighted) | jog | 6.4 | 3 | 83.2 | 38.6 | -44.7 | 29.9 | 16.6 |
| babel3 | T5gh (weighted) | wave * | 1.6 | 3 | 22.6 | 28.4 | +5.8 | 2.0 | 2.1 |
| babel3 | T5gh (weighted) | dance | 43.1 | 3 | 30.1 | 30.4 | +0.3 | 63.6 | 61.2 |
| babel3 | T5gh (weighted) | gesture | 25.8 | 3 | 32.9 | 33.4 | +0.5 | 38.3 | 36.4 |
| babel3 | T5gh (weighted) | none | 23.1 | 3 | 39.2 | 40.5 | +1.2 | 41.0 | 42.4 |

## Tokenizer diagnostics (label-free)

| dataset | tokenizer | seed | pre-PCA variance share by block | share of PCA-64 variance by block | occupied codes | PCA-64 explained | windows left unrotated |
|---|---|---|---|---|---|---|---|
| babel2 | t5_both | 1538574472 | — | dist 0.374 / vertical 0.396 / heading 0.230 | 512 | 0.960 | — |
| babel3 | t5_both | 1538574472 | — | dist 0.391 / vertical 0.368 / heading 0.241 | 512 | 0.954 | — |
| babel2 | t5_both_std | 111 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.923 / vertical 0.075 / heading 0.002 | 512 | 0.930 | — |
| babel2 | t5_both_std | 1538574472 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.923 / vertical 0.075 / heading 0.002 | 512 | 0.930 | — |
| babel2 | t5_both_std | 222 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.923 / vertical 0.075 / heading 0.002 | 512 | 0.930 | — |
| babel3 | t5_both_std | 111 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.924 / vertical 0.075 / heading 0.001 | 512 | 0.923 | — |
| babel3 | t5_both_std | 1538574472 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.924 / vertical 0.075 / heading 0.001 | 512 | 0.923 | — |
| babel3 | t5_both_std | 222 | dist 0.920 / vertical 0.077 / heading 0.003 | dist 0.924 / vertical 0.075 / heading 0.001 | 512 | 0.923 | — |
| babel2 | t6_heading_canonical | 111 | all 1.000 | all 1.000 | 512 | 0.998 | 0.0001 |
| babel2 | t6_heading_canonical | 1538574472 | all 1.000 | all 1.000 | 512 | 0.998 | 0.0001 |
| babel2 | t6_heading_canonical | 222 | all 1.000 | all 1.000 | 512 | 0.998 | 0.0001 |
| babel3 | t6_heading_canonical | 111 | all 1.000 | all 1.000 | 512 | 0.996 | 0.0017 |
| babel3 | t6_heading_canonical | 1538574472 | all 1.000 | all 1.000 | 512 | 0.996 | 0.0017 |
| babel3 | t6_heading_canonical | 222 | all 1.000 | all 1.000 | 512 | 0.997 | 0.0017 |
| lara | t6_heading_canonical | 111 | all 1.000 | all 1.000 | 512 | 0.996 | 0.0000 |
| lara | t6_heading_canonical | 1538574472 | all 1.000 | all 1.000 | 512 | 0.996 | 0.0000 |
| lara | t6_heading_canonical | 222 | all 1.000 | all 1.000 | 512 | 0.996 | 0.0000 |
| babel2 | t6s_heading_canonical_stats | 111 | all 1.000 | all 1.000 | 512 | 0.990 | 0.0001 |
| babel2 | t6s_heading_canonical_stats | 1538574472 | all 1.000 | all 1.000 | 512 | 0.990 | 0.0001 |
| babel2 | t6s_heading_canonical_stats | 222 | all 1.000 | all 1.000 | 512 | 0.990 | 0.0001 |
| babel3 | t6s_heading_canonical_stats | 111 | all 1.000 | all 1.000 | 512 | 0.988 | 0.0017 |
| babel3 | t6s_heading_canonical_stats | 1538574472 | all 1.000 | all 1.000 | 512 | 0.988 | 0.0017 |
| babel3 | t6s_heading_canonical_stats | 222 | all 1.000 | all 1.000 | 512 | 0.988 | 0.0017 |
| lara | t6s_heading_canonical_stats | 111 | all 1.000 | all 1.000 | 512 | 0.981 | 0.0000 |
| lara | t6s_heading_canonical_stats | 1538574472 | all 1.000 | all 1.000 | 512 | 0.981 | 0.0000 |
| lara | t6s_heading_canonical_stats | 222 | all 1.000 | all 1.000 | 512 | 0.981 | 0.0000 |

