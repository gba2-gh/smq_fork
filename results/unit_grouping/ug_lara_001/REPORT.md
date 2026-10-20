# Unit-grouping test ug_lara_001


M = 24 states grouped into C = 8; LARa; held-out evaluation recordings; final solver endpoint; one Hungarian mapping over the evaluation recordings. `b0_c` = Milestone 1 C-state B0 cells (read, not re-run).

## Primary split (mean over seeds in parentheses where shown)

| arm | seed | MoF | F1@50 | balanced acc | classes with a group | ARI vs oracle | pred/true segments |
|---|---|---|---|---|---|---|---|
| b0_c | 111 | 39.4 | 18.8 | 47.0 | - | - | - |
| group_temporal | 111 | 32.3 | 10.2 | 28.5 | 3 | 0.07 | 2.23 |
| group_shuffled | 111 | 30.5 | 9.1 | 27.5 | 3 | 0.01 | 2.50 |
| group_appearance | 111 | 36.4 | 11.8 | 40.2 | 4 | 0.19 | 2.84 |
| group_random (100, size-matched) | 111 | mean 29.8 / p95 33.7 | mean 9.1 / p95 11.2 | - | - | - | - |
| (24 raw states: oracle many-to-one MoF bound) | 111 | 58.4 | - | - | - | - | - |
| b0_c | 222 | 38.4 | 20.1 | 46.3 | - | - | - |
| group_temporal | 222 | 35.2 | 13.4 | 39.0 | 4 | 0.13 | 2.26 |
| group_shuffled | 222 | 40.2 | 16.4 | 43.6 | 5 | 0.17 | 2.32 |
| group_appearance | 222 | 37.4 | 15.4 | 47.1 | 4 | 0.13 | 2.83 |
| group_random (100, size-matched) | 222 | mean 28.7 / p95 34.3 | mean 9.0 / p95 11.0 | - | - | - | - |
| (24 raw states: oracle many-to-one MoF bound) | 222 | 59.3 | - | - | - | - | - |
| b0_c | 1538574472 | 38.8 | 20.1 | 45.8 | - | - | - |
| group_temporal | 1538574472 | 34.6 | 14.2 | 41.4 | 4 | 0.14 | 2.42 |
| group_shuffled | 1538574472 | 34.8 | 11.0 | 34.9 | 4 | 0.18 | 2.44 |
| group_appearance | 1538574472 | 33.8 | 13.6 | 43.1 | 4 | 0.20 | 2.78 |
| group_random (100, size-matched) | 1538574472 | mean 28.1 / p95 33.1 | mean 9.0 / p95 11.5 | - | - | - | - |
| (24 raw states: oracle many-to-one MoF bound) | 1538574472 | 58.1 | - | - | - | - | - |

## Decision (declared rule, primary split)

**STOP**

- a_vs_b0: fails
- b_vs_appearance: fails
- c_vs_shuffled: fails
- d_vs_random_p95: fails

## Secondary: subject-disjoint folds (descriptive, mean over 4 folds)

Rule applied descriptively: **STOP**.

## Smoothing sensitivity (MoF/F1@50 at emission multipliers 0.5 and 2; multiplier 1 is the table above)

| arm | seed | x0.5 | x2 |
|---|---|---|---|
| group_temporal | 111 | 30.5/13.2 | 32.8/6.9 |
| group_shuffled | 111 | 29.7/13.1 | 30.6/5.9 |
| group_appearance | 111 | 37.1/16.4 | 35.4/7.8 |
| group_temporal | 222 | 35.8/17.6 | 35.4/9.8 |
| group_shuffled | 222 | 41.0/21.9 | 40.0/12.2 |
| group_appearance | 222 | 36.8/22.0 | 37.2/10.5 |
| group_temporal | 1538574472 | 35.3/19.4 | 33.4/8.2 |
| group_shuffled | 1538574472 | 35.6/15.7 | 33.5/7.0 |
| group_appearance | 1538574472 | 34.2/19.6 | 33.1/8.7 |

Scope: temporal grouping of transport states (symmetric affinity, reversal-invariant), not ordered composition and not grouping of raw tokens. The oracle is an upper bound on MoF for hard relabellings of the fixed 24 states only. Three seeds on one split are not confidence intervals.