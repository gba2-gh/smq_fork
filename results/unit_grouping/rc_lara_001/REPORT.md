# Residual co-occurrence grouping rc_lara_001


LARa; the 24 states, prototypes and fitting plans of ug_lara_001, re-decoded on the held-out recordings; groupings into 8 actions; `b0_c` = Milestone 1 8-state B0.

| seed | arm | MoF | F1@50 | balanced acc | ARI vs oracle | classes with a group |
|---|---|---|---|---|---|---|
| 111 | b0_c | 39.4 | 18.8 | 47.0 | - | - |
| 111 | group_residual | 34.2 | 12.2 | 41.9 | 0.32 | 5 |
| 111 | group_temporal | 32.3 | 10.2 | 28.5 | 0.07 | 3 |
| 111 | group_shuffled | 30.5 | 9.1 | 27.5 | 0.01 | 3 |
| 111 | group_appearance | 36.4 | 11.8 | 40.2 | 0.19 | 4 |
| 111 | random (size-matched to residual) | mean 30.5 / p95 36.9 | mean 9.3 / p95 11.4 | - | - | - |
| 111 | residual share of affinity / ARI residual vs temporal / parity exact | 0.035 | 0.08 | True | | |
| 222 | b0_c | 38.4 | 20.1 | 46.3 | - | - |
| 222 | group_residual | 34.5 | 12.0 | 33.2 | 0.08 | 4 |
| 222 | group_temporal | 35.2 | 13.4 | 39.0 | 0.13 | 4 |
| 222 | group_shuffled | 40.2 | 16.4 | 43.6 | 0.17 | 5 |
| 222 | group_appearance | 37.4 | 15.4 | 47.1 | 0.13 | 4 |
| 222 | random (size-matched to residual) | mean 28.9 / p95 33.1 | mean 9.2 / p95 11.3 | - | - | - |
| 222 | residual share of affinity / ARI residual vs temporal / parity exact | 0.041 | 0.22 | True | | |
| 1538574472 | b0_c | 38.8 | 20.1 | 45.8 | - | - |
| 1538574472 | group_residual | 35.9 | 11.8 | 35.3 | 0.15 | 3 |
| 1538574472 | group_temporal | 34.6 | 14.2 | 41.4 | 0.14 | 4 |
| 1538574472 | group_shuffled | 34.8 | 11.0 | 34.9 | 0.18 | 4 |
| 1538574472 | group_appearance | 33.8 | 13.6 | 43.1 | 0.20 | 4 |
| 1538574472 | random (size-matched to residual) | mean 28.2 / p95 33.3 | mean 9.1 / p95 10.8 | - | - | - |
| 1538574472 | residual share of affinity / ARI residual vs temporal / parity exact | 0.056 | 0.39 | True | | |

## Decision (declared rule, primary split)

**STOP**

- a_vs_b0: fails
- b_vs_appearance: fails
- c_vs_shuffled: fails
- d_vs_random_p95: fails
- residual minus temporal (mean MoF, F1@50): +0.78, -0.57 (reported, not gated)

Subject-disjoint folds, rule applied descriptively: **STOP**.