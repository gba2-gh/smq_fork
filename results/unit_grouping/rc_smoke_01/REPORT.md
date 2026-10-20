# Residual co-occurrence grouping rc_smoke_01

SMOKE RUN: plumbing only, not evidence.
LARa; the 24 states, prototypes and fitting plans of ug_lara_001, re-decoded on the held-out recordings; groupings into 8 actions; `b0_c` = Milestone 1 8-state B0.

| seed | arm | MoF | F1@50 | balanced acc | ARI vs oracle | classes with a group |
|---|---|---|---|---|---|---|
| 111 | b0_c | 39.4 | 18.8 | 47.0 | - | - |
| 111 | group_residual | 59.9 | 31.0 | 57.8 | 0.13 | 7 |
| 111 | group_temporal | 44.7 | 25.2 | 37.0 | 0.16 | 4 |
| 111 | group_shuffled | 33.3 | 13.7 | 23.3 | 0.18 | 4 |
| 111 | group_appearance | 41.0 | 14.7 | 38.9 | 0.20 | 6 |
| 111 | random (size-matched to residual) | mean 43.4 / p95 49.6 | mean 18.0 / p95 26.1 | - | - | - |
| 111 | residual share of affinity / ARI residual vs temporal / parity exact | 0.035 | 0.08 | False | | |

## Decision (declared rule, primary split)

**not computed (smoke run)**
