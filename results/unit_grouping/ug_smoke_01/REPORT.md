# Unit-grouping test ug_smoke_01

SMOKE RUN: plumbing only, not evidence.
M = 24 states grouped into C = 8; LARa; held-out evaluation recordings; final solver endpoint; one Hungarian mapping over the evaluation recordings. `b0_c` = Milestone 1 C-state B0 cells (read, not re-run).

## Primary split (mean over seeds in parentheses where shown)

| arm | seed | MoF | F1@50 | balanced acc | classes with a group | ARI vs oracle | pred/true segments |
|---|---|---|---|---|---|---|---|
| b0_c | 111 | 39.4 | 18.8 | 47.0 | - | - | - |
| group_temporal | 111 | 47.5 | 10.1 | 36.9 | 4 | 0.26 | 3.72 |
| group_shuffled | 111 | 48.8 | 11.8 | 42.0 | 4 | 0.17 | 3.31 |
| group_appearance | 111 | 41.0 | 7.2 | 45.1 | 2 | 0.01 | 4.95 |
| group_random (100, size-matched) | 111 | mean 38.2 / p95 46.0 | mean 6.2 / p95 7.8 | - | - | - | - |
| (24 raw states: oracle many-to-one MoF bound) | 111 | 69.3 | - | - | - | - | - |

## Decision (declared rule, primary split)

**not computed (smoke run)**


## Smoothing sensitivity (MoF/F1@50 at emission multipliers 0.5 and 2; multiplier 1 is the table above)

| arm | seed | x0.5 | x2 |
|---|---|---|---|
| group_temporal | 111 | 47.2/13.0 | 49.4/10.4 |
| group_shuffled | 111 | 48.4/15.7 | 48.8/10.8 |
| group_appearance | 111 | 40.2/7.1 | 40.9/7.2 |

Scope: temporal grouping of transport states (symmetric affinity, reversal-invariant), not ordered composition and not grouping of raw tokens. The oracle is an upper bound on MoF for hard relabellings of the fixed 24 states only. Three seeds on one split are not confidence intervals.