# Milestone 1 run m1_lara_001

Scale `full`

## Evaluation population, final-solver endpoint (one Hungarian mapping fitted globally over the evaluation recordings)

| arm | seed | status | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| b0_centre | 111 | complete | 39.38 | 27.93 | 28.96 | 24.83 | 18.83 |
| b0_centre | 222 | complete | 38.43 | 31.18 | 30.54 | 26.30 | 20.09 |
| b0_centre | 1538574472 | complete | 38.79 | 29.49 | 29.34 | 25.94 | 20.12 |
| m1_bag | 111 | complete | 38.50 | 22.63 | 23.67 | 19.72 | 14.51 |
| m1_bag | 222 | complete | 38.49 | 22.40 | 22.70 | 18.74 | 13.56 |
| m1_bag | 1538574472 | complete | 38.67 | 22.79 | 23.59 | 19.41 | 14.15 |
| m1_ordered | 111 | complete | 38.69 | 21.58 | 21.95 | 18.39 | 13.59 |
| m1_ordered | 222 | complete | 38.38 | 22.16 | 22.26 | 18.48 | 13.70 |
| m1_ordered | 1538574472 | complete | 38.02 | 21.27 | 21.99 | 17.88 | 12.89 |
| occupancy | 111 | complete | 39.24 | 31.65 | 31.19 | 27.79 | 21.09 |
| occupancy | 222 | complete | 39.08 | 31.95 | 31.34 | 27.27 | 20.67 |
| occupancy | 1538574472 | complete | 37.43 | 33.37 | 32.12 | 28.07 | 21.57 |

## Frozen feasibility gate (declared before the run)

**FAILED**

- Ordered minus matched B0-centre (need mean F1@50 >= +5, mean MoF >= -2, all seeds positive): 111: F1 -5.24 / MoF -0.69, 222: F1 -6.39 / MoF -0.05, 1538574472: F1 -7.23 / MoF -0.77; mean F1 -6.29, mean MoF -0.50, SD 0.82/0.32 -> fail
- Ordered minus bag (need mean F1@50 >= +3, mean MoF >= -2, all seeds positive): 111: F1 -0.92 / MoF +0.19, 222: F1 +0.14 / MoF -0.11, 1538574472: F1 -1.26 / MoF -0.65; mean F1 -0.68, mean MoF -0.19, SD 0.60/0.35 -> fail
- Ordered minus occupancy (need mean F1@50 >= +3, mean MoF >= -2, all seeds positive): 111: F1 -7.50 / MoF -0.55, 222: F1 -6.97 / MoF -0.70, 1538574472: F1 -8.68 / MoF +0.59; mean F1 -7.72, mean MoF -0.22, SD 0.71/0.58 -> fail

## Secondary: final-decode cost multipliers (primary split, frozen models; multiplier 1 is the primary result)

| arm | seed | multiplier | MoF | F1@50 |
|---|---|---|---|---|
| b0_centre | 111 | 0.5 | 40.05 | 26.15 |
| b0_centre | 111 | 1.0 | 39.38 | 18.83 |
| b0_centre | 111 | 2.0 | 38.63 | 12.39 |
| b0_centre | 222 | 0.5 | 38.28 | 25.00 |
| b0_centre | 222 | 1.0 | 38.43 | 20.09 |
| b0_centre | 222 | 2.0 | 38.02 | 13.28 |
| b0_centre | 1538574472 | 0.5 | 38.72 | 24.63 |
| b0_centre | 1538574472 | 1.0 | 38.79 | 20.12 |
| b0_centre | 1538574472 | 2.0 | 38.09 | 12.40 |
| m1_bag | 111 | 0.5 | 38.40 | 19.57 |
| m1_bag | 111 | 1.0 | 38.50 | 14.51 |
| m1_bag | 111 | 2.0 | 38.20 | 11.85 |
| m1_bag | 222 | 0.5 | 39.25 | 17.90 |
| m1_bag | 222 | 1.0 | 38.49 | 13.56 |
| m1_bag | 222 | 2.0 | 38.17 | 12.20 |
| m1_bag | 1538574472 | 0.5 | 38.95 | 18.75 |
| m1_bag | 1538574472 | 1.0 | 38.67 | 14.15 |
| m1_bag | 1538574472 | 2.0 | 38.72 | 11.08 |
| m1_ordered | 111 | 0.5 | 38.69 | 17.28 |
| m1_ordered | 111 | 1.0 | 38.69 | 13.59 |
| m1_ordered | 111 | 2.0 | 38.15 | 10.86 |
| m1_ordered | 222 | 0.5 | 39.23 | 17.37 |
| m1_ordered | 222 | 1.0 | 38.38 | 13.70 |
| m1_ordered | 222 | 2.0 | 38.07 | 11.09 |
| m1_ordered | 1538574472 | 0.5 | 38.18 | 16.40 |
| m1_ordered | 1538574472 | 1.0 | 38.02 | 12.89 |
| m1_ordered | 1538574472 | 2.0 | 38.40 | 10.04 |
| occupancy | 111 | 0.5 | 39.15 | 26.34 |
| occupancy | 111 | 1.0 | 39.24 | 21.09 |
| occupancy | 111 | 2.0 | 38.78 | 15.79 |
| occupancy | 222 | 0.5 | 39.66 | 25.96 |
| occupancy | 222 | 1.0 | 39.08 | 20.67 |
| occupancy | 222 | 2.0 | 38.31 | 15.88 |
| occupancy | 1538574472 | 0.5 | 37.26 | 23.62 |
| occupancy | 1538574472 | 1.0 | 37.43 | 21.57 |
| occupancy | 1538574472 | 2.0 | 37.17 | 15.42 |

## Secondary: subject-disjoint folds (mean over 4 folds of 4 held-out subjects; one Hungarian mapping per fold)

The tokenizer was still fitted on all recordings, so this is not fully inductive.

| arm | seed | MoF | Edit | F1@50 |
|---|---|---|---|---|
| b0_centre | 111 | 44.20 | 29.96 | 18.32 |
| b0_centre | 222 | 43.02 | 29.11 | 17.67 |
| b0_centre | 1538574472 | 42.04 | 29.70 | 18.55 |
| m1_bag | 111 | 42.29 | 23.41 | 12.34 |
| m1_bag | 222 | 40.34 | 23.54 | 12.62 |
| m1_bag | 1538574472 | 40.06 | 23.59 | 12.46 |
| m1_ordered | 111 | 42.88 | 21.27 | 11.08 |
| m1_ordered | 222 | 40.66 | 20.93 | 11.49 |
| m1_ordered | 1538574472 | 40.49 | 21.98 | 11.35 |
| occupancy | 111 | 43.05 | 34.21 | 21.54 |
| occupancy | 222 | 41.73 | 33.06 | 21.14 |
| occupancy | 1538574472 | 39.77 | 32.51 | 18.93 |

Descriptive comparisons (same thresholds as the gate, not a gate):

- Ordered minus matched B0-centre: mean F1@50 -6.87, mean MoF -1.74, per-seed F1 -7.24, -6.18, -7.19 -> does not meet the thresholds
- Ordered minus bag: mean F1@50 -1.16, mean MoF +0.45, per-seed F1 -1.25, -1.13, -1.10 -> does not meet the thresholds
- Ordered minus occupancy: mean F1@50 -9.23, mean MoF -0.17, per-seed F1 -10.45, -9.65, -7.58 -> does not meet the thresholds

Three initialization seeds on one fixed split are not confidence intervals. Details: cells.csv, training.csv, transport.csv, per-cell diagnostics.json.