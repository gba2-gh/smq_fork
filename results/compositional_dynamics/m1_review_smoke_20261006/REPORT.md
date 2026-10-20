# Milestone 1 run m1_review_smoke_20261006

Scale `smoke` (**SMOKE: plumbing check, not evidence**)

## Evaluation population, final-solver endpoint (one Hungarian mapping fitted globally over the evaluation recordings)

| arm | seed | status | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| b0_centre | 111 | complete | 45.97 | 21.80 | 21.46 | 18.03 | 13.73 |
| m1_bag | 111 | complete | 47.88 | 19.94 | 20.95 | 17.90 | 12.57 |
| m1_ordered | 111 | complete | 48.01 | 19.91 | 20.87 | 17.84 | 12.52 |
| occupancy | 111 | complete | 48.07 | 20.62 | 22.22 | 17.78 | 13.33 |

## Frozen feasibility gate (declared before the run)

**not computed (smoke run)**


## Secondary: final-decode cost multipliers (primary split, frozen models; multiplier 1 is the primary result)

| arm | seed | multiplier | MoF | F1@50 |
|---|---|---|---|---|
| b0_centre | 111 | 0.5 | 46.53 | 21.02 |
| b0_centre | 111 | 1.0 | 45.97 | 13.73 |
| b0_centre | 111 | 2.0 | 45.22 | 11.30 |
| m1_bag | 111 | 0.5 | 46.42 | 16.96 |
| m1_bag | 111 | 1.0 | 47.88 | 12.57 |
| m1_bag | 111 | 2.0 | 46.60 | 10.70 |
| m1_ordered | 111 | 0.5 | 46.95 | 17.58 |
| m1_ordered | 111 | 1.0 | 48.01 | 12.52 |
| m1_ordered | 111 | 2.0 | 46.90 | 10.49 |
| occupancy | 111 | 0.5 | 46.75 | 16.20 |
| occupancy | 111 | 1.0 | 48.07 | 13.33 |
| occupancy | 111 | 2.0 | 46.57 | 10.92 |

Three initialization seeds on one fixed split are not confidence intervals. Details: cells.csv, training.csv, transport.csv, per-cell diagnostics.json.