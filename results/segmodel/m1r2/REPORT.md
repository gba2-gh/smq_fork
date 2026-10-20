# Milestone 1 rerun: corrected segment objective

## Ten-line summary

1. This report covers the corrected per-window segment objective only.
2. Saved result rows: 82; planned cells: 120; omitted/failed: 38.
3. D7 and input acceptance results are recorded in acceptance_inputs.json.
4. Standardization off gate: fail.
5. Standardization on gate: fail.
6. Gate decisions use only soft K=500, base temperature, transductive observed/control pairs.
7. Lambda was selected by released-code run duration; labels did not select it.
8. Seed SD is initialization sensitivity, not sampling uncertainty.
9. Degenerate cells remain reported and are excluded from gate evidence.
10. No result here authorizes Milestone 2 automatically.

## Gate status

```json
{
  "False": {
    "outcome": "fail",
    "f1_improvements": [
      -4.813333333333333,
      -3.450000000000003
    ],
    "mof_drops": [
      -2.2466666666666555,
      15.053333333333335
    ],
    "reasons": [
      "hugadb observed: degenerate",
      "lara permutation: degenerate"
    ]
  },
  "True": {
    "outcome": "fail",
    "f1_improvements": [
      -9.776666666666667,
      -1.2400000000000002
    ],
    "mof_drops": [
      3.626666666666665,
      10.996666666666666
    ],
    "reasons": [
      "hugadb permutation: degenerate",
      "lara permutation: degenerate"
    ]
  }
}
```

## Recovery and verification

Recovered 82/82 artifact-backed cells without fitting or decoding. Original failure rows and manifest are preserved in before_recovery/. Each recovered cell passed frame/boundary alignment, Hungarian-optimal mapping, objective-trace and label-free lambda-selection checks. The logging exception overwrote per-cell runtimes; those remain unavailable, not zero. The original experiment deadline and manifest remain unchanged. See recovery_audit.json. Subject-disjoint metrics use a separate evaluation-label Hungarian mapping for each fold; they are not a deployable label mapping. The lambda_start_action_runs field refers to fitting data and lambda_final_action_runs to evaluation data; these populations differ for subject-disjoint rows and cannot measure rate drift directly. Iteration-cap and empty-state flags are reported diagnostics; the brief does not automatically equate either with numerical failure.

## Same-checkpoint baselines

| Dataset | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| hugadb | 41.99 | 36.16 | 38.50 | 31.54 | 24.30 |
| lara | 37.38 | 39.40 | 34.69 | 28.40 | 16.39 |

## Primary metrics

| Dataset | Recording z-score | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| hugadb | False | 44.24 +/- 1.73 | 35.00 +/- 4.59 | 31.53 +/- 9.34 | 26.05 +/- 8.65 | 19.49 +/- 7.59 |
| lara | False | 22.33 +/- 0.77 | 24.00 +/- 0.76 | 22.39 +/- 1.23 | 18.41 +/- 1.08 | 12.94 +/- 0.64 |
| hugadb | True | 38.36 +/- 0.10 | 36.93 +/- 0.42 | 23.14 +/- 0.11 | 17.85 +/- 0.08 | 14.52 +/- 0.01 |
| lara | True | 26.38 +/- 0.91 | 28.99 +/- 0.65 | 27.65 +/- 1.58 | 22.28 +/- 1.60 | 15.15 +/- 1.48 |

## Guard diagnostics

| Dataset | Recording z-score | DP/GT | runs/GT | mean duration | median duration | rounds |
|---|---|---|---|---|---|---|
| hugadb | False | 2.793 | 2.719 | 2.36 | 0.50 | 9, 36, 11 |
| lara | False | 1.375 | 1.353 | 2.76 | 1.28 | 9, 11, 13 |
| hugadb | True | 1.789 | 1.737 | 2.86 | 1.00 | 6, 6, 6 |
| lara | True | 1.550 | 1.546 | 2.42 | 1.60 | 18, 11, 17 |

## Permutation separation

| Dataset | Recording z-score | paired F1@50 differences | mean | population SD | separated |
|---|---|---|---|---|---|
| hugadb | False | 17.24, 6.26, 25.52 | 16.34 | 7.89 | True |
| lara | False | 10.84, 10.68, 11.77 | 11.10 | 0.48 | True |
| hugadb | True | 13.49, 13.68, 12.82 | 13.33 | 0.37 | True |
| lara | True | 12.45, 9.85, 13.62 | 11.97 | 1.58 | True |

## Lambda curves

All 1066 evaluated lambda points, including all five label-based diagnostic metrics, are in [lambda_curves.csv](lambda_curves.csv). Only the label-free selected point enters the gate.

## Full grid

All 82 executed cell rows are in [cells.csv](cells.csv). Transductive and subject-disjoint protocols are separate columns.

| Cell | Status | MoF | Edit | F1@10 | F1@25 | F1@50 | runs/GT | Rounds | Cap reached |
|---|---|---|---|---|---|---|---|---|---|
| hugadb_transductive_std0_soft_k500_t1_s111_observed | degenerate | 43.11 | 35.78 | 35.32 | 28.33 | 20.29 | 2.034 | 9 | False |
| hugadb_transductive_std0_soft_k500_t1_s222_observed | degenerate | 46.68 | 29.03 | 18.68 | 14.50 | 9.81 | 4.738 | 36 | False |
| hugadb_transductive_std0_soft_k500_t1_s1538574472_observed | complete | 42.92 | 40.19 | 40.60 | 35.31 | 28.36 | 1.385 | 11 | False |
| hugadb_transductive_std1_soft_k500_t1_s111_observed | complete | 38.36 | 37.50 | 23.12 | 17.95 | 14.52 | 1.733 | 6 | False |
| hugadb_transductive_std1_soft_k500_t1_s222_observed | complete | 38.24 | 36.78 | 23.28 | 17.84 | 14.51 | 1.741 | 6 | False |
| hugadb_transductive_std1_soft_k500_t1_s1538574472_observed | complete | 38.49 | 36.51 | 23.01 | 17.75 | 14.54 | 1.736 | 6 | False |
| lara_transductive_std0_soft_k500_t1_s111_observed | complete | 21.82 | 23.59 | 21.70 | 17.63 | 12.45 | 1.407 | 9 | False |
| lara_transductive_std0_soft_k500_t1_s222_observed | complete | 21.74 | 23.34 | 21.36 | 17.67 | 12.53 | 1.377 | 11 | False |
| lara_transductive_std0_soft_k500_t1_s1538574472_observed | complete | 23.42 | 25.07 | 24.12 | 19.94 | 13.84 | 1.276 | 13 | False |
| lara_transductive_std1_soft_k500_t1_s111_observed | complete | 26.73 | 29.24 | 28.41 | 22.90 | 15.74 | 1.533 | 18 | False |
| lara_transductive_std1_soft_k500_t1_s222_observed | complete | 25.14 | 28.09 | 25.44 | 20.08 | 13.12 | 1.571 | 11 | False |
| lara_transductive_std1_soft_k500_t1_s1538574472_observed | complete | 27.28 | 29.63 | 29.09 | 23.85 | 16.59 | 1.532 | 17 | False |
| hugadb_transductive_std0_soft_k500_t1_s111_permutation | complete | 16.39 | 15.62 | 13.41 | 8.69 | 3.05 | 1.366 | 38 | False |
| hugadb_transductive_std0_soft_k500_t1_s222_permutation | complete | 16.54 | 16.92 | 14.47 | 9.54 | 3.55 | 1.474 | 20 | False |
| hugadb_transductive_std0_soft_k500_t1_s1538574472_permutation | complete | 15.33 | 15.34 | 13.47 | 8.02 | 2.84 | 1.564 | 43 | False |
| hugadb_transductive_std1_soft_k500_t1_s111_permutation | degenerate | 18.72 | 11.98 | 7.91 | 3.77 | 1.03 | 8.260 | 27 | False |
| hugadb_transductive_std1_soft_k500_t1_s222_permutation | degenerate | 18.61 | 10.48 | 7.30 | 3.25 | 0.83 | 9.077 | 27 | False |
| hugadb_transductive_std1_soft_k500_t1_s1538574472_permutation | degenerate | 19.77 | 14.34 | 9.78 | 5.46 | 1.72 | 6.475 | 52 | False |
| lara_transductive_std0_soft_k500_t1_s111_permutation | degenerate | 22.41 | 17.63 | 12.69 | 6.04 | 1.61 | 5.157 | 48 | False |
| lara_transductive_std0_soft_k500_t1_s222_permutation | degenerate | 21.04 | 19.57 | 13.29 | 6.66 | 1.85 | 4.509 | 60 | True |
| lara_transductive_std0_soft_k500_t1_s1538574472_permutation | degenerate | 21.27 | 16.99 | 12.09 | 6.44 | 2.07 | 5.223 | 37 | False |
| lara_transductive_std1_soft_k500_t1_s111_permutation | degenerate | 22.52 | 27.74 | 17.37 | 10.71 | 3.29 | 2.027 | 60 | True |
| lara_transductive_std1_soft_k500_t1_s222_permutation | degenerate | 26.42 | 27.77 | 18.08 | 10.80 | 3.27 | 2.201 | 45 | False |
| lara_transductive_std1_soft_k500_t1_s1538574472_permutation | degenerate | 20.19 | 26.85 | 17.56 | 9.94 | 2.97 | 2.106 | 49 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s111_observed_f0 | degenerate | 41.69 | 25.64 | 12.21 | 7.87 | 5.73 | 7.140 | 40 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s111_observed_f1 | degenerate | 49.61 | 26.96 | 27.86 | 21.63 | 14.71 | 3.483 | 19 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s111_observed_f2 | degenerate | 45.62 | 32.54 | 23.88 | 19.88 | 14.75 | 3.332 | 25 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s111_observed_f3 | degenerate | 50.46 | 30.17 | 25.83 | 20.60 | 15.43 | 2.182 | 52 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s222_observed_f0 | degenerate | 48.26 | 24.73 | 12.37 | 8.88 | 6.55 | 7.149 | 19 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s222_observed_f1 | degenerate | 48.21 | 27.53 | 20.68 | 16.10 | 10.28 | 4.396 | 32 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s222_observed_f2 | degenerate | 50.25 | 34.20 | 25.65 | 20.43 | 15.11 | 3.216 | 46 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s222_observed_f3 | degenerate | 50.46 | 32.82 | 23.69 | 19.18 | 14.27 | 2.393 | 35 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f0 | degenerate | 38.64 | 22.52 | 12.37 | 7.09 | 4.96 | 7.053 | 30 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f1 | degenerate | 47.86 | 30.41 | 19.71 | 15.72 | 11.19 | 4.251 | 23 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f2 | degenerate | 51.20 | 38.21 | 28.92 | 23.47 | 18.02 | 2.536 | 46 | False |
| hugadb_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f3 | complete | 51.80 | 31.58 | 33.53 | 25.26 | 16.21 | 1.648 | 11 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s111_observed_f0 | complete | 45.05 | 32.24 | 25.83 | 19.66 | 13.99 | 1.754 | 5 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s111_observed_f1 | degenerate | 42.15 | 38.47 | 23.68 | 18.30 | 14.04 | 2.126 | 5 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s111_observed_f2 | complete | 35.98 | 33.67 | 23.02 | 18.34 | 12.85 | 1.862 | 5 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s111_observed_f3 | complete | 43.15 | 38.72 | 23.14 | 18.32 | 15.48 | 1.654 | 7 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s222_observed_f0 | complete | 45.17 | 32.11 | 27.07 | 20.41 | 14.46 | 1.789 | 4 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s222_observed_f1 | complete | 43.30 | 39.92 | 28.06 | 23.52 | 17.29 | 1.980 | 13 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s222_observed_f2 | complete | 37.20 | 34.62 | 28.12 | 23.10 | 17.59 | 1.784 | 6 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s222_observed_f3 | complete | 52.34 | 45.20 | 31.51 | 26.31 | 23.92 | 1.768 | 7 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f0 | complete | 46.82 | 32.70 | 28.65 | 21.76 | 15.73 | 1.752 | 7 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f1 | complete | 42.88 | 39.24 | 24.67 | 18.94 | 13.88 | 1.974 | 4 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f2 | complete | 37.22 | 34.31 | 28.14 | 22.97 | 16.83 | 1.816 | 8 | False |
| hugadb_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f3 | complete | 53.01 | 45.94 | 31.37 | 26.68 | 24.13 | 1.773 | 10 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s111_observed_f0 | complete | 35.93 | 28.11 | 25.59 | 19.66 | 11.75 | 1.975 | 7 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s111_observed_f1 | complete | 25.03 | 20.08 | 19.30 | 15.07 | 9.81 | 1.730 | 7 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s111_observed_f2 | complete | 29.41 | 31.95 | 31.32 | 26.66 | 18.55 | 1.473 | 60 | True |
| lara_subject_disjoint_std0_soft_k500_t1_s111_observed_f3 | complete | 26.89 | 27.90 | 25.36 | 19.70 | 13.41 | 1.355 | 6 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s222_observed_f0 | degenerate | 35.23 | 27.79 | 25.25 | 19.36 | 11.47 | 2.017 | 9 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s222_observed_f1 | complete | 24.45 | 23.97 | 19.97 | 14.38 | 9.13 | 1.754 | 7 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s222_observed_f2 | complete | 27.48 | 27.53 | 24.85 | 20.98 | 14.95 | 1.729 | 5 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s222_observed_f3 | complete | 27.39 | 26.75 | 24.34 | 19.17 | 12.73 | 1.509 | 11 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f0 | complete | 32.16 | 28.18 | 23.14 | 17.30 | 9.87 | 1.982 | 6 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f1 | complete | 24.27 | 19.59 | 18.93 | 14.50 | 9.58 | 1.735 | 8 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f2 | complete | 27.79 | 29.74 | 28.09 | 23.27 | 16.27 | 1.645 | 22 | False |
| lara_subject_disjoint_std0_soft_k500_t1_s1538574472_observed_f3 | complete | 26.30 | 27.79 | 24.79 | 19.44 | 13.21 | 1.354 | 12 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s111_observed_f0 | complete | 30.68 | 31.14 | 28.30 | 22.05 | 14.12 | 1.715 | 14 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s111_observed_f1 | complete | 24.91 | 28.47 | 25.79 | 19.23 | 11.24 | 1.581 | 16 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s111_observed_f2 | complete | 28.94 | 31.94 | 33.00 | 28.61 | 21.09 | 1.307 | 22 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s111_observed_f3 | complete | 27.35 | 28.58 | 27.08 | 21.38 | 14.13 | 1.595 | 15 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s222_observed_f0 | complete | 30.26 | 31.13 | 27.95 | 21.89 | 14.18 | 1.705 | 8 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s222_observed_f1 | complete | 24.77 | 28.60 | 25.86 | 19.31 | 11.40 | 1.592 | 14 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s222_observed_f2 | complete | 29.15 | 31.90 | 33.46 | 29.34 | 21.68 | 1.295 | 14 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s222_observed_f3 | complete | 27.62 | 28.49 | 27.12 | 21.25 | 14.20 | 1.606 | 22 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f0 | complete | 29.95 | 30.23 | 26.90 | 21.06 | 13.88 | 1.706 | 9 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f1 | complete | 24.02 | 27.04 | 23.70 | 18.34 | 10.98 | 1.601 | 18 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f2 | complete | 29.21 | 32.40 | 33.79 | 29.59 | 22.19 | 1.309 | 9 | False |
| lara_subject_disjoint_std1_soft_k500_t1_s1538574472_observed_f3 | complete | 28.14 | 28.68 | 27.19 | 21.57 | 14.47 | 1.612 | 23 | False |
| hugadb_transductive_std0_hard_k500_t1_s111_observed | complete | 35.89 | 19.93 | 17.11 | 13.75 | 10.88 | 0.713 | 2 | False |
| hugadb_transductive_std0_hard_k500_t1_s222_observed | complete | 31.49 | 19.84 | 11.79 | 8.14 | 5.09 | 0.563 | 2 | False |
| hugadb_transductive_std0_hard_k500_t1_s1538574472_observed | complete | 33.96 | 20.67 | 14.72 | 11.33 | 8.40 | 0.681 | 2 | False |
| hugadb_transductive_std0_soft_k500_t2_s111_observed | degenerate | 41.73 | 22.49 | 14.27 | 10.94 | 8.46 | 5.618 | 57 | False |
| hugadb_transductive_std0_soft_k500_t2_s222_observed | degenerate | 42.79 | 24.15 | 14.04 | 10.58 | 7.47 | 6.245 | 48 | False |
| hugadb_transductive_std0_soft_k500_t2_s1538574472_observed | degenerate | 42.51 | 24.26 | 12.49 | 9.33 | 6.86 | 7.116 | 60 | True |
| hugadb_transductive_std0_soft_k200_t1_s111_observed | degenerate | 48.85 | 33.11 | 32.76 | 28.06 | 22.49 | 2.023 | 8 | False |
| hugadb_transductive_std0_soft_k200_t1_s222_observed | degenerate | 40.66 | 30.47 | 16.43 | 14.31 | 11.83 | 4.250 | 41 | False |
| hugadb_transductive_std0_soft_k200_t1_s1538574472_observed | degenerate | 53.59 | 36.26 | 36.19 | 31.80 | 26.19 | 2.070 | 6 | False |
| hugadb_transductive_std0_soft_k1000_t1_s111_observed | degenerate | 41.09 | 26.96 | 16.39 | 11.83 | 7.33 | 4.016 | 31 | False |

## Open questions

- Does any corrected-objective advantage survive subject-disjoint fitting?
- Are failures associated with duration-cap concentration or with weak separation from permutation controls?

## What these numbers cannot establish

These frozen-encoder results do not establish causal sources of dataset differences or a general limit on segment models. HuGaDB and LARa differ in class count, subject count and modality. Labels enter D7 provenance, Hungarian mapping, curve diagnostics, scoring and post-fit guards; they do not select lambda or another model setting.

## Not run

- `hugadb_transductive_std0_soft_k1000_t1_s222_observed`: five-hour execution deadline reached
- `hugadb_transductive_std0_soft_k1000_t1_s1538574472_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_hard_k500_t1_s111_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_hard_k500_t1_s222_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_hard_k500_t1_s1538574472_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k500_t2_s111_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k500_t2_s222_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k500_t2_s1538574472_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k200_t1_s111_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k200_t1_s222_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k200_t1_s1538574472_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k1000_t1_s111_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k1000_t1_s222_observed`: five-hour execution deadline reached
- `hugadb_transductive_std1_soft_k1000_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std0_hard_k500_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std0_hard_k500_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std0_hard_k500_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k500_t2_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k500_t2_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k500_t2_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k200_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k200_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k200_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k1000_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k1000_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std0_soft_k1000_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std1_hard_k500_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std1_hard_k500_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std1_hard_k500_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k500_t2_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k500_t2_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k500_t2_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k200_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k200_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k200_t1_s1538574472_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k1000_t1_s111_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k1000_t1_s222_observed`: five-hour execution deadline reached
- `lara_transductive_std1_soft_k1000_t1_s1538574472_observed`: five-hour execution deadline reached
