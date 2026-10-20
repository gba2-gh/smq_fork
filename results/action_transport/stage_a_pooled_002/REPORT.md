# Stage A report -- stage_a_pooled_002

Generated from saved artifacts only (no refitting). Runner protocol version 1.3.1, protocol `pooled`, 1 launch(es) on SC448095.

**Coverage:** 130/130 planned prediction sets scored; 0 fit(s) not run.

## Summary of what this run can and cannot show

- F1@50 is the declared focal metric; all five metrics are tabulated below.
- Setting T is primary. E re-decodes the same T-fitted parameters with different coefficients; it is a bundled sensitivity, not an E-fitted model.
- A0+filter is one fixed smoother. Beating it is not superiority over all temporal models.
- A-cont and A1 use different cost geometries under equal coefficients.
- SMQ is a system comparison (different tokenizer and resolution), not an isolated K intervention.
- Seed SD describes initialization sensitivity, not sampling uncertainty.

## SMQ baseline (same checkpoint, same evaluator)

| Dataset | MoF | Edit | F1@10 | F1@25 | F1@50 | pred/GT runs |
|---|---|---|---|---|---|---|
| hugadb | 41.99 | 36.16 | 38.50 | 31.54 | 24.30 | 1.64 |
| lara | 37.38 | 39.40 | 34.69 | 28.40 | 16.39 | 1.33 |

Provenance check: the segmodel audit reported 41.99 / 24.30 (HuGaDB MoF / F1@50) and 37.38 / 16.39 (LARa) for the same codes and mapping.

## Comparisons (Pooled)

### Pooled: F1@50, setting T (primary)

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 24.30 | 2.66 ± 0.27 (n=3) | 6.75 ± 0.69 (n=3) | 3.73 ± 1.74 (n=3) | 4.80 ± 0.00 (n=3) | 4.85 ± 2.68 (n=3) |
| hugadb | on | 24.30 | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 6.74 ± 1.01 (n=3) | 3.17 ± 0.23 (n=3) | 4.93 ± 0.39 (n=3) |
| lara | off | 16.39 | 6.08 ± 0.50 (n=3) | 13.05 ± 1.18 (n=3) | 9.86 ± 0.96 (n=3) | 11.09 ± 0.81 (n=3) | 17.21 ± 1.58 (n=3) |
| lara | on | 16.39 | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 7.78 ± 0.67 (n=3) | 8.36 ± 1.15 (n=3) | 17.05 ± 0.73 (n=3) |

### Pooled: F1@50, setting E (secondary sensitivity)

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 24.30 | 2.64 ± 0.29 (n=3) | 6.73 ± 0.71 (n=3) | 5.73 ± 1.84 (n=3) | 7.06 ± 0.03 (n=3) | 22.30 ± 7.58 (n=3) |
| hugadb | on | 24.30 | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 9.66 ± 1.16 (n=3) | 3.56 ± 0.12 (n=3) | 24.32 ± 2.44 (n=3) |
| lara | off | 16.39 | 6.09 ± 0.51 (n=3) | 13.06 ± 1.19 (n=3) | 11.90 ± 1.18 (n=3) | 12.58 ± 1.00 (n=3) | 16.63 ± 1.86 (n=3) |
| lara | on | 16.39 | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 10.69 ± 0.99 (n=3) | 10.95 ± 0.83 (n=3) | 17.21 ± 0.83 (n=3) |

#### Paired seed differences (F1@50, setting T)

1. **Claim** (fine vs class-matched vocabulary, same decoder):

| Dataset | Norm | A1 - A1-KC per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -2.50, -2.10, +1.38 | -1.07 | 1/3 |
| hugadb | on | +5.24, +2.51, +2.97 | +3.57 | 3/3 |
| lara | off | +1.20, -1.98, -2.93 | -1.24 | 1/3 |
| lara | on | +0.32, +1.00, -3.06 | -0.58 | 2/3 |

2. **Temporal assignment** (transport vs the fixed smoother):

| Dataset | Norm | A1 - A0+filter per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -4.03, -3.51, -1.54 | -3.03 | 0/3 |
| hugadb | on | +2.56, +0.25, +0.40 | +1.07 | 3/3 |
| lara | off | -3.27, -2.19, -4.12 | -3.19 | 0/3 |
| lara | on | -3.95, -7.88, -6.35 | -6.06 | 0/3 |

3. **Discretization** (hard codes vs continuous features, same decoder):

| Dataset | Norm | A1 - A-cont per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -0.67, -0.24, -2.46 | -1.12 | 0/3 |
| hugadb | on | +2.78, +0.66, +1.98 | +1.81 | 3/3 |
| lara | off | -8.21, -7.08, -6.76 | -7.35 | 0/3 |
| lara | on | -9.79, -7.44, -10.57 | -9.27 | 0/3 |

### All five metrics, setting T (Pooled)

**MoF**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 41.99 | 28.12 ± 1.23 (n=3) | 32.86 ± 0.94 (n=3) | 37.71 ± 4.08 (n=3) | 41.52 ± 0.05 (n=3) | 34.97 ± 3.46 (n=3) |
| hugadb | on | 41.99 | 27.60 ± 1.08 (n=3) | 32.46 ± 1.78 (n=3) | 40.72 ± 0.95 (n=3) | 32.07 ± 0.57 (n=3) | 35.07 ± 1.23 (n=3) |
| lara | off | 37.38 | 21.96 ± 0.58 (n=3) | 22.68 ± 0.81 (n=3) | 27.73 ± 0.75 (n=3) | 29.49 ± 0.44 (n=3) | 28.30 ± 0.79 (n=3) |
| lara | on | 37.38 | 26.97 ± 2.03 (n=3) | 28.52 ± 2.12 (n=3) | 29.62 ± 0.33 (n=3) | 28.38 ± 1.23 (n=3) | 26.16 ± 0.54 (n=3) |

**Edit**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 36.16 | 13.03 ± 1.48 (n=3) | 18.34 ± 0.76 (n=3) | 13.94 ± 6.98 (n=3) | 16.97 ± 0.04 (n=3) | 15.08 ± 6.34 (n=3) |
| hugadb | on | 36.16 | 10.83 ± 0.34 (n=3) | 16.90 ± 0.81 (n=3) | 16.89 ± 2.12 (n=3) | 14.08 ± 1.31 (n=3) | 17.64 ± 0.37 (n=3) |
| lara | off | 39.40 | 18.13 ± 0.26 (n=3) | 28.47 ± 0.03 (n=3) | 26.49 ± 0.95 (n=3) | 29.92 ± 0.32 (n=3) | 30.36 ± 0.42 (n=3) |
| lara | on | 39.40 | 15.71 ± 0.70 (n=3) | 31.76 ± 1.89 (n=3) | 23.19 ± 0.26 (n=3) | 24.89 ± 1.90 (n=3) | 34.93 ± 0.63 (n=3) |

**F1@10**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 38.50 | 5.91 ± 0.67 (n=3) | 12.96 ± 0.34 (n=3) | 7.19 ± 2.04 (n=3) | 7.32 ± 0.01 (n=3) | 9.15 ± 2.77 (n=3) |
| hugadb | on | 38.50 | 5.36 ± 0.41 (n=3) | 12.56 ± 0.20 (n=3) | 12.24 ± 1.27 (n=3) | 6.34 ± 0.36 (n=3) | 8.37 ± 0.90 (n=3) |
| lara | off | 34.69 | 15.51 ± 0.72 (n=3) | 24.42 ± 1.55 (n=3) | 22.37 ± 0.80 (n=3) | 27.78 ± 0.12 (n=3) | 29.87 ± 1.35 (n=3) |
| lara | on | 34.69 | 15.42 ± 1.26 (n=3) | 28.44 ± 2.60 (n=3) | 21.08 ± 0.57 (n=3) | 22.98 ± 1.61 (n=3) | 31.43 ± 0.64 (n=3) |

**F1@25**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 31.54 | 3.23 ± 0.21 (n=3) | 9.28 ± 0.34 (n=3) | 4.73 ± 1.96 (n=3) | 5.03 ± 0.00 (n=3) | 5.57 ± 2.75 (n=3) |
| hugadb | on | 31.54 | 3.02 ± 0.21 (n=3) | 7.99 ± 0.24 (n=3) | 8.45 ± 1.05 (n=3) | 3.72 ± 0.35 (n=3) | 5.43 ± 0.41 (n=3) |
| lara | off | 28.40 | 10.67 ± 0.65 (n=3) | 19.55 ± 1.54 (n=3) | 16.84 ± 1.16 (n=3) | 19.74 ± 0.53 (n=3) | 24.88 ± 1.55 (n=3) |
| lara | on | 28.40 | 9.44 ± 1.30 (n=3) | 22.01 ± 2.47 (n=3) | 14.15 ± 0.66 (n=3) | 15.57 ± 1.54 (n=3) | 25.22 ± 0.61 (n=3) |

**F1@50**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 24.30 | 2.66 ± 0.27 (n=3) | 6.75 ± 0.69 (n=3) | 3.73 ± 1.74 (n=3) | 4.80 ± 0.00 (n=3) | 4.85 ± 2.68 (n=3) |
| hugadb | on | 24.30 | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 6.74 ± 1.01 (n=3) | 3.17 ± 0.23 (n=3) | 4.93 ± 0.39 (n=3) |
| lara | off | 16.39 | 6.08 ± 0.50 (n=3) | 13.05 ± 1.18 (n=3) | 9.86 ± 0.96 (n=3) | 11.09 ± 0.81 (n=3) | 17.21 ± 1.58 (n=3) |
| lara | on | 16.39 | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 7.78 ± 0.67 (n=3) | 8.36 ± 1.15 (n=3) | 17.05 ± 0.73 (n=3) |

### Segmentation diagnostics, setting T (Pooled)

Predicted runs merge adjacent equal raw states. A pred/GT ratio far above 1 means over-segmentation; states used / largest state expose collapse; MI(state;subject) is descriptive only (subject IDs are metadata).

**pred/GT runs**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 1.639 | 10.638 ± 0.099 (n=3) | 4.660 ± 0.085 (n=3) | 7.741 ± 0.109 (n=3) | 9.173 ± 0.008 (n=3) | 5.760 ± 0.470 (n=3) |
| hugadb | on | 1.639 | 12.564 ± 0.187 (n=3) | 6.223 ± 0.218 (n=3) | 8.338 ± 0.295 (n=3) | 11.347 ± 0.156 (n=3) | 8.465 ± 0.405 (n=3) |
| lara | off | 1.327 | 3.705 ± 0.108 (n=3) | 1.355 ± 0.044 (n=3) | 2.042 ± 0.099 (n=3) | 1.544 ± 0.075 (n=3) | 0.701 ± 0.007 (n=3) |
| lara | on | 1.327 | 5.536 ± 0.158 (n=3) | 1.847 ± 0.055 (n=3) | 3.466 ± 0.011 (n=3) | 2.895 ± 0.321 (n=3) | 1.019 ± 0.016 (n=3) |

**median run (s)**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 2.000 | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) |
| hugadb | on | 2.000 | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) | 0.250 ± 0.000 (n=3) |
| lara | off | 2.000 | 0.480 ± 0.000 (n=3) | 1.680 ± 0.000 (n=3) | 0.720 ± 0.000 (n=3) | 1.040 ± 0.113 (n=3) | 2.960 ± 0.113 (n=3) |
| lara | on | 2.000 | 0.240 ± 0.000 (n=3) | 1.440 ± 0.000 (n=3) | 0.480 ± 0.000 (n=3) | 0.720 ± 0.000 (n=3) | 2.640 ± 0.000 (n=3) |

**GT median run (s)**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 3.467 | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) |
| hugadb | on | 3.467 | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) | 3.467 ± 0.000 (n=3) |
| lara | off | 2.200 | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) |
| lara | on | 2.200 | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) | 2.200 ± 0.000 (n=3) |

**states used**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 10.000 | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) |
| hugadb | on | 10.000 | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) |
| lara | off | 8.000 | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) |
| lara | on | 8.000 | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) |

**largest state**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 0.339 | 0.136 ± 0.005 (n=3) | 0.140 ± 0.005 (n=3) | 0.410 ± 0.049 (n=3) | 0.366 ± 0.000 (n=3) | 0.437 ± 0.060 (n=3) |
| hugadb | on | 0.339 | 0.124 ± 0.012 (n=3) | 0.152 ± 0.015 (n=3) | 0.240 ± 0.005 (n=3) | 0.236 ± 0.009 (n=3) | 0.193 ± 0.031 (n=3) |
| lara | off | 0.316 | 0.147 ± 0.005 (n=3) | 0.156 ± 0.008 (n=3) | 0.263 ± 0.014 (n=3) | 0.275 ± 0.043 (n=3) | 0.223 ± 0.016 (n=3) |
| lara | on | 0.316 | 0.194 ± 0.012 (n=3) | 0.209 ± 0.016 (n=3) | 0.258 ± 0.013 (n=3) | 0.256 ± 0.039 (n=3) | 0.249 ± 0.006 (n=3) |

**MI(state;subject)**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | 0.183 | 0.316 ± 0.023 (n=3) | 0.490 ± 0.035 (n=3) | 0.218 ± 0.022 (n=3) | 0.176 ± 0.001 (n=3) | 0.150 ± 0.008 (n=3) |
| hugadb | on | 0.183 | 0.052 ± 0.006 (n=3) | 0.088 ± 0.003 (n=3) | 0.079 ± 0.009 (n=3) | 0.054 ± 0.001 (n=3) | 0.040 ± 0.014 (n=3) |
| lara | off | 0.121 | 0.171 ± 0.015 (n=3) | 0.199 ± 0.018 (n=3) | 0.459 ± 0.022 (n=3) | 0.534 ± 0.016 (n=3) | 0.478 ± 0.011 (n=3) |
| lara | on | 0.121 | 0.032 ± 0.003 (n=3) | 0.044 ± 0.007 (n=3) | 0.025 ± 0.002 (n=3) | 0.020 ± 0.001 (n=3) | 0.016 ± 0.002 (n=3) |

## Permutation adjacency diagnostic (not a primary comparison)

A1 refit on within-recording permuted codes (seed 111), states restored to the original timeline before scoring. Compare with A1 seed 111 in the same row: the gap is what true temporal adjacency contributes, including emission-refitting effects.

| Dataset | Norm | Setting | A1 seed 111 F1@50 | A1-permuted F1@50 | A1 run ratio | permuted run ratio |
|---|---|---|---|---|---|---|
| hugadb | off | E | 3.22 | 1.01 | 5.48 | 7.08 |
| hugadb | off | T | 2.30 | 1.76 | 7.83 | 8.73 |
| hugadb | on | E | 10.77 | 2.61 | 6.10 | 7.15 |
| hugadb | on | T | 8.11 | 3.91 | 7.94 | 8.75 |
| lara | off | E | 13.57 | 2.55 | 1.39 | 2.41 |
| lara | off | T | 11.19 | 7.94 | 2.02 | 2.14 |
| lara | on | E | 10.87 | 2.08 | 2.35 | 6.65 |
| lara | on | T | 7.84 | 4.03 | 3.46 | 5.66 |

## Convergence

Fit outer status counts prediction-set rows (each fit appears once per setting). Final solves count recordings. `capped` = declared cap reached with a finite, nonincreasing objective; capped results are included in every table above.

| Arm | Setting | fit outer: converged / capped / invalid | final solves (recordings): converged / capped / invalid | max residual | median outer iters |
|---|---|---|---|---|---|
| A1 | E | 0 / 12 / 0 | 2622 / 2196 / 0 | 3.3e-01 | 50 |
| A1 | T | 0 / 12 / 0 | 4604 / 214 / 0 | 1.1e-01 | 50 |
| A1-KC | E | 8 / 4 / 0 | 2400 / 2418 / 0 | 3.3e-01 | 38 |
| A1-KC | T | 8 / 4 / 0 | 4599 / 219 / 0 | 6.0e-02 | 38 |
| A-cont | E | 0 / 12 / 0 | 928 / 3890 / 0 | 5.3e-01 | 50 |
| A-cont | T | 0 / 12 / 0 | 3046 / 1772 / 0 | 1.8e-01 | 50 |
| A0 | E | 1 / 11 / 0 | 4818 / 0 / 0 | 8.6e-06 | 50 |
| A0 | T | 1 / 11 / 0 | 4818 / 0 / 0 | 7.7e-06 | 50 |
| A1-permuted | E | 0 / 4 / 0 | 404 / 1202 / 0 | 3.7e-01 | 50 |
| A1-permuted | T | 0 / 4 / 0 | 1519 / 87 / 0 | 3.1e-02 | 50 |

## Reproducibility against `stage_a_pooled_001`

126 shared prediction sets. Largest absolute metric change: MoF 0.01, Edit 0.45, F1@10 0.32, F1@25 0.26, F1@50 0.05.

26 metric values changed by more than 0.01 points (expected only where near-tied states flip under the exact-log fix):

| Prediction set | Metric | before | after |
|---|---|---|---|
| hugadb_raw_pooled_pooled_seed111_continuous_asot__E | Edit | 29.13 | 29.14 |
| hugadb_raw_pooled_pooled_seed111_continuous_asot__E | F1@10 | 30.53 | 30.51 |
| hugadb_raw_pooled_pooled_seed111_continuous_asot__E | F1@25 | 25.49 | 25.43 |
| hugadb_raw_pooled_pooled_seed111_continuous_asot__E | F1@50 | 18.43 | 18.40 |
| lara_raw_pooled_pooled_seed111_categorical_asot__T | Edit | 26.13 | 26.14 |
| lara_raw_pooled_pooled_seed111_categorical_asot__T | F1@10 | 22.90 | 22.92 |
| lara_raw_pooled_pooled_seed111_categorical_asot__T | F1@25 | 18.27 | 18.29 |
| lara_raw_pooled_pooled_seed111_continuous_asot__E | F1@25 | 26.92 | 26.91 |
| lara_raw_pooled_pooled_seed222_categorical_asot__T | F1@10 | 21.25 | 21.24 |
| lara_raw_pooled_pooled_seed222_continuous_asot__E | F1@10 | 27.41 | 27.40 |
| hugadb_norm_pooled_pooled_seed111_continuous_asot__E | Edit | 40.74 | 40.70 |
| hugadb_norm_pooled_pooled_seed111_continuous_asot__E | F1@10 | 40.16 | 40.11 |
| hugadb_norm_pooled_pooled_seed111_continuous_asot__E | F1@25 | 35.61 | 35.56 |
| hugadb_norm_pooled_pooled_seed222_continuous_asot__E | Edit | 38.46 | 38.44 |
| hugadb_norm_pooled_pooled_seed222_continuous_asot__E | F1@10 | 36.49 | 36.54 |
| hugadb_norm_pooled_pooled_seed222_continuous_asot__E | F1@25 | 30.17 | 30.23 |
| hugadb_norm_pooled_pooled_seed222_continuous_asot__E | F1@50 | 23.79 | 23.82 |
| hugadb_norm_pooled_pooled_seed1538574472_categorical_asot__T | Edit | 14.80 | 14.35 |
| hugadb_norm_pooled_pooled_seed1538574472_categorical_asot__T | F1@10 | 10.94 | 10.62 |
| hugadb_norm_pooled_pooled_seed1538574472_categorical_asot__T | F1@25 | 7.35 | 7.09 |
| hugadb_norm_pooled_pooled_seed1538574472_categorical_asot__T | F1@50 | 5.77 | 5.72 |
| hugadb_norm_pooled_pooled_seed1538574472_continuous_asot__E | Edit | 41.58 | 41.64 |
| hugadb_norm_pooled_pooled_seed1538574472_continuous_asot__E | F1@10 | 36.50 | 36.53 |
| hugadb_norm_pooled_pooled_seed1538574472_continuous_asot__E | F1@25 | 28.88 | 28.91 |
| hugadb_norm_pooled_pooled_seed1538574472_continuous_asot__E | F1@50 | 21.60 | 21.62 |
| lara_norm_pooled_pooled_seed1538574472_categorical_asot__E | MoF | 29.93 | 29.94 |

## Limitations

- Protocol `pooled` only; subject-disjoint transfer not run.
- The frozen encoder is transductive; per-recording normalization and inference are offline.
- Local temporal consistency is not a learned action grammar.
- Every planned arm and setting is shown above or marked not run; nothing was selected by score.
