# Stage A report -- stage_a_pooled_001

Generated from saved artifacts only (no refitting). Runner protocol version 1.3, protocol `pooled`, 1 launch(es).

**Coverage:** 126/128 planned prediction sets scored; 1 fit(s) not run.

| Not-run cell | Reason |
|---|---|
| lara_norm_pooled_pooled_seed111_permuted_asot | projected 604s exceeds 202s remaining |

## Summary of what this run can and cannot show

- F1@50 is the declared focal metric; all five metrics are tabulated below.
- Setting T is primary. E re-decodes the same T-fitted parameters with different coefficients; it is a bundled sensitivity, not an E-fitted model.
- A0+filter is one fixed smoother. Beating it is not superiority over all temporal models.
- A-cont and A1 use different cost geometries under equal coefficients.
- SMQ is a system comparison (different tokenizer and resolution), not an isolated K intervention.
- Seed SD describes initialization sensitivity, not sampling uncertainty.

## SMQ baseline

**Not scored in this run** (v1.3 runner). Q1/Q2 same-checkpoint values, cited for orientation only: HuGaDB 42.0 / 24.3, LARa 37.4 / 16.4 (MoF / F1@50).

## Comparisons (Pooled)

### Pooled: F1@50, setting T (primary)

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 2.66 ± 0.27 (n=3) | 6.75 ± 0.69 (n=3) | 3.73 ± 1.74 (n=3) | 4.80 ± 0.00 (n=3) | 4.85 ± 2.68 (n=3) |
| hugadb | on | -- | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 6.76 ± 0.99 (n=3) | 3.17 ± 0.23 (n=3) | 4.93 ± 0.39 (n=3) |
| lara | off | -- | 6.08 ± 0.50 (n=3) | 13.05 ± 1.18 (n=3) | 9.85 ± 0.95 (n=3) | 11.09 ± 0.81 (n=3) | 17.21 ± 1.58 (n=3) |
| lara | on | -- | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 7.78 ± 0.67 (n=3) | 8.36 ± 1.15 (n=3) | 17.05 ± 0.73 (n=3) |

### Pooled: F1@50, setting E (secondary sensitivity)

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 2.64 ± 0.29 (n=3) | 6.73 ± 0.71 (n=3) | 5.73 ± 1.84 (n=3) | 7.06 ± 0.03 (n=3) | 22.31 ± 7.58 (n=3) |
| hugadb | on | -- | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 9.66 ± 1.16 (n=3) | 3.56 ± 0.12 (n=3) | 24.31 ± 2.45 (n=3) |
| lara | off | -- | 6.09 ± 0.51 (n=3) | 13.06 ± 1.19 (n=3) | 11.91 ± 1.19 (n=3) | 12.58 ± 1.00 (n=3) | 16.63 ± 1.86 (n=3) |
| lara | on | -- | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 10.69 ± 0.99 (n=3) | 10.95 ± 0.83 (n=3) | 17.21 ± 0.83 (n=3) |

#### Paired seed differences (F1@50, setting T)

1. **Claim** (fine vs class-matched vocabulary, same decoder):

| Dataset | Norm | A1 - A1-KC per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -2.50, -2.10, +1.38 | -1.07 | 1/3 |
| hugadb | on | +5.24, +2.56, +2.97 | +3.59 | 3/3 |
| lara | off | +1.19, -1.98, -2.93 | -1.24 | 1/3 |
| lara | on | +0.32, +1.00, -3.06 | -0.58 | 2/3 |

2. **Temporal assignment** (transport vs the fixed smoother):

| Dataset | Norm | A1 - A0+filter per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -4.03, -3.51, -1.54 | -3.03 | 0/3 |
| hugadb | on | +2.56, +0.30, +0.40 | +1.09 | 3/3 |
| lara | off | -3.28, -2.19, -4.12 | -3.20 | 0/3 |
| lara | on | -3.95, -7.88, -6.35 | -6.06 | 0/3 |

3. **Discretization** (hard codes vs continuous features, same decoder):

| Dataset | Norm | A1 - A-cont per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | -0.67, -0.24, -2.46 | -1.12 | 0/3 |
| hugadb | on | +2.78, +0.71, +1.98 | +1.82 | 3/3 |
| lara | off | -8.22, -7.08, -6.76 | -7.35 | 0/3 |
| lara | on | -9.79, -7.44, -10.57 | -9.27 | 0/3 |

### All five metrics, setting T (Pooled)

**MoF**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 28.12 ± 1.23 (n=3) | 32.86 ± 0.94 (n=3) | 37.71 ± 4.08 (n=3) | 41.52 ± 0.05 (n=3) | 34.97 ± 3.46 (n=3) |
| hugadb | on | -- | 27.60 ± 1.08 (n=3) | 32.46 ± 1.78 (n=3) | 40.72 ± 0.95 (n=3) | 32.07 ± 0.57 (n=3) | 35.07 ± 1.23 (n=3) |
| lara | off | -- | 21.96 ± 0.58 (n=3) | 22.68 ± 0.81 (n=3) | 27.73 ± 0.75 (n=3) | 29.49 ± 0.44 (n=3) | 28.30 ± 0.79 (n=3) |
| lara | on | -- | 26.97 ± 2.03 (n=3) | 28.52 ± 2.12 (n=3) | 29.62 ± 0.33 (n=3) | 28.38 ± 1.23 (n=3) | 26.16 ± 0.54 (n=3) |

**Edit**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 13.03 ± 1.48 (n=3) | 18.34 ± 0.76 (n=3) | 13.94 ± 6.98 (n=3) | 16.97 ± 0.04 (n=3) | 15.08 ± 6.34 (n=3) |
| hugadb | on | -- | 10.83 ± 0.34 (n=3) | 16.90 ± 0.81 (n=3) | 17.04 ± 1.94 (n=3) | 14.08 ± 1.31 (n=3) | 17.64 ± 0.37 (n=3) |
| lara | off | -- | 18.13 ± 0.26 (n=3) | 28.47 ± 0.03 (n=3) | 26.48 ± 0.96 (n=3) | 29.92 ± 0.32 (n=3) | 30.36 ± 0.42 (n=3) |
| lara | on | -- | 15.71 ± 0.70 (n=3) | 31.76 ± 1.89 (n=3) | 23.19 ± 0.26 (n=3) | 24.89 ± 1.90 (n=3) | 34.93 ± 0.63 (n=3) |

**F1@10**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 5.91 ± 0.67 (n=3) | 12.96 ± 0.34 (n=3) | 7.19 ± 2.04 (n=3) | 7.32 ± 0.01 (n=3) | 9.15 ± 2.77 (n=3) |
| hugadb | on | -- | 5.36 ± 0.41 (n=3) | 12.56 ± 0.20 (n=3) | 12.34 ± 1.14 (n=3) | 6.34 ± 0.36 (n=3) | 8.37 ± 0.90 (n=3) |
| lara | off | -- | 15.51 ± 0.72 (n=3) | 24.42 ± 1.55 (n=3) | 22.37 ± 0.79 (n=3) | 27.78 ± 0.12 (n=3) | 29.87 ± 1.35 (n=3) |
| lara | on | -- | 15.42 ± 1.26 (n=3) | 28.44 ± 2.60 (n=3) | 21.08 ± 0.57 (n=3) | 22.98 ± 1.61 (n=3) | 31.43 ± 0.64 (n=3) |

**F1@25**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 3.23 ± 0.21 (n=3) | 9.28 ± 0.34 (n=3) | 4.73 ± 1.96 (n=3) | 5.03 ± 0.00 (n=3) | 5.57 ± 2.75 (n=3) |
| hugadb | on | -- | 3.02 ± 0.21 (n=3) | 7.99 ± 0.24 (n=3) | 8.53 ± 0.94 (n=3) | 3.72 ± 0.35 (n=3) | 5.43 ± 0.41 (n=3) |
| lara | off | -- | 10.67 ± 0.65 (n=3) | 19.55 ± 1.54 (n=3) | 16.83 ± 1.16 (n=3) | 19.74 ± 0.53 (n=3) | 24.88 ± 1.55 (n=3) |
| lara | on | -- | 9.44 ± 1.30 (n=3) | 22.01 ± 2.47 (n=3) | 14.15 ± 0.66 (n=3) | 15.57 ± 1.54 (n=3) | 25.22 ± 0.61 (n=3) |

**F1@50**

| Dataset | Norm | SMQ | A0 | A0+filter | A1 | A1-KC | A-cont |
|---|---|---|---|---|---|---|---|
| hugadb | off | -- | 2.66 ± 0.27 (n=3) | 6.75 ± 0.69 (n=3) | 3.73 ± 1.74 (n=3) | 4.80 ± 0.00 (n=3) | 4.85 ± 2.68 (n=3) |
| hugadb | on | -- | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 6.76 ± 0.99 (n=3) | 3.17 ± 0.23 (n=3) | 4.93 ± 0.39 (n=3) |
| lara | off | -- | 6.08 ± 0.50 (n=3) | 13.05 ± 1.18 (n=3) | 9.85 ± 0.95 (n=3) | 11.09 ± 0.81 (n=3) | 17.21 ± 1.58 (n=3) |
| lara | on | -- | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 7.78 ± 0.67 (n=3) | 8.36 ± 1.15 (n=3) | 17.05 ± 0.73 (n=3) |

## Permutation adjacency diagnostic (not a primary comparison)

A1 refit on within-recording permuted codes (seed 111), states restored to the original timeline before scoring. Compare with A1 seed 111 in the same row: the gap is what true temporal adjacency contributes, including emission-refitting effects.

| Dataset | Norm | Setting | A1 seed 111 F1@50 | A1-permuted F1@50 | A1 run ratio | permuted run ratio |
|---|---|---|---|---|---|---|
| hugadb | off | E | 3.22 | 1.01 | -- | -- |
| hugadb | off | T | 2.30 | 1.76 | -- | -- |
| hugadb | on | E | 10.77 | 2.61 | -- | -- |
| hugadb | on | T | 8.11 | 3.91 | -- | -- |
| lara | off | E | 13.58 | 2.55 | -- | -- |
| lara | off | T | 11.18 | 7.94 | -- | -- |

## Convergence

Fit outer status counts prediction-set rows (each fit appears once per setting). Final solves count recordings. `capped` = declared cap reached with a finite, nonincreasing objective; capped results are included in every table above.

| Arm | Setting | fit outer: converged / capped / invalid | final solves (recordings): converged / capped / invalid | max residual | median outer iters |
|---|---|---|---|---|---|
| A1 | E | 0 / 12 / 0 | 0 / 0 / 0 | -- | -- |
| A1 | T | 0 / 12 / 0 | 0 / 0 / 0 | -- | -- |
| A1-KC | E | 8 / 4 / 0 | 0 / 0 / 0 | -- | -- |
| A1-KC | T | 8 / 4 / 0 | 0 / 0 / 0 | -- | -- |
| A-cont | E | 0 / 12 / 0 | 0 / 0 / 0 | -- | -- |
| A-cont | T | 0 / 12 / 0 | 0 / 0 / 0 | -- | -- |
| A0 | E | 1 / 11 / 0 | 0 / 0 / 0 | -- | -- |
| A0 | T | 1 / 11 / 0 | 0 / 0 / 0 | -- | -- |
| A1-permuted | E | 0 / 3 / 0 | 0 / 0 / 0 | -- | -- |
| A1-permuted | T | 0 / 3 / 0 | 0 / 0 / 0 | -- | -- |

## Limitations

- Protocol `pooled` only; subject-disjoint transfer not run.
- The frozen encoder is transductive; per-recording normalization and inference are offline.
- Local temporal consistency is not a learned action grammar.
- Every planned arm and setting is shown above or marked not run; nothing was selected by score.
