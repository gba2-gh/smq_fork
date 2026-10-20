# Time-scale inspection, phase 4: input representation — ts4_v1

Status `finished`. Radius: `current`. Positions-plus-velocity family, built fresh per cell; the position-only baseline is **not rebuilt** here -- `context` comes from `rkx_v2`'s saved results, `pca`/`centre` from `ts1_v1`'s Phase 1 `inputs` stage, both at the same radius. HuGaDB is not rebuilt (it is already derivative IMU data); its position-only numbers from `rkx_v2`/Phase 1 stand as the control.

## babel2

| arm | pos seeds | position per seed | position mean±sd | vel seeds | +velocity per seed | +velocity mean±sd | Δ (paired) | +velocity pred segs/rec |
|---|---|---|---|---|---|---|---|---|
| context | 3 | 16.68 / 11.12 / 17.46 | 15.09 ± 3.46 | 3 | 20.84 / 13.73 / 20.23 | 18.27 ± 3.94 | +3.18 (+4.2 / +2.6 / +2.8) | 1.9 |
| pca | 3 | 19.06 / 17.99 / 19.09 | 18.71 ± 0.63 | 3 | 20.88 / 18.95 / 18.32 | 19.38 ± 1.33 | +0.67 (+1.8 / +1.0 / -0.8) | 2.9 |
| centre | 3 | 20.10 / 18.57 / 20.30 | 19.66 ± 0.95 | 3 | 22.64 / 20.40 / 19.35 | 20.80 ± 1.68 | +1.14 (+2.5 / +1.8 / -0.9) | 2.7 |

## babel3

| arm | pos seeds | position per seed | position mean±sd | vel seeds | +velocity per seed | +velocity mean±sd | Δ (paired) | +velocity pred segs/rec |
|---|---|---|---|---|---|---|---|---|
| context | 3 | 14.78 / 16.04 / 15.15 | 15.32 ± 0.65 | 3 | 13.14 / 13.99 / 13.26 | 13.46 ± 0.46 | -1.86 (-1.6 / -2.0 / -1.9) | 2.7 |
| pca | 3 | 17.10 / 17.06 / 18.17 | 17.44 ± 0.63 | 3 | 19.18 / 18.09 / 18.87 | 18.71 ± 0.56 | +1.27 (+2.1 / +1.0 / +0.7) | 3.8 |
| centre | 3 | 17.92 / 17.62 / 17.11 | 17.55 ± 0.41 | 3 | 9.70 / 17.13 / 18.50 | 15.11 ± 4.74 | -2.44 (-8.2 / -0.5 / +1.4) | 4.1 |

## lara

| arm | pos seeds | position per seed | position mean±sd | vel seeds | +velocity per seed | +velocity mean±sd | Δ (paired) | +velocity pred segs/rec |
|---|---|---|---|---|---|---|---|---|
| context | 3 | 5.03 / 4.60 / 5.55 | 5.06 ± 0.48 | 3 | 7.27 / 8.65 / 8.69 | 8.20 ± 0.81 | +3.14 (+2.2 / +4.1 / +3.1) | 18.2 |
| pca | 3 | 10.16 / 9.76 / 9.84 | 9.92 ± 0.21 | 3 | 9.99 / 8.63 / 10.28 | 9.63 ± 0.88 | -0.29 (-0.2 / -1.1 / +0.4) | 18.9 |
| centre | 3 | 10.30 / 9.94 / 10.28 | 10.17 ± 0.20 | 3 | 10.34 / 10.18 / 10.36 | 10.29 ± 0.10 | +0.12 (+0.0 / +0.2 / +0.1) | 22.4 |

## Per-token action information (Phase 0.4 probes, balanced accuracy, mean over seeds)

Position-only numbers from `ts0_v1`'s Phase 0 `probes.csv`; +velocity from this run. Same grouped split, caps and classifiers. The H4 rule uses the linear probe on the PCA feature.

| dataset | probe | representation | position | +velocity | Δ | seeds (pos/vel) |
|---|---|---|---|---|---|---|
| babel2 | linear | pca | 0.422 | 0.349 | -0.073 | 3/3 |
| babel2 | linear | centre | 0.413 | 0.396 | -0.016 | 3/3 |
| babel2 | linear | onehot | 0.418 | 0.468 | +0.050 | 3/3 |
| babel2 | linear | context | 0.367 | 0.414 | +0.047 | 3/3 |
| babel2 | nn | pca | 0.520 | 0.624 | +0.104 | 3/3 |
| babel2 | nn | centre | 0.477 | 0.507 | +0.030 | 3/3 |
| babel2 | nn | onehot | 0.478 | 0.489 | +0.012 | 3/3 |
| babel2 | nn | context | 0.461 | 0.507 | +0.046 | 3/3 |
| babel3 | linear | pca | 0.379 | 0.342 | -0.038 | 3/3 |
| babel3 | linear | centre | 0.355 | 0.340 | -0.015 | 3/3 |
| babel3 | linear | onehot | 0.345 | 0.368 | +0.023 | 3/3 |
| babel3 | linear | context | 0.323 | 0.396 | +0.074 | 3/3 |
| babel3 | nn | pca | 0.420 | 0.458 | +0.038 | 3/3 |
| babel3 | nn | centre | 0.333 | 0.339 | +0.006 | 3/3 |
| babel3 | nn | onehot | 0.380 | 0.387 | +0.007 | 3/3 |
| babel3 | nn | context | 0.380 | 0.426 | +0.046 | 3/3 |
| lara | linear | pca | 0.464 | 0.370 | -0.094 | 3/3 |
| lara | linear | centre | 0.355 | 0.364 | +0.008 | 3/3 |
| lara | linear | onehot | 0.330 | 0.405 | +0.074 | 3/3 |
| lara | linear | context | 0.271 | 0.398 | +0.127 | 3/3 |
| lara | nn | pca | 0.445 | 0.507 | +0.062 | 3/3 |
| lara | nn | centre | 0.323 | 0.355 | +0.032 | 3/3 |
| lara | nn | onehot | 0.343 | 0.417 | +0.073 | 3/3 |
| lara | nn | context | 0.338 | 0.439 | +0.100 | 3/3 |

## H4 reading (bar: ≥ 3 F1@50 points on the mean, all 3 seeds in the same direction)

- babel2 / context: position 15.09 ± 3.46 -> +velocity 18.27 ± 3.94 (Δ +3.18) -> **helps** (bar ±3, paired, all 3 seeds)
- babel2 / pca: position 18.71 ± 0.63 -> +velocity 19.38 ± 1.33 (Δ +0.67) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- babel2 / centre: position 19.66 ± 0.95 -> +velocity 20.80 ± 1.68 (Δ +1.14) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- babel3 / context: position 15.32 ± 0.65 -> +velocity 13.46 ± 0.46 (Δ -1.86) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- babel3 / pca: position 17.44 ± 0.63 -> +velocity 18.71 ± 0.56 (Δ +1.27) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- babel3 / centre: position 17.55 ± 0.41 -> +velocity 15.11 ± 4.74 (Δ -2.44) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- lara / context: position 5.06 ± 0.48 -> +velocity 8.20 ± 0.81 (Δ +3.14) -> **helps** (bar ±3, paired, all 3 seeds)
- lara / pca: position 9.92 ± 0.21 -> +velocity 9.63 ± 0.88 (Δ -0.29) -> **no clear effect** (bar ±3, paired, all 3 seeds)
- lara / centre: position 10.17 ± 0.20 -> +velocity 10.29 ± 0.10 (Δ +0.12) -> **no clear effect** (bar ±3, paired, all 3 seeds)

**H4 (position-only input lacks the motion signal IMU input has): partly supported (F1@50 only, probe did not rise).** Rule: velocity helps F1@50 on BABEL-2 or LARa by the bar in some arm, and the linear probe on the PCA feature gains action information on that dataset. BABEL-3 is reported but not part of the declared rule.
