# Time-scale inspection, phase 1: OT kernel radius — ts1_v1

Status `finished`; 54 finished units. Source run: `rkx_v2` (validation-selected checkpoints, last final-solve rung = 4,000, outer cap 50). Only the kernel radius (and, in the beta table, β) changes; a = 0.7, λ = 0.05, ε = 0.07 and 25 inner steps are frozen. **BABEL-2 is the development set**: settings are chosen on it only, and every other dataset is reported with those settings frozen.

## Gates

| gate | status | cell | max |Δ| vs source ladder | F1@50 now / source |
|---|---|---|---|---|
| direct_pca_reproduces_task2 | passed | raw_kmeans__direct_continuous__seed111 | 0.0 | 3.97 / 3.97 |

A setting whose half-width equals the current one on every recording runs the identical solve; it reuses the source run's decode (`= current` column) once these gates pass.

## Screen — babel2 (development)

| setting | = current | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs current, paired (per seed) | MoF | Edit | F1@10 | pred segs / rec | GT segs / rec | states used | capped fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | yes | 3 | 15.09 ± 3.46 | 16.68 / 11.12 / 17.46 | +0.00 (+0.0 / +0.0 / +0.0) | 32.60 | 29.46 | 28.91 | 1.9 | 3.7 | 1.73 | 3 |

## Screen — babel3 (held out from selection)

| setting | = current | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs current, paired (per seed) | MoF | Edit | F1@10 | pred segs / rec | GT segs / rec | states used | capped fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | yes | 3 | 15.32 ± 0.65 | 14.78 / 16.04 / 15.15 | +0.00 (+0.0 / +0.0 / +0.0) | 32.12 | 28.08 | 30.98 | 2.7 | 3.7 | 2.12 | 3 |

## No-contextualizer inputs

`pca`: the patch's PCA-64 feature, mean-subtracted and L2-normalized (the repository's `direct_continuous` arm). `centre`: the K-means centre of its code, same treatment (`direct_quantized`). `context`: the selected checkpoint's contextual embedding (the `rkx_v2` decode). The OT fit, solver settings and scorer are identical; no checkpoint is involved for `pca` and `centre`. Δ is paired by seed against `context` at the same radius (the context column at a non-current radius exists only if the screen/confirm stage decoded it).

### hugadb

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 46.23 ± 3.30 | 50.00 / 44.83 / 43.85 |  | 48.88 | 42.68 | 9.0 | 10.5 | 3.78 |
| current | centre | 3 | 3.96 ± 0.02 | 3.95 / 3.96 / 3.98 | -42.26 (-46.0 / -40.9 / -39.9) | 34.83 | 17.54 | 118.1 | 10.5 | 6.62 |
| current | pca | 3 | 3.97 ± 0.00 | 3.97 / 3.97 / 3.97 | -42.26 (-46.0 / -40.9 / -39.9) | 34.52 | 17.45 | 117.9 | 10.5 | 6.83 |

### babel1

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 14.25 ± 1.44 | 15.89 / 13.21 / 13.64 |  | 31.32 | 30.63 | 2.6 | 5.1 | 2.14 |
| current | centre | 3 | 18.14 ± 1.21 | 17.43 / 17.45 / 19.54 | +3.89 (+1.5 / +4.2 / +5.9) | 36.89 | 32.69 | 3.1 | 5.1 | 2.14 |
| current | pca | 3 | 17.49 ± 1.35 | 16.74 / 16.67 / 19.05 | +3.24 (+0.8 / +3.5 / +5.4) | 36.76 | 30.52 | 3.3 | 5.1 | 2.50 |

### babel2

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 15.09 ± 3.46 | 16.68 / 11.12 / 17.46 |  | 32.60 | 29.46 | 1.9 | 3.7 | 1.73 |
| current | centre | 3 | 19.66 ± 0.95 | 20.10 / 18.57 / 20.30 | +4.57 (+3.4 / +7.5 / +2.8) | 34.16 | 33.23 | 2.7 | 3.7 | 1.93 |
| current | pca | 3 | 18.71 ± 0.63 | 19.06 / 17.99 / 19.09 | +3.63 (+2.4 / +6.9 / +1.6) | 34.23 | 32.25 | 3.0 | 3.7 | 2.24 |

### babel3

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 15.32 ± 0.65 | 14.78 / 16.04 / 15.15 |  | 32.12 | 28.08 | 2.7 | 3.7 | 2.12 |
| current | centre | 3 | 17.55 ± 0.41 | 17.92 / 17.62 / 17.11 | +2.23 (+3.1 / +1.6 / +2.0) | 34.40 | 36.16 | 4.3 | 3.7 | 2.14 |
| current | pca | 3 | 17.44 ± 0.63 | 17.10 / 17.06 / 18.17 | +2.12 (+2.3 / +1.0 / +3.0) | 34.30 | 34.11 | 4.3 | 3.7 | 2.49 |

### lara (global)

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 5.06 ± 0.48 | 5.03 / 4.60 / 5.55 |  | 20.23 | 22.50 | 14.8 | 32.1 | 5.81 |
| current | centre | 3 | 10.17 ± 0.20 | 10.30 / 9.94 / 10.28 | +5.11 (+5.3 / +5.3 / +4.7) | 27.10 | 28.45 | 19.7 | 32.1 | 5.91 |
| current | pca | 3 | 9.92 ± 0.21 | 10.16 / 9.76 / 9.84 | +4.86 (+5.1 / +5.2 / +4.3) | 25.94 | 27.25 | 18.8 | 32.1 | 5.94 |

### lara (per_recording)

| radius | input | seeds | F1@50 mean ± sd | per seed (111 / 222 / 1538574472) | Δ vs context, paired | MoF | Edit | pred segs / rec | GT segs / rec | states used |
|---|---|---|---|---|---|---|---|---|---|---|
| current | context | 3 | 7.13 ± 0.40 | 7.32 / 7.41 / 6.67 |  | 21.25 | 26.88 | 18.4 | 32.1 | 6.67 |
| current | centre | 3 | 9.12 ± 0.38 | 8.87 / 8.93 / 9.56 | +1.99 (+1.5 / +1.5 / +2.9) | 21.81 | 29.98 | 33.7 | 32.1 | 7.37 |
| current | pca | 3 | 9.71 ± 0.46 | 9.18 / 10.01 / 9.95 | +2.58 (+1.9 / +2.6 / +3.3) | 20.47 | 28.84 | 26.1 | 32.1 | 7.37 |

### Reading at the current radius (bar: ≥ 3 F1@50 points on the mean and in every seed)

- hugadb: context − centre = +42.26 (+46.0 / +40.9 / +39.9) → the contextualizer adds ≥ 3 points; pca − centre = +0.01 → quantization does not clearly cost ≥ 3 points.
- babel1: context − centre = -3.89 (-1.5 / -4.2 / -5.9) → the contextualizer does not clearly add ≥ 3 points; pca − centre = -0.65 → quantization does not clearly cost ≥ 3 points.
- babel2: context − centre = -4.57 (-3.4 / -7.5 / -2.8) → the contextualizer does not clearly add ≥ 3 points; pca − centre = -0.94 → quantization does not clearly cost ≥ 3 points.
- babel3: context − centre = -2.23 (-3.1 / -1.6 / -2.0) → the contextualizer does not clearly add ≥ 3 points; pca − centre = -0.11 → quantization does not clearly cost ≥ 3 points.
- lara global: context − centre = -5.11 (-5.3 / -5.3 / -4.7) → the contextualizer does not clearly add ≥ 3 points; pca − centre = -0.25 → quantization does not clearly cost ≥ 3 points.
- lara per_recording: context − centre = -1.99 (-1.5 / -1.5 / -2.9) → the contextualizer does not clearly add ≥ 3 points; pca − centre = +0.59 → quantization does not clearly cost ≥ 3 points.

## Declared reading

- The BABEL-2 screen is not finished (three seeds of every setting are needed); no verdict.

The 3-point bar comes from the observed seed-to-seed spread (up to about 6 points on HuGaDB and BABEL-2). A cross-dataset rule additionally needs one family within 1 point of the best setting on every dataset; compare the confirm tables. Decision point 1 is yours.
