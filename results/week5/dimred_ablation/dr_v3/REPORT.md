# Dimensionality-reduction ablation — `dr_v3`

Status: running; last rung of the OT ladder; seeds 111, 222, 1538574472.

## Gates

| gate | status | detail |
|---|---|---|
| g3 | passed |  |
| g_babel2 | passed | all checks equal; codes differ in 0 window(s) (agreement 1.000000); max centroid diff 7.63e-06 |
| g_babel3 | passed | all checks equal; codes differ in 0 window(s) (agreement 1.000000); max centroid diff 9.54e-07 |
| g_hugadb | passed | all checks equal; codes differ in 0 window(s) (agreement 1.000000); max centroid diff 1.53e-05 |
| g_lara | passed | all checks equal; codes differ in 0 window(s) (agreement 1.000000); max centroid diff 7.63e-06 |

## F1@50 difference from `base`, all datasets

| variant | babel2 | babel3 | hugadb |
|---|---|---|---|
| base F1@50 | 30.68 ± 0.74 | 26.04 ± 3.70 | 43.42 ± 8.66 |
| identity | -1.66 | -8.13 ✗ | n/a |
| random64 | -5.50 ✗ | -10.65 ✗ | n/a |
| sfa64 | -4.49 ✗ | -5.44 | n/a |
| kpca64 | -4.38 ✗ | -7.60 ✗ | n/a |
| pca_full | n/a | n/a | n/a |
| whiten_flip | n/a | n/a | n/a |
| pca16 | n/a | n/a | n/a |
| pca32 | n/a | n/a | n/a |
| pca128 | n/a | n/a | n/a |
| wrpca64 | n/a | n/a | n/a |
| rc_pca64 | n/a | n/a | n/a |
| nostd_pca64 | n/a | n/a | n/a |
| pca8 | n/a | n/a | n/a |
| pca256 | n/a | n/a | n/a |
| var95 | n/a | n/a | n/a |
| whiten_half | n/a | n/a | n/a |

✓ / ✗: better / worse than `base` by the bar (mean ≥ 3.0 F1@50 and every seed the same sign).

## babel2

Anchor: relational features, 8-frame windows, PCA-64, K-means-512, arm `context` (guard 0, span 2). Source: ts5_v1 phase 5 stage D, T2 span 2, F1@50 30.68; SMQ 27.5.

Re-run of the anchor (original unit -> this run's `base`): seed 111: 31.48 -> 31.48; seed 222: 30.02 -> 30.02; seed 1538574472: 30.54 -> 30.54.

| stage | variant | dims | retained var | codes used | NMI(code;action) | NMI(code;recording) | F1@50 | Δ F1@50 vs base (per seed) | reading | MoF | Edit | F1@10 | F1@25 | segments pred/gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | base (PCA-64, randomized SVD (anchor)) | 64 | 0.941 | 512 | 0.247 | 0.620 | 30.68 ± 0.74 | reference | — | 44.18 ± 6.45 | 40.34 ± 4.53 | 44.90 ± 2.12 | 40.57 ± 1.65 | 3.0 / 3.7 |
| M | identity (no reduction) | 900 | 1.000 | 512 | 0.248 | 0.624 | 29.02 ± 1.15 | -1.66 (-1.6 / -0.6 / -2.8) | no clear difference | 40.24 ± 1.86 | 39.03 ± 3.19 | 43.73 ± 2.44 | 38.96 ± 0.97 | 3.0 / 3.7 |
| M | random64 (random projection-64) | 64 | 0.072 | 512 | 0.240 | 0.617 | 25.18 ± 2.46 | -5.50 (-5.0 / -3.3 / -8.2) | **worse** | 39.45 ± 4.33 | 37.88 ± 1.14 | 41.44 ± 1.58 | 34.97 ± 2.63 | 3.0 / 3.7 |
| M | sfa64 (slow features-64 (whitened)) | 64 | 0.381 | 512 | 0.246 | 0.707 | 26.19 ± 0.25 | -4.49 (-5.1 / -4.1 / -4.3) | **worse** | 35.96 ± 3.72 | 37.34 ± 3.84 | 40.99 ± 1.41 | 35.72 ± 0.37 | 2.7 / 3.7 |
| M | kpca64 (kernel PCA-64 (Nystrom RBF)) | 64 | 0.793 | 512 | 0.243 | 0.636 | 26.30 ± 3.09 | -4.38 (-8.5 / -3.1 / -1.5) | **worse** | 38.67 ± 2.61 | 36.71 ± 3.49 | 41.13 ± 3.75 | 36.15 ± 4.02 | 2.9 / 3.7 |

## babel3

Anchor: raw features, 8-frame windows, whitened PCA-64, K-means-512, arm `pca`. Source: ts5_v1 phase 5 stage A, T1 / pca, F1@50 26.04; SMQ 19.4.

Re-run of the anchor (original unit -> this run's `base`): seed 111: 23.55 -> 23.55; seed 222: 24.28 -> 24.28; seed 1538574472: 30.30 -> 30.30.

| stage | variant | dims | retained var | codes used | NMI(code;action) | NMI(code;recording) | F1@50 | Δ F1@50 vs base (per seed) | reading | MoF | Edit | F1@10 | F1@25 | segments pred/gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | base (PCA-64, randomized SVD (anchor)) | 64 | 0.996 | 512 | 0.214 | 0.637 | 26.04 ± 3.70 | reference | — | 42.68 ± 5.93 | 38.79 ± 2.83 | 43.59 ± 3.61 | 36.61 ± 4.86 | 3.7 / 3.7 |
| M | identity (no reduction) | 600 | 1.000 | 512 | 0.208 | 0.655 | 17.91 ± 0.58 | -8.13 (-5.4 / -7.0 / -12.0) | **worse** | 34.37 ± 0.19 | 34.38 ± 0.22 | 35.68 ± 0.67 | 26.22 ± 0.74 | 4.2 / 3.7 |
| M | random64 (random projection-64) | 64 | 0.108 | 512 | 0.203 | 0.645 | 15.40 ± 0.83 | -10.65 (-8.1 / -9.7 / -14.1) | **worse** | 32.50 ± 1.76 | 31.45 ± 1.41 | 32.98 ± 1.06 | 23.88 ± 1.07 | 4.8 / 3.7 |
| M | sfa64 (slow features-64 (whitened)) | 64 | 0.129 | 512 | 0.254 | 0.748 | 20.61 ± 7.75 | -5.44 (-6.3 / +5.2 / -15.2) | no clear difference | 37.57 ± 8.89 | 36.34 ± 3.44 | 39.16 ± 5.48 | 31.87 ± 8.00 | 3.7 / 3.7 |
| M | kpca64 (kernel PCA-64 (Nystrom RBF)) | 64 | 0.886 | 512 | 0.211 | 0.658 | 18.45 ± 1.09 | -7.60 (-4.5 / -5.2 / -13.1) | **worse** | 35.70 ± 3.26 | 32.56 ± 1.24 | 36.05 ± 0.67 | 28.32 ± 1.32 | 4.0 / 3.7 |

## hugadb

Anchor: vq_latent features, 15-frame windows, PCA-64, K-means-512, arm `context` (guard 14, span 8). Source: task2_v1 vq_kmeans / context_id (seed 111 only), F1@50 49.55; SMQ 24.3.

Re-run of the anchor: seed 111 49.55 (task2_v1) -> 49.55 (this run's `base`; seeds 222 / 1538574472 are new).

| stage | variant | dims | retained var | codes used | NMI(code;action) | NMI(code;recording) | NMI(code;subject) | F1@50 | Δ F1@50 vs base (per seed) | reading | MoF | Edit | F1@10 | F1@25 | segments pred/gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | base (PCA-64, randomized SVD (anchor)) | 64 | 0.979 | 512 | 0.398 | 0.490 | 0.348 | 43.42 ± 8.66 (n=2) | reference | — | 48.03 ± 6.17 | 41.97 ± 6.97 | 48.28 ± 7.45 | 46.66 ± 7.50 | 8.8 / 10.5 |

## How to read this

* Only two things differ between variants: **which subspace** the window vector is reduced to, and **how its axes are
  scaled**. A rotation inside the kept subspace changes nothing downstream (K-means, centroid lookup and the
  mean-centred cosine OT cost are all rotation-invariant), so e.g. ICA would equal whitened PCA and is not run.
* For the `context` anchors (HuGaDB, LARa, BABEL-2) the reduction reaches the result only through the **K-means
  partition** (and the initial code grouping): the contextualizer sees code identities. For BABEL-3 (`pca` arm) the
  reduced vector **is** the OT feature, so the reduction acts on the cost directly.
* `sfa64` is whitened by construction: read it against `whiten_flip` where the anchor is unwhitened (HuGaDB, LARa,
  BABEL-2) and against `base` on BABEL-3. The same holds for `whiten_half`, which sits between the two.
* `identity` keeps every coordinate without scaling; on BABEL-3 it is also unwhitened, so its difference from `base`
  mixes "no reduction" with "no whitening" (`whiten_flip` isolates the second).
* `wrpca64` changes only the axes (fitted on recording-centred windows); `rc_pca64` also removes each recording's mean
  from the data. NMI(code; recording / subject) shows whether either actually reduced recording identity in the codes.
* NMI columns are evaluation-only descriptions of the tokens, computed after preparation; nothing is fitted on them.
