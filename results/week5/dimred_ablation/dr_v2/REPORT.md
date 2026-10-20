# Dimensionality-reduction ablation — `dr_v2`

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

| variant | babel2 |
|---|---|
| base F1@50 | 30.68 ± 0.74 |
| identity | n/a |
| random64 | n/a |
| sfa64 | n/a |
| kpca64 | n/a |
| pca_full | n/a |
| whiten_flip | n/a |
| pca16 | n/a |
| pca32 | n/a |
| pca128 | n/a |
| wrpca64 | n/a |
| rc_pca64 | n/a |
| nostd_pca64 | n/a |
| pca8 | n/a |
| pca256 | n/a |
| var95 | n/a |
| whiten_half | n/a |

✓ / ✗: better / worse than `base` by the bar (mean ≥ 3.0 F1@50 and every seed the same sign).

## babel2

Anchor: relational features, 8-frame windows, PCA-64, K-means-512, arm `context` (guard 0, span 2). Source: ts5_v1 phase 5 stage D, T2 span 2, F1@50 30.68; SMQ 27.5.

Re-run of the anchor (original unit -> this run's `base`): seed 111: 31.48 -> 31.48; seed 222: 30.02 -> 30.02; seed 1538574472: 30.54 -> 30.54.

| stage | variant | dims | retained var | codes used | NMI(code;action) | NMI(code;recording) | F1@50 | Δ F1@50 vs base (per seed) | reading | MoF | Edit | F1@10 | F1@25 | segments pred/gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | base (PCA-64, randomized SVD (anchor)) | 64 | 0.941 | 512 | 0.247 | 0.620 | 30.68 ± 0.74 | reference | — | 44.18 ± 6.45 | 40.34 ± 4.53 | 44.90 ± 2.12 | 40.57 ± 1.65 | 3.0 / 3.7 |

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
