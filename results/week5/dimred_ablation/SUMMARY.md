# Dimensionality-reduction ablation: results to date (2026-10-08)

Question: why PCA-64 with randomized SVD, on 10k fitting windows, as the step between standardization and K-means-512.
Three parts. Parts 1 and 2 are complete; part 3 (`dr_v3`) has 33 of 60 units and is not running.

## 1. Why randomized SVD (token level, complete: 4 datasets x 3 seeds)
Source: `solver_check/REPORT.md`.

| dataset | variance captured, randomized - exact | mean subspace cosine | codes ARI vs exact SVD | ARI from changing only the K-means seed | exact SVD fit time |
|---|---|---|---|---|---|
| BABEL-2 | -2.7e-05 | 0.9994 | 0.655 | 0.643 | 6.6x slower |
| BABEL-3 | -5.6e-06 | 0.9994 | 0.547 | 0.542 | 5.6x slower |
| HuGaDB | -3.8e-07 | 1.0000 | 0.736 | 0.744 | 11.3x slower |
| LARa | -5.6e-05 | 0.9986 | 0.399 | 0.389 | 4.5x slower |

Reading: the randomized and exact solvers keep the same subspace and variance; the codes differ from exact-SVD codes no
more than a K-means seed change makes them differ. ARPACK gives the same result. The solver does not matter.

## 2. Does the 10k-window cap matter (token level, complete: HuGaDB and LARa; BABEL has about 10k windows in total)
Source: `sample_size_check/REPORT.md`. Fitted on training recordings, scored on held-out validation recordings.

| dataset | fitting windows | fit time | held-out relative error | ARI between two K-means seeds | codes used (of 512) |
|---|---|---|---|---|---|
| HuGaDB | 10,000 | 6 s | 0.203 | 0.966 | 332 |
| HuGaDB | 40,000 | 23 s | 0.191 | 0.928 | 340 |
| HuGaDB | all (67,962) | 39 s | 0.189 | 0.981 | 336 |
| LARa | 10,000 | 6 s | 0.413 | 0.633 | 492 |
| LARa | 40,000 | 28 s | 0.385 | 0.609 | 490 |
| LARa | all (197,500) | 124 s | 0.370 | 0.693 | 494 |

Cross-size ARI against the smaller size's seed ARI: HuGaDB 0.948 / 0.943 / 0.949 vs 0.966 / 0.966 / 0.928 (no more than
seed noise); LARa 0.568 / 0.536 / 0.609 vs 0.633 / 0.633 / 0.609 (10k vs larger moves codes more than seed noise; 40k
vs all does not).

Reading: harmless for HuGaDB; about 10% lower held-out error for LARa with all windows, still falling at 197k. Code
instability (seed ARI) does not improve with sample size, so it is intrinsic to K-means-512 on this data. Token-level
evidence only; no decoding.

## 3. Why PCA (decoded, `dr_v3`, 33 of 60 units; F1@50, last rung, 3 seeds)
Per-run scores of every metric: `dr_v3/SCORES.md`.

| dataset | variant | F1@50 mean | change from base (per seed) | reading by the 3.0 bar |
|---|---|---|---|---|
| BABEL-2 | base (PCA-64) | 30.68 +- 0.74 | | |
| BABEL-2 | no reduction | 29.02 | -1.66 (-1.6 / -0.6 / -2.8) | no clear difference |
| BABEL-2 | random projection-64 | 25.18 | -5.50 (-5.0 / -3.4 / -8.2) | worse |
| BABEL-2 | slow features-64 | 26.19 | -4.49 (-5.1 / -4.1 / -4.3) | worse |
| BABEL-2 | kernel PCA-64 | 26.30 | -4.38 (-8.5 / -3.2 / -1.5) | worse |
| BABEL-3 | base (whitened PCA-64) | 26.04 +- 3.70 | | |
| BABEL-3 | no reduction | 17.91 | -8.13 (-5.4 / -7.0 / -12.0) | worse |
| BABEL-3 | random projection-64 | 15.40 | -10.65 (-8.1 / -9.7 / -14.1) | worse |
| BABEL-3 | slow features-64 | 20.61 +- 7.75 | -5.44 (-6.3 / +5.2 / -15.2) | no clear difference (erratic) |
| BABEL-3 | kernel PCA-64 | 18.45 | -7.60 (-4.5 / -5.2 / -13.1) | worse |
| HuGaDB | base (PCA-64) | 44.73 +- 6.53 | | alternatives not run yet |
| LARa | base | not run yet | | |

Reading so far: no alternative beats PCA on BABEL-2/3; random projection is worst everywhere (PCA's variance ordering
matters, not only the dimension). The 49.55 anchor on HuGaDB was a favourable seed (49.55 / 37.30 / 47.35).
Confounds: BABEL-3's anchor is whitened, so "no reduction" also differs in scaling; kernel PCA and random projection are
scaled differently from PCA. HuGaDB and LARa have no alternative results yet.

## Not yet answered
- HuGaDB and LARa alternatives (27 units, about 5 h): `python -m script.week5.dimred_ablation.run_dimred --run-id dr_v3 --hours 10`
  resumes the stalled run (last unit written 11:45; cause unknown, possibly the same BLAS crash on a full-dimension
  preparation).
- Why 64 dimensions: only the T3/T3b result (PCA-256 vs PCA-64 on LARa, -0.29 F1@50 with the contextualizer) exists; the
  dimension curve (PCA-16/32/128/256) is implemented but not run (stage X, about 7 h).
- Decoded effect of fitting on all windows for LARa.
