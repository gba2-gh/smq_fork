# Justifying the reduction step: why PCA, and why randomized SVD

Code: `script/week5/dimred_ablation/`. Results: `results/week5/dimred_ablation/solver_check/REPORT.md` (solver) and
`results/week5/dimred_ablation/<run-id>/REPORT.md` (method).

## Question

Every result reduces the standardized window vector with **PCA-64 fitted by sklearn's randomized SVD** before
K-means-512. We were asked to justify two choices:

1. **The algorithm.** Why randomized SVD rather than an exact solver? (stage S)
2. **The method.** Why PCA at all, rather than no reduction or another linear reduction? (stage M)

This is a justification, not a search. Each dataset's best configuration is held fixed, and only the reduction step
changes. Budget: 10 h at most.

## Anchors (held fixed)

Taken from `outputs/all_runs/all_runs.xlsx`, sheet "relevant results".

| dataset | features | window | reduction | arm | anchor F1@50 |
|---|---|---|---|---|---|
| HuGaDB | fsq_pilot_002 VQ latents (epoch 30), 1,440-d | 15 | PCA-64 | `context` (guard 14, span 8) | 49.55 (seed 111 only) |
| LARa | T2 inter-joint distances, 693-d | 12 | PCA-64 | `context` (span 2) | 24.02 |
| BABEL-2 | T2 inter-joint distances, 900-d | 8 | PCA-64 | `context` (span 2) | 30.68 |
| BABEL-3 | raw frames, 600-d | 8 | whitened PCA-64 | `pca` (no contextualizer) | 26.04 |

Everything else is unchanged and shared across datasets:
- K-means-512 on at most 10k complete windows.
- The contextualizer trained for 5000 updates, with the best-validation checkpoint.
- OT under setting T, with the radius set to fps // window and the 500/1000/2000/4000 rung ladder.
- Seeds 111, 222 and 1538574472.

Gates (passed on 2026-10-07): the `base` preparation reproduces each anchor's saved tokens.
- BABEL-2, BABEL-3 and HuGaDB match exactly.
- LARa matches exactly except for one window in about 220k, which sits on a K-means tie that a 1e-5 centroid
  difference flips.

## Stage S: the solver (token level, about 15 min, no decoding)

`solver_check.py` takes each dataset and seed and fits PCA-64 on the anchor's exact fitting sample with three solvers:
`randomized` (the anchor), `full` (exact LAPACK SVD) and `arpack` (exact truncated SVD). It reports:
- **Fit time** for each solver.
- **Variance captured** by each solver.
- **Subspace agreement:** the cosines of the principal angles between the randomized and the exact 64-d subspaces.
- **Code agreement:** K-means-512 is fitted on each solver's projection with the same seed, every window is assigned,
  and the codes are compared with the anchor's (agreement and ARI).
- **A noise floor for the codes:** the ARI between the anchor's codes and the anchor refitted with nothing changed
  except the K-means seed.

**Reading rule:**
- **Solver ARI at or above the K-means-seed ARI:** the solver changes the tokens less than K-means' own initialization
  does. The solver choice is then justified on speed alone, and decoding the exact-SVD variant (`pca_full`) is not
  informative.
- **Solver ARI clearly below the noise floor:** run stage S of `run_dimred` (`pca_full`, decoded) on the cheap datasets.

**Result (2026-10-07, 4 datasets × 3 seeds):** the randomized solver is equivalent to exact SVD at the token level, so
the decoded `pca_full` stage is not needed. Per-cell values are in
`results/week5/dimred_ablation/solver_check/REPORT.md`.

| dataset | variance captured, randomized − exact | mean subspace cosine | codes ARI vs exact | K-means seed reference ARI | exact SVD fit time |
|---|---|---|---|---|---|
| BABEL-2 | −2.7e-05 | 0.9994 | 0.655 | 0.643 | 6.6× slower |
| BABEL-3 | −5.6e-06 | 0.9994 | 0.547 | 0.542 | 5.6× slower |
| HuGaDB | −3.8e-07 | 1.0000 | 0.736 | 0.744 | 11.3× slower |
| LARa | −5.6e-05 | 0.9986 | 0.399 | 0.389 | 4.5× slower |

## Stage M: the method (decoded, about 9.3 h)

| key | reduction | what it answers |
|---|---|---|
| `base` | PCA-64, randomized SVD (the anchor's whitening) | the anchor, re-run under this code; the paired reference |
| `identity` | none: K-means on the full standardized vector (600–1,440-d) | is any reduction needed? |
| `random64` | Gaussian random projection to 64 | is it PCA's variance ordering that matters, or just having 64 dimensions? |
| `sfa64` | slow feature analysis: whitened PCA-128, then the 64 directions with the least window-to-window change | a principled alternative whose criterion (temporal persistence) is closer to segmentation than variance is |
| `kpca64` | kernel PCA, RBF kernel, Nyström form: 2000 landmark windows from the fitting sample, bandwidth = median pairwise landmark distance (label-free), 64 components, the anchor's scaling | would a nonlinear reduction do better? |

`kpca64` token-level check on real data (seed 111): fit about 9 s, projecting all of LARa's 219,500 windows 14 s, 512 codes
occupied everywhere; the codes differ from PCA's by an ARI of 0.52 (BABEL-2), 0.37 (BABEL-3), 0.67 (HuGaDB) and 0.36
(LARa), within the range the solver check measured for a change of K-means seed alone. The reported "retained variance"
for this variant is the share of the centred kernel spectrum on the landmarks (72–88%), not a share of input variance.

Notes on reading these variants:
- **Rotations are irrelevant.** Any rotation inside the kept subspace leaves K-means, the centroid lookup and the
  mean-centred cosine OT unchanged, so ICA or other rotations of PCA would add nothing.
- **Where the reduction acts.** On HuGaDB, LARa and BABEL-2 the reduction acts only through the K-means partition. On
  BABEL-3 it is the OT feature itself.
- **SFA's whitening.** `sfa64` is whitened by construction. On the unwhitened anchors part of any difference can come
  from whitening; `whiten_flip` (stage X) isolates that if needed.
- **The decision bar.** Each variant is compared with `base`, paired by seed. A difference counts only if the mean
  F1@50 difference is at least 3.0 and all three seeds have the same sign.
  - PCA is **justified** if no alternative is better by that bar.
  - PCA is **necessary** if `identity` or `random64` is worse by it.
- **HuGaDB.** This also gives the first 3-seed estimate of HuGaDB's 49.55, which was a single-seed result.

## Stage X: declared, not run

These are defined in `reducers.VARIANTS`, so a later run can add them under the same run id:
- PCA-8, 16, 32, 128 and 256;
- the PCA dimension that reaches 95% variance;
- whitening switched the other way, and half whitening;
- within-recording PCA and recording-centred data;
- no per-coordinate scaling.

## Commands

```powershell
# solver (already run once; rerun only if the code or data change)
& $env:SMQ_PYTHON -m script.week5.dimred_ablation.solver_check
# method, ~7.5 h
.\script\week5\dimred_ablation\run_dimred.ps1 -RunId dr_v1 -DryRun
.\script\week5\dimred_ablation\run_dimred.ps1 -RunId dr_v1 -Hours 10
```
