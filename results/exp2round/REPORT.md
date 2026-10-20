# Independent unit-vocabulary sweep

## Outcome

**Negative result; Gate 2 failed.** The best HuGaDB cell was W=15, K_u=20 at 47.96 MoF (SD 0.41), versus the recomputed D7 majority-code reference of 44.96. The improvement was 3.00, below the required +5.00.

![MoF vocabulary sweep](units_sweep_mof.png)

## Gates

- Gate 0: PASS (0.999801 agreement; threshold 0.999).
- Gate 1: PASS (47.58 mean MoF; 1.16 from 46.42).
- Gate 2: FAIL (HuGaDB improvement 3.00; required 5.00 on both datasets).

## Full primary curve (mean across three seeds)

| W | K_u | MoF | SD | F1@50 | Units/GT segment | NMI action | NMI subject |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 10 | 47.37 | 0.05 | 42.24 | 20.75 | 0.483 | 0.124 |
| 15 | 20 | 47.96 | 0.41 | 42.62 | 20.75 | 0.451 | 0.159 |
| 15 | 50 | 43.94 | 0.81 | 39.09 | 20.75 | 0.411 | 0.213 |
| 15 | 100 | 42.40 | 1.19 | 41.51 | 20.75 | 0.391 | 0.267 |
| 15 | 500 | 38.63 | 2.43 | 34.53 | 20.75 | 0.353 | 0.381 |
| 15 | 1000 | 37.99 | 2.40 | 36.33 | 20.75 | 0.343 | 0.401 |
| 30 | 10 | 46.39 | 1.12 | 41.16 | 10.85 | 0.497 | 0.127 |
| 30 | 20 | 46.04 | 0.34 | 36.24 | 10.85 | 0.446 | 0.155 |
| 30 | 50 | 43.55 | 0.79 | 34.46 | 10.85 | 0.408 | 0.229 |
| 30 | 100 | 42.39 | 0.62 | 41.88 | 10.85 | 0.399 | 0.288 |
| 30 | 500 | 37.08 | 3.90 | 30.98 | 10.85 | 0.366 | 0.405 |
| 30 | 1000 | 35.48 | 2.38 | 30.07 | 10.85 | 0.357 | 0.428 |
| 60 | 10 | 47.58 | 1.38 | 36.85 | 5.90 | 0.496 | 0.127 |
| 60 | 20 | 43.46 | 1.04 | 35.76 | 5.90 | 0.449 | 0.169 |
| 60 | 50 | 41.98 | 0.59 | 26.38 | 5.90 | 0.407 | 0.253 |
| 60 | 100 | 38.15 | 2.06 | 31.29 | 5.90 | 0.408 | 0.303 |
| 60 | 500 | 33.12 | 1.67 | 19.11 | 5.90 | 0.377 | 0.422 |
| 60 | 1000 | 28.91 | 1.38 | 10.05 | 5.90 | 0.367 | 0.449 |

## Controls

| Control | Best mean MoF | W | K_u |
|---|---:|---:|---:|
| Majority independent unit | 46.63 | 15 | 10 |
| Continuous PCA-64 segment mean | 43.82 | 30 | n/a |
| Shuffled units | 21.08 | 15 | 100 |
| Length-weighted segment clustering | 47.41 | 15 | 20 |

## Supported

- Gate 0 verifies that dumped pre-quantisation latents reproduce cached unit assignments.
- Real unit structure matters: shuffled-unit controls collapse well below the primary cells.
- At the best cell, unit/action NMI exceeds unit/subject NMI; identity nuisance is not the main explanation there.
- Shorter windows increase observations per GT segment, but do not deliver the preregistered +5 MoF improvement.

## Not supported

- A larger independent vocabulary does not make action identity substantially recoverable under this protocol.
- Performance does not improve monotonically with vocabulary size; the largest vocabularies often degrade it.
- Gate 2 does not justify expansion to retrained T1 seeds or BABEL-1.

## Planned but not run

- LARa released checkpoint: stopped because HuGaDB already made the two-dataset Gate 2 conjunction impossible.
- Three T1 K=C seeds and BABEL-1: prohibited after Gate 2 failure.
- W=patch, K_u=100 without PCA: raw-dimensional k-means was terminated by the OS; no result is claimed.

## Protocol notes

- CPU only; fixed seeds 111, 222, and 1538574472.
- Incomplete terminal windows were excluded from vocabulary/PCA fitting and zero-padded only for assignment, preserving frame alignment.
- The explicit `models/hugadb/epoch-30.model` checkpoint differs from `models/pretrained/hugadb.model` behind the original 44.85 table; therefore the same-checkpoint D7 reference (44.96) is primary here.
