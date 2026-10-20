# MASQ replication (Tier 0): masq_hu1

Status `finished`. Training code `14712f110377f9ea`, scoring code `9f2e138ac2219a09`; analysis hash at launch `7a85a5ae610b3200` (now `7a85a5ae610b3200`).
Checkpoint: final epoch (30 for results; 2 for smoke), global Hungarian mapping (primary, as in the published tables), MoF and F1@50 co-primary. Bar: mean difference >= 3 with every seed in the same direction, read only with three seeds.

## Gates

| gate | status |
|---|---|
| G1 | passed |
| G2 | passed |
| G3 | passed |
| G6 | passed |
| G7 | passed |
| G4 | passed |

## Cells (global Hungarian; seed mean ± SD, SD = population SD over seeds)

### HuGaDB

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 3 | 41.76 ± 5.14 | 36.53 ± 4.40 | 40.12 ± 6.17 | 31.78 ± 5.91 | 22.83 ± 5.62 | 13.09 ± 2.24 | 10.50 ± 0.00 | 9.33 ± 0.47 |
| V2a | 3 | 42.84 ± 5.87 | 36.09 ± 4.86 | 40.85 ± 7.35 | 31.20 ± 7.98 | 20.80 ± 7.83 | 11.91 ± 1.13 | 10.50 ± 0.00 | 9.00 ± 0.82 |
| V2b | 3 | 45.03 ± 0.99 | 38.83 ± 4.74 | 42.10 ± 4.50 | 34.26 ± 4.69 | 25.26 ± 4.63 | 10.73 ± 1.41 | 10.50 ± 0.00 | 8.33 ± 0.47 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 52.3 | 45.1 | 52.1 | 47.0 | 39.5 |
| SMQ released checkpoint (re-scored, e0) | 41.99 | 36.15 | 38.5 | 31.54 | 24.3 |
| SMQ retrained, earlier 3 seeds (mean ± SD) | 41.6 ± 1.13 | — | — | — | 21.5 ± 2.54 |

### Per-sequence (local) Hungarian, reported separately and never mixed with the global scores

**HuGaDB**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| V1 | 60.47 ± 2.51 | 46.60 ± 3.07 | 54.28 ± 4.12 | 45.99 ± 4.66 | 32.32 ± 4.61 |
| V2a | 59.56 ± 3.36 | 43.07 ± 4.13 | 51.58 ± 4.70 | 43.61 ± 5.75 | 28.52 ± 6.65 |
| V2b | 62.56 ± 1.16 | 48.57 ± 5.59 | 56.93 ± 5.72 | 49.34 ± 6.54 | 34.91 ± 6.19 |

## Comparisons (A minus B, paired by seed; MoF and F1@50 co-primary)

### R1: M-pub minus S0: does MASQ's training change improve SMQ?

- **hugadb**: MoF — -> no cells; F1@50 — -> no cells

### R2: M-pub mean vs published MASQ (reproduced if within 3 points on MoF and F1@50)

- **hugadb**: incomplete

### R3: S0 (our retrained SMQ) vs the references


### R4: not run (stage 2)

## Every metric

| id | dataset | A − B | metric | Δ (per seed) | n | reading |
|---|---|---|---|---|---|---|

## R5: identity diagnostics (descriptive; subject only for HuGaDB and LARa)

Code AMI = chance-corrected mutual information of native codes with the metadata. Latent NN share = same-recording share of nearest neighbours of patch-mean latents (adjacent patches excluded). The per-class block uses labels (analysis only).

| dataset | arm | n | recording AMI | subject AMI | per-class subject AMI (labels) | latent NN share |
|---|---|---|---|---|---|---|
| hugadb | V1 | 3 | 0.27 ± 0.01 | 0.11 ± 0.02 | 0.20 ± 0.06 | 0.37 ± 0.00 |
| hugadb | V2a | 3 | 0.25 ± 0.02 | 0.09 ± 0.01 | 0.14 ± 0.01 | 0.36 ± 0.01 |
| hugadb | V2b | 3 | 0.25 ± 0.02 | 0.09 ± 0.01 | 0.15 ± 0.01 | 0.35 ± 0.01 |

M-pub minus S0 (per seed):

| dataset | measure | Δ | n |
|---|---|---|---|

## Training (final epoch losses)

| dataset | arm | seed | rec | commit | vel | dead-code replacements | EMA codes in use | seconds |
|---|---|---|---|---|---|---|---|---|
| hugadb | V1 | 111 | 87882.29624 | 23148.37914 | 3789.26711 | 1647 | 10 | 526 |
| hugadb | V1 | 222 | 91399.85717 | 24608.62027 | 3915.49740 | 1533 | 10 | 462 |
| hugadb | V1 | 1538574472 | 104507.12356 | 25168.40812 | 3738.57281 | 1580 | 10 | 168 |
| hugadb | V2a | 111 | 85726.72737 | 24384.61923 | 481.54359 | 1611 | 10 | 480 |
| hugadb | V2a | 222 | 89732.41355 | 22607.22191 | 424.73920 | 1519 | 10 | 491 |
| hugadb | V2a | 1538574472 | 96722.83551 | 23814.09682 | 518.31134 | 1540 | 10 | 171 |
| hugadb | V2b | 111 | 95385.03711 | 24382.88037 | 1758.97098 | 1492 | 10 | 481 |
| hugadb | V2b | 222 | 89559.69001 | 23299.18775 | 2010.90684 | 1633 | 10 | 313 |
| hugadb | V2b | 1538574472 | 90025.16912 | 26697.87856 | 1926.24802 | 1453 | 10 | 169 |

## Verdict (computed from the declared rule)

Not computed: the verdict needs S0 and M-pub with three seeds on all five datasets (Stage 1 incomplete).

## Caveats

All results here are *development* results (protocol P7): every dataset informed earlier decisions and no untouched evaluation set exists. The published MASQ numbers are single runs from a paper that is not peer reviewed; our M arms are a reimplementation from the paper text with the SMQ backbone and defaults, so a gap to the published numbers may come from undocumented details. These experiments cannot show *why* MASQ works, only whether its training changes reproduce and which component matters; they say nothing about our OT decoder on MASQ latents (Tier 1).
