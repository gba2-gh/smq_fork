# MASQ replication (Tier 0): masq_v1

Status `budget_stop`. Training code `14712f110377f9ea`, scoring code `9f2e138ac2219a09`; analysis hash at launch `7a85a5ae610b3200` (now `7a85a5ae610b3200`).
Checkpoint: final epoch (30 for results; 2 for smoke), global Hungarian mapping (primary, as in the published tables), MoF and F1@50 co-primary. Bar: mean difference >= 3 with every seed in the same direction, read only with three seeds.

## Gates

| gate | status |
|---|---|
| G1 | passed |
| G2 | passed |
| G3 | passed |
| G4 | passed |

## Not run

- `M-fix__babel1__seed1538574472`: budget: projected to cross the deadline
- `M-fix__babel1__seed222`: budget: projected to cross the deadline
- `M-fix__lara__seed111`: budget: projected to cross the deadline
- `M-fix__lara__seed1538574472`: budget: projected to cross the deadline
- `M-fix__lara__seed222`: budget: projected to cross the deadline

## Cells (global Hungarian; seed mean ± SD, SD = population SD over seeds)

### BABEL-3

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 3 | 38.14 ± 0.43 | 32.61 ± 8.34 | 33.18 ± 7.46 | 25.12 ± 5.55 | 17.23 ± 3.85 | 4.83 ± 0.26 | 3.70 ± 0.00 | 5.00 ± 0.00 |
| M-pub | 3 | 39.67 ± 2.78 | 25.10 ± 8.48 | 24.87 ± 7.69 | 19.04 ± 4.28 | 11.96 ± 2.15 | 3.85 ± 1.08 | 3.70 ± 0.00 | 4.33 ± 0.94 |
| M-JL | 3 | 41.07 ± 5.45 | 29.51 ± 2.79 | 30.25 ± 3.13 | 25.55 ± 0.25 | 19.58 ± 1.78 | 2.91 ± 1.74 | 3.70 ± 0.00 | 4.33 ± 0.47 |
| M-fix | 3 | 37.88 ± 0.13 | 31.18 ± 6.91 | 31.89 ± 7.65 | 22.53 ± 4.88 | 14.49 ± 2.64 | 5.38 ± 0.11 | 3.70 ± 0.00 | 5.00 ± 0.00 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 47.7 | 26.9 | 28.3 | 27.5 | 24.5 |
| SMQ released checkpoint (re-scored, e0) | 40.37 | 38.6 | 37.58 | 28.59 | 19.35 |

### BABEL-2

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 3 | 39.86 ± 3.67 | 31.65 ± 0.98 | 36.09 ± 2.28 | 28.37 ± 1.52 | 17.81 ± 0.80 | 3.16 ± 0.10 | 3.74 ± 0.00 | 5.00 ± 0.00 |
| M-pub | 3 | 38.46 ± 4.50 | 30.10 ± 2.44 | 33.29 ± 3.33 | 26.24 ± 3.61 | 17.52 ± 4.02 | 3.64 ± 0.28 | 3.74 ± 0.00 | 5.00 ± 0.00 |
| M-JL | 3 | 38.92 ± 5.74 | 27.89 ± 4.17 | 33.91 ± 5.01 | 28.42 ± 4.30 | 18.64 ± 2.89 | 3.48 ± 0.16 | 3.74 ± 0.00 | 5.00 ± 0.00 |
| M-fix | 3 | 37.34 ± 5.02 | 31.08 ± 0.98 | 34.20 ± 1.18 | 26.91 ± 1.08 | 17.43 ± 1.45 | 3.18 ± 0.45 | 3.74 ± 0.00 | 5.00 ± 0.00 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 55.8 | 38.1 | 46.1 | 38.7 | 28.4 |
| SMQ released checkpoint (re-scored, e0) | 49.17 | 37.81 | 43.7 | 37.42 | 27.52 |

### HuGaDB

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 3 | 40.72 ± 1.54 | 32.78 ± 2.58 | 34.77 ± 2.49 | 27.68 ± 1.78 | 22.04 ± 1.48 | 17.56 ± 0.89 | 10.50 ± 0.00 | 10.00 ± 0.00 |
| M-pub | 3 | 39.74 ± 1.52 | 34.41 ± 1.99 | 36.89 ± 1.27 | 28.95 ± 1.42 | 21.12 ± 2.42 | 16.24 ± 0.81 | 10.50 ± 0.00 | 10.00 ± 0.00 |
| M-JL | 3 | 45.24 ± 0.92 | 37.36 ± 1.00 | 42.59 ± 0.40 | 35.73 ± 0.83 | 27.56 ± 1.56 | 12.83 ± 1.73 | 10.50 ± 0.00 | 9.33 ± 0.47 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 52.3 | 45.1 | 52.1 | 47.0 | 39.5 |
| SMQ released checkpoint (re-scored, e0) | 41.99 | 36.15 | 38.5 | 31.54 | 24.3 |
| SMQ retrained, earlier 3 seeds (mean ± SD) | 41.6 ± 1.13 | — | — | — | 21.5 ± 2.54 |

### BABEL-1

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 3 | 32.95 ± 1.76 | 27.20 ± 0.77 | 31.96 ± 2.80 | 24.98 ± 2.85 | 13.00 ± 2.08 | 3.45 ± 0.34 | 5.06 ± 0.00 | 5.00 ± 0.00 |
| M-pub | 3 | 38.51 ± 2.74 | 31.77 ± 3.73 | 35.88 ± 2.77 | 27.98 ± 2.33 | 14.98 ± 0.86 | 3.34 ± 0.35 | 5.06 ± 0.00 | 5.00 ± 0.00 |
| M-JL | 3 | 37.01 ± 2.54 | 29.69 ± 2.21 | 31.88 ± 2.59 | 24.71 ± 1.70 | 13.60 ± 0.71 | 3.39 ± 0.44 | 5.06 ± 0.00 | 5.00 ± 0.00 |
| M-fix | 1 | 35.91 | 27.00 | 31.07 | 24.64 | 12.89 | 3.23 | 5.06 | 5.00 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 46.7 | 33.1 | 39.6 | 34.0 | 21.0 |
| SMQ released checkpoint (re-scored, e0) | 36.4 | 38.25 | 40.71 | 32.6 | 22.38 |

### LARa

| arm | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segs/rec | gt segs/rec | occupied codes |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 3 | 36.81 ± 1.26 | 38.76 ± 0.75 | 33.80 ± 0.87 | 27.49 ± 0.89 | 15.82 ± 0.65 | 42.26 ± 1.01 | 32.13 ± 0.00 | 8.00 ± 0.00 |
| M-pub | 3 | 37.03 ± 1.09 | 38.74 ± 1.16 | 34.72 ± 1.09 | 27.83 ± 1.13 | 15.56 ± 0.90 | 39.92 ± 4.36 | 32.13 ± 0.00 | 7.67 ± 0.47 |
| M-JL | 3 | 35.33 ± 2.67 | 38.77 ± 3.39 | 33.73 ± 4.46 | 26.01 ± 4.85 | 13.91 ± 3.78 | 36.78 ± 9.27 | 32.13 ± 0.00 | 7.33 ± 0.94 |

| reference | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| MASQ published (single run) | 47.2 | 42.4 | 42.0 | 35.4 | 22.2 |
| SMQ released checkpoint (re-scored, e0) | 37.38 | 39.4 | 34.69 | 28.4 | 16.39 |
| SMQ retrained, earlier 3 seeds (mean ± SD) | 36.26 ± 1.61 | — | — | — | 15.75 ± 0.75 |

### Per-sequence (local) Hungarian, reported separately and never mixed with the global scores

**BABEL-3**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| S0 | 71.08 ± 0.42 | 48.80 ± 0.35 | 51.94 ± 0.28 | 46.46 ± 0.49 | 33.77 ± 0.51 |
| M-pub | 71.63 ± 3.57 | 45.97 ± 3.09 | 47.45 ± 2.85 | 42.12 ± 3.03 | 29.65 ± 3.43 |
| M-JL | 74.70 ± 4.70 | 44.92 ± 1.85 | 50.01 ± 0.92 | 45.15 ± 0.62 | 34.61 ± 2.66 |
| M-fix | 68.57 ± 0.65 | 46.03 ± 0.56 | 48.93 ± 0.44 | 42.88 ± 0.62 | 30.12 ± 0.31 |

**BABEL-2**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| S0 | 70.09 ± 1.44 | 53.49 ± 1.56 | 60.97 ± 1.87 | 55.55 ± 2.11 | 40.15 ± 2.44 |
| M-pub | 68.51 ± 1.95 | 50.42 ± 2.41 | 57.21 ± 1.02 | 54.04 ± 1.31 | 38.82 ± 1.59 |
| M-JL | 68.64 ± 3.51 | 50.04 ± 3.39 | 58.36 ± 3.24 | 54.33 ± 3.54 | 39.78 ± 4.14 |
| M-fix | 69.73 ± 1.73 | 53.19 ± 1.27 | 58.89 ± 2.05 | 54.54 ± 2.58 | 40.39 ± 2.67 |

**HuGaDB**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| S0 | 54.01 ± 1.57 | 41.45 ± 3.00 | 48.33 ± 2.02 | 39.58 ± 2.00 | 28.92 ± 2.13 |
| M-pub | 57.11 ± 1.31 | 44.22 ± 1.66 | 51.78 ± 1.14 | 43.52 ± 1.47 | 30.47 ± 1.28 |
| M-JL | 61.52 ± 2.21 | 46.78 ± 2.28 | 56.71 ± 1.17 | 49.55 ± 1.55 | 36.07 ± 2.46 |

**BABEL-1**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| S0 | 60.61 ± 0.59 | 44.77 ± 1.29 | 53.55 ± 0.44 | 48.08 ± 0.37 | 29.64 ± 0.97 |
| M-pub | 61.24 ± 0.98 | 45.66 ± 2.82 | 53.29 ± 2.10 | 47.98 ± 2.24 | 29.08 ± 2.09 |
| M-JL | 62.27 ± 1.90 | 45.49 ± 1.56 | 53.53 ± 1.91 | 48.19 ± 2.21 | 29.62 ± 2.78 |
| M-fix | 61.15 | 44.44 | 53.20 | 48.34 | 29.80 |

**LARa**

| arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| S0 | 51.35 ± 0.75 | 44.49 ± 0.70 | 44.60 ± 0.75 | 38.10 ± 0.74 | 23.30 ± 0.47 |
| M-pub | 50.69 ± 1.76 | 45.57 ± 2.13 | 44.58 ± 1.61 | 37.39 ± 1.61 | 22.51 ± 1.08 |
| M-JL | 50.07 ± 1.31 | 44.67 ± 1.49 | 42.70 ± 3.50 | 35.23 ± 3.72 | 20.16 ± 3.31 |

## Comparisons (A minus B, paired by seed; MoF and F1@50 co-primary)

### R1: M-pub minus S0: does MASQ's training change improve SMQ?

- **babel3**: MoF +1.53 (+1.4 / +5.5 / -2.4) -> within the bar: comparable performance, not equivalence; F1@50 -5.27 (-9.6 / -1.2 / -5.0) -> reverse passes the bar
- **babel2**: MoF -1.40 (-8.4 / -0.1 / +4.3) -> within the bar: comparable performance, not equivalence; F1@50 -0.29 (+0.1 / -6.3 / +5.3) -> within the bar: comparable performance, not equivalence
- **hugadb**: MoF -0.98 (-2.0 / +0.5 / -1.4) -> within the bar: comparable performance, not equivalence; F1@50 -0.92 (-1.8 / +1.9 / -2.9) -> within the bar: comparable performance, not equivalence
- **babel1**: MoF +5.56 (+4.6 / +7.0 / +5.1) -> passes the bar; F1@50 +1.99 (+3.4 / +2.3 / +0.3) -> within the bar: comparable performance, not equivalence
- **lara**: MoF +0.22 (+3.2 / -2.5 / -0.0) -> within the bar: comparable performance, not equivalence; F1@50 -0.26 (+0.0 / -1.4 / +0.6) -> within the bar: comparable performance, not equivalence

### R2: M-pub mean vs published MASQ (reproduced if within 3 points on MoF and F1@50)

- **babel3**: ours MoF 39.7 / F1@50 12.0; published 47.7 / 24.5 -> not reproduced
- **babel2**: ours MoF 38.5 / F1@50 17.5; published 55.8 / 28.4 -> not reproduced
- **hugadb**: ours MoF 39.7 / F1@50 21.1; published 52.3 / 39.5 -> not reproduced
- **babel1**: ours MoF 38.5 / F1@50 15.0; published 46.7 / 21.0 -> not reproduced
- **lara**: ours MoF 37.0 / F1@50 15.6; published 47.2 / 22.2 -> not reproduced

### R3: S0 (our retrained SMQ) vs the references

- **babel3**: S0 MoF 38.14 ± 0.43, F1@50 17.23 ± 3.85; released SMQ 40.4 / 19.4
- **babel2**: S0 MoF 39.86 ± 3.67, F1@50 17.81 ± 0.80; released SMQ 49.2 / 27.5
- **hugadb**: S0 MoF 40.72 ± 1.54, F1@50 22.04 ± 1.48; released SMQ 42.0 / 24.3; G5 passed {'MoF': True, 'F1@50': True}
- **babel1**: S0 MoF 32.95 ± 1.76, F1@50 13.00 ± 2.08; released SMQ 36.4 / 22.4
- **lara**: S0 MoF 36.81 ± 1.26, F1@50 15.82 ± 0.65; released SMQ 37.4 / 16.4; G5 passed {'MoF': True, 'F1@50': True}

### R4a: M-pub minus M-JL: the velocity loss's contribution

- **babel3**: MoF -1.40 (+6.2 / -2.8 / -7.6) -> within the bar: comparable performance, not equivalence; F1@50 -7.62 (-6.8 / -10.5 / -5.5) -> reverse passes the bar
- **babel2**: MoF -0.46 (-2.5 / -11.4 / +12.4) -> within the bar: comparable performance, not equivalence; F1@50 -1.12 (-5.0 / -5.3 / +6.9) -> within the bar: comparable performance, not equivalence
- **hugadb**: MoF -5.50 (-4.3 / -3.4 / -8.9) -> reverse passes the bar; F1@50 -6.44 (-4.8 / -2.6 / -11.9) -> reverse passes the bar
- **babel1**: MoF +1.50 (+0.5 / +6.1 / -2.2) -> within the bar: comparable performance, not equivalence; F1@50 +1.38 (+0.0 / +1.5 / +2.6) -> within the bar: comparable performance, not equivalence
- **lara**: MoF +1.69 (+3.9 / +2.9 / -1.8) -> within the bar: comparable performance, not equivalence; F1@50 +1.65 (+6.2 / -0.3 / -0.9) -> within the bar: comparable performance, not equivalence

### R4b: M-fix minus M-pub: does the published per-dataset p matter?

- **babel3**: MoF -1.79 (-1.7 / -5.1 / +1.4) -> within the bar: comparable performance, not equivalence; F1@50 +2.53 (+5.9 / +0.1 / +1.5) -> within the bar: comparable performance, not equivalence
- **babel2**: MoF -1.12 (+0.7 / +9.0 / -13.0) -> within the bar: comparable performance, not equivalence; F1@50 -0.09 (-1.9 / +6.6 / -5.0) -> within the bar: comparable performance, not equivalence
- **hugadb**: MoF — -> no cells; F1@50 — -> no cells
- **babel1**: MoF +0.69 (+0.7) -> incomplete (1 seed); F1@50 -1.61 (-1.6) -> incomplete (1 seed)
- **lara**: MoF — -> no cells; F1@50 — -> no cells

## Every metric

| id | dataset | A − B | metric | Δ (per seed) | n | reading |
|---|---|---|---|---|---|---|
| R1 | babel3 | M-pub - S0 | MoF | +1.53 (+1.4 / +5.5 / -2.4) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel3 | M-pub - S0 | Edit | -7.51 (-21.0 / -0.2 / -1.3) | 3 | reverse passes the bar |
| R1 | babel3 | M-pub - S0 | F1@10 | -8.31 (-21.1 / -0.8 / -3.1) | 3 | reverse passes the bar |
| R1 | babel3 | M-pub - S0 | F1@25 | -6.08 (-15.0 / +0.9 / -4.2) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel3 | M-pub - S0 | F1@50 | -5.27 (-9.6 / -1.2 / -5.0) | 3 | reverse passes the bar |
| R1 | babel2 | M-pub - S0 | MoF | -1.40 (-8.4 / -0.1 / +4.3) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel2 | M-pub - S0 | Edit | -1.55 (-2.4 / -3.3 / +1.1) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel2 | M-pub - S0 | F1@10 | -2.80 (-3.3 / -5.7 / +0.6) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel2 | M-pub - S0 | F1@25 | -2.13 (-1.6 / -7.5 / +2.7) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel2 | M-pub - S0 | F1@50 | -0.29 (+0.1 / -6.3 / +5.3) | 3 | within the bar: comparable performance, not equivalence |
| R1 | hugadb | M-pub - S0 | MoF | -0.98 (-2.0 / +0.5 / -1.4) | 3 | within the bar: comparable performance, not equivalence |
| R1 | hugadb | M-pub - S0 | Edit | +1.64 (+0.7 / +2.2 / +1.9) | 3 | within the bar: comparable performance, not equivalence |
| R1 | hugadb | M-pub - S0 | F1@10 | +2.12 (+0.7 / +1.8 / +3.8) | 3 | within the bar: comparable performance, not equivalence |
| R1 | hugadb | M-pub - S0 | F1@25 | +1.28 (-0.0 / +2.7 / +1.1) | 3 | within the bar: comparable performance, not equivalence |
| R1 | hugadb | M-pub - S0 | F1@50 | -0.92 (-1.8 / +1.9 / -2.9) | 3 | within the bar: comparable performance, not equivalence |
| R1 | babel1 | M-pub - S0 | MoF | +5.56 (+4.6 / +7.0 / +5.1) | 3 | passes the bar |
| R1 | babel1 | M-pub - S0 | Edit | +4.57 (+2.8 / +2.2 / +8.7) | 3 | passes the bar |
| R1 | babel1 | M-pub - S0 | F1@10 | +3.92 (+5.0 / +2.8 / +3.9) | 3 | passes the bar |
| R1 | babel1 | M-pub - S0 | F1@25 | +3.00 (+4.5 / +1.9 / +2.6) | 3 | passes the bar |
| R1 | babel1 | M-pub - S0 | F1@50 | +1.99 (+3.4 / +2.3 / +0.3) | 3 | within the bar: comparable performance, not equivalence |
| R1 | lara | M-pub - S0 | MoF | +0.22 (+3.2 / -2.5 / -0.0) | 3 | within the bar: comparable performance, not equivalence |
| R1 | lara | M-pub - S0 | Edit | -0.01 (+0.8 / -2.6 / +1.7) | 3 | within the bar: comparable performance, not equivalence |
| R1 | lara | M-pub - S0 | F1@10 | +0.92 (+2.5 / -1.3 / +1.5) | 3 | within the bar: comparable performance, not equivalence |
| R1 | lara | M-pub - S0 | F1@25 | +0.34 (+1.7 / -1.9 / +1.2) | 3 | within the bar: comparable performance, not equivalence |
| R1 | lara | M-pub - S0 | F1@50 | -0.26 (+0.0 / -1.4 / +0.6) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel3 | M-pub - M-JL | MoF | -1.40 (+6.2 / -2.8 / -7.6) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel3 | M-pub - M-JL | Edit | -4.41 (-15.2 / -5.4 / +7.4) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel3 | M-pub - M-JL | F1@10 | -5.38 (-16.6 / -4.4 / +4.9) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel3 | M-pub - M-JL | F1@25 | -6.51 (-11.0 / -7.3 / -1.2) | 3 | reverse passes the bar |
| R4a | babel3 | M-pub - M-JL | F1@50 | -7.62 (-6.8 / -10.5 / -5.5) | 3 | reverse passes the bar |
| R4a | babel2 | M-pub - M-JL | MoF | -0.46 (-2.5 / -11.4 / +12.4) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel2 | M-pub - M-JL | Edit | +2.21 (-5.0 / +0.9 / +10.8) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel2 | M-pub - M-JL | F1@10 | -0.62 (-10.7 / -0.8 / +9.6) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel2 | M-pub - M-JL | F1@25 | -2.18 (-9.0 / -5.4 / +7.9) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel2 | M-pub - M-JL | F1@50 | -1.12 (-5.0 / -5.3 / +6.9) | 3 | within the bar: comparable performance, not equivalence |
| R4a | hugadb | M-pub - M-JL | MoF | -5.50 (-4.3 / -3.4 / -8.9) | 3 | reverse passes the bar |
| R4a | hugadb | M-pub - M-JL | Edit | -2.95 (+0.4 / -2.7 / -6.5) | 3 | within the bar: comparable performance, not equivalence |
| R4a | hugadb | M-pub - M-JL | F1@10 | -5.70 (-3.4 / -6.6 / -7.1) | 3 | reverse passes the bar |
| R4a | hugadb | M-pub - M-JL | F1@25 | -6.77 (-5.1 / -5.3 / -10.0) | 3 | reverse passes the bar |
| R4a | hugadb | M-pub - M-JL | F1@50 | -6.44 (-4.8 / -2.6 / -11.9) | 3 | reverse passes the bar |
| R4a | babel1 | M-pub - M-JL | MoF | +1.50 (+0.5 / +6.1 / -2.2) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel1 | M-pub - M-JL | Edit | +2.07 (-1.6 / -2.7 / +10.5) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel1 | M-pub - M-JL | F1@10 | +4.00 (+3.9 / -2.0 / +10.1) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel1 | M-pub - M-JL | F1@25 | +3.26 (+2.3 / -0.7 / +8.2) | 3 | within the bar: comparable performance, not equivalence |
| R4a | babel1 | M-pub - M-JL | F1@50 | +1.38 (+0.0 / +1.5 / +2.6) | 3 | within the bar: comparable performance, not equivalence |
| R4a | lara | M-pub - M-JL | MoF | +1.69 (+3.9 / +2.9 / -1.8) | 3 | within the bar: comparable performance, not equivalence |
| R4a | lara | M-pub - M-JL | Edit | -0.03 (+4.9 / -2.8 / -2.1) | 3 | within the bar: comparable performance, not equivalence |
| R4a | lara | M-pub - M-JL | F1@10 | +0.99 (+7.6 / -2.6 / -2.1) | 3 | within the bar: comparable performance, not equivalence |
| R4a | lara | M-pub - M-JL | F1@25 | +1.82 (+8.6 / -1.5 / -1.6) | 3 | within the bar: comparable performance, not equivalence |
| R4a | lara | M-pub - M-JL | F1@50 | +1.65 (+6.2 / -0.3 / -0.9) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel3 | M-fix - M-pub | MoF | -1.79 (-1.7 / -5.1 / +1.4) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel3 | M-fix - M-pub | Edit | +6.08 (+15.8 / +1.2 / +1.2) | 3 | passes the bar |
| R4b | babel3 | M-fix - M-pub | F1@10 | +7.01 (+18.7 / -0.7 / +3.0) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel3 | M-fix - M-pub | F1@25 | +3.48 (+11.1 / -2.5 / +1.8) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel3 | M-fix - M-pub | F1@50 | +2.53 (+5.9 / +0.1 / +1.5) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel2 | M-fix - M-pub | MoF | -1.12 (+0.7 / +9.0 / -13.0) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel2 | M-fix - M-pub | Edit | +0.98 (+4.0 / +2.8 / -3.8) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel2 | M-fix - M-pub | F1@10 | +0.91 (+5.0 / +0.1 / -2.4) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel2 | M-fix - M-pub | F1@25 | +0.67 (+0.6 / +4.9 / -3.5) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel2 | M-fix - M-pub | F1@50 | -0.09 (-1.9 / +6.6 / -5.0) | 3 | within the bar: comparable performance, not equivalence |
| R4b | babel1 | M-fix - M-pub | MoF | +0.69 (+0.7) | 1 | incomplete (1 seed) |
| R4b | babel1 | M-fix - M-pub | Edit | -2.59 (-2.6) | 1 | incomplete (1 seed) |
| R4b | babel1 | M-fix - M-pub | F1@10 | -3.29 (-3.3) | 1 | incomplete (1 seed) |
| R4b | babel1 | M-fix - M-pub | F1@25 | -1.72 (-1.7) | 1 | incomplete (1 seed) |
| R4b | babel1 | M-fix - M-pub | F1@50 | -1.61 (-1.6) | 1 | incomplete (1 seed) |

## R5: identity diagnostics (descriptive; subject only for HuGaDB and LARa)

Code AMI = chance-corrected mutual information of native codes with the metadata. Latent NN share = same-recording share of nearest neighbours of patch-mean latents (adjacent patches excluded). The per-class block uses labels (analysis only).

| dataset | arm | n | recording AMI | subject AMI | per-class subject AMI (labels) | latent NN share |
|---|---|---|---|---|---|---|
| babel3 | S0 | 3 | 0.23 ± 0.01 | — | — | 0.44 ± 0.00 |
| babel3 | M-pub | 3 | 0.15 ± 0.05 | — | — | 0.47 ± 0.01 |
| babel3 | M-JL | 3 | 0.11 ± 0.08 | — | — | 0.49 ± 0.00 |
| babel3 | M-fix | 3 | 0.21 ± 0.01 | — | — | 0.46 ± 0.01 |
| babel2 | S0 | 3 | 0.20 ± 0.03 | — | — | 0.48 ± 0.00 |
| babel2 | M-pub | 3 | 0.22 ± 0.02 | — | — | 0.48 ± 0.01 |
| babel2 | M-JL | 3 | 0.21 ± 0.02 | — | — | 0.46 ± 0.02 |
| babel2 | M-fix | 3 | 0.22 ± 0.02 | — | — | 0.47 ± 0.00 |
| hugadb | S0 | 3 | 0.26 ± 0.00 | 0.08 ± 0.00 | 0.11 ± 0.00 | 0.45 ± 0.03 |
| hugadb | M-pub | 3 | 0.25 ± 0.02 | 0.11 ± 0.03 | 0.18 ± 0.07 | 0.50 ± 0.00 |
| hugadb | M-JL | 3 | 0.26 ± 0.02 | 0.10 ± 0.02 | 0.18 ± 0.02 | 0.37 ± 0.01 |
| babel1 | S0 | 3 | 0.14 ± 0.01 | — | — | 0.24 ± 0.00 |
| babel1 | M-pub | 3 | 0.11 ± 0.01 | — | — | 0.22 ± 0.01 |
| babel1 | M-JL | 3 | 0.14 ± 0.01 | — | — | 0.24 ± 0.00 |
| babel1 | M-fix | 1 | 0.14 | — | — | 0.23 |
| lara | S0 | 3 | 0.08 ± 0.00 | 0.05 ± 0.00 | 0.09 ± 0.00 | 0.39 ± 0.01 |
| lara | M-pub | 3 | 0.07 ± 0.01 | 0.05 ± 0.00 | 0.08 ± 0.01 | 0.45 ± 0.02 |
| lara | M-JL | 3 | 0.06 ± 0.02 | 0.04 ± 0.01 | 0.07 ± 0.01 | 0.44 ± 0.01 |

M-pub minus S0 (per seed):

| dataset | measure | Δ | n |
|---|---|---|---|
| babel3 | recording_ami | -0.08 (-0.1 / -0.1 / -0.0) | 3 |
| babel3 | latent_same_recording_nn_share | +0.03 (+0.0 / +0.0 / +0.0) | 3 |
| babel2 | recording_ami | +0.02 (+0.1 / -0.0 / -0.0) | 3 |
| babel2 | latent_same_recording_nn_share | -0.00 (+0.0 / -0.0 / +0.0) | 3 |
| hugadb | recording_ami | -0.01 (-0.0 / -0.0 / +0.0) | 3 |
| hugadb | subject_ami | +0.03 (+0.0 / -0.0 / +0.1) | 3 |
| hugadb | latent_same_recording_nn_share | +0.05 (+0.0 / +0.0 / +0.1) | 3 |
| babel1 | recording_ami | -0.03 (-0.0 / -0.0 / -0.0) | 3 |
| babel1 | latent_same_recording_nn_share | -0.02 (-0.0 / -0.0 / -0.0) | 3 |
| lara | recording_ami | -0.01 (-0.0 / -0.0 / -0.0) | 3 |
| lara | subject_ami | -0.00 (-0.0 / -0.0 / +0.0) | 3 |
| lara | latent_same_recording_nn_share | +0.06 (+0.1 / +0.0 / +0.0) | 3 |

## Training (final epoch losses)

| dataset | arm | seed | rec | commit | vel | dead-code replacements | EMA codes in use | seconds |
|---|---|---|---|---|---|---|---|---|
| babel1 | M-JL | 111 | 12.13989 | 2.38246 | 0.00000 | 296 | 5 | 3055 |
| babel1 | M-JL | 222 | 10.42180 | 3.21196 | 0.00000 | 339 | 5 | 4277 |
| babel1 | M-JL | 1538574472 | 17.98683 | 3.15321 | 0.00000 | 268 | 5 | 3188 |
| babel1 | M-fix | 111 | 11.01271 | 3.02639 | 1.44593 | 191 | 5 | 2996 |
| babel1 | M-pub | 111 | 22.50012 | 3.82847 | 1.05860 | 419 | 5 | 2791 |
| babel1 | M-pub | 222 | 25.48686 | 4.35607 | 1.27244 | 324 | 5 | 3744 |
| babel1 | M-pub | 1538574472 | 20.10565 | 6.13104 | 1.71853 | 291 | 5 | 2984 |
| babel1 | S0 | 111 | 7.25349 | 3.20058 | 0.00000 | 149 | 5 | 2698 |
| babel1 | S0 | 222 | 6.64746 | 2.80782 | 0.00000 | 151 | 5 | 3738 |
| babel1 | S0 | 1538574472 | 7.36675 | 3.05362 | 0.00000 | 130 | 5 | 2971 |
| babel2 | M-JL | 111 | 28.41092 | 11.04867 | 0.00000 | 117 | 5 | 680 |
| babel2 | M-JL | 222 | 28.40422 | 7.74367 | 0.00000 | 117 | 5 | 510 |
| babel2 | M-JL | 1538574472 | 35.40856 | 8.11325 | 0.00000 | 110 | 5 | 483 |
| babel2 | M-fix | 111 | 28.21053 | 6.83150 | 2.32952 | 42 | 5 | 722 |
| babel2 | M-fix | 222 | 33.94931 | 8.14941 | 2.26790 | 78 | 5 | 516 |
| babel2 | M-fix | 1538574472 | 29.78639 | 6.70936 | 2.23474 | 58 | 5 | 491 |
| babel2 | M-pub | 111 | 45.32888 | 9.74784 | 3.03130 | 200 | 5 | 668 |
| babel2 | M-pub | 222 | 36.32831 | 8.54448 | 4.86565 | 169 | 5 | 484 |
| babel2 | M-pub | 1538574472 | 32.57576 | 7.18411 | 3.19453 | 141 | 5 | 483 |
| babel2 | S0 | 111 | 18.24174 | 6.11259 | 0.00000 | 32 | 5 | 693 |
| babel2 | S0 | 222 | 13.26053 | 6.19512 | 0.00000 | 36 | 5 | 503 |
| babel2 | S0 | 1538574472 | 18.26015 | 4.57722 | 0.00000 | 35 | 5 | 486 |
| babel3 | M-JL | 111 | 63.11108 | 27.30859 | 0.00000 | 110 | 5 | 62 |
| babel3 | M-JL | 222 | 60.05113 | 11.46817 | 0.00000 | 86 | 5 | 212 |
| babel3 | M-JL | 1538574472 | 78.31562 | 25.24644 | 0.00000 | 77 | 5 | 180 |
| babel3 | M-fix | 111 | 57.15897 | 15.14871 | 4.06329 | 58 | 5 | 63 |
| babel3 | M-fix | 222 | 46.35668 | 14.86879 | 3.82840 | 44 | 5 | 193 |
| babel3 | M-fix | 1538574472 | 50.87634 | 12.36212 | 5.04439 | 41 | 5 | 191 |
| babel3 | M-pub | 111 | 80.90600 | 15.91078 | 5.15226 | 133 | 5 | 61 |
| babel3 | M-pub | 222 | 78.74551 | 14.44072 | 5.52842 | 99 | 5 | 181 |
| babel3 | M-pub | 1538574472 | 79.83703 | 15.99458 | 5.35931 | 96 | 5 | 182 |
| babel3 | S0 | 111 | 30.18574 | 6.86081 | 0.00000 | 32 | 5 | 62 |
| babel3 | S0 | 222 | 31.00322 | 7.50702 | 0.00000 | 16 | 5 | 208 |
| babel3 | S0 | 1538574472 | 29.30026 | 6.43230 | 0.00000 | 31 | 5 | 182 |
| hugadb | M-JL | 111 | 89399.44056 | 26840.55346 | 0.00000 | 1568 | 10 | 440 |
| hugadb | M-JL | 222 | 90335.20440 | 22795.25231 | 0.00000 | 1534 | 10 | 470 |
| hugadb | M-JL | 1538574472 | 89891.44663 | 24151.06262 | 0.00000 | 1491 | 10 | 167 |
| hugadb | M-pub | 111 | 91569.54322 | 30345.43911 | 173048.98157 | 1588 | 10 | 445 |
| hugadb | M-pub | 222 | 99705.42281 | 28196.29021 | 177981.40039 | 1578 | 10 | 446 |
| hugadb | M-pub | 1538574472 | 94828.29721 | 33193.19202 | 176691.65018 | 1591 | 10 | 168 |
| hugadb | S0 | 111 | 52121.99002 | 18954.29059 | 0.00000 | 821 | 10 | 444 |
| hugadb | S0 | 222 | 54597.52293 | 18723.34789 | 0.00000 | 824 | 10 | 443 |
| hugadb | S0 | 1538574472 | 61955.07632 | 17151.57221 | 0.00000 | 823 | 10 | 167 |
| lara | M-JL | 111 | 56.91555 | 4.67602 | 0.00000 | 390 | 8 | 791 |
| lara | M-JL | 222 | 20.28394 | 3.61951 | 0.00000 | 276 | 8 | 786 |
| lara | M-JL | 1538574472 | 29.77443 | 3.69258 | 0.00000 | 229 | 8 | 784 |
| lara | M-pub | 111 | 34.55433 | 4.95887 | 1.32303 | 235 | 8 | 781 |
| lara | M-pub | 222 | 32.30524 | 5.07578 | 1.41589 | 186 | 8 | 784 |
| lara | M-pub | 1538574472 | 31.85719 | 4.92730 | 1.34141 | 197 | 8 | 785 |
| lara | S0 | 111 | 16.15078 | 5.59655 | 0.00000 | 89 | 8 | 786 |
| lara | S0 | 222 | 15.83049 | 5.47862 | 0.00000 | 76 | 8 | 779 |
| lara | S0 | 1538574472 | 16.08958 | 5.61438 | 0.00000 | 70 | 8 | 782 |

## Verdict (computed from the declared rule)

**partly replicated.** R1 passes on MoF in 1 of 5 datasets (babel1); R2 reproduced on 0.

Rule: replicated = R1 passes on MoF in >= 3 of 5 datasets including LARa and R2 holds somewhere; partly = 1-2 passes, or >= 3 passes without LARa, or the replicated pattern with R2 failing everywhere; not replicated = no pass.

## Caveats

All results here are *development* results (protocol P7): every dataset informed earlier decisions and no untouched evaluation set exists. The published MASQ numbers are single runs from a paper that is not peer reviewed; our M arms are a reimplementation from the paper text with the SMQ backbone and defaults, so a gap to the published numbers may come from undocumented details. These experiments cannot show *why* MASQ works, only whether its training changes reproduce and which component matters; they say nothing about our OT decoder on MASQ latents (Tier 1).
