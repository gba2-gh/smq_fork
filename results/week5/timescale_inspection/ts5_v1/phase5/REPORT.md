# Time-scale inspection, phase 5: the tokenizer — ts5_v1

Status `finished`. Radius `current`, scale `full`. Baselines are read, not rebuilt: T0 `context` from `rkx_v2`, T0 `centre`/`pca` from `ts1_v1` Phase 1 `inputs`. The primary number is F1@50 (selected checkpoint, last rung); the bar is ≥ 3 points on the mean with all 3 seeds in the same direction. T4's `context` arm is trained with guard 14 / span 16 and is not comparable with T0's.

## Gates

| gate | status | detail |
|---|---|---|
| g1 | passed | checks {'same_recordings': True, 'codes': True, 'continuous': True, 'starts_lengths': True, 'lookup': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True} |
| g2_babel2 | passed | MoF 0.0000, Edit 0.0000, F1@10 0.0000, F1@25 0.0000, F1@50 0.0000 |
| g2_babel3 | passed | MoF 0.0000, Edit 0.0000, F1@10 0.0000, F1@25 0.0000, F1@50 0.0000 |
| g2_hugadb | passed | MoF 0.0000, Edit 0.0000, F1@10 0.0000, F1@25 0.0000, F1@50 0.0000 |
| g2_lara | passed | MoF 0.0000, Edit 0.0000, F1@10 0.0000, F1@25 0.0000, F1@50 0.0000 |
| g3 | passed | median across-joint extent: channels 3-5 609.889 vs 0-2 315.753 |

## F1@50 by dataset

### babel2

| tokenizer | arm | F1@50 per seed (111 / 222 / 1538574472) | mean ± sd | Δ vs T0, same arm (paired) | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|
| T0 raw (baseline) | context | 16.68 / 11.12 / 17.46 | 15.09 ± 3.46 | — | — | — |
| T0 raw (baseline) | centre | 20.10 / 18.57 / 20.30 | 19.66 ± 0.95 | — | — | — |
| T0 raw (baseline) | pca | 19.06 / 17.99 / 19.09 | 18.71 ± 0.63 | — | — | — |
| T1 whiten | context | 15.37 / 18.75 / 13.13 | 15.75 ± 2.83 | +0.66 (-1.3 / +7.6 / -4.3) | 2.0 | 3.7 |
| T1 whiten | centre | 18.08 / 17.72 / 17.29 | 17.70 ± 0.40 | -1.96 (-2.0 / -0.9 / -3.0) | 3.1 | 3.7 |
| T1 whiten | pca | 14.43 / 22.30 / 22.41 | 19.71 ± 4.58 | +1.00 (-4.6 / +4.3 / +3.3) | 2.9 | 3.7 |
| T2 relational | context | 16.33 / 26.16 / 21.75 | 21.41 ± 4.92 | +6.33 (-0.4 / +15.0 / +4.3) | 2.2 | 3.7 |
| T2 relational | centre | 30.63 / 18.77 / 30.75 (early stop: seed(s) 222) | 26.72 ± 6.88 | +7.06 (+10.5 / +0.2 / +10.4) | 5.3 | 3.7 |
| T2 relational | pca | 26.27 / 18.29 / 30.17 (early stop: seed(s) 222) | 24.91 ± 6.06 | +6.20 (+7.2 / +0.3 / +11.1) | 5.0 | 3.7 |
| T4 smq | context (guard 14, span 16: not comparable with T0) | 11.55 / 15.68 / 14.12 | 13.78 ± 2.09 | n/a | 1.9 | 3.7 |
| T4 smq | centre | 12.74 / 16.09 / 12.98 | 13.94 ± 1.87 | -5.72 (-7.4 / -2.5 / -7.3) | 4.0 | 3.7 |
| T4 smq | pca | 16.43 / 15.66 / 12.54 | 14.88 ± 2.06 | -3.84 (-2.6 / -2.3 / -6.6) | 3.7 | 3.7 |

### babel3

| tokenizer | arm | F1@50 per seed (111 / 222 / 1538574472) | mean ± sd | Δ vs T0, same arm (paired) | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|
| T0 raw (baseline) | context | 14.78 / 16.04 / 15.15 | 15.32 ± 0.65 | — | — | — |
| T0 raw (baseline) | centre | 17.92 / 17.62 / 17.11 | 17.55 ± 0.41 | — | — | — |
| T0 raw (baseline) | pca | 17.10 / 17.06 / 18.17 | 17.44 ± 0.63 | — | — | — |
| T1 whiten | context | 23.04 / 13.41 / 12.71 | 16.39 ± 5.77 | +1.06 (+8.3 / -2.6 / -2.4) | 2.7 | 3.7 |
| T1 whiten | centre | 21.55 / 27.53 / 26.68 | 25.25 ± 3.24 | +7.70 (+3.6 / +9.9 / +9.6) | 4.6 | 3.7 |
| T1 whiten | pca | 23.55 / 24.28 / 30.30 | 26.04 ± 3.70 | +8.60 (+6.4 / +7.2 / +12.1) | 3.7 | 3.7 |
| T2 relational | context | 19.02 / 18.40 / 20.58 | 19.33 ± 1.12 | +4.01 (+4.2 / +2.4 / +5.4) | 2.6 | 3.7 |
| T2 relational | centre | 20.08 / 21.95 / 21.80 | 21.28 ± 1.04 | +3.73 (+2.2 / +4.3 / +4.7) | 8.1 | 3.7 |
| T2 relational | pca | 23.85 / 24.81 / 25.87 | 24.84 ± 1.01 | +7.40 (+6.8 / +7.8 / +7.7) | 6.3 | 3.7 |
| T4 smq | context (guard 14, span 16: not comparable with T0) | 14.29 / 16.29 / 12.58 | 14.39 ± 1.86 | n/a | 2.6 | 3.7 |
| T4 smq | centre | 8.91 / 8.93 / 18.10 (early stop: seed(s) 111, 222) | 11.98 ± 5.30 | -5.57 (-9.0 / -8.7 / +1.0) | 5.8 | 3.7 |
| T4 smq | pca | 9.90 / 9.86 / 15.46 (early stop: seed(s) 111, 222) | 11.74 ± 3.22 | -5.70 (-7.2 / -7.2 / -2.7) | 5.4 | 3.7 |

### lara

| tokenizer | arm | F1@50 per seed (111 / 222 / 1538574472) | mean ± sd | Δ vs T0, same arm (paired) | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|
| T0 raw (baseline) | context | 5.03 / 4.60 / 5.55 | 5.06 ± 0.48 | — | — | — |
| T0 raw (baseline) | centre | 10.30 / 9.94 / 10.28 | 10.17 ± 0.20 | — | — | — |
| T0 raw (baseline) | pca | 10.16 / 9.76 / 9.84 | 9.92 ± 0.21 | — | — | — |
| T2 relational | context | 17.99 / 18.70 / 20.59 | 19.09 ± 1.34 | +14.03 (+13.0 / +14.1 / +15.0) | 21.2 | 32.1 |
| T2 relational | centre | 18.39 / 16.38 / 18.64 | 17.80 ± 1.24 | +7.63 (+8.1 / +6.4 / +8.4) | 64.2 | 32.1 |
| T3 lara_pos256 | context | 11.14 / 10.83 / 9.83 | 10.60 ± 0.68 | +5.54 (+6.1 / +6.2 / +4.3) | 22.1 | 32.1 |
| T3 lara_pos256 | centre | 11.38 / 11.85 / 11.97 | 11.73 ± 0.31 | +1.56 (+1.1 / +1.9 / +1.7) | 30.2 | 32.1 |
| T3 lara_pos256 | pca | 10.76 / 11.83 / 11.67 | 11.42 ± 0.58 | +1.50 (+0.6 / +2.1 / +1.8) | 27.6 | 32.1 |
| T3b lara_pos64 | context | 10.21 / 11.06 / 11.40 | 10.89 ± 0.61 | +5.83 (+5.2 / +6.5 / +5.9) | 22.3 | 32.1 |
| T3b lara_pos64 | centre | 10.95 / 11.80 / 11.64 | 11.46 ± 0.45 | +1.29 (+0.6 / +1.9 / +1.4) | 30.6 | 32.1 |
| T4 smq | context (guard 14, span 16: not comparable with T0) | 4.78 / 4.57 / 4.31 | 4.55 ± 0.24 | n/a | 13.4 | 32.1 |
| T4 smq | centre | 14.39 / 18.45 / 15.93 | 16.26 ± 2.05 | +6.08 (+4.1 / +8.5 / +5.7) | 28.9 | 32.1 |

### hugadb

| tokenizer | arm | F1@50 per seed (111 / 222 / 1538574472) | mean ± sd | Δ vs T0, same arm (paired) | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|
| T0 raw (baseline) | context | 50.00 / 44.83 / 43.85 | 46.23 ± 3.30 | — | — | — |
| T0 raw (baseline) | centre | 3.95 / 3.96 / 3.98 | 3.96 ± 0.02 | — | — | — |
| T0 raw (baseline) | pca | 3.97 / 3.97 / 3.97 | 3.97 ± 0.00 | — | — | — |
| T1 whiten | context | 52.66 / n/a / n/a | 52.66 ± 0.00 | n/a | 8.9 | 10.5 |
| T1 whiten | centre | 8.36 / n/a / n/a | 8.36 ± 0.00 | n/a | 60.4 | 10.5 |
| T4 smq | context (guard 14, span 16: not comparable with T0) | 23.19 / n/a / n/a | 23.19 ± 0.00 | n/a | 7.6 | 10.5 |
| T4 smq | centre | 3.26 / n/a / n/a | 3.26 ± 0.00 | n/a | 67.1 | 10.5 |

### HuGaDB transfer check (descriptive, seed 111 only)

- T1 `context` 52.66 vs T0 50.00: **ok** (flag below −6)
- T4 `context` 23.19 vs T0 50.00: **FLAGGED** (flag below −6) — different masking

## H6: the tokenizer limits skeleton datasets (F1@50 alone decides)

- babel2 / T1 centre: Δ vs T0 -1.96 (-2.0 / -0.9 / -3.0) -> **does not beat T0 by the bar**
- babel2 / T1 context: Δ vs T0 +0.66 (-1.3 / +7.6 / -4.3) -> **does not beat T0 by the bar**
- babel2 / T2 centre: Δ vs T0 +7.06 (+10.5 / +0.2 / +10.4) -> **passes, fragile (smallest per-seed gain < 1 point)**
- babel2 / T2 context: Δ vs T0 +6.33 (-0.4 / +15.0 / +4.3) -> **does not beat T0 by the bar**
- babel2 / T4 centre: Δ vs T0 -5.72 (-7.4 / -2.5 / -7.3) -> **does not beat T0 by the bar**
- lara / T2 centre: Δ vs T0 +7.63 (+8.1 / +6.4 / +8.4) -> **beats T0 by the bar**
- lara / T2 context: Δ vs T0 +14.03 (+13.0 / +14.1 / +15.0) -> **beats T0 by the bar**
- lara / T3 centre: Δ vs T0 +1.56 (+1.1 / +1.9 / +1.7) -> **does not beat T0 by the bar**
- lara / T3 context: Δ vs T0 +5.54 (+6.1 / +6.2 / +4.3) -> **beats T0 by the bar**
- lara / T3b centre: Δ vs T0 +1.29 (+0.6 / +1.9 / +1.4) -> **does not beat T0 by the bar**
- lara / T3b context: Δ vs T0 +5.83 (+5.2 / +6.5 / +5.9) -> **beats T0 by the bar**
- lara / T4 centre: Δ vs T0 +6.08 (+4.1 / +8.5 / +5.7) -> **beats T0 by the bar**

**H6: supported by babel2/T2, lara/T2, lara/T3, lara/T3b, lara/T4.**

## H7: was the contextualizer's failure conditional on the tokens?

- babel2 / T1: context − centre = -1.95 (-2.7 / +1.0 / -4.2) -> **neutral**; T0's gap -4.57; moved by +2.62
- babel2 / T2: context − centre = -5.30 (-14.3 / +7.4 / -9.0) -> **neutral**; T0's gap -4.57; moved by -0.73
- babel3 / T1: context − centre = -8.87 (+1.5 / -14.1 / -14.0) -> **neutral**; T0's gap -2.23; moved by -6.64
- babel3 / T2: context − centre = -1.94 (-1.1 / -3.6 / -1.2) -> **neutral**; T0's gap -2.23; moved by +0.28
- lara / T2: context − centre = +1.29 (-0.4 / +2.3 / +1.9) -> **neutral**; T0's gap -5.11; moved by +6.40 -> **H7 supported on this tokenizer** (it also satisfies H6)
- lara / T3: context − centre = -1.13 (-0.2 / -1.0 / -2.1) -> **neutral**; T0's gap -5.11; moved by +3.98 -> **H7 supported on this tokenizer** (it also satisfies H6)
- lara / T3b: context − centre = -0.57 (-0.7 / -0.7 / -0.2) -> **neutral**; T0's gap -5.11; moved by +4.54 -> **H7 supported on this tokenizer** (it also satisfies H6)
- babel2 / T4 (descriptive; other masking): context − centre = -0.15 over 3 seed(s)
- babel3 / T4 (descriptive; other masking): context − centre = +2.41 over 3 seed(s)
- lara / T4 (descriptive; other masking): context − centre = -11.70 over 3 seed(s)
- hugadb / T4 (descriptive; other masking): context − centre = +19.93 over 1 seed(s)

## LARa: what T3 and T3b separate

- LARa / context: truncation effect (T3 − T3b) -0.29 (+0.9 / -0.2 / -1.6); orientation effect (T3b − T0) +5.83 (+5.2 / +6.5 / +5.9)
- LARa / centre: truncation effect (T3 − T3b) +0.27 (+0.4 / +0.0 / +0.3); orientation effect (T3b − T0) +1.29 (+0.6 / +1.9 / +1.4)

## Per-token probes on the PCA feature (balanced accuracy, T0 → new; mechanism label, not part of H6)

For T1 (whitening), the linear probe and the whitened 1-NN probe are invariant to PCA whitening by construction -- both standardize the feature before fitting, so whitening the PCA space ahead of that changes nothing. Only the plain 1-NN probe can move for T1, and T1's label is computed from it alone (the table still shows all three; the invariant two read unchanged, not 'risen').

| dataset | tokenizer | linear | 1-NN | 1-NN whitened | label | seeds probed |
|---|---|---|---|---|---|---|
| babel2 | T1 | 0.422 → 0.422 | 0.520 → 0.628 | 0.627 → 0.627 | some rise (1/3) | 3 |
| babel2 | T2 | 0.422 → 0.552 | 0.520 → 0.586 | 0.627 → 0.549 | some rise (2/3) | 3 |
| babel2 | T4 | 0.422 → 0.483 | 0.520 → 0.498 | 0.627 → 0.575 | some rise (1/3) | 3 |
| babel3 | T1 | 0.379 → 0.379 | 0.420 → 0.473 | 0.474 → 0.474 | some rise (1/3) | 3 |
| babel3 | T2 | 0.379 → 0.501 | 0.420 → 0.495 | 0.474 → 0.472 | some rise (2/3) | 3 |
| babel3 | T4 | 0.379 → 0.422 | 0.420 → 0.438 | 0.474 → 0.444 | some rise (2/3) | 3 |
| hugadb | T1 | 0.609 → 0.609 | 0.682 → 0.691 | 0.687 → 0.684 | some rise (1/3) | 1 |
| hugadb | T4 | 0.609 → 0.687 | 0.682 → 0.675 | 0.687 → 0.716 | some rise (2/3) | 1 |
| lara | T2 | 0.464 → 0.590 | 0.445 → 0.569 | 0.455 → 0.565 | all rise | 3 |
| lara | T3 | 0.464 → 0.507 | 0.445 → 0.535 | 0.455 → 0.550 | all rise | 3 |
| lara | T3b | 0.464 → 0.490 | 0.445 → 0.533 | 0.455 → 0.576 | all rise | 3 |
| lara | T4 | 0.464 → 0.552 | 0.445 → 0.502 | 0.455 → 0.518 | all rise | 3 |

## Verification on every metric, not F1@50 alone

F1@50 is the pre-declared metric for every verdict above (H6, H7, H2'). This table re-runs the same paired comparisons on MoF, Edit, F1@10 and F1@25 too, read from the identical saved rows, so a result driven by the choice of F1@50 would show up here as a metric that moves the other way.

| comparison | metric | new mean ± sd | base mean ± sd | Δ (paired) | bar |
|---|---|---|---|---|---|
| babel2 T2 centre vs T0 | MoF | 53.94 ± 6.86 | 34.16 ± 1.03 | +19.78 (+23.8 / +10.7 / +24.9) | beats by bar |
| babel2 T2 centre vs T0 | Edit | 39.62 ± 7.14 | 33.23 ± 0.44 | +6.39 (+10.3 / -2.3 / +11.2) | no |
| babel2 T2 centre vs T0 | F1@10 | 41.28 ± 7.32 | 30.15 ± 0.33 | +11.13 (+13.8 / +3.2 / +16.5) | beats by bar |
| babel2 T2 centre vs T0 | F1@25 | 35.99 ± 7.86 | 26.18 ± 0.37 | +9.82 (+13.3 / +1.2 / +15.0) | beats by bar |
| babel2 T2 centre vs T0 | F1@50 | 26.72 ± 6.88 | 19.66 ± 0.95 | +7.06 (+10.5 / +0.2 / +10.4) | beats by bar |
| babel2 T2 context vs T0 | MoF | 34.52 ± 1.30 | 32.60 ± 1.46 | +1.92 (+2.1 / +1.9 / +1.8) | no |
| babel2 T2 context vs T0 | Edit | 32.32 ± 5.02 | 29.46 ± 1.75 | +2.86 (-2.3 / +6.4 / +4.5) | no |
| babel2 T2 context vs T0 | F1@10 | 36.31 ± 5.25 | 28.91 ± 2.88 | +7.40 (+0.0 / +16.1 / +6.1) | beats by bar |
| babel2 T2 context vs T0 | F1@25 | 30.21 ± 6.93 | 22.70 ± 3.18 | +7.51 (-0.5 / +18.8 / +4.3) | no |
| babel2 T2 context vs T0 | F1@50 | 21.41 ± 4.92 | 15.09 ± 3.46 | +6.33 (-0.4 / +15.0 / +4.3) | no |
| babel3 T2 context vs T0 | MoF | 36.17 ± 2.13 | 32.12 ± 0.92 | +4.05 (+1.4 / +4.3 / +6.4) | beats by bar |
| babel3 T2 context vs T0 | Edit | 34.02 ± 2.40 | 28.08 ± 2.12 | +5.93 (+7.6 / +1.0 / +9.2) | beats by bar |
| babel3 T2 context vs T0 | F1@10 | 36.34 ± 1.11 | 30.98 ± 1.84 | +5.37 (+4.2 / +4.1 / +7.8) | beats by bar |
| babel3 T2 context vs T0 | F1@25 | 29.32 ± 1.59 | 23.62 ± 1.44 | +5.69 (+7.2 / +2.3 / +7.5) | beats by bar |
| babel3 T2 context vs T0 | F1@50 | 19.33 ± 1.12 | 15.32 ± 0.65 | +4.01 (+4.2 / +2.4 / +5.4) | beats by bar |
| lara T2 centre vs T0 | MoF | 37.84 ± 0.53 | 27.10 ± 0.30 | +10.74 (+11.6 / +10.1 / +10.6) | beats by bar |
| lara T2 centre vs T0 | Edit | 28.28 ± 0.02 | 28.45 ± 0.49 | -0.17 (-0.6 / -0.3 / +0.4) | no |
| lara T2 centre vs T0 | F1@10 | 28.63 ± 0.34 | 23.32 ± 0.89 | +5.31 (+4.3 / +6.1 / +5.6) | beats by bar |
| lara T2 centre vs T0 | F1@25 | 24.35 ± 0.84 | 17.64 ± 0.52 | +6.70 (+6.4 / +6.3 / +7.4) | beats by bar |
| lara T2 centre vs T0 | F1@50 | 17.80 ± 1.24 | 10.17 ± 0.20 | +7.63 (+8.1 / +6.4 / +8.4) | beats by bar |
| lara T2 context vs T0 | MoF | 33.36 ± 2.24 | 20.23 ± 1.05 | +13.13 (+11.8 / +12.9 / +14.7) | beats by bar |
| lara T2 context vs T0 | Edit | 32.70 ± 0.58 | 22.50 ± 0.48 | +10.20 (+9.6 / +10.7 / +10.3) | beats by bar |
| lara T2 context vs T0 | F1@10 | 34.76 ± 2.26 | 17.23 ± 1.13 | +17.53 (+15.5 / +17.9 / +19.3) | beats by bar |
| lara T2 context vs T0 | F1@25 | 29.81 ± 2.19 | 11.59 ± 0.82 | +18.21 (+16.7 / +18.0 / +19.9) | beats by bar |
| lara T2 context vs T0 | F1@50 | 19.09 ± 1.34 | 5.06 ± 0.48 | +14.03 (+13.0 / +14.1 / +15.0) | beats by bar |
| lara T3b context vs T0 | MoF | 25.61 ± 0.89 | 20.23 ± 1.05 | +5.38 (+5.3 / +5.6 / +5.3) | beats by bar |
| lara T3b context vs T0 | Edit | 29.85 ± 0.58 | 22.50 ± 0.48 | +7.35 (+6.8 / +7.7 / +7.5) | beats by bar |
| lara T3b context vs T0 | F1@10 | 25.44 ± 1.25 | 17.23 ± 1.13 | +8.22 (+6.4 / +9.8 / +8.4) | beats by bar |
| lara T3b context vs T0 | F1@25 | 19.83 ± 0.93 | 11.59 ± 0.82 | +8.24 (+7.2 / +9.3 / +8.2) | beats by bar |
| lara T3b context vs T0 | F1@50 | 10.89 ± 0.61 | 5.06 ± 0.48 | +5.83 (+5.2 / +6.5 / +5.9) | beats by bar |
| lara T4 centre vs T0 | MoF | 27.66 ± 1.32 | 27.10 ± 0.30 | +0.56 (-0.4 / +1.6 / +0.4) | no |
| lara T4 centre vs T0 | Edit | 31.71 ± 1.93 | 28.45 ± 0.49 | +3.26 (+1.1 / +5.2 / +3.5) | beats by bar |
| lara T4 centre vs T0 | F1@10 | 29.23 ± 2.91 | 23.32 ± 0.89 | +5.92 (+2.4 / +9.9 / +5.5) | beats by bar |
| lara T4 centre vs T0 | F1@25 | 24.07 ± 2.57 | 17.64 ± 0.52 | +6.43 (+3.5 / +9.7 / +6.1) | beats by bar |
| lara T4 centre vs T0 | F1@50 | 16.26 ± 2.05 | 10.17 ± 0.20 | +6.08 (+4.1 / +8.5 / +5.7) | beats by bar |
| babel2 T2 span 2 vs span 8 | MoF | 44.18 ± 6.45 | 34.52 ± 1.30 | +9.65 (+15.0 / +1.1 / +12.9) | beats by bar |
| babel2 T2 span 2 vs span 8 | Edit | 40.34 ± 4.53 | 32.32 ± 5.02 | +8.02 (+18.6 / +1.7 / +3.8) | beats by bar |
| babel2 T2 span 2 vs span 8 | F1@10 | 44.90 ± 2.12 | 36.31 ± 5.25 | +8.59 (+16.1 / +1.8 / +7.9) | beats by bar |
| babel2 T2 span 2 vs span 8 | F1@25 | 40.57 ± 1.65 | 30.21 ± 6.93 | +10.36 (+18.1 / +1.8 / +11.2) | beats by bar |
| babel2 T2 span 2 vs span 8 | F1@50 | 30.68 ± 0.74 | 21.41 ± 4.92 | +9.27 (+15.2 / +3.9 / +8.8) | beats by bar |
| babel2 T2 span 4 vs span 8 | MoF | 43.83 ± 5.80 | 34.52 ± 1.30 | +9.30 (+13.3 / +1.6 / +13.0) | beats by bar |
| babel2 T2 span 4 vs span 8 | Edit | 38.19 ± 2.29 | 32.32 ± 5.02 | +5.87 (+12.4 / +1.1 / +4.1) | beats by bar |
| babel2 T2 span 4 vs span 8 | F1@10 | 43.32 ± 0.83 | 36.31 ± 5.25 | +7.01 (+11.2 / +2.3 / +7.5) | beats by bar |
| babel2 T2 span 4 vs span 8 | F1@25 | 39.86 ± 0.94 | 30.21 ± 6.93 | +9.64 (+14.4 / +2.5 / +12.0) | beats by bar |
| babel2 T2 span 4 vs span 8 | F1@50 | 30.45 ± 0.35 | 21.41 ± 4.92 | +9.04 (+13.8 / +4.3 / +9.1) | beats by bar |
| babel3 T2 span 2 vs span 8 | MoF | 36.23 ± 3.27 | 36.17 ± 2.13 | +0.06 (-0.2 / -1.1 / +1.5) | no |
| babel3 T2 span 2 vs span 8 | Edit | 41.09 ± 2.22 | 34.02 ± 2.40 | +7.07 (+3.7 / +9.2 / +8.3) | beats by bar |
| babel3 T2 span 2 vs span 8 | F1@10 | 43.80 ± 2.14 | 36.34 ± 1.11 | +7.46 (+5.0 / +8.1 / +9.2) | beats by bar |
| babel3 T2 span 2 vs span 8 | F1@25 | 36.83 ± 3.05 | 29.32 ± 1.59 | +7.51 (+3.3 / +9.1 / +10.1) | beats by bar |
| babel3 T2 span 2 vs span 8 | F1@50 | 23.23 ± 3.39 | 19.33 ± 1.12 | +3.90 (+1.0 / +4.4 / +6.2) | beats by bar |


## Span (stage D) and window (stage E) on the selected tokenizer

Selected on BABEL-2 only, frozen in `selection.json`: **t2_relational** (T2). Span 8 and window default come from stage A (or the T0 baselines).

### Span (`context`, F1@50)

| dataset | mask span | per seed | mean ± sd | Δ vs span 8 (paired) |
|---|---|---|---|---|
| babel2 | 2 | 31.48 / 30.02 / 30.54 | 30.68 ± 0.74 | +9.27 (+15.2 / +3.9 / +8.8) |
| babel2 | 4 | 30.11 / 30.45 / 30.80 | 30.45 ± 0.35 | +9.04 (+13.8 / +4.3 / +9.1) |
| babel2 | 8 | 16.33 / 26.16 / 21.75 | 21.41 ± 4.92 | — |
| babel3 | 2 | 20.06 / 22.83 / 26.80 | 23.23 ± 3.39 | +3.90 (+1.0 / +4.4 / +6.2) |
| babel3 | 4 | 18.15 / 19.20 / 24.34 | 20.56 ± 3.31 | +1.23 (-0.9 / +0.8 / +3.8) |
| babel3 | 8 | 19.02 / 18.40 / 20.58 | 19.33 ± 1.12 | — |


Per-dataset bar result (before the combined verdict):

- babel2: span(s) 2, 4 beat span 8 by the bar (span 2 Δ +9.27 (+15.2 / +3.9 / +8.8); span 4 Δ +9.04 (+13.8 / +4.3 / +9.1))
- babel3: span(s) 2 beat span 8 by the bar (span 2 Δ +3.90 (+1.0 / +4.4 / +6.2); span 4 Δ +1.23 (-0.9 / +0.8 / +3.8))
### Masked-loss boundary split, recomputed per span (diagnostic)

| dataset | span | seeds | share of scored targets crossing a boundary | within loss | crossing loss | scored targets |
|---|---|---|---|---|---|---|
| babel2 | 2 | 3 | 0.069 | 3.284 | 4.043 | 304 |
| babel2 | 4 | 3 | 0.155 | 3.488 | 4.007 | 304 |
| babel2 | 8 | 3 | 0.171 | 3.499 | 5.341 | 304 |
| babel3 | 2 | 3 | 0.036 | 4.488 | 4.271 | 329 |
| babel3 | 4 | 3 | 0.234 | 4.647 | 4.554 | 329 |
| babel3 | 8 | 3 | 0.295 | 4.761 | 5.208 | 329 |

### Window (F1@50)

The OT radius in seconds is nearly constant across these windows (about 0.93s / 0.80s / 1.00s for windows 4/8/15 at the current rule), so it is a minor confound here. The larger confounds are the mask duration and the context length in seconds, both of which change with the window at a fixed span; the mask-duration and span-ratio columns make that explicit.

| dataset | window (frames) | mask duration (s) | span ratio | arm | per seed | mean ± sd |
|---|---|---|---|---|---|---|
| babel2 | 4 | 1.07 | 0.63 | context | 30.81 / 31.49 / 31.82 | 31.37 ± 0.52 |
| babel2 | 4 | 1.07 | 0.63 | centre | 13.40 / 21.80 / 14.82 | 16.67 ± 4.50 |
| babel2 | 8 | 2.13 | 1.25 | context | 16.33 / 26.16 / 21.75 | 21.41 ± 4.92 |
| babel2 | 8 | 2.13 | 1.25 | centre | 30.63 / 18.77 / 30.75 | 26.72 ± 6.88 |
| babel2 | 15 | 4.00 | 2.35 | context | 10.87 / 10.73 / 11.87 | 11.16 ± 0.62 |
| babel2 | 15 | 4.00 | 2.35 | centre | 26.61 / 20.94 / 31.96 | 26.50 ± 5.51 |
| babel3 | 4 | 1.07 | 0.60 | context | 21.96 / 24.60 / 13.98 | 20.18 ± 5.53 |
| babel3 | 4 | 1.07 | 0.60 | centre | 15.23 / 16.14 / 15.98 | 15.78 ± 0.49 |
| babel3 | 8 | 2.13 | 1.21 | context | 19.02 / 18.40 / 20.58 | 19.33 ± 1.12 |
| babel3 | 8 | 2.13 | 1.21 | centre | 20.08 / 21.95 / 21.80 | 21.28 ± 1.04 |
| babel3 | 15 | 4.00 | 2.26 | context | 12.92 / 15.40 / 15.26 | 14.53 ± 1.39 |
| babel3 | 15 | 4.00 | 2.26 | centre | 27.72 / 28.29 / 13.55 | 23.19 ± 8.35 |

**H2′ (a shorter mask span helps, on the selected tokenizer): supported (spans 2, 4 beat span 8 on BABEL-2; BABEL-3 moves the same way); BABEL-3 also beats span 8 by the bar on its own at span(s) 2.**

## Reference: SMQ's released checkpoint (re-scored here, global Hungarian) vs. the best-of-39 arm found here (optimistic; not a pre-declared comparison)

| dataset | SMQ F1@50 (released checkpoint) | best-of-39 here (mean, 3 seeds) | which |
|---|---|---|---|
| babel2 | 27.5 | 26.72 | T2 centre |
| babel3 | 19.4 | 26.04 | T1 pca |
| lara | 16.4 | 19.09 | T2 context |
| hugadb | 24.3 | 46.23 | T0 context |

