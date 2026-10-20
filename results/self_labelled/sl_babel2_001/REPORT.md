# Self-labelled embedding test sl_babel2_001


BABEL-2; frozen-MotionBERT window features; matched B0 decode on held-out recording folds (mean over folds), final endpoint, one Hungarian mapping per fold. `F0` = frozen MotionBERT K-means codes from pe_babel2_001 (verified identity; declared transductive tokenization). `F1` and the learned arms fit every preprocessing step on the fold's fitting recordings only.

| arm | seed | MoF | F1@50 | balanced acc | recording NMI | NN share (shared context) | NN share (support-disjoint, eligible) | collapsed folds |
|---|---|---|---|---|---|---|---|---|
| F0 | 111 | 54.22 | 23.34 | 46.52 | 0.29 | 0.28 | 0.24 | 0 |
| F0 | 222 | 55.38 | 25.28 | 48.73 | 0.29 | 0.29 | 0.26 | 0 |
| F0 | 333 | 54.07 | 24.43 | 44.47 | 0.29 | 0.28 | 0.22 | 0 |
| F0 | 444 | 54.52 | 25.98 | 42.42 | 0.30 | 0.28 | 0.23 | 0 |
| F0 | 1538574472 | 54.53 | 23.40 | 50.20 | 0.29 | 0.28 | 0.23 | 0 |
| F1 | 111 | 46.06 | 21.60 | 37.67 | 0.30 | 0.39 | 0.32 | 0 |
| F1 | 222 | 46.28 | 21.61 | 37.45 | 0.30 | 0.39 | 0.32 | 0 |
| F1 | 333 | 46.27 | 21.82 | 37.52 | 0.30 | 0.39 | 0.32 | 0 |
| F1 | 444 | 46.33 | 21.60 | 37.67 | 0.30 | 0.39 | 0.32 | 0 |
| F1 | 1538574472 | 46.28 | 21.59 | 37.66 | 0.30 | 0.39 | 0.32 | 0 |
| L_full | 111 | 39.37 | 21.88 | 38.39 | 0.36 | 0.21 | 0.15 | 0 |
| L_full | 222 | 37.68 | 20.44 | 39.93 | 0.32 | 0.21 | 0.16 | 0 |
| L_full | 333 | 41.34 | 22.54 | 40.39 | 0.33 | 0.20 | 0.14 | 0 |
| L_full | 444 | 37.15 | 18.52 | 36.67 | 0.33 | 0.20 | 0.13 | 0 |
| L_full | 1538574472 | 41.54 | 23.84 | 41.03 | 0.33 | 0.20 | 0.16 | 0 |
| L_noaug | 111 | 34.17 | 16.78 | 33.87 | 0.34 | 0.23 | 0.15 | 0 |
| L_noaug | 222 | 37.08 | 18.35 | 35.74 | 0.37 | 0.22 | 0.17 | 0 |
| L_noaug | 333 | 35.60 | 18.76 | 35.31 | 0.32 | 0.21 | 0.17 | 0 |
| L_noaug | 444 | 38.68 | 20.21 | 37.29 | 0.34 | 0.24 | 0.17 | 0 |
| L_noaug | 1538574472 | 35.94 | 21.26 | 41.51 | 0.36 | 0.21 | 0.14 | 0 |
| L_noeq | 111 | 34.05 | 16.68 | 30.30 | 0.04 | 0.08 | 0.04 | 0 |
| L_noeq | 222 | 32.69 | 17.08 | 33.96 | 0.04 | 0.08 | 0.04 | 0 |
| L_noeq | 333 | 34.19 | 15.59 | 32.69 | 0.04 | 0.08 | 0.05 | 0 |
| L_noeq | 444 | 34.33 | 16.80 | 35.08 | 0.05 | 0.08 | 0.05 | 0 |
| L_noeq | 1538574472 | 32.88 | 17.75 | 34.10 | 0.05 | 0.09 | 0.06 | 0 |

## Decision (rules declared in the plan)

**STOP**

- L_full vs_F0: mean MoF -15.13, mean F1@50 -3.04; per-seed MoF -14.8, -17.7, -13.0, -12.7, -17.4, F1@50 -1.5, -4.8, +0.4, -1.9, -7.5 -> rule fails
- L_full vs_F1: mean MoF -6.83, mean F1@50 -0.20; per-seed MoF -6.7, -8.6, -4.7, -4.9, -9.2, F1@50 +0.3, -1.2, +2.2, +0.7, -3.1 -> rule fails
- L_full vs_L_noeq: mean MoF +5.79, mean F1@50 +4.67; per-seed MoF +5.3, +5.0, +8.7, +7.1, +2.8, F1@50 +5.2, +3.4, +6.1, +7.0, +1.7 -> rule holds
- L_full vs_L_noaug: mean MoF +3.12, mean F1@50 +2.37; per-seed MoF +5.2, +0.6, +5.6, +5.7, -1.5, F1@50 +5.1, +2.1, +2.6, +3.8, -1.7 -> rule holds
- adopt (vs F0, no collapsed L_full cell): False; collapsed L_full cells: 0; learning claim (vs F1): False; equipartition claim: True; augmentation claim: True

Development data (BABEL-2 was diagnosed with labels in earlier tests). Five seeds measure initialization and batch-order sensitivity on the same recordings. The shortcut diagnostics are reported, not gated; the shared-context neighbour share can still reflect shared MotionBERT source frames, and the support-disjoint share is computed only over queries that have an eligible same-recording candidate. Batch equipartition does not by itself prevent recording-based states: several recordings can occupy different states while the batch stays balanced.