# Pretrained-encoder test pe_babel2_001


BABEL-2; matched B0 (C = 5) fitted on four recording folds and decoded on the held-out fold, mean over the folds run; final endpoint; one Hungarian mapping per fold. Probes use labels and are diagnostics only.

Folds run: fold0, fold1, fold2, fold3, fold4.

| input | seed | B0 MoF | B0 F1@50 | balanced acc | action probe (continuous) | code table | majority |
|---|---|---|---|---|---|---|---|
| mb | 111 | 54.2 | 23.3 | 46.5 | 76.8 | 74.2 | 51.2 |
| mb | 222 | 55.4 | 25.3 | 48.7 | 76.8 | 71.8 | 51.2 |
| mb | 1538574472 | 54.5 | 23.4 | 50.2 | 77.0 | 72.7 | 51.2 |
| t2 | 111 | 50.4 | 24.7 | 47.8 | 77.2 | 73.6 | 51.2 |
| t2 | 222 | 49.5 | 24.4 | 45.2 | 77.3 | 74.4 | 51.2 |
| t2 | 1538574472 | 44.9 | 18.8 | 48.4 | 77.4 | 74.0 | 51.2 |

## Decision (declared rules)

**STOP**

- mb − t2 MoF per seed: +3.85, +5.85, +9.60 (mean +6.43); F1@50: -1.41, +0.92, +4.59 (mean +1.37); bar +3 on both, all seeds positive
- supervised probe gain mb − t2 (mean over seeds): -0.5 (INFORMATION_ONLY needs ≥ 5)

One encoder, one checkpoint, one pooling, one view, frozen. The encoder saw AMASS motions (including BABEL) without labels; the tokenizer is fitted on all recordings. Recording-disjoint folds share performers (BABEL has no subject IDs). Three seeds are not confidence intervals.