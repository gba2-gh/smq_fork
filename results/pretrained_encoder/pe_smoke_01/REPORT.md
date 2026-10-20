# Pretrained-encoder test pe_smoke_01

SMOKE RUN: plumbing only, not evidence.
BABEL-2; matched B0 (C = 5) fitted on four recording folds and decoded on the held-out fold, mean over the folds run; final endpoint; one Hungarian mapping per fold. Probes use labels and are diagnostics only.

Folds run: fold0.

| input | seed | B0 MoF | B0 F1@50 | balanced acc | action probe (continuous) | code table | majority |
|---|---|---|---|---|---|---|---|
| mb | 111 | 44.9 | 24.4 | 40.1 | 75.3 | 74.2 | 51.2 |
| t2 | 111 | 47.9 | 29.6 | 50.8 | 75.7 | 73.6 | 51.2 |

## Decision (declared rules)

**not computed (smoke run)**


One encoder, one checkpoint, one pooling, one view, frozen. The encoder saw AMASS motions (including BABEL) without labels; the tokenizer is fitted on all recordings. Recording-disjoint folds share performers (BABEL has no subject IDs). Three seeds are not confidence intervals.