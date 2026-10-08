# Pretrained-encoder test: results (run pe_babel2_001, BABEL-2)

8 October 2026 · Plan: [PRETRAINED_ENCODER_TEST_PLAN.md](PRETRAINED_ENCODER_TEST_PLAN.md) · Raw outputs: `results/pretrained_encoder/pe_babel2_001/` (REPORT.md, decision.json, probes/, b0__*/cells/).

## Outcome

**STOP**, by the declared rule. All 30 B0 cells (2 inputs × 3 seeds × 5 folds) and 6 probes completed; verification passed for the run's code hash.

| input | MoF per seed | mean | F1@50 per seed | mean | supervised probe | code-table probe |
|---|---|---|---|---|---|---|
| t2 | 50.4 / 49.5 / 44.9 | 48.3 | 24.7 / 24.4 / 18.8 | 22.6 | 77.3 | 74.0 |
| mb (MotionBERT, frozen) | 54.2 / 55.4 / 54.5 | 54.7 | 23.3 / 25.3 / 23.4 | 24.0 | 76.9 | 72.9 |
| mb − t2 | +3.85 / +5.85 / +9.60 | **+6.43** | −1.41 / +0.92 / +4.59 | **+1.37** | **−0.5** | |

- **ADOPT_ENCODER fails** because of F1@50: the mean is +1.4 (bar +3) and one seed is negative. MoF alone would have passed (+6.4, all seeds positive).
- **INFORMATION_ONLY fails:** the supervised probe gain is −0.5 (needs ≥ +5).

## What the results show

1. **MotionBERT features carry no more linearly decodable action information than T2.** The supervised probe is 76.9 vs 77.3, the code-table probe 72.9 vs 74.0, and many-to-one purity of the decoded states is similar (about 64 for both).
2. **The MoF gain is real but is not an information gain.** MotionBERT decodes are higher on MoF in every seed and much more stable across seeds (54.2–55.4 vs 44.9–50.4). Part of the mean gain comes from T2's weak seed 1538574472 (+9.6 there, +3.9 and +5.9 elsewhere). With purity unchanged, the gain comes from states that match the classes better under one-to-one mapping. This is the same pattern as `t2_centered` on LARa.
3. **Segmentation quality does not improve.** F1@50 is flat on average. T2's F1@50 swings strongly by fold (32–37 on folds 0–1, 16–20 on folds 2–4 for seeds 111/222). MotionBERT's is more uniform (19–30). Predicted ÷ true segments: 1.32 vs 1.39.
4. **Both decoders sit close to the frame-majority baseline.** On these folds 51.0% of frames belong to the majority class. T2's mean B0 MoF (48.3) is below that baseline; MotionBERT's (54.7) is slightly above it.
5. **Identity association is not reduced.** Recording NMI of the decoded states is 0.29 (mb) vs 0.24 (t2); action NMI is 0.25 vs 0.24.

## Notes

- The historical T2 BABEL-2 B0 result (about 57 MoF) was measured on the 27-recording preparation split. Over all 255 recordings in 5 folds, T2 averages 48.3, so that split was favourable.
- The decode multipliers listed as a diagnostic in the plan were not computed: the reused B0 cell computes them only for the primary split, and this test used folds only. This does not affect the verdict.
- Limits, as declared: one encoder, one checkpoint, one pooling (mean over joints), one view, frozen; the encoder saw AMASS (including BABEL) motions without labels; recording folds share performers.

## Reading in context

This is the fifth consecutive test in which a change of decoder, grouping or input leaves the supervised probe essentially unchanged and moves label-free decoding by at most a few points:

| test | dataset | supervised probe | label-free decode vs majority baseline |
|---|---|---|---|
| Milestone 1 (ordered dynamics) | LARa | – | below |
| input representation (bodyscale, centred) | LARa | 65–67 | below |
| unit grouping (24 → 8) | LARa | – | below |
| pretrained encoder (MotionBERT) | BABEL-2 | 77 vs 77 | T2 below, MotionBERT slightly above |

The information needed to classify windows is present in every representation tried (supervised probes are 20–26 points above the majority baseline). What none of the label-free methods do is organise it into the benchmark's action classes much better than the majority baseline.
