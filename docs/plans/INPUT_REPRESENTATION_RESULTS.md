# Input-representation test: results (run irt_lara_001, LARa)

7 October 2026 · Plan and declared rules: [INPUT_REPRESENTATION_TEST_PLAN.md](INPUT_REPRESENTATION_TEST_PLAN.md) · Raw outputs: `results/input_representation/irt_lara_001/` (REPORT.md, decision.json, probes/, b0__*/cells/). T2 decode figures are the Milestone 1 reference cells (`results/compositional_dynamics/m1_lara_001`), read, not re-run.

## Outcome

**Declared verdict: ORGANISATION_PROBLEM.** Neither new input passes the adoption bar, and the supervised probe exceeds the unsupervised MoF by 24–27 points (threshold 15). All 9 cells completed; the B0 solves converged on all 44 evaluation recordings (one capped for `t2_centered`).

## Held-out decode (44 evaluation recordings, mean of 3 seeds, final endpoint)

| input | MoF | Edit | F1@10 | F1@25 | F1@50 | predicted ÷ true segments |
|---|---|---|---|---|---|---|
| t2 | 38.9 | 29.5 | 29.6 | 25.7 | 19.7 | 1.82 |
| t2_bodyscale | 40.7 | 25.1 | 24.0 | 18.0 | 11.6 | 2.96 |
| t2_centered | 41.8 | 28.5 | 25.9 | 22.2 | 15.7 | 2.15 |

Paired by seed against t2 (bar: mean ≥ +3 on both MoF and F1@50, all seeds positive):

| input | MoF per seed | mean | F1@50 per seed | mean | passes |
|---|---|---|---|---|---|
| t2_bodyscale | +1.75, +3.72, +0.03 | +1.83 | −7.94, −6.19, −10.14 | −8.09 | no |
| t2_centered | +3.16, +3.45, +2.19 | +2.93 | −3.44, −4.39, −4.10 | −3.98 | no |

MoF rises slightly for both inputs (below the bar), while F1@50 and Edit fall: the new inputs fragment the segmentation more. Seed spread in MoF is about 1–2 points, so the `t2_centered` MoF gain is consistent in sign but small.

## Probes (label-using diagnostics; mean over folds/seeds)

| input | action probe (continuous) | action probe (code table) | majority baseline | subject probe (chance 7.7) |
|---|---|---|---|---|
| t2 | 65.8 | 62.3 | 43.5 | **99.3** |
| t2_bodyscale | 65.1 | 63.4 | 43.5 | **98.5** |
| t2_centered | 67.5 | 61.4 | 43.5 | **31.9** |

Assignment association on the evaluation recordings (after decoding, NMI): recording 0.24 / 0.11 / 0.12 and subject 0.25 / 0.08 / 0.09 for t2 / t2_bodyscale / t2_centered. Action NMI: 0.23 / 0.25 / 0.22.

## Revised reading (re-analysis of saved outputs, 7 October)

The first version of this section over-read the run. Three re-analyses on saved files (no retraining) change it.

**1. The dominant loss is a granularity mismatch, not missing organisation.** Handling(centred) is 46% of evaluation frames. With T2 (seed 111), 5 of the 8 decoded states have Handling(centred) as their most frequent class, together holding about 70% of frames. One-to-one Hungarian mapping can assign only one of them to that class, so the other four are scored against classes they barely contain. The same holds for `t2_centered` (6 of 8 states). Meanwhile Synchronization (1.7%), None (5%) and Handling(upwards) (6.7%) get no state of their own. The decoder subdivides the dominant action into sub-states, which is what a unit model would be expected to do, and the evaluation penalises it.

| input | one-to-one MoF | many-to-one MoF | majority (eval frames) |
|---|---|---|---|
| t2 | 38.9 | 54.3 | 46.1 |
| t2_bodyscale | 40.7 | 54.7 | 46.1 |
| t2_centered | 41.8 | 51.9 | 46.1 |

About 15 points are lost to the one-to-one constraint alone. The states are purer than the majority baseline by about 6–9 points, against about 22 for the supervised linear probe. The C-state decoder with a near-uniform relaxed marginal (q = 1/C) has no way to group sub-states into actions or to give rare actions a state.

**2. F1@50 differences between inputs are confounded with the emission/temporal balance.** Decoding the same frozen costs at multipliers 0.5 / 1 / 2 moves T2's F1@50 from 25.3 to 19.7 to 12.7, while MoF moves by under 1 point. Each input produces a different cosine-cost spread, so the same coefficients give each a different amount of smoothing. At multiplier 0.5 the inputs' F1@50 gaps to T2 shrink (bodyscale −4.6, centered −3.4) but remain negative. MoF is the cleaner signal for comparing inputs. On MoF, `t2_centered` is +2.9 (consistent in sign, just under the bar), but its many-to-one purity is lower than T2's: the gain comes from a better state-to-class match, not purer states. No input makes the states purer.

**3. The identity claims were not supported and are withdrawn.**
- Body-scale normalisation removes global size, not body proportions. Relative limb lengths stay in the distances, so the 98.5% subject probe after normalisation says proportions identify people, not that size is irrelevant.
- The 32% subject probe on `t2_centered` is linear, on a feature dominated by pose and orientation. It does not show identity was removed.
- The halved recording/subject NMI of decoded states coincides with more fragmentation (1.8 → 2.2–3.0 predicted segments per true segment), which lowers association with constant per-recording labels mechanically.

So the test never verifiably removed identity, and it says nothing about whether removing identity would help. The earlier statement "removing identity did not recover actions" is retracted.

**What stands:**
- K-means-512 codes keep nearly all linearly decodable per-window action information (code table 61–63% vs 65–67%).
- Simple input changes do not make decoded states purer.
- The per-window supervised ceiling is itself modest: 65–67% against a 43.5% majority, without temporal context.

## Revised implications

- The declared rule's verdict (ORGANISATION_PROBLEM) stands as computed. Its practical meaning is narrower: the largest measurable loss is that the decoder's C states and near-uniform prior do not match LARa's action granularity and imbalance. A better encoder alone would not fix that.
- This bears directly on the compositional premise. The decoder already finds sub-units of the dominant action. What the pipeline lacks is a label-free way to **group units into actions**, or to allow more states than actions without breaking the one-to-one evaluation. This hierarchical step (units → actions) is the untested part, and it is cheaper to test than an end-to-end encoder.
- LARa is a hard place to judge this, because one class is 46% of frames. A conclusion from LARa alone should not be generalised without one confirmation dataset.
- Not run: any end-to-end training, a hierarchical grouping test, a heading-invariant or proportion-normalised input, a nonlinear identity probe, or a temporal-context probe.
