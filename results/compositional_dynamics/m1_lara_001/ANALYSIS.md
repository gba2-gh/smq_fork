# Interpretation of m1_lara_001

7 October 2026. Analysis of saved outputs; no new training or parameter selection.

## Decision

The declared numerical feasibility gate fails decisively. Do not proceed automatically to learned units/encoder training with this decoder. Ordered action-conditioned next-code likelihood does not improve segmentation under this protocol. This is a negative result for the implemented mechanism, not proof that discrete units or action composition are impossible.

All 60 cells completed: four arms, three seeds and five populations (primary split plus four subject-disjoint folds). Summed cell runtime is 5.95 hours. All neural cells record 5,000 updates, all score files use the 4,000-budget final endpoint, and no cells are listed as omitted. Historical parity remains unverified because its output-writing step crashed, as explained below.

## Main results

Primary evaluation uses 44 recordings, with one global evaluation-label Hungarian mapping per arm/seed. Values below are means and population SD across the three seeds; these are initialization sensitivity, not sampling uncertainty.

| Arm | MoF | F1@50 | Edit | Predicted / GT segment count |
| --- | --- | --- | --- | --- |
| Matched B0-centre | 38.87 ± 0.39 | 19.68 ± 0.60 | 29.53 | 1.82 |
| Occupancy | 38.58 ± 0.82 | 21.11 ± 0.37 | 32.32 | 1.62 |
| Bag history | 38.55 ± 0.08 | 14.07 ± 0.39 | 22.61 | 3.02 |
| Ordered history | 38.36 ± 0.27 | 13.39 ± 0.36 | 21.67 | 3.19 |

Ordered minus B0 is -6.29 F1@50; ordered minus occupancy is -7.72. Both differences are negative in all three seeds. Ordered minus bag is -0.68 on average, negative in two seeds and only +0.14 in the other. This is not a marginal failure of an aggressive +3/+5 threshold.

Occupancy improves F1@50 over B0 by 1.43 points on average, with similar MoF. That is a useful local control result, not a large architectural contribution or a matched comparison to external SOTA.

## Secondary checks agree with the primary failure

Subject-disjoint F1@50, first averaging the four folds within each seed and then averaging seeds:

| Arm | F1@50 |
| --- | --- |
| Matched B0-centre | 18.18 |
| Occupancy | 20.54 |
| Bag history | 12.47 |
| Ordered history | 11.31 |

Ordered is worse than bag in all 12 seed/fold pairs, and worse than occupancy in all 12. This supports the negative ranking on unseen decoder-training subjects. Tokenization remains transductive, and each fold has its own oracle mapping; absolute fold scores should not be compared to the primary split as a clean measure of subject-transfer difficulty.

Halving emission strength improves every arm's mean F1@50:

| Arm | 0.5× emission | Declared 1× | 2× emission |
| --- | --- | --- | --- |
| Matched B0-centre | 25.26 | 19.68 | 12.69 |
| Occupancy | 25.31 | 21.11 | 15.70 |
| Bag history | 18.74 | 14.07 | 11.71 |
| Ordered history | 17.02 | 13.39 | 10.66 |

Weaker emissions make the existing temporal/marginal/entropy terms stronger relative to emissions. The sensitivity shows a consequential regularization balance, but it does not rescue the dynamics model: it remains substantially worse at every matched multiplier. Do not promote 0.5× to a retrospectively selected primary result. Occupancy's small advantage over B0 also largely disappears at 0.5×.

## What the diagnostics establish

**The failure is strongly associated with fragmentation.** MoF changes little, while ordered dynamics approximately doubles the segment-count ratio relative to occupancy. Edit and every segmental F1 threshold deteriorate. This is consistent with action evidence changing too often within an action, although counts alone do not identify the precise erroneous boundaries.

**Prediction fit and segmentation utility diverge.** Mean assigned-state conditional NLL is 1.88 fitting / 2.04 evaluation for ordered, versus 2.32 / 2.46 for bag. Ordered better explains observed codes under its inferred states, but segments worse. These are assigned-state NLLs, not an independent marginal likelihood under a common externally fixed state assignment. They support an objective/segmentation mismatch, not a claim of superior semantic representation.

**Ordinary fitting-set memorization is not the complete explanation.** Ordered fitting F1@50 is only 13.53 versus evaluation 13.39; occupancy is 21.59 fitting versus 21.11 evaluation. The dynamics model already has poor segmentation on the fitting population. Separate oracle mappings and population composition limit interpretation of the gap, and some NLL overfitting remains possible.

**The final segmentation weakly depends on the observed order.** Shuffling each target's history reduces ordered F1@50 from 13.39 to 13.04 on average, with 93.83% state agreement. The trained bag control is also better. Order-sensitive prediction is present in the architecture, but this run provides no evidence that it produces useful action-state assignments.

**Increased identity association is not supported as the explanation for this run's final states.** From B0 to ordered, mean subject NMI decreases 0.2542 → 0.2217 and recording NMI 0.2413 → 0.2183, while action NMI slightly increases 0.2285 → 0.2417. These measurements concern final state assignments, not all information in hidden representations. They neither establish identity invariance nor justify saying identity retention caused the F1 loss.

**Emission strength changes during self-training despite a frozen calibration factor.** The median scaled state-cost range grows from about 1.00 at the first refresh to 1.45 at the last recorded ordered refresh; occupancy remains near 1.02. This is permitted by the declared protocol, not an implementation bug. It offers a mechanism for greater fragmentation, but the matched multiplier results show that this one sensitivity is not enough to recover a competitive dynamics result.

My interpretation is that the learned transition likelihood supplies locally useful predictive distinctions that do not translate into stable action evidence. An improvement shared by all action heads cannot distinguish action states; only differences between heads enter their relative OT preference. The observed improvement in prediction loss therefore does not require useful action composition. The remaining action-dependent evidence can also vary with phases or transitions inside an action. This is a plausible explanation consistent with the diagnostics, not a proved causal decomposition.

## Verification and implementation caveats

- The run uses reviewed package hash `f16c3fdbf5974f04`; all 60 stored cell identities reconstruct correctly using that run hash and the current preparation contents. The newer working-tree hash differs because parity serialization was subsequently edited. The saved run must retain its original provenance.
- Independently rescoring the saved seed-111 ordered and occupancy frame predictions reproduces all five stored metrics exactly.
- Across all saved evaluation endpoints, 5,713 recording-solves are converged and 83 capped. The three primary ordered cells have only 2 capped evaluation recordings out of 132 solves. These results are finite-budget results; widespread failure to reach the solver criterion is not an evidenced explanation for the primary ranking.
- The overnight log records `TypeError: Object of type bool_ is not JSON serializable` in `parity.py` while writing `parity.json`, after numerical computation. The launcher continued. This explains `parity: null`; it is neither a demonstrated numerical mismatch nor a passed reproduction. A serialization fix is now present in the working file, but no completed parity artifact was available when checked.
- The prior review's runner and artifact limitations remain relevant. All main cells completed, so optional failure handling did not erase cells in this run; the audited identities/scores show no observed cache contamination. Saved `R_final` remains the last training refresh rather than final decoding responsibilities, so it must not be used as the final posterior. Those limitations do not provide evidence that the observed ranking is an implementation artifact.

## Next steps

1. Honor the stop rule: close this particular one-step action-conditioned likelihood pilot as unsuccessful. Do not launch learned-unit/encoder training on the assumption that a larger representation will repair it.
2. Complete historical parity with the serialization correction and preserve its result under explicit provenance. This closes the verification gap; it does not require retraining the 60 completed cells.
3. Retain matched B0 and occupancy as inexpensive baselines. Keep their emission-strength sensitivities visible and distinguish primary from secondary outcomes.
4. Use the result to constrain the next research proposal: it must justify why its objective should recover stable action identity across multiple fine movements, rather than merely predict fine codes. A segment-level formulation could address that mismatch, but would be a new hypothesis requiring a novelty and feasibility argument, not an automatic continuation or a recommendation to add more mechanisms now.

This run answers a useful architectural question and saves a larger unsuccessful training campaign. It does not by itself supply the major positive contribution sought for a top-conference paper.
