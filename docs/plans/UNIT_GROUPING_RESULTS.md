# Unit-to-action grouping test: results (run ug_lara_001, LARa)

7 October 2026 · Plan: [UNIT_GROUPING_TEST_PLAN.md](UNIT_GROUPING_TEST_PLAN.md) (v2) · Raw outputs: `results/unit_grouping/ug_lara_001/` (REPORT.md, gates.json, cells/).

## Outcome

**STOP**, by the declared rule. All four criteria fail, and the subject-disjoint folds give the same verdict. All 15 cells completed (3 seeds × primary + 4 folds), and verification passed for the run's code hash.

| arm (primary split, mean of 3 seeds) | MoF | F1@50 | balanced accuracy | classes with a group |
|---|---|---|---|---|
| b0_c (8-state B0, Milestone 1) | 38.9 | 19.7 | 46.4 | – |
| group_temporal | 34.0 | 12.6 | 36.3 | 3–4 |
| group_shuffled (adjacency destroyed) | 35.2 | 12.2 | 35.3 | 3–5 |
| group_appearance | 35.9 | 13.6 | 43.5 | 4 |
| size-matched random (100 per seed) | 28.9 (p95 33.7) | 9.0 (p95 11.2) | – | – |

Folds (mean of 12 cells): temporal 38.4 / 13.8, shuffled 37.1 / 12.4, appearance 40.9 / 16.2. Temporal minus B0 is −4.7 MoF and −4.4 F1@50.

## What the results show

1. **No label-free grouping beats fitting 8 states directly.** Every grouping falls below `b0_c` on both metrics, in the primary split and in the folds.
2. **Local temporal adjacency adds nothing beyond recording membership.** Temporal and shuffled groupings are within about 1 point of each other in both readouts. The temporal grouping also fails to clear the 95th percentile of size-matched random partitions in every seed (MoF 32.3 / 35.2 / 34.6 vs p95 33.7 / 34.3 / 33.1).
3. **The headroom exists, but these signals do not find it.** Relabelled with the oracle many-to-one mapping, the 24 states reach 58–59 MoF, against about 54 for the 8-state decoder. So the over-complete states are purer. But the label-free groupings agree poorly with the oracle's state-to-class assignment (ARI 0.07–0.14 temporal, 0.13–0.20 appearance), and their many-to-one purity (49–54) is no better than fitting 8 states directly (54.3).
4. **Rare classes are still not recovered.** Only 3–5 of the 8 classes receive a group.
5. **Smoothing does not change the ranking.** At emission multipliers 0.5 and 2, temporal stays at or below appearance and shuffled in two of three seeds.

## Reading in context

This is the second negative test of temporal structure among T2 units on LARa, at the roughly 2-second scale:
- Milestone 1: order of preceding units did not add action evidence.
- This test: symmetric co-occurrence of states did not group them into actions beyond recording membership.

The input-representation test found that the 512 codes keep most linearly decodable action information, and that the transport states are purer than the 8-way decode suggests. What none of the label-free signals tried so far can do is turn that purity into action identity on LARa.

Limits of the conclusion (as declared):
- STOP refers to this fixed-horizon (h = 8 windows) spectral grouping of B0 transport states with M = 24. It is not a verdict on unit grouping in general.
- No other horizon, state count or grouping method was run, and none should be run now in response to this result.
- LARa's class structure (Handling(centred) 46%; Handling up/centred/down differ mainly by arm height; Cart and None depend on context outside the skeleton) may limit any skeleton-only, label-free method. This is a hypothesis, not a finding.
