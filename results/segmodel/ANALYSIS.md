# Milestone 1 results audit

The observed primary gate fails. The run also failed its lambda-calibration target and the implementation/reporting checks below; these results do not constitute a complete, protocol-compliant test of the planned milestone.

## Population and duplicate handling

The log contains 25 entries: 3 D7 entries and 22 experiment entries representing 12 unique cells. Each cell has exactly three unique seeds per dataset/standardization setting. This audit takes the last logged entry per configuration and seed, without selecting by score. All 12 retained entries reproduce all five logged metrics exactly from saved frame predictions, with correct frame counts. Some repeated LARa fits differ slightly, so repeated rows must not be interpreted as independent seeds. The original report incorrectly counts four or six seeds.

Only transductive soft K=500, temperature multiplier 1 was run. HuGaDB: 10 observed classes, 18 subjects, wearable IMU inputs. LARa: 8 classes, 16 subjects, optical motion capture. Dataset differences are not controlled comparisons.

## Metrics

Mean +/- population SD across three unique seeds. Seed variation is initialization sensitivity, not sampling uncertainty.

| Dataset | Recording z-score | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| hugadb | SMQ baseline | 41.99 | 36.16 | 38.50 | 31.54 | 24.30 |
| hugadb | False | 44.01 +/- 2.01 | 29.21 +/- 3.20 | 36.46 +/- 5.12 | 30.33 +/- 6.01 | 27.16 +/- 6.29 |
| hugadb | True | 48.21 +/- 0.64 | 33.72 +/- 0.10 | 41.17 +/- 1.84 | 30.01 +/- 0.20 | 26.53 +/- 0.44 |
| lara | SMQ baseline | 37.38 | 39.40 | 34.69 | 28.40 | 16.39 |
| lara | False | 20.08 +/- 0.16 | 14.90 +/- 0.45 | 13.73 +/- 0.18 | 8.84 +/- 0.06 | 4.32 +/- 0.14 |
| lara | True | 26.66 +/- 0.68 | 24.71 +/- 0.55 | 23.93 +/- 0.15 | 16.86 +/- 0.52 | 7.81 +/- 1.01 |

## Gate

D7 reproduces 44.85 MoF for HuGaDB and 41.66 for LARa. D7 is a provenance reference, not the M1 baseline. The baseline scores above were recomputed from same-checkpoint cache codes using dataset-level Hungarian mapping.

HuGaDB F1@50 changes by +2.86 points without recording standardization and +2.23 with it, below the required +3. LARa MoF changes by -17.30 and -10.72 points respectively, exceeding the permitted 2-point drop. Neither setting passes. One favorable HuGaDB seed does not replace the three-seed aggregate. Milestone 2 is not supported by this gate.

## Calibration and segmentation guards

All 12 cells have lambda=0 and within_2pct=false. Initial counts are approximately 3,200-3,800 versus target 19,143 on HuGaDB and 8,900-10,500 versus target 52,680 on LARa. With frozen candidate costs, increasing a nonnegative per-segment penalty cannot increase the optimal count. The target was therefore unattainable within the implemented search. The runner nevertheless marked these cells complete.

The logged count is the number of internal DP blocks. Adjacent blocks assigned the same label merge when scored. The table gives ratios for both definitions; the final action-run ratio violates the lower guard of 0.5 in every cell.

| Dataset | Recording z-score | Mean DP-block / GT ratio | Mean action-run / GT ratio | Range of mean action duration, seconds |
|---|---|---|---|---|
| hugadb | False | 0.766 | 0.388 | 11.83-14.75 |
| hugadb | True | 0.782 | 0.459 | 10.52-11.18 |
| lara | False | 0.538 | 0.243 | 15.27-15.50 |
| lara | True | 0.555 | 0.416 | 8.51-9.30 |

The ten-second limit applies only to internal DP blocks. Final constant-label runs reach 207.87 seconds on HuGaDB and 120 seconds on LARa. This explains why the block-count guard does not protect the scored segmentation.

A structural property of the implemented cost helps explain merging: for normalized histogram h and nonnegative centroid c, squared distance is 1 + ||c||^2 - 2 sum(c_i sqrt(h_i)), which is convex in h. Consequently, pooling two same-state blocks cannot increase their duration-weighted cost, when the pooled block is admissible. A nonnegative segment penalty further favors pooling. This is an objective property, not proof that it alone caused the measured score losses.

All saved objective traces are nonincreasing and no state is empty in any saved round. Seven cells reach the 20-round cap with relative improvement still above the stopping threshold; five satisfy that threshold. A capped fit should not be called converged. This alone does not determine distance to an optimum.

## Implementation and reporting limitations

- The previous report falsely says no work was omitted. Hard assignments, permutation controls, inductive evaluation and the remaining K/temperature grid have no results here.
- The manifest records 6 h 37 min 57 s, exceeding the six-hour total budget. The runner checks time only before a cell and does not interrupt it at the deadline.
- The log order and duplicates are consistent with overlapping runs, but no run IDs exist to establish their provenance. My earlier reply told you to stop the process without actually stopping it; that handoff was inadequate.
- Saved models contain centroids, prototypes, lambda, fit IDs and traces, but omit scaler/PCA parameters, actual temperature, posteriors and assignments. Exact reuse is consequently limited.
- Final returned centroids have just been updated, while returned partitions were decoded against the preceding centroids. At the iteration cap these are different model states.
- Initialization uses a one-window first chunk and then four-window chunks, not exact fixed one-second chunks. Four LARa windows span 0.96 seconds.
- Saved HuGaDB boundary metadata ends beyond the real frame count for 331 recordings in each cell. Saved label-array lengths and scores are correct; terminal edge metadata is padded.
- The current code materializes all flattened windows and large distance tensors, rather than bounded batches.

## Open questions

- How much of the observed undersegmentation follows from the histogram objective versus initialization and iteration truncation?
- Would a protocol that makes the target count feasible change the boundary metrics? These results do not answer that question.

## What these numbers cannot establish

The scores describe this executed transductive decoder. They do not invalidate the previously measured action information in the units, establish that segment-level decoding generally fails, or isolate the causal source of a dataset difference. Labels enter D7 ground-truth construction, Hungarian mapping, scoring and the GT-based guard; all frozen-encoder results remain transductive.

No experiment was rerun for this audit. Numerical verification is saved in analysis_audit.json.
