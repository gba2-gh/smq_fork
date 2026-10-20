# Milestone 1: segmental clustering

## Status

This report contains only completed transductive cells. Ground-truth labels are used only for the D7 check and evaluation mapping/scoring.

## D7 provenance gate

- hugadb: MoF 44.85; status complete.
- lara: MoF 41.66; status complete.
- lara: MoF 41.66; status complete.

## Completed primary cells

| Dataset | recording z-score | K | temperature | seeds | MoF mean (population SD) | F1@50 mean (population SD) |
|---|---:|---:|---:|---:|---:|---:|
| hugadb | False | 500 | 1 | 4 | 43.75 (1.80) | 26.70 (5.50) |
| hugadb | True | 500 | 1 | 6 | 48.21 (0.64) | 26.53 (0.44) |
| lara | False | 500 | 1 | 6 | 20.09 (0.16) | 4.32 (0.13) |
| lara | True | 500 | 1 | 6 | 26.66 (0.69) | 7.80 (0.99) |

## Not run or invalid

- None.

## Open questions

- Does the primary gate hold after the planned hard-assignment and subject-disjoint evaluations?
- How sensitive are the findings to vocabulary size and temperature?

## What these numbers cannot establish

- These transductive, oracle-mapped measurements do not establish deployment performance or a causal source of any dataset difference.
