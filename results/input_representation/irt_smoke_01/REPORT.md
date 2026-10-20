# Input-representation test irt_smoke_01

SMOKE RUN: plumbing only, not evidence.
LARa, held-out matched B0 decode (44 evaluation recordings, final endpoint; T2 = Milestone 1 reference cells), label-using probes as diagnostics.

| input | seed | B0 MoF | B0 F1@50 | action probe (continuous) | action probe (code table) | majority | subject probe (chance) |
|---|---|---|---|---|---|---|---|
| t2 | 111 | 39.4 | 18.8 | - | - | - | - (-) |
| t2 | 222 | 38.4 | 20.1 | - | - | - | - (-) |
| t2 | 1538574472 | 38.8 | 20.1 | - | - | - | - (-) |
| t2_bodyscale | 111 | 57.0 | 12.1 | 61.6 | 62.9 | 43.5 | 95.3 (7.7) |

## Decision (declared rules)

**not computed (smoke run)**


The probes use action labels and subject metadata and are upper-bound diagnostics, never a selection signal. t2_centered is not heading-invariant, so a win could come from orientation rather than posture.