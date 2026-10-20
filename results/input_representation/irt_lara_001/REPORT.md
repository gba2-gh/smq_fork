# Input-representation test irt_lara_001


LARa, held-out matched B0 decode (44 evaluation recordings, final endpoint; T2 = Milestone 1 reference cells), label-using probes as diagnostics.

| input | seed | B0 MoF | B0 F1@50 | action probe (continuous) | action probe (code table) | majority | subject probe (chance) |
|---|---|---|---|---|---|---|---|
| t2 | 111 | 39.4 | 18.8 | 65.9 | 62.1 | 43.5 | 99.3 (7.7) |
| t2 | 222 | 38.4 | 20.1 | 65.8 | 62.1 | 43.5 | 99.3 (7.7) |
| t2 | 1538574472 | 38.8 | 20.1 | 65.8 | 62.7 | 43.5 | 99.3 (7.7) |
| t2_bodyscale | 111 | 41.1 | 10.9 | 65.2 | 62.9 | 43.5 | 98.5 (7.7) |
| t2_bodyscale | 222 | 42.1 | 13.9 | 65.1 | 63.5 | 43.5 | 98.6 (7.7) |
| t2_bodyscale | 1538574472 | 38.8 | 10.0 | 65.1 | 63.7 | 43.5 | 98.5 (7.7) |
| t2_centered | 111 | 42.5 | 15.4 | 67.5 | 61.2 | 43.5 | 32.2 (7.7) |
| t2_centered | 222 | 41.9 | 15.7 | 67.5 | 61.2 | 43.5 | 31.9 (7.7) |
| t2_centered | 1538574472 | 41.0 | 16.0 | 67.5 | 61.7 | 43.5 | 31.5 (7.7) |

## Decision (declared rules)

**ORGANISATION_PROBLEM**

- t2_bodyscale − t2: mean MoF +1.83, mean F1@50 -8.09, per-seed MoF +1.75, +3.72, +0.03, F1@50 -7.94, -6.19, -10.14 -> does not pass (bar +3 on both, all seeds positive)
- t2_centered − t2: mean MoF +2.93, mean F1@50 -3.98, per-seed MoF +3.16, +3.45, +2.19, F1@50 -3.44, -4.39, -4.10 -> does not pass (bar +3 on both, all seeds positive)
- supervised probe minus unsupervised MoF (mean over seeds): t2 +27.0, t2_bodyscale +24.4, t2_centered +25.7; threshold 15

The probes use action labels and subject metadata and are upper-bound diagnostics, never a selection signal. t2_centered is not heading-invariant, so a win could come from orientation rather than posture.