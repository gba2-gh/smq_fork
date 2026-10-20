# Stage C report -- stage_c_pooled_001

Generated from saved artifacts only (no refitting). Runner version 1.0, stage_b_run `stage_b_003`, stage_a_run `stage_a_pooled_002`.

**Coverage:** 80/80 planned prediction sets scored; 0 fit(s) not run.

## 1. Precommitted decision (plan §4, v1.7)

**Neither the primary bar nor the partial-result clause is met.** Per plan §4 v1.7, the discrete-code-plus-transport direction is not supported by this evidence.

### Primary bar: contextual ASOT vs A-cont and SMQ (F1@50, T, pooled, mean of 3 seeds)

| Dataset | Norm | contextual_asot mean F1@50 | A-cont mean F1@50 | beats A-cont | SMQ F1@50 | beats SMQ | wins vs A-cont | wins vs SMQ |
|---|---|---|---|---|---|---|---|---|
| hugadb | off | 37.00 | 4.85 | YES | 24.30 | YES | 3/3 | 3/3 |
| hugadb | on | 29.16 | 4.93 | YES | 24.30 | YES | 3/3 | 3/3 |
| lara | off | 7.77 | 17.21 | no | 16.39 | no | 0/3 | 0/3 |
| lara | on | 11.34 | 17.05 | no | 16.39 | no | 0/3 | 0/3 |

### Partial-result clause: context-only fragmentation ratio in [0.5, 2]

| Dataset | Norm | context_only pred/GT median run ratio (mean of 3 seeds) | in [0.5, 2] |
|---|---|---|---|
| hugadb | off | 0.07 | no |
| hugadb | on | 0.07 | no |
| lara | off | 0.33 | no |
| lara | on | 0.22 | no |

## 2. Comparison matrix (representation x temporal treatment), setting T, F1@50

| Representation | No temporal | Fixed filter | Temporal transport |
|---|---|---|---|
| Hard K=500 (hugadb/raw) | 2.66 ± 0.27 (n=3) | 6.75 ± 0.69 (n=3) | 3.73 ± 1.74 (n=3) |
| B-code context (hugadb/raw) | 8.46 ± 0.46 (n=3) | 29.69 ± 1.41 (n=3) | 37.00 ± 2.31 (n=3) |
| Hard K=500 (hugadb/norm) | 2.55 ± 0.07 (n=3) | 5.67 ± 0.23 (n=3) | 6.74 ± 1.01 (n=3) |
| B-code context (hugadb/norm) | 6.87 ± 0.43 (n=3) | 25.10 ± 2.33 (n=3) | 29.16 ± 5.45 (n=3) |
| Hard K=500 (lara/raw) | 6.08 ± 0.50 (n=3) | 13.05 ± 1.18 (n=3) | 9.86 ± 0.96 (n=3) |
| B-code context (lara/raw) | 4.96 ± 0.85 (n=3) | 7.56 ± 1.59 (n=3) | 7.77 ± 1.18 (n=3) |
| Hard K=500 (lara/norm) | 4.77 ± 0.88 (n=3) | 13.84 ± 1.94 (n=3) | 7.78 ± 0.67 (n=3) |
| B-code context (lara/norm) | 5.20 ± 0.22 (n=3) | 8.46 ± 0.62 (n=3) | 11.34 ± 0.34 (n=3) |

## 3. Paired seed differences (F1@50, setting T)

**Adding transport to context** (contextual_asot - context_only):

| Dataset | Norm | contextual ASOT - context-only per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | +24.67, +30.94, +30.00 | +28.54 | 3/3 |
| hugadb | on | +18.02, +18.58, +30.29 | +22.30 | 3/3 |
| lara | off | +2.75, +3.76, +1.93 | +2.81 | 3/3 |
| lara | on | +6.34, +6.05, +6.02 | +6.14 | 3/3 |

**Transport vs the fixed smoother on context** (contextual_asot - context_only_filter):

| Dataset | Norm | contextual ASOT - context-only + filter per seed | mean | wins |
|---|---|---|---|---|
| hugadb | off | +2.42, +11.21, +8.28 | +7.30 | 3/3 |
| hugadb | on | +2.07, -2.13, +12.26 | +4.07 | 2/3 |
| lara | off | -0.87, +1.45, +0.05 | +0.21 | 2/3 |
| lara | on | +4.13, +2.24, +2.26 | +2.88 | 3/3 |

## 4. All five metrics (setting T)

**MoF**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 39.04 ± 1.03 (n=3) | 42.53 ± 0.91 (n=3) | 41.38 ± 3.71 (n=3) |
| hugadb | on | 36.00 ± 3.33 (n=3) | 39.18 ± 4.04 (n=3) | 37.08 ± 1.93 (n=3) |
| lara | off | 19.97 ± 1.36 (n=3) | 20.15 ± 1.43 (n=3) | 21.13 ± 1.89 (n=3) |
| lara | on | 22.03 ± 0.72 (n=3) | 22.35 ± 0.79 (n=3) | 25.59 ± 0.42 (n=3) |

**Edit**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 16.16 ± 1.11 (n=3) | 33.03 ± 0.96 (n=3) | 37.15 ± 2.07 (n=3) |
| hugadb | on | 14.22 ± 0.61 (n=3) | 32.90 ± 0.56 (n=3) | 38.80 ± 2.63 (n=3) |
| lara | off | 22.98 ± 0.54 (n=3) | 26.92 ± 0.69 (n=3) | 23.90 ± 0.18 (n=3) |
| lara | on | 22.01 ± 0.30 (n=3) | 30.25 ± 0.53 (n=3) | 28.41 ± 1.16 (n=3) |

**F1@10**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 13.99 ± 0.66 (n=3) | 35.99 ± 0.91 (n=3) | 43.12 ± 2.24 (n=3) |
| hugadb | on | 13.61 ± 0.71 (n=3) | 35.43 ± 2.09 (n=3) | 40.72 ± 4.01 (n=3) |
| lara | off | 15.48 ± 0.59 (n=3) | 20.09 ± 1.16 (n=3) | 21.88 ± 1.17 (n=3) |
| lara | on | 16.28 ± 0.20 (n=3) | 23.24 ± 0.30 (n=3) | 26.39 ± 0.43 (n=3) |

**F1@25**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 10.97 ± 0.42 (n=3) | 33.12 ± 1.19 (n=3) | 40.56 ± 2.16 (n=3) |
| hugadb | on | 9.54 ± 0.74 (n=3) | 30.75 ± 2.67 (n=3) | 35.84 ± 5.02 (n=3) |
| lara | off | 10.16 ± 0.92 (n=3) | 14.67 ± 1.75 (n=3) | 15.76 ± 1.66 (n=3) |
| lara | on | 10.70 ± 0.09 (n=3) | 16.66 ± 0.61 (n=3) | 20.63 ± 0.29 (n=3) |

**F1@50**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 8.46 ± 0.46 (n=3) | 29.69 ± 1.41 (n=3) | 37.00 ± 2.31 (n=3) |
| hugadb | on | 6.87 ± 0.43 (n=3) | 25.10 ± 2.33 (n=3) | 29.16 ± 5.45 (n=3) |
| lara | off | 4.96 ± 0.85 (n=3) | 7.56 ± 1.59 (n=3) | 7.77 ± 1.18 (n=3) |
| lara | on | 5.20 ± 0.22 (n=3) | 8.46 ± 0.62 (n=3) | 11.34 ± 0.34 (n=3) |

## 5. Fragmentation diagnostics (setting T)

**pred/GT runs**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 6.119 ± 0.106 (n=3) | 1.739 ± 0.112 (n=3) | 0.794 ± 0.014 (n=3) |
| hugadb | on | 6.420 ± 0.223 (n=3) | 1.849 ± 0.111 (n=3) | 0.901 ± 0.050 (n=3) |
| lara | off | 2.075 ± 0.022 (n=3) | 0.830 ± 0.001 (n=3) | 0.446 ± 0.007 (n=3) |
| lara | on | 3.041 ± 0.165 (n=3) | 1.123 ± 0.049 (n=3) | 0.592 ± 0.009 (n=3) |

**median run (s)**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 0.250 ± 0.000 (n=3) | 1.500 ± 0.204 (n=3) | 3.917 ± 0.118 (n=3) |
| hugadb | on | 0.250 ± 0.000 (n=3) | 1.917 ± 0.118 (n=3) | 4.000 ± 0.000 (n=3) |
| lara | off | 0.720 ± 0.000 (n=3) | 2.640 ± 0.000 (n=3) | 5.680 ± 0.113 (n=3) |
| lara | on | 0.480 ± 0.000 (n=3) | 2.240 ± 0.113 (n=3) | 4.800 ± 0.000 (n=3) |

**states used**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) |
| hugadb | on | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) | 10.000 ± 0.000 (n=3) |
| lara | off | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) |
| lara | on | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) | 8.000 ± 0.000 (n=3) |

**largest state**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 0.160 ± 0.003 (n=3) | 0.184 ± 0.004 (n=3) | 0.273 ± 0.025 (n=3) |
| hugadb | on | 0.149 ± 0.006 (n=3) | 0.170 ± 0.006 (n=3) | 0.145 ± 0.020 (n=3) |
| lara | off | 0.138 ± 0.003 (n=3) | 0.140 ± 0.005 (n=3) | 0.177 ± 0.011 (n=3) |
| lara | on | 0.182 ± 0.015 (n=3) | 0.186 ± 0.016 (n=3) | 0.213 ± 0.010 (n=3) |

**MI(state;subject)**

| Dataset | Norm | context-only | context-only + filter | contextual ASOT |
|---|---|---|---|---|
| hugadb | off | 0.319 ± 0.039 (n=3) | 0.384 ± 0.041 (n=3) | 0.361 ± 0.014 (n=3) |
| hugadb | on | 0.083 ± 0.024 (n=3) | 0.119 ± 0.035 (n=3) | 0.151 ± 0.012 (n=3) |
| lara | off | 0.271 ± 0.039 (n=3) | 0.294 ± 0.039 (n=3) | 0.294 ± 0.025 (n=3) |
| lara | on | 0.023 ± 0.007 (n=3) | 0.027 ± 0.007 (n=3) | 0.028 ± 0.007 (n=3) |

## 6. Contextual-order diagnostic (seed 111, pooled only)

Stage B's frozen checkpoint fed permuted codes, embeddings restored to original order, prototypes refit, transport with the original chronological kernel.

| Dataset | Norm | Setting | contextual_asot (seed 111) F1@50 | order-diagnostic F1@50 |
|---|---|---|---|---|
| hugadb | off | E | 33.35 | 16.99 |
| hugadb | off | T | 33.78 | 16.57 |
| hugadb | on | E | 25.38 | 15.98 |
| hugadb | on | T | 24.61 | 15.54 |
| lara | off | E | 8.95 | 8.91 |
| lara | off | T | 8.90 | 9.49 |
| lara | on | E | 11.66 | 14.04 |
| lara | on | T | 11.75 | 13.91 |

## 7. `stage_b_003` summary

| Cell | stopped_by | steps_run | model_acc | copy_acc | margin |
|---|---|---|---|---|---|
| hugadb_norm_pooled_seed111 | patience | 3600 | 0.301 | 0.281 | +0.020 |
| hugadb_norm_pooled_seed1538574472 | patience | 4100 | 0.292 | 0.264 | +0.027 |
| hugadb_norm_pooled_seed222 | patience | 4500 | 0.277 | 0.261 | +0.016 |
| hugadb_raw_pooled_seed111 | step_cap | 5000 | 0.511 | 0.518 | -0.007 |
| hugadb_raw_pooled_seed1538574472 | step_cap | 5000 | 0.542 | 0.516 | +0.026 |
| hugadb_raw_pooled_seed222 | step_cap | 5000 | 0.511 | 0.515 | -0.005 |
| lara_norm_pooled_seed111 | step_cap | 5000 | 0.279 | 0.297 | -0.018 |
| lara_norm_pooled_seed1538574472 | step_cap | 5000 | 0.290 | 0.310 | -0.021 |
| lara_norm_pooled_seed222 | step_cap | 5000 | 0.283 | 0.298 | -0.015 |
| lara_raw_pooled_seed111 | step_cap | 5000 | 0.494 | 0.535 | -0.041 |
| lara_raw_pooled_seed1538574472 | step_cap | 5000 | 0.484 | 0.521 | -0.037 |
| lara_raw_pooled_seed222 | step_cap | 5000 | 0.473 | 0.522 | -0.049 |

## 8. Limitations

- No K=C contextualizer exists, so beating A1-KC cannot isolate vocabulary size from context.
- The frozen encoder is transductive; per-recording normalization and inference are offline.
- Equal T/E coefficients do not calibrate the 128-d contextual cosine cost the same way as the 64-d PCA cosine or the categorical cost.
- Pooled protocol only; subject-disjoint Stage C was not run.
- One selected contextualizer, chosen by a label-free tuning criterion on one seed.
