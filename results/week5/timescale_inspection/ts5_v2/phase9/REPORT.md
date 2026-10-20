# Time-scale inspection, phase 9: iterative contextual target refinement — ts5_v2

Status `finished`. Rules declared in PHASE9_LEARNED_UNITS.md (Revision 3) before the run. Primary metric F1@50 at the last rung, mask span 2, paired by seed; the bar is a mean gain >= 3 with every seed in the same direction, read only with three seeds. 'Within the bar' means comparable performance, not equivalence. This phase tests contextual target refinement; it cannot show discovery of local motion units (every arm reads T2's PCA-64 summaries and I2's codes are contextual).

## Gates

| gate | status | detail |
|---|---|---|
| g3 | passed |  |
| g9a | passed | {'selected_step': True, 'history': True, 'tensors': True, 'ladder': True} |
| g9b | passed | {'context_continuous_masked_input_ignored': True, 'context_continuous_visible_vector_effect': True, 'context_quantized_masked_input_ignored': True, 'context_quantized_visible_vector_effect': True, 'context_continuous_training_effect': True, 'context_quantized_ |
| g9c | passed | {'code_input_equals_phase5_extractor': True, 'centroid_input_equals_phase5_extractor': True, 'continuous_input_equals_independent_extractor': True, 'layers_shapes': True, 'layers_finite': True, 'layer4_normalizes_to_extractor_output': True} |
| g9d | passed | {'permutation_invariant': True, 'independent_near_zero': True, 'shuffled_near_zero': True} |
| g9e | passed | {'grouping_equals_t2': True, 'lookup_equals_per_code_centres': True, 'dims': True} |
| g9f_babel2_111 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_babel2_1538574472 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_babel2_222 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_babel3_111 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_babel3_1538574472 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_babel3_222 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_lara_111 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_lara_1538574472 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9f_lara_222 | passed | {'lookup_dim': True, 'mean_dim': True, 'rms_scalar': True, 'rms_equal_base': True, 'codes_in_range': True, 'grouping_len': True, 'model_input_dim': True} |
| g9g | passed | {'batch_plan_identical': True, 'backbone_and_head_init_identical': True, 'projection_init_identical': True} |


## Cells (F1@50, span 2). `tg` = T2-code grouping initializes OT, `ig` = I2-code grouping

| dataset | arm | grouping | per seed | n | mean | pred segs/rec | gt segs/rec | same-recording NN share |
|---|---|---|---|---|---|---|---|---|
| babel2 | R (T2 ids) | t2 (native) | 111:31.5 / 222:30.0 / 1538574472:30.5 | 3 | 30.68 | 3.0 | 3.7 | — |
| babel2 | G (geometry-init ids) | tg | 111:31.0 / 222:19.4 / 1538574472:36.1 | 3 | 28.82 | 2.9 | 3.7 | 0.540 |
| babel2 | I1q (centroid) | tg | 111:20.8 / 222:25.0 / 1538574472:32.9 | 3 | 26.25 | 2.9 | 3.7 | 0.537 |
| babel2 | I1 (continuous) | tg | 111:30.5 / 222:20.1 / 1538574472:31.8 | 3 | 27.47 | 3.1 | 3.7 | 0.643 |
| babel2 | I1 (continuous) | ig | 111:28.4 / 222:26.4 / 1538574472:24.4 | 3 | 26.42 | 3.0 | 3.7 | 0.643 |
| babel2 | I2 (refined targets) | tg | 111:28.8 / 222:15.7 / 1538574472:31.9 | 3 | 25.46 | 3.0 | 3.7 | 0.688 |
| babel2 | I2 (refined targets) | ig | 111:28.6 / 222:30.1 / 1538574472:22.1 | 3 | 26.95 | 3.0 | 3.7 | 0.688 |
| babel3 | R (T2 ids) | t2 (native) | 111:20.1 / 222:22.8 / 1538574472:26.8 | 3 | 23.23 | 3.3 | 3.7 | — |
| babel3 | G (geometry-init ids) | tg | 111:26.4 / 222:15.0 / 1538574472:29.8 | 3 | 23.75 | 3.5 | 3.7 | 0.693 |
| babel3 | I1q (centroid) | tg | 111:28.0 / 222:16.0 / 1538574472:28.6 | 3 | 24.23 | 3.4 | 3.7 | 0.700 |
| babel3 | I1 (continuous) | tg | 111:23.0 / 222:15.8 / 1538574472:26.7 | 3 | 21.84 | 3.2 | 3.7 | 0.782 |
| babel3 | I1 (continuous) | ig | 111:25.0 / 222:17.2 / 1538574472:27.6 | 3 | 23.29 | 3.2 | 3.7 | 0.782 |
| babel3 | I2 (refined targets) | tg | 111:23.8 / 222:14.9 / 1538574472:27.0 | 3 | 21.90 | 3.0 | 3.7 | 0.834 |
| babel3 | I2 (refined targets) | ig | 111:25.4 / 222:17.0 / 1538574472:27.3 | 3 | 23.25 | 2.9 | 3.7 | 0.834 |
| lara | R (T2 ids) | t2 (native) | 111:23.4 / 222:22.6 / 1538574472:26.0 | 3 | 24.02 | 24.7 | 32.1 | — |
| lara | G (geometry-init ids) | tg | 111:24.1 / 222:21.8 / 1538574472:26.7 | 3 | 24.20 | 24.7 | 32.1 | 0.196 |
| lara | I1q (centroid) | tg | 111:23.5 / 222:20.5 / 1538574472:26.0 | 3 | 23.34 | 24.6 | 32.1 | 0.194 |
| lara | I1 (continuous) | tg | 111:22.5 / 222:22.1 / 1538574472:22.3 | 3 | 22.33 | 25.8 | 32.1 | 0.280 |
| lara | I1 (continuous) | ig | 111:23.8 / 222:24.1 / 1538574472:27.8 | 3 | 25.23 | 27.2 | 32.1 | 0.280 |
| lara | I2 (refined targets) | tg | 111:24.1 / 222:22.0 / 1538574472:26.2 | 3 | 24.11 | 27.3 | 32.1 | 0.279 |
| lara | I2 (refined targets) | ig | 111:23.4 / 222:25.8 / 1538574472:28.5 | 3 | 25.90 | 28.0 | 32.1 | 0.279 |

## Comparisons (A minus B, paired by seed)

### Q1a: G vs R: initial geometry (same parameterization)

- **babel2**: -1.86 (-0.5 / -10.6 / +5.6) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +0.52 (+6.4 / -7.8 / +3.0) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +0.18 (+0.6 / -0.8 / +0.8) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q1b: I1q vs G: free table vs input tied to the centroid (same geometric start)

- **babel2**: -2.57 (-10.1 / +5.7 / -3.2) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +0.48 (+1.6 / +1.0 / -1.1) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: -0.86 (-0.5 / -1.3 / -0.7) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q1c: I1 vs I1q: detail inside the K-means cell (actual features vs centroid)

- **babel2**: +1.22 (+9.7 / -4.9 / -1.1) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: -2.39 (-5.1 / -0.2 / -1.9) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: -1.01 (-1.0 / +1.6 / -3.7) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q2: I2 vs I1, both with the I2-code grouping: do refined contextual targets help? (primary)

- **babel2**: +0.53 (+0.2 / +3.6 / -2.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: -0.04 (+0.4 / -0.2 / -0.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +0.67 (-0.5 / +1.7 / +0.7) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q2r: I2 vs I1, both with the T2-code grouping (robustness)

- **babel2**: -2.01 (-1.7 / -4.4 / +0.1) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +0.06 (+0.8 / -0.9 / +0.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +1.78 (+1.6 / -0.1 / +3.9) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q3a (descriptive): I2 (I2-code grouping) vs R (descriptive: the total effect)

- **babel2**: -3.73 (-2.8 / +0.1 / -8.4) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +0.02 (+5.3 / -5.8 / +0.5) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +1.88 (-0.1 / +3.2 / +2.5) over 3 seeds -> within the bar: comparable performance, not equivalence

### Q3b (descriptive): I2 (T2-code grouping) vs R (descriptive)

- **babel2**: -5.22 (-2.7 / -14.3 / +1.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: -1.33 (+3.7 / -7.9 / +0.2) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +0.09 (+0.7 / -0.6 / +0.2) over 3 seeds -> within the bar: comparable performance, not equivalence

### D1 (descriptive): I1: I2-code grouping vs T2-code grouping (OT-initialization effect, descriptive)

- **babel2**: -1.06 (-2.1 / +6.3 / -7.4) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +1.45 (+2.0 / +1.4 / +0.9) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +2.90 (+1.3 / +1.9 / +5.5) over 3 seeds -> within the bar: comparable performance, not equivalence

### D2 (descriptive): I2: I2-code grouping vs T2-code grouping (OT-initialization effect, descriptive)

- **babel2**: +1.49 (-0.2 / +14.4 / -9.8) over 3 seeds -> within the bar: comparable performance, not equivalence
- **babel3**: +1.35 (+1.6 / +2.1 / +0.3) over 3 seeds -> within the bar: comparable performance, not equivalence
- **lara**: +1.79 (-0.8 / +3.8 / +2.4) over 3 seeds -> within the bar: comparable performance, not equivalence

## Every metric

| id | dataset | A − B | metric | Δ (per seed) | n | reading |
|---|---|---|---|---|---|---|
| Q1a | babel2 | g/tg − R/tg | MoF | +1.24 (-3.0 / +0.4 / +6.4) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel2 | g/tg − R/tg | Edit | +0.77 (-3.6 / -2.2 / +8.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel2 | g/tg − R/tg | F1@10 | -1.01 (-0.7 / -10.1 / +7.8) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel2 | g/tg − R/tg | F1@25 | -2.00 (-1.2 / -12.5 / +7.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel2 | g/tg − R/tg | F1@50 | -1.86 (-0.5 / -10.6 / +5.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel3 | g/tg − R/tg | MoF | -0.42 (+0.9 / -1.6 / -0.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel3 | g/tg − R/tg | Edit | -4.29 (+2.8 / -15.7 / +0.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel3 | g/tg − R/tg | F1@10 | -4.79 (+1.7 / -16.0 / -0.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel3 | g/tg − R/tg | F1@25 | -3.00 (+2.9 / -13.2 / +1.2) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | babel3 | g/tg − R/tg | F1@50 | +0.52 (+6.4 / -7.8 / +3.0) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | lara | g/tg − R/tg | MoF | -0.05 (-0.6 / +0.9 / -0.4) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | lara | g/tg − R/tg | Edit | -0.99 (-0.9 / -4.0 / +1.9) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | lara | g/tg − R/tg | F1@10 | -2.18 (-1.5 / -3.9 / -1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | lara | g/tg − R/tg | F1@25 | -2.27 (-1.7 / -3.4 / -1.7) | 3 | within the bar: comparable performance, not equivalence |
| Q1a | lara | g/tg − R/tg | F1@50 | +0.18 (+0.6 / -0.8 / +0.8) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel2 | i1q/tg − g/tg | MoF | -4.95 (-7.5 / -0.1 / -7.2) | 3 | **B better (reverse passes the bar)** |
| Q1b | babel2 | i1q/tg − g/tg | Edit | -3.46 (-3.5 / -1.7 / -5.2) | 3 | **B better (reverse passes the bar)** |
| Q1b | babel2 | i1q/tg − g/tg | F1@10 | -4.09 (-10.1 / +3.1 / -5.3) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel2 | i1q/tg − g/tg | F1@25 | -4.04 (-11.8 / +5.0 / -5.3) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel2 | i1q/tg − g/tg | F1@50 | -2.57 (-10.1 / +5.7 / -3.2) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel3 | i1q/tg − g/tg | MoF | -0.01 (-0.8 / +0.7 / +0.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel3 | i1q/tg − g/tg | Edit | +0.71 (+3.3 / -0.1 / -1.0) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel3 | i1q/tg − g/tg | F1@10 | +0.58 (+1.1 / +0.6 / +0.0) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel3 | i1q/tg − g/tg | F1@25 | +0.24 (+1.4 / -0.4 / -0.3) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | babel3 | i1q/tg − g/tg | F1@50 | +0.48 (+1.6 / +1.0 / -1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | lara | i1q/tg − g/tg | MoF | -0.58 (+1.0 / -1.2 / -1.5) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | lara | i1q/tg − g/tg | Edit | -0.32 (+0.6 / -1.2 / -0.4) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | lara | i1q/tg − g/tg | F1@10 | -1.16 (-0.4 / -1.9 / -1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | lara | i1q/tg − g/tg | F1@25 | -1.06 (-0.2 / -1.8 / -1.2) | 3 | within the bar: comparable performance, not equivalence |
| Q1b | lara | i1q/tg − g/tg | F1@50 | -0.86 (-0.5 / -1.3 / -0.7) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel2 | i1/tg − i1q/tg | MoF | +3.39 (+7.9 / +2.8 / -0.5) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel2 | i1/tg − i1q/tg | Edit | +1.02 (+1.4 / +2.7 / -1.0) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel2 | i1/tg − i1q/tg | F1@10 | +1.70 (+6.5 / -1.1 / -0.3) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel2 | i1/tg − i1q/tg | F1@25 | +1.71 (+10.1 / -4.4 / -0.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel2 | i1/tg − i1q/tg | F1@50 | +1.22 (+9.7 / -4.9 / -1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel3 | i1/tg − i1q/tg | MoF | +1.88 (+1.5 / +1.5 / +2.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel3 | i1/tg − i1q/tg | Edit | -3.51 (-8.1 / +0.4 / -2.8) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel3 | i1/tg − i1q/tg | F1@10 | -2.06 (-4.1 / +0.6 / -2.7) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel3 | i1/tg − i1q/tg | F1@25 | -2.46 (-4.4 / +0.6 / -3.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | babel3 | i1/tg − i1q/tg | F1@50 | -2.39 (-5.1 / -0.2 / -1.9) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | lara | i1/tg − i1q/tg | MoF | -0.54 (-0.4 / +0.3 / -1.5) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | lara | i1/tg − i1q/tg | Edit | -0.65 (-1.9 / +4.5 / -4.6) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | lara | i1/tg − i1q/tg | F1@10 | -0.49 (-1.1 / +4.1 / -4.5) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | lara | i1/tg − i1q/tg | F1@25 | -0.58 (-0.8 / +3.2 / -4.1) | 3 | within the bar: comparable performance, not equivalence |
| Q1c | lara | i1/tg − i1q/tg | F1@50 | -1.01 (-1.0 / +1.6 / -3.7) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | i2/ig − i1/ig | MoF | -0.27 (+0.5 / -4.6 / +3.2) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | i2/ig − i1/ig | Edit | -2.06 (-2.3 / -7.3 / +3.4) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | i2/ig − i1/ig | F1@10 | -0.93 (-1.7 / -2.9 / +1.8) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | i2/ig − i1/ig | F1@25 | +0.71 (+0.7 / +1.2 / +0.2) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel2 | i2/ig − i1/ig | F1@50 | +0.53 (+0.2 / +3.6 / -2.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | i2/ig − i1/ig | MoF | -2.16 (-4.4 / -2.3 / +0.2) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | i2/ig − i1/ig | Edit | +1.30 (+1.4 / +2.0 / +0.5) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | i2/ig − i1/ig | F1@10 | +0.36 (-0.4 / +1.1 / +0.4) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | i2/ig − i1/ig | F1@25 | +0.56 (+0.8 / -0.4 / +1.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | babel3 | i2/ig − i1/ig | F1@50 | -0.04 (+0.4 / -0.2 / -0.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | i2/ig − i1/ig | MoF | -0.66 (-2.7 / +0.6 / +0.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | i2/ig − i1/ig | Edit | +0.74 (-1.0 / +1.3 / +1.8) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | i2/ig − i1/ig | F1@10 | +2.00 (-1.6 / +4.5 / +3.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | i2/ig − i1/ig | F1@25 | +1.44 (-1.2 / +3.6 / +1.9) | 3 | within the bar: comparable performance, not equivalence |
| Q2 | lara | i2/ig − i1/ig | F1@50 | +0.67 (-0.5 / +1.7 / +0.7) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel2 | i2/tg − i1/tg | MoF | -1.58 (-2.1 / -4.6 / +1.9) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel2 | i2/tg − i1/tg | Edit | -1.40 (-2.0 / -3.3 / +1.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel2 | i2/tg − i1/tg | F1@10 | -1.18 (-1.0 / -4.3 / +1.7) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel2 | i2/tg − i1/tg | F1@25 | -1.55 (-1.1 / -4.7 / +1.2) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel2 | i2/tg − i1/tg | F1@50 | -2.01 (-1.7 / -4.4 / +0.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel3 | i2/tg − i1/tg | MoF | -1.30 (-1.9 / -0.6 / -1.4) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel3 | i2/tg − i1/tg | Edit | +2.40 (+1.7 / +4.2 / +1.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel3 | i2/tg − i1/tg | F1@10 | +1.03 (+0.5 / +3.1 / -0.4) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel3 | i2/tg − i1/tg | F1@25 | +0.62 (+1.2 / +1.0 / -0.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | babel3 | i2/tg − i1/tg | F1@50 | +0.06 (+0.8 / -0.9 / +0.3) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | lara | i2/tg − i1/tg | MoF | -0.66 (-1.0 / -2.0 / +1.0) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | lara | i2/tg − i1/tg | Edit | +0.91 (+2.1 / -0.9 / +1.6) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | lara | i2/tg − i1/tg | F1@10 | +1.42 (+1.8 / -1.5 / +3.9) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | lara | i2/tg − i1/tg | F1@25 | +1.66 (+1.7 / -0.9 / +4.1) | 3 | within the bar: comparable performance, not equivalence |
| Q2r | lara | i2/tg − i1/tg | F1@50 | +1.78 (+1.6 / -0.1 / +3.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel2 | i2/ig − R/tg | MoF | -4.08 (-3.3 / +1.7 / -10.6) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel2 | i2/ig − R/tg | Edit | -1.86 (-5.2 / -1.0 / +0.6) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel2 | i2/ig − R/tg | F1@10 | -2.52 (-3.8 / -1.7 / -2.1) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel2 | i2/ig − R/tg | F1@25 | -3.73 (-3.9 / -1.3 / -6.0) | 3 | **B better (reverse passes the bar)** |
| Q3a | babel2 | i2/ig − R/tg | F1@50 | -3.73 (-2.8 / +0.1 / -8.4) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel3 | i2/ig − R/tg | MoF | +0.57 (-1.8 / +0.3 / +3.2) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel3 | i2/ig − R/tg | Edit | -4.12 (+1.0 / -11.8 / -1.5) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel3 | i2/ig − R/tg | F1@10 | -4.73 (+0.2 / -12.4 / -2.0) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel3 | i2/ig − R/tg | F1@25 | -3.31 (+3.0 / -11.5 / -1.4) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | babel3 | i2/ig − R/tg | F1@50 | +0.02 (+5.3 / -5.8 / +0.5) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | lara | i2/ig − R/tg | MoF | +1.24 (-1.6 / +4.0 / +1.4) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | lara | i2/ig − R/tg | Edit | -0.01 (-2.3 / -0.4 / +2.7) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | lara | i2/ig − R/tg | F1@10 | +0.79 (-2.8 / +0.6 / +4.6) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | lara | i2/ig − R/tg | F1@25 | +0.84 (-2.2 / +1.2 / +3.5) | 3 | within the bar: comparable performance, not equivalence |
| Q3a | lara | i2/ig − R/tg | F1@50 | +1.88 (-0.1 / +3.2 / +2.5) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel2 | i2/tg − R/tg | MoF | -1.90 (-4.7 / -1.6 / +0.6) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel2 | i2/tg − R/tg | Edit | -3.08 (-7.7 / -4.5 / +3.0) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel2 | i2/tg − R/tg | F1@10 | -4.58 (-5.3 / -12.3 / +3.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel2 | i2/tg − R/tg | F1@25 | -5.88 (-4.0 / -16.6 / +2.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel2 | i2/tg − R/tg | F1@50 | -5.22 (-2.7 / -14.3 / +1.3) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel3 | i2/tg − R/tg | MoF | +0.16 (-0.3 / +0.0 / +0.7) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel3 | i2/tg − R/tg | Edit | -4.68 (-0.3 / -11.3 / -2.4) | 3 | **B better (reverse passes the bar)** |
| Q3b | babel3 | i2/tg − R/tg | F1@10 | -5.24 (-0.9 / -11.7 / -3.2) | 3 | **B better (reverse passes the bar)** |
| Q3b | babel3 | i2/tg − R/tg | F1@25 | -4.60 (+1.1 / -12.0 / -2.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | babel3 | i2/tg − R/tg | F1@50 | -1.33 (+3.7 / -7.9 / +0.2) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | lara | i2/tg − R/tg | MoF | -1.84 (-1.1 / -1.9 / -2.4) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | lara | i2/tg − R/tg | Edit | -1.05 (-0.0 / -1.6 / -1.5) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | lara | i2/tg − R/tg | F1@10 | -2.41 (-1.2 / -3.2 / -2.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | lara | i2/tg − R/tg | F1@25 | -2.24 (-1.0 / -2.9 / -2.9) | 3 | within the bar: comparable performance, not equivalence |
| Q3b | lara | i2/tg − R/tg | F1@50 | +0.09 (+0.7 / -0.6 / +0.2) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | i1/ig − i1/tg | MoF | -3.50 (-1.2 / +3.2 / -12.5) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | i1/ig − i1/tg | Edit | +1.88 (+2.9 / +7.5 / -4.7) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | i1/ig − i1/tg | F1@10 | +1.81 (+2.2 / +9.3 / -6.1) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | i1/ig − i1/tg | F1@25 | -0.11 (-1.8 / +9.3 / -7.9) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel2 | i1/ig − i1/tg | F1@50 | -1.06 (-2.1 / +6.3 / -7.4) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel3 | i1/ig − i1/tg | MoF | +1.28 (+0.9 / +2.0 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel3 | i1/ig − i1/tg | Edit | +1.67 (+1.6 / +1.6 / +1.7) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel3 | i1/ig − i1/tg | F1@10 | +1.18 (+1.9 / +1.2 / +0.4) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel3 | i1/ig − i1/tg | F1@25 | +1.36 (+2.3 / +1.9 / -0.1) | 3 | within the bar: comparable performance, not equivalence |
| D1 | babel3 | i1/ig − i1/tg | F1@50 | +1.45 (+2.0 / +1.4 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| D1 | lara | i1/ig − i1/tg | MoF | +3.08 (+1.1 / +3.4 / +4.8) | 3 | **A better (passes the bar)** |
| D1 | lara | i1/ig − i1/tg | Edit | +1.20 (+0.8 / -1.0 / +3.9) | 3 | within the bar: comparable performance, not equivalence |
| D1 | lara | i1/ig − i1/tg | F1@10 | +2.61 (+1.8 / -2.3 / +8.3) | 3 | within the bar: comparable performance, not equivalence |
| D1 | lara | i1/ig − i1/tg | F1@25 | +3.31 (+1.7 / -0.4 / +8.6) | 3 | within the bar: comparable performance, not equivalence |
| D1 | lara | i1/ig − i1/tg | F1@50 | +2.90 (+1.3 / +1.9 / +5.5) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | i2/ig − i2/tg | MoF | -2.18 (+1.4 / +3.2 / -11.2) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | i2/ig − i2/tg | Edit | +1.23 (+2.5 / +3.5 / -2.4) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | i2/ig − i2/tg | F1@10 | +2.06 (+1.5 / +10.7 / -6.0) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | i2/ig − i2/tg | F1@25 | +2.15 (+0.1 / +15.2 / -8.9) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel2 | i2/ig − i2/tg | F1@50 | +1.49 (-0.2 / +14.4 / -9.8) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | i2/ig − i2/tg | MoF | +0.42 (-1.5 / +0.3 / +2.5) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | i2/ig − i2/tg | Edit | +0.57 (+1.3 / -0.6 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | i2/ig − i2/tg | F1@10 | +0.51 (+1.1 / -0.7 / +1.2) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | i2/ig − i2/tg | F1@25 | +1.30 (+1.9 / +0.5 / +1.5) | 3 | within the bar: comparable performance, not equivalence |
| D2 | babel3 | i2/ig − i2/tg | F1@50 | +1.35 (+1.6 / +2.1 / +0.3) | 3 | within the bar: comparable performance, not equivalence |
| D2 | lara | i2/ig − i2/tg | MoF | +3.09 (-0.5 / +5.9 / +3.8) | 3 | within the bar: comparable performance, not equivalence |
| D2 | lara | i2/ig − i2/tg | Edit | +1.03 (-2.3 / +1.2 / +4.2) | 3 | within the bar: comparable performance, not equivalence |
| D2 | lara | i2/ig − i2/tg | F1@10 | +3.20 (-1.7 / +3.7 / +7.5) | 3 | within the bar: comparable performance, not equivalence |
| D2 | lara | i2/ig − i2/tg | F1@25 | +3.09 (-1.2 / +4.1 / +6.4) | 3 | within the bar: comparable performance, not equivalence |
| D2 | lara | i2/ig − i2/tg | F1@50 | +1.79 (-0.8 / +3.8 / +2.4) | 3 | within the bar: comparable performance, not equivalence |

## Training (validation loss is comparable only within an arm; masked accuracy is the difficulty of the prediction task)

| dataset | arm | seed | selected step | val loss | masked-code val accuracy | seconds |
|---|---|---|---|---|---|---|
| babel2 | g | 111 | 1200 | 2.953 | 0.304 | 89 |
| babel2 | g | 1538574472 | 1000 | 3.002 | 0.290 | 90 |
| babel2 | g | 222 | 800 | 3.183 | 0.363 | 91 |
| babel2 | i1 | 111 | 1200 | 2.899 | 0.331 | 93 |
| babel2 | i1 | 1538574472 | 1100 | 2.976 | 0.318 | 97 |
| babel2 | i1 | 222 | 900 | 3.086 | 0.367 | 96 |
| babel2 | i1q | 111 | 1200 | 2.961 | 0.291 | 93 |
| babel2 | i1q | 1538574472 | 1300 | 3.010 | 0.330 | 96 |
| babel2 | i1q | 222 | 1000 | 3.165 | 0.349 | 95 |
| babel2 | i2 | 111 | 1400 | 3.827 | 0.291 | 96 |
| babel2 | i2 | 1538574472 | 1500 | 3.441 | 0.337 | 95 |
| babel2 | i2 | 222 | 1100 | 3.921 | 0.318 | 95 |
| babel3 | g | 111 | 1100 | 4.006 | 0.202 | 96 |
| babel3 | g | 1538574472 | 800 | 3.777 | 0.244 | 98 |
| babel3 | g | 222 | 1000 | 3.957 | 0.217 | 97 |
| babel3 | i1 | 111 | 700 | 4.214 | 0.175 | 102 |
| babel3 | i1 | 1538574472 | 800 | 3.922 | 0.193 | 101 |
| babel3 | i1 | 222 | 1000 | 4.108 | 0.150 | 102 |
| babel3 | i1q | 111 | 1000 | 4.003 | 0.214 | 104 |
| babel3 | i1q | 1538574472 | 800 | 3.814 | 0.231 | 101 |
| babel3 | i1q | 222 | 1000 | 3.986 | 0.183 | 105 |
| babel3 | i2 | 111 | 1000 | 5.314 | 0.083 | 100 |
| babel3 | i2 | 1538574472 | 600 | 5.314 | 0.098 | 97 |
| babel3 | i2 | 222 | 700 | 5.249 | 0.150 | 100 |
| lara | g | 111 | 4800 | 2.503 | 0.385 | 143 |
| lara | g | 1538574472 | 5000 | 2.441 | 0.393 | 147 |
| lara | g | 222 | 5000 | 2.444 | 0.387 | 149 |
| lara | i1 | 111 | 4600 | 2.202 | 0.418 | 155 |
| lara | i1 | 1538574472 | 4700 | 2.184 | 0.414 | 156 |
| lara | i1 | 222 | 4800 | 2.189 | 0.413 | 154 |
| lara | i1q | 111 | 4800 | 2.498 | 0.384 | 158 |
| lara | i1q | 1538574472 | 5000 | 2.439 | 0.391 | 162 |
| lara | i1q | 222 | 5000 | 2.447 | 0.386 | 155 |
| lara | i2 | 111 | 4900 | 1.901 | 0.463 | 156 |
| lara | i2 | 1538574472 | 4900 | 1.918 | 0.448 | 160 |
| lara | i2 | 222 | 5000 | 1.793 | 0.475 | 157 |


## Layer diagnostics (label-free; they choose nothing — layer 2 was fixed in advance)

High cross-seed agreement is not read as quality: stable recording identity or coarse posture would also score highly. Association with recording and subject (normalized mutual information of codes with that metadata; nearest-neighbour share ignoring temporal neighbours) shows whether units separate recordings rather than actions. With 512 codes and few windows the NMI of unrelated labelings is far above 0 (finite-sample bias), so cross-seed NMI is read only against the chance value shown for T2 (one seed's codes shuffled across windows) and against T2 at the same window count.

| dataset | codes | occupied | perplexity | cross-seed NMI | recording NMI | subject NMI | same-recording NN share |
|---|---|---|---|---|---|---|---|
| babel2 | T2 codes (reference) | 512 | 260.0 | 0.868 (chance 0.381, 10003 windows) | 0.620 | — | 0.325 |
| babel2 | I1 layer 1 | 512 | 409.5 | 0.852 | 0.705 | — | 0.435 |
| babel2 | I1 layer 2 (targets) | 512 | 413.9 | 0.867 | 0.747 | — | 0.506 |
| babel2 | I1 layer 3 | 512 | 417.0 | 0.881 | 0.788 | — | 0.579 |
| babel2 | I1 layer 4 | 512 | 415.2 | 0.895 | 0.820 | — | 0.643 |
| babel3 | T2 codes (reference) | 512 | 305.7 | 0.843 (chance 0.408, 10386 windows) | 0.608 | — | 0.450 |
| babel3 | I1 layer 1 | 512 | 440.6 | 0.839 | 0.699 | — | 0.577 |
| babel3 | I1 layer 2 (targets) | 512 | 444.5 | 0.863 | 0.768 | — | 0.674 |
| babel3 | I1 layer 3 | 512 | 433.9 | 0.879 | 0.813 | — | 0.735 |
| babel3 | I1 layer 4 | 512 | 420.6 | 0.898 | 0.844 | — | 0.782 |
| lara | T2 codes (reference) | 512 | 348.3 | 0.688 (chance 0.077, 219500 windows) | 0.370 | 0.356 | 0.163 |
| lara | I1 layer 1 | 512 | 462.1 | 0.666 | 0.372 | 0.330 | 0.196 |
| lara | I1 layer 2 (targets) | 512 | 463.4 | 0.665 | 0.398 | 0.362 | 0.217 |
| lara | I1 layer 3 | 512 | 464.6 | 0.685 | 0.449 | 0.436 | 0.242 |
| lara | I1 layer 4 | 512 | 465.3 | 0.704 | 0.500 | 0.499 | 0.280 |


## Per-class frame recall against R (uses labels; analysis only)

Global Hungarian mapping as the evaluator; small classes are noisy.

| dataset | arm/grouping | class | frame share % | n seeds | R recall | recall | Δ recall |
|---|---|---|---|---|---|---|---|
| babel2 | i1/tg | sit | 50.7 | 3 | 48.6 | 45.2 | -3.4 |
| babel2 | i1/tg | run | 8.5 | 3 | 56.1 | 75.4 | +19.4 |
| babel2 | i1/tg | stand_up | 9.5 | 3 | 24.0 | 22.8 | -1.1 |
| babel2 | i1/tg | kick | 3.9 | 3 | 15.4 | 26.8 | +11.4 |
| babel2 | i1/tg | none | 27.4 | 3 | 43.4 | 41.3 | -2.1 |
| babel2 | i2/ig | sit | 50.7 | 3 | 48.6 | 40.5 | -8.1 |
| babel2 | i2/ig | run | 8.5 | 3 | 56.1 | 55.1 | -1.0 |
| babel2 | i2/ig | stand_up | 9.5 | 3 | 24.0 | 27.5 | +3.5 |
| babel2 | i2/ig | kick | 3.9 | 3 | 15.4 | 4.7 | -10.7 |
| babel2 | i2/ig | none | 27.4 | 3 | 43.4 | 44.1 | +0.6 |
| babel3 | i1/tg | jog | 6.4 | 3 | 83.2 | 84.9 | +1.7 |
| babel3 | i1/tg | wave | 1.6 | 3 | 22.6 | 29.7 | +7.1 |
| babel3 | i1/tg | dance | 43.1 | 3 | 30.1 | 30.4 | +0.3 |
| babel3 | i1/tg | gesture | 25.8 | 3 | 32.9 | 40.1 | +7.2 |
| babel3 | i1/tg | none | 23.1 | 3 | 39.2 | 36.0 | -3.3 |
| babel3 | i2/ig | jog | 6.4 | 3 | 83.2 | 85.9 | +2.7 |
| babel3 | i2/ig | wave | 1.6 | 3 | 22.6 | 16.6 | -6.0 |
| babel3 | i2/ig | dance | 43.1 | 3 | 30.1 | 32.3 | +2.2 |
| babel3 | i2/ig | gesture | 25.8 | 3 | 32.9 | 32.1 | -0.8 |
| babel3 | i2/ig | none | 23.1 | 3 | 39.2 | 38.3 | -1.0 |
| lara | i1/tg | Standing | 11.2 | 3 | 26.2 | 31.7 | +5.6 |
| lara | i1/tg | Walking | 12.4 | 3 | 63.1 | 34.6 | -28.5 |
| lara | i1/tg | Cart | 11.9 | 3 | 20.2 | 57.0 | +36.9 |
| lara | i1/tg | Handling(upwards) | 7.8 | 3 | 55.6 | 38.5 | -17.2 |
| lara | i1/tg | Handling(centred) | 43.7 | 3 | 26.4 | 24.8 | -1.6 |
| lara | i1/tg | Handling(downwards) | 7.0 | 3 | 57.8 | 58.0 | +0.3 |
| lara | i1/tg | Synchronization | 1.8 | 3 | 98.8 | 66.0 | -32.8 |
| lara | i1/tg | None | 4.2 | 3 | 12.4 | 11.5 | -0.9 |
| lara | i2/ig | Standing | 11.2 | 3 | 26.2 | 29.1 | +2.9 |
| lara | i2/ig | Walking | 12.4 | 3 | 63.1 | 44.5 | -18.6 |
| lara | i2/ig | Cart | 11.9 | 3 | 20.2 | 52.5 | +32.4 |
| lara | i2/ig | Handling(upwards) | 7.8 | 3 | 55.6 | 56.7 | +1.1 |
| lara | i2/ig | Handling(centred) | 43.7 | 3 | 26.4 | 25.6 | -0.8 |
| lara | i2/ig | Handling(downwards) | 7.0 | 3 | 57.8 | 61.8 | +4.0 |
| lara | i2/ig | Synchronization | 1.8 | 3 | 98.8 | 64.7 | -34.0 |
| lara | i2/ig | None | 4.2 | 3 | 12.4 | 12.0 | -0.4 |
