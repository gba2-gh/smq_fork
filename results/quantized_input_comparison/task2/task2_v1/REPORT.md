# Quantized-input comparison: task2_v1

Compiled 2026-09-30T11:23:56+13:00 from saved results. No training, OT refitting, or experiment files were changed. Scores are percentages. This report supersedes the earlier partial snapshot for this run.

## Summary

1. All 40 planned main cells and both historical compatibility jobs completed; the main table contains 160 unique cell/rung rows.
2. All 40 cells passed the current artifact/configuration, finite-metric and recording/frame-coverage checks. The current implementation hash matches the run manifest.
3. Native VQ quantized-vector contextual inputs reach **41.27 ± 1.76 F1@50**, versus **37.04 ± 3.26** for native IDs, across three downstream seeds.
4. That F1 gain is not a consistent gain in frame accuracy: mean MoF is **42.79** versus **43.19** for IDs.
5. Freely trained embeddings initialized from VQ geometry reach **34.52 ± 4.45 F1@50**; that variant does not improve on the random-ID embedding baseline.
6. Native FSQ quantized-vector context reaches only **5.12 F1@50**, versus **34.03** for native IDs at the same seed. Its embedding diagnostics show strong temporal similarity and low effective rank.
7. The largest observed F1@50 is **49.55** for VQ-encoder K-means IDs with context, measured at one downstream seed; this is not an established ranking over all seeds.
8. Contextual IDs improve F1@50 over direct categorical OT in every family at matched seed 111. Quantized vectors without context do not produce comparable segment scores.
9. Native VQ uses **384/512** codes; native FSQ uses **23/512**. Nominal alphabet size does not imply equal effective use.
10. **38/40 outer OT fits are capped**. The comparisons describe this fixed optimization budget, one dataset, one autoencoder seed, and a pooled transductive protocol.

## Protocol and provenance

HuGaDB only: 364 recordings, 1,137,986 frames, 76,035 windows, 60 Hz, and 10 action classes including background. Inputs are IMU channels. Every family uses non-overlapping 15-frame (0.25-second) tokens with partial terminal windows retained. VQ and FSQ encoders come from `fsq_pilot_002`, trained for 30 epochs at autoencoder seed 111 with 15-frame quantization and nominal K=512. This run does not retrain those encoders. Raw K-means has no encoder.

| Identity | Value |
|---|---|
| Task 2 manifest | [manifest.json](manifest.json) |
| Source artifacts | [qic_v3 manifest](../../qic_v3/manifest.json) |
| Effective config hash | 4e375edbc2063079 |
| Implementation hash | ece3209d4ed9c785 |
| Task 1 manifest identity | 84e28c120053d109 |
| VQ checkpoint SHA-256 | c58b9e1755eca77825d9234e49637fe59323a912f56600c7bff21a46533c1c13 |
| FSQ checkpoint SHA-256 | 71db8d36f955fea3462b3a9f4f6c31cb576f98e41ac900201c07f15b67cba7a5 |
| Main metrics | [segmentation.csv](segmentation.csv) |
| Training metrics | [contextualizer.csv](contextualizer.csv) |
| Per-cell predictions, fitted OT parameters, checkpoints, diagnostics | [cells/](cells/) |
| Historical reproduction | [historical/diff.csv](historical/diff.csv) |
| Main results SHA-256 | fbe7e29a23c967abd5c3e68fe0abf0f2b2c98ad48839ae3690b8af82caaada60 |

All contextual arms predict the same family-specific discrete targets. ID arms use a trainable embedding table. Quantized and continuous arms share the projection initialization, backbone and head initialization, masking scheme and optimization budget within each family/seed. Quantized vectors are divided by the same scalar RMS as continuous features. Geometry-initialized IDs begin with the quantized arm's projected lookup table, then train freely. Native VQ/FSQ vectors are 1,440-dimensional reconstructed patch vectors; K-means vectors are PCA-64 features. Native FSQ's upstream scalar bottleneck has three coordinates.

The fixed recording train/validation split contains 335 training and 29 validation recordings. The contextualizer uses width 128, four layers, four attention heads, symmetric ALiBi, masked prediction with receptive-field eligibility, 5,000 optimizer updates, and best eligible validation loss for checkpoint selection. This validation split selects the contextualizer checkpoint; final OT fitting/scoring is pooled over the full collection. It is not a held-out action-segmentation evaluation.

Direct categorical OT uses code IDs and categorical emissions. Direct vector OT uses shared-mean centering and L2 normalization, with cosine costs. All contextual OT arms use contextualizer embeddings and cosine costs. They fit 10 action states, sharing the initial code grouping within a family. Coefficients are a=0.7, beta=0.3, lambda=0.05 and epsilon=0.07. Outer fitting is capped at 50 iterations with 25 inner steps; final solves use saved rungs 500/1,000/2,000/4,000. Tables below use 4,000.

Action labels enter Hungarian evaluation mapping and diagnostics, not contextualizer training or model selection. Pooled preprocessing, vocabulary fitting, frozen-encoder exposure and OT fitting make these transductive results. Native VQ contextual arms have three downstream seeds; every other family and all direct arms have one. Population SD measures downstream initialization sensitivity conditional on the fixed encoder/tokenizer, not sampling uncertainty.

## Contextual OT results

| Family | Input arm | Seeds | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| vq_native | context_id | 3 | 43.19 ± 1.95 | 39.21 ± 2.74 | 44.60 ± 3.43 | 41.87 ± 3.53 | 37.04 ± 3.26 |
| vq_native | context_quantized | 3 | 42.79 ± 0.94 | 40.23 ± 0.73 | 47.32 ± 1.19 | 44.85 ± 2.02 | 41.27 ± 1.76 |
| vq_native | context_continuous | 3 | 41.13 ± 0.96 | 36.85 ± 1.99 | 43.61 ± 1.39 | 40.84 ± 1.19 | 37.38 ± 1.50 |
| vq_native | context_geometry_init | 3 | 39.80 ± 1.92 | 36.46 ± 2.28 | 41.54 ± 4.08 | 38.07 ± 4.04 | 34.52 ± 4.45 |
| fsq_native | context_id | 1 | 46.11 | 33.28 | 42.75 | 37.51 | 34.03 |
| fsq_native | context_quantized | 1 | 33.14 | 17.08 | 20.71 | 13.52 | 5.12 |
| fsq_native | context_continuous | 1 | 43.02 | 37.10 | 44.74 | 40.26 | 34.74 |
| vq_kmeans | context_id | 1 | 52.39 | 46.90 | 53.55 | 51.96 | 49.55 |
| vq_kmeans | context_quantized | 1 | 45.13 | 44.63 | 49.14 | 47.69 | 45.40 |
| vq_kmeans | context_continuous | 1 | 37.53 | 30.92 | 37.10 | 34.15 | 31.58 |
| fsq_kmeans | context_id | 1 | 44.76 | 39.56 | 47.05 | 45.33 | 42.18 |
| fsq_kmeans | context_quantized | 1 | 42.78 | 30.09 | 38.06 | 34.00 | 32.42 |
| fsq_kmeans | context_continuous | 1 | 43.88 | 33.30 | 41.69 | 38.27 | 35.80 |
| raw_kmeans | context_id | 1 | 43.87 | 36.76 | 42.71 | 39.04 | 36.08 |
| raw_kmeans | context_quantized | 1 | 38.73 | 33.13 | 38.88 | 35.59 | 30.95 |
| raw_kmeans | context_continuous | 1 | 28.58 | 21.40 | 25.06 | 21.32 | 16.78 |
| raw_kmeans | context_geometry_init | 1 | 34.59 | 30.15 | 34.80 | 32.21 | 28.42 |

## Direct OT results

All direct results use seed 111. `direct_id` is categorical OT; the other two arms are cosine-cost OT.

| Family | Input arm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| vq_native | direct_id | 37.54 | 18.30 | 5.62 | 4.40 | 4.05 |
| vq_native | direct_quantized | 42.26 | 17.60 | 6.14 | 4.96 | 4.87 |
| vq_native | direct_continuous | 40.43 | 18.01 | 5.91 | 5.08 | 4.95 |
| fsq_native | direct_id | 41.83 | 18.77 | 12.55 | 8.40 | 6.33 |
| fsq_native | direct_quantized | 30.16 | 20.32 | 14.17 | 8.77 | 4.54 |
| fsq_native | direct_continuous | 46.52 | 23.36 | 17.47 | 14.56 | 13.63 |
| vq_kmeans | direct_id | 39.10 | 21.46 | 6.77 | 5.43 | 4.61 |
| vq_kmeans | direct_quantized | 40.81 | 17.47 | 5.91 | 4.95 | 4.86 |
| vq_kmeans | direct_continuous | 39.94 | 17.90 | 5.74 | 5.09 | 4.95 |
| fsq_kmeans | direct_id | 45.47 | 24.24 | 10.01 | 7.84 | 6.61 |
| fsq_kmeans | direct_quantized | 41.66 | 17.28 | 7.04 | 5.33 | 5.20 |
| fsq_kmeans | direct_continuous | 43.58 | 18.43 | 9.84 | 7.82 | 7.50 |
| raw_kmeans | direct_id | 31.71 | 7.68 | 3.22 | 1.97 | 1.64 |
| raw_kmeans | direct_quantized | 34.55 | 17.58 | 4.59 | 4.03 | 3.95 |
| raw_kmeans | direct_continuous | 34.53 | 17.37 | 4.56 | 4.04 | 3.97 |

## Matched differences and interpretation

| Family | Context IDs − direct IDs (seed 111) | Context quantized − context IDs (available paired seeds) | Context continuous − context quantized (available paired seeds) |
|---|---|---|---|
| vq_native | +31.45 | +4.23 | -3.89 |
| fsq_native | +27.70 | -28.91 | +29.62 |
| vq_kmeans | +44.94 | -4.15 | -13.82 |
| fsq_kmeans | +35.57 | -9.76 | +3.38 |
| raw_kmeans | +34.44 | -5.13 | -14.17 |

### Native VQ

Quantized-vector minus ID F1@50 differences at seeds 111, 222 and 1538574472 are +3.31, +8.74, +0.63 points (mean +4.23). These are three comparisons of the same encoder checkpoint. The quantized lookup is a deterministic function of the IDs, so this does not demonstrate added information: it demonstrates an advantage for this parameterization and supplied geometry under the tested training procedure.

The freely trained geometry-initialized arm does not preserve that advantage. Its lower mean score shows that adding geometry at initialization alone was insufficient here. It does not establish why: shared projection structure, regularization, learning trajectory and geometry retention all differ after initialization. Continuous input does not outperform the quantized arm on mean F1@50. That is compatible with useful quantization, but does not prove the removed detail is subject nuisance or irrelevant information.


### Native FSQ

The quantized-vector contextual arm has F1@50 5.12, adjacent embedding cosine similarity 0.9975, covariance entropy effective rank 2.63 and median predicted run length 7.50 seconds. The ID contextual arm has F1@50 34.03, similarity 0.9087, effective rank 10.11 and median run length 4.25 seconds. Ground-truth median duration is 3.47 seconds. These observations are consistent with poorly differentiated contextual embeddings and overly long predictions in the vector arm; they do not establish a single causal explanation.

This is the repaired pilot FSQ adapter, with non-affine LayerNorm across three scalars and 23 occupied native codes. The three-coordinate constraint limits the upstream geometry; the exported 1,440-dimensional inverse-projected vector is not 1,440 independent scalar degrees of freedom. This run supplies no geometry-initialized free-ID control for FSQ, so it cannot separate input geometry limitations from projection restrictions there. It is not evidence that FSQ generally fails.


### Encoder K-means and raw inputs

At seed 111, VQ-encoder K-means IDs reach 49.55 F1@50 versus 36.08 for raw K-means IDs in this protocol. This does not reproduce the broad claim that the encoder is invariably detrimental. Conversely, a single seed and a different masking/training protocol cannot reverse the previous encoder-sweep conclusion in general. Geometry inputs help native VQ but lower F1@50 versus ID inputs in all three K-means families at the evaluated seed. There is no universal geometry-input gain across tokenizers.

Continuous input also does not universally help: it beats quantized input for FSQ-native and FSQ-K-means, but loses for VQ-native, VQ-K-means and raw K-means. These are outcomes of the trained input/readout combinations, not information-theoretic rankings.

## Tokenizer and embedding diagnostics

| Family | Nominal codes | Occupied codes | Effective codes: exp(entropy) | Largest window share |
|---|---|---|---|---|
| vq_native | 512 | 384 | 159.83 | 6.47% |
| fsq_native | 512 | 23 | 11.24 | 17.52% |
| vq_kmeans | 512 | 512 | 192.98 | 6.13% |
| fsq_kmeans | 512 | 512 | 148.18 | 9.37% |
| raw_kmeans | 512 | 512 | 178.88 | 4.68% |

Counts include terminal windows. These saved Task 1 assignments are the relevant occupancy population for this run; small differences from the earlier pilot audit can reflect extraction/boundary handling and are not evidence of retraining.

| Family | Arm (seed 111) | Selected step | Validation loss | Effective rank | Adjacent cosine | Predicted median run (s) |
|---|---|---|---|---|---|---|
| vq_native | context_id | 4400 | 2.1927 | 43.94 | 0.6867 | 4.00 |
| vq_native | context_quantized | 4500 | 2.1606 | 28.72 | 0.6651 | 4.00 |
| vq_native | context_continuous | 4000 | 2.1518 | 32.11 | 0.6997 | 4.00 |
| vq_native | context_geometry_init | 4400 | 2.1349 | 27.95 | 0.8289 | 4.25 |
| fsq_native | context_id | 4200 | 0.8508 | 10.11 | 0.9087 | 4.25 |
| fsq_native | context_quantized | 4900 | 1.0282 | 2.63 | 0.9975 | 7.50 |
| fsq_native | context_continuous | 4000 | 0.8176 | 15.41 | 0.8488 | 4.50 |
| vq_kmeans | context_id | 4400 | 1.9554 | 42.83 | 0.7401 | 3.75 |
| vq_kmeans | context_quantized | 4800 | 1.9047 | 31.37 | 0.7810 | 4.00 |
| vq_kmeans | context_continuous | 3700 | 1.9602 | 33.64 | 0.7615 | 4.50 |
| fsq_kmeans | context_id | 5000 | 1.9658 | 40.88 | 0.7152 | 3.75 |
| fsq_kmeans | context_quantized | 4400 | 1.9242 | 29.34 | 0.7972 | 4.25 |
| fsq_kmeans | context_continuous | 4900 | 1.9191 | 32.23 | 0.7851 | 4.25 |
| raw_kmeans | context_id | 4400 | 1.8159 | 43.65 | 0.7353 | 4.00 |
| raw_kmeans | context_quantized | 4700 | 1.7647 | 35.33 | 0.7088 | 4.00 |
| raw_kmeans | context_continuous | 4800 | 1.7692 | 39.42 | 0.6728 | 4.00 |
| raw_kmeans | context_geometry_init | 4700 | 1.7615 | 35.75 | 0.7112 | 4.00 |

Effective rank is exp(entropy) of the centered covariance eigenspectrum on a fixed sample of at most 512 Task 1 fitting windows. It describes contextual output embeddings, not raw input-vector rank. Median run length pools all runs across recordings. Validation losses are comparable within a tokenizer family; different target distributions preclude treating cross-family loss as an action-quality ranking.

## Optimization, artifact verification and historical reproduction

All 40 planned cells have complete artifacts and four distinct solver rungs; all 160 metric rows are finite. Prediction recording sets and frame/window lengths match Task 1. The recorded implementation hash `ece3209d4ed9c785` equals the current implementation hash. This verifies saved artifact completeness and provenance checks; it is not an independent rescore of every prediction. No failed-cell ledger or final invalid solver status was found.

Outer fit status: 2 converged and 38 capped. At the final rung, there are 13,897 converged and 663 capped recording solves across the 40 cells (4.55% capped). Recording solves are repeated uses of the same population across cells, not independent samples. Final continuation cannot certify convergence or global optimality of the capped outer alternating fits.

| Metric | Largest absolute cell change, 500 → 4,000 steps (pp) |
|---|---|
| MoF | 0.68 |
| Edit | 1.83 |
| F1@10 | 2.64 |
| F1@25 | 2.70 |
| F1@50 | 2.22 |

All contextualizers completed 5,000 optimizer updates. The saved geometry-initialization equivalence checks and training counts are retained in each `info.json`; no additional training or numerical interventions were performed for this report.

Both historical jobs completed. Across 16 stream/method/rung comparison rows and all five reported metrics, the maximum absolute historical difference is **0.00 points**, at the saved CSV precision. This reproduces the historical seed-111 helpers. It does not make the new main-matrix protocol equivalent to those helpers.

| Stream | Method | Historical F1@50 | New protocol F1@50 (seed111) | Difference (pp) |
|---|---|---|---|---|
| vq_native | categorical_asot | 4.05 | 4.05 | +0.00 |
| vq_native | contextual_asot | 36.77 | 35.50 | -1.27 |
| raw | categorical_asot | 1.64 | 1.64 | +0.00 |
| raw | contextual_asot | 50.82 | 36.08 | -14.74 |

The protocol gap includes receptive-field-aware masking, initialization/control choices and training/selection differences. The compatibility check rules out an inability to reproduce the old helpers, but does not isolate which new choice explains the changed contextual scores.

## Missing comparisons and limits

- No planned main cell is missing. Both planned historical checks are complete. Geometry-initialized arms were planned only for native VQ and raw K-means.
- LARa was not part of this run. The other four families have only one downstream seed; none of the families has multiple independent autoencoder training seeds here.
- No native VQ-10 at 15 frames, native VQ-512 at 60 frames, or released VQ-10 native-code OT comparison is included. Patch duration, vocabulary size and training effects remain unseparated.
- The released SMQ baseline and original Stage C have different tokenizers, training histories and solver/protocol details. They are historical references, not matched controls.
- The native FSQ adapter has low realized code use and constrained geometry. Its outcome cannot establish a general limit of FSQ.
- The run does not identify whether context gains come from ordered composition, masked training, denoising, smoothing or additional model capacity; no new order-destruction or matched pooling control is included.
- No evaluation here demonstrates out-of-subject generalization, a new SOTA, an information-theoretic ceiling, or novelty relative to other papers.

## Research questions answered at this scope

| Question | What this run supports |
|---|---|
| Can native learned short-patch codes support action decoding? | Yes for this native VQ checkpoint: contextual inputs recover substantially higher segment F1 than direct OT. |
| Does providing code-vector geometry always help? | No. It helps native VQ F1 across the three-seed mean, but not MoF, and it hurts several other families. |
| Does freeing a geometry-initialized ID table improve results? | Not for native VQ or raw K-means in the evaluated configurations. |
| Are continuous pre-quantization inputs necessarily superior? | No under these matched family-specific contextual objectives and budgets. This is not a proof of information loss or denoising. |
| Did shorter training patches improve the encoder? | Not isolated: there is no matched 15-versus-60-frame training comparison. |

## Open questions

- Does the native VQ quantized-input F1 advantage persist across independently trained tokenizers, more subjects or LARa?
- Does a less constrained FSQ adapter or a free geometry-initialized FSQ table avoid the observed low-rank contextual output?
- Which change between the historical and main contextualizer protocols explains their score gap, particularly for raw tokens?
- Does changing patch duration at fixed vocabulary and training budget improve native-code decoding?
- Do larger outer optimization budgets change the ordering of these arms?

## Complete seed-level final results

| Family | Arm | Seed | MoF | Edit | F1@10 | F1@25 | F1@50 | Outer status | Final capped recordings |
|---|---|---|---|---|---|---|---|---|---|
| vq_native | context_id | 111 | 42.98 | 39.43 | 43.29 | 40.46 | 35.5 | capped | 10 |
| vq_native | context_id | 222 | 40.92 | 35.75 | 41.2 | 38.42 | 34.05 | capped | 23 |
| vq_native | context_id | 1538574472 | 45.68 | 42.45 | 49.3 | 46.72 | 41.58 | capped | 20 |
| vq_native | context_quantized | 111 | 41.51 | 39.47 | 46.09 | 42.14 | 38.81 | capped | 14 |
| vq_native | context_quantized | 222 | 43.14 | 41.21 | 48.93 | 46.98 | 42.79 | capped | 30 |
| vq_native | context_quantized | 1538574472 | 43.73 | 40.01 | 46.95 | 45.43 | 42.21 | capped | 17 |
| vq_native | context_continuous | 111 | 40.14 | 34.12 | 41.65 | 39.26 | 35.82 | capped | 23 |
| vq_native | context_continuous | 222 | 42.42 | 37.62 | 44.59 | 41.12 | 36.92 | capped | 9 |
| vq_native | context_continuous | 1538574472 | 40.82 | 38.81 | 44.59 | 42.13 | 39.4 | capped | 23 |
| fsq_native | context_id | 111 | 46.11 | 33.28 | 42.75 | 37.51 | 34.03 | capped | 8 |
| fsq_native | context_quantized | 111 | 33.14 | 17.08 | 20.71 | 13.52 | 5.12 | capped | 211 |
| fsq_native | context_continuous | 111 | 43.02 | 37.1 | 44.74 | 40.26 | 34.74 | capped | 57 |
| vq_kmeans | context_id | 111 | 52.39 | 46.9 | 53.55 | 51.96 | 49.55 | capped | 10 |
| vq_kmeans | context_quantized | 111 | 45.13 | 44.63 | 49.14 | 47.69 | 45.4 | capped | 9 |
| vq_kmeans | context_continuous | 111 | 37.53 | 30.92 | 37.1 | 34.15 | 31.58 | capped | 25 |
| fsq_kmeans | context_id | 111 | 44.76 | 39.56 | 47.05 | 45.33 | 42.18 | capped | 8 |
| fsq_kmeans | context_quantized | 111 | 42.78 | 30.09 | 38.06 | 34.0 | 32.42 | capped | 2 |
| fsq_kmeans | context_continuous | 111 | 43.88 | 33.3 | 41.69 | 38.27 | 35.8 | capped | 10 |
| raw_kmeans | context_id | 111 | 43.87 | 36.76 | 42.71 | 39.04 | 36.08 | capped | 10 |
| raw_kmeans | context_quantized | 111 | 38.73 | 33.13 | 38.88 | 35.59 | 30.95 | capped | 10 |
| raw_kmeans | context_continuous | 111 | 28.58 | 21.4 | 25.06 | 21.32 | 16.78 | capped | 32 |
| vq_native | context_geometry_init | 111 | 40.59 | 39.68 | 47.3 | 43.63 | 40.27 | capped | 17 |
| vq_native | context_geometry_init | 222 | 37.15 | 34.79 | 38.91 | 36.4 | 33.86 | capped | 23 |
| vq_native | context_geometry_init | 1538574472 | 41.65 | 34.92 | 38.42 | 34.17 | 29.44 | capped | 15 |
| raw_kmeans | context_geometry_init | 111 | 34.59 | 30.15 | 34.8 | 32.21 | 28.42 | capped | 14 |
| vq_native | direct_id | 111 | 37.54 | 18.3 | 5.62 | 4.4 | 4.05 | capped | 0 |
| vq_native | direct_quantized | 111 | 42.26 | 17.6 | 6.14 | 4.96 | 4.87 | converged | 0 |
| vq_native | direct_continuous | 111 | 40.43 | 18.01 | 5.91 | 5.08 | 4.95 | capped | 0 |
| fsq_native | direct_id | 111 | 41.83 | 18.77 | 12.55 | 8.4 | 6.33 | capped | 1 |
| fsq_native | direct_quantized | 111 | 30.16 | 20.32 | 14.17 | 8.77 | 4.54 | converged | 0 |
| fsq_native | direct_continuous | 111 | 46.52 | 23.36 | 17.47 | 14.56 | 13.63 | capped | 16 |
| vq_kmeans | direct_id | 111 | 39.1 | 21.46 | 6.77 | 5.43 | 4.61 | capped | 0 |
| vq_kmeans | direct_quantized | 111 | 40.81 | 17.47 | 5.91 | 4.95 | 4.86 | capped | 7 |
| vq_kmeans | direct_continuous | 111 | 39.94 | 17.9 | 5.74 | 5.09 | 4.95 | capped | 1 |
| fsq_kmeans | direct_id | 111 | 45.47 | 24.24 | 10.01 | 7.84 | 6.61 | capped | 0 |
| fsq_kmeans | direct_quantized | 111 | 41.66 | 17.28 | 7.04 | 5.33 | 5.2 | capped | 1 |
| fsq_kmeans | direct_continuous | 111 | 43.58 | 18.43 | 9.84 | 7.82 | 7.5 | capped | 4 |
| raw_kmeans | direct_id | 111 | 31.71 | 7.68 | 3.22 | 1.97 | 1.64 | capped | 0 |
| raw_kmeans | direct_quantized | 111 | 34.55 | 17.58 | 4.59 | 4.03 | 3.95 | capped | 1 |
| raw_kmeans | direct_continuous | 111 | 34.53 | 17.37 | 4.56 | 4.04 | 3.97 | capped | 2 |
