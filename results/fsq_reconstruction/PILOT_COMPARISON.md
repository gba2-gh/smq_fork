# FSQ pilot comparison

28 September 2026. Analysis of fsq_pilot_001, fsq_pilot_002 and the two smoke runs. No training or implementation changes were performed. Existing results are preserved.

## Assessment

Pilot 001 failed at the native FSQ bottleneck. Pilot 002 restores encoder gradients and permits native-code decoding, but its FSQ adaptation underperforms matched VQ in both native and encoder-K-means contextual segmentation. The repaired adapter uses only 23 of its nominal 512 codes on the complete dataset. Neither run supports progression to LARa under the plan's rule.

These are two different implementations at the same autoencoder seed, not two independent training replicates. Both retain SMQ's reconstruction target. They do not evaluate alternative encoder objectives or settle whether properly configured FSQ could work.

## Coverage and comparisons

Both full runs use HuGaDB, 15-frame patches, 30 autoencoder epochs, autoencoder seed 111, and three downstream seeds. HuGaDB has 10 action classes and IMU inputs; no LARa result exists in this directory.

- Pilot 001: 18 completed method/seed combinations, each with four solver endpoints; three explicit native-FSQ categorical failures and three unrecorded native-FSQ contextual omissions. The results CSV is now named segmentation001.csv.
- Pilot 002: all 24 method/seed combinations and 96 unique solver rows are present. At the final endpoint, the aggregate recording-level status counts are 8,551 converged and 185 capped; completion is not universal numerical convergence. No final invalid status is recorded. Outer-fit convergence traces were not saved.
- Smoke runs use two epochs, smaller encoders, 32 nominal codes and one downstream seed. They verify execution only and are not performance comparisons with the full pilots. fsq_smoke_test has five occupied native-FSQ prototypes; fsq_smoke_fix has one, according to their failure rows. Native contextual decoding is skipped in both.

## Contextual ASOT results

Final 4,000-step ladder endpoint, mean +/- population SD over the three contextualizer/OT seeds. K-means was actually fitted once at seed 111 per encoder, not refitted per downstream seed.

| Run | Stream | F1@50 | MoF |
|---|---|---:|---:|
| 001 | VQ native | 43.16 +/- 1.04 | 44.61 +/- 3.72 |
| 001 | VQ encoder K-means | 47.84 +/- 1.34 | 50.74 +/- 1.33 |
| 001 | FSQ native | Unavailable | Unavailable |
| 001 | FSQ encoder K-means | 50.77 +/- 2.08 | 48.42 +/- 1.88 |
| 002 | VQ native | 41.90 +/- 4.24 | 45.11 +/- 3.00 |
| 002 | VQ encoder K-means | 46.53 +/- 2.92 | 46.74 +/- 2.22 |
| 002 | FSQ native | 29.72 +/- 3.27 | 41.29 +/- 3.54 |
| 002 | FSQ encoder K-means | 41.32 +/- 5.50 | 45.34 +/- 2.48 |

Within pilot 002, paired FSQ-minus-VQ differences:

| Stream | F1@50 differences, seeds 111 / 222 / 1538574472 | Mean F1@50 difference | Mean MoF difference |
|---|---|---:|---:|
| Native | -2.44 / -13.94 / -20.14 | -12.17 | -3.82 |
| Encoder K-means | -0.61 / -3.73 / -11.29 | -5.21 | -1.40 |

FSQ loses F1@50 at every matched downstream seed in both streams. No FSQ stream passes the progression rule. These are conditional comparisons at one autoencoder seed and one fitted vocabulary, not estimates of training-population uncertainty.

Pilot 001's encoder-K-means gain of 2.93 F1 points came with a 2.33-point MoF loss and a failed native tokenizer. Its larger F1 is not evidence of superior native FSQ discretization.

The VQ checkpoint tensors at epoch 30 are exactly equal between pilots, and their categorical scores and saved reconstruction diagnostics match. Their contextual scores nevertheless differ. The intervening source change routes contextualizer training/inference to the selected device instead of default CPU. Original manifests omit device/source identities, so exact runtime provenance cannot be established. This changed routing is a concrete confound: do not attribute cross-run differences solely to the FSQ normalization change.

## Native-code usage and training health

A fresh CPU, evaluation-mode audit covered all 364 recordings, 76,035 patches and 1,137,986 real frames, one recording at a time. Encoder output was repacked as in SMQ and passed through the checkpoint's quantizer projection. Pilot 001 was evaluated with the historical direct projection; pilot 002 with the current non-affine three-coordinate LayerNorm. Original source hashes were not saved, so these architecture assignments are reconstructed from the code history and earlier run audit.

The audit results are saved in PILOT_COMPARISON_AUDIT.json, including all per-code and per-coordinate counts.

| Diagnostic | Pilot 001 FSQ | Pilot 002 FSQ |
|---|---:|---:|
| Occupied native codes | 1 | 23 |
| Window-level code entropy, nats | 0 | 2.4192 |
| Effective code count, exp(entropy) | 1 | 11.24 |
| Largest code's window share | 100% | 17.51% |
| Maximum absolute pre-bound scalar | 2,878,743 | 1.414214 |
| Encoder parameter values changed, epoch 20 -> 30 | 0 | 399,896 |
| Reconstruction MSE | 398,160,243.84 | 306,680,336.14 |
| Reconstruction MSE / VQ MSE | 6.70 | 5.16 |

Pilot 002 improves FSQ reconstruction by 22.98% relative to pilot 001 and has nonzero fixed-batch encoder reconstruction gradients at epochs 10, 20 and 30. Its late encoder parameters continue changing. It therefore fixes the original saturated-bound/zero-gradient failure. However, reconstruction remains much worse than VQ, and code use remains narrow. VQ's reconstruction MSE is 59,402,489.67 in both runs.

The observed 23-code occupancy is about 4.5% of the nominal alphabet; effective usage is lower because counts are unequal. This is not complete collapse, but it is not a successfully utilized 512-code representation either.

### What the normalization changes

The repair adds non-affine LayerNorm directly across the three scalars, before the FSQ bound. Those three outputs have zero mean and nearly fixed variance when the input variance exceeds epsilon. Apart from epsilon's small radial effect, the representation is restricted to a circle in a plane: it has roughly one angular degree of freedom, rather than three freely varying scalar coordinates.

That changes the accessible product-grid geometry. Nominal alphabet size 512 does not establish capacity matching with VQ512. The audit does not prove that LayerNorm alone explains every unused code or all downstream losses, but its restriction is architectural, not a learned property of the motion data.

The logged saturation_frac counts |z| > 3 after this normalization. For three non-affine normalized scalars, |z| is bounded by approximately sqrt(2), so this flag is guaranteed to stay zero for finite inputs. Zero saturation therefore provides no evidence of healthy code utilization. The smoke_fix run demonstrates this directly: zero flagged saturation coexists with one occupied native code.

## What the decoder results establish

Pilot 002 mean F1@50 for direct categorical versus contextual ASOT:

| Stream | Categorical | Contextual |
|---|---:|---:|
| VQ native | 4.47 | 41.90 |
| VQ encoder K-means | 2.43 | 46.53 |
| FSQ native | 5.98 | 29.72 |
| FSQ encoder K-means | 3.81 | 41.32 |

The complete learned-context pipeline is substantially more useful than direct categorical emissions on all four streams under this configuration. This comparison does not isolate temporal ordering, masked prediction, smoothing or model capacity individually.

FSQ's native categorical F1 is slightly higher than VQ's, but both values are low and that ordering reverses under the stronger contextual decoder. A single decoder endpoint is not a representation-quality ranking.

Within pilot 002, final-solve continuation from 500 to 4,000 steps changes F1@50 by at most 0.17 points across the scored cells. Simply extending those final solves is unlikely to explain the much larger FSQ-VQ gaps; this does not establish global optimization or resolve unsaved outer-fit convergence.

Both VQ and FSQ encoder-K-means streams beat their own native-token contextual results. This shows that the trained native quantizers are not the best tested tokenization for the downstream task. It does not establish that discretization itself is harmful: the better alternatives are also discrete K-means units. No matched continuous-feature decoder is included here.

## Remaining implementation limitations

- fit_pca_kmeans fixes KMeans random_state to preprocessing_seed and runs before the downstream loop. Three downstream seeds do not include vocabulary fitting variability.
- The manifests retain pilot_version 1.0 and the same configuration hash despite a substantive adapter change. Non-affine LayerNorm has no state-dict parameters, so loading weights cannot validate which forward computation belongs to a checkpoint.
- Actual FSQ occupancy was not recorded by the runner; the audit above fills that diagnostic gap. VQ active_codes counts positive post-replacement EMA buffers, not observed assignments. No per-step replacement audit was saved.
- Artifacts retain action prototypes/groupings but not predictions, fitted tokenizer transforms, contextualizer checkpoints or inference embeddings. Saved segmentation scores cannot be independently rescored without rerunning substantial computation.
- Resume accepts completed jobs by row count, without validating checkpoint/source/configuration identities. Resuming after a source change can mix incompatible results; row completeness alone does not rule this out. No such mixture is proven here.
- The released-encoder reference and epoch-0 encoder-K-means control remain absent. Improvements over historical Stage C are not controlled encoder-training effects or evidence of SOTA.

## Recommended decision

Do not expand the current FSQ variant to LARa or more seeds on these results. The native-FSQ hypothesis has no positive support here; the first run's upstream F1 signal did not persist as a matched FSQ advantage in the repaired run.

If one final repair is affordable, first test a small, versioned normalization/calibration change that controls projection scale without normalizing the three FSQ coordinates jointly onto a near-circle. Examples to evaluate in a short feasibility check are scaling calibrated on fitting data or normalization in the wider feature space before the projection. These are proposed interventions, not demonstrated fixes. Require live reconstruction gradients and broad actual coordinate/code usage before another downstream grid; no arbitrary occupancy threshold guarantees action semantics.

Otherwise retain the working VQ-encoder/K-means/contextual-ASOT result and stop the FSQ branch for this deadline. An epoch-0 encoder control or released-encoder matched reference would answer the outstanding encoder-training question more directly than another uncontrolled FSQ run. Neither alternative was executed during this analysis.

The strongest supported finding is practical: learned context recovers useful segment predictions from these discrete streams, while the tested native FSQ adapters have not delivered the proposed benefit. General claims about FSQ, information loss from discretization, encoder optimality or alternative reconstruction objectives remain unanswered.
