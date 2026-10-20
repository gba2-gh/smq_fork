# FSQ pilot 001: result and implementation audit

Analysis date: 28 September 2026. No training was launched and no original results or implementation files were changed. This analysis reads the saved CSVs, checkpoints, artifact keys and implementation, including the working-tree diff.

## Main finding

The collapsed native FSQ codes are not outperforming VQ. The better F1@50 belongs to K-means applied to encoder features **before** the FSQ projection and quantizer. That path replaces the failed FSQ tokenizer with a working, separately fitted K-means tokenizer, then learns a new contextualizer and OT decoder.

The native FSQ stream has one occupied prototype according to all three saved downstream failures. Its categorical decoder could not initialize ten action states. The implementation returns at that failure, so the contextual native-FSQ jobs were never attempted; they are missing, not low-scoring results.

## Final contextual results

HuGaDB, autoencoder seed 111; final 4,000-step ladder endpoint. Mean and population SD over the three contextualizer/OT seeds. The encoder K-means vocabulary actually stays fixed at seed 111, contrary to the intended three-tokenizer-seed protocol.

| Stream | F1@50 | MoF |
|---|---:|---:|
| VQ native | 43.16 +/- 1.04 | 44.61 +/- 3.72 |
| VQ encoder -> K-means | 47.84 +/- 1.34 | 50.74 +/- 1.33 |
| FSQ encoder -> K-means | 50.77 +/- 2.08 | 48.42 +/- 1.88 |
| FSQ native | Unavailable: one occupied code | Unavailable |

Paired FSQ-encoder-K-means minus VQ-encoder-K-means differences:

| Downstream seed | F1@50 difference | MoF difference |
|---|---:|---:|
| 111 | -1.87 | -6.24 |
| 222 | +5.06 | +1.62 |
| 1538574472 | +5.59 | -2.36 |
| Mean | +2.93 | -2.33 |

This is a segmentation metric tradeoff, not uniform superiority. The mean MoF loss exceeds the plan's one-point allowance, so this does not qualify for its LARa progression rule even on the encoder-K-means stream. A malfunctioning native bottleneck also prevents treating the run as a healthy FSQ-versus-VQ comparison.

VQ encoder K-means also exceeds VQ native F1@50 by 4.69 points and MoF by 6.13 points. This supports testing tokenizer/readout compatibility; both compared paths remain discrete, so it is not a continuous-versus-discrete result.

## Evidence for saturation and interrupted encoder learning

Saved reconstruction MSE under the joint-distance target is 59,402,489.67 for VQ and 398,160,243.84 for FSQ: FSQ is about 6.70 times worse. These are dataset-wide, unweighted valid-element errors, not directly comparable to the weighted padded training loss.

Saved fixed-batch reconstruction gradients at the FSQ encoder output are zero at epochs 10, 20 and 30. To examine the mechanism, the saved FSQ checkpoints were loaded into the historical adapter configuration (linear projection directly into FSQ, with the newly added LayerNorm replaced by Identity in memory only). The current working tree adds LayerNorm, but that is not the architecture represented by this run.

On the implementation's fixed diagnostic batch, 373 valid patches:

| Epoch | Median absolute pre-bound scalar | Maximum absolute scalar | Fraction with exactly saturated float32 tanh |
|---|---:|---:|---:|
| 0 | 175.47 | 405.65 | 99.285% |
| 10 | 26,822.83 | 61,123.99 | 100% |
| 20 | 553,955.25 | 1,623,836.50 | 100% |
| 30 | 553,955.25 | 1,623,836.50 | 100% |

The saturation diagnostic evaluated the shifted tanh (approximately z + 0.144 for eight levels). This is a fixed-batch diagnostic, not a saved whole-dataset activation census. Padded positions in the frame-ID tensor have a default ID and must not be counted as additional occupied motion codes.

Checkpoint parameter comparisons establish:

- FSQ encoder weights changed between epochs 0 and 10, and between 10 and 20.
- All 399,896 encoder parameter values are exactly identical at epochs 20 and 30. All 4,323 input-projection parameter values are also identical at those endpoints.
- The final encoder differs from initialization by about 26.88% in relative parameter L2 norm. It is therefore incorrect to describe the final features as simply untouched random features.
- Zero gradients on the diagnostic batch at epoch 10 do not establish zero gradients on every training batch at that epoch. Dropout, other recordings and optimizer state can also affect the trajectory. The weight changes before epoch 20 confirm that distinction.

The supported explanation is a severe scale mismatch entering the saturating bound, with loss of the reconstruction gradient through that bottleneck and eventual cessation of encoder updates. Rounding's straight-through estimator does not bypass the tanh derivative. This does not demonstrate that three scalar coordinates or finite quantization must fail under healthy training.

The decoder and output adapter can still change and fit a generic reconstruction after encoder gradients disappear. Declining reconstruction training loss alone does not demonstrate successful encoder learning.

The review's suggestion that VQ is dominated by commitment is also not supported throughout training: its saved fixed-batch commitment/reconstruction gradient-norm ratio is about 0.15-0.16 at epochs 10-30, although it is about 79 at initialization. These local diagnostics do not determine dataset-wide optimizer updates.

## Why pre-FSQ features can still work

A constant FSQ output does not imply a constant encoder output. Many distinct high-dimensional patches can be mapped to the same saturated corner by the three-coordinate projection. K-means reads those distinct upstream patches and does not pass through that projection or tanh.

Thus the scored path is:

    motion -> partially trained encoder -> standardization/PCA -> K-means-512
           -> fresh masked contextualizer -> contextual ASOT

The failed native path is:

    motion -> encoder -> FSQ projection/bound -> one code

On the observed population, a constant native code has zero empirical entropy and cannot distinguish action categories by code identity. In this specific run, the combined projection/bound/quantization mapping destroys the variation needed by the native stream. That does not isolate rounding as the cause, establish general information loss from every quantizer, or show that continuous OT would perform better. The higher-scoring alternative still uses K-means discretization.

The observed K-means F1 advantage could reflect retained local structure, a different encoder-training trajectory, geometry or downstream fitting sensitivity. These results do not identify its cause or establish beneficial FSQ training. Epoch-0 encoder K-means under the same downstream recipe is the most direct missing control for whether training improved this upstream representation.

## Implementation and reporting limitations

1. **Tokenizer seeds are not replicated.** fit_pca_kmeans uses cfg.preprocessing_seed for KMeans and is called once per arm before the downstream loop. These are three contextualizer/OT seeds conditional on one vocabulary and one encoder, not the planned full downstream replication.
2. **Native FSQ contextual failures are not recorded.** The grouping failure returns before contextualizer training and writes only a categorical unavailable row. not_run.csv is empty despite those missing jobs. There are 18 completed method/seed combinations with four ladder rows each, three explicit unavailable categorical combinations and three absent contextual combinations.
3. **Actual occupancy is inadequately logged.** The VQ active_codes field counts positive EMA buffers after replacement. Resetting dead-code counts to one makes this unsuitable evidence that all 512 codes were actually assigned. FSQ occupancy entries are blank. Per-step replacement logging is absent.
4. **Artifacts are insufficient for independent rescoring.** The NPZ files contain categorical parameters/groupings and contextual prototypes; native-FSQ files are empty. They do not contain predictions, contextualizer checkpoints/embeddings, or fitted tokenizer transforms. The reported scores can be summarized but not fully reproduced or rescored from these artifacts alone.
5. **Provenance is incomplete.** The manifest records configuration but not the required source/checkpoint hashes, package versions or split/sample identities. Checkpoints save model weights, not optimizer/RNG state. No complete REPORT.md was produced by the runner.
6. **Source has changed since the run.** The working-tree FSQ adapter adds non-affine LayerNorm across the three scalar coordinates, and downstream training now has GPU routing. Loading old weights with that parameter-free normalization can succeed while silently changing the computation. Preserve this run and use a new protocol/run identity for any repair.
7. **The new normalization is not neutral.** LayerNorm over three coordinates removes their common offset and largely fixes their scale, constraining the accessible scalar geometry. Its non-affine normalized coordinates have magnitude at most approximately sqrt(2), so a post-normalization |z| > 3 warning cannot detect directional/code collapse. Measure actual level occupancy, code entropy and reconstruction gradients, not that warning alone. This modification has not been evaluated here.

No independent rescore of segmentation predictions was possible because the predictions were not saved. The numerical summaries above are faithful to segmentation.csv, with the listed limitations.

## Next action

Do not expand to LARa on the strength of these scores. First make a small, explicitly versioned FSQ training repair that keeps inputs to the bound in a usable range. Verify live encoder gradients, nonconstant real-patch assignments and reconstruction on a short run before launching another downstream grid. Do not silently interpret the existing three-coordinate LayerNorm edit as the old model or assume its saturation warning proves health.

Fix tokenizer-seed handling and artifact/status saving before the next complete comparison. An epoch-0 encoder-K-means control would test whether the promising upstream result requires the current training at all; it requires new downstream work and was not run during this audit.

The current useful finding is that pre-bottleneck features can remain decodable after the native FSQ pathway has failed. The run does not establish an advantage for functioning FSQ, an advantage for continuous representations over discrete units, or that encoder objectives no longer matter.
