# FSQ reconstruction pilot: a bounded feasibility comparison

Date: 27 September 2026. Revision 3. Status: plan only; this revision does not launch implementation or training.

This revision supersedes the earlier three-arm, three-seed proposal. The required experiment is HuGaDB, one paired autoencoder seed, two autoencoders: VQ512 and FSQ512 with levels [8,8,8]. Evaluate each frozen autoencoder with three matched downstream seeds. FSQ9 remains removed. A released-encoder reference, LARa and autoencoder replication are prioritized follow-ups within the same 36-hour budget.

## 1. Question and scope

Before investing further in OT, test whether an FSQ adaptation of SMQ offers a useful reconstruction, native-token, or encoder-feature signal relative to a freshly trained VQ reference.

Answer three questions:

1. Can this FSQ adaptation train stably and reconstruct reasonably under the existing SMQ reconstruction objective?
2. Do its native codes support better segmentation than matched native VQ codes?
3. Do K-means units from its encoder support better segmentation than K-means units from the VQ encoder?

This is an opportunity screen, not a publishable superiority test. One autoencoder seed can demonstrate operational feasibility and motivate follow-up; it cannot establish training robustness or rule out FSQ after a weak result. Three downstream seeds characterize downstream initialization sensitivity conditional on those fixed autoencoders. Reconstruction and code utilization do not by themselves establish action semantics.

Use HuGaDB first, 15-frame patches, the existing input channels and raw latent preprocessing. Use LARa with 12-frame patches only under the follow-up rule below. Train on one RTX 4090; cap aggregate numerical-library CPU threads at eight or the available logical count, whichever is smaller.

Keep code concise, readable and self-contained. Reuse model blocks, data loaders, tokenization helpers, contextualizer and OT/scoring implementations. Preserve historical code, checkpoints and results.

This pilot retains the reconstruction objective. It does not compare MAE-style, contrastive or other encoder-training objectives; a null result cannot establish that changing the encoder is unnecessary.

Out of scope: FSQ9, a quantizer-layout sweep, MAE, extra data, BABEL, pose conversion, new reconstruction or temporal objectives, end-to-end OT training, new decoders, hyperparameter searches, and continuous-ASOT arms.

## 2. What the comparison changes

SMQ uses 16 continuous channels per joint and concatenates them across joints before patch quantization.

| Dataset | Joints | Features per frame | Pilot patch | Flattened encoder patch |
|---|---:|---:|---:|---:|
| HuGaDB | 6 | 96 | 15 frames | 1,440 coordinates |
| LARa | 22 | 352 | 12 frames | 4,224 coordinates |

The encoder retains these dimensions in both arms. Only the FSQ quantizer adapter projects a complete patch to three scalar coordinates.

- VQ512 learns 512 patch vectors in the full patch space.
- FSQ512 projects a patch to three scalars, bounds and rounds each to eight levels, and projects the resulting grid point back for the existing decoder.

Both transmit one of 512 symbols per patch: at most nine bits of code identity. They do not have equal parameterization or optimization objectives. VQ has freely learned code vectors; FSQ has a fixed product grid plus learned adapters.

With a linear FSQ output adapter, the reconstructed latent patches lie in an affine subspace of dimension at most three before the decoder. The nonlinear decoder can map these points to complex motions; the final reconstructed motion is not restricted to a three-dimensional linear subspace. This is a real structural difference, not proof of a performance disadvantage.

The earlier low-dimensional projection experiment used a different setup. It motivates inspection but does not predict this jointly trained model's outcome. Do not treat it as evidence that three-coordinate FSQ must fail.

The first test therefore compares two complete quantizer adaptations under a shared backbone and reconstruction target. It does not isolate rounding from nearest-neighbor assignment. A later adapter-matched VQ control would be needed for that attribution.

## 3. Required experiment

| Setting | Fixed choice |
|---|---|
| Dataset | HuGaDB |
| Patch length | 15 frames |
| Arms | VQ512; FSQ512 [8,8,8] |
| Autoencoder training seed | 111 |
| Training endpoint | Epoch 30 |
| Preprocessing and sample seed | 111 |
| K-means, contextualizer and OT seeds | 111, 222, 1538574472, with separate RNG streams |
| Primary downstream endpoint | Contextual ASOT F1@50; report all other metrics |
| Other downstream evaluation | Direct categorical OT |
| Normalization | No per-recording latent z-scoring |

Required totals: **two autoencoder fits, four stream types, 12 contextualizer fits, 12 categorical OT fits and 12 contextual ASOT fits**. Each native token assignment is fixed across downstream seeds; each encoder-K-means stream has a vocabulary fitted at each downstream seed. Final solver checkpoints are continuations, not additional model arms.

Pair downstream seeds across the two autoencoders. Within a downstream replicate, K-means, contextualizer and OT each use separately namespaced RNG streams derived from that seed. Keep fitting samples/PCA and recording splits fixed at 111. The resulting variation combines downstream sources; it does not isolate any one stage.

For each autoencoder, evaluate:

1. Its native quantizer IDs.
2. K-means-512 IDs from its full encoder patches before quantization or FSQ projection.

Compare native with native and encoder K-means with encoder K-means. Within-model native-versus-K-means differences are descriptive: their fitting procedures, geometries and downstream models differ.

There is no mandatory three-autoencoder-seed gate. Report the three downstream values, their mean and population SD, and paired FSQ-minus-VQ differences. Also report the mean and population SD of those paired differences; marginal spreads alone do not determine paired uncertainty. Label all summaries as conditional on autoencoder seed 111, not training robustness, sampling uncertainty or statistical significance.

## 4. Matched autoencoder training

Train both arms from scratch. Copy exactly identical initial encoder and decoder tensors into the two models and save their initialization hashes. Implement the matching, rather than relying on a single initial manual_seed call:

1. Construct the shared encoder/decoder weights once, then copy them into both models. Initialize VQ eagerly with the existing random initialization law and initialize FSQ adapters in isolated CPU/CUDA RNG contexts. The VQ initialized flag, embedding, EMA sums and counts must be valid before epoch 0; do not consume a training update merely to initialize them.
2. After imports and initialization, explicitly reseed Python, NumPy, torch CPU and all CUDA generators identically immediately before training each arm. batch_gen.py resets Python's RNG at import, so import-time state is not the training seed.
3. Save a sorted recording list and precompute each epoch's order with a dedicated local RNG. Feed that schedule to both arms rather than depending on os.listdir order or a shared random.shuffle stream.
4. Isolate all quantizer random operations from the training dropout RNG. The current replacement helper already uses a local generator seeded at 42 on each forward; preserve and record that behavior. Verify no quantizer path advances the global dropout stream after eager initialization.
5. Save and restore CPU, CUDA, Python, NumPy and local-generator states on resume. Diagnostics must restore RNG and model state. Test shared dropout-mask equality on a short paired training run; matching weights alone is insufficient.

Use separate training processes if convenient, but do not assume that separate processes alone guarantee matching. Numerical equality across hardware is not promised.

Retain the repository's SMQ configuration:

- Backbone width 128, three layers, latent width 16 per joint.
- Existing input channels, sampling rate, masks and decoder.
- Dataset batch size 8; no microbatching in the primary comparison.
- Adam learning rate 5e-4; preserve and record the remaining optimizer defaults.
- Thirty epochs, with checkpoints at epochs 0, 10, 20 and 30.
- Existing joint-distance reconstruction target and coefficient 0.001.
- VQ commitment coefficient 1.0, EMA decay 0.5, dead-code threshold **1**, representative replacement and sampling quantile 0.5; use the existing default initialization path, kmeans=False.
- FSQ commitment loss zero, no learned lookup table, EMA or dead-code replacement.
- Temporal-consistency loss zero in both arms.

The threshold of 1 is a declared adaptation to the 512-code alphabet, not a reproduction of the released small-alphabet configuration. At roughly 1,670 valid patches per HuGaDB batch, uniform usage is about 3.3 patches/code; threshold 10 exceeds that mean. A replacement reset to 10 followed by EMA decay 0.5 can immediately fall below 10 again. Threshold 1 approximately retains the released threshold-to-mean-occupancy ratio. Verify actual valid-patch counts from the scheduled batches, including partial terminal patches, rather than treating those approximate counts as measured pilot results.

Log valid patches, assignments, EMA-count quantiles, expired/replaced codes and repeat-to-fill events each step. After epoch 1, flag replacement fractions above 5% for inspection, but do not use 5% as a universal correctness gate or tune the threshold to force it. Examine persistent near-total replacement before accepting VQ as a usable reference; do not report an FSQ advantage against a demonstrably malfunctioning VQ run. Use threshold 1 on conditional LARa too, with its own occupancy audit; no score-driven threshold sweep.

Record every resolved setting in the manifest; do not inherit undocumented command-line defaults. Preserve the existing reconstruction reduction, including padding treatment. Additionally report an unweighted-by-loss-coefficient, valid-frame reconstruction diagnostic with its exact denominator. Do not silently replace the training objective with that diagnostic.

Evaluate reconstruction and extract every feature/token stream with model.eval() and one unpadded recording per batch (quantizer-internal terminal padding is retained). Use inference/no-grad mode except for the explicit gradient diagnostics below. Apply the same extraction policy to the released-encoder reference. The initial biased 1x1 convolution precedes masking, so cross-recording batch padding can affect terminal features; do not change training masking as an incidental fix in this pilot. Test repeated extraction and fixed recording order for exact native-ID agreement.

Do not compare only stochastic training-batch losses. Save separate reconstruction and commitment terms, not just total loss, because the total objectives differ.

On a fixed saved batch at epochs 0, 10, 20 and 30, measure gradients of the weighted reconstruction term and weighted commitment term separately with respect to the same prequantization encoder output. Use eval mode with autograd enabled, no optimizer step, no EMA/replacement update, and restore all RNG/model state afterward. Report valid-coordinate gradient L2 norm and RMS, their ratio (undefined if the denominator is zero), and gradient cosine when both are nonzero. FSQ's commitment gradient is exactly zero. The forward diagnostic must leave checkpoint hashes unchanged.

These norms describe the local training signal, not Adam parameter-update magnitudes or causal attribution. VQ receives reconstruction plus commitment gradients while FSQ receives reconstruction alone. Adam can approximately cancel a global loss rescaling, but epsilon and other numerical/optimizer effects prevent claiming that the 0.001 factor is exactly irrelevant.

Use epoch 30 for downstream work regardless of action scores. Record whether reconstruction is still improving at the cap. Do not extend one arm's epochs or change its learning rate after seeing results. A fixed-budget comparison is not a comparison of fully optimized models.

The released SMQ checkpoints are historical references only: their alphabet and patch sizes differ. This pilot does not need to rerun D7; verify the unchanged evaluator against saved historical predictions instead.

## 5. FSQ implementation contract

For each encoder patch:

```text
SMQ encoder -> [W,D] -> flatten [W*D]
            -> Linear(W*D,3)
            -> reference FSQ bounding + rounding with straight-through gradients
            -> three quantized scalars, levels [8,8,8]
            -> Linear(3,W*D) -> reshape [W,D]
            -> existing SMQ decoder
```

Use biased linear adapters and record their initialization. Follow a pinned FSQ reference implementation for the even-level bound, offset, rounding and value normalization; a generic tanh-plus-round implementation is not sufficient without parity checks.

The author implementation is JAX; adding JAX to the training environment is not required. Use independently derived, committed formula-based fixtures covering the eight-level offsets, bounds, ties and all code values. If parity was checked only against these fixtures, report formula parity rather than claiming execution parity with JAX.

Use mixed-radix indexing to produce one integer ID in 0..511 per patch. A mask/padding token for the contextualizer is additional infrastructure, not a 513th motion unit.

Preserve SMQ's partial-terminal-patch policy, padding, cropping and frame masks. Fully padded patches contribute neither occupancy nor fitting samples. Retain every real frame in evaluation.

Return the shapes and attributes expected by the existing SMQ trainer, including zero commitment and temporal losses. Where VQ-specific distances are unavailable, make that explicit rather than fabricating distances.

Cache full encoder patches or a bounded-memory equivalent, pre-bound projection values, bounded pre-rounding coordinates, quantized coordinates, patch IDs and recording/start/length metadata. The output adapter's patch vectors are latent templates; do not call them independently decoded motions when the temporal decoder uses neighboring context.

## 6. Preprocessing and token streams

Use the same recording/start identifiers for fitting the two encoder K-means vocabularies. Uniformly sample at most 10,000 complete patches without replacement with seed 111. Fit per-coordinate standardization and randomized PCA-64 on that sample only; reduce the PCA dimension to the maximum feasible rank if necessary and record it.

Fit full KMeans with K=512, k-means++ initialization, n_init=5, max_iter=300, tol=1e-4 and Lloyd's algorithm. Record convergence and occupied clusters. Do not substitute MiniBatchKMeans or silently reduce K. If fewer than 512 fitting patches exist, mark the comparison unavailable.

Transform and assign in bounded batches. Exclude incomplete terminal patches from fitting; zero-pad them for assignment and retain their real frames in scoring. Save fitted scalers, PCA models, prototypes, sample identifiers and assignments.

For native streams, IDs come directly from their saved autoencoder. Do not infer them from historical prediction caches. Verify exact equality between cached and recomputed IDs.

Keep preprocessing randomness fixed at 111 across all downstream seeds and any second autoencoder seed; only downstream clustering/model randomness follows the downstream replicate seed. If autoencoder seed 222 is added, it receives the same three downstream seeds. No extra K-means fit on the three FSQ scalars is required for this initial screen. Saving those coordinates permits later diagnosis without retraining.

### Released-encoder reference

After the required paired HuGaDB comparison, prioritize one additional encoder-K-means-512 stream from the released SMQ checkpoint, evaluated at the same three downstream seeds and through both decoders. This requires no autoencoder training but does require three fresh vocabularies/contextualizers and the corresponding OT fits; it is not a zero-cost reference. If completed, HuGaDB totals become 15 contextualizers and 15 fits per decoder.

Verify the full checkpoint hash against the existing released-checkpoint provenance (HuGaDB prefix 22dfd5639c718177; LARa prefix a45392ad30e27bda), recording/frame identities and single-recording eval extraction. Reuse cached encoder features only if all of those properties match. Refit K=512; old K=500 tokenizer/contextualizer artifacts are not interchangeable.

This reference asks whether the newly trained encoder pipeline improves over the released C-code encoder under the new downstream protocol. Differences also include the original training patch size and checkpoint training history; they do not isolate alphabet size or quantizer type. Present it separately from the matched VQ512/FSQ512 comparison. Historical Stage C numbers remain unmatched references.

This reference is budget-conditional, runs before LARa or a second autoencoder seed, and does not enter the FSQ-versus-VQ progression threshold. If it cannot fit, record the retraining-versus-released-encoder question as unanswered.

## 7. Matched downstream evaluation

### Direct categorical OT

Use the existing categorical ASOT objective with 512 emission categories, the existing dataset action count (HuGaDB 10; LARa 8, including background), frame-mass weighting and log(512) cost normalization.

Initialize action states consistently: calculate each occupied unit's frame-weighted mean in that arm's encoder-window PCA space and cluster these means into C groups with unit frame counts as weights, using the existing initialization helper. Record that even native-code decoding uses encoder geometry for initialization. Never calculate distances between numerical code IDs.

Use the existing symmetric emission prior, with total pseudocount per state equal to dataset fps, distributed across all 512 categories. Empty units have no initialization weight but remain in the emission alphabet.

### Contextual ASOT

Train a fresh masked-code contextualizer for each of the four streams. The old stage_b_003 embedding table cannot be reused: new code IDs have no alignment with its vocabulary.

Reuse the selected configuration through stageb_configs.resolve_by_name:
postLN_cosine_wu500_lr0.0003_alibi. Preserve the existing selected model architecture, token inference/stitching, recording split and label-free checkpoint selection. Save the fully resolved model and training configurations and their hashes.

Use alphabet 512, context length 128, stride 64, mask span 8, mask fraction 0.30 and at most 5,000 optimizer updates. Preserve evaluation every 100 steps and patience of 1,000 steps. Match recording memberships, chunk order and masks across streams; equal caps and stopping rules need not yield identical update counts. Report actual counts.

Fit the existing contextual ASOT decoder using cosine cost **1 - z dot mu**, with normalized vectors, not half that cost. Preserve the existing temporal kernel and per-recording normalization. Padding must not change a recording's kernel amplitude.

Both downstream methods treat native codes as categorical IDs. Stage B learns an embedding per ID; categorical OT has no explicit FSQ-grid distance. Thus this tests FSQ as a learned token generator, not a decoder designed to exploit its grid geometry. The encoder-based OT initialization supplies separate geometry, which must also be disclosed.

### Solver settings for both methods

Use setting T only: emission coefficient 0.7, temporal coefficient 0.3, marginal KL coefficient 0.05, entropy coefficient 0.07. No coefficient search or E sensitivity.

Use 50 outer fitting iterations, 25 inner steps, and the existing stopping tolerances. Cold-start the final solve using the verified frozen-study initialization, with a maximum of 4,000 steps and saved diagnostics at 500, 1,000, 2,000 and 4,000 where reached. Preserve legitimate early stopping. Apply the same policy to every arm and record convergence or capping separately.

The primary value is the final available endpoint under this rule. Do not select a solver checkpoint by action score or describe a warm-versus-cold initialization difference as a quantizer effect.

Do not modify global historical K=500 settings or rely on parent-process variable changes reaching spawned workers. Pass an explicit experiment configuration with K=512 through every helper and artifact identity.

## 8. Diagnostics and limits of interpretation

For each arm report:

- Checkpoint reconstruction error under the existing objective, valid-frame reconstruction error, and training trajectories.
- Active codes, occupancy distribution and entropy, effective code count exp(entropy), and code-run lengths.
- FSQ per-coordinate level use and saturation; VQ dead-code replacement counts.
- Parameter counts separated into encoder, decoder and quantizer/adapters; runtime and peak memory.
- MoF, Edit, F1@10/25/50, predicted/ground-truth segment-count ratio, and all solver status flags for both streams and both OT methods.

Do not add supervised probe training or an oracle-prototype grid to this feasibility screen. If an implementation problem is suspected, use targeted saved-feature diagnostics and document them separately from the planned comparison.

Interpret patterns cautiously:

| Observation | Supported interpretation |
|---|---|
| Better native FSQ segmentation | A useful signal for this complete FSQ adaptation and decoder |
| Better FSQ-encoder K-means, weak native FSQ | Encoder-training opportunity; native FSQ utility remains unproven |
| Better reconstruction, no segmentation gain | Reconstruction improvement did not establish better action units |
| High occupancy, no segmentation gain | Code usage alone was insufficient |
| Weak native FSQ relative to its encoder K-means | A discrepancy worth investigating; not proof that projection alone caused it |
| Weak results with strongly improving loss at epoch 30 | An inconclusive fixed-budget outcome |
| Weak results with numerically healthy training | No demonstrated opportunity under this configuration; not a general rejection of FSQ |

All primary training and discovery are pooled/transductive. No encoder generalization, unseen-subject transfer or data-sufficiency claim follows. Contextualizer recording-held-out validation does not undo the encoder's pooled exposure.

Action labels are used for evaluation and the follow-up allocation rule below. They must not select checkpoints, quantizer layouts, decoder coefficients or unfavorable-result omissions. Hungarian matching is evaluation only. Historical cross-dataset comparisons must state differences in class counts, subjects and modalities.

## 9. Follow-up rule: exploration, not confirmation

Complete the entire HuGaDB autoencoder-seed-111 comparison, including all three downstream seeds, before allocating follow-up work.

A **promising segmentation signal** requires, in the same stream: a mean paired contextual ASOT gain of at least 2 F1@50 percentage points across all three downstream seeds, positive F1@50 differences in at least two seeds, and a mean paired MoF difference no worse than -1 point. This is a practical time-allocation threshold, not a significance test. Always report individual paired differences, the other stream and categorical results.

The frozen study showed substantial HuGaDB downstream spread (population SD about 3.95 raw and 5.46 normalized). This motivates replication; it does not prove that a particular paired 2-point difference is a coin flip, because the covariance between matched arms is unknown. Raising a single-seed cutoff to 5 is not a substitute for measuring sensitivity. Three downstream seeds still leave autoencoder-training variance unmeasured.

Use the following fixed priority:

1. Complete the released-encoder HuGaDB reference first if it fits. If there is a promising signal, then run the same two-arm, four-stream comparison on LARa at autoencoder seed 111 and all three downstream seeds, only if the complete paired comparison fits before the compute deadline. Its result remains exploratory.
2. If HuGaDB is ambiguous, a second complete paired HuGaDB seed (222) has priority over LARa. Define ambiguous, when the promising rule is not met, as at least one contextual stream having absolute mean paired F1@50 difference below 2 points, or a mean gain of at least 2 points failing the sign-consistency or MoF condition. A second autoencoder pair must receive the same three downstream seeds. This replication is optional and budget-dependent.
3. After a promising HuGaDB result and completed LARa follow-up, a paired HuGaDB seed 222 is the next optional job if time remains.
4. Otherwise stop and report. Do not add a third autoencoder seed or a new quantizer variant within this pilot.

If an ambiguous result receives a second autoencoder seed, first average paired differences over downstream seeds within each autoencoder pair. Progression requires the same stream to have an equally weighted two-pair mean F1@50 gain of at least 2 points, a positive mean F1@50 difference for each autoencoder pair, positive downstream differences in at least two of three seeds within each pair, and a two-pair mean MoF difference no worse than -1 point. Show the nested results (autoencoder seed, downstream seed); do not treat six downstream runs as six independently trained autoencoders. Two autoencoder seeds still provide limited robustness evidence.

Numerical failures, corrupt artifacts or unfinished autoencoder training take precedence over score-based allocation. Fix demonstrated implementation defects and rerun affected matched comparisons within the budget; do not retune methods to rescue scores. Merely reaching the predeclared OT iteration cap is a budgeted result, not automatically an implementation failure.

If the required follow-up cannot fit, record it as not run. Never run just the favorable arm without its matched reference. A result from one autoencoder pair may justify ending this time-limited branch, but cannot establish that FSQ cannot work.

## 10. Implementation organization and checks

Add a small isolated package under script/fsq_reconstruction/:

| File | Responsibility |
|---|---|
| quantizers.py | FSQ math and the SMQ-compatible patch adapter |
| experiment.py | Explicit configuration, paired training, extraction and scheduling |
| evaluate.py | Shared tokenizer/contextualizer/OT integration and reporting |
| tests.py | Targeted mathematical and integration checks |
| run.ps1 | One entry point for validate, smoke, run, resume and analyze |
| README.md | Exact commands, protocol, outputs and resume behavior |

Prefer direct imports and small functions over a new framework. Keep historical modules unchanged where adapters suffice. Do not duplicate the trainer or OT solver wholesale. The empty script/exp2_fsq/fsq_train.py is not a working implementation to rely on.

Before full execution verify:

1. Wrapped VQ matches the existing model's reconstruction, IDs, loss and gradients on a fixed batch with both instances configured at threshold 1 and identically initialized.
2. FSQ matches the documented reference formulas/fixtures on random inputs and even-level edge cases; all 512 ID/value round trips pass; straight-through gradients are finite and adapters update.
3. Both arms share encoder/decoder initialization, data order and tested dropout RNG behavior; eager initialization and diagnostic calls do not consume training RNG or mutate checkpoints.
4. Padding, terminal patches and real-frame coverage match the declared policy; all extraction uses eval mode and one recording per batch. Audit VQ replacement rates before interpreting outcomes.
5. Saved and recomputed native IDs agree exactly; cache identities include code hashes, model hashes, data IDs, patch lengths and configuration.
6. All codebook-dependent paths use 512, including logits, masks, emission priors and cost normalization.
7. Fresh contextualizers use identical splits/masks and cannot load a historical token embedding by accident.
8. Perfect synthetic predictions score correctly; an existing saved prediction reproduces its historical score with the reused evaluator.
9. Tiny smoke settings and deadlines propagate to subprocesses; resume skips only verified completed artifacts.

The user launches the full experiment. This document is not authorization to start training in the planning turn.

## 11. Budget and execution

Maximum allocation: 36 elapsed hours from the first implementation or compute action for this pilot. Persist that timestamp and the deadline before work starts; do not reset it on resume. Plan editing alone does not start the implementation clock. Reserve the final three hours for verification and reporting.

Priorities:

1. Implementation, parity checks, tiny smoke run and measured runtime projection.
2. Complete HuGaDB autoencoder seed 111, including all four streams, three downstream seeds and both OT methods.
3. Budget-conditional released-encoder reference, then LARa or paired autoencoder seed 222 according to section 9.
4. Report.

Run expensive GPU jobs sequentially. Cache checkpoints, token streams, contextual embeddings and solver outputs incrementally. Use measured autoencoder, contextualizer and OT timings to estimate the complete required comparison; neither old class-sized VQ runtimes nor the review's suggested 1-2 minute contextualizer timings are verified forecasts. Do not assume autoencoder training is the expensive stage. Admit the full three-downstream-seed grid only from measured estimates.

Do not start a job projected to cross the compute deadline. If the required HuGaDB comparison is projected not to fit, report that before a full launch and propose a concrete smaller protocol; do not silently remove the VQ reference, native stream or contextual decoder. Mark interrupted work incomplete rather than presenting it as a completed pilot.

No action-score-driven early stopping of an arm. Stop at the budget even if results are inconclusive.

## 12. Outputs and report

Write only to results/fsq_reconstruction/<run_id>/ with a new protocol identity for this revision:

- manifest.json: source and checkpoint hashes, pinned FSQ source, resolved settings, hardware/packages, recording IDs, sample IDs, splits, seeds, initialization hashes and deadlines.
- Checkpoints, optimizer/RNG state, encoder/projection caches, native IDs, fitted PCA/K-means artifacts, contextualizer checkpoints and embeddings.
- training.csv, reconstruction.csv, gradient_diagnostics.csv, occupancy.csv, segmentation.csv and runtime/status logs, with autoencoder_seed and downstream_seed in applicable rows.
- Raw and mapped predictions, solver traces, and not_run.csv for every omitted, failed or incomplete planned job.
- REPORT.md: a short decision summary, complete matched tables, reconstruction curves, code utilization, numerical status, limitations and follow-up status.

Keep reconstruction results, native-code segmentation and encoder-K-means segmentation visibly separate. Report percentage-point differences, not relative percentages, for segmentation comparisons. Label autoencoder seeds and downstream seeds separately. Report mean/population SD over the three downstream seeds and over paired differences, conditional on the fixed autoencoders. One- or two-downstream-seed partial cells are incomplete and cannot trigger progression.

Conclude whether the screen found a reason to invest further and which stream supplied it. Do not claim FSQ is intrinsically superior/inferior, that reconstruction guarantees semantics, that the projection has been causally isolated, or that this swap alone constitutes a novel contribution.

## Revision 3 review decisions

Accepted: adapt the VQ dead-code threshold before results; replicate downstream stages while keeping one initial autoencoder seed; explicit eager initialization/RNG isolation; eval-mode single-recording extraction; a budget-conditional released-encoder reference; per-loss encoder-gradient diagnostics; categorical-ID interpretation; and formula-based FSQ parity without adding JAX.

Qualified: a 5% replacement rate is a diagnostic flag, not a universal validity bound; downstream spread does not quantify the variance of paired differences; Adam is not exactly invariant to scaling; runtime claims need measurement. The reported frozen-study smoke configuration bug motivates explicit subprocess configuration tests. This plan does not independently certify its historical full run as unaffected without a saved-config audit.

## References

- [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/html/2309.15505v2): standard FSQ construction; Appendix A.4.1 recommends [8,8,8] for a 512-symbol alphabet. Its image results motivate the test, not a motion-performance guarantee.
- [Author reference implementation](https://github.com/google-research/google-research/tree/master/fsq): pin the exact revision used during implementation.
- Local architecture and training: src/model/smq.py, src/model/motion_quantizer.py, main.py, model.py.
- Selected contextualizer: script/action_transport/stageb_configs.py.
- Decoder configuration: script/action_transport/config.py; reuse with explicit pilot overrides.
- Current diagnostic evidence: results/frozen_study/frozen_001/ANALYSIS.md.
