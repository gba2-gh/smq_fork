# Q1/Q2 agent instructions: motion-unit information and readout diagnostics

## 1. Objective, scope and provenance

Measure what the units retain about action and subject, and how action recovery changes with vocabulary size, representation and readout. Deliver measurements and limitations, not a recommendation about the project’s contribution.

Use **HuGaDB and LARa, frozen released checkpoints, CPU only**. Do not run merge curves, decoders, duration models, BABEL experiments or encoder retraining.

Preserve all existing scripts and results. Put new code under `script/exp2round/q1q2/` and all new outputs under `results/exp2round/q1q2/`. Do not overwrite the existing `results/exp2round/REPORT.md`.

Reuse applicable functions from the existing latent dumper, vocabulary sweep and scoring helpers. Do not run the old summarizer: its gates, dataset assumptions and conclusions are specific to the previous experiment.

**Checkpoint policy**

- HuGaDB: `models/pretrained/hugadb.model`, SHA-256 beginning `22dfd5639c718177`.
- LARa: `models/pretrained/lara.model`, SHA-256 beginning `a45392ad30e27bda`.
- Record complete hashes in the manifest.
- The existing HuGaDB `latents_hugadb_epoch30.npz` uses a different checkpoint. Do not reuse it for this round.
- Reuse other latent caches only when checkpoint hash, recording identities, frame counts and latent layout are verified. Otherwise extract new latents on CPU.
- Save predictions generated from the same checkpoint as the latents. Do not substitute historical predictions merely because agreement is high.

For each dataset, verify that quantizing the cached latents reproduces the same-checkpoint model predictions exactly. Compare separately with the historical released-prediction cache; require frame agreement ≥0.999.

Recompute D7: ground-truth segments assigned their majority original SMQ code, followed by dataset-level Hungarian matching. Require MoF within 1.5 points of **44.85 for HuGaDB** and **41.66 for LARa**. If a gate fails, stop work on that dataset and report the discrepancy. Continue the other dataset if its gates pass.

The old `exp2round` scores are historical references, not matched controls: this round changes the clustering fit policy and uses the released HuGaDB checkpoint.

## 2. Shared experimental rules

**Grid and randomness**

- HuGaDB windows: \(W\in\{60,30,15\}\).
- LARa windows: \(W\in\{50,25,12\}\). Use integer floor for quarter-patch windows.
- Vocabulary sizes: \(K_u\in\{10,20,50,100,500,1000\}\).
- Clustering seeds: **111, 222, 1538574472**.
- Fix splits, subsampling and PCA randomness at seed **111**, independently of the clustering seed.
- Use the same selected fitting windows across vocabulary sizes and clustering seeds for a given dataset, window size, representation and fold.
- Record runtime, sample counts, convergence status and cache identity per cell.

**Windows and preprocessing**

Use non-overlapping windows of flattened frame features. Exclude incomplete terminal windows from fitting; zero-pad them before applying the fitted transformations for assignment. Retain every frame in evaluation.

For each fitting population, sample uniformly without replacement at most **10,000 complete windows**. Save their recording/start-frame identifiers. Fit per-coordinate standardization and randomized PCA-64 on this sample only. Use 64 components, or the maximum feasible rank if smaller, recording the actual dimension and explained variance.

Transform and assign all windows in bounded batches. Do not materialize the full collection of high-dimensional flattened windows in memory.

Use full `KMeans` at every \(K_u\), with k-means++ initialization, `n_init=5`, `max_iter=300`, `tol=1e-4`, and Lloyd’s algorithm. Fit on the fixed sample; assign all windows afterward. Do not switch to MiniBatchKMeans.

For final segment clustering, use the same KMeans settings with \(C\) clusters, where \(C\) is the dataset’s existing action-class count, including `none`/`None`.

**Hard and soft segment representations**

The hard histogram is frame-weighted: each window contributes according to its overlap in frames with the ground-truth segment. Normalize to sum to one, then take the elementwise square root before clustering or logistic regression.

For soft assignments, use squared Euclidean distances in PCA space:

\[
p(k\mid x)=\operatorname{softmax}_k(-\|x-c_k\|^2/\tau).
\]

Set the base temperature to the median strictly positive difference between the second-nearest and nearest prototype distances on fitting windows. If no positive difference exists, mark the soft cell undefined.

Use multipliers **0.5, 1 and 2**. Aggregate using the same overlap-frame weights, normalization and square-root transform as hard histograms. Hard and soft comparisons must share the fitted scaler, PCA and vocabulary.

Define the continuous segment representation as the overlap-frame-weighted mean of PCA-64 window features. Do not call it a mean of the original frame latents.

**Evaluation populations**

Keep two settings distinct:

- **Pooled diagnostics:** preprocessing, vocabulary and segment clustering fit on the whole collection without action labels; ground-truth segments and evaluation label mapping make these oracle-boundary, transductive diagnostics.
- **Cross-validation diagnostics:** fit preprocessing, vocabulary, temperature and readout exclusively on fitting recordings. The frozen encoder remains transductive.

Use four folds for both recording-grouped and subject-grouped cross-validation. Construct folds without action labels: sort groups by decreasing frame count, break ties using seed 111, and greedily assign each group to the fold with the fewest frames. Save memberships and reuse them across all compared representations and seeds.

## 3. Tasks

### Task A — Information diagnostics

The old sweep did not save independent unit assignments. Recompute them under this round’s shared pipeline; do not attempt to infer exact information quantities from saved NMI summaries.

For each pooled hard-vocabulary cell, use one observation per assigned window. Its action is the majority frame label in that window; its subject comes from the recording identifier. Include terminal windows consistently and record their count.

Compute:

- \(I(U;A)\), \(I(U;S)\), and \(I(U;S\mid A)\), in nats.
- \(H(U)\), \(H(A\mid U)\), and \(H(S\mid U)\), in nats.
- Arithmetic-normalized NMI and AMI for unit–action and unit–subject assignments.
- Number of windows, occupied units and unit occupancy distribution.

For marginal information diagnostics, generate **20 global permutations of the observed unit assignments**, preserving exact unit counts. For conditional subject information, generate **20 permutations within action groups**. Use permutation seeds 111–130, fixed across clustering seeds.

Report observed values and permutation-reference means and SDs. Do not use the previous `shuffled_histogram` cells as these references: they sampled independent units and did not save the required quantities. Do not call every reference a “chance floor”; marginal entropy is unchanged by count-preserving permutation, and AMI already includes a chance adjustment.

These permutations are descriptive reference distributions, not valid uncertainty estimates for independent temporal observations.

### Task B — Matched pooled sweep on both datasets

Run the full shared grid, subject to the budget policy below.

For every cell, report the original scoring helpers’ MoF, Edit and F1@10/25/50, plus Task A’s diagnostics.

Run these explicit controls:

- **Majority-unit control:** assign the dominant unit to each ground-truth segment. Report both Hungarian mapping and evaluation-label many-to-one mapping. Label the latter an oracle mapping; quantify their difference without assuming its cause.
- **Continuous control:** cluster continuous PCA-64 segment means into \(C\) clusters.
- **Permutation control:** globally permute window assignments using `default_rng(seed + 10000000)`, preserving exact counts, then repeat histogram construction and segment clustering.
- **Length-weighted control:** repeat final hard-histogram segment clustering with segment duration as sample weight at every cell. Do not select a “best cell” using evaluation scores.

Do not run the historical no-PCA control.

### Task C — Soft assignment and supervised readouts

Use both datasets, every \(K_u\), and only full- and quarter-patch windows.

For pooled clustering, compare hard histograms, soft histograms at the three temperatures, and continuous segment means.

For both cross-validation groupings, fit class-balanced logistic regression to:

1. Hard segment histograms.
2. Continuous PCA-64 segment means.
3. Soft segment histograms at each temperature.

Use L2 regularization, inverse regularization strength `C=1`, an intercept, LBFGS, `max_iter=2000`, and `tol=1e-4`. Use multinomial logistic regression when there are more than two training classes. Do not tune hyperparameters using evaluation labels or add feature standardization after the representations defined above.

Each training segment is one sample; class balancing operates on training-segment counts. Report:

- Out-of-fold segment balanced accuracy.
- Frame MoF after broadcasting each segment prediction over its frames.
- Per-class recall, fitting/evaluation class support, and fold sizes.

Compute primary metrics from pooled out-of-fold predictions within each seed. If a training fold lacks a class, retain that fold and report the missing class; do not alter splits. If fewer than two training classes are present, or optimization fails to converge, mark the affected result invalid and report it.

Continuous readouts are deterministic under the fixed preprocessing and splits. Compute once and mark clustering-seed spread as not applicable rather than inventing three independent fits.

Call these **supervised readout diagnostics**, not information-theoretic ceilings. The recording-versus-subject fold gap is a protocol difference; it does not isolate the effect of previously seeing a person.

### Task D — Restricted Q1 controls

Use \(K_u\in\{10,100,1000\}\) and full- and quarter-patch windows.

1. **Raw input features:** repeat the pooled hard-histogram sweep and Task A diagnostics using the model’s existing raw input channels, without the encoder. Apply the same fitting, windowing, PCA and clustering rules.

2. **Per-recording latent standardization:** z-score each latent coordinate over that recording before windowing, then apply the shared pipeline. Use population SD; replace SD below \(10^{-8}\) with 1. Repeat the pooled hard-histogram sweep and information diagnostics. This uses the complete recording and is not an online method.

3. **Within-subject segment clustering:** keep the pooled latent vocabulary and hard histograms fixed. Refit only the final \(C\)-cluster segment model separately within each subject; Hungarian-match separately per subject and pool frame predictions. Compare with pooled clustering from the identical cell. If a subject has fewer than \(C\) segments, mark that subject’s fit unavailable and report coverage. State that separate mappings add oracle flexibility.

4. **Unit reuse:** reuse Task C’s subject-disjoint vocabularies. For each held-out subject, a unit is “used” if at least one complete window receives that unit. Report the fraction of all \(K_u\) units used by at least half the held-out subjects, using the ceiling of half their count.

   For each unit, first divide its assignment count in each subject by that subject’s complete-window count; then normalize these rates across subjects and compute subject-distribution entropy in nats. Also report entropy divided by the log of the number of held-out subjects. Units unused by every held-out subject have undefined entropy and must be reported separately. Retain fold-level results because different folds fit different vocabularies.

## 4. Budget, failure handling and verification

The hard limit is **eight elapsed hours from the first implementation or compute action**, including validation and reporting. Reserve the final **30 minutes** for reporting. Run expensive fits sequentially and cap numerical-library threads at eight or the available logical CPU count, whichever is smaller.

Execution priority:

1. Provenance, gates, implementation checks and artifact caching.
2. Task A/B/C core: full-patch \(K_u=10,100\), both datasets, all three seeds; Task C base temperature only.
3. Remaining Task A/B/C base-temperature cells.
4. Task D.
5. Task C’s 0.5× and 2× temperature sensitivities.

Within each stage, alternate datasets and complete all three seeds of a configuration before starting the next. Process remaining vocabulary sizes in order **10, 100, 1000, 20, 50, 500**, skipping completed values; prioritize full-patch, then quarter-patch, then half-patch windows where applicable.

If runtime projections exceed the remaining budget, omit **Task D’s \(K_u=1000\)** first, then extra temperatures, then remaining unstarted work in reverse priority order. Task D has no \(K_u=500\). Do not start an expensive job projected to cross the compute deadline. Enforce the deadline and mark interrupted cells incomplete.

This omission policy is the only permission to shrink the grid. Do not substitute methods, representations, seeds or hyperparameters after failure. Missing data, memory failures and nonconvergence become explicit “not run” or “invalid” entries. Report one- or two-seed partial cells as incomplete; do not present them as three-seed results.

Verify before full execution:

- Checkpoint/cache identity, quantization agreement and frame alignment.
- Subject parsing and absence of group leakage.
- Fitting-only preprocessing and temperature selection in cross-validation.
- Histogram mass, boundary-overlap weighting and hard/soft aggregation consistency.
- Information calculations on known independent and deterministic examples.
- Exact count preservation in permutation controls.
- Many-to-one versus Hungarian mapping on a deliberately fragmented synthetic example.
- Out-of-fold prediction coverage and both segment- and frame-level scoring.

Save completed artifacts incrementally, including fitted transformations, prototypes, assignments, contingency counts, fold manifests and predictions. Reuse verified artifacts rather than refitting equivalent cells.

## 5. Outputs and reporting rules

Write under `results/exp2round/q1q2/`:

- `manifest.json`: provenance, full hashes, package versions, splits, sample identifiers, configuration and deadline.
- `q1q2_cells.csv`: seed-level clustering metrics and cell status.
- `q1q2_information.csv`: observed and permutation-reference information measures.
- `supervised_readouts.csv`: fold-level and pooled out-of-fold metrics.
- `unit_reuse.csv`: fold-level reuse and per-unit entropy.
- `not_run.csv`: every omitted, failed, invalid or incomplete planned cell and its reason.
- Cached artifacts and runtime logs.
- Three figures: information versus \(K_u\); clustering and supervised frame MoF versus \(K_u\), with balanced accuracy in a separate panel; unit reuse versus \(K_u\).
- `REPORT.md`: ten-line summary, tables, **Open questions**, **What these numbers cannot establish**, and **Not run**.

Keep pooled clustering and cross-validation results visibly separate in figures and tables. Report mean and population SD across the three clustering seeds where applicable; distinguish permutation SD and fold variation. Seed variation measures initialization sensitivity, not sampling uncertainty.

Disclose action-label use in ground-truth segment construction, Hungarian and many-to-one mappings, supervised fitting/evaluation, information diagnostics and conditional permutations. Subject identifiers may be used for grouping and diagnostics. No labels may select unsupervised hyperparameters or omit unfavorable results.

All frozen-encoder results remain transductive. Raw-feature results have no encoder exposure, but pooled raw-feature fitting is also transductive. Every HuGaDB–LARa comparison must note class count, subject count and modality; HuGaDB cannot establish body-shape effects.

Permitted statements describe measured differences and limitations. Do not claim which component is “the bottleneck,” what “dominates,” what the project should do next, or what constitutes the contribution. Put possible interpretations under **Open questions**, phrased as questions.
