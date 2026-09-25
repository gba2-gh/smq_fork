# Fine-to-coarse motion segmentation
## Three-stage research and implementation plan

25 September 2026 | Version 1.0 | Planning document; no experiment launched

### Purpose

Test whether a sequence of fine motion codes can support action segmentation without equating individual codes with actions. Keep tokenization fixed while introducing temporal assignment, contextual representation learning, and then their combination. The immediate deliverable is a reliable categorical optimal-transport baseline; the eventual research claim depends on the controlled comparisons, generalization, and later tokenizer evidence.

This plan fixes the three-stage order discussed with the project owner. Numerical defaults introduced here are prospective design choices, not previously validated optima or settings taken unchanged from a paper. Freeze them in a versioned configuration before accessing new action scores. Implementation agents should deliver readable scripts and one execution entry point; the user launches the full experiments.

## 1. Architecture and evidence boundary

The shared input pipeline is: existing sensor or motion-capture channels → frozen released SMQ encoder → standardized/PCA-transformed windows of prequantization latents → frozen K-means vocabulary → hard code sequence. Stages A–C consume that code sequence. They do not receive continuous encoder features as an additional decoder input.

| Stage | Added component | Output and question |
| --- | --- | --- |
| A — Categorical transport | Action-state distributions over codes; alternating temporal transport and emission updates | Can temporal assignment organize fixed codes into action segments? |
| B — Masked-code context | Small bidirectional Transformer trained to predict masked code spans | Does ordered code context produce useful representations beyond individual code identity? |
| C — Contextual transport | Stage A transport machinery with costs from frozen Stage B embeddings | Does context add value beyond temporal regularization, and does transport add value beyond context? |

The exp2round and Q1/Q2 vocabulary studies concern posthoc K-means units. They do not establish the quality of a 500-state FSQ bottleneck. Use K=500 as a fixed starting vocabulary, not a discovered optimum. Oracle-boundary diagnostics establish recoverable associations under their stated protocols; they do not establish unsupervised boundary recovery or subject-invariant primitives.

The released checkpoints used here have learned VQ quantization. The proposed K-means tokenizer uses their prequantization features and is a different pipeline from native SMQ, HistVQ, or a trained FSQ system. A future matched FSQ replacement is a separate experiment after these stages, with the same downstream decoder and evaluation. Do not label the current pipeline FSQ.

Existing Milestone 1 and corrected-rerun artifacts remain historical comparisons. Stage A replaces the segment dynamic program with temporal transport. It does not inherit the old lambda calibration, maximum segment duration, or performance gate. Keep previous reports intact and describe their exact protocols when comparing them.

## 2. Shared experimental contract

### Data, exposure, and provenance

| Dataset | Action states, including background | Subjects in existing collection | Modality | Sampling and code window |
| --- | --- | --- | --- | --- |
| HuGaDB | 10 | 18 | Wearable IMU | 60 Hz; 15 frames, 0.25 s |
| LARa | 8 | 16 | Optical motion capture | 50 Hz; 12 frames, 0.24 s |

Verify these counts against the loaded recording inventory. Different class inventories, subject counts, and modalities prevent a controlled causal comparison between datasets. HuGaDB cannot establish body-shape effects. Supplying the action-state count is an explicit assumption; this plan does not discover the number of actions.

Use models/pretrained/hugadb.model with SHA-256 22dfd5639c718177c12bc48debf67371c4bf7ca64af2d7de4e83ebacae55d7d2, and models/pretrained/lara.model with SHA-256 a45392ad30e27bda400adcab541aba087e4ae56b68837547cc65ca77ff5bfa65. Verify complete hashes, recording identities, frame counts, latent layout, and tokenizer configuration before reuse. Same-checkpoint predictions are the SMQ baseline. Historical D7 values are provenance checks, not an action-segmentation baseline. Reuse a verified D7 artifact when its entire dependency chain matches; recompute only when that chain changes or cannot be verified.

Run recording-level latent standardization off and on as separate, fully reported conditions. For the on condition, z-score each coordinate over its complete recording with population SD, replacing SD below 1e-8 by 1. This accesses the entire recording and is an offline operation. It does not establish identity removal.

### Frozen vocabulary and temporal alignment

Use hard K=500 assignments and seeds 111, 222, and 1538574472. Fix sampling, PCA, and fold randomness at 111. Per fitting population, uniformly select at most 10,000 complete, non-overlapping flattened windows. Fit per-coordinate scaling and randomized PCA with 64 components or the maximum feasible rank. Fit full Lloyd KMeans with k-means++, n_init=5, max_iter=300, tol=1e-4. Assign all windows in bounded batches. Exclude terminal partial windows from fitting; zero-pad for assignment, and retain their true frame counts for every later operation.

Save the sample identifiers, scaler, PCA, prototypes, hard assignments, and real frame spans. Reuse equivalent verified artifacts. A seed denotes a paired pipeline replicate, including vocabulary and decoder initialization; its variation is not isolated decoder uncertainty. Do not introduce a vocabulary-size sweep or soft-assignment grid in these stages.

### Two distinct evaluation settings

The primary pooled setting fits the vocabulary and stage-specific models on the full unlabeled collection. It is transductive. All recordings are segmented without ground-truth boundaries.

The secondary subject-disjoint setting uses the saved four subject-grouped folds if their inventory and construction are verified. Otherwise rebuild label-free folds by decreasing group frame count, seed-111 tie breaking, and greedy allocation to the least-loaded fold. Fit vocabulary, Stage A emissions, Stage B training/validation, and Stage C prototypes exclusively inside the fitting partition. Held-out recordings contribute only to their own transport inference and optional per-recording standardization. The frozen pretrained encoder may already have seen these subjects: call this downstream subject-disjoint evaluation, not an inductive encoder evaluation.

Never use labels or ground-truth segments to fit, initialize, select hyperparameters, choose checkpoints, set run counts, or decide which results to omit. Keep label arrays outside fitting APIs. Use labels only in the evaluator and clearly disclose Hungarian mapping, metrics, and any diagnostic boundary comparisons.

### Evaluation and reporting

Broadcast each window's argmax state to its actual frames, then merge contiguous identical states. Score MoF, Edit, and F1@10/25/50 using the existing scorer. Keep None/none and the existing background handling explicit. Save raw states, mapped predictions, mappings, and the scorer identity. Do not concatenate recordings before segment-level scoring.

Use one dataset-level Hungarian mapping in the pooled setting. In subject-disjoint evaluation, state identities differ across fitted models: map each held-out fold separately, then aggregate matched recording predictions using the scorer's original aggregation. Disclose the extra oracle flexibility of fold-specific mappings. Never align fold states by labels before model fitting. Report fold scores and pooled out-of-fold results separately.

Report all seeds, mean and population SD, paired differences, action-run counts, duration distributions, state occupancy, occupancy entropy, objective traces, convergence status, runtime, and coverage. A partial seed group is incomplete. Seed SD describes initialization sensitivity; fold variation is neither seed SD nor an independent-sample confidence interval. Predeclare F1@50 as the focal segmentation metric while displaying all five metrics; MoF gains alone do not demonstrate better segmentation.

## 3. Stage A — Categorical temporal transport

### Model and initialization

Let u_t be a hard code, n_t its real frame count, and theta[a,u] the code distribution for action state a. Theta is shared across fitting recordings of one dataset, normalization condition, seed, and fold. The emission cost is D[t,a] = -log(theta[a,u_t]) / log(500). This fixed normalization avoids fitting a changing cost scale to each iteration or recording.

Initialize from non-overlapping, exactly one-second blocks of the fitting recordings. Build frame-overlap-weighted code histograms, normalize, and take their square roots. Fit C-cluster KMeans with the vocabulary's stated KMeans settings; use at most 10,000 complete blocks sampled with seed 111. Use these assignments to pool raw frame-weighted code counts into initial theta. Add total pseudocount eta=fps per state, distributed uniformly over 500 codes. If there are fewer than C complete initialization blocks, mark the configuration unavailable. No ground-truth segments enter initialization.

### Explicit objective and transport semantics

For recording r with N_r real frames, define p_t=n_t/N_r. The coupling T has one row per window and C columns; T is nonnegative and each row sums to p_t. Conditional assignments are R[t,a]=T[t,a]/p_t. The action marginal m_a=sum_t T[t,a] is relaxed toward q_a=1/C through a KL penalty. No hard constraint requires every action to appear.

Minimize the following per-recording objective with theta fixed:

F_r = a * <D,T> + (beta/2) * sum[t,s,a,b] V[t,s] T[t,a] T[s,b] 1[a != b] + lambda * KL(m || q) - epsilon * H(T).

Use a=0.7, beta=0.3, lambda=0.05, epsilon=0.07. H(T)=-sum T(log T-1); use generalized KL, sum(m log(m/q)-m+q). These are fixed design defaults in the normalized-cost system, not a claim that ASOT's original tuning transfers to this task.

Use a one-second neighborhood half-width. For L windows, set b=min(L-1, max(1, floor(fps/W))) and V[t,s]=L/b for 0<|t-s|<=b, otherwise zero. When L=1, set V=0. This is the reference ASOT kernel with radius b/L, expressed in fixed approximate physical time. Save the achieved radius in seconds. Do not use the optional monotonic action-order prior. A neighborhood consistency penalty favors nearby agreement; it does not learn a directional action grammar.

The complete fitting objective is sum_r N_r F_r plus the smoothing penalty -a*eta/(500*log(500))*sum[a,u] log(theta[a,u]). This definition makes the frame-weighted emission update consistent with the declared cost normalization and smoothing strength:

theta[a,u] = (sum[r,t] n_t R[t,a] 1[u_t=u] + eta/500) / (sum[r,t] n_t R[t,a] + eta).

Alternate transport on every fitting recording with this pooled update. Retain the previous T as the next inner initialization. Start the first transport iteration from T=p*q. After the final theta update, run transport again before saving predictions. Never describe a capped iterate as a global optimum.

### Solver and reference implementation

Use the official ASOT repository as the numerical reference, pinned to an exact commit and with its license retained. Keep upstream files unmodified under third_party/action_seg_ot and expose a thin adapter in the experiment package. Its full video training framework and datasets are unnecessary. ASOT supplies temporal assignment; the categorical emissions and alternating fitting procedure here are project adaptations. [1, 2]

Maintain a small equal-mass parity mode that matches the pinned reference's kernel, initialization, step sizes, and iteration count. Production mode must support real frame masses and independently checked objective/gradient consistency. Inspect the pinned objective helper: the current public implementation applies a factor of one half to a quantity containing both linear and quadratic terms. Verify this against its gradient rather than trusting the logged scalar; record any corrected objective implementation in the adapter, preserving upstream code.

Use float64 validation and stable log-domain row projections. For production mirror descent, start step size at 1, halve on objective increase, and allow at most 30 backtracks per step. Cap inner solves at 500 iterations. Require both row-centered gradient max norm <=1e-5 and relative objective change <=1e-7 for five accepted steps; cap outer iterations at 50 and require relative complete-objective change <=1e-6 for three iterations. These are numerical stopping rules, not action-score selection. Log residuals and capped/nonconverged results explicitly. A backtracking failure or nonfinite value invalidates the cell; do not substitute another model. Use convolution or prefix sums for the temporal term, never a dense L-by-L matrix.

### Matched conditions and execution matrix

| Condition | Unary, marginal, and entropy terms | Temporal term | Input order |
| --- | --- | --- | --- |
| A0 — No temporal term | Identical settings and initial theta to A1 | beta=0; keep a=0.7 | Observed |
| A1 — Categorical ASOT | Fixed settings above | beta=0.3 | Observed |
| A2 — Permuted control | Same initialization procedure and settings; refit emissions | beta=0.3 | Within-recording permutation |

For A2, permute only complete equal-length windows, leaving the partial terminal window fixed. This preserves exact code counts and frame-weighted counts within each recording. Use default_rng(seed+10000000) over sorted recording names and save the permutations. Keep labels in their original positions for evaluation. This null changes local code–label alignment as well as temporal structure; it is not an isolated test of a learned grammar. In subject-disjoint runs, shuffle fitting and held-out code sequences, with no transfer of windows across recordings or folds.

The primary matrix has 2 datasets × 2 normalization conditions × 3 seeds × 3 methods = 36 pooled fits. The four-fold secondary matrix adds 144 fits, for 180 planned method fits. Cache vocabulary and initialization once per matching fitting population; fit each method separately. SMQ reference scoring is separate. Include HistVQ only when compatible predictions and their provenance are available; otherwise mark the comparison unavailable.

Complete pooled comparisons first, then subject-disjoint comparisons, alternating datasets and completing three paired seeds per configuration. Use CPU and at most eight numerical threads. The execution launcher receives the runtime budget explicitly; reserve 30 minutes for reporting and stop before projected work crosses the compute deadline. Do not inherit the old experiment's elapsed-time clock or silently drop a normalization condition or seed. Record every unstarted or interrupted cell.

### Acceptance and interpretation

Before real fits, pass reference parity, objective finite-difference, frame-mass, terminal-window, permutation, initialization, leakage, and evaluator round-trip tests. Include planted sequences with missing actions, repeated actions, unequal durations, and a single window. Tests should verify numerical contracts and expose behavior; they must not assert that unsupervised optimization always recovers ground truth.

Engineering completion requires reproducible artifacts and an honest status for every planned cell. Scientific assessment compares A1 with A0, A2, and the same-checkpoint SMQ baseline across seeds and datasets. Temporal gains accompanied by worse segment metrics, near-single-state solutions, or poor transfer must remain visible. There is no newly invented numerical publication gate. A negative result can still justify completing the already planned context comparison; an invalid solver must be repaired before proceeding.

## 4. Stage B — Masked-code contextual representation

### Architecture and objective

Freeze each Stage A tokenizer. Feed only code IDs, position information, and a learned mask token to a small bidirectional Transformer. Initial architecture: embedding width 128, four encoder layers, four heads, feed-forward width 512, dropout 0.1, maximum context 128 windows, and a linear 500-class prediction head. Use sinusoidal positions. The representation is the final hidden state before the prediction head.

Train with cross-entropy only at masked positions, weighting each position by its real frame count. Mask 30% of valid positions using contiguous spans of eight tokens; merge overlaps and trim deterministically to the requested count. Replace every selected input token with MASK. Padding is neither a target nor an attention key. Never cross recording boundaries. Freeze code assignments throughout; do not recluster targets or train the original encoder.

This borrows HuBERT's masked prediction of discrete targets, but it is not a faithful HuBERT reproduction: the proposed input is already discrete, whereas HuBERT uses speech inputs with hidden-unit targets. The adaptation must be described explicitly. Low masked-code loss alone does not establish action semantics or subject invariance. [3]

### Training, validation, and inference

Within each fitting population, reserve approximately 10% of frames as validation by selecting whole recordings using a seed-111 shuffled recording list until reaching the target; retain at least one fitting and one validation recording. Save this split and reuse it across methods. Validation and fitting records must never overlap; outer held-out subjects must never enter this split. If insufficient recordings exist, report the configuration unavailable.

Use AdamW, learning rate 3e-4, weight decay 0.01, batch size 64, gradient norm clipping at 1, maximum 50 epochs, and patience 8 on fixed-mask validation cross-entropy. Save the lowest-validation-loss checkpoint, breaking ties toward the earlier epoch. No action labels choose the checkpoint. Log code-frequency and neighboring-code prediction baselines to detect an easy repetition task. Hardware and the runtime budget are declared before training; do not silently alter batch size or architecture after a failure.

At inference, disable dropout and use unmasked inputs. Cover each recording with length-128 chunks at stride 64, adding a final anchored chunk if needed. Average hidden states for tokens seen in multiple chunks, then L2-normalize each vector, retaining zero-vector flags. Keep the last partial token and exact frame alignment. The model uses future context and is offline.

Train the same three paired seeds for both normalization conditions and datasets; repeat within each downstream subject-disjoint fitting fold. Reuse one learned contextualizer for all Stage C arms requiring that representation. Report training curves, held-recording prediction performance, embedding variance/effective rank, runtime, and coverage. Export a strict code-sequence-to-embedding interface plus checkpoint and tokenizer hashes.

### Stage B checkpoint

Accept implementation only after tests verify masking, padding, chunk stitching, absence of label access, fold isolation, and deterministic reload inference. Inspect collapsed or trivial prediction behavior and report it; do not choose architecture settings from action scores. Stage C is the planned downstream test of usefulness, so masked-code accuracy is not a standalone acceptance claim for segmentation.

## 5. Stage C — Contextual temporal transport

### Frozen-context model

Freeze the selected Stage B checkpoint. Replace categorical emission costs with D[t,a]=(1-z_t dot mu_a)/2, where z_t and mu_a are unit vectors. Retain Stage A's frame masses, local temporal kernel, marginal penalty, entropy term, solver, state count, frame decoding, and evaluator. The bounded cosine cost is a declared new cost geometry; equal regularization coefficients do not make the two representations identically calibrated.

Initialize C prototypes by KMeans on at most 10,000 complete one-second frame-weighted means of normalized embeddings, using the shared sampling and KMeans settings; normalize centers. Alternate transport with mu_a=normalize(sum[r,t] n_t R[t,a] z_t). A zero update retains the previous prototype and is flagged. Recompute final transport after the final update. The categorical pseudocount prior does not carry over to continuous prototypes. Do not train the contextualizer on ASOT pseudo-labels in this stage.

### Comparisons that determine the result

| Representation | No temporal term | Temporal transport |
| --- | --- | --- |
| Hard code identity | Stage A0 | Stage A1 |
| Frozen masked-code context | Context-only assignment, beta=0 | Contextual ASOT, beta=0.3 |

Reuse the completed Stage A predictions. Fit both contextual arms with paired initialization and the same numerical budgets. Compare the contextual temporal arm with each single-component arm, preserving both normalization conditions and all seeds. This separates measured incremental effects; it does not prove a mechanism from one aggregate score.

Add a declared order-destruction diagnostic by permuting complete code windows within each recording before contextual inference, retaining the terminal token. Keep the trained contextualizer fixed, refit Stage C prototypes on the permuted fitting inputs, and evaluate permuted held-out inputs. Report it separately from Stage A's refitted categorical null. This is an out-of-distribution diagnostic for the contextualizer, not proof that it learned a semantic action grammar.

Report the complete five-metric matrix, matched seed differences, downstream subject-disjoint scores, occupancy, run durations, solver residuals, and runtime. Do not choose the temporal radius or context checkpoint after inspecting action labels. Any later sensitivity study must have a new protocol/version and retain the primary results.

### Research decision after Stage C

If contextual transport improves both temporal-only and context-only comparisons across the declared protocols, the supported finding concerns the combination of code context and temporal assignment in this frozen-tokenizer pipeline. Establishing a publishable contribution still requires positioning against closely related methods, compatible SMQ/HistVQ comparisons, and evidence beyond a favorable dataset or metric.

If gains are limited to pooled fitting or one metric, report that limitation before expanding the architecture. If the components do not improve segmentation, keep the negative evidence: no stage guarantees a top-conference contribution. A later matched tokenizer study can compare K-means, native learned VQ, and a genuinely trained FSQ bottleneck without changing the decoder. Pose normalization or encoder-loss changes would be another controlled axis, not evidence supplied by this plan.

## 6. Implementation, artifacts, and handoff

Use script/action_transport/ for new experiment code and results/action_transport/<run_id>/ for all outputs. Keep the code small: explicit configuration, token/cache loading, a transport adapter, categorical/context models, and a runner/evaluator boundary. Prefer short typed functions and small data records to plugin registries, deep inheritance, or copied training frameworks. Reuse verified scoring and windowing functions; do not import old runners or summarizers for their side effects.

Every run must save a configuration hash, source revision, upstream commit/license, full checkpoint hashes, package versions, recording/fold manifests, cache dependency hashes, initialization, prototypes/emissions, per-recording assignments and frame predictions, objective/residual traces, runtime logs, seed-level results, and a planned-cell status table. Use atomic writes and dependency-checked resume; never silently reuse an artifact under different settings.

Deliver one PowerShell execution script with explicit stage, protocol, hours, and resume options. A dry run prints the full grid, verifies caches, and estimates cost without fitting. A validation-only mode runs synthetic tests and tiny smoke checks. Full runs are launched by the user. Keep Stage A CPU-only; request no new hardware as part of implementing it. Checkpoint at safe boundaries and preserve interrupted partial outputs with an incomplete status.

The report must distinguish pooled and downstream subject-disjoint results; include a short factual summary, all comparison tables, convergence and coverage, limitations, open questions, and not-run cells. Generate action-timeline examples from a fixed seed-111 recording selection, not examples selected for favorable scores. Include an architecture figure, a five-metric comparison figure, and occupancy/duration diagnostics.

Stage A's detailed agent instructions are supplied in STAGE_A_AGENT_INSTRUCTIONS.md. Stage B and C numerical defaults should be reviewed for feasibility at their implementation handoff using runtime and label-free diagnostics; any change becomes a new frozen configuration before action evaluation, not silent tuning of this document.

## 7. Sources and scope of borrowing

[1] Xu and Gould (2024), Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation. Source of the temporal transport formulation and relaxed action-marginal approach. https://arxiv.org/abs/2404.01518

[2] Official ASOT implementation, mingu6/action_seg_ot. Numerical reference and license source; pin the actual commit at implementation. https://github.com/mingu6/action_seg_ot — solver: https://github.com/mingu6/action_seg_ot/blob/main/src/asot.py

[3] Hsu et al. (2021), HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. Source of masked hidden-unit prediction, not a claim that discrete-input Stage B reproduces the speech architecture. https://arxiv.org/abs/2106.07447

[4] Local project evidence: Current_State_Discrete_Motion_Project.docx; results/exp2round/q1q2/REPORT.md and its detailed tables; results/segmodel/ANALYSIS.md; and results/segmodel/m1r2/ artifacts. Read each result with its original protocol and validity status. No new result is asserted by this planning document.

[5] Kinoshita et al. (2026), Action Motifs, supplied in related work/. Its pose/shape-normalization and hierarchical representation direction remains a future input/encoder axis. None of its architectural components is silently included in Stages A–C.
