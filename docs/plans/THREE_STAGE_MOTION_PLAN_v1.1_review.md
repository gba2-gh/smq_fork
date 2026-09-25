# Fine-to-coarse motion segmentation
## Three-stage research and implementation plan

25 September 2026 | Version 1.1 | Planning document; no experiment launched

### Purpose

Test whether a fine, class-count-agnostic vocabulary of motion codes, decoded with temporal assignment, segments actions better than class-matched quantization, without equating individual codes with actions. Keep tokenization fixed while introducing temporal assignment, contextual representation learning, and then their combination. The immediate deliverable is a reliable categorical optimal-transport baseline with discriminating controls; the eventual research claim depends on the controlled comparisons, generalization, and later tokenizer evidence.

This plan fixes the three-stage order discussed with the project owner. Numerical defaults introduced here are prospective design choices, not previously validated optima or settings taken unchanged from a paper. Freeze them in a versioned configuration before accessing new action scores. Implementation agents should deliver readable scripts and one execution entry point; the user launches the full experiments.

### Changes from v1.0

Version 1.0 is preserved as THREE_STAGE_MOTION_PLAN_v1.0.md (and the original .docx). Version 1.1 was revised after a review of the project evidence and related work, before any Stage A implementation or action score existed. No change below was chosen from action scores.

1. Declared research claim: fine vocabulary plus temporal decoding versus class-matched quantization (§1).
2. Stage A controls: the within-recording permutation (A2) moves from the primary matrix to a one-seed sanity check. Added A0+filter (fixed mode filter), a sticky categorical HMM, A1-KC (the same decoder on a K=C vocabulary), and A-cont (the same decoder on the continuous PCA windows the tokenizer consumes) (§3).
3. Initialization: vocabulary-level code grouping replaces K-means on one-second code histograms. A one-second block holds about four codes, so its 500-bin histogram is a near-one-hot vector (§3).
4. Two pre-registered coefficient settings: ASOT's training settings (primary) and its evaluation settings (secondary). Both are applied at the final re-solve and both are reported (§3).
5. Solver: torch float64, on CUDA when available. Generalized-EM inner schedule during fitting, with a tight solve only for final inference. Capped cells stay in the primary aggregate, with a flag (§3).
6. An explicit user decision point after pooled Stage A, before any subject-disjoint fit (§3, §6).
7. Stage B: a second contextualizer (B-hubert: continuous input, code targets) as a competing arm, plus label-free subject-shortcut diagnostics. Stage C adds filter baselines (§4, §5).
8. Unmatched published references (SMQ, HiST-VQ, MASQ) are recorded, with BABEL named as a requirement before any publication claim (§5, §7).

## 1. Architecture, claim, and evidence boundary

The shared input pipeline is: existing sensor or motion-capture channels → frozen released SMQ encoder → standardized/PCA-transformed windows of prequantization latents → frozen K-means vocabulary → hard code sequence. Stages A–C consume that code sequence. The only exceptions are the declared competing arms A-cont and B-hubert. They consume the same standardized PCA windows the tokenizer consumes, and are never combined with the code input.

| Stage | Added component | Output and question |
| --- | --- | --- |
| A — Categorical transport | Action-state distributions over codes; alternating temporal transport and emission updates; matched controls | Can temporal assignment organize fixed fine codes into action segments, beyond trivial smoothing, an HMM, a class-matched vocabulary, and the undiscretized features? |
| B — Masked-code context | Small bidirectional Transformer trained to predict masked code spans (code input, and a continuous-input variant) | Does ordered context produce useful representations beyond individual code identity? |
| C — Contextual transport | Stage A transport machinery with costs from frozen Stage B embeddings | Does context add value beyond temporal regularization, and does transport add value beyond context? |

**Declared claim.** The research claim under test is that a class-count-agnostic fine vocabulary (K=500), decoded by temporal assignment, segments better than class-matched quantization. There are two matched comparisons for this claim:
1. The same-checkpoint SMQ baseline (a learned VQ with K equal to the number of actions).
2. A1-KC: the identical decoder on a K=C K-means vocabulary.

A-cont and B-hubert are competitors, not merely references. If they match or beat the code-based arms, the discretization is not what helps, and the report must say so. A second argument for fine vocabularies is that one tokenizer can serve several action granularities (different C) without retraining. This is a future study, not part of Stages A–C.

**Prior evidence that this plan must answer.** At Ku=500, pooled clustering of hard histograms at oracle boundaries was weak (HuGaDB F1@50 35.6 ± 6.9; LARa 6.2). Hard histograms at Ku=10–20 and continuous PCA features did better. At Ku=500, codes carry more subject than action information (I(U;S) > I(U;A) on both datasets). The strength of K=500 appears only under supervised readouts. Stage A is therefore expected to be at risk. Its controls are chosen so that a negative result remains interpretable.

The exp2round and Q1/Q2 vocabulary studies concern posthoc K-means units. They do not establish the quality of a 500-state FSQ bottleneck. Use K=500 as a fixed starting vocabulary, not a discovered optimum. Oracle-boundary diagnostics establish recoverable associations under their stated protocols; they do not establish unsupervised boundary recovery or subject-invariant primitives.

The released checkpoints used here have learned VQ quantization. The proposed K-means tokenizer uses their prequantization features and is a different pipeline from native SMQ, HiST-VQ, or a trained FSQ system. A future matched FSQ replacement is a separate experiment after these stages, with the same downstream decoder and evaluation. Do not label the current pipeline FSQ.

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

Use hard assignments and seeds 111, 222, and 1538574472. The primary vocabulary is K=500. The only other vocabulary is K=C (10 for HuGaDB, 8 for LARa), used solely by A1-KC. This is a matched control for the declared claim, not a vocabulary sweep. Fix sampling, PCA, and fold randomness at 111. Per fitting population, uniformly select at most 10,000 complete, non-overlapping flattened windows. Fit per-coordinate scaling and randomized PCA with 64 components or the maximum feasible rank. Fit full Lloyd KMeans with k-means++, n_init=5, max_iter=300, tol=1e-4. Assign all windows in bounded batches. Exclude terminal partial windows from fitting; zero-pad for assignment, and retain their true frame counts for every later operation. Both vocabularies of one fitting population share the scaler, PCA, and sample.

Save the sample identifiers, scaler, PCA, prototypes, hard assignments, PCA-space window vectors needed by A-cont (or a verified way to regenerate them), and real frame spans. Reuse equivalent verified artifacts. A seed denotes a paired pipeline replicate, including vocabulary and decoder initialization; its variation is not isolated decoder uncertainty. Do not introduce any further vocabulary sizes or a soft-assignment grid in these stages.

### Two distinct evaluation settings

The primary pooled setting fits the vocabulary and stage-specific models on the full unlabeled collection. It is transductive. All recordings are segmented without ground-truth boundaries.

The secondary subject-disjoint setting uses the saved four subject-grouped folds if their inventory and construction are verified. Otherwise rebuild label-free folds by decreasing group frame count, seed-111 tie breaking, and greedy allocation to the least-loaded fold. Fit vocabulary, Stage A emissions/prototypes, Stage B training/validation, and Stage C prototypes exclusively inside the fitting partition. Held-out recordings contribute only to their own transport inference and optional per-recording standardization. The frozen pretrained encoder may already have seen these subjects: call this downstream subject-disjoint evaluation, not an inductive encoder evaluation. Subject-disjoint fits start only after the user's decision point (§3).

Never use labels or ground-truth segments to fit, initialize, select hyperparameters, choose checkpoints, set run counts, or decide which results to omit. Keep label arrays outside fitting APIs. Use labels only in the evaluator and clearly disclose Hungarian mapping, metrics, and any diagnostic boundary comparisons. Subject IDs are recording metadata. They may define folds and label-free diagnostics, but never select a model, checkpoint, or setting.

### Evaluation and reporting

Broadcast each window's argmax state to its actual frames, then merge contiguous identical states. Score MoF, Edit, and F1@10/25/50 using the existing scorer. Keep None/none and the existing background handling explicit. Save raw states, mapped predictions, mappings, and the scorer identity. Do not concatenate recordings before segment-level scoring.

Use one dataset-level Hungarian mapping in the pooled setting. In subject-disjoint evaluation, state identities differ across fitted models: map each held-out fold separately, then aggregate matched recording predictions using the scorer's original aggregation. Disclose the extra oracle flexibility of fold-specific mappings. Never align fold states by labels before model fitting. Report fold scores and pooled out-of-fold results separately.

Report all seeds, mean and population SD, paired differences, action-run counts, duration distributions, state occupancy, occupancy entropy, objective traces, convergence status, runtime, device, and coverage. A partial seed group is incomplete. Seed SD describes initialization sensitivity; fold variation is neither seed SD nor an independent-sample confidence interval. Predeclare F1@50 as the focal segmentation metric while displaying all five metrics; MoF gains alone do not demonstrate better segmentation.

## 3. Stage A — Categorical temporal transport

### Model

Let u_t be a hard code, n_t its real frame count, and theta[a,u] the code distribution for action state a. Theta is shared across fitting recordings of one dataset, normalization condition, vocabulary, seed, and fold. The emission cost is D[t,a] = -log(theta[a,u_t]) / log(K). The fixed log(K) normalization avoids fitting a changing cost scale to each iteration or recording. With K=500 it gives a cost range of roughly [0, 2], comparable to ASOT's cosine cost.

### Vocabulary-level initialization

Initialize from the vocabulary, not from time blocks. Let c_u be the frame-weighted count of code u over the fitting population. Cluster the K prototypes (in the tokenizer's PCA space) into C groups g(u), using weighted KMeans with sample weights c_u, the vocabulary's stated KMeans settings, and the pipeline seed. Codes with c_u=0 take the group of their nearest center. Initial theta[a,u] = (c_u 1[g(u)=a] + eta/K) / (sum_u c_u 1[g(u)=a] + eta), with total pseudocount eta=fps per state distributed uniformly over the K codes. If fewer than C codes have c_u>0, mark the configuration unavailable. No ground-truth segment or label enters initialization.

Rationale, declared before scoring: a one-second block contains about four windows, so its K=500 histogram is a near-one-hot vector. K-means on such vectors groups blocks by exactly shared codes, and at K=500 shared codes mostly indicate the same subject. Grouping prototypes instead uses the geometry of the vocabulary. The grouping g is invariant to within-recording permutation and is shared by every arm of a configuration. That gives paired initialization across A0, HMM, A1, and A-cont.

### Explicit objective and transport semantics

For recording r with N_r real frames, define p_t=n_t/N_r. The coupling T has one row per window and C columns; T is nonnegative and each row sums to p_t. Conditional assignments are R[t,a]=T[t,a]/p_t. The action marginal m_a=sum_t T[t,a] is relaxed toward q_a=1/C through a KL penalty. No hard constraint requires every action to appear.

Minimize the following per-recording objective with theta fixed:

F_r = a * <D,T> + (beta/2) * sum[t,s,a,b] V[t,s] T[t,a] T[s,b] 1[a != b] + lambda * KL(m || q) - epsilon * H(T).

H(T)=-sum T(log T-1); use generalized KL, sum(m log(m/q)-m+q). In ASOT's parameterization, a=1-alpha and beta=alpha. The upstream scalar objective multiplies both linear and quadratic terms by one half, while its gradient does not; the objective above matches the gradient.

**Two pre-registered coefficient settings.** Neither is chosen from action scores:

| Setting | Lineage | a | beta | lambda | epsilon | Role |
| --- | --- | --- | --- | --- | --- | --- |
| T | ASOT training defaults (alpha=0.3) | 0.7 | 0.3 | 0.05 | 0.07 | Fitting, and primary final decoding |
| E | ASOT evaluation defaults (alpha=0.6) | 0.4 | 0.6 | 0.01 | 0.04 | Secondary final decoding |

Every arm is fitted under setting T. After the last update of theta (or of the prototypes), run the final re-solve once under T and once under E, from the same fitted parameters. For arms with no temporal term, beta stays 0 under both settings, and the other coefficients follow the setting. Both prediction sets are scored and reported in full. Primary comparisons use T. A supported claim must hold under T, and its direction must be reported under E. The weaker E marginal (lambda=0.01) is more tolerant of recordings that contain only a few of the C actions. This is a declared reason for including E, not a prediction of its score.

Use a one-second neighborhood half-width. For L windows, set b=min(L-1, max(1, floor(fps/W))) and V[t,s]=L/b for 0<|t-s|<=b, otherwise zero. When L=1, set V=0. This is the reference ASOT kernel with radius b/L, expressed in fixed approximate physical time. Save the achieved radius in seconds. Do not use the optional monotonic action-order prior. A neighborhood consistency penalty favors nearby agreement; it does not learn a directional action grammar.

The complete fitting objective is sum_r N_r F_r plus the smoothing penalty -a*eta/(K*log(K))*sum[a,u] log(theta[a,u]). This definition makes the frame-weighted emission update consistent with the declared cost normalization and smoothing strength:

theta[a,u] = (sum[r,t] n_t R[t,a] 1[u_t=u] + eta/K) / (sum[r,t] n_t R[t,a] + eta).

Alternate transport on every fitting recording with this pooled update. Retain the previous T as the next inner initialization. Start the first transport iteration from T=p*q. Never describe a capped iterate as a global optimum.

### Solver and reference implementation

Use the official ASOT repository as the numerical reference, pinned to commit 0c4c86b1037eb4c3e262197aeee115b31e220c9d (a local clone exists at Behaviour/action_seg_ot) and with its license retained. Keep upstream files unmodified under third_party/action_seg_ot and expose a thin adapter in the experiment package. Its full video training framework and datasets are unnecessary. ASOT supplies temporal assignment; the categorical emissions and alternating fitting procedure here are project adaptations. [1, 2]

Maintain a small equal-mass parity mode that matches the pinned reference's kernel, initialization, step sizes, and iteration count. Production mode must support real frame masses and independently checked objective/gradient consistency. Record the corrected objective implementation in the adapter, preserving upstream code.

Implement the solver in torch float64. Run production solves on CUDA when available (the project workstation has an RTX 4090), otherwise on CPU. Record the device per cell. Batch recordings with padding and masks. Keep a per-recording step size and per-recording stopping state, and implement the temporal term with 1-D convolution, never a dense L-by-L matrix. Parity and finite-difference tests run in float64 on CPU. Cap CPU numerical threads at eight.

Production mirror descent starts each step at step size 1, halves on an increase in that recording's objective, and allows at most 30 backtracks.
- **During fitting (generalized EM):** at most 25 accepted steps per recording per outer iteration, warm-started. Each accepted step decreases F_r, so every outer iteration decreases the complete objective. Cap outer iterations at 50. Stop after three consecutive outer iterations with relative complete-objective change <=1e-6 (denominator max(1,|previous|)).
- **Final inference (both settings):** at most 500 accepted steps. Stop when the row-centered gradient max norm is <=1e-5 and relative objective change is <=1e-7 for five accepted steps.

**Cell status:**
- *converged*: all stopping rules met.
- *capped*: an iteration cap was reached, with a finite, nonincreasing accepted-objective trace. Capped cells stay in the primary aggregate, flagged and counted.
- *invalid*: a nonfinite value, a backtracking failure, or an objective increase. Invalid cells are excluded and listed.

These are numerical rules, not action-score selection. Log residuals. Do not substitute another model after a failure.

### Arms and execution matrix

| Arm | Representation and cost | Temporal structure | Fitting | Question |
| --- | --- | --- | --- | --- |
| SMQ | Same-checkpoint learned VQ, K=C | none | released | Class-matched external baseline |
| A0 | K=500 codes; categorical | none (beta=0) | alternating, paired init | Emission-only reference |
| A0+filter | A0's raw states | fixed centred mode filter, half-width b | none beyond A0 | Does transport beat trivial smoothing? |
| HMM | K=500 codes; categorical | sticky fixed transitions | Baum–Welch on theta | Does transport beat the canonical discrete-sequence model? |
| A1 | K=500 codes; categorical | ASOT | alternating, paired init | Main arm |
| A1-KC | K=C codes; categorical | ASOT | alternating | Fine versus class-matched vocabulary, same decoder |
| A-cont | normalized PCA-64 windows; cosine | ASOT | alternating prototypes, paired init | Does discretization help or hurt? |

**A0+filter.** Take A0's window-level raw states after its final re-solve (under T and under E separately). For each window t, count frame-weighted votes (weight n_s) among windows s of the same recording with |t-s|<=b. Choose the state with the most votes. On a tie, keep A0's state at t if it is among the tied states, otherwise take the smallest state ID. Apply one pass and do not iterate. This arm involves no fitting.

**HMM.** Use C states, the identical initial theta as A0/A1, a fixed uniform initial-state distribution, and a fixed transition matrix with A[a,a]=s and A[a,b]=(1-s)/(C-1), where s=1-1/(2b+1). The expected dwell is therefore one kernel diameter, (2b+1) windows. The transitions are not learned. Scale each window's emission log-likelihood by n_t/W, so a terminal partial window counts in proportion to its real frames. Update theta by Baum–Welch with frame-weighted expected counts n_t*gamma[t,a] and the same eta/K pseudocount (MAP). Run forward–backward in the log domain in float64. Cap outer iterations at 50, stopping after three consecutive iterations with relative change of the complete log-posterior <=1e-6. Decode by per-window argmax of the posterior marginal gamma, matching A1's argmax R. This HMM is a comparison arm, not a duration model: there is no HSMM, no learned transition matrix, and no segment DP.

**A1-KC.** Identical to A1, but with the K=C vocabulary of the same fitting population and seed, the cost normalization log(C), and eta distributed over C codes. Its initialization applies the same vocabulary-level rule. With K=C, each prototype forms its own group.

**A-cont.** Let z_t be the L2-normalized standardized PCA-64 window vector produced by the tokenizer transform for the same normalization condition and fitting population. Zero vectors are flagged and keep cost 1/2 for every state. The cost is D[t,a]=(1-z_t dot mu_a)/2. Initialize mu_a=normalize(sum over fitting windows with g(u_t)=a of n_t z_t), using the same K=500 grouping g as A1 for paired initialization. Update mu_a=normalize(sum[r,t] n_t R[t,a] z_t). A zero update retains the previous prototype and is flagged. Transport settings, solver, and budgets are identical to A1. Equal coefficients do not make the cosine and categorical costs identically calibrated; disclose this.

**Permutation sanity check (not a primary comparison).** Fit A1 on within-recording permuted inputs, pooled, with seed 111 only, for both datasets and both normalizations. Permute only complete equal-length windows, leaving the partial terminal window fixed. Use default_rng(seed+10000000) over sorted recording names and save the permutations. Keep labels in their original positions. This destroys the local code–label alignment as well as the temporal structure, so the check is expected to fall near chance. Report it in a separate sanity table.

**Matrix.**
- **Pooled:** 2 datasets × 2 normalization conditions × 3 seeds × 5 fitted arms (A0, HMM, A1, A1-KC, A-cont) = 60 fits, plus 4 permutation sanity fits = 64 fits. Each pooled configuration yields 11 scored prediction sets: A0, A0+filter, A1, A1-KC and A-cont under both T and E, plus the HMM.
- **Subject-disjoint:** 4 folds × 60 = 240 fits.
- **Caches:** build vocabularies, PCA windows, and the grouping g once per matching fitting population, and fit each arm separately. SMQ reference scoring is separate.
- **External references:** a matched HiST-VQ or MASQ comparison needs compatible predictions and their provenance; otherwise mark it unavailable. Published paper-table numbers may appear only in a clearly labelled unmatched-reference table.

**Order and decision point.** Run the pooled matrix first, alternating datasets and completing three paired seeds per configuration. The runner then stops. The user reads the pooled report and decides whether to launch the subject-disjoint matrix. The launch must name the completed pooled run it follows, which makes the decision explicit and traceable. There is no automatic numerical gate. The execution launcher receives the runtime budget explicitly; reserve 30 minutes for reporting and stop before projected work crosses the compute deadline. The dry run must project cost from a timing probe before a full launch. Do not inherit the old experiment's elapsed-time clock or silently drop a normalization condition, seed, arm, or setting. Record every unstarted or interrupted cell.

### Acceptance and interpretation

Before real fits, pass these tests:
- reference parity, and objective finite differences
- frame-mass and terminal-window handling
- vocabulary-level initialization
- the HMM forward–backward, checked against brute-force enumeration on tiny sequences
- mode-filter tie rules
- A-cont cost and prototype updates
- the permutation check
- leakage, and an evaluator round trip

Include planted sequences with missing actions, repeated actions, unequal durations, and a single window. Tests verify numerical contracts and expose behavior; they must not assert that unsupervised optimization always recovers ground truth.

Engineering completion requires reproducible artifacts and an honest status for every planned cell. The scientific assessment has three parts:
1. **The declared claim:** A1 against SMQ and A1-KC.
2. **The temporal mechanism:** A1 against A0+filter and the HMM. A0 is reported, but because it has no temporal term, beating it alone is not evidence for transport.
3. **The value of discretization:** A1 against A-cont.

Temporal gains accompanied by worse segment metrics, near-single-state solutions, subject-aligned states, or poor transfer must remain visible. A negative result can still justify completing the planned context comparison; an invalid solver must be repaired before proceeding.

## 4. Stage B — Masked-code contextual representation

### Architecture and objective

Freeze each Stage A K=500 tokenizer. Train two contextualizers with the identical architecture and training recipe:
- **B-code:** input is code IDs only, with a learned mask token.
- **B-hubert:** input is the standardized PCA-64 window vectors, linearly projected to the model width, with masked positions replaced by a learned mask vector. This is the HuBERT recipe of continuous input and hidden-unit targets, and it is a competing arm under the declared claim.

Both use position information and predict the same K=500 code targets. Initial architecture: embedding width 128, four encoder layers, four heads, feed-forward width 512, dropout 0.1, maximum context 128 windows, and a linear 500-class prediction head. Use sinusoidal positions. The representation is the final hidden state before the prediction head.

Train with cross-entropy only at masked positions, weighting each position by its real frame count. Mask 30% of valid positions using contiguous spans of eight tokens; merge overlaps and trim deterministically to the requested count. Replace every selected input position with the mask token or vector. Padding is neither a target nor an attention key. Never cross recording boundaries. Freeze code assignments throughout; do not recluster targets or train the original encoder.

B-code borrows HuBERT's masked prediction of discrete targets, but it is not a faithful HuBERT reproduction, because its input is already discrete. B-hubert is closer, but it still uses frozen K-means targets from a frozen encoder, and there are no target-refinement iterations. Low masked-code loss alone does not establish action semantics or subject invariance. [3]

### Training, validation, and inference

Within each fitting population, reserve approximately 10% of frames as validation by selecting whole recordings using a seed-111 shuffled recording list until reaching the target; retain at least one fitting and one validation recording. Save this split and reuse it across methods. Validation and fitting records must never overlap; outer held-out subjects must never enter this split. If insufficient recordings exist, report the configuration unavailable.

Use AdamW, learning rate 3e-4, weight decay 0.01, batch size 64, gradient norm clipping at 1, maximum 50 epochs, and patience 8 on fixed-mask validation cross-entropy. Save the lowest-validation-loss checkpoint, breaking ties toward the earlier epoch. No action labels choose the checkpoint. Log code-frequency and neighboring-code prediction baselines to detect an easy repetition task. HuGaDB has roughly 76k tokens in total, so expect early stopping to act quickly; this is not a reason to change the architecture after seeing scores. Hardware and the runtime budget are declared before training; do not silently alter batch size or architecture after a failure.

At inference, disable dropout and use unmasked inputs. Cover each recording with length-128 chunks at stride 64, adding a final anchored chunk if needed. Average hidden states for tokens seen in multiple chunks, then L2-normalize each vector, retaining zero-vector flags. Keep the last partial token and exact frame alignment. The model uses future context and is offline.

Train the same three paired seeds for both normalization conditions and datasets; repeat within each downstream subject-disjoint fitting fold. Reuse one learned contextualizer per variant for all Stage C arms requiring that representation. Report training curves, held-recording prediction performance, embedding variance/effective rank, runtime, and coverage. Export a strict sequence-to-embedding interface plus checkpoint and tokenizer hashes.

**Subject-shortcut diagnostics.** These are label-free with respect to actions, and they are for reporting only. Masked prediction over subject-heavy codes can be solved partly by recognizing the subject.
- **Subject probe:** report subject-decoding accuracy from a fixed linear probe (the logistic-regression settings of the Q1/Q2 readouts, with a recording-grouped four-fold split and subjects appearing on both sides). Fit it on one-hot codes, on B-code embeddings, and on B-hubert embeddings.
- **Neighbour ratio:** report the fraction of each window's 10 nearest neighbours (cosine) that come from the same recording, and the fraction from the same subject.

These diagnostics never select checkpoints or settings.

### Stage B checkpoint

Accept implementation only after tests verify masking, padding, chunk stitching, absence of label access, fold isolation, deterministic reload inference, and B-hubert's input projection and masking. Inspect collapsed or trivial prediction behavior and report it; do not choose architecture settings from action scores. Stage C is the planned downstream test of usefulness, so masked-code accuracy is not a standalone acceptance claim for segmentation.

## 5. Stage C — Contextual temporal transport

### Frozen-context model

Freeze the Stage B checkpoints. Replace categorical emission costs with D[t,a]=(1-z_t dot mu_a)/2, where z_t and mu_a are unit vectors. Retain Stage A's frame masses, local temporal kernel, marginal penalty, entropy term, solver, generalized-EM schedule, both coefficient settings, state count, frame decoding, and evaluator. The bounded cosine cost is a declared new cost geometry; equal regularization coefficients do not make the two representations identically calibrated.

Initialize prototypes as mu_a=normalize(sum over fitting windows with g(u_t)=a of n_t z_t), using Stage A's vocabulary-level grouping g, so initialization is paired with A0/A1/A-cont. Alternate transport with mu_a=normalize(sum[r,t] n_t R[t,a] z_t). A zero update retains the previous prototype and is flagged. Recompute the final transport under settings T and E after the final update. The categorical pseudocount prior does not carry over to continuous prototypes. Do not train the contextualizer on ASOT pseudo-labels in this stage.

### Comparisons that determine the result

| Representation | No temporal term | Fixed mode filter | Temporal transport |
| --- | --- | --- | --- |
| Hard code identity, K=500 | A0 | A0+filter | A1 |
| Hard code identity, K=C | — | — | A1-KC |
| Continuous PCA windows | — | — | A-cont |
| B-code context | Context-only, beta=0 | Context-only + filter | Contextual ASOT |
| B-hubert context | Context-only, beta=0 | Context-only + filter | Contextual ASOT |

Reuse the completed Stage A predictions. Fit the contextual arms with paired initialization and the same numerical budgets. Compare each contextual temporal arm with its context-only and filter arms, with A1, and with A-cont, preserving both normalization conditions, both coefficient settings, and all seeds. This separates measured incremental effects; it does not prove a mechanism from one aggregate score.

Add a declared order-destruction diagnostic for each contextualizer by permuting complete code windows (and, for B-hubert, the matching PCA windows) within each recording before contextual inference, retaining the terminal token. Keep the trained contextualizer fixed, refit Stage C prototypes on the permuted fitting inputs, and evaluate permuted held-out inputs. Report it separately from Stage A's sanity check. This is an out-of-distribution diagnostic for the contextualizer, not proof that it learned a semantic action grammar.

Report the complete five-metric matrix, matched seed differences, downstream subject-disjoint scores, occupancy, run durations, solver residuals, and runtime. Do not choose the temporal radius or context checkpoint after inspecting action labels. Any later sensitivity study must have a new protocol/version and retain the primary results.

### Research decision after Stage C

**When the declared claim is supported.** Under setting T, a K=500 code-based arm (A1 or contextual ASOT on B-code) must improve F1@50 over both the same-checkpoint SMQ baseline and A1-KC:
- across seeds and on both datasets
- in the pooled setting and in the subject-disjoint setting, if run
- with its direction reported under E

If A-cont or B-hubert matches or exceeds the code-based arms, the supported finding concerns temporal decoding of these features, not discretization, and must be reported as such.

**Where the field stands.** The current unmatched published references are:

| Method | HuGaDB MoF / F1@50 | LARa MoF / F1@50 |
| --- | --- | --- |
| SMQ | 42.0 / 24.3 | 37.4 / 16.4 |
| HiST-VQ | 48.2 / 28.3 | 45.9 / 19.3 |
| MASQ | 52.3 / 39.5 | 47.2 / 22.2 |

HiST-VQ already learns a fine-to-coarse (sub-action to action) quantization end to end. A contribution here must therefore rest on post-hoc decoding over a frozen, class-count-agnostic vocabulary, and on evidence beyond one favorable dataset or metric. All three papers also evaluate BABEL, so BABEL is required before any publication claim.

**When results are limited or negative.** If gains are limited to pooled fitting, one setting, or one metric, report that limitation before expanding the architecture. If the components do not improve segmentation, keep the negative evidence: no stage guarantees a top-conference contribution. Two later studies are possible, each a separate controlled axis:
- A matched tokenizer study comparing K-means, native learned VQ, and a genuinely trained FSQ bottleneck, without changing the decoder.
- A multi-granularity study (several C over one frozen vocabulary).

Pose normalization or encoder-loss changes would be yet another controlled axis, not evidence supplied by this plan.

## 6. Implementation, artifacts, and handoff

Use script/action_transport/ for new experiment code and results/action_transport/<run_id>/ for all outputs. Keep the code small: explicit configuration, token/cache loading, a transport adapter, categorical/continuous/HMM models, and a runner/evaluator boundary. Prefer short typed functions and small data records to plugin registries, deep inheritance, or copied training frameworks. Reuse verified scoring and windowing functions; do not import old runners or summarizers for their side effects.

Every run must save a configuration hash, source revision, upstream commit/license, full checkpoint hashes, package versions, device, recording/fold manifests, cache dependency hashes, initialization and grouping, prototypes/emissions, per-recording assignments and frame predictions for each setting, objective/residual traces, runtime logs, seed-level results, and a planned-cell status table. Use atomic writes and dependency-checked resume; never silently reuse an artifact under different settings.

Deliver one PowerShell execution script with explicit stage, protocol, hours, and resume options. The protocols are pooled and subject_disjoint, and there is no combined option. A subject-disjoint launch must name the completed pooled run it follows. A dry run prints the full grid, verifies caches, runs a short timing probe, and estimates cost without fitting. A validation-only mode runs synthetic tests and tiny smoke checks. Full runs are launched by the user. Stage A may use the existing CUDA GPU through torch and falls back to CPU; request no new hardware. Checkpoint at safe boundaries and preserve interrupted partial outputs with an incomplete status.

The report must include:
- pooled and downstream subject-disjoint results, kept separate
- settings T and E, kept separate, with T primary
- a short factual summary
- all comparison tables, organized by the three questions in §3
- convergence (converged/capped/invalid) and coverage
- limitations, open questions, and not-run cells
- action-timeline examples from a fixed seed-111 recording selection, not examples selected for favorable scores
- an architecture figure, a five-metric comparison figure, and occupancy/duration diagnostics

Stage A's detailed agent instructions are supplied in STAGE_A_AGENT_INSTRUCTIONS.md (version 1.1). Stage B and C numerical defaults should be reviewed for feasibility at their implementation handoff using runtime and label-free diagnostics; any change becomes a new frozen configuration before action evaluation, not silent tuning of this document.

## 7. Sources and scope of borrowing

[1] Xu and Gould (2024), Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation. Source of the temporal transport formulation, relaxed action-marginal approach, and the training/evaluation coefficient lineages. https://arxiv.org/abs/2404.01518

[2] Official ASOT implementation, mingu6/action_seg_ot, commit 0c4c86b1037eb4c3e262197aeee115b31e220c9d. Numerical reference and license source. https://github.com/mingu6/action_seg_ot — solver: src/asot.py; coefficient defaults: src/train.py and run_bf.sh.

[3] Hsu et al. (2021), HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. Source of masked hidden-unit prediction; B-hubert adopts its continuous-input/discrete-target structure without target refinement. https://arxiv.org/abs/2106.07447

[4] Local project evidence: Current_State_Discrete_Motion_Project.docx; results/exp2round/q1q2/REPORT.md and its detailed tables; results/segmodel/ANALYSIS.md; and results/segmodel/m1r2/ artifacts. Read each result with its original protocol and validity status. No new result is asserted by this planning document.

[5] Kinoshita et al. (2026), Action Motifs, supplied in related work/. Its pose/shape-normalization and hierarchical representation direction remains a future input/encoder axis. None of its architectural components is silently included in Stages A–C.

[6] Gökay et al. (2025), Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation (SMQ). Source of the released checkpoints and the class-matched baseline.

[7] Ahmed et al. (2026), Unsupervised Skeleton-Based Action Segmentation via Hierarchical Spatiotemporal Vector Quantization (HiST-VQ). Closest prior fine-to-coarse design; unmatched published reference.

[8] Qin et al. (2026), MASQ: Mask-Aware Spatiotemporal Quantization for Unsupervised Skeleton Action Segmentation. Current published state of the art on HuGaDB and LARa; unmatched published reference.
