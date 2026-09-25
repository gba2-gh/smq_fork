# Fine-to-coarse motion segmentation
## Three-stage research and implementation plan

25 September 2026 | Version 1.2 | Planning document; no experiment launched

### Purpose

Test whether a fine, class-count-agnostic vocabulary of motion codes, decoded with temporal assignment, segments actions better than class-matched quantization, without equating individual codes with actions. Keep tokenization fixed while introducing temporal assignment, contextual representation learning, and then their combination. The immediate deliverable is a reliable categorical optimal-transport baseline with discriminating controls; the eventual research claim depends on the controlled comparisons, generalization, and later tokenizer evidence.

This plan fixes the three-stage order discussed with the project owner. Numerical defaults introduced here are prospective design choices, not previously validated optima or settings taken unchanged from a paper. Freeze them in a versioned configuration before accessing new action scores. Implementation agents should deliver readable scripts and one execution entry point; the user launches the full experiments.

### Review outcome and scope

Version 1.0 and the reviewed version 1.1 are preserved. Version 1.2 accepts the review's fixed-smoother, K=C, and continuous-feature controls; geometry-based initialization; efficient alternating optimization; and explicit numerical reporting. It defers the HMM, a second continuous-input contextualizer, subject classifiers, and BABEL execution. These additions would enlarge the architecture before the discrete pipeline is tested. REVIEW_RESPONSE_v1.2.md records each decision and the factual corrections.

The principal objective remains learning coarse action structure from fine discrete motion sequences. Stage A supplies local consistency, Stage B learns context from code sequences, and Stage C combines them. A1-KC is a targeted control, not a vocabulary sweep. A-cont is a separate reference, not a continuous-input replacement for the proposed architecture. K=500, both normalization conditions, and all three seeds remain fixed.

## 1. Architecture, claim, and evidence boundary

The shared input pipeline is: existing sensor or motion-capture channels → frozen released SMQ encoder → standardized/PCA-transformed windows of prequantization latents → frozen K-means vocabulary → hard code sequence. Stages A–C consume that code sequence. The sole continuous-window comparison is A-cont in Stage A. It is never combined with the code input. Vocabulary prototype geometry is also used once for initialization; categorical fitting and inference subsequently receive code IDs and frame masses only. This is an explicit geometric initialization prior, not a claim that code IDs alone determine the initial model.

| Stage | Added component | Output and question |
| --- | --- | --- |
| A — Categorical transport | Action-state distributions over codes; alternating temporal transport and emission updates; matched controls | Can temporal assignment organize fixed fine codes into action segments, beyond a fixed smoother, a class-matched vocabulary, and a continuous-feature reference? |
| B — Masked-code context | Small bidirectional Transformer trained to predict masked code spans (code input only) | Does ordered context produce useful representations beyond individual code identity? |
| C — Contextual transport | Stage A transport machinery with costs from frozen Stage B embeddings | Does context add value beyond temporal regularization, and does transport add value beyond context? |

**Hypothesis, not assumed result.** A fixed fine vocabulary may support action segmentation when several codes can express one action and contextual evidence can change the assignment of a code. A1 versus A1-KC tests fine versus class-count-sized vocabularies under the declared pipeline. The same-checkpoint SMQ baseline is a system comparison: its tokenizer, objective, and temporal resolution also differ. It cannot by itself isolate the effect of K. The encoder was trained inside SMQ; choosing K independently of C does not make the full representation independent of the original class-count choice.

A-cont tests a normalized continuous representation with cosine emissions. It is a useful competitor, but categorical negative log-likelihood and cosine costs have different geometries. Neither winning nor losing this comparison establishes a universal causal benefit of discretization. If the code arm does not win, report that its segmentation advantage over this continuous competitor was not demonstrated. Reusability across action granularities remains a future hypothesis.

**Prior evidence, qualified by protocol.** In Q1/Q2, latent K=500 hard oracle-segment histograms at the planned windows gave F1@50 35.57 ± 6.86 on HuGaDB (W=15) and 6.19 ± 0.20 on LARa (W=12). K=10 hard histograms gave 48.33 ± 0.43 and 23.26 ± 2.57. These are warning signs for hard histogram clustering, not results for the proposed transport decoder. K=500 soft histograms at base temperature gave 61.90 ± 4.54 and 20.15 ± 2.59, so useful K=500 results were not confined to supervised readouts. All values are pooled oracle-boundary diagnostics, mean ± population SD across three seeds, not matched free-boundary baselines.

At these windows, I(U;S) exceeded I(U;A): 1.680 versus 1.428 nats on HuGaDB, and 1.812 versus 0.647 on LARa. Subject association merits monitoring. The inequality does not establish that shared codes mostly identify subjects, that subject variation caused the clustering results, or that all useful action information is absent. The targets have different marginal entropies, and codes may encode both. Sources: Q1/Q2 clustering_summary.csv and information_summary.csv, representation=latent, windows=15/12.

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

Let u_t be a hard code, n_t its real frame count, and theta[a,u] the code distribution for action state a. Theta is shared across fitting recordings of one dataset, normalization condition, vocabulary, seed, and fold. The emission cost is D[t,a] = -log(theta[a,u_t]) / log(500), for both K=500 and K=C. A shared fixed denominator keeps the unary weight per nat constant across the vocabulary-size comparison; do not normalize the K=C arm by log(C). It is not a bounded cosine-like cost. With state count mass M_a and the stated smoothing, the maximum possible cost for an unobserved code is log(K*(M_a+eta)/eta)/log(500), which can exceed 2. Compare fitting cost distributions and objective components without tuning on action scores.

### Vocabulary-level initialization

Initialize from the vocabulary, not from time blocks. Let c_u be the frame-weighted count of code u over the fitting population. Cluster the occupied prototypes (in the tokenizer's PCA space) into C groups g(u), using weighted KMeans with sample weights c_u, the vocabulary's stated KMeans settings, and the pipeline seed. Codes with c_u=0 take the group of their nearest center. Initial theta[a,u] = (c_u 1[g(u)=a] + eta/K) / (sum_u c_u 1[g(u)=a] + eta), with total pseudocount eta=fps per state distributed uniformly over the K codes. If fewer than C distinct occupied prototypes exist, mark the configuration unavailable. At K=C with distinct occupied prototypes, set g(u)=u directly; this is the exact one-prototype-per-state grouping, without a redundant KMeans fit. No ground-truth segment or label enters initialization.

Rationale, declared before scoring: a one-second histogram has at most four contributing windows on HuGaDB and five on LARa, including boundary overlaps. It is sparse in 500 dimensions, but not necessarily near-one-hot. Prototype grouping uses the fitted vocabulary geometry, is invariant to permutation of recording order, and gives a shared initialization for A0/A1/A-cont. This is a defensible initialization choice, not a demonstrated repair of subject bias. Save the geometry dependency and grouping; do not add an initialization sweep.

### Explicit objective and transport semantics

For recording r with N_r real frames, define p_t=n_t/N_r. The coupling T has one row per window and C columns; T is nonnegative and each row sums to p_t. Conditional assignments are R[t,a]=T[t,a]/p_t. The action marginal m_a=sum_t T[t,a] is relaxed toward q_a=1/C through a KL penalty. No hard constraint requires every action to appear.

Minimize the following per-recording objective with theta fixed:

F_r = a * <D,T> + (beta/2) * sum[t,s,a,b] V[t,s] T[t,a] T[s,b] 1[a != b] + lambda * KL(m || q) - epsilon * H(T).

H(T)=-sum T(log T-1); use generalized KL, sum(m log(m/q)-m+q). In ASOT's parameterization, a=1-alpha and beta=alpha. The upstream scalar objective multiplies both linear and quadratic terms by one half, while its gradient does not; the objective above matches the gradient.

**Two pre-registered coefficient settings.** Neither is chosen from action scores:

| Setting | Lineage | a | beta | lambda | epsilon | Role |
| --- | --- | --- | --- | --- | --- | --- |
| T | Pinned CLI training defaults (alpha=0.3) | 0.7 | 0.3 | 0.05 | 0.07 | Fitting, and primary final decoding |
| E | Pinned CLI evaluation defaults (alpha=0.6) | 0.4 | 0.6 | 0.01 | 0.04 | Secondary final decoding |

Every arm is fitted under setting T. After the last update of theta (or of the prototypes), run the final re-solve once under T and once under E, from the same fitted parameters. For arms with no temporal term, beta stays 0 under both settings, and the other coefficients follow the setting. Both prediction sets are scored and reported in full. Primary comparisons use T. A supported claim must hold under T, and its direction must be reported under E. E is a bundled decoding sensitivity: it changes four coefficients at once and does not isolate the marginal penalty. It does not refit the emissions, so it is not evidence about fitting the model under E. These values are src/train.py command-line defaults at the pinned commit; constructor defaults and run_bf.sh overrides differ. They are not universally validated ASOT settings for these categorical costs. Compute primary T predictions before E. Always enumerate and report missing E results rather than selecting settings by score.

Use a one-second neighborhood half-width. For L windows, set b=min(L-1, max(1, floor(fps/W))) and V[t,s]=L/b for 0<|t-s|<=b, otherwise zero. When L=1, set V=0. This is the reference ASOT kernel with radius b/L, expressed in fixed approximate physical time. Save the achieved radius in seconds. Do not use the optional monotonic action-order prior. A neighborhood consistency penalty favors nearby agreement; it does not learn a directional action grammar.

The complete fitting objective is sum_r N_r F_r plus the smoothing penalty -a*eta/(K*log(500))*sum[a,u] log(theta[a,u]). This definition makes the frame-weighted emission update consistent with the declared cost normalization and smoothing strength:

theta[a,u] = (sum[r,t] n_t R[t,a] 1[u_t=u] + eta/K) / (sum[r,t] n_t R[t,a] + eta).

Alternate transport on every fitting recording with this pooled update. Retain the previous T as the next inner initialization. Start the first transport iteration from T=p*q. Never describe a capped iterate as a global optimum.

### Solver and reference implementation

Use the official ASOT repository as the numerical reference, pinned to commit 0c4c86b1037eb4c3e262197aeee115b31e220c9d (a local clone exists at Behaviour/action_seg_ot) and with its license retained. Keep upstream files unmodified under third_party/action_seg_ot and expose a thin adapter in the experiment package. Its full video training framework and datasets are unnecessary. ASOT supplies temporal assignment; the categorical emissions and alternating fitting procedure here are project adaptations. [1, 2]

Maintain a small equal-mass parity mode that matches the pinned reference's kernel, initialization, step sizes, and iteration count. Production mode must support real frame masses and independently checked objective/gradient consistency. Record the corrected objective implementation in the adapter, preserving upstream code.

Implement in torch float64 with CPU as the default. An explicit CUDA launch may use the existing GPU after numerical parity and a label-blind timing probe; do not assume float64 GPU speedup. Freeze the device for a run and never silently switch on failure. Implement correctness per recording first, then optional bounded batches with masks and per-recording steps/stopping. Kernel amplitude must use each recording's true L, not padded batch length. Implement the temporal term with convolution or prefix sums, never a dense L-by-L matrix. CPU parity/finite-difference tests precede acceleration. Cap CPU numerical threads at min(8, logical CPU count).

Production mirror descent starts each step at step size 1, halves on an increase in that recording's objective, and allows at most 30 backtracks.
- **During fitting (inexact alternating minimization):** at most 25 accepted steps per recording per outer iteration, warm-started. Each accepted step is nonincreasing within a documented floating-point tolerance; the exact emission update must also be nonincreasing. Verify the complete weighted objective, including its prior. This is inexact block-coordinate optimization, not ordinary likelihood EM. Cap outer iterations at 50. Stop after three consecutive outer iterations with relative complete-objective change <=1e-6 (denominator max(1,|previous|)).
- **Final inference (both settings):** at most 500 accepted steps. Stop when the row-centered gradient max norm is <=1e-5 and relative objective change is <=1e-7 for five accepted steps.

**Cell status:**
- *converged*: the declared outer objective-change and final stationarity criteria are met; this is operational convergence, not a global-optimum certificate.
- *capped*: the outer or final-inference cap was reached, with finite feasible iterates and a nonincreasing accepted-objective trace. Include these in a predeclared budgeted-algorithm table, with status counts and residuals visible. Also show convergence-qualified paired comparisons restricted to the same seeds; never present that selected subset as the full-grid result. The planned 25-step inner schedule is not itself a cap failure. An interrupted cell is not capped.
- *invalid*: nonfinite values, infeasibility, backtracking failure, or an objective increase exceeding numerical tolerance. Exclude the affected metric from valid aggregates but retain every planned cell and reason. Keep fitting and final-inference status separate for T and E; an E failure cannot invalidate a completed T result.

These are numerical rules, not action-score selection. Log residuals. Do not substitute another model after a failure.

### Arms and execution matrix

| Arm | Representation and cost | Temporal structure | Fitting | Question |
| --- | --- | --- | --- | --- |
| SMQ | Same-checkpoint learned VQ, K=C | none | released | Class-matched external baseline |
| A0 | K=500 codes; categorical | none (beta=0) | alternating, paired init | Emission-only reference |
| A0+filter | A0's raw states | fixed centred mode filter, half-width b | none beyond A0 | Does transport beat trivial smoothing? |
| A1 | K=500 codes; categorical | ASOT | alternating, paired init | Main arm |
| A1-KC | K=C codes; categorical | ASOT | alternating | Fine versus class-matched vocabulary, same decoder |
| A-cont | normalized PCA-64 windows; cosine | ASOT | alternating prototypes, paired init | Does discretization help or hurt? |

**A0+filter.** Take A0's window-level raw states after its final re-solve (under T and under E separately). For each window t, count frame-weighted votes (weight n_s) among windows s of the same recording with |t-s|<=b. Choose the state with the most votes. On a tie, keep A0's state at t if it is among the tied states, otherwise take the smallest state ID. Apply one pass and do not iterate. This arm involves no fitting.

**Deferred HMM.** A fixed sticky HMM is a worthwhile later sequence-model competitor, but adds a separate objective, implementation, and validation path. It is not part of this Stage A brief. Beating A0+filter will support a comparison with that particular smoother, not superiority over HMMs or all temporal methods. The review's proposed HMM also needs an explicit prior scale consistent with n_t/W likelihood weights and a geometric-duration interpretation before implementation.

**A1-KC.** Identical to A1, but with the K=C vocabulary of the same fitting population and seed, the shared cost normalization log(500), and eta distributed over C codes. Its initialization applies the same vocabulary-level rule. With K=C, distinct occupied prototypes map directly to states. The denominator and total pseudocount are fixed across K; the per-code pseudocount changes with alphabet size. This controls the avoidable cost-scale change, while alphabet size and emission capacity necessarily differ.

**A-cont.** Let z_t be the L2-normalized standardized PCA-64 window vector produced by the tokenizer transform for the same normalization condition and fitting population. Zero vectors are flagged and keep cost 1/2 for every state. The cost is D[t,a]=(1-z_t dot mu_a)/2. Initialize mu_a=normalize(sum over fitting windows with g(u_t)=a of n_t z_t), using the same K=500 grouping g as A1 for paired initialization. Update mu_a=normalize(sum[r,t] n_t R[t,a] z_t). A zero initial state mean makes A-cont unavailable and is reported; a zero later update retains the previous unit prototype and is flagged. Transport settings, solver, and budgets are identical to A1. Equal coefficients do not make the cosine and categorical costs identically calibrated; disclose this.

**Permutation sanity check (not a primary comparison).** Fit A1 on within-recording permuted inputs, pooled, with seed 111 only, for both datasets and both normalizations. Permute only complete equal-length windows, leaving the partial terminal window fixed. Use default_rng(seed+10000000) over sorted recording names and save the permutations. Keep labels in their original positions. This destroys the local code–label alignment as well as the temporal structure, so it cannot isolate temporal modeling. Recording-level action/subject composition remains, and no near-chance outcome is guaranteed. Report it in a separate sanity table.

**Matrix and scope.** There are four fitted arms: A0, A1, A1-KC, and A-cont. The derived A0+filter requires no fit. Pooled: 2 datasets × 2 normalization conditions × 3 seeds × 4 arms = 48 fits, plus four seed-111 permutation sanity fits = 52. Subject-disjoint: four folds × 48 = 192 fits. Total planned Stage A fits: 244. Each regular configuration yields ten prediction sets (four fitted arms plus filter, each under T and E); each permutation fit yields two. Keep fit IDs distinct from prediction-set IDs. Build shared tokenizer/preprocessing/grouping artifacts once. Score SMQ separately; compatible HiST-VQ/MASQ exports are optional matched system comparisons, not prerequisites.

**Order and decision point.** Run pooled T first, then the pooled sanity checks and E sensitivity, alternating datasets and completing three paired seeds per configuration. The runner stops after this protocol. The user may launch subject-disjoint evaluation separately with the same frozen settings and the pooled run ID. This decision manages scope and budget; it does not authorize changing settings, dropping losing arms, or claiming transfer without completing the folds. If transfer is not run, list it as not run and restrict conclusions to pooled fitting.

The launcher receives an explicit runtime budget, reserves 30 minutes for reporting, and records all unstarted/interrupted cells. Dry run lists the matrix and estimates from existing timings without fitting. An explicit timing-probe mode may run small label-blind jobs. Primary T work has priority over E; missing E cells remain visible. Never shrink the declared primary grid by score.

### Acceptance and interpretation

Before real fits, pass these tests:
- reference parity, and objective finite differences
- frame-mass and terminal-window handling
- vocabulary-level initialization
- mode-filter tie rules
- A-cont cost and prototype updates
- the permutation check
- leakage, and an evaluator round trip

Include planted sequences with missing actions, repeated actions, unequal durations, and a single window. Tests verify numerical contracts and expose behavior; they must not assert that unsupervised optimization always recovers ground truth.

Engineering completion requires reproducible artifacts and an honest status for every planned cell. The scientific assessment has three parts:
1. **The declared claim:** A1 against SMQ and A1-KC.
2. **Temporal assignment:** A1 against A0 identifies the effect of adding the declared structural term under paired fitting. A1 against A0+filter asks whether it improves on this fixed smoother. Neither proves superiority over all sequence models.
3. **The value of discretization:** A1 against A-cont.

Temporal gains accompanied by worse segment metrics, near-single-state solutions, subject-aligned states, or poor transfer must remain visible. A negative result can still justify completing the planned context comparison; an invalid solver must be repaired before proceeding.

## 4. Stage B — Masked-code contextual representation

### Architecture and objective

Freeze each Stage A K=500 tokenizer. Train one contextualizer, B-code, whose input is code IDs and a learned mask token. A continuous-input masked-prediction competitor is deferred to a separately specified follow-up; it is not another primary architecture in this plan.

Use position information and predict the same K=500 code targets. Initial architecture: embedding width 128, four encoder layers, four heads, feed-forward width 512, dropout 0.1, maximum context 128 windows, and a linear 500-class prediction head. Use sinusoidal positions. The representation is the final hidden state before the prediction head.

Train with cross-entropy only at masked positions, weighting each position by its real frame count. Mask 30% of valid positions using contiguous spans of eight tokens; merge overlaps and trim deterministically to the requested count. Replace every selected input position with the mask token. Padding is neither a target nor an attention key. Never cross recording boundaries. Freeze code assignments throughout; do not recluster targets or train the original encoder.

B-code borrows HuBERT's masked prediction of discrete targets, but it is not a faithful HuBERT reproduction, because its input is already discrete. There are no target-refinement iterations. Low masked-code loss alone does not establish action semantics or subject invariance. [3]

### Training, validation, and inference

Within each fitting population, reserve approximately 10% of frames as validation by selecting whole recordings using a seed-111 shuffled recording list until reaching the target; retain at least one fitting and one validation recording. Save this split and reuse it across methods. Validation and fitting records must never overlap; outer held-out subjects must never enter this split. If insufficient recordings exist, report the configuration unavailable.

Use AdamW, learning rate 3e-4, weight decay 0.01, batch size 64, gradient norm clipping at 1, maximum 50 epochs, and patience 8 on fixed-mask validation cross-entropy. Save the lowest-validation-loss checkpoint, breaking ties toward the earlier epoch. No action labels choose the checkpoint. Log code-frequency and neighboring-code prediction baselines to detect an easy repetition task. HuGaDB's existing W=15 cache contains approximately 76k tokens; overlapping training chunks do not create independent observations. Do not infer convergence speed or adequacy of training data from that count alone. Hardware and the runtime budget are declared before training; do not silently alter batch size or architecture after a failure.

At inference, disable dropout and use unmasked inputs. Cover each recording with length-128 chunks at stride 64, adding a final anchored chunk if needed. Average hidden states for tokens seen in multiple chunks, then L2-normalize each vector, retaining zero-vector flags. Keep the last partial token and exact frame alignment. The model uses future context and is offline.

Train the same three paired seeds for both normalization conditions and datasets; repeat within each downstream subject-disjoint fitting fold. Reuse one learned contextualizer for all Stage C arms requiring that representation. Report training curves, held-recording prediction performance, embedding variance/effective rank, runtime, and coverage. Export a strict sequence-to-embedding interface plus checkpoint and tokenizer hashes.

**Subject-association diagnostics.** Keep diagnostics descriptive and inexpensive. Save frame-weighted subject-by-predicted-state counts and mutual information at evaluation. For contextual embeddings, sample at most 10,000 complete windows uniformly with seed 111, including recording/subject IDs. Report cosine 10-nearest-neighbor same-recording/same-subject rates before and after excluding same-recording candidates, exclude self-neighbors, and report the corresponding sampling-composition reference rates. This exposes local repetition and cross-recording association separately. Subject metadata are labels for these diagnostics, even though action annotations are not used. Do not use them to select checkpoints or settings. A fitted subject classifier is deferred; it would add split and optimization choices. None of these diagnostics isolates subject identity from action/recording conditions.

The internal validation split excludes recordings from contextualizer gradient updates, but the frozen tokenizer may already have used them. Describe it as contextualizer validation, not an independently held-out tokenizer evaluation.

### Stage B checkpoint

Accept implementation only after tests verify masking, padding, chunk stitching, absence of label access, fold isolation, deterministic reload inference. Inspect collapsed or trivial prediction behavior and report it; do not choose architecture settings from action scores. Stage C is the planned downstream test of usefulness, so masked-code accuracy is not a standalone acceptance claim for segmentation.

## 5. Stage C — Contextual temporal transport

### Frozen-context model

Freeze the Stage B checkpoint for each configuration. Replace categorical emission costs with D[t,a]=(1-z_t dot mu_a)/2, where z_t and mu_a are unit vectors. Retain Stage A's frame masses, local temporal kernel, marginal penalty, entropy term, solver, inexact alternating schedule, both coefficient settings, state count, frame decoding, and evaluator. The bounded cosine cost is a declared new cost geometry; equal regularization coefficients do not make the two representations identically calibrated.

Initialize prototypes as mu_a=normalize(sum over fitting windows with g(u_t)=a of n_t z_t), using Stage A's vocabulary-level grouping g, so initialization is paired with A0/A1/A-cont. Alternate transport with mu_a=normalize(sum[r,t] n_t R[t,a] z_t). A zero initial state mean makes that contextual arm unavailable; a zero later update retains the previous unit prototype and is flagged. Recompute the final transport under settings T and E after the final update. The categorical pseudocount prior does not carry over to continuous prototypes. Do not train the contextualizer on ASOT pseudo-labels in this stage.

### Comparisons that determine the result

| Representation | No temporal term | Fixed mode filter | Temporal transport |
| --- | --- | --- | --- |
| Hard code identity, K=500 | A0 | A0+filter | A1 |
| Hard code identity, K=C | — | — | A1-KC |
| Continuous PCA windows | — | — | A-cont |
| B-code context | Context-only, beta=0 | Context-only + filter | Contextual ASOT |

Reuse the completed Stage A predictions. Fit the contextual arms with paired initialization and the same numerical budgets. Compare each contextual temporal arm with its context-only and filter arms, with A1, and with A-cont, preserving both normalization conditions, both coefficient settings, and all seeds. This separates measured incremental effects; it does not prove a mechanism from one aggregate score.

Add a pooled, seed-111-only order-destruction diagnostic for B-code by permuting complete code windows within each recording before contextual inference, retaining the terminal token. Keep the trained contextualizer fixed, refit Stage C prototypes on the permuted fitting inputs, and evaluate permuted held-out inputs. Report it separately from Stage A's sanity check. This is an out-of-distribution diagnostic for the contextualizer, not proof that it learned a semantic action grammar.

Report the complete five-metric matrix, matched seed differences, downstream subject-disjoint scores, occupancy, run durations, solver residuals, and runtime. Do not choose the temporal radius or context checkpoint after inspecting action labels. Any later sensitivity study must have a new protocol/version and retain the primary results.

### Research decision after Stage C

**What each comparison can establish.** Stage A can test A1 versus A1-KC under a shared decoder family. Stage C tests adding context at K=500 and adding transport to that context. Because there is no K=C contextualizer, beating A1-KC with Stage C cannot isolate vocabulary size: context also changed. Preserve this boundary rather than adding another contextualizer grid.

Report F1@50 paired differences and all five metrics under primary T, across the declared seeds and both datasets. Show E directions separately. Restrict transfer claims to completed subject-disjoint runs. A strong positive result requires agreement across the relevant controls and no concealed collapse or numerical failure; it is not a pre-written publication conclusion. A-cont matching or exceeding A1 means no segmentation advantage over this continuous competitor was demonstrated, not that discretization has no possible value.

**Where the field stands.** The current unmatched published references are:

| Method | HuGaDB MoF / F1@50 | LARa MoF / F1@50 |
| --- | --- | --- |
| SMQ | 42.0 / 24.3 | 37.4 / 16.4 |
| HiST-VQ | 48.2 / 28.3 | 45.9 / 19.3 |
| MASQ | 52.3 / 39.5 | 47.2 / 22.2 |

HiST-VQ already learns a fine-to-coarse (sub-action to action) quantization end to end. A contribution here must therefore rest on post-hoc decoding over a frozen, class-count-agnostic vocabulary, and on evidence beyond one favorable dataset or metric. These papers evaluate BABEL, making it a natural later benchmark for comparable positioning. BABEL is outside Stages A–C and is not a universal prerequisite for publishing a narrower, properly supported result. Do not advertise state of the art from unmatched paper-table numbers.

**When results are limited or negative.** If gains are limited to pooled fitting, one setting, or one metric, report that limitation before expanding the architecture. If the components do not improve segmentation, keep the negative evidence: no stage guarantees a top-conference contribution. Two later studies are possible, each a separate controlled axis:
- A matched tokenizer study comparing K-means, native learned VQ, and a genuinely trained FSQ bottleneck, without changing the decoder.
- A multi-granularity study (several C over one frozen vocabulary).

Pose normalization or encoder-loss changes would be yet another controlled axis, not evidence supplied by this plan.

## 6. Implementation, artifacts, and handoff

Use script/action_transport/ for new experiment code and results/action_transport/<run_id>/ for all outputs. Keep the code small: explicit configuration, token/cache loading, a transport adapter, categorical/continuous models, and a runner/evaluator boundary. Prefer short typed functions and small data records to plugin registries, deep inheritance, or copied training frameworks. Reuse verified scoring and windowing functions; do not import old runners or summarizers for their side effects.

Every run must save a configuration hash, source revision, upstream commit/license, full checkpoint hashes, package versions, device, recording/fold manifests, cache dependency hashes, initialization and grouping, prototypes/emissions, per-recording assignments and frame predictions for each setting, objective/residual traces, runtime logs, seed-level results, and a planned-cell status table. Use atomic writes and dependency-checked resume; never silently reuse an artifact under different settings.

Deliver one PowerShell execution script with explicit stage, protocol, hours, and resume options. The protocols are pooled and subject_disjoint, and there is no combined option. A subject-disjoint launch must name the completed pooled run it follows. A dry run prints the full grid and verifies caches without fitting. A separate explicit timing probe estimates cost using tiny label-blind jobs. A validation-only mode runs synthetic tests and tiny smoke checks. Full runs are launched by the user. Stage A defaults to CPU; CUDA is an explicit verified run option, with no silent fallback or new hardware request. Checkpoint at safe boundaries and preserve interrupted partial outputs with an incomplete status.

The report must include:
- pooled and downstream subject-disjoint results, kept separate
- settings T and E, kept separate, with T primary
- a short factual summary
- all comparison tables, organized by the three questions in §3
- budgeted-algorithm results, convergence-qualified paired comparisons, status counts, and coverage
- limitations, open questions, and not-run cells
- action-timeline examples from a fixed seed-111 recording selection, not examples selected for favorable scores
- an architecture figure, a five-metric comparison figure, and occupancy/duration diagnostics

Stage A's detailed agent instructions are supplied in STAGE_A_AGENT_INSTRUCTIONS.md (version 1.2). Stage B and C numerical defaults should be reviewed for feasibility at their implementation handoff using runtime and label-free diagnostics; any change becomes a new frozen configuration before action evaluation, not silent tuning of this document.

## 7. Sources and scope of borrowing

[1] Xu and Gould (2024), Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation. Source of the temporal transport formulation, relaxed action-marginal approach, and the temporal structural prior. https://arxiv.org/abs/2404.01518

[2] Official ASOT implementation, mingu6/action_seg_ot, commit 0c4c86b1037eb4c3e262197aeee115b31e220c9d. Numerical reference and license source. https://github.com/mingu6/action_seg_ot — solver: src/asot.py; coefficient lineage: src/train.py CLI defaults; constructor defaults and run_bf.sh use different values.

[3] Hsu et al. (2021), HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. Source of masked hidden-unit prediction. The planned discrete-input B-code model is an adaptation, not a HuBERT reproduction. https://arxiv.org/abs/2106.07447

[4] Local project evidence: Current_State_Discrete_Motion_Project.docx; results/exp2round/q1q2/REPORT.md and its detailed tables; results/segmodel/ANALYSIS.md; and results/segmodel/m1r2/ artifacts. Read each result with its original protocol and validity status. No new result is asserted by this planning document.

[5] Kinoshita et al. (2026), Action Motifs, supplied in related work/. Its pose/shape-normalization and hierarchical representation direction remains a future input/encoder axis. None of its architectural components is silently included in Stages A–C.

[6] Gökay et al. (2025), Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation (SMQ). Source of the released checkpoints and the class-matched baseline.

[7] Ahmed et al. (2026), Unsupervised Skeleton-Based Action Segmentation via Hierarchical Spatiotemporal Vector Quantization (HiST-VQ). Prior fine-to-coarse design; unmatched published reference. https://arxiv.org/html/2604.15196v1

[8] Qin et al. (2026), MASQ: Mask-Aware Spatiotemporal Quantization for Unsupervised Skeleton Action Segmentation. Recent reported benchmark results; unmatched published reference. https://arxiv.org/html/2608.29891v1
