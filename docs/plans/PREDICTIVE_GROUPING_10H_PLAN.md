# Ten-hour experiment: predictive temporal grouping

Date: 27 September 2026. Status: experiment design; implementation and execution are separate deliverables.

## 1. Decision this run must support

Determine whether a learned, label-free signal can place temporal consistency more usefully than fixed ASOT, simple feature changes, or local code-frequency changes. This is the first bounded test of the proposed adaptive composition mechanism for the seven-week project.

**Primary hypothesis:** with contextual features, action prototypes, transport coefficients and temporal-penalty mass fixed, predictive disagreement between the two sides of a candidate boundary improves segmentation when used to redistribute the temporal penalty.

**Mechanism hypothesis:** the placement of this signal matters. A displaced version of the same signal should be weaker, and a learned predictor should add value beyond an empirical code-histogram signal.

This is a decoder experiment on the complete HuGaDB and LARa collections. It is not a new MAE/FSQ training experiment, a transfer evaluation, or a complete demonstration of the proposed paper architecture. Prediction disagreement is a candidate boundary signal, not an assumed semantic boundary probability. A negative result concerns this particular signal and decoder intervention.

## 2. Scope and artifacts

- Primary: raw inputs, both datasets, seeds 111, 222, 1538574472.
- Secondary: per-recording-normalized inputs, both datasets, seed 111 only, if the budget permits. These are deliberately one-seed sensitivity results, not incomplete three-seed estimates.
- Use every recording and every real frame. Do not shorten recordings, subsample the evaluation population, or select favorable subjects.
- Reuse Stage A tokenizer identities and Stage B `stage_b_003` embeddings/checkpoints, with Stage C `stage_c_pooled_001` action prototypes.
- K=500, HuGaDB W=15 at 60 Hz, LARa W=12 at 50 Hz. These are approximately quarter-second tokens, not interchangeable frame rates.
- Action states: 10 for HuGaDB and 8 for LARa. HuGaDB has 18 subjects and IMU inputs; LARa has 16 subjects and optical motion-capture inputs. Include every observed class, including LARa `None`.
- Freeze encoders, vocabularies, contextualizers and action prototypes. Do not update the prototypes separately for the new kernels.
- Train only the small code-distribution forecaster defined below.
- Primary coefficients T: a=0.7, beta=0.3, lambda=0.05, epsilon=0.07. Do not run E, vocabulary sweeps or coefficient sweeps.
- New code belongs in `script/action_transport/predictive_grouping/`; outputs in `results/predictive_grouping/<run_id>/`. Preserve all historical code and results.

The saved Stage C prototypes came from capped outer fits. Holding them fixed isolates decoding changes, but does not resolve outer-fit nonconvergence. No result from this experiment may be described as a converged comparison of fully refitted architectures.

## 3. The learned signal

### 3.1 Forecaster and objective

Train a small model to predict the distribution of codes in the next H=4 complete windows from the preceding M=16 complete windows. Physical horizons are 1.00 s / 4.00 s on HuGaDB and 0.96 s / 3.84 s on LARa.

For a training example, the target is the normalized empirical histogram of the four future codes. Optimize soft-target cross entropy, `-sum_k target[k] * log_softmax(logits)[k]`. The forecast target uses no action labels or ground-truth boundaries.

Architecture: code embedding 64 dimensions; concatenate the 16 embeddings in temporal order and an 8-dimensional learned direction embedding; Linear(1032,256), GELU, LayerNorm(256), Linear(256,500). There is no action classifier, attention stack, decoder or additional representation loss.

Use one model per dataset/normalization/seed. Train on both forward and reversed recordings; the direction embedding distinguishes these orientations. Reverse only for constructing forecasting examples, never for scoring segmentation.

Training settings: AdamW, learning rate 3e-4, weight decay 0.01, batch size 256, gradient-norm clipping at 1, 500 warm-up updates followed by cosine decay to 3e-5 at update 5000. Cap at 5000 optimizer updates. Validate every 100 updates; stop after 1000 updates without improved validation cross entropy; restore the lowest-validation-loss checkpoint. No action score enters selection.

Within each batch, choose direction uniformly, then sample a valid complete-window example uniformly from the fitting population, with replacement. Save RNG states. Use the cell seed for initialization and training randomness. All three pipeline seeds include different upstream vocabularies/contextualizers, so their spread is not attributable solely to this forecaster's initialization.

### 3.2 Recording split and validation

Create five recording-grouped folds separately per dataset: order recordings by decreasing real-frame count, break ties with seed 111, and greedily place each recording into the least-loaded fold. Fold 0 is validation; folds 1-4 are fitting. Reuse memberships across normalizations and seeds. Do not use action labels or subject labels to construct these folds.

Fit only on fitting recordings. Select a fixed uniform sample of at most 10,000 valid examples from each direction on the validation recordings, seed 111, and reuse example identifiers across compared forecasters. Report validation cross entropy for the learned forecast, fitting-population unigram frequencies, and the empirical histogram of the 16 visible context codes. For these two baselines use additive 1/500 pseudocount per code (total pseudocount 1) before normalization.

Do not drop a forecaster because it fails to beat those baselines. Report its downstream result and forecasting failure. An invalid numerical fit is a different, explicitly invalid condition.

The downstream decoding remains pooled and transductive, including exposure through the frozen encoder/contextualizer. This recording split checks forecasting generalization only; it does not make the segmentation evaluation held out.

### 3.3 Predictive disagreement at a gap

Index a gap by i, between windows i-1 and i. On gaps with at least M complete windows on both sides, compute:

- `q_left`: forward forecast from codes i-M through i-1.
- `q_right`: backward forecast from reversed codes i through i+M-1.
- `s_predictive[i] = JS(q_left, q_right)`, with natural logarithms and zero terms handled exactly.

Each distribution predicts code frequencies on the other side using only its own side. This compares their forecasts; it is not the likelihood of one shared observed target and must not be called that. A periodic motion can also create disagreement: whether the signal identifies useful grouping changes is the experiment.

For comparable nonlearned signals, on the identical eligible gaps compute:

- Histogram signal: JS between the empirical code histograms in those same 16 windows on each side, with the same additive smoothing defined above.
- Geometry signal: `1 - z[i-1] dot z[i]`, where z is the saved unit-normalized contextual embedding. Unlike the forecaster inputs, these embeddings already contain bidirectional context; disclose this difference.

The forecaster never receives z, action prototypes, actions or subject IDs.

### 3.4 Label-free scale calibration

For each signal and cell, sample up to 10,000 eligible fitting-recording gaps uniformly without replacement at seed 111. Reuse gap IDs across signals and pipeline seeds. Fit its empirical midrank CDF `F(s) = (count(sample < s) + 0.5*count(sample == s))/n`.

Convert a score to an adjacent-gap affinity `g[i] = 1 - 0.9*F(s[i])`, between 0.1 and 1. Larger disagreement weakens local consistency. This deliberately fixes a scale convention rather than tuning temperature on action scores. Save score distributions and calibrators. If the calibration sample is empty or has exactly zero range, declare it uninformative and set all that signal's affinities to 1, including edge gaps; this must reduce to the fixed kernel after normalization. Do not amplify a constant calibration signal through a rank transform.

For ineligible edge gaps use g=1 in all adaptive arms. A terminal partial window is retained for decoding but cannot supply a forecasting example or an eligible gap. Short recordings with no eligible gaps use the fixed kernel in every arm and are counted explicitly.

## 4. Five matched decoder arms

| Arm | Temporal affinities | Purpose |
|---|---|---|
| F: fixed | Existing one-second ASOT band | Matched fixed-kernel reference |
| G: geometry | Adjacent contextual-feature change | Does ordinary feature contrast suffice? |
| H: histogram | Left/right empirical code-frequency disagreement | Does a simple nonlearned code signal suffice? |
| P: predictive | Forecaster disagreement | Proposed intervention |
| S: displaced predictive | Circularly displaced P affinities within each recording | Does placement matter beyond having nonuniform weights? |

For S, shift only the contiguous eligible-gap affinity sequence, retaining its values and counts exactly. Choose a nonzero circular shift using a stable SHA-256-derived seed from `(dataset, normalization, pipeline_seed, recording_id, 10000000)`; never Python's process-dependent `hash`. When possible require both circular distances to exceed the kernel half-width b. If this is impossible, use any nonzero shift and report the case. With fewer than two eligible gaps, S equals P and contributes no displacement evidence. Predictions, codes, features and labels remain in their original order. Use one displacement per cell; it is a descriptive control, not a permutation significance test.

### Kernel construction and equal-strength matching

Let `V0[t,s] = L/b` for `0 < |t-s| <= b`, zero otherwise, with `b=min(L-1,max(1,floor(fps/W)))`. L is the recording's true window count. If L=1, all kernels are zero.

For an adaptive arm and positive lag d<=b, define the raw pair affinity as the minimum adjacent-gap affinity crossed between t and t+d. Normalize **separately for every recording and lag**, using real frame masses p:

`m_d = sum_t p[t]*p[t+d]*raw[t,t+d] / sum_t p[t]*p[t+d]`

`V[t,t+d] = V[t+d,t] = (L/b) * raw[t,t+d] / m_d`.

This preserves `sum_t p[t]*p[t+d]*V[t,t+d]` exactly relative to F at every lag. The radius, distance profile and coupling-independent temporal-penalty mass therefore match. Some individual weights can exceed V0: this is redistribution, not merely weakening every edge. It does not force the achieved objective's temporal component to be equal, since assignments differ.

Store symmetric band weights, never dense L-by-L matrices. Implement one band-application function shared by the objective and gradient; test against an independent dense reference. Shift affinities before building and normalizing S's kernel. Do not claim the final S/P pair-weight histograms are identical; only adjacent-gap affinity counts and normalized lag masses are matched.

## 5. Frozen costs and optimization

For every arm in a cell use exactly the same saved contextual z and Stage C contextual-ASOT prototypes mu. Re-normalize z in float64 as Stage C does. Verify prototype norms; do not silently change mu. Preserve the existing zero-feature cost convention.

`D[t,a] = 1 - z[t] dot mu[a]`.

Use the current corrected log-domain transport objective, including its marginal and entropy terms. Only V differs. Initialize every arm from `T=p*q`, not from its historical predictions or another arm's solution. Keep q uniform over the dataset's C states.

- Primary budget: at most 2000 accepted mirror steps per recording.
- Save a prediction/diagnostic checkpoint at 500 steps, then continue the same state to 2000; if it converges earlier, reuse that converged state for later checkpoints and record actual steps.
- Keep existing stopping tolerances: centered-gradient residual <=1e-5, relative objective change <=1e-7, patience 5, existing backtracking settings. Use the existing solver's exact combined stopping semantics and record them in the manifest.
- Final precision: float64. Transport runs on CPU with at most eight numerical threads; GPU is used for forecasting only. Do not opportunistically switch backend after examining results.
- Save final log couplings, residuals, objective components, step counts and raw state predictions for every recording.
- Check deadlines inside the solver. A partial recording collection is incomplete and cannot yield a primary full-dataset score.

**Higher-budget sensitivity:** for both raw datasets, seed 111, extend all five arms from their 2000-step states to at most 5000 steps, with unchanged objective and parameters. Perform this before normalized-input sensitivities. Report whether score differences and frame assignments stabilize; this tests final-solve sensitivity only. Do not infer stationarity from the step cap or from small changes in F1 alone.

Re-score historical Stage C outputs as external references, not matched controls: their warm starts, final-step budgets and fitted kernels differ. The primary comparator is newly decoded F.

## 6. Grid, runtime and deadline

The ten-hour clock starts at the execution script's first preflight action. Implementation is completed and reviewed before launch; it is not part of the ten-hour run. Persist the original UTC launch/deadline and derive a process-local monotonic deadline from remaining time at each launch. Reporting must finish by hour 10; expensive work stops by hour 9.5. A resume does not reset the budget or exclude time between launches.

Planned primary grid: 2 datasets x 3 seeds x 5 arms = **30 full-dataset decoder runs**, with six forecasters. The 500-step checkpoint is a checkpoint of these runs, not another independent fit.

Planned higher-budget sensitivity: 2 datasets x seed 111 x 5 arms = **10 continuations**.

Planned secondary normalized grid: 2 datasets x seed 111 x 5 arms = **10 decoder runs**, with two additional forecasters. No higher-budget extension of this sensitivity.

Planning allocation, not a guaranteed runtime:

| Elapsed time | Work |
|---|---|
| 0:00-0:30 | Provenance, numerical tests and representative timing |
| 0:30-1:30 | Raw forecasters, validation and affinity caches |
| 1:30-7:30 | Complete the 30 primary decoder runs |
| 7:30-9:30 | Higher-budget sensitivity, then normalized sensitivity if feasible |
| 9:30-10:00 | Aggregate verified artifacts, figures, limitations and report |

Historical Stage C timings were approximately 377-391 seconds per HuGaDB contextual-ASOT fit and 657-669 seconds per LARa fit, including alternating fitting and both final settings. These are feasibility evidence only: they are not timings of this new variable-band kernel or its 2000-step cold-start solve.

Preflight timing uses label-free median-length and 95th-percentile-length recordings from each dataset, selected by lengths only. Include all five kernels. Cache reusable work. Project full-run cost by windows and observed step costs, use the slower observed cost per window and a 1.5 safety factor, and refresh the projection after every completed cell.

Execute paired five-arm bundles: HuGaDB seed111, LARa seed111, HuGaDB seed222, LARa seed222, HuGaDB seed1538574472, LARa seed1538574472. Rotate arm order deterministically by seed to avoid always leaving the same comparator last. Keep all expensive jobs sequential.

Do not start a bundle projected to cross 9.5 hours. Drop normalized sensitivities first, then 5000-step continuations. If the primary grid still cannot finish, preserve completed paired bundles and mark every remaining cell not run. Never reduce dataset size, change a hyperparameter or present one/two seeds as a three-seed result. Report the primary decision as incomplete when the required paired seeds are missing.

## 7. Evaluation and decision rules

Use the original dataset-level Hungarian mapping and existing MoF, Edit and F1@10/25/50 helpers. Primary endpoint: F1@50 at the 2000-step budget. Include raw state predictions and matching tables. Ground-truth action labels enter scoring and post hoc diagnostics only; they never fit the predictor, calibrate affinities, choose checkpoints or tune kernels.

Report seed-level values, paired P-F, P-G, P-H and P-S differences, means and population SD. Pipeline-seed variation is sensitivity, not sampling uncertainty. Do not pool different datasets, checkpoints or solver budgets into one mean.

Also report predicted/GT segment-count ratio, segment-duration distributions, state occupancy, per-class recall and short/medium/long GT-segment performance (within-dataset duration tertiles computed only for reporting). These diagnostics must not select the method or exclude classes. Plot gate strength around GT boundaries versus interior positions, explicitly as post hoc evidence; boundary correlation alone is not the primary success criterion.

Predeclared engineering decision thresholds, not significance tests:

- **Strong progression evidence:** P-F is at least +2 F1@50 points on both datasets in the three-seed mean; P-F is positive in at least two of three seeds on each dataset; P-G, P-H and P-S are each positive in mean on each dataset; and mean MoF does not fall by more than 1 point against F on either dataset.
- **Conditional result:** improvement appears on only one dataset, or P does not beat G/H/S. Identify the exact limitation; do not call a nonlearned-gate improvement evidence for the forecaster. A finding that all adaptive kernels help supports studying placement, not the necessity of learned prediction.
- **No progression evidence for this proposal:** complete primary results do not satisfy either a replicated dataset-specific P-F gain or an advantage over the simple controls. Stop expanding this forecaster/kernel design; preserve it as a negative control when moving to the representation/quantizer experiment.
- **Optimization unresolved:** if paired comparisons reverse sign between 500 and 2000 steps, or available 5000-step extensions reverse their sign, qualify the affected decision as unstable. If the higher-budget check is omitted while relevant runs remain capped, conclusions are explicitly conditional on the 2000-step algorithm. No claim about fully optimized performance is permitted.

The +2/-1 thresholds express a practical investment criterion for seven weeks; they are not published standards, confidence intervals or evidence of SOTA. Passing this experiment earns a full refit and transfer test of the mechanism. It does not establish a new architecture's SOTA or prove FSQ useful.

## 8. Required checks before expensive execution

1. Verify full hashes of source files, released encoders, vocabulary artifacts, Stage B checkpoint and every consumed embedding file, Stage C mu, and source CSVs. Verify code hashes, recording identities/order, frame lengths, finite values and expected latent dimensions. New artifact identity includes all these dependencies and the frozen configuration.
2. Verify forecaster examples do not cross recording boundaries or include terminal partial windows; targets do not leak into context; validation recordings are excluded from fitting; reverse indexing and both gap sides match explicit toy sequences.
3. Check JS nonnegativity, symmetry and equality at identical distributions; CDF tie behavior; constant-signal reduction to F; exact adjacent-affinity count preservation for S; every real frame receives a prediction.
4. Compare band kernel application against a dense toy reference, validate per-lag mass equality and symmetry, and finite-difference the full transport objective against the implemented gradient. Check single-window and short-recording cases.
5. Reproduce the original convolution kernel using the new band implementation at all-one affinities, including lengths where fractional-radius rounding previously failed.
6. Verify all arms have identical costs, mu, p, q, coefficients and initial coupling. Check finite, nonincreasing objectives and row-mass conservation; record numerical failures explicitly.
7. Replay at least one saved Stage C prediction set per dataset through the scorer; require agreement with the stored metrics at their recorded precision. This checks scoring, not a requirement that the new cold-start F output equal historical output.

No new D7 model extraction is needed when the already-passed provenance gates and their immutable dependencies are verified. If a required cache fails identity verification, do not silently substitute a checkpoint or retrain an encoder inside this run.

## 9. Implementation and outputs

Keep the implementation self-contained: a small configuration/identity module, forecaster and signal module, band-kernel/decoder module, runner, reporter and focused tests. Reuse the existing dataset loader, tokenizer identity checks, scoring and exact objective where possible. Do not modify the legacy experiment's mathematical behavior. Avoid a general experiment framework or hidden global configuration mutation.

Provide one PowerShell execution entry point, `script/action_transport/predictive_grouping/run.ps1`, with preflight, run ID, hours and resume options. It must run preflight, execute the planned priority order, persist results incrementally and generate the report even after a budget stop. This path is a planned deliverable; the script does not exist merely because this plan names it. The user launches the actual run.

Required outputs:

- `manifest.json`, `planned_cells.csv`, `cells.csv`, `not_run.csv` and runtime events.
- Recording splits, forecasting training curves/checkpoints/validation metrics, sampled example and calibration IDs.
- Per-recording scores/affinities, kernel normalization constants, predictions, log couplings, solver diagnostics and dependency hashes.
- Paired effect table for all four primary comparisons, plus separate historical and normalized-sensitivity tables.
- Three figures: F1@50 and MoF by arm; fragmentation and duration diagnostics; score/residual/prediction changes across solver budgets.
- `REPORT.md`: exact coverage, answer to the two hypotheses, practical progression decision, optimization limits, failed controls and not-run entries. Include the next proposed action, without claiming a validated mechanism when its controls fail.

## 10. Relationship to the seven-week project

This run tests the proposed composition mechanism before spending weeks on a new encoder. It deliberately keeps its interpretation narrower than the eventual paper claim. It adds full-data evidence across two existing modalities, not evidence that more pretraining data helps. A larger-data MAE/quantizer experiment remains necessary and must separately control bottleneck capacity, bitrate, continuous inputs and exposure to evaluation data.

ASOT supplies the transport objective; masked-code modeling motivates using predictions as a grouping signal. Neither a new name nor a favorable pilot establishes novelty relative to Action Motifs, HiST-VQ, CLOT or boundary-aware segmentation. A positive outcome must be followed by the specific novelty audit and held-out evaluation already required in the seven-week proposal.
