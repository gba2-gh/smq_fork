# Agent instructions — Milestone 1: fixed-unit action dynamics

Version 2.2 · 6 October 2026 — reduced LARa feasibility protocol (changes: [v2.1](M1_AGENT_INSTRUCTIONS_CHANGES_v2.1.md), [v2.2](M1_AGENT_INSTRUCTIONS_CHANGES_v2.2.md))

Parent: [Compositional action dynamics proposal](COMPOSITIONAL_ACTION_DYNAMICS_PROPOSAL.md), especially Sections 4, 6, 9 and 10.

## 1. Assignment and stopping point

Implement the minimum sequence decoder, its matched controls, verification and a single execution script. The question it prepares us to test is:

> Does action-specific prediction of ordered local-unit sequences provide better action evidence than occupancy and order-agnostic prediction, and improve on a matched fixed-centroid decoder?

**Finish implementation, run tests and a bounded smoke run, then hand the execution command to the user. Do not launch the full feasibility experiment.** The frozen three-seed LARa run and the decision gate in Section 10 are the next execution step. A smoke score is not evidence for the research hypothesis.

Use existing T2 relational K-means units. Do not train an encoder, VQ/FSQ, a new vocabulary or separate masked-prediction contextualizer; add retargeting; change the temporal kernel; introduce adaptive duration; or expand the dataset/seed grid. The new causal history network is part of this milestone. The novelty hypothesis concerns the eventual architecture, not these individually established components.

Keep new code readable, concise and self-contained under `script/compositional_dynamics/`; outputs under `results/compositional_dynamics/<run_id>/`. Preserve existing scripts, manifests and results. Import narrow, verified helpers rather than copying historical runners or modifying files included in earlier implementation hashes.

B1 centroid self-training, other datasets, alternative initialization, emission-scale sweeps and bit-exact resume are deferred. They are not prerequisites for this first test.

The numerical settings below are **new, declared feasibility defaults**, not settings established by earlier experiments. Freeze them in configuration before scored runs. Document any necessary deviation; do not silently substitute a different experiment.

## 2. Inputs, existing split and leakage boundary

Use LARa only, seeds **111, 222, 1538574472**, and each seed's existing preparation:

`results/week5/timescale_inspection/ts5_v1/phase5/prep/t2_relational/lara__t2_relational__seed<seed>/`

Family `tok5_t2_relational`, K=512, 12-frame windows at 50 FPS, C=8. Verify against manifests. Each recording supplies codes, starts, real lengths, frame count and shared lookup[512,64]. Reuse the existing vector-input transformation exactly; do not introduce a new centering operation. The fixed lookup is a deterministic function of code ID. No within-cell continuous feature may enter an action model. One declared exception: the matched B0 initialization (Section 3) uses the existing code-level grouping helper, which averages continuous PCA features per code over fitting recordings. It produces only a code-to-state grouping used to initialize B0; no continuous feature enters any cost, model input or training target.

**Use the existing split.json without regenerating it.** Direct inspection on 6 October found all three files byte-identical: 395 fitting recordings, 44 evaluation recordings, split_seed=111, SHA-256 `04f8a605d3881dea040e6e26fa0a5ab908b4268562ba3a5bdb9eec7f8d2940aa`. Both sides contain all 16 subjects. This corrects the assessment's claim that each seed has a different split. Verify membership equality and disjointness in preflight; a changed source requires explicit reconciliation, not automatic acceptance.

The source field named `val` is the evaluation population here: do not use it for checkpoint selection. Parse subjects from the Sxx filename field and verify complete parsing. This is **held-out decoder evaluation with transductive tokenization**, not unseen-subject evaluation. Tokenizer fitting used all recordings; preserve and disclose that exposure. Seed spread combines vocabulary/model initialization effects, not split variation or sampling uncertainty.

Decoder prototypes, occupancy, model gradients, initial responsibilities, cost calibration and every assignment refresh use **only the 395 fitting recordings**. Evaluation inputs never enter these operations. Freeze the model and scale before decoding the 44 evaluation recordings. Never return their plans to training. Action labels are restricted to final scoring and post-fit diagnostics; known C is the declared benchmark convention.

Reuse the preparation Loader behind a small adapter. Verify source hashes, IDs, starts/lengths, lookup dimensions and exact frame coverage. Close NPZ handles. Missing sources fail preflight rather than triggering new tokenization. Retain terminal windows with real lengths and never connect histories across recordings. Save both populations and their complete source identities.

## 3. Four arms

| ID | Model | Purpose |
| --- | --- | --- |
| `b0_centre` | Fixed T2 centroid ASOT, refitted on fitting recordings only | Matched performance baseline and common initialization. |
| `m1_ordered` | Occupancy plus action-conditioned causal unit prediction | Proposed mechanism. |
| `m1_bag` | Same model with canonically sorted history | Trained order-agnostic control with identical history multiset, parameters and exposure. |
| `occupancy` | Action-specific categorical occupancy without dynamics | Single-unit evidence control. |

Within each seed all arms share tokenizer, fitting/evaluation membership, frame weighting and transport policy. Ordered and bag share backbone initialization, target-window draws and update budget. Occupancy uses closed-form updates; do not call its compute matched to neural training.

Reproduce the historical pooled B0 as a separate seed-111 parity gate. **Do not reuse its fitted prototypes or plans to initialize the held-out experiment.** Build the new initial C-state grouping with the same helper that produced the historical grouping, restricted to fitting recordings: `script/quantized_input_comparison/fit.py::initial_grouping`, called with `codes_by_name`, `lengths_by_name` and `pca_by_name` containing **only the 395 fitting recordings**, and the preparation's `num_codes`, `num_states`, fps, window and seed. The helper computes frame-weighted per-code centres in the family's continuous PCA space and clusters them with `categorical.build_grouping` (weighted KMeans(C), k-means++, n_init=5, max_iter=300, tol=1e-4, Lloyd). Do not substitute a different clustering, and do not cluster lookup rows instead: that would change the initializer relative to historical B0. If the helper cannot run on the fitting subset (for example fewer than C occupied codes), fail preflight and report it. Then build initial prototypes with `continuous.build_mu0` on fitting recordings and run the existing continuous ASOT fit. Do not reuse the preparation's `grouping.npy`, which was fitted on all recordings. Use each recording's saved `continuous` array (the family's PCA features, stored as float32) as `pca_by_name`. Save the grouping, per-code centres/counts and the fitting population. Preflight verifies the construction by calling the helper on **all** recordings and comparing with the saved `grouping.npy`. Equality is expected. If they differ, the preparation computed centres from float64 PCA features, so report the number of differing codes as a documented numerical discrepancy rather than silently accepting or forcing the match.

The newly fitted B0's soft plans on fitting recordings initialize all three new arms. Use the existing B0 fitting settings: 25 inner steps, outer cap 50, patience 3 and relative tolerance 1e-6. Its frozen prototypes decode both populations independently. Historical LARa 18.4 F1@50 is a parity reference, **not the held-out acceptance baseline**. Contextualizer scores are historical context, not the performance target.

## 4. Exact finite-history model

Use the fixed 64-d quantized lookup, followed by a trainable linear projection to 128 dimensions and a single-layer GRU of hidden width 128, no dropout. No recording/subject embeddings or global recording summary.

History length is `H = floor(2.0 * fps / window + 0.5)`: eight preceding windows, or 1.92 s on LARa. Record these realized values. This limits conditioning evidence, not predicted segment duration.

For target t, ordered dynamics consumes exactly `c[max(0,t-H):t]`. Reset the recurrent state for every target history. Do not run a streaming GRU over the entire recording: that would give unbounded support. Batch histories, mask padding correctly, and use zero hidden state for empty history. No current or future code may reach the dynamics predictor.

For `m1_bag`, sort the available history using one saved, seed-111 random permutation of vocabulary IDs as the sorting key, then run the **same GRU**. Repeated codes and history length remain unchanged. This is an order-invariant control, not a claim that code numbers have geometry. It must produce identical predictions for every permutation of the same history multiset.

For action a, use a residual adapter

`v_a = h + Wup_a(GELU(Wdown_a(h)))`,

with bottleneck width 16, then shared `Linear(128,512)` plus action-specific output bias. Use normal seeded initialization, distinct adapters, and stable `log_softmax`; no identical-head initialization that preserves exact state symmetry. Save parameter counts. Shared backbone tensors must initialize identically across applicable arms regardless of construction order.

The occupancy probabilities are `theta[a,k]`. Update from detached responsibilities with a Dirichlet pseudocount of **1 per code per action**:

`theta[a,k] ∝ 1 + sum_t (length[t]/window) * R[t,a] * 1[c[t]=k]`.

Counts use complete-window equivalents, so the pseudocount is one complete window per code per action; terminal windows contribute proportionally. This prevents infinite likelihoods; it does not enforce uniform state usage. Normalize over k. Fix `alpha=0.5`:

`D_raw[t,a] = -0.5*log(theta[a,c[t]]) - 0.5*log(p_a(c[t] | history[t]))`.

Assign this evidence to **t**, not t−1. Occupancy uses `D_raw=-log(theta[a,c[t]])`. For empty history, retain the action prior implied by the predictor and occupancy; never drop the first window.

Do not add a separate powerful action-free predictor to the main model. Compute copy and state-averaged prediction diagnostics without feeding them into costs. Subtracting an action-independent value from every row cannot change a fixed-row-mass OT solution.

## 5. Transport and cost scale

Reuse `script/action_transport/transport.py`. Keep CPU float64 solves, real per-recording row masses `p[t]=length[t]/sum(length)`, uniform `q[a]=1/C`, and `Coeffs(a=0.7,beta=0.3,lam=0.05,eps=0.07)`.

Use the existing `kernel_half_width` and `build_kernel` with each recording's **true L**. The existing radius is approximately one second; do not confuse it with the new two-second history. No padded-length kernel amplitudes. Preserve row constraints and the relaxed column marginal, including solver validity checks.

NLLs and cosine costs have different scales. For each new arm, after the first 500 model updates (or initial occupancy fit), evaluate raw costs on the same saved, uniformly sampled set of at most 10,000 complete **fitting** windows, with sampling seed 111 and the cell's verified fitting population. Let

`s = median(positive(max_a D_raw[t,a] - min_a D_raw[t,a]))`.

Require finite `s >= 1e-6`; otherwise mark calibration undefined and stop that cell. Freeze s for the rest of that arm's training/inference. Use

`D[t,a] = (D_raw[t,a] - min_a D_raw[t,a]) / s`.

Row subtraction is harmless under fixed row masses; division changes relative regularization and is a declared protocol choice. Do not clip costs or recompute s at every refresh. Save raw/scaled distributions and saturation/near-equality diagnostics at every refresh. All three likelihood arms use this policy. Matched and historical B0 keep the original cosine-cost convention; disclose this difference when comparing to B0.

## 6. Fitting and frozen evaluation

1. Fit matched B0 on fitting recordings only. Cache its parameters and soft fitting plans, R0=T0/p, with R0 rows summing to one. All new arms share this R0; no ground-truth mapping enters initialization.
2. Initialize theta from fitting R0. Warm-fit each neural arm for 500 updates with R0 fixed, then calibrate its scale on fitting windows. Calibrate occupancy after its initial count fit.
3. Refresh **fitting** plans from current costs using 25 accepted inner steps, warm-started from previous fitting plans. Invalid/backtracking failure stops the cell. Fixed-step success is budgeted optimization, not convergence.
4. Detach R=T/p; update theta and train 500 more updates. Repeat to **5,000 total updates**, including warm fitting. Refresh after updates 500, 1,000, ..., 4,500. The primary model is the final checkpoint; neither evaluation NLL nor action scores select a checkpoint.
5. Freeze everything. Decode fitting and evaluation recordings separately from the same cold-start convention. Retain 500 and up-to-4,000 final-solver endpoints as a continuation, using established ladder semantics. Primary scoring uses the final 4,000-budget endpoint, including earlier convergence. Save status, residual, objective traces and actual iterations; capped is not converged.

Training: AdamW, constant LR 3e-4, weight decay 0.01, batch 256 target windows, gradient norm clip 1, float32 without mixed precision. Draw targets with replacement proportional to real frame length over **fitting recordings only**, then average losses without additional length weighting. Match target IDs across ordered/bag. Use independent saved RNG streams for sampling, initialization and diagnostics.

Dynamics minimizes detached-responsibility-weighted conditional NLL, retaining its 0.5 factor. Occupancy updates by the window-equivalent counts above. Occupancy-only performs the same nine assignment/count refreshes without invented gradient updates. This is alternating pseudo-target training, not differentiation through OT or guaranteed monotone EM.

Cache only verified completed cells atomically. An interrupted cell restarts from its recorded initialization; partial checkpoints may be retained for inspection but are not resumable success artifacts. No bit-exact GPU-resume requirement. Use deterministic settings where supported and save the actual settings. Benchmark CPU and GPU execution rather than asserting either is faster.

## 7. Verification before handoff

Implement focused tests of scientific contracts rather than tests that merely repeat formulas:

| Check | Required behavior |
| --- | --- |
| Alignment/coverage | Every real frame is covered exactly once; first/terminal windows and variable lengths work. |
| Causality | Changing current/future codes leaves the conditional distribution at t unchanged; changing current code may change its gathered likelihood. |
| Finite support | Changing codes before t−H or in another recording cannot change the output at t. |
| Order control | Bag outputs are invariant to history permutation; ordered model can distinguish equal-multiset histories. |
| Synthetic learnability | On known synthetic processes with equal unit occupancy but different order, action-conditioned training can learn different next-unit distributions. Synthetic state labels are allowed only inside this unit test. This is not a test of unsupervised semantic recovery. |
| Soft assignments | R rows sum to one; T rows equal p; no gradients pass through plans; manually calculated weighted counts/losses agree. |
| Scale | Calibration IDs/scales are saved and verified when reusing a completed cell; row-constant offsets leave transport unchanged within numerical tolerance. |
| Model dependence | Adapters receive nonzero gradients; separate all-head relabelling (a symmetry) from collapsing heads to one shared predictor (a meaningful intervention). |
| Scoring | Existing dataset-global Hungarian evaluator reproduces hand-built segment/frame examples, including `None` conventions; no per-recording remapping. |
| Caching | Completed-cell reuse verifies source/config hashes; interrupted cells restart and cannot be mistaken for completed results. |
| Population isolation | Changing evaluation codes or labels cannot change initial fitting plans, calibrated scales, learned parameters or fitting plans. Evaluation decoding occurs only after model freeze. |

Historical parity source: `results/week5/timescale_inspection/ts5_v1/phase5/units/lara__t2_relational__seed111__centre/info.json`. Reproduce its original pooled preparation/grouping and endpoint settings. Require identical stored metrics under the same environment, or stop and document the numerical/environment discrepancy rather than widening tolerances to force a pass. Retain new frame predictions. Keep this parity job and artifacts isolated from matched-B0 fitting; pooled plans must never enter the experiment.

Keep scoring separate from training. Tests should demonstrate that changing evaluation labels leaves learned models, calibrated scales and plans unchanged.

## 8. Diagnostics and saved artifacts

Diagnostics are split into two tiers. **Tier A is implemented in this milestone**: the gate metrics and the minimum needed to interpret a pass or fail. **Tier B is not implemented now.** The run saves the artifacts Tier B needs, and a separate analysis script computes it later from saved files, only if the gate outcome needs explaining. No Tier B item may require retraining or reloading the model; a Tier B decode uses saved cost components and the existing solver.

### 8.1 Saved artifacts (required)

- Complete source/cache hashes, package/device/thread versions, configuration, fitting populations, sample IDs, RNG states and run identity.
- Native window codes, starts/lengths or verified references; initial/final responsibilities; theta; learned parameters; raw/scaled costs; solver traces; raw window and frame predictions.
- Frozen-endpoint GRU history representations `h` for every window of both populations (float16 is acceptable), with recording and window index, so Tier B representation diagnostics need no model reload.
- Training losses, gradient norms, predictive accuracy, state mass and posterior entropy per refresh. Report occupancy and dynamics contributions separately.

### 8.2 Tier A: computed in this milestone

- Primary MoF, Edit and F1@10/25/50 use **one Hungarian mapping fitted globally over the 44 evaluation recordings only**, separately per arm/seed; no per-recording remapping. Save mappings, confusion counts, per-class recall/support and segment counts. This is oracle evaluation mapping, not a learned semantic classifier.
- Secondary fitting scores, using a separately labelled global mapping fitted on the fitting population. Neither mapping is used to update any model.
- At the frozen endpoint, fitting and evaluation occupancy NLL and conditional NLL, with the same definition and weighting, evaluating every window once (not resampled training batches). A fitting/evaluation gap is evidence to inspect for overfitting, not a pure measurement of memorization: population composition can also affect it.
- Initial/final assignment agreement on both populations under their common state IDs. For evaluation, initial assignments are the frozen matched B0 decode.
- Assignment–recording, assignment–subject and assignment–action MI and arithmetic NMI, before and after learning, with contingency counts. This is a few lines on saved predictions and addresses the project's recurring identity-shortcut problem. Association does not by itself establish a harmful shortcut: action composition differs between recordings.
- Post-training order sensitivity: permute only each target's preceding history, keeping the target code and time position fixed, regenerate costs and decode the evaluation recordings. Use saved seed-111 permutations. Transport adjacency and labels remain unchanged, so this isolates corruption of predictive order evidence. It is distribution-shift sensitivity, complemented by the trained bag arm, not proof by itself.

### 8.3 Tier B: deferred to a later analysis script

Not implemented in this milestone. Each item must be computable from the Section 8.1 artifacts without retraining; items that decode do so with saved cost components and the existing solver, without reloading the model. Save per-window, per-action occupancy and conditional log-probabilities separately (not only their combined cost) so the state-averaged decode below needs no model.

- Cross-applied mapping scores: the fitting-derived mapping applied to both populations, to inspect the gap without changing the mapping.
- Posterior-weighted NLL and mixture NLL using a state prior estimated from fitting assignments only. Posterior-weighted NLL uses costs that helped choose the posterior, so it is not an independent predictive test.
- Same-recording nearest-neighbour preference on saved history representations, excluding self and any overlapping input supports, with one fixed query/reference sampling policy for all compared arms, reporting available neighbours and ties, and keeping fitting/evaluation separate. Not on C-state hard assignments.
- Adapter output differences and state-conditioned code/transition summaries.
- Post-training state-dependence: replace all saved conditional distributions with their arithmetic mean across actions while retaining occupancy, and decode. Do not claim that globally permuting all state labels should harm globally matched scores.

Provide `manifest.json`, `cells.csv`, `training.csv`, `transport.csv`, `gates.json`, `not_run.csv`, and per-cell artifacts. Failed, invalid, capped, complete and interrupted states must remain distinguishable. Write atomic outputs, reuse only verified completed cells and refuse incompatible source/config identities. Do not overwrite earlier runs or convert a partial run into a completed row.

## 9. Implementation structure and execution

Prefer a small package: `config.py`, `data.py`, `model.py`, `train.py`, `decode.py`, `evaluate.py`, `run.py`, focused tests, `README.md`, and `run.ps1`. Combine trivial modules if clearer. Do not build a generic experiment framework, copy a whole historical trainer, or add dependencies when PyTorch/NumPy and existing helpers suffice.

Reuse these entry points where their semantics match:

- `script/quantized_input_comparison/artifacts.py`: preparation access.
- `script/action_transport/continuous.py`: B0 costs/prototype fitting.
- `script/action_transport/transport.py`: solver and temporal kernel.
- `script/week5/timescale_inspection/run_phase7.py::fit_and_score_keep`: reference behavior for cached predictions/metrics; avoid constructing its full runner.
- Existing scoring helpers reached from those paths: global mapping and frame metrics.

Implement one PowerShell entry point with mutually exclusive `-Preflight`, `-Smoke`, `-DryRun` and `-Run` modes, plus `-RunId`, `-Arms`, `-Seeds`, `-Device`, `-Hours`, and `-ReuseCompleted`. Preflight must not fall through into training. Default invocation without an execution mode prints help. Use the existing `smq` environment or an explicit Python executable, not an implicit different interpreter.

Intended full-run command, to be supplied only after implementation checks:

```powershell
.\script\compositional_dynamics\run.ps1 -Run -RunId m1_lara_001 -Arms b0_centre,m1_ordered,m1_bag,occupancy -Seeds 111,222,1538574472 -Device cuda -Hours 10
```

Neural fitting may use the available GPU; transport remains CPU. Cap numerical-library threads at eight or available CPUs, whichever is smaller. Run expensive cells sequentially in fresh child processes, propagate nonzero exits, and close all archive handles. Enforce elapsed deadlines inside training and transport; save interrupted status and do not launch jobs projected to exceed the remaining limit. Estimate from actual smoke/reference timings, including every refresh over fitting recordings and final decoding of both populations. The ten-hour command is a cap, **not a runtime promise**; output a dry-run estimate before the user executes it.

Smoke uses a separately identified reduced population, updates and solver caps, marked `smoke=true`; it exercises every arm, final decoding, scoring and interrupted-cell handling without masquerading as a full result. Use seed 111. Bound smoke to approximately ten minutes. A historical B0 parity check necessarily uses the original population and is separate from this reduced smoke; report whether it was actually executed.

Tier B diagnostics (Section 8.3) are not part of this handoff; only verify that the artifacts they need are saved. At handoff provide: files created, passed/failed/unrun gates, smoke outcomes, measured runtime/memory and projected full cost, documented deviations, and the exact preflight/full-run/completed-cell-reuse commands. **Stop there.** No full training is authorized by this implementation assignment. The following gate decides whether learned units have earned the next investment.

## 10. Frozen feasibility gate

After implementation/smoke verification, the user runs the four arms on LARa at all three seeds. Compare paired differences within each seed, on the identical 44-recording evaluation population and primary final-solver endpoint.

| Comparison | Required mean difference across three seeds | Required seed consistency |
| --- | --- | --- |
| Ordered minus matched B0-centre | F1@50 >= +5 points; MoF >= -2 points | F1@50 difference strictly positive in all three seeds |
| Ordered minus bag | F1@50 >= +3 points; MoF >= -2 points | F1@50 difference strictly positive in all three seeds |
| Ordered minus occupancy | F1@50 >= +3 points; MoF >= -2 points | F1@50 difference strictly positive in all three seeds |

All requirements must pass. The MoF tolerance applies to each paired comparison's mean; it is not a per-seed improvement requirement. Report individual differences and population SD of paired differences. Three initialization runs are not confidence intervals or evidence of statistical significance.

Do not infer noise magnitude from the historical pooled B0 scores: matched held-out scores must be computed anew. Partial seeds, invalid cells or failed verification make the gate **inconclusive**, not passed. Capped but numerically valid, fully executed budgets may be scored, with their status visible.

If the complete, verified run fails the gate, stop progression to learned units under this hypothesis. Do not automatically launch more datasets, a larger model, extra histories or another tuning grid. A passed gate permits proposing the next milestone; it does not authorize that training or establish publication novelty.

The evaluation recordings are development data once this gate is inspected. Disclose that for any later confirmation. The ten-hour cap is a safety limit, not a runtime estimate; report measured per-cell/full-run projections in the dry run.

## 11. Secondary readouts for the overnight budget (v2.2)

These use spare compute and are **declared before any scored run**. They are descriptive: they cannot rescue a failed Section 10 gate and cannot overturn a passed one. If they disagree with the gate, report both and the disagreement.

1. **Subject-disjoint folds.** The 16 LARa subjects, parsed from file names, are shuffled with seed 111 and dealt round-robin into 4 folds of 4. For fold k, the 4 held-out subjects' recordings are evaluated and every other recording is the fitting population. Same four arms, same three seeds, same settings. One Hungarian mapping per fold, fitted over that fold's evaluation recordings. Report the mean over folds per arm and seed, and the Section 10 comparisons applied descriptively. The tokenizer is still fitted on all recordings, so this is not a fully inductive test.
2. **Final-decode multipliers 0.5 and 2** on the frozen scaled emission cost, primary split only, decode only (no retraining). Multiplier 1 is the primary result. These are sensitivity outputs, not candidates selected by score.
3. **Historical B0 parity** (Section 7) runs before the experiment in the overnight launcher; a mismatch is reported and does not block the run.

Execution order is primary split first (all seeds), then folds, so an overrun removes secondary cells first. The overnight launcher is `run_overnight.ps1`.

## Revision record

Version 2 adopts LARa-only fixed-split evaluation; removes B1, alternate initialization, scale sweeps and exact resume; uses window-equivalent smoothing; adds paired three-seed stopping thresholds, secondary fitting diagnostics and identity checks. Repository verification corrected the evaluator's split claim: the same seed-111 split is used by all three preparations.

Version 2.2 adds Section 11 (subject-disjoint folds, decode multipliers, overnight launcher); see [M1_AGENT_INSTRUCTIONS_CHANGES_v2.2.md](M1_AGENT_INSTRUCTIONS_CHANGES_v2.2.md).

Version 2.1 makes two changes, documented in [M1_AGENT_INSTRUCTIONS_CHANGES_v2.1.md](M1_AGENT_INSTRUCTIONS_CHANGES_v2.1.md). First, matched B0 is initialized only with the existing `initial_grouping` helper on fitting recordings, the lookup-row KMeans fallback is removed, and the code-level grouping is declared as the one exception to the continuous-feature rule. Second, diagnostics are split into Tier A (implemented now) and Tier B (deferred to a later analysis script over saved artifacts).
