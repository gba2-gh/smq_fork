# Agent instructions: contextualizer search calibration

Implement [revision 2 of the experimental design](CONTEXTUALIZER_SEARCH_CALIBRATION.md). Deliver concise, readable experiment code, focused verification and one PowerShell launch script. **Implement and verify; leave the full experiment for the user to launch.** No full-data anchor training or largest-model timing probe during implementation.

## 1. Scope and structure

- New code: `script/week5/contextualizer_calibration/`.
- New outputs: `results/week/contextualizer_calibration/<run_id>/`.
- Read-only scientific inputs: `results/week5/quantized_input_comparison/qic_v3/`, `results/quantized_input_comparison/task2/task2_v1/` and referenced pilot002 checkpoints.
- Preserve existing runs and reports. Reuse functions rather than copying the whole Task 2 runner. Small backward-compatible shared-helper extensions are allowed when necessary; preserve old defaults and test them.
- No tokenizer retraining, FSQ or LARa runs, new OT objective, future search execution, or action-label gradients.

Suggested modules: configuration/planning, training/checkpoints, evaluation, runner, reporting and tests. Combine small modules where clearer; do not build a generic experiment framework. Add a short README documenting provenance and commands.

Inspect/reuse `quantized_input_comparison/{artifacts,contextual,masking,ot_arms}.py`, `action_transport/{stageb_model,stageb_data,stageb_configs,transport}.py` and `fsq_reconstruction/evaluate.py`. Chunk length, mask radius and inference stride currently have default/fixed assumptions: trace every path rather than changing only one constant.

## 2. Configurations and enumeration

Use explicit typed effective configurations; no per-job global mutation. Enumerate **14 training jobs / at most 58 OT fits** by default. `--include-combined-target-match` adds one training job and up to two OT fits. The throughput probe is separately counted. Save the full plan before execution, including optional choices and checkpoint-role placeholders.

| Pipeline | Family / input arm | Guard radius |
|---|---|---:|
| Native VQ | `vq_native` / `context_quantized` | 14 |
| Primary raw | `raw_kmeans` / `context_id` | 0 |
| Raw reproduction only | `raw_kmeans` / `context_id` | 14 |

Implement the main four configurations and additional jobs exactly as listed in the plan. Stable job IDs must encode pipeline, context, span, guard, horizon, seed and target-count policy. The two 3k jobs use primary guards, seed 111 and a genuine 3k cosine horizon. Seed-222 references run 5k. Optional combined count matching must not be enabled implicitly.

Resolve and record all existing Task 2 settings: initialization, model width/layers/heads, FF dimension, ALiBi, post-LN, untied head, dropout, AdamW parameters, clipping, warmup, schedule floor, validation cadence and masks. Hash full effective settings rather than depending on a mutable config name. Check source manifests and checkpoint hashes.

## 3. Chunking, training and exposure

Keep the existing recording split without fallback reassignment. Explicitly pass training/inference length and stride: 128/64 or 64/32. Preserve final-anchored coverage and embedding aggregation. No hidden `MAX_CONTEXT` remaining in an altered path.

Match 8,192 nominal token positions per full optimizer update: batch 64×128 or 128×64. Preserve Task 2's epoch shuffling, batch-tail handling and initialization/RNG reset for anchors. Log smaller epoch tails. If accumulation is required for non-anchor jobs, accumulate frame-weighted loss numerators, normalize by the effective batch's total weight, clip once and update once. Do not call accumulated dropout numerically identical to a monolithic batch. Anchor batching remains historical unless a hardware limitation is documented as a reproduction limitation.

Empty-target batches do not advance the optimizer or schedule. Use a bounded failure policy for persistently empty data. Save attempted batches, accepted updates, valid/padded token counts, selected/eligible targets, real-frame weights, skipped chunks, realized spans, position coverage, timings and peak memory.

Pass guard radius into both training and validation mask construction. Preserve the historical mask sampler for this pilot, including its trimming behavior; audit actual masks rather than silently replacing the algorithm. The raw radius-zero change does not alter the frozen tokenizer cache.

### Matched-target-count sensitivity

Pair native-VQ long-mask training with the span-8 reference at the same context, seed, batch order and schedule. Generate the reference mask/count and long selected mask from isolated deterministic streams on each effective batch. Keep every selected long-mask input masked, but uniformly subsample its eligible supervised positions to the reference count. Subsampling must not consume model/dropout or batch-order RNG.

If the long mask supplies fewer eligible positions, resample its selected mask through a recorded independent retry stream, at most 100 attempts. If exact matching remains impossible, mark this sensitivity unavailable with the batch identity; do not lower the quota or remove guard protection silently. Mirror reference empty-batch skips. Record retries and disclose that feasibility conditioning changes the mask distribution. Count equality does not ensure identical terminal-window frame weights; report both.

Use ordinary fixed validation masks for this configuration, without target thinning. Save stepwise reference/achieved target counts. Optional combined sensitivity pairs with the short-context span-8 reference. It must not change either primary main-grid fit.

## 4. Checkpoint saving, schedules and replay

Run the designated horizon without patience termination. Validate every 100 updates and at the endpoint. Save fixed 1k/3k/5k checkpoints where applicable and a distinct checkpoint on every strict validation improvement. Preserve all best states needed for replay, with step, loss, architecture/input configuration and tensor identity.

Replay patience 1,000 over the saved history: strict improvement updates the best; earliest ties are retained; the first event satisfying `step - best_step >= 1000` is the simulated stop. Select the best checkpoint available then. If there is no stop, use the best through the endpoint. Save stopping and selected steps separately. Replay must not perturb training. It models stopping the original schedule, not rescheduling a shorter run.

Checkpoint roles are aliases. Deduplicate evaluation only when tensor identity, architecture, input/preprocessing, inference settings, grouping and solver config all agree. Equal losses, steps or filenames across runs are insufficient.

For exact training resume, save model, optimizer, scheduler, RNGs, epoch permutation/cursor, update count and selection history atomically. If exact intra-fit resume cannot be verified, restart that interrupted fit deterministically from initialization and disclose lost compute. Do not resume only weights and call the trajectory equivalent.

## 5. OT and reproduction gates

Reuse pooled full-recording OT fitting, groupings, prototype initialization, frame broadcasting, Hungarian mapping and original MoF/Edit/F1 scoring. Action labels enter scoring/diagnostics only, never training or loss-based checkpoint selection. Save embeddings/zero flags, initial/final prototypes, objective traces, statuses and raw window/frame predictions.

Use 10 states; coefficients a=0.7, beta=0.3, lambda=0.05, epsilon=0.07; 50 outer iterations and 25 inner steps with existing stopping rules. Preserve final rung boundaries `(500,1000,2000,4000)` for reproduction consistency. Display 500 and 4000 as primary endpoints. Rungs continue transport with fixed prototypes; do not refit per rung. Invalid outer/final solves invalidate the affected evaluation before scoring. Capped outcomes remain explicit budgeted results.

The four outer-sensitivity jobs use the final 5k checkpoint of reference/combined × both primary pipelines. Run from identical saved initialization with outer cap 150 for straightforward reproducibility; do not resume an incomplete solver state. These are four additional OT fits, not contextualizer fits.

At the start of user-launched full execution, run native-VQ reference and guarded raw reference. Their validation-selected, seed-111, rung-4000 F1@50 targets are **38.81** and **36.08**. Compare all five metrics, selected step, validation trajectory and available checkpoint/embedding identities against saved Task 2 artifacts. Additional checkpoint logging must leave RNG/updates unchanged.

Exact agreement at stored metric precision is the default gate. A mismatch blocks interpretation of dependent changed cells and must be saved with environment/numerical evidence. Do not automatically blame GPU nondeterminism, silently widen tolerances, or modify settings to recover the score. Match device class, batching and numerical modes where available; document reproducibility limitations honestly.

## 6. Future-search declaration and bounded probe

Provide a **metadata-only** generator for the plan's future search, not a training sweep. Verify 10,368 nominal definitions. Include the frozen reference plus the first 11 feasible unique candidates in a seed-111 permutation of the remaining definitions. Record rejection reasons and duplicate effective configurations.

Use physical durations, actual fps/window, ceiling conversion and explicit realized durations. For guarded isolated spans, at least `2*ceil(2*radius/window)+1` tokens are needed for an eligible interior target. Apply this conservative nominal rule and audit actual masks, truncated sequences and validation eligibility without action labels. Raw radius zero does not inherit guarded restrictions. Feasible spaces can differ by tokenizer; disclose this. No segmentation score may admit/reject a candidate.

Add an explicit `position_mode='none'` model path if missing. It must omit both sinusoidal additions and ALiBi bias. Preserve and regression-test old modes/defaults. Do not silently use ALiBi as the fallback for `none`.

Provide a separate `--probe` job: native-VQ vector input, width 256, eight layers, four heads, FF 1024, context 256, stride 128, effective batch 32, 100 updates. Record first-10 versus subsequent throughput, peak memory, one validation pass and embedding extraction throughput. No OT or performance ranking. A memory failure is recorded, not silently repaired by changing the candidate. An explicit accumulation probe may be a separate config. Short timing is an extrapolation, not a bound or convergence certificate.

## 7. Runner, deadlines and cache safety

Deliver `run_calibration.ps1` and a module CLI supporting preflight, dry-run, smoke, probe, run ID, hours, resume, optional combined-count control, CPU fallback and source-run IDs. Defaults point to qic_v3/task2_v1. Resolve the smq Python environment, propagate exit errors and cap numerical-library threads at min(8, available CPUs).

GPU training is allowed; OT stays CPU. Run expensive jobs sequentially. Full execution requires an explicit elapsed-time budget including validation and output writing. Reserve a documented final writing allowance. Measure reference runtime, update projections, and avoid starting jobs likely to cross the compute deadline. Save interrupted/unstarted reasons; do not silently shrink the matrix or replace hyperparameters.

Freeze implementation identity once per invocation, including shared dependencies, packages and device details. Check source changes at job boundaries. Never restamp old artifacts with current hashes. Resume identity includes source caches/checkpoints, effective configs, splits and optional-job settings. Reuse requires artifact presence, finite metrics, expected checkpoint/rung coverage and exact recording/frame/window coverage. Remove stale completion/aggregate rows before retrying failed verification; prevent duplicate rows.

## 8. Verification and permitted work

Run focused tests. A separately identified synthetic or cached-slice smoke of **at most 10 minutes** is permitted. It must not be presented as either full-data anchor or scientific evidence. Do not launch the full run, anchor training or large-model probe during implementation.

Verify:

1. Default and optional planned counts (14/58 and 15/60) before checkpoint deduplication; probe separate.
2. Default model/mask/chunk/init parity with Task 2 on fixed small inputs; checkpoint logging leaves RNG/updates unchanged.
3. Real 3k/5k schedule horizons, accepted-update counting, weighted accumulation and empty-target behavior.
4. Radius-zero/radius-14 support, span eligibility, matched counts/retries and isolated subsampling RNG.
5. Train/inference stride, end anchoring, padding and full real-frame coverage.
6. Replay histories with early stops, later improvements, ties and no stop; each selected checkpoint exists.
7. No-position behavior and regression tests for ALiBi/sinusoidal paths.
8. Cache identity/deduplication, interruption/resume and invalid-solver rejection.
9. Label-free search enumeration, duration conversion/deduplication and logged feasibility rejections.
10. Bounded-smoke prediction rescoring matches saved metrics and incomplete jobs remain explicit.

Record exact checks and outcomes. Full-data reproduction acceptance remains a first step of the user's execution, not something inferred from synthetic tests.

## 9. Outputs and final handover

Incrementally save `manifest.json`, `planned_training.csv`, `planned_evaluations.csv`, `training.csv`, `checkpoint_policies.csv`, `segmentation.csv`, `exposure.csv`, `reproduction.csv`, `runtime.csv`, `not_run.csv`, logs and fitted artifacts. Save the future-search declaration/rejections without launching it.

Write a Markdown report from saved results distinguishing main configurations, target-count sensitivity, genuine schedule comparisons, policy replay, reference-seed replication and solver-budget effects. Cite manifests/results and disclose missing cells, pooling, unequal seed coverage and capped fits. Do not modify the original Task 2 report or claim tokenizer superiority from calibration.

Deliver implementation files, tests, smoke results, limitations and **one exact PowerShell command for the user to start the full run**, with a new run ID, explicit hours and optional-control choice. Also document separate preflight/dry-run/probe commands. Do not start the full experiment yourself.
