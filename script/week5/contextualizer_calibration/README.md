# Contextualizer search calibration

Implements [revision 2 of the design](../../../docs/plans/week5/CONTEXTUALIZER_SEARCH_CALIBRATION/CONTEXTUALIZER_SEARCH_CALIBRATION.md)
following [the agent instructions](../../../docs/plans/week5/CONTEXTUALIZER_SEARCH_CALIBRATION/CONTEXTUALIZER_SEARCH_CALIBRATION_AGENT_INSTRUCTIONS.md).
It calibrates the contextualizer-search procedure (training budget, checkpoint
policy, masking, solver sensitivity, cost) before the later native-VQ
window / vocabulary comparison. It trains no tokenizer, runs no FSQ or LARa job,
adds no OT objective and executes none of the future search.

## Provenance and paths

* Inputs (read-only): the frozen `qic_v3` artifacts and the `task2_v1` run. They are looked up under
  `results/week5/quantized_input_comparison/` first and then `results/quantized_input_comparison/`;
  today they live at the second location. Pilot002 checkpoint hashes are re-verified against the qic manifest.
* Outputs: `results/week5/contextualizer_calibration/<run_id>/`. (The instructions' `results/week/...` reads as a
  typo for `week5`, matching `script/week5/` and the input path.)
* Reused unchanged: `quantized_input_comparison/{artifacts,contextual,masking,ot_arms,labels}.py`,
  `action_transport/{continuous,transport,stageb_*}.py`, `fsq_reconstruction/evaluate.py`.
* One shared-code extension: `action_transport/stageb_model.py` gains `position_mode="none"` (no sinusoidal
  addition, no ALiBi bias, never an ALiBi fallback). The sinusoidal and ALiBi paths are untouched and
  regression-tested against a from-scratch statement of the pre-`none` forward pass.

| Module | Role |
|---|---|
| `plan.py` | typed job / evaluation specs, the 14 + 58 (15 + 60) plan, stable ids, effective-settings identities |
| `train.py` | `train_core` (Task 2 trajectory, explicit context/stride/batch/guard/horizon, checkpoints, exposure, matched targets, accumulation), patience replay |
| `evaluate.py` | context/stride-aware embeddings, pooled OT fit + rung ladder, evaluation identities, Task 2 comparison |
| `search_space.py` | metadata-only declaration of the later search (10,368 definitions, 12 per tokenizer) |
| `run_calibration.py` / `.ps1` | CLI, schedule, deadline projection, resume, gate, aggregates, probe |
| `report.py` / `tests.py` | report from saved artifacts / 22 focused tests + smoke verification |

## What the run does

Sequential. Order: native-VQ reference, guarded raw reference, **reproduction gate**, remaining jobs (each followed by its
planned decodes), then the four outer-cap-150 solver fits.

* **Jobs (14; 15 with `-IncludeCombinedTargetMatch`):** two pipelines (native VQ quantized vectors, guard 14; raw K-means IDs,
  primary guard 0) × Reference / Short context / Long mask / Combined, seed 111; guarded raw reproduction reference;
  genuine 3,000-update-horizon references; seed-222 references; native-VQ matched-target long mask.
* **Planned OT fits (58; 60):** checkpoint policies per job (`ckpt_1k/3k/5k`, `selected` = full-history validation best,
  `replay` = patience-1000 replay, `final`) plus four outer-cap-150 fits. Aliased roles (e.g. `selected == ckpt_5k`) stay
  separate planned rows; they share one realized evaluation only when the identity (checkpoint tensors, architecture,
  input/preprocessing, context+stride, grouping, solver, recordings) is identical. Equal losses, steps or file names never merge.
* **Checkpoints:** one file per strict validation improvement, plus fixed-step and final checkpoints, each with a tensor
  hash. Replay is an offline pure function of the saved validation history: strict improvement, earliest ties,
  stop at the first evaluation with `step - best_step >= 1000`, else the best through the endpoint.
* **Reproduction gate:** the two references' validation-selected rung-4000 decodes are compared with the saved Task 2 cells
  (`vq_native__context_quantized__seed111` → 38.81, `raw_kmeans__context_id__seed111` → 36.08): all five metrics at every
  rung (exact), selected step, validation trajectory, checkpoint tensors, embeddings. The gate is exact agreement of
  metrics and selected step; trajectory/tensor/embedding differences are recorded for investigation. On failure the run
  stops before dependent cells (`--continue-on-repro-mismatch` overrides) and writes
  `reproduction_mismatch_environment.json`. Nothing is attributed to GPU nondeterminism without evidence, and no
  tolerance is widened.
* **Resume:** `--resume` refuses any change of sources, effective configs, split, optional-job flags or implementation.
  Exact intra-fit resume is **not** implemented: an interrupted fit is restarted deterministically from initialization,
  the lost compute is logged in `restarts.jsonl`. Completed fits/decodes are reverified (tensor hashes, finite metrics,
  rung/prediction coverage) before reuse; aggregates are rebuilt from artifacts, so no duplicate or stale rows survive.
* **Verification at every step:** aggregates include an evaluation only if its job, checkpoint tensors, identity and
  artifact contents (array keys, shapes, finiteness, integer prediction types, state ranges) verify now; a stale completion
  record is deleted. A training job that fails is not retried in the same invocation and its decodes are recorded as
  unavailable with the cause. A failed or invalid reference decode produces a failed gate, never a crash, and the run still
  writes its outputs. The resume identity includes the environment (device, CUDA/cuDNN, package versions, threads), so a
  CPU/GPU or PyTorch change is refused. Source tokenizer checkpoint hashes must match the qic manifest before any training.
* **Deadline:** a full run requires `-Hours`. The last 600 s are reserved for writing. Before each unit the runner projects
  its cost from measured per-update / per-decode times (first units unmeasured by design) and records
  `not_run.csv` reasons instead of starting a unit likely to cross the deadline. No cell is shrunk or substituted.

## Commands

All from the repository root.

```powershell
# focused tests (27), no data-heavy work
.\script\week5\contextualizer_calibration\run_calibration.ps1 -Preflight

# plan + search declaration + source/checkpoint checks, no training
.\script\week5\contextualizer_calibration\run_calibration.ps1 -DryRun -RunId ctx_cal_dry

# width-256 / 8-layer / context-256 memory + throughput probe (100 updates, no OT); not a convergence certificate
.\script\week5\contextualizer_calibration\run_calibration.ps1 -Probe -RunId ctx_cal_probe
#   optional accumulation variant:  -ProbeMicroChunks 8

# bounded smoke (12 recordings, 40 updates, 2 rungs): not scientific evidence
.\script\week5\contextualizer_calibration\run_calibration.ps1 -Smoke -RunId ctx_cal_smoke

# FULL RUN (default plan: 14 jobs / <=58 OT fits, optional combined control NOT included)
.\script\week5\contextualizer_calibration\run_calibration.ps1 -RunId ctx_cal_v1 -Hours 9

# same with the optional combined target-count control (15 jobs / <=60 OT fits)
.\script\week5\contextualizer_calibration\run_calibration.ps1 -RunId ctx_cal_v1 -Hours 9 -IncludeCombinedTargetMatch

# continue an interrupted / deadline-stopped run (same flags as the original launch)
.\script\week5\contextualizer_calibration\run_calibration.ps1 -RunId ctx_cal_v1 -Hours 9 -Resume
```

Other flags: `-QicRunId`, `-Task2RunId` (defaults `qic_v3`, `task2_v1`), `-Cpu`, `-ContinueOnReproMismatch`, `-PythonExe`.
Numerical-library threads are capped at `min(8, CPUs)`; OT stays on the CPU, training uses CUDA when present.

## Outputs per run

`manifest.json`, `planned_training.csv`, `planned_evaluations.csv`, `training.csv`, `checkpoint_policies.csv`,
`segmentation.csv`, `exposure.csv`, `reproduction.csv/json`, `runtime.csv`, `not_run.csv`, `search_declaration.json`,
`search_rejections.csv`, `REPORT.md`, plus `jobs/<job_id>/` (checkpoints, ledger, per-update exposure arrays,
mask/position audits), `evals/<identity>/` (embeddings, initial/final prototypes, objective traces, raw window/frame
predictions, ladder) and `eval_records/`. Disk: roughly 4 MB per saved checkpoint (one per strict validation
improvement), so expect on the order of 1–3 GB for the default plan.

## Verification performed during implementation

* `-Preflight`: **22/27 tests pass**, under both a bare `python -m` launch and the `.ps1` launch (thread count differs;
  two initially thread-fragile checks were rewritten to be environment-independent).
  Covered: planned counts 14/58 and 15/60; stable ids and identities; model/optimizer parity with Task 2; **trajectory
  parity with `train_vector_contextualizer`** (history, best step, best and last tensors identical, id and vector arms);
  checkpoint logging leaves training unchanged; real 3k/5k horizons and accepted-update counting; weighted
  accumulation; empty-target batches and bounded failure; radius 0/14 support, span eligibility (runs of 1/4 give 0
  targets, 5 gives 1, 8 gives 4), radius reaching both training and validation masks; matched counts, retry stream,
  explicit unavailability, isolated RNG, pairing with the span-8 job; stride / end anchoring / coverage and
  **embedding parity with `extract_vector_embeddings`** at (128, 64); replay (stop, later improvement, ties, no stop, best at
  step 4,400, boundary); `position_mode="none"` (permutation-equivariant, padding-safe, never ALiBi) and unchanged old modes;
  evaluation identities; interrupted-fit restart and explicit deadline; invalid solves never scored or marked done;
  search declaration (10,368 definitions, ceiling conversion, label-free); the reproduction gate run against Task 2's own
  saved cells (passes when identical, fails on a 0.01 metric or one-step change); the probe code path on a tiny config.
* Smoke `ctx_cal_smoke_final` (82 s, 12 recordings, 40 updates, rungs 20/40, outer caps 3/6): 14/14 jobs, 58/58 planned fits
  (34 distinct realized evaluations after exact alias de-duplication), 34/34 saved predictions rescored to their saved
  metrics, no unexplained unit. On an earlier smoke run of the same code path (since deleted): a resume reused every unit (13 s), and a launch with an
  already-expired budget produced explicit `not_run` reasons that a following `--resume` completed. The dry run found checkpoint hashes
  matching, both Task 2 reproduction cells present and a complete 12-candidate declaration for each tokenizer.
  The smoke directory is about 0.55 GB and can be deleted.

## Limits to keep in mind

* The 6.5–7 h figure in the design is an estimate from Task 2 file timestamps; the runner measures the first jobs and
  projects the rest, and `-Hours 9` leaves margin rather than certifying it.
* The reproduction gate is **not verified on full data**: it is the first thing the full run does.
* Task 2 did not record its device or package versions, so a mismatch cannot be diagnosed from artifacts alone;
  `reproduction_mismatch_environment.json` records the current environment for comparison.
* The synthetic and smoke runs are not anchor reproductions and not evidence about tokenizers.
* The search declaration shows feasibility, not that 5,000 updates train a given candidate adequately.
