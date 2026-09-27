# Agent instructions: overnight frozen-representation study (v1.0)

**Task.** Implement one runner that executes every job below and **only saves** outputs.

The runner must not produce any of these:
- a report or summary table;
- a verdict, or an evaluation of a threshold;
- metric printouts on the console. Log only job status and timing.

**Scope limits.**
- Nothing new is trained. The encoders, tokenizers and `stage_b_003` are frozen.
- Do not launch the full run: the user launches it. You run only `-ValidateOnly` and `-Smoke`.
- If this document conflicts with the code, record the conflict in `README_FROZEN_STUDY.md`; do not silently change the method.

## 1. Files to create

```text
script/action_transport/frozen_study.py        # everything: jobs, workers, scheduler, saving
script/action_transport/tests_frozen_study.py   # -ValidateOnly suite
script/action_transport/run_frozen_study.ps1    # the ONE launcher (validate / smoke / full)
script/action_transport/README_FROZEN_STUDY.md  # how to launch; what gets saved
```

**Reuse existing code; do not copy it.**

| Module | Functions to reuse |
|---|---|
| `continuous` | `l2_normalize`, `build_mu0`, `cosine_cost`, `prototype_update`, `fit_continuous`, `ContinuousRecordingInput` |
| `categorical` | `recording_tensors` |
| `transport` | `solve_final`, `cold_start_log`, `objective_log`, `Coeffs` |
| `contextual` | `load_stage_b_cell`, `load_embeddings`, `context_inputs`, `verify_stage_b_manifest`, `verify_tokenizer_continuity`, `verify_stage_a_grouping` |
| `run` | `get_loaded`, `get_tokenizer`, `continuous_inputs` |
| `evaluate` | `broadcast_to_frames`, `score_pooled`, `segment_diagnostics` |
| `script.repro.d_error_decomposition` | `score` (identity-mapped scoring) |

`budget.check_deadline` provides the cooperative deadline.

## 2. Inputs and fail-fast identity checks

| Input | Path |
|---|---|
| Stage A PCA prototypes (A-cont) | `results/action_transport/stage_a_pooled_002/artifacts/{ds}_{norm}_pooled_pooled_seed{s}_continuous_asot.npz` (`mu`, `grouping`) |
| Stage C context prototypes | `results/action_transport_stage_c/stage_c_pooled_001/artifacts/{ds}_{norm}_pooled_pooled_seed{s}_contextual_asot.npz` (`mu`) |
| Contextual embeddings | `stage_b_003`, read via `contextual.load_stage_b_cell` / `load_embeddings` |
| PCA windows and codes | the tokenizer cache, via `run.get_tokenizer` |

`{ds}` is `hugadb` or `lara`, `{norm}` is `raw` or `norm`, and seeds are 111, 222 and 1538574472.

At startup, for all 12 (dataset, normalize, seed) cells, run these checks. Abort the whole run on any failure:
- `verify_stage_b_manifest`;
- `load_stage_b_cell`;
- `verify_tokenizer_continuity`;
- `verify_stage_a_grouping`;
- an assertion that `pca_windows` are identical across the three seed caches of each (dataset, normalize).

**Labels are 1-indexed.** Use GT label IDs directly as class IDs, and never index `mapping.txt` from 0.

## 3. Definitions (declared; do not change after scoring)

- **Window label:** the majority frame label within the window (`np.bincount(...).argmax()`; ties go to the lowest ID). Used only for oracle prototypes and the probe.
- **Representations:**
  - `pca`: `l2_normalize(pca_windows)`.
  - `pool_h{h}` for h ∈ {2, 8, 32, 64} windows, a centred box of 2h+1 windows (about 1, 4, 16 and 32 s). Within each recording: `pooled[t] = Σ_{|s−t|≤h} n_s·x[s] / Σ n_s`, where `x` is the raw `pca_windows` and `n_s` the window's frame count. Truncate at recording edges, include the terminal partial window with its weight, then apply `l2_normalize`.
  - `context`: `contextual.context_inputs` on the `stage_b_003` embeddings.
- **Prototype sources:**
  - `disc_saved`: saved `mu` (the Stage A A-cont `mu` for `pca`; the Stage C `mu` for `context`).
  - `disc_fit`: for `pool_*` and the extended fits. `mu0 = continuous.build_mu0(g, recs, C)` with the Stage A grouping `g`, then `fit_continuous` under setting T with `FIT_INNER_STEPS`, `FIT_OUTER_PATIENCE` and `FIT_OUTER_RELTOL`, and `outer_cap` = 50 (150 for extended fits). C is `config.num_actions`. A zero initial mean marks the job `unavailable`.
  - `oracle`: for each class present in the ground truth, `normalize(Σ n_t·z_t)` over windows carrying that label. C is the number of present classes. Labels enter **only** here and in the probe.
- **Ladder** (the only inference procedure used in this study; setting T only):
  - **Start and rungs:** cold start `transport.cold_start_log(p, q)`, then cumulative rungs at [500, 1000, 2000, 4000] total steps.
  - **Per recording:** keep `log_t` and call `solve_final(log_t, …, max_steps=rung − steps_done)` with the default tolerances.
  - **Convergence:** once a recording reports `converged`, freeze it; it is not re-solved at later rungs. The patience streak resets at each rung boundary; record this in the README.
  - **Deadline:** call `check_deadline` before each recording.
  - **Saved after each rung:**
    - `objective_total = Σ_r N_r·objective_log(...)`;
    - maximum residual and frame-weighted median residual;
    - counts of converged, capped and invalid recordings;
    - `frac_changed`, the frame-weighted fraction of windows whose argmax changed since the previous rung;
    - the five metrics under Hungarian mapping (`score_pooled`);
    - `segment_diagnostics`;
    - for `oracle` only, the five metrics under identity mapping (state index → present-class label ID, scored with `score`).
- **Probe** (one job per dataset and normalization, seed 111):
  - **Features:** every representation above.
  - **Folds:** four subject-grouped folds from `loaded.fold_of`.
  - **Readouts:**
    - `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)` with `sample_weight = n_t`;
    - a nearest-centroid readout whose centroids are built on the training folds.
  - **Training subsample:** at most 50,000 training windows per fold, drawn with `default_rng(111)`.
  - **Saved per fold:** frame-weighted accuracy and frame-weighted macro-F1.

## 4. Jobs (priority order; IDs are deterministic)

| Priority | Job ID pattern | Content | Count |
|---|---|---|---|
| P0 | `check_baselines` | Re-score saved T predictions (Stage A `continuous_asot`; Stage C `contextual_asot` and `context_only`; all seeds) and compare against their `cells.csv` rows. Save `baseline_check.csv`. **Abort if any metric differs by more than 1e-9.** | 1 |
| P0 | `pilot` | A 500-step cold T solve on the first 20 recordings of each dataset, using `pca` with `disc_saved`. Save `pilot.json` (seconds per recording-step per dataset). | 1 |
| P1 | `L_{ds}_{norm}_s111_{pca\|context}_{disc_saved\|oracle}` | Ladder | 16 |
| P2 | `F_{ds}_{norm}_s111_pool_h{h}` | `disc_fit` (outer cap 50), then ladder | 16 |
| P2 | `L_{ds}_{norm}_s111_pool_h{h}_oracle` | Ladder | 16 |
| P3 | `X_{ds}_raw_s111_{pca\|context}` | Extended `disc_fit` (outer cap 150), then ladder | 4 |
| P4 | `probe_{ds}_{norm}` | Probe | 4 |
| P5 | `L_{ds}_{norm}_s{222\|1538574472}_{pca\|context}_disc_saved`, `L_…_context_oracle` | Ladder | 24 |
| P5 | `F_{ds}_{norm}_s{222\|1538574472}_pool_h{32\|64}` | `disc_fit` (outer cap 50), then ladder | 16 |

Oracle prototypes on `pca` and `pool_*` do not depend on the seed, because the PCA windows are identical across seed caches (asserted in §2). They therefore run for seed 111 only.

## 5. Execution

- **Launch mode:** one command runs everything: identity checks → P0 → the queue from P1 to P5.
- **Workers:**
  - use a `ProcessPoolExecutor` with the `spawn` start method, `--workers` (default 6) and `--threads-per-worker` (default 5);
  - the launcher sets `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` to the threads-per-worker value before Python starts;
  - each worker's initializer calls `torch.set_num_threads`;
  - device: CPU.
- **Scheduling:** submit jobs in priority order, keeping at most `workers` jobs in flight. Before submitting a job, project its cost from `pilot.json` (steps × recordings × seconds per recording-step, plus fit time at 50 outer iterations × 25 inner steps). Skip the job if the projection exceeds the time remaining, and record it in `not_run.csv` with the reason.
- **Deadline:** `--hours H`. The hard stop is H − 0.5 h, passed to workers as a cooperative deadline.
  - A ladder interrupted mid-rung keeps its completed rungs and is marked `partial`.
  - An interrupted fit saves nothing and is marked `interrupted`.
  - The last 30 minutes are reserved for flushing files only. The runner does no reporting.
- **Resume:** `--resume` skips job IDs whose status is `complete`. Rerun `partial` jobs from scratch. Refuse to resume if `manifest.json` identity differs (source hashes, declared config, input run IDs).
- **Writes:**
  - workers return rows; only the parent appends to CSV files, followed by `fsync`;
  - the parent is also the only writer of `status.csv`;
  - arrays go to one npz per job, written atomically (temp file, then `os.replace`).

## 6. What gets saved (`results/frozen_study/<run_id>/`)

```text
manifest.json       declared config (§3 values, rungs, h list, probe settings), source sha256, input run IDs
                    (stage_a_pooled_002, stage_b_003 + config_hash, stage_c_pooled_001), host, packages,
                    workers/threads, hours, launches; the declared analysis rules below, copied verbatim
jobs.csv            every planned job: job_id, priority, dataset, normalize, seed, representation, h, prototype_source, outer_cap
status.csv          job_id, status (complete|partial|interrupted|unavailable|skipped|error), start, end, seconds, error
rows.csv            one row per (job, rung): job ids and params, rung_steps, objective_total, residual_max,
                    residual_median_w, n_converged, n_capped, n_invalid, frac_changed, MoF, Edit, F1@10/25/50
                    (Hungarian), id_MoF..id_F1@50 (oracle only), segment diagnostics
fits.csv            per fit job: outer_iterations, outer_status, last_rel_change, stale prototypes, runtime
probe.csv           per (dataset, normalize, representation, readout, fold): accuracy_w, macro_f1_w, n_train, n_test
baseline_check.csv  prediction_set_id, metric, saved, recomputed, abs_diff
not_run.csv         job_id, reason
pilot.json
artifacts/{job_id}.npz   prototypes used (mu, mu0 where applicable), fit objective trace, per-rung raw window
                         states (object array per recording, original order), recording names
logs/run.log        status and timing only
```

**Declared analysis rules.** Write these into `manifest.json` verbatim; **do not evaluate them in code**:
- Q1: pooling matches context if the best-h `pool` result (`disc_fit`, final rung) is within 3 F1@50 points of `context` (`disc_saved`, final rung) on HuGaDB, in both normalizations.
- Q2: the discovery gap is large if `oracle` minus `disc_*` is at least 10 F1@50 points at the final rung, for at least one representation, on both datasets.
- Q3: LARa stays poor with labels if `context` `oracle` is below `pca` `oracle` at the final rung, in both normalizations.

## 7. Tests (`-ValidateOnly`; all must pass)

1. **Pooling:** hand-checked weighted average including edge truncation and the terminal partial window; `h=0` equals `pca`; no averaging across recordings.
2. **Oracle prototypes:** hand-computed on a toy input; only present classes are used; 1-indexed label IDs are handled correctly; identity scoring maps state index → label ID.
3. **Ladder:** on a synthetic recording, cumulative rungs (30 + 30 steps) give the same `log_t` as a single 60-step `solve_final` (atol 1e-12, with tolerances set so no early stop occurs); converged recordings are not re-solved; `frac_changed` is 0 when states are unchanged.
4. **Scheduler:** job enumeration counts match §4 (98 including P0); IDs are deterministic; `--resume` skips `complete` jobs and reruns `partial` ones; a simulated deadline yields `partial` with its completed rungs saved.
5. **Probe:** training and test subjects are disjoint within every fold; the subsample is deterministic.
6. **Label separation:** a structural check that the `disc_fit` and ladder code paths receive no labels.
7. **Identity:** a tampered `stage_b_003` index or a mismatched grouping aborts the run.

**`-Smoke`:**
- run on 3 recordings per dataset, seed 111, with rungs [10, 20], `outer_cap` 2, h ∈ {2}, and 2 workers;
- run every job type once, writing to `results/frozen_study/_smoke/`;
- the smoke run must complete end to end, and all output files must exist with the schemas above.

## 8. Launcher and handoff

```powershell
.\script\action_transport\run_frozen_study.ps1 -ValidateOnly
.\script\action_transport\run_frozen_study.ps1 -Smoke
.\script\action_transport\run_frozen_study.ps1 -Hours 8 -RunId frozen_001            # user runs this
.\script\action_transport\run_frozen_study.ps1 -Hours 4 -RunId frozen_001 -Resume     # if needed
```

- **Launcher behaviour:** resolve the `smq` environment exactly as `run_stage_c.ps1` does, set the per-worker thread variables, and run from the repository root.
- **Guards:** `-Hours` must be greater than 0.5, and `-RunId` must not end in `_smoke` unless `-Smoke` is given.

**Handoff message.** List:
- the files created;
- test results;
- smoke runtime and pilot timings;
- the exact launch command.

Report no metric values: not from the smoke run, and not from anything else.
