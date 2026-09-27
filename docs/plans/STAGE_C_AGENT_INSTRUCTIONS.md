# Agent instructions — Stage B freeze (`stage_b_003`) and Stage C: contextual temporal transport (version 1.0)

Implement two things, in order:
- **Part 1:** freeze the Stage B contextualizer selected by `optstudy_002` as run `stage_b_003`.
- **Part 2:** implement Stage C of `THREE_STAGE_MOTION_PLAN.md` §5 on those frozen embeddings.

Deliver concise, readable code in the style of the existing Stage A/B modules, meaningful tests, a README and launchers. Run validation, preflight and tiny smoke checks only. **Do not launch `stage_b_003` or the Stage C grid: the user will run both.** These instructions, the plan (v1.6, §4–§6) and `STAGE_A_AGENT_INSTRUCTIONS.md` (v1.3; referred to below as "Stage A instructions") are the specification. Record any unresolved conflict rather than silently changing the method.

## 0. What this closes, and what must be fixed before any Stage C score exists

Stage C is the planned downstream test of whether B-code context helps segmentation (plan §4, v1.6 correction). It is scored against the precommitted criterion in plan §4. Three items must be recorded in the plan as a v1.7 note **before** the Stage C grid is launched. None of them may be revisited after a Stage C action score exists.

**0.1 The Phase 1 stop rule is superseded (a record of a decision already taken).** `optstudy_002/DESIGN.json` says "if no config beats copy on tuning accuracy in all cells, recommend stopping Stage B". No configuration did: LARa fell 0.5 and 1.3 points short of copy. Plan v1.6 had already made copy accuracy a diagnostic, not a Stage C gate. On 26 September 2026 the user decided to proceed to Stage C with the selected configuration. Record:
- the conflict;
- the decision;
- that it was taken on label-free evidence only.

**0.2 Operationalizing the primary bar (PROPOSED — the user confirms or amends before launch).** Plan §4 says contextual transport must beat A-cont and SMQ on F1@50, under T, on both datasets, pooled. It does not say how normalization conditions and seeds enter. Proposed reading, the strict one, consistent with "this plan does not average across datasets":
- **Unit.** One comparison per (dataset, normalization), under T, pooled. The statistic is the mean F1@50 over the three paired seeds.
- **vs A-cont.** Contextual ASOT's three-seed mean exceeds A-cont's three-seed mean (same dataset, normalization and seeds; taken from `stage_a_pooled_002`).
- **vs SMQ.** Contextual ASOT's three-seed mean exceeds the same-checkpoint SMQ F1@50 (HuGaDB 24.30, LARa 16.39).
- **Outcome.** The bar is met only if all **8** comparisons hold: 2 datasets × 2 normalizations × {A-cont, SMQ}. Report seed-level win counts beside every mean, and apply no significance test with n=3. If only one normalization condition clears, report that as a failure of the bar, stating which condition cleared.

**0.3 Operationalizing the partial-result clause (PROPOSED — same status).** "The context-only arm already removes most of Stage A's fragmentation" means:
- for `context_only` under T, in every (dataset, normalization);
- the three-seed mean of `pred_run_median_s / gt_run_median_s` (existing `segment_diagnostics` columns) lies in [0.5, 2].

This is reported as partial only if 0.2 fails.

## Part 1 — `stage_b_003`: freeze the selected contextualizer

### 1.1 Why a new run is needed

`optstudy_002` saved learning curves only, and trained on a tuning subset with seed 111 only. `stage_b_002` has checkpoints, but for the v1.5 sinusoidal configuration. `run_stage_b.py` always uses `TrainConfig()`. So no frozen selected contextualizer exists. `stage_b_001` and `stage_b_002` are **not** Stage C inputs and must not be substituted.

### 1.2 The frozen configuration (no changes)

Take exactly the `TrainConfig` that `stageb_optstudy.grid_configs()` produces for `postLN_cosine_wu500_lr0.0003_alibi`:

| Field | Value |
|---|---|
| Architecture | d_model 128, 4 layers, 4 heads, ff 512, dropout 0.1, context 128; post-LN, untied head, ALiBi positions |
| Optimizer | AdamW, lr 3e-4, wd 0.01, batch 64, clip 1.0 |
| Schedule | 500-step linear warmup, then cosine decay to step 5000 |
| Training length | step mode, `max_steps=5000`, `eval_every=100`, `patience_steps=1000` |
| Masking | 30% of valid positions, span 8 |

Its `config_hash` must equal **`0689bb38dfdd031b`**, the hash stored in `optstudy_002/sweep.csv` for every row of that configuration. The runner refuses to start if it does not.

**Do not raise `max_steps`**, even though 3 of the 4 tuning cells stopped at the step cap. The cosine horizon is part of the selected configuration. A longer budget is a different configuration and belongs to `STAGE_B_SCALING_CURVE_PLAN.md`. Record the stop reason per cell and report cap hits.

### 1.3 Data and splits

- **Same splits as `stage_b_002`.** Train on the official training recordings of `stage_b_002/splits/` and select checkpoints on its official validation recordings. That split was never used by the optimization study.
  - Copy the four split files into `stage_b_003/splits/`.
  - Assert that `build_validation_split` regenerates them identically.
  - The official training set includes the study's tuning recordings; say so in the README.
- **Paired seeds.** Seeds are 111, 222 and 1538574472, with the model seed paired to the tokenizer seed as in `stage_b_002`. This gives 12 models.
- **Tokenizer continuity gate.**
  - **Why:** Stage C reuses Stage A's grouping `g` and Stage A's predictions as paired controls. Tokenizer caches were deleted and rebuilt after the v1.6 cache-identity fix.
  - **Check:** for each of the 12 (dataset, normalization, seed) cells, recompute `g` and A-cont `mu0` from the current cache, exactly as `run.run_continuous_arm` does. Require `g` to equal the grouping saved in `stage_a_pooled_002/artifacts/*_continuous_asot.npz` exactly, and `mu0` to match to 1e-12.
  - **Already verified:** a check on 26 September 2026 found exact equality for all four seed-111 cells. Seeds 222 and 1538574472 are not rebuilt yet.
  - **On failure:** that cell's Stage A controls are not paired. Stop the affected cells and report; do not proceed silently.
  - **Record:** save a per-cell `codes_sha256` (over the concatenated `codes_fine`, in recording order) in the checkpoint's `extra` and in the manifest.

### 1.4 Code changes (`run_stage_b.py`, launcher, tests)

- **Select the configuration explicitly.** Add `--train-config {v1.5,selected}`. `selected` resolves the name from `optstudy_002/selection.json` through `grid_configs()` and checks the hash from §1.2. `v1.5` keeps today's default, so `stage_b_002` stays reproducible.
  - `stageb_optstudy` already imports `run_stage_b`, so avoid the circular import: move `GRID`, `STEP`, `grid_configs` and `config_hash` into a small side-effect-free module, or import lazily.
- **Make the effective config the identity.** `identity_snapshot()` must use the effective `TrainConfig`, not `TrainConfig()`, and must add `config_hash` and `selected_from` (study ID plus the `selection.json` sha256). Resume and mixing refusal then work unchanged.
- **Save step-mode fields.** `results.json` must also save `best_step`, `steps_run` (or the history length) and `config_hash`. In step mode, `best_epoch` alone is misleading.
- **Record what each embedding came from.** Write `embeddings/<cell>/INDEX.json`: the recording list in tokenizer order, L per recording, the checkpoint sha256, the tokenizer `cache_identity`, `codes_sha256`, the ALiBi/position mode and the `extract_embeddings` source hash.
- **Show the effective config.** `--dry-run` prints the effective config and its hash; `--preflight` checks that `optstudy_002/selection.json` exists and matches.
- **Subject-association diagnostics (plan §4, still not computed).** Add `--diagnostics --run-id stage_b_003`. It reads saved embeddings only and uses no action labels (subject IDs are metadata):
  - sample at most 10,000 complete windows uniformly with seed 111;
  - report cosine 10-NN same-recording and same-subject rates, before and after excluding same-recording candidates, with self-neighbors excluded;
  - report the sampling-composition reference rates;
  - write `subject_association.json`.

  It must be run before the Stage C report is written. It never selects anything.
- **Launcher.** `run_stage_b.ps1` gains `-TrainConfig selected|v1.5` (default `v1.5`) and `-Diagnostics`.
- **Tests (added to `tests_stageb.py`).**
  - `selected` resolves to hash `0689bb38dfdd031b`, and a tampered `selection.json` is refused.
  - Resuming a `v1.5` run with `selected`, or the reverse, is refused.
  - Reload inference with an ALiBi checkpoint reproduces saved embeddings exactly. This extends the existing sinusoidal test.
  - The tokenizer continuity gate passes on a synthetic case and raises on a one-code perturbation.
  - The NN diagnostic on a toy set gives hand-checked rates.
  - A `_smoke` run of the selected config through the real runner uses 20 recordings and 50 steps. Its step-mode keys are present and its manifest identity is correct.

### 1.5 Expected cost

Each tuning fit took 44–55 s on the GPU for up to 5000 steps. Official training sets are somewhat larger, but steps are capped. Allow about 1–2 min per model including embedding extraction, so about 15–30 min for all 12. `-Hours 1` leaves margin.

## Part 2 — Stage C: contextual temporal transport

### 2.1 Scope and file boundaries

- **New code** goes in `script/action_transport/`: suggested `contextual.py` (embedding loading, identity chain, order diagnostic), `run_stage_c.py` (CLI), `report_stage_c.py`, `tests_stagec.py`, `run_stage_c.ps1` and `README_STAGE_C.md`. **Outputs** go to `results/action_transport_stage_c/<run_id>/`.
- **Reuse, do not copy:**
  - `transport` (solver, kernel, objective);
  - `continuous` (`l2_normalize`, `build_mu0`, `cosine_cost`, `prototype_update`, `fit_continuous`);
  - `categorical.fit_alternating` / `infer_final`;
  - `smoothing.mode_filter`;
  - `permute.build_permutation`;
  - `evaluate` (the only label boundary);
  - `data` (provenance, loading, tokenizer cache);
  - the scoring and prediction-saving helpers in `run.py`, if importable without side effects (otherwise factor them out).
- **Stage A is not refit.** Its prediction sets and `cells.csv` rows are read from `stage_a_pooled_002` as the comparison arms.
- **Not allowed:**
  - training or fine-tuning the contextualizer (including on ASOT pseudo-labels);
  - a second contextualizer (not `stage_b_002`, not another configuration);
  - changing T/E coefficients, C, radius b, budgets or initialization rule;
  - soft-code inputs, extra grids or new temporal-null arms;
  - choosing anything from action scores.
- **Pooled protocol only.** Subject-disjoint Stage C requires per-fold contextualizers (plan §4). It is not run, matching Stage A's decision (plan §3.5), and must be listed as not run.

### 2.2 Inputs and the identity chain

For each (dataset, normalization, seed), build the fitting inputs as `continuous.ContinuousRecordingInput` with:
- **codes and lengths:** `codes`/`lengths` from the tokenizer cache, exactly as Stage A's A-cont used them;
- **embeddings:** `z` from `stage_b_003/embeddings/<cell>/<recording>.npz`, cast to float64 and re-L2-normalized with `continuous.l2_normalize`;
- **zero flags:** the logical OR of the stored `zero_flags` and the re-normalization flag.

Refuse to run a cell, recording it as `invalid_input` with a reason, unless all of these hold:
1. `stage_b_003/manifest.json` identity has `config_hash == 0689bb38dfdd031b` and `stageb_version` is current. Every checkpoint file's sha256 matches `INDEX.json`.
2. `INDEX.json` recording order and `codes_sha256` equal the current tokenizer cache's. Per recording, embedding rows equal L.
3. The tokenizer continuity gate (§1.3) passes for this cell.
4. The embedding `dim` is 128 and every non-zero row has a finite norm within 1e-5 of 1 before float64 re-normalization.

Record in the manifest:
- the `stage_b_003` run ID, identity, per-cell checkpoint and embedding-index hashes;
- the `stage_a_pooled_002` run ID and artifact hashes for the groupings used;
- tokenizer `cache_identity` and `codes_sha256`;
- the protocol version (introduce `STAGEC_VERSION = "1.0"`), source hashes, packages and device.

### 2.3 Model (plan §5 "Frozen-context model")

- **Cost.** `D[t,a] = 1 − z_t·mu_a`, via `continuous.cosine_cost`. It lies in [0,2] with no halving, and zero inputs cost 1 for every state.
- **Initialization.** `mu0[a] = normalize(Σ_{fitting t: g[code_t]=a} n_t z_t)`, with `g` the **Stage A grouping loaded from `stage_a_pooled_002`**, after the continuity gate has proved it equals the recomputed one. A zero initial mean makes the arm `unavailable`, with the reason recorded.
- **Updates and fitting.** Alternate transport with `continuous.prototype_update`. A zero later update keeps the previous prototype and is flagged `stale`. There is no prior term.
- **Transport and solving.** Everything else is identical to Stage A's A-cont:
  - frame masses, kernel V = L/b with `b=min(L−1,max(1,fps//W))`, and the marginal and entropy terms;
  - the log-domain solver, backtracking and the inexact schedule: 25 inner steps, 50 outer iterations, stop after 3 × relative change ≤1e-6;
  - fitting under T;
  - final re-solves under both T and E from the fitted prototypes: up to 500 steps, residual ≤1e-5, relative change ≤1e-7 for 5 steps, warm-started;
  - the status vocabulary (`converged`, `capped`, `invalid`, `unavailable`, `interrupted`, `unstarted`).
- **Device.** CPU by default, as in Stage A. CUDA is an explicit, verified option only.
- **Calibration.** Equal coefficients do not calibrate the 128-d contextual cosine cost the same way as the 64-d PCA cosine or the categorical cost; disclose this.

### 2.4 Arms and planned cells

Per (dataset, normalization, seed), pooled:

| Arm ID | Fitted? | Temporal term | Prediction sets |
|---|---|---|---|
| `context_only` | yes, β=0 (all other coefficients as the active setting) | none | T, E |
| `context_only_filter` | no, derived from `context_only` raw states with `smoothing.mode_filter`, half-width b, Stage A's tie rule | fixed 1-s mode filter | T, E |
| `contextual_asot` | yes | ASOT, β per setting | T, E |

The order diagnostic is `contextual_order_perm`, described in §2.5. It is pooled, seed 111 only, both datasets and normalizations, and yields T and E prediction sets.

Counts:
- **Fits:** 2 datasets × 2 normalizations × 3 seeds × 2 fitted arms = **24 fits**, plus **4 diagnostic fits = 28**.
- **Prediction sets:** 12 configurations × 6 = 72, plus 4 × 2 = 8, giving **80**.
- **Enumeration:** enumerate all 28 fits before execution, with distinct fit IDs and prediction-set IDs.

Order of execution:
1. `contextual_asot` T fits first, alternating datasets and completing three paired seeds per configuration;
2. then `context_only`;
3. then the four diagnostics;
4. E final re-solves are computed right after each fit, but T has priority under the deadline.

Never shrink the grid by score.

### 2.5 Contextual-order diagnostic (plan §5, paragraph 3)

For seed 111, each (dataset, normalization):
1. **Build the permutation.** Use `permute.build_permutation` with `default_rng(seed + 10000000)` over sorted recording names, exactly as in Stage A, keeping the terminal partial window in place.
   - Require the permutations to equal the `perm` arrays saved in `stage_a_pooled_002/artifacts/*_seed111_permuted_asot.npz`.
   - They are the same code sequences, so a mismatch is an implementation error.
2. **Extract embeddings from the permuted codes.** Feed `codes_perm = codes[perm]` through the **frozen `stage_b_003` checkpoint** with the unchanged `extract_embeddings`, which applies chunking, stride and ALiBi as usual.
3. **Restore window order.** `Z_restored[perm[j]] = Z_perm[j]`, which is equivalently `Z_restored = Z_perm[inverse_perm]`. Save both `Z_perm` and `Z_restored` with explicit `order` metadata under `artifacts/order_diagnostic/`.
4. **Refit on the restored embeddings.** Refit `contextual_asot` prototypes on `Z_restored`, using the original grouping `g`, frame spans and fitting population, with a freshly computed `mu0` from `Z_restored`.
5. **Solve and score in original order.** Run transport with the **original chronological kernel**, then score on original labels. Predictions are already in original order, so **do not apply the inverse permutation again**.

This changes only the order available to the contextualizer; transport adjacency is held fixed. Describe it in the report as sensitivity to disrupted context for a fixed contextualizer, an out-of-distribution intervention, not evidence of learned action grammar.

### 2.6 Evaluation and comparisons

- **Scoring.**
  - Score every Stage C prediction set with `evaluate.score_pooled` and `evaluate.segment_diagnostics`, using one dataset-level Hungarian mapping per prediction set.
  - Use Stage A's `cells.csv` rows for A0, A0+filter, A1, A1-KC, A-cont and SMQ.
  - Before using those rows, re-score one Stage A prediction set per (dataset, normalization) from its saved `.npz` and require the metrics to match `cells.csv` to 1e-9. This proves the evaluator and population are unchanged.
- **Report (`report_stage_c.py`, from artifacts only).** Sections in this order:
  1. **Precommitted criterion first.** The §0.2 table: 8 comparisons, the three-seed means, seed wins, met or not met. Then the §0.3 partial-result table. The overall verdict line uses exactly one of three phrasings from plan §4: primary bar met / partial (context, not transport) / neither.
  2. **The §5 comparison matrix** (representation × temporal treatment), five metrics, T and E in separate tables, both normalizations, three-seed mean ± population SD.
  3. **Paired seed differences (F1@50, T).**
     - `contextual_asot` − `context_only`: adding transport to context.
     - `contextual_asot` − `context_only_filter`: transport versus the fixed smoother on context.
     - `context_only` − A0: adding context without a temporal term.
     - `contextual_asot` − A1 and `contextual_asot` − A-cont.
  4. **Fragmentation diagnostics per arm:** run ratio, predicted and GT median run durations, p10/p90, occupied states, max state fraction, and MI(state; subject).
  5. **The order-diagnostic table:** original versus restored-shuffled context, seed 111.
  6. **Convergence and coverage:** fit and final status counts, residuals, outer iterations, stale-prototype flags, runtimes and not-run cells.
  7. **The `stage_b_003` summary:** per-cell stop reason (patience versus step cap), masked accuracy versus copy, span-depth breakdown, effective rank, adjacent-cosine diagnostics, and the subject-association results from §1.4.
  8. **Limitations and interpretation boundaries (plan §5 "What each comparison can establish"):**
     - no K=C contextualizer, so beating A1-KC does not isolate vocabulary size;
     - transductive frozen encoder, offline normalization and inference;
     - uncalibrated cost scales;
     - pooled only; subject-disjoint not run;
     - single selected contextualizer, chosen by a label-free tuning criterion on one seed.
- **Fixed-selection timelines.** Timeline examples come from a seed-111 recording selection saved before any scoring, not chosen by score.
- **Framing.** Do not claim state of the art from unmatched paper tables.

### 2.7 Required tests (`tests_stagec.py`, run by `-ValidateOnly`)

1. **Identity chain.**
   - A `stage_b` run with the wrong `config_hash`, a checkpoint sha mismatch, a reordered `INDEX.json`, a `codes_sha256` mismatch, or an embedding with the wrong L is refused. Each case gets its own synthetic fixture.
   - The continuity gate raises on a perturbed grouping.
2. **Inputs.** Float64 re-normalization keeps zero flags. The cost is 0/1/2 for equal/orthogonal/opposite unit vectors and 1 for zero inputs. `mu0` matches a hand computation on a toy grouping. A zero initial mean gives `unavailable`.
3. **Arms.** `context_only` has β=0 in both T and E with the other coefficients unchanged. `context_only_filter` equals `mode_filter` applied to saved `context_only` raw states (hand-checked votes and ties). With T fixed, a prototype update does not increase the objective on 128-d inputs.
4. **Order diagnostic.**
   - The permutation reproduces Stage A's saved seed-111 `perm` for a real recording list (or a synthetic list with the same RNG convention, if caches are absent in CI).
   - An **identity contextualizer**, whose embedding depends only on the code, makes `Z_restored` equal the original embeddings exactly.
   - With a synthetic position-dependent contextualizer, restoration lands each row back at its original index.
   - A synthetic case where applying the inverse permutation a second time changes the metrics, proving predictions are not double-restored.
   - Transport in the diagnostic uses the original chronological kernel, which is checked by comparing its V support with the unpermuted case.
5. **Label separation.** Structural check: `contextual.py` and the fitting path never import `evaluate`. Fitting APIs take no labels.
6. **Stage A reuse.** Re-scoring a saved Stage A prediction set reproduces its `cells.csv` row (§2.6), on a tiny synthetic run directory in unit tests and on the real run in `-Preflight`.
7. **Operations.**
   - Resume refuses a changed `STAGEC_VERSION`, `stage_b` identity or Stage A run.
   - A simulated deadline marks cells as `interrupted`, not `capped`.
   - A `_smoke` run namespace stays disjoint from full runs.
   - Saved predictions reproduce all five metrics after reload.

A real-data smoke test (a few recordings per dataset, one fit per arm, with a stub scorer as in Stage A) may verify shapes and time fits for the dry-run projection. It must not evaluate action scores.

### 2.8 Artifacts

```text
results/action_transport_stage_c/<run_id>/
  manifest.json        identity chain (§2.2), grid, settings, device, launches, hours budget
  planned_cells.csv    28 fit IDs, 80 prediction-set IDs, intended comparisons
  cells.csv            Stage A columns + stage_b run/config hash; five metrics, status, residuals, diagnostics
  not_run.csv          unavailable / invalid_input / interrupted / unstarted, with reasons
  artifacts/           per fit: mu0, mu, grouping source hash, objective trace, stale flags
  artifacts/order_diagnostic/   perm, inverse_perm, Z_perm, Z_restored (order-tagged)
  predictions/         per prediction set: raw window states (original order), mapping, metadata
  REPORT.md, figures/
```

### 2.9 Launcher and runtime

`run_stage_c.ps1`:
- **Modes and flags:** `-Preflight`, `-ValidateOnly`, `-DryRun`, `-TimingProbe`, `-Report`, `-Hours`, `-RunId`, `-Resume`, `-StageBRun` (required) and `-StageARun` (required), with `-Device cpu|cuda` (default cpu).
- **Environment:** resolve the `smq` env as the Stage A launcher does, and set thread caps before Python starts.
- **Deadline:** reserve 30 min for reporting and use cooperative deadline checks. Reuse Stage A's projection-from-observed-timings logic.

The cost estimate uses Stage A pooled medians per fit on this machine, including final T/E solves:

| Dataset | A0 | A-cont |
|---|---|---|
| HuGaDB | 158 s | 355 s |
| LARa | 242 s | 574 s |

`context_only` and `contextual_asot` should cost about the same as A0 and A-cont, respectively. The 128-d cost adds little next to transport. So 24 fits take about 2.2 h, plus about 0.5 h for the diagnostics and their re-extraction, **about 3 h total**. Suggest `-Hours 5`.

Commands, which the user runs:

```powershell
# Part 1
.\script\action_transport\run_stage_b.ps1 -ValidateOnly
.\script\action_transport\run_stage_b.ps1 -DryRun -TrainConfig selected
.\script\action_transport\run_stage_b.ps1 -TrainConfig selected -Hours 1 -RunId stage_b_003
.\script\action_transport\run_stage_b.ps1 -Diagnostics -RunId stage_b_003
# Part 2 (after recording §0 in the plan)
.\script\action_transport\run_stage_c.ps1 -Preflight -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.\script\action_transport\run_stage_c.ps1 -ValidateOnly
.\script\action_transport\run_stage_c.ps1 -DryRun -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.\script\action_transport\run_stage_c.ps1 -StageBRun stage_b_003 -StageARun stage_a_pooled_002 -Hours 5 -RunId stage_c_pooled_001
.\script\action_transport\run_stage_c.ps1 -Report -RunId stage_c_pooled_001
```

## 3. Handoff report

Finish by reporting:
- files created or changed;
- test results for Parts 1 and 2;
- the continuity-gate outcome for all 12 cells, once the caches exist;
- smoke timings;
- remaining limitations;
- the exact launch commands;
- anything in this document that could not be satisfied, with the affected cells kept `unavailable`.
