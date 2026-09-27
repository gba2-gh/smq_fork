# `stage_b_003` and Stage C: contextual temporal transport

Implements `docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md` (version 1.0), which
covers two things:

1. **`stage_b_003`**: freezing the optimization study's selected (post-LN,
   cosine-warmup, ALiBi) configuration on the official split, all 12 cells
   -- an extension of `run_stage_b.py`, not a new module.
2. **Stage C**: contextual ASOT on those frozen embeddings, per
   `THREE_STAGE_MOTION_PLAN.md` §5. This is the actual downstream test of
   whether B-code's context helps segmentation; masked-code accuracy is a
   diagnostic, not this test (plan §4, version 1.6).

## Layout

```text
stageb_configs.py    GRID/BASELINE/config_hash/grid_configs, factored out of stageb_optstudy.py
                      so run_stage_b.py can resolve --train-config selected without a circular import
contextual.py         identity chain (stage_b_003 <-> stage_a_pooled_002 <-> current tokenizer cache),
                      context-embedding inputs, coeffs_for, the order-destruction diagnostic
run_stage_c.py        CLI: --validate-only / --preflight / --dry-run / full run
report_stage_c.py     REPORT.md from saved artifacts only (no refitting)
tests_stagec.py       validation suite (14 tests)
run_stage_c.ps1        the launcher (plus --report)
```

`run_stage_b.py` gained `--train-config {v1.5,selected}`, `--study-id`,
`--split-run`, and `--diagnostics` (subject-association NN diagnostics on
saved embeddings). `STAGEB_VERSION` is now `"1.7"`.

## Part 1: `stage_b_003`

`optstudy_002` selected `postLN_cosine_wu500_lr0.0003_alibi` (hash
`0689bb38dfdd031b`) but saved no checkpoints, and trained on a tuning subset
with seed 111 only. `stage_b_002` has checkpoints, but for the original
sinusoidal configuration. `stage_b_003` trains the selected configuration,
unchanged (including its 5000-step cosine horizon -- **do not raise
`max_steps`**, since the schedule length is part of the selected
configuration), on `stage_b_002`'s official train/validation split, for all
12 (dataset, normalize, seed) cells.

```powershell
.\script\action_transport\run_stage_b.ps1 -ValidateOnly
.\script\action_transport\run_stage_b.ps1 -Preflight -TrainConfig selected -SplitRun stage_b_002
.\script\action_transport\run_stage_b.ps1 -DryRun -TrainConfig selected
.\script\action_transport\run_stage_b.ps1 -TrainConfig selected -SplitRun stage_b_002 -Hours 1 -RunId stage_b_003
.\script\action_transport\run_stage_b.ps1 -Diagnostics -RunId stage_b_003
```

**Identity.** `--train-config selected` resolves the configuration by name
through `stageb_configs.grid_configs()` and checks its hash against the row
`optstudy_002/sweep.csv` actually recorded for it -- a changed grid or
`TrainConfig` default is refused, not silently retrained under a different
meaning of "selected". `run_one` now also writes `codes_sha256` (the exact
code sequences trained on) and `cache_identity` into each checkpoint's
`extra`, and an `embeddings/<cell>/INDEX.json` with the recording order,
per-recording lengths, checkpoint sha256, `config_hash`, and position mode
-- everything Stage C's identity chain checks before trusting a cell's
embeddings.

**Split reuse.** `--split-run stage_b_002` makes `get_or_build_split`
recompute the split (deterministic given the seed and recording list) and
then *check* it against `stage_b_002`'s saved split file, refusing if they
differ, rather than blindly copying. On this machine the recording lists are
identical, so this always succeeds; it exists to catch a future dataset or
split-code change.

**Verified before handoff:** a real (not stubbed) smoke pass of `run_one`
with the selected configuration truncated to 50-60 steps, on full HuGaDB,
training in under a minute on CPU. It produced a correct checkpoint,
`INDEX.json`, and per-recording embeddings, and `--preflight
-TrainConfig selected` correctly resolves and hash-checks the configuration.
23/23 Stage B tests pass, including the new resolve/tamper/split-reuse/
smoke/NN-diagnostic tests.

**Cost.** Each tuning fit took 44-55s for up to 5000 steps in the
optimization study; the official training sets are similar in scale.
Expect roughly 1-2 minutes per model including embedding extraction, so
15-30 minutes for all 12 cells -- `-Hours 1` leaves margin.

## Part 2: Stage C

Reuses (does not copy) Stage A's `transport`, `categorical.infer_final`,
`continuous` (cosine cost, `build_mu0`, `prototype_update`,
`fit_continuous`), `smoothing.mode_filter`, `permute`, `evaluate`, and
several of `run.py`'s generic per-cell helpers (`_base_row`,
`score_frames_and_row`, `_save_states`, `_final_stats`, `_set_status`).
Trains nothing; fits only Stage C's own cosine-cost prototypes.

### Arms and grid (pooled only; 28 fits, 80 prediction sets)

| Arm | Fitted? | beta | Derived from |
|---|---|---|---|
| `context_only` | yes | 0 | -- |
| `context_only_filter` | no | -- | `context_only`'s raw states, `smoothing.mode_filter` |
| `contextual_asot` | yes | setting's beta | -- |
| `contextual_order_perm` | yes (diagnostic) | setting's beta | permuted codes -> frozen `stage_b_003` checkpoint -> restored embeddings -> refit |

2 datasets x 2 normalizations x 3 seeds x 2 fitted arms = 24, plus 4
order-diagnostic fits (seed 111 only) = 28. Each fitted/diagnostic cell
yields a T and an E prediction set; `context_only` additionally yields a
filter prediction set per setting: 24x2 + 12x2 + 4x2 = 80.

### Identity chain (checked per cell before fitting; a failure marks the cell `invalid_input`, not silently scored)

1. `stage_b_003/manifest.json`'s `config_hash` equals the frozen selected
   hash.
2. The cell's `embeddings/<cell>/INDEX.json` config_hash matches, its
   checkpoint's actual sha256 matches the recorded one, and every listed
   recording's embedding file has shape `(L, 128)`.
3. The current tokenizer cache's `cache_identity` and recomputed
   `codes_sha256` match what `stage_b_003` recorded -- proves the codes
   Stage C reads are the exact codes the embeddings were extracted from.
4. The grouping `g` recomputed from the current tokenizer cache equals the
   grouping `stage_a_pooled_002`'s A-cont artifact actually saved -- proves
   Stage A's paired initialization is still paired.

### The order diagnostic

Builds the identical within-recording permutation Stage A's
`permuted_asot` used (checked against Stage A's saved `perm`, when present),
feeds the permuted codes through the frozen `stage_b_003` checkpoint's
unmodified chunking/inference path, restores embeddings to original window
order (`Z_restored[perm[j]] = Z_perm[j]`), refits contextual prototypes on
`Z_restored` with the original grouping and original chronological kernel,
then scores in original order with no second restoration.

### Running it

```powershell
.\script\action_transport\run_stage_c.ps1 -ValidateOnly
.\script\action_transport\run_stage_c.ps1 -Preflight -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.\script\action_transport\run_stage_c.ps1 -DryRun -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.\script\action_transport\run_stage_c.ps1 -StageBRun stage_b_003 -StageARun stage_a_pooled_002 -Hours 5 -RunId stage_c_pooled_001
.\script\action_transport\run_stage_c.ps1 -Report -RunId stage_c_pooled_001
```

**Cost.** Projected from Stage A's observed A0/A-cont medians on this
machine (`context_only` ~ A0's cost; `contextual_asot` ~ A-cont's cost):
about 2.2h for the 24 pooled fits plus a comparable amount for the 4
order-diagnostic fits (each pays a `contextual_asot`-sized fit plus one
CPU inference pass per recording through the frozen checkpoint) -- **about
3h total**. `-Hours 5` leaves margin.

### What a run produces (`results/action_transport_stage_c/<run_id>/`)

```text
manifest.json      identity chain summary, stage_b_run/stage_a_run, settings, device, launches
planned_cells.csv  28 fit IDs, 80 prediction-set IDs
cells.csv          same columns as Stage A's cells.csv (five metrics, status, residuals, diagnostics)
not_run.csv        unavailable / invalid_input / interrupted / unstarted, with reasons
artifacts/         per fit: mu, mu0, grouping, objective trace, stale-prototype flags (+ perm for the diagnostic)
predictions/       per prediction set: raw window states (original order), mapping, metadata
REPORT.md          written by --report
```

### Verified before handoff

A full pipeline smoke test (not stubbed) trained a truncated `stage_b_003`-
style checkpoint on real HuGaDB, then ran all three arm types
(`context_only` with its derived filter, `contextual_asot`, and
`contextual_order_perm`) against it end to end, including scoring one
resulting prediction set with the real evaluator. `--dry-run` and
`--preflight` were exercised against both the real `stage_a_pooled_002` and
a deliberately wrong `stage_b_run` (`stage_b_002`, correctly refused: wrong
`config_hash`). 14/14 Stage C tests pass, covering identity-chain rejection
(wrong hash, tampered checkpoint, shape mismatch), the cosine-cost
convention at 128-d, `beta=0` for `context_only`, a hand-computed `mu0`,
the zero-initial-mean unavailability rule, exact order-diagnostic
restoration (an identity-contextualizer round trip and a synthetic
non-identity permutation where skipping or double-applying restoration
changes the result), the unchanged-kernel argument for the order
diagnostic, the `contextual.py` import allow-list (no evaluator, no other
unexpected module), Stage A re-scoring reproducing its own `cells.csv` row,
manifest refusal on a different `stage_b_run`/`stage_a_run`, and the
28-fit/80-prediction-set enumeration.

## Not covered

- Subject-disjoint Stage C is not implemented, matching Stage A's decision
  (plan §3.5) and Stage B's (`README_STAGE_B.md`, "Not covered").
- The `--diagnostics` subject-association output is not consumed by Stage
  C's report; it is summarized separately in §7 of `REPORT.md` only as
  `stage_b_003`'s own masked-accuracy/copy table, not the NN rates.
- A real, full-scale `stage_b_003` and Stage C run have not been launched
  by this handoff -- the smoke tests above used truncated step budgets and
  small/synthetic fixtures. The user runs the real grids.
