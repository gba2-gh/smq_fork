# Time-scale inspection

Plan: `docs/plans/week5/TIMESCALE_INSPECTION/TIMESCALE_INSPECTION.md`.

## Phase 0 (implemented): label-only ceilings and diagnostics

Reads ground truth and the saved artifacts of `results/week5/raw_kmeans_cross_dataset/rkx_v2`
(preparation, selected checkpoints, embeddings, predictions). It never writes into that run,
and it does no OT decoding or training. Everything is a label-aware diagnostic; nothing selects a setting.

    python -m script.week5.timescale_inspection.run_phase0 --run-id ts0_v1

Options: `--datasets`, `--parts grid smooth profile probes masked`, `--source-run`, `--cpu`.

Output in `results/week5/timescale_inspection/<run_id>/phase0/`:

| file | content |
|---|---|
| `grid_ceilings.csv` | 0.1 perfect token labels on 4/8/15/30-frame grids (BABEL), 15/60 (HuGaDB), 12/50 (LARa) |
| `smoothing_ceilings.csv` | 0.2 mode filter on perfect token labels: fixed 1-4 tokens; 2/4/8% of recording length, with and without a 1-token floor |
| `profile_summary.csv`, `profile_per_class.csv`, `profile_error_types.csv`, `profile_tertiles.csv` | 0.3 failure profile of every verified decode |
| `probes.csv` | 0.4 linear and 1-NN probes on PCA / K-means centre / one-hot code / contextual embedding |
| `masked_loss_split.csv` | 0.4 validation masked loss split by whether the masked run crosses a GT boundary |
| `manifest.json`, `REPORT.md` | the source decodes used (a snapshot) and the tables |

Built-in consistency checks, reported in `REPORT.md`:
- F1@50 recomputed from saved predictions must equal the saved last-rung value.
- The recomputed validation loss must equal the selected step's saved validation loss.

Tests: `python -m script.week5.timescale_inspection.tests` (7 tests).

## Phase 1 (implemented): OT kernel radius, re-decoding saved checkpoints

No training. Reads the verified selected-checkpoint embeddings of `rkx_v2`, repeats the full contextual OT fit
(outer EM, then the rung ladder 500/1000/2000/4000) with only the kernel half-width changed, and scores with the
repository scorer. Settings (`radius.py`): `current` (fps // window), `fixed_0..4` tokens, `rel_2/4/8` (percent of the
recording's token count, floored at 1 token), `sec_0.25/0.5/1.0` (fixed in seconds). The solver's own function is
replaced inside a scope (shared modules are untouched, so earlier run identities stay valid) and every half-width it
returns is recorded per unit.

    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage screen -DryRun
    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage screen -Hours 4
    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage confirm beta -Hours 7

| stage | content |
|---|---|
| `gate` (runs first, automatically for screen/confirm/beta) | fresh decodes at the `current` radius on HuGaDB seed 111 and BABEL-2 seed 111 must equal the source run's ladders exactly (smoke on BABEL-2: max difference 0.0) |
| `screen` | BABEL-2 (development) and BABEL-3, 3 seeds, 12 settings |
| `confirm` | the best fixed-seconds and the best relative setting **chosen on BABEL-2 only** (written to `selection.json`), plus `current`, on BABEL-1, HuGaDB, LARa (global) |
| `beta` | BABEL-2, the two chosen radii, β in {0.1, 0.5} |
| `inputs` | no-contextualizer inputs at the current radius, every dataset (LARa both normalizations), 3 seeds: `pca` (PCA-64 feature, mean-subtracted, L2-normalized) and `centre` (K-means centre, same treatment). These are the repository's `direct_continuous` / `direct_quantized` arms. No checkpoint is involved. The context baseline is added as a free alias of the source decode. `--datasets` restricts it |
| `inputs_chosen` | the same two inputs at the two radii chosen on BABEL-2 (needs the finished screen) |

Settings that resolve to the same half-width on every recording of a dataset run the identical solve, so they are
decoded once (`aliased_of` in the unit record). `current` itself reuses the source run's decode once the gates pass.
Resume: re-run the same command; a unit is reused only when its identity (source decode, half-width signature, β,
implementation hash) matches. `--hours` is required; completed units are always kept.

The `inputs` stages have their own gate: `pca` on HuGaDB seed 111 at the current radius must equal Task 2's
`direct_continuous` decode of the same family (smoke: F1@50 3.97, max difference 0.0 over all rungs and metrics).
Planned counts (`-DryRun`): `inputs` is 54 units, 36 real decodes plus 18 free baselines.

Tests: `python -m script.week5.timescale_inspection.tests1` (10 tests).

## Phase 4 (implemented): input representation (velocity)

Builds a second raw-patch family, positions plus first differences (velocities, frame-rate scaled, zero at frame 0),
for BABEL-2, BABEL-3 and LARa (global), 3 seeds each, trains the frozen reference contextualizer on it exactly as
`raw_kmeans_cross_dataset` does, and decodes three ways at the current OT radius (Decision 1 is not finalized yet;
`--radius` can override it once it is): `context` (the trained embedding), `pca_velocity` and `centre_velocity`
(the same no-contextualizer decoder as Phase 1's `inputs` stage, built on the velocity feature / its K-means centre).

HuGaDB is not rebuilt -- it is already derivative (IMU) data and serves as its own control via `rkx_v2`. The
position-only baseline for BABEL/LARa is **not rebuilt** either: `context` is read from `rkx_v2`'s saved results and
`pca`/`centre` from Phase 1's `inputs` stage (`ts1_v1` by default), both at the same radius, so the comparison is
against the exact numbers already reported rather than a second, noisier vocabulary draw.

    .\script\week5\timescale_inspection\run_phase4.ps1 -RunId ts4_v1 -DryRun
    .\script\week5\timescale_inspection\run_phase4.ps1 -RunId ts4_v1 -Hours 6

`-Arms context pca_velocity centre_velocity` (default: all three) restricts which arms run; `-ContextRun` /
`-DirectRun` point the report at different baseline runs. 9 cells x 3 arms = 27 units. Resumable: a cell's training is
reused once its checkpoint hash verifies, and each arm's decode is reused once its identity (checkpoint, arm, radius,
implementation hash) matches.

A regression test checks that the velocity extractor with velocity switched off reproduces
`raw_kmeans_cross_dataset.prepare.extract_raw` exactly, so the only change from the position-only family is the
added channels.

Tests: `python -m script.week5.timescale_inspection.tests4` (7 tests).


## Phase 5 (implemented): the tokenizer, then span and window

Design, gates and reading rules: `docs/plans/week5/TIMESCALE_INSPECTION/PHASE5_TOKENIZER.md`. Phases 0-4 only ever
varied the contextualizer on one tokenizer; this phase varies the tokenizer (`tokenizers.py`):

| key | source | PCA | datasets |
|---|---|---|---|
| `t0_raw` | flattened raw patch (baseline; never rebuilt in a run, only for G1 and a T0 selection) | 64 | all |
| `t1_whiten` | raw patch, PCA **whitened**, K-means in the whitened space | 64 | BABEL-2/3, HuGaDB |
| `t2_relational` | per-window inter-joint distances: mean, sd, mean \|delta\| over the window's own frames | 64 | BABEL-2/3, LARa |
| `t3_lara_pos256` | LARa position channels (3-5) only | 256 | LARa |
| `t3b_lara_pos64` | LARa position channels only | 64 | LARa |
| `t4_smq` | SMQ's released encoder, pre-quantization latents (16 per joint) | 64 | all four |

Each tokenizer is its own family (`tok5_<key>`) in the unchanged qic artifact layout, so no shared module or earlier
run identity is touched. T4's `context` arm uses guard 14 / span 16 (the encoder's receptive field would otherwise
leak) and is compared only within T4; every other `context` arm is the frozen reference (guard 0, span 8).

    .\script\week5	imescale_inspectionun_phase5.ps1 -RunId ts5_v1 -DryRun
    .\script\week5	imescale_inspectionun_phase5.ps1 -RunId ts5_v1 -Hours 10
    .\script\week5	imescale_inspectionun_phase5.ps1 -RunId ts5_smoke -Smoke      # plumbing check, capped recordings

Stages (`stage_items` in `run_phase5.py`), run in order so a budget stop leaves every earlier stage complete: G gates,
A BABEL-2/3 (T1, T2, T4 x 3 seeds x context/centre/pca), B HuGaDB control (seed 111), C1 LARa (T3 with `pca`, T2, T4),
P1 probes, C2 LARa T3b, D span {2, 4} on the selected tokenizer, E window {4, 15}, P2 probes. The selection for D/E
is made on BABEL-2 only, among T0/T1/T2, and frozen in `selection.json` as soon as stage A completes. `-DryRun` prints the
unit list and the time projection (matches the plan: 2.5 / 3.0 / 7.9 / 9.6 / 10.4 / 11.6 h cumulative).

Gates (`gates5.py`, results in `gates.json`): **G1** the generalized preparation reproduces `rkx_v2`'s BABEL-2 seed-111
artifacts (codes, PCA features, grouping, split exactly; K-means centres to 1e-4, because the original preparation's
own centres differ by 7.6e-6 between two runs); **G2** SMQ's checkpoint loads verbatim and its own codes, scored by the
exact producer `script/repro/e0_metrics.py`, match `results/e0/{dataset}_pretrained.json` global within 0.005 (blocks T4 on
that dataset); **G3** LARa channels 3-5 have the larger across-joint extent (blocks T2/T3/T3b on LARa). A crashing gate
counts as failed.

Outputs under `results/week5/timescale_inspection/<run>/phase5/`: `prep/<tokenizer>/<cell>`, `jobs/`, `units/`,
`probes/` (per-token probes incl. a whitened 1-NN, and the T0 whitened-1-NN reference), `span_diag/` (Phase 0.4b's
masked-loss split per span), `gates.json`, `selection.json`, `REPORT.md` (H6 on F1@50 alone, probes as an
all/some/none mechanism label, H7, H2', T3 vs T3b, HuGaDB transfer check, SMQ reference).

Tests: `python -m script.week5.timescale_inspection.tests5` (15 tests; includes G1 on the real BABEL-2 data and SMQ
latent extraction on HuGaDB).
