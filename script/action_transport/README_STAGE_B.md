# Stage B: masked-code context (B-code)

Implements Stage B of `docs/plans/THREE_STAGE_MOTION_PLAN.md` v1.4 §4, and
the precommitted Stage B decision criterion added there before this ran.
Reuses Stage A's cached K=500 tokenizer; trains a small bidirectional
Transformer to predict masked codes from surrounding context. No action
label enters this stage anywhere -- see `tests_stageb.py`'s structural
check.

## Layout

```text
stageb_data.py    chunking (shared by train/infer), span masking, the frozen train/val split
stageb_model.py   the Transformer (embedding+mask token, sinusoidal positions, encoder, head)
stageb_train.py   the training loop, checkpointing, frequency/neighbor baselines
stageb_infer.py   full-sequence embedding extraction (chunk+stride, average, L2-normalize)
run_stage_b.py    CLI: --validate-only / --preflight / --dry-run / full run
tests_stageb.py   validation suite (11 tests)
run_stage_b.ps1   the launcher
```

## Why Stage B, and the bar it has to clear

Stage A's negative result (plan §3.5) was diagnosed as under-smoothing of
noisy single-window (0.25 s) evidence relative to a too-weak one-second
temporal term -- not evidence that codes lack action information. B-code
pools evidence over up to 128 codes (~32 s), which is the direct test of
that diagnosis. The plan now fixes, before any Stage B result exists, what
counts as clearing it (plan §4, "Precommitted decision criterion"):
contextual transport (Stage C) on B-code must beat both A-cont and SMQ on
F1@50, under T, on both datasets. A partial result (context alone removes
most of the fragmentation, without beating that bar) is reported as such,
not inflated.

## What's implemented and verified

- **v1.5:** max_epochs 50 -> 500 (50 was binding in every `stage_b_001` cell), a label-free sanity gate (validation masked accuracy must beat the neighbor-copy baseline before Stage C), and automatic masked-accuracy / stop-reason / embedding-smoothness / effective-rank logging in `results.json`. See plan §4.
- `-ValidateOnly`: **11/11 tests pass**, covering chunk coverage/anchoring,
  that chunks never cross a recording boundary and padding is never a real
  code, the mask's exact target count and its deterministic trim rule, the
  validation split (deterministic, disjoint, keeps >=1 of each), that
  padding is never masked/targeted, a structural check that no Stage B
  module imports the evaluator, hand-checked single-chunk embedding
  averaging, exact reproduction of embeddings after a checkpoint
  save/reload, and that training measurably reduces the fixed validation
  loss below its value before any update.
- `-Preflight`: passes on this machine (both provenance gates, all 12
  tokenizer caches present, CUDA available).
- **Real-data smoke** (not stubbed -- Stage B never touches labels, so
  there is nothing to stub): 20 recordings per dataset, 6 epochs, on GPU.
  Both datasets trained in ~1-1.5s, validation loss fell from ~5.6/5.2 to
  ~4.2/4.1 (nats; ln(500)=6.2 is the random-code-guess floor). The
  neighbor-repetition baseline reached 33-40% accuracy at masked positions
  on this tiny subset -- worth watching once the full run reports it, per
  the plan's own warning about detecting "an easy repetition task" (§4).
  Embeddings extracted for all 20 recordings with correct shape and unit
  norm; checkpoint reload reproduced them exactly.

## What changed in 1.6 (deep optimization study + external audit response)

`stage_b_002` (v1.5) stopped by patience in every cell but scored below the
copy-nearest-unmasked-code baseline in 11 of 12. An optimization study
(`stageb_optstudy.py`) found the cause and an external audit separately
found two implementation defects. All are fixed.

**The optimization finding.** Phase 0 showed the model memorizes fine
(>99% on 16 held-fixed chunks) but cannot learn to copy on synthetic data
where copying is near-optimal, even with a long budget and no early
stopping (Phase 0c, 15,000 steps): the unchanged model (sinusoidal
positions, post-LN, constant lr) reached 0.02 accuracy against a 0.47 copy
baseline. Positional inductive bias is implicated: with ALiBi (relative
position bias) instead of absolute sinusoidal positions, plus pre-LN,
weight tying and warmup+cosine decay, the same long run reached 0.40 against
0.47 -- close, on synthetic data where matching copy is the ceiling, not a
segmentation result. **A real-data sweep to select one configuration for
Stage B has not been run yet** -- see "Not covered" below.

**Two bugs found while building the study, both fixed and tested:**
- The "fixed" validation mask was seeded with Python's `hash()`, which is
  randomized per process (`PYTHONHASHSEED`). `stage_b_001` and `stage_b_002`
  were each internally consistent but not a strictly matched comparison to
  each other. `stable_seed` now uses CRC32; `test_stable_seed_is_process_independent`
  checks this across subprocesses.
- A custom ALiBi implementation was silently broken by torch 2.1's
  attention fast path in eval mode: the fast path drops a float attention
  bias, so eval-mode outputs were position-blind and produced NaNs, while
  train-mode outputs (which don't take the fast path with dropout>0) were
  fine. This is invisible at the small model sizes typical of a unit test.
  Fixed by routing through PyTorch's own reference attention computation
  when a bias is used. `test_train_and_eval_mode_compute_the_same_function`
  checks this at production size (d_model=128, 4 heads) for every
  position/norm/tying combination; the invalid first Phase 0c file is kept
  as `phase0c_INVALID_alibi_fastpath_bug.json`.

**Audit-confirmed defects, fixed:**
- Run identity didn't reflect the mask fix: `STAGEB_VERSION` stayed "1.5",
  and resume matched cells by key only, so old- and new-mask results could
  mix in one run undetected. `run_stage_b.py` now snapshots version,
  tokenizer version, the effective `TrainConfig`, and source file hashes
  into the manifest, and refuses to resume or reuse a run whose manifest
  identity differs from the current code.
- The optimization study's `--smoke` mode could write into the same
  `--study-id` as a full sweep; `phase1_sweep` only checked `(cell,
  config)`, not what that row was actually trained under. `--smoke` now
  forces a `_smoke`-suffixed, disjoint study directory, and every sweep row
  now carries a `config_hash` of its effective `TrainConfig`; resuming with
  a changed grid, changed source, or a stale/smoke row raises instead of
  silently accepting it. Per-fit learning curves are now saved under
  `curves/`, not just summary rows.
- The final, smaller minibatch of each epoch was silently dropped
  (`len // batch_size` floor division) -- seen in shuffling across epochs,
  but never trained in its own tail position. Fixed with `math.ceil`.
- Training-time validation ran as one unbatched forward pass over the
  whole validation set; now batched (`val_eval_batch`, default 256).
- The deadline was checked only between cells, so one very long training
  run couldn't be interrupted; `train()` now checks a cooperative deadline
  once per optimizer step.
- A docstring said the overfit diagnostic (Phase 0a) uses fixed training
  masks; training masks resample every step as in ordinary training --
  only the fixed 16-chunk evaluation set has a fixed mask. Corrected.

## Declared choices beyond the plan's numbers

- **Device defaults to CUDA when available** (`-Device auto`, the
  default), not Stage A's CPU-default convention: this is ordinary
  gradient-descent training, not the ASOT solver Stage A's instructions
  specifically constrained to CPU-by-default. Pass `-Device cpu` to force
  CPU.
- The training-chunk scheme is the same overlapping stride-64 length-128
  windows used for inference (the plan's own caveat -- "overlapping
  training chunks do not create independent observations" -- only makes
  sense if training already overlaps; a non-overlapping scheme was not
  specified separately).

## Running it

```powershell
.\script\action_transport\run_stage_b.ps1 -Preflight
.\script\action_transport\run_stage_b.ps1 -ValidateOnly
.\script\action_transport\run_stage_b.ps1 -DryRun
.\script\action_transport\run_stage_b.ps1 -Hours 4 -RunId stage_b_001
.\script\action_transport\run_stage_b.ps1 -Hours 2 -RunId stage_b_001 -Resume
```

12 models total (2 datasets x 2 normalizations x 3 seeds), each up to 50
epochs with patience 8. On the tiny smoke subset, 6 epochs on ~60-125
chunks took ~1-1.5s on this GPU; the full pooled populations are far
larger (HuGaDB ~76k tokens, LARa larger still), so budget accordingly --
`-DryRun` does not yet project a time estimate the way Stage A's does,
since no full-scale timing probe has been run.

## What a run produces (`results/action_transport_stage_b/<run_id>/`)

```text
manifest.json      versions, every launch (host, device)
splits/            the frozen train/val recording split per (dataset, normalize)
checkpoints/        one .pt per (dataset, normalize, seed): weights, config, training info
embeddings/<key>/   one .npz per recording: unit embedding [L,128] + zero_flags, for every
                    recording in the tokenizer's population -- this is Stage C's frozen input
results.json        best_epoch/best_val_loss/diagnostics/runtime per cell, updated as it runs
```

## Not covered

- No held-out subject-disjoint fitting fold trains a separate model yet;
  only the pooled population does. Subject-disjoint Stage B would repeat
  this per fold, per plan §4 ("repeat within each downstream subject-disjoint
  fitting fold") -- deferred, matching Stage A's decision not to run
  subject-disjoint (plan §3.5).
- Stage C (contextual transport on these embeddings) is not implemented
  yet; per the v1.6 plan correction, it is the actual test of whether
  B-code's context helps segmentation, not the copy-accuracy diagnostic.
- The plan's subject-association diagnostics (nearest-neighbor
  same-recording/same-subject rates) are not yet computed on the trained
  embeddings.
- **The Phase 1 real-data sweep (32 configurations x 4 cells) has not been
  run.** Phase 0/0c narrowed the factors worth testing on synthetic data
  (ALiBi, pre-LN, tying, warmup+cosine); Phase 1 is what selects one
  configuration on real motion codes, on a tuning split that excludes the
  official validation split. Launch with:
  `python -m script.action_transport.stageb_optstudy --study-id <id> --phase 1`
  (runs `--phase 0c` first if not already present for that `--study-id`).
  Expect roughly 2-3 hours for the full 128 runs at the current step
  budget (5000 steps/run, 4 cells).
- `stage_b_002`'s checkpoints/embeddings are unaffected by the v1.6 fixes
  (its masks were internally consistent within that run) but should not be
  treated as matched against a future run under a different `stageb_version`.
