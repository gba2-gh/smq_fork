# FSQ reconstruction pilot

Implements the **required experiment** in
`docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md` (Revision 3, §3): one HuGaDB
autoencoder seed, two quantizers (VQ512 vs FSQ512 with levels [8,8,8]), four
streams (native codes and K-means-512 codes from each encoder), three
downstream seeds, and both categorical and contextual ASOT.

## Files

- `quantizers.py` -- FSQ math (`FSQ`), the SMQ-compatible patch adapter
  (`PatchFSQAdapter`, `FSQSMQModel`), and the RNG-isolation / matched-init
  helpers (`isolated_rng`, `reseed_all`, `build_shared_backbone`,
  `build_vq_arm`, `build_fsq_arm`).
- `experiment.py` -- configuration (`PilotConfig`), the fixed-order batch
  generator, matched training (`train_one_arm`), gradient diagnostics,
  extraction (`extract_recording`/`extract_dataset`), K-means-512 fitting,
  and the end-to-end scheduler (`run_required_pilot`) plus the CLI.
- `evaluate.py` -- the log(512) categorical-OT cost/fit, the frame-weighted
  PCA-space per-code centers, both cumulative-rung final-solve ladders
  (categorical and contextual), and the fresh-contextualizer wrapper around
  `stageb_train.train`.
- `tests.py` -- the `-ValidateOnly` suite.
- `run.ps1` -- the one launcher (validate / smoke / run / resume).

## Launching

```powershell
.\script\fsq_reconstruction\run.ps1 -ValidateOnly
.\script\fsq_reconstruction\run.ps1 -Smoke
.\script\fsq_reconstruction\run.ps1 -Hours 30 -RunId fsq_pilot_001            # the user runs this
.\script\fsq_reconstruction\run.ps1 -Hours 30 -RunId fsq_pilot_001 -Resume     # if needed
```

`-Hours` must leave at least 3 hours of margin (§11 reserves the final 3
hours for verification/reporting); the cooperative deadline passed to
training and to every downstream job is `Hours - 3h`.

## What gets saved (`results/fsq_reconstruction/<RunId>/`)

`manifest.json`, `backbone_init_hash.json`, `checkpoints/{vq,fsq}/epoch-*.model`,
`kmeans_{vq,fsq}_meta.json`, `training.csv`, `gradient_diagnostics.csv`,
`occupancy.csv`, `reconstruction.csv`, `segmentation.csv` (appended
incrementally, one row per (stream, downstream seed, method, rung)),
`not_run.csv`, `artifacts/{stream}_s{seed}.npz` (fitted `theta`/`mu` and the
grouping used to initialize them).

## RNG matching (§4)

The only global-RNG consumer inside `SMQModel.forward` for either quantizer
is dropout in the encoder/decoder (`ms_tcn.py`'s `nn.Dropout()`, p=0.5).
`SkeletonMotionQuantizer`'s own randomness (`init_embed_`'s kaiming draw,
`expire_codes_`'s replacement sampling) is either run eagerly inside an
`isolated_rng` context before training starts, or already uses a local
`torch.Generator` seeded to 42 on every forward call (unmodified
`motion_quantizer.py` behavior) -- neither touches the shared stream.
`PatchFSQAdapter` makes no random draws during `forward` at all. So:
encoder/decoder weights start identical (copied from one canonical
`build_shared_backbone` construction), each arm's own quantizer is
initialized inside its own `isolated_rng` context, and `reseed_all` is
called once, identically, immediately before each arm's `train_one_arm`
call. Given the same fixed per-epoch batch order (`precompute_epoch_orders`,
independent of `batch_gen.py`'s import-time `random.seed(42)`), dropout
masks are therefore identical step-for-step between the two arms.

## Dead-code threshold (§4)

`vq_dead_code_threshold` defaults to **1**, not the historical published
value of 10. At K=512 with 15-frame HuGaDB patches, a batch of 8 recordings
has roughly 1,670 valid patches, i.e. about 3.3 patches/code under uniform
usage -- below a threshold of 10, which caused near-total codebook
replacement every step in an earlier draft of this pilot (see the plan's
Revision 2 review). Threshold 1 keeps the historical threshold-to-mean-
occupancy ratio (~0.24x the mean). `occupancy.csv`'s `below_threshold`
column is a diagnostic flag (`replacement_warn_fraction`, default 5%), not
a validity gate: inspect it before trusting a VQ result, but it is not
auto-tuned or used to reject a run.

## K=512, not K=500 (§7)

`categorical.py`'s `emission_cost`/`prior_penalty`/`fit_categorical` hardcode
`LOG_500 = math.log(500)` for the historical K=500 pipeline and must not be
edited. `evaluate.py` therefore reimplements only those two small cost/prior
functions parameterized by `log_k = math.log(512)`, reusing the already
K-agnostic engine underneath (`categorical.fit_alternating`,
`categorical.infer_final`, `categorical.emission_update`,
`categorical.build_grouping`, `categorical.build_theta0`) unmodified.

## Scope of this implementation

This implements §3's **required** experiment end to end: matched VQ512 vs
FSQ512 training, four streams, three downstream seeds, both OT methods, all
diagnostics in §8, and the §10 test suite. Three follow-up branches the plan
describes are **not** auto-scheduled by `run_required_pilot`, since the plan
itself says "the user launches the full experiment... this document is not
authorization to start training":

1. **LARa** (§9 priority 1) and **a second autoencoder seed 222** (§9
   priority 2/3) use the same `PilotConfig`/`run_required_pilot` machinery
   with `dataset="lara"` or `autoencoder_seed=222`; they are not implemented
   as a separate automated escalation path. Decide from `segmentation.csv`
   per §9's promising-signal rule, then launch the next stage as its own
   `-RunId`.
2. **The released-encoder reference** (§6 "Released-encoder reference") is
   not implemented in this pass. It requires reusing
   `script/exp2round/q1q2/core.py`'s cached SMQ latents (the same pipeline
   `script/action_transport/data.py` uses) rather than this pilot's own
   `extract_dataset`, and a fresh K-means-512 fit on those features.
3. **§9's fixed-priority auto-escalation** (LARa only if HuGaDB is
   promising, a second seed only if ambiguous, etc.) is not automated here.
   Each stage is a separate, auditable launch; nothing decides by itself
   whether to spend the next block of compute.

All three are straightforward extensions of the existing `PilotConfig` /
`run_stream_downstream` / `run_required_pilot` functions, not a different
architecture -- see the docstrings there.

## Verification performed before handoff

- 17/17 `tests.py` checks pass (`-ValidateOnly`): FSQ round-trip and
  saturation on all 512 grid points, straight-through gradient finiteness,
  RNG isolation exactness, matched-arm encoder/decoder equality with
  independent quantizer initialization, epoch-order determinism, the
  log(512) cost, extraction determinism and full frame coverage (including
  a non-multiple-of-window terminal patch), per-code PCA-center correctness
  on unoccupied codes, and a real identity check against `data.py` on
  HuGaDB.
- An end-to-end `-Smoke` run (tiny model, 2 epochs, K=32, one downstream
  seed, `final_rungs=(5,10)`, `contextualizer_max_steps=20`) against the
  real HuGaDB dataset -- see the handoff message for its outcome and timing.
- The full `-Hours 30` run has not been launched; that is the user's call
  per the plan's own instruction.
