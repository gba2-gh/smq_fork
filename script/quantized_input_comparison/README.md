# Quantized-input comparison — Task 1 (extraction, preprocessing, receptive-field masking)

Implements only the data-preparation layer for the quantized-input
comparison experiment: extraction of five input-representation "families"
from the frozen HuGaDB VQ/FSQ epoch-30 checkpoints
(`results/fsq_reconstruction/fsq_pilot_002/`), their shared preprocessing
(standardizer + PCA-64 + K-means-512), conditioning statistics, an initial
code-to-action grouping, and a reusable receptive-field-aware masking
eligibility helper. It does **not** train a contextualizer, run OT, or score
segmentation — that is Task 2/3/4.

## Files

- `config.py` — fixed scope (dataset=hugadb, window=15, K=512, PCA-64,
  seed=111 throughout, the verified receptive-field constants) and
  `PrepConfig`.
- `extract.py` — loads the frozen VQ/FSQ checkpoints (eval mode, gradients
  disabled, verified to load every tensor including EMA buffers verbatim)
  and runs the existing `extract_dataset`/`extract_recording`
  (`script/fsq_reconstruction/experiment.py`) and `extract_raw_dataset`
  (`script/fsq_reconstruction/encoder_sweep.py`) helpers unmodified.
- `fit.py` — shared-identifier sampling (<=10,000 complete windows, seed
  111, shared across the three preprocessing populations), per-population
  StandardScaler+PCA-64+KMeans-512 fitting, pooled-mean/RMS conditioning
  stats, the shared recording-level validation split
  (`stageb_data.build_validation_split`), and the initial grouping
  (`categorical.build_grouping` on frame-weighted per-code PCA means, via
  `evaluate.per_code_pca_centers`).
- `families.py` — assembles the five families' per-recording records and
  their fixed `[512, feature_dim]` quantized-vector lookup tables.
- `masking.py` — the reusable receptive-field eligibility helper
  (`eligible_loss_positions`, `build_eligible_mask`), built on the existing
  `stageb_data.build_mask` (30%/span-8).
- `artifacts.py` — manifest + per-recording array files (never one giant
  in-memory object) and the `Loader` class Task 2 should read through.
- `prepare.py` — CLI entrypoint (`--preflight` / `--smoke` / `--full` /
  `--resume`).
- `tests.py` — preflight unit tests (masking/receptive-field/FSQ math, no
  checkpoint needed) and `run_real_data_checks(run_dir)` (real-checkpoint,
  real-data verification, run automatically after `--smoke`).

## Post-review correction (qic_v2 supersedes qic_v1)

An independent review of the original `qic_v1` run found four real
problems, fixed here; see the "Verification outcomes" and "Artifact
interface for Task 2" sections below for the corrected behavior. Summary of
what changed and what the review got right/wrong, verified directly against
this codebase and `qic_v1`'s real saved arrays:

1. **RMS was a per-coordinate array, not the single shared scalar the spec
   requires.** Confirmed: `qic_v1/families/vq_native/rms.npy` is shape
   `(1440,)`, min 44.88, max 737.39 (matches the review's numbers almost
   exactly). `fit.conditioning_stats` computed `np.sqrt(np.mean(train**2,
   axis=0))` (per-coordinate). Fixed to a single scalar
   `sqrt(mean(x^2))` pooled over every coordinate of every complete
   training window, floored at 1.0 below 1e-8, shared by both the
   quantized and continuous inputs of a family. `qic_v1` is left as-is
   (do not use its `rms.npy` for anything); `qic_v2` has the corrected
   scalar (see per-family values below).
2. **`--resume` did not stop on a manifest mismatch.** Confirmed: the old
   `if resume and manifest_matches(...): ... elif ... and not resume: raise`
   had no `else` -- a resume whose config/checkpoints/data did not match an
   existing `manifest.json` fell through silently and went on to reuse
   whatever cached `.npz`/`sample_ids.json` files happened to exist. Fixed:
   `prepare.py` now raises `SystemExit` immediately whenever `--resume` is
   passed and an existing `manifest.json` does not match (verified by
   testing `--resume` with a changed `--recording-cap` against a live run:
   exit code 1, no writes). `manifest_matches` also now compares the exact
   recording-name set (not just config/checkpoint hashes), and a separate
   integrity check compares the loaded `sample_ids.json`'s hash against the
   prior manifest's recorded `sample_ids_hash` before reusing it, raising on
   mismatch.
3. **The README's receptive-field claim was wrong; the perturbation test
   was missing.** The old text asserted "the extraction pipeline never
   reads outside a window's own raw frames" -- false: `extract_recording`
   runs the whole recording through the dilated-conv encoder in one forward
   pass before chunking into patches, so a patch's `flat`/`ids` **does**
   depend on up to 14 frames on each side crossing into neighboring
   patches' raw support. That is exactly why `masking.py`'s
   receptive-field-aware eligibility helper exists at all -- masking would
   be unnecessary if patches were truly independent. Corrected below, and a
   real perturb-and-diff integration test
   (`tests._test_receptive_field_perturbation_real`, run automatically
   after `--smoke` and manually re-run against `qic_v2`) now perturbs the
   raw frames supporting a genuinely eligible masked target (found via
   `masking.eligible_loss_positions` on real window grids, not asserted by
   construction) and asserts every other window whose support is disjoint
   from the perturbed range is unchanged to float32 tolerance, while the
   perturbed target's own feature does change (positive control).
4. **Several tests checked weaker properties than claimed.** Confirmed:
   the old native-VQ check compared two lookups from the same saved table
   against each other (trivially self-consistent), not against the live
   model; the K-means check verified only `lookup.shape`. Fixed:
   `tests.run_real_data_checks` now diffs the saved `vq_native` lookup
   against `vq_model.vq._embedding` read directly from the live frozen
   checkpoint, diffs the saved `fsq_native` lookup against a fresh
   `out_proj` forward pass on the same sampled grid indices, and diffs
   every K-means family's saved lookup against its
   `preprocessing/<pop>_kmeans.npz`'s `cluster_centers_` values directly
   (`np.array_equal`, not shape-only).

Other deviation-table follow-ups fixed at the same time: `occupancy.npz`
now saves `window_counts` (unweighted, integer per-window occupancy) as a
separate array alongside the existing frame-weighted `counts`; `--smoke`'s
raw-feature extraction is now genuinely bounded (`extract_raw_dataset`
takes an explicit `names` subset instead of loading every recording under
`features_path` and subsetting afterward); and `fit.conditioning_stats`
now accumulates the pooled mean and RMS incrementally (one window's
`[d]`-shaped array at a time, float64 running sums) instead of stacking
every training-window coordinate into one big float64 array up front.

**`qic_v1` is preserved, unmodified, at
`results/quantized_input_comparison/qic_v1/`** -- its conditioning
statistics (the per-coordinate `rms.npy`) do not match the corrected spec
and it should not be read by Task 2.

## Second-round review correction (qic_v3 supersedes qic_v2 for --resume; qic_v2's data is still valid)

A second, independent review of `qic_v2` found the *data* correct but the
*cache/resume* machinery unsafe for a real resumable full run. Four issues,
all fixed here (see `artifacts.py`'s module docstring and `prepare.py`'s
module docstring for the resume-safety contract in full):

1. **Old, incompatible artifacts could pass `manifest_matches()`.**
   Confirmed: `qic_v1`'s (and `qic_v2`'s) manifest has no `schema_version`
   key at all, and neither recorded per-recording frame counts, so a
   changed data directory under an unchanged recording-name set went
   undetected. Fixed: `config.ARTIFACT_SCHEMA_VERSION` (currently `2`) is
   now written into every manifest and checked first in
   `manifest_matches()` -- a manifest missing this field (i.e. every
   `qic_v1`/`qic_v2` manifest) reads back as `schema_version: None`, which
   never equals `ARTIFACT_SCHEMA_VERSION`, so **both `qic_v1` and `qic_v2`
   now correctly fail `--resume` compatibility** (verified below). This
   does not mean `qic_v2`'s saved arrays are wrong -- it means a *new* run
   cannot silently treat them as a resumable cache without re-verification.
   `manifest_matches()` also now compares `recording_frame_counts` (a cheap
   `mmap`-based per-recording frame count, read directly off each raw
   feature file's shape before any extraction runs), not just the
   recording-name *set*, so a same-named-but-edited data directory is
   caught too.
2. **`--resume` without a manifest could still reach reuse code paths.**
   Confirmed: the old `if resume and manifest_path.exists(): ... elif ...
   and not resume: raise` had no branch for "`--resume` passed, manifest
   missing, but partial `preprocessing/`/`families/` files already exist"
   -- that case fell through both conditions and the population/sample-id
   reuse checks further down (keyed on bare `resume`, not on a verified
   match) could reuse those unverifiable files. Fixed: a single `can_reuse`
   boolean, true only after a verified matching manifest is confirmed, now
   gates every reuse decision in `prepare.py`; `--resume` against a
   `run_dir` with partial artifacts and no manifest is refused outright
   (`SystemExit`, verified below) rather than trusting them.
3. **`--resume` regenerated already-complete family artifacts, with
   arithmetic that drifted ~1e-5 from a fresh run.** Confirmed: the old
   per-family loop always called `build_family_records`/`build_lookup`
   regardless of `resume`, and the resume path's `_load_population`
   rebuilt scaler/PCA from saved arrays via a hand-rolled
   `(X - mean) @ components.T` affine map, which silently computed in a
   different floating-point precision than sklearn's in-place
   `X -= mean_; X /= scale_` (in-place ops downcast to the left operand's
   dtype; the manual map's out-of-place broadcast promotes to float64) --
   independently reproduced here as a `9.4e-6`-`1.1e-5` max abs diff on
   real `qic_smoke_v2`/`qic_v2` data, consistent with the reviewer's
   `9.54e-6`. Fixed two ways: (a) `artifacts.family_artifacts_complete`
   checks each family's on-disk artifacts (files present, `rms.npy` a
   scalar, `lookup.npy` shaped `[num_codes, *]`, every recording present)
   and `prepare.py` skips regeneration for any family that's already
   complete and valid, never overwriting it; (b) `_load_population` now
   reconstructs **real** `sklearn.StandardScaler`/`PCA`/`KMeans` objects
   (attributes assigned directly, not fit) instead of manual-map stand-ins,
   so a resumed population's `.transform()`/`.predict()` run the *exact
   same code path* as a freshly-fit population's -- fresh and resumed
   arithmetic are no longer two different implementations. Verified below:
   resumed-vs-sklearn transform diff is now exactly `0.000e+00`, and a
   full-dataset `--resume` against an already-complete `qic_v3` reproduces
   every family file byte-for-byte (`sha256` over all 1,857 files,
   confirmed identical).
4. **Extraction held all three feature populations in memory
   simultaneously.** Confirmed: `extract.extract_all` built and returned
   `vq_encoder`/`fsq_encoder`/`raw` together in one dict, and `prepare.py`
   kept all three resident for the whole run. Fixed:
   `extract.extract_population(pop_name, ...)` now extracts exactly one
   population; `prepare.py` processes "raw" (fit + build `raw_kmeans` +
   save), frees it (`del raw`), then "vq_encoder" (fit + build
   `vq_native`/`vq_kmeans` + save), frees it, then "fsq_encoder" the same
   way -- at most one population's full per-window array set is resident
   at a time, never all three.

**`qic_v3`** (same code path, same real HuGaDB data, same checkpoints as
`qic_v2`) is the run produced under these fixes and is what Task 2 should
use. `qic_v2` is left in place, untouched; its numerical artifacts are
still correct (this round of issues was about cache/resume robustness, not
`qic_v2`'s data), but **`qic_v2`'s own manifest now also fails the new,
stricter `--resume` compatibility check**, for the same structural reason
`qic_v1`'s does (no `schema_version`/`recording_frame_counts` fields,
since it predates this fix) -- `qic_v2` should be read directly via
`Loader` (which never calls `manifest_matches`), just not used as a
`--resume` cache target for a new run. All commands and verification
numbers in the rest of this document are from `qic_v2` unless marked
`qic_v3`; the two are numerically equivalent (same code, same data,
same checkpoints) and the interface shape below is unchanged.

### Second-round verification (real commands, real output)

```
PY=/c/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe
$PY -m script.quantized_input_comparison.prepare --preflight                       # 8/8 passed
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v3            # fresh, 50.5s, 364 recordings
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v1 --resume   # rejected, exit 1 (schema_version)
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v2 --resume   # rejected, exit 1 (schema_version)
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v3 --resume   # accepted, 0.7s, nothing regenerated
```

- **(a) Resume against `qic_v1`**: `SystemExit`, exit code 1 --
  `"...does not match the current schema version, config, checkpoints,
  recording set or per-recording frame counts; refusing to reuse
  mismatched preprocessing/family artifacts under --resume."` -- and no
  extraction ran (the schema-version/frame-count check happens before any
  heavy extraction).
- **(b) Resume against a `run_dir` with partial artifacts but no
  manifest** (simulated: a `preprocessing/sample_ids.json` written, no
  `manifest.json`): `SystemExit`, exit code 1 -- `"...contains
  preprocessing/family artifacts but no manifest.json -- most likely an
  earlier run crashed before completing... These artifacts cannot be
  verified, so --resume refuses to reuse them."` No file was read as
  reusable; `run_dir` was left with only the pre-existing file.
- **(c) Resume of a genuinely matching, complete `qic_v3`**: manifest
  verified, all 5 families reported `"artifacts complete and valid,
  skipping regeneration"`, and the whole run short-circuits before any
  extraction (`"all 5 families already complete and valid; nothing to
  (re)compute"`), completing in 0.7s vs. 50.5s fresh. `sha256sum` over
  every one of the 1,857 files under `results/quantized_input_comparison/
  qic_v3/` before and after this `--resume` is identical, byte-for-byte,
  for every file including `manifest.json`.
- Real-data checks (`tests.run_real_data_checks`) re-run against the full
  `qic_v3` output: all pass, including the rewritten resume-arithmetic
  regression test (test 11: `prepare._load_population`'s reconstructed
  `scaler`/`pca`/`kmeans` are asserted to be real `sklearn` instances, and
  its transform is diffed against an independently-reconstructed sklearn
  transform on the same saved parameters and the same input -- measured
  `0.000e+00`, down from the `~1e-5` hand-rolled-map drift) and the
  receptive-field perturbation check (target window support `[46,88]`,
  205 disjoint-support windows unchanged).

## Post-review correction (qic_v2 supersedes qic_v1)

## Commands actually run

```
# conda env "smq" python, no python on PATH — see script/fsq_reconstruction/run.ps1's Resolve-Python
PY=/c/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe

$PY -m script.quantized_input_comparison.prepare --preflight
$PY -m script.quantized_input_comparison.prepare --smoke --run-id qic_smoke_v2 --recording-cap 8
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v2
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v2 --resume   # verified idempotent
$PY -m script.quantized_input_comparison.prepare --full --run-id qic_v2           # (no --resume) correctly refuses
```

Preflight: 8/8 tests passed. Smoke (8 real HuGaDB recordings, real epoch-30
VQ/FSQ checkpoints): prepared in 8.2s, all real-data checks passed, including
the new receptive-field perturbation test (target window's support
`[46,88]`, 205 disjoint-support windows confirmed unchanged, target window
itself changed as expected) and the resume-transform-equivalence check
(max abs diff vs a from-scratch sklearn `transform` on the same saved
parameters, real data: `1.132e-05`). Full preparation (all 364 HuGaDB
recordings): prepared in 49.5s (resume: 26.1s, reusing the fitted
preprocessing rather than refitting, correctly detected as matching); a
plain re-run without `--resume` correctly refuses (exit code 1); a manual
`--resume` with a deliberately changed `--recording-cap` against the
existing `qic_smoke_v2` manifest correctly refuses with exit code 1
(`does not match the current config, checkpoints or recording set`) instead
of silently reusing mismatched artifacts. A follow-up
`tests.run_real_data_checks(...)` run against the full `qic_v2` output also
passed (resume-transform max abs diff on real data: `1.036e-05`; same
perturbation-test window/target). Results live under
`results/quantized_input_comparison/qic_smoke_v2/` and
`results/quantized_input_comparison/qic_v2/`. The original, now-superseded
`qic_smoke_test`/`qic_v1` runs (per-coordinate `rms.npy`, weaker tests) are
left in place, untouched, under their original directories.

Full preparation was launched directly (it completes in under a minute, well
inside the "launch it yourself if it's a few minutes" allowance) rather than
left for the user; `--resume` was verified idempotent on top of it, and a
plain re-run without `--resume` correctly refuses to overwrite an existing
`manifest.json`.

## Bugs found and fixed during the real-data smoke test

- `families.fsq_native_lookup` built its index tensor on CPU while the
  frozen FSQ checkpoint had been moved to CUDA (this machine has a GPU), so
  `out_proj` raised a device-mismatch error. Fixed by reading the device off
  `adapter.out_proj.weight` and placing both the index and grid-point
  tensors on it. This only surfaces when CUDA is available, i.e. exactly the
  class of bug the instructions warned "the code should work" claims miss.
- The masking preflight tests' first draft of
  `test_masking_eligible_when_span_masked_together` had the wrong expected
  eligible/ineligible indices: at window=15, radius=14, two windows must be
  >=3 apart in index (45 raw frames of start separation, since each
  window's support already spans 43 frames) before their supports stop
  overlapping — not 2 apart, which the first draft assumed. Recomputed by
  hand from `window_support` and corrected the test.

## Deviations from the spec's literal wording (and why)

- **Native families' own-PCA for grouping.** The spec says native families'
  grouping "use[s] PCA fitted on their own encoder patches." Rather than
  fitting a *fourth* PCA, this reuses the `vq_encoder`/`fsq_encoder`
  population PCA (already fit on that arm's own encoder patches) for the
  matching native family's grouping projection — those *are* "PCA fitted on
  their own encoder patches," and fitting a duplicate PCA on the same
  population would only add noise from a second random SVD draw with no
  spec-stated benefit. `fsq_kmeans`/`vq_kmeans` reuse the same PCA object for
  the identical reason (spec: "Fit separate standardizers and ... PCA-64
  ... for raw, VQ-encoder and FSQ-encoder features" — three populations, not
  five).
- **K-means resume reconstruction.** `prepare.py --resume` rebuilds a
  `FittedPopulation`-shaped object from the saved `.npz` arrays using
  `types.SimpleNamespace` stand-ins for `StandardScaler`/`PCA` (only
  `mean_`/`scale_`/`components_`/`transform` are exercised downstream) rather
  than re-pickling the full sklearn objects. This keeps the artifact format
  plain `.npz` (readable without unpickling untrusted objects) at the cost of
  not reloading `KMeans.predict`'s internals verbatim — `assign_kmeans`
  re-predicts from the saved `cluster_centers_` via `KMeans.predict`, which
  matches spec's "K-means lookups equal fitted centroids" requirement
  exactly (that's what "reload identically" tests check).
- **Occupancy/assignment counts.** The spec asks for "code occupancy and
  assignment counts" without a schema. Saved as
  `families/<family>/occupancy.npz` with `counts` (frame-weighted
  per-code assignment mass, from `per_code_pca_centers`), `window_counts`
  (unweighted, integer per-code window count — added after review: the two
  support different occupancy/usage statistics and neither substitutes for
  the other) and `n_occupied` (codes with counts>0, consistent between both
  arrays by construction).
- **`recording_cap` for `--smoke`.** Not in the spec's artifact interface;
  it is a `prepare.py` CLI-only knob (`PrepConfig.recording_cap`) that
  restricts extraction to the first N sorted recordings for a fast smoke
  run. It is never written into a full/`--resume` run's manifest as
  anything other than `null`, so it cannot silently narrow a full
  preparation.

## Artifact interface for Task 2

`from script.quantized_input_comparison.artifacts import Loader`

```python
loader = Loader(run_dir)                     # results/quantized_input_comparison/<run_id>
loader.families()                             # ["vq_native","fsq_native","vq_kmeans","fsq_kmeans","raw_kmeans"]
loader.recording_names()                      # all recordings extracted into this run
loader.frame_count(name)                      # int, real raw-frame count
loader.split_membership(name)                 # "train" | "val"
loader.split                                  # {"train":[...], "val":[...], "frame_counts":{name:int}}

loader.lookup(family)     # np.float32 [512, feature_dim]  -- fixed quantized-vector table
loader.mean(family)       # np.float32 [feature_dim]       -- pooled fitting-window mean (direct-OT centering)
loader.rms(family)        # np.float32 SCALAR (0-d array)  -- single train-split coordinate RMS, floor 1e-8->1,
                           #   shared by BOTH the quantized lookup vectors and the continuous features of this
                           #   family (contextual scaling). CHANGED from qic_v1: was mistakenly [feature_dim].
loader.grouping(family)   # np.int32   [512]                -- seed-111 initial code->action grouping
loader.occupancy(family)  # {"counts": float64[512] (frame-weighted), "window_counts": int64[512] (unweighted
                           #   window count, added after review), "n_occupied": int}

rec = loader.recording(family, name)
# rec["codes"]      int64   [P]   code id per window (native ids, or K-means assignment)
# rec["continuous"] float32 [P,d] d=1440 for *_native, d=64 for *_kmeans (already in the family's own PCA space)
# rec["starts"]     int32   [P]   raw-frame window start (non-overlapping, stride=window=15)
# rec["lengths"]    int32   [P]   real frame count per window (== 15 except a zero-padded terminal window)
```

`feature_dim` per family (verified against the real checkpoints:
`latent_dim=16, num_joints=6, num_person=1, window=15` -> native dim
`15*16*6*1 = 1440`, matching the spec's expected 1,440 coordinates):

| family | quantized/continuous dim |
|---|---|
| `vq_native` | 1440 |
| `fsq_native` | 1440 |
| `vq_kmeans` | 64 |
| `fsq_kmeans` | 64 |
| `raw_kmeans` | 64 |

`manifest.json` (`artifacts.read_manifest(run_dir)`): `config` (the
`PrepConfig` used), `checkpoint_sha256` (`{"vq":..., "fsq":...}`),
`sample_ids_hash`, `feature_dims`, `recording_names`, `num_windows` (per
family, per recording, window count), `receptive_radius`/`receptive_field`,
`native_dim_expected`, `families`. `manifest_matches(run_dir, prep_cfg,
checkpoint_hashes, recording_names)` is what `--resume` (and any Task 2
cache check) should call before reusing a run's preprocessing/families --
it now also compares the exact recording-name set, not just config and
checkpoint hashes, and `prepare.py --resume` raises `SystemExit` (stops
before any reuse or write) on any mismatch rather than warning and
continuing; it separately verifies the loaded `sample_ids.json`'s hash
against the prior manifest's `sample_ids_hash` before reusing the sampled
fitting windows.

**Receptive-field masking**, for Task 2's masked-code training loop:

```python
from script.quantized_input_comparison.masking import build_eligible_mask, eligible_loss_positions
selected, eligible = build_eligible_mask(starts, lengths, valid_len, frame_count, rng, radius=14)
# selected: True at every position build_mask chose (30%, span 8) -- mask ALL of these as input.
# eligible: subset of selected whose raw-frame support (clipped to [0, frame_count-1], radius 14)
#           is disjoint from every visible window's support in the same chunk -- compute loss
#           ONLY at these positions. eligible.sum() == 0 is a legitimate, expected outcome for a
#           heavily-edge-masked chunk; skip that chunk's loss entirely rather than treating it as
#           an error.
```

Verified: encoder receptive field is 29 raw frames (radius 14, from two
`MultiStageModel` stages x 3 `DilatedResidualLayer`s with dilations 1,2,4,
kernel 3 each: per-stage radius `1+2+4=7`, two stages -> `14`); a complete
15-frame patch depends on up to 43 raw frames (`15 + 2*14`). Both match the
spec's stated expectations and are asserted in
`tests.test_receptive_field_matches_architecture` against the actual
`num_layers` read from the source run's config (not hardcoded twice).

**Correction (post-review): extraction genuinely crosses patch boundaries.**
An earlier draft of this README claimed the extraction pipeline "never
reads outside a window's own raw frames," reasoning that indexing `flat`
per patch only looks at that patch's own slice. That is true of the
reshape/slice step but false of the pipeline as a whole:
`experiment.extract_recording` runs the *entire* recording's raw frames
through the dilated-conv encoder in a single forward pass (`model(x,
mask)` over the full `(1, C, T, V, M)` tensor) and only reshapes into
per-patch chunks afterward -- so a patch's pre-quantization feature
depends on up to 14 raw frames on each side of its own `[start,
start+15)` slice, i.e. it *can* and *does* read into a neighboring
patch's raw support. This is exactly the property `masking.py`'s
receptive-field-aware eligibility helper exists to guard against (it
would be pointless if patches truly never shared raw-frame support). This
is now checked directly, not just argued from code structure: see
`tests._test_receptive_field_perturbation_real`, run automatically after
`--smoke` and included in `run_real_data_checks`.

## Verification outcomes

Preflight (`tests.run_all()`, 8 tests, synthetic + pure-math, no checkpoint):
masking disjointness against a single visible neighbor, span-interior
eligibility vs. span-edge ineligibility (hand-verified against
`window_support`'s exact arithmetic), explicit zero-eligible-targets
result, recording-boundary support clipping, padded positions never
eligible, mask/eligibility reproducibility under a fixed seed, the
receptive-field constant computed from the real encoder architecture
(`num_layers=3`, dilations `1,2,4`, two stages -> radius 14, field 29, max
patch support 43), and the FSQ index<->code round trip. All passed.

Real-data (`tests.run_real_data_checks`, run automatically after `--smoke`
and again manually against the full `qic_v2` run):

- Frozen VQ model's full `state_dict` (weights + EMA buffers `cluster_size`/
  `embed_avg`/`_embedding`/`initted`) is bit-identical before and after an
  eval-mode, `no_grad` extraction forward pass.
- **Native VQ lookup diffed directly against the live model**: sampled code
  ids' rows in the saved `vq_native` lookup equal the corresponding rows of
  `vq_model.vq._embedding` read fresh off the frozen checkpoint (not merely
  two reads of the same saved table). Identical native VQ code ids
  occurring in real data also map to identical looked-up vectors.
- **Native FSQ lookup diffed directly against a live forward pass**:
  sampled grid indices' rows in the saved `fsq_native` lookup equal a fresh
  `adapter.out_proj(adapter.fsq.indexes_to_codes(idx))` forward pass on the
  same indices through the live frozen adapter.
- **K-means family lookups diffed value-for-value** (`np.array_equal`, not
  shape-only) against `preprocessing/<pop>_kmeans.npz`'s
  `cluster_centers_`, for all three K-means families.
- Native VQ and native FSQ lookup dimension is exactly 1440, as the spec
  expects.
- Every recording's window lengths sum to exactly its real frame count, and
  its terminal window start+window covers that frame count (terminal
  zero-padding handled correctly, not dropped).
- Train/val split is disjoint and exactly reproducible from
  `frame_counts` + seed 111 via `fit.shared_validation_split`.
- Cached lookup/grouping arrays reload byte-identical across two separate
  `Loader` instances.
- The first 200 saved fitting-sample identifiers are all complete
  (`length == window`) windows, confirming terminal/partial windows were
  excluded from the preprocessing sample.
- `rms.npy` is a scalar (`shape == ()`) for every family, not a
  per-coordinate array.
- `occupancy.npz`'s `counts` (frame-weighted) and `window_counts`
  (unweighted) agree on exactly which codes are occupied.
- **Resume-path transform equivalence, measured, not merely asserted**:
  for each population, a real extracted (pre-PCA) feature vector for one
  recording is pushed through (a) the true `sklearn.StandardScaler`/`PCA`
  reconstructed from the saved `mean_`/`scale_`/`components_` arrays and
  (b) the manual affine map `prepare._load_population`'s resume path
  actually uses at runtime; the max absolute difference is measured and
  asserted `< 1e-3`. Measured: `1.132e-05` (`qic_smoke_v2`, 8 recordings)
  and `1.036e-05` (`qic_v2`, all 364 recordings) — small but real,
  consistent with the reviewer's independently-measured `9.54e-6` PCA
  reproduction figure. **This is not exact equivalence** and the interface
  above does not claim it is; downstream code relying on bit-identical
  fresh-vs-resumed PCA coordinates should not assume that.
- **Receptive-field perturbation, against real checkpoints, not argued
  from code structure**: an eligible masked target is found on a real
  window grid via `masking.eligible_loss_positions` (not assumed to
  exist), its raw-frame support is perturbed with large noise, and the
  recording is re-extracted through the live frozen encoder. Every other
  window whose support is disjoint from the perturbed range is confirmed
  unchanged (`atol=1e-5`), and the perturbed target's own feature is
  confirmed to have changed (positive control, so the test cannot pass
  vacuously). Measured on `qic_smoke_v2`/`qic_v2`: target window support
  `[46, 88]`, 205 disjoint-support windows checked and unchanged.

## Still needed (Task 3 / 4, not implemented here)

Cross-family reporting/comparison beyond Task 2's own `segmentation.csv`/
`contextualizer.csv`.

# Task 2 — contextual training and OT decoding

Implements and verifies (preflight + a reduced-budget smoke matrix; the
full matrix was NOT launched -- see "Handover" below) training six arms per
family (`context_id`, `context_quantized`, `context_continuous`,
`direct_id`, `direct_quantized`, `direct_continuous`, plus
`context_geometry_init` for `vq_native`/`raw_kmeans` only) on top of Task
1's `qic_v3` artifacts, OT decoding, and segmentation scoring. Builds
directly on the "Artifact interface for Task 2" section above and on
`masking.build_eligible_mask`.

## Files

- `script/action_transport/stageb_model.py` (EXTENDED, not replaced) --
  `ModelConfig` gained `input_mode` ("id", unchanged default, vs "vector")
  and `input_dim`. `CodeContextModel.forward()` (the id path) is
  byte-identical to the pre-Task-2 model (verified: same hidden state AND
  logits, both position modes, train and eval mode, dropout RNG order
  included -- see `tests_task2.test_id_path_backward_compatible`). A new
  `forward_vectors()` method (shared backbone/head code via
  `_run_backbone`/`_head`) adds the vector-input path for
  `context_quantized`/`context_continuous`: a learned `nn.Linear` projection
  plus a learned mask token that completely replaces a masked position's
  projected input (spec: "Replace masked projected inputs completely with
  the learned mask token"). `context_id` and `context_geometry_init` both
  reuse the unmodified id path (`forward()`/`nn.Embedding`) -- architecturally
  identical trainable embedding tables, differing only in their initial
  weight values, so no new branch was needed for `context_geometry_init`.
- `script/quantized_input_comparison/contextual.py` -- matched
  initialization (`build_matched_init`/`build_arm_model`: one shared
  backbone/head/mask-token template and one shared projection template,
  each drawn once inside `isolated_rng` per family+seed, per spec "Copy
  identical backbone, output-head and mask-token initialization across
  matched arms" / "Copy identical projection initialization between the
  quantized and continuous arms"), the `context_geometry_init` construction
  (`E_c = W(q_c/RMS)+b`) and its rtol=1e-5/atol=1e-6 equivalence check
  against `context_quantized` (`verify_geometry_init_matches_quantized`),
  receptive-field-aware chunk/mask building (`make_vector_chunks`/
  `chunk_masks`, calling `masking.build_eligible_mask` directly -- not a
  reimplementation), the training loop (`train_vector_contextualizer`:
  AdamW lr=3e-4/wd=0.01/default betas+eps, batch 64, grad clip 1, cosine
  schedule with 500-step warmup over exactly 5,000 updates, validate every
  100, best-checkpoint-by-lowest-eligible-target-val-loss with earliest-tie
  selection, no early stopping in the main matrix (`patience_steps` set to
  effectively infinite) -- historical compatibility jobs pass
  `patience_steps=1000` separately, see below), real-frame-weighted
  cross-entropy at ELIGIBLE positions only (`masked_loss`, returns `None` --
  batch skipped, counted in `skipped_train_chunks` -- when a batch has zero
  eligible targets), and inference (`extract_vector_embeddings`: every
  window retained, overlapping length-128/stride-64 final-anchored chunks
  averaged then L2-normalized, mirroring
  `script.action_transport.stageb_infer.extract_embeddings` exactly for the
  vector path).
- `script/quantized_input_comparison/ot_arms.py` -- direct/contextual OT
  wiring reusing `script.action_transport.categorical`/`continuous`/
  `transport` and `script.fsq_reconstruction.evaluate`'s K-agnostic
  (`log_k`-parameterized) categorical helpers (K=512 here, never
  touching `categorical.py`'s hardcoded K=500 `LOG_500`). `direct_id`/
  `context_id` are categorical arms (theta over the 512 codes, `log(512)`
  cost, eta=60 pseudocount); `direct_quantized`/`direct_continuous` are
  continuous/cosine arms directly on Task 1's raw feature vectors (minus
  the shared pooled mean, L2-normalized); `context_quantized`/
  `context_continuous`/`context_geometry_init` are continuous/cosine arms on
  the matching contextualizer's already-L2-normalized embeddings. Task 1's
  grouping (`loader.grouping(family)`) is shared across every arm of a
  family (never re-derived); each arm computes its own initial prototypes
  (`theta0`/`mu0`) under that fixed grouping from its own features.
  Coefficients a=0.7/beta=0.3/lambda=0.05/eps=0.07 (== the existing
  `SETTING_T`/`FIT_SETTING`), 10 action states, fit budget 50 outer x 25
  inner steps / reltol 1e-6 / patience 3 (== the existing
  `FIT_OUTER_CAP`/`FIT_INNER_STEPS`/`FIT_OUTER_RELTOL`/`FIT_OUTER_PATIENCE`),
  final cold-start rung ladder 500/1000/2000/4000 via
  `fsq_reconstruction.evaluate.categorical_final_ladder`/
  `contextual_final_ladder` with strict invalid-solver rejection. CPU, thread-capped at 8
  (`set_cpu_thread_cap`).
- `script/quantized_input_comparison/labels.py` -- the only Task 2 module
  Task 2's OWN training/selection code ever imports labels from is
  `ot_arms.py`'s two scoring functions
  (`fit_and_score_categorical`/`score_continuous`), which import it
  LOCALLY (inside the function, not at module level) -- verified by a
  static AST check
  (`tests_task2.test_labels_module_not_imported_by_training_modules`) that
  `contextual.py` never imports it and that none of `ot_arms.py`'s fitting
  functions (`fit_categorical_k`, `fit_continuous_arm`,
  `theta0_from_grouping`, `mu0_from_grouping`, the `*_recordings*`
  builders) do either. Labels come from
  `script.exp2round.q1q2.core.load_dataset("hugadb", ...)`, the same
  source `script/action_transport/data.py` uses; verified the recording-name
  sets agree exactly with `qic_v3`'s (364/364, 0 missing, 0 extra).
- `script/quantized_input_comparison/historical_compat.py` -- the two
  seed-111 compatibility jobs (spec: "native-VQ IDs and raw-K-means IDs
  through historical helpers, using original masking, initialization and
  patience_steps=1000"). Deliberately bypasses `qic_v3`/`contextual.py`/
  `ot_arms.py` entirely and replays
  `script/fsq_reconstruction/encoder_sweep.py`'s OWN extraction
  (`extract_representation`, representations `"vq_e30"`/`"raw"`, the same
  epoch-30 VQ checkpoint Task 1 uses) and
  `script.fsq_reconstruction.experiment.run_stream_downstream` (the
  pre-Task-2 `CodeContextModel` id-only path, `stageb_data.build_mask`,
  `PilotConfig.contextualizer_patience_steps=1000` default) directly -- a
  genuine reproducibility check of the historical helpers, not a
  re-derivation through this experiment's new code. Diffs the resulting
  rows against `results/fsq_reconstruction/encoder_sweep/encoder_sweep_001/
  segmentation.csv` at seed 111 (`diff_against_historical`); per spec, no
  tolerance is invented -- only the measured MoF/Edit/F1@10/25/50
  differences are saved (`historical/diff.csv`), for the reader to judge.
- `script/quantized_input_comparison/run_task2.py` -- the CLI orchestrator
  (`--preflight`/`--smoke`/`--resume`/`--run-id`/`--qic-run-id`/`--hours`/
  `--cpu`). Predeclares every cell up front (25 contextualizer fits + 15
  direct-arm cells = 40 main-matrix segmentation evaluations, plus the 2
  historical jobs = 27 fits / 42 evaluations total, matching the spec's
  counts exactly -- verified by `enumerate_context_cells`/
  `enumerate_direct_cells`'s output size). Resume safety mirrors Task 1's
  pattern exactly: a single `manifest_matches` gate (schema_version +
  effective-config hash + a hash of the qic source run's OWN manifest, so a
  same-named-but-different `qic_v3` is caught) decides whether ANY cell
  output may be treated as reusable; a `run_dir` with cell artifacts but no
  `manifest.json` is refused outright, never trusted; completed cells are
  tracked in `done.csv` and skipped only when their per-cell config hash,
  rung metrics, prediction coverage, and required artifacts pass
  `cell_complete`. Invalid completed cells have stale ledger rows removed
  before retrying.
  `--smoke` only ever shrinks per-cell BUDGETS (`max_steps=40`,
  `eval_every=10`, `rungs=(20,40)`, `recording_cap=8`, native-VQ seeds
  collapsed to `(111,)`) -- it still enumerates and runs every family and
  every arm TYPE at least once, never omitting a family/arm-type
  combination. Every failed/invalid/interrupted/not-run cell is recorded
  explicitly in `failed.csv` (reason + status), never silently dropped.
- `script/quantized_input_comparison/run_task2.ps1` -- PowerShell launcher
  (`-Preflight`/`-Smoke`/`-Resume`/`-RunId`/`-QicRunId`/`-Hours`/`-Cpu`),
  same `Resolve-Python`/CPU-thread-cap conventions as
  `script/action_transport/run_stage_b.ps1`.
- `script/quantized_input_comparison/tests_task2.py` -- preflight tests
  (`--preflight`), synthetic/fast, no checkpoint or full training run:
  id-path backward compatibility (real tensor equality against a
  from-scratch reimplementation of the pre-Task-2 forward pass), the four
  init-pairing invariants (backbone/head/mask-token identity across all
  four context arms, projection identity between quantized/continuous, the
  geometry_init<->quantized rtol=1e-5/atol=1e-6 equivalence, verified
  max-abs-diff 0.0 on a synthetic family), that isolated-RNG model
  construction does not perturb an unrelated numpy stream, that
  `contextual.chunk_masks` genuinely calls `masking.build_eligible_mask`
  (source-inspected AND behaviorally cross-checked against a direct call on
  the same chunk), and the label-leakage static checks described above.

## Arm / family / seed matrix (spec, unchanged by --smoke's budget shrink)

| | vq_native | fsq_native | vq_kmeans | fsq_kmeans | raw_kmeans |
|---|---|---|---|---|---|
| context_id/quantized/continuous | 3 seeds | 1 seed | 1 seed | 1 seed | 1 seed |
| context_geometry_init | 3 seeds | -- | -- | -- | 1 seed |
| direct_id/quantized/continuous | 1 seed | 1 seed | 1 seed | 1 seed | 1 seed |

25 contextualizer fits (21 + 4 geometry_init), 15 direct-arm OT cells = 40
main-matrix segmentation evaluations, + 2 historical-compatibility jobs =
27 fits / 42 evaluations total.

## Verified so far (real commands, real output; NOT the full run)

Preflight (`python -m script.quantized_input_comparison.run_task2
--preflight`, `tests_task2.run_all()`, 6 tests): all passed, including the
id-path backward-compatibility check and the init-pairing tensor-equality
checks (max abs diff 0.0 between a fresh `context_geometry_init` and a
fresh `context_quantized` model's initial logits on the same input).

Ad hoc real-data checks (against real `qic_v3`, before the smoke matrix was
wired into the orchestrator): a tiny `raw_kmeans` slice (6-8 recordings) end
to end for `direct_id` (categorical OT, real Hungarian-scored MoF/Edit),
`direct_continuous` (cosine OT), and `context_quantized` (contextualizer
train -> embeddings -> cosine OT -> scoring) all ran successfully and
produced finite, plausible MoF/Edit/F1 values with real ground truth.

Smoke matrix (`-Smoke -RunId t2_smoke_v1`, `max_steps=40`, `rungs=(20,40)`,
`recording_cap=8`, native-VQ seeds collapsed to 111), real GPU, real
`qic_v3` data, real ground truth: **32/32 main-matrix cells + 2/2
historical-compatibility jobs completed**, 1524.7s (25.4 min) total on the
first pass. One real bug found and fixed by the smoke run itself: both
`context_geometry_init` cells (`vq_native`, `raw_kmeans`) failed with a
CUDA/CPU device-mismatch `RuntimeError` inside
`verify_geometry_init_matches_quantized` (its probe tensors were built on
CPU while the models being compared were on CUDA -- exactly the class of
GPU-only bug Task 1's README independently ran into for the same reason:
"this only surfaces when CUDA is available"). Fixed (tensors now placed on
each model's own device, compared after `.cpu()`); re-ran with `--resume`
and both cells completed in 11.1s, producing finite MoF/Edit/F1 values and
a verified geometry-init/quantized initial-output match. A second
`--resume` pass confirmed full idempotency: 0.0s, 124/124 segmentation
rows already present, 0 failed cells, nothing recomputed. A stale-failed-
ledger bug (a cell that failed then later succeeded under `--resume` left
its old `failed.csv` row behind) was also found and fixed during this
verification (`run_task2.run`'s failed-rows load now prunes any `cell_id`
already in `done.csv`).

Every family x every arm type (`context_id/quantized/continuous`,
`context_geometry_init` for `vq_native`/`raw_kmeans`,
`direct_id/quantized/continuous`) produced real, finite, plausible
MoF/Edit/F1@10/25/50 values against real Hungarian-mapped HuGaDB ground
truth (spot-checked: `segmentation.csv` has exactly the 30 (family, arm)
pairs the main matrix predeclares). The historical-compatibility jobs ran
the REAL (unreduced) `contextualizer_max_steps=5000`/
`contextualizer_patience_steps=1000` budget (only `final_rungs` was
reduced under `--smoke`, per the code as it existed when this run started;
a later edit that would also shrink the historical contextualizer budget
under `--smoke` did not affect this already-running process) -- so the two
historical jobs' ~9-10 minutes each are a REAL full-scale (364-recording)
timing data point, not a smoke extrapolation; their diff-vs-CSV rows all
read `no_historical_row` because the historical CSV only has rows at rungs
500/1000/2000/4000 and this run's `final_rungs` was reduced to (20, 40) --
a genuine reproducibility diff against the historical CSV numbers is
therefore deferred to the full run, as already documented below.

## Deviations from the spec's literal wording (and why)

- **`context_geometry_init` needs no new model branch.** The spec describes
  it as "a freely trainable embedding table initialized to the matching
  quantized arm's initial projected vectors" and says to "remove any
  dependency on the projection" after initialization. That is architecturally
  identical to `context_id`'s `nn.Embedding(K+1, d_model)` -- both are plain
  trainable lookup tables with no projection dependency -- differing only in
  their initial weight VALUES. Rather than adding a third `input_mode` to
  `CodeContextModel`, `context_geometry_init` reuses `input_mode="id"`
  unmodified and only its initial embedding weights are set externally (from
  a throwaway `context_quantized`-matched projection, applied once, then
  discarded) in `contextual.build_arm_model`.
- **OT arm-family assignment (categorical vs continuous) inferred, not
  restated verbatim in the spec.** The spec names six arms + geometry_init
  and separately says "Contextual OT uses normalized contextual embeddings"
  and "For direct vector arms, subtract Task 1's shared continuous-feature
  mean and L2-normalize," but does not spell out which OT solver
  (categorical vs continuous/cosine) each of the six arms uses. Read as:
  `direct_id` = categorical (raw code ids, the historical `categorical_asot`
  shape); `direct_quantized`/`direct_continuous` = continuous/cosine directly
  on Task 1's own feature vectors (no contextualizer); all four `context_*`
  arms = continuous/cosine on the trained contextualizer's embeddings (the
  historical `contextual_asot` shape, generalized to non-id contextualizer
  inputs). This keeps every contextual arm's OT identical in kind (only the
  embeddings differ), which is what makes the four context arms a fair,
  matched comparison.
- **No early stopping in the main matrix implemented as `patience_steps` set
  effectively infinite** (`10**9`), reusing `train_vector_contextualizer`'s
  existing patience-based stopping code path rather than adding a second,
  separate "no stopping" branch -- `stopped_by` is `step_cap` in practice for
  every main-matrix cell (verified: `steps_run == max_steps` whenever
  training completes without a `DeadlineExceeded`/exception).
- **Historical-compatibility rungs under `--smoke`.** Since `--smoke`
  reduces `cfg.rungs` for the main matrix, the historical-compatibility jobs
  also use the reduced rungs under `--smoke` (`cfg_overrides=dict(final_rungs=cfg.rungs)`)
  rather than always running the historical helper's real 500/1000/2000/4000
  ladder -- a full-length historical-compatibility diff is deferred to the
  full run, like the rest of the main matrix's real numbers.

## Handover: commands, disk, timing, and blocking issues for the full run

See the coordinating report for the exact preflight/full-run commands,
measured smoke timing, disk-requirement estimate, and any blocking issues --
not duplicated here to avoid the two going out of sync. **The full run
(`-Hours <N> -RunId <...>`) was NOT launched**; only `-Preflight` and
`-Smoke` were run, per instructions.
