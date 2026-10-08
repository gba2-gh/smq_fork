# Pretrained motion encoder as the information source (BABEL-2)

Draft · 7 October 2026 · Code so far: `script/pretrained_encoder/encoder.py`; weights: `external/motionbert/` (Apache-2.0).

## 1. Question

On LARa, every label-free decoder and grouping we tried ties or loses to the same 8-state B0 on T2 features (Milestone 1, input-representation test, unit-grouping test). One candidate explanation is that the information source, not the decoder, is the limit. This test changes only the source:

> Do frozen features from a large-scale pretrained motion encoder, passed through the unchanged tokenizer and matched B0 decoder, beat T2 relational features?

**Why BABEL-2, not LARa.**
- BABEL is AMASS-derived (NTU-25 joints, 30 fps, z up), so it maps directly onto MotionBERT's H36M-17 input.
- LARa's custom marker skeleton would add a retargeting risk.
- LARa's labels partly depend on objects outside the skeleton.
- One dataset only, as before.

**What has already failed.** A pretrained encoder trained on the same small dataset (Phase 5 T4, SMQ's released encoder) lost to T2 by about 18 MoF on BABEL-2 (39 vs 57). So the case for MotionBERT rests on the scale and diversity of its pretraining, not on being pretrained.

## 2. Encoder (frozen, no fine-tuning)

- **Checkpoint:** MotionBERT DSTformer, pretraining checkpoint `MB_release` (`checkpoint/pretrain/MB_release/latest_epoch.bin`, SHA-256 `02a9ee01…eaf8`). It was pretrained by 2D→3D lifting and masked reconstruction on AMASS, Human3.6M, InstaVariety and PoseTrack. **No action labels.** The NTU fine-tuned checkpoints are not used.
- **Exposure, disclosed:** AMASS includes the BABEL motions, so the encoder has seen these motions as 2D projections, without labels. This is a transductive exposure of the same kind as the tokenizer's.
- **Input path (matches MotionBERT's own AMASS pretraining, `tools/convert_amass.py` and `dataset_motion_3d.py`):**
  1. NTU-25 to H36M-17 by joint correspondence.
  2. mm to m, then the AMASS camera convention (`v @ [[1,0,0],[0,0,1],[0,-1,0]]`, × 0.298).
  3. 2D input (x, y, confidence 1), with no crop-scaling.
  4. Resample 30 → 60 fps.
  5. 243-frame clips at stride 81, with overlaps averaged.
- **Feature:** per-frame representation (17 joints × 512), mean over joints → 512-d per frame, back at 30 fps. Then the same window statistics as T2 (mean, sd, mean |Δ| over each 8-frame window).

**Feasibility check (done, label-free).** I used the encoder's own 3D head on 20 recordings, as a sanity check only:

| Check | Result |
|---|---|
| x, y round trip | exact (correlation 1.00) |
| Depth correlation, view rotations 0° / 90° / 270° | +0.57 / +0.58 / +0.65 |
| Depth correlation, 180° view (facing away) | −0.10 |
| Axis convention: lifting error (`amass`) | 145 mm, against 288 and 367 mm for the alternatives |
| Lifting error vs a flat-depth baseline | does not beat it in absolute terms |
| Speed | about 0.1 s per recording (≈ 30 s for all 255) |

What this supports:
- The joint mapping and axis convention are consistent, so the encoder is reading the skeleton as a body.
- The pretraining-only checkpoint lifts BABEL poses only moderately, and fails for back-facing views. That is a property of the encoder, not a bug, and it is a reason to expect a modest effect.

**Declared view:** the input view is rotated 270° about the vertical axis. It is chosen by the label-free lifting check above (best depth correlation and lowest lifting error of the four views) and frozen before any scored run.

## 3. Design

Both inputs go through the unchanged Phase 5 pipeline: standardize → PCA-64 → K-means-512, the same sample cap and the same seeds (111, 222, 1538574472).

| Input | Per-window feature |
|---|---|
| `t2` | existing T2 relational (Phase 5 preparation, rebuilt only if needed for the folds) |
| `mb` | MotionBERT window statistics (1536-d before PCA) |

**Population: five recording-disjoint folds**, not the preparation split. The existing BABEL-2 evaluation split is 27 recordings (8,009 frames, about 4.5 minutes), too small for a +3 bar.
- The 255 recordings are shuffled with seed 111 and dealt into 5 folds.
- For each fold, the matched B0 (Milestone 1 machinery, C = 5) is fitted on the other four folds and decoded on the held-out fold, with one Hungarian mapping per fold.
- Every recording is scored exactly once.
- BABEL has no subject IDs, so folds are by recording, not subject.

**Diagnostics** (label-using, interpretation only):
- a recording-grouped supervised linear action probe and a code-table probe for each input (as in the input-representation test);
- many-to-one MoF;
- balanced accuracy and per-class recall;
- predicted ÷ true segments;
- decode multipliers 0.5 and 2;
- recording NMI of the decoded states.

## 4. Decision rule (declared before any scored run)

Paired by seed, using the mean over the 5 folds.

| Outcome | Condition | Next step |
|---|---|---|
| **ADOPT_ENCODER** | `mb − t2` ≥ +3 mean on MoF **and** F1@50, all three seeds positive on both | MotionBERT features become the input any later decoder or grouping must beat; rerun the strongest earlier ideas on them. |
| **INFORMATION_ONLY** | ADOPT fails, but the `mb` supervised probe exceeds the `t2` probe by ≥ +5 points (mean over seeds) | The encoder carries more action information that the label-free decoder does not use. Report; the next question is organisation on these features, not another source. |
| **STOP** | neither | Frozen MotionBERT features do not help here. Stop this source; do not try other layers, pooling or views in response. |
| **INCONCLUSIVE** | missing seeds or folds, invalid solves, failed verification | Report what is missing. |

**Seed noise warning.** Historical BABEL-2 B0 MoF on T2 varies widely across seeds (57.4 / 46.0 / 58.4). The `t2` and `mb` tokenizers share seeds but not features, so the pairing reduces noise only partly. The fold mean helps. All-seeds-positive is a demanding condition here, and that is intended.

## 5. Limits

- One encoder, one checkpoint, one pooling (mean over joints), one view, frozen. A negative result is about this configuration.
- The encoder has seen AMASS (including BABEL) motions without labels, and the tokenizer is fitted on all recordings. Both are transductive and disclosed.
- Recording-disjoint folds share performers across folds, because BABEL has no subject IDs.
- Three seeds are not confidence intervals.

## 6. Implementation and cost

Two pieces of new code, reusing the M1 B0 code, the decode ladder and scoring:
- **Feature extraction and preparation:** register an `mb` source in the Phase 5 tokenizer pipeline, as the input-representation test did.
- **The fold driver:** dataset-generic overrides of the Milestone 1 settings (C = 5, window 8, 30 fps), with folds by recording.

Tests: the joint mapping and axis convention, label-free fold construction, evaluation isolation, the decision rule, and that features are identical across reruns.

About half a day including smoke. Runtime is about 2 inputs × 3 seeds × 5 folds × 2–3 min B0 ≈ 1–1.5 h, plus feature extraction (about 1 min) and probes.

## 7. Implementation notes (7 October 2026)

- Code: `script/pretrained_encoder/` (`encoder.py`, `features.py`, `run.py`, `run.ps1`, `tests.py`, `config.py`). The Milestone 1 B0 fit, decode ladder and scoring are reused with run-time overrides (BABEL-2, window 8, 30 fps, C = 5, recording folds); nothing in `script/compositional_dynamics` was edited for this test.
- The "subject" association in the reused diagnostics is the recording association, because BABEL has no subject IDs.
- Per-recording MotionBERT features are cached under `results/pretrained_encoder/feature_cache/` and shared by the three seeds.
- Tests: 9 pass (joint mapping and axis convention, exact checkpoint loading and hash, x/y round trip on real motion, deterministic per-recording features, window statistics, fold construction, run-time overrides, decision rule, installation). Smoke run (both inputs, seed 111, fold0) passed; its numbers are not evidence.
- Run: `.\script\pretrained_encoder\run.ps1 -Run -RunId pe_babel2_001 -Hours 4 -ReuseCompleted *>&1 | Tee-Object pe.log`. The gate is INCONCLUSIVE unless `results/pretrained_encoder/verification.json` matches the code hash (written by the tests).
