# MotionBERT clip-context test (BABEL-2)

Version 2 · 8 October 2026 (changes in Section 7) · Follows [PRETRAINED_ENCODER_RESULTS.md](PRETRAINED_ENCODER_RESULTS.md) and the per-class diagnosis of run `pe_babel2_001`.

## 1. Hypothesis

Frozen MotionBERT features lose run and stand_up because each frame's feature attends over a 243-frame clip (about 4 s at 60 fps). That blurs short segments into their neighbours.

Evidence from `pe_babel2_001`, which was diagnosed with evaluation labels, so BABEL-2 is development data:

| Recall, segments shorter than 1 s | T2 | MotionBERT |
|---|---|---|
| run | 0.72 | 0.37 |
| stand_up | 0.40 | 0.11 |

- On segments longer than 2 s, the two inputs are equal.
- Sit (51% of frames, mostly long segments) gains from the smoothing.
- A smaller, separate effect: run seen head-on is poorly recovered (0.50 vs 0.93), probably because the 2D input hides depth. This test does not address it.

**Prediction:** shorter MotionBERT inference clips improve how well short run and stand_up events are recovered, in the features and in the decode. Long-segment classes may lose part of their gain. The size of the change should follow the clip length (dose–response).

**What the intervention changes.** It is a **clip-context intervention**: shortening the clip changes three things together:
- the clip length;
- the support gained by averaging overlapping clips. With stride L/3, a frame in the interior of a recording draws on about 5L/3 of input: about 6.75 s for 243, 2.25 s for 81, 0.75 s for 27 (recording ends and window aggregation change this locally);
- which part of the temporal positional embedding the model sees, which also means a shift from its pretraining input distribution.

Available support does not show that the network actually smooths over that whole interval. So a positive result supports "shorter inference clips improve short-event recovery in this pipeline", not "attention erased these events".

## 2. Arms

Everything except the clip length is frozen from `pe_babel2_001`:
- checkpoint `MB_release`, 270° view, mean over joints;
- 30 → 60 fps, then window statistics over 8 frames;
- PCA-64 → K-means-512 → matched B0 with C = 5;
- the same 5 recording folds.

| Arm | MotionBERT clip (60 fps) | Stride | Context | Role |
|---|---|---|---|---|
| `t2` | – | – | 0.27 s (8 frames) | reference |
| `mb243` | 243 | 81 | ~4.0 s | current MotionBERT (cells reused from `pe_babel2_001` for its 3 seeds) |
| `mb81` | 81 | 27 | ~1.35 s | **hypothesis** |
| `mb27` | 27 | 9 | ~0.45 s | dose–response end point |
| `t2_mb` | T2 + `mb243` | – | both | pipeline candidate: combine the local, view-invariant features with the smooth ones (secondary) |

- **Stride rule:** the stride is always one third of the clip, the same overlap as the original, and overlapping clips are averaged. MotionBERT's temporal positional embedding is sliced to the clip length, as its code does for any length up to 243.
- **`t2_mb` construction:**
  1. Standardize each source and fit PCA-32 for each on the same fitting sample.
  2. Centre each PCA block and divide it by its own fitting-sample RMS norm (two saved scalars). Equal dimension counts alone do not give equal contributions, because the retained eigenvalues can differ; the RMS scaling equalizes each block's average contribution to Euclidean K-means distances while keeping the geometry within each block.
  3. Concatenate to 64 dimensions, then K-means-512.

  Each block's share of the squared distances is reported. The fusion arm tests whether the two sources complement each other. It is secondary and does not test the context hypothesis.

**Seeds:** 111, 222 and 1538574472 (existing), plus **333 and 444** (new), for 5 seeds in total. The new seeds need fresh T2 and MotionBERT preparations built with the unchanged Phase 5 procedure. The existing `mb243` and `t2` cells are reused for the original three seeds. The five seeds measure preparation and initialization sensitivity on the same recordings; they are not five independent replications.

**Exposure (declared).** The preparation samples windows and fits the scaler, PCA and K-means across all recordings. The supervised probe and B0 are then fitted on recording folds. The protocol is therefore **held-out readout and decoder evaluation with transductive tokenization**, not fully held-out representation learning. This policy is kept so the original cells can be reused. Within each seed, every arm must use identical K-means sample window identifiers; this is verified in preflight.

**Preserving the 243 path.** The clip length becomes a parameter, so the original path must be reproduced exactly:
- Before any reused cell is trusted, re-extract `mb243` features for a fixed set of recordings with the new code and require them to be bit-identical to the cached ones.
- Feature-cache identities include the clip length, stride, checkpoint hash, view, resampling, pooling and overlap aggregation.
- Reused B0 cells are accepted only if their recorded source hash matches.

## 3. Measurements

Labels are used for evaluation and diagnosis only.

**Duration buckets (all measurements).**
- Buckets come from ground-truth segment durations, never predicted ones: under 1 s, 1–2 s, and over 2 s; "short" means under 2 s.
- Frames take their segment's bucket.
- A window takes the bucket and label of the ground-truth segment covering most of its frames. Windows that cross a ground-truth boundary are flagged. Every probe result is also reported with crossing windows excluded.
- For each bucket and class, report segment counts, frame counts and window counts.

**Representation level (independent of the decoder).**
- **One** supervised linear classifier per fold, fitted on all fitting windows of the PCA-64 representation. The same recording folds are used as in the earlier tests. Its held-out predictions are then stratified by duration bucket.
- The probe tests the PCA-64 representation that the tokenizer and decoder actually receive, not all the information in MotionBERT's raw features.
- **Primary (R):** the macro-average of run recall and stand_up recall on short-segment windows.
- **Secondary:** overall accuracy on short and on long windows, and per-class recall in every bucket.
- Label-free: feature autocorrelation at 0.5 s and 1 s lag.

**Decode level (matched B0, mean over folds).**
- **Primary (D):** the macro-average of run recall and stand_up recall on frames in short segments.
- **Secondary:**
  - frame-pooled run + stand_up recall;
  - MoF, Edit, F1@10/25/50 and balanced accuracy;
  - per-class recall by bucket;
  - predicted ÷ true segments.

**Feasibility, before any scored run (label-free).**
- The x/y round trip and lifting depth correlation on 20 recordings, for each clip length.
- These are numerical diagnostics only: shorter clips are inputs MotionBERT was not pretrained on, and good lifting does not validate the quality of the action features.

## 4. Declared decision rules

### 4.1 Mechanism (does the shorter clip help short events?)

Compare `mb81` against `mb243`, paired by seed:
- **(R) representation:** the primary R endpoint rises by **at least 0.05** (macro recall) on average, positive in at least 4 of 5 seeds.
- **(D) decode:** the primary D endpoint rises by **at least 0.08** on average, positive in at least 4 of 5 seeds.

| Outcome | Condition | What it supports |
|---|---|---|
| **CLIP_CONTEXT_HELPS** | R and D | Shorter inference clips improve short-event recovery for run and stand_up in this pipeline. It does not show which component (attention, overlap averaging, positional exposure) removed those events. |
| **FEATURES_ONLY** | R, not D | The probed PCA-64 features recover short events, but something downstream loses them: quantization, prototype discovery, transport or mapping. It does not single out temporal smoothing. |
| **NO_RECOVERY** | not R | This intervention did not recover the signal. It does **not** rule out excessive context as a cause, because shorter clips also change positional exposure and the input distribution. |

`mb27` is reported as dose–response. It is not part of the rule.

### 4.2 Pipeline (what becomes the current baseline?)

Each arm is compared with the current baseline, `mb243`, paired by seed over 5 seeds:
- **Adopt** if mean MoF and mean F1@50 are each **≥ −1**, and at least one of them is **≥ +2** with at least 4 of 5 seeds positive.
- If several arms qualify, adopt the one with the larger MoF + F1@50 sum.
- Otherwise `mb243` stays.

This is a development decision, not a claim. The accumulated pipeline is later tested once, frozen, on BABEL-3 or BABEL-1. Both datasets have influenced earlier phases, so that test gives **cross-dataset evidence for a frozen pipeline**, not untouched confirmation.

## 5. Limits

- One checkpoint, one view and one pooling; only the context length varies (plus the combination arm).
- Shortening the context changes MotionBERT's input distribution as well as its context. The feasibility check and the dose–response pattern help separate these, but cannot fully.
- The decoder's temporal kernel (±3 windows, about ±0.8 s, so about 1.6 s across) also smooths short events, equally in every arm. It is a regularizer, not an enforced minimum duration. That is why the representation-level probe is part of the mechanism rule.
- Recording folds share performers (BABEL has no subject IDs). The encoder saw AMASS motions without labels.

## 6. Implementation and cost

Code changes in `script/pretrained_encoder/`:
- make the clip length a parameter, with a separate feature cache per clip length;
- register `mb81`, `mb27` and the block-PCA `t2_mb` preparation;
- build the new-seed preparations;
- add the short/long-segment probe split, the autocorrelation measure and segment-length recall;
- update the report with both decision rules;
- add tests:
  - the stride rule;
  - cache identity and separation;
  - bit-identical `mb243` features after the refactor;
  - RMS-balanced fusion blocks;
  - GT-duration bucketing, including boundary-crossing windows;
  - a single probe fit stratified afterwards;
  - identical sample identifiers across arms;
  - both decision rules.

About half a day including smoke.

Runtime is about 95 new B0 cells:
- `t2` and `mb243`: 2 new seeds each;
- `mb81`, `mb27` and `t2_mb`: 5 seeds each;
- every seed–arm pair runs 5 folds.

At about 2 minutes per cell that is roughly 3–3.5 h, plus feature extraction (about 1 minute per clip length) and probes. This fits one night. Primary arms (`mb81`, then `mb243`'s new seeds) run first, and `t2_mb` last.

## 7. Changes from version 1

| Change | Reason |
|---|---|
| Verdicts renamed and narrowed (CLIP_CONTEXT_HELPS / FEATURES_ONLY / NO_RECOVERY) | The test cannot attribute an effect to attention specifically. FEATURES_ONLY points to everything after the probed features, not only smoothing. A failure of R does not rule out context. |
| R and D share one primary population: the macro-average of run and stand_up recall on segments under 2 s | Before, R covered all short-segment classes, so sit gains could pass R while run and stand_up stayed poor. |
| One probe per fold, stratified afterwards; buckets from GT durations; boundary-crossing windows flagged and also excluded; support counts reported | Separate short/long probes answer a different question, and bucketing has to be unambiguous. |
| Described as a clip-context intervention, with the overlap support stated (about 5L/3) | Clip length, overlap-averaging support and positional exposure change together. |
| Fusion blocks are RMS-scaled after per-source PCA, and each block's distance share is reported | Equal dimension counts do not give equal contributions to K-means distances. |
| Transductive tokenization declared; identical sample identifiers verified; the probe tests PCA-64 | States the protocol as it is. |
| Bit-identical parity for the 243 path, and cache identities that include every extraction setting | Reused cells are only valid if the original path is preserved. |
| Decoder radius written as ±0.8 s, not a minimum duration; lifting check described as a diagnostic only; the later BABEL-1/3 run described as cross-dataset evidence; seeds described as sensitivity, not replications | Accuracy of wording. |
