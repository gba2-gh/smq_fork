# Phase 9: iterative contextual target refinement (a diagnostic)

Revision 3 — 4 October 2026 (two rounds of evaluator corrections applied; see the end). **Implemented; development
stage complete** (run `ts5_v2`, finished 5 October 2026; results below). The prospective BABEL-1 stage P has not been run.
Development stages projected **~7.9 h** under a **10-hour cap**. The prospective BABEL-1 stage is a separate run
(~3.9 h). Parents: `PHASE8C_SYMMETRY_TEST.md` (results: `results/week5/timescale_inspection/ts5_v1/phase8c/REPORT.md`) and
`PHASE7_LABEL_FREE_SCALE.md`.

This phase is the **representation track**. The time-scale track (mask span selection, learned temporal support:
Phase 7a, K1) is planned separately. To keep the two apart, the mask span is fixed here at the frozen Phase 7 recipe.

## Development results (`ts5_v2`, 5 October 2026)

Report: `results/week5/timescale_inspection/ts5_v2/phase9/REPORT.md`. All gates passed (G3, G9a-G9g, and G9f for each of
the 9 iteration-2 families). Every planned cell ran: 36 trainings and 54 decodes, with nothing incomplete. The analysis
hash at launch (`96db3bc17e14621d`) equals the hash of the code that wrote the report.

**Execution (deviation from the launch line below).** The first launch (`ts5_v1`) and the first `ts5_v2` attempt died
from native crashes during stage B: an OpenBLAS access violation (0xC0000005) and heap corruption (0xC0000374), neither
leaving a Python traceback. The run was completed under `resume_phase9.py`. That driver runs each stage in a fresh
process and retries a crashed stage. It is outside the implementation hash, which was unchanged
(`e11dfd7b61988706`), and units are cached by identity, so the resumed results equal an uninterrupted run. The root
cause of the crashes is unknown.

**F1@50 (mean of seeds 111 / 222 / 1538574472, last rung).** R is the Phase 7 T2 reference.

| Dataset | R | G | I1q | I1 [T2 grouping] | I1 [I2 grouping] | I2 [T2 grouping] | I2 [I2 grouping] |
|---|---|---|---|---|---|---|---|
| BABEL-2 | 30.68 | 28.82 | 26.25 | 27.47 | 26.42 | 25.46 | 26.95 |
| BABEL-3 | 23.23 | 23.75 | 24.23 | 21.84 | 23.29 | 21.90 | 23.25 |
| LARa | 24.02 | 24.20 | 23.34 | 22.33 | 25.23 | 24.11 | 25.90 |

**Declared comparisons (F1@50).** Every one is **within the bar** on all three datasets.
- **Q2 (primary, I2 vs I1, I2-code grouping):** BABEL-2 +0.53 (+0.2 / +3.6 / −2.3); BABEL-3 −0.04 (+0.4 / −0.2 / −0.3);
  LARa +0.67 (−0.5 / +1.7 / +0.7).
- **Q2r (T2-code grouping):** −2.01, +0.06, +1.78.
- **Q1a** (G vs R): −1.86, +0.52, +0.18. **Q1b** (I1q vs G): −2.57, +0.48, −0.86. **Q1c** (I1 vs I1q): +1.22, −2.39,
  −1.01.
- **Secondary metrics:** one declared contrast passes the bar in reverse. On BABEL-2, I1q is worse than G on MoF (−4.95)
  and Edit (−3.46), with all seeds negative.

**Descriptive.**
- **Q3a (I2 at the I2-code grouping vs R):** BABEL-2 −3.73 (−2.8 / +0.1 / −8.4); BABEL-3 +0.02; LARa +1.88
  (−0.1 / +3.2 / +2.5). LARa I2 [I2 grouping] is the highest LARa mean recorded (25.90), but it is within the bar of R.
- **OT initialization (D1, D2):** on LARa, the I2-code grouping helps both models: I1 MoF +3.08 (passes the bar), F1@50
  +2.90 (within); I2 F1@50 +1.79. This is the largest effect on LARa in the phase, and it comes from the decoder's
  starting prototypes, not from the targets.

**Shortcut check (declared failure mode): observed.** The iteration-2 units are more recording-specific than T2's.
Recording association rises with depth through the I1 model, and layer 2 sits above T2 on every measure.

| Measure (T2 codes → I1 layer-2 targets) | BABEL-2 | BABEL-3 | LARa |
|---|---|---|---|
| Recording NMI | 0.620 → 0.747 | 0.608 → 0.768 | 0.370 → 0.398 |
| Subject NMI | — | — | 0.356 → 0.362 |
| Same-recording nearest-neighbour share | 0.325 → 0.506 | 0.450 → 0.674 | 0.163 → 0.217 |
| Cross-seed NMI (chance) | 0.868 → 0.867 (0.381) | 0.843 → 0.863 (0.408) | 0.688 → 0.665 (0.077) |

The I2 models' embeddings reach same-recording neighbour shares of 0.69 (BABEL-2) and 0.83 (BABEL-3). Agreement across
seeds does not increase, so the refined targets are not more stable; they are more tied to the recording. This matches
the Q1/Q2 finding that K-means codes carry more subject than action information.

**Other observations.**
- On LARa, every arm selected its checkpoint at 4,600-5,000 of 5,000 updates, so training was budget-limited there;
  BABEL selected at 600-1,500.
- On BABEL-3, the I2 targets are barely predictable: masked-code accuracy 0.08-0.15, with validation loss about 5.3
  against ln 512 ≈ 6.2.
- On LARa, I1 and I2 trade recall between classes rather than improving uniformly. Cart gains 32-37 points, while
  Walking (−19 to −29) and Synchronization (−33 to −34) lose.

**Reading.** One round of contextual target refinement does not improve segmentation at this budget. Neither does
replacing code ids with centroid or continuous T2 input. T2 with the code-id contextualizer stays the recipe. The
phase's main finding is mechanistic: re-clustering contextual states amplifies recording identity, the same shortcut
seen in the earlier code analysis.

**Stage P** is declared to run whatever these results show. Its dry run plans gates plus BABEL-1 (3 I1 and 3 I2
trainings, 9 decodes) at about 3.9 h. Launch: `run_phase9.ps1 -RunId ts5_v2 -SourceRun ts5_v1 -Stage P -Hours 5`. Output:
`results/week5/timescale_inspection/ts5_v2/phase9p/`.

## Scope: what this phase can and cannot show

Phase 9 tests **iterative contextual pseudo-label refinement** (the HuBERT recipe, Hsu et al. 2021) on top of T2, with
the model budget and OT fixed.

- **It can show:**
  - how the contextualizer's input parameterization matters: code identity, centroid vector or actual features (Q1);
  - whether re-clustering the contextualizer's internal states into new training targets helps segmentation (Q2).
- **It cannot show** that the system discovers better fine-grained motion units ("movemes") that are later composed
  into actions:
  - every arm still reads T2's **PCA-64 window summaries**, so within-window temporal order, and whatever the
    summaries discarded, cannot be recovered;
  - I2's targets are clusters of **contextual** transformer states, so a window's new code can depend on its
    neighbouring windows. They are contextual targets, not local units.

A better I2 score would be evidence that refining contextual targets helps. A claim about learned local units needs
the separate experiment under **Follow-up**, with a local-unit control and an independent evaluation.

## Why

Phases 5-8c tested hand-built tokenizers. T2 (per-window statistics of inter-joint distances, PCA-64, K-means-512)
is the best and has held up against every hand-built variant:

- Phase 7 (BABEL-1, prospective): T2 beat raw coordinates by +12.3 F1@50. The contextualizer added value on T2 (+4.99
  over T2 `centre`) but not on T0 (−3.9).
- Phase 8a: appending gravity and heading channels with block weighting lost 2-8 points.
- Phase 8c: heading canonicalization did not help raw coordinates (T6 vs T0 within ±0.8). Canonical coordinates with
  T2's temporal summary stayed below T2 (BABEL-2 −9.55, all seeds), and unweighted T5gh at best tied T2.

Two properties of the pipeline remain, and both are what HuBERT targets:

1. **What counts as relevant is decided by variance.** PCA and K-means keep whatever varies most.
2. **The contextualizer only sees code identities, and its targets never improve.**

## Fixed throughout

- **Iteration-0 tokens:** T2 (`t2_relational`), the existing Phase 5/6 preparations (continuous PCA-64 window features,
  K-means-512 codes). Each window's features use only its own frames.
- **Backbone and training:** the Phase 5 reference contextualizer and training procedure (128 wide, 4 layers, 4 heads,
  feed-forward 512, post-LayerNorm, ALiBi, dropout 0.1, context 128, stride 64, batch 64, 5,000 updates, learning rate
  3e-4, 500 warm-up steps, cosine schedule, 30% masking). **Mask span 2** on BABEL-2/3 and LARa, not re-tuned.
- **Matched across arms within a seed:**
  - backbone, head, mask-token and input-projection initialization (`build_matched_init` from the seed);
  - training chunks, batch order and masks. These come from the seed and epoch only, and eligibility depends on
    window positions, not codes, so I2 sees the same masks as I1.

  Gate G9g checks this.
- **Checkpoint:** selected by validation masked-prediction loss, compared only within one arm.
- **Transductive targets, stated explicitly.** T2's K-means was fitted on complete windows sampled from all recordings,
  training and validation (`fitmod.sample_window_ids` over every recording, no labels). So the validation loss of every
  arm, including the existing reference R, is transductive with respect to its targets. I2's K-means uses the same
  sampled windows, keeping the protocol identical. No labels are involved anywhere.
- **Embeddings for OT:** final-layer hidden states, averaged over overlapping chunks, L2-normalized (unchanged).
- **OT:** unchanged (pooled continuous-cosine fit, current radius, outer cap 50, rungs 500/1,000/2,000/4,000, known
  class count). The code-to-state grouping that initializes it is named per decode (see **Decodes**). Frame predictions
  are saved.
- **Seeds:** 111, 222, 1538574472, all three for every cell.
- **Iterations: exactly two.** No score-triggered third iteration. No labels enter fitting, checkpoint selection, layer
  choice or the iteration count.

## Arms

| arm | input to the contextualizer | targets | differs from the row above in |
|---|---|---|---|
| **R** `t2_codes` (reference, read) | T2 code identity, free embedding table, default initialization | T2 codes | — (Phase 5/6 units, span 2) |
| **G** `t2_geometry_init` | T2 code identity, free embedding table **initialized** to the projected centroids W(q_c/RMS)+b | T2 codes | initial geometry (same parameterization as R) |
| **I1q** `t2_centroid` | the code's K-means centroid vector ÷ RMS through a trainable projection | T2 codes | parameterization: free table → tied to the centroid (same geometric start as G) |
| **I1** `t2_continuous` | the window's own T2 continuous PCA-64 features ÷ the same RMS, through the same projection | T2 codes | detail inside the K-means cell: centroid → actual features |
| **I2** `t2_continuous_it2` | the window's T2 continuous features | K-means-512 on I1's layer-2 states | targets: T2 codes → contextual refined codes |

Masked positions are replaced entirely by the learned mask token in every arm, so no model sees a masked window's
input. The ladder R → G → I1q → I1 changes one thing per step:
- **R → G:** initial geometry only;
- **G → I1q:** whether the input stays tied to the centroid;
- **I1q → I1:** whether the actual features add detail beyond the centroid.

**Iteration-2 targets.** For each seed, the trained I1 model (validation-selected checkpoint) encodes every window of
every recording with no masking, averaging overlapping chunks as for the embeddings. Hidden states are taken from all
four layers.
- **Layer 2 is preselected for the targets.** HuBERT Base took its second-iteration targets from the 6th of its 12
  layers; layer 2 of 4 is the same relative depth. Layers 1, 3 and 4 are clustered only for diagnostics.
- **K-means:** K = 512 (as T2, so only the targets change), fitted on the same sampled windows as T2's K-means, with the
  seed's random state and T2's preparation thread count. Every window gets a code.
- **The I2 preparation family** keeps the loader's dimensional contract:

  | artifact | content | dimension |
  |---|---|---|
  | `codes` | iteration-2 codes | — |
  | `continuous` | T2's continuous features, unchanged (the model input) | 64 |
  | `lookup` | for each new code, the frame-weighted mean of T2's continuous features over its windows (as `per_code_pca_centers`) | 64 |
  | mean, RMS | from `continuous`, identical to T2's | 64 / scalar |
  | grouping | the existing `initial_grouping` from the new codes and T2's continuous features | — |
  | `teacher_centroids.npz` (separate) | layer-2 K-means centroids | 128 |

  The trainer takes `input_dim` from `lookup.shape[1]`, so storing the 128-d centroids as `lookup` would build a
  128-input model fed 64-d features. Gate G9f checks the dimensions.

## Trainer and extractor: a gated copy, not an edit

The Phase 5 trainer (`contextualizer_calibration.train.train_core`) defines `feature_of(chunk)` inside the function as
`lookup[chunk.codes]`, and `run_training_job` passes `lookup` only to `context_quantized`. The Phase 5 extractor
(`evaluate.extract_embeddings`) also reads `lookup[codes]`. Neither can be made to feed `chunk.continuous`, or to pass
the centroids that `context_geometry_init` needs, by subclassing or wrapping. A change is therefore unavoidable.

- **Chosen: a line-for-line copy** in `trainer9.py` of `train_core`, `run_training_job` and `extract_embeddings`, with
  two additions:
  - a `feature_of` option (default: the existing `lookup[chunk.codes]`; `continuous` for I1 and I2);
  - passing `lookup_over_rms` for the geometry-initialized arm.

  The shared files are **not** edited. Their contents are hashed into every Phase 5-8c implementation identity
  (`TRAIN_FILES`), so an edit, even backward-compatible, would make all earlier runs fail their resume and identity
  checks.
- **Gate G9a proves the copy's default path is the original**, by retraining rather than only re-decoding. Training is
  bit-reproducible on this machine: ctx_cal_v1's reproduction retrained models with identical checkpoint tensors and
  metrics.
- Same discipline as Phase 7 (`fit_and_score_keep`) and Phase 8a (`fit_population8`).
- If you prefer the shared edit instead (a default-preserving `feature_of` option in `train.py` and `evaluate.py`),
  the same gates apply. The cost is that Phases 5-8c could no longer be resumed or re-verified under their recorded
  identities.

## Decodes: separating target learning from OT initialization

OT starts from prototypes built through the code-to-state grouping (`continuous.build_mu0`), so I2's codes change OT's
initialization as well as its embeddings:

| embeddings | T2-code grouping | I2-code grouping |
|---|---|---|
| R, G, I1q | ✓ | — |
| I1 | ✓ | ✓ |
| I2 | ✓ | ✓ |

- **Q2 primary:** I2 vs I1, both with the **I2-code grouping**. Robustness: both with the T2-code grouping.
- **Initialization effect (descriptive):** each model under its two groupings.

## Declared comparisons

Primary metric F1@50 at the last rung, paired by seed. The bar is a mean gain ≥ 3 with every seed in the same
direction. Within the bar is read as comparable performance, not equivalence. All five metrics are reported.

- **Q1a: initial geometry. G vs R.**
- **Q1b: parameterization, at the same geometric start. I1q vs G.** A free embedding table vs an input tied to the
  centroid through a projection.
- **Q1c: detail inside the K-means cell. I1 vs I1q.** Same parameterization; actual features vs centroid.
- **Q2: does training on refined contextual targets help? I2 vs I1,** at the I2-code grouping (primary) and the T2-code
  grouping (robustness). The phase's main question.
- **Q3 (descriptive):** I2 vs R, the total effect against the current pipeline.
- **Q4 (stage P, prospective):** I1 and I2 vs R on BABEL-1, with the same bar and the BABEL-1 caveat.

Q1a-Q1c are read as effects of those specific contrasts, not as the value of "geometry" or "continuous input" in general.

## Diagnostics (label-free unless stated)

- **All four layers of I1 (diagnostic only; they choose nothing):**
  - occupancy (occupied codes, perplexity);
  - cross-seed code agreement (normalized mutual information);
  - association with recording identity (normalized mutual information between codes and recording, and the
    same-recording share of nearest neighbours);
  - on LARa, association with subject (subject is metadata, not an action label).

  High cross-seed agreement is not read as quality: stable recording identity or coarse posture would also score
  highly. T2's codes are reported on the same measures as the reference.
- Masked-prediction accuracy and validation loss per arm (within arm only).
- **Shortcut check:** recording and subject association for the iteration-2 codes and every arm's embeddings, against
  T2. If iteration-2 units become more recording-specific than T2's, that is reported as a failure mode.
- Predicted vs ground-truth segments per recording.
- Per-class recall against R (uses labels; analysis only).

## Known risks, stated in advance

- **Continuous input may make span-2 masking too easy.** A masked window can be interpolated from its continuous
  neighbours, so the model may learn local smoothing. Masked-prediction accuracy is reported. The span belongs to the
  time-scale track.
- **Iteration-2 clusters may encode recording identity;** see the shortcut checks.
- **Late checkpoints overfit on BABEL-2/3,** so validation selection is kept.
- **The OT decoder remains a ceiling.** Phase 9 changes OT's input, not OT.

## Gates (preflight; all before stage A)

- **G9a retrained reproduction:** the copied trainer's default path (`context_id`, span 2) retrains BABEL-2 seed 111.
  It must reproduce ts5_v1's Phase 5 job exactly:
  - selected checkpoint step;
  - validation-loss history;
  - selected checkpoint's tensor SHA-256;
  - after decoding, every rung and metric of the saved unit.

  About 4 minutes.
- **G9b input and masking:**
  - changing a masked window's input leaves every output unchanged, in every vector arm;
  - in I1, changing a **visible** window's continuous vector while holding its code fixed changes both the training
    loss and the inference embeddings;
  - in I1q the same change leaves both unchanged, because its input is the code's centroid.
- **G9c extraction:**
  - (i) code input: the copied extractor's normalized output equals `evaluate.extract_embeddings`'s exactly;
  - (ii) continuous input: it equals the independent continuous-input extractor
    `quantized_input_comparison.contextual.extract_vector_embeddings(arm="context_continuous")` exactly, for the same
    model and recording;
  - (iii) raw per-layer states have the declared shapes and are finite, and layer 4's raw states normalize to (i)/(ii).
- **G9d agreement:** the cross-seed agreement function is permutation-invariant (relabelled identical codes give 1.0) and
  gives about 0 for independent random codes (unit test).
- **G9e grouping path:** with iteration-2 codes replaced by T2's codes, the I2 family builder reproduces T2's grouping exactly,
  and its `lookup` equals the per-code centres that `initial_grouping` itself computes from T2's codes and features. (It
  is not T2's K-means `lookup`: those centroids are means of the sampled fitting windows, not frame-weighted means over
  all windows.)
- **G9f dimensional contract** (every I2 family): `lookup.shape[1]` = `continuous.shape[1]` = the model's input
  projection width = the mean's length = 64; the RMS equals T2's; codes lie in [0, 512).
- **G9g matching:** for one seed, the first 50 training batches have identical chunk order and identical mask indices
  across G, I1q, I1 and I2. G, I1q and I1 start from identical backbone and head tensors.

## Stages and order

Ordered by dataset, so a budget stop leaves BABEL-2 and BABEL-3 complete rather than every dataset half done.

| Stage | Work | Estimate |
|---|---|---|
| G | Gates G9a-G9g | ~15 min |
| BABEL-2 | A: G, I1q, I1 × 3 seeds (9 trainings, 9 decodes); B: 4-layer features, 12 K-means fits, the I2 family; C: I2 × 3 seeds (3 trainings; 9 decodes) | ~73 min |
| BABEL-3 | the same | ~67 min |
| LARa | the same | ~310 min |
| Report | diagnostics, report | ~10 min |

**Development total about 7.9 h** (about 8.4 h if Stage B hits its worst case), with a **10-hour cap.**
- The first training and decode of each dataset are timed, and the remaining work is re-projected after them.
- A job projected to cross the cap is not started.
- Incomplete cells are reported as incomplete.

**Stage P (separate run, about 3.9 h):** prospective BABEL-1. I1 and I2 (two iterations, layer 2, fixed now), 3 seeds;
I1 decoded with both groupings, I2 with the I2-code grouping. It runs whatever the development results show. The BABEL-1
T2 code reference already exists (Phase 7). BABEL-1 is not untouched: earlier phases analysed it, and 10 recordings are
byte-identical to BABEL-2/3 recordings.

## Budget basis

- **Unit costs** measured in Phases 5-8c (train / `context` decode, minutes): BABEL-2 1.5 / 2.5; BABEL-3 1.6 / 2.1; LARa
  2.2 / 14.4; BABEL-1 1.5 / about 20.
- **Stage B, measured:** one K-means-512 fit on 10,000 × 128 features at the preparation thread count took 12 s
  (22 iterations), and assigning 220,000 windows took 0.1 s. That was on synthetic data, and real features may need
  more iterations.
- **Stage B, budgeted:** 12 fits per dataset at up to 60 s each, plus extraction and diagnostics, about 10 min (BABEL)
  and 25 min (LARa). The worst case adds about 30 min in total. The preflight smoke run times a real fit, and the dry
  run uses that measurement.

Launch (once implemented): `run_phase9.ps1 -RunId ts5_v1 -Hours 10`. Output:
`results/week5/timescale_inspection/ts5_v1/phase9/`.

## Implementation notes

- New files only: `trainer9.py` (the gated copy), `units9.py` (per-layer features, iteration-2 K-means, the I2 family,
  agreement and association measures), `run_phase9.py`, `report9.py`, `tests9.py`, `run_phase9.ps1`.
- Decoding with a non-default grouping passes the other family's grouping and codes to `build_mu0`. The embeddings are
  unchanged.

## Follow-up (not this phase): learned local tokens

A separate experiment would test learned local motion units directly:
- **Input:** the ordered per-frame relational trajectory inside each window (BABEL: 8 frames × 300 inter-joint
  distances) instead of T2's summary statistics.
- **Encoder:** window-local, with a receptive field that never crosses the window. It is trained without labels, and
  its output is quantized into tokens.
- **Downstream:** those tokens go through the unchanged contextualizer and OT.
- **Controls:**
  - a local-unit control (the same encoder with order destroyed by shuffling frames within each window);
  - T2 itself;
  - an independent evaluation set.

Only that design can support a claim about discovered, composable movemes.

## Open decisions

1. **HuGaDB:** T2 is undefined for IMU data; Phase 9 could run on T0 as iteration 0 there, at span 8. Default: left out.
2. **Iteration-2 codes as a tokenizer** for the code-input contextualizer: a different question (learned discretizer vs
   learned targets). Default: not run.
3. **Shared-trainer edit vs gated copy:** default is the gated copy (see above).

## Revision history

**Revision 2 (first evaluation).**
1. Scope and title narrowed to iterative contextual target refinement; local-unit claims moved to a follow-up.
2. Centroid-input control added.
3. Label-triggered third iteration removed.
4. Cross-seed NMI layer selector removed; layer 2 preselected; four-layer measures are diagnostics.
5. I1 and I2 decoded with the same groupings.
6. K-means fitting population stated, with transductive validation loss for every arm.
7. G9c and G9e corrected or added.

**Revision 3 (second evaluation).**
1. **Trainer.** The continuous-input path cannot be injected into the existing trainer or extractor. It is now a
   line-for-line copy with a `feature_of` option, chosen over a shared edit because the shared files' hashes are part
   of every Phase 5-8c identity. G9a now retrains R and requires identical checkpoint tensors, not only an identical
   re-decode.
2. **I2 artifact.** Its `lookup` is the 64-d per-code mean of T2's features. The 128-d teacher centroids are stored
   separately. G9f asserts the dimensional contract.
3. **Stage B.** Budgeted for four layers' K-means (12 fits per dataset), based on a timing benchmark. The first real
   fit is timed in preflight.
4. **G9c.** Now also checks continuous-input extraction against an independent extractor that uses continuous vectors.
5. **Q1a.** It conflated geometry with parameterization. The geometry-initialized ID arm (G) is added, so the ladder
   separates initial geometry (G vs R), parameterization (I1q vs G) and within-cell detail (I1 vs I1q). Initialization,
   chunks and masks are matched across arms (G9g), and G9b checks that visible continuous detail reaches I1's outputs
   but not I1q's.
6. **Budget.** About 7.9 h under a 10-hour cap, ordered by dataset, re-projected after the first timed jobs. BABEL-1
   stays a separate run.
