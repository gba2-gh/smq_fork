# Time-scale mismatch: inspection plan for BABEL and LARa

Revision 4 — 3 October 2026 (adds the correction after Phase 4 and points to `PHASE5_TOKENIZER.md`). Revision 3 — 2 October 2026. Phase 0 and the Phase 1 `screen`/`gate`/`inputs` stages are implemented and have
produced results (below); `confirm`, `beta` and `inputs_chosen` are implemented but not yet run; Phase 4 is
implemented but not yet run at scale (one smoke cell); Phases 2-3 are still design only. Changes from Revision 1:
added "Results obtained so far" after Phase 0.4's per-token probes showed the contextualizer adds nothing on
BABEL/LARa, this added a no-contextualizer decoding arm (`pca`/`centre`) to Phase 1 that the original Revision 1
design did not include (§"No-contextualizer input arms" under Phase 1), and the same two arms were carried into
Phase 4 so it also separates "more information in the representation" from "the contextualizer can use it"
(§"Implementation" under Phase 4).

## Objective

The frozen HuGaDB configuration does not transfer (`rkx_v2`, in progress). The working explanation is that every time scale in the pipeline was implicitly set for HuGaDB's segment lengths, and those scales are too long for BABEL's segments. This plan tests that explanation component by component, before anything is tuned.

The scales in question:
- the token window (patch length);
- the contextualizer's mask span;
- the OT kernel radius (the temporal smoothing neighbourhood).

The plan answers three questions, in this order:

1. **How much of the gap is a ceiling imposed by the scales themselves, independent of features?** Answered from labels only.
2. **Which component causes the loss:** the decoder radius, the mask span, the token window, or the input representation?
3. **Is there one label-free rule for these scales that works on all five datasets** without hurting HuGaDB?

It does not choose a final configuration. Each phase ends at a decision point for the user.

## Evidence so far

### `rkx_v2` (frozen HuGaDB configuration)

Validation-selected checkpoint, last rung (4,000), F1@50:

| | seed 111 / 222 / 1538574472 | mean | SMQ (local) | ASOT (published) |
|---|---|---|---|---|
| HuGaDB | 50.00 / 44.83 / 43.85 | 46.2 | 24.3 | 3.0 |
| BABEL-1 | 15.89 / 13.21 / 13.64 | 14.2 | 22.4 | 24.5 |
| BABEL-2 | 16.68 / 11.12 / 17.46 | 15.1 | 27.5 | 23.4 |
| BABEL-3 | 14.78 / 16.04 / 15.15 | 15.3 | 19.4 | 14.3 |
| LARa global | 5.03 (seed 111 only; the run is still going) | — | 16.4 | 5.7 |

The run is still going, so LARa per-recording normalization and the other LARa seeds are not in yet.

### Segment statistics (ground truth, `script/week5/segment_stats.py`)

| | median segment | mean | under 1 s | under 2 s | mask span (8 tokens) | span ÷ median |
|---|---|---|---|---|---|---|
| HuGaDB | 3.47 s | 4.96 s | 4.2% | 21.5% | 2.00 s | **0.58** |
| LARa | 2.20 s | 3.73 s | 14.3% | 43.5% | 1.92 s | 0.87 |
| BABEL-1 | 1.20 s | 1.91 s | 40.0% | 72.6% | 2.13 s | **1.78** |
| BABEL-2 | 1.70 s | 2.76 s | 28.9% | 58.5% | 2.13 s | 1.25 |
| BABEL-3 | 1.77 s | 4.06 s | 30.4% | 54.2% | 2.13 s | 1.21 |

### Symptoms in `rkx_v2` that point at scale

- **Under-segmentation.** The decoder predicts too few segments and uses too few states:

  | | predicted segments / rec. | GT segments / rec. | states used | classes present |
  |---|---|---|---|---|
  | BABEL-1 | 2.5 | 5.1 | 2.1 | 2.75 |
  | LARa | 15 | 32 | — | — |

- **The OT kernel is wider than most BABEL segments.** Its half-width is 3 tokens (±0.8 s), so it spans about 1.9 s.
- **ASOT's relative radius shows the mirror-image pattern.** ASOT smooths over 4% of each recording's length, which is about 0.3 s on BABEL and 4.8 s on LARa. It is strong on BABEL and weak on LARa and HuGaDB. Our fixed ~1 s does the opposite.

### Evidence against "it is only the contextualizer fit"

- **Padding is harmless.** Padded tokens are excluded from attention (`stageb_model.py:58`), chunks never cross recordings (`stageb_data.py:49`), and positions are relative (ALiBi). A 128-token context on a 30-token clip is therefore equivalent to a 32-token one; the padding only wastes compute.
- **Overfitting is real but does not explain the gap.** BABEL-2/3 validation loss rises after step 600–1,100. But BABEL-1 has as many tokens as HuGaDB (72k vs 76k), does not overfit (about 48 scorings per token vs about 150 on HuGaDB), and is just as poor (14.2).
- **LARa fits well yet scores 5 F1.** Its validation loss is a healthy 1.86.

### Confounds already visible

- **The span ratio does not order the BABEL subsets.** BABEL-1 has the worst ratio (1.78) and scores 14.2; BABEL-3 has 1.21 and scores 15.3. Data size, class mix and the `none` class vary at the same time.
- **The sensors differ.** HuGaDB is IMU data (accelerometer + gyroscope, already motion signals); LARa and BABEL are root-centred joint positions (LARa also has orientations).
- **BABEL has a background class.** `none` makes up 23–27% of BABEL frames and has no consistent motion.

## Hypotheses

| ID | Hypothesis | Predicts | Tested in |
|---|---|---|---|
| H0 | The token grid and smoothing make a high F1@50 impossible on BABEL, whatever the features | label-only ceilings on BABEL far below HuGaDB's | Phase 0 |
| H1 | The OT radius is too wide for BABEL (and possibly LARa) | a smaller radius raises F1@50 and the predicted segment count on BABEL, without hurting HuGaDB at its current radius | Phase 1 |
| H2 | A mask span longer than typical segments makes masked prediction a cross-action guess, so embeddings carry less action information | shorter spans raise action information in the embeddings and raise F1@50, most on BABEL-1 | Phase 2 |
| H3 | 0.27 s tokens are too coarse or too fine for BABEL's segment boundaries | changing the window changes F1@50 when span and radius are held fixed in seconds | Phase 3 |
| H4 | Position-only skeleton input lacks the motion signal that IMU input has | adding joint velocities helps BABEL and LARa, not HuGaDB | Phase 4 |
| H5 | The BABEL-specific background class, not scale, limits BABEL | most errors involve `none`; F1@50 without `none` is much higher | Phase 0 |

H1–H3 are the scale hypotheses. H4 and H5 are the main alternatives and are tested so a scale result is not over-read.

## Results obtained so far

`rkx_v2` finished (36/36 decodes, both port gates passed). Phase 0 (`ts0_v1`) finished. Phase 1's `gate` and `screen`
stages finished; `inputs` is nearly finished (missing BABEL-1 centre seed 1538574472 and both LARa normalizations at
the time of writing). `confirm`, `beta` and `inputs_chosen` have not run yet.

### `rkx_v2` final numbers (F1@50, selected checkpoint, mean of 3 seeds)

| | ours | SMQ (local) |
|---|---|---|
| HuGaDB | 46.2 | 24.3 |
| BABEL-1 / 2 / 3 | 14.2 / 15.1 / 15.3 | 22.4 / 27.5 / 19.4 |
| LARa global | 5.1 | 16.4 |
| LARa per-recording | 7.1 | 16.4 |

Declared reading: **does not transfer**.

### Phase 0: where the points are lost (BABEL-2 as the running example: 15.1 scored)

| step | F1@50 | loss |
|---|---|---|
| perfect token labels (0.1, 8-frame grid) | 96.9 | −3.1 from 100 (the grid itself) |
| + current ±3-token smoothing, still perfect labels (0.2) | 81.5 | −15.4 (the scale/radius effect) |
| what we actually score | 15.1 | **−66.4 (not explained by scale)** |

Per-dataset grid/smoothing ceilings and per-token action probes (0.4, balanced accuracy, PCA feature, 1-NN,
mean of 3 seeds) that support "the remaining loss is the representation, not the scales":

| | grid ceiling | + smoothing ceiling | our score | per-token probe (PCA) |
|---|---|---|---|---|
| HuGaDB | 100 | 98.2 | 46.2 | 0.68 |
| BABEL-1/2/3 | 97/97/97 | 79/82/78 | 14/15/15 | 0.52/0.52/0.42 |
| LARa | 99.9 | 88.1 (ASOT's 4%-of-length rule: 23.5) | 5.1-7.1 | 0.45 |

Segment-outcome classification (0.3) on BABEL-2: only 12% of true segments are hit, 24% are merged into a neighbour
(the smoothing failure), 49% get the wrong label even when correctly bounded (the feature failure) -- including 47%
of segments >=2s long, which no plausible radius would erase. Masked-loss-by-boundary-crossing (0.4) found BABEL-2
has *fewer* crossing targets than HuGaDB (17% vs 39%) yet a much higher within-segment loss (3.4-3.9 vs 1.45), so
crossing spans are not the main reason the contextualizer struggles there -- this demoted H2 (mask span) relative to
H4 (input representation).

**H5 (background class) not supported:** `none` is 23-27% of BABEL frames and 40-53% of errors (over-represented),
but excluding it from scoring does not consistently raise F1@50 (up on BABEL-1, mixed on BABEL-2/3).

### Phase 1 `screen` and `inputs`: the radius and the no-contextualizer arms

Screen (BABEL-2/3, current vs best setting found so far) and gate results are in `results/week5/timescale_inspection/ts1_v1/phase1/REPORT.md`
(gates passed exactly: HuGaDB and BABEL-2 current-radius re-decodes equal `rkx_v2`'s saved ladders, max diff 0.0).

The `inputs` stage (added after Phase 0's probes -- see "No-contextualizer input arms" under Phase 1) decodes the raw
PCA-64 feature (`pca`) and the K-means centre (`centre`) directly, with no contextualizer, at the current radius:

| | context (trained) | `pca` | `centre` |
|---|---|---|---|
| HuGaDB | **46.2** | 3.97 | 3.96 |
| BABEL-1 (2/3 seeds so far) | 14.3 | 16.7 | **17.4** |
| BABEL-2 | 15.1 | 18.7 | **19.7** |
| BABEL-3 | 15.3 | 17.4 | **17.6** |

**Reading:** on HuGaDB the contextualizer is the entire result (46 vs 4 without it -- without it the decoder
fragments into ~118 segments/recording against 10.5 true). On BABEL the contextualizer is a net *negative*: every
seed with both arms finished scores higher without it (by 2-4.6 points on the mean), though this falls short of the
plan's 3-point bar on BABEL-1/3 with only 2-3 seeds in. Quantization (`centre` vs `pca`) costs nothing on BABEL,
matching the 0.4 probes. Even the best BABEL arm (`centre`, ~17-20) stays below SMQ (19-28) and below published ASOT
on BABEL-1/2 (24.5/23.4), so this is not a fix -- it only shows the trained contextualizer is not the reason we are
below SMQ there.

### What this changes versus the original plan

- **H1-H3 (the scale hypotheses) are now secondary.** Real (Phase 0.2's ~15-point smoothing loss on BABEL), but they
  are not what separates us from SMQ; most of the loss is in the per-token representation.
- **H4 (input representation) is promoted** ahead of Phase 2/3 — see "Revised phase order" below.
- **Phase 1 gained a design element not in Revision 1:** the no-contextualizer `pca`/`centre` decoding arms, documented
  under Phase 1 below. These answer a question Revision 1 did not ask: whether the contextualizer helps *at all* on
  a given dataset, independent of its radius/span/window settings.

### Correction after Phase 4 (3 October): the tokenizer, not the contextualizer

Above, and in the Phase 4 analysis, the contextualizer was called "the consistent problem on skeleton data". That
over-reached. Every comparison used **one** tokenizer: a flattened raw patch, standardized, PCA-64, then Euclidean
K-means. The context-vs-`centre` gap only shows that the contextualizer hurts *with those codes*. Three findings
point upstream instead:
- **BABEL:** 68% of the dominant PCA variance is between-recording nuisance.
- **LARa:** PCA-64 lowers the linear probe from 0.529 to 0.466.
- **The contextualizer is token-sensitive.** Interventions that cleaned up the tokens (LARa per-recording
  normalization, velocity) helped the contextualizer specifically, not the direct arms.

The "decoder ceiling regardless of representation" claim is withdrawn: all representations shared the same tokenizer
design. The H4 probe verdict also depends on the classifier (linear vs 1-NN) and on whitening (BABEL-2 1-NN:
unscaled 0.508, whitened 0.625), not only on the representation. Span and window are not closed: Phase 0.4b was
indirect evidence, and BABEL-2 fails despite code–action MI close to HuGaDB's (0.66 vs 0.69). The follow-up is
`PHASE5_TOKENIZER.md`: whitened, relational, LARa-positions/PCA-256 and SMQ-encoder tokenizers, then span on the
best of them.

**Caveat on Phase 4's velocity result.** The first-frame velocity of each window is computed from the previous
window's last frame, so a masked token's code depends slightly on the visible token before it. This is weak (one
frame's difference, in addition to the dependence smooth motion already gives adjacent windows) but it makes the
contextualizer's task a little easier than the position-only family's. Phase 4's contextualizer gain (+3.1 to +3.2 on
BABEL-2 and LARa) is therefore an upper-end estimate. If velocity is reused, compute each window's first-frame
velocity within the window.

### Revised phase order (recommendation, not yet adopted as the plan's declared order; superseded by Phase 5)

1. Finish Phase 1 `inputs` (LARa), then `confirm`/`beta`/`inputs_chosen` (still informative: the decoder under-segments
   on BABEL regardless of input, so a better radius may still help the best no-contextualizer arm).
2. Phase 4 (velocities) before Phases 2-3: the per-token probes (0.4) are the one finding consistent across every
   piece of evidence so far, and only Phase 4 tests it directly.
3. A short Phase 2 check (BABEL-2/3 only, ~1h) to close out H2, since 0.4b is suggestive but not an intervention.
4. Phase 3 deferred; its ceiling bound (0.1) is already small (~3 points on BABEL), and a window change overlaps with
   Phase 4's representation question.

## Phase 0: label-only ceilings and diagnostics (no training, no decoding)

Everything here reads saved files: ground truth, and `rkx_v2` predictions and embeddings (`evals/<id>/predictions.npz`, `embeddings.npz`). All of it is label-aware **diagnostics** and never used to choose settings.

### 0.1 Ceilings from the token grid

Downsample each ground-truth sequence to the token grid by majority label per window, broadcast it back to frames, and score it against the original ground truth.

- **Windows:** 4, 8, 15 and 30 frames on BABEL (0.13 / 0.27 / 0.5 / 1 s). Use the matching windows on HuGaDB (15) and LARa (12).
- **Reading:** this is the best F1@50 possible if every token were labelled perfectly. 30 frames is SMQ's 1-s patch, so this also tells us what SMQ's grid costs.

### 0.2 Ceilings from smoothing

Apply a centred mode filter of half-width r tokens to the token-grid ground truth and score it.

- **Radii:** r = 1, 2, 3, 4 tokens, plus ASOT-style relative radii of 2 / 4 / 8% of recording length (with and without a 1-token floor).
- **Reading:** how much true structure a smoothing neighbourhood of that size destroys, even with perfect evidence. The OT term is not a mode filter, so this is an indicative bound, not an exact one. If r = 3 already costs BABEL many points on oracle labels, H1 is supported before any decoding.

### 0.3 Failure profile of the existing predictions

From `rkx_v2` predictions (3 seeds, both roles):

- **Segment lengths:** predicted vs ground-truth distribution per dataset (histograms, medians, share under 1 s).
- **Per-class results:** F1@50 and recall after the dataset-level Hungarian mapping.
- **Background errors (H5):** the share of errors involving `none`, and F1@50 recomputed with `none` frames excluded from scoring.
- **Clip length:** F1@50 stratified by clip-length tertile, to see whether short clips are the failures.
- **Error type:** missed short segments vs merged neighbours vs wrong labels on correctly bounded segments. A segment counts as merged when its frames carry the label of an adjacent ground-truth segment.

### 0.4 Action information per token, by representation

Measure how much a single token says about its action. Use a subject- or recording-grouped split (BABEL has no subject IDs, so group by recording) and two probes:
- a linear probe;
- a 1-NN probe.

Run them on four representations of the same tokens:

| Representation | Source |
|---|---|
| (a) PCA-64 patch features | saved preparation |
| (b) K-means centre of the patch's code | saved preparation |
| (c) one-hot code | saved preparation |
| (d) contextual embedding | `rkx_v2` selected checkpoint |

- **Masked-prediction split (H2):** for (d), also report the masked-prediction loss on validation tokens, split by whether the masked span crosses a ground-truth boundary. This takes one forward pass per checkpoint with the saved validation mask seeds, which is cheap.
- **Reading:** if (d) beats (a)–(c) on HuGaDB but not on BABEL, the contextualizer adds information on HuGaDB only, which fits H2.

**Decision point 0.** If the BABEL ceilings at the current grid and radius are already near our scores, the scale explanation is confirmed and Phase 1 becomes a calibration study. If the ceilings are high and errors are dominated by `none` or wrong labels, H5/H4 move ahead of H1–H3.

## Phase 1: OT radius, re-decoding existing checkpoints (no training)

### Design

Use the `rkx_v2` selected checkpoints and their saved embeddings. Re-run the full contextual OT fit (outer EM with cap 50, then the final ladder at rungs 500/1000/2000/4000) with only the kernel changed. The fit is repeated, not just the final solve, because the prototypes are fitted with the kernel in place.

Kernel settings (a radius of 0 means the temporal term is off):

| Family | Settings |
|---|---|
| Fixed, in tokens | 0, 1, 2, 3, 4 (3 is the current BABEL value, 4 is HuGaDB's and LARa's) |
| Relative (ASOT-style) | 2%, 4%, 8% of the recording's token count, floored at 1 token |
| Fixed, in seconds | ±0.25, ±0.5, ±1.0 s, rounded to tokens per dataset |

Everything else stays frozen: β = 0.3, a = 0.7, λ = 0.05, ε = 0.07, 25 inner steps.

The kernel weight is L / b (`transport.py:47`). A smaller b therefore increases the weight per neighbour while the total stays fixed. That is the existing normalization and is kept, so changing b changes *reach*, not total strength. The report states this.

### Matrix and order

1. **Screen:** BABEL-2 and BABEL-3, 3 seeds, all 11 settings. That is 66 fits at about 2.5 min each, roughly 2.8 h.
2. **Confirm:** BABEL-1, HuGaDB and LARa global (as many seeds as `rkx_v2` has finished), on three settings fixed in advance: the current one, plus two chosen as follows.
   - The best **fixed-seconds** setting and the best **relative** setting from the screen, judged by mean F1@50 on BABEL-2 only.
   - BABEL-3, BABEL-1, HuGaDB and LARa are then reported with those settings frozen.
   - Cost: about 9 × 21 min on BABEL-1, 9 × 8.5 min on HuGaDB and up to 9 × 17 min on LARa, roughly 7 h in total.
3. **β cross-check:** β ∈ {0.1, 0.3, 0.5} on BABEL-2 at the two selected radii, 3 seeds. That is 18 fits, about 45 min. It shows whether the radius result is really a strength result.

### Reported per setting

- MoF, Edit, F1@10/25/50;
- predicted vs ground-truth segments per recording, and states used;
- capped-fit counts;
- the change against the current radius, paired by seed.

### Reading rule (declared now)

- **H1 supported:** on BABEL-2 the best setting beats the current radius by at least 3 F1@50 points on the mean, in all 3 seeds; and the held-out BABEL-3 improves in the same direction.
- **Transfer check:** the frozen setting does not lower HuGaDB by more than 3 points (the observed seed spread is about 6).
- **A cross-dataset rule exists:** one family (fixed seconds or relative) is within 1 point of the best setting on every dataset.

**Decision point 1.** Pick the radius rule to carry into Phases 2–4, or keep the current one if H1 is not supported.

### No-contextualizer input arms (added after Phase 0; not in the original Phase 1 design)

Phase 0.4's per-token probes found the contextualizer adds little or negative information on BABEL and LARa. This is
a question Phase 1 did not originally ask: it answers "does the contextualizer help at all here", separate from which
radius/span/window it is tuned with. Implemented as two more arms of the same decoder, alongside the radius settings:

| arm | input, no checkpoint involved | repository equivalent |
|---|---|---|
| `pca` | the patch's own PCA-64 feature, mean-subtracted, L2-normalized | `direct_continuous` |
| `centre` | the K-means centre of its code, same treatment | `direct_quantized` |

Everything else (OT fit, solver settings, rungs, scorer) is identical to the contextual decode, so a gap against the
`context` arm at the same radius isolates the contextualizer's contribution.

Two new stages:
- **`inputs`:** `pca` and `centre` on every dataset (LARa both normalizations), 3 seeds, current radius. 54 units,
  36 real decodes (18 `context`-radius baselines are free aliases of the already-decoded `rkx_v2`/screen checkpoints).
- **`inputs_chosen`:** the same two arms at the two radii chosen in Decision point 1 (needs `selection.json` from the
  screen).

A dedicated gate (`direct_pca_reproduces_task2`) checks the `pca` arm on HuGaDB/global/seed 111 against Task 2's
saved `direct_continuous` decode of the same `raw_kmeans` family before either stage runs. It passed exactly
(max difference 0.0) on first launch.

Reading rule (same 3-point bar as Decision point 1, applied per dataset, paired by seed): the contextualizer "helps"
on a dataset if `context` beats both `pca` and `centre` by ≥3 points on the mean and in every seed; it "hurts" if
either no-contextualizer arm beats `context` by the same bar. Results so far are in "Results obtained so far" above.

## Phase 2: mask span (retraining, BABEL-cheap)

### Design

Retrain the contextualizer with mask span 2, 4 and 8 tokens (8 exists in `rkx_v2`). The mask fraction stays at 0.30 and everything else is frozen. Decode each checkpoint at two radii: the current one and the Decision-1 rule. That separates the span effect from the radius effect.

| Dataset | Spans | Seeds | Training | Decoding |
|---|---|---|---|---|
| BABEL-2, BABEL-3 | 2, 4 (new) + 8 (reuse) | 3 | 12 runs × ~1.5 min | 36 decodes × ~2.5 min |
| BABEL-1 | 2, 4 (new) | 3 | 6 × ~2 min | 12 × ~21 min |
| HuGaDB (control) | 4 (new) | 3 | 3 × ~2.7 min | 6 × ~8.5 min |

That is about 6.5 h, dominated by BABEL-1 decoding. If time is short, BABEL-1 can run with 1 seed first.

### Also recorded

- Phase 0.4 probes on the new embeddings: per-token action information, and the loss split by boundary crossing.
- The validation-loss curve, to see whether shorter spans also delay overfitting on BABEL-2/3.

### Reading rule

- **H2 supported:** on BABEL-2 the best shorter span beats span 8 by at least 3 points (mean, all seeds) at the same radius; the probes show more action information per token; and BABEL-1 and BABEL-3 move in the same direction.
- **The span ratio matters:** HuGaDB at span 4 (ratio 0.29) does not drop by more than 3 points.

**Decision point 2.** Choose the span rule (in tokens, seconds, or relative to recording length).

## Phase 3: token window (preparation + retraining)

### Design

BABEL windows of 4, 8 and 15 frames (0.13 / 0.27 / 0.50 s). The span and radius are held **in seconds** at the Decision 1–2 values, so only the grid changes. Each window needs a new tokenizer (PCA-64 + K-means-512, about 10–20 s), training and decoding.

At 4 frames the token count doubles, so the tokens per clip (about 54–86) still fit in the 128 context. At 15 frames a clip is about 14–23 tokens. In that case K = 512 can exceed what the data supports, so K = 256 is recorded as a secondary arm.

### Matrix

BABEL-2 and BABEL-3, 3 seeds, windows 4 and 15 (8 exists): 12 preparations, 12 trainings, 12 decodes, about 1 h. BABEL-1 at the selected window only, if it differs from 8.

### Reading rule

- **H3 supported:** a window other than 8 changes F1@50 by at least 3 points on BABEL-2 (mean, all seeds), in the same direction on BABEL-3.
- **Ceiling check:** the Phase 0.1 ceiling for that window must allow the gain; if it does not, the result is suspect.

## Phase 4: input representation (H4, runs alongside Phase 3)

### Design

Raw patches of positions vs positions plus first differences (velocities, frame-rate scaled) for BABEL and LARa. HuGaDB stays IMU-only as the control, since it is already derivative data. Same tokenizer recipe, same contextualizer, Decision 1–2 settings.

The new patch dimensions:

| | positions only | positions + velocities |
|---|---|---|
| BABEL | 600 | 1,200 |
| LARa | 1,584 | 3,168 |

PCA-64 keeps the token dimension fixed.

### Matrix

BABEL-2, BABEL-3 and LARa global, 3 seeds, one new arm. About 1 h on BABEL plus about 1.5 h on LARa (decoding dominates).

### Implementation (added when Phase 4 was built; extends the "one new arm" above to three)

`script/week5/timescale_inspection/velocity_prepare.py` builds the velocity family (`raw_kmeans_velocity`) with the
same sampling/PCA/K-means/split/grouping recipe as `raw_kmeans_cross_dataset.prepare`; with velocity switched off it
is byte-identical to that module's `extract_raw` (tested directly). `run_phase4.py` trains the frozen reference
contextualizer on it (reusing `train.run_training_job` unchanged) and decodes three arms, not one, reusing Phase 1's
machinery directly:

- `context` -- the trained contextualizer's embedding (needs training, like the original design);
- `pca_velocity` / `centre_velocity` -- the same no-contextualizer decode Phase 1 added for the position-only family
  (`run_phase1.fit_and_score`), built on the velocity feature / its K-means centre instead. Free to add once the
  family exists, and they isolate whether a gain comes from the representation itself or from the contextualizer
  learning something new on it.

The radius defaults to `current` (Decision 1 is not finalized); `--radius` can select a Phase 1 setting once it is.
The position-only baseline is read from already-reported results rather than rebuilt: `context` from `rkx_v2`,
`pca`/`centre` from Phase 1's `inputs` stage -- rebuilding it would add a second, independent vocabulary-seed draw
and make the paired comparison noisier, not cleaner (the vocabulary seed alone is known to move F1 by 8-12 points).
HuGaDB is not rebuilt at all, exactly as designed above.

### Reading rule

- **H4 supported:** velocities add at least 3 points on BABEL-2 or LARa (mean, all seeds), and the Phase 0.4 linear probe on the PCA features shows more action information per token.
- Applied per arm (`context`, `pca_velocity`, `centre_velocity`) against its position-only counterpart, paired by seed: this separates "the representation carries more action information" (the `pca`/`centre` comparison) from "the contextualizer can also exploit it" (the `context` comparison), which the original one-arm design could not.

## LARa track

LARa's failure is not mainly a scale problem: its span ratio is 0.87, yet it scores 5 F1@50 with a good validation loss. It shares Phase 0 (ceilings, failure profile, probes) and Phase 1 (radius), where ASOT's 4.8 s relative radius predicts damage. Its likely specific causes are:
- 32 segments and 6.4 classes per recording;
- one class (Handling, centred) covering 44% of frames;
- upper-body handling actions that differ only subtly;
- two normalizations still running in `rkx_v2`.

Further LARa work is planned only after Decision 0, from its failure profile.

## Selection and leakage rules

- **Phase 0 is label-aware by design and selects nothing.** It sets bounds and explains failures.
- **Settings are chosen on BABEL-2 only.** It is the cheapest subset and has the median span ratio. BABEL-1, BABEL-3, HuGaDB and LARa are reported with the chosen settings frozen. BABEL-2 numbers are labelled "development" in every table.
- **Candidate rules are label-free.** They are defined in tokens, seconds, or as a fraction of recording length, never from ground-truth segment lengths. The segment statistics above motivate the grids but are not inputs to any rule.
- **Every comparison is paired by seed,** with all 3 seeds reported. The 3-point bar is set from the observed spread: up to 6 points on HuGaDB and BABEL-2 seed-to-seed, and about 5 points between checkpoints.
- **Only the validation-selected checkpoint is primary.** The final checkpoint is secondary, as in `rkx_v2`.

## Implementation notes

- **Package:** `script/week5/timescale_inspection/`, with one entry point per phase. Write to `results/week5/timescale_inspection/<run_id>/`; never write into `rkx_v2`.
- **Radius override:** `categorical.recording_tensors` computes the half-width internally from `rec.fps` and `rec.window` (`categorical.py:108`). Add an explicit half-width function to the decode path instead of patching globals. The function becomes part of the evaluation identity, so a re-decode can never be mistaken for an `rkx_v2` record.
- **Reuse:** read `rkx_v2` preparation, checkpoints and embeddings through the existing loaders, verifying their manifests and checkpoint hashes. As in `rkx_v2`, preparation is pinned to 32 threads.
- **Regression gate:** re-decoding HuGaDB seed 111 at the current radius through the new path must reproduce `rkx_v2`'s 50.00 exactly before anything else runs. BABEL-2/seed 111 is gated the same way. Both passed exactly.
- **Direct-input gate:** the `pca`/`centre` arms (above) have their own gate against Task 2's saved `direct_continuous` decode of the same `raw_kmeans` family on HuGaDB/global/111. Passed exactly.
- **Ceilings:** Phase 0.1–0.2 use the repository scorer with the same dataset-level Hungarian matching (for oracle labels the mapping is the identity, which is checked).
- **Tests:**
  - the radius override at the current value equals the existing kernel;
  - the relative radius is floored correctly on a 16-token clip;
  - a mode filter with r = 0 leaves the labels unchanged;
  - the token-grid ceiling at a window of 1 frame scores 100;
  - velocity features have the expected shape, and the first frame is handled by zero padding.

## Runtime summary

| Phase | Training | Decoding | Approximate wall time |
|---|---|---|---|
| 0 | — | — | under 1 h (CPU; probes dominate) |
| 1 | — | 66 (screen) + ~27 (confirm) + 18 (β) + 36 (inputs) + 36 (inputs_chosen) fits | ~14 h (screen 2.8 h, confirm ~7 h, β 0.75 h, inputs ~2.5 h, inputs_chosen ~2.5 h) |
| 2 | 21 runs | 54 decodes | ~6.5 h (BABEL-1 dominates) |
| 3 | 12 runs | 12 decodes | ~1 h |
| 4 | 9 runs | 9 decodes | ~2.5 h |

Each phase is launched separately by the user after the previous decision point. Phase 1 can start as soon as Phase 0 is reviewed. Its LARa part needs `rkx_v2` to finish first.

## Out of scope

- Model-size and dropout sweeps for BABEL-2/3 overfitting. These come only if Phase 2 leaves a gap between BABEL-1 and BABEL-2/3.
- Native VQ, FSQ or learned-encoder tokens on BABEL/LARa.
- Changing the number of OT states away from the number of classes.
- Any per-dataset tuning of a, λ or ε.
