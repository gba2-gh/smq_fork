# PROTOCOL — segmentation-first motion representation, first execution block

Written **before** the full Experiment 1 runs were computed; nothing below was tuned on evaluation labels.
Transparency note: one smoke run of the finished Experiment 1 code on BABEL-3 (the smallest set) was executed
to debug the pipeline before this file was written. No design choice was changed as a result, and BABEL-3 is
re-run with identical frozen code for the reported results.

Research question: *does segmenting motion into meaningful temporal intervals before quantization preserve
useful behavioural structure better than quantizing fixed-duration patches?*
**H1 (segmentation):** meaningful boundaries give better segment representations than fixed or randomized boundaries.
**H2 (discretization):** quantizing those representations gives a modelling advantage over continuous embeddings.
Neither hypothesis, if supported, establishes action atoms, syllables or semantic primitives.

## 1. Scope

* Method: SMQ (released checkpoints) plus controlled changes to the segmentation/quantization stages.
* Datasets: LARa (8 classes, 439 recordings, 16 participants) and the three existing BABEL subsets (K = 5 each).
  HuGaDB is out of scope. No new annotations, architectures or datasets.
* Existing annotations are used only for (a) evaluation and (b) the explicitly labelled **oracle** partition.

## 2. Setting (named accurately)

**Transductive.** The frozen encoder (released weights, `models/pretrained/<dataset>.model`) was trained without
labels on all recordings of the dataset, which include every evaluation recording. Fitting the scaler and the
k-means codebook on a subset of recordings does **not** make the encoder inductive. Any claim from this block
is a claim about the segment representation given a transductively-trained frozen encoder.

**Folds.** LARa: 4 folds, each holding out 4 whole participants (participant permutation seed 0). BABEL: 5 folds of
recordings (seed 0); no participant identifiers exist. For each fold, the scaler and k-means are fitted on the other
folds' recordings and applied unchanged to the held-out recordings. Everything is evaluated out-of-fold and pooled.
The three BABEL subsets are analysed separately, are **not independent replications** (they share recordings and
almost certainly participants/source collections), and BABEL-1 (1,960 recordings) is the primary BABEL set.

## 3. Shared segment representation (identical for every condition)

1. Frame-level latent `z` (T × D; D = 352 LARa, 400 BABEL) = the VQ input of the frozen encoder, one row per native
   frame (encoder: two stages × dilations 1,2,4, kernel 3 ⇒ receptive field ±14 frames; edge frames see zero padding).
2. Per-dimension standardisation, mean/SD fitted on **fitting-recording frames only**, same for all conditions.
3. Each segment's normalised extent is split into **8 bins**; L ≥ 8 frames: exact area-average of the piecewise-constant
   frame signal over each bin; L < 8: linear interpolation at bin centres; flatten in temporal order (8·D).
   Verified on controlled examples (`script/seg/tests.py`): monotone ramps stay monotone, reversal flips bin order, time
   reversal is much farther than a speed change, an asymmetric shape is separated from its reversal at equal mean.
   The bin count is a default, **not swept**.
4. Distance: Euclidean on the flattened vector. Clustering: k-means, `K` = the dataset's existing action-class count
   (LARa 8, BABEL 5), k-means++ with 5 inits, 3 clustering seeds (0,1,2), each segment one unweighted sample
   (duration-weighted fitting is a sensitivity, `--weighting duration`). K = C is a benchmark-comparability budget, not
   a claim about the number of primitives. A 4K sensitivity is deferred until after the primary result.
5. **Prototype** of a segment = its k-means centroid. "Continuous" = the 8·D vector; "quantized" = the prototype.
6. Reference: **Original SMQ** = the released codes (one code per 1 s patch), never mixed with the controlled
   conditions. Three levels are kept distinct: (i) original SMQ, (ii) controlled representation on fixed windows,
   (iii) controlled representation on alternative boundaries.

**Fixed windows:** one-second grid (LARa 50, BABEL 30 frames); the final partial window is kept as its **own shorter
segment** (LARa has none: all recordings are exactly 6,000 frames).
**Oracle:** maximal constant-label runs of the existing annotation (every class incl. `None`/`none` is an ordinary
segment; runs are never split or discarded).
**Random (oracle-matched):** permute the oracle durations within each recording and re-accumulate (exact cover).
10 prespecified randomizations (seeds 1000–1009, per recording). Reported: mean over randomizations, SD across
randomizations, and per recording the number of distinct arrangements, duration CV and fraction of randomizations equal
to the oracle. Uses annotation-derived durations ⇒ an **oracle-matched diagnostic, not an unsupervised method**.

## 4. Experiment 1 — oracle boundary diagnostic

Conditions: Fixed, Oracle, Random; Original SMQ as reference. Continuous and quantized readouts for each.

**Primary endpoint — cross-recording retrieval, common-anchor evaluation (principal cross-condition result).**
Anchors: one timestamp per second of recording, uniformly jittered inside its second, drawn from a seeded RNG
independent of segmentation and labels (`ANCHOR_SEED=20260921`). An anchor is represented by the segment containing
it and carries the existing frame label at that timestamp. Queries and galleries are the same anchors for every condition
(sha256 stored; identical `n` per recording/class asserted). Gallery excludes the query's group: **participant (LARa)**,
**recording (BABEL, no participant ids)**. Metric: **class-balanced precision@10** (k = 10 gallery anchors; items carrying
several anchors are used fractionally at the cut-off = expected precision under random tie-breaking); mean over classes
with ≥ 20 queries; the random-ranking **chance** level under the same exclusion is reported and `lift = p − chance`.
Per-class precision, query counts and chance are exported.

**Complementary — segment-level retrieval.** Query/gallery = segments with action purity ≥ **0.8** (the same rule for all
conditions); relevance = majority label. Sensitivity: purity ≥ 0 (all segments, mixed-label included). Per-class
coverage (queries per class) reported.

**Secondary.** Permutation-invariant NMI and ARI between clusters and labels (frame level, per fold, mean over folds);
train-fitted cluster→label transfer (majority label on **fitting** recordings, applied to held-out recordings; uses fitting
labels); standard benchmark mapping (fold-level Hungarian, uses evaluation labels, reported separately); continuous-versus-
quantized retrieval difference.

**Uncertainty.** Paired resampling (2,000 draws, seed 12345) of independent **groups** (participants for LARa; recordings
for BABEL) with replacement; the same draw is applied to every condition. Point estimate and 95 % percentile interval of
differences. Clustering-seed SD (3 seeds) and randomization SD (10 draws) are reported **separately**; frames, segments
within a recording, and overlapping BABEL subsets are never treated as independent. The Hungarian mapping is re-estimated
on every draw for MoF; F1 mappings (Experiment 2) are held fixed at the point-sample mapping (interval is conditional on it).

**Implementation checks (asserted / saved in the results):** every frame in exactly one segment; durations sum to recording
length; oracle/random share count and duration multiset; labels never enter scaler or k-means (they enter only through the
declared oracle partition and evaluation); common-anchor query/gallery identical across conditions; no participant leakage
between fitting and evaluation folds (LARa); original-SMQ frame codes reproduce the cached E0 predictions exactly (0 mismatches).

**Interpretation rules (fixed in advance).**
* H1 supported at the tested representation iff oracle > random **and** oracle > fixed on the continuous common-anchor
  lift with paired 95 % intervals excluding 0. Oracle and random both > fixed, oracle ≈ random ⇒ pooling duration/rate.
* Discretization problem iff the continuous oracle advantage is present while the quantized one is absent (interval includes 0).
* Little oracle benefit (interval includes 0 or effect negligible vs. seed SD) ⇒ weakens the case for this recipe,
  conditional on this representation. It is not evidence that no segmentation-first representation can work.
* Oracle scores are never read as boundary discovery.

## 5. Experiment 2 — one label-free segmentation method (feasibility pilot)

**Objective** `Σ_s LinearFitError(z_{a_s:b_s}) + λ · N_segments`, minimised **exactly** by dynamic programming over segment
ends (O(T · (L_max − L_min))), on the standardised frozen latent trajectory (§3, same scaler as the fold).
* `LinearFitError` = residual sum of squares of a per-dimension least-squares line against time, **summed over the
  segment's frames and averaged over the D dimensions** (computed in float64 from prefix sums, closed form). Averaging over D
  makes the scale independent of latent width (352 vs 400 dims); units are standardised-variance × frames, so λ has the
  same meaning in both datasets.
* Bounds: min **0.25 s** ⇒ `ceil(0.25·fps)` frames (LARa 13, BABEL 8); max **10 s** ⇒ `floor(10·fps)` (LARa 500, BABEL 300).
  Engineering constraints, not behavioural durations. Fraction of segments at each bound is recorded.
* λ: chosen **per fold on unlabelled fitting recordings only** (all fitting recordings; see Deviations D1) by bisection on log λ so
  the mean segment rate matches the one-second grid's rate on the same recordings (±2 %); frozen before any evaluation label
  is inspected. If the rate cannot be matched, the achieved rate is reported and no label-based tuning is done.
* Exactness validated against brute force on 36 small exactly solvable cases (`tests.py`); no approximation is used.

Conditions (same scaler, K, fit policy, 3 seeds, metrics): (1) Original SMQ; (2) Fixed windows + shared encoder + k-means;
(3) DP variable-duration segments + shared encoder + k-means; (4) Random partitions matched to (3)'s per-recording count and
duration multiset (10 randomizations).

**Primary endpoint: segmental F1@50** (published MS-TCN/SMQ implementation, `src/model/eval_utils.py`) under the fold-level
Hungarian benchmark mapping (uses evaluation labels — a benchmark convention, not deployment). Secondary: F1@10/25, Edit,
MoF, common-anchor retrieval, class-agnostic boundary precision/recall (**one-to-one** matching, ±0.5 s; ±1 s sensitivity;
empty-prediction and empty-ground-truth recordings contribute FN / FP; boundaries = code changes of the frame-level
cluster sequence; raw cut points are also exported), and the train-fitted-mapping version of every segmental metric.
Duration histograms and bound saturation are descriptive only.

Pilot: one released checkpoint per dataset, 3 clustering seeds; small seed-sensitive differences are not findings.

**Continuation rule (project decision threshold, not a significance criterion).** On a dataset, "advance" requires all of:
(a) F1@50(DP) − F1@50(Fixed) ≥ **2.0 points** with paired 95 % interval excluding 0;
(b) Edit difference ≥ 0 and MoF difference ≥ −1 point ("compatible");
(c) the matched random partition does not explain it: F1@50(DP-random) − F1@50(Fixed) < ½ (DP − Fixed), and
DP − DP-random has a paired interval excluding 0.
Success on one dataset with a modest unresolved effect on the other justifies targeted replication only, stated as such.
An isolated best score on one metric or dataset is not an advance.

## 6. Experiment 3 — not launched in this block

Would only start if the earlier experiments give a credible reason. Prerequisites recorded in the manifest: causal segment
extraction (or fixed observed prefix), scaler/segmentation/codebook fitted on training recordings only, **recordings excluded from
encoder training** (no released/T1 checkpoint qualifies except the four LARa subject-split encoders of E2), persistence and
constant-velocity baselines, one fixed horizon chosen on development data, independent final test split.

## 7. Stopping rule

After the manifest, Experiments 1–2 and the decision report the block stops. No replication with retraining and no
Experiment 3 are launched. If unsupported, the hypothesis that failed is identified; no automatic new losses, larger codebooks,
new datasets or hierarchy are added.

## 8. Deviations from this protocol (added chronologically; none was made after inspecting an evaluation metric of the affected experiment)

* **D1 (Experiment 2, lambda selection sample).** The protocol first specified matching the event rate on at most 40 evenly
  spaced fitting recordings. A debugging run of the finished Experiment 2 code on BABEL-3 showed, from unlabelled rates alone,
  that the 40-recording subsample matched the full fitting-set rate only within about 9 % (e.g. 1.12 vs 1.03 segments/s). The rule
  was therefore changed to match on **all** fitting recordings (computationally cheap). Only unlabelled rates were inspected; no
  Experiment 2 F1/Edit/retrieval value had been looked at when the change was made, and BABEL-3 is re-run under the final rule.
