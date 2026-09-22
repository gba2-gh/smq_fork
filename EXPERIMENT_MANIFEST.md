# EXPERIMENT MANIFEST

Data exposure, splits, checkpoints, evaluation conventions and unresolved limitations for the segmentation-first
programme. Sources: `results/CONSOLIDATED_EXPERIMENT_REPORT.md`, `results/characterization/REPORT.md`, the run
configurations under `models/`, and a fresh audit of the data (`script/seg/audit_data.py` →
`results/seg/data_audit.json`). Companion files: [PROTOCOL.md](PROTOCOL.md), [RESULTS.md](RESULTS.md),
[DECISION.md](DECISION.md). Environment: Python 3.10.14, torch 2.1.0+cu121, one RTX 4090; repo commit `f389b8b` plus the
uncommitted working tree (see "Provenance" below).

## 1. Data

| | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---:|---:|---:|---:|
| Recordings | 439 | 1,960 | 255 | 183 |
| Native fps (preprocessed) | 50 (200 Hz ÷ 4) | 30 | 30 | 30 |
| Input per frame | 22 joints × 6 channels (channels 3:6 hip-relative xyz, verified zero at the root; the physical meaning of channels 0:3 is not inferred) | 25 joints × 3 (xyz, mm) | same | same |
| Recording length | **all exactly 6,000 frames = 120 s** | 41–999 frames (median 209 = 7.0 s) | 38–996 (median 255 = 8.5 s) | 75–990 (median 341 = 11.4 s) |
| Total frames | 2,634,000 | 568,435 | 79,118 | 82,424 |
| Action classes C (=K) | 8 | 5 | 5 | 5 |
| Classes | Standing, Walking, Cart, Handling(up), Handling(centred), Handling(down), Synchronization, None | walk, stand, turn, jump, none | sit, run, stand_up, kick, none | jog, wave, dance, gesture, none |
| Frame share of the background-like class | None 4.2 %, Synchronization 1.8 % | none 22.9 % | none 27.4 % | none 23.1 % |
| Existing annotation runs / recording (mean, median) | 32.1, 32 | 5.1, 5 | 3.7, 3 | 3.7, 3 |
| Recordings with a single run | 1 | 122 | 53 | 11 |
| Recordings with < 4 runs | 3 | 639 (33 %) | 146 (57 %) | 148 (81 %) |
| Median run duration (s) | 2.2 | 1.2 | 1.7 | 1.8 |
| Participant identifiers | **yes** — 16 participants from the file name `L<scenario>_S<participant>_R<rec>` (14–30 recordings each; 3 scenarios) | **none** | **none** | **none** |

**Background handling.** There is no separate background label: `None` (LARa) and `none` (BABEL) are ordinary classes
counted in C, exactly as in the benchmark protocol (`get_labels_start_end_time` is called with a `bg_class` that matches
no label). The programme keeps them as classes. BABEL preprocessing dropped sequences with > 50 % `none` frames. LARa
frames with no active one-hot label were written as `None`. Annotation-derived "runs" are maximal constant-label
intervals, so a `none` interval is a run like any other.

**Lengths and annotation density matter.** LARa is long, dense (≈ 32 annotated runs per recording) and has participant
identifiers. BABEL is short and sparse: the duration-permutation null of Experiment 1 is degenerate for a large share of
BABEL recordings (a single run: permutation is the identity; with n runs there are ≤ n! arrangements). The number of
distinct arrangements, duration CV and the fraction of randomizations identical to the oracle are stored per recording
(`raw/exp1_*.pkl`, `meta.checks.null_informativeness`) and a sensitivity restricted to "informative-null" recordings (≥ 4 runs
and duration CV ≥ 0.25) is reported.

### BABEL subset overlap

Subsets were compared by recording id and by SHA-1 of the feature array:

| pair | shared recording ids | of which identical features | share of the smaller subset |
|---|---:|---:|---:|
| BABEL-1 ∩ BABEL-2 | 3 | 3 | 1.2 % |
| BABEL-1 ∩ BABEL-3 | 7 | 7 | 3.8 % |
| BABEL-2 ∩ BABEL-3 | 2 | 2 | 1.1 % |

Recording-level overlap is therefore small (exact-duplicate features only). **Participant-level overlap cannot be determined**:
the BABEL sequence ids and the data preparation (`src/data/babel.py`) carry no subject identifier, so recordings from the same
performer may sit in different subsets or in different folds of one subset — this is unresolved, not shown to be absent. The three
subsets are different action vocabularies built from one common source collection (BABEL); they must **not** be treated as three
independent dataset replications. BABEL-1 is the primary BABEL set (largest); BABEL-2/3 are reported as
related but non-independent subsets. Because BABEL has no participant identifiers, BABEL retrieval excludes only the query's own
**recording**, not its participant; participant-level leakage between fitting/evaluation recordings is unresolved for BABEL.

## 2. Encoder: output shape and temporal support

* Encoder = `MultiStageModel` (two single-stage TCNs, each 1×1 conv → 3 dilated residual layers, dilations 1, 2, 4, kernel 3 → 1×1 conv),
  applied independently to every joint (`src/model/smq.py`). Output (per recording): **(T, D)**, one 16-d latent per joint per native
  frame, concatenated over joints: **D = 352 (LARa), 400 (BABEL)**. This is the vector that is patch-quantized (the VQ input).
* Temporal support: each stage has radius 1+2+4 = 7, so the encoder has **receptive field ±14 frames (29 frames)** per output frame
  (same value as reported in the characterization study). Zero padding is used at recording edges (mask = 1 everywhere at inference).
  Frame *t* therefore depends on raw frames *t−14 … t+14*. The latent trajectory used for segmentation and segment encoding is
  the **full-length** latent; nothing is cropped.
* Original SMQ quantization: non-overlapping 1 s patches over the latent (`W` = 50 / 30 frames), Euclidean-summed patch distance to K codes,
  the final partial patch zero-padded in latent space. The frame-level codes reproduce the cached E0 predictions exactly
  (0 mismatches in 2,634,000 / 568,435 / 79,118 / 82,424 frames, `script/seg/extract.py`).

## 3. Checkpoints and what they were trained on

All SMQ training in this repository is **unsupervised and transductive**: `BatchGenerator.read_data()` reads every file in
`data/<dataset>/features/`, so every checkpoint below except the E2 subject splits was trained on **all recordings of its
dataset, including any recording later used for evaluation or fold-level codebook fitting**.

| Checkpoint | Dataset(s) | Trained on | sha256 | Use here |
|---|---|---|---|---|
| `models/pretrained/lara.model` | LARa | all 439 recordings (undocumented in repo; inferred — see below) | `a45392ad…fa65` | **Primary LARa encoder (frozen)** |
| `models/pretrained/babel1.model` | BABEL-1 | all 1,960 (inferred) | `f8bba85e…a6e6` | **Primary BABEL-1 encoder** |
| `models/pretrained/babel2.model` | BABEL-2 | all 255 (inferred) | `d7f580fa…7ca6` | related non-independent subset |
| `models/pretrained/babel3.model` | BABEL-3 | all 183 (inferred) | `fd40db56…1f72` | related non-independent subset |
| `models/exp/T1/lara/K8_s{1538574472,111,222}` | LARa | all 439, K=C=8, 30 epochs | see `results/characterization/manifest.json` | seed replication if the direction is supported |
| `models/exp/T1/babel1/K5_s{1538574472,111}` | BABEL-1 | all 1,960, K=5 | — | seed replication (2 seeds) |
| `models/e2_splits/lara_split{0..3}/lara/epoch-30.model` | LARa | **13 participants only** (349–368 recordings); test participants {06,08,10}, {02,06,14}, {01,08,13}, {03,08,11} unseen | — | the only participant-held-out encoders; candidates for Experiment 3 |
| `models/e0_seeds/…`, T3/T4 checkpoints, HuGaDB models | — | all recordings | — | not used |

Full hashes are in `results/seg/cache/<dataset>/_meta.npz` (`ckpt_sha256`) and `results/seg/frozen_code_hashes.txt`.
Provenance limits: the training set of the *released* weights is not documented in the repository. It is inferred from (i) the
published protocol (train and evaluate on the same benchmark collection) and (ii) the E0 result that these weights reproduce
the published transductive scores to within 0.71 points (LARa 37.38/39.40/34.69/28.40/16.39). The released checkpoints
therefore must be treated as **having seen every evaluation recording**. Historical bit-identity of a released checkpoint
with the paper's run is not established (see the consolidated report).

## 4. Splits used in this programme

Setting: **transductive**. Splitting recordings only governs the *scaler* and the *k-means codebook* fitted on top of the frozen
encoder; it does **not** make the encoder inductive.

* LARa: 4 folds, each holding out **4 whole participants** (participant permutation `default_rng(0)`); scaler and k-means are fitted on the other
  12 participants; no participant appears in both fitting and evaluation (asserted).
* BABEL-1/2/3: 5 folds of **recordings** (`default_rng(0)` permutation); participant identity unknown, so fitting and evaluation
  folds may share participants.
* Retrieval galleries: same-fold evaluation anchors of other **participants** (LARa) or other **recordings** (BABEL).
* All reported numbers are out-of-fold and pooled; there is no separate untouched test split in this block. **Forecasting (Experiment 3) needs
  recordings excluded from encoder training** (not provided by any released/T1 checkpoint).

## 5. Evaluation conventions

* Segmental F1@{10,25,50}, Edit, MoF: the repository's MS-TCN-derived `f_score` / `edit_score` (`src/model/eval_utils.py`), unchanged.
  Known convention: `f_score` lets a prediction whose best-IoU ground-truth segment was already matched count as a false positive rather
  than matching another segment; this is the published behaviour and was kept for comparability. Frame-level metrics include every class
  (`None`/`none` are not excluded).
* Cluster→label mapping: (a) **fold-level Hungarian** on the fold's evaluation recordings (uses evaluation labels; benchmark convention, closest
  to the published dataset-level Hungarian, and reported separately); (b) **train-fitted** majority-label transfer from fitting recordings (uses
  fitting labels only). Published-protocol numbers (dataset-level Hungarian on all recordings, `results/e0`) remain the reference for the original
  SMQ and are not mixed with these.
* Boundary precision/recall: **new one-to-one** matching (`script/seg/metrics.py::match_boundaries`); empty predictions/ground truth are
  counted; tolerance ±0.5 s (±1 s sensitivity).
* Retrieval: class-balanced precision@10 with group exclusion and chance level (PROTOCOL.md §4).
* Uncertainty: paired resampling of participants (LARa) / recordings (BABEL); model/clustering-seed and randomization variability reported separately.

### Status of the earlier boundary-scoring defects

The consolidated report identifies three defects in the earlier diagnostics (`script/repro/d_boundary_vs_label.py`):
(1) sequences with an empty prediction or empty ground-truth boundary set were **skipped** (277 true boundaries lost on HuGaDB, 853 true + 286
predicted on BABEL-1, 37 predicted on LARa); (2) `boundary_f1` tests nearest neighbours independently, so duplicate predictions are all "hits"
(predictions at 9 and 11 for one boundary at 10 score 100 instead of 66.7); (3) the AUC/top-N diagnostics use oracle counts.
**They have not been corrected in the repository**; the original outputs are preserved as-is. Nothing in this programme reuses them.
`script/seg/metrics.py` implements one-to-one matching with empty-set handling and is unit-tested on the counter-example
(`results/seg/implementation_checks.json`: TP 1, FP 1, FN 0 → F1 66.7; empty sides → FN or FP only).

## 6. Reused verified components

E0 reproduction of the released checkpoints and its saved predictions (used only as a code-equality check); the original evaluation code
(`f_score`, `edit_score`, `create_correspondences`; tested against a re-implementation of Hungarian mapping); the characterization study's finding
that encoder assignments are stable to small noise and 2-frame shifts on LARa (context only — it did not establish movement categories).
Not reused: HuGaDB (out of scope), the earlier posture-matching design, any codebook-size sweep.

## 7. Experiment 3 execution estimate (not launched)

Prerequisite: encoders whose training excluded the forecasting evaluation recordings.
* **LARa:** the four E2 subject-split encoders already exist (13 training participants each) — **no retraining** for one held-out split; 3 training
  seeds would need 3 × 18 min ≈ 1 h per split on this machine (T1 runner log: 17.5–19.8 min per 30-epoch LARa run).
* **BABEL-1:** no held-out-recording encoder exists. One held-out split × 3 seeds × ≈ 14.4 min ≈ 45 min; BABEL-2/3 (smaller) ≈ 10 min each.
* Predictor training (three input variants × 3 seeds × sweep) is small next to encoder training. Retraining is therefore ≈ 1–3 GPU-hours in total and does not
  threaten the 8-week schedule; **the binding constraints are causal segment extraction and the writing/checking time, not compute.**

## 8. Unresolved limitations (carried into RESULTS.md and DECISION.md)

1. **Transductive encoder.** Every encoder used here has seen the evaluation recordings (unsupervised). Results characterize segment
   representations given such an encoder, not generalization to unseen people or recordings.
2. **No participant identifiers for BABEL**; BABEL fold splits and retrieval exclusions are recording-level; participant leakage possible.
3. **BABEL annotations are short and sparse** (median 3–5 runs); the duration-matched random null is weak or degenerate for many recordings.
4. **BABEL subsets are related** (common source, shared recordings), not independent replications.
5. **One encoder checkpoint per dataset** in the pilot (released weights); encoder-training seed variance is not measured in this block.
6. Fold-level Hungarian mapping and train-fitted mapping both use labels (evaluation and fitting labels respectively) — readouts, not unsupervised methods.
7. The frozen released encoder's edge frames see zero padding; the ±14-frame receptive field means segment features near boundaries share raw-input
   context with neighbouring segments (an unavoidable property of a frozen offline encoder; relevant to causal use in Experiment 3).
8. Historical training provenance of the released weights is undocumented (see §3).

## 9. Provenance

The working tree is `f389b8b` + uncommitted changes (`main.py`, `model.py`, `src/model/{motion_quantizer,ms_tcn,smq}.py`, `results/`, `script/`); the
uncommitted model-code changes are the memory-only edits and the default-off T2 option audited in the consolidated report. New code for this
programme is in `script/seg/` with SHA-256 hashes in `results/seg/frozen_code_hashes.txt`.
