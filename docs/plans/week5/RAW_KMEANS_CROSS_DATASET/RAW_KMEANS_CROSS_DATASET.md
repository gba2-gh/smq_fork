# Raw K-means tokens across datasets: LARa and BABEL

Revision 1 — 2 October 2026. Design only; nothing launched.

## Objective

Test whether the best HuGaDB configuration transfers to LARa and the three BABEL subsets, with every setting frozen in advance. That configuration is raw patches → K-means-512 IDs → masked contextualizer → contextual OT, at guard radius 0.

Every recommendation so far, including "K-means rather than native VQ codes" and "raw is competitive with learned encoders", rests on HuGaDB: one checkpoint family and mostly one seed. This plan asks one question: **does the configuration still work on other datasets, without re-tuning?** It does not tune anything on LARa or BABEL.

## Motivating evidence (HuGaDB, development)

| Configuration (seed 111 unless noted) | F1@50 | Source |
|---|---|---|
| Raw K-means IDs, guard 0, reference contextualizer | 50.00 (seed 222: 52.96) | `results/week5/contextualizer_calibration/ctx_cal_v1` |
| Raw K-means IDs, guard 14 | 36.08 | same; reproduces Task 2 |
| Native VQ quantized vectors, guard 14 | 38.81 | same |
| VQ-encoder K-means IDs, guard 14 | 49.55 | `results/quantized_input_comparison/task2/task2_v1` |
| SMQ released checkpoint, dataset-level Hungarian | 24.30 | `results/e0/hugadb_pretrained.json` |

Raw K-means needs no encoder training. The tokenizer is an unsupervised fit: a global standardizer, PCA-64 and K-means-512. The contextualizer and the OT fit are still trained per dataset and seed. Without the contextualizer, OT on raw codes scores 1.6–4 F1@50, so the contextualizer stays.

## Datasets (local copies)

| | HuGaDB | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---|---|---|---|---|
| Recordings | 364 | 439 | 1,960 | 255 | 183 |
| Frames | ~1.14 M | 2.63 M | 568 k | 79 k | 82 k |
| Median length | 47 s | 120 s | 7 s | 8.5 s | 11 s |
| fps | 60 | 50 | 30 | 30 | 30 |
| Channels per frame | 6 joints × 6 | 22 joints × 6 | 25 joints × 3 | 25 × 3 | 25 × 3 |
| Classes (incl. background) | 10 | 8 | 5 | 5 | 5 |
| Token window (frames / s) | 15 / 0.250 | 12 / 0.240 | **8 / 0.267** | 8 / 0.267 | 8 / 0.267 |
| Raw patch dimension | 540 | 1,584 | 600 | 600 | 600 |
| Tokens (approx.) | 76 k | 219 k | 71 k | 9.9 k | 10.3 k |

The BABEL subset classes are walk/stand/turn/jump, sit/run/stand_up/kick and jog/wave/dance/gesture, each with `none`.

**Data identity is already verified.** Re-scoring the released SMQ checkpoints with the repository scorer and dataset-level Hungarian matching reproduces the published numbers. Those are `results/e0/*_pretrained.json`, "global":

| | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---|---|---|---|
| Local re-score F1@50 | 16.39 | 22.38 | 27.52 | 19.35 |
| SMQ paper F1@50 | 16.4 | 22.3 | 27.4 | 19.3 |

The local BABEL data are SMQ's subsets, with sequences of more than 50% background already excluded. The scorer treats `none` as an ordinary class, as SMQ's evaluator does.

**Prior use, disclosed.** LARa is not untouched: it informed the Stage B contextualizer selection and the normalization findings. BABEL has been used for SMQ reproduction and code diagnostics, but never to choose contextualizer, masking or OT settings. Both are prospective tests of a frozen configuration. Neither is a held-out test set in the strict sense.

## Frozen configuration

These are carried over from HuGaDB unchanged and declared before any LARa or BABEL score exists:

| Component | Setting |
|---|---|
| Tokenizer | raw frame patches → global StandardScaler → randomized PCA-64 → K-means K=512 (n_init 5, Lloyd); fit on ≤10,000 complete training-split windows |
| Contextualizer | `postLN_cosine_wu500_lr0.0003_alibi`: width 128, 4 layers, 4 heads, FF 512, dropout 0.1, untied head; AdamW 3e-4 / 0.01, cosine schedule with 500 warmup and a 0.1 floor |
| Masking | fraction 0.30, span 8 tokens, **guard radius 0**; raw patches do not overlap, so this is leak-free |
| Context | 128 tokens, training and inference stride 64, final chunk anchored to the end, batch 64 |
| Budget | 5,000 accepted updates, validation every 100, no early termination |
| Validation split | whole recordings, 10% of frames, split seed 111, the same for all seeds |
| OT | continuous cosine OT on L2-normalized contextual embeddings; a=0.7, beta=0.3, lambda=0.05, epsilon=0.07; 25 inner steps, **outer cap 50**; final rungs 500 / 1,000 / 2,000 / 4,000 with fixed prototypes |
| OT states | number of classes: LARa 8, BABEL 5, HuGaDB 10 |
| Temporal kernel | the existing rule, half-width = fps // window: HuGaDB 4 (1.00 s), LARa 4 (0.96 s), BABEL 3 (0.80 s) |
| Initial grouping | the existing `build_grouping` on frame-weighted per-code PCA means, refit for each vocabulary |
| Scoring | frame broadcast, one dataset-level Hungarian mapping, MoF / Edit / F1@{10,25,50}; F1@50 is the focal metric |

The only per-dataset differences are the window (≈0.25 s), the class count and what follows from the frame rate. Per-dataset tuning is out of scope.

## Declared factors

* **LARa normalization, two levels.**
  - `global`: global standardizer only, as on HuGaDB.
  - `per_recording`: each channel z-scored by its own recording's mean and standard deviation (floor 1e-8) on raw frames before patching. The global standardizer, PCA and K-means are then fit on the normalized patches.

  This is label-free and transductive per recording. Reuse the Stage C helper if it matches this definition exactly, and record which one is used. Both levels are primary; neither is chosen afterwards.
* **Seeds: three paired seeds**, {111, 222, 1538574472}. Each seed drives both the vocabulary (sample, PCA, K-means, grouping) and the contextualizer (initialization, batch order, masks). The K-means vocabulary seed alone moved HuGaDB F1@50 by 8–12 points, so seeds that vary only the contextualizer would understate the spread. Paired seeds measure the total variance; they do not separate its sources.
* **HuGaDB anchor:** the same paired-seed protocol on HuGaDB, so all datasets share one seed protocol. Seed 111 doubles as the port check.

## Matrix

| Dataset | Normalization | Seeds | Training jobs | OT fits (selected + final) |
|---|---|---|---|---|
| HuGaDB | global | 3 | 3 | 6 |
| LARa | global, per_recording | 3 each | 6 | 12 |
| BABEL-1 / 2 / 3 | global | 3 each | 9 | 18 |
| **Total** | | | **18** | **36** |

Checkpoints:

- The **validation-selected** checkpoint (full-history best, strict improvement, earliest tie) is primary.
- The **final** 5,000-update checkpoint is secondary. The calibration showed checkpoint-to-checkpoint swings of up to 8 points, so reporting both shows how much the conclusion depends on the policy.

Evaluations share results only when checkpoint tensors, inputs, inference and solver settings are identical.

## Gates before scientific interpretation

1. **Port regression of the tokenizer.** The generalized data preparation, run on HuGaDB with seed 111, must reproduce `qic_v3`'s `raw_kmeans` family exactly: codes, lookup table, mean, RMS, grouping and the split.
2. **Port regression of the pipeline.** The HuGaDB seed-111 job must reproduce the `ctx_cal_v1` job `raw__ctx128_span8_g0_h5000_seed111_tcnone`: validation-selected step 4100 and all five metrics at every rung, with F1@50 = 50.00.
3. **Label loading and scoring for LARa and BABEL.** Re-scoring the saved SMQ predictions (`results/preds/*_pretrained.npz`) with the new label loaders must reproduce the "global" numbers in `results/e0/*_pretrained.json`.

A gate failure blocks interpretation of the dependent dataset. The runner stops by default; an explicit flag overrides. Mismatches are investigated and saved with environment evidence, never fixed by changing settings to recover a score.

## Comparison context

Published F1@50. These numbers come from different protocols: SMQ, MASQ and HiST-VQ train end-to-end with K equal to the number of classes and 1-second patches.

| Method | HuGaDB | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---|---|---|---|---|
| ASOT (as run in SMQ) | 3.0 | 5.7 | 24.5 | 23.4 | 14.3 |
| SMQ (re-scored locally) | 24.3 | 16.4 | 22.4 | 27.5 | 19.4 |
| HiST-VQ | 28.3 | 19.3 | 22.7 | 30.4 | 24.1 |
| MASQ | 39.5 | 22.2 | 21.0 | 28.4 | 24.5 |

**Protocol differences to state in any comparison:**

- Our OT fit pools all recordings, including the validation recordings, so it is transductive.
- HuGaDB settings were chosen on HuGaDB development scores.
- Our tokens are 0.25 s, not 1 s.

The comparison is informative, but it is not a matched benchmark result.

## Declared reading guide (not a gate)

Decided before scoring, on the mean of three seeds at the primary checkpoint and the last rung:

* **Transfers:** above the local SMQ re-score on LARa under at least one normalization, and on at least two of the three BABEL subsets, with every seed of those cells above SMQ. The HuGaDB tokenizer recommendation then extends to these datasets.
* **Does not transfer:** below SMQ on LARa under both normalizations, or on at least two BABEL subsets. The recommendation stays HuGaDB-only, and the cause is diagnosed (see the risks below) before any change.
* **Anything between** is reported as mixed. The normalization factor and the checkpoint policy are shown as they came out. Neither is re-chosen per dataset after scoring.

## Risks, and what is recorded for each

| Risk | Why | Recorded |
|---|---|---|
| BABEL sequences are short (median ~28 tokens) | most of each 128-token chunk is padding, so supervised tokens per update fall about 4–5× | valid-token fraction, eligible targets, batch tails |
| BABEL-2/3 overfitting | ~10 k tokens and ~230 training chunks, so 5,000 updates is about 1,000+ epochs | full validation history; selected versus final |
| K=512 on BABEL-2/3 | ~20 tokens per code on average | occupancy, perplexity and singleton codes per vocabulary |
| Uniform action prior per recording | a 7-second clip holds 1–3 of 5 states | per-recording predicted state counts and run lengths |
| LARa codes may encode the subject more than the action | earlier K≈500 analyses: 1.86 vs 0.71 nats | label-free subject probe on the codes (subject IDs are metadata), as a diagnostic only |
| Outer OT fit never converges | all calibration fits ended capped at 50 and at 150; cap 150 was up to 2.9 points worse | outer status, objective traces, per-recording final-solve statuses |
| Seed variance | 3–12 points on HuGaDB | all seeds reported, never only the mean |

None of these is "fixed" by retuning inside this plan. Remedies such as a K sweep for BABEL, a shorter context or a different prior are follow-ups, each with its own plan.

## Runtime estimate

This is scaled from `ctx_cal_v1` (HuGaDB: about 2 min to train and about 6.3 min per decode, mostly CPU OT) by tokens × states. Re-measure after the first job of each dataset.

| Dataset | Training | OT | Subtotal |
|---|---|---|---|
| HuGaDB | 3 × 2 min | 6 × 6.3 min | ~45 min |
| LARa | 6 × 2–3 min | 12 × ~15 min | ~3.3 h |
| BABEL-1 | 3 × 2 min | 6 × ~5 min (1,960 recordings add overhead) | ~35 min |
| BABEL-2/3 | 6 × 2 min | 12 × ~1.5 min | ~30 min |
| Preparation | | 5 datasets × seeds × levels | ~20 min |
| **Total** | | | **~5.5 h** (launch with `-Hours 8`) |

## Implementation scope

Reuse the calibration package (`script/week5/contextualizer_calibration/`), extended to take a dataset, with backward-compatible HuGaDB defaults. The current HuGaDB-only places are:

- the label loader (`quantized_input_comparison/labels.py`);
- the OT frame rate, window and 10 states (`ot_arms.py`);
- the data preparation (`quantized_input_comparison/prepare.py`, `fit.py`, `families.py`);
- the q1q2 dataset loader, which has no BABEL entry.

Generalize each with an explicit dataset argument. Never use per-job global mutation, and leave HuGaDB results byte-identical (gates 1 and 2).

* New code: `script/week5/raw_kmeans_cross_dataset/`. New results: `results/week5/raw_kmeans_cross_dataset/<run_id>/`. Earlier runs are preserved.
* Data preparation runs per (dataset, normalization, seed) and is cached by identity: dataset hash, normalization, seed, window, K, PCA dimension and sample cap.
* Output files mirror the calibration run: manifest with environment and resume identity, planned and completed CSVs, exposure, per-recording predictions, embeddings, prototypes, traces, runtime, `not_run.csv`, and a report generated from saved results.
* Preflight tests cover:
  - gates 1–3 on cached data;
  - BABEL windowing, with the last partial window kept as in `qic_v3`;
  - per-recording normalization;
  - class counts and the label-to-ID mapping per subset;
  - the kernel half-width per dataset.

  A bounded smoke run of at most 10 minutes may use a few recordings per dataset.

Separate agent instructions will follow this plan. The user launches the full run.

## Out of scope

Encoder or VQ training, FSQ, native VQ, the patch-local encoder, K or window sweeps, contextualizer search on LARa or BABEL, subject-disjoint splits, and claims of a benchmark result.
