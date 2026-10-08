# Stage B scaling curve: is the contextualizer data-limited?

26 September 2026 | Version 1.0 | Planning document; nothing launched

## 1. Question and why it comes first

Stage B trains a 0.92M-parameter masked-code Transformer on about 76k tokens (HuGaDB) or 220k tokens (LARa). That is 0.08-0.24 tokens per parameter. Masked-modeling methods are normally trained at tens to thousands of tokens per parameter. The results so far (the stage_b_002 copy-baseline shortfall, and the ALiBi advantage in `optstudy_002`) may therefore be small-data effects rather than properties of the approach.

This study answers three label-free questions before any further investment in Stage B, Stage C, or a pretraining pivot:

- **Q1 (data-limited?)** Does held-out masked-code performance still improve as training data doubles, up to the largest amount available here?
- **Q2 (diversity or volume?)** At matched hours, do more *subjects* help more than more *hours per subject*?
- **Q3 (is ALiBi a small-data crutch?)** Does ALiBi's advantage over sinusoidal positions shrink as data grows?

The answers decide between two directions:
- Re-scoping toward pretraining on large, multi-subject motion corpora, if Q1 is positive and Q2 favours diversity.
- Continuing to evaluate the pipeline in the current regime (Stage C, plan §4 criterion), if Q1 is flat.

No action label is used anywhere in this study.

## 2. Why this design

- **Everything available locally tops out near 15 hours.**
  - LARa has 14.6 h across 16 subjects, about 1 h each.
  - HuGaDB has 5.3 h across 18 subjects, 0.1-0.44 h each.
  - The BABEL subsets used by SMQ total only 5.3, 0.7 and 0.8 h. Their recordings are bare sequence IDs with no subject metadata, so they cannot extend the curve or control diversity.
  - Datasets cannot be pooled, because each has its own frozen SMQ encoder and tokenizer.
- **A factorial of subjects × hours-per-subject within LARa** is the only way to get a usable range (16× in hours) while separating diversity from volume. Its iso-hour diagonals compare 3, 6 and 12 subjects at the same total training time.
- **Held-out subjects, not held-out recordings,** are the evaluation set. The hypothesis is about generalizing across people, and every Stage B result so far used recordings from the same subjects for training and validation.

## 3. Data, held-out set, and fixed elements

| Element | Choice | Reason |
|---|---|---|
| Primary dataset | LARa, raw (no per-recording z-score) | Largest and most uniform per-subject volume; raw is where the copy gap was largest |
| Evaluation set | LARa subject fold 0: S07, S11, S13, S14 (3.43 h, 103 recordings) | Existing deterministic fold (plan §2 rule); never used for training, tokenizer fitting, or early stopping |
| Training pool | The other 12 subjects (11.2 h) | |
| Tokenizer | K=500 K-means, seed 111, fit on the 12 training subjects only (`build_or_load_tokenizer(..., "subject_disjoint", fold=0, 111)`) | Held-out subjects never shape the vocabulary. Deliberately **fixed across all subsets**: this study measures contextualizer scaling at a fixed vocabulary, not tokenizer scaling |
| Encoder | Frozen released SMQ LARa checkpoint | It was trained on all 16 subjects (transductive). This lifts all points roughly uniformly and is disclosed; it does not bias slopes toward any arm |
| Evaluation masks | Fixed CRC32 masks (`mask_base` 111), identical for every run | The copy baseline on the evaluation set is therefore one constant, a fixed anchor for every point |

## 4. Factors and grid

**Subset construction (nested).** Each *draw* (d = 1, 2, 3, seeded 111, 222, 1538574472) fixes a random order of the 12 training subjects and a random order of each subject's recordings. A cell (S, F) takes the first S subjects in that order and, from each, the first ⌈F × n_recordings⌉ of that subject's recordings. Subsets are therefore nested within a draw: every smaller cell is contained in the larger ones. This keeps "which subjects happened to be picked" from masquerading as a data effect.

| Subjects S \ fraction per subject F | 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| 3 | ~0.7 h | ~1.4 h | ~2.8 h |
| 6 | ~1.4 h | ~2.8 h | ~5.6 h |
| 12 | ~2.8 h | ~5.6 h | ~11.2 h |

Diagonals of equal hours give the diversity-vs-volume contrasts:
- 1.4 h: (3, 0.5) vs (6, 0.25)
- 2.8 h: (3, 1.0) vs (6, 0.5) vs (12, 0.25)
- 5.6 h: (6, 1.0) vs (12, 0.5)

Actual hours per subset are computed from real frame counts and reported; the table's values are nominal.

**Configurations (2).** They differ in one factor only:
- **C0, original positions:** sinusoidal, post-LN, constant lr 3e-4, untied head. This is the plan's B-code specification, run in step mode.
- **C1, ALiBi:** identical except ALiBi positions.

C1 substitutes a constant learning rate for the cosine schedule of `optstudy_002`'s selected configuration, for two reasons. A cosine schedule tied to a step budget would interact with the longer budget below. And Phase 1 found the schedule effect negligible: mean tuning loss 3.075 vs 3.097, and `postLN_constant_lr0.0003_alibi` was within 0.004 loss of the selected configuration. This substitution is declared here, before any run.

**Run count.**
- 9 cells × 3 draws, minus the 2 duplicate copies of (12, 1.0), which is identical across draws, gives 25 subsets.
- 25 subsets × 2 configurations = 50 runs.
- Plus 2 extra model seeds (222, 1538574472) at (12, 1.0) for each configuration: 4 runs. These estimate model-seed noise separately from subset-draw noise.
- **Total: 54 runs on LARa.**

## 5. Training budget and early stopping

- Step mode: at most **30,000 steps**, evaluation every 250 steps, **patience 3,000 steps**. In `optstudy_002`, the 5,000-step cap bound in 73 of 128 runs. A binding cap would make the curve measure compute, not data.
- Early stopping uses an **inner validation split** carved from each training subset: about 10% of its frames, as whole recordings, seed 111, via the existing `build_validation_split`. It never uses the held-out subjects, so the evaluation set plays no part in any selection.
- Batch 64, AdamW, weight decay 0.01, gradient clip 1.0, dropout 0.1, mask 30% in spans of 8: all unchanged from the plan.
- Model seed 111 except for the seed replicates above.
- A run that stops at the 30,000-step cap is flagged, and its point is treated as a lower bound, not a converged value.

## 6. Metrics (all on the held-out subjects, fixed masks)

**Primary:**
- **held-out masked-code cross-entropy**: the training objective, and the most sensitive measure of the curve's shape;
- **margin over the copy baseline** in masked-code accuracy.

**Secondary:**
- margin at span depth ≥4 (positions far from any unmasked code: the part that requires context rather than repetition) and at depth 1;
- training-probe accuracy on a fixed sample of each run's own training chunks (the generalization gap);
- embedding adjacent-window cosine and effective rank on the evaluation set;
- steps run, stop reason, runtime;
- actual training hours, subjects and recordings per subset.

## 7. Pre-declared analysis and decision rules

These are fixed now and written to the study's `DESIGN.json` before the first run. For each configuration and metric, fit

    metric ≈ a + b_S · log2(S) + b_F · log2(F)

over all 25 subset points. Uncertainty comes from 2,000 bootstrap resamples of draws: whole draws are resampled, so subject-composition noise is carried through. Seed noise is reported alongside from the (12, 1.0) replicates.

- **D1, data-limited (Q1).** The *total-hours slope* (b_S + b_F: both factors doubling together) is examined on the primary metrics.
  - **"Still data-limited at ~11 h"** requires two things: the 90% bootstrap interval of the slope excludes zero in the improving direction, and the final doubling from (6, 1.0) to (12, 1.0) improves held-out cross-entropy by more than twice the seed standard deviation at (12, 1.0).
  - **"Not data-limited in this range"** applies if the interval includes zero, or if the final doubling is within noise.
- **D2, diversity vs volume (Q2).** Compare b_S with b_F, using the bootstrap interval of the difference, and check the iso-hour diagonals directly.
  - "Diversity-driven" if b_S > b_F with the interval excluding zero, and the 12-subject cell beats the 3-subject cell at 2.8 h.
  - "Volume-driven" if the reverse holds.
  - Otherwise "not distinguishable".
- **D3, ALiBi as a small-data crutch (Q3).** Let Δ = metric(C1) − metric(C0), computed per subset, paired by draw. Regress Δ on log2(total hours).
  - "Crutch-like" if Δ's advantage shrinks significantly with data (the 90% interval of the slope excludes zero in the shrinking direction).
  - "Scale-independent in this range" if not.
- **D4, extrapolation guard.**
  - No claim is made beyond about 2× the largest point (roughly 22 h).
  - A positive slope does not state how much data would suffice.
  - An estimate of where the margin crosses zero is reported only if the crossing falls inside the measured range.
- **D5, replication.** A conclusion is labelled "replicated" only if HuGaDB (Phase B) gives the same direction. With HuGaDB's much smaller volume this is weak evidence, and it is reported as such.

## 8. What each outcome means for the project

| Outcome | Implication |
|---|---|
| Data-limited and diversity-driven | The Stage B shortfall is plausibly a scale problem. Re-scope toward pretraining the encoder, tokenizer (FSQ becomes relevant) and contextualizer on large multi-subject corpora, with the skeleton path (AMASS → LARa/BABEL) first. Stage C on the current small-data models becomes lower priority. |
| Data-limited but volume-driven | More hours of similar people help. Pretraining is still indicated, but subject diversity is not the lever. Revisit why codes carry more subject than action information. |
| Not data-limited | The Stage B limitation is not scale within reach, and pretraining is unlikely to rescue it. Proceed to Stage C as the planned usefulness test with the current arm, or close the discrete-context direction under plan §4's criterion. |
| ALiBi crutch-like | Treat Stage B's positional choice as a small-data accommodation. It should not be carried into a pretraining design as a finding. |
| Inconclusive (wide intervals over a 16× range) | Say so. Neither pivot is justified by this study. External data would be needed to settle Q1. |

## 9. Confounds and limitations (disclosed, not solved)

- **Fixed tokenizer.** It was fit on all 12 training subjects, even for 3-subject cells. This isolates contextualizer scaling, but a vocabulary informed by more subjects may flatter small cells.
- **Transductive encoder.** The SMQ encoder saw all 16 subjects, including the held-out ones.
- **Narrow range.** Twelve training subjects and 11 h are far from pretraining scale, and at most a 4× range in subjects. A flat curve here does not prove a flat curve at 1,000 h.
- **Label-free proxies.** Held-out masked-code loss and the copy margin are not segmentation. The audit's point stands: copy accuracy is neither necessary nor sufficient for segmentation. Downstream usefulness is still measured only by Stage C.
- **A single held-out fold.** Fold 0 (4 subjects) is the evaluation set; other folds are not rotated in, to keep cost bounded. This limits how general the result is across subjects.
- **Compute per example.** A smaller subset gets more epochs within the same step budget. Patience-based stopping on its own inner validation split is the control; the stop reasons are reported.

## 10. Phases and runtime

Measured cost is about 11 ms per step for both configurations on this GPU (from `optstudy_002`).

| Phase | Content | Runs | Estimate |
|---|---|---|---|
| A (primary) | LARa raw, full grid | 54 | ~2.5-4.5 h (at most 5.5 min per run at the 30k cap; small subsets stop earlier) |
| B (replication) | HuGaDB raw, same design, fold 0 held out (5 subjects), training pool of 13 subjects with S ∈ {3, 6, 13} | 54 | ~1.5-3 h |
| C (optional) | LARa normalized, same design | 54 | ~2.5-4.5 h |

Phase A is analysed before B or C is launched. B and C are pre-declared here and do not depend on A's outcome, so running them is a budget decision, not a result-driven one.

## 11. Implementation and verification (before launch)

- **New script `script/action_transport/stageb_scaling.py`.** It reuses `stageb_train.train`, `stageb_data.make_chunks` and `build_validation_split`, the subject-disjoint tokenizer path, and the v1.6 identity machinery:
  - `config_hash` per run and `source_sha256` in `DESIGN.json`;
  - refusal to resume under a changed design or changed code;
  - a separate `_smoke` namespace.
- **Outputs:**
  - `DESIGN.json` (written first, containing sections 3-7 verbatim);
  - `subsets.json` (exact recordings and hours for every cell and draw);
  - `runs.csv`;
  - per-run learning curves;
  - `analysis.json` (fits, bootstrap intervals, and the D1-D4 verdicts computed by code, not by hand);
  - `REPORT.md` with the scaling plots: metric against log2(hours), coloured by subject count, one panel per configuration, plus the Δ(C1 − C0) panel.
- **Modes:**
  - `--dry-run` prints the grid with actual hours per cell, without training;
  - `--smoke` runs 2 cells × 1 configuration × 500 steps;
  - `--analyze` recomputes the analysis from `runs.csv` only.
- **Tests required before launch:**
  - subsets are nested within a draw and deterministic across processes;
  - no training, inner-validation or tokenizer-fitting recording belongs to a held-out subject;
  - hours accounting matches frame counts;
  - evaluation masks and the copy baseline are identical across runs;
  - the tokenizer's fit population excludes fold 0;
  - the analysis reproduces known slopes on synthetic runs, and its bootstrap resamples draws, not individual runs.

## 12. Out of scope

- Pretraining on external corpora (AMASS, Capture-24). This study decides whether that is warranted.
- Tokenizer scaling, and FSQ.
- Stage C.
- Any action-label metric.
- Resolving the pending Phase 1 decision (literal stop rule vs plan v1.6). This study is independent of it: it uses Phase 1 only to choose the ALiBi configuration C1.
