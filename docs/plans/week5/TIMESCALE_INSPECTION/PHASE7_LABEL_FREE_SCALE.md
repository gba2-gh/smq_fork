# Phase 7: a prospective BABEL-1 evaluation, and an exploratory screen of label-free span selectors

Revision 2 — 3 October 2026 (corrections from an external review applied before launch; see the end). Implemented in `script/week5/timescale_inspection/run_phase7.py`; not launched. Sized for
one **10-hour** overnight run (dry run projects 6.9 h). Parent: `PHASE6_SPAN_CONFIRMATION.md`.

## Why

Phases 5 and 6 found that relational tokens (T2) and a short mask span raise F1@50 on BABEL-2, BABEL-3 and LARa,
while HuGaDB prefers the long span. Two problems remain before this can be a paper claim:

1. **No independent evaluation.** Every configuration was developed and confirmed on the same four datasets.
2. **The span was chosen with labels.** The span ratio divides by the ground-truth median segment length. A
   reviewer can dismiss the span finding unless the span can be chosen without labels.

Phase 7 addresses (1) with a **prospective evaluation of the frozen recipe on BABEL-1**. This is not an untouched
test set: BABEL-1's T0 `context`/`centre` scores, oracle ceilings and ground-truth duration statistics were examined in
rkx_v2 and Phases 0-1, and 3 of its 1,960 recordings are byte-identical to BABEL-2 recordings and 7 to BABEL-3
recordings (feature-file hashes). What is new is that the T2 / short-span recipe was frozen on BABEL-2/3 and LARa
before it was ever run on BABEL-1. The overlap is disclosed in the report. It answers (2) with Stage 0 of the label-free programme:
re-decode the existing checkpoints with frame predictions saved, then score label-free criteria, declared here,
against the spans that F1@50 preferred.

**Correction carried in:** the BABEL-2 median segment length is 1.70 s, not 1.20 s (1.20 s is BABEL-1). The Phase 6
span-ratio table was corrected. BABEL-2 and BABEL-3 have nearly equal medians (1.70 / 1.77 s), so their different
span gains are not evidence for the ratio.

**The duration threshold is treated as unsupported.** Under Phase 6's declared rule, LARa's span-2 gain (+4.93, all
seeds) means the "ratio ~1" threshold looks wrong; the first Phase 6 report mis-stated this by moving the reference
after the fact. So Phase 7 makes no prediction from the ratio about the size of BABEL-1's gain (its median segment is
1.20 s, the shortest of all datasets, which is reported descriptively only).

## Part H: prospective evaluation on BABEL-1

New tokenizer keys registered at runtime only (no shared file is edited): `t2_relational_b1` and `t0_raw_b1`.
They are identical to T2 and T0 and are defined for BABEL-1.

| Stage | Cells | Seeds |
|---|---|---|
| H1 | T2 span 2 `context`, T2 span 8 `context`, T2 `centre` | 111, 222, 1538574472 |
| H2 | T0 span 2 `context` | same 3 |
| H3 | T2 span 2 `context`, T2 span 8 `context`, T0 span 8 `context` | 333, 444 (extends to 5 seeds) |

References that are read, not rebuilt:
- T0 span 8 `context` at the original seeds: `rkx_v2`.
- T0 `centre`: `ts1_v1`.
- SMQ: `results/e0/babel1_pretrained.json`, global, F1@50 22.38.

**Declared predictions.** The primary metric is F1@50 at the last rung, paired by seed. The bar is a gain of ≥ 3 on
the mean, with every seed in the same direction.

- **P1:** T2 span 2 beats T2 span 8 by the bar. The span ratio is 1.78 at span 8, the largest of any dataset.
- **P2:** T2 span 2 beats T0 span 8 (the original pipeline) by the bar.
- **P3 (descriptive):** T0 span 2 against T0 span 8.
- **P4:** T2 span 2 `context` beats T2 `centre` (the same tokens without contextualization) by the bar. This is the
  more consequential question: does contextualization add value on BABEL-1, beyond the tokenizer? It needs no extra
  fitting (H1 already decodes `centre`).
- **Tokenizer x span factorial (descriptive):** span effect on T0, span effect on T2, and their interaction, on the
  3 original seeds where all four cells exist (T0 span 2 is not run at seeds 333/444, so H3 does not extend it).
- **Known risk, stated in advance:** T2 drops orientation, and BABEL-1 has a `turn` class. A P2 failure driven by
  `turn` would be a tokenizer limitation, not a span failure.
- P1 and P2 are read at 3 seeds after H1/H2, and again at 5 seeds after H3, if H3 finishes.

## Part S0: label-free span selectors (exploratory, retrospective screening)

**Re-decodes**, with frame predictions saved. Only seeds present at every candidate span can enter the criteria, so seeds 333/444 (no span-4 checkpoints) are not re-decoded. Each re-decode must reproduce its saved unit's ladder exactly, on
every rung and metric. A unit that does not reproduce is excluded and reported.

| Dataset | Tokenizer | Spans | Source of checkpoints |
|---|---|---|---|
| BABEL-2, BABEL-3 | T2 | 2, 4, 8 (3 original seeds) | ts5_v1 phase 5 (span 8 from stage A, 2/4 from stage D) |
| LARa | T2 | 2, 4, 8 | phase 5 (span 8), phase 6 (2, 4) |
| HuGaDB | T0 | 2, 4 | phase 6; span 8 predictions already saved by rkx_v2 |
| BABEL-1 | T2 | 2, 8 | from Part H (decoded with predictions natively) |

**This is screening, not evidence of label-free selection.** The selectors use no labels, but the spans they are
scored against were chosen by F1@50, and choosing among K1/K3/K4 by their scores is itself a label-based selection. A
selector that passes is a candidate for a prospective test, nothing more.

**Selectors (declared now; none has been computed yet).** For dataset *d* and candidate span *s*, using the seeds
available at every candidate span of *d*:

- **K1, cross-seed stability.** For every pair of vocabulary seeds, map one seed's predicted states onto the
  other's by Hungarian matching on frame co-occurrence, pooled over the dataset. This matches predictions to
  predictions and uses no labels. Score segmental F1@50 between the two segmentations, averaged over both
  directions. K1(s) is the mean over seed pairs. Pick: argmax over *s*.
- **K3, self-consistency.** m(s) is the median predicted segment duration in seconds, pooled over recordings and
  seeds. Pick: the largest *s* whose mask duration (span × window ÷ fps) is ≤ m(s). If none qualifies, the smallest
  span. Known risk, stated in advance: a long span makes OT predict longer segments, which favours long spans (a
  positive-feedback trap).
- **K4, feature time scale (model-free).** τ is the lag, in seconds, at which the mean autocorrelation of the
  standardized continuous token features (the tokenizer's PCA space) first drops below 0.5. It is computed per
  recording and the median is taken. Pick: the largest *s* whose mask duration is ≤ τ. If none qualifies, the
  smallest span.
- **Constant references:** always the shortest span, and always span 8.

**Diagnostics reported with every selector** (the selectors cannot see these failure modes):
- K1 scores identical collapsed segmentations (one state everywhere) as 100, so occupancy is reported per span:
  states per recording, share of single-state recordings, predicted runs per recording, states used.
- K3 can justify the long span that caused over-merging; predicted run counts are reported next to it.
- K4 coverage: the number of recordings where the decorrelation time is defined, per seed. An undefined K4 on a dataset
  counts as a wrong pick there.

**Declared cells and completeness.** Spans: BABEL-1 {2, 8}; BABEL-2, BABEL-3, LARa and HuGaDB {2, 4, 8}. A dataset
is complete only if every declared span is present, at least 3 seeds are common to all of them, and every cell covers
the identical recordings with identical frame counts. Missing or non-reproducing cells make that dataset
**incomplete**; `always_8` is always span 8.

**Gate.** The acceptable set on a dataset is every declared span whose mean F1@50 is within 3 points of that dataset's
best span (same seeds). Status per selector: **fail** if its pick is wrong or undefined on any complete dataset;
otherwise **incomplete** if any declared dataset is incomplete; otherwise **pass**. An incomplete screen is never
reported as a pass. Expected baselines: "always shortest" fails HuGaDB, and "always 8" fails BABEL/LARa, so a criterion must
beat both constants to count.

These rules cannot be changed after the run. The selector and verdict code (`criteria7.py`, `report7.py`) is hashed
(the analysis hash), recorded at launch in the manifest and with `criteria.json`; the report flags any mismatch. Any
new selector proposed after seeing the results is reported as a new exploratory analysis.

## Order and budget

| Stage | Content | Estimate | Cumulative |
|---|---|---|---|
| G | G7: generalized T0 preparation on BABEL-1 reproduces rkx_v2's (codes and features exact, centres to 1e-4) | 2 min | 0.05 h |
| H1 | BABEL-1 T2 (span 2, span 8, `centre`) × 3 seeds | ~100 min | 1.7 h |
| H2 | BABEL-1 T0 span 2 × 3 (completes the 3-seed comparison before the re-decodes) | ~30 min | 2.2 h |
| S0 | 33 re-decodes with predictions | ~3.7 h | 5.9 h |
| H3 | BABEL-1 seeds 333, 444 | ~58 min | 6.9 h |
| K | criteria and report (CPU, minutes) | ~5 min | 7.0 h |

Launch with `run_phase7.ps1 -RunId ts5_v1 -Hours 10`. Output goes to `results/week5/timescale_inspection/ts5_v1/phase7/`.

## Revision 2: corrections from an external review (before launch)

1. BABEL-1 is not an untouched test set (prior T0 analysis; 10 byte-identical recordings with BABEL-2/3). Part H is a
   prospective evaluation of a frozen recipe on a previously analysed benchmark, and the report discloses this.
2. Phase 6's LARa verdict was corrected to "threshold looks wrong" (the report had moved the declared reference); the
   duration threshold is treated as unsupported and makes no prediction for BABEL-1.
3. Incomplete data can no longer yield a selector pass (declared spans, >= 3 common seeds, identical coverage;
   `always_8` no longer substitutes another span).
4. Added P4 (context vs the same tokenizer without context) and the 3-seed tokenizer x span factorial.
5. Selectors are labelled exploratory; collapse/occupancy, run-count and K4-coverage diagnostics are reported.
6. The selector and verdict code is covered by an analysis hash.
