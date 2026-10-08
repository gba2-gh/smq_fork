# Phase 6: confirming the span rule, then the final report

Revision 1 — 3 October 2026. Implementation instructions; nothing implemented or launched. Sized for one
**6-hour** budgeted run (projected ~4.7 h). Parent plans: `TIMESCALE_INSPECTION.md`, `PHASE5_TOKENIZER.md`.
Phase 5 results: `results/week5/timescale_inspection/ts5_v1/phase5/REPORT.md`.

## Why this phase exists

Phase 5 (`ts5_v1`) found two effects:
- **Tokenizer:** inter-joint distances (T2) beat the raw tokenizer (T0) on BABEL-2, BABEL-3 and LARa.
- **Span:** on T2, the contextualizer's F1@50 rises steadily as the mask gets shorter in seconds. On BABEL-2 it goes
  from 11.2 at 4.0 s, to 21.4 at 2.1 s, to about 31 at 1.1 s or less. Two different routes to 1.07 s (span 4 at
  window 8, span 8 at window 4) agree.

The proposed mechanism is that a mask longer than a typical segment makes the contextualizer over-smooth, so OT
predicts too few segments. On BABEL at span 8, OT predicts about 2.2 segments per recording against 3.7 in the
ground truth. At ≤ 1.07 s it predicts about 3.

That claim rests on two datasets, one tokenizer and three seeds, and was found after the fact. This phase tests it
where it makes predictions it could fail, then closes the report. Four items, in priority order:

1. **Headline seeds.** Extend the BABEL-2/3 result to 5 seeds, as a full 2×2 design (tokenizer T0/T2 × span 8/2).
2. **Span 2 on T0.** Does the span effect need T2 tokens? This is part of the same 2×2.
3. **HuGaDB negative control.** Its span 8 is already short relative to its segments (ratio 0.58), so the rule
   predicts no gain.
4. **LARa out-of-sample test.** Its span 8 is borderline (ratio 0.87), so the rule predicts a small gain at most.

Then the report fixes and the final write-up.

## Reference quantities (declared now)

Median ground-truth segment length, from `script/week5/segment_stats.py`: BABEL-2 1.70 s (corrected after the run; an earlier draft used BABEL-1's 1.20 s), BABEL-3 1.77 s, LARa 2.2 s,
HuGaDB 3.47 s.

Mask duration = span × window ÷ fps. The **span ratio** = mask duration ÷ median segment length.

| dataset | window, fps | span 2 | span 4 | span 8 |
|---|---|---|---|---|
| BABEL-2 | 8, 30 | 0.53 s / 0.31 | 1.07 s / 0.63 | 2.13 s / 1.25 |
| BABEL-3 | 8, 30 | 0.53 s / 0.30 | 1.07 s / 0.60 | 2.13 s / 1.20 |
| LARa | 12, 50 | 0.48 s / 0.22 | 0.96 s / 0.44 | 1.92 s / **0.87** |
| HuGaDB | 15, 60 | 0.50 s / 0.14 | 1.00 s / 0.29 | 2.00 s / **0.58** |

The working rule, stated so it can fail: **the contextualizer over-smooths once the span ratio is above about 1.**
Below that, shortening the span gives no large gain.

## Matrix

New seeds: **333 and 444**, declared now and never replaced. The original seeds are 111, 222 and 1538574472.
Radius `current`, outer cap 50, rungs 500/1,000/2,000/4,000, `context` jobs at guard 0. Everything else is unchanged
from Phase 5.

| Stage | Content | Units | Estimate | Cumulative |
|---|---|---|---|---|
| R | reproduction gates R1, R2 (below) | — | ~3 min | 0.05 h |
| S1a | BABEL-2/3, **T0 span 2** × original 3 seeds (`context`) | 6 trainings + 6 decodes | ~26 min | 0.5 h |
| S1b | BABEL-2/3 × new seeds {333, 444}: T2 span 8, T2 span 2, T0 span 8, T0 span 2 (`context`), plus T2 `centre` and T0 `centre` | 16 trainings, 24 decodes | ~80 min | 1.8 h |
| S2 | **HuGaDB**, T0, spans {2, 4} × 3 original seeds (`context`) | 6 trainings + 6 decodes | ~66 min | 2.9 h |
| S3 | **LARa**, T2, spans {2, 4} × 3 original seeds (`context`) | 6 trainings + 6 decodes | ~106 min | 4.7 h |

Unit costs are the Phase 5 table: train 1.5 / 1.6 / 2.2 / 2.7 min, decode `context` 2.5 / 2.1 / 14.4 / 8.0 min,
`centre` 2.1 / 1.5 / 10.6 / 4.6 min, for BABEL-2 / BABEL-3 / LARa / HuGaDB.

After S1, BABEL-2 and BABEL-3 each have a complete 2×2 (tokenizer × span) with 5 seeds per cell. The `centre` arm
has 5 seeds for both tokenizers.

**Where the references come from (read, not rebuilt), original seeds:**
- T2 span 8 `context`, T2 `centre`: `ts5_v1/phase5/units`.
- T2 span 2 and 4: `ts5_v1` stage D.
- T0 span 8 `context`: `rkx_v2`.
- T0 `centre`: `ts1_v1` Phase 1 `inputs`.
- HuGaDB T0 span 8 `context`: `rkx_v2` (50.00 / 44.83 / 43.85).
- LARa T2 span 8 `context`: `ts5_v1`.

The order follows importance: the paper's headline first (S1), then the negative control (S2, cheaper), then LARa (S3,
the longest). A budget stop always leaves earlier stages complete.

Out of scope: T2 `pca` at the new seeds (`centre` is the declared no-context arm), LARa T0 span 2, HuGaDB T1, BABEL-1,
radius changes, any further window values.

## Reading rules (declared before the run)

**Post-run correction (3 October):** the first Phase 6 report compared LARa against BABEL-3's updated 5-seed gain (+5.30) instead of the declared +3.9, and called LARa consistent. Under the declared rule LARa span 2 (+4.93, all seeds positive) means **the threshold looks wrong**. The report code now uses the declared value.

The primary number is F1@50, selected checkpoint, last rung, paired by seed. The bar is ≥ 3 points on the mean, with
every seed in the same direction. With 5 seeds, the report also gives the number of positive seeds and an exact
two-sided sign-test p-value. Neither is part of a verdict.

- **H2′-confirm (headline).** On BABEL-2, T2 span 2 beats T2 span 8 by the bar over all 5 seeds. BABEL-3 is reported
  on the same rule.
- **H9: does the span effect need the tokenizer?** Using the 2×2 on each BABEL dataset, the report gives:
  - the span effect on T0 (T0 span 2 − T0 span 8) and on T2;
  - the tokenizer effect at span 8 and at span 2;
  - the interaction (difference of differences), paired by seed.

  Labels:
  - *Span effect is tokenizer-independent:* T0 span 2 beats T0 span 8 by the bar.
  - *Needs relational tokens:* T0 span 2 is within ±3 of T0 span 8 while T2's span effect passes.
  - *Reverses:* T0 span 2 loses by the bar.
- **H8a: HuGaDB negative control.** Prediction: neither span 2 nor span 4 beats span 8 by the bar.
  - If either does, the ratio threshold is wrong: shorter spans would help everywhere, not only past ratio ~1.
  - A loss by the bar is **consistent** with the smoothing mechanism. It would mean HuGaDB needs the smoothing a longer
    span gives, which sets a lower limit on the useful span. It is reported as such, not as a pass or fail.
  - Note: HuGaDB's span-8 seed spread is about 3.3 points, so only all-seed movements are read.
- **H8b: LARa out-of-sample.** Prediction: the better of span 2 / span 4 is within −3 to +3 of span 8, or gains
  modestly, and in any case gains less than BABEL-2 did (+9.3).
  - *Refuted (short spans hurt):* span 2 or 4 loses by the bar.
  - *Threshold wrong:* span 2 or 4 gains ≥ 3 and more than BABEL-3's +3.9.
  - Note: a modest gain is the predicted outcome, not a failure.
- **Descriptive, every cell:** predicted segments per recording against ground truth. The mechanism predicts the
  segment count moves towards the ground-truth count as the span shortens on BABEL. On HuGaDB it predicts no move, or
  a move above the ground truth.

## Implementation instructions

### Do not edit (their hashes give `ts5_v1` its identity)

- `run_phase5.py`
- `tokenizers.py`
- `gates5.py`
- `radius.py`
- every shared module listed in `run_phase5.TRAIN_FILES`

`report5.py` is not in the hash and may be edited (see the report fixes below). If a shared helper is needed, copy
it into the new module.

### New files in `script/week5/timescale_inspection/`

**1. `run_phase6.py`**

- Import `run_phase5 as R5` and reuse `R5.Cell`, `R5.mk_cell`, `R5.make_job`, `R5.select_tokenizer` (unused) and the
  `Runner` methods.
- Define `Phase6Runner(R5.Runner)`. Its `__init__` must **not** call `super().__init__`, because that hard-codes the
  `phase5` directory and its cell set. Instead, set these attributes:
  - `out = results/week5/timescale_inspection/<run_id>/phase6`, with subdirectories `prep`, `jobs`, `units`;
  - `scale = P.FULL`;
  - `smoke` (`--smoke` gives `replace(P.SMOKE, recording_cap=None)` plus the dataset's `smoke_cap`);
  - `deadline`;
  - `impl`, a hash over `OWN_FILES = (tokenizers, gates5, run_phase5, run_phase6, radius)` plus `R5.TRAIN_FILES`;
  - `radius = RAD.CURRENT`;
  - `seeds`, `job_infos`, `loaders`, `not_run`, `gates_path`, `gates`, `manifest_path`;
  - `base_cells`, the union of all cells in S1a–S3.

  Then call `self.check_resume()`. The inherited `check_resume` writes `phase=5`; override it to write `phase=6`.
- `stage_items(stage, seeds)` for stages S1a, S1b, S2, S3 exactly as in the matrix. Items are `R5.Item(cell, arms)`:
  - S1a: `mk_cell(ds, "t0_raw", s, span=2)`, arms `("context",)`.
  - S1b, for each new seed and BABEL dataset:
    - `mk_cell(ds, "t2_relational", s)` with arms `("context", "centre")`;
    - `mk_cell(ds, "t2_relational", s, span=2)` with `("context",)`;
    - `mk_cell(ds, "t0_raw", s)` with `("context", "centre")`;
    - `mk_cell(ds, "t0_raw", s, span=2)` with `("context",)`.
  - S2: `mk_cell("hugadb", "t0_raw", s, span=sp)` for sp ∈ {2, 4}, `("context",)`.
  - S3: `mk_cell("lara", "t2_relational", s, span=sp)` for sp ∈ {2, 4}, `("context",)`.
  - `t0_raw` is defined for all four datasets in `TOK.SPECS`, and HuGaDB's T0 is the raw IMU patch, so S2 needs no
    new tokenizer.
- **Preparation reuse (read-only).** Override `prepare(cell)` to look in order:
  1. `ts5_v1/phase5/prep/<tok>/<prep_id>`;
  2. for `babel2__t0_raw__seed111` only, the G1 directory `ts5_v1/phase5/prep/babel2__t0_raw__seed111` (Phase 5's G1
     wrote it outside the `<tok>/` subfolder);
  3. otherwise build in `phase6/prep/<tok>/<prep_id>` via `TOK.prepare_cell`.

  A reused directory must pass `TOK.is_prepared` (exact prep-key match). Never write into `ts5_v1/phase5`. Reuse
  matters because the span-2/4 units of Phase 5 were trained on those exact artifacts, so a span comparison at an
  original seed keeps identical tokens on both sides.
- **Gates.** Reuse `R5.Runner._gate`, writing to `phase6/gates.json`.
  - **R1:** rebuild `babel3__t2_relational__seed111` in a scratch folder under `phase6/gates/` and compare with
    `ts5_v1`'s copy:
    - codes, continuous features, starts/lengths, mean, RMS, grouping and split must match exactly;
    - the K-means centres (`lookup`) must match to 1e-4, the same tolerance as G1 and for the same reason.

    If R1 fails, S1b is blocked, because new-seed cells would be built by code that does not reproduce Phase 5.
  - **R2:** decode `babel2__t2_relational__seed111` `centre` from the reused preparation. Every metric at every rung
    must equal `ts5_v1/phase5/units/babel2__t2_relational__seed111__centre/info.json` (F1@50 30.63) exactly. If R2
    fails, nothing runs.
  - Gates G2/G3 are not needed: no T4, and LARa's T2 inherits Phase 5's passed G3, which the report cites.
- `run_item`, `ensure_train`, `ensure_decode` and `build_recs` are inherited unchanged. The unit identity already
  includes `identity_of(cell)` and hence the span. `job_dir` and `unit_dir` resolve under `phase6/` because `self.out`
  does.
- `execute()`: gates, then S1a → S1b → S2 → S3. Then write the manifest (`finished` or `budget_stop`), then
  `report6.write_report`.
- `dry_run()`: list every unit and print a per-stage and cumulative projection from `R5.COST`. Reused preparations
  cost 0. Prints should match the table above to within 10%.
- CLI: `--run-id` (default `ts5_v1`, so the output sits next to Phase 5), `--hours` (required unless `--dry-run` or
  `--smoke`), `--smoke`, `--cpu`, `--dry-run`, `--context-run rkx_v2`, `--direct-run ts1_v1`, `--phase5-dir`
  (default `<run>/phase5`).

**2. `report6.py`** (reads phase5 and phase6 units, plus the rkx_v2/ts1_v1 baselines; writes `phase6/REPORT.md`)

- A combined unit loader keyed `(dataset, tokenizer, arm, window, span) → {seed: row}`.
  - Merge phase5 and phase6 units; on a clash, error out, never silently prefer one.
  - Add the T0 baselines at the original seeds through `report5.Results.f1`.
- **Sections, in order:**
  1. Gates R1/R2.
  2. **Headline:** a 5-seed table on BABEL-2 and BABEL-3 with the four 2×2 cells (`context`) plus `centre` for T0
     and T2. Per seed, mean ± sd, then the H2′-confirm verdict, positive-seed count and sign-test p.
  3. **H9:** span effect on T0 and on T2, tokenizer effect at span 8 and at span 2, and the interaction, each with
     paired deltas and verdict labels.
  4. **Span-rule table:** every `context` cell, across BABEL-2/3 and LARa (T2) and HuGaDB (T0), with spans 2/4/8 and
     the Phase 5 windows 4/15. Columns:
     - mask duration in seconds;
     - span ratio;
     - F1@50 mean ± sd and seeds;
     - predicted segments per recording, and the ground-truth count.

     Sort each dataset by span ratio.
  5. **H8a (HuGaDB)** and **H8b (LARa)** verdicts, with the wording above.
  6. **SMQ reference** (see the fixes below).
- Every verdict is computed by a pure function from per-seed dicts, so it can be tested on synthetic numbers.

**3. `tests6.py`**

- Stage contents and counts:
  - S1a has 6 items;
  - S1b has 16 items and 24 arm-units;
  - S2 has 6 items and S3 has 6;
  - every S1b item has seed 333 or 444, and no item outside S1b does.
- `prepare` prefers a verifying `ts5_v1` directory and never writes into it. Test with a temporary fake `phase5` tree
  whose manifest does not verify; the runner must then build in `phase6`.
- Unit identity differs between span 2 and span 4 for the same cell, and differs from the Phase 5 identity of the
  same cell id. That is intended: different `impl`, never a silent cache hit across phases.
- The merged loader raises on a duplicate key.
- H9 labels on synthetic 2×2 numbers: independent / needs relational tokens / reverses.
- H8a and H8b labels on synthetic numbers, including "a loss is consistent" for HuGaDB.
- Sign-test p-values against a hand calculation: 5/5 gives p = 0.0625 two-sided.
- The span-ratio column matches the declared table above.
- An expired budget starts nothing, and stages run in order.
- R2 on the real data: the `centre` decode of BABEL-2 T2 seed 111 equals the saved unit. Skip if `ts5_v1` is absent.

**4. `run_phase6.ps1`** — copy `run_phase5.ps1` and change the module name. The default `-RunId` is `ts5_v1`.

### Report fixes to `report5.py` (then regenerate `ts5_v1/phase5/REPORT.md`)

1. **SMQ table.**
   - Consider every finished `context`/`centre`/`pca` unit with ≥ 3 seeds, including stage D/E variants.
   - Title it "best of N arms (selected after the fact; optimistic)", with N printed.
   - Add a separate row for the pre-declared headline (T2 span 2).
   - Change the label to "SMQ released checkpoint, re-scored here with dataset-level Hungarian
     (`results/e0/*_pretrained.json`, `global`)".
   - Drop HuGaDB's single-seed rows from "best".
2. **T1 probe degeneracy.** For T1, the linear probe and the whitened 1-NN probe are invariant to PCA whitening by
   construction: both standardize the features first. Show them as "invariant (n/a)" and compute T1's
   all/some/none label from the plain 1-NN probe alone. State the reason under the table.
3. **H6 fragility marker.** Flag a pass whose smallest per-seed delta is below 1 point as "passes, fragile"
   (BABEL-2 T2 `centre`: +10.5 / +0.2 / +10.4).
4. **Early-stopped OT marker.** Mark seeds whose OT status is `converged` with fewer than 50 outer iterations. They
   explain most of the seed spread on the direct arms (BABEL-2 T2 seed 222 `centre`/`pca`; BABEL-3 T4 `centre`).
5. **H2′ per dataset.** Give the bar result for each dataset as well as the combined verdict. BABEL-3 span 2 passes on
   its own (+3.9; +1.0 / +4.4 / +6.2).
6. **Window confound wording.** The OT radius in seconds is nearly constant across windows (0.93 / 0.80 / 1.00 s for
   windows 4 / 8 / 15). The larger confounds are the mask duration and the context length in seconds. Add the mask
   duration and span ratio columns to the window table.
7. **Ground-truth segments.** Add a `gt segs/rec` column next to `pred segs/rec` in every dataset table.

Run `report5.write_report` on `ts5_v1/phase5` afterwards and check that only the edited sections changed. Every
F1@50 must be byte-identical to the current report.

### Verification before handing over

- `tests5` still passes 15/15, and `tests6` passes.
- `run_phase6 --smoke` completes end to end in ≤ 10 min. It uses capped recordings; its numbers are not results.
- `run_phase6 --dry-run` projection is about 4.7 h.
- **Do not launch the full run; the user launches it:**
  `.\script\week5\timescale_inspection\run_phase6.ps1 -RunId ts5_v1 -Hours 6`

## The final write-up (after the run)

When `phase6/REPORT.md` exists:

1. **Write `docs/plans/week5/TIMESCALE_INSPECTION/RESULTS.md`.** This is the closing document for Phases 0–6, written
   only from saved files. Sections:
   - what was asked;
   - what was ruled out: the Phase 0 ceilings, `none` not being the main cause, radius left untested;
   - the tokenizer result (H6, T3 vs T3b, T4);
   - the span rule with its dose-response table and the HuGaDB/LARa tests;
   - the 5-seed headline against SMQ;
   - what remains open;
   - corrections made along the way: the contextualizer-failure claim, the "span 8 scores nothing" claim, the PCA
     truncation premise on LARa.
2. **Update `TIMESCALE_INSPECTION.md`:** a short "Results" section pointing to `RESULTS.md`, and mark Phases 2/3 (span,
   window) as answered by Phases 5–6.
3. **Every number in `RESULTS.md` must trace to a unit file or a report table.** Before handing over, cross-check the
   headline table against the unit JSONs with a script.
