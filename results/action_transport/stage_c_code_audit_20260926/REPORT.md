# Stage C implementation review

26 September 2026. Scope: `contextual.py`, `run_stage_c.py`, `report_stage_c.py`, `tests_stagec.py`, launchers and the Stage B freeze/diagnostic dependencies. No production source was changed and no experiment grid was launched. No completed Stage C run, or `stage_b_003`, was found in the standard results directories at review time.

**Assessment: the mathematical adapter is largely consistent with the intended method, but the runner and report are not ready for a full experiment.** In particular, the report can claim success from incomplete seed groups, resume can mix changed inputs, and interrupted secondary solves discard completed primary results. Several issues raised in the preceding instructions review remain present in code.

## Verified behavior

- `contextual.coeffs_for` sets beta to zero for context-only in both T and E, retaining other coefficients.
- Costs use `1 - z @ mu.T`, with no half scaling. Initialization and prototype updates reuse the existing frame-weighted continuous machinery.
- Both fitted arms fit under T and perform fixed-prototype final solves under T and E. The fixed filter is derived from the corresponding context-only raw states.
- The order diagnostic permutes codes, extracts embeddings, restores embedding order, refits prototypes and applies chronological transport. There is no second restoration of predictions in this path.
- Enumeration gives 28 fits and 80 prediction sets.
- The existing Stage C suite passes **14/14**. Its real-data Stage A re-scoring check also passes. These tests do not cover the report completeness, changed-content resume, invalid numeric inputs or interruption scenarios reproduced below.

## P1 — Incorrect research conclusions from incomplete results

Sources: `report_stage_c.py:51–103,168–188`.

`mean_by`, `primary_bar_table` and `partial_result_table` average whichever rows are present. They do not require the exact three seed IDs or reject duplicates. The output nevertheless calls these three-seed means.

**Reproduction:** supplying only seed 111 for contextual ASOT in each of the four conditions, with score 30 against A-cont 10 and SMQ 20, makes `primary_bar_table(...)[1]` return `True`. Only four contextual results are present, rather than the required twelve.

The opposite failure occurs with absent results: both criterion functions return `False`, leading `build_report` to conclude that neither criterion is met and the research direction is unsupported. Absence of evidence is being reported as a negative experiment. Coverage also counts all CSV prediction IDs as scored, including `invalid_input` rows without metrics.

**Required correction:** validate unique expected seeds, full recording coverage and finite metrics for every required comparison. Add an explicit incomplete/not-assessable outcome. Show seed count with every aggregate. Keep finite capped results under the declared budgeted protocol, but never silently turn an incomplete seed group into a complete result.

The partial verdict additionally infers “context, not transport, is doing the work” solely from predicted/GT median-duration ratios. Similar durations do not establish correct boundaries or action assignments, and do not isolate the incremental transport effect. Report the duration criterion separately and use the paired segmentation comparisons for the measured transport effect.

## P1 — Resume does not identify the actual experiment artifacts

Sources: `run_stage_c.py:270–302`; `run_stage_b.py:45,247`.

Stage C resume compares only the Stage C version and the names of the Stage A/B runs. It does not compare model checkpoints, embedding/index contents, Stage A artifacts, solver/source hashes or effective settings with the previous launch. A changed upstream run under the same directory name is accepted.

**Reproduction:** write a Stage C manifest against a synthetic Stage B run; change that Stage B manifest's source hash while retaining the run name and selected configuration hash; call `write_manifest` with resume. It succeeds.

Stage B similarly stores source hashes but does not compare them: `IDENTITY_FIELDS` contains only version, tokenizer version and TrainConfig. Architecture implementation can change without changing those fields. Existing split files are returned without checking them against `split_source` on resume.

**Required correction:** freeze and compare a dependency identity comprising the effective configuration, source/solver hashes, selected-study identity, split contents, tokenizer/code identities, checkpoint hashes and per-recording embedding hashes. Stage C must snapshot the relevant Stage A comparison artifacts as well. Store the dependency references in each cell/prediction artifact. Refuse changes under an existing run ID instead of continuing with mixed results.

## P1 — A deadline during E discards a finished T solve and the fit

Sources: `run_stage_c.py:89–119,170–205,391–397`.

`run_context_arm` builds an in-memory output containing T, then computes E, and returns only after both finish. `run_one_cell` saves the fitted artifacts and predictions only after that return. If E reaches the deadline, the exception skips all saving, including the completed T result and fitted prototypes. The order-diagnostic path has the same structure.

**Reproduction:** replace the fit with a small fixture, let final T complete, and raise `DeadlineExceeded` during E. The function propagates the exception after visiting `['T', 'E']`, without returning T. The enclosing runner records the whole fit as not run.

This contradicts both T priority and incremental artifact caching. It can waste the last substantial fit of a budgeted run and force refitting on resume.

**Required correction:** persist the fit when fitting ends, then atomically persist each final prediction set as it completes. Track and resume at the fit/setting level. Preserve completed T outputs when E is interrupted. Add cooperative checks during order-diagnostic embedding extraction and after each recording; the current extraction loop accepts no deadline.

## P1 — Embedding validation permits modified and nonfinite inputs

Sources: `contextual.py:42–93,154–170`.

The loader checks the selected configuration string, the checkpoint file hash, existence and declared embedding shape. It does not hash embedding contents or verify finite values, unit norms, zero-flag shape/consistency, or the current recording's expected window count independently of INDEX.json. `context_inputs` silently normalizes whatever arrays it receives.

**Reproduction:** a valid synthetic embedding file was replaced by one of the same shape containing NaNs, without changing the checkpoint or index. `load_stage_b_cell` accepted it. Passing it through `context_inputs` propagated NaNs. A row with norm 2 was silently normalized rather than rejected by the required pre-normalization check.

Nonfinite data may fail later as an optimizer result instead of being identified as invalid input. A finite same-shaped substitution can silently change the experiment entirely.

**Required correction:** verify finite values, exact shapes against the current recording grid, boolean zero flags and pre-normalization norms; enforce consistent treatment of flagged zero rows. Compare per-file hashes with an immutable index. Check the recorded Stage B/source/architecture identity and checkpoint metadata as well as the selected TrainConfig hash.

## P2 — The report is incomplete and can crash on an expected failure row

Sources: `report_stage_c.py:134–144,205–256`.

`order_diagnostic_table` formats `num(r['F1@50'])` with `:.2f` without first excluding or displaying unavailable values. An `invalid_input` diagnostic row has a blank metric.

**Reproduction:** such a row raises `TypeError: unsupported format string passed to NoneType.__format__`. This prevents the final report precisely when a failed diagnostic needs documenting.

The report also omits substantial requested content:

- E-setting main-arm metric tables; only the order diagnostic displays E.
- Full five-metric comparisons with A-cont, A1-KC and SMQ; the representation matrix contains only hard K=500 and contextual arms for T F1@50.
- Paired context-only vs A0 and contextual ASOT vs A1/A-cont difference tables.
- Fit/final convergence counts, residuals, runtime and detailed completeness tables.
- Saved subject-association diagnostics, span-depth results and full embedding diagnostics.
- Figures and fixed-selection timelines.
- The stopping-implementation difference between historical Stage A controls and the corrected solver used by Stage C.

**Required correction:** make the report robust to every planned status and complete the declared comparison matrix. Report historical controls with their numerical-protocol caveat; re-scoring confirms their evaluator, not equivalence of their original stopping procedure.

## P2 — Required subject-neighbor diagnostics allocate a huge dense matrix

Source: `run_stage_b.py:342–376`, especially `sims = norm[sample] @ norm.T`.

The command loads all embeddings as float64 and computes all sampled-query/all-window similarities in one allocation. At the cached dataset sizes and 10,000 queries, this matrix alone is approximately **5.67 GiB for HuGaDB** and **16.35 GiB for LARa**, before input arrays, matrix-product workspace and sorting temporaries. It then fully sorts the candidate list twice for each query.

This is a deterministic allocation requirement, not an observed OOM in this audit. It is nevertheless unnecessary for a 10-neighbor diagnostic and risks making a required reporting step disproportionately expensive.

The sampler includes terminal partial windows, contrary to the specified complete-window population. Its single RNG advances across cells, so compared representations/seeds use different query samples. The reported composition references also omit the conditional same-subject reference after excluding same-recording candidates.

**Required correction:** batch queries (and candidates if needed), retain only top-k candidates, sample shared complete-window IDs per dataset, and compute the reference under the actual candidate exclusions. Report coverage when fewer than k eligible candidates exist instead of admitting excluded `-inf` entries.

## P2 — Several advertised provenance/preflight checks are absent or not connected

Sources: `contextual.py:125–150,202–220`; `run_stage_c.py:431–473`.

- Stage A continuity checks grouping `g` only. The specified additional A-cont `mu0` consistency check is absent. Even both checks together would not prove historical code-sequence identity; the limitation from the instructions review remains.
- `check_permutation_matches_stage_a` is defined but never called by the runner. Even if called, it silently returns for a missing artifact and ignores absent recording names. The README nevertheless says the permutations are checked.
- Preflight checks file presence/shape and checkpoint hashes, but does not run tokenizer continuity, Stage A grouping checks or `verify_stage_a_reuse`. Having only some of the 12 Stage B cells is a warning unless none are available.
- `verify_stage_a_reuse` itself warns and continues if fewer than four dataset/normalization conditions can be checked.
- Stage B can train all twelve models before Stage C discovers that the Stage A continuity checks fail; there is no corresponding pre-training gate in `run_one`.

**Required correction:** invoke the existing checks at the documented boundaries; verify complete expected name sets and make the production preflight result explicit. Distinguish checks skipped for synthetic fixtures from checks required for real production inputs.

## Additional operational gaps

- `_load_completed` treats every CSV row as completed, regardless of status. An `invalid_input` row is permanently skipped on resume even if a missing Stage B artifact is later supplied. Define a retry policy explicitly.
- `not_run.csv` is built only for the current invocation, excludes invalid/unavailable cells returned as CSV rows, and is overwritten on resume. Derive an exhaustive current status table from planned IDs plus completed/failed artifacts.
- Cells are saved only after both final solves, and a partial CSV append can cause duplicate prediction IDs after rerunning a partially written fit. Require atomic per-prediction records or deduplicate before aggregation.
- The production runner has no dedicated smoke mode or smoke-namespace enforcement. `-TimingProbe` is absent from the launcher/parser even though it is part of the instructions.
- `resolve_train_config('selected')` validates the chosen name against its sweep hash but does not require the pinned selected name/hash. A changed selection pointing to another legitimate sweep row can train successfully, only to be rejected by Stage C afterward. Pin the intended study choice before training.

## Minimal repair order

1. Fix completeness-aware reporting and invalid-row rendering, with regression fixtures.
2. Freeze and enforce artifact/source identity; add numeric input checks.
3. Persist fitting and T/E results incrementally, with an interruption/resume test.
4. Connect continuity/permutation/preflight checks and make failure/retry status bookkeeping complete.
5. Batch subject-neighbor diagnostics and finish the declared report outputs.

Retain the current mathematical arms and grid. The review does not identify a need for an architecture search or a change to the coefficients. These are correctness, reproducibility and completion fixes before committing compute to the planned experiment.
