# Frozen-representation study

Implements `docs/plans/FROZEN_STUDY_AGENT_INSTRUCTIONS.md` (v1.0): discovery
vs. oracle vs. pooling on frozen `stage_a_pooled_002` / `stage_b_003` /
`stage_c_pooled_001` artifacts, with a cumulative-rung convergence ladder.
Nothing is trained. **This runner only saves outputs.** It never prints a
report, a verdict, or a metric value; console/log output is job status and
timing only.

## Files

- `frozen_study.py` -- everything: representations, prototypes, the ladder,
  the probe, job enumeration, the scheduler, and all file writing.
- `tests_frozen_study.py` -- the `-ValidateOnly` suite (synthetic fixtures;
  no dependence on real datasets except the final identity test, which only
  needs a nonexistent `stage_b_run` name).
- `run_frozen_study.ps1` -- the one launcher (validate / smoke / full).

## Launching

```powershell
.\script\action_transport\run_frozen_study.ps1 -ValidateOnly
.\script\action_transport\run_frozen_study.ps1 -Smoke
.\script\action_transport\run_frozen_study.ps1 -Hours 8 -RunId frozen_001            # the user runs this
.\script\action_transport\run_frozen_study.ps1 -Hours 4 -RunId frozen_001 -Resume     # if needed
```

`-Hours` is a total wall-clock budget; the cooperative solver deadline is
`Hours - 0.5h`, and the last 30 minutes are reserved for flushing files
(no reporting happens in that reserve -- there is no report).

## What gets saved (`results/frozen_study/<RunId>/`)

`manifest.json`, `jobs.csv`, `status.csv`, `rows.csv`, `fits.csv`,
`probe.csv`, `baseline_check.csv`, `not_run.csv`, `pilot.json`,
`artifacts/{job_id}.npz`, `logs/run.log` -- schemas exactly as specified in
the instructions document §6. `manifest.json` also carries the declared
Q1/Q2/Q3 analysis thresholds verbatim; **the runner never evaluates them.**

## 98 jobs, six priority tiers

P0 (`check_baselines`, `pilot`, synchronous in the parent process) -> P1
(16 seed-111 ladders on `pca`/`context` x `disc_saved`/`oracle`) -> P2 (16
pooled fit+ladder jobs at h in {2,8,32,64}, plus 16 pooled-oracle ladders)
-> P3 (4 extended fits, outer_cap 150, raw only) -> P4 (4 subject-grouped
probes) -> P5 (40 jobs replicating the seed-111 population at seeds 222 and
1538574472). Job IDs are deterministic strings (`enumerate_jobs()` returns
the same 98 IDs on every call); `jobs.csv` records the full plan up front.

## Judgment calls made while implementing this document

The instructions leave a few operational details to the implementer's
judgment. Recorded here per the instructions' own conflict-resolution rule
("record the conflict... do not silently change the method"):

1. **P0 jobs run synchronously in the parent, not through the worker pool.**
   `check_baselines` must abort the *entire* run on a metric mismatch, and
   `pilot`'s timing numbers gate every other job's cost projection --
   both are naturally single-shot, whole-run-scoped operations, so they run
   before the `ProcessPoolExecutor` starts rather than being one submitted
   job among many.
2. **Pilot dataset selection.** The pilot's "first 20 recordings" is run
   once per dataset at `normalize=False`, seed 111, using the `pca`
   representation with `disc_saved` prototypes (Stage A's A-cont `mu`). The
   instructions do not name a normalization for the pilot; raw was chosen
   because normalization does not materially change window count or step
   cost.
3. **Fit coefficients for `disc_fit`.** `run_disc_fit` uses
   `transport.Coeffs` built directly from `config.FIT_SETTING` (setting T,
   full beta) rather than routing through `contextual.coeffs_for`, since
   that helper's only special case (`arm == "context_only"` zeroes beta) does
   not apply to any representation in this study -- pooled/pca/context
   `disc_fit` are all full contextual-ASOT-style fits.
4. **`-Smoke` job set.** Rather than re-running the full 98-job schedule at
   tiny rungs, `-Smoke` runs a fixed 7-job list (`enumerate_smoke_jobs()`)
   that exercises every job *kind* once (`check_baselines`, `pilot`,
   `ladder` x2 -- disc_saved and oracle, `fit_ladder` x2 -- pooled and
   extended, `probe`), on 3 recordings/dataset, rungs `[10, 20]`,
   `outer_cap=2`, `h=2`, 2 workers. This satisfies "run every job type once"
   without inventing 98 tiny variants of a schedule the smoke run does not
   need to validate on its own (the full 98-job enumeration is checked
   separately and exactly by `tests_frozen_study.py`).
5. **`--resume` also retries `error`/`unavailable`/`interrupted` jobs**, not
   only `partial` ones. The instructions specify only that `partial` jobs
   restart from scratch; retrying any non-`complete` job on resume is a
   strict superset of that behavior and never silently drops a job.
6. **Ladder tolerances are not parameterized.** Every ladder solve (`L_*`,
   the ladder half of `F_*`/`X_*`) always uses `config.FINAL_GRAD_TOL`,
   `config.FINAL_OBJ_RELTOL`, and `config.FINAL_PATIENCE` -- "the default
   tolerances" per §3. The patience streak is reset at each rung boundary
   as a direct consequence of each rung being its own `transport.solve_final`
   call (documented in the instructions and verified by
   `test_deadline_yields_partial_with_completed_rungs_saved` and the
   cumulative-vs-single-solve equality test).
7. **Cost projection for `probe` jobs** uses a fixed 300-second placeholder
   (the pilot only measures transport-solve timing, not `sklearn` fit time),
   so probe jobs are effectively always scheduled unless the deadline itself
   has already passed.

## Tests

```powershell
.\script\action_transport\run_frozen_study.ps1 -ValidateOnly
```

17 tests covering pooling (hand-checked weighted average, edge truncation,
terminal-partial-window weighting, `h=0` identity), oracle prototypes
(hand-computed, present-classes-only, 1-indexed label handling, identity
state->label mapping), the ladder (cumulative-rung/single-solve numerical
equality with tolerances forced never to trigger early stopping; frozen
converged recordings; zero `frac_changed` when unchanged; a simulated
deadline preserving completed rungs), the scheduler (98-job count and
determinism, per-priority counts, resume semantics, the smoke job set),
the probe (subject-disjoint folds, deterministic subsampling, hand-computed
frame-weighted metrics), a structural label-separation check on the
fitting/representation code paths, and a real identity-abort check.
