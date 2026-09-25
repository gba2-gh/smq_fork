# Stage A: categorical/continuous temporal transport

Implements Stage A of `docs/plans/THREE_STAGE_MOTION_PLAN.md` and
`docs/plans/STAGE_A_AGENT_INSTRUCTIONS.md`, both version 1.3. This directory
is the whole deliverable; nothing here has launched a full run.

## Layout

```text
config.py       explicit dataclasses, T/E coefficient settings, planned-cell enumeration
data.py         provenance gate, dataset loading, folds, tokenizer build/cache
transport.py    torch float64 mirror-descent solver, objective/gradient, kernel
categorical.py  A0/A1/A1-KC: vocabulary-level init, emission cost/update, alternator
continuous.py   A-cont: cosine cost, prototype init/update
smoothing.py    A0+filter: fixed frame-weighted mode filter
permute.py      permuted_asot: within-recording permutation + restoration
evaluate.py     the only module that receives action labels: Hungarian mapping, scorer,
                subject-state MI diagnostic
budget.py       cooperative deadline check (DeadlineExceeded)
run.py          CLI: --validate-only / --dry-run / --timing-probe / full run
tests.py        the validation suite run by --validate-only
run_stage_a.ps1 the one user-facing execution script
```

`third_party/action_seg_ot/` holds the pinned upstream ASOT source
(commit `0c4c86b1037eb4c3e262197aeee115b31e220c9d`, matching the local clone's
HEAD at `C:\Users\gzaz976\Documents\repo\Behaviour\action_seg_ot`), copied
unchanged with its license, for parity testing only. Production code does
not import or modify it.

## What was verified

`-ValidateOnly` runs 20 synthetic tests, all passing:

- the temporal-kernel convolution against a dense band-matrix reference;
- the documented v1.3 rounding trap in `int(L*(b/L))` (b=4 fails to recover
  itself at L=49/98/103, holds at L=16/32/64) -- production never uses this
  path, it builds the kernel from the integer half-width directly;
- the declared objective against its gradient by finite differences, under
  both coefficient settings and with beta=0;
- that mirror descent keeps each recording's row masses exactly at `p`
  (max error <1e-9);
- the cosine-cost conventions (equal->0, orthogonal->1, opposite->2,
  zero-input->1, matching the v1.3 [0,2] convention, not v1.2's halved one);
- that the pooled emission update and the A-cont prototype update do not
  increase the objective with T fixed;
- the vocabulary grouping (K=C identity shortcut, the "too few occupied
  prototypes" unavailable path, theta0's pseudocount/normalization by hand);
- the mode filter's hand-computed votes and tie rule;
- the permutation round trip (`perm`/`inverse_perm`/`restore_states`) and
  that A0 (no temporal coupling) is equivariant under it: theta and restored
  raw states match the unpermuted fit to float64 tolerance;
- **exact parity** (max |R_ref - R_ours| = 0.0) against the pinned upstream
  solver in equal-mass mode on a tiny case, once the reference's own
  `T *= nnz` output rescaling is accounted for (see `tests.py`'s docstring
  on that comparison -- getting this wrong first gave a spurious ~15x
  "mismatch" that was a unit-convention bug in the test, not the solver);
- an evaluator round trip: identity prediction scores MoF=100, and a pure
  relabelling recovers MoF=100 through the Hungarian mapping.

I also ran a tiny real-data smoke check (6 recordings per dataset, a fresh
small tokenizer, `outer_cap=5`) exercising A0/A1/A1-KC/A-cont end to end on
both HuGaDB and LARa: the same-checkpoint provenance gate passed for both
datasets, all fits produced finite theta/mu and correctly shaped predictions,
and most small fits ended `capped` rather than `converged` at this tiny
`outer_cap` -- expected, and worth watching in the real run too, since the
default `outer_cap=50`/`patience=3`/`reltol=1e-6` schedule is a real
stopping bar. Per instructions §6, this smoke check verified shapes and
timing only; it did not evaluate or report any action score, and its
scratch script was deleted afterward.

`-DryRun` correctly enumerates 52 pooled fits (128 prediction sets) and 192
subject-disjoint fits (480 prediction sets), matching the plan's counts
exactly. `-TimingProbe` ran on real (tiny-subset) data and wrote
`results/action_transport/cache/timings.json`, which `-DryRun` then uses to
project a cost estimate.

## Known limitations (stated, not hidden)

- **Test coverage.** The suite above covers the numerically load-bearing
  contracts. It does **not** yet cover: atomic-write recovery under a
  simulated crash, resume correctly refusing a changed configuration, cell
  deduplication, a simulated mid-run deadline interruption, or CPU/CUDA
  numerical agreement (no CUDA exercised here, though the code path is
  device-parametric and `torch.cuda.is_available()` is `True` on this
  machine). These are mechanically straightforward given the rest of the
  implementation, but were not exercised.
- **Deadline granularity.** The cooperative deadline check
  (`budget.check_deadline`) is checked once per recording inside the
  fitting/inference loops, not per mirror-descent step. A single
  recording's up-to-500-step final solve cannot be interrupted mid-solve.
  This is coarser than ideal but bounds the worst case to one recording's
  solve time, which is small relative to an hours-scale budget.
- **Batching.** Per instructions §5 ("implement per-recording correctness
  first ... batching optional"), the solver loops over recordings rather
  than padding them into a batch. This is simpler and was what got verified
  against the reference; it is also slower than a padded-batch
  implementation would be. The timing probe's projection already reflects
  this per-recording cost, so `-DryRun`'s hour estimate accounts for it.
- **Permutation prediction-set count.** The plan does not explicitly state
  whether `permuted_asot` yields one or two scored prediction sets. This
  implementation computes both T and E (2 sets per permutation fit, 8 total
  across the 4 permutation fits) since the marginal cost is one extra
  final re-solve; the primary sanity comparison should still use T.

## Launch commands

```powershell
# Read-only checks (no fitting):
.\script\action_transport\run_stage_a.ps1 -ValidateOnly
.\script\action_transport\run_stage_a.ps1 -DryRun -Protocol pooled
.\script\action_transport\run_stage_a.ps1 -DryRun -Protocol subject_disjoint

# Real but tiny label-blind timing probe (updates the -DryRun cost estimate):
.\script\action_transport\run_stage_a.ps1 -TimingProbe

# Full pooled run (52 fits, 128 scored prediction sets). Pick -Hours to
# comfortably exceed the -DryRun projection; it already reserves 30 minutes
# for reporting on top of whatever you pass.
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 4 -RunId stage_a_pooled_001

# Resume an interrupted/partial pooled run under the same RunId:
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 4 -RunId stage_a_pooled_001 -Resume

# Subject-disjoint (192 fits) -- only after reading the pooled REPORT.md and
# deciding to continue; must name the completed pooled run:
.\script\action_transport\run_stage_a.ps1 -Protocol subject_disjoint -AfterPooledRun stage_a_pooled_001 -Hours 8 -RunId stage_a_sd_001

# Explicit CUDA (after -ValidateOnly has passed on this machine; CPU is default):
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 4 -RunId stage_a_pooled_001 -Device cuda
```

I have not run any of the four full-run commands above -- per the
instructions, the user launches the full experiment. Everything before that
line (`-ValidateOnly`, `-DryRun`, `-TimingProbe`) has actually been run
during implementation, with output shown above.

## What the runner produces

Under `results/action_transport/<run_id>/`:

```text
manifest.json       config/protocol-version/device/hours/packages/AfterPooledRun
planned_cells.csv   every planned fit and its prediction-set ids
cells.csv           one row per scored prediction set: all five metrics, status,
                    fit_outer_status, runtime, device
not_run.csv         cells skipped by the deadline projection or interrupted mid-fit
artifacts/          theta/mu/grouping/permutations per fit
predictions/        per-fit JSON with the same rows as cells.csv (for reload without refitting)
```

`REPORT.md` and `figures/` are not generated by this handoff -- the
instructions ask for readable scripts, tests, and a README at this stage;
the report-generation step (reading `cells.csv`/`not_run.csv` without
refitting, producing the plan's comparison tables and figures) is a small,
separable follow-up once real `cells.csv` rows exist from an actual launch.
I did not want to write a report generator against data that does not yet
exist and risk it silently assuming a shape the real run doesn't produce.
