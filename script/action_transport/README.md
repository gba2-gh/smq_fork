# Stage A: categorical/continuous temporal transport

Implements Stage A of `docs/plans/THREE_STAGE_MOTION_PLAN.md` and
`docs/plans/STAGE_A_AGENT_INSTRUCTIONS.md` (both version 1.3). Runner
version **1.3.1** = the v1.3 method with the implementation fixes listed
below. The declared objective, coefficients, arms, budgets and stopping
rules are unchanged.

## Layout

```text
config.py       frozen settings (T/E), arms, cell enumeration, versions
data.py         provenance gate, dataset loading, folds, tokenizer build/cache
transport.py    torch float64 log-domain mirror-descent solver, objective/gradient, kernel
categorical.py  A0/A1/A1-KC: vocabulary-level init, emissions, shared alternating fit
continuous.py   A-cont: cosine cost, prototype init/update
smoothing.py    A0+filter: fixed frame-weighted mode filter
permute.py      permuted_asot: within-recording permutation + restoration
evaluate.py     the only module that sees action labels: Hungarian mapping, scorer, diagnostics
budget.py       cooperative deadline check
run.py          CLI: --preflight / --validate-only / --dry-run / --timing-probe / full run
report.py       REPORT.md from saved artifacts only (no refitting)
tests.py        validation suite (26 tests)
run_stage_a.ps1 the one user-facing launcher
```

`third_party/action_seg_ot/` holds the pinned upstream ASOT source (commit
`0c4c86b1037eb4c3e262197aeee115b31e220c9d`), unchanged, used only by the
parity test.

## What changed in 1.3.1 (and why)

1. **Solver bug: every final solve was labelled `capped`.** v1.3 computed
   the entropy gradient as `log(T + 1e-12)`. At the optimum, many entries of
   T are legitimately below 1e-12 (a/eps = 10 with categorical costs up to
   ~2.5). For those entries the floored log gave a false gradient, so the
   row-centered residual could never reach 1e-5. All 126 v1.3 prediction
   sets therefore ran their full 500 final steps and were marked `capped`,
   even where the objective had stopped changing to machine precision. The
   solver now keeps log T as its state and uses the exact log (the
   instructions already asked for "stable log-domain mirror descent").
   Argmax states are unaffected except where two states were effectively
   tied, so v1.3 metrics should reproduce closely. The `--compare-to`
   report section checks this.
2. **SMQ baseline scoring added.** The plan's first comparison (A1 vs SMQ)
   had no matched number in v1.3. The runner now scores the released
   checkpoint's own per-frame codes with the same evaluator, first, as two
   fit-free jobs. On full LARa it reproduces the segmodel audit exactly
   (MoF 37.38, F1@50 16.39). HuGaDB should give 41.99 / 24.30; the report
   prints both as a provenance check.
3. **Raw states and diagnostics saved.** v1.3 saved only metric rows, so
   nothing could be re-analysed without refitting. Now each prediction set
   gets `predictions/<id>.npz`: raw per-window states on the original
   timeline, the Hungarian mapping, and metadata. `cells.csv` gains:
   - predicted vs ground-truth run counts and their ratio
   - run-duration quantiles (seconds)
   - occupied states, occupancy entropy and largest-state fraction
   - frame-weighted MI(state; subject)
   - fit and final-solve convergence details: status counts, max and
     frame-weighted median residual, mean steps
4. **Subject-disjoint scoring fixed (not used by the pooled rerun).** Fold
   cells now score only their held-out recordings. Once all four folds of a
   group exist, an out-of-fold row is aggregated with one Hungarian mapping
   per fold. v1.3 scored fitting recordings too.
5. **Operational fixes:**
   - Datasets alternate in the schedule, as instructions §3 require.
   - The deadline projection uses the median runtime of the same
     (dataset, arm).
   - `--resume` refuses a `cells.csv` or manifest from a different runner
     version.
   - The manifest records every launch and its host.
   - `--preflight` checks a machine before a run.
   - The launcher finds the `smq` env on another machine (`-PythonExe` or
     `$env:SMQ_PYTHON` override).
   - The launcher runs reports directly with `-Report`.

**Known remaining limitation:** with the temporal term, the final solve
converges linearly (about 0.99 contraction per step on a synthetic
300-window case), so reaching the 1e-5 residual can take more than the
declared 500 steps from a cold start. Final solves are warm-started from
the fitted state. In the smoke test, T-setting solves mostly converged
while E-setting solves (whose smaller epsilon also contracts more slowly)
were often still `capped`. Those rows are now genuinely capped (residual
recorded), not artefacts. The budget was **not** raised: the plan freezes
it, so any change is your decision (label-free, but still a protocol
change).

## What changed in 1.6 (external audit response, 26 September 2026)

An external audit of `results/action_transport/audit_20260926/` found and I confirmed:

1. **Outer stopping objective mixed parameter states.** `fit_alternating`'s outer trace summed each recording's inner-solve objective at the pre-update parameters and added the prior at the post-update parameters -- not the declared `sum_r N_r*F(T_new;params_new) + prior(params_new)` of the returned iterate. Reproduced exactly against the audit's independent case (2 recordings, K=6, C=3, one outer iteration): old value 17.305001509113488, correct value 7.008690581479591, matching to 1e-9. Fixed by recomputing the objective at the already-solved couplings under the updated parameters (one extra objective evaluation per recording per outer iteration, no re-solving). Two regression tests added (`test_outer_objective_matches_returned_iterate`, `test_outer_objective_nonincreasing_across_iterations`).
2. **Tokenizer-cache identity was too weak.** Only fitting indices and per-recording window counts were checked, so a same-shaped but reordered or differently-sourced dataset could pass silently. Now also checks the exact recording-name list and the checkpoint/latent-cache identity string; caches saved before this fix are rejected with a clear message rather than silently reused (all 12 on this machine were deleted and will rebuild deterministically on first use).

**Impact on `stage_a_pooled_002`:** 43 of 52 fits ran to the 50-iteration outer cap; their per-iteration updates are unaffected by a stopping-trace fix. The 9 that stopped early may have stopped at a different iteration under the corrected trace. This has not been re-run; whether any reported score changes is not yet established. The plan (§3.5) now states this rather than asserting the fix explains the score deficit.

## Verification done before handoff

- `-ValidateOnly`: **26/26 tests pass**. The suite covers:
  - reference parity against the pinned upstream solver (max |ΔR| = 1.4e-11;
    the remainder is the reference's own `log(T+1e-12)`)
  - finite-difference objective/gradient checks under T and E
  - the log-domain objective matching the declared formula term by term
  - the convergence-bug regression
  - fold-specific mapping, the segment diagnostics, enumeration counts and
    alternation, and the resume guard
  - the earlier v1.3 tests
- `-Preflight` on this machine: 0 failures, 0 warnings.
- A plumbing smoke test ran every arm on a 6-recording real subset for both
  datasets:
  - the scorer was stubbed for fitted arms, so no model's action score was
    computed;
  - the real scorer, diagnostics and out-of-fold aggregation were exercised
    on random states;
  - the SMQ baseline job ran on full LARa;
  - A1-KC was correctly `unavailable` on the tiny HuGaDB subset (only 5 of
    10 codes occupied), as declared; on full data all are occupied.

## Running the pooled rerun on another computer

### 1. Code

The code is in git. Commit and push these changes on this machine, then on
the other machine pull and check out the same commit. `models/` and
`results/` are gitignored, so they must be copied separately (next step).

### 2. Copy these files (not in git; keep the same relative paths)

| Path | Size | Needed for |
|---|---|---|
| `models/pretrained/hugadb.model`, `models/pretrained/lara.model` | 8 MB | provenance gate (SHA-256 checked) |
| `results/preds/hugadb_pretrained.npz`, `results/preds/lara_pretrained.npz` | <1 MB | recording order, labels, historical agreement |
| `results/exp2round/q1q2/cache/hugadb/` | 418 MB | HuGaDB latents (the released-checkpoint cache, *not* `results/seg/cache/hugadb`) |
| `results/seg/cache/lara/` | 3.5 GB | LARa latents |
| `results/action_transport/cache/tokenizers/` | 410 MB | strongly recommended: identical codes to this machine. If rebuilt elsewhere, KMeans can differ with BLAS/CPU |
| `results/action_transport/stage_a_pooled_001/` | small | optional: lets the report check reproducibility against the v1.3 run |

The 18 GB `data/` folder is **not** needed, as long as the latent caches
above are present. Without them, the gate would try to re-extract latents
from `data/`.

### 3. Environment

A conda env named `smq` from `environment.yml`, or pass `-PythonExe`. The
reference versions are Python 3.10.14, numpy 1.26.4, scikit-learn 1.2.2 and
torch 2.1.0; `-Preflight` warns if they differ.

### 4. Commands (from the repo root, in PowerShell)

```powershell
.\script\action_transport\run_stage_a.ps1 -Preflight           # must show 0 failures
.\script\action_transport\run_stage_a.ps1 -ValidateOnly        # must show 26/26
.\script\action_transport\run_stage_a.ps1 -DryRun -Protocol pooled
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_002
# if it stops at the deadline, continue with a new budget (same RunId):
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 4 -RunId stage_a_pooled_002 -Resume
# report (with a reproducibility check against the v1.3 run):
.\script\action_transport\run_stage_a.ps1 -Report -RunId stage_a_pooled_002 -CompareTo stage_a_pooled_001
```

On runtime: v1.3 took about 5.3 h of fitting for 51 of the 52 fits on this
machine (8 threads), with every final solve forced to 500 steps. 1.3.1
stops final solves early where they converge, so expect less. `-Hours 8`
leaves margin, and `-Resume` never redoes finished fits. Do not pass
`-Resume` pointing at `stage_a_pooled_001`: the runner refuses it because
the runner version and columns differ.

## What a run produces (`results/action_transport/<run_id>/`)

```text
manifest.json       versions, settings, every launch (host, hours, packages)
planned_cells.csv   52 fits (+2 SMQ baseline jobs) and their prediction-set ids
cells.csv           one row per scored prediction set: 5 metrics, status, convergence, diagnostics
not_run.csv         fits skipped by the deadline projection or interrupted, with reasons
artifacts/          per fit: theta/mu, theta0/mu0, grouping, code counts, objective trace, permutations
predictions/        per prediction set: raw window states (original timeline) + mapping
REPORT.md           written by -Report
```

## Not covered

- Subject-disjoint was not run. Do not run it for Stage A unless the pooled
  results justify it (plan §3 decision point).
- Tests do not cover crash recovery of atomic writes, a simulated mid-run
  deadline interruption, out-of-fold aggregation on real fold fits, or
  CPU/CUDA agreement.
- Figures and timeline examples from the plan's reporting list are not
  generated.
