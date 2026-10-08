# Milestone 1 implementation review

Reviewed 6 October 2026 against instructions v2.2. Package identity: `f16c3fdbf5974f04`.

## Verdict

The scientific core implements the intended fixed-unit experiment coherently. No direct evaluation-to-training leakage, causal-history alignment error, bag-control order leak, or incorrect pseudocount weighting was found. All 18 existing tests pass, all three real-data preflights pass, and a new current-code CUDA smoke run completes all four primary-split arms.

The implementation is **not yet ready for an unattended decision run**. The main blockers concern result preservation and the evidence accepted by the gate, rather than a demonstrated error in the model likelihood. Optional failures can invalidate already-computed primary results; cache identity checks can be bypassed or overwrite existing results; and saved “final plans” are not the final decoded plans.

This is a review, not a code repair. No implementation files or experimental settings were changed. No full training or historical full-population parity run was launched.

## Verification performed

- Read all new implementation modules, tests, launchers, README and v2.1/v2.2 changes, plus relevant upstream grouping, transport and scoring helpers.
- Ran all 18 contract tests without replacing the existing verification file: 18 passed.
- Ran real-data preflight at 111, 222 and 1538574472. Each has 395 fitting/44 evaluation recordings, 219,500 windows, 16 subjects on each side. All-recording grouping reconstruction differs from saved grouping on **zero codes** at all three seeds.
- Ran a fresh reduced CUDA smoke under a three-minute cap: `results/compositional_dynamics/m1_review_smoke_20261006/`. All four arms complete; the report correctly does not evaluate the feasibility gate. These are 60-update/60-final-step plumbing results, not hypothesis evidence.
- Injected an optional multiplier failure using temporary synthetic fixtures: primary fitting/evaluation decodes finish, then the cell returns invalid and does not save primary scores.
- Tested cache and gate boundaries with temporary/synthetic fixtures: incompatible completion metadata gets overwritten; a completion stub without models/plans/predictions passes the cache predicate; stale smoke-shaped cells can produce a PASSED gate when presented under a full-run request.
- Inspected actual saved artifacts and replayed both float16 scaled costs and reconstructed float16 NLL components on the fresh smoke evaluation population. Both reproduce all 2,000 smoke evaluation window states. This does not establish precision equivalence at the full 4,000-step endpoint.

## Findings, ordered by importance

### 1. High: optional decodes can destroy the primary result

Locations: `run.py:158`, `run.py:217`, `run.py:225`.

Both primary decodes and scores are computed before multiplier sensitivities, but completion and primary artifacts are written only after sensitivities and order diagnostics. One shared exception handler marks the entire cell invalid/interrupted. The matched B0 dependency then also prevents downstream arms from using an otherwise successful baseline.

Reproduction: an injected failure on multiplier 0.5 produced decode calls `[1.0, 1.0, 0.5]`, cell status `invalid`, exit 4, and no `scores.json`, despite successful primary fitting/evaluation decodes.

This contradicts v2.2's requirement that secondary results cannot overturn the primary gate. Save and commit the primary scientific result before optional work. Give every secondary job an independent identity/status/deadline; a failed sensitivity must leave the primary eligible. Ideally schedule optional multipliers after all primary cells. Treat required order diagnostics explicitly if they fail, preserving the computed primary artifacts even when verification remains incomplete.

### 2. High: caching does not enforce the promised provenance boundary

Locations: `run.py:64`, `run.py:88`, `run.py:95`, `run.py:103`; `config.py:75`.

`completed()` verifies only info status, identity equality and existence of `scores.json`. An empty scores file plus completion metadata passes even when all scientific artifacts are missing. Dependent cells accept any B0 marked complete and take its recorded identity without checking that it is the currently expected B0 for this source/configuration. This matters when running selected arms against existing baseline cells.

More seriously, when `-ReuseCompleted` encounters an identity mismatch, execution falls through and writes `status=running` over the old completion metadata instead of refusing the incompatible cell. A controlled test confirmed this overwrite before any fitting began. Run-level checks reject a changed local package hash, but do not protect every source mismatch or independently invoked cell.

The code hash excludes imported grouping, solver, evaluator and label-loading implementations. Changing those numerical dependencies can leave old cells reusable under an unchanged local package identity. Device/environment changes are also not part of the cell identity.

Fix: hash the actually used numerical dependencies; validate current expected B0 identity and required artifact hashes; reject incompatible existing cells; require a new run ID or explicit replacement policy. Completion should be an atomic manifest of verified outputs, not merely an info flag plus one filename.

### 3. High: the gate trusts cells without verifying their experimental identity

Locations: `report.py:46`, `report.py:68`, `report.py:162`.

The gate thresholds and paired arithmetic are implemented correctly, but the gate does not check each cell's code/source identity, smoke/full scale, completed update budget, evaluation membership or final endpoint. It trusts score dictionaries loaded from the run directory and checks only the separately stored test-verification hash.

Reproduction: synthetic cells explicitly marked with stale code identity and smoke status, containing a 60-step endpoint, return **PASSED** when `evaluate_gate(..., smoke=False, verification=current_pass)` is called with favorable scores. This is a validation-boundary test, not evidence that existing smoke results have already been reported as full results.

Fix: validate the complete expected cell matrix against the run manifest, exact seed set, common split, full scale, primary endpoint and current dependency identity before computing a conclusive verdict. Persist preflight evidence and distinguish verified primary results from completed-but-unverified data.

Historical parity also needs an explicit policy. v2.2 permits continuing after mismatch, while the earlier verification language says failed checks make the decision inconclusive. The code currently treats parity as informational even for the verdict, and does not validate the parity artifact's current identity. Continuing computation can be allowed while withholding a verified verdict until a mismatch is explained. No saved completed parity result was present during this review.

### 4. Medium: “final_plans” are stale training plans; final inference evidence is discarded

Locations: `run.py:139`, `run.py:203`; `decode.py:42`, `decode.py:51`.

`final_plans.npz/R_final` comes from the last training refresh, before the last 500 model updates in a full run. It is not the 5,000-update model's final 4,000-budget decode. Evaluation soft plans are never saved. In the fresh smoke, its argmax differs from the actual final fitting decode on **567/6,000 windows**. That is expected for different iterates, but the filename/field is misleading for later analysis.

`decode_ladder` discards the full objective trace and does not return the soft plans. The runner further reduces its per-recording diagnostics to aggregate step counts/status counts and a maximum residual. The specified per-recording solver evidence, final responsibilities and some deferred posterior diagnostics are therefore not directly available from saved final outputs.

Fix: name the existing artifact `last_training_responsibilities`, save actual final T or R on both populations with recording offsets and endpoint identity, and retain per-recording solver traces/residuals/status. Preserve raw cost components at float32 or better for later solver replays; float16 history representations remain acceptable. No smoke prediction change from float16 costs was observed, so reduced precision is a reproducibility limitation, not a demonstrated scoring error here.

### 5. Medium: the elapsed-time limit is cooperative and can be exceeded

Locations: `train.py:226`, `train.py:269`; `run.py:333`; `run_overnight.ps1:32`.

There are no deadline checks within a training block or a batched whole-population cost evaluation. Warm fitting occurs before the first outer-loop deadline check. A final solver call can consume its entire remaining iteration budget before the next recording-level check. The parent waits on the child without a timeout.

Preflight/parity also have no overall deadline enforcement. The overnight script explicitly grants at least 0.1 further hours even if the original budget has already expired. This is not a hard ten-hour limit.

Fix: propagate one absolute deadline, check it inside update/batch loops, use bounded solver chunks or an external watchdog, and return without launching when no time remains. Preserve primary completion before optional work. Deadline-skipped runs currently can also return exit zero because skipped work does not update the driver's worst exit status; the incomplete state should be machine-visible.

### 6. Medium: tests and saved diagnostics cover less than the handoff claims

Locations: `tests.py:65`, `tests.py:350`; `train.py:163`; `evaluate.py:74`.

Two assertions contain `or True`. The coverage test catches even its own deliberately raised assertion, so that negative check cannot fail; the matched-arm assertion is also vacuous. Other tests do independently cover some of the same invariants, so this does not invalidate all 18 tests. Add tests for the failure paths above rather than relying on the aggregate count.

Several promised outputs are absent: grouping per-code centres/counts are discarded; MI/NMI contingency tables are not saved; predictive accuracy/copy diagnostics and a complete RNG/source manifest are missing. Some can be reconstructed from codes/predictions, but the implementation should either save them or accurately narrow the contract. The current assigned-state NLL is clearly defined in code; it is not a marginal held-out likelihood independent of the decoded assignment, and should remain labelled that way in analysis.

## What is implemented correctly

The fitting and evaluation populations are separated through matched-B0 fitting, occupancy updates, model gradients, scale calibration and assignment refreshes. Evaluation labels enter only after inference in the main run. Subject folds are constructed from filenames and keep subjects disjoint while correctly disclosing pooled tokenizer exposure.

Histories exclude current/future codes, reset per target, respect H=8 and recording boundaries, and keep the target code at its own time position. The bag arm preserves the history multiset and uses the same parameters/initialization. Its invariance and the ordered model's synthetic learnability pass tests. Window-equivalent occupancy smoothing, detached responsibility weighting, sampled frame weighting, the fixed calibration rule and alpha=0.5 cost are consistent with the instructions.

Transport uses the existing CPU float64 solver, true per-recording kernels and row masses. The new final ladder matches the historical ladder on the synthetic parity test. Scoring uses a population-global Hungarian mapping; the evaluation mapping is separate from the fitting mapping. Primary gate mean margins, the MoF tolerance and the all-three-positive requirement match the declared thresholds.

## Recommended disposition

Keep the architecture and frozen experimental settings. Fix findings 1–4 before investing in the full decision run; enforce the promised budget and repair the verification gaps as part of the same runner work. Re-run focused fault-path tests and a fresh smoke, then perform the historical parity check under the clarified verification policy. The existing smoke scores neither support nor refute the action-dynamics hypothesis.
