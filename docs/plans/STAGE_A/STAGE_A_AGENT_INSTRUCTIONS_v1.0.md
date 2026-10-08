# Agent instructions — Stage A: categorical temporal transport

Implement Stage A of `THREE_STAGE_MOTION_PLAN.md`. Deliver concise, readable, self-contained code, meaningful tests, a README, and one execution script. Run validation and small synthetic checks only. **Do not launch the full experiment: the user will run it.** Do not implement Stages B or C now. These instructions and the companion plan are the specification; record any unresolved conflict instead of silently changing the method.

## 1. Scope and file boundaries

- New code: `script/action_transport/`. New outputs: `results/action_transport/<run_id>/`. Preserve every existing experiment script and result, including old reports.
- Frozen released encoders, hard K-means K=500 codes, HuGaDB and LARa, CPU only, threads capped at `min(8, logical_cpu_count)` before importing numerical libraries.
- No encoder retraining, FSQ training, soft assignments, K sweep, HMM/HSMM, segment DP, duration cap, monotonic action-order prior, pose conversion, or old lambda calibration.
- Keep labels outside fitting APIs. Ground-truth segments, labels, and action scores cannot initialize or select anything. C=10/8 is supplied as a known dataset action count, including background.
- Read applicable repository instructions first. Review `script/exp2round/q1q2/core.py`, its scoring dependencies, and existing cache manifests. Reuse verified primitives, not the old runners/summarizers. Avoid importing modules with execution side effects.

Suggested compact layout (combine files when clearer):

```text
script/action_transport/
  __init__.py
  config.py          # explicit dataclasses and planned-cell enumeration
  data.py            # cache provenance, folds, hard tokens, frame spans
  transport.py       # thin ASOT adapter, objective, solver, diagnostics
  categorical.py     # initialization, emissions, alternating fitting
  run.py             # stages of execution, resume, budget, atomic writes
  evaluate.py        # only boundary allowed to receive action labels
  tests.py
  run_stage_a.ps1    # the one user-facing execution script
  README.md
```

Use small functions, named shapes/units, type hints, and docstrings explaining mathematical choices. Avoid generic plugin systems, nested configuration factories, large classes, and copies of the whole upstream training framework.

## 2. Provenance and reusable token data

Verify full checkpoint SHA-256:

```text
hugadb: 22dfd5639c718177c12bc48debf67371c4bf7ca64af2d7de4e83ebacae55d7d2
lara:   a45392ad30e27bda400adcab541aba087e4ae56b68837547cc65ca77ff5bfa65
```

Paths are `models/pretrained/hugadb.model` and `models/pretrained/lara.model`. Verify recording names, frame counts, latent layout, code/preprocessing identity, and actual class/subject inventory. Expected HuGaDB: 18 subjects, 10 classes, IMU, 60 Hz. Expected LARa: 16 subjects, 8 classes, optical motion capture, 50 Hz. Report discrepancies; do not silently filter recordings.

Reuse latent and token artifacts only if their dependencies match. In particular, never reuse the old epoch30 HuGaDB cache. Predictions for the SMQ reference must come from the same checkpoint. If a prior quantization/alignment/D7 verification has the identical full dependency chain, reference that artifact rather than rerunning it. Otherwise verify quantization equality, alignment, and D7 using the established procedure (44.85/41.66 MoF, tolerance 1.5 points). Stop the affected dataset on failure, retaining the other. D7 is an oracle-boundary provenance check, not the SMQ segmentation baseline.

Represent a recording as `{name, subject, fps, frame_count, code_ids, starts, lengths}`; fitting objects must contain no labels. All intervals are half-open frame intervals. Assert exact coverage of every frame once and positive real lengths.

Use W=15 for HuGaDB and W=12 for LARa. Run per-recording latent z-score both off and on. The on condition uses the complete recording, population SD, and SD values below 1e-8 replaced by 1. It is an offline preprocessing condition.

If token artifacts need creating, uniformly sample at most 10,000 complete flattened windows, fit StandardScaler and randomized PCA-64 (maximum feasible rank if smaller), then full KMeans with K=500, k-means++, n_init=5, max_iter=300, tol=1e-4, algorithm='lloyd'. Fix sampling/PCA seed 111; vocabulary seeds are 111, 222, 1538574472. Fit only within the fitting population. Save sample `(recording,start)` IDs. Process all transformations in bounded batches; incomplete terminal windows are excluded from fitting and zero-padded for assignment. Never materialize all flattened windows at once.

Cache a tokenizer once per dataset, normalization condition, seed, and fitting partition; share it across all methods. Do not infer equivalent identity from filenames alone.

## 3. Protocol and planned cells

Implement both protocols, with pooled first:

1. `pooled`: vocabulary and emissions fit on all unlabeled recordings; inference on all recordings.
2. `subject_disjoint`: reuse verified Q1/Q2 four-fold subject memberships; otherwise construct folds by decreasing subject frame count, seed-111 tie breaking, and allocation to the least-loaded fold. Fit vocabulary and emissions on fitting recordings only. Infer held-out recordings using frozen emissions. No overlap of subjects or recordings.

Use three methods per matching configuration:

- `no_temporal`: beta=0; every other coefficient unchanged.
- `categorical_asot`: beta=0.3.
- `permuted_asot`: beta=0.3; refit on shuffled sequences.

The first two share the identical initial theta. For the null, use the same initialization procedure on shuffled inputs. With `default_rng(seed + 10000000)`, visit sorted recording names and permute complete equal-length tokens only; retain a terminal partial token in its original position. Save permutation indices. Preserve exact counts and real-frame-weighted counts within each recording. Labels stay at their original positions. In subject-disjoint folds, apply the transformation independently within both fitting and held-out recordings; no token crosses a recording boundary.

There are 36 pooled method fits and 144 subject-disjoint method fits (180 total). Enumerate these before execution, plus separately identified SMQ baseline-scoring jobs. Each method fits independently. Complete three paired seeds for a configuration and alternate datasets. Never choose normalization, seed, or protocol from action scores.

## 4. Initialization and categorical model

Build complete, non-overlapping one-second blocks starting at recording frame zero; do not use action boundaries. Accumulate code counts by actual frame overlap, normalize to sum one, then square root. Select at most 10,000 blocks uniformly without replacement using seed 111. Fit C-cluster KMeans with the same settings as above and the pipeline seed. Derive initial emission counts from the selected blocks' assignments and their unsquared raw frame counts. Save block IDs, centers, and theta. Fewer than C complete blocks makes the configuration unavailable; do not change C.

For each state, add total pseudocount `eta=fps`, equally divided across K=500 codes. Thus theta rows are strictly positive and sum to one. Let `a=0.7` and define:

```text
D[t,a_state] = -log(theta[a_state, code[t]]) / log(K)
```

Use the fixed `log(K)` scale for every recording and outer iteration. Do not refit a cost normalizer to current costs.

For each recording, let N be the real frame count, L the window count, `p[t]=length[t]/N`, and `q[a_state]=1/C`. T is a coupling with row sums p; R is conditional assignment with `R=T/p[:,None]`. Never confuse T and R when updating counts or comparing the reference implementation.

Declare and implement the objective exactly:

```text
F(T;theta) = 0.7 * sum(D*T)
           + beta/2 * sum_{t,s,a,b}(V[t,s]*T[t,a]*T[s,b]*1[a!=b])
           + 0.05 * sum_a(m[a]*log(m[a]/q[a]) - m[a] + q[a])
           - 0.07 * H(T)
m = T.sum(axis=0)
H(T) = -sum(T*(log(T)-1))
```

For L>1, `b=min(L-1,max(1,floor(fps/W)))`; set `V[t,s]=L/b` when `0<abs(t-s)<=b`, else zero. For L=1, V=0. Save `b`, `b*W/fps`, and `b/L`. Implement multiplication by V with convolution or prefix sums; no dense L-by-L arrays. Rows of V are not additionally normalized at recording edges.

The fitting objective is `sum_r N_r*F_r - 0.7*eta/(K*log(K))*sum(log(theta))`. Its emission update is:

```text
count[a,u] = sum_{fitting r,t}(length[t]*R[t,a]*1[code[t]==u])
theta[a,u] = (count[a,u] + eta/K) / (count[a,:].sum() + eta)
```

Alternate solving every fitting recording and applying this one pooled update. Warm-start each recording's coupling from its previous iterate; initialize first solves with `p[:,None]*q[None,:]`. Re-solve after the final theta update, including for the no-temporal method. Held-out inference never changes theta. Broadcast argmax R over real frames and merge adjacent equal states for action runs.

## 5. ASOT reference and numerical requirements

Use https://github.com/mingu6/action_seg_ot as reference. If absent, obtain the required public source and license under `third_party/action_seg_ot/`, pin an exact commit, and record hashes. Keep upstream files unchanged. Its video datasets/training framework are not required. If retrieval is blocked, complete unaffected work and report the missing dependency; do not claim reference parity.

Implement two clearly separate numerical modes:

- **Parity test mode:** equal temporal masses, identical kernel/radius, costs, initialization, iteration count, and fixed reference step size. Compare conditional assignments against the pinned implementation. Start with float64 tiny inputs; require max absolute difference <=1e-8, or document precision-based tolerance if the reference forces a lower dtype. Do not relax tolerance to conceal an algorithm difference.
- **Production mode:** weighted temporal masses and the explicit objective above, with stable log-domain mirror descent and row projection. Preserve the solver lineage but document every adaptation. In particular, inspect the upstream scalar objective's half factor on linear-plus-quadratic terms and verify against the gradient. Use an independently checked objective for production convergence; do not edit upstream to conceal a discrepancy.

Start trial step size at 1 and halve on an increase in the declared objective, allowing 30 backtracks. Use float64 for the initial implementation and all correctness tests. Cap inner solves at 500 accepted iterations. Require row-centered gradient max norm <=1e-5 and relative objective change <=1e-7 for five iterations. Compute relative change using denominator `max(1,abs(previous_objective))`. Cap outer iterations at 50 and require relative complete-objective change <=1e-6 for three iterations; final inference must also satisfy the inner criterion. Save the stopping reason and actual residual, not only a boolean.

Require finite nonnegative mass, row-marginal max error <=1e-10, normalized theta, and finite costs. Handle zeros with mathematically documented limits; do not hide nonfinite states via blanket `nan_to_num`. Backtracking failure or nonfinite outputs invalidate the affected cell. Capped optimization is `nonconverged`; retain its diagnostic predictions but exclude it from the primary complete-cell aggregate. Never present it as a converged optimum. Do not change coefficients, C, initialization, or the representation after a failure.

## 6. Required validation before full-run handoff

Write targeted tests for the actual failure modes:

1. Identity/alignment: checkpoint and cache mismatch rejection, subject parsing, exact frame coverage, terminal token, and same-checkpoint baseline lineage. Use synthetic cache manifests where possible; avoid re-extracting encoders during unit tests.
2. Label separation: fitting APIs accept no action labels; outer held-out tokens never update preprocessing or theta. Verify group disjointness and artifact keys across folds.
3. Histograms: hand-computed one-second overlaps, mass conservation, no GT boundary access, pseudocount update, and square-root only for initialization KMeans.
4. Math: finite differences of every objective component and the combined objective at positive T, including feasible row-sum-preserving directions; emission update decreases the complete objective with T fixed; beta=0 retains unary/marginal/entropy coefficients.
5. Transport: upstream equal-mass parity, constant/unequal frame masses, row sums, missing/repeated planted actions, unequal durations, one-token recording, and stable finite objective traces. Verify empirical behavior rather than assert guaranteed unsupervised recovery.
6. Null: exact token counts, frame-weighted counts, retained terminal position, deterministic seed behavior, no cross-recording swaps, and null fitting on the transformed training inputs.
7. Evaluation: frame broadcasts and run merging; identity/permuted-label synthetic scoring; consistent Hungarian mapping; fold-specific state IDs; all five metrics reproducible from reloaded predictions. Never treat arbitrary raw state IDs as aligned across folds.
8. Operations: atomic-write recovery, resume refuses changed configuration/dependencies, cell deduplication, simulated deadline interruption, partial seeds marked incomplete, and saved predictions matching the final theta.

A tiny real-data smoke check may verify loading/assignment/inference shapes without evaluating action scores. Keep it explicitly separate from synthetic unit tests and the full experimental matrix. Report what actually ran.

## 7. Evaluation and artifacts

The evaluator reuses the established Hungarian matching and MoF/Edit/F1@10/25/50 scorer. Evaluate the same-checkpoint SMQ predictions on exactly the same recordings. Reuse verified baseline scores only if scorer and population match. HistVQ is optional when a compatible export exists; mark unavailable otherwise. Do not compare a paper-table number as a matched result.

Pooled evaluation gets one dataset-level Hungarian mapping. Subject-disjoint results use one mapping per held-out fold, then aggregate recording-level matched outputs with the scorer's original reduction. Report this additional oracle mapping flexibility. Keep recordings separate for Edit and F1; save mappings and both raw and mapped frame predictions.

Write at least:

```text
manifest.json                # config/source/dependency hashes, grid, runtime budget
planned_cells.csv            # unique cell IDs and intended comparisons
cells.csv                    # all five metrics, status, runtime, iterations, residuals
not_run.csv                  # unavailable/failed/nonconverged/interrupted/unstarted cells
splits/                      # recording and subject memberships
artifacts/                   # tokenizer, init, theta, permutations, T or R
predictions/                 # real-frame predictions plus raw state IDs
traces/                      # objective components, residuals, accepted steps, occupancy
logs/
REPORT.md
figures/
```

Capture kernel radius, eta, coefficients, cost scale, true frame counts, sample IDs, occupied states, occupancy entropy, action-run counts and duration quantiles, peak memory where supported, and elapsed time. Define status values once and include a reason. Generate reports only from artifacts, without refitting. Use explicit unique cell IDs and never count duplicate log rows as extra seeds.

The report includes all normalization/protocol/method conditions, paired seed differences, means/population SD across exactly three complete seeds, fold variation separately, coverage, convergence, not-run cells, limitations, and open questions. Figures separate pooled and downstream subject-disjoint results; show all five metrics and occupancy/durations. Choose timeline example recordings using a saved seed-111 selection before scoring. Disclose that the frozen encoder remains transductive, that per-recording normalization and inference are offline, and that local temporal consistency is not a learned action grammar. Do not claim a publication contribution from a metric gain.

## 8. One execution script and final handoff

Create `run_stage_a.ps1` with `-ValidateOnly`, `-DryRun`, `-Protocol pooled|subject_disjoint|all`, `-Hours`, `-RunId`, and `-Resume`. Require an explicit hours budget for a full run, with at least 30 minutes reserved for reporting. Do not inherit the old experiment's deadline. Dry run lists planned cells and cache status without fitting. Validation-only mode cannot launch the real grid. Locate/use the existing project environment and print the resolved executable; do not rely on a Microsoft Store Python alias.

Before each expensive job, use observed comparable timings to project completion; skip and record jobs projected to cross the compute deadline. On the first job with no estimate, state that uncertainty and use cooperative deadline checks inside fitting/assignment loops. Keep fits sequential. Persist completed work, handle interruption, and attempt reporting from saved artifacts within the reserved time. Resume must preserve cell identity and supply a new explicit runtime budget, without restarting already verified fits.

Document exact copyable commands, for example the following interface once implemented:

```powershell
.\script\action_transport\run_stage_a.ps1 -ValidateOnly
.\script\action_transport\run_stage_a.ps1 -DryRun -Protocol all
.\script\action_transport\run_stage_a.ps1 -Protocol all -Hours 8 -RunId stage_a_001
.\script\action_transport\run_stage_a.ps1 -Protocol all -Hours 8 -RunId stage_a_001 -Resume
```

Eight hours here is an example explicit user launch budget, not permission for the agent to launch the run. Do not run those full-grid commands during implementation.

Finish by reporting files created, numerical/parity test results, remaining limitations, exact user launch commands, and what the runner will produce. If a requirement cannot be satisfied, identify it precisely and keep affected cells unavailable. Deliver an implementation that is easy to inspect; do not add a new framework to hide unresolved assumptions.
