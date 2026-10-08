# Agent instructions — Stage A: categorical temporal transport (version 1.1)

Implement Stage A of `THREE_STAGE_MOTION_PLAN.md` (version 1.1). Deliver concise, readable, self-contained code, meaningful tests, a README, and one execution script. Run validation and small synthetic checks only. **Do not launch the full experiment: the user will run it.** Do not implement Stages B or C now. These instructions and the companion plan are the specification; record any unresolved conflict instead of silently changing the method. Version 1.0 of these instructions is preserved as `STAGE_A_AGENT_INSTRUCTIONS_v1.0.md`.

## 1. Scope and file boundaries

- New code: `script/action_transport/`. New outputs: `results/action_transport/<run_id>/`. Preserve every existing experiment script and result, including old reports.
- Frozen released encoders, hard K-means codes (K=500 primary, K=C only for A1-KC), HuGaDB and LARa. Torch float64 solver on CUDA when available, CPU otherwise. CPU numerical threads capped at `min(8, logical_cpu_count)` before importing numerical libraries.
- **Not allowed:** encoder retraining, FSQ training, soft assignments, other vocabulary sizes, HSMM or duration models, learned HMM transitions, segment DP, duration caps, monotonic action-order priors, pose conversion, or the old lambda calibration.
- **Allowed:** the sticky categorical HMM specified in §4, which is a comparison arm.
- Keep labels outside fitting APIs. Ground-truth segments, labels, and action scores cannot initialize or select anything. C=10/8 is supplied as a known dataset action count, including background.
- Read applicable repository instructions first. Review `script/exp2round/q1q2/core.py`, its scoring dependencies, and existing cache manifests. Reuse verified primitives, not the old runners/summarizers. Avoid importing modules with execution side effects.

Suggested compact layout (combine files when clearer):

```text
script/action_transport/
  __init__.py
  config.py          # explicit dataclasses, settings T/E, planned-cell enumeration
  data.py            # cache provenance, folds, hard tokens, PCA windows, frame spans
  transport.py       # thin ASOT adapter, objective, batched torch solver, diagnostics
  categorical.py     # vocabulary-level init, emissions, alternating fitting (A0, A1, A1-KC)
  continuous.py      # A-cont prototypes and cosine cost
  hmm.py             # sticky categorical HMM (fixed transitions, Baum–Welch on theta)
  smoothing.py       # A0+filter mode filter
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

Represent a recording as `{name, subject, fps, frame_count, code_ids, starts, lengths}`. A-cont additionally needs `pca_windows` (float, L×64 or the feasible rank). Fitting objects must contain no labels. All intervals are half-open frame intervals. Assert exact coverage of every frame once and positive real lengths.

Use W=15 for HuGaDB and W=12 for LARa. Run per-recording latent z-score both off and on. The on condition uses the complete recording, population SD, and SD values below 1e-8 replaced by 1. It is an offline preprocessing condition.

If token artifacts need creating, uniformly sample at most 10,000 complete flattened windows, fit StandardScaler and randomized PCA-64 (maximum feasible rank if smaller), then full KMeans with K=500 (and separately K=C), k-means++, n_init=5, max_iter=300, tol=1e-4, algorithm='lloyd'. Fix sampling/PCA seed 111; vocabulary seeds are 111, 222, 1538574472. Fit only within the fitting population. Both vocabularies share the scaler, PCA, and sample. Save sample `(recording,start)` IDs. Process all transformations in bounded batches; incomplete terminal windows are excluded from fitting and zero-padded for assignment. Never materialize all flattened windows at once. PCA windows for A-cont are the same scaler+PCA outputs, stored per recording or regenerated with a verified hash.

Cache a tokenizer once per dataset, normalization condition, seed, vocabulary size, and fitting partition; share it across all arms. Do not infer equivalent identity from filenames alone.

## 3. Protocols, decision point, and planned cells

Implement both protocols as separate launches:

1. `pooled`: vocabulary, emissions, and prototypes are fit on all unlabeled recordings; inference runs on all recordings. This protocol runs first.
2. `subject_disjoint`: reuse verified Q1/Q2 four-fold subject memberships; otherwise construct folds by decreasing subject frame count, seed-111 tie breaking, and allocation to the least-loaded fold. Fit vocabulary, emissions, and prototypes on fitting recordings only. Infer held-out recordings using frozen parameters. No overlap of subjects or recordings.

The subject-disjoint protocol starts only after the user has read the pooled report and chosen to continue. The script accepts `-Protocol subject_disjoint` only together with `-AfterPooledRun <run_id>`. It verifies that the named pooled run exists and has a report, and records that ID in the manifest. There is no `all` option and no automatic numerical gate.

Fitted arms per matching configuration (dataset × normalization × seed × fitting population):

- `no_temporal` (A0): K=500 codes, beta=0, every other coefficient as in the fitting setting.
- `hmm`: K=500 codes, sticky fixed transitions (§4).
- `categorical_asot` (A1): K=500 codes, beta per setting.
- `categorical_asot_kc` (A1-KC): K=C codes, otherwise identical to A1.
- `continuous_asot` (A-cont): cosine cost on normalized PCA windows, otherwise identical to A1.

Derived arm with no fit: `no_temporal_filter` (A0+filter), computed from A0's saved raw states.

Coefficient settings, both frozen before scoring:

```text
T (primary, fitting + final decoding):   a=0.7  beta=0.3  lambda=0.05  epsilon=0.07
E (secondary, final decoding only):      a=0.4  beta=0.6  lambda=0.01  epsilon=0.04
```

Fit every transport arm under T. Then run the final re-solve under both T and E from the same fitted parameters. For A0, beta=0 under both settings. Save and score both prediction sets. The HMM has no setting and yields one prediction set. Thus each configuration yields 11 scored prediction sets.

Permutation sanity check: `permuted_asot`, i.e. A1 refit on within-recording permuted inputs. It runs pooled only, seed 111 only, for both datasets and both normalizations. With `default_rng(seed + 10000000)`, visit sorted recording names and permute complete equal-length tokens only (and the matching PCA windows, if stored alongside). Retain a terminal partial token in its original position. Save permutation indices. Preserve exact counts and real-frame-weighted counts within each recording. Labels stay at their original positions. Report this in a separate sanity table, not among the primary comparisons.

Counts to enumerate before execution, plus separately identified SMQ baseline-scoring jobs:

- pooled: 2 datasets × 2 normalizations × 3 seeds × 5 fitted arms = 60, plus 4 permutation sanity fits = **64 fits**
- subject_disjoint: × 4 folds = **240 fits**

Each fit is independent. Complete three paired seeds for a configuration and alternate datasets. Never choose normalization, seed, setting, arm, or protocol from action scores.

## 4. Initialization and models

### Vocabulary-level initialization (shared)

For each fitting population and vocabulary:
1. Compute frame-weighted code counts `c[u] = sum over fitting windows of length[t]*1[code[t]==u]`.
2. Fit C-cluster KMeans on the K prototypes in PCA space with `sample_weight=c`, using the vocabulary KMeans settings and the pipeline seed.
3. Codes with `c[u]==0` take the group of the nearest fitted center.

Save the group vector `g`, the centers, and the counts. If fewer than C codes have `c[u]>0`, the configuration is unavailable; do not change C. For K=C, the grouping reduces to one prototype per group; verify this rather than assume it. No label, ground-truth boundary, or time block enters initialization. `g` is invariant under within-recording permutation; assert this in a test.

Initial emissions, with total pseudocount `eta=fps` per state divided equally over the K codes:

```text
theta0[a,u] = (c[u]*1[g[u]==a] + eta/K) / (sum_u c[u]*1[g[u]==a] + eta)
```

A0, A1 and the HMM share `theta0`. A-cont initial prototypes use the same K=500 grouping:

```text
mu0[a] = normalize(sum over fitting windows with g[code[t]]==a of length[t]*z[t])
```

Here z is the L2-normalized PCA window. Zero vectors are flagged; they contribute nothing and take cost 1/2.

### Categorical transport (A0, A1, A1-KC)

```text
D[t,a_state] = -log(theta[a_state, code[t]]) / log(K)
```

Use the fixed `log(K)` scale for every recording and outer iteration. Do not refit a cost normalizer to current costs.

For each recording, let N be the real frame count, L the window count, `p[t]=length[t]/N`, and `q[a_state]=1/C`. T is a coupling with row sums p; R is conditional assignment with `R=T/p[:,None]`. Never confuse T and R when updating counts or comparing the reference implementation.

Declare and implement the objective exactly, with (a, beta, lambda, epsilon) taken from the active setting:

```text
F(T;theta) = a * sum(D*T)
           + beta/2 * sum_{t,s,a,b}(V[t,s]*T[t,a]*T[s,b]*1[a!=b])
           + lambda * sum_a(m[a]*log(m[a]/q[a]) - m[a] + q[a])
           - epsilon * H(T)
m = T.sum(axis=0)
H(T) = -sum(T*(log(T)-1))
```

For L>1, `b=min(L-1,max(1,floor(fps/W)))`; set `V[t,s]=L/b` when `0<abs(t-s)<=b`, else zero. For L=1, V=0. Save `b`, `b*W/fps`, and `b/L`. Implement multiplication by V with 1-D convolution over masked, padded batches; no dense L-by-L arrays. Rows of V are not additionally normalized at recording edges.

The fitting objective (setting T) is `sum_r N_r*F_r - a*eta/(K*log(K))*sum(log(theta))`. Its emission update is:

```text
count[a,u] = sum_{fitting r,t}(length[t]*R[t,a]*1[code[t]==u])
theta[a,u] = (count[a,u] + eta/K) / (count[a,:].sum() + eta)
```

Alternate solving every fitting recording and applying this one pooled update. Warm-start each recording's coupling from its previous iterate; initialize first solves with `p[:,None]*q[None,:]`. After the final theta update, re-solve under settings T and E, including for A0. Held-out inference never changes theta. Broadcast argmax R over real frames and merge adjacent equal states for action runs.

### Continuous transport (A-cont)

`D[t,a]=(1-z[t]·mu[a])/2`, with identical transport, settings, solver, and budgets. The prototype update is `mu[a]=normalize(sum_{fitting r,t} length[t]*R[t,a]*z[t])`. A zero update keeps the previous prototype and is flagged. The complete fitting objective is `sum_r N_r*F_r` with no prior term; verify that each prototype update does not increase it with T fixed (this holds because it is the normalized weighted mean under the cosine cost). Disclose that equal coefficients do not calibrate the cosine and categorical costs identically.

### Sticky categorical HMM

- **States:** C states, starting from `theta0`, with a fixed uniform initial distribution.
- **Fixed transitions:** `A[a,a]=s`, `A[a,b]=(1-s)/(C-1)`, where `s=1-1/(2b+1)` and b is the kernel half-width. Record s and the implied expected dwell in seconds.
- **Emissions:** each window's emission log-likelihood is `(length[t]/W)*log(theta[a,code[t]])`.
- **Fitting:** forward–backward in the log domain in float64. The M-step updates theta only: `theta[a,u]=(sum length[t]*gamma[t,a]*1[code[t]==u] + eta/K)/(sum length[t]*gamma[t,a] + eta)`.
- **Stopping:** at most 50 iterations; stop after three consecutive iterations with relative change in the complete log-posterior `<=1e-6`.
- **Decoding:** per-window argmax of gamma. Held-out recordings use frozen theta.
- **Status:** a decreasing log-posterior or a nonfinite value invalidates the cell.

### Mode filter (A0+filter)

Apply the filter separately to A0's final raw states under T and under E. For window t, compute frame-weighted votes `sum_{s: |t-s|<=b, same recording} length[s]*1[state[s]==a]`. Take the argmax. On ties, keep the original state at t if it is tied, else the smallest state ID. Run a single pass with no iteration. Save the filtered raw states as their own prediction set.

## 5. ASOT reference and numerical requirements

Use the upstream reference pinned at commit `0c4c86b1037eb4c3e262197aeee115b31e220c9d`. A local clone exists at `C:\Users\gzaz976\Documents\repo\Behaviour\action_seg_ot`. Copy the required source and license into `third_party/action_seg_ot/` unchanged, and record the commit and file hashes. Its video datasets/training framework are not required. If the source is unavailable, complete unaffected work and report the missing dependency; do not claim reference parity.

Implement two clearly separate numerical modes:

- **Parity test mode:** equal temporal masses, identical kernel/radius, costs, initialization, iteration count, and the reference step-size rule (`4/max(grad)` at the first iteration, fixed afterwards). Compare conditional assignments against the pinned implementation, on CPU in float64 with tiny inputs. Require a max absolute difference <=1e-8. Do not relax the tolerance to conceal an algorithm difference.
- **Production mode:** weighted temporal masses and the explicit objective above, using stable log-domain mirror descent and row projection in torch float64. Run on CUDA if available (record the device) and fall back to CPU. Batch recordings with padding and masks, keeping a per-recording step size and stopping state. The upstream scalar objective multiplies the linear term by one half while its gradient does not; use the independently checked objective above, and do not edit upstream code.

Backtracking: for each recording, start the trial step size at 1, halve it on an increase in that recording's F, and allow at most 30 backtracks. Use relative change with denominator `max(1,abs(previous))`.

- **Fitting (generalized EM):** at most 25 accepted steps per recording per outer iteration, warm-started. Cap outer iterations at 50. Stop after three consecutive outer iterations with relative complete-objective change `<=1e-6`.
- **Final inference (T and E):** at most 500 accepted steps. Stop when the row-centered gradient max norm is `<=1e-5` and relative objective change is `<=1e-7` for five accepted steps.

Save the stopping reason and actual residuals, not only a boolean.

Require finite nonnegative mass, row-marginal max error <=1e-10, normalized theta, unit prototypes (A-cont), and finite costs. Handle zeros with mathematically documented limits; do not hide nonfinite states via blanket `nan_to_num`.

Status values, defined once:
- `converged`: all stopping rules met.
- `capped`: an iteration cap was reached, with a finite, nonincreasing accepted-objective trace. Included in the primary aggregate, flagged, and counted.
- `invalid`: a nonfinite value, a backtracking failure, or an objective increase. Excluded and listed.
- `unavailable`, `interrupted`, `unstarted`: with a reason.

Do not change coefficients, C, initialization, or the representation after a failure.

## 6. Required validation before full-run handoff

Write targeted tests for the actual failure modes:

1. Identity/alignment: checkpoint and cache mismatch rejection, subject parsing, exact frame coverage, terminal token, same-checkpoint baseline lineage, K=500 and K=C vocabularies sharing scaler/PCA/sample, and PCA-window/code alignment. Use synthetic cache manifests where possible; avoid re-extracting encoders during unit tests.
2. Label separation: fitting APIs accept no action labels; outer held-out tokens never update preprocessing, theta, or prototypes. Verify group disjointness and artifact keys across folds. A subject-disjoint launch without a valid `-AfterPooledRun` is refused.
3. Initialization: hand-computed weighted prototype grouping on a toy vocabulary, zero-count code handling, invariance of `g` under permutation, K=C grouping, `theta0` normalization and pseudocount, and `mu0` normalization.
4. Math: finite differences of every objective component and the combined objective at positive T, under both settings, including feasible row-sum-preserving directions. With T fixed, the emission update and the A-cont prototype update each do not increase the complete objective. beta=0 retains the unary, marginal, and entropy coefficients of the active setting.
5. Transport: upstream equal-mass parity; constant and unequal frame masses; row sums; missing and repeated planted actions; unequal durations; a one-token recording; batched and padded results equal to per-recording results; CPU and CUDA agreement to 1e-10 when CUDA exists; stable finite objective traces; the generalized-EM schedule decreasing the complete objective across outer iterations. Verify empirical behavior rather than assert guaranteed unsupervised recovery.
6. HMM: forward–backward marginals and log-likelihood against brute-force enumeration on tiny sequences; a nondecreasing log-posterior over Baum–Welch iterations; transitions unchanged; terminal-window scaling.
7. Mode filter: hand-computed votes, tie rules, recording boundaries, frame weights, and idempotence not assumed.
8. Null: exact token counts, frame-weighted counts, retained terminal position, deterministic seed behavior, no cross-recording swaps, and null fitting on the transformed training inputs.
9. Evaluation: frame broadcasts and run merging; identity/permuted-label synthetic scoring; consistent Hungarian mapping; fold-specific state IDs; all five metrics reproducible from reloaded predictions for every prediction set, including T and E. Never treat arbitrary raw state IDs as aligned across folds.
10. Operations: atomic-write recovery; resume refuses changed configuration/dependencies; cell deduplication; simulated deadline interruption; partial seeds marked incomplete; saved predictions matching the final parameters.

A tiny real-data smoke check may verify loading/assignment/inference shapes and time one fit per arm, for the dry-run cost projection, without evaluating action scores. Keep it explicitly separate from synthetic unit tests and the full experimental matrix. Report what actually ran.

## 7. Evaluation and artifacts

The evaluator reuses the established Hungarian matching and MoF/Edit/F1@10/25/50 scorer. Evaluate the same-checkpoint SMQ predictions on exactly the same recordings. Reuse verified baseline scores only if scorer and population match. HiST-VQ/MASQ are matched comparisons only when a compatible export exists; mark them unavailable otherwise. Published paper-table numbers may appear only in a separately labelled unmatched-reference table, never as a matched result.

Pooled evaluation gets one dataset-level Hungarian mapping per prediction set. Subject-disjoint results use one mapping per held-out fold, then aggregate recording-level matched outputs with the scorer's original reduction. Report this additional oracle mapping flexibility. Keep recordings separate for Edit and F1; save mappings and both raw and mapped frame predictions.

Write at least:

```text
manifest.json                # config/source/dependency hashes, grid, device, runtime budget, AfterPooledRun
planned_cells.csv            # unique fit IDs, prediction-set IDs, intended comparisons
cells.csv                    # all five metrics per prediction set, status, runtime, iterations, residuals
not_run.csv                  # unavailable/invalid/interrupted/unstarted cells, with reasons
splits/                      # recording and subject memberships
artifacts/                   # tokenizers, grouping g, theta0/mu0, theta/mu, HMM transitions, permutations, T or R
predictions/                 # real-frame predictions plus raw state IDs, per setting
traces/                      # objective components, residuals, accepted steps, occupancy
logs/
REPORT.md
figures/
```

Capture kernel radius, eta, coefficients per setting, cost scale, HMM self-transition, true frame counts, sample IDs, occupied states, occupancy entropy, action-run counts and duration quantiles, device, peak memory where supported, and elapsed time. As a label-free diagnostic, also save the subject-state mutual information (subject IDs are metadata) to expose subject-aligned states. Generate reports only from artifacts, without refitting. Use explicit unique cell IDs, and never count duplicate log rows as extra seeds.

The report organizes comparisons by the plan's three questions:
1. **The claim:** A1 against SMQ and against A1-KC.
2. **The temporal mechanism:** A1 against A0+filter and against the HMM, with A0 shown.
3. **Discretization:** A1 against A-cont.

Setting T is primary and setting E is reported in full alongside it. The report must also include:
- paired seed differences, and means/population SD across exactly three complete seeds
- fold variation, reported separately
- coverage, convergence status counts, and not-run cells
- the permutation sanity table
- limitations and open questions

Figures separate pooled and downstream subject-disjoint results; show all five metrics and occupancy/durations. Choose timeline example recordings using a saved seed-111 selection before scoring. Disclose that:
- the frozen encoder remains transductive
- per-recording normalization and inference are offline
- local temporal consistency is not a learned action grammar
- the HMM is a fixed-transition comparison, not a duration model

Do not claim a publication contribution from a metric gain.

## 8. One execution script and final handoff

Create `run_stage_a.ps1` with `-ValidateOnly`, `-DryRun`, `-Protocol pooled|subject_disjoint`, `-AfterPooledRun`, `-Hours`, `-RunId`, `-Resume`, and optional `-Device auto|cpu|cuda` (default auto).
- Require an explicit hours budget for a full run, with at least 30 minutes reserved for reporting. Do not inherit the old experiment's deadline.
- Dry run lists planned cells and cache status, runs a short timing probe per arm, and projects cost without fitting the grid.
- Validation-only mode cannot launch the real grid.
- Locate/use the existing project environment (`C:\Users\gzaz976\AppData\Local\anaconda3\envs\smq\python.exe`) and print the resolved executable. Do not rely on a Microsoft Store Python alias.
- Set `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` to 8 before Python starts. Unset thread caps have caused segfaults on this machine.

Before each expensive job, use observed comparable timings to project completion; skip and record jobs projected to cross the compute deadline. On the first job with no estimate, state that uncertainty and use cooperative deadline checks inside fitting/assignment loops. Keep fits sequential. Persist completed work, handle interruption, and attempt reporting from saved artifacts within the reserved time. Resume must preserve cell identity and supply a new explicit runtime budget, without restarting already verified fits.

Document exact copyable commands, for example the following interface once implemented:

```powershell
.\script\action_transport\run_stage_a.ps1 -ValidateOnly
.\script\action_transport\run_stage_a.ps1 -DryRun -Protocol pooled
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_001
.\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_001 -Resume
# only after reading the pooled report and deciding to continue:
.\script\action_transport\run_stage_a.ps1 -Protocol subject_disjoint -AfterPooledRun stage_a_pooled_001 -Hours 12 -RunId stage_a_sd_001
```

The hour values here are example explicit user launch budgets, not permission for the agent to launch the run. Do not run those full-grid commands during implementation.

Finish by reporting:
- files created
- numerical, parity and HMM test results
- measured timing-probe costs and the projected pooled runtime
- remaining limitations
- exact user launch commands
- what the runner will produce

If a requirement cannot be satisfied, identify it precisely and keep affected cells unavailable. Deliver an implementation that is easy to inspect; do not add a new framework to hide unresolved assumptions.
