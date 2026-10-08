# Agent instructions — Stage A: categorical temporal transport (version 1.3)

Implement Stage A of `THREE_STAGE_MOTION_PLAN.md` (version 1.3). Deliver concise, readable, self-contained code, meaningful tests, a README, and one execution script. Run validation and small synthetic checks only. **Do not launch the full experiment: the user will run it.** Do not implement Stages B or C now. These instructions and the companion plan are the specification; record any unresolved conflict instead of silently changing the method. Versions 1.0, the reviewed 1.1, and 1.2 are preserved. See `REVIEW_RESPONSE_v1.2.md` for scope decisions and `REVIEW_RESPONSE_v1.3.md` for the pre-experiment numerical and permutation corrections.

## 1. Scope and file boundaries

- New code: `script/action_transport/`. New outputs: `results/action_transport/<run_id>/`. Preserve every existing experiment script and result, including old reports.
- Frozen released encoders, hard K-means codes (K=500 primary, K=C only for A1-KC), HuGaDB and LARa. Torch float64 solver defaults to CPU; CUDA is an explicit, verified launch option. CPU numerical threads capped at `min(8, logical_cpu_count)` before importing numerical libraries.
- **Not allowed:** encoder retraining, FSQ training, soft assignments, other vocabulary sizes, HSMM or duration models, learned HMM transitions, segment DP, duration caps, monotonic action-order priors, pose conversion, or the old lambda calibration.
- HMMs and continuous-input contextualizers are deferred. Do not implement them in Stage A.
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

The subject-disjoint protocol starts only after the user has read the pooled report and chosen to continue. The script accepts `-Protocol subject_disjoint` only together with `-AfterPooledRun <run_id>`. It verifies that the named pooled run exists and has a report, and records that ID in the manifest. There is no `all` option and no automatic numerical gate. This is a budget/scope decision: reuse the frozen settings and all primary arms if continued, otherwise mark transfer not run. A pooled-score inspection does not authorize tuning before claiming generalization.

Fitted arms per matching configuration (dataset × normalization × seed × fitting population):

- `no_temporal` (A0): K=500 codes, beta=0, every other coefficient as in the fitting setting.
- `categorical_asot` (A1): K=500 codes, beta per setting.
- `categorical_asot_kc` (A1-KC): K=C codes, otherwise identical to A1.
- `continuous_asot` (A-cont): cosine cost on normalized PCA windows, otherwise identical to A1.

Derived arm with no fit: `no_temporal_filter` (A0+filter), computed from A0's saved raw states.

Coefficient settings, both frozen before scoring:

```text
T (primary, fitting + final decoding):   a=0.7  beta=0.3  lambda=0.05  epsilon=0.07
E (secondary, final decoding only):      a=0.4  beta=0.6  lambda=0.01  epsilon=0.04
```

Fit every transport arm under T. Then run the final re-solve under both T and E from the same fitted parameters. For A0, beta=0 under both settings. Save and score both prediction sets. There are four fitted arms plus the derived filter; each configuration yields ten prediction sets. T is primary; E is a bundled sensitivity on fixed T-fitted parameters, not an E-trained model or an isolated marginal-penalty test. These settings come from the pinned src/train.py CLI defaults, not the constructor or run_bf.sh overrides. Compute T first; list E jobs not reached within the budget.

Permutation adjacency diagnostic: `permuted_asot`, i.e. A1 refit on within-recording permuted inputs, pooled only, seed 111 only, both datasets and normalizations. Reuse the exact observed tokenizer, grouping g and theta0. Use `default_rng(seed + 10000000)` over sorted recording names. Permute complete equal-length windows only, with any aligned stored PCA rows, retaining the terminal partial token in place. Keep origin metadata separate from shuffled slot indices: the temporal kernel acts on the latter. Preserve token counts and frame-weighted counts.

Save `perm` and `inverse_perm` with this mandatory convention:

```python
# perm[j] = original window index occupying shuffled position j
inverse_perm = np.argsort(perm)
codes_perm = codes[perm]
# Fit and infer in shuffled order; do not use original timestamps in V.
R_restored = R_perm[inverse_perm]  # equivalently: R_restored[perm] = R_perm
state_original = R_restored.argmax(axis=1)
# Broadcast with ORIGINAL starts/lengths, then merge runs, map labels and score.
```

Labels stay on the original timeline. Do not score shuffled-order predictions against them. Store both posterior orders with explicit metadata; traces refer to the shuffled optimization space. This is a paired adjacency diagnostic, including emission refitting effects and finite optimization, not a chance baseline. A0 must pass a synthetic exact-arithmetic equivariance test numerically (including its emission updates): compare theta and restored probabilities at float64 atol/rtol 1e-8 under paired initialization and a fixed update schedule. Compare raw state IDs before Hungarian mapping, with exact hard labels away from ties. Also exercise stopping rules on a stable synthetic case; do not promise bitwise equality near thresholds. This adds no full-grid fit.

Counts to enumerate before execution, plus separately identified SMQ baseline-scoring jobs:

- pooled: 2 datasets × 2 normalizations × 3 seeds × 4 fitted arms = 48, plus 4 permutation diagnostic fits = **52 fits**
- subject_disjoint: × 4 folds = **192 fits** (244 total planned fits)

Each fit is independent. Complete three paired seeds for a configuration and alternate datasets. Never choose normalization, seed, setting, arm, or protocol from action scores.

Include protocol version 1.3, cosine cost convention, direct integer kernel construction, and permutation restoration mode in artifact identity. Do not reuse v1.2 half-scale cosine fits or un-restored permutation metrics as current results.

## 4. Initialization and models

### Vocabulary-level initialization (shared)

For each fitting population and vocabulary:
1. Compute frame-weighted code counts `c[u] = sum over fitting windows of length[t]*1[code[t]==u]`.
2. Fit C-cluster KMeans on the occupied prototypes in PCA space with `sample_weight=c`, using the vocabulary KMeans settings and the pipeline seed. At K=C with distinct occupied prototypes, assign `g[u]=u` directly instead.
3. Codes with `c[u]==0` take the group of the nearest fitted center.

Save the group vector `g`, the centers, and the counts. If fewer than C distinct occupied prototypes exist, the configuration is unavailable; do not change C. At K=C, validate the direct one-prototype-per-state grouping. No label, ground-truth boundary, or time block enters initialization. `g` is invariant under within-recording permutation; assert this in a test.

Initial emissions, with total pseudocount `eta=fps` per state divided equally over the K codes:

```text
theta0[a,u] = (c[u]*1[g[u]==a] + eta/K) / (sum_u c[u]*1[g[u]==a] + eta)
```

A0 and A1 share `theta0`. Geometry is a declared initialization input; categorical fitting/inference thereafter consume only code IDs and masses. This is not a geometry-free initialization. A-cont initial prototypes use the same K=500 grouping:

```text
mu0[a] = normalize(sum over fitting windows with g[code[t]]==a of length[t]*z[t])
```

Here z is the L2-normalized PCA window. Zero vectors are flagged; they contribute nothing and take cost 1.

### Categorical transport (A0, A1, A1-KC)

```text
D[t,a_state] = -log(theta[a_state, code[t]]) / log(500)
```

Use the common fixed `log(500)` scale for both K=500 and K=C, every recording, and every outer iteration. Do not refit a cost normalizer to current costs. This cost is not bounded by 2. Save fitting cost quantiles and objective components. Do not switch the K=C denominator to log(C). Fixing cost scale and total smoothing avoids an unnecessary penalty-scale confound; per-code pseudocount and emission capacity still depend on alphabet size.

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

For L>1, `b=min(L-1,max(1,floor(fps/W)))`; set `V[t,s]=L/b` when `0<abs(t-s)<=b`, else zero. For L=1, V=0. Save `b`, `b*W/fps`, and `b/L`. Implement per-recording multiplication by V first, using 1-D convolution or prefix sums; batching is optional after equivalence tests. Construct support directly from integer b and amplitude from each recording's true L/b. In padded batches, mask padding and retain these per-recording values; never call the reference constructor with N_padded to define a recording's kernel. Do not recover b by int(L*(b/L)). No dense L-by-L arrays. Rows of V are not additionally normalized at recording edges.

The fitting objective (setting T) is `sum_r N_r*F_r - a*eta/(K*log(500))*sum(log(theta))`. Its emission update is:

```text
count[a,u] = sum_{fitting r,t}(length[t]*R[t,a]*1[code[t]==u])
theta[a,u] = (count[a,u] + eta/K) / (count[a,:].sum() + eta)
```

Alternate solving every fitting recording and applying this one pooled update. Warm-start each recording's coupling from its previous iterate; initialize first solves with `p[:,None]*q[None,:]`. After the final theta update, re-solve under settings T and E, including for A0. Held-out inference never changes theta. Broadcast argmax R over real frames and merge adjacent equal states for action runs.

### Continuous transport (A-cont)

`D[t,a]=1-z[t]·mu[a]`, in [0,2] for unit vectors, matching the pinned upstream unary convention. Do not divide by two. Zero input vectors have cost 1 for every state. Keep T/E coefficients, transport, solver, and budgets unchanged. The prototype update is `mu[a]=normalize(sum_{fitting r,t} length[t]*R[t,a]*z[t])`. A zero initial state mean makes the continuous arm unavailable, with a recorded reason; a zero later update keeps the previous unit prototype and is flagged. The complete fitting objective is `sum_r N_r*F_r` with no prior term; verify that each prototype update does not increase it with T fixed (this holds because it is the normalized weighted mean under the cosine cost). Disclose that equal coefficients do not calibrate the cosine and categorical costs identically.

### Deferred model family

Do not implement the review's sticky HMM. It would add a separate likelihood and implicit geometric-duration prior. A0+filter is the bounded temporal comparator here; conclusions must name that particular smoother, not all temporal models.

### Mode filter (A0+filter)

Apply the filter separately to A0's final raw states under T and under E. For window t, compute frame-weighted votes `sum_{s: |t-s|<=b, same recording} length[s]*1[state[s]==a]`. Take the argmax. On ties, keep the original state at t if it is tied, else the smallest state ID. Run a single pass with no iteration. Save the filtered raw states as their own prediction set.

## 5. ASOT reference and numerical requirements

Use the upstream reference pinned at commit `0c4c86b1037eb4c3e262197aeee115b31e220c9d`. A local clone exists at `C:\Users\gzaz976\Documents\repo\Behaviour\action_seg_ot`. Copy the required source and license into `third_party/action_seg_ot/` unchanged, and record the commit and file hashes. Its video datasets/training framework are not required. If the source is unavailable, complete unaffected work and report the missing dependency; do not claim reference parity.

Implement two clearly separate numerical modes:

- **Parity test mode:** equal temporal masses, identical kernel/radius, costs, initialization, iteration count, and the reference step-size rule (`4/max(grad)` at the first iteration, fixed afterwards). Compare conditional assignments against the pinned implementation, on CPU in float64 with tiny inputs. Ensure the upstream-created kernel/marginals are also float64 (scope and restore the default dtype in the test harness without editing upstream). Before calling upstream, assert `int(L*(b/L)) == b` for each parity case (e.g. b=4, L=16,32,64) and compare actual kernel support/amplitude. If the assertion fails, that case is unsuitable for reference parity; do not change b or nudge r. Production still supports those lengths through direct integer construction; include b=4,L=49 as a separate regression case. Require a max absolute difference <=1e-8. Do not relax the tolerance to conceal an algorithm difference.
- **Production mode:** weighted temporal masses and the explicit objective above, using stable log-domain mirror descent and row projection in torch float64. Default to CPU. CUDA requires an explicit launch option, successful numerical checks, and a label-blind timing probe. Record/freeze the selected device per run; no silent fallback. Implement per-recording correctness first. If batching, use bounded padding/masks, per-recording step/stopping, and each true recording length L in the kernel amplitude; padded length must not change its objective. The upstream scalar objective multiplies the linear term by one half while its gradient does not; use the independently checked objective above, and do not edit upstream code.

Backtracking: for each recording, start the trial step size at 1, halve it on an increase in that recording's F exceeding `1e-12*max(1,abs(F_old))`, and allow at most 30 backtracks. Use relative change with denominator `max(1,abs(previous))`.

- **Fitting (inexact alternating minimization):** at most 25 accepted steps per recording per outer iteration, warm-started. Cap outer iterations at 50. Stop after three consecutive outer iterations with relative complete-objective change `<=1e-6`.
- **Final inference (T and E):** at most 500 accepted steps. Stop when the row-centered gradient max norm is `<=1e-5` and relative objective change is `<=1e-7` for five accepted steps.

Save the stopping reason and actual residuals, not only a boolean. Define the row-centered residual as `max(abs(grad - grad.mean(axis=1,keepdims=True)))` over valid entries only. Report maximum and frame-weighted median residual across recordings. This is inexact block-coordinate optimization; the 25-step inner schedule is planned, not an optimization-failure status.

Require finite nonnegative mass, row-marginal max error <=1e-10, normalized theta, unit prototypes (A-cont), and finite costs. Handle zeros with mathematically documented limits; do not hide nonfinite states via blanket `nan_to_num`.

Status values, defined once:
- `converged`: the declared outer and final stopping rules met; no global-optimum claim.
- `capped`: outer or final-inference cap reached with feasible, finite iterates and a nonincreasing accepted-objective trace. Include in the explicitly labelled budgeted-algorithm table with status counts/residuals. Also report convergence-qualified comparisons on the same paired seeds, without promoting that selected subset to the main result. Deadline-interrupted jobs are incomplete, not capped.
- `invalid`: nonfinite values, infeasibility, backtracking failure, or an objective increase exceeding tolerance. Exclude invalid metrics and retain the cell/reason. Record fitting status and T/E inference status separately so E cannot invalidate T.
- `unavailable`, `interrupted`, `unstarted`: with a reason.

The main table is the budgeted-algorithm table, including feasible capped fits with per-arm cap counts and residuals. The convergence-qualified paired subset is secondary and may be small or empty; report its actual size or absence without claiming full-grid convergence. Summarize outer/final iteration counts, final stationarity residuals, and last-three-outer-iteration relative objective changes. Treat cap frequency as an observed quantity; do not loosen tolerances or enlarge iteration budgets based on action scores.

Do not change coefficients, C, initialization, or the representation after a failure.

## 6. Required validation before full-run handoff

Write targeted tests for the actual failure modes:

1. Identity/alignment: checkpoint and cache mismatch rejection, subject parsing, exact frame coverage, terminal token, same-checkpoint baseline lineage, K=500 and K=C vocabularies sharing scaler/PCA/sample, and PCA-window/code alignment. Use synthetic cache manifests where possible; avoid re-extracting encoders during unit tests.
2. Label separation: fitting APIs accept no action labels; outer held-out tokens never update preprocessing, theta, or prototypes. Verify group disjointness and artifact keys across folds. A subject-disjoint launch without a valid `-AfterPooledRun` is refused.
3. Initialization: hand-computed weighted prototype grouping on a toy vocabulary, zero-count code handling, invariance of `g` under permutation, K=C grouping, `theta0` normalization and pseudocount, and `mu0` normalization.
4. Math: finite differences of every objective component and the combined objective at positive T, under both settings, including feasible row-sum-preserving directions. With T fixed, the emission update and the A-cont prototype update each do not increase the complete objective. beta=0 retains the unary, marginal, and entropy coefficients of the active setting. Test cosine costs explicitly: equal unit vectors -> 0, orthogonal -> 1, opposite -> 2, zero input -> 1, with the same convention required for future Stage C.
5. Transport: upstream equal-mass parity with the int(L*r) guard; production width/amplitude regression at L=49,b=4; constant and unequal frame masses; row sums; missing and repeated planted actions; unequal durations; a one-token recording; batched and padded results equal to per-recording results; CPU and CUDA agreement on small fixed cases at atol=1e-8, rtol=1e-6 when explicitly enabled; verify objective/gradients and posterior probabilities, and flag argmax ties rather than demanding identical hard labels there; stable finite objective traces; the inexact alternating schedule decreasing the complete objective across outer iterations. Verify empirical behavior rather than assert guaranteed unsupervised recovery.
6. Mode filter: hand-computed votes, tie rules, recording boundaries, frame weights, and idempotence not assumed.
7. Adjacency diagnostic: exact token/frame-weighted counts; fixed terminal token; deterministic permutations; no cross-recording swaps; exact forward/inverse round trip; original-span frame broadcast and scoring after restoration; paired initialization; A0 objective/update/posterior equivariance at the stated tolerance, comparing raw states before any label mapping. Include a synthetic known nonidentity permutation where failing to restore predictions changes metrics, so the evaluator cannot silently score the wrong timeline.
8. Evaluation: frame broadcasts and run merging; identity/permuted-label synthetic scoring; consistent Hungarian mapping; fold-specific state IDs; all five metrics reproducible from reloaded predictions for every prediction set, including T and E. Never treat arbitrary raw state IDs as aligned across folds.
9. Operations: atomic-write recovery; resume refuses changed configuration/dependencies; cell deduplication; simulated deadline interruption; partial seeds marked incomplete; saved predictions matching the final parameters.

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
artifacts/                   # tokenizers, grouping g, theta0/mu0, theta/mu, perm/inverse_perm, T or R
predictions/                 # original-order frame predictions/raw IDs, per setting; explicit order tags
traces/                      # objective components, residuals, accepted steps, occupancy
logs/
REPORT.md
figures/
```

Capture kernel radius, eta, coefficients per setting, cost scale, true frame counts, sample IDs, occupied states, occupancy entropy, action-run counts and duration quantiles, device, peak memory where supported, and elapsed time. As an action-annotation-free diagnostic, save frame-weighted subject-by-state counts and subject-state mutual information. Subject IDs are labels for this diagnostic; do not interpret the association as a causal subject-identity effect. Generate reports only from artifacts, without refitting. Use explicit unique cell IDs, and never count duplicate log rows as extra seeds.

The report organizes comparisons by the plan's three questions:
1. **The claim:** A1 against SMQ and against A1-KC.
2. **Temporal assignment:** A1 against A0 and A0+filter; superiority is only over the specified smoother.
3. **Discretization:** A1 against A-cont.

Setting T is primary and setting E is reported in full alongside it. The report must also include:
- paired seed differences and means/population SD across exactly three completed seeds, with capped counts visible in the budgeted table; convergence-qualified paired subsets are secondary and list their actual seed counts
- fold variation, reported separately
- coverage, outer/final cap counts, residual summaries and late objective progress; secondary convergence-qualified subset size (including zero), and not-run cells
- the paired adjacency-diagnostic table
- limitations and open questions

Figures separate pooled and downstream subject-disjoint results; show all five metrics and occupancy/durations. Choose timeline example recordings using a saved seed-111 selection before scoring. Disclose that:
- the frozen encoder remains transductive
- per-recording normalization and inference are offline
- local temporal consistency is not a learned action grammar
- A-cont differs in emission geometry, K=C differs in alphabet/emission capacity despite sharing the cost denominator and total pseudocount, and SMQ is a matched-population system comparison, not an isolated K intervention

Do not claim a publication contribution from a metric gain.

## 8. One execution script and final handoff

Create `run_stage_a.ps1` with `-ValidateOnly`, `-DryRun`, `-Protocol pooled|subject_disjoint`, `-AfterPooledRun`, `-Hours`, `-RunId`, `-Resume`, `-TimingProbe`, and optional `-Device cpu|cuda` (default cpu).
- Require an explicit hours budget for a full run, with at least 30 minutes reserved for reporting. Do not inherit the old experiment's deadline.
- Dry run lists planned cells/cache status and projects from existing timing records without fitting. `-TimingProbe` is a separate explicit mode that runs tiny label-blind jobs; it never launches the grid or evaluates action scores.
- Validation-only mode cannot launch the real grid.
- Locate/use the existing project environment (`C:\Users\gzaz976\AppData\Local\anaconda3\envs\smq\python.exe`) and print the resolved executable. Do not rely on a Microsoft Store Python alias.
- Set numerical-library thread caps to min(8, logical CPU count), including `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS`, before Python starts.

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
- numerical, parity, filter, and continuous-update test results
- measured timing-probe costs if explicitly run, otherwise the projection's missing timing evidence
- remaining limitations
- exact user launch commands
- what the runner will produce

If a requirement cannot be satisfied, identify it precisely and keep affected cells unavailable. Deliver an implementation that is easy to inspect; do not add a new framework to hide unresolved assumptions.
