# Pre-experiment corrections — version 1.3

All four comments are accepted. Version 1.2 is archived; the active plan, Stage A brief, and Word document now specify version 1.3. The architecture, primary hyperparameters, stopping tolerances, and 244 planned Stage A fits are unchanged. This revision changes definitions before any Stage A experiment is run.

## 1. Cosine scale: corrected throughout

A-cont and all Stage C cosine costs now use `D = 1 - z @ mu.T`, range [0,2] for unit vectors, matching the unary convention in the pinned ASOT `src/train.py`. A zero input vector has constant cost 1, not 1/2. T/E coefficients remain unchanged. The normalized weighted-mean prototype update remains the same because multiplication of the cosine objective by a positive constant does not change its minimizer with assignments fixed.

The original factor of one half did change the relative strength of the other terms. Removing it corrects that departure; it does not prove that categorical negative log-likelihood and cosine costs are now statistically calibrated, or that the upstream coefficients are optimal for either arm. The direction of the eventual metric change remains empirical. Required cost tests cover equal, orthogonal, opposite, and zero input vectors.

## 2. Permutations: restore correspondence, separate the two mechanisms

The earlier check mixed loss of code–label correspondence with loss of adjacency. The corrected diagnostic is not a chance reference. Its permutation convention is explicit:

```python
# perm[j] identifies the original window moved into shuffled slot j
inverse_perm = np.argsort(perm)
codes_perm = codes[perm]
R_restored = R_perm[inverse_perm]   # equivalently R_restored[perm] = R_perm
```

Only complete equal-length windows move; the partial terminal window stays fixed. Save both index arrays and original spans. Do not let stored original timestamps reconstruct chronological adjacency inside the shuffled model.

**Stage A:** reuse the observed tokenizer and initialization, refit/infer with shuffled adjacency, restore posterior rows, then argmax, broadcast to original frames, merge runs, Hungarian-map, and score. Store shuffled-space optimization traces separately from original-order evaluation predictions. The comparison measures the effect of replacing observed adjacency under this fitting procedure, including the response of the fitted emissions and numerical optimization.

**A0 equivariance:** without the structural term, the objective and pooled emission update are invariant to row permutation. With paired initialization and operations, restored results are identical in exact arithmetic. Floating-point reductions and stopping thresholds can prevent bitwise equality. Require synthetic fixed-schedule and stable stopping-rule tests, compare theta and restored probabilities at float64 tolerance 1e-8, and compare raw labels before Hungarian mapping; flag near-ties. This is an implementation test, not a new full-grid fitted arm.

**Stage C:** there is one useful refinement to the suggested inverse-prediction fix. If both the contextualizer and transport process shuffled order, restoring only final predictions still mixes disruption of context with disruption of transport adjacency. Instead, feed shuffled codes into the frozen contextualizer, inverse-permute its output embeddings, then refit/infer the existing diagnostic Stage C arm on those restored embeddings using the original chronological transport kernel. Predictions are already chronological; do not invert twice. This isolates the change in context presented to the downstream decoder while retaining transport adjacency. Initialization uses the original code grouping with restored embeddings.

Stage C's diagnostic remains pooled-only, seed 111, both datasets and normalization conditions. The erroneous mention of held-out recordings has been removed. No extra contextualizer or transport arm is introduced. Inverse indexing has small linear bookkeeping cost; the existing diagnostic fits remain the same in number. Positional/chunk context also changes under shuffling, so the diagnostic is sensitivity to ordered input context, not proof of a learned semantic grammar.

## 3. Kernel rounding: guarded parity, direct production width

Reproduced the reported **286 failures for L=5 through 2999 with b=4** in Python binary64, starting at 49, 98, 103, 107. The pinned reference computes `int(L*r)` and therefore can truncate below the intended integer when `r=b/L`.

Reference parity now requires `assert int(L*(b/L)) == b` for the chosen inputs, followed by direct kernel support/amplitude comparison. Examples L=16,32,64 with b=4 satisfy the guard. Failing lengths do not justify changing the experimental b, silently adjusting r, or declaring the production solver incorrect.

Production constructs support from integer b and amplitude from true L/b directly. A separate regression at L=49,b=4 checks that production retains the intended four-window half-width. The upstream source stays unchanged. Float64 dtype handling in the reference harness remains required independently of this rounding issue.

## 4. Batching: one consistent instruction

The conflicting mandatory-batching sentence is replaced. Establish per-recording correctness first; convolution or prefix sums are permitted. Batching is optional after equivalence tests. Each recording retains its true L, integer b, amplitude L/b, masses, mask, step size, and stopping state. A padded batch length cannot define its kernel. Tests compare unequal-length batched results against separate inference.

## Reporting expectation

The budgeted-algorithm table is explicitly the main table, with finite feasible capped results, paired differences, and visible status counts. Summaries include outer/final iteration counts, stationarity residuals, and late objective progress. The convergence-qualified paired subset is secondary and may be sparse or empty; report that coverage without presenting it as the full matrix. Outer caps and final-inference caps remain distinct, and a failed E sensitivity does not invalidate a completed T result.

No tolerance or iteration-budget change is made. A high cap rate is plausible but must be measured rather than asserted before execution.

## Verification and boundaries

Checked the pinned upstream cosine expression and kernel constructor, reproduced the rounding issue, and checked permutation restoration and A0 objective/update invariants with tiny synthetic arrays. These are arithmetic/definition checks, not experimental results or a claim that the Stage A solver is implemented. Updated document consistency and Word structure were checked; visual Word rendering remains unavailable because the rendering dependencies are absent. No full experiment was launched.
