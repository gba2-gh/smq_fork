# Changes to the Milestone 1 agent instructions: v2 → v2.1

6 October 2026 · Applies to [COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md](COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md)

v2.1 makes two changes to v2 and leaves the scientific design unchanged. The question, the arms, the split, the training schedule, the transport settings and the Section 10 gate are all as in v2.

## Change 1: matched B0 initialization uses only the existing helper (Sections 2 and 3)

### What v2 said
Build the initial grouping on fitting recordings with "the existing deterministic initialization helper if it supports an explicit fitting population. Otherwise use full KMeans(C) … on frozen quantized vectors".

### What v2.1 says
- Use `script/quantized_input_comparison/fit.py::initial_grouping` with `codes_by_name`, `lengths_by_name` and `pca_by_name` restricted to the 395 fitting recordings. `pca_by_name` is each recording's saved `continuous` array.
- The fallback is removed. If the helper cannot run on the fitting subset, preflight fails.
- Then call `continuous.build_mu0` on the fitting recordings and run the existing continuous ASOT fit.
- **Preflight check:** the helper run on all recordings must reproduce the preparation's `grouping.npy`. Any mismatch is reported as a numerical discrepancy, with the number of differing codes. It is never forced to match.
- **Section 2:** this code-level grouping is declared the one exception to "no within-cell continuous feature may enter an action model". It only initializes B0, and no continuous feature reaches any cost, model input or training target.

### Why
- **The helper already accepts a population.** It takes per-recording dictionaries, so passing only the fitting recordings is the explicit fitting population. The "if it supports" condition was always met, and the fallback branch was dead code that an agent might still have chosen.
- **The fallback would have changed the method.** `initial_grouping` clusters frame-weighted per-code means of the continuous PCA features, through `categorical.build_grouping`. The fallback clustered lookup rows, which are different vectors. Matched B0 would then have used a different initializer from historical B0, so the matched baseline would differ from the historical one in two ways at once: population and method.
- **The helper read literally violated v2's own rule.** v2's continuous-feature rule would have forbidden the historical helper, because it averages PCA features per code. Declaring the exception keeps the instructions consistent with the method being matched.
- **The equality check makes the construction verifiable.** The stored `continuous` arrays are float32. The preparation computed centres from float64 PCA projections (`prepare.py:160`). A mismatch is therefore possible, and it has to be visible rather than assumed away.

## Change 2: diagnostics are split into Tier A (now) and Tier B (deferred) (Sections 8 and 9)

### What v2 said
All of the following were required in the implementation:
- two mappings, each applied across both populations;
- occupancy and conditional NLL, posterior-weighted NLL and mixture NLL;
- assignment agreement, and MI/NMI against recording, subject and action;
- nearest-neighbour preference on history representations, excluding overlapping supports;
- an order-perturbation decode and a state-averaged decode;
- adapter differences and transition summaries.

### What v2.1 says

| Tier | Contents | When |
| --- | --- | --- |
| A | Gate metrics: evaluation-population mapping, per-class recall, segment counts | This milestone |
| A | Secondary fitting-population scores, with their own mapping | This milestone |
| A | Fitting vs evaluation occupancy and conditional NLL at the frozen endpoint | This milestone |
| A | Initial/final assignment agreement on both populations | This milestone |
| A | Assignment–recording/subject/action MI and NMI, before and after learning | This milestone |
| A | Order-sensitivity decode on evaluation recordings | This milestone |
| B | Cross-applied mapping scores | Later analysis script, from saved artifacts |
| B | Posterior-weighted and mixture NLL | Later analysis script, from saved artifacts |
| B | Nearest-neighbour preference on history representations | Later analysis script, from saved artifacts |
| B | Adapter differences, transition summaries | Later analysis script, from saved artifacts |
| B | State-averaged decode | Later analysis script, from saved artifacts |

The run must also save what Tier B needs, without retraining or reloading the model:
- frozen-endpoint history representations `h` for every window of both populations (float16 allowed);
- per-window, per-action occupancy and conditional log-probabilities, stored separately rather than only as the combined cost.

Section 9 states that Tier B is not part of the handoff; the handoff only verifies that these artifacts exist.

### Why
- **Diagnostics had become the largest implementation cost.** The scope was reduced in v2, but Section 8 still required about a dozen analyses, several of them non-trivial: support-excluded nearest neighbours, mixture likelihoods and an extra decode path. With the 11 test families on top, that brought the estimate back above a day.
- **Tier A is what the gate needs.** It covers deciding the gate and interpreting a pass or fail:
  - the gate metrics themselves;
  - whether the model overfits the fitting recordings (NLL gap and secondary scores);
  - whether learning moved away from the B0 initialization (agreement);
  - whether order evidence is used at test time (order sensitivity, alongside the trained bag control).
- **Identity MI/NMI stays in Tier A on purpose.** It is cheap, a few lines on saved predictions, and it addresses the problem this project has identified twice: codes and the contextualizer encoding subject and recording identity.
- **Tier B explains outcomes rather than deciding them.** Saving its inputs means it can still be run later without retraining, but only if the gate result needs explaining.

## Not changed

These remain as in v2:
- LARa only, with the existing byte-identical `split.json` (395 fitting and 44 evaluation recordings, all 16 subjects on both sides);
- seeds 111, 222 and 1538574472;
- arms `b0_centre`, `m1_ordered`, `m1_bag` and `occupancy`;
- the history model and adapters;
- window-equivalent pseudocounts;
- cost calibration;
- the transport coefficients and kernel;
- 5,000 updates with refreshes every 500;
- the verification table;
- historical B0 parity as a separate check;
- the Section 10 gate and stop rule.
