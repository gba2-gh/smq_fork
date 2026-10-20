# Stage C analysis: `stage_c_pooled_001`

26 September 2026. This analysis uses only saved artifacts: this run's `cells.csv`, `stage_a_pooled_002/cells.csv`, and `stage_b_003/results.json`. Nothing was refit. The full generated tables are in [REPORT.md](REPORT.md).

## Run integrity

| Check | Result |
|---|---|
| Coverage | 80/80 prediction sets; 0 not run; 0 invalid |
| Contextualizer identity | `stage_b_003`, hash `0689bb38dfdd031b`; all 12 cells passed the identity checks |
| Paired initialization | The grouping `g` is identical to Stage A's A-cont grouping in all cells |
| Order-diagnostic permutation | Identical to Stage A's `permuted_asot` permutation. Checked offline, because the runtime check was not called; see "Errors" |
| Evaluator | Re-scoring a saved Stage A prediction set reproduced its `cells.csv` row exactly |
| Earlier aborted launch (18:32) | Crashed before writing anything, because `stage_b_003` did not exist yet. The real run started fresh at 19:25 and finished at 22:32. **No effect on results.** |

## Main results (F1@50, setting T, pooled, mean ± SD over 3 seeds)

| Dataset / norm | SMQ | A0+filter | A1 (codes + ASOT) | A-cont (PCA + ASOT) | Context-only | Context-only + filter | **Contextual ASOT** |
|---|---|---|---|---|---|---|---|
| HuGaDB / raw | 24.30 | 6.75 | 3.73 | 4.85 | 8.46 ± 0.46 | 29.69 ± 1.41 | **37.00 ± 2.31** |
| HuGaDB / norm | 24.30 | 5.67 | 6.74 | 4.93 | 6.87 ± 0.43 | 25.10 ± 2.33 | **29.16 ± 5.45** |
| LARa / raw | 16.39 | 13.05 | 9.86 | 17.21 | 4.96 ± 0.85 | 7.56 ± 1.59 | **7.77 ± 1.18** |
| LARa / norm | 16.39 | 13.84 | 7.78 | 17.05 | 5.20 ± 0.22 | 8.46 ± 0.62 | **11.34 ± 0.34** |

Contextual ASOT's other metrics under T:

| Dataset / norm | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| HuGaDB / raw | 41.4 | 37.2 | 43.1 | 40.6 | 37.0 |
| HuGaDB / norm | 37.1 | 38.8 | 40.7 | 35.8 | 29.2 |
| LARa / raw | 21.1 | 23.9 | 21.9 | 15.8 | 7.8 |
| LARa / norm | 25.6 | 28.4 | 26.4 | 20.6 | 11.3 |
| SMQ, HuGaDB | 42.0 | 36.2 | -- | -- | 24.3 |
| SMQ, LARa | 37.4 | 39.4 | -- | -- | 16.4 |

Under setting E the direction is the same:
- **HuGaDB:** contextual ASOT scores 37.4 / 29.7, against A-cont's 22.3 / 24.3.
- **LARa:** contextual ASOT scores 7.5 / 11.4, against A-cont's 16.6 / 17.2.

## Precommitted decision (plan §4, v1.7)

| Comparison (three-seed mean F1@50, T) | HuGaDB raw | HuGaDB norm | LARa raw | LARa norm |
|---|---|---|---|---|
| Contextual ASOT beats A-cont | yes (3/3 seeds) | yes (3/3) | **no** (0/3) | **no** (0/3) |
| Contextual ASOT beats SMQ | yes (3/3) | yes (3/3) | **no** (0/3) | **no** (0/3) |
| Context-only median-run ratio in [0.5, 2] | 0.07 (no) | 0.07 (no) | 0.33 (no) | 0.22 (no) |

**Verdict: neither the primary bar nor the partial-result clause is met.**
- **Primary bar:** 4 of the 8 required comparisons hold, all on HuGaDB; all 4 on LARa fail.
- **Partial-result clause:** the partial result ("context alone removes fragmentation") is not what happened. Context alone stays fragmented; context combined with transport is what removes the fragmentation.

## Fragmentation: the same transport on different representations (T)

| Dataset / norm | A1: pred/GT runs, median run | A-cont | Context-only | **Contextual ASOT** | GT median run |
|---|---|---|---|---|---|
| HuGaDB / raw | 7.74, 0.25 s | 5.76, 0.25 s | 6.12, 0.25 s | **0.79, 3.92 s** | 3.47 s |
| HuGaDB / norm | 8.34, 0.25 s | 8.46, 0.25 s | 6.42, 0.25 s | **0.90, 4.00 s** | 3.47 s |
| LARa / raw | 2.04, 0.72 s | 0.70, 2.96 s | 2.08, 0.72 s | **0.45, 5.68 s** | 2.20 s |
| LARa / norm | 3.47, 0.48 s | 1.02, 2.64 s | 3.04, 0.48 s | **0.59, 4.80 s** | 2.20 s |

On HuGaDB, the same solver, coefficients, and initialization go from one-window runs (0.25 s) to runs of about GT length (about 4 s) once the emissions are contextual. On LARa, contextual ASOT **under**-segments: it produces half the GT number of runs, each about twice as long as GT runs.

## Incremental effects (paired by seed, F1@50, T)

| Effect | HuGaDB raw | HuGaDB norm | LARa raw | LARa norm |
|---|---|---|---|---|
| Transport on context (contextual ASOT − context-only) | +28.5 (3/3) | +22.3 (3/3) | +2.8 (3/3) | +6.1 (3/3) |
| Transport vs the fixed filter on context | +7.3 (3/3) | +4.1 (2/3) | +0.2 (2/3) | +2.9 (3/3) |
| Context vs PCA under the same ASOT (contextual ASOT − A-cont, means) | +32.2 | +24.2 | −9.4 | −5.7 |

## Contextual-order diagnostic (seed 111)

In this diagnostic, the frozen contextualizer sees the same codes in shuffled order, and the embeddings are restored to their original order before refitting.

| Dataset / norm | Original order, F1@50 T | Shuffled context, F1@50 T | Change |
|---|---|---|---|
| HuGaDB / raw | 33.78 | 16.57 | −17.2 |
| HuGaDB / norm | 24.61 | 15.54 | −9.1 |
| LARa / raw | 8.90 | 9.49 | +0.6 |
| LARa / norm | 11.75 | 13.91 | +2.2 |

On HuGaDB, about half the result depends on the order the contextualizer sees. On LARa, order contributes nothing. Even with shuffled context, HuGaDB (15.5–16.6) remains far above A1 and A-cont (3.7–6.7). What produces that remaining gain is not established. Shuffling also changes the contextualizer's positional input, so the shuffled condition is out of distribution. Attributing the residual to "pooling" requires the temporally pooled PCA control, which has not been run.

## Errors and caveats that affect interpretation

1. **Contextual ASOT is less converged than the Stage A arms it is compared with.**
   - **Outer loop:** every transport fit, in both stages, hit the 50-iteration outer cap. Contextual ASOT's last relative change (median 1.7e-3) is about 3× larger than A-cont's (6.2e-4).
   - **Final solves:** 3217/4818 recordings were capped under T (A-cont: 1772; A1: 214), and 4530/4818 under E. The weighted-median residual reached 2.2e-2 under E.
   - **Status:** these are valid `capped` results (finite, nonincreasing objectives), and the budgets were identical by design. But the Stage C numbers describe a budgeted algorithm further from stationarity than Stage A's, and they could move under a larger budget, in either direction.
2. **The runtime permutation cross-check was not called.** `contextual.check_permutation_matches_stage_a` exists but `run_order_diagnostic` never calls it. I verified the check offline and the permutations are identical, so this has **no effect on results**. It should still be wired in.
3. **Stage A's outer-trace bug (v1.6.1 fix) does not affect these comparisons.** Stage A ran before the fix. The affected fits are those that stopped early. All A1 and A-cont fits, and 11/12 A0 fits, ran to the cap, so their updates are identical to the fixed code. Only A1-KC (8 fits converged early) could differ, and A1-KC is not in the Stage C bar.
4. **The LARa contextualizer is weak.**
   - **Masked accuracy:** all 6 LARa `stage_b_003` cells are below the copy baseline (margins −0.015 to −0.049), and all hit the 5000-step cap.
   - **Order:** the order diagnostic shows no order benefit on LARa.
   - **What this does and does not show:** the LARa failure is consistent with a contextualizer that did not learn useful context there. It is not evidence about transport on good context.
5. **The comparisons are not fully matched.** Results are pooled and transductive, and subject-disjoint evaluation was not run. The 128-d contextual cosine cost is not calibrated like the 64-d PCA cost. The HuGaDB-normalized seed SD is large (±5.45).

## What the evidence supports

- **The precommitted claim fails.** Contextual transport does not beat A-cont and SMQ on both datasets. Per §4, the discrete-code + transport direction is not established at K=500.
- **A real, replicated HuGaDB effect.** On HuGaDB, contextual ASOT beats SMQ by +12.7 (raw) and +4.9 (norm) F1@50 in 3/3 seeds each. Against A-cont the margins are +32.2 and +24.2. Those comparisons use the same transport, coefficients, and initialization. The F1@50 gain comes from segment structure (Edit 37–39 vs SMQ 36), not frame accuracy (MoF 41.4 vs SMQ 42.0).
- **The Stage A mechanism question is answered for HuGaDB.** Stage A's fragmentation came from per-window emissions, not from the transport. Swapping only the representation turns 0.25 s runs into GT-length runs. On context, transport also beats the fixed smoother, which it did not do on codes in Stage A.
- **The HuGaDB effect is not purely a learned-sequence effect.** Order accounts for about half of it; the other half survives shuffling. A simple temporal-pooling baseline (for example, a moving average of PCA windows fed to A-cont) was not tested and could explain part of the gain. That baseline is the critical next control before attributing the effect to the masked-code model.
- **For publication, this is a one-dataset result.** On HuGaDB, 37.0 F1@50 sits between the published HiST-VQ (28.3) and MASQ (39.5) numbers, but those references are unmatched. MoF is below all published systems.

## Why LARa fails: an oracle-controlled diagnosis

26 September 2026, post-hoc. This uses ground-truth action labels only to build oracle class prototypes and to read out recall/confusion -- descriptive, not part of any fitted or selected setting. Full numbers in `lara_diagnosis.json`.

**Method:** for each (dataset, normalize), build the *oracle* class centroid for each representation (PCA-64 windows, and the frozen contextual embeddings) as the frame-weighted mean unit vector of all windows whose majority ground-truth label is that class. This is a label-derived reference prototype, not a performance ceiling. A class centroid is not the accuracy-maximizing prototype; multiple prototypes per class, other metrics or nonlinear readouts could do better. Then (a) classify by nearest centroid (no transport) and (b) feed those oracle centroids into the *identical* Stage C transport (T, same solver/coefficients) as fixed prototypes, and score with the identity mapping (no Hungarian needed, since prototypes already carry the class label).

### Oracle-prototype intervention (label-derived centroids, fixed transport)

| Dataset / norm | Oracle centroid accuracy: PCA | Oracle centroid accuracy: context | Oracle-prototype transport F1@50: PCA | Oracle-prototype transport F1@50: context |
|---|---|---|---|---|
| HuGaDB / raw | 0.475 | **0.644** | 30.2 | **61.9** |
| HuGaDB / norm | 0.321 | **0.661** | 21.8 | **61.3** |
| LARa / raw | **0.386** | 0.360 | **28.9** | 17.5 |
| LARa / norm | 0.346 | 0.402 | **29.1** | 21.3 |

On HuGaDB, swapping PCA windows for context embeddings roughly doubles both oracle accuracy and oracle-transport F1@50. **On LARa, oracle-transport F1@50 is lower with context than with PCA (17.5 vs 28.9 raw; 21.3 vs 29.1 norm).** Nearest-centroid accuracy does not fall uniformly, however: it drops slightly on raw (0.386 -> 0.360) and *rises* on norm (0.346 -> 0.402). This points to an interaction between representation geometry and this decoder, not to uniformly worse separability.

Two limits apply to this test:
- **It does not rule out optimization.** The oracle final solves were cold-started (no warm start), and every LARa context run hit the 500-step cap (439/439). LARa PCA was nearly as capped (416 and 425 of 439), as were the HuGaDB runs (269–310 of 364).
- **It does not show that LARa's action information is lost.** High inter-centroid cosine may reflect a shared embedding component rather than irreversible collapse.

The HuGaDB oracle MoF (63–66) also shows that MoF can improve through discovery or decoding, not only through the encoder.

*Revision (27 September 2026):* earlier wording here and in "Bottom line" said the LARa embedding was "intrinsically less class-separable". That claim is withdrawn; this test does not support it.

### The mechanism: class centroids collapse together in LARa's context space

| Dataset / norm | Mean cosine between different-class centroids: PCA | Mean cosine between different-class centroids: context |
|---|---|---|
| HuGaDB / raw | -0.05 (near orthogonal) | 0.52 |
| HuGaDB / norm | -0.08 | 0.38 |
| LARa / raw | -0.07 (near orthogonal) | **0.65** |
| LARa / norm | 0.01 | **0.69** |

In LARa's context space, different action classes point in almost the same direction (cosine 0.65-0.69) -- far more collapsed than HuGaDB's already-elevated 0.38-0.52, and starting from PCA space where LARa's classes were about as separated as HuGaDB's. Per-class oracle recall (PCA -> context) is uneven rather than uniformly worse:
- **Standing** drops (0.37 -> 0.18 raw, 0.39 -> 0.26 norm).
- **Handling(centred)**, the 44% majority class, stays low (0.31 -> 0.28 raw; 0.19 -> 0.27 norm).
- **Synchronization** rises sharply (0.50 -> 0.77 raw, 0.61 -> 0.88 norm).
- Walking, Cart and Handling(upwards) change modestly.

Context helps some classes and hurts others, while the class centroids become far less distinct overall.

*Correction (27 September 2026):* an earlier version of this paragraph attached these recalls to the wrong classes. LARa and HuGaDB ground-truth label IDs are 1-indexed, but the diagnosis script indexed the mapping file from 0, which shifted every class name by one. That version described a "collapse toward None/background"; that description was wrong. The accuracy, transport and centroid-cosine numbers above do not depend on class names and are unaffected. The confusion matrices shown in chat on 26 September carried the same shift and omitted the true None class.

### Why this dataset is harder for a masked-code contextualizer

| | HuGaDB | LARa |
|---|---|---|
| Median segment duration | 3.47 s | 2.20 s |
| Mean segments per recording | 10.5 | 32.1 |
| B-code receptive field (~128 codes) | ~32 s | ~31 s |
| `stage_b_003` masked-accuracy margin over copy | -0.007 to +0.027 | -0.049 to -0.015 (all 6 cells below copy) |

LARa switches actions almost twice as often (a new segment roughly every 3.7 s of a 120 s recording) as HuGaDB does, so each ~31 s context window spans many more ground-truth segments on LARa. Masked-code prediction over that window is also measurably harder to learn on LARa (all 6 cells score below the copy baseline, vs. roughly break-even on HuGaDB). And LARa's own class definitions (three ordinal phases of the same manual-handling motion: upwards/centred/downwards) are kinematically closer to each other than HuGaDB's classes (running, sitting, bicycling, ...), so a pooled representation trained without labels has less natural incentive to keep them apart -- and, per the centroid-collapse numbers above, it does not.

### Bottom line

On HuGaDB, label-derived prototypes on the same representation and fixed transport raise F1@50 from 37 to 62. That shows large discovery headroom without any change to the temporal kernel.

On LARa, the context representation works worse with this cosine-centroid decoder than PCA does. Whether the cause is lost information, geometry that is incompatible with a single-centroid cosine readout, or unconverged optimization (every context oracle solve was capped) is unresolved. Distinguishing these needs a more flexible fitting-only readout and warm-started, larger-budget solves. Candidate representation-side changes, untested:

1. A smaller context window and/or a per-dataset masking scheme matched to LARa's faster segment rate; the current 30 s window and masking objective are shared across datasets and not tuned per dataset.
2. A check on whether LARa's masked-code task is dominated by the 44%-majority "Handling(centred)" class, which would let the model minimize loss while ignoring the rarer phase distinctions.