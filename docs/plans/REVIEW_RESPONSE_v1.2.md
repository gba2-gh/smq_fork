# Response to the motion-plan review

25 September 2026 | Revised plan and Stage A brief: version 1.2

## Verdict

Accept the review's demand for more discriminating comparisons, but narrow the architectural expansion and correct several interpretations. The principal objective remains **learning coarse action structure from a fine discrete motion sequence**. Stage A tests categorical temporal assignment, Stage B learns context from code IDs, and Stage C combines them. Continuous features remain a declared reference arm; they do not replace the discrete input to the proposed model.

The linked `THREE_STAGE_MOTION_PLAN_v1.0.md` is the preserved original. The substantive review changes were present in the active plan and agent brief as version 1.1. Both reviewed files have been archived as `*_v1.1_review.md`; the active files are now version 1.2. No experiment or model training was launched during this review.

## Decisions on the proposed changes

| Proposed change | Decision | Reason and resulting requirement |
| --- | --- | --- |
| Explicit fine-versus-class-sized vocabulary hypothesis | Accept, qualify | K=500 versus K=C is useful. SMQ is a system comparison, not an isolated K intervention; its encoder/tokenizer and temporal resolution differ. A fixed K independent of C does not erase the frozen encoder's original training assumptions. |
| Fixed mode-filter baseline | Accept | Costs no additional fit and tests whether the structured method beats this particular smoothing rule. It does not stand in for all temporal models. |
| K=C transport control | Accept | Directly tests whether the large vocabulary helps within the stated decoder family. Share samples/scaler/PCA. Correct the review's log(K) scaling: both arms now divide by log(500), retaining the same penalty scale per nat and total smoothing mass. Alphabet capacity still differs. |
| Continuous PCA transport arm | Accept as a reference | Helps determine whether the discrete arm has a measured advantage over a plausible continuous competitor. Different emission geometry means the result cannot establish a universal causal effect of quantization. Keep it separate from the proposed code-input architecture. |
| Mandatory sticky HMM | Defer | Valuable later, but creates a separate objective, fitting algorithm, and tests. The fixed smoother provides a bounded initial comparison. Do not claim superiority over HMMs without running one. The review's HMM specification also needs corrections below. |
| Reduce permutations to a sanity check | Accept | Shuffling destroys local code–action alignment as well as temporal structure. It is not a clean test of a learned grammar. Keep the four pooled seed-111 runs and retain recording composition in the interpretation. |
| Prototype-grouping initialization | Accept, qualify | Provides deterministic paired geometric initialization without GT boundaries. Sparse short histograms are a reasonable concern, but neither “near-one-hot” nor “mostly subject identity” is established. Declare that prototype geometry enters initialization. |
| Two ASOT coefficient settings | Accept as primary plus sensitivity | The numbers match the pinned CLI defaults. Fit under T, report T first, then re-decode under E. E changes four terms and does not test E-trained models or isolate one regularizer. Do not choose the better setting after scoring. |
| 25-step inner fitting schedule | Accept, rename | Warm-started nonincreasing steps plus exact parameter updates are inexact alternating minimization. This is not ordinary likelihood EM. Final stationarity and outer progress are reported separately. |
| Include capped fits | Accept with separate interpretation | The main table evaluates the specified budgeted algorithm, with cap counts and residuals visible. A second table reports convergence-qualified paired comparisons. Do not equate a finite decreasing trace with stationarity or silently remove difficult seeds. |
| Automatic GPU execution and exact cross-device agreement | Modify | CPU remains the default; CUDA is explicit after parity/timing checks. Record the device and prohibit silent fallback. Compare small-case probabilities/objectives with declared numerical tolerances; hard labels near ties need not agree. |
| Stop after pooled evaluation before transfer | Accept as scope management | A separate user launch is consistent with the requested workflow. Continue with unchanged settings/all primary arms. If transfer is omitted, it stays not run and no transfer claim is supported. Pooled inspection is not a new tuning set. |
| A second continuous-input contextualizer | Defer | This doubles the contextual representation branch before the main code-input architecture is tested. It is a valid later competitor, not required to answer the immediate discrete-sequence question. |
| Subject-shortcut diagnostics | Accept a bounded descriptive version | Save subject/state contingencies and fixed-sample neighborhood associations. Exclude self and separate same-recording repetition from cross-recording neighbors. Defer an additional fitted subject classifier. These use subject labels, though no action annotations. |
| Stage C filter baselines | Accept | Derive them from saved context-only assignments, with no new model training. |
| Related-method positioning and BABEL | Accept positioning; defer BABEL | HiST-VQ already has a fine-to-coarse hierarchy. BABEL is a sensible later benchmark, not a universal condition for any publishable result. Keep these stages on HuGaDB/LARa and label published numbers unmatched. |

## Findings that require correction, not just a design preference

### 1. K=500 did not work only with supervised readouts

The review correctly reports weak hard-histogram results at the intended window sizes, but its stronger conclusion omits the soft-histogram results:

| Q1/Q2 pooled oracle-boundary condition | HuGaDB W=15 F1@50 | LARa W=12 F1@50 |
| --- | --- | --- |
| K=10 hard histograms | 48.33 ± 0.43 | 23.26 ± 2.57 |
| K=500 hard histograms | 35.57 ± 6.86 | 6.19 ± 0.20 |
| K=500 soft histograms, base temperature | 61.90 ± 4.54 | 20.15 ± 2.59 |

Values were checked in `results/exp2round/q1q2/report_details/clustering_summary.csv`, representation `latent`; SD is population SD across the three seeds. Soft assignments change the representation and these runs use oracle segments. They do not validate the proposed hard-code decoder. They do refute the unqualified assertion that K=500's usefulness appears only under supervised readout. The revision retains hard codes as the primary input and does not add a soft grid.

### 2. Subject information is a concern, not a causal explanation

At these windows the information table does show I(U;S)>I(U;A): HuGaDB 1.680>1.428 nats, LARa 1.812>0.647. That does not mean a matching pair of codes mostly indicates subject identity, nor that identity caused an action-clustering failure. The marginal target entropies differ and the associations can coexist. Conditional information and normalized diagnostics are relevant context, not a causal identification strategy. The plan now says what was measured without claiming a mechanism.

Source: `results/exp2round/q1q2/report_details/information_summary.csv`, latent K=500, W=15/12. Any later subject probe would also be a supervised diagnostic with respect to subject labels, even if action labels were absent.

### 3. Sparse histograms are not necessarily near-one-hot

A complete one-second block contains four contributing code windows in HuGaDB and up to five in LARa because the 12-frame windows cross 50-frame block boundaries. A histogram could put mass on several different codes. Sparsity supports concern about unstable initialization, but not the stated near-one-hot or subject-dominance conclusion. Geometry-based grouping is accepted for its coherent shared initialization, not because the previous initialization was mathematically invalid.

The new initialization consumes codebook geometry. Categorical assignment still operates on IDs, but the whole fitting procedure must not be advertised as purely symbolic or invariant to arbitrary changes of codebook geometry.

### 4. Dividing negative log-likelihood by log(K) does not bound it by 2

Under v1.1's smoothing and normalization, an unobserved code in state a has probability `(eta/K)/(M_a+eta)`. Its cost is `log(K*(M_a+eta)/eta)/log(K)`, dependent on fitted count mass M_a. For example K=500, eta=60 and M_a=1,000,000 gives approximately 2.56. A cosine cost divided by two is in [0,1]; these are not interchangeable scales. The revision uses the same fixed log(500) denominator for both categorical vocabulary sizes, rather than the review's K-dependent log(K). This removes an avoidable penalty-scale confound without adding a sweep. It also requires cost distributions and component magnitudes to be logged; the continuous comparison still changes emission geometry.

### 5. The coefficient lineage is specific, not universal

Verified the local ASOT clone at commit `0c4c86b1037eb4c3e262197aeee115b31e220c9d`. The T/E numbers match `src/train.py` command-line defaults. The class constructor differs, and `run_bf.sh` overrides alpha-train to 0.4, alpha-eval to 0.7, and training action-marginal weight to 0.1. Consequently the review's proposed values are legitimate prospective settings, but not the sole original training/evaluation protocol.

The scalar-objective concern is real in the inspected code: applying one half to the inner product of the combined structural-plus-unary gradient also halves the linear cost. Production must evaluate the objective consistent with the gradient. Under fixed row masses, the reference KL gradient's added constant has no effect after projection; parity tests must compare the projected updates, with matching dtype/kernel/step sizes.

Source: [pinned ASOT code](https://github.com/mingu6/action_seg_ot/tree/0c4c86b1037eb4c3e262197aeee115b31e220c9d). The original structural term and the corrected production objective remain in scope.

### 6. The proposed HMM needs a consistent prior and duration description

The review weights emission log likelihood by n_t/W but gives the M-step counts n_t and total pseudocount eta. To obtain that update, the prior contribution in the corresponding weighted objective must use coefficient `eta/(W*K)` on each log theta, not `eta/K`. State the weighted objective explicitly if this arm is later implemented.

A sticky first-order HMM also has an implicit geometric dwell-time distribution. Fixed transitions do not make it duration-free. With self-transition s, mean dwell is 1/(1-s) windows. This is distinct from an explicit HSMM duration model, but still a duration assumption. These are reasons to specify the comparator carefully, not arguments that HMMs are unsuitable.

### 7. Stage C versus K=C would confound context and vocabulary size

The revised Stage A contrast A1 versus A1-KC is the relevant coarse/fine comparison. A contextual K=500 arm beating a noncontextual K=C arm changes two components. It cannot isolate a fine-vocabulary advantage. Version 1.2 states this limitation rather than adding another contextualizer grid.

## Resulting scope and implementation burden

Stage A has four fitted arms: A0, A1, A1-KC, A-cont; A0+filter is derived. This is 48 pooled fits plus four permutation sanity fits, and 192 subject-disjoint fits if that protocol is launched: **244 planned fits**, versus the review's 304. Each regular configuration has ten scored prediction sets across T/E, while each permutation fit has two. Fit count and final-decoding runtime are reported separately.

Stage B trains only B-code. Stage C reuses it for context-only and contextual-transport fits, plus a derived filter and bounded permutation diagnostic. No HMM implementation, new encoder, FSQ training, second contextualizer, or new dataset is required to complete these stages.

The stronger controls do not turn this into proof of a paper contribution. HiST-VQ's existing hierarchy makes “fine-to-coarse” alone insufficient positioning; the proposed frozen-tokenizer context/decoding formulation must earn its contribution through results and comparison. See the [HiST-VQ paper](https://arxiv.org/html/2604.15196v1) and [MASQ paper](https://arxiv.org/html/2608.29891v1) for the verified related-method context. Their reported benchmark numbers remain unmatched references here.

## Deliverables and limits

Updated `THREE_STAGE_MOTION_PLAN.md`, its Word counterpart, and `STAGE_A_AGENT_INSTRUCTIONS.md`. Preserved the reviewed versions and the original Word document. This review checked document consistency, saved Q1/Q2 summary rows, the pinned ASOT source, and primary related-paper sources. It did not run model fits or establish that any new proposed setting performs well.
