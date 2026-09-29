# Research opportunities from SMQ, BehaveMAE and LAPS

28 September 2026. Research assessment, not an instruction to start experiments.

## Scope and recommendation

This review inspected selected reports, machine-readable results and relevant implementation paths in all three repositories. It did not independently rerun every experiment. Later audits qualify earlier reports; probe F1, boundary-tolerance F1 and segmental F1 are not interchangeable.

The strongest method opportunity is **learning reusable code phrases through the reconstruction decoder**, with an explicit test of whether phrase identity remains useful across contexts. The most ambitious alternative is **unsupervised discovery of overlapping movement components with separate temporal progressions**. A third, scientifically substantive direction is **determining when a reconstruction code can legitimately be interpreted as a movement primitive**.

These are competing project scopes. They are not a proposal to implement three architectures. Each has a specific prospective contribution and a comparison that can invalidate it. The literature screen identifies close precedents and narrows the proposed contribution; it does not certify a first-ever claim.

## 1. Evidence that changes the opportunity assessment

### BehaveMAE: useful motion can be accessible through the decoder but poorly exposed by isolated codes

On Shot7M2, the original FSQ one-hot code probe scores 41.6 +/- 0.4 all-F1. Poses reconstructed from that model's quantized sequence score 58.5 after PCA and probing, versus 60.8 for the matched original root-centred poses. These are different readouts and temporal supports, so their difference is not an estimate of information loss. It establishes a constructive recovery path through the decoder.

The source `experiments/fsq_exp2/exp2/part_a.py` actually loads cached quantized vectors and calls the reconstruction decoder; this is not a probe of the prequantization encoder. The comparison is therefore directly relevant to the original discrete-code proposal.

Changing the context decoder to a single-token decoder raises one-hot all-F1 from 41.6 to 51.0 while position reconstruction error rises from approximately 13.3 to 26.9 mm. The intervention also changes architecture and capacity, so a future causal test must control those changes. Nevertheless, this is a strong observed trade-off to investigate.

Sources:

- [Experiment 2 summary](C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/fsq_exp2/SUMMARY.md)
- [Decode-and-probe implementation](C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/fsq_exp2/exp2/part_a.py)
- [Independent review](C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/research_review_2026_09_20/RESEARCH_REVIEW.md)

### BehaveMAE: local content is useful; ordered composition remains unestablished

On hBABEL, a discrete centre token plus a local histogram scores 22.96 Top-30 versus 20.26 for the centre alone. The paired reported gain is 2.70 points, with a recording/seed bootstrap interval of [1.43, 3.91]. The trained continuous-input/continuous-target and discrete-input/discrete-target temporal models score 18.91 and 12.21 respectively. These systems are not parameter matched; the discrete input embedding overfits and the masking task has documented target-support overlap.

This makes local bags an essential control for a new method. It does not establish that order is intrinsically unhelpful. The older code-boundary and fine-tuning conclusions also need the qualifications in the independent review: count-limited boundary metrics, evaluation-coordinate mismatch, and incompletely paired fine-tuning crops.

Sources:

- [Core and follow-up findings](C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/exp6_discrete_motion/FINDINGS.md)
- [Saved local-context statistics](C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/exp6_discrete_motion/manifests/post_core_stats.json)

### SMQ: native codes already support useful downstream segmentation

HuGaDB pilot 002 native VQ achieves 41.90 +/- 4.24 F1@50 with contextual ASOT, compared with 4.47 under the categorical arm. The comparison changes contextual representation and emissions together; it does not isolate ordering. Nevertheless, it rules out treating all native-code processing as a demonstrated failure.

In the separate encoder sweep, raw-input K-means tokens plus the learned contextualizer achieve 46.88 +/- 2.87, versus 39.61 +/- 3.07 for epoch-30 encoder K-means. These are matched within that sweep, but should not be subtracted from pilot means as a matched intervention. The later encoder's empirical action MI is slightly higher despite worse segmentation. Thus simple marginal action association is not enough to select a tokenizer for this decoder.

The SMQ FSQ failures do not settle FSQ as a family: pilot 001 uses one code, and pilot 002 uses 23 of nominal 512 with a restrictive three-coordinate LayerNorm. In contrast, BehaveMAE's FSQ runs include high-occupancy models. Avoid pooling these into one claim that FSQ always collapses.

Sources:

- [Pilot comparison](../../results/fsq_reconstruction/PILOT_COMPARISON.md)
- [Encoder sweep](../../results/fsq_reconstruction/encoder_sweep/encoder_sweep_001/ANALYSIS.md)
- [Frozen study](../../results/frozen_study/frozen_001/ANALYSIS.md)

### LAPS: cross-context reuse is a concrete unresolved capability

The 27-video Breakfast diagnostic includes 123 filtered action instances. Its code-histogram cosine similarity averages 0.478 for the same action in different activities and 0.487 for different actions in different activities. Cross-video retrieval top-1 is 7.32% against its reported 4.17% reference. Adding bigrams reduces top-1 to 5.69%.

Same-video/different-action similarity is 0.668. This motivates testing contextual dependence; it does not isolate whether the cause is scene, person, tracking, objects, or temporal phase. The sample is exploratory, and Breakfast boundary F1 at 2/5 seconds is not SMQ's segmental IoU F1. The local LAPS runs are not a reproduction of the paper's benchmark protocol.

Sources:

- [Saved similarity results](C:/Users/gzaz976/Documents/repo/Behaviour/LAPS/output/segment_action_similarity/report_default.json)
- [Bigram extension](C:/Users/gzaz976/Documents/repo/Behaviour/LAPS/output/segment_action_similarity/report_extensions.json)
- [Pair taxonomy and implementation](C:/Users/gzaz976/Documents/repo/Behaviour/LAPS/scripts/analyze_segment_action_similarity.py)

## 2. Opportunity A: learn a dictionary of decoder-grounded code phrases

**Question:** Can distinct short code strings be recognized as instances of the same movement, without assuming that individual IDs already have stable semantic meanings?

Keep a healthy tokenizer and its decoder frozen. Treat a short code span as the object to interpret. Use the decoder to determine the local movement produced by that span and how replacing it changes the surrounding reconstruction. Learn phrase types that group different strings with compatible movement effects, while preserving distinct motion and transition patterns.

A concrete starting mechanism is a context-conditioned substitution metric: compare two observed candidate phrases inside several compatible contexts, measuring both their decoded movement difference within the span and the disturbance outside it. Distil these comparisons into a compact phrase model. Begin with a fixed span related to the decoder support; do not add learned boundary selection to the first implementation. Compose the resulting phrase types for action discovery using a fixed downstream decoder.

Only plausible contexts should be used. Arbitrary token splicing can create off-distribution decoder artifacts; cycle consistency alone cannot establish real semantic equivalence. Physically different motions must remain distinguishable. Raw observations can validate reconstruction effects during development, but deployment inputs must remain codes plus the declared frozen decoder/model assets.

**Prospective novelty:** context-dependent equivalence of *code phrases* as a learned interface for unsupervised action discovery. Neither exact code-string equality nor proximity of isolated codebook vectors defines the primitive. A successful method would recover reusable movement types from an existing reconstruction codec without action-label supervision or replacing its encoder.

**Closest work and required distinction:**

- [Action Motifs](https://arxiv.org/abs/2604.28173) already clusters learned atoms and mines frequent categorical patterns. A new contribution must show why decoder-grounded phrase equivalence handles alternative code strings better than frequent-pattern grouping.
- [MoGeFlow](https://arxiv.org/abs/2606.11656) already exploits motion codebook geometry and measures decoded changes from code replacement. Merely using code vectors or decoder distances would be insufficient. The proposed distinction is context-conditioned phrase equivalence and unsupervised discovery/reuse, rather than geometry-aware text-conditioned generation.
- [D-CLOT](https://arxiv.org/abs/2608.05877) already refines representations and action prototypes. Leave that decoder fixed in the first comparison so the claimed mechanism is the phrase interface.

**Decisive evidence:** better cross-context retrieval and action segmentation than (i) isolated IDs, (ii) local histograms, (iii) code-vector contextualization, and especially (iv) reconstructing motion then applying the same feature extraction/discovery method. Match effective raw-frame support, tokenizer rate, downstream capacity and label exposure. If simple decode-then-cluster explains the benefit, the proposed mechanism has not earned its complexity or novelty.

**Feasibility:** strongest fit to existing healthy BehaveMAE FSQ and SMQ VQ checkpoints. Decoder queries can be cached; no foundation-model training is required initially. A one-week feasibility allocation is reasonable as a project decision, but runtime needs profiling. LAPS is a later stress test, not a compulsory first implementation target.

## 3. Opportunity B: discover overlapping movement components with separate clocks

**Question:** Is forcing concurrent movements into one whole-body token stream and one action assignment an avoidable obstacle to compositional reuse?

Use a small number of anatomical token streams, each with its own latent progress through a motif. Learn sparse coordination between streams so that a persistent locomotion component can coexist with a brief hand interaction. An action is a learned arrangement of component trajectories; the model need not enumerate every combination as a separate primitive.

The architectural challenge is joint label-free assignment of local phase and cross-part coordination, with non-degenerate reconstruction or prediction constraints. Independent part clustering followed by concatenation would be a baseline, not the proposed contribution. A coupled factorial/segmental model is another necessary baseline.

**Prospective novelty:** unsupervised discovery of asynchronously composed discrete movement components that transfer to previously unobserved combinations. The contribution must be demonstrated through component reuse and composition generalization, not simply a better pooled score after splitting joints.

**Closest work and required distinction:**

- [LAC](https://arxiv.org/abs/2308.14500) already learns composable skeleton representations and uses synthesized motions for representation learning, followed by downstream fine-tuning. The target here is label-free discovery and segmentation of overlapping components.
- [HiST-VQ](https://arxiv.org/abs/2604.15196) already learns a subaction/action quantization hierarchy; adding another quantization level is insufficient.
- Factorial motion models are longstanding prior art. See [Vollmer et al.](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2012/Vollmer-KI-2012.pdf). The new mechanism must specify what it adds to factorial decomposition.
- [FrankenMotion](https://arxiv.org/abs/2601.10909) addresses asynchronous body-part composition in generation. Its part-level annotations are produced through an automated LLM pipeline, so they are a potential evaluation resource, not unquestioned human ground truth.

**Decisive evidence:** withhold natural combinations of component movements during representation training, retain the constituent components in other combinations, then test discovery on the withheld combinations. Hold out performers as well where identities and sample size permit. Labels may define an evaluation split, but must not train the discovery model. Fix semantic matching on a separate mapping population for a transfer claim. Synthetic recombinations are useful mechanism checks but cannot be the only positive result.

**Evidence strength and feasibility:** this is a structural hypothesis motivated by multilabel Shot7M2/hBABEL tasks and the reuse problem, not a demonstrated cause of current failure. It has the largest architectural ambition and the largest deadline risk. It changes the single-state output assumption, so use overlapping-label datasets. HuGaDB is IMU data and cannot be treated as a complete pose/shape test; LARa has different class and subject counts and optical motion-capture inputs.

## 4. Opportunity C: establish when motion codes are interpretable as primitives

**Question:** Which apparent properties of a movement vocabulary belong to the motion, and which depend on an arbitrary encoding convention or reconstruction decoder?

An exact control can separate these. For pairs of code IDs in an alphabet of size K, define

`R(a,b) = (a, (a+b) mod K)`.

This is invertible: `R^-1(u,v) = (u, (v-u) mod K)`. Leave an odd terminal token unchanged. If the original decoder is D, use `D_R = D composed with R^-1`. The reconstructed sequence is identical, the alphabet size and fixed-width token count are unchanged, and no motion information is deleted. Yet individual-token associations and code-transition statistics can change.

This is a mathematical control, not an experimental result or a new information-theory theorem. It does not preserve marginal code counts, actual entropy-coded bitrate, one-token accessibility, or unadjusted decoder support. Compare equal physical observation support and report the inverse wrapper's cost. Do not demand invariance of a one-token readout when its available information was intentionally changed.

Use such controls with matched-capacity decoder-context interventions and healthy VQ/FSQ/K-means models to establish how reconstruction, semantic accessibility, temporal locality and reuse separate. Build an evaluation criterion that rewards stable discovery across equivalent encodings while still discriminating physically different movements.

**Prospective novelty:** a controlled account of the semantic identifiability of motion primitives, paired with a practical corrected evaluation or discovery method. Existing results supply motivating cases; a cross-codec mechanism and a prospective prediction are needed to turn them into a scientific contribution.

**Closest work and required distinction:** [Effective Context in Transformers](https://arxiv.org/abs/2605.13485) already studies lossless recoding and finite-context prediction. [What Matters for Latent Actions](https://arxiv.org/abs/2608.19613) already conducts a broad robotic latent-action design/proxy study. The proposed question is specifically whether reconstruction-trained movement symbols and their boundaries warrant primitive-level interpretation. A generic survey of failed tokenizers or the elementary invertibility observation alone would be too weak.

**Decisive evidence:** demonstrate the predicted separation across healthy codecs; use controlled interventions rather than model-family correlations; show that the proposed criterion predicts cross-context discovery or that a corrected reader restores performance without changing the encoded motion. Include an independently chosen evaluation population because previous benchmarks have already influenced many design decisions.

**Feasibility:** makes the greatest use of existing runs and requires the least new training. It is a research-question paper route, not a new-SOTA claim. It can also supply the evaluation backbone for Opportunity A rather than becoming a separate project.

## 5. Allocation and exclusions

My preferred scope is **A as the method, with the narrow controls from C as the scientific explanation**. Choose B instead only if the priority is the more ambitious compositional architecture and the project accepts the added implementation and evaluation risk.

Use one existing healthy codec first. Freeze tokenizer, input exposure and downstream decoder during the mechanism comparison. Expand only after a result distinguishes the mechanism from decode-then-cluster and local pooling. A publication claim then needs independent data/populations, repeated training, appropriate motion baselines including MASQ/HiST-VQ, and a matched D-CLOT comparison where feasible. None of the existing tables establishes new SOTA under such a protocol.

Do not make another generic VQ-versus-FSQ sweep, vocabulary increase, commitment sweep, smoothness loss, masked-code model, or second hierarchy the headline contribution. Existing work and this project's experiments already cover these ingredients extensively.

The tempting audio-inspired semantic/kinematic split is also not an unoccupied contribution: [SeMoCo](https://arxiv.org/abs/2608.24334) explicitly transfers that design to motion, while [X-Tokenizer](https://arxiv.org/abs/2606.14752) separates semantic and reconstruction roles for robot actions. Likewise, simply adding action-centric cycle consistency overlaps [CycleMimic](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Learning_a_Unified_Latent_Action_Space_from_Videos_with_Action-centric_CVPR_2026_paper.pdf).

The positive opportunity is to specify and demonstrate what makes a discrete movement representation *reusable for discovery*. The existing repositories contain unusually useful controlled evidence for that question. They do not yet provide its answer, but they make a focused mechanism study considerably better grounded than another unrestricted architecture search.
