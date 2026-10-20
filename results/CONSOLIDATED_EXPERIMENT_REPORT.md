# SMQ experiment record and implementation audit

**Scope:** the research sequence from [the reproduction and diagnosis report](REPORT_reproduction_and_diagnosis.md), through [T1–T4 preregistration](PREREG_T1-T4.md), to [the final report](FINAL_REPORT.md). Compiled and audited on 21 September 2026 against the current source, saved predictions, run configurations, and JSON/CSV results. The original reports and experiment outputs are preserved.

**Main finding:** SMQ's discrete codes retain substantially less action information than its continuous representation. A finer patch grid alone cannot explain or repair the large frame-accuracy deficit. Larger codebooks recover some information, but converting it into useful action segments remains dataset-dependent. No tested intervention establishes a general improvement across frame accuracy and segment quality. Temporal-consistency training remains untested.

**Audit finding:** the principal numerical tables are reproducible from the saved artifacts, but several interpretations in the earlier reports are too strong. Boundary diagnostics contain implementation problems; the duration distribution is not established to be geometric; the JSD “floor” is not a mathematical floor; and the initial report's claim that many-to-one mapping does not improve segment metrics is contradicted by HuGaDB's saved results.

## 1. Evidence, protocol, and completion status

The original run environment was Python 3.10.14, PyTorch 2.1.0+cu121, a single RTX 4090, and repository commit `f389b8b` plus local changes. This audit concerns the current working tree and its artifacts; it does not independently establish the historical identity of every training revision or published checkpoint. Paper values below are the reference values recorded in the supplied reports, not a new external literature verification.

| Dataset | Sequences in released-checkpoint evaluation | Action classes C | Native fps | Default patch |
|---|---:|---:|---:|---:|
| HuGaDB | 364 | 10 | 60 | 60 frames / 1 s |
| LARa | 439 | 8 | 50 | 50 frames / 1 s |
| BABEL-1 | 1,960 | 5 | 30 | 30 frames / 1 s |
| BABEL-2 | 255 | 5 | 30 | 30 frames / 1 s |
| BABEL-3 | 183 | 5 | 30 | 30 frames / 1 s |

MoF is frame accuracy; Edit measures agreement of collapsed label sequences; F1@10/25/50 measures labeled-segment agreement at the corresponding IoU thresholds. These scores are percentages, higher is better. JSD here means **100 × the square root of base-2 Jensen–Shannon divergence** between segment-duration histograms; lower is better. Changes in percentage scores are percentage points.

The primary evaluation uses one **dataset-level Hungarian assignment**, estimated using benchmark labels. It is a clustering evaluation convention, not a learned deployment-time classifier. Per-sequence assignment is a different, more permissive protocol. Many-to-one mapping and oracle relabeling use additional ground-truth assistance and are diagnostic readouts, not deployable unsupervised methods. Probes train supervised classifiers with folds grouped by recording; they do not establish generalization to unseen subjects or datasets. The unsupervised representation and temporal models are fitted on the benchmark collection being analyzed.

| Experiment | Research question | Work actually completed | Status |
|---|---|---|---|
| E0 released weights | Can published segmentation scores be recovered, and with which matching protocol? | Five datasets, local and global matching | Completed |
| E0 retraining | How much do results vary across training runs? | Three seeds each on HuGaDB/LARa; one additional BABEL-1 artifact | Completed for the two-dataset variance study |
| E1 JSD audit | Which duration metric convention fits the reference, and what does it measure? | Convention sweep, five-dataset scoring, nulls, grid oracle, histograms, Wasserstein distance | Completed, interpretation qualified below |
| Fig. 3 | Do selected examples display the reported qualitative errors? | Four GT/SMQ panels | Completed for SMQ only |
| D1–D8 | Is the limitation grid resolution, boundaries, labels, code capacity, or duration structure? | Grid and relabeling oracles, boundary diagnostics, smoothing, priors, probes and occupancy | Completed; some diagnostics require correction |
| T1 | Does increasing K recover useful action information? | 32 main SMQ runs: 8 baseline and 24 enlarged-codebook runs | Provisional; enlarged K has two seeds |
| T1 control | Does learned quantization beat raw-feature clustering? | 45 k-means runs: 5 K values × 3 datasets × 3 seeds | Completed |
| T1 threshold check | Does dead-code replacement suppress large codebooks? | Three extra SMQ runs at K=16C, threshold 10 | Exploratory, one seed per dataset |
| T2 | Does an adjacent-patch consistency loss improve action identity? | Loss implementation and weight calculation; no treatment results | **Not run** |
| T3 | Can HMM/HSMM decoding improve temporal segmentation? | 11 checkpoints × 8 decoder settings = 88 treatments | Completed on available checkpoints |
| T4 | Does granularity depend on patch duration or sampling rate? | 36 new runs plus the 8 shared T1 baselines | New conditions: three seeds; BABEL-1 baseline: two |

Training for T1/T4 used 30 epochs, Adam with learning rate 0.0005, latent dimension 16, 128 feature maps, three layers, EMA decay 0.5, commitment weight 1, and reconstruction weight 0.001. The 71 available T1/T4 configurations include the threshold checks; all have `micro_batch_size=null` and `tc_weight=0`.

The preregistered decision rule required a difference in means exceeding the larger seed SD **and** nonoverlapping mean ± SD bands. It was specified for three seeds, then applied at two or more. This is a descriptive rule, **not a statistical significance test or confidence interval**. There is no multiple-comparison correction. “No detectable effect” is not proof of equality. Welch tests were promised in preregistration but are not produced by the current summarizer.

## 2. E0 — reproduction and training variability

**Question and implementation.** Score released checkpoints using the repository's original segmentation metrics, distinguishing global from local label assignment. Separately retrain using three seeds (`1538574472`, `111`, `222`) to estimate variability. Implementation: [dump_predictions.py](../script/repro/dump_predictions.py), [e0_metrics.py](../script/repro/e0_metrics.py), and [eval_utils.py](../src/model/eval_utils.py). Evidence: [E0 JSON directory](e0/).

| Dataset | Published MoF / Edit / F1@10 / F1@25 / F1@50 | Reproduced |
|---|---|---|
| HuGaDB | 42.0 / 36.1 / 38.5 / 31.5 / 24.3 | 41.99 / 36.15 / 38.50 / 31.54 / 24.30 |
| LARa | 37.4 / 39.4 / 34.7 / 28.4 / 16.4 | 37.38 / 39.40 / 34.69 / 28.40 / 16.39 |
| BABEL-1 | 36.6 / 38.5 / 40.9 / 32.8 / 22.3 | 36.40 / 38.25 / 40.71 / 32.60 / 22.38 |
| BABEL-2 | 49.1 / 37.8 / 43.8 / 37.4 / 27.4 | 49.17 / 37.81 / 43.70 / 37.42 / 27.52 |
| BABEL-3 | 40.6 / 38.6 / 38.0 / 29.3 / 19.3 | 40.37 / 38.60 / 37.58 / 28.59 / 19.35 |

The global protocol closely recovers the reference: maximum differences are 0.05, 0.02, 0.25, 0.12 and 0.71 points respectively. For example, HuGaDB's local MoF is 55.27 instead of global 41.99. Local scores must not be mixed with the published baseline.

Three-seed retraining statistics below were recalculated from the saved E0 JSONs using sample SD; a few hundredth-place values differ from the original prose because it rounded intermediate results.

| Dataset | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| HuGaDB | 41.59 ± 1.13 | 33.74 ± 2.39 | 35.67 ± 1.94 | 28.01 ± 1.80 | 21.50 ± 2.54 |
| LARa | 36.26 ± 1.61 | 38.71 ± 0.95 | 33.67 ± 0.99 | 27.35 ± 1.02 | 15.75 ± 0.75 |

**Correctness and limitations.** This audit rescored all five cached released-checkpoint prediction files and reproduced all five global metrics within 0.00051 points of saved values. It did not rerun GPU inference or retraining. Near-identical scores support numerical reproduction; they do not prove that the released weights are exactly the weights used in a paper. The BABEL offsets do not by themselves identify checkpoint differences as the cause. Nondeterministic cuDNN is enabled, and microbatch use in the earlier E0 training batch was not recorded. Microbatching changes VQ EMA updates and is not equivalent to full-batch training.

**Analysis.** Reproduction is strong enough to support subsequent diagnostics. E0 and T1 are separate training batches and must retain their own baselines and SDs. A small number of seeds gives a variability estimate, not a universal significance threshold.

## 3. E1 — duration metric convention and validity

**Question and implementation.** Can the reported SMQ JSD values be reconstructed, and can that statistic distinguish better temporal segmentation? The sweep compares logarithm bases and histogram grids. The chosen implementation uses per-video 20-frame bins through the longest observed segment, an overflow bin, normalized histograms, base-2 JS distance, and an activity-weighted aggregate. Within each activity, videos receive equal weight; activity means are weighted by total frames. LARa uses filename scenarios; HuGaDB/BABEL effectively have one group. See [jsd_sweep.py](../script/repro/jsd_sweep.py), [e1_jsd.py](../script/repro/e1_jsd.py), [convention results](e1_jsd/convention_sweep.json), and [duration results](e1_jsd/jsd_summary.csv).

| Dataset | Published JSD | Reproduced | Split-GT reference | Matched-count duration null | Majority-patch oracle | Pooled W1, frames |
|---|---:|---:|---:|---:|---:|---:|
| HuGaDB | 87.1 | 86.93 | 79.58 | 83.29 | 73.12 | 164.16 |
| LARa | 74.2 | 73.89 | 66.45 | 54.85 | 68.04 | 50.63 |
| BABEL-1 | 72.8 | 74.51 | 67.86 | 72.69 | 52.47 | 20.56 |
| BABEL-2 | 77.0 | 78.71 | 81.56 | 82.06 | 44.47 | 11.71 |
| BABEL-3 | 81.9 | 83.27 | 69.13 | 85.04 | 46.91 | 39.77 |

The split-GT reference randomly divides a video's GT durations into two halves, repeated 50 times, when at least four segments exist. The matched-count null draws as many durations as SMQ predicts from the activity's pooled GT durations. The grid oracle assigns each patch its majority GT class. W1 is the pooled Wasserstein-1 distance between predicted and GT durations; it uses a different aggregation from per-video JSD.

Histograms show the expected lattice effect: HuGaDB predictions concentrate at 60-frame multiples, with 45.8% of runs at one patch, 21.4% at two and 10.4% at three. In BABEL-1, GT places 25.2% of segments in the first 20-frame bin versus 10.8% for SMQ. Figures: [duration and sequence histograms](../figures/repro_e1_jsd/).

**Correctness and issues.** The JS formula and cached scores are consistent; the audit independently recomputed JSD for all five datasets. The selected convention is a good fit among those examined, not uniquely proven to be the authors' implementation. HuGaDB/LARa differ from the reference by 0.17/0.31, so “exactly reproduced” is inaccurate. Under the assumed ×100 scaling, natural-log JS distance cannot exceed 83.26, which supports base 2 against a cited value of 97.4.

The following qualifications matter:

- The “sparsity floor” is an empirical split-sample reference, **not a lower bound**: identical duration histograms score zero. It uses half-sized samples and excludes videos with fewer than four GT segments from its valid means. Its dataset composition is therefore not identical to the main score.
- The matched-count null uses GT, including the evaluated collection, and need not produce durations summing to the video's length. It is a histogram control, not a segmentation algorithm.
- Majority-patch labeling maximizes frame agreement on that grid. Its JSD is not a minimum attainable JSD, nor is its F1 a proven maximum. “Patch ceiling” should be restricted to MoF.
- The 5.85-point LARa and 13.81-point HuGaDB oracle-to-SMQ gaps are observed reference gaps, not the metric's entire dynamic range of 0–100.
- The cited HuGaDB method difference 89.0−87.1=1.9 is larger than one SMQ seed SD of 1.38. Without uncertainty for both methods it is not a significance comparison.
- W1 avoids bin edges but still discards class identity and temporal placement. Report it in seconds or normalized units for cross-dataset comparison.

**Analysis.** JSD measures duration-distribution similarity, not action segmentation quality. Sparse videos, bin placement and patch size strongly affect it. The LARa null score and the T4 disagreement between JSD and F1 demonstrate why it should accompany, rather than replace, label and segment metrics. These results do not independently establish or refute HiST-VQ's overall performance: that model was not run.

## 4. Qualitative reproduction

**Question and method.** Does SMQ display the described transition confusion on concrete examples? [fig3_qualitative.py](../script/repro/fig3_qualitative.py) plots GT and globally mapped predictions for HuGaDB, LARa, BABEL-1 and BABEL-3. The examples include `HuGaDB_v2_various_01_00`, `L02_S15_R04`, and BABEL-1 `929`; BABEL-3 is selected by proximity to global MoF. Output: [four-panel figure](../figures/repro_fig3/fig3_smq.svg).

**Results and correctness.** The HuGaDB example displays confusion between sitting-down and standing-up transitions. The use of the same global mapping as E0 is appropriate. The BABEL-3 selection is reproducible but still uses GT for illustrative selection. Only the SMQ row was reproduced; this is not a visual replication of all competing methods.

**Analysis.** The figure illustrates a plausible label-confusion mechanism. Four selected sequences do not quantify its prevalence or establish its cause.

## 5. D1–D8 — diagnosis of the errors

### 5.1 D1: patch-grid oracle

**Question.** How much frame accuracy is unavoidably lost by emitting one label per patch?

**Method.** Assign the majority GT class independently to each patch and evaluate. [d_error_decomposition.py](../script/repro/d_error_decomposition.py); [raw decomposition](diagnostics/error_decomposition.json).

| Dataset | SMQ MoF / F1@50 | Grid oracle MoF / F1@50 | Irreducible grid loss in MoF | Gap from SMQ to grid oracle |
|---|---|---:|---:|---:|
| HuGaDB | 41.99 / 24.30 | 95.48 / 97.68 | 4.52 | 53.49 |
| LARa | 37.38 / 16.39 | 93.83 / 93.17 | 6.17 | 56.45 |
| BABEL-1 | 36.40 / 22.38 | 90.17 / 81.35 | 9.83 | 53.77 |

**Assessment.** Correct as an MoF upper bound at a fixed grid. F1/Edit/JSD are scores of this particular oracle, not optimized bounds. HuGaDB loses 58.01 MoF points relative to perfection, decomposed into 4.52 grid points plus 53.49 points below the grid optimum; earlier prose sometimes conflates these quantities.

**Analysis.** Grid resolution alone explains a small part of the observed frame error. The remaining gap includes label discovery and temporal decisions within the allowable grid; it cannot all be assigned to quantization by subtraction.

### 5.2 D2 and D6: boundary counts and representation change signals

**Question.** Are predicted boundaries near real transitions, and does the encoder improve a simple transition detector?

**Method.** Count nearest-boundary distances within half a patch. Separately compare changes between neighboring window means in raw inputs and latent features, using AUC and top-N peak picking. The latter is evaluated on at most 120 sampled sequences and uses the true number of boundaries as N. Source: [d_boundary_vs_label.py](../script/repro/d_boundary_vs_label.py).

| Dataset | Predicted boundaries within tolerance | GT boundaries covered | Latent / raw change AUC | Latent top-N / actual SMQ boundary score |
|---|---:|---:|---|---|
| HuGaDB | 31.3% | 58.6% | 0.620 / 0.587 | 22.6 / 35.3 |
| LARa | 41.4% | 57.0% | 0.692 / 0.685 | 48.3 / 46.7 |
| BABEL-1 | 53.8% | 46.0% | 0.531 / 0.567 | 38.0 / 41.4 |

**Implementation defects.** These are the original diagnostic values, not corrected conventional boundary precision/recall:

1. D2 skips a sequence whenever either side has no boundaries. The audit found that this omits **277 true boundaries in HuGaDB**, **37 predicted boundaries in LARa**, and **853 true plus 286 predicted boundaries in BABEL-1**. The corresponding omitted sequence counts are 40, 1 and 352. Empty-side cases should contribute missed or false boundaries.
2. `boundary_f1` independently tests nearest neighbors instead of enforcing one-to-one matches. Predictions at frames 9 and 11 against a single true boundary at 10, tolerance 1, score **100** in the implementation; one-to-one matching gives **66.67**. Dense duplicate predictions can inflate this score.
3. AUC labels all frames within tolerance as positive, so it measures proximity to transitions under one fixed signal. Top-N uses oracle segment counts and is not directly comparable to an unconstrained detector. The raw/latent comparison is also restricted to the sampled eligible sequences.

**Analysis.** This particular change detector gains little from the encoder. The experiment does not establish that no other boundary information exists or that boundary improvement has little headroom. The observed missed/spurious-boundary rates already argue against calling boundaries “solved.”

### 5.3 D7: oracle labels versus oracle segmentation

**Question.** Which intervention recovers more accuracy: relabeling predicted runs or imposing GT segmentation on current codes?

| Dataset | Original MoF / F1@50 | Predicted runs, majority GT label | GT runs, majority SMQ code then Hungarian |
|---|---|---|---|
| HuGaDB | 41.99 / 24.30 | 78.78 / 79.17 | 44.85 / 41.74 |
| LARa | 37.38 / 16.39 | 84.40 / 68.83 | 41.66 / 27.92 |
| BABEL-1 | 36.40 / 22.38 | 76.86 / 51.48 | 37.52 / 27.82 |

**Correctness and limitation.** The implementation performs the stated relabeling, but the interventions are asymmetric. Each predicted run gets its own GT-assisted label, while GT runs retain a globally constrained code mapping. Relabeling neighboring predicted runs to the same class removes their boundary during scoring. Likewise, adjacent GT runs receiving the same code merge. These are useful oracle interventions, **not a pure additive separation of fixed boundaries from fixed labels**.

**Analysis.** Considerably more accuracy is recoverable through GT-assisted relabeling than through majority-code assignment on GT segments. This supports a substantial code-to-action problem. It does not establish that actual boundary localization is already adequate.

### 5.4 D8: action probes, code occupancy and many-to-one mapping

**Question.** Is action identity present before quantization, and does it survive conversion to a single code?

**Method.** Five-fold recording-grouped logistic regression on patch-mean raw features, patch-mean latents, and one-hot codes. Compute NMI, code usage and majority-class coverage. The original diagnostic uses up to 250 sampled sequences and only complete patches. Source: [d_what_codes_encode.py](../script/repro/d_what_codes_encode.py); [diagnostic JSONs](diagnostics/).

| Dataset | Majority baseline | Raw probe | Latent probe | Code probe | Latent−code | Code/GT NMI | Distinct majority classes / K | Perplexity |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| HuGaDB | 28.19 | 78.79 | 79.50 | 56.95 | 22.55 | 0.480 | 6 / 10 | 7.66 |
| LARa | 43.52 | 64.95 | 70.44 | 47.71 | 22.73 | 0.184 | 3 / 8 | 5.76 |
| BABEL-1 | 44.14 | 62.62 | 62.20 | 52.17 | 10.03 | 0.117 | 2 / 5 | 3.93 |

No unused codes were observed in these probe samples. All codes being used does not mean they represent distinct action classes. Hungarian assignment remains mathematically feasible; “unsatisfiable” in the earlier reports should mean **a one-code-per-action semantic interpretation is unsupported**, not that the assignment solver has no solution.

The later released-checkpoint analysis also reports full-frame many-to-one mapping:

| Dataset | Global MoF → many-to-one | Edit → many-to-one | F1@50 → many-to-one |
|---|---|---|---|
| HuGaDB | 41.99 → 56.55 | 36.15 → 40.00 | 24.30 → 34.94 |
| LARa | 37.38 → 48.06 | 39.40 → 33.31 | 16.39 → 16.02 |
| BABEL-1 | 36.40 → 48.72 | 38.25 → 31.30 | 22.38 → 21.23 |

**Correction.** The initial report says Edit/F1 stay flat or drop under many-to-one mapping. HuGaDB contradicts that statement: F1@50 rises by **10.65** points. Evidence: [released-checkpoint analysis](exp/analysis/pretrained/). The mapping uses evaluation GT and cannot be claimed as an unsupervised method improvement.

**Assessment and analysis.** Fold-specific feature scaling is correctly fitted only on training folds. However, these are ordinary accuracies, not balanced accuracy, and grouping by recording does not prevent subject overlap across folds. The encoder was trained unsupervised on the dataset being probed. The accuracy gap is evidence of reduced **linearly decodable action identity**, not a direct information-theoretic measurement or proof that K alone causes the loss. LARa's latent improves over raw features by 5.49 points; a blanket claim that the encoder does not help is unwarranted. Later T1 probes include partial patches and should not be numerically pooled with this earlier probe batch.

### 5.5 D3: mode filtering

**Question.** Can simple removal of short code fluctuations recover segment quality without retraining?

| Filter width in patches | HuGaDB MoF / F1@50 | LARa MoF / F1@50 | BABEL-1 MoF / F1@50 |
|---|---|---|---|
| None | 41.99 / 24.30 | 37.38 / 16.39 | 36.40 / 22.38 |
| 3 | 42.40 / 26.57 | 37.52 / 16.31 | 35.82 / 22.29 |
| 5 | 42.65 / 29.33 | 36.38 / 13.05 | 34.64 / 15.51 |
| 7 | 41.57 / 22.70 | 35.77 / 9.69 | 34.01 / 13.71 |

**Assessment.** The patch-mode filter and re-estimated global mapping implement the intended post-hoc comparison. Width 5 improves HuGaDB F1@50 by 5.03 but worsens JSD from 86.93 to 90.01. Width selection is exploratory, on one released checkpoint. LARa/BABEL-1 do not benefit in F1@50. This is a dataset-specific observation, not a validated general recipe.

### 5.6 D4–D5: duration law and class-prior mismatch

**Question.** Do persistence and occupancy explain the duration and frame errors?

| Dataset | Self-transition probability | Geometric implied mean, patches | Observed mean, patches | GT mean, patches | Predicted CV / geometric CV | Prior TV distance | MoF bound from fixed mapped priors |
|---|---:|---:|---:|---:|---|---:|---:|
| HuGaDB | 0.6859 | 3.18 | 3.03 | 4.96 | 2.89 / 0.83 | 0.199 | 80.08 |
| LARa | 0.6500 | 2.86 | 2.81 | 3.73 | 1.53 / 0.81 | 0.209 | 79.10 |
| BABEL-1 | 0.6999 | 3.33 | 2.59 | 1.91 | 1.18 / 0.84 | 0.184 | 81.62 |

**Assessment.** The descriptive calculations are useful. However, `1/(1−p_stay)` is a fitted geometric-model mean, not the observed mean. The initial reports sometimes call 3.18/2.86/3.33 the observed runs; the actual values are 3.03/2.81/2.59. The large CV disagreement, especially HuGaDB, does not support “near-geometric by construction.” A temporal encoder produces correlated patch representations; independent nearest-code decisions do not imply a memoryless stochastic sequence. No goodness-of-fit test establishes a geometric law.

**Analysis.** HuGaDB/LARa oversegment on average, while BABEL-1 undersegments. The fixed mapped class-prior mismatch limits attainable frame overlap but leaves ample error beyond that bound. Neither statistic alone identifies a causal mechanism.

## 6. T1 — larger codebooks, merging, and raw-feature controls

**Question.** Does K>C preserve more action identity, is the benefit attributable to learned representations, and can it survive label-free merging?

**Design.** K/C ∈ {1,2,4,8,16}; HuGaDB C=10, LARa C=8, BABEL-1 C=5. Evaluate: (a) original bijective matching; (b) GT-assisted many-to-one; (c1) average-linkage geometry merge to C; (c2) spectral clustering of the symmetrized code-transition graph to C. Also measure probes and code usage. K=C makes both merges the identity. K>C has two seeds; K=C has three on HuGaDB/LARa and two on BABEL-1. Code: [make_jobs.py](../script/exp/make_jobs.py), [analyze_run.py](../script/exp/analyze_run.py), [kmeans_control.py](../script/exp/kmeans_control.py).

The dead-code threshold follows `max(1, round(10*C/K))`, giving 10, 5, 2, 1, 1. Thus the main sweep changes K together with a prespecified threshold schedule; its effect is not a pure K-only intervention.

The table gives seed means; full SDs and all published readouts are in [SUMMARY.md](exp/SUMMARY.md) and [per-run JSONs](exp/analysis/T1/).

| Dataset | K | n | Bijective MoF | Geometry MoF / F1@50 | Temporal MoF / F1@50 | SMQ code probe | Raw k-means code probe | Latent probe |
|---|---:|---:|---:|---|---|---:|---:|---:|
| HuGaDB | 10 | 3 | 40.7 | 40.7 / 22.0 | 40.7 / 22.0 | 56.9 | 51.5 | 76.5 |
| HuGaDB | 20 | 2 | 31.5 | 31.7 / 8.8 | 43.8 / 25.0 | 58.3 | 53.3 | 78.8 |
| HuGaDB | 40 | 2 | 19.3 | 30.6 / 7.5 | 37.9 / 20.1 | 59.1 | 54.7 | 77.5 |
| HuGaDB | 80 | 2 | 21.0 | 33.4 / 9.0 | 42.7 / 25.4 | 63.9 | 64.3 | 78.2 |
| HuGaDB | 160 | 2 | 13.2 | 33.1 / 9.4 | 41.4 / 22.6 | 63.4 | 68.1 | 79.1 |
| LARa | 8 | 3 | 36.8 | 36.8 / 15.8 | 36.8 / 15.8 | 47.9 | 43.7 | 70.3 |
| LARa | 16 | 2 | 23.2 | 44.4 / 16.6 | 32.9 / 15.6 | 52.1 | 45.8 | 70.8 |
| LARa | 32 | 2 | 14.9 | 45.5 / 12.2 | 32.8 / 16.8 | 52.9 | 45.5 | 70.7 |
| LARa | 64 | 2 | 10.2 | 45.8 / 13.2 | 31.6 / 16.7 | 56.1 | 46.2 | 70.8 |
| LARa | 128 | 2 | 7.5 | 46.8 / 13.2 | 30.0 / 16.4 | 57.2 | 46.8 | 70.8 |
| BABEL-1 | 5 | 2 | 34.5 | 34.5 / 14.7 | 34.5 / 14.7 | 44.4 | 43.4 | 65.3 |
| BABEL-1 | 10 | 2 | 28.4 | 42.6 / 10.7 | 35.0 / 16.9 | 48.2 | 46.5 | 63.9 |
| BABEL-1 | 20 | 2 | 23.3 | 43.1 / 10.8 | 34.7 / 15.4 | 49.7 | 51.3 | 65.2 |
| BABEL-1 | 40 | 2 | 19.7 | 42.8 / 15.0 | 37.9 / 16.8 | 52.4 | 55.9 | 64.8 |
| BABEL-1 | 80 | 2 | 15.7 | 42.3 / 12.0 | 34.5 / 16.7 | 54.8 | 59.2 | 65.3 |

**Results.** Increasing K improves code-probe accuracy overall but does not close the latent gap. Geometry merging improves MoF by approximately 8–10 points on LARa/BABEL-1, while damaging HuGaDB. Geometry-merge JSD rises to 93.9–94.3 on HuGaDB, 76.6–79.8 on LARa and 83.7–85.7 on BABEL-1, compared with baselines 89.5, 74.0 and 81.4. LARa's high-K geometry merges also reduce F1@50; BABEL-1 F1 differences do not meet the recorded rule. Temporal merging is better suited to HuGaDB, with scattered gains; BABEL-1 K=40 is an exception to any claim of “no temporal-merge MoF gain.”

**Control result.** Raw-feature k-means has three seeds at every K. SMQ has a clear probe advantage throughout LARa, including 57.2 versus 46.8 at K=128. On HuGaDB the advantage exists at smaller K, but not at K=80/160 under the recorded rule. On BABEL-1 no SMQ advantage is detected, and k-means wins at K=80. This supports a dataset-dependent learned-representation advantage, not the blanket conclusion that learning helps only on LARa at every vocabulary size.

**Implementation assessment.** The geometry distance matches the SMQ quantizer's sum of per-frame Euclidean distances. Both merges are label-free before benchmark Hungarian scoring, and all eight K=C identity checks pass. Extra codes map to distinct dummy labels in the unmerged bijection; the falling score is an expected protocol penalty, not evidence that representations necessarily deteriorate. Evaluation at K>C is defined even without merging, but is restrictive.

The k-means comparison is useful but not a perfectly isolated encoder ablation: raw channels are globally standardized, k-means optimizes flattened-patch Euclidean distortion, and its last patch uses edge padding, whereas SMQ quantizes zero-padded latent patches with a different distance. Global normalization and clustering are transductive. A matched preprocessing/distance control and clustering of frozen learned latents would better separate representation learning from clustering mechanics. Selecting a merge using benchmark scores would add label-dependent model selection, even though the merge algorithm itself is unsupervised.

**Analysis.** Larger vocabulary helps preserve class information; usable segmentation depends on how codes are grouped. No single merge delivers a consistent joint benefit across these datasets. Two-seed comparisons and multiple conditions make isolated wins provisional. A deterministic merge cannot create boundaries relative to its own unmerged code stream, so the final report's proposed fragmentation mechanism is untested; comparisons against separately trained K=C models require direct segment-count analysis.

### 6.1 Threshold/churn check

**Question.** Is a fixed replacement threshold of 10 compatible with a large vocabulary?

| Dataset, K=16C | Perplexity, threshold 10 → 1 | Unused codes, 10 → 1 | Code probe, 10 → 1 | Geometry MoF, 10 → 1 |
|---|---|---|---|---|
| HuGaDB, K=160 | 18.3 → 73.9 | 110 → 17 | 57.0 → 60.7 | 40.8 → 33.1 |
| LARa, K=128 | 53.4 → 88.8 | 50 → 0 | 53.3 → 57.1 | 38.3 → 46.8 |
| BABEL-1, K=80 | 7.1 → 44.2 | 69 → 1 | 46.6 → 54.9 | 34.0 → 42.5 |

**Assessment and analysis.** One nominally matched seed per dataset shows substantial loss of vocabulary utilization with threshold 10. It also changes merge performance in opposite directions across datasets. This supports threshold sensitivity and the rationale for scaling; it does not prove that this exact schedule is necessary or optimal. There are no training-time replacement-count trajectories establishing churn directly. Use a replicated K × threshold design before attributing all gains to K.

## 7. T2 — temporal-consistency objective, not run

**Question and planned implementation.** Add symmetric cross-entropy between adjacent patches' soft code assignments plus `log(K)−H(mean assignment)` as a diversity term. The codebook is detached for this loss so gradients reach the encoder; EMA remains the codebook update. Compare baseline, λ₁, 10λ₁, and zero reconstruction weight with λ₁. Commitment loss remains active in the last condition, so “consistency-only” is shorthand, not the entire objective. Source: [motion_quantizer.py](../src/model/motion_quantizer.py).

| Dataset | Final baseline commitment loss, reported | λ₁ = loss / log C |
|---|---:|---:|
| HuGaDB | 17,151.6 | 7,448.8 |
| LARa | 5.614 | 2.700 |
| BABEL-1 | 2.815 | 1.749 |

**Observed outcome.** No T2 treatment JSONs exist. The final report records that it was skipped because native input/loss scales differed greatly and the proposed weight was judged difficult to interpret. This is a design decision, not a null experimental result.

**Implementation and reasoning issues.** A large difference in λ across datasets is not alone proof that loss-scale matching fails: adapting λ to different magnitudes is its stated purpose. Loss values do not establish relative gradient influence. No treatment runs or gradient measurements demonstrate instability or nonportability of this rule.

The diversity term also does **not guarantee prevention of collapse**. If every patch has the same assignment distribution q, the smoothness term is H(q), while the diversity term is log K−H(q); their sum is log K for every constant q. Uniform assignments and a fully collapsed assignment therefore have the same limiting objective value, apart from numerical epsilon. A temporally consistent, balanced partition can do better, but optimization is not guaranteed to find it.

**Analysis.** The objective-level research question remains open. Before running it, verify gradient scales, finite gradients near zero distances, and behavior on constant-assignment examples. Any input-normalized variant needs its own matching baseline. A revised preregistration can document these decisions without treating an unrun experiment as evidence of failure.

## 8. T3 — HMM and HSMM duration-aware decoding

**Question.** Can temporal transition and explicit-duration models improve the existing codes without retraining the encoder?

**Design and correctness.** Emissions are log-softmax of negative standardized patch-to-code distance at β∈{0.5,1,2,4}. The HMM estimates transitions including self-loops from argmax codes. The HSMM fits shifted negative-binomial durations (Poisson fallback when underdispersed), between-state transitions, and three hard-EM decode/refit rounds with a 100-patch maximum. These operations use predicted codes, not action GT; fitting and evaluation use the same collection. Source: [t3_hsmm.py](../script/exp/t3_hsmm.py), [T3 results](exp/analysis/T3/).

The dynamic-programming recurrence passed **40 new exhaustive small-case comparisons** in this audit. This validates the decoder recurrence for those cases, not the statistical suitability of fitted durations or emissions. Clipping long observed durations, pooled fallback for rare states, uniform initial state probabilities, and untreated sequence-edge censoring remain modeling assumptions.

Representative sharp-emission results, β=4; entries are mean paired decoder-minus-argmax differences ± sample SD across retrained checkpoints:

| Dataset | n | Decoder | ΔMoF | ΔEdit | ΔF1@50 | ΔJSD |
|---|---:|---|---:|---:|---:|---:|
| HuGaDB | 3 | HMM | +2.1 ± 0.8 | −2.6 ± 1.7 | +1.3 ± 0.7 | +1.5 ± 1.3 |
| HuGaDB | 3 | HSMM | +2.2 ± 0.9 | −2.3 ± 2.7 | +2.2 ± 0.2 | +1.1 ± 1.1 |
| LARa | 3 | HMM | +2.8 ± 0.2 | −4.2 ± 0.6 | +1.9 ± 0.4 | +3.3 ± 0.4 |
| LARa | 3 | HSMM | +2.4 ± 0.3 | −10.0 ± 1.5 | −0.2 ± 0.4 | +5.3 ± 0.6 |
| BABEL-1 | 2 | HMM | +0.7 ± 0.4 | −2.8 ± 3.2 | −1.5 ± 2.1 | +2.9 ± 0.6 |
| BABEL-1 | 2 | HSMM | +0.6 ± 1.0 | −0.9 ± 0.1 | −0.6 ± 0.3 | +1.6 ± 2.1 |

All four β settings and both checkpoint sources are retained in [the complete T3 tables](exp/SUMMARY.md#t3--duration-aware-decoding-post-hoc). Low β strongly undersegments. At β=0.5, retrained HMM/HSMM F1@50 changes are −17.5/−19.9 on HuGaDB and −14.6/−14.6 on LARa; JSD rises approximately 9–10 and 25–26 respectively. HuGaDB's released HSMM fitted mean duration grows to about 29 patches through refitting. BABEL-1's released checkpoint loses MoF at every decoder setting, whereas retrained checkpoints gain MoF, including +5.9 for HMM β=0.5 while segment metrics worsen.

**Analysis and limits.** This family of fitted decoders does not provide a consistent joint segmentation improvement. LARa HMM β=4 has a recorded F1 benefit but worse Edit/JSD. MoF can improve while segments deteriorate, especially under dominant-class smoothing. Such an improvement is not inherently suspicious and does not contradict a non-additive oracle analysis. The displayed deltas are paired, while the verdict rule compares unpaired score-level bands; this can hide consistent within-checkpoint improvements. A future paired analysis needs more seeds and a declared comparison policy. Failure of these emissions and hard-EM fits does not rule out duration modeling in general.

## 9. T4 — patch duration versus frame rate

**Question and design.** Separately vary the temporal unit and the sampling density. Sweep P uses 0.5/1/2-second patches at native fps. Sweep R keeps a one-second patch and strides both features and GT: HuGaDB 30/15 fps, LARa 25/10 fps, BABEL-1 15/10 fps. Epochs and nominal optimizer steps are held fixed. JSD rescales run lengths back to native-frame units; duration readouts are in seconds. Sources: [prep_rates.py](../script/exp/prep_rates.py), [T4 JSONs](exp/analysis/T4/).

Means are shown below; each new condition has n=3. The reused native baseline has n=3 for HuGaDB/LARa and n=2 for BABEL-1. Full SDs are in [SUMMARY.md](exp/SUMMARY.md).

| Dataset | Condition | MoF | Edit | F1@50 | JSD | Mean predicted duration, s | Grid-oracle MoF |
|---|---|---:|---:|---:|---:|---:|---:|
| HuGaDB | 0.5 s, 60 fps | 40.3 | 22.1 | 9.3 | 95.1 | 1.05 | 97.8 |
| HuGaDB | 1 s, 60 fps | 40.7 | 32.8 | 22.0 | 89.5 | 2.97 | 95.5 |
| HuGaDB | 2 s, 60 fps | 42.6 | 39.1 | 26.1 | 91.6 | 4.89 | 90.9 |
| HuGaDB | 1 s, 30 fps | 42.3 | 36.0 | 24.0 | 88.6 | 3.17 | 95.5 |
| HuGaDB | 1 s, 15 fps | 41.0 | 36.6 | 24.5 | 88.2 | 3.75 | 95.5 |
| LARa | 0.5 s, 50 fps | 35.5 | 35.0 | 13.6 | 63.0 | 1.98 | 96.9 |
| LARa | 1 s, 50 fps | 36.8 | 38.8 | 15.8 | 74.0 | 2.84 | 93.8 |
| LARa | 2 s, 50 fps | 30.7 | 35.2 | 14.5 | 84.5 | 4.26 | 87.5 |
| LARa | 1 s, 25 fps | 35.6 | 39.1 | 16.1 | 73.7 | 2.68 | 93.8 |
| LARa | 1 s, 10 fps | 33.2 | 38.5 | 15.9 | 73.7 | 2.84 | 93.8 |
| BABEL-1 | 0.5 s, 30 fps | 34.8 | 29.0 | 12.3 | 81.9 | 2.62 | 94.9 |
| BABEL-1 | 1 s, 30 fps | 34.5 | 28.2 | 14.7 | 81.4 | 2.97 | 90.2 |
| BABEL-1 | 2 s, 30 fps | 32.0 | 30.7 | 14.3 | 83.5 | 3.20 | 82.6 |
| BABEL-1 | 1 s, 15 fps | 32.2 | 30.1 | 13.3 | 78.6 | 2.66 | 90.2 |
| BABEL-1 | 1 s, 10 fps | 32.7 | 28.9 | 15.2 | 77.4 | 2.53 | 90.2 |

**Results and analysis.** HuGaDB duration responds strongly to patch size: 1.05→2.97→4.89 seconds, against GT mean 4.96. LARa responds 1.98→2.84→4.26 seconds, against GT 3.73. BABEL-1 is much flatter, 2.62→2.97→3.20 seconds, against GT 1.91. HuGaDB's 2-second patches improve F1@50 while its oracle frame ceiling falls; LARa's half-second patches improve JSD by approximately 11 points while worsening F1@50. These outcomes separate duration-distribution fit from segmentation quality.

**Correctness and limitations.** Feature/GT striding is aligned, patch durations remain one second in Sweep R, and JSD length rescaling is implemented. However:

- Downsampling uses plain stride without anti-alias filtering, and segmentation metrics are evaluated on downsampled GT. This changes both input bandwidth and the evaluation discretization; native-frame evaluation would be a useful additional comparison.
- Changing patch size also changes codebook parameter count, numbers of training patches, and EMA counts relative to the fixed replacement threshold. Matching epochs/steps does not isolate physical unit duration alone.
- “MoF is unaffected by either” is false for LARa under the report's own rule: 2-second patches give 30.7±3.5 versus 36.8±1.5; 10 fps gives 33.2±2.0. Both are marked decreases.
- “Indifferent to frame rate” is too strong: HuGaDB mean predicted duration rises from 2.97 to 3.75 seconds. No duration-equivalence test was reported.

The evidence supports **sensitivity to patch duration, strongest on HuGaDB**, not a universal conclusion that the discovered behavioral unit is exactly the patch. The weaker BABEL-1 response remains unresolved.

## 10. Implementation and reporting issues to address

| Priority | Issue | Required action | Affected conclusion |
|---|---|---|---|
| High | D2 excludes empty-boundary cases; D6 allows duplicate matches | Count all sequences; implement one-to-one boundary matching and recompute | Boundary precision/recall and “little headroom” |
| High | Oracle relabeling can merge segments; interventions are asymmetric | Describe them as assisted interventions; evaluate explicit fixed-boundary representations if needed | Pure label-versus-boundary decomposition |
| High | HuGaDB many-to-one segment improvement omitted | Use saved full-frame values: F1@50 24.30→34.94 | Claim that mapping only improves MoF |
| High | T1/threshold conclusions rely on two/one seeds | Complete missing preregistered seeds, then replicate threshold conditions | Robustness of codebook/merge gains |
| High before T2 | Diversity loss does not guarantee anti-collapse; weight rationale unvalidated | Check constant-assignment behavior and relative gradients; preregister revised design | Any future objective-level conclusion |
| Medium | JSD references described as bounds and unique reconstruction | Rename empirical references and qualify convention identification | Metric validity and cross-paper comparisons |
| Medium | Geometric law asserted from persistence | Distinguish observed/fitted means; test the distribution directly | Rationale for duration models |
| Medium | k-means control differs in scaling, padding and metric | Add matched preprocessing/distance and frozen-latent controls | Attribution of benefit to representation learning |
| Medium | T4 changes more than temporal units | Control patch exposure/EMA and add filtered/native-frame comparisons | Causal interpretation of unit duration |
| Medium | Descriptive rule presented as effect significance; promised Welch output absent | Report rule as descriptive, preserve n, predefine paired comparisons and uncertainty | Reliability of isolated gains/nulls |
| Medium | Training revision and E0 microbatch history incompletely recorded | Archive code/config/checkpoint hashes and environment; rerun E0 baseline if exact training provenance is needed | Reproducibility beyond cached scoring |
| Low | Summary header says T3 n=2 and T4 all n=3 | State T3 n=3/3/2; T4 baseline n=3/3/2; update stale summarizer docstring | Accurate experiment inventory |

The memory changes—distance evaluation without autograd, indexed codebook lookup, chunked distances, and activation checkpointing—are consistent with the intended computation by source inspection. The final report describes prior bit-equivalence tests, but this audit did not repeat those training tests. Default-off T2 does not affect the completed runs. The original evaluation protocol intentionally retains its original behavior; corrected diagnostics should be versioned separately so reproduction scores remain traceable.

## 11. Earlier work referenced by this research sequence

These artifacts predate the requested reproduction-to-final-report sequence. They are recorded here to avoid mistaking them for missing T1–T4 results. They were inspected as background, not independently rerun or fully audited.

| Earlier work | Question and available result | Relationship to this report |
|---|---|---|
| E1 parts A/B | GT duration statistics and an earlier duration-metric decomposition; [earlier summary](../figures/e1_part_b/summary.csv) reports JSD 89.54/83.52/82.00 for HuGaDB/LARa/BABEL-1 | Different checkpoints/conventions; do not pool with audited E1. Earlier HuGaDB claim of no duration headroom is not established by those histogram references. |
| E2 split preparation | [e2_prepare_splits.py](../script/e2_prepare_splits.py) prepares subject splits | Preparation is not evidence that a held-out-subject training study was completed; no such result is established by the four main reports. |
| E3 subject decodability | Saved summaries give subject prediction from code histograms 21.70% versus action-histogram baseline 8.52% on HuGaDB; 30.52% versus 18.89% on LARa | Suggests subject-associated structure; scale-normalized retraining control explicitly not run. Does not prove an identity-specific causal mechanism. [Artifacts](../figures/e3_subject_decodability/) |
| E4 preliminary sweep | Single-seed patch/rate sweep scored with local Hungarian | Superseded for published-protocol comparisons by T4. [Earlier table](../figures/e4_sweeps/summary.csv) |
| E5.0 feasibility gate | Median codes per GT segment: 4 on HuGaDB, 3 on LARa; 51.3%/72.1% have fewer than five; both recorded gates fail | A feasibility result, not completion of a subsequent segment-representation intervention. [Gate table](../figures/e5_0_gate/gate_summary.csv) |

## 12. Overall conclusion and next research steps

The strongest supported conclusion is that **the learned code partition and its correspondence to action classes are major limitations**. The grid oracle and continuous-versus-discrete probes support that diagnosis through different measurements. Larger K partially improves action decoding, but the remaining latent gap, raw-feature controls and sensitivity to merging show that vocabulary size alone is not a complete remedy.

Boundary quality remains imperfect, and the current diagnostics cannot justify deprioritizing it categorically. JSD reveals duration bias but is insufficient for ranking segmentations. Smoothing and the tested HMM/HSMM decoders yield tradeoffs rather than a general solution. Patch-duration manipulation strongly affects some datasets; sampling and patch-exposure confounds limit the causal claim. T2 supplies no evidence either way because it was not run.

The next defensible sequence is: first correct and regenerate the boundary diagnostics and reporting errors; then complete missing replication and isolate threshold/control differences; then run a revised, instrumented T2 or a prespecified learned hierarchical merge. Evaluate any new method jointly on MoF, Edit, F1, duration statistics and label-free model selection. A hierarchical codebook is a motivated hypothesis, not an outcome demonstrated by the experiments here.

## 13. Audit performed for this report and evidence index

This compilation added a reproducible CPU audit, [audit_consolidated_report.py](../script/repro/audit_consolidated_report.py), and its [machine-readable output](CONSOLIDATED_AUDIT.json). It:

- independently aggregated **459** rows of `summary_long.csv` from per-run JSONs, finding **zero mismatches** in counts, means or sample SDs;
- rescored cached predictions on **all five datasets**, recovering saved global MoF/Edit/F1 and JSD, and verified the full-frame many-to-one scores on HuGaDB/LARa/BABEL-1;
- checked all **eight K=C baseline artifacts** for identical merged and unmerged scores;
- inspected **71 training configurations**, confirming 30 epochs, no microbatching and no temporal-consistency treatment;
- reproduced the duplicate-boundary counterexample and counted boundaries omitted by D2;
- verified the HSMM recurrence against exhaustive enumeration on **40** new random small problems;
- recalculated E0 seed statistics directly from saved JSONs.

The audit validates saved numerical evidence and selected implementation properties. It is not a new training replication, a rerun of all probes/controls, or certification of every model modification. The experiments and original result files were not changed.

Run from the repository root in PowerShell:

```powershell
& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/repro/audit_consolidated_report.py
```

| Evidence | Location |
|---|---|
| Initial hypotheses and diagnostics | [REPORT_reproduction_and_diagnosis.md](REPORT_reproduction_and_diagnosis.md) |
| Prespecified design and decision rule | [PREREG_T1-T4.md](PREREG_T1-T4.md) |
| Original final interpretation and deviations | [FINAL_REPORT.md](FINAL_REPORT.md) |
| Complete T1–T4 generated tables | [exp/SUMMARY.md](exp/SUMMARY.md) |
| Aggregated numeric data | [exp/summary_long.csv](exp/summary_long.csv) |
| E0 metrics and cached predictions | [e0/](e0/), [preds/](preds/) |
| JSD scores and convention sweep | [e1_jsd/](e1_jsd/) |
| Diagnostic results | [diagnostics/](diagnostics/) |
| Individual experiment analyses | [exp/analysis/](exp/analysis/) |
| Run configurations and training losses | [models/exp/](../models/exp/) |
| Audit evidence generated for this report | [CONSOLIDATED_AUDIT.json](CONSOLIDATED_AUDIT.json) |
