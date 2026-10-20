# What do SMQ's discrete codes represent?

**Characterization completed: 21 September 2026.** Frozen one-second K=C baselines, three seeds, HuGaDB and LARa. No model training or downstream experiments were performed.

**Verdict:** this study does **not establish reproducible local movement categories beyond static posture**. On LARa, the frequency-weighted posture-controlled movement advantage changes sign across seeds and the fixed matching rules retain fewer than 0.7% of anchor candidates. The across-action subset is smaller still. This is an inconclusive characterization of movement reuse, not evidence that the codes are meaningless or exclusively posture-based. Assignments are highly stable under the tested small noise perturbations. HuGaDB's skeletal-posture question is untestable from its IMU-only inputs; as requested, it contributes stability and signal exemplars only.

The main limitation is identifiable: there is very little matched support for comparing different codes at nearly identical posture within the same donor recording and across people. The available observations cannot distinguish absence of movement information from the restrictions of this matched population.

## 1. Frozen experimental unit and exposure

The [protocol](PROTOCOL.md) was saved before grouping outcomes. The [manifest](manifest.json) records its SHA256, all six checkpoint/export hashes, exact model configurations, and participant splits. Models are the T1 epoch-30 baselines at seeds 1538574472, 111 and 222: LARa K=8/W=50, HuGaDB K=10/W=60. Released weights, subject-split weights and enlarged codebooks were not substituted after seeing results.

The encoder has two stages with kernel-3 convolutions at dilations 1,2,4: a **29-frame receptive field, radius 14**. Patch p represents native preprocessed rows [pW,(p+1)W); code assignment additionally depends on 14 rows of context on each side. Only complete odd-indexed patches with complete context are eligible, so even the assignment-dependency intervals of successive eligible patches do not overlap. LARa's original preprocessing samples every fourth 200-Hz row; native index t corresponds to original row 4t. HuGaDB uses the pipeline's nominal 60 Hz.

All selected SMQ models trained on all native recordings. The calibration/evaluation split here is **unseen only to descriptor scaling and posture k-means**, not to SMQ. It cannot demonstrate generalization to participants unseen during unsupervised training.

| Dataset | Recordings | Calibration people | Evaluation people | Calibration patches | Evaluation patches | Descriptor scope |
|---|---|---|---|---|---|---|
| lara | 439 | 5 | 11 | 7257 | 18644 | hip-relative xyz |
| hugadb | 364 | 6 | 12 | 3473 | 5790 | IMU signals; no joint posture |

LARa calibration participants: 12, 06, 13, 01, 11. Primary participant dyads: 08/04, 02/09, 07/03, 10/14, 16/05. Participant 15 remains in stability/exemplars but is unpaired for primary inference. These dyads contain 16,874 eligible patches; sampling at most 12 evenly spaced anchors per recording gives **3,432 anchor candidates** per seed and analysis.

LARa positional channels 3:6 are root-relative to joint 21, verified zero at the root throughout the data. The first three channels are preserved in model inputs; their exact physical meaning is not inferred from the variable name in preprocessing. No rotations, mirroring or input normalization changes were introduced. Native coordinate units are retained without assuming millimeters. HuGaDB contains six IMU channels at six sensor sites, not joint positions. One extreme sensor value at recording `HuGaDB_v2_various_10_08`, frame 21, affects patch 0; that patch is excluded by the prespecified complete-context/odd-patch rule, including from descriptor scaling. Model inputs/checkpoints were not repaired.

## 2. Measurement and implementation checks

LARa posture is the patch-mean configuration of the 21 non-root joints. Coordinate scaling is fitted to calibration patch means, with a floor of 5% of their median nonzero SD. Ordered displacement is position(t)−position(0); forward-difference velocity retains sign and time order. Both use the same coordinate scales. The two fixed-length RMS distances are normalized by calibration cross-person median distances: **0.801269** for displacement and **2.751475** for velocity. The primary composite is their equal-weighted mean. Distances and differences are dimensionless relative to these declared scales; raw standardized velocity has units of scaled coordinate per second.

The sensitivity distance is forward-only endpoint-constrained DTW with a ±5-frame band (0.1 s), retaining joint identity, displacement direction and velocity. It permits modest local speed variation, not time reversal. It minimizes summed local squared costs and reports the square root of their mean on that path; it is not an exact minimum-average-path objective.

Controlled examples were checked before grouping: all have the same mean posture. Identical motion scores zero. Opposite direction gives composite **1.033** versus **0.068** for modest speed variation; corresponding DTW distances are **1.072** and **0.081**. Each displacement and velocity component has the same ordering. These checks establish the required basic discrimination, not validation against human judgments. [Metric-check values](metric_checks.json).

Stability inference encodes the complete raw context and quantizes only its central represented patch, avoiding a shifted quantizer-grid origin. **9,927** clean crop/checkpoint assignments match the original full-recording exports exactly. Common raw frames under temporal shift have exactly equal coordinates/signals and recomputed aligned descriptors. This directly validates the practical receptive-field/crop alignment used here.

## 3. Primary results: movement after posture and participant controls

For an anchor, the same-code donor must come from the other participant of its fixed dyad and lie within posture RMS 0.50. The different-code control must come from the **same donor recording**, lie within 0.25 of that donor and 0.50 of the anchor, and match the anchor–donor posture distance within 0.05. No patch is reused within an analysis. If the nearest same-code donor has no valid control, the anchor is unmatched; calipers were not relaxed. The action-restricted analysis requires both donor and control majority labels to differ from the anchor's.

**Effect = control movement distance − same-code movement distance. Positive favors same-code movement similarity.** The following is the primary table for both datasets. Numbers are frequency-weighted within seed. Intervals resample entire participant dyads, retaining dependence among all their recordings and both pair endpoints; they are separate from model-seed variability.

| Dataset | Grouping | Seed | Action restriction | Matched / 3,432 anchors | Dyads with matches | Same-code D | Control D | Difference [95% cluster interval] |
|---|---|---|---|---|---|---|---|---|
| LARa | SMQ | 1538574472 | all | 18 (0.52%) | 3 | 0.811 | 0.765 | -0.046 [-0.467, 0.072] |
| LARa | SMQ | 111 | all | 22 (0.64%) | 3 | 0.649 | 0.705 | 0.056 [-0.016, 0.083] |
| LARa | SMQ | 222 | all | 21 (0.61%) | 3 | 0.699 | 0.757 | 0.059 [-0.036, 0.312] |
| LARa | SMQ | 1538574472 | different actions | 8 (0.23%) | 3 | 1.209 | 1.168 | -0.042 [-0.113, 0.327] |
| LARa | SMQ | 111 | different actions | 12 (0.35%) | 2 | 0.926 | 1.006 | 0.080 [0.071, 0.083] |
| LARa | SMQ | 222 | different actions | 13 (0.38%) | 3 | 0.831 | 0.963 | 0.132 [-0.001, 0.320] |
| LARa | Posture k-means | 1538574472 | all | 10 (0.29%) | 3 | 0.469 | 0.463 | -0.006 [-0.136, 0.313] |
| LARa | Posture k-means | 111 | all | 10 (0.29%) | 3 | 0.554 | 0.477 | -0.077 [-0.183, -0.064] |
| LARa | Posture k-means | 222 | all | 7 (0.20%) | 3 | 0.758 | 0.689 | -0.069 [-0.136, -0.021] |
| LARa | Posture k-means | 1538574472 | different actions | 3 (0.09%) | 2 | 0.749 | 0.713 | -0.036 [-0.211, 0.313] |
| LARa | Posture k-means | 111 | different actions | 9 (0.26%) | 3 | 0.779 | 0.839 | 0.060 [-0.069, 0.212] |
| LARa | Posture k-means | 222 | different actions | 5 (0.15%) | 2 | 0.848 | 0.774 | -0.075 [-0.109, -0.024] |
| HuGaDB | SMQ | all 3 | — | unavailable | — | — | — | No skeletal posture/trajectory observations |

Only **three of five dyads** contribute to the all-action SMQ estimates, and two or three to the across-action estimates. With so few informative clusters, percentile intervals are descriptive and fragile. For example, seed 111's positive across-action interval comes from only 12 triples in two dyads; it is not robust evidence of reusable movement categories. Empty bootstrap draws are excluded and their count is recorded. Seeds reuse the evaluation collection, so their observations must not be pooled as independent replications.

Static-posture k-means is fitted only on calibration posture, separately with the same three seeds and K=8. It has only 7–10 matched triples without action restrictions and 3–9 across actions. Its residual effects also vary, particularly in the across-action subset. Different groupings admit different matched populations, so directly subtracting the two methods' aggregate effects would not be a controlled test of learned movement information.

### Component distances and temporal-alignment sensitivity

Each cell below is same-code / control / paired difference; displacement and velocity are standardized-coordinate RMS distances before their calibration median normalization. The last column gives the normalized DTW difference and its cluster interval. Both components and every control result are available in [primary_summary.csv](primary_summary.csv).

| Seed | Subset | Displacement: same / control / Δ | Velocity: same / control / Δ | DTW Δ [95% interval] |
|---|---|---|---|---|
| 1538574472 | all | 0.663 / 0.589 / -0.075 | 2.184 / 2.188 / 0.004 | -0.048 [-0.713, 0.064] |
| 111 | all | 0.534 / 0.585 / 0.051 | 1.736 / 1.867 / 0.131 | 0.055 [-0.021, 0.082] |
| 222 | all | 0.546 / 0.582 / 0.036 | 1.970 / 2.169 / 0.199 | 0.061 [-0.044, 0.350] |
| 1538574472 | different actions | 0.919 / 0.850 / -0.069 | 3.498 / 3.506 / 0.009 | -0.072 [-0.199, 0.325] |
| 111 | different actions | 0.747 / 0.828 / 0.081 | 2.530 / 2.691 / 0.161 | 0.075 [0.066, 0.078] |
| 222 | different actions | 0.630 / 0.740 / 0.110 | 2.410 / 2.757 / 0.347 | 0.137 [-0.010, 0.319] |

### Frequency weighting, code balance and seed variability

Frequency weighting gives each accepted anchor triple equal weight; code balance gives each represented code equal weight. This is frequency in the **matched sample**, not inverse-probability extrapolation to all native patches. Missing codes cannot receive an estimable movement effect.

| Grouping | Subset | Weighting | Effects: default / 111 / 222 | Across-seed mean ± SD |
|---|---|---|---|---|
| smq | all_actions | frequency | -0.046 / 0.056 / 0.059 | 0.023 ± 0.059 |
| smq | all_actions | code_balanced | -0.032 / 0.047 / 0.038 | 0.018 ± 0.044 |
| smq | different_actions | frequency | -0.042 / 0.080 / 0.132 | 0.057 ± 0.089 |
| smq | different_actions | code_balanced | 0.010 / 0.156 / 0.131 | 0.099 ± 0.078 |
| posture_kmeans | all_actions | frequency | -0.006 / -0.077 / -0.069 | -0.051 ± 0.039 |
| posture_kmeans | all_actions | code_balanced | -0.051 / -0.110 / -0.109 | -0.090 ± 0.034 |
| posture_kmeans | different_actions | frequency | -0.036 / 0.060 / -0.075 | -0.017 ± 0.069 |
| posture_kmeans | different_actions | code_balanced | -0.036 / 0.105 / -0.104 | -0.012 ± 0.107 |

These SDs describe three fixed model/control fits. They are not uncertainty across evaluation people, which is reported in the cluster intervals. The code-balanced across-action point estimate is positive in all three seeds; that is a weak favorable signal worth recording. It covers only 3–5 represented codes and 2–3 informative dyads per seed, while frequency-weighted estimates change sign. It therefore does not establish a robust population-level movement-reuse effect.

### Coverage and matching quality

Across SMQ seeds, about 1,995–2,004 anchors fail the same-code posture caliper in the unrestricted analysis; another 1,403–1,412 have no different-code control meeting the fixed rules. These failure categories account for nearly all unmatched anchors. Native orientation/body configuration, fixed participant pairing and the same-recording requirement can all limit support. Low support does not show that the codes are wrong or posture-only.

| Seed | Subset | Triples | Recordings | Codes / 8 | Mean posture same | Mean posture control | Control−same posture | Mean minimum action purity |
|---|---|---|---|---|---|---|---|---|
| 1538574472 | all_actions | 18 | 27 | 7 | 0.429 | 0.433 | 0.004 | 0.872 |
| 1538574472 | different_actions | 8 | 16 | 3 | 0.442 | 0.446 | 0.004 | 0.895 |
| 111 | all_actions | 22 | 32 | 6 | 0.407 | 0.418 | 0.011 | 0.846 |
| 111 | different_actions | 12 | 21 | 5 | 0.396 | 0.410 | 0.014 | 0.743 |
| 222 | all_actions | 21 | 38 | 7 | 0.412 | 0.418 | 0.006 | 0.830 |
| 222 | different_actions | 13 | 23 | 5 | 0.425 | 0.427 | 0.001 | 0.868 |

Posture distance imbalance is small by the frozen absolute caliper but not identically zero; control posture is slightly farther on average in most cells. There is no causal adjustment for remaining within-caliper differences. Each reported triple includes its three raw intervals, subject/recording IDs, action labels/purity and all distances in [primary_pairs.csv](primary_pairs.csv). [Coverage and exclusion counts](primary_coverage.csv).

The across-action subset uses majority labels, including potentially `None` and `Synchronization`, so not every different-label triple is a clear transition across substantive action contexts. The following purely descriptive counts expose this limitation without changing the frozen analysis:

| Seed | Across-label triples | No None/Synchronization in any member | Also ≥80% label purity in all members |
|---|---|---|---|
| 1538574472 | 8 | 7 | 6 |
| 111 | 12 | 11 | 4 |
| 222 | 13 | 8 | 6 |

## 4. Assignment stability, conditional on actual descriptor change

At most six fixed eligible patches per evaluation recording were tested: **1,896 LARa patches / 316 recordings / 11 people**, and **1,413 HuGaDB patches / 243 recordings / 12 people**. Three HuGaDB patches lack complete +2-frame shifted context, leaving 1,410 for that condition. Pose/signal noise uses the same deterministic draw across model seeds. No Hungarian remapping of code IDs is used.

LARa Gaussian xyz noise is 0.1% or 0.5% of calibration RMS root-relative position, **316.285 native coordinate units**: input SDs about 0.316 and 1.581. Root subtraction preserves zero hip and leaves the first three input channels unchanged. It increases effective non-root noise; measured full-skeleton RMS relative to the scale is about 0.138%/0.690%. HuGaDB uses each native channel/sensor's calibration SD; it is signal noise, not pose noise. Gaussian additions are floating point even though HuGaDB files store integers. Measured amplitudes are recorded in the validation JSONs.

Agreement below is across-seed mean ± sample SD in percentage points. Descriptor changes are the mean over paired evaluation windows; they are seed-independent because inputs and perturbations are shared. HuGaDB quantities are signal-mean / signal-displacement / signal-derivative analogues, not skeletal measurements.

| Dataset | Perturbation | Patches/seed | Agreement %, mean ± seed SD | Posture/signal mean Δ | Displacement Δ | Velocity/derivative Δ | Composite descriptor Δ |
|---|---|---|---|---|---|---|---|
| lara | shift_2 | 1896 | 97.679 ± 0.105 | 0.028 | 0.090 | 1.072 | 0.251 |
| lara | noise_0.001 | 1896 | 100.000 ± 0.000 | 0.001 | 0.015 | 0.766 | 0.148 |
| lara | noise_0.005 | 1896 | 99.982 ± 0.030 | 0.007 | 0.074 | 3.824 | 0.741 |
| hugadb | shift_2 | 1410 | 91.868 ± 1.046 | 0.182 | 4.996 | 226.504 | 1.079 |
| hugadb | noise_0.001 | 1413 | 100.000 ± 0.000 | 0.001 | 0.006 | 0.380 | 0.002 |
| hugadb | noise_0.005 | 1413 | 99.976 ± 0.041 | 0.003 | 0.031 | 1.900 | 0.008 |

A +2-frame shift is 40 ms on LARa and nominally 33.3 ms on HuGaDB. Full relative-time windows have changed content and phase, whereas their explicitly aligned shared raw frames agree exactly. HuGaDB's composite change is about **1.079 calibration-distance units**, so its approximately 8% assignment changes do not establish instability under negligible motion change. LARa's 0.5% pose noise causes a large velocity-descriptor change despite essentially unchanged code IDs; this shows resistance to that high-frequency perturbation, not proof of semantic invariance.

The tables below retain per-seed participant-cluster 95% intervals for shift agreement and show code-balanced shift agreement. Full noise intervals, descriptor intervals and per-code frequencies/agreements are exported. A bootstrap interval [100%,100%] when no flips were observed is an empirical resampling result, not proof of zero population error.

| Dataset | Seed | Frequency-weighted shift agreement % [95% interval] | Code-balanced agreement % [95% interval] |
|---|---|---|---|
| lara | 1538574472 | 97.68 [97.02, 98.39] | 97.58 [96.64, 98.48] |
| lara | 111 | 97.57 [96.63, 98.40] | 97.61 [96.49, 98.55] |
| lara | 222 | 97.78 [97.08, 98.40] | 97.01 [95.43, 98.20] |
| hugadb | 1538574472 | 91.49 [89.76, 93.10] | 87.54 [85.98, 89.22] |
| hugadb | 111 | 93.05 [91.32, 94.67] | 90.25 [88.24, 92.05] |
| hugadb | 222 | 91.06 [89.41, 92.97] | 86.64 [82.83, 89.04] |

## 5. Movement and signal exemplars without outcome selection

[Open the montage index](../../figures/characterization/exemplars/index.html). It contains **54 montages**, one for every code of every checkpoint, with **324 full-patch occurrences**. Every montage includes six distinct evaluation participants; none of the 54 codes is absent. Available evaluation-patch counts range from 97 to 6,821 across codebooks. SHA256 ordering first maximizes participant diversity, then recording diversity. [Selection, counts and exact frame provenance](exemplars.csv).

LARa plots all 21 non-root joint tracks through the complete second, with purple-to-yellow time, start/end markers, and signed displacement heatmaps. They preserve the native coordinate system and do not infer missing skeleton connectivity. HuGaDB plots all 36 complete ordered sensor traces, with calibration-only standardization explicitly restricted to display. Near-flat traces below mean near-flat at that shared display scale, not literally zero motion.

The fixed qualitative subset is code 0, the most frequent code, and the least frequent nonempty code in the default-seed model (deduplicated). It was chosen by the protocol, not by visual coherence:

| Dataset/code | Eligible patches | Observed common structure and variation |
|---|---|---|
| LARa 0 | 2120 | Mix of small changes and substantial articulated movement; no visually uniform direction. |
| LARa 6, most frequent | 5824 | Nearly stationary configurations and larger trajectories coexist; no single obvious transition. |
| LARa 3, least frequent | 617 | Some nearly stationary examples, others strong arcs; opposite signed displacement components appear across people. |
| HuGaDB 3, most frequent | 1755 | All six selected occurrences have nearly flat sensor traces at the shared scale. |
| HuGaDB 0, least frequent | 340 | Varied transients and one nearly flat example; no uniform waveform apparent. |

These descriptions are qualitative evidence of heterogeneous occurrences, not semantic labels or a complete assessment of within-code consistency. They neither validate a recurring primitive nor prove its absence. A code number in one seed has no established correspondence to the same number in another seed.

## 6. Relative support for the three explanations

The explanations are not mutually exclusive:

| Explanation | Evidence in this run | Supported conclusion |
|---|---|---|
| Movement-sensitive representation | SMQ residual movement effects positive for two seeds and negative for one; very low matching support; DTW does not supply uniform positive evidence. | Reproducible cross-person movement categories beyond posture are not established. |
| Posture-sensitive representation | Tight same-recording different-code posture controls are rare; static-posture control also has low coverage; examples mix static configurations and dynamics. | Posture remains a plausible contributor, but this run cannot show that posture is the dominant or sole organizing variable. |
| Nuisance-sensitive or unstable partition | Code agreement is near100% under declared small noise; temporal shifts change HuGaDB signal descriptors substantially; all subjects were seen by SMQ. | Strong small-noise assignment instability is not supported. Participant/recording dependence and genuinely unseen-person reproducibility remain unresolved. |

The primary contrast controls participant and recording identity between its two alternatives by design. It does not estimate all recording-specific variation, and no direct within-person versus across-person matched effect was prespecified. The limited match yield is not itself a nuisance-dependence effect. Root-relative xyz controls also leave any information in the first three native LARa channels unmeasured.

**Supported:** the frozen models can be examined with exactly aligned raw patches; the chosen movement distances distinguish direction from modest speed differences on controlled examples; tested small-noise code agreement is high; eligible code occurrences are distributed across people; the strict matched comparison has very limited support and no consistently signed movement benefit across seeds.

**Unsupported:** that SMQ learns reusable local movement primitives, that it merely quantizes static posture, that poor action recognition makes its codes meaningless, that sharing codes across labels proves reuse, or that these results demonstrate variable-duration syllables, hierarchy, or unseen-subject generalization.

## 7. Audit, deviations and deliverables

The only substantive scope adjustment was the user-approved exclusion of HuGaDB's skeletal-posture primary test. Protocol calipers, model selection and sampling were not changed after outcomes. Technical implementation corrections materialized cached code arrays once for efficient reading and applied HuGaDB Gaussian noise in floating point; final stability outputs use the corrected implementation. Raw-to-float32 cache rounding on LARa was validated after the declared cast and recorded, not treated as a data mismatch.

The [artifact audit](audit.json) validates all **12** matched analyses, **138** triples across analyses (not mutually independent across seeds/subsets), all **96** primary summary rows, frozen checkpoint/protocol hashes, matching calipers, no within-analysis patch reuse, cross-participant and donor-recording constraints, action restrictions, 54 montage files and their participant coverage. It also records descriptive action-label quality counts. Both primary and stability intervals resample clusters rather than individual pairs. Only two or three primary dyads contain matches, so uncertainty estimates should not be used as strong hypothesis tests.

Reproduce from the repository root with the existing `smq` Python environment:

```powershell
$smqPython = 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe'
& $smqPython -B script/characterize/prepare.py
& $smqPython -B script/characterize/primary.py
& $smqPython -B script/characterize/stability.py --dataset lara
& $smqPython -B script/characterize/stability.py --dataset hugadb
& $smqPython -B script/characterize/exemplars.py
& $smqPython -B script/characterize/audit.py
& $smqPython -B script/characterize/make_report.py
```

[Protocol](PROTOCOL.md) · [Frozen manifest](manifest.json) · [Primary distances and intervals](primary_summary.csv) · [Coverage/exclusions](primary_coverage.csv) · [All per-code primary results](primary_per_code.csv) · [Matched triples](primary_pairs.csv) · [LARa stability](stability_lara_summary.csv) · [HuGaDB stability](stability_hugadb_summary.csv) · [LARa per-code stability](stability_lara_per_code.csv) · [HuGaDB per-code stability](stability_hugadb_per_code.csv) · [Montages](../../figures/characterization/exemplars/index.html).

This run stops at characterization. Resolving the unanswered question would require a separately designed comparison with adequate matched support and, for HuGaDB, actual pose observations; neither is supplied by declaring the present sparse effects decisive.

## Appendix: every LARa SMQ code, unrestricted-action primary comparison

Counts expose codes with no matched evidence. Intervals below are exploratory, especially with fewer than three informative dyads. The separate CSV includes the action-restricted subset, posture control, and both distance components for every code.

| Seed | Code | Evaluation occurrences | Candidate anchors | Matched triples | Informative dyads | Movement Δ [95% interval] |
|---|---|---|---|---|---|---|
| 1538574472 | 0 | 2120 | 434 | 3 | 1 | 0.095 [0.095, 0.095] |
| 1538574472 | 1 | 1060 | 187 | 1 | 1 | -0.124 [-0.124, -0.124] |
| 1538574472 | 2 | 902 | 168 | 0 | 0 | — [—, —] |
| 1538574472 | 3 | 617 | 176 | 1 | 1 | -0.016 [-0.016, -0.016] |
| 1538574472 | 4 | 5638 | 1086 | 3 | 2 | -0.079 [-0.283, 0.327] |
| 1538574472 | 5 | 1086 | 195 | 3 | 1 | 0.100 [0.100, 0.100] |
| 1538574472 | 6 | 5824 | 926 | 5 | 3 | -0.210 [-0.467, -0.096] |
| 1538574472 | 7 | 1397 | 260 | 2 | 1 | 0.009 [0.009, 0.009] |
| 111 | 0 | 2066 | 416 | 4 | 1 | 0.078 [0.078, 0.078] |
| 111 | 1 | 6821 | 1275 | 4 | 2 | -0.022 [-0.057, 0.083] |
| 111 | 2 | 1389 | 256 | 4 | 1 | 0.075 [0.075, 0.075] |
| 111 | 3 | 826 | 147 | 0 | 0 | — [—, —] |
| 111 | 4 | 1147 | 195 | 4 | 1 | 0.134 [0.134, 0.134] |
| 111 | 5 | 699 | 128 | 0 | 0 | — [—, —] |
| 111 | 6 | 5017 | 826 | 5 | 1 | 0.036 [0.036, 0.036] |
| 111 | 7 | 679 | 189 | 1 | 1 | -0.016 [-0.016, -0.016] |
| 222 | 0 | 550 | 165 | 1 | 1 | -0.016 [-0.016, -0.016] |
| 222 | 1 | 5953 | 1160 | 2 | 2 | 0.312 [0.297, 0.327] |
| 222 | 2 | 1748 | 357 | 4 | 2 | -0.059 [-0.113, 0.102] |
| 222 | 3 | 1215 | 236 | 2 | 1 | 0.009 [0.009, 0.009] |
| 222 | 4 | 6208 | 970 | 7 | 3 | 0.163 [0.053, 0.424] |
| 222 | 5 | 781 | 141 | 0 | 0 | — [—, —] |
| 222 | 6 | 980 | 173 | 2 | 1 | -0.119 [-0.119, -0.119] |
| 222 | 7 | 1209 | 230 | 3 | 1 | -0.021 [-0.021, -0.021] |

