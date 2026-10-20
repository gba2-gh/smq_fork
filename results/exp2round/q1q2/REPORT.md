# Q1/Q2: motion-unit information and readout diagnostics

## Ten-line summary

1. Both checkpoint/quantization gates passed; D7 reproduced 44.85% MoF on HuGaDB and 41.66% on LARa.
2. HuGaDB has 10 observed classes / 18 subjects / IMU inputs; LARa has 8 classes / 16 subjects / optical motion-capture inputs.
3. At full-patch windows, action MI rises from Ku=10 to 1000 on both datasets; subject MI and conditional subject MI rise as well.
4. HuGaDB full-patch action AMI falls across those endpoints, while LARa action AMI rises; raw MI and AMI describe different quantities.
5. Full-patch hard pooled MoF changes from 49.23 ± 0.36 to 34.95 ± 0.55 on HuGaDB, and from 28.79 ± 0.02 to 38.05 ± 2.55 on LARa.
6. Full-patch hard subject-grouped readout MoF changes from 41.69 ± 0.45 to 79.16 ± 0.20 on HuGaDB, and from 26.62 ± 0.59 to 50.56 ± 0.62 on LARa.
7. Soft assignment, duration weighting, window size and grouping change scores; all matched comparisons and temperature settings are retained below.
8. Oracle many-to-one and Hungarian majority-unit mappings differ substantially; raw, recording-standardized and within-subject controls are reported separately.
9. The fraction of units shared by at least half the held-out subjects declines with vocabulary size in the displayed fold-specific reuse diagnostics.
10. All eight continuous supervised configurations are invalid (32 nonconverged folds); artifact and timing gaps prevent claiming full protocol compliance.

## Reading this report

HuGaDB: 10 observed classes, 18 subjects, wearable IMU channels. LARa: 8 classes including None, 16 subjects, optical motion-capture channels. Cross-dataset comparisons differ in class count, subject count and modality; HuGaDB cannot establish body-shape effects. All score comparisons below are descriptive. Percentages are used for MoF, Edit, F1 and balanced accuracy; differences are percentage points (pp). “±” denotes population SD across exactly the three clustering seeds unless explicitly labelled otherwise. Seed SD measures initialization sensitivity, not sampling uncertainty. No cell is selected as “best”.

This revision reads the existing measurements and saved artifacts without fitting models or rerunning experiments. The [previous report](report_details/REPORT_previous.md) is preserved. Full tables: [A: information and permutation references](report_details/A_information.md), [B: all five clustering metrics](report_details/B_clustering.md), [C: all supervised settings](report_details/C_supervised.md), [D: reuse by fold](report_details/D_unit_reuse.md). Numerical audit: [report_audit.json](report_details/report_audit.json).

## 1. Provenance, populations and scope

| Dataset | Recordings | Frames | GT segments | Exact quantization agreement | Historical agreement | D7 MoF | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 364 | 1137986 | 3821 | 1 | 0.999894551 | 44.85 | passed |
| lara | 439 | 2634000 | 14105 | 1 | 1.000000000 | 41.66 | passed |

**hugadb checkpoint:** `models\pretrained\hugadb.model`; SHA-256 `22dfd5639c718177c12bc48debf67371c4bf7ca64af2d7de4e83ebacae55d7d2`. Cache: `results\exp2round\q1q2\cache\hugadb`.

**lara checkpoint:** `models\pretrained\lara.model`; SHA-256 `a45392ad30e27bda400adcab541aba087e4ae56b68837547cc65ca77ff5bfa65`. Cache: `results\seg\cache\lara`.

The gates require exact same-checkpoint latent quantization, historical agreement ≥0.999, and D7 MoF within 1.5 pp of the reference. HuGaDB uses a fresh released-checkpoint cache rather than the incompatible epoch-30 dump. LARa reuses the verified cache. D7 uses the original SMQ codebook, whereas the new vocabulary majority-unit control uses Ku clusters; they are different measurements.

The existing `exp2round` scores are historical references, not matched controls: both the clustering fit policy and HuGaDB checkpoint differ. CPU-only scope excludes merge curves, decoders, duration models, BABEL and encoder training.

Recorded software: Python 3.10.14, NumPy 1.26.4, scikit-learn 1.2.2, PyTorch 2.1.0+cu121; numerical-library thread cap 8. A CUDA-capable package version does not imply GPU execution.

### Action support

| Dataset | Class ID | Class | Segments | Frames |
| --- | --- | --- | --- | --- |
| hugadb | 1 | walking | 959 | 314966 |
| hugadb | 2 | running | 71 | 65917 |
| hugadb | 3 | going_up | 436 | 135127 |
| hugadb | 4 | going_down | 468 | 119819 |
| hugadb | 5 | sitting | 340 | 82819 |
| hugadb | 6 | sitting_down | 268 | 18180 |
| hugadb | 7 | standing_up | 278 | 20603 |
| hugadb | 8 | standing | 802 | 222649 |
| hugadb | 10 | up_by_elevator | 108 | 89144 |
| hugadb | 11 | down_by_elevator | 91 | 68762 |
| lara | 1 | Standing | 3124 | 295488 |
| lara | 2 | Walking | 1666 | 325489 |
| lara | 3 | Cart | 940 | 313685 |
| lara | 4 | Handling(upwards) | 1489 | 205557 |
| lara | 5 | Handling(centred) | 4830 | 1151079 |
| lara | 6 | Handling(downwards) | 1322 | 183612 |
| lara | 7 | Synchronization | 439 | 48168 |
| lara | 8 | None | 295 | 110922 |

HuGaDB IDs are non-contiguous: 1–8, 10 and 11. Mapping entries 9 (bicycling) and 12 (sitting_in_car) have no observations in this evaluated collection. No HuGaDB none class is present. LARa class 8 (`None`) is included throughout. No observed class was removed.

## 2. Methods and interpretation of metrics

**Shared grid.** Windows are HuGaDB 60/30/15 and LARa 50/25/12; Ku=10,20,50,100,500,1000; clustering seeds 111,222,1538574472. Sampling, PCA and splits use 111. Task C uses only full/quarter windows; raw and per-recording controls use Ku=10,100,1000 at those windows.

**Preprocessing and vocabulary.** Non-overlapping flattened windows; incomplete tails excluded from fitting and zero-padded for assignment, with all real frames retained in evaluation. Each fitting population samples 10,000 complete windows uniformly without replacement. Coordinate standardization and randomized PCA-64 fit only this sample. Full Lloyd KMeans uses k-means++, n_init=5, max_iter=300, tol=1e-4 for every Ku and for C-class final segment clustering. Sample IDs and transformations are saved. PCA explained variance varies by representation; see [preprocessing.csv](report_details/preprocessing.csv).

**Segment features.** Hard and soft histograms weight each window by its actual frame overlap with the GT segment, normalize to one, then take the elementwise square root. Soft probabilities use squared PCA-space Euclidean distances and temperature equal to 0.5/1/2 times the median strictly positive second-nearest-minus-nearest distance gap on fitting windows. Comparisons share the vocabulary and transforms. Continuous features are overlap-weighted means of PCA window features, not original frame-latent means.

**Pooled versus CV.** Pooled preprocessing, vocabularies and segment clustering fit the whole collection without action supervision, but GT boundaries and evaluation mapping are label-based. CV fits preprocessing, vocabulary, temperature and readout on fitting recordings only. Four folds are built by descending group frame count, seed-111 tie breaking and greedy frame balancing. Frozen encoder exposure remains transductive in either setting.

**Supervised fitting.** Each training segment is one sample; class balancing uses segment counts. Logistic regression uses L2, C=1, intercept, LBFGS, max_iter=2000, tol=1e-4, multinomial treatment for more than two classes, and no further feature standardization. Primary BA is class-average segment recall from pooled OOF predictions. Frame MoF weights each correctly predicted segment by its duration. Neither is an information-theoretic ceiling.

**Scoring.** Existing helpers produce dataset frame MoF; Edit averaged over recordings; F1 from pooled TP/FP/FN at IoU 0.10/0.25/0.50. Adjacent segments with equal predictions can merge in the scored frame sequence. Hungarian matching is dataset-wide and one-to-one; unmatched unit IDs map to a sentinel. Many-to-one assigns each unit its most frequent evaluation class, explicitly an oracle mapping. All five clustering metrics are available in Appendix B. Saved individual clustering scores are rounded to 0.01 before reporting seed summaries.

**Information.** One observation per assigned window, including terminal windows; window action is its majority real-frame label, subject is parsed from recording ID. MI/CMI and entropies use natural logs. NMI uses arithmetic normalization; AMI includes chance adjustment. Twenty global count-preserving permutations (seeds 111–130) supply marginal references; within-action permutations supply conditional subject references. Reference SD describes permutations, not independent temporal uncertainty; unit entropy is invariant under global permutation.

### Pooled preprocessing and window counts

| Dataset | Representation | W | Assigned windows | Terminal windows | Fitting windows | PCA dim | Explained variance % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | latent | 15 | 76035 | 331 | 10000 | 64 | 98.14 |
| hugadb | latent | 30 | 38106 | 344 | 10000 | 64 | 97.14 |
| hugadb | latent | 60 | 19143 | 354 | 10000 | 64 | 94.69 |
| hugadb | per_recording | 15 | 76035 | 331 | 10000 | 64 | 93.99 |
| hugadb | per_recording | 60 | 19143 | 354 | 10000 | 64 | 86.04 |
| hugadb | raw | 15 | 76035 | 331 | 10000 | 64 | 71.16 |
| hugadb | raw | 60 | 19143 | 354 | 10000 | 64 | 54.60 |
| lara | latent | 12 | 219500 | 0 | 10000 | 64 | 99.27 |
| lara | latent | 25 | 105360 | 0 | 10000 | 64 | 98.66 |
| lara | latent | 50 | 52680 | 0 | 10000 | 64 | 97.13 |
| lara | per_recording | 12 | 219500 | 0 | 10000 | 64 | 97.33 |
| lara | per_recording | 50 | 52680 | 0 | 10000 | 64 | 92.74 |
| lara | raw | 12 | 219500 | 0 | 10000 | 64 | 97.84 |
| lara | raw | 50 | 52680 | 0 | 10000 | 64 | 94.43 |

## 3. Task A — action and subject information

The following full-patch table shows observed information and its corresponding reference mean. All three windows and restricted representations, every entropy, arithmetic NMI, AMI, and reference mean/SD are in Appendix A. Occupancy distributions are preserved in `q1q2_information.csv`.

| Dataset | Ku | I(U;A) | I(U;S) | I(U;S given A) | Ref I(U;A) | Ref I(U;S) | Ref conditional | AMI action | AMI subject |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 10 | 0.985 ± 0.004 | 0.204 ± 0.013 | 0.233 ± 0.014 | 0.002 | 0.004 | 0.016 | 0.495 ± 0.007 | 0.084 ± 0.006 |
| hugadb | 20 | 1.015 ± 0.006 | 0.233 ± 0.006 | 0.277 ± 0.020 | 0.005 | 0.008 | 0.029 | 0.455 ± 0.002 | 0.085 ± 0.002 |
| hugadb | 50 | 1.158 ± 0.008 | 0.635 ± 0.012 | 0.686 ± 0.012 | 0.012 | 0.022 | 0.062 | 0.431 ± 0.003 | 0.201 ± 0.004 |
| hugadb | 100 | 1.289 ± 0.002 | 0.975 ± 0.038 | 1.029 ± 0.041 | 0.025 | 0.047 | 0.112 | 0.426 ± 0.004 | 0.276 ± 0.009 |
| hugadb | 500 | 1.509 ± 0.006 | 1.810 ± 0.004 | 1.831 ± 0.002 | 0.122 | 0.233 | 0.330 | 0.388 ± 0.001 | 0.407 ± 0.001 |
| hugadb | 1000 | 1.569 ± 0.003 | 2.094 ± 0.003 | 2.097 ± 0.002 | 0.221 | 0.409 | 0.501 | 0.361 ± 0.001 | 0.426 ± 0.001 |
| lara | 10 | 0.293 ± 0.007 | 0.606 ± 0.035 | 0.665 ± 0.045 | 0.001 | 0.001 | 0.009 | 0.156 ± 0.001 | 0.251 ± 0.012 |
| lara | 20 | 0.345 ± 0.006 | 0.871 ± 0.008 | 0.936 ± 0.003 | 0.001 | 0.003 | 0.018 | 0.155 ± 0.002 | 0.317 ± 0.005 |
| lara | 50 | 0.432 ± 0.004 | 1.178 ± 0.013 | 1.239 ± 0.014 | 0.003 | 0.007 | 0.044 | 0.160 ± 0.002 | 0.367 ± 0.004 |
| lara | 100 | 0.525 ± 0.003 | 1.395 ± 0.004 | 1.458 ± 0.005 | 0.007 | 0.014 | 0.080 | 0.171 ± 0.002 | 0.390 ± 0.001 |
| lara | 500 | 0.707 ± 0.002 | 1.864 ± 0.006 | 1.926 ± 0.005 | 0.034 | 0.075 | 0.265 | 0.179 ± 0.001 | 0.421 ± 0.001 |
| lara | 1000 | 0.798 ± 0.003 | 2.029 ± 0.006 | 2.085 ± 0.004 | 0.069 | 0.148 | 0.412 | 0.180 ± 0.001 | 0.419 ± 0.001 |

Full-patch action MI increases from 0.985 to 1.569 nats on HuGaDB and 0.293 to 0.798 on LARa. The count-preserving references also vary with Ku, so unadjusted MI increases alone do not establish improved action invariance. Conditional subject MI remains above its within-action reference in these cells. Action and subject associations can coexist; their magnitudes do not establish causal attribution.

![Information across all window sizes](report_details/information_all_windows.png)

### Occupancy and conditional entropies at the prespecified restricted Ku values

| Dataset | W | Ku | Occupied units | H(U) | H(A given U) | H(S given U) | Largest unit % of windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 15 | 10 | 10.0 ± 0.0 | 1.966 ± 0.001 | 1.088 ± 0.001 | 2.669 ± 0.001 | 36.74 ± 0.05 |
| hugadb | 15 | 100 | 100.0 ± 0.0 | 3.933 ± 0.050 | 0.787 ± 0.009 | 1.967 ± 0.035 | 12.79 ± 2.63 |
| hugadb | 15 | 1000 | 1000.0 ± 0.0 | 5.853 ± 0.008 | 0.548 ± 0.003 | 0.930 ± 0.005 | 3.66 ± 0.19 |
| hugadb | 30 | 10 | 10.0 ± 0.0 | 1.941 ± 0.018 | 1.047 ± 0.005 | 2.639 ± 0.001 | 35.54 ± 0.00 |
| hugadb | 30 | 100 | 100.0 ± 0.0 | 3.900 ± 0.032 | 0.758 ± 0.002 | 1.948 ± 0.033 | 12.32 ± 0.24 |
| hugadb | 30 | 1000 | 1000.0 ± 0.0 | 5.868 ± 0.005 | 0.491 ± 0.001 | 0.813 ± 0.005 | 3.64 ± 0.07 |
| hugadb | 60 | 10 | 10.0 ± 0.0 | 1.946 ± 0.037 | 1.042 ± 0.004 | 2.641 ± 0.013 | 35.83 ± 0.13 |
| hugadb | 60 | 100 | 100.0 ± 0.0 | 3.963 ± 0.053 | 0.738 ± 0.002 | 1.870 ± 0.038 | 13.98 ± 2.47 |
| hugadb | 60 | 1000 | 1000.0 ± 0.0 | 5.890 ± 0.003 | 0.458 ± 0.003 | 0.750 ± 0.003 | 3.51 ± 0.16 |
| lara | 12 | 10 | 10.0 ± 0.0 | 2.005 ± 0.057 | 1.429 ± 0.006 | 2.182 ± 0.031 | 20.25 ± 1.06 |
| lara | 12 | 100 | 100.0 ± 0.0 | 4.335 ± 0.020 | 1.229 ± 0.004 | 1.393 ± 0.013 | 2.77 ± 0.24 |
| lara | 12 | 1000 | 1000.0 ± 0.0 | 6.561 ± 0.004 | 0.988 ± 0.003 | 0.775 ± 0.004 | 0.67 ± 0.01 |
| lara | 25 | 10 | 10.0 ± 0.0 | 2.036 ± 0.050 | 1.420 ± 0.008 | 2.200 ± 0.038 | 19.97 ± 1.13 |
| lara | 25 | 100 | 100.0 ± 0.0 | 4.343 ± 0.011 | 1.224 ± 0.005 | 1.378 ± 0.016 | 2.97 ± 0.23 |
| lara | 25 | 1000 | 1000.0 ± 0.0 | 6.545 ± 0.005 | 0.958 ± 0.003 | 0.753 ± 0.014 | 0.68 ± 0.01 |
| lara | 50 | 10 | 10.0 ± 0.0 | 2.052 ± 0.056 | 1.417 ± 0.007 | 2.153 ± 0.035 | 19.59 ± 0.84 |
| lara | 50 | 100 | 100.0 ± 0.0 | 4.345 ± 0.027 | 1.186 ± 0.003 | 1.364 ± 0.004 | 3.06 ± 0.10 |
| lara | 50 | 1000 | 1000.0 ± 0.0 | 6.517 ± 0.010 | 0.912 ± 0.003 | 0.730 ± 0.006 | 0.76 ± 0.03 |

## 4. Task B — matched pooled sweep and controls

### Hard histograms: all windows and vocabulary sizes

| Dataset | W | Ku | MoF | Edit | F1@10 | F1@25 | F1@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | 49.23 ± 0.36 | 46.25 ± 4.49 | 48.12 ± 0.76 | 46.02 ± 0.76 | 44.62 ± 0.55 |
| hugadb | 60 | 20 | 46.26 ± 1.19 | 44.93 ± 2.37 | 47.85 ± 1.82 | 46.26 ± 1.87 | 43.81 ± 1.95 |
| hugadb | 60 | 50 | 42.38 ± 4.13 | 37.98 ± 2.82 | 40.59 ± 2.18 | 38.60 ± 2.30 | 36.20 ± 3.04 |
| hugadb | 60 | 100 | 42.02 ± 1.77 | 34.77 ± 4.63 | 42.45 ± 4.56 | 39.12 ± 5.31 | 35.80 ± 5.22 |
| hugadb | 60 | 500 | 36.55 ± 1.43 | 26.06 ± 2.31 | 30.14 ± 4.25 | 26.68 ± 4.70 | 22.98 ± 4.48 |
| hugadb | 60 | 1000 | 34.95 ± 0.55 | 19.26 ± 1.77 | 19.09 ± 4.48 | 15.83 ± 4.34 | 12.53 ± 4.47 |
| hugadb | 30 | 10 | 50.52 ± 0.91 | 51.22 ± 0.77 | 50.81 ± 1.34 | 47.91 ± 0.99 | 46.46 ± 1.06 |
| hugadb | 30 | 20 | 49.28 ± 0.27 | 49.31 ± 0.72 | 51.47 ± 0.48 | 49.08 ± 0.94 | 47.44 ± 1.25 |
| hugadb | 30 | 50 | 45.93 ± 1.20 | 39.25 ± 0.63 | 43.82 ± 1.33 | 41.49 ± 1.31 | 39.29 ± 1.49 |
| hugadb | 30 | 100 | 42.83 ± 1.44 | 38.48 ± 2.92 | 42.15 ± 3.47 | 39.25 ± 3.73 | 37.40 ± 4.12 |
| hugadb | 30 | 500 | 36.84 ± 4.11 | 27.33 ± 2.44 | 32.10 ± 4.58 | 29.44 ± 4.38 | 25.99 ± 4.10 |
| hugadb | 30 | 1000 | 36.54 ± 1.49 | 23.89 ± 1.33 | 24.87 ± 0.53 | 21.65 ± 0.63 | 18.39 ± 0.51 |
| hugadb | 15 | 10 | 49.24 ± 0.24 | 48.67 ± 0.74 | 51.87 ± 0.31 | 49.78 ± 0.50 | 48.33 ± 0.43 |
| hugadb | 15 | 20 | 50.49 ± 2.96 | 49.29 ± 3.23 | 56.34 ± 1.99 | 54.68 ± 1.88 | 53.55 ± 1.88 |
| hugadb | 15 | 50 | 52.38 ± 1.86 | 47.25 ± 5.15 | 53.94 ± 5.82 | 51.70 ± 6.30 | 50.28 ± 6.50 |
| hugadb | 15 | 100 | 41.24 ± 2.44 | 38.41 ± 5.30 | 44.00 ± 4.45 | 41.01 ± 5.65 | 39.05 ± 6.51 |
| hugadb | 15 | 500 | 38.01 ± 1.38 | 36.21 ± 3.74 | 39.81 ± 5.39 | 37.40 ± 6.12 | 35.57 ± 6.86 |
| hugadb | 15 | 1000 | 30.49 ± 3.21 | 23.10 ± 2.33 | 20.94 ± 4.06 | 18.67 ± 3.80 | 15.88 ± 3.40 |
| lara | 50 | 10 | 28.79 ± 0.02 | 29.90 ± 1.56 | 29.31 ± 2.07 | 26.55 ± 2.06 | 22.49 ± 1.97 |
| lara | 50 | 20 | 23.21 ± 1.63 | 24.81 ± 1.69 | 20.39 ± 0.42 | 17.38 ± 0.35 | 13.38 ± 0.12 |
| lara | 50 | 50 | 31.43 ± 2.18 | 22.75 ± 1.23 | 16.93 ± 1.29 | 12.41 ± 0.99 | 7.34 ± 0.64 |
| lara | 50 | 100 | 32.59 ± 1.13 | 21.32 ± 1.99 | 15.68 ± 1.58 | 11.22 ± 0.97 | 6.77 ± 0.48 |
| lara | 50 | 500 | 40.03 ± 0.59 | 12.54 ± 1.24 | 12.33 ± 1.28 | 7.98 ± 0.97 | 4.88 ± 0.51 |
| lara | 50 | 1000 | 38.05 ± 2.55 | 11.65 ± 2.88 | 9.91 ± 2.03 | 6.33 ± 1.57 | 3.61 ± 0.97 |
| lara | 25 | 10 | 30.19 ± 0.85 | 30.59 ± 2.50 | 30.94 ± 2.65 | 28.20 ± 2.72 | 24.04 ± 2.66 |
| lara | 25 | 20 | 23.88 ± 1.50 | 24.87 ± 1.13 | 22.20 ± 0.78 | 19.29 ± 0.91 | 15.19 ± 0.78 |
| lara | 25 | 50 | 28.08 ± 4.31 | 24.09 ± 0.84 | 16.36 ± 0.36 | 12.17 ± 0.17 | 7.70 ± 0.15 |
| lara | 25 | 100 | 29.37 ± 0.82 | 23.04 ± 1.97 | 17.26 ± 1.29 | 12.65 ± 1.01 | 8.28 ± 0.82 |
| lara | 25 | 500 | 38.65 ± 2.17 | 15.87 ± 1.00 | 14.74 ± 1.22 | 10.36 ± 1.01 | 6.50 ± 1.18 |
| lara | 25 | 1000 | 38.24 ± 2.39 | 14.85 ± 0.54 | 13.36 ± 0.90 | 8.98 ± 0.97 | 5.45 ± 0.63 |
| lara | 12 | 10 | 30.42 ± 1.01 | 29.74 ± 2.03 | 30.06 ± 2.80 | 27.36 ± 2.87 | 23.26 ± 2.57 |
| lara | 12 | 20 | 24.73 ± 2.63 | 24.05 ± 3.23 | 22.72 ± 5.04 | 20.06 ± 5.34 | 16.10 ± 5.10 |
| lara | 12 | 50 | 25.35 ± 2.75 | 21.63 ± 1.26 | 16.06 ± 0.88 | 11.85 ± 0.75 | 7.75 ± 0.37 |
| lara | 12 | 100 | 31.76 ± 2.55 | 21.34 ± 1.25 | 16.26 ± 0.77 | 11.84 ± 0.58 | 7.26 ± 0.48 |
| lara | 12 | 500 | 37.79 ± 0.93 | 17.47 ± 0.70 | 15.12 ± 0.42 | 10.75 ± 0.33 | 6.19 ± 0.20 |
| lara | 12 | 1000 | 40.11 ± 0.66 | 15.06 ± 0.36 | 13.84 ± 0.14 | 9.50 ± 0.20 | 5.27 ± 0.36 |

HuGaDB full-patch hard MoF decreases between Ku=10 and 1000, while LARa increases across those endpoints. The relationship is not uniformly monotonic across either full grid. Window changes also produce different results; no window is chosen from evaluation scores.

### Majority-unit mapping: identical predictions, different oracle mapping constraints

| Dataset | W | Ku | Hungarian MoF | Oracle many-to-one MoF | Many-to-one minus Hungarian pp |
| --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | 42.66 ± 0.46 | 59.69 ± 0.53 | 17.03 ± 0.76 |
| hugadb | 60 | 20 | 35.64 ± 0.16 | 59.95 ± 0.25 | 24.31 ± 0.30 |
| hugadb | 60 | 50 | 28.32 ± 0.15 | 68.23 ± 0.10 | 39.91 ± 0.19 |
| hugadb | 60 | 100 | 22.94 ± 1.93 | 74.53 ± 0.50 | 51.59 ± 2.24 |
| hugadb | 60 | 500 | 11.31 ± 0.14 | 83.12 ± 0.07 | 71.81 ± 0.07 |
| hugadb | 60 | 1000 | 7.55 ± 0.02 | 84.59 ± 0.10 | 77.04 ± 0.10 |
| hugadb | 30 | 10 | 44.84 ± 0.32 | 60.12 ± 0.01 | 15.28 ± 0.31 |
| hugadb | 30 | 20 | 37.46 ± 1.44 | 60.17 ± 0.20 | 22.71 ± 1.39 |
| hugadb | 30 | 50 | 27.91 ± 0.68 | 67.02 ± 1.01 | 39.11 ± 0.52 |
| hugadb | 30 | 100 | 23.53 ± 1.23 | 74.17 ± 0.43 | 50.64 ± 1.64 |
| hugadb | 30 | 500 | 12.88 ± 0.52 | 82.81 ± 0.59 | 69.93 ± 0.94 |
| hugadb | 30 | 1000 | 8.49 ± 0.38 | 84.43 ± 0.11 | 75.95 ± 0.39 |
| hugadb | 15 | 10 | 46.94 ± 0.23 | 62.01 ± 0.15 | 15.07 ± 0.27 |
| hugadb | 15 | 20 | 42.51 ± 0.40 | 66.13 ± 0.74 | 23.62 ± 0.95 |
| hugadb | 15 | 50 | 34.72 ± 2.75 | 71.44 ± 0.84 | 36.72 ± 2.06 |
| hugadb | 15 | 100 | 25.10 ± 0.95 | 75.36 ± 0.70 | 50.26 ± 0.84 |
| hugadb | 15 | 500 | 10.21 ± 0.54 | 81.93 ± 0.28 | 71.72 ± 0.36 |
| hugadb | 15 | 1000 | 8.30 ± 0.24 | 82.85 ± 0.09 | 74.55 ± 0.22 |
| lara | 50 | 10 | 27.16 ± 1.03 | 49.99 ± 0.43 | 22.83 ± 0.61 |
| lara | 50 | 20 | 19.62 ± 0.61 | 50.92 ± 0.56 | 31.29 ± 1.17 |
| lara | 50 | 50 | 10.95 ± 0.72 | 54.84 ± 0.29 | 43.89 ± 1.02 |
| lara | 50 | 100 | 7.44 ± 0.42 | 59.26 ± 0.16 | 51.82 ± 0.53 |
| lara | 50 | 500 | 3.47 ± 0.08 | 67.43 ± 0.40 | 63.96 ± 0.41 |
| lara | 50 | 1000 | 2.80 ± 0.14 | 71.66 ± 0.18 | 68.86 ± 0.05 |
| lara | 25 | 10 | 28.05 ± 1.08 | 49.83 ± 0.70 | 21.78 ± 0.87 |
| lara | 25 | 20 | 21.26 ± 1.52 | 52.18 ± 0.58 | 30.92 ± 0.95 |
| lara | 25 | 50 | 11.82 ± 0.61 | 55.09 ± 0.15 | 43.27 ± 0.47 |
| lara | 25 | 100 | 7.83 ± 0.57 | 57.65 ± 0.45 | 49.82 ± 0.17 |
| lara | 25 | 500 | 3.54 ± 0.08 | 66.85 ± 0.11 | 63.32 ± 0.18 |
| lara | 25 | 1000 | 2.62 ± 0.17 | 71.54 ± 0.36 | 68.92 ± 0.52 |
| lara | 12 | 10 | 28.20 ± 0.82 | 50.68 ± 0.67 | 22.48 ± 0.15 |
| lara | 12 | 20 | 20.59 ± 1.48 | 51.98 ± 0.54 | 31.39 ± 0.96 |
| lara | 12 | 50 | 12.04 ± 0.58 | 54.94 ± 0.51 | 42.90 ± 0.55 |
| lara | 12 | 100 | 8.09 ± 0.35 | 58.61 ± 0.23 | 50.52 ± 0.45 |
| lara | 12 | 500 | 3.56 ± 0.21 | 67.70 ± 0.12 | 64.14 ± 0.09 |
| lara | 12 | 1000 | 2.75 ± 0.07 | 71.46 ± 0.12 | 68.72 ± 0.15 |

At full-patch Ku=1000, the mapping difference is much larger than at Ku=10. The table quantifies that difference without assigning its cause. The many-to-one mapping uses evaluation labels and is not a deployable classifier or a CV result.

### Count-preserving permutation and duration weighting

Each contrast pairs the same dataset/window/Ku/seed. The permutation control uses one global count-preserving shuffle per clustering seed (`seed + 10000000`), then rebuilds segment histograms and fits final clusters. This differs from Task A’s 20-permutation information references. Length weighting changes only the final histogram KMeans sample weights to segment durations.

| Dataset | W | Ku | Hard MoF | Permuted MoF | Hard minus permuted pp | Duration-weighted MoF | Weighted minus hard pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | 49.23 ± 0.36 | 17.80 ± 0.22 | 31.42 ± 0.42 | 47.56 ± 2.52 | -1.66 ± 2.33 |
| hugadb | 60 | 20 | 46.26 ± 1.19 | 15.64 ± 0.34 | 30.62 ± 0.90 | 44.15 ± 0.68 | -2.11 ± 1.63 |
| hugadb | 60 | 50 | 42.38 ± 4.13 | 19.36 ± 0.72 | 23.02 ± 4.04 | 46.57 ± 2.01 | 4.19 ± 2.12 |
| hugadb | 60 | 100 | 42.02 ± 1.77 | 17.73 ± 1.03 | 24.28 ± 2.78 | 41.35 ± 2.17 | -0.67 ± 3.48 |
| hugadb | 60 | 500 | 36.55 ± 1.43 | 15.32 ± 0.42 | 21.24 ± 1.72 | 29.35 ± 4.47 | -7.20 ± 5.48 |
| hugadb | 60 | 1000 | 34.95 ± 0.55 | 15.43 ± 0.09 | 19.52 ± 0.48 | 33.15 ± 2.68 | -1.80 ± 3.03 |
| hugadb | 30 | 10 | 50.52 ± 0.91 | 19.45 ± 0.24 | 31.07 ± 0.98 | 49.50 ± 0.08 | -1.03 ± 0.98 |
| hugadb | 30 | 20 | 49.28 ± 0.27 | 19.61 ± 0.61 | 29.66 ± 0.46 | 47.31 ± 1.21 | -1.96 ± 1.40 |
| hugadb | 30 | 50 | 45.93 ± 1.20 | 19.24 ± 0.62 | 26.69 ± 0.62 | 46.09 ± 0.88 | 0.16 ± 1.03 |
| hugadb | 30 | 100 | 42.83 ± 1.44 | 18.83 ± 0.21 | 24.01 ± 1.54 | 42.92 ± 1.55 | 0.09 ± 2.30 |
| hugadb | 30 | 500 | 36.84 ± 4.11 | 16.97 ± 0.27 | 19.86 ± 4.35 | 36.92 ± 2.76 | 0.08 ± 6.63 |
| hugadb | 30 | 1000 | 36.54 ± 1.49 | 16.53 ± 1.26 | 20.02 ± 2.05 | 30.61 ± 4.11 | -5.93 ± 4.40 |
| hugadb | 15 | 10 | 49.24 ± 0.24 | 20.29 ± 0.85 | 28.96 ± 0.85 | 48.84 ± 0.29 | -0.40 ± 0.34 |
| hugadb | 15 | 20 | 50.49 ± 2.96 | 20.17 ± 0.21 | 30.33 ± 3.03 | 50.00 ± 1.58 | -0.49 ± 2.00 |
| hugadb | 15 | 50 | 52.38 ± 1.86 | 20.52 ± 0.40 | 31.86 ± 2.26 | 52.14 ± 1.93 | -0.24 ± 0.56 |
| hugadb | 15 | 100 | 41.24 ± 2.44 | 20.85 ± 0.70 | 20.40 ± 1.92 | 46.50 ± 1.16 | 5.25 ± 3.12 |
| hugadb | 15 | 500 | 38.01 ± 1.38 | 19.60 ± 0.48 | 18.41 ± 1.80 | 37.10 ± 1.33 | -0.91 ± 1.15 |
| hugadb | 15 | 1000 | 30.49 ± 3.21 | 18.85 ± 0.36 | 11.64 ± 2.91 | 28.41 ± 2.16 | -2.08 ± 4.76 |
| lara | 50 | 10 | 28.79 ± 0.02 | 31.47 ± 0.46 | -2.68 ± 0.46 | 28.96 ± 0.12 | 0.17 ± 0.15 |
| lara | 50 | 20 | 23.21 ± 1.63 | 20.58 ± 1.24 | 2.63 ± 0.39 | 22.03 ± 1.86 | -1.18 ± 2.66 |
| lara | 50 | 50 | 31.43 ± 2.18 | 19.88 ± 2.01 | 11.56 ± 3.49 | 22.02 ± 1.19 | -9.41 ± 3.15 |
| lara | 50 | 100 | 32.59 ± 1.13 | 24.78 ± 0.65 | 7.81 ± 1.44 | 26.68 ± 0.67 | -5.90 ± 0.71 |
| lara | 50 | 500 | 40.03 ± 0.59 | 30.70 ± 1.98 | 9.33 ± 2.39 | 34.82 ± 1.34 | -5.21 ± 1.89 |
| lara | 50 | 1000 | 38.05 ± 2.55 | 37.81 ± 1.66 | 0.24 ± 3.86 | 36.01 ± 1.87 | -2.04 ± 2.85 |
| lara | 25 | 10 | 30.19 ± 0.85 | 33.98 ± 0.39 | -3.79 ± 1.17 | 30.36 ± 0.82 | 0.17 ± 0.08 |
| lara | 25 | 20 | 23.88 ± 1.50 | 33.22 ± 0.17 | -9.35 ± 1.60 | 23.44 ± 1.32 | -0.44 ± 0.37 |
| lara | 25 | 50 | 28.08 ± 4.31 | 29.51 ± 0.47 | -1.43 ± 3.89 | 21.60 ± 0.80 | -6.48 ± 3.93 |
| lara | 25 | 100 | 29.37 ± 0.82 | 25.45 ± 0.25 | 3.92 ± 1.05 | 25.34 ± 1.70 | -4.03 ± 1.39 |
| lara | 25 | 500 | 38.65 ± 2.17 | 24.83 ± 2.23 | 13.82 ± 3.74 | 31.96 ± 2.04 | -6.68 ± 0.22 |
| lara | 25 | 1000 | 38.24 ± 2.39 | 26.85 ± 4.25 | 11.40 ± 6.35 | 33.84 ± 1.75 | -4.40 ± 1.60 |
| lara | 12 | 10 | 30.42 ± 1.01 | 35.44 ± 0.80 | -5.02 ± 1.36 | 31.16 ± 0.90 | 0.74 ± 0.48 |
| lara | 12 | 20 | 24.73 ± 2.63 | 35.60 ± 0.07 | -10.87 ± 2.65 | 23.74 ± 2.66 | -0.99 ± 0.79 |
| lara | 12 | 50 | 25.35 ± 2.75 | 34.12 ± 0.07 | -8.77 ± 2.69 | 21.12 ± 0.87 | -4.22 ± 2.18 |
| lara | 12 | 100 | 31.76 ± 2.55 | 33.08 ± 0.23 | -1.33 ± 2.72 | 22.86 ± 2.75 | -8.89 ± 2.94 |
| lara | 12 | 500 | 37.79 ± 0.93 | 26.79 ± 1.87 | 11.01 ± 2.46 | 30.19 ± 3.13 | -7.60 ± 2.32 |
| lara | 12 | 1000 | 40.11 ± 0.66 | 29.10 ± 1.22 | 11.01 ± 1.46 | 34.89 ± 2.34 | -5.22 ± 2.96 |

The permutation control is not universally below the observed score: for example, LARa W=12/Ku=10 has hard MoF 30.42 versus permuted 35.44 (seed means). It therefore should not be labelled a universal chance floor. Duration weighting also changes scores in both directions across cells. These descriptive comparisons do not establish a cause.

### Continuous pooled segment clustering

Continuous representations have no vocabulary-size parameter. Their final C-cluster KMeans still has three initialization seeds; this seed variation is applicable to pooled clustering.

| Dataset | W | MoF | Edit | F1@10 | F1@25 | F1@50 |
| --- | --- | --- | --- | --- | --- | --- |
| hugadb | 15 | 49.41 ± 0.21 | 41.08 ± 0.31 | 50.16 ± 0.26 | 47.43 ± 0.42 | 46.32 ± 0.50 |
| hugadb | 30 | 49.47 ± 0.07 | 41.24 ± 0.24 | 50.18 ± 0.06 | 47.53 ± 0.06 | 46.54 ± 0.03 |
| hugadb | 60 | 48.85 ± 0.41 | 42.85 ± 0.13 | 49.63 ± 0.54 | 47.49 ± 0.45 | 45.87 ± 0.36 |
| lara | 12 | 30.78 ± 0.39 | 30.67 ± 0.74 | 31.34 ± 1.56 | 28.22 ± 1.56 | 23.70 ± 1.56 |
| lara | 25 | 30.48 ± 0.43 | 29.16 ± 1.41 | 29.39 ± 2.06 | 26.00 ± 2.18 | 21.61 ± 2.15 |
| lara | 50 | 30.08 ± 0.39 | 29.35 ± 0.57 | 29.21 ± 1.38 | 25.94 ± 1.36 | 21.32 ± 1.37 |

## 5. Task C — hard/soft comparisons and supervised readouts

### Pooled clustering and temperature sensitivity

Hard and all soft temperatures share preprocessing and prototypes. The continuous pooled values are in §4. All five scores for each temperature are in Appendix B.

| Dataset | W | Ku | Hard MoF | Soft 0.5× MoF | Soft 1× MoF | Soft 2× MoF | Base-soft minus hard pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | 49.23 ± 0.36 | 50.62 ± 0.19 | 49.95 ± 1.23 | 48.13 ± 0.37 | 0.72 ± 0.93 |
| hugadb | 60 | 20 | 46.26 ± 1.19 | 49.40 ± 1.89 | 48.95 ± 1.25 | 48.78 ± 1.20 | 2.69 ± 0.15 |
| hugadb | 60 | 50 | 42.38 ± 4.13 | 48.79 ± 0.12 | 48.99 ± 0.21 | 50.82 ± 0.15 | 6.61 ± 4.33 |
| hugadb | 60 | 100 | 42.02 ± 1.77 | 50.82 ± 3.53 | 53.35 ± 1.68 | 51.14 ± 0.15 | 11.33 ± 3.36 |
| hugadb | 60 | 500 | 36.55 ± 1.43 | 48.55 ± 3.08 | 51.54 ± 2.55 | 51.55 ± 1.42 | 14.99 ± 3.07 |
| hugadb | 60 | 1000 | 34.95 ± 0.55 | 46.82 ± 2.18 | 50.22 ± 2.49 | 52.17 ± 2.51 | 15.27 ± 3.01 |
| hugadb | 15 | 10 | 49.24 ± 0.24 | 49.25 ± 0.07 | 47.58 ± 0.13 | 47.43 ± 0.07 | -1.67 ± 0.33 |
| hugadb | 15 | 20 | 50.49 ± 2.96 | 48.96 ± 0.59 | 47.71 ± 0.54 | 47.54 ± 0.61 | -2.78 ± 2.64 |
| hugadb | 15 | 50 | 52.38 ± 1.86 | 54.19 ± 0.46 | 54.00 ± 0.60 | 51.44 ± 0.49 | 1.61 ± 1.96 |
| hugadb | 15 | 100 | 41.24 ± 2.44 | 55.19 ± 0.37 | 54.84 ± 0.84 | 54.42 ± 0.22 | 13.60 ± 1.60 |
| hugadb | 15 | 500 | 38.01 ± 1.38 | 51.20 ± 3.20 | 54.84 ± 0.71 | 55.23 ± 0.68 | 16.83 ± 1.24 |
| hugadb | 15 | 1000 | 30.49 ± 3.21 | 47.76 ± 0.18 | 54.07 ± 1.08 | 54.04 ± 0.96 | 23.58 ± 3.74 |
| lara | 50 | 10 | 28.79 ± 0.02 | 28.84 ± 0.02 | 29.06 ± 0.20 | 29.55 ± 1.55 | 0.27 ± 0.18 |
| lara | 50 | 20 | 23.21 ± 1.63 | 28.27 ± 0.44 | 30.33 ± 0.08 | 30.61 ± 1.05 | 7.12 ± 1.55 |
| lara | 50 | 50 | 31.43 ± 2.18 | 26.75 ± 2.50 | 30.71 ± 0.56 | 31.71 ± 0.10 | -0.72 ± 2.10 |
| lara | 50 | 100 | 32.59 ± 1.13 | 22.47 ± 0.54 | 30.31 ± 1.34 | 31.85 ± 0.31 | -2.28 ± 1.92 |
| lara | 50 | 500 | 40.03 ± 0.59 | 22.18 ± 0.99 | 25.56 ± 2.48 | 31.83 ± 0.41 | -14.47 ± 2.89 |
| lara | 50 | 1000 | 38.05 ± 2.55 | 23.87 ± 0.30 | 25.04 ± 1.98 | 31.33 ± 0.10 | -13.00 ± 4.52 |
| lara | 12 | 10 | 30.42 ± 1.01 | 29.66 ± 0.28 | 29.55 ± 0.44 | 30.16 ± 0.61 | -0.87 ± 1.23 |
| lara | 12 | 20 | 24.73 ± 2.63 | 29.23 ± 0.74 | 30.97 ± 0.61 | 30.54 ± 0.69 | 6.24 ± 3.20 |
| lara | 12 | 50 | 25.35 ± 2.75 | 29.30 ± 0.50 | 30.59 ± 0.40 | 31.65 ± 0.80 | 5.25 ± 2.84 |
| lara | 12 | 100 | 31.76 ± 2.55 | 22.79 ± 0.53 | 30.10 ± 0.06 | 31.93 ± 0.18 | -1.65 ± 2.61 |
| lara | 12 | 500 | 37.79 ± 0.93 | 24.65 ± 0.32 | 27.24 ± 2.73 | 30.75 ± 0.09 | -10.56 ± 3.65 |
| lara | 12 | 1000 | 40.11 ± 0.66 | 22.99 ± 1.58 | 26.65 ± 2.14 | 30.72 ± 0.22 | -13.46 ± 2.39 |

At the prespecified full-patch Ku=100 comparison, HuGaDB hard/base-soft pooled MoF is 42.02/53.35, while LARa is 32.59/30.31. The sign of the hard-to-soft difference therefore differs in this matched comparison. These are seed means; paired SDs are in the last table column. No temperature was selected by evaluation performance.

### Cross-validation fold populations

| Dataset | Grouping | Fold | Fit recordings | Test recordings | Fit segments | Test segments | Test frames | Held-out subjects | Missing fit classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | recording | 0 | 273 | 91 | 2851 | 970 | 284540 | subjects may overlap | [] |
| hugadb | recording | 1 | 273 | 91 | 2884 | 937 | 284478 | subjects may overlap | [] |
| hugadb | recording | 2 | 273 | 91 | 2868 | 953 | 284485 | subjects may overlap | [] |
| hugadb | recording | 3 | 273 | 91 | 2860 | 961 | 284483 | subjects may overlap | [] |
| hugadb | subject | 0 | 265 | 99 | 2809 | 1012 | 295437 | 01, 03, 06, 10, 15 | [] |
| hugadb | subject | 1 | 274 | 90 | 2905 | 916 | 294859 | 02, 04, 11, 12, 13 | [] |
| hugadb | subject | 2 | 273 | 91 | 2804 | 1017 | 273662 | 05, 08, 09, 17 | [] |
| hugadb | subject | 3 | 280 | 84 | 2945 | 876 | 274028 | 07, 14, 16, 18 | [] |
| lara | recording | 0 | 329 | 110 | 10626 | 3479 | 660000 | subjects may overlap | [] |
| lara | recording | 1 | 329 | 110 | 10503 | 3602 | 660000 | subjects may overlap | [] |
| lara | recording | 2 | 329 | 110 | 10479 | 3626 | 660000 | subjects may overlap | [] |
| lara | recording | 3 | 330 | 109 | 10707 | 3398 | 654000 | subjects may overlap | [] |
| lara | subject | 0 | 336 | 103 | 11325 | 2780 | 618000 | S07, S11, S13, S14 | [] |
| lara | subject | 1 | 325 | 114 | 10701 | 3404 | 684000 | S06, S08, S10, S16 | [] |
| lara | subject | 2 | 325 | 114 | 9439 | 4666 | 684000 | S01, S02, S03, S05 | [] |
| lara | subject | 3 | 331 | 108 | 10850 | 3255 | 648000 | S04, S09, S12, S15 | [] |

All folds retain every observed training class. Subject folds are disjoint by parsed ID, and saved fitting-window IDs fall exclusively in fitting recordings; these were checked in this report audit. Recording folds allow the same subject in fitting and evaluation recordings. [Per-class support](report_details/class_support.csv) and original fold-level training/evaluation supports remain available.

### Hard and base-soft readouts: both grouping protocols, all windows and Ku

Each entry aggregates three independently initialized vocabularies. Metrics are computed on pooled OOF predictions within each seed, not averages of fold percentages.

| Dataset | W | Ku | Readout | Recording MoF | Subject MoF | Recording minus subject pp | Recording BA | Subject BA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | hard | 42.24 ± 0.49 | 41.69 ± 0.45 | 0.55 ± 0.92 | 55.19 ± 0.41 | 54.78 ± 0.25 |
| hugadb | 60 | 10 | soft | 46.55 ± 0.15 | 46.47 ± 0.02 | 0.08 ± 0.13 | 59.57 ± 0.08 | 59.62 ± 0.04 |
| hugadb | 60 | 20 | hard | 43.35 ± 0.08 | 41.26 ± 0.70 | 2.09 ± 0.66 | 54.91 ± 0.53 | 53.50 ± 0.55 |
| hugadb | 60 | 20 | soft | 48.12 ± 0.19 | 47.69 ± 0.13 | 0.43 ± 0.19 | 61.20 ± 0.32 | 60.28 ± 0.14 |
| hugadb | 60 | 50 | hard | 58.80 ± 0.48 | 57.82 ± 0.61 | 0.98 ± 1.04 | 65.24 ± 0.62 | 62.58 ± 1.08 |
| hugadb | 60 | 50 | soft | 66.22 ± 0.35 | 63.00 ± 1.00 | 3.21 ± 1.32 | 71.00 ± 0.98 | 67.62 ± 0.32 |
| hugadb | 60 | 100 | hard | 69.11 ± 0.75 | 66.08 ± 0.36 | 3.03 ± 0.57 | 72.97 ± 0.85 | 69.88 ± 0.50 |
| hugadb | 60 | 100 | soft | 73.07 ± 0.80 | 69.24 ± 0.09 | 3.83 ± 0.80 | 80.25 ± 0.58 | 75.86 ± 0.20 |
| hugadb | 60 | 500 | hard | 77.69 ± 0.18 | 79.19 ± 0.61 | -1.50 ± 0.46 | 80.34 ± 0.28 | 77.76 ± 0.55 |
| hugadb | 60 | 500 | soft | 78.91 ± 0.28 | 78.25 ± 0.35 | 0.66 ± 0.61 | 82.50 ± 0.19 | 80.37 ± 0.24 |
| hugadb | 60 | 1000 | hard | 77.64 ± 0.09 | 79.16 ± 0.20 | -1.53 ± 0.17 | 81.12 ± 0.12 | 78.38 ± 0.30 |
| hugadb | 60 | 1000 | soft | 79.13 ± 0.13 | 79.03 ± 0.16 | 0.10 ± 0.25 | 82.74 ± 0.12 | 80.27 ± 0.14 |
| hugadb | 15 | 10 | hard | 45.15 ± 0.51 | 40.95 ± 0.16 | 4.20 ± 0.67 | 59.34 ± 0.08 | 56.44 ± 0.07 |
| hugadb | 15 | 10 | soft | 56.81 ± 0.67 | 54.46 ± 0.06 | 2.35 ± 0.73 | 62.46 ± 0.41 | 59.27 ± 0.02 |
| hugadb | 15 | 20 | hard | 55.79 ± 0.94 | 55.50 ± 1.86 | 0.29 ± 2.60 | 63.04 ± 0.33 | 62.50 ± 0.16 |
| hugadb | 15 | 20 | soft | 64.23 ± 0.39 | 64.54 ± 1.25 | -0.31 ± 1.57 | 67.19 ± 0.31 | 65.15 ± 0.60 |
| hugadb | 15 | 50 | hard | 70.93 ± 1.19 | 66.89 ± 0.35 | 4.04 ± 1.25 | 69.44 ± 0.82 | 67.79 ± 0.96 |
| hugadb | 15 | 50 | soft | 75.54 ± 0.20 | 70.36 ± 1.39 | 5.18 ± 1.49 | 74.36 ± 0.43 | 71.13 ± 0.52 |
| hugadb | 15 | 100 | hard | 75.98 ± 0.91 | 73.47 ± 0.51 | 2.51 ± 1.41 | 73.79 ± 0.41 | 69.64 ± 0.57 |
| hugadb | 15 | 100 | soft | 79.22 ± 0.04 | 76.31 ± 0.45 | 2.91 ± 0.48 | 77.33 ± 0.29 | 74.33 ± 1.01 |
| hugadb | 15 | 500 | hard | 77.73 ± 0.39 | 80.99 ± 0.66 | -3.27 ± 0.27 | 76.58 ± 0.74 | 74.28 ± 0.40 |
| hugadb | 15 | 500 | soft | 80.02 ± 0.16 | 81.50 ± 0.13 | -1.48 ± 0.10 | 78.44 ± 0.44 | 77.19 ± 0.38 |
| hugadb | 15 | 1000 | hard | 77.48 ± 0.25 | 80.16 ± 0.32 | -2.68 ± 0.34 | 78.23 ± 0.90 | 75.00 ± 0.08 |
| hugadb | 15 | 1000 | soft | 80.54 ± 0.06 | 82.18 ± 0.13 | -1.64 ± 0.12 | 80.48 ± 0.23 | 78.54 ± 0.53 |
| lara | 50 | 10 | hard | 29.91 ± 0.66 | 26.62 ± 0.59 | 3.30 ± 0.36 | 42.67 ± 0.16 | 40.30 ± 0.53 |
| lara | 50 | 10 | soft | 38.01 ± 0.91 | 35.91 ± 0.56 | 2.10 ± 1.08 | 47.06 ± 0.50 | 44.80 ± 0.22 |
| lara | 50 | 20 | hard | 35.68 ± 0.59 | 32.35 ± 0.86 | 3.33 ± 0.52 | 46.45 ± 0.47 | 43.47 ± 0.22 |
| lara | 50 | 20 | soft | 44.75 ± 0.69 | 43.18 ± 1.13 | 1.57 ± 1.19 | 51.52 ± 0.37 | 49.25 ± 0.50 |
| lara | 50 | 50 | hard | 42.55 ± 0.19 | 39.14 ± 0.32 | 3.40 ± 0.16 | 50.48 ± 0.14 | 46.66 ± 0.85 |
| lara | 50 | 50 | soft | 51.53 ± 0.12 | 49.27 ± 0.34 | 2.26 ± 0.29 | 56.66 ± 0.12 | 53.75 ± 0.88 |
| lara | 50 | 100 | hard | 46.73 ± 0.77 | 43.10 ± 1.06 | 3.62 ± 1.72 | 53.58 ± 0.75 | 49.14 ± 0.49 |
| lara | 50 | 100 | soft | 54.66 ± 0.04 | 52.28 ± 0.43 | 2.38 ± 0.39 | 60.09 ± 0.22 | 55.95 ± 0.17 |
| lara | 50 | 500 | hard | 54.83 ± 0.25 | 47.27 ± 0.52 | 7.56 ± 0.47 | 58.63 ± 0.31 | 52.09 ± 0.15 |
| lara | 50 | 500 | soft | 63.03 ± 0.27 | 57.80 ± 0.66 | 5.23 ± 0.67 | 65.08 ± 0.49 | 59.25 ± 0.51 |
| lara | 50 | 1000 | hard | 58.11 ± 0.20 | 50.56 ± 0.62 | 7.54 ± 0.47 | 60.35 ± 0.23 | 53.45 ± 0.32 |
| lara | 50 | 1000 | soft | 65.71 ± 0.24 | 60.84 ± 0.35 | 4.87 ± 0.44 | 67.31 ± 0.30 | 61.19 ± 0.10 |
| lara | 12 | 10 | hard | 36.49 ± 0.75 | 33.07 ± 0.40 | 3.41 ± 0.77 | 47.39 ± 0.41 | 45.22 ± 0.37 |
| lara | 12 | 10 | soft | 45.00 ± 0.74 | 42.74 ± 0.62 | 2.26 ± 1.00 | 51.60 ± 0.26 | 49.19 ± 0.52 |
| lara | 12 | 20 | hard | 42.28 ± 0.29 | 37.32 ± 0.98 | 4.96 ± 1.09 | 51.77 ± 0.12 | 47.24 ± 0.69 |
| lara | 12 | 20 | soft | 51.10 ± 0.18 | 47.88 ± 0.20 | 3.22 ± 0.34 | 56.92 ± 0.36 | 52.72 ± 0.48 |
| lara | 12 | 50 | hard | 48.90 ± 0.61 | 45.66 ± 0.41 | 3.24 ± 0.69 | 55.47 ± 0.44 | 52.62 ± 1.00 |
| lara | 12 | 50 | soft | 56.07 ± 0.25 | 54.96 ± 0.59 | 1.11 ± 0.41 | 61.29 ± 0.17 | 58.99 ± 0.86 |
| lara | 12 | 100 | hard | 53.65 ± 0.54 | 49.89 ± 0.45 | 3.76 ± 0.69 | 60.09 ± 0.75 | 56.19 ± 0.37 |
| lara | 12 | 100 | soft | 60.39 ± 0.67 | 58.28 ± 0.91 | 2.11 ± 1.10 | 65.48 ± 0.82 | 61.87 ± 0.51 |
| lara | 12 | 500 | hard | 63.38 ± 0.03 | 56.39 ± 0.56 | 6.99 ± 0.56 | 67.15 ± 0.12 | 60.54 ± 0.28 |
| lara | 12 | 500 | soft | 68.72 ± 0.31 | 65.22 ± 0.31 | 3.50 ± 0.43 | 71.71 ± 0.08 | 66.45 ± 0.18 |
| lara | 12 | 1000 | hard | 65.99 ± 0.18 | 59.32 ± 0.68 | 6.67 ± 0.87 | 68.33 ± 0.19 | 61.74 ± 0.26 |
| lara | 12 | 1000 | soft | 70.34 ± 0.14 | 66.47 ± 0.39 | 3.87 ± 0.49 | 72.42 ± 0.07 | 67.35 ± 0.14 |

The grouping difference is a protocol difference, not an isolated estimate of previously seeing a person. For example, full-patch hard subject-grouped MoF at Ku=1000 is 79.16 on HuGaDB and 50.56 on LARa, while their pooled hard-clustering MoF values are 34.95 and 38.05. Both the fitting protocol and readout differ across these settings; the contrast does not isolate one component.

At full-patch Ku=100, hard/base-soft subject-grouped MoF is 66.08/69.24 on HuGaDB and 43.10/52.28 on LARa. Corresponding balanced accuracy is 69.88/75.86 and 49.14/55.95. Thus the LARa soft comparison at this setting differs between pooled clustering and supervised readout; this is a measured protocol-dependent difference, not a causal diagnosis.

### Supervised temperature sensitivity

Below are paired differences from the same seed’s base-temperature soft readout. Absolute MoF and BA at all three temperatures are in Appendix C.

| Dataset | W | Grouping | Ku | ΔMoF 0.5×−1× | ΔMoF 2×−1× | ΔBA 0.5×−1× | ΔBA 2×−1× |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | recording | 10 | 1.19 ± 0.13 | 0.18 ± 0.20 | 0.51 ± 0.09 | -0.49 ± 0.03 |
| hugadb | 60 | recording | 20 | 0.79 ± 1.03 | -0.83 ± 0.20 | -0.52 ± 0.44 | -2.10 ± 0.28 |
| hugadb | 60 | recording | 50 | -3.78 ± 0.32 | 0.96 ± 0.33 | -1.68 ± 0.52 | 1.05 ± 0.40 |
| hugadb | 60 | recording | 100 | -1.88 ± 0.30 | 1.44 ± 0.60 | -2.06 ± 0.18 | -0.60 ± 0.42 |
| hugadb | 60 | recording | 500 | 0.08 ± 0.36 | -0.78 ± 0.29 | -0.34 ± 0.05 | -0.08 ± 0.37 |
| hugadb | 60 | recording | 1000 | 0.39 ± 0.15 | -0.66 ± 0.16 | -0.23 ± 0.11 | -0.26 ± 0.14 |
| hugadb | 60 | subject | 10 | 0.88 ± 0.36 | -1.16 ± 0.02 | -0.17 ± 0.25 | -2.44 ± 0.13 |
| hugadb | 60 | subject | 20 | 1.66 ± 0.23 | -2.44 ± 0.22 | 0.50 ± 0.40 | -3.14 ± 0.26 |
| hugadb | 60 | subject | 50 | -1.79 ± 0.42 | -0.38 ± 0.78 | -1.55 ± 0.19 | 0.02 ± 0.24 |
| hugadb | 60 | subject | 100 | -0.45 ± 0.34 | -0.20 ± 1.14 | -0.04 ± 0.58 | -0.75 ± 0.53 |
| hugadb | 60 | subject | 500 | 1.76 ± 0.43 | -3.51 ± 0.37 | 0.50 ± 0.11 | -1.73 ± 0.06 |
| hugadb | 60 | subject | 1000 | 1.56 ± 0.29 | -3.17 ± 0.19 | 0.97 ± 0.13 | -1.27 ± 0.24 |
| hugadb | 15 | recording | 10 | -2.99 ± 0.04 | 1.99 ± 0.05 | -1.37 ± 0.16 | -0.10 ± 0.11 |
| hugadb | 15 | recording | 20 | 2.12 ± 0.17 | -2.18 ± 0.51 | 0.03 ± 0.20 | -2.89 ± 0.20 |
| hugadb | 15 | recording | 50 | -0.67 ± 0.51 | -0.04 ± 0.36 | -1.27 ± 0.32 | 0.07 ± 0.24 |
| hugadb | 15 | recording | 100 | -0.34 ± 0.72 | -0.54 ± 0.21 | -0.76 ± 0.55 | -0.35 ± 0.28 |
| hugadb | 15 | recording | 500 | -0.11 ± 0.14 | 0.39 ± 0.33 | -0.29 ± 0.27 | -0.27 ± 0.34 |
| hugadb | 15 | recording | 1000 | -0.05 ± 0.10 | -0.00 ± 0.09 | -0.16 ± 0.36 | -1.38 ± 0.20 |
| hugadb | 15 | subject | 10 | -2.21 ± 0.05 | 3.28 ± 0.08 | -1.14 ± 0.07 | -0.13 ± 0.07 |
| hugadb | 15 | subject | 20 | 0.61 ± 0.95 | -2.42 ± 0.42 | 0.18 ± 0.38 | -2.34 ± 1.17 |
| hugadb | 15 | subject | 50 | -0.08 ± 0.46 | -0.22 ± 1.22 | -0.89 ± 0.23 | -1.49 ± 0.46 |
| hugadb | 15 | subject | 100 | 0.25 ± 0.14 | -1.44 ± 0.42 | -0.13 ± 0.27 | -1.10 ± 0.80 |
| hugadb | 15 | subject | 500 | 0.79 ± 0.07 | -1.07 ± 0.08 | 0.09 ± 0.27 | -1.27 ± 0.17 |
| hugadb | 15 | subject | 1000 | 0.85 ± 0.18 | -1.32 ± 0.13 | 0.16 ± 0.21 | -1.86 ± 0.38 |
| lara | 50 | recording | 10 | -2.25 ± 0.41 | 1.96 ± 0.29 | -1.90 ± 0.18 | 1.55 ± 0.26 |
| lara | 50 | recording | 20 | -3.38 ± 0.64 | 2.53 ± 0.23 | -2.37 ± 0.33 | 1.17 ± 0.10 |
| lara | 50 | recording | 50 | -3.02 ± 0.42 | 1.97 ± 0.11 | -2.48 ± 0.19 | 1.01 ± 0.30 |
| lara | 50 | recording | 100 | -3.12 ± 0.35 | 1.28 ± 0.13 | -2.27 ± 0.01 | 0.55 ± 0.30 |
| lara | 50 | recording | 500 | -1.00 ± 0.78 | -2.13 ± 0.31 | -1.73 ± 0.22 | -0.74 ± 0.06 |
| lara | 50 | recording | 1000 | 0.10 ± 0.27 | -3.34 ± 0.32 | -1.33 ± 0.28 | -1.41 ± 0.21 |
| lara | 50 | subject | 10 | -3.73 ± 0.32 | 1.53 ± 0.37 | -2.19 ± 0.17 | 1.66 ± 0.26 |
| lara | 50 | subject | 20 | -3.75 ± 0.88 | 1.60 ± 0.20 | -2.52 ± 0.14 | 1.11 ± 0.15 |
| lara | 50 | subject | 50 | -3.87 ± 0.40 | 1.27 ± 0.20 | -2.85 ± 0.17 | 1.10 ± 0.46 |
| lara | 50 | subject | 100 | -2.90 ± 0.74 | 0.42 ± 0.44 | -2.09 ± 0.27 | 0.80 ± 0.08 |
| lara | 50 | subject | 500 | -2.60 ± 0.30 | -1.49 ± 0.42 | -2.15 ± 0.40 | 0.39 ± 0.08 |
| lara | 50 | subject | 1000 | -1.65 ± 0.48 | -2.18 ± 0.22 | -2.52 ± 0.12 | 0.28 ± 0.06 |
| lara | 12 | recording | 10 | -1.32 ± 0.40 | -0.95 ± 1.09 | -1.68 ± 0.08 | 1.24 ± 0.43 |
| lara | 12 | recording | 20 | -3.84 ± 0.10 | 2.23 ± 0.57 | -2.16 ± 0.11 | 1.36 ± 0.22 |
| lara | 12 | recording | 50 | -3.47 ± 0.38 | 1.85 ± 0.34 | -2.53 ± 0.31 | 0.71 ± 0.23 |
| lara | 12 | recording | 100 | -2.48 ± 0.40 | 0.84 ± 0.23 | -2.08 ± 0.26 | 0.20 ± 0.20 |
| lara | 12 | recording | 500 | 0.03 ± 0.26 | -2.24 ± 0.22 | -0.88 ± 0.10 | -1.43 ± 0.15 |
| lara | 12 | recording | 1000 | 0.83 ± 0.24 | -3.06 ± 0.19 | -0.35 ± 0.33 | -1.79 ± 0.13 |
| lara | 12 | subject | 10 | -1.37 ± 0.59 | 0.34 ± 0.64 | -1.42 ± 0.06 | 1.81 ± 0.24 |
| lara | 12 | subject | 20 | -3.90 ± 0.84 | 2.05 ± 0.49 | -2.38 ± 0.53 | 1.34 ± 0.20 |
| lara | 12 | subject | 50 | -3.70 ± 0.47 | 0.58 ± 0.80 | -2.15 ± 0.29 | 0.46 ± 0.23 |
| lara | 12 | subject | 100 | -3.33 ± 0.50 | 0.73 ± 0.53 | -2.02 ± 0.07 | 0.67 ± 0.26 |
| lara | 12 | subject | 500 | -2.04 ± 0.46 | -2.52 ± 0.17 | -2.13 ± 0.16 | -0.25 ± 0.13 |
| lara | 12 | subject | 1000 | -0.47 ± 0.20 | -3.64 ± 0.41 | -1.57 ± 0.27 | -0.69 ± 0.13 |

![All temperatures and both CV grouping protocols](report_details/readout_all_temperatures.png)

![Balanced accuracy in a separate figure](report_details/balanced_accuracy.png)

### Continuous supervised readouts — invalid, not comparable evidence

Every one of the 32 continuous fold fits emitted a convergence warning at max_iter=2000. Consequently all eight pooled OOF configurations are invalid. The table retains their computed values solely for failure audit; they must not be used as valid continuous-versus-histogram comparisons. No replacement regularization, scaling, solver, or iteration limit was used. Seed spread is not applicable.

| Dataset | W | Grouping | Nonconverged folds | Invalid OOF MoF | Invalid OOF BA | Status |
| --- | --- | --- | --- | --- | --- | --- |
| hugadb | 15 | recording | 4/4 | 73.74 | 78.77 | INVALID |
| hugadb | 60 | recording | 4/4 | 72.18 | 78.07 | INVALID |
| hugadb | 15 | subject | 4/4 | 69.81 | 71.48 | INVALID |
| hugadb | 60 | subject | 4/4 | 66.61 | 71.03 | INVALID |
| lara | 12 | recording | 4/4 | 70.90 | 72.73 | INVALID |
| lara | 50 | recording | 4/4 | 67.17 | 69.93 | INVALID |
| lara | 12 | subject | 4/4 | 64.68 | 66.66 | INVALID |
| lara | 50 | subject | 4/4 | 61.09 | 65.17 | INVALID |

### Per-class recall and support

Appendix C includes full-patch hard/base-soft subject-grouped recall at every Ku and every class, named using the dataset mapping files. [per_class_recall.csv](report_details/per_class_recall.csv) includes all valid temperatures, both windows and both groupings. Class supports are segment counts; frame imbalance is shown separately in §1. BA and frame MoF should be read together because one averages classes over segments and the other weights duration.

## 6. Task D — restricted Q1 controls

### Raw features and per-recording latent standardization

Raw controls use the existing model-input channels without the encoder. The standardized-latent control computes each coordinate’s population mean/SD over the complete recording (SD below 1e-8 replaced with 1) before windowing. Both then use the same sampled standardization/PCA/KMeans pipeline. Recording standardization uses future frames and is not an online method. These are pooled diagnostics; raw features have no encoder exposure, but their fitting is transductive.

| Dataset | W | Ku | Representation | Hard MoF | ΔMoF vs latent pp | I(U;A) | I(U;S) | I(U;S given A) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | latent | 49.23 ± 0.36 | reference | 0.985 ± 0.004 | 0.204 ± 0.013 | 0.233 ± 0.014 |
| hugadb | 60 | 10 | raw | 46.65 ± 0.43 | -2.57 ± 0.67 | 0.733 ± 0.008 | 0.064 ± 0.002 | 0.116 ± 0.003 |
| hugadb | 60 | 10 | per_recording | 37.43 ± 2.80 | -11.80 ± 2.65 | 0.770 ± 0.028 | 0.053 ± 0.009 | 0.108 ± 0.005 |
| hugadb | 60 | 100 | latent | 42.02 ± 1.77 | reference | 1.289 ± 0.002 | 0.975 ± 0.038 | 1.029 ± 0.041 |
| hugadb | 60 | 100 | raw | 43.99 ± 3.50 | 1.98 ± 2.32 | 1.165 ± 0.017 | 0.787 ± 0.045 | 0.836 ± 0.052 |
| hugadb | 60 | 100 | per_recording | 59.15 ± 3.71 | 17.13 ± 1.96 | 1.346 ± 0.002 | 0.336 ± 0.007 | 0.432 ± 0.007 |
| hugadb | 60 | 1000 | latent | 34.95 ± 0.55 | reference | 1.569 ± 0.003 | 2.094 ± 0.003 | 2.097 ± 0.002 |
| hugadb | 60 | 1000 | raw | 33.64 ± 2.04 | -1.31 ± 2.19 | 1.491 ± 0.006 | 2.061 ± 0.023 | 2.063 ± 0.020 |
| hugadb | 60 | 1000 | per_recording | 37.61 ± 7.74 | 2.66 ± 7.83 | 1.626 ± 0.001 | 1.362 ± 0.015 | 1.441 ± 0.015 |
| hugadb | 15 | 10 | latent | 49.24 ± 0.24 | reference | 0.940 ± 0.001 | 0.175 ± 0.001 | 0.209 ± 0.001 |
| hugadb | 15 | 10 | raw | 45.83 ± 0.56 | -3.41 ± 0.35 | 0.619 ± 0.003 | 0.083 ± 0.003 | 0.140 ± 0.004 |
| hugadb | 15 | 10 | per_recording | 46.87 ± 1.28 | -2.38 ± 1.20 | 0.650 ± 0.018 | 0.054 ± 0.000 | 0.092 ± 0.001 |
| hugadb | 15 | 100 | latent | 41.24 ± 2.44 | reference | 1.241 ± 0.009 | 0.877 ± 0.035 | 0.921 ± 0.036 |
| hugadb | 15 | 100 | raw | 49.89 ± 4.63 | 8.65 ± 6.81 | 1.156 ± 0.008 | 0.795 ± 0.078 | 0.811 ± 0.083 |
| hugadb | 15 | 100 | per_recording | 59.54 ± 4.67 | 18.30 ± 7.04 | 1.237 ± 0.007 | 0.311 ± 0.007 | 0.391 ± 0.012 |
| hugadb | 15 | 1000 | latent | 30.49 ± 3.21 | reference | 1.480 ± 0.003 | 1.914 ± 0.005 | 1.937 ± 0.004 |
| hugadb | 15 | 1000 | raw | 34.30 ± 2.16 | 3.81 ± 5.36 | 1.465 ± 0.004 | 1.931 ± 0.011 | 1.918 ± 0.009 |
| hugadb | 15 | 1000 | per_recording | 45.71 ± 1.32 | 15.22 ± 4.08 | 1.526 ± 0.003 | 1.165 ± 0.004 | 1.251 ± 0.004 |
| lara | 50 | 10 | latent | 28.79 ± 0.02 | reference | 0.293 ± 0.007 | 0.606 ± 0.035 | 0.665 ± 0.045 |
| lara | 50 | 10 | raw | 26.78 ± 0.26 | -2.01 ± 0.26 | 0.160 ± 0.001 | 0.122 ± 0.001 | 0.176 ± 0.000 |
| lara | 50 | 10 | per_recording | 29.33 ± 0.15 | 0.54 ± 0.15 | 0.308 ± 0.000 | 0.027 ± 0.000 | 0.071 ± 0.000 |
| lara | 50 | 100 | latent | 32.59 ± 1.13 | reference | 0.525 ± 0.003 | 1.395 ± 0.004 | 1.458 ± 0.005 |
| lara | 50 | 100 | raw | 32.67 ± 1.00 | 0.08 ± 1.19 | 0.347 ± 0.004 | 0.771 ± 0.009 | 0.935 ± 0.007 |
| lara | 50 | 100 | per_recording | 24.00 ± 0.95 | -8.59 ± 1.99 | 0.526 ± 0.001 | 0.199 ± 0.005 | 0.335 ± 0.004 |
| lara | 50 | 1000 | latent | 38.05 ± 2.55 | reference | 0.798 ± 0.003 | 2.029 ± 0.006 | 2.085 ± 0.004 |
| lara | 50 | 1000 | raw | 40.68 ± 0.72 | 2.63 ± 3.26 | 0.651 ± 0.003 | 1.660 ± 0.003 | 1.849 ± 0.001 |
| lara | 50 | 1000 | per_recording | 35.29 ± 3.51 | -2.75 ± 0.99 | 0.711 ± 0.002 | 0.603 ± 0.005 | 0.887 ± 0.004 |
| lara | 12 | 10 | latent | 30.42 ± 1.01 | reference | 0.283 ± 0.006 | 0.576 ± 0.031 | 0.626 ± 0.041 |
| lara | 12 | 10 | raw | 26.75 ± 0.02 | -3.67 ± 1.02 | 0.160 ± 0.000 | 0.118 ± 0.000 | 0.172 ± 0.000 |
| lara | 12 | 10 | per_recording | 32.02 ± 0.08 | 1.60 ± 1.08 | 0.294 ± 0.000 | 0.027 ± 0.000 | 0.065 ± 0.000 |
| lara | 12 | 100 | latent | 31.76 ± 2.55 | reference | 0.482 ± 0.004 | 1.366 ± 0.013 | 1.420 ± 0.012 |
| lara | 12 | 100 | raw | 29.72 ± 3.13 | -2.04 ± 3.35 | 0.346 ± 0.006 | 0.733 ± 0.012 | 0.881 ± 0.012 |
| lara | 12 | 100 | per_recording | 28.86 ± 2.95 | -2.90 ± 3.81 | 0.470 ± 0.002 | 0.177 ± 0.005 | 0.284 ± 0.006 |
| lara | 12 | 1000 | latent | 40.11 ± 0.66 | reference | 0.723 ± 0.003 | 1.984 ± 0.004 | 2.026 ± 0.003 |
| lara | 12 | 1000 | raw | 41.74 ± 0.75 | 1.63 ± 0.44 | 0.620 ± 0.003 | 1.606 ± 0.006 | 1.766 ± 0.006 |
| lara | 12 | 1000 | per_recording | 28.21 ± 1.13 | -11.90 ± 1.76 | 0.613 ± 0.001 | 0.530 ± 0.006 | 0.758 ± 0.003 |

At full-patch Ku=100, HuGaDB latent/raw/per-recording-standardized MoF is 42.02/43.99/59.15; LARa is 32.59/32.67/24.00. Per-recording standardization therefore changes the score in opposite directions in this prescribed comparison. All seeds, other restricted Ku values and window scales are retained above.

These controls alter representation and the subsequent fitted geometry. The raw-feature comparison is not an isolated causal estimate of encoder benefit. The per-recording control is not evidence that subject identity has been removed; its subject-information measurements are reported directly.

### Within-subject segment clustering

The pooled latent vocabulary and hard histograms remain fixed; only final C-cluster segment models and Hungarian mappings are fitted separately per subject. Coverage is 100% of frames in every saved cell, and no subject had fewer than C segments. Separate mappings add oracle flexibility.

| Dataset | W | Ku | Pooled MoF | Within-subject MoF | Within minus pooled pp | Within Edit | Within F1@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 60 | 10 | 49.23 ± 0.36 | 55.84 ± 0.88 | 6.62 ± 0.79 | 52.38 ± 0.69 | 50.26 ± 0.23 |
| hugadb | 60 | 100 | 42.02 ± 1.77 | 58.20 ± 0.45 | 16.18 ± 1.95 | 53.46 ± 0.42 | 55.24 ± 0.88 |
| hugadb | 60 | 1000 | 34.95 ± 0.55 | 57.51 ± 0.97 | 22.56 ± 1.49 | 49.44 ± 0.90 | 49.15 ± 1.56 |
| hugadb | 15 | 10 | 49.24 ± 0.24 | 60.21 ± 0.57 | 10.97 ± 0.64 | 58.84 ± 0.65 | 56.70 ± 0.39 |
| hugadb | 15 | 100 | 41.24 ± 2.44 | 68.51 ± 0.70 | 27.27 ± 1.86 | 63.65 ± 0.81 | 64.66 ± 0.95 |
| hugadb | 15 | 1000 | 30.49 ± 3.21 | 66.47 ± 0.52 | 35.98 ± 3.29 | 62.07 ± 0.45 | 65.22 ± 1.10 |
| lara | 50 | 10 | 28.79 ± 0.02 | 39.80 ± 0.17 | 11.01 ± 0.17 | 36.94 ± 0.12 | 27.43 ± 0.58 |
| lara | 50 | 100 | 32.59 ± 1.13 | 33.86 ± 0.71 | 1.27 ± 1.62 | 30.65 ± 0.48 | 21.13 ± 0.59 |
| lara | 50 | 1000 | 38.05 ± 2.55 | 29.03 ± 0.62 | -9.01 ± 2.48 | 27.93 ± 0.28 | 14.72 ± 0.63 |
| lara | 12 | 10 | 30.42 ± 1.01 | 40.94 ± 0.46 | 10.52 ± 0.95 | 38.67 ± 0.33 | 30.05 ± 0.79 |
| lara | 12 | 100 | 31.76 ± 2.55 | 35.12 ± 0.64 | 3.37 ± 2.61 | 32.79 ± 0.32 | 24.78 ± 0.66 |
| lara | 12 | 1000 | 40.11 ± 0.66 | 30.61 ± 0.25 | -9.50 ± 0.91 | 29.02 ± 0.35 | 18.37 ± 0.13 |

Within-subject MoF is not uniformly higher: at LARa full-patch Ku=1000 it is 29.03 versus pooled 38.05. The extra oracle flexibility therefore does not itself guarantee a higher score after this clustering fit.

### Unit reuse across held-out subjects

For a fold, a unit is shared if it occurs in at least ceil(held-subject-count/2) subjects, using complete windows only. Per-unit entropy first divides each subject’s unit count by that subject’s complete-window count, normalizes these rates across subjects, then computes entropy in nats and divides by log(number of held subjects). Units unused everywhere have undefined entropy and are counted separately.

The original plan restricts Task D to Ku=10,100,1000. The implementation also saved reuse at Ku=20,50,500 by reusing Task C vocabularies; these are explicitly additional descriptive outputs. The primary table below uses the restricted values. Fold-specific three-seed means/SD and all extra Ku values are in Appendix D. The figure separates folds instead of mixing seed and fold variation.

![Reuse with separate fold curves](report_details/reuse_by_fold.png)

| Dataset | W | Ku | Fold | Subjects | Reuse fraction | Unused units | Mean entropy, used units | Mean normalized entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 15 | 10 | 0 | 5 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.154 ± 0.001 | 0.717 ± 0.001 |
| hugadb | 15 | 10 | 1 | 5 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.443 ± 0.000 | 0.896 ± 0.000 |
| hugadb | 15 | 10 | 2 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.279 ± 0.000 | 0.922 ± 0.000 |
| hugadb | 15 | 10 | 3 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.263 ± 0.000 | 0.911 ± 0.000 |
| hugadb | 15 | 100 | 0 | 5 | 0.403 ± 0.012 | 13.7 ± 1.7 | 0.617 ± 0.023 | 0.384 ± 0.015 |
| hugadb | 15 | 100 | 1 | 5 | 0.860 ± 0.008 | 4.7 ± 2.5 | 0.959 ± 0.028 | 0.596 ± 0.017 |
| hugadb | 15 | 100 | 2 | 4 | 0.947 ± 0.005 | 1.0 ± 0.8 | 0.832 ± 0.014 | 0.600 ± 0.010 |
| hugadb | 15 | 100 | 3 | 4 | 0.850 ± 0.008 | 6.0 ± 0.8 | 0.811 ± 0.017 | 0.585 ± 0.012 |
| hugadb | 15 | 1000 | 0 | 5 | 0.123 ± 0.004 | 489.0 ± 2.2 | 0.365 ± 0.006 | 0.227 ± 0.004 |
| hugadb | 15 | 1000 | 1 | 5 | 0.251 ± 0.009 | 344.3 ± 1.7 | 0.487 ± 0.008 | 0.303 ± 0.005 |
| hugadb | 15 | 1000 | 2 | 4 | 0.465 ± 0.003 | 201.7 ± 4.5 | 0.373 ± 0.005 | 0.269 ± 0.003 |
| hugadb | 15 | 1000 | 3 | 4 | 0.369 ± 0.011 | 384.7 ± 5.7 | 0.434 ± 0.010 | 0.313 ± 0.007 |
| hugadb | 60 | 10 | 0 | 5 | 0.900 ± 0.000 | 0.3 ± 0.5 | 1.090 ± 0.046 | 0.677 ± 0.028 |
| hugadb | 60 | 10 | 1 | 5 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.376 ± 0.002 | 0.855 ± 0.001 |
| hugadb | 60 | 10 | 2 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.259 ± 0.006 | 0.908 ± 0.004 |
| hugadb | 60 | 10 | 3 | 4 | 0.967 ± 0.047 | 0.3 ± 0.5 | 1.293 ± 0.006 | 0.933 ± 0.005 |
| hugadb | 60 | 100 | 0 | 5 | 0.293 ± 0.026 | 16.7 ± 1.2 | 0.479 ± 0.008 | 0.297 ± 0.005 |
| hugadb | 60 | 100 | 1 | 5 | 0.667 ± 0.012 | 10.3 ± 1.2 | 0.834 ± 0.005 | 0.518 ± 0.003 |
| hugadb | 60 | 100 | 2 | 4 | 0.860 ± 0.008 | 3.0 ± 0.8 | 0.777 ± 0.019 | 0.560 ± 0.014 |
| hugadb | 60 | 100 | 3 | 4 | 0.663 ± 0.005 | 9.3 ± 2.4 | 0.666 ± 0.029 | 0.480 ± 0.021 |
| hugadb | 60 | 1000 | 0 | 5 | 0.071 ± 0.002 | 629.0 ± 7.8 | 0.269 ± 0.006 | 0.167 ± 0.004 |
| hugadb | 60 | 1000 | 1 | 5 | 0.126 ± 0.005 | 536.7 ± 2.6 | 0.455 ± 0.012 | 0.283 ± 0.007 |
| hugadb | 60 | 1000 | 2 | 4 | 0.242 ± 0.001 | 394.7 ± 4.5 | 0.275 ± 0.003 | 0.198 ± 0.002 |
| hugadb | 60 | 1000 | 3 | 4 | 0.212 ± 0.005 | 561.7 ± 9.8 | 0.360 ± 0.011 | 0.259 ± 0.008 |
| lara | 12 | 10 | 0 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.096 ± 0.006 | 0.791 ± 0.004 |
| lara | 12 | 10 | 1 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 0.956 ± 0.042 | 0.689 ± 0.030 |
| lara | 12 | 10 | 2 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.024 ± 0.036 | 0.739 ± 0.026 |
| lara | 12 | 10 | 3 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 0.951 ± 0.000 | 0.686 ± 0.000 |
| lara | 12 | 100 | 0 | 4 | 0.930 ± 0.014 | 0.7 ± 0.9 | 0.729 ± 0.014 | 0.526 ± 0.010 |
| lara | 12 | 100 | 1 | 4 | 0.800 ± 0.016 | 2.7 ± 0.5 | 0.516 ± 0.015 | 0.372 ± 0.011 |
| lara | 12 | 100 | 2 | 4 | 0.800 ± 0.022 | 4.7 ± 1.2 | 0.539 ± 0.003 | 0.389 ± 0.002 |
| lara | 12 | 100 | 3 | 4 | 0.870 ± 0.008 | 5.3 ± 0.5 | 0.567 ± 0.015 | 0.409 ± 0.011 |
| lara | 12 | 1000 | 0 | 4 | 0.540 ± 0.007 | 143.3 ± 3.9 | 0.437 ± 0.004 | 0.315 ± 0.003 |
| lara | 12 | 1000 | 1 | 4 | 0.357 ± 0.002 | 230.3 ± 7.1 | 0.276 ± 0.002 | 0.199 ± 0.001 |
| lara | 12 | 1000 | 2 | 4 | 0.442 ± 0.002 | 171.3 ± 4.6 | 0.301 ± 0.002 | 0.217 ± 0.002 |
| lara | 12 | 1000 | 3 | 4 | 0.445 ± 0.005 | 167.3 ± 3.9 | 0.276 ± 0.007 | 0.199 ± 0.005 |
| lara | 50 | 10 | 0 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 1.082 ± 0.007 | 0.780 ± 0.005 |
| lara | 50 | 10 | 1 | 4 | 1.000 ± 0.000 | 0.0 ± 0.0 | 0.885 ± 0.001 | 0.638 ± 0.001 |
| lara | 50 | 10 | 2 | 4 | 0.967 ± 0.047 | 0.0 ± 0.0 | 0.975 ± 0.057 | 0.703 ± 0.041 |
| lara | 50 | 10 | 3 | 4 | 0.900 ± 0.000 | 0.0 ± 0.0 | 0.761 ± 0.003 | 0.549 ± 0.002 |
| lara | 50 | 100 | 0 | 4 | 0.847 ± 0.005 | 1.0 ± 0.0 | 0.711 ± 0.011 | 0.513 ± 0.008 |
| lara | 50 | 100 | 1 | 4 | 0.700 ± 0.037 | 7.3 ± 1.7 | 0.531 ± 0.024 | 0.383 ± 0.017 |
| lara | 50 | 100 | 2 | 4 | 0.780 ± 0.008 | 6.3 ± 1.9 | 0.594 ± 0.026 | 0.429 ± 0.019 |
| lara | 50 | 100 | 3 | 4 | 0.823 ± 0.012 | 7.7 ± 1.2 | 0.544 ± 0.008 | 0.392 ± 0.005 |
| lara | 50 | 1000 | 0 | 4 | 0.399 ± 0.003 | 268.0 ± 3.6 | 0.402 ± 0.007 | 0.290 ± 0.005 |
| lara | 50 | 1000 | 1 | 4 | 0.275 ± 0.003 | 345.3 ± 3.7 | 0.267 ± 0.010 | 0.193 ± 0.007 |
| lara | 50 | 1000 | 2 | 4 | 0.314 ± 0.005 | 279.3 ± 6.2 | 0.281 ± 0.006 | 0.202 ± 0.005 |
| lara | 50 | 1000 | 3 | 4 | 0.285 ± 0.009 | 289.7 ± 2.5 | 0.229 ± 0.006 | 0.165 ± 0.004 |

## 7. Coverage, validation and reproducibility limits

| Table | Expected rows | Observed | Missing | Unexpected | Duplicate keys |
| --- | --- | --- | --- | --- | --- |
| pooled | 882 | 882 | 0 | 0 | 0 |
| information | 180 | 180 | 0 | 0 | 0 |
| supervised | 2920 | 2920 | 0 | 0 | 0 |
| reuse | 80640 | 80640 | 0 | 0 | 0 |

Coverage was checked by the full dataset/representation/window/Ku/seed/readout/temperature/fold key sets, not merely by total row counts. The reuse expected set includes the explicitly additional Ku=20,50,500 outputs. Of 2,920 supervised rows, 2,304 valid fold rows and 576 valid pooled OOF rows describe hard/soft readouts; 32 invalid fold rows and 8 invalid pooled OOF rows describe continuous readouts. The 40 invalid rows are eight failed configurations, not 40 independent experiments.

The prior implementation tests cover information examples, count preservation, overlap histograms, mapping fragmentation and folds. This report additionally verifies occupancy sums, confusion totals, per-class recall/BA, real fold segment support, subject disjointness and complete fitting-window membership. Frame MoF cannot be independently reconstructed from confusion matrices alone because they omit durations; individual saved prediction sequences are needed for that audit.

The 756 saved vocabulary artifacts have iteration counts from 8 to 128; none reached the 300-iteration limit. This does not verify final segment-clustering convergence. The archived smoke outputs with incorrect non-contiguous HuGaDB class indices are excluded; source CSVs are the consolidated, corrected measurements.

### Gaps in the saved evidence

- Vocabulary artifacts retain centers, per-recording window assignments, inertia and iteration counts; preprocessing artifacts retain sample IDs and transformation arrays. Fold memberships are saved. The current audit checks their population membership; it does not rerun quantization or model fitting.

- The runners compute final segment labels and logistic predictions but do not persist them. Final segment-clustering models and logistic coefficients are also not saved. Thus the requested incremental prediction/model archive is incomplete, even though metric rows cover the grid.

- Information contingency matrices are not saved explicitly. They could be reconstructed from assignments and source labels, but the delivered table stores only the derived quantities and occupancy distributions.

- Final segment-clustering convergence is not recorded. “Complete” pooled status indicates a recorded score, not verified convergence of every final KMeans. Saved vocabulary iteration counts provide a narrower diagnostic.

- Runtime fields mix individual and cumulative cell timings; summing them double-counts shared work. Some CV rows omit cache identity/sample/convergence metadata requested per cell. Exact total compute time and execution-priority compliance cannot be reconstructed reliably.

### Timing inconsistency

The source manifest records start `2026-09-23T11:26:12Z` and completion `2026-09-23T15:55:56.720035Z`. Their difference is 4.50 hours. Completion precedes its recorded deadline `2026-09-23T19:26:12Z`, yet `deadline_compliance` is false and the note claims the deadline was exceeded. Those fields contradict each other. The previous report’s overrun claim is unsupported by these timestamps. The historical manifest is preserved; the report audit records this inconsistency rather than inventing a corrected run history. Reporting time in this revision is separate from the original run.

## Open questions

- What accounts for the different Ku trends in pooled clustering and supervised readouts within each dataset?

- How much of the observed subject association persists with encoders trained without held-out subjects?

- How would these measurements change without ground-truth segment boundaries and evaluation-label mappings?

- Why do the count-preserving histogram permutation controls exceed observed MoF in some LARa cells?

- What explains the hard/soft and temperature differences across windows and datasets?

- Would a separately specified converged continuous-readout protocol change its comparison with histogram readouts?

## What these numbers cannot establish

They do not establish a causal bottleneck, which component dominates, a project contribution, or what the project should do next. They do not identify body-shape effects from HuGaDB IMU inputs. Action/subject MI depends on the observed window population and finite-sample partition; raw values at different Ku are not controlled estimates of invariance. All encoder results remain transductive. Pooled raw fitting is transductive despite no encoder exposure. The CV grouping gap changes the protocol rather than isolating previous exposure to a person.

Oracle boundaries, Hungarian mappings, oracle many-to-one mappings, supervised fitting/evaluation, information labels and within-action permutations all use action labels. Subject IDs are used for grouping, separate subject fits and diagnostics. These diagnostics do not measure unsupervised boundary discovery or end-to-end action segmentation without labels. Seed spreads and permutation spreads are descriptive; neither is sampling uncertainty. Invalid continuous readouts cannot support a valid comparison or an information ceiling.

## Not run

No planned measurement keys are missing from the saved score grid. The 8 continuous supervised configurations are invalid because every fold failed the prescribed convergence criterion; their 40 rows are retained in [not_run.csv](not_run.csv). No substitute method was fitted in this reporting revision. Required individual predictions, final fitted readout models, explicit contingency artifacts and full convergence/runtime logs are unavailable as detailed above. Experiments outside Q1/Q2 scope were not requested and are not treated as omissions.

## Files and reproduction

- Source measurements: [clustering](q1q2_cells.csv), [information](q1q2_information.csv), [supervised](supervised_readouts.csv), [per-unit reuse](unit_reuse.csv), [manifest](manifest.json).

- Complete readable tables: [A information](report_details/A_information.md), [B clustering](report_details/B_clustering.md), [C supervised](report_details/C_supervised.md), [D reuse](report_details/D_unit_reuse.md).

- Derived CSVs: [clustering summaries](report_details/clustering_summary.csv), [information summaries](report_details/information_summary.csv), [supervised summaries](report_details/supervised_summary.csv), [recall](report_details/per_class_recall.csv), [fold class support/recall](report_details/fold_class_support_recall.csv), [folds](report_details/folds.csv), [preprocessing](report_details/preprocessing.csv), [reuse per fold/seed](report_details/reuse_fold_seed_summary.csv).

- Regenerate this report without refitting: `python script/exp2round/q1q2/report_complete.py` in the existing SMQ environment. [Audit and source hashes](report_details/report_audit.json) identify the exact input files.
