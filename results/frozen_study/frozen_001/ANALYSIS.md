# Frozen study: analysis of frozen_001

Analysis date: 27 September 2026. Based on saved results and inspection of the implementation; no models were refit and no experiments were launched for this analysis.

## Main conclusion

The tested PCA averaging controls do not explain the HuGaDB contextual segmentation gain. HuGaDB retains a large gap between discovered and label-derived prototypes. LARa remains weak under contextual cosine-prototype transport, but its normalized contextual features remain competitive with PCA under a supervised linear readout. The results support investigating action discovery and representation/readout compatibility; they do not establish that LARa's action information has irreversibly disappeared or that adaptive boundaries are the necessary next mechanism.

## Coverage and checks

- All 98 planned jobs are recorded as complete: 92 segmentation jobs, four probe jobs, baseline verification and the timing pilot.
- All 368 segmentation rows are unique job/rung pairs; each segmentation job has four rungs. No invalid recording solves are reported.
- All 92 segmentation artifacts exist, have the expected 364 HuGaDB or 439 LARa recording identities, finite prototypes and predictions for all four rungs.
- All 180 baseline metric checks (36 prediction sets, five metrics each) have exactly zero discrepancy. These checks rescore historical predictions; they do not establish that newly solved predictions reproduce the historical initialization protocol.
- No recorded objective increases occur between successive rungs within a job. Objective values across representations or different fixed prototypes are not performance rankings.
- The manifest records a nine-hour allowance. The launch at 03:18:11 and last completion at 07:22:01 indicate approximately 4 hours 4 minutes elapsed.
- There are no omitted jobs. Completion is operational completion, not numerical convergence: all 36 newly fitted prototype models reached their outer caps.

HuGaDB has 10 observed action classes and 18 subjects with IMU inputs. LARa has eight classes and 16 subjects with optical motion-capture inputs. Differences between datasets cannot isolate modality, subject diversity or class difficulty.

## 1. Final results with fixed saved prototypes

F1@50 at the 4,000-step cumulative final-solve rung. Values are mean and population SD over three pipeline seeds, except PCA oracle results, which use seed 111 only. Oracle means are not an unsupervised method or a mathematical ceiling. The contextual oracle's identity and Hungarian mappings give the same numbers here.

| Dataset / normalization | PCA, discovered | Context, discovered | Context, oracle | PCA, oracle (one seed, identity mapping) |
|---|---:|---:|---:|---:|
| HuGaDB / raw | 4.87 +/- 2.29 | 41.11 +/- 3.95 | 64.17 +/- 3.07 | 30.22 |
| HuGaDB / norm | 4.70 +/- 0.58 | 34.24 +/- 5.46 | 62.52 +/- 0.84 | 22.21 |
| LARa / raw | 17.89 +/- 1.00 | 8.16 +/- 0.55 | 15.90 +/- 1.40 | 28.93 |
| LARa / norm | 17.93 +/- 0.28 | 11.57 +/- 0.41 | 22.14 +/- 0.70 | 29.15 |

The paired contextual oracle-minus-discovered gaps are +23.05 points on HuGaDB/raw and +28.28 on HuGaDB/norm. All six HuGaDB seed-level gaps exceed 20 points. LARa gaps average +7.74 raw and +10.57 normalized. Thus better class-aware prototypes help both datasets, but they do not make contextual LARa competitive with its PCA oracle.

Class means are optimal for a particular within-class cosine-distortion calculation, not for classification accuracy or segmentation F1. These results do not exclude richer emission models, different metrics, multiple prototypes per action, or nonlinear readouts.

Other metrics for discovered contextual prototypes:

| Dataset / normalization | MoF | Edit | F1@10 | F1@25 | F1@50 | Predicted/GT runs |
|---|---:|---:|---:|---:|---:|---:|
| HuGaDB / raw | 41.67 | 39.35 | 45.99 | 44.45 | 41.11 | 0.85 |
| HuGaDB / norm | 39.74 | 40.75 | 43.89 | 39.44 | 34.24 | 0.94 |
| LARa / raw | 20.38 | 23.73 | 21.26 | 15.64 | 8.16 | 0.48 |
| LARa / norm | 24.58 | 28.65 | 26.05 | 20.27 | 11.57 | 0.67 |

## 2. Pooling does not match learned context on HuGaDB

Matched seed-111, discovered-prototype F1@50 at the final rung:

| Dataset / normalization | PCA | Pool h=2 | Pool h=8 | Pool h=32 | Pool h=64 | Context |
|---|---:|---:|---:|---:|---:|---:|
| HuGaDB / raw | 3.30 | 21.57 | 13.53 | 4.87 | 2.52 | 36.02 |
| HuGaDB / norm | 5.28 | 23.90 | 20.39 | 9.66 | 7.49 | 31.07 |
| LARa / raw | 19.30 | 17.89 | 10.56 | 1.73 | 0.86 | 8.71 |
| LARa / norm | 18.22 | 17.75 | 9.80 | 2.83 | 1.87 | 11.72 |

Here h is the half-width, so the shortest averaging window spans five token windows: 1.25 seconds on HuGaDB and 1.20 seconds on LARa. Averaging is frame-weighted, centered, truncated at recording edges and followed by L2 normalization. The longest window spans 32.25/30.96 seconds respectively.

The best tested HuGaDB pooling width is h=2, but it remains 14.45/raw and 7.17/norm points below matched contextual results. Thus the declared within-three-points pooling rule is not met. Even with oracle prototypes, h=2 reaches 37.72/34.78 versus contextual 61.84/61.35 at seed 111.

Only h=32 and h=64 pooling were replicated across three seeds; their poor performance persists. The stronger h=2 and h=8 controls have one seed. Choosing the best width by evaluation score is an exploratory envelope, not an independently selected deployable setting. This excludes the tested box averages as a complete explanation of the HuGaDB effect, not every possible pooling operator or width.

## 3. More final iterations do not explain the dataset contrast

Mean contextual F1@50 with saved prototypes:

| Dataset / normalization | Original Stage C | Cold start, 500 steps | Cold start, 4,000 steps |
|---|---:|---:|---:|
| HuGaDB / raw | 37.00 | 40.82 | 41.11 |
| HuGaDB / norm | 29.16 | 34.06 | 34.24 |
| LARa / raw | 7.77 | 8.19 | 8.16 |
| LARa / norm | 11.34 | 11.61 | 11.57 |

This is an important protocol distinction: Stage C initializes final decoding from the transport state retained during alternating fitting. This study starts from the independent coupling and then continues that solve across rungs. Most of the HuGaDB score difference from Stage C is already present at 500 steps. It must not be reported as a benefit of increasing the iteration cap alone. Initialization sensitivity is plausible for this nonconvex objective; the historical warm-start path was not extended to 4,000 steps here.

Within this study, all 92 job-level Hungarian F1@50 changes from 500 to 4,000 steps are at most one point in absolute value. For saved contextual prototypes, the mean changes are +0.29, +0.18, -0.03 and -0.04 points in the table's order. Many residuals and convergence counts improve substantially, while predictions and scores change little.

Convergence is still incomplete. At the final rung, contextual oracle solves remain capped for an average 231/439 LARa/raw recordings and 155/439 LARa/norm recordings. Score stability over this ladder is not proof of a global optimum.

The four extended outer fits use seed 111, raw inputs only, and refit from initialization for 150 outer iterations. Compared with the saved 50-iteration prototypes evaluated under the same final cold-start protocol:

| Dataset / representation | Saved fit F1@50 | 150-iteration refit F1@50 |
|---|---:|---:|
| HuGaDB / PCA | 3.30 | 3.28 |
| HuGaDB / context | 36.02 | 35.18 |
| LARa / PCA | 19.30 | 18.35 |
| LARa / context | 8.71 | 8.35 |

All remain outer-capped. This limited sensitivity check finds no improvement from simply extending the existing alternating procedure.

## 4. Supervised probes qualify the representation diagnosis

Arithmetic means of four subject-grouped fold accuracies, in percent, for standardized logistic regression:

| Dataset / normalization | PCA | Pool h=2 | Context |
|---|---:|---:|---:|
| HuGaDB / raw | 68.71 | 66.48 | 61.17 |
| HuGaDB / norm | 58.55 | 65.24 | 73.99 |
| LARa / raw | 60.80 | 63.96 | 44.54 |
| LARa / norm | 56.76 | 59.95 | 58.32 |

These are duration-weighted majority-window classification accuracies, not actual frame MoF or segment F1. Windows crossing boundaries receive their majority label for the entire window weight. The table averages fold scores, not pooled out-of-fold predictions.

Raw LARa context is substantially weaker than PCA on all four fold accuracies. Normalized LARa context is broadly comparable to PCA and slightly below the shortest pooling control. Therefore the blanket conclusion that LARa's contextual representation has lost action identity is not supported. Its cosine-centroid transport performance and supervised linear accessibility diverge.

HuGaDB also shows a divergence: raw context improves segmentation despite lower linear-probe accuracy than PCA, whereas normalized context improves both. A representation's utility depends on the readout and temporal objective, not a single universal quality ranking.

Probe limitations:

- Only readout fitting is subject-disjoint. The encoder, PCA/vocabulary and contextualizer have transductive exposure; this is not an inductive transfer test.
- Standardization fits within each training fold. C=1 is fixed; there is no inner hyperparameter selection.
- Training windows are subsampled to 50,000 separately as a shared RNG advances through representations. The selected training examples are not identical across representations.
- The runner saves no logistic-regression iteration counts, convergence status, coefficients or out-of-fold predictions. A successful job does not certify optimizer convergence. Probe comparisons remain provisional until that numerical omission is checked; it does not invalidate the separately saved transport results.

## 5. Interpretation of the declared rules

- Q1, pooling within three HuGaDB F1@50 points in both normalizations: not met in the complete seed-111 width comparison.
- Q2, a ten-point oracle/discovery gap for at least one representation on both datasets: met under the literal condition-level reading, including normalized LARa PCA (29.15 minus 18.22 = 10.93). It is not uniform: raw LARa PCA has a 9.63-point gap, and raw contextual LARa has smaller gaps. Do not turn the existential threshold into a universal bottleneck claim.
- Q3, contextual LARa oracle transport below PCA oracle transport in both normalizations: met. It describes the prescribed cosine-prototype decoder, not all information accessible from those representations.

## 6. Audit limitations and practical next step

No observed missing-job, nonfinite-prototype or invalid-transport failure explains the main pattern. The historical 1-indexed class-name error is avoided here by using actual class IDs. Remaining reporting/provenance issues include absent source SHA-256 values in the manifest despite the instructions requesting them, identical start/end timestamps written at job completion, and the unaudited probe convergence noted above. Oracle identity and Hungarian metrics can differ; they must remain labeled separately.

The study did not test BABEL, new pretrained encoders, FSQ, an adaptive temporal kernel, multiple prototypes per action or a new discovery objective. It supports neither a cross-dataset transfer claim nor an FSQ advantage.

The next methodological target should be action discovery and its compatibility with the feature geometry, keeping the encoder fixed initially. HuGaDB's oracle headroom survives the pooling and iteration checks. LARa's normalized probes motivate checking the decoder geometry before assuming that a new encoder or more data is necessary. First close the small evidence gaps: audit probe convergence with the same objective, replicate the h=2 control, and preserve cold/warm decoding as explicitly separate protocols. These are verification tasks, not another unrestricted search. A proposed discovery method must demonstrate gains beyond initialization changes and simple metric corrections before being treated as the paper's novelty.

Sources: `rows.csv`, `fits.csv`, `probe.csv`, `baseline_check.csv`, `status.csv`, `manifest.json`, saved NPZ artifacts, and `script/action_transport/frozen_study.py` plus its dependencies.
