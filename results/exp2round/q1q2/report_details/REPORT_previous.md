# Q1/Q2 motion-unit diagnostics

1. Both released-checkpoint provenance gates passed for HuGaDB and LARa.
2. Same-checkpoint quantization agreement was 1.000 and 1.000, respectively.
3. Historical prediction agreement was 0.999895 for HuGaDB and 1.000000 for LARa.
4. D7 MoF reproduced 44.85 for HuGaDB and 41.66 for LARa.
5. Information diagnostics use count-preserving global and within-action permutation references.
6. Pooled clustering results use oracle ground-truth segment boundaries and collection-wide unsupervised fitting.
7. Recording- and subject-grouped readouts use four folds with fitting-only preprocessing, vocabularies, temperatures and classifiers.
8. Hard, soft and continuous representations are reported separately; soft results include 0.5x, 1x and 2x temperatures.
9. Seed spreads below are population SDs for initialization sensitivity, while fold results remain individually available.
10. There are 40 invalid supervised rows due to the prescribed convergence rule; details are retained in not_run.csv.

## Provenance and gates

| Dataset | Checkpoint SHA-256 | Recordings | Frames | Quantization agreement | Historical agreement | D7 MoF | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hugadb | 22dfd5639c718177c12bc48debf67371c4bf7ca64af2d7de4e83ebacae55d7d2 | 364.000000 | 1137986.000000 | 1.000000 | 0.999895 | 44.850000 | True |
| lara | a45392ad30e27bda400adcab541aba087e4ae56b68837547cc65ca77ff5bfa65 | 439.000000 | 2634000.000000 | 1.000000 | 1.000000 | 41.660000 | True |

## Full-patch latent information

| Dataset | Ku | I(U;A) mean | seed SD | seeds |
| --- | --- | --- | --- | --- |
| hugadb | 10.0000 | 0.9846 | 0.0042 | 3.0000 |
| hugadb | 20.0000 | 1.0155 | 0.0063 | 3.0000 |
| hugadb | 50.0000 | 1.1575 | 0.0080 | 3.0000 |
| hugadb | 100.0000 | 1.2885 | 0.0020 | 3.0000 |
| hugadb | 500.0000 | 1.5095 | 0.0059 | 3.0000 |
| hugadb | 1000.0000 | 1.5691 | 0.0033 | 3.0000 |
| lara | 10.0000 | 0.2932 | 0.0068 | 3.0000 |
| lara | 20.0000 | 0.3454 | 0.0060 | 3.0000 |
| lara | 50.0000 | 0.4321 | 0.0038 | 3.0000 |
| lara | 100.0000 | 0.5246 | 0.0031 | 3.0000 |
| lara | 500.0000 | 0.7070 | 0.0015 | 3.0000 |
| lara | 1000.0000 | 0.7983 | 0.0031 | 3.0000 |

## Full-patch pooled hard clustering

| Dataset | Ku | MoF mean | seed SD | seeds |
| --- | --- | --- | --- | --- |
| hugadb | 10.00 | 49.23 | 0.36 | 3.00 |
| hugadb | 20.00 | 46.26 | 1.19 | 3.00 |
| hugadb | 50.00 | 42.38 | 4.13 | 3.00 |
| hugadb | 100.00 | 42.02 | 1.77 | 3.00 |
| hugadb | 500.00 | 36.55 | 1.43 | 3.00 |
| hugadb | 1000.00 | 34.95 | 0.55 | 3.00 |
| lara | 10.00 | 28.79 | 0.02 | 3.00 |
| lara | 20.00 | 23.21 | 1.63 | 3.00 |
| lara | 50.00 | 31.43 | 2.18 | 3.00 |
| lara | 100.00 | 32.59 | 1.13 | 3.00 |
| lara | 500.00 | 40.03 | 0.59 | 3.00 |
| lara | 1000.00 | 38.05 | 2.55 | 3.00 |

## Full-patch subject-grouped hard readout

| Dataset | Ku | frame MoF mean | seed SD | balanced accuracy mean | seed SD (BA) |
| --- | --- | --- | --- | --- | --- |
| hugadb | 10.00 | 41.69 | 0.45 | 54.78 | 0.25 |
| hugadb | 20.00 | 41.26 | 0.70 | 53.50 | 0.55 |
| hugadb | 50.00 | 57.82 | 0.61 | 62.58 | 1.08 |
| hugadb | 100.00 | 66.08 | 0.36 | 69.88 | 0.50 |
| hugadb | 500.00 | 79.19 | 0.61 | 77.76 | 0.55 |
| hugadb | 1000.00 | 79.16 | 0.20 | 78.38 | 0.30 |
| lara | 10.00 | 26.62 | 0.59 | 40.30 | 0.53 |
| lara | 20.00 | 32.35 | 0.86 | 43.47 | 0.22 |
| lara | 50.00 | 39.14 | 0.32 | 46.66 | 0.85 |
| lara | 100.00 | 43.10 | 1.06 | 49.14 | 0.49 |
| lara | 500.00 | 47.27 | 0.52 | 52.09 | 0.15 |
| lara | 1000.00 | 50.56 | 0.62 | 53.45 | 0.32 |

## Unit reuse

| Dataset | Window | Ku | fraction mean | fold/seed population SD | unused units mean |
| --- | --- | --- | --- | --- | --- |
| hugadb | 15.0000 | 10.0000 | 1.0000 | 0.0000 | 0.0000 |
| hugadb | 15.0000 | 20.0000 | 0.9375 | 0.0681 | 0.6667 |
| hugadb | 15.0000 | 50.0000 | 0.8833 | 0.1385 | 1.5000 |
| hugadb | 15.0000 | 100.0000 | 0.7650 | 0.2123 | 6.3333 |
| hugadb | 15.0000 | 500.0000 | 0.4407 | 0.1733 | 116.1667 |
| hugadb | 15.0000 | 1000.0000 | 0.3019 | 0.1284 | 354.9167 |
| hugadb | 60.0000 | 10.0000 | 0.9667 | 0.0471 | 0.1667 |
| hugadb | 60.0000 | 20.0000 | 0.9292 | 0.0803 | 0.7500 |
| hugadb | 60.0000 | 50.0000 | 0.8050 | 0.1896 | 2.0833 |
| hugadb | 60.0000 | 100.0000 | 0.6208 | 0.2057 | 9.8333 |
| hugadb | 60.0000 | 500.0000 | 0.2775 | 0.1240 | 200.0000 |
| hugadb | 60.0000 | 1000.0000 | 0.1630 | 0.0682 | 530.5000 |
| lara | 12.0000 | 10.0000 | 1.0000 | 0.0000 | 0.0000 |
| lara | 12.0000 | 20.0000 | 0.9708 | 0.0380 | 0.4167 |
| lara | 12.0000 | 50.0000 | 0.9100 | 0.0436 | 1.1667 |
| lara | 12.0000 | 100.0000 | 0.8500 | 0.0566 | 3.3333 |
| lara | 12.0000 | 500.0000 | 0.5788 | 0.0723 | 56.2500 |
| lara | 12.0000 | 1000.0000 | 0.4462 | 0.0649 | 178.0833 |
| lara | 50.0000 | 10.0000 | 0.9667 | 0.0471 | 0.0000 |
| lara | 50.0000 | 20.0000 | 0.9125 | 0.0650 | 0.6667 |
| lara | 50.0000 | 50.0000 | 0.8633 | 0.0672 | 1.7500 |
| lara | 50.0000 | 100.0000 | 0.7875 | 0.0595 | 5.5833 |
| lara | 50.0000 | 500.0000 | 0.4482 | 0.0570 | 95.0000 |
| lara | 50.0000 | 1000.0000 | 0.3181 | 0.0489 | 295.5833 |

## Protocol disclosures

Ground-truth action labels define segments, information variables and conditional permutations; they are also used for Hungarian and oracle many-to-one evaluation mappings and for supervised fitting/evaluation. Subject identifiers define grouping and subject diagnostics. No evaluation label selected an unsupervised hyperparameter or removed a result. Pooled diagnostics are oracle-boundary and transductive. Cross-validation preprocessing, vocabulary, temperature and readout use fitting recordings only, while their frozen encoders remain transductive. Raw-feature runs have no encoder exposure, but their pooled fitting remains transductive.

HuGaDB has 10 observed action classes and 18 subjects with wearable IMU channels; LARa has 8 observed action classes and 16 subjects with optical motion-capture skeleton channels. Every HuGaDB-LARa comparison is descriptive because class count, subject count and modality differ. HuGaDB does not establish body-shape effects. The recording-versus-subject fold gap is a protocol difference and does not isolate prior exposure to a person. Separate within-subject Hungarian mappings add oracle flexibility.

## Open questions

- How much of the observed action and subject association persists under evaluation designs that avoid oracle segment boundaries?
- Which dataset or modality properties account for differences between HuGaDB and LARa?
- How stable are the measured patterns across checkpoints, independent samples and non-transductive encoders?
- Why do hard, soft and continuous readouts differ at particular vocabulary sizes and window scales?

## What these numbers cannot establish

These diagnostics do not identify a unique bottleneck, establish causal effects, determine which component dominates, define the project contribution, or prescribe the next project step. Permutation distributions are descriptive references rather than uncertainty estimates for independent temporal samples. Seed variation measures initialization sensitivity rather than sampling uncertainty. Supervised readouts are diagnostics rather than information-theoretic ceilings.

## Not run

| status | reason | rows |
| --- | --- | --- |
| invalid | logistic regression did not converge | 40.00 |

See `not_run.csv` for cell-level details.

All planned cells were executed. The eight-hour elapsed deadline was exceeded because execution was paused and later resumed on the user's instruction to finish; the manifest records the original deadline and this deviation.
