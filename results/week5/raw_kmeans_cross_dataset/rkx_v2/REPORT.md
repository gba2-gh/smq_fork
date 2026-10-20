# Raw K-means tokens across datasets — rkx_v2

Manifest: `manifest.json` (status `finished`, implementation `ddbf2c205499cbdb`). Cells with verified training: 18/18; unfinished units: 0 (`not_run.csv`).

Frozen HuGaDB configuration on every dataset (raw patches, K-means-512, reference contextualizer, guard 0, 5,000 updates, outer cap 50). Only window, frame rate and class count differ by dataset. Tokenizer and OT fitting pool all recordings, validation included, so decoding is transductive; HuGaDB settings were chosen on HuGaDB development scores; LARa is not untouched (it informed earlier choices).

## Gates

| dataset | labels = saved SMQ ground truth | max |metric diff| vs saved SMQ scores | passed |
|---|---|---|---|
| babel1 | True | 0.0026 | True |
| babel2 | True | 0.0043 | True |
| babel3 | True | 0.0020 | True |
| hugadb | True | 0.0049 | True |
| lara | True | 0.0041 | True |

**Gate 1 — tokenizer port vs qic_v3:** passed

**Gate 2 — pipeline port vs ctx_cal_v1:** passed (F1@50 50.00 vs 50.00)

## F1@50 by dataset (mean ± sd over three paired seeds)

Validation-selected checkpoint (primary) and final 5,000-update checkpoint (secondary), last final-solve rung (4,000 in a full run). SMQ is the released checkpoint re-scored locally; the published numbers use different protocols (end-to-end training, 1-second patches, K = number of classes).

| dataset | normalization | seeds | selected | selected per seed | final | SMQ (local) | ASOT (pub.) | HiST-VQ (pub.) | MASQ (pub.) |
|---|---|---|---|---|---|---|---|---|---|
| hugadb | global | 3 | 46.23 ± 3.30 | 50.00 / 44.83 / 43.85 | 45.73 ± 2.55 | 24.30 | 3.00 | 28.30 | 39.50 |
| babel1 | global | 3 | 14.25 ± 1.44 | 15.89 / 13.21 / 13.64 | 14.48 ± 2.11 | 22.38 | 24.50 | 22.70 | 21.00 |
| babel2 | global | 3 | 15.09 ± 3.46 | 16.68 / 11.12 / 17.46 | 17.58 ± 1.37 | 27.52 | 23.40 | 30.40 | 28.40 |
| babel3 | global | 3 | 15.32 ± 0.65 | 14.78 / 16.04 / 15.15 | 12.01 ± 1.76 | 19.35 | 14.30 | 24.10 | 24.50 |
| lara | global | 3 | 5.06 ± 0.48 | 5.03 / 4.60 / 5.55 | 5.14 ± 0.53 | 16.39 | 5.70 | 19.30 | 22.20 |
| lara | per_recording | 3 | 7.13 ± 0.40 | 7.32 / 7.41 / 6.67 | 7.08 ± 0.41 | 16.39 | 5.70 | 19.30 | 22.20 |

## Declared reading

**does not transfer** (selected checkpoint, last rung, mean of three seeds).

- LARa groups above SMQ (mean and every seed): 0/2; BABEL subsets above SMQ: 0/3; LARa below SMQ under both normalizations: True; BABEL subsets below SMQ (mean): 3/3

The rule was fixed before scoring; an intermediate outcome is reported as mixed, and the normalization factor and checkpoint policy are shown as they came out.

## All metrics (mean over seeds, last rung)

| dataset | norm | checkpoint | MoF | Edit | F1@10 | F1@25 | F1@50 | pred segments / rec. | GT segments / rec. | outer-capped fits |
|---|---|---|---|---|---|---|---|---|---|---|
| hugadb | global | selected | 48.88 | 42.68 | 50.21 | 48.75 | 46.23 | 9.0 | 10.5 | 3 |
| hugadb | global | final | 48.12 | 42.06 | 49.89 | 48.40 | 45.73 | 9.0 | 10.5 | 3 |
| babel1 | global | selected | 31.32 | 30.63 | 35.06 | 26.61 | 14.25 | 2.6 | 5.1 | 3 |
| babel1 | global | final | 31.64 | 30.81 | 35.33 | 27.01 | 14.48 | 2.6 | 5.1 | 3 |
| babel2 | global | selected | 32.60 | 29.46 | 28.91 | 22.70 | 15.09 | 1.9 | 3.7 | 3 |
| babel2 | global | final | 30.51 | 31.87 | 32.31 | 25.34 | 17.58 | 2.9 | 3.7 | 3 |
| babel3 | global | selected | 32.12 | 28.08 | 30.98 | 23.62 | 15.32 | 2.7 | 3.7 | 3 |
| babel3 | global | final | 27.12 | 28.82 | 30.34 | 22.18 | 12.01 | 3.6 | 3.7 | 3 |
| lara | global | selected | 20.23 | 22.50 | 17.23 | 11.59 | 5.06 | 14.8 | 32.1 | 3 |
| lara | global | final | 20.15 | 22.64 | 17.23 | 11.64 | 5.14 | 14.8 | 32.1 | 3 |
| lara | per_recording | selected | 21.25 | 26.88 | 21.27 | 15.13 | 7.13 | 18.4 | 32.1 | 3 |
| lara | per_recording | final | 21.16 | 26.84 | 21.25 | 15.16 | 7.08 | 18.4 | 32.1 | 3 |

## Per-cell diagnostics

| cell | selected step | val loss | valid-token fraction | eligible / update | occupied codes | perplexity | singleton codes | MI code-action | MI code-subject |
|---|---|---|---|---|---|---|---|---|---|
| hugadb__global__seed111 | 4100 | 1.810 | 0.971 | 2289 | 512 | 179 | 2 | 1.400 | 1.633 |
| hugadb__global__seed222 | 4900 | 1.815 | 0.971 | 2289 | 512 | 177 | 1 | 1.388 | 1.622 |
| hugadb__global__seed1538574472 | 5000 | 1.830 | 0.971 | 2289 | 512 | 173 | 4 | 1.402 | 1.560 |
| babel1__global__seed111 | 4800 | 3.089 | 0.287 | 693 | 512 | 310 | 5 | 0.507 | n/a |
| babel1__global__seed222 | 4600 | 3.019 | 0.287 | 693 | 512 | 303 | 7 | 0.509 | n/a |
| babel1__global__seed1538574472 | 4300 | 3.056 | 0.287 | 693 | 512 | 305 | 3 | 0.504 | n/a |
| babel2__global__seed111 | 1100 | 3.856 | 0.308 | 675 | 512 | 298 | 15 | 0.826 | n/a |
| babel2__global__seed222 | 800 | 3.750 | 0.308 | 675 | 512 | 290 | 17 | 0.826 | n/a |
| babel2__global__seed1538574472 | 800 | 4.336 | 0.308 | 675 | 512 | 297 | 13 | 0.828 | n/a |
| babel3__global__seed111 | 600 | 4.710 | 0.453 | 928 | 512 | 323 | 19 | 0.731 | n/a |
| babel3__global__seed222 | 700 | 4.768 | 0.453 | 928 | 512 | 336 | 9 | 0.748 | n/a |
| babel3__global__seed1538574472 | 900 | 4.868 | 0.453 | 928 | 512 | 328 | 20 | 0.734 | n/a |
| lara__global__seed111 | 4700 | 1.855 | 1.000 | 2388 | 512 | 398 | 0 | 0.520 | 1.340 |
| lara__global__seed222 | 4900 | 1.911 | 1.000 | 2388 | 512 | 403 | 0 | 0.525 | 1.338 |
| lara__global__seed1538574472 | 4900 | 1.893 | 1.000 | 2388 | 512 | 402 | 0 | 0.515 | 1.331 |
| lara__per_recording__seed111 | 5000 | 2.464 | 1.000 | 2388 | 512 | 345 | 0 | 0.489 | 0.711 |
| lara__per_recording__seed222 | 4600 | 2.547 | 1.000 | 2388 | 512 | 345 | 0 | 0.487 | 0.709 |
| lara__per_recording__seed1538574472 | 4600 | 2.541 | 1.000 | 2388 | 512 | 338 | 0 | 0.476 | 0.702 |

The subject probe is a label-aware diagnostic of what the codes carry; it never entered selection.

## Missing cells and limits

All planned units completed.

Limits: three paired seeds (each drives the vocabulary and the contextualizer); the outer OT fit is budget-capped in essentially every fit; BABEL clips are short (mostly padding in a 128-token context); published numbers come from different protocols. This run does not tune anything on LARa or BABEL.
