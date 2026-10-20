# HuGaDB-focused follow-up (velocity-term variants): masq_hu1

Status `finished`. Variant code `330faa72efd1692a`, training code `14712f110377f9ea`. References S0, M-pub and M-JL are read from `masq_v1` (same code, settings and seeds). Global Hungarian mapping, final epoch, MoF and F1@50 co-primary; bar: mean difference >= 3 with every seed in the same direction.

**Decision rule (declared before the run):** The MASQ track continues only if a variant passes the bar over M-JL on both MoF and F1@50 (mean difference >= 3 with all three seeds in the same direction). Otherwise it stops on HuGaDB: no other dataset is run.

## Cells (HuGaDB, global Hungarian; seed mean ± population SD)

| arm | from | n seeds | MoF | Edit | F1@10 | F1@25 | F1@50 | MoF per seed | F1@50 per seed |
|---|---|---|---|---|---|---|---|---|---|
| S0 | masq_v1 | 3 | 40.72 ± 1.54 | 32.78 ± 2.58 | 34.77 ± 2.49 | 27.68 ± 1.78 | 22.04 ± 1.48 | 42.7 / 40.5 / 39.0 | 24.1 / 21.4 / 20.6 |
| M-pub | masq_v1 | 3 | 39.74 ± 1.52 | 34.41 ± 1.99 | 36.89 ± 1.27 | 28.95 ± 1.42 | 21.12 ± 2.42 | 40.7 / 41.0 / 37.6 | 22.3 / 23.3 / 17.7 |
| M-JL | masq_v1 | 3 | 45.24 ± 0.92 | 37.36 ± 1.00 | 42.59 ± 0.40 | 35.73 ± 0.83 | 27.56 ± 1.56 | 44.9 / 44.3 / 46.5 | 27.1 / 25.9 / 29.7 |
| V1 | masq_hu1 | 3 | 41.76 ± 5.14 | 36.53 ± 4.40 | 40.12 ± 6.17 | 31.78 ± 5.91 | 22.83 ± 5.62 | 35.1 / 47.6 / 42.6 | 16.7 / 30.3 / 21.5 |
| V2a | masq_hu1 | 3 | 42.84 ± 5.87 | 36.09 ± 4.86 | 40.85 ± 7.35 | 31.20 ± 7.98 | 20.80 ± 7.83 | 50.3 / 35.9 / 42.3 | 31.9 / 15.2 / 15.3 |
| V2b | masq_hu1 | 3 | 45.03 ± 0.99 | 38.83 ± 4.74 | 42.10 ± 4.50 | 34.26 ± 4.69 | 25.26 ± 4.63 | 44.0 / 46.3 / 44.8 | 18.8 / 29.2 / 27.8 |

Published MASQ (single run): MoF 52.3, Edit 45.1, F1@10 52.1, F1@25 47.0, F1@50 39.5.

## Calibration (label-free, at initialization: weight x term = rho x reconstruction)

| arm | seed | term | rho | reconstruction(init) | term(init) | weight |
|---|---|---|---|---|---|---|
| V1 | 111 | coord_scalefree | 0.0016 | 6.201e+05 | 1.808e+08 | 5.489e-06 |
| V1 | 222 | coord_scalefree | 0.0016 | 6.201e+05 | 1.808e+08 | 5.489e-06 |
| V1 | 1538574472 | coord_scalefree | 0.0016 | 6.201e+05 | 1.808e+08 | 5.489e-06 |
| V2a | 111 | latent_smooth | 0.01 | 6.201e+05 | 0.2726 | 2.275e+04 |
| V2a | 222 | latent_smooth | 0.01 | 6.201e+05 | 0.2796 | 2.218e+04 |
| V2a | 1538574472 | latent_smooth | 0.01 | 6.201e+05 | 0.268 | 2.314e+04 |
| V2b | 111 | latent_smooth | 0.1 | 6.201e+05 | 0.2726 | 2.275e+05 |
| V2b | 222 | latent_smooth | 0.1 | 6.201e+05 | 0.2796 | 2.218e+05 |
| V2b | 1538574472 | latent_smooth | 0.1 | 6.201e+05 | 0.268 | 2.314e+05 |

## Comparisons (A minus B, paired by seed)

- **V1 − M-JL** (does the extra term add anything to joint dropout alone? (decision comparison)): MoF -3.48 (-9.9 / +3.3 / -3.9) -> within the bar: comparable performance, not equivalence; F1@50 -4.74 (-10.4 / +4.3 / -8.2) -> within the bar: comparable performance, not equivalence
- **V1 − S0** (total effect over our SMQ retrain): MoF +1.05 (-7.6 / +7.1 / +3.6) -> within the bar: comparable performance, not equivalence; F1@50 +0.78 (-7.4 / +8.9 / +0.9) -> within the bar: comparable performance, not equivalence
- **V1 − M-pub** (against MASQ's published-weight velocity loss (scale hypothesis)): MoF +2.03 (-5.6 / +6.6 / +5.0) -> within the bar: comparable performance, not equivalence; F1@50 +1.71 (-5.6 / +7.0 / +3.8) -> within the bar: comparable performance, not equivalence
- **V2a − M-JL** (does the extra term add anything to joint dropout alone? (decision comparison)): MoF -2.41 (+5.3 / -8.4 / -4.2) -> within the bar: comparable performance, not equivalence; F1@50 -6.77 (+4.8 / -10.7 / -14.4) -> within the bar: comparable performance, not equivalence
- **V2a − S0** (total effect over our SMQ retrain): MoF +2.12 (+7.5 / -4.5 / +3.4) -> within the bar: comparable performance, not equivalence; F1@50 -1.24 (+7.8 / -6.2 / -5.3) -> within the bar: comparable performance, not equivalence
- **V2a − M-pub** (against MASQ's published-weight velocity loss (scale hypothesis)): MoF +3.10 (+9.6 / -5.0 / +4.7) -> within the bar: comparable performance, not equivalence; F1@50 -0.32 (+9.6 / -8.1 / -2.4) -> within the bar: comparable performance, not equivalence
- **V2b − M-JL** (does the extra term add anything to joint dropout alone? (decision comparison)): MoF -0.22 (-1.0 / +2.0 / -1.7) -> within the bar: comparable performance, not equivalence; F1@50 -2.30 (-8.3 / +3.2 / -1.8) -> within the bar: comparable performance, not equivalence
- **V2b − S0** (total effect over our SMQ retrain): MoF +4.31 (+1.2 / +5.9 / +5.8) -> passes the bar; F1@50 +3.22 (-5.3 / +7.8 / +7.2) -> within the bar: comparable performance, not equivalence
- **V2b − M-pub** (against MASQ's published-weight velocity loss (scale hypothesis)): MoF +5.29 (+3.3 / +5.4 / +7.2) -> passes the bar; F1@50 +4.14 (-3.5 / +5.9 / +10.1) -> within the bar: comparable performance, not equivalence

### Every metric

| A − B | metric | Δ (per seed) | n | reading |
|---|---|---|---|---|
| V1 − M-JL | MoF | -3.48 (-9.9 / +3.3 / -3.9) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-JL | Edit | -0.83 (-5.5 / +5.4 / -2.3) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-JL | F1@10 | -2.46 (-8.6 / +5.3 / -4.0) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-JL | F1@25 | -3.95 (-9.8 / +4.5 / -6.6) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-JL | F1@50 | -4.74 (-10.4 / +4.3 / -8.2) | 3 | within the bar: comparable performance, not equivalence |
| V1 − S0 | MoF | +1.05 (-7.6 / +7.1 / +3.6) | 3 | within the bar: comparable performance, not equivalence |
| V1 − S0 | Edit | +3.76 (-5.2 / +10.3 / +6.1) | 3 | within the bar: comparable performance, not equivalence |
| V1 − S0 | F1@10 | +5.36 (-4.5 / +13.7 / +6.8) | 3 | within the bar: comparable performance, not equivalence |
| V1 − S0 | F1@25 | +4.10 (-4.7 / +12.5 / +4.5) | 3 | within the bar: comparable performance, not equivalence |
| V1 − S0 | F1@50 | +0.78 (-7.4 / +8.9 / +0.9) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-pub | MoF | +2.03 (-5.6 / +6.6 / +5.0) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-pub | Edit | +2.12 (-5.9 / +8.1 / +4.2) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-pub | F1@10 | +3.24 (-5.2 / +11.9 / +3.0) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-pub | F1@25 | +2.83 (-4.7 / +9.8 / +3.4) | 3 | within the bar: comparable performance, not equivalence |
| V1 − M-pub | F1@50 | +1.71 (-5.6 / +7.0 / +3.8) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-JL | MoF | -2.41 (+5.3 / -8.4 / -4.2) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-JL | Edit | -1.27 (+6.1 / -5.2 / -4.7) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-JL | F1@10 | -1.73 (+9.1 / -8.7 / -5.5) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-JL | F1@25 | -4.53 (+7.3 / -9.9 / -11.0) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-JL | F1@50 | -6.77 (+4.8 / -10.7 / -14.4) | 3 | within the bar: comparable performance, not equivalence |
| V2a − S0 | MoF | +2.12 (+7.5 / -4.5 / +3.4) | 3 | within the bar: comparable performance, not equivalence |
| V2a − S0 | Edit | +3.32 (+6.4 / -0.2 / +3.8) | 3 | within the bar: comparable performance, not equivalence |
| V2a − S0 | F1@10 | +6.09 (+13.2 / -0.3 / +5.4) | 3 | within the bar: comparable performance, not equivalence |
| V2a − S0 | F1@25 | +3.52 (+12.4 / -1.9 / +0.1) | 3 | within the bar: comparable performance, not equivalence |
| V2a − S0 | F1@50 | -1.24 (+7.8 / -6.2 / -5.3) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-pub | MoF | +3.10 (+9.6 / -5.0 / +4.7) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-pub | Edit | +1.68 (+5.7 / -2.5 / +1.8) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-pub | F1@10 | +3.97 (+12.5 / -2.1 / +1.6) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-pub | F1@25 | +2.24 (+12.4 / -4.6 / -1.1) | 3 | within the bar: comparable performance, not equivalence |
| V2a − M-pub | F1@50 | -0.32 (+9.6 / -8.1 / -2.4) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-JL | MoF | -0.22 (-1.0 / +2.0 / -1.7) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-JL | Edit | +1.47 (-3.1 / +8.5 / -1.0) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-JL | F1@10 | -0.49 (-6.2 / +3.1 / +1.7) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-JL | F1@25 | -1.47 (-7.5 / +2.6 / +0.6) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-JL | F1@50 | -2.30 (-8.3 / +3.2 / -1.8) | 3 | within the bar: comparable performance, not equivalence |
| V2b − S0 | MoF | +4.31 (+1.2 / +5.9 / +5.8) | 3 | passes the bar |
| V2b − S0 | Edit | +6.06 (-2.7 / +13.5 / +7.4) | 3 | within the bar: comparable performance, not equivalence |
| V2b − S0 | F1@10 | +7.33 (-2.1 / +11.5 / +12.5) | 3 | within the bar: comparable performance, not equivalence |
| V2b − S0 | F1@25 | +6.58 (-2.5 / +10.6 / +11.6) | 3 | within the bar: comparable performance, not equivalence |
| V2b − S0 | F1@50 | +3.22 (-5.3 / +7.8 / +7.2) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-pub | MoF | +5.29 (+3.3 / +5.4 / +7.2) | 3 | passes the bar |
| V2b − M-pub | Edit | +4.42 (-3.4 / +11.2 / +5.5) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-pub | F1@10 | +5.21 (-2.8 / +9.7 / +8.7) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-pub | F1@25 | +5.31 (-2.4 / +7.8 / +10.5) | 3 | within the bar: comparable performance, not equivalence |
| V2b − M-pub | F1@50 | +4.14 (-3.5 / +5.9 / +10.1) | 3 | within the bar: comparable performance, not equivalence |

## Training (final epoch; 'extra' = the variant's added term, weighted)

| arm | seed | rec | commit | extra / vel | dead-code replacements | seconds |
|---|---|---|---|---|---|---|
| S0 | 111 | 52122.0 | 18954.3 | 0.0 | 821 | 444 |
| S0 | 222 | 54597.5 | 18723.3 | 0.0 | 824 | 443 |
| S0 | 1538574472 | 61955.1 | 17151.6 | 0.0 | 823 | 167 |
| M-pub | 111 | 91569.5 | 30345.4 | 173049.0 | 1588 | 445 |
| M-pub | 222 | 99705.4 | 28196.3 | 177981.4 | 1578 | 446 |
| M-pub | 1538574472 | 94828.3 | 33193.2 | 176691.7 | 1591 | 168 |
| M-JL | 111 | 89399.4 | 26840.6 | 0.0 | 1568 | 440 |
| M-JL | 222 | 90335.2 | 22795.3 | 0.0 | 1534 | 470 |
| M-JL | 1538574472 | 89891.4 | 24151.1 | 0.0 | 1491 | 167 |
| V1 | 111 | 87882.3 | 23148.4 | 3789.3 | 1647 | 526 |
| V1 | 222 | 91399.9 | 24608.6 | 3915.5 | 1533 | 462 |
| V1 | 1538574472 | 104507.1 | 25168.4 | 3738.6 | 1580 | 168 |
| V2a | 111 | 85726.7 | 24384.6 | 481.5 | 1611 | 480 |
| V2a | 222 | 89732.4 | 22607.2 | 424.7 | 1519 | 491 |
| V2a | 1538574472 | 96722.8 | 23814.1 | 518.3 | 1540 | 171 |
| V2b | 111 | 95385.0 | 24382.9 | 1759.0 | 1492 | 481 |
| V2b | 222 | 89559.7 | 23299.2 | 2010.9 | 1633 | 313 |
| V2b | 1538574472 | 90025.2 | 26697.9 | 1926.2 | 1453 | 169 |

## Decision (computed from the declared rule)

**Stop on HuGaDB.** No variant passes the bar over joint dropout alone on both MoF and F1@50, so the velocity term adds nothing measurable in either reading; the MASQ track is not extended to other datasets.

## Caveats

All results here are *development* results (protocol P7): every dataset informed earlier decisions and no untouched evaluation set exists. The published MASQ numbers are single runs from a paper that is not peer reviewed; our M arms are a reimplementation from the paper text with the SMQ backbone and defaults, so a gap to the published numbers may come from undocumented details. These experiments cannot show *why* MASQ works, only whether its training changes reproduce and which component matters; they say nothing about our OT decoder on MASQ latents (Tier 1).

The variants are our readings of an ambiguous specification (equation 5 versus the paper's description of a smoothness prior); neither is known to be the authors' implementation. rho values are declared choices, all of which are reported.
