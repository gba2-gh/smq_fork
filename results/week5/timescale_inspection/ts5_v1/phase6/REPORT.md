# Time-scale inspection, phase 6: confirming the span rule — ts5_v1

Status `finished`. Reuses `results\week5\timescale_inspection\ts5_v1\phase5` read-only for any preparation it already built. The primary number is F1@50 (selected checkpoint, last rung), paired by seed; the bar is >= 3 points on the mean with every seed in the same direction. With 5 seeds the report also gives the positive-seed count and an exact two-sided sign-test p-value; neither is part of a verdict.

## Gates

| gate | status | detail |
|---|---|---|
| g3 | passed | median across-joint extent: channels 3-5 609.889 vs 0-2 315.753 (inherited from phase5) |
| r1 | passed | checks {'same_recordings': True, 'codes': True, 'continuous': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True, 'lookup': True} |
| r2 | passed | 500/MoF 0.0, 500/Edit 0.0, 500/F1_10 0.0, 500/F1_25 0.0, 500/F1_50 0.0, 1000/MoF 0.0, 1000/Edit 0.0, 1000/F1_10 0.0, 1000/F1_25 0.0, 1000/F1_50 0.0, 2000/MoF 0.0, 2000/Edit 0.0, 2000/F1_10 0.0, 2000/F1_25 0.0, 2000/F1_50 0.0, 4000/MoF 0.0, 4000/Edit 0.0, 4000/F1_10 0.0, 4000/F1_25 0.0, 4000/F1_50 0.0 |


## Headline: BABEL-2/3, 5 seeds, tokenizer x span (`context`)

Original seeds 111/222/1538574472 plus the new seeds 333/444 (S1a extends T0 span 2 to all 3 original seeds; S1b adds both tokenizers x both spans at the new seeds).

| dataset | tokenizer | arm | span | F1@50 per seed (sorted: 111/222/333/444/1538574472) | mean ± sd | n seeds |
|---|---|---|---|---|---|---|
| babel2 | T0 | context | 8 | 16.7 / 11.1 / 12.7 / 11.2 / 17.5 | 13.84 ± 3.03 | 5 |
| babel2 | T0 | context | 2 | 15.1 / 20.7 / 17.2 / 16.9 / 16.8 | 17.36 ± 2.05 | 5 |
| babel2 | T2 | context | 8 | 16.3 / 26.2 / 27.7 / 15.2 / 21.8 | 21.43 ± 5.65 | 5 |
| babel2 | T2 | context | 2 | 31.5 / 30.0 / 29.3 / 25.6 / 30.5 | 29.39 ± 2.26 | 5 |
| babel3 | T0 | context | 8 | 14.8 / 16.0 / 13.3 / 17.1 / 15.2 | 15.28 ± 1.42 | 5 |
| babel3 | T0 | context | 2 | 13.6 / 11.7 / 13.3 / 16.3 / 14.7 | 13.90 ± 1.70 | 5 |
| babel3 | T2 | context | 8 | 19.0 / 18.4 / 15.7 / 19.1 / 20.6 | 18.56 ± 1.80 | 5 |
| babel3 | T2 | context | 2 | 20.1 / 22.8 / 23.5 / 26.1 / 26.8 | 23.86 ± 2.72 | 5 |

### `centre` (no-contextualizer) at span 8, 5 seeds where available

| dataset | tokenizer | arm | span | F1@50 per seed (sorted: 111/222/333/444/1538574472) | mean ± sd | n seeds |
|---|---|---|---|---|---|---|
| babel2 | T0 | centre | 8 | 20.1 / 18.6 / 19.9 / 20.4 / 20.3 | 19.86 ± 0.74 | 5 |
| babel2 | T2 | centre | 8 | 30.6 / 18.8 / 18.6 / 17.9 / 30.8 | 23.31 ± 6.74 | 5 |
| babel3 | T0 | centre | 8 | 17.9 / 17.6 / 17.0 / 17.4 / 17.1 | 17.40 ± 0.38 | 5 |
| babel3 | T2 | centre | 8 | 20.1 / 21.9 / 21.4 / 22.2 / 21.8 | 21.49 ± 0.84 | 5 |

### H2'-confirm

- babel2: T2 span 2 - T2 span 8 = +7.97 (+15.2 / +3.9 / +1.6 / +10.4 / +8.8) over 5 seeds -> **beats span 8 by the bar** (5/5 seeds positive, sign-test p=0.0625)
- babel3: T2 span 2 - T2 span 8 = +5.30 (+1.0 / +4.4 / +7.8 / +7.0 / +6.2) over 5 seeds -> **beats span 8 by the bar** (5/5 seeds positive, sign-test p=0.0625)

**H2'-confirm: confirmed.**

## H9: does the span effect need the tokenizer?

Using the 2x2 (tokenizer x span) at the seeds both sides have.

### babel2

- span effect on T0 (span2 - span8): +3.52 (-1.5 / +9.6 / +4.5 / +5.6 / -0.6)
- span effect on T2 (span2 - span8): +7.97 (+15.2 / +3.9 / +1.6 / +10.4 / +8.8)
- tokenizer effect at span 8 (T2 - T0): +7.59 (-0.4 / +15.0 / +15.0 / +3.9 / +4.3)
- tokenizer effect at span 2 (T2 - T0): +12.04 (+16.4 / +9.3 / +12.1 / +8.7 / +13.7)
- interaction ((T2 span2-span8) - (T0 span2-span8)), paired by seed: +4.45 (+16.7 / -5.7 / -2.9 / +4.8 / +9.4)
- **mixed (does not fit a declared label)**

### babel3

- span effect on T0 (span2 - span8): -1.38 (-1.2 / -4.4 / -0.0 / -0.9 / -0.5)
- span effect on T2 (span2 - span8): +5.30 (+1.0 / +4.4 / +7.8 / +7.0 / +6.2)
- tokenizer effect at span 8 (T2 - T0): +3.28 (+4.2 / +2.4 / +2.3 / +2.0 / +5.4)
- tokenizer effect at span 2 (T2 - T0): +9.96 (+6.5 / +11.2 / +10.2 / +9.9 / +12.1)
- interaction ((T2 span2-span8) - (T0 span2-span8)), paired by seed: +6.68 (+2.2 / +8.8 / +7.8 / +7.9 / +6.7)
- **needs relational tokens (T0 span 2 is within the bar of T0 span 8; T2's span effect passes)**

## Span-rule table: F1@50 and predicted vs ground-truth segments, by span ratio

`context` arm only.

| dataset | tokenizer | span | mask (s) | span ratio | F1@50 mean ± sd | n seeds | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|---|---|---|
| babel2 | T0 | 2 | 0.53 | 0.31 | 17.36 ± 2.05 | 5 | 2.5 | 3.7 |
| babel2 | T2 | 2 | 0.53 | 0.31 | 29.39 ± 2.26 | 5 | 3.0 | 3.7 |
| babel2 | T2 | 4 | 1.07 | 0.63 | 30.45 ± 0.35 | 3 | 2.9 | 3.7 |
| babel2 | T0 | 8 | 2.13 | 1.25 | 13.84 ± 3.03 | 5 | 1.9 | 3.7 |
| babel2 | T2 | 8 | 2.13 | 1.25 | 21.43 ± 5.65 | 5 | 2.2 | 3.7 |
| babel3 | T0 | 2 | 0.53 | 0.30 | 13.90 ± 1.70 | 5 | 3.3 | 3.7 |
| babel3 | T2 | 2 | 0.53 | 0.30 | 23.86 ± 2.72 | 5 | 3.4 | 3.7 |
| babel3 | T2 | 4 | 1.07 | 0.60 | 20.56 ± 3.31 | 3 | 3.1 | 3.7 |
| babel3 | T0 | 8 | 2.13 | 1.21 | 15.28 ± 1.42 | 5 | 2.6 | 3.7 |
| babel3 | T2 | 8 | 2.13 | 1.21 | 18.56 ± 1.80 | 5 | 2.5 | 3.7 |
| hugadb | T0 | 2 | 0.50 | 0.14 | 39.90 ± 1.40 | 3 | 10.3 | 10.5 |
| hugadb | T0 | 4 | 1.00 | 0.29 | 43.10 ± 4.45 | 3 | 10.0 | 10.5 |
| hugadb | T0 | 8 | 2.00 | 0.58 | 46.23 ± 3.30 | 3 | n/a | n/a |
| lara | T2 | 2 | 0.48 | 0.22 | 24.02 ± 1.75 | 3 | 24.7 | 32.1 |
| lara | T2 | 4 | 0.96 | 0.44 | 22.10 ± 0.90 | 3 | 23.3 | 32.1 |
| lara | T2 | 8 | 1.92 | 0.87 | 19.09 ± 1.34 | 3 | 21.2 | 32.1 |


## H8a: HuGaDB negative control, and H8b: LARa out-of-sample

- HuGaDB span 2 vs span 8: -6.33 (-11.7 / -4.3 / -3.0) -> **loses by the bar -> consistent with the smoothing mechanism** (not a pass/fail; HuGaDB needs the smoothing a longer span gives, setting a lower limit on the useful span)
- HuGaDB span 4 vs span 8: -3.13 (-9.7 / +3.4 / -3.1) -> within the bar -> no effect either way

- LARa span 2 vs span 8: +4.93 (+5.5 / +3.9 / +5.4) -> **threshold looks wrong** (gains >= 3 and more than BABEL-3's declared span-2 gain of +3.90); for information, BABEL-3's gain with 5 seeds is +5.30 (not the declared reference)
- LARa span 4 vs span 8: +3.01 (+4.1 / +2.5 / +2.4) -> consistent with the prediction (a modest gain at most, no loss); for information, BABEL-3's gain with 5 seeds is +5.30 (not the declared reference)

## Verification on every metric, not F1@50 alone

F1@50 is the pre-declared metric for H2'-confirm, H8a and H8b. This re-runs the same paired span comparisons on MoF, Edit, F1@10 and F1@25, read from the identical saved rows.

| comparison | metric | new mean | base mean | Δ (paired) | bar |
|---|---|---|---|---|---|
| babel2 T2 span2 vs span8 | MoF | 41.53 | 34.76 | +6.78 (+15.0 / +1.1 / +1.7 / +3.2 / +12.9) | beats by bar |
| babel2 T2 span2 vs span8 | Edit | 38.60 | 32.09 | +6.51 (+18.6 / +1.7 / +3.9 / +4.6 / +3.8) | beats by bar |
| babel2 T2 span2 vs span8 | F1@10 | 43.94 | 36.78 | +7.16 (+16.1 / +1.8 / +3.4 / +6.6 / +7.9) | beats by bar |
| babel2 T2 span2 vs span8 | F1@25 | 39.79 | 30.73 | +9.06 (+18.1 / +1.8 / +4.5 / +9.7 / +11.2) | beats by bar |
| babel2 T2 span2 vs span8 | F1@50 | 29.39 | 21.43 | +7.97 (+15.2 / +3.9 / +1.6 / +10.4 / +8.8) | beats by bar |
| babel3 T2 span2 vs span8 | MoF | 36.44 | 35.79 | +0.65 (-0.2 / -1.1 / +1.4 / +1.7 / +1.5) | no |
| babel3 T2 span2 vs span8 | Edit | 41.45 | 32.54 | +8.91 (+3.7 / +9.2 / +8.1 / +15.2 / +8.3) | beats by bar |
| babel3 T2 span2 vs span8 | F1@10 | 44.02 | 35.29 | +8.72 (+5.0 / +8.1 / +9.1 / +12.1 / +9.2) | beats by bar |
| babel3 T2 span2 vs span8 | F1@25 | 36.89 | 28.43 | +8.47 (+3.3 / +9.1 / +10.3 / +9.5 / +10.1) | beats by bar |
| babel3 T2 span2 vs span8 | F1@50 | 23.86 | 18.56 | +5.30 (+1.0 / +4.4 / +7.8 / +7.0 / +6.2) | beats by bar |
| hugadb T0 span2 vs span8 | MoF | 42.68 | 48.88 | -6.21 (-9.5 / -5.4 / -3.7) | no |
| hugadb T0 span2 vs span8 | Edit | 39.91 | 42.68 | -2.77 (-7.1 / -0.4 / -0.8) | no |
| hugadb T0 span2 vs span8 | F1@10 | 47.88 | 50.21 | -2.34 (-7.4 / +0.6 / -0.1) | no |
| hugadb T0 span2 vs span8 | F1@25 | 44.86 | 48.75 | -3.90 (-9.1 / -0.6 / -2.0) | no |
| hugadb T0 span2 vs span8 | F1@50 | 39.90 | 46.23 | -6.33 (-11.7 / -4.3 / -3.0) | no |
| hugadb T0 span4 vs span8 | MoF | 47.19 | 48.88 | -1.70 (-4.0 / -0.8 / -0.3) | no |
| hugadb T0 span4 vs span8 | Edit | 41.89 | 42.68 | -0.78 (-4.8 / +2.9 / -0.5) | no |
| hugadb T0 span4 vs span8 | F1@10 | 48.48 | 50.21 | -1.74 (-7.0 / +4.2 / -2.4) | no |
| hugadb T0 span4 vs span8 | F1@25 | 46.15 | 48.75 | -2.61 (-9.1 / +4.4 / -3.1) | no |
| hugadb T0 span4 vs span8 | F1@50 | 43.10 | 46.23 | -3.13 (-9.7 / +3.4 / -3.1) | no |
| lara T2 span2 vs span8 | MoF | 35.39 | 33.36 | +2.03 (+3.6 / +1.5 / +1.0) | no |
| lara T2 span2 vs span8 | Edit | 37.40 | 32.70 | +4.70 (+5.8 / +4.1 / +4.2) | beats by bar |
| lara T2 span2 vs span8 | F1@10 | 40.01 | 34.76 | +5.25 (+6.1 / +5.4 / +4.2) | beats by bar |
| lara T2 span2 vs span8 | F1@25 | 35.31 | 29.81 | +5.50 (+6.1 / +5.4 / +5.0) | beats by bar |
| lara T2 span2 vs span8 | F1@50 | 24.02 | 19.09 | +4.93 (+5.5 / +3.9 / +5.4) | beats by bar |
| lara T2 span4 vs span8 | MoF | 35.07 | 33.36 | +1.71 (+2.6 / +1.4 / +1.1) | no |
| lara T2 span4 vs span8 | Edit | 35.06 | 32.70 | +2.37 (+3.4 / +2.5 / +1.2) | no |
| lara T2 span4 vs span8 | F1@10 | 37.49 | 34.76 | +2.73 (+4.0 / +2.9 / +1.4) | no |
| lara T2 span4 vs span8 | F1@25 | 32.74 | 29.81 | +2.93 (+4.3 / +2.8 / +1.7) | no |
| lara T2 span4 vs span8 | F1@50 | 22.10 | 19.09 | +3.01 (+4.1 / +2.5 / +2.4) | beats by bar |


## OT fits that stopped before the 50-iteration cap (may explain seed-level outliers)

- babel2/t2_relational/centre span8 seed1538574472: converged at 40 iterations
- babel2/t2_relational/centre span8 seed222: converged at 48 iterations
- babel2/t2_relational/pca span8 seed222: converged at 29 iterations
- babel3/t0_raw/centre span8 seed333: converged at 39 iterations
- babel3/t0_raw/centre span8 seed444: converged at 23 iterations
- babel3/t4_smq/centre span16 seed111: converged at 30 iterations
- babel3/t4_smq/centre span16 seed222: converged at 27 iterations
- babel3/t4_smq/pca span16 seed111: converged at 49 iterations
- babel3/t4_smq/pca span16 seed222: converged at 26 iterations

## Reference: SMQ's local re-score (not a declared criterion)

| dataset | SMQ F1@50 (released checkpoint, global Hungarian) |
|---|---|
| babel2 | 27.5 |
| babel3 | 19.4 |
| hugadb | 24.3 |
| lara | 16.4 |


- Pre-declared headline (BABEL-2, T2, span 2, `context`): 29.39 mean over 5 seeds, vs SMQ 27.5.
- Same cell on BABEL-3: 23.86 mean over 5 seeds, vs SMQ 19.4.
