# Time-scale inspection, phase 7: BABEL-1 prospective evaluation and label-free span screening — ts5_v1

Status `budget_stop`. Rules declared in PHASE7_LABEL_FREE_SCALE.md before the run. Primary metric F1@50 at the last rung; bar >= 3 points on the mean with every seed in the same direction.

## Gates

| gate | status | detail |
|---|---|---|
| g7 | passed | {'same_recordings': True, 'codes': True, 'continuous': True, 'starts': True, 'lengths': True, 'mean': True, 'rms': True, 'grouping': True, 'split': True, 'lookup': True} |


## Part H: prospective evaluation of the frozen T2 / short-span recipe on BABEL-1

**Not an untouched test set.** BABEL-1's T0 `context`/`centre` scores, oracle ceilings and ground-truth duration statistics were already examined in rkx_v2, Phase 0 and Phase 1, and 3 of its 1,960 recordings are byte-identical to BABEL-2 recordings and 7 to BABEL-3 recordings. What is new is that the T2 / short-span recipe was frozen on BABEL-2/3 and LARa before it was ever run here.

SMQ released checkpoint on BABEL-1 (global Hungarian): F1@50 22.38. T0 span 8 at seeds 111/222/1538574472 is rkx_v2; at 333/444 it is this phase (gate G7).

| cell | F1@50 per seed | mean ± sd | n seeds | pred segs/rec | gt segs/rec |
|---|---|---|---|---|---|
| T0 span 8 context | 111:15.9 / 222:13.2 / 333:13.2 / 1538574472:13.6 | 13.98 ± 1.29 | 4 | 2.3 | 5.1 |
| T0 span 2 context | 111:14.7 / 222:16.0 / 1538574472:16.6 | 15.77 ± 0.96 | 3 | 2.8 | 5.1 |
| T0 centre | 111:17.4 / 222:17.4 / 1538574472:19.5 | 18.14 ± 1.21 | 3 | — | — |
| T2 span 8 context | 111:28.4 / 222:26.0 / 333:26.4 / 1538574472:24.4 | 26.31 ± 1.63 | 4 | 2.6 | 5.1 |
| T2 span 2 context | 111:28.4 / 222:29.5 / 333:23.1 / 444:33.4 / 1538574472:27.3 | 28.35 ± 3.73 | 5 | 3.1 | 5.1 |
| T2 centre | 111:21.1 / 222:24.6 / 1538574472:24.4 | 23.40 ± 1.96 | 3 | 5.2 | 5.1 |

### Declared comparisons

- **P1** (T2 span 2 vs T2 span 8), original seeds: +2.10 (+0.0 / +3.5 / +2.8) over 3 seeds -> **does not pass the bar**
- **P1**, all seeds: +0.77 (+0.0 / +3.5 / -3.2 / +2.8) over 4 seeds -> **does not pass the bar**
- **P2** (T2 span 2 vs T0 span 8, the original pipeline), original seeds: +14.15 (+12.5 / +16.3 / +13.6) over 3 seeds -> **passes the bar**
- **P2**, all seeds: +13.09 (+12.5 / +16.3 / +9.9 / +13.6) over 4 seeds -> **passes the bar**
- **P3** (descriptive; T0 span 2 vs T0 span 8): +1.52 (-1.2 / +2.8 / +2.9) over 3 seeds -> **does not pass the bar**
- **P4** (does contextualization add value? T2 span 2 `context` vs T2 `centre`, same tokens, no context): +4.99 (+7.3 / +4.9 / +2.8) over 3 seeds -> **passes the bar**

### Tokenizer x span factorial (the 3 seeds with all four cells; T0 span 2 has no seeds 333/444)

- span effect on T0 (span 2 - span 8): +1.52 (-1.2 / +2.8 / +2.9)
- span effect on T2 (span 2 - span 8): +2.10 (+0.0 / +3.5 / +2.8)
- interaction (T2 span effect - T0 span effect): +0.58 (+1.2 / +0.7 / -0.1)

### Every metric (same paired comparisons, original seeds)

| comparison | metric | Δ (paired) | bar |
|---|---|---|---|
| P1 T2 span2 vs span8 | MoF | +0.42 (-0.6 / -2.0 / +3.9) | no |
| P1 T2 span2 vs span8 | Edit | +3.66 (+2.8 / +3.4 / +4.7) | passes |
| P1 T2 span2 vs span8 | F1@10 | +2.18 (+2.1 / +1.2 / +3.2) | no |
| P1 T2 span2 vs span8 | F1@25 | +2.63 (+2.1 / +2.1 / +3.7) | no |
| P1 T2 span2 vs span8 | F1@50 | +2.10 (+0.0 / +3.5 / +2.8) | no |
| P2 T2 span2 vs T0 span8 | MoF | +10.59 (+12.2 / +8.8 / +10.8) | passes |
| P2 T2 span2 vs T0 span8 | Edit | +9.99 (+8.9 / +10.5 / +10.6) | passes |
| P2 T2 span2 vs T0 span8 | F1@10 | +9.35 (+9.4 / +9.9 / +8.7) | passes |
| P2 T2 span2 vs T0 span8 | F1@25 | +12.72 (+12.8 / +13.3 / +12.1) | passes |
| P2 T2 span2 vs T0 span8 | F1@50 | +14.15 (+12.5 / +16.3 / +13.6) | passes |
| P4 T2 span2 context vs T2 centre | MoF | +1.52 (+7.8 / -2.0 / -1.2) | no |
| P4 T2 span2 context vs T2 centre | Edit | +1.93 (+2.8 / +1.7 / +1.4) | no |
| P4 T2 span2 context vs T2 centre | F1@10 | +4.14 (+7.0 / +2.5 / +3.0) | passes |
| P4 T2 span2 context vs T2 centre | F1@25 | +6.23 (+9.2 / +4.9 / +4.5) | passes |
| P4 T2 span2 context vs T2 centre | F1@50 | +4.99 (+7.3 / +4.9 / +2.8) | passes |

## Part S0: re-decodes with predictions saved

33/33 re-decodes reproduced their saved ladder exactly (every rung, every metric); 33 were planned. Only reproduced units enter the screening.

## Part K: label-free span selectors — exploratory, retrospective screening

These selectors use no labels, but the spans they are scored against were chosen by F1@50, and choosing among K1/K3/K4 by how well they score is itself a label-based selection. Any 'pass' here is a candidate for a prospective test, not evidence of label-free selection.

Analysis hash (criteria7.py + report7.py): `62199ec349588bfd`

| dataset | span | mask (s) | F1@50 | acceptable | K1 stability | K3 median pred seg (s) | pred runs/rec | states/rec | share 1-state recs | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| babel1 | 2 | 0.53 | 27.07 | yes | 47.48 | 2.40 | 3.1 | 2.43 | 0.13 | 4 |
| babel1 | 8 | 2.13 | 26.31 | yes | 51.16 | 2.67 | 2.6 | 2.14 | 0.21 | 4 |
| babel2 | 2 | 0.53 | 30.68 | yes | 52.64 | 2.67 | 3.0 | 2.23 | 0.16 | 3 |
| babel2 | 4 | 1.07 | 30.45 | yes | 49.13 | 2.70 | 2.9 | 2.14 | 0.18 | 3 |
| babel2 | 8 | 2.13 | 21.41 |  | 31.00 | 3.47 | 2.2 | 1.77 | 0.44 | 3 |
| babel3 | 2 | 0.53 | 23.23 | yes | 42.96 | 3.20 | 3.3 | 2.48 | 0.12 | 3 |
| babel3 | 4 | 1.07 | 20.56 | yes | 36.91 | 3.60 | 3.1 | 2.36 | 0.20 | 3 |
| babel3 | 8 | 2.13 | 19.33 |  | 27.63 | 4.00 | 2.6 | 2.10 | 0.33 | 3 |
| lara | 2 | 0.48 | 24.02 | yes | 41.50 | 3.60 | 24.7 | 7.15 | 0.00 | 3 |
| lara | 4 | 0.96 | 22.10 | yes | 39.40 | 3.84 | 23.3 | 7.00 | 0.00 | 3 |
| lara | 8 | 1.92 | 19.09 |  | 34.71 | 4.08 | 21.2 | 6.85 | 0.00 | 3 |
| hugadb | 2 | 0.50 | 39.90 |  | 35.79 | 3.50 | 10.3 | 5.14 | 0.05 | 3 |
| hugadb | 4 | 1.00 | 43.10 |  | 40.42 | 3.50 | 10.0 | 4.42 | 0.05 | 3 |
| hugadb | 8 | 2.00 | 46.23 | yes | 48.77 | 3.75 | 9.0 | 3.78 | 0.07 | 3 |


K4 coverage (recordings where the decorrelation time is defined):

| dataset | K4 tau (s) | coverage |
|---|---|---|
| babel1 | 0.27 | seed 111: 1960/1960 recordings; seed 222: 1960/1960 recordings; seed 333: 1960/1960 recordings; seed 1538574472: 1960/1960 recordings |
| babel2 | 0.27 | seed 111: 255/255 recordings; seed 222: 255/255 recordings; seed 1538574472: 255/255 recordings |
| babel3 | 0.27 | seed 111: 183/183 recordings; seed 222: 183/183 recordings; seed 1538574472: 183/183 recordings |
| lara | 0.24 | seed 111: 439/439 recordings; seed 222: 439/439 recordings; seed 1538574472: 439/439 recordings |
| hugadb | 0.25 | seed 111: 364/364 recordings; seed 222: 364/364 recordings; seed 1538574472: 364/364 recordings |


### Picks and gate

| selector | babel1 | babel2 | babel3 | lara | hugadb | right | status |
|---|---|---|---|---|---|---|---|
| K1 | 8 ✓ | 2 ✓ | 2 ✓ | 2 ✓ | 8 ✓ | 5/5 | **pass** |
| K3 | 8 ✓ | 8 ✗ | 8 ✗ | 8 ✗ | 8 ✓ | 2/5 | **fail** |
| K4 | 2 ✓ | 2 ✓ | 2 ✓ | 2 ✓ | 2 ✗ | 4/5 | **fail** |
| always_shortest | 2 ✓ | 2 ✓ | 2 ✓ | 2 ✓ | 2 ✗ | 4/5 | **fail** |
| always_8 | 8 ✓ | 8 ✗ | 8 ✗ | 8 ✗ | 8 ✓ | 2/5 | **fail** |


K1 = cross-seed segmentation stability (argmax; can score 100 on collapsed segmentations, hence the occupancy columns); K3 = largest span whose mask fits inside the model's own median predicted segment (over-merging can justify the long span that caused it); K4 = largest span whose mask fits inside the feature decorrelation time (persistence need not mark action boundaries). An undefined pick counts as wrong. Constants are references.
