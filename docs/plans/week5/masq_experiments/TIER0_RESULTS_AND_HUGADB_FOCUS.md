# Tier 0 MASQ replication: what was done, what it showed, and the HuGaDB-focused follow-up

6 October 2026 · Version 1. Records `masq_v1` (finished, budget stop) and the code for the follow-up stage `hu`. The follow-up has
**not been run**; it is implemented, tested and smoke-tested.

Parents: `TIER0_PROTOCOL_MASQ_REPLICATION.md` (plan and Part A protocol), `AGENT_INSTRUCTIONS.md` (implementation brief),
`docs/plans/COMPOSITIONAL_ACTION_DYNAMICS_PROPOSAL.md` (the architecture programme Tier 0 feeds).

## 1. Summary

- **Question.** Does MASQ's training recipe (joint-level structured dropout, JLSD, plus a mask-aware velocity loss) improve SMQ,
  and do its published numbers reproduce on this repository's SMQ code?
- **Answer.** No. The computed verdict is *partly replicated*: MASQ's published settings pass the bar over our SMQ retrain on one
  of five datasets (BABEL-1, MoF +5.6), and none of the published numbers is reproduced (ours are 8-17 MoF points lower).
- **Most useful finding.** The velocity loss, implemented as the paper's equation 5, is unlearnable at this architecture's patch
  bottleneck and its size depends on the data scale. JLSD alone helps on HuGaDB (+4.5 MoF, +5.5 F1@50, all seeds) and BABEL-1
  (+4.1 MoF); adding the velocity term makes HuGaDB worse. This comparison was not declared in advance (exploratory).
- **Decision.** Stop Stage 2 (5 of 27 units unrun; they cannot change the verdict). Run one focused test on HuGaDB instead
  (section 6), with a declared stop rule.
- **What went wrong in planning** is in section 5: the full five-dataset grid was run before the new terms were checked on a single
  dataset.

## 2. What was built

All code is in `script/week5/masq_experiments/` (nothing outside it was edited). It was reviewed against SMQ's originals once
after the first implementation; that review found and fixed three defects (a scoring-code change would have blocked a resume; deterministic
Python errors were retried three times; a cached smoke rerun overwrote the saved timings).

| file | role |
|---|---|
| `model_masq.py` | `MASQModel(SMQModel)`: `forward` is SMQ's verbatim, plus JLSD after the encoder (one Bernoulli draw per sample, person and joint, broadcast over time and channels; kept joints scaled by 1/(1-p); off in eval) |
| `losses.py` | mask-aware velocity loss (equation 5): squared L2 over channels, averaged over valid (frame pair, joint) points; loss assembly |
| `train_masq.py` | one training unit, a copy of `model.Trainer.train` plus the velocity term; seeding in `main.py`'s order |
| `score.py` | one scoring unit through the same path as the `results/e0` numbers; gate G4 |
| `diagnostics.py` | subject and recording AMI of codes; same-recording neighbour share of latents |
| `run.py`, `run.ps1`, `run_all.ps1` | driver (gates, smoke, stages), per-process isolation with retries on native crashes, atomic cached units, chained launcher |
| `report.py`, `sheet.py` | REPORT.md with R1-R5, G5 and the computed verdict; spreadsheet sheet (not yet written to the real workbook) |
| `tests.py` | 10 tests |

**Gates (all passed before any scored run).**

| gate | what | status |
|---|---|---|
| G1 | MASQModel with p = 0 and no velocity loss equals SMQModel exactly (outputs, indices, losses, every gradient; train and eval) | passed; confirmed to fail when JLSD is on |
| G2 | JLSD mechanics (dropped rows zero at every time step and channel, kept rows scaled, drop rate, eval off, row mapping) | passed |
| G3 | velocity loss vs a hand computation | passed |
| G4 | released SMQ checkpoints (HuGaDB, LARa) re-scored through our scorer reproduce `results/e0`, global and local | passed, max difference 0.0 |
| G5 | S0 within 2 SD of the earlier retrained spread (HuGaDB, LARa) | passed |

G1-G3 run on CPU in float32 because SMQ's quantizer builds float32 one-hots and is not float64-clean; G1 still demands bitwise
equality.

## 3. The run `masq_v1`

Stage 1 (S0, M-pub) and part of Stage 2 (M-JL, M-fix), launched 6 October 00:27, stopped at 15:15 on its 6-hour budget. 52 of 57
planned units completed every step (train, score, diagnose); the logs (UTF-8, decoded and searched) contain no errors or retries. Unrun: `M-fix` on BABEL-1 (seeds 222,
1538574472) and LARa (all seeds).

Measured cost per unit (train + score + diagnose, minutes): BABEL-3 1-4, BABEL-2 8-12, HuGaDB 3-8, LARa 13, BABEL-1 45-71. The
first estimates were BABEL-1 12 and BABEL-2 5, which is why the run overran; `config.UNIT_MINUTES` now holds the measured values.
Peak GPU memory 10-17.5 GB.

### Results (seed means, global Hungarian, final epoch)

MoF / F1@50:

| Dataset | S0 (SMQ retrain) | M-pub (MASQ published settings) | M-JL (dropout only) | MASQ published | SMQ released checkpoint |
|---|---|---|---|---|---|
| HuGaDB | 40.7 / 22.0 | 39.7 / 21.1 | **45.2 / 27.6** | 52.3 / 39.5 | 42.0 / 24.3 |
| LARa | 36.8 / 15.8 | 37.0 / 15.6 | 35.3 / 13.9 | 47.2 / 22.2 | 37.4 / 16.4 |
| BABEL-1 | 33.0 / 13.0 | **38.5** / 15.0 | 37.0 / 13.6 | 46.7 / 21.0 | 36.4 / 22.4 |
| BABEL-2 | 39.9 / 17.8 | 38.5 / 17.5 | 38.9 / 18.6 | 55.8 / 28.4 | 49.2 / 27.5 |
| BABEL-3 | 38.1 / 17.2 | 39.7 / 12.0 | 41.1 / 19.6 | 47.7 / 24.5 | 40.4 / 19.4 |

Declared comparisons (paired by seed; bar = mean ≥ 3 with all three seeds in the same direction):

- **R1, M-pub − S0, MoF:** passes on BABEL-1 (+5.56, all seeds positive); within the bar on the other four. On F1@50 it is
  reversed on BABEL-3 (−5.27).
- **R2, reproduced?** No dataset is within 3 points of the published MoF and F1@50.
- **R4a, M-pub − M-JL (the velocity loss):** reverse passes on HuGaDB (MoF −5.50, F1@50 −6.44, all seeds negative) and on
  BABEL-3 F1@50 (−7.62); within the bar on the rest.
- **R4b, M-fix − M-pub:** incomplete. BABEL-3 and BABEL-2 are within the bar on MoF and F1@50 (on BABEL-3 Edit passes, +6.08); BABEL-1 has one seed; HuGaDB and LARa were not run.
- **Exploratory, not declared:** M-JL − S0 on HuGaDB passes on MoF (+4.52) and F1@50 (+5.52), all seeds; BABEL-1 MoF +4.06.

**Identity diagnostics (R5).** JLSD did not reduce subject or recording association: subject AMI on LARa is 0.05 for S0 and M-pub
(0.04 for M-JL); on HuGaDB it rose from 0.08 (S0) to 0.11 (M-pub). The same-recording neighbour share of latents rose on LARa from
0.39 to 0.45. Under the plan's rule, body-retargeting consistency remains the only identity mechanism in the programme.

**S0 on BABEL is weak.** SMQ retrained with the repository defaults lands well below the released BABEL checkpoints (BABEL-2 F1@50
17.8 against 27.5; BABEL-1 13.0 against 22.4), so BABEL comparisons start from a weak baseline. The released models were probably
trained with settings the repository does not record. HuGaDB and LARa match the earlier retrained spread (G5 passed).

## 4. Why the velocity loss failed to help: diagnostics on the saved checkpoints

No code bug was found: `forward` differs from SMQ's only by the JLSD block, the mask mapping is verified exhaustively, and G1-G4
pass. Read-only diagnostics on the saved checkpoints (12 recordings per dataset, seed 111) show a specification problem instead.

**The loss is on very different scales per dataset.** Initial velocity/reconstruction ratio (at an all-zero decoder output):

| dataset | feature std | velocity loss / distance reconstruction at init |
|---|---|---|
| HuGaDB | 9,992 (raw IMU values) | 0.27 |
| LARa | 231 | 0.0002 |
| BABEL-3 | 323 | 0.0018 |
| BABEL-1 | 307 | 0.0014 |

With MASQ's λ = 0.001 the velocity term ends about twice the reconstruction loss on HuGaDB (173k against 92k) and about 6% of it on BABEL.

**The loss is unlearnable here.** Velocity loss ÷ true velocity energy (1.0 = no better than predicting zero velocity):

| | HuGaDB | BABEL-2 |
|---|---|---|
| S0 (never trained on it) | 1.002 | 1.455 |
| M-JL | 1.001 | 1.395 |
| M-pub (trained on it) | 1.001 | 1.206 |
| correlation of decoded and true velocity, M-pub | 0.004 | −0.024 |

The trained model reproduces the true velocity no better than S0, and the logged velocity loss on HuGaDB is nearly flat (181,459
in epoch 1, 173,049 in epoch 30) while the reconstruction loss falls 58%. A 1-second patch quantizer cannot represent frame-level
velocity, so the term is a large constant with noisy gradients.

**JLSD makes training and test assignments disagree.** When JLSD is applied to the latents before the (eval-mode) quantizer, only
44-60% of patches keep their code (HuGaDB M-JL 0.46, M-pub 0.60; BABEL-2 0.46 and 0.44). This is inherent to the method.

**Where the specification is ambiguous.** Equation 5 matches decoded to true velocity, which cannot be a smoothness prior. The
paper's discussion calls the term "a first-order smoothness prior" acting "in a continuous feature space" before quantization.
The authors may apply it to the pre-quantization latents or in a normalized space. This is an inference from the text, not
confirmed by their code.

## 5. What was planned badly

1. **No single-dataset gate.** The plan launched 30 units (Stage 1) and 27 (Stage 2) over five datasets before checking the new
   terms on one. The HuGaDB logs already showed the velocity term dominating training; it should have stopped there.
2. **Time estimates were guesses.** BABEL-1 took 45-71 minutes per unit against 12 estimated, BABEL-2 8-12 against 5; the run
   exceeded its budget, and BABEL-1 alone used 9.0 of the 14.2 hours of training (BABEL-2 1.9, LARa 2.0, HuGaDB 0.9, BABEL-3 0.5).
3. **λ was applied without checking the loss scale on each dataset.** The scale table above is a ten-minute measurement.
4. **The plan declared no stop rule.** R1-R5 and the verdict can read "partly replicated" without saying what to do next.

## 6. The HuGaDB-focused follow-up (stage `hu`): implemented, not run

**Scope.** HuGaDB only: cheapest (3-8 minutes per unit) and the largest gap to MASQ's published numbers (our best, M-JL, is
45.2 / 27.6; published 52.3 / 39.5). Nine units: three variants × three seeds, about 1.2 h by the worst-case estimate.

**Arms.** All three are M-JL (JLSD at the published HuGaDB rate 0.25, no coordinate velocity loss) plus one calibrated term:

| arm | term | rho |
|---|---|---|
| V1 | equation 5 as written (coordinate velocity matching), scale-free weight | 0.0016 |
| V2a | latent smoothness prior on pre-quantization latents of visible joints | 0.01 |
| V2b | the same | 0.1 |

**Calibration rule (label-free, scale-free).** The weight is set at initialization so that weight × term(init) = rho × SMQ's
weighted reconstruction loss at an all-zero decoder output, on a fixed sample (the first 16 recordings by file name, first 2000
frames). The latent term is the first-difference energy divided by the energy of the latent trajectory (a ratio, so shrinking the latents
cannot reduce it). Calibration runs under a forked random stream and calls only the encoder, so each variant uses the same
random stream, codebook initialization and batch order as the matching M-JL unit and differs only by the extra term.

**Why these values.** rho = 0.0016 for V1 is the initial ratio that MASQ's published λ = 0.001 gives on BABEL-1 and BABEL-3
(measured 0.0014 and 0.0018): it is "the published weight, with the dataset scale removed". On HuGaDB it gives a weight of about
5e-6, 180 times smaller than 0.001. rho = 0.01 and 0.1 for V2 are declared strengths, both reported; the latent term's value at
initialization is 0.27 (a ratio), so V2b is about 10% of the initial reconstruction loss.

**Decision rule (declared before the run).** The MASQ track continues only if a variant passes the bar over M-JL on both MoF and
F1@50 (mean difference ≥ 3, all three seeds in the same direction). Otherwise it stops on HuGaDB; no other dataset is run.
`report_hu.py` computes this.

**Expected outcome, stated now.** V1 should land near M-JL (a term that cannot be learned, at a weight that cannot dominate), which
would confirm the scale explanation. V2 is the only arm with a chance of adding value. V2b's strength is the main risk: the latent
ratio rose to about 1.5 within 6 smoke batches, which puts the V2b term near the size of the reconstruction loss.

**Not claimed.** These variants are our readings of an ambiguous specification; neither is known to be the authors' implementation.
A pass would show that a scale-free or latent-space term helps HuGaDB, not that MASQ is reproduced.

**Implementation.** New files only: `losses_variants.py`, `train_variants.py`, `report_hu.py`; edits to `config.py` (arms, calibration
constants, measured time table), `run.py`/`run.ps1` (stage `hu`, gates G6/G7), `tests.py` (tests 9-10), `sheet.py`, `README.md`.
Variant code is hashed separately and only into variant identities. **Regression check:** all 52 `masq_v1` units still carry the
identities the current code computes, so the finished run's results remain valid. Tests: 10 of 10 pass; test 10 was confirmed to
fail when the row mapping is broken. A smoke run covering all seven arms (the four baseline arms on BABEL-3, the three variants on
HuGaDB) took 4.0 minutes; its throwaway folders were deleted.

**Launch (user).**

```
.\script\week5\masq_experiments\run.ps1 -RunId masq_hu1 -Stage hu -Hours 2 *>&1 | Tee-Object masq_hu1.log
```

Output: `results/week5/masq_experiments/masq_hu1/` (`REPORT.md`, and `REPORT_HU.md` with the decision). Gates run first
(about 3 minutes). Windows PowerShell 5.1 `Tee-Object` writes the log as UTF-16.

## 7. Known limitations and loose ends

- `masq_v1` Stage 2 is incomplete by choice (5 `M-fix` units). Completing it takes about 3.1 h and answers only whether the published
  per-dataset dropout rate matters.
- The spreadsheet sheet "MASQ replication" has not been written to the real workbook (`sheet.py` is tested on a staged copy only).
- `AGENT_INSTRUCTIONS.md` still describes a single code hash; the code now has separate training, scoring and variant identities, and the
  per-unit time table is the measured one.
- The BABEL baseline gap (section 3) is unexplained.
- Everything here is a development result (protocol P7): all datasets informed earlier decisions and no untouched evaluation set exists.
