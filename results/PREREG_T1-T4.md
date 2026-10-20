# Pre-registration — experiments T1–T4

Written before any T1–T4 training run or analysis. Follow-ups to
`REPORT_reproduction_and_diagnosis.md` §5. Any departure from this document is
reported as a departure, with its reason.

## Shared rules

* **Seeds**: 1538574472 (repo default), 111, 222 — the E0 seeds. Every condition,
  including each experiment's own baseline, is trained fresh by the same runner so
  that baseline and treatment differ only in the manipulated setting.
* **Everything else is the published configuration** (30 epochs, lr 5e-4,
  latent 16, 128 maps, 3 layers, decay 0.5, commit 1.0, rec 0.001, joint-distance
  reconstruction). Only the final checkpoint is kept (`--save_every 30`).
* **Datasets**: HuGaDB, LARa, BABEL-1. C = number of GT classes = 10 / 8 / 5.
* **Primary protocol**: the published dataset-level bijective Hungarian (MoF,
  Edit, F1@{10,25,50}), plus JSD with the convention in the report §2.
* **Effect rule**: a difference is called an effect only when the difference of
  3-seed means exceeds the larger of the two conditions' seed sds **and** the
  mean ± sd bands do not overlap. Otherwise it is reported as "no detectable
  effect". Welch t is reported but not used as the decision rule (n = 3).
* **Nulls are results.** No condition is added, dropped, or re-run because of
  its outcome.

### Code changes (all default-off; published behaviour unchanged)

* `euclidean_dist` is chunked over the codebook for memory. Every published
  configuration fits in one chunk, so it runs the original computation exactly.
* `--dead_code_threshold` (default 10), `--tc_weight` (default 0), `--tc_beta`
  (default 1), `--save_every` (default 5). Per-epoch loss terms go to
  `losses.jsonl`; the resolved config goes to `config.json`.

---

## T1 — Decouple codebook size from class count

**Question.** Does a codebook larger than C carry more action information, and
does that survive an unsupervised merge back to C?

**Conditions.** K ∈ {C, 2C, 4C, 8C, 16C} → HuGaDB {10…160}, LARa {8…128},
BABEL-1 {5…80}. 3 seeds each.

**Dead-code threshold.** The published threshold (10) is an absolute EMA count
per batch, set for K = C. At 16C the expected count per code falls below the
threshold and every code would be replaced every step. Primary runs therefore
use `max(1, round(10·C/K))` → {10, 5, 2, 1, 1}, which keeps the threshold at the
same fraction of expected per-code usage. **Check**: one extra run per dataset at
K = 16C with the published threshold 10 (seed 1538574472), to show what the
unscaled setting does. That is a stated deviation, not a tuned one.

**Readouts, all reported for every run:**
1. (a) Bijective Hungarian, as published. For K > C the voting table is padded
   and surplus codes score as wrong. This is the protocol as-is; expect it to fall.
2. (b) Many-to-one (each code → its majority class). **Uses GT for the mapping
   and inflates trivially with K** (K = #patches is the patch-grid oracle), so it
   is interpreted only against the control below.
3. (c1) Geometry merge — agglomerative clustering (average linkage) of the K
   codebook entries into C groups under the quantiser's own patch distance,
   then (a). Unsupervised.
4. (c2) Temporal merge — spectral clustering (C groups) of the symmetrised
   code-transition count graph between consecutive patches, then (a).
   Unsupervised. At K = C both merges are the identity, which serves as a
   correctness check.
5. Code one-hot probe and pre-quantisation latent probe (5-fold grouped by
   sequence, patch-level, as in the report §4.3).
6. Codebook perplexity, dead codes at evaluation.

**Control.** Mini-batch k-means on flattened raw-feature patches at every K
(3 seeds), put through readouts 1–5 identically (the geometry merge uses
centroid distances). This separates "a larger K helps" from "a *learned* larger
K helps".

**Predictions and what falsifies them.**
* The code probe rises toward the latent probe as K grows. *Falsified* if it is
  flat in K.
* The rise is due to learning. *Falsified* if the SMQ code probe at K is not
  above the k-means control's code probe at the same K by the effect rule.
* The gain is usable without labels. *Falsified* if neither c1 nor c2 at any
  K > C beats the K = C baseline under the effect rule on MoF or F1@50. **This
  is the decision-relevant readout.** There is no selection of a "best K": all K
  are reported, and a claim about a specific K has to hold across all three seeds.

---

## T2 — Temporal-consistency objective

**Question.** Does a label-free term pushing temporally adjacent patches onto
the same code make codes carry action identity?

**Loss** (`SkeletonMotionQuantizer._temporal_consistency_loss`). Soft assignment
q_i = softmax(−β · standardised distance of patch i to each code). The codebook
is detached, so only the encoder gets gradient and the EMA update is untouched.
L_tc = mean symmetric cross-entropy between adjacent patches of the same sequence
+ (log K − H(batch-mean q)). The second term prevents collapse onto one code.
β = 1, fixed.

**Weight.** λ is set from loss *magnitudes*, never from outcomes. Let
c = the final-epoch commitment loss of the T1 K = C seed-1538574472 run for that
dataset. L_tc is O(log K) at initialisation, so λ₁ = c / log C matches the two
terms in scale.

**Conditions** (K = C, 3 seeds): baseline (λ = 0; reuses the T1 K = C runs,
same runner), λ₁, 10·λ₁, and consistency-only (λ₁ with reconstruction weight 0).

**Readouts.** As T1 readouts 1, 5, 6, plus JSD.
**Prediction.** The code probe and NMI rise. *Falsified* if neither beats the
baseline under the effect rule at either weight. If the probe rises but MoF does
not, that is reported as "the bijection binds", not as a failure of T2.
**Expected failure mode, stated in advance.** Consistency-only will most likely
collapse or encode time and sequence identity. It is an ablation, not a candidate.

---

## T3 — Duration-aware decoding (post-hoc, no training)

**Inputs.** Per-patch distances to every code, from the released checkpoints
and from the three T1 K = C seeds. Emission log-probability =
log softmax(−β · standardised distance).

**Decoders.**
* argmax (published)
* HMM Viterbi — transition matrix (including self-transitions) estimated from
  the argmax code sequence, pooled over the dataset
* HSMM (explicit-duration) Viterbi — per-code negative-binomial duration model
  plus between-code transitions, both fitted by 3 iterations of hard EM starting
  from the argmax runs, maximum duration 100 patches

HMM vs HSMM separates "any smoothing" from "an explicit duration model".

**β ∈ {0.5, 1, 2, 4}. All four are reported; none is selected.** No GT is used
anywhere in decoding or fitting.

**Prediction (from report §4.2).** Edit, F1 and JSD change; MoF does not
(|ΔMoF| within the seed sd). A MoF gain larger than the seed sd would contradict
the boundary-vs-label decomposition and would be investigated before being
reported as a win.

---

## T4 — Unit duration vs sampling rate

Replaces the earlier single-seed E4, which scored with the per-sequence
Hungarian and is therefore not comparable to published numbers.

**Sweep P (patch duration, native frame rate).** Patch ∈ {0.5 s, 1 s, 2 s} →
HuGaDB {30, 60, 120}, LARa {25, 50, 100}, BABEL-1 {15, 30, 60} frames.
*Held constant:* frames, epochs, optimiser steps. *Varies:* patches per sequence (×2, ×½).

**Sweep R (frame rate, patch fixed at 1 s).** Downsample features and GT by
stride r: HuGaDB r ∈ {2, 4} (30, 15 fps), LARa r ∈ {2, 5} (25, 10 fps), BABEL-1
r ∈ {2, 3} (15, 10 fps), with patch = fps frames. The strides differ by dataset
because the patch has to be an integer number of frames.
*Held constant:* patches per sequence, epochs, optimiser steps. *Varies:* frames (÷r).

The 1 s native condition is shared with T1 K = C. 3 seeds per condition.

**Readouts.** Published metrics on the (possibly downsampled) GT. Predicted and GT
segment durations **in seconds**. JSD recomputed at native-frame resolution
(lengths × r) so the 20-frame bins mean the same thing in every condition. Plus
the patch-grid oracle for each condition.

**Stop condition (from experiment.md).** If predicted segment durations in
seconds are flat across Sweep P, the discovered unit is not the patch. That is a
strong finding in the opposite direction from expectation, and it is reported
without searching for a configuration that behaves as predicted.
