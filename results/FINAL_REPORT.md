# SMQ: reproduction, metric audit, error diagnosis, and intervention experiments

**Subject.** Skeleton Motion Quantization (SMQ; Gökay et al., ICCV 2025) for unsupervised
skeleton-based temporal action segmentation, and the segment-length (JSD) comparison used
against it by HiST-VQ (Ahmed et al., 2026).

**Provenance.** Repo commit `f389b8b` plus the local changes listed in §8.2. Environment: conda
env `smq`, Python 3.10.14, torch 2.1.0+cu121, single RTX 4090. Datasets: HuGaDB, LARa,
BABEL subsets 1–3, preprocessed by the repo's own scripts. Every number below is produced by
scripts in `script/repro/` and `script/exp/`, with raw outputs under `results/`.

**Pre-registration.** `results/PREREG_T1-T4.md`, written before any T1–T4 run. Deviations are
listed in §8.1.

---

## 0. Executive summary

1. **The published results reproduce exactly.** HuGaDB and LARa match the paper to two decimals
   on all five metrics; the three BABEL subsets match to ≤0.71. The evaluation protocol behind the
   published tables is the dataset-level ("global") Hungarian matching, not the per-sequence variant
   the repo also prints.
2. **The JSD metric was reverse-engineered and corroborated**, and then found to be close to
   useless for the comparisons it is used for. A near-perfect segmentation (95.5 MoF) scores 73.1
   on HuGaDB against SMQ's 86.9 — the entire range between excellent and mediocre is ~14 points,
   against a seed spread of ±1.4. On LARa, drawing segment lengths i.i.d. from the pooled
   ground-truth distribution beats SMQ by 19 points.
3. **Both papers misattribute SMQ's temporal error.** They blame the patch grid. Measured, the
   1-second grid costs 4.5 MoF on HuGaDB out of ~54 lost. Given SMQ's own boundaries and perfect
   labels the model reaches 78.8 MoF; given perfect boundaries and its own labels, 44.9. **The
   boundaries are the part that works.**
4. **The failure is quantisation granularity.** A linear probe reads the action from the
   continuous pre-quantisation latent at 79.5% and from the emitted code at 57.0%: the snap-to-code
   step with K = number-of-classes destroys ~22 points of action information the encoder already had.
5. **Enlarging the codebook raises frame accuracy at the cost of segment quality, and mostly not
   because of learning (T1, provisional, 2 seeds).** With an unsupervised geometry merge back to C,
   MoF rises by ~8–10 points on LARa and BABEL-1 — but the same merge makes JSD *worse* on all three
   datasets (LARa 74.0 → 76.6–79.8; BABEL-1 81.4 → 83.7–85.7) and lowers Edit and F1@50 on
   HuGaDB and LARa. The alternative (temporal) merge spares segment quality but does not raise MoF
   there. No single merge rule works across datasets, and on HuGaDB and BABEL-1 plain k-means on raw
   patches matches or beats the learned codebook.
6. **Duration-aware decoding gives no dependable gain (T3).** Soft-emission HMM/HSMM decoders
   under-segment badly at low sharpness (Edit −8 to −32, JSD +9 to +26). At the sharpest setting
   (β = 4) they are indistinguishable from argmax on HuGaDB, help LARa's F1@50 by +1.9 with the HMM
   only, and disagree in sign between released and retrained BABEL-1 checkpoints.
7. **The discovered unit is the patch (T4, 3 seeds).** On HuGaDB and LARa, predicted segment
   duration scales with the hand-set patch size (HuGaDB 1.0 → 3.0 → 4.9 s for 0.5 → 1 → 2 s patches;
   about 2–3 patches per segment throughout) and is indifferent to frame rate; MoF is unaffected by
   either. BABEL-1 does not follow (2.6 → 3.2 s), for reasons the data cannot resolve. JSD tracks the
   predicted duration rather than segmentation quality: on LARa a 0.5 s patch improves JSD by 11
   points while F1@50 gets worse.
8. **T2 (temporal-consistency objective) was not run**, for the reason given in §6: its
   pre-registered weight is not comparable across datasets, so any result would be uninterpretable.

---

## 1. Reproduction of the published numbers

Released checkpoints (`models/pretrained/`) scored with the repo's own metric code, unmodified.

| Dataset | Metric order: MoF / Edit / F1@10 / F1@25 / F1@50 | Published | Reproduced | max Δ |
|---|---|---|---|---|
| HuGaDB | | 42.0 / 36.1 / 38.5 / 31.5 / 24.3 | **41.99 / 36.15 / 38.50 / 31.54 / 24.30** | 0.05 |
| LARa | | 37.4 / 39.4 / 34.7 / 28.4 / 16.4 | **37.38 / 39.40 / 34.69 / 28.40 / 16.39** | 0.02 |
| BABEL-1 | | 36.6 / 38.5 / 40.9 / 32.8 / 22.3 | 36.40 / 38.25 / 40.71 / 32.60 / 22.38 | 0.25 |
| BABEL-2 | | 49.1 / 37.8 / 43.8 / 37.4 / 27.4 | 49.17 / 37.81 / 43.70 / 37.42 / 27.52 | 0.12 |
| BABEL-3 | | 40.6 / 38.6 / 38.0 / 29.3 / 19.3 | 40.37 / 38.60 / 37.58 / 28.59 / 19.35 | 0.71 |

HuGaDB and LARa reproduce to the second decimal, so those released weights are the ones behind the
tables. The BABEL checkpoints are near-equivalent but not identical reruns — which also explains a
systematic +1.4…+1.7 offset in their JSD (§2).

**Protocol note.** The repo prints both a per-sequence ("local") and a dataset-level ("global")
Hungarian matching. Only the global one matches the papers; the local one scores 10–25 points
higher. An earlier in-repo analysis (`figures/e4_sweeps/summary.csv`) used the local protocol and
is therefore not comparable to published numbers.

### 1.1 Retraining variance

Retraining from scratch with the published configuration, 3 seeds (`1538574472`, `111`, `222`):

| | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| HuGaDB retrained | 41.60 ± 1.13 | 33.74 ± 2.40 | 35.67 ± 1.94 | 28.02 ± 1.80 | 21.50 ± 2.54 |
| HuGaDB published | 42.0 | 36.1 | 38.5 | 31.5 | 24.3 |
| LARa retrained | 36.26 ± 1.61 | 38.71 ± 0.95 | 33.67 ± 1.00 | 27.35 ± 1.02 | 15.75 ± 0.75 |
| LARa published | 37.4 | 39.4 | 34.7 | 28.4 | 16.4 |

Retraining lands within ~1 sd of published on MoF and Edit; the published F1s sit at the top of the
retrained spread. **A seed does not pin a run here:** the repo sets `cudnn.deterministic = False`,
so the same seed gave MoF 41.6 in one batch of runs and 39.0 in another. All comparisons in this
report therefore use within-batch seed spread, never a single run.

---

## 2. The JSD metric: convention, corroboration, and what it can resolve

HiST-VQ defines the metric only as "histograms (with 20-frame bins) … averaged per activity and
then combined into a frame-weighted average". Three choices are left open. Sweeping them against
all five published SMQ values at once (`script/repro/jsd_sweep.py`) identifies exactly one setting:

* **log base 2**, not e — base e caps the distance at 83.3 and so cannot produce HVQ's published
  97.4. This is a hard constraint, not a preference.
* **an untruncated 20-frame grid** (0 … the longest segment in that video) plus one overflow bin.
  Truncating the grid moves HuGaDB by up to 21 points.
* **"activity" = the grouping present in the file names**: LARa scenario (L01/L02/L03); HuGaDB is a
  single category; BABEL has none. Histograms are per video over *all* segments, each normalised to 1.

| | HuGaDB | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---|---|---|---|---|
| SMQ published (HiST-VQ Tab. 3) | 87.1 | 74.2 | 72.8 | 77.0 | 81.9 |
| **Reproduced** | **86.93** | **73.89** | 74.51 | 78.71 | 83.27 |

The two datasets whose checkpoints reproduce §1 exactly also reproduce JSD exactly. Background
exclusion, edge-segment exclusion and minimum-segment-count variants were all tested and fit worse.

### 2.1 The metric has almost no dynamic range

| | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| SMQ | 86.93 | 73.89 | 74.51 |
| Sparsity floor (two disjoint halves of the *same* ground truth) | 79.58 | 66.45 | 67.86 |
| Matched-count null (lengths drawn from the pooled GT distribution) | 83.29 | **54.85** | 72.69 |
| **Patch-grid oracle** (GT majority-voted onto the patch grid — 95.5 MoF) | **73.12** | **68.04** | **52.47** |
| SMQ seed sd | ±1.38 | ±0.04 | — |

Consequences, which should accompany any use of this metric:

1. **A 95.5-MoF segmentation scores 73.1.** The band between near-perfect and SMQ is 13.8 points on
   HuGaDB and 5.9 on LARa. The published SMQ→HiST-VQ differences (HuGaDB 87.1→89.0, i.e. HiST-VQ
   *worse*; LARa 74.2→73.8) sit inside that band and, on HuGaDB, inside SMQ's own seed spread.
2. **On LARa, i.i.d. draws from the pooled GT length distribution score 54.9 — 19 points better than
   SMQ.** SMQ's length distribution is not noisy, it is systematically mis-shaped.
3. **On BABEL-2 the sparsity floor (81.6) exceeds SMQ's score (78.7)**: videos there have so few
   segments that two halves of the truth disagree more than SMQ disagrees with the whole.

### 2.2 Segment-length histograms

`figures/repro_e1_jsd/` (SVG + PNG), per dataset: ground truth vs SMQ vs the patch-grid oracle on
the metric's own 20-frame grid, plus sequence-length distributions. The BABEL-1 panel independently
confirms HiST-VQ's stated observation (GT puts 25.2% of segments in the first bin, SMQ 10.8%, and
SMQ over-fills the second). The HuGaDB panel shows the mechanism: **SMQ's predicted lengths are a
comb at exact multiples of the 60-frame patch** (45.8% at 60f, 21.4% at 120f, 10.4% at 180f)
against a broad, heavy-tailed ground truth (median 208f, 41% ≥ 240f).

---

## 3. Qualitative reproduction (paper Fig. 3)

`figures/repro_fig3/fig3_smq.svg` — four panels matching the paper's (a) HuGaDB (b) LARa
(c) BABEL-1 (d) BABEL-3, ground truth above SMQ, predictions mapped through the same dataset-level
Hungarian assignment used for the reported MoF. SMQ row only, as requested; CTE/TOT/ASOT are not
reproduced. The HuGaDB panel uses `HuGaDB_v2_various_01_00`, the example the paper's text
describes, and reproduces the described behaviour (SMQ mixing `standing_up` and `sitting_down`).
BABEL-3 has no named example, so it is chosen by rule — the sequence whose frame agreement is
closest to that dataset's global MoF — not by eye.

---

## 4. Diagnosis: where the temporal error actually comes from

All of §4 is post-hoc on the released checkpoints: no retraining.
Raw output: `results/diagnostics/*.json`.

### 4.1 The patch grid is not the bottleneck

Both papers attribute the error to patch-aligned boundaries. SMQ: *"Since actions are assigned at
the patch level, transitions can only occur at patch boundaries. Improving the detection of action
boundaries needs to be investigated in future work."* Majority-voting the ground truth onto the
patch grid measures that claim directly:

| | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| SMQ | 41.99 MoF / 24.30 F1@50 | 37.38 / 16.39 | 36.40 / 22.38 |
| **Patch-grid oracle** | **95.48 / 97.68** | **93.83 / 93.17** | **90.17 / 81.35** |

The 1-second grid costs **4.5 MoF and 2.3 F1@50** on HuGaDB. The other ~53 MoF are lost elsewhere.

### 4.2 It is a labelling error, not a boundary error

Solving one half of the problem at a time (HuGaDB):

| | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|
| SMQ (published) | 41.99 | 36.15 | 38.50 | 24.30 |
| SMQ **boundaries** + oracle labels | **78.78** | 79.82 | 87.63 | **79.17** |
| GT boundaries + SMQ **labels** | 44.85 | 43.71 | 46.32 | 41.74 |
| Both oracle (patch grid) | 95.48 | 99.52 | 99.70 | 97.68 |

Same ordering on LARa (84.40 vs 41.66 MoF) and BABEL-1 (76.86 vs 37.52).

Consistently, boundary information is barely present anywhere in the model. Using the change in the
encoder's pre-quantisation latent as a boundary detector gives ROC AUC 0.62 / 0.69 / 0.53
(HuGaDB / LARa / BABEL-1) against GT boundaries — and the *raw input features* give 0.59 / 0.69 /
0.57. The encoder adds essentially no boundary information, and SMQ's actual boundaries (top-N
F1 35.3 / 46.7 / 41.4) already match or beat peak-picking that signal (22.6 / 48.3 / 38.0).

### 4.3 Where the label information goes

Patch-level linear probes, 5-fold grouped by sequence:

| → majority GT class of the patch | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| chance (majority class) | 28.19 | 43.52 | 44.14 |
| raw features, mean-pooled | 78.79 | 64.95 | 62.62 |
| **pre-quantisation latent** | **79.50** | **70.44** | **62.20** |
| **one-hot code** | **56.95** | **47.71** | **52.17** |
| NMI(code, GT class) | 0.480 | 0.184 | 0.117 |
| codes with a *distinct* majority class | **6/10** | **3/8** | **2/5** |

* The action information **is** in the latent; quantising to K = C discards 22.6 / 22.7 / 10.0 points.
* Not codebook collapse — perplexity 7.66/10, 5.76/8, 3.93/5, zero dead codes.
* Because only 6/10, 3/8 and 2/5 codes have a distinct majority class, the **1:1 Hungarian matching
  is unsatisfiable**: several GT classes have no code that represents them. Relaxing to many-to-one
  moves MoF by +14.6 / +10.7 / +12.3.
* The encoder barely earns its keep: the latent probes within 0.7 points of the raw features on
  HuGaDB and BABEL-1. Reconstruction in joint-distance space preserves the input; it does not
  abstract it.

### 4.4 Smoothing and the duration law

A mode filter over the patch-code sequence, no retraining: HuGaDB gains a free **+5.0 F1@50** at
width 5 (24.30 → 29.33; 45.6% of its predicted runs are exactly one patch long). LARa and BABEL-1
lose from any smoothing. The predicted duration law is near-geometric by construction —
self-transition probability 0.686 / 0.650 / 0.700, mean runs 3.18 / 2.86 / 3.33 patches against GT
means of 4.96 / 3.73 / 1.91 — because each patch is quantised independently.

---

## 5. T1 — decoupling codebook size from class count

Full tables: `results/exp/SUMMARY.md`; figure: `figures/exp/t1_codebook_size.svg`.
**Concluded at 2 seeds** (3 for some K = C cells), a deviation from the pre-registered n = 3; every
cell prints its own n and verdicts are provisional. Readouts: (a) the published bijective Hungarian;
(b) many-to-one (GT-assisted, control-relative only); (c1) an unsupervised merge of the codebook to
C groups by average-linkage clustering under the quantiser's own distance; (c2) an unsupervised
merge by spectral clustering of the code-transition graph; plus linear probes and a **mini-batch
k-means control on raw patches at every K**.

### 5.1 Results

Effect rule applied at each K > C against K = C ("▲/▼" = better/worse; "—" = no effect). Baselines:
HuGaDB F1@50 22.0, Edit 32.8, JSD 89.5; LARa 15.8, 38.8, 74.0; BABEL-1 14.7, 28.2, 81.4.

| | K = C MoF | (c1) geometry merge, K > C | (c2) temporal merge, K > C | Code probe vs k-means control |
|---|---|---|---|---|
| **LARa** | 36.8 ± 1.5 | **MoF 44.4 → 46.8 (▲ every K).** JSD 76.6–79.8 (▼ every K); Edit ▼ at K ≥ 32; F1@50 ▼ at K ≥ 64 | MoF 30.0–32.9 (▼ at K ≥ 64); JSD flat except ▼ at K = 16; Edit ▼ at K ≥ 64; F1@50 ▲ at K = 32 only | SMQ ahead at **every** K (52.1 → 57.2 vs 45.5–46.8) |
| **BABEL-1** | 34.5 ± 1.1 | **MoF 42.3 → 43.1 (▲ every K).** JSD 83.7–85.7 (▼ at 3 of 4 K); Edit and F1@50 —  | MoF — except ▲ at K = 40; JSD ▲ at K = 20 only; Edit ▲ at K = 10, 40; F1@50 — | no effect; k-means **ahead** at K = 80 |
| **HuGaDB** | 40.7 ± 1.9 | MoF ▼ every K (30.6–33.4). JSD 93.9–94.3 (▼ every K); Edit and F1@50 ▼ every K | **MoF ▲ at K = 20 (43.8), — at 80/160, ▼ at 40.** JSD —; F1@50 ▲ at K = 20, 80; Edit ▲ at K = 160 | SMQ ahead to K = 40; no effect at 80/160 |

Supporting observations:

* **The published protocol (a) collapses as K grows** — HuGaDB 40.7 → 13.2, LARa 36.8 → 7.5 — as it
  must: surplus codes are matched to padded dummy labels and score as wrong. Any K > C therefore
  requires a merge step to be evaluated at all.
* **No codebook collapse at any K.** Perplexity tracks K (HuGaDB 7.7 → 71.0, LARa 5.6 → 91.9); unused
  codes stay ≤17/160. The dead-code threshold scaled with K (§8.1) did its job.
* **The latent probe is flat in K** (HuGaDB 76.5 → 79.1, LARa 70.3 → 70.8): the encoder is unchanged
  by codebook size, exactly as expected. The code probe rises but **never reaches the latent**
  (LARa 57.2 vs 70.8; HuGaDB 63.4 vs 79.1). 16× more codes narrows the quantisation gap; it does
  not close it.

### 5.2 Reading

The pre-registered hypothesis had three parts, and they come apart:

1. *Do codes carry more action information as K grows?* **Yes**, on all three datasets.
2. *Is that because of learning?* **Only on LARa.** There the learned codebook beats k-means on raw
   patches at every K by ~10 points. On HuGaDB and BABEL-1 the control matches or overtakes it —
   so most of the gain there is the trivial consequence of a larger vocabulary, not of the model.
3. *Is the gain usable without labels?* **Partly.** A merge recovers ~+8–10 MoF on LARa and
   BABEL-1, but **the merge that works differs by dataset** — geometry on LARa/BABEL-1, temporal on
   HuGaDB, each actively harmful on the other. With no label-free way to choose, this is not yet a
   method.

**MoF and segment quality trade off.** The merge that produces the MoF gain (geometry, on LARa and
BABEL-1) is the one that damages segment-level metrics: JSD worsens by 2.6–5.8 points on LARa and
2.3–4.3 on BABEL-1, and Edit and F1@50 fall on LARa. A plausible reading — untested here — is that
merging codes by codebook proximity groups together motions that alternate in time, so a
frame-accurate label sequence becomes more fragmented. The merge that respects temporal structure
(c2) keeps segment metrics roughly intact but gives no MoF gain on LARa or BABEL-1. On HuGaDB, c2
shows scattered gains (MoF at K = 20; F1@50 at K = 20 and 80; Edit at K = 160) with no consistent
pattern across K, which is what noise across 4 conditions × several metrics would look like — with
2 seeds and no multiple-comparison correction these should not be read as effects.

So enlarging the codebook can be made to raise *which frames get which label*, but no configuration
tested here improved segment quality reliably. That is consistent with §4.2: boundaries were never
what the codebook was limiting, and adding codes does not fix the boundaries the way it fixes frame
accuracy.

### 5.3 Dead-code threshold check

The main T1 runs scale the dead-code threshold with K (§8.1). To test that this was necessary, one
extra run per dataset at K = 16C used the published threshold of 10 (same seed as the scaled run; n = 1
each, so read differences against the T1 seed spread):

| | threshold | perplexity | codes unused at eval | (c1) MoF | (c2) MoF | code probe | latent probe |
|---|---|---|---|---|---|---|---|
| HuGaDB K = 160 | 10 (published) | **18.3** | **110 / 160** | 40.8 | 42.4 | 57.0 | 78.9 |
| | 1 (scaled) | 73.9 | 17 / 160 | 33.1 | 41.1 | 60.7 | 79.0 |
| LARa K = 128 | 10 | **53.4** | **50 / 128** | 38.3 | 27.8 | 53.3 | 70.6 |
| | 1 | 88.8 | 0 / 128 | 46.8 | 29.6 | 57.1 | 70.9 |
| BABEL-1 K = 80 | 10 | **7.1** | **69 / 80** | 34.0 | 35.1 | 46.6 | 64.8 |
| | 1 | 44.2 | 1 / 80 | 42.5 | 36.0 | 54.9 | 63.3 |

* **Confirmed.** With the published threshold, 50–110 codes end up unused and perplexity falls by 40–84%
  (BABEL-1: 44.2 → 7.1, an effective vocabulary barely above its class count of 5). The enlarged
  codebook is mostly not being used, so the threshold has to scale with K for T1 to test anything.
* The code probe is lower under the published threshold on all three (by 3.7–8.3 points), while the
  latent probe is unchanged — consistent with the loss occurring at quantisation, as elsewhere.
* **But the threshold is a confound for T1's merge results.** HuGaDB (c1) MoF is 40.8 under the
  published threshold versus 33.1 under the scaled one (the scaled runs' own seed spread at K = 160
  is 0.1), and LARa/BABEL-1 (c1) go the other way (38.3 vs 46.8; 34.0 vs 42.5). So the K > C merge
  conclusions in §5.1 are conditional on this choice, and the HuGaDB result in particular — "the
  geometry merge hurts" — is not robust to it. This was checked at K = 16C only, with one seed.
* (a) MoF is *higher* with the published threshold (29.8 vs 12.5 on HuGaDB) — for the wrong reason:
  fewer live codes means fewer surplus codes to be scored as wrong under the bijective matching.

---

## 5B. T3 — duration-aware decoding (post-hoc, no training)

Decoders over SMQ's per-patch code distances, none using ground truth: argmax (published), an HMM
Viterbi with self-transitions, and an explicit-duration HSMM Viterbi fitted by 3 rounds of hard EM.
Emissions are a softmax over standardised distances with sharpness β ∈ {0.5, 1, 2, 4}; all four are
reported, none selected. The HSMM decoder was checked against brute-force enumeration on 40 random
problems (exact match). Evaluated on the released checkpoints and on the K = C runs from T1 (n = 3
seeds on HuGaDB and LARa, n = 2 on BABEL-1). Δ = decoder − argmax on the same checkpoint.

| | β = 0.5 | β = 4 |
|---|---|---|
| **HuGaDB** (n = 3) | Edit −18 to −22 ▼, F1@50 −18 to −20 ▼, JSD +9 to +10 ▼; MoF — | all of MoF (+2.1/+2.2), Edit, F1@50, JSD: no effect |
| **LARa** (n = 3) | Edit −32 ▼, F1@50 −15 ▼, JSD +25 ▼; MoF +0.3 to +3.1 (no effect by the rule) | Edit −4 to −10 ▼, JSD +3 to +5 ▼; F1@50 **+1.9 ▲ (HMM only)**, HSMM —; MoF +2.4 to +2.8 (no effect by the rule) |
| **BABEL-1** (n = 2) | MoF **+5.9 ▲ (HMM)**, Edit −8 ▼, F1@50 −6 to −9 ▼, JSD +12 ▼ | all: no effect (MoF +0.6/+0.7) |

* **Low sharpness is destructive.** With soft emissions the hard-EM step inflates fitted durations
  roughly tenfold in three rounds (HuGaDB released, β = 0.5: mean duration 2.8 → 29 patches), so the
  decoder collapses the sequence to 6–33% of the true segment count.
* **At β = 4 the decoders are close to argmax** — mostly indistinguishable within seed noise, with
  isolated exceptions (LARa F1@50 +1.9 for the HMM; BABEL-1 MoF at other β).
* **The released and retrained BABEL-1 checkpoints disagree in sign** on MoF: released checkpoint
  −0.8 to −5.9 at all eight decoder settings; retrained seeds +0.6 to +5.9 at all eight (▲ at four
  of them). The retrained K = C baselines are weaker (34.5 vs 36.4 MoF), which leaves more for
  smoothing to repair; this is a reason not to generalise the BABEL-1 gain.
* On LARa the MoF change is small but consistently positive (+2.4 to +3.4 across seeds). The
  pre-registered rule compares seed *levels* and so calls it no effect; it is not a paired test and is
  conservative for paired deltas. No paired test was run.

**Conclusion.** No duration-aware decoder tested gives a dependable improvement in Edit, F1@50 or
JSD, and most settings make them substantially worse. This runs against the pre-registered
prediction that these metrics would move and MoF would not; what moved was mostly for the worse. A
hard-EM duration model on soft emissions is unstable; a better-behaved fit (fixed rather than
re-estimated durations, or soft-EM) is untested.

---

## 6. T2 — temporal-consistency objective: **not run**

Pre-registered as: add a label-free term pulling temporally adjacent patches toward the same code,
with weight λ₁ = (final-epoch commitment loss) / log C, i.e. matched to the commitment term's scale.

Applying that rule to the measured losses gives:

| | final commitment loss | λ₁ |
|---|---|---|
| HuGaDB | 17151.6 | **7448.8** |
| LARa | 5.614 | 2.700 |
| BABEL-1 | 2.815 | 1.749 |

**λ₁ differs by a factor of ~2,800–4,300 across datasets.** None of the three datasets is
normalised; they simply sit in very different native units. Measured over a sample of each dataset's
feature files:

| | feature std (overall) | per-channel std |
|---|---|---|
| HuGaDB (raw IMU) | **9,742** | 5,617 – 10,894 |
| LARa (root-centred mocap) | 232 | 98 – 475 |
| BABEL-1 (root-centred SMPL joints) | 303 | 199 – 399 |

HuGaDB's inputs are ~30–40× larger than the others, so a squared-distance quantity such as the
commitment loss is ~1,000–1,800× larger from the input scale alone. The observed loss ratio (3,055×
vs LARa, 6,093× vs BABEL-1) is larger still, so scale accounts for the bulk of it but not all — the
remainder depends on the learned latent geometry and was not separated out. Either way, λ₁ is an
artefact of native units rather than of anything about the datasets' difficulty.

The consequence is that "the same condition" would mean a different intervention strength on each
dataset, and a null on HuGaDB could not be distinguished from a badly scaled weight. Since the
pre-registration forbids retuning toward an expected result, and a scale-matching rule that does not
transfer across datasets cannot be fixed after seeing outcomes, **the experiment was not run rather
than run uninterpretably.**

This is itself a finding about the codebase: **loss-scale-derived hyperparameters are not portable
across these three datasets**, which is worth knowing before anyone tunes loss weights on HuGaDB.

To run T2 properly, the weight would have to be set by a scale-free criterion — matching gradient
norms rather than loss values, or z-scoring input features per channel first (which changes the
published configuration and would need its own baseline). Either is a clean follow-up; neither is a
variant of what was pre-registered.

---

## 7. T4 — unit duration vs frame rate

*36 runs (3 datasets × 4 conditions × 3 seeds, plus the shared 1 s native baseline from T1's K = C
runs); complete, 0 failures. Full tables: `results/exp/SUMMARY.md`; figure:
`figures/exp/t4_patch_duration.svg`. BABEL-1's baseline has n = 2, everything else n = 3.*

Design: **Sweep P** varies patch duration (0.5 s / 1 s / 2 s) at the native frame rate, holding
frames, epochs and optimiser steps constant. **Sweep R** varies frame rate (native, ~½, ~¼) with the
patch fixed at 1 second, holding patches per sequence, epochs and optimiser steps constant. The 1 s
native condition is shared with T1's K = C runs. Durations are reported in seconds and JSD is
recomputed at native-frame resolution so conditions are comparable.

Pre-registered stop condition: if predicted segment durations in seconds are flat across Sweep P,
the discovered unit is not the patch — a strong result in the opposite direction from expectation,
to be reported without searching for a configuration that behaves as predicted.

### 7.1 Sweep P — the discovered duration follows the patch on two of three datasets

Mean predicted segment duration (seconds), mean ± sd over 3 seeds; ground-truth mean in the header:

| patch | HuGaDB (GT 4.96 s) | LARa (GT 3.73 s) | BABEL-1 (GT 1.91 s) |
|---|---|---|---|
| 0.5 s | 1.0 ± 0.1 | 2.0 ± 0.1 | 2.6 ± 0.4 |
| 1 s (native) | 3.0 ± 0.2 | 2.8 ± 0.1 | 3.0 ± 0.2 (n = 2) |
| 2 s | 4.9 ± 0.3 | 4.3 ± 0.3 | 3.2 ± 0.2 |
| mean run length, in patches | 2.0 → 3.0 → 2.5 | 4.0 → 2.8 → 2.1 | 5.2 → 3.0 → 1.6 |

* **The stop condition (flat durations in seconds) is not met on HuGaDB or LARa.** A 4× change in
  patch duration moves the discovered segment length by 4.9× on HuGaDB and 2.1× on LARa. On HuGaDB
  the mean run is 2–3 patches at every setting — the "discovered unit" is the patch to within noise.
* **BABEL-1 is the exception.** Predicted duration moves only 2.6 → 3.2 s (+23%, roughly 2 sd) for
  a 4× patch change, so on this dataset the duration is close to patch-independent. This should not
  be read as SMQ finding a real timescale there: the ground truth is 1.9 s on average with 40% of
  its segments shorter than one patch, and SMQ over-predicts duration at every setting (2.6–3.2 s
  vs 1.9 s). It is equally consistent with a floor set by the codebook rather than by behaviour, and
  this experiment cannot tell those apart.
* **Only on HuGaDB does a 2 s patch happen to land on the ground-truth mean (4.9 vs 4.96 s).** On
  LARa it overshoots (4.3 vs 3.7 s) and on BABEL-1 no setting comes near. There is no evidence that
  the duration converges on anything the data contain.

### 7.2 Segment quality does not improve with any patch size

Effect rule against the 1 s native baseline (n = 3 unless noted):

| | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| **0.5 s** MoF / F1@50 / JSD | 40.3 (—) / 9.3 (▼) / 95.1 (▼) | 35.5 (—) / 13.6 (▼) / 63.0 (▲) | 34.8 (—) / 12.3 (—) / 81.9 (—) |
| **1 s** (baseline) | 40.7 / 22.0 / 89.5 | 36.8 / 15.8 / 74.0 | 34.5 / 14.7 / 81.4 |
| **2 s** MoF / F1@50 / JSD | 42.6 (—) / 26.1 (▲) / 91.6 (▼) | 30.7 (▼) / 14.5 (—) / 84.5 (▼) | 32.0 (—) / 14.3 (—) / 83.5 (▼) |
| patch-grid oracle MoF, 0.5 / 1 / 2 s | 97.8 / 95.5 / 90.9 | 96.9 / 93.8 / 87.5 | 94.9 / 90.2 / 82.6 |

* **MoF is unaffected by patch size** except a drop on LARa at 2 s (36.8 → 30.7). The one clear gain
  is HuGaDB F1@50 at 2 s (22.0 → 26.1), where the longer patch happens to match its long segments; it
  costs JSD and lowers the oracle ceiling.
* **JSD follows the predicted duration, not the segmentation quality.** On LARa the 0.5 s patch
  *improves* JSD by 11 points (74.0 → 63.0) while F1@50 gets *worse* (15.8 → 13.6) and MoF is
  unchanged; the 2 s patch does the reverse. This is a second, independent demonstration of §2.1:
  the metric rewards a shorter segment length regardless of whether the segments are right.
* The patch-grid ceiling falls steadily as the patch grows (HuGaDB 97.8 → 90.9), so a coarser unit
  does cost some attainable accuracy, but SMQ is nowhere near either ceiling.

### 7.3 Sweep R — frame rate is incidental

Patch fixed at 1 s; frames per second reduced by the factor shown (HuGaDB 60 → 30 / 15 fps; LARa
50 → 25 / 10 fps; BABEL-1 30 → 15 / 10 fps):

| | HuGaDB R2 / R4 | LARa R2 / R5 | BABEL-1 R2 / R3 |
|---|---|---|---|
| MoF (baseline 40.7 / 36.8 / 34.5) | 42.3 / 41.0 (—) | 35.6 (—) / 33.2 (▼) | 32.2 / 32.7 (—) |
| F1@50 (baseline 22.0 / 15.8 / 14.7) | 24.0 / 24.5 (—) | 16.1 / 15.9 (—) | 13.3 / 15.2 (—) |
| predicted duration, s (baseline 3.0 / 2.8 / 3.0) | 3.2 / 3.8 | 2.7 / 2.8 | 2.7 / 2.5 |

Neither MoF nor F1@50 moves with the frame rate, with one exception (LARa MoF at 10 fps, 36.8 →
33.2). Predicted durations are flat on LARa and BABEL-1 and drift up on HuGaDB (3.0 → 3.8 s, roughly
+27%, at 15 fps). The LARa JSD "improvement" (73.7 vs 74.0) is 0.3 points against a baseline sd of
0.0 — an artefact of the effect rule when one condition has almost no seed variance, not a result.

### 7.4 Reading

The pre-registered "cleanest outcome" is a dissociation — patch extent moves the results, frame rate
does not. On HuGaDB and LARa that is what happened: predicted duration tracks the patch and is
nearly indifferent to frame rate. The practical consequence is that **the segment length SMQ
discovers, and every duration-based number computed from it, including JSD, is set by a hand-chosen
constant** (the 1 s patch), not by anything in the behaviour. BABEL-1 does not show it, for reasons
the data cannot resolve.

Limits on this reading. (i) Sweep P changes the number of patches per sequence in inverse
proportion to the patch (frames, epochs and optimiser steps are fixed, but the codebook sees 2×
more or ½ as many patches), so a training-set-size contribution is not excluded. (ii) The 4 s
condition from the earlier single-seed run (predicted 12.9 s) was not repeated; that run also used
the per-sequence Hungarian protocol and is not comparable. (iii) Three datasets and three patch sizes
is a small design; the HuGaDB result is clean, the others less so.

---

## 8. Deviations, and changes to the code

### 8.1 Deviations from the pre-registration

| Deviation | Reason |
|---|---|
| **T1 concluded at n = 2** for K > C (K = C has n = 3 on HuGaDB and LARa, n = 2 on BABEL-1); the effect rule was pre-registered at n = 3 | At the user's direction, to free the GPU. Seeds 1 and 2 agreed closely, but a 2-seed sd is a weak estimate, so T1 verdicts are **provisional**. `summarize.py` was changed to issue verdicts at n ≥ 2 (`MIN_SEEDS`) and prints n in every cell. |
| **T3 ran on whatever K = C seeds existed** (n = 3 on HuGaDB and LARa, n = 2 on BABEL-1) | It consumes the T1 K = C runs. |
| **T2 not run** | §6. Skipped at the user's direction, on the scale argument in §6. |
| **Dead-code churn check — run after the fact** (3 runs, seed 1538574472, K = 16C, published threshold 10) | Motivated the threshold scaling max(1, round(10·C/K)) used in all T1 runs. Result in §5.3: the published threshold does degrade large codebooks, so the scaling was needed. It also shows that T1's merge results depend on the threshold. |
| Dead-code threshold scaled with K | As above — stated in the pre-registration, not introduced afterwards. |

### 8.2 Code changes, and why results are unaffected

All are default-off or memory-only. The published configuration runs the original code path.

| Change | Verification |
|---|---|
| Distances computed under `no_grad` (they only feed an `argmax`; no loss uses them) | bit-identical outputs, gradients, codebook and EMA counts over 6 training steps, all three dataset shapes |
| Codebook lookup by indexing instead of a one-hot product over a (patches × K × window × dim) tensor | same test |
| Distance computation chunked over the codebook | every published configuration fits in one chunk, running the original computation exactly |
| Optional TCN activation checkpointing (`--grad_checkpoint`) | bit-identical loss, all gradients, all parameters and codes over 4 Adam steps on CPU, with dropout and dead-code replacement active |
| New flags: `--dead_code_threshold`, `--tc_weight`, `--tc_beta`, `--save_every`, `--gpu_mem_fraction`, `--num_actions` | all default to published behaviour |

Inherited from the working tree before this work: `cudnn.deterministic = False` (so seeds do not pin
runs — see §1.1), and an optional `--micro_batch_size` whose VQ EMA semantics are **not** equivalent
to the published configuration. It was not used in any T1–T4 run; whether the earlier E0 seed runs
used it is unrecorded.

The memory work was needed because the published LARa configuration requires ~20 GB of TCN
activations, more than this GPU has beside a desktop. All T1–T4 runs were capped at 12 GB, at
below-normal priority, one at a time.

---

## 9. What this means

**For SMQ.** The paper's own stated limitation — that transitions can only occur at patch boundaries
— accounts for about 4.5 of the ~54 missing MoF points on HuGaDB. The binding constraint is that a
codebook sized to the number of action classes, trained purely for reconstruction, does not
partition motion by action; the 1:1 evaluation matching then converts that into unrecoverable
errors. Work aimed at boundary detection is aimed at the part that already works.

**For HiST-VQ's comparison.** Its headline segment-length metric cannot resolve the effects it is
used to claim. Between a 95-MoF oracle and SMQ there are 5.9–13.8 JSD points; seed noise is up to
1.4; on one dataset the metric's own sparsity floor exceeds a real method's score, and on another a
random draw from the correct length distribution beats the method by 19 points. Any future use of
this metric should report the patch-grid-oracle value and the seed spread beside it, and prefer the
bin-free Wasserstein-1 (already computed in `results/e1_jsd/jsd_summary.csv`).

**For the obvious fix.** Enlarging the codebook does recover part of the discarded information, and
on LARa it does so through genuine learning. But the merge that turns this into a MoF gain also
worsens JSD, Edit and F1@50, the merge that spares segment quality gives no MoF gain, and the right
merge currently has to be chosen per dataset by looking at labels. It is a partial, not yet usable,
remedy — and all of it rests on 2 seeds.

**What we would do next**, in order: (i) close the loop on granularity by testing whether a
*hierarchical* codebook — fine codes with a learned coarse level, rather than a post-hoc merge —
recovers segment quality rather than only frame accuracy; (ii) re-run T2 with a scale-free weight
criterion and per-channel normalised inputs, since the objective-level fix remains untested;
(iii) treat segment quality directly, since nothing tested here improved F1@50 or JSD reliably and the
MoF-raising merge lowered them.

---

## 10. Reproducing this report

```bash
PY=.../envs/smq/python.exe
# §1-§4: reproduction, JSD, Fig. 3, diagnosis (released checkpoints, no training)
for ds in hugadb lara babel1 babel2 babel3; do
  $PY script/repro/dump_predictions.py --dataset $ds --ckpt models/pretrained/$ds.model \
      --out results/preds/${ds}_pretrained.npz
  $PY script/repro/e0_metrics.py --preds results/preds/${ds}_pretrained.npz --out results/e0/${ds}_pretrained.json
done
$PY script/repro/jsd_sweep.py                                    # §2 convention
$PY script/repro/e1_jsd.py --preds results/preds/*_pretrained.npz
$PY script/repro/fig3_qualitative.py
$PY script/repro/d_error_decomposition.py --preds results/preds/{hugadb,lara,babel1}_pretrained.npz
for ds in hugadb lara babel1; do
  $PY script/repro/d_boundary_vs_label.py --dataset $ds --ckpt models/pretrained/$ds.model \
      --preds results/preds/${ds}_pretrained.npz
  $PY script/repro/d_what_codes_encode.py --dataset $ds --ckpt models/pretrained/$ds.model \
      --preds results/preds/${ds}_pretrained.npz
done

# §5-§7: T1/T4 (training). Resumable; finished runs are skipped.
$PY script/exp/make_jobs.py T1 && $PY script/exp/make_jobs.py T4
$PY script/exp/run_jobs.py models/exp/jobs_T1.jsonl models/exp/jobs_T4.jsonl   # one job at a time, 12 GB cap
$PY script/exp/lowprio.py script/exp/analysis_watcher.py                       # readouts, controls, T3
$PY script/exp/summarize.py                                                    # tables + figures
```

| Artifact | Path |
|---|---|
| This report | `results/FINAL_REPORT.md` |
| Stage-1 report (reproduction + diagnosis, more detail) | `results/REPORT_reproduction_and_diagnosis.md` |
| Pre-registration | `results/PREREG_T1-T4.md` |
| Auto-generated T1–T4 tables | `results/exp/SUMMARY.md`, `results/exp/summary_long.csv` |
| Per-run readouts | `results/exp/analysis/**.json` |
| Diagnosis raw output | `results/diagnostics/*.json` |
| Figures | `figures/repro_e1_jsd/`, `figures/repro_fig3/`, `figures/exp/` |
