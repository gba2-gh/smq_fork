# SMQ: reproduction, JSD corroboration, Fig. 3, and a diagnosis of the temporal error

Repo commit at time of run: `f389b8b` (working tree has the local `main.py` /
`model.py` modifications listed in §0.3).
Environment: conda env `smq`, Python 3.10.14, torch 2.1.0+cu121, RTX 4090.
All numbers below come from `script/repro/*` and the JSON/CSV under `results/`.

---

## 1. E0 — reproduction of the published numbers

Evaluated with the repo's own metric code, unmodified, on the released
checkpoints in `models/pretrained/`. The published protocol is the
**dataset-level (global) Hungarian** matching — the per-sequence ("local")
variant the code also prints scores 10–25 points higher and is *not* what the
papers report.

| Dataset | Metric | Published | Reproduced | Δ |
|---|---|---|---|---|
| HuGaDB | MoF / Edit / F1@10 / F1@25 / F1@50 | 42.0 / 36.1 / 38.5 / 31.5 / 24.3 | **41.99 / 36.15 / 38.50 / 31.54 / 24.30** | ≤0.05 |
| LARa | | 37.4 / 39.4 / 34.7 / 28.4 / 16.4 | **37.38 / 39.40 / 34.69 / 28.40 / 16.39** | ≤0.02 |
| BABEL-1 | | 36.6 / 38.5 / 40.9 / 32.8 / 22.3 | **36.40 / 38.25 / 40.71 / 32.60 / 22.38** | ≤0.25 |
| BABEL-2 | | 49.1 / 37.8 / 43.8 / 37.4 / 27.4 | **49.17 / 37.81 / 43.70 / 37.42 / 27.52** | ≤0.12 |
| BABEL-3 | | 40.6 / 38.6 / 38.0 / 29.3 / 19.3 | **40.37 / 38.60 / 37.58 / 28.59 / 19.35** | ≤0.71 |

HuGaDB and LARa reproduce to the second decimal — those released checkpoints are
the exact ones behind the tables. The three BABEL checkpoints are within 0.7 but
not exact, so they are near-equivalent reruns rather than the table's own
weights. This matters in §2.

### 1.1 Seed variance (retraining from scratch, 3 seeds, published config)

Seeds `1538574472` (default), `111`, `222`; 30 epochs; nothing else changed.

| Dataset | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| HuGaDB (mean ± sd) | 41.60 ± 1.13 | 33.74 ± 2.40 | 35.67 ± 1.94 | 28.02 ± 1.80 | 21.50 ± 2.54 |
| HuGaDB published | 42.0 | 36.1 | 38.5 | 31.5 | 24.3 |
| LARa (mean ± sd) | 36.26 ± 1.61 | 38.71 ± 0.95 | 33.67 ± 1.00 | 27.35 ± 1.02 | 15.75 ± 0.75 |
| LARa published | 37.4 | 39.4 | 34.7 | 28.4 | 16.4 |

Retraining lands within ~1 sd of published on MoF and Edit; the published
F1s sit at the top of the retrained spread (+1.1 sd on HuGaDB F1@50).
**Significance threshold for anything downstream: ~1.1–1.6 MoF, ~2.5 F1@50 on
HuGaDB, ~0.8 F1@50 on LARa.**

---

## 2. JSD — convention identified and corroborated

HiST-VQ specifies the metric only as "histograms (with 20-frame bins) ...
averaged per activity and then combined into a frame-weighted average". Three
things are left open. `script/repro/jsd_sweep.py` sweeps them against all five
published SMQ values at once; exactly one setting fits:

* **log base 2**, not e. Base e caps the distance at 83.3×100 and therefore
  cannot produce HVQ's published 97.4 — this is a hard constraint, not a
  preference.
* **untruncated 20-frame grid** (0 … longest segment in that video) plus one
  overflow bin. Truncating the grid at 10–30 bins moves HuGaDB by up to 21 points.
* **"activity" = the grouping present in the file names**: LARa scenario
  (L01/L02/L03); HuGaDB is a single category (`various`); BABEL has none. For
  HuGaDB and BABEL the frame-weighted step is therefore a plain mean over videos.

Per-video histograms over *all* segments (not per class), each normalised to 1.

| | HuGaDB | LARa | BABEL-1 | BABEL-2 | BABEL-3 |
|---|---|---|---|---|---|
| SMQ published (HiST-VQ Tab. 3) | 87.1 | 74.2 | 72.8 | 77.0 | 81.9 |
| **Reproduced** | **86.93** | **73.89** | 74.51 | 78.71 | 83.27 |
| Δ | −0.17 | −0.31 | +1.71 | +1.71 | +1.37 |

The two datasets whose checkpoints reproduce the MoF table exactly also
reproduce JSD exactly. The three BABEL subsets are offset by a consistent
+1.4…+1.7, in line with their checkpoints not being the table's own weights.
Background-exclusion, edge-segment-exclusion and minimum-segment-count variants
were all tested and all fit *worse* (MAE 1.32–5.37 vs 1.05), so the convention
above stands.

### 2.1 The metric has almost no dynamic range

Computed alongside (`results/e1_jsd/jsd_summary.csv`):

| | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| SMQ | 86.93 | 73.89 | 74.51 |
| Sparsity floor (two disjoint halves of the *same* GT) | 79.58 | 66.45 | 67.86 |
| Matched-count null (lengths drawn from the pooled GT distribution) | 83.29 | 54.85 | 72.69 |
| **Patch-grid oracle** (GT majority-voted onto the patch grid — 95.5 MoF!) | **73.12** | **68.04** | **52.47** |
| SMQ seed sd (3 seeds) | ±1.38 | ±0.04 | — |

Three things follow, and they should be stated in any writeup that uses JSD:

1. **A near-perfect segmentation scores 73.1 on HuGaDB.** The whole band between
   "95.5 MoF oracle" and "42.0 MoF SMQ" is 13.8 JSD points wide, and 5.9 points
   wide on LARa. The published SMQ→HiST-VQ differences (HuGaDB 87.1→89.0, i.e.
   HiST-VQ *worse*; LARa 74.2→73.8, 0.4 better) sit inside that band and, on
   HuGaDB, inside SMQ's own ±1.38 seed spread.
2. On LARa, simply **drawing segment lengths i.i.d. from the pooled ground-truth
   length distribution scores 54.9 — 19 points better than SMQ's 73.9.** SMQ's
   length distribution is not noisy, it is systematically mis-shaped.
3. On BABEL-2 the sparsity floor (81.6) is *above* SMQ's score (78.7), because
   videos there carry so few GT segments that two halves of the truth disagree
   more than SMQ does with the whole. The metric is dominated by sample size.

### 2.2 Histograms

`figures/repro_e1_jsd/` — per dataset, segment-length histograms (GT vs SMQ vs
patch-grid oracle) on the metric's own 20-frame grid, and sequence-length
histograms. SVG + PNG.

The BABEL-1 panel independently confirms HiST-VQ's stated observation: GT puts
25.2% of segments in the first bin and SMQ only 10.8%, while SMQ over-fills the
second bin (33.6% vs 28.7%). The HuGaDB panel shows the mechanism plainly —
**SMQ's predicted lengths are a comb at exact multiples of the 60-frame patch**
(45.8% at 60f, 21.4% at 120f, 10.4% at 180f), against a broad heavy-tailed GT
(median 208f, 41% ≥ 240f).

---

## 3. Figure 3 — qualitative reproduction (SMQ row only)

`figures/repro_fig3/fig3_smq.svg` / `.png`. Four panels matching the paper's
(a) HuGaDB (b) LARa (c) BABEL Subset-1 (d) BABEL Subset-3, each showing the GT
strip above the SMQ strip, with predictions mapped through the same
dataset-level Hungarian assignment used for the reported MoF.

Sequences: the HuGaDB panel uses `HuGaDB_v2_various_01_00`, which is the example
the paper's text describes ("it mixes standing up and sitting down") — and the
reproduction shows exactly that, SMQ labelling the sit→stand transitions
`standing_up` where the GT alternates `sitting_down`/`standing_up`. LARa
`L02_S15_R04` and BABEL-1 `929` are the repo's own published examples. The
BABEL-3 panel has no named example, so it is selected by rule, not by eye: the
sequence whose frame agreement is closest to that dataset's global MoF.

---

## 4. Diagnosis: what actually introduces the temporal error

All of §4 is post-hoc on the cached code sequences plus one forward pass for
latents — no retraining. `results/diagnostics/*.json`.

### 4.1 The patch grid is not the bottleneck

Both papers attribute the temporal error to the patch grid. SMQ: *"Since actions
are assigned at the patch level, transitions can only occur at patch boundaries.
Improving the detection of action boundaries needs to be investigated in future
work."* That is measurable — majority-vote the GT onto the patch grid and score it:

| | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| SMQ | 41.99 MoF / 24.30 F1@50 | 37.38 / 16.39 | 36.40 / 22.38 |
| **Patch-grid oracle** | **95.48 / 97.68** | **93.83 / 93.17** | **90.17 / 81.35** |

The 1-second patch grid costs **4.5 MoF and 2.3 F1@50** on HuGaDB. The remaining
53.5 MoF and 73.4 F1@50 are lost to something else.

### 4.2 It is a labelling error, not a boundary error

Solve one half of the problem at a time:

| HuGaDB | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|
| SMQ (published) | 41.99 | 36.15 | 38.50 | 24.30 |
| SMQ **boundaries** + oracle labels | **78.78** | 79.82 | 87.63 | **79.17** |
| GT boundaries + SMQ **labels** | 44.85 | 43.71 | 46.32 | 41.74 |
| Both oracle (patch grid) | 95.48 | 99.52 | 99.70 | 97.68 |

Same ordering on LARa (84.40 vs 41.66 MoF) and BABEL-1 (76.86 vs 37.52).

**SMQ's boundaries already support ~77–84 MoF and 51–79 F1@50. Its codes,
handed perfect boundaries, reach only 38–45.** The temporal localisation is the
part that works; the code→action correspondence is the part that fails.

Consistent with that, boundary information is barely present anywhere in the
model. Using the change in the encoder's pre-quantisation latent as a boundary
detector gives ROC AUC 0.62 / 0.69 / 0.53 (HuGaDB / LARa / BABEL-1) against GT
boundaries — and the *raw input features* give 0.59 / 0.69 / 0.57. The encoder
adds essentially no boundary information, and SMQ's actual boundaries (top-N F1
35.3 / 46.7 / 41.4) are already at or above what peak-picking that signal
achieves (22.6 / 48.3 / 38.0). There is little headroom left on boundaries.

### 4.3 Where the label information goes: quantisation at K = #classes

Patch-level linear probes, 5-fold grouped by sequence (`d_what_codes_encode.py`):

| → majority GT class of the patch | HuGaDB | LARa | BABEL-1 |
|---|---|---|---|
| chance (majority class) | 28.19 | 43.52 | 44.14 |
| raw features, mean-pooled | 78.79 | 64.95 | 62.62 |
| **pre-quantisation latent, mean-pooled** | **79.50** | **70.44** | **62.20** |
| **one-hot code** | **56.95** | **47.71** | **52.17** |
| NMI(code, GT class) | 0.480 | 0.184 | 0.117 |
| codes used / codebook | 10/10 | 8/8 | 5/5 |
| codes with a *distinct* majority class | **6/10** | **3/8** | **2/5** |

Read together with §4.2 this is the whole story:

* The action information **is** in the latent (79.5% on HuGaDB vs 28% chance).
* Quantising to K = number-of-classes throws away **22.6 points** of it on
  HuGaDB, 22.7 on LARa, 10.0 on BABEL-1.
* It is not codebook collapse — perplexity 7.66/10, 5.76/8, 3.93/5, zero dead
  codes. The codes are all used; they just do not partition by action.
* Because only 6/10, 3/8 and 2/5 codes have a distinct majority class, the
  **1:1 Hungarian bijection is unsatisfiable**: several GT classes cannot be
  matched to any code that actually represents them. Relaxing the bijection to a
  many-to-one map (each code → its own majority class) moves MoF by
  **+14.6 / +10.7 / +12.3** — while Edit and F1 stay flat or drop, because
  merging codes removes segments rather than placing them better.
* The encoder is also barely earning its keep: the latent probes within 0.7
  points of the raw features on HuGaDB and BABEL-1. Reconstruction in
  joint-distance space preserves the input; it does not abstract it.

### 4.4 Smoothing is a dead end (mostly)

A mode filter over the patch-level code sequence, no retraining:

| width | HuGaDB MoF / F1@50 | LARa | BABEL-1 |
|---|---|---|---|
| none | 41.99 / 24.30 | 37.38 / 16.39 | 36.40 / 22.38 |
| 3 | 42.40 / 26.57 | 37.52 / 16.31 | 35.82 / 22.29 |
| 5 | **42.65 / 29.33** | 36.38 / 13.05 | 34.64 / 15.51 |
| 7 | 41.57 / 22.70 | 35.77 / 9.69 | 34.01 / 13.71 |

HuGaDB gains a free **+5.0 F1@50** at width 5 — 45.6% of its predicted runs are
exactly one patch long and 9.5% are sandwiched singletons. LARa and BABEL lose
from any smoothing. Worth applying on HuGaDB, not a general fix, and it does not
touch the labelling problem.

The predicted duration law is near-geometric by construction: self-transition
probability 0.686 / 0.650 / 0.700, giving mean runs of 3.18 / 2.86 / 3.33
patches against GT means of 4.96 / 3.73 / 1.91. Each patch is quantised
independently, so there is no duration model at all.

### 4.5 Summary of the diagnosis

The published account — that temporal error comes from boundaries being locked
to the patch grid — accounts for about 4.5 of the ~54 missing MoF points on
HuGaDB. The dominant cause is that **the codebook, sized to the number of action
classes and trained purely for reconstruction, does not partition motion by
action**, and the evaluation's 1:1 matching then converts that into
unrecoverable errors. HiST-VQ's hierarchy addresses the symptom this points at
(more codes at a finer level) but its headline JSD metric cannot resolve the
effect: the metric's full range between a 95-MoF oracle and SMQ is 5.9–13.8
points, with a seed sd of up to 1.4.

---

## 5. Proposed experiments

Ordered by (evidence that it targets the real bottleneck) × (cost). Training is
cheap here: 30 epochs is ~8 min on HuGaDB, ~13 min on LARa, ~10 min on BABEL-1
on one RTX 4090, so a 3-seed sweep over 4 settings × 3 datasets is ≈6 GPU-hours.

**Every comparison is against the §1.1 seed sd, not against zero.**

### T1 — Decouple codebook size from class count *(highest value)*

The direct test of §4.3. Retrain with `--num_actions` K ∈ {#classes, 2×, 4×, 8×,
16×}, 3 seeds, all three datasets. Evaluate three ways: the published bijective
Hungarian; many-to-one (code → majority class); and Hungarian after merging codes
to #classes by agglomerative clustering of codebook vectors (an unsupervised
merge, so it stays a legitimate unsupervised protocol).

*Predicted, and what falsifies it:* the one-hot-code probe should climb toward
the 79.5 / 70.4 / 62.2 latent ceiling as K grows. If it plateaus well below —
the codes stop gaining action information past some K — then the encoder, not
the codebook, is the limit, and T2 becomes the priority instead. Report the
probe accuracy alongside MoF: it is the mechanistic readout, MoF is the outcome.
Watch for collapse (perplexity, dead-code count) at large K.

### T2 — Give the quantiser a temporal-consistency objective

§4.3 shows the encoder preserves rather than abstracts (latent ≈ raw). Add to
the existing reconstruction + commitment loss a term that pulls patches within a
temporally contiguous window toward the same code and pushes distant patches
apart — a self-supervised proxy for "same action", requiring no labels. Ablate:
reconstruction only (baseline), +consistency, consistency only.

*Readout:* the one-hot-code probe and NMI(code, class), not just MoF. If the
probe rises without MoF rising, the bijection is the binding constraint and T1's
matching variants are what unlock it.

### T3 — Add a duration/transition model over codes

§4.4: durations are geometric because patches are quantised independently.
Decode the patch-code sequence with an HSMM (explicit duration distribution
fitted unsupervised to the code sequence itself) instead of argmax-per-patch.
This is post-hoc on existing checkpoints — hours, not GPU-days.

*Caveat, from §4.2:* boundaries are not the bottleneck, so expect this to move
Edit, F1 and JSD but **not** MoF. That is still worth having, because Edit/F1/JSD
are where SMQ looks weakest and where HiST-VQ claims its win. Treat a MoF gain
here as suspicious.

### T4 — Patch size vs. frame rate (disentangling the unit from the sampling)

§2.2 shows predicted lengths are a comb on the patch grid, but §4.1 shows the
grid costs little in MoF. So the open question is whether patch size matters for
*label* quality, not boundary quality. Sweep patch size ∈ {0.5 s, 1 s, 2 s} at
fixed frame rate, and frame rate at fixed patch *duration*. Hold the number of
patches seen in training constant across conditions, or the comparison confounds
timescale with training-set size — state explicitly how this was done.

*Report durations in seconds, never frames*, or the conditions are incomparable.

### T5 — Retire or recalibrate JSD in this project's reporting

Not a training experiment, a reporting decision. §2.1 shows JSD spans ~6–14
points between a 95-MoF oracle and SMQ, with a 1.4-point seed sd on HuGaDB and a
sparsity floor that can exceed a real method's score. Recommendation: keep
reporting raw JSD for comparability with the two papers, but always alongside
(a) the patch-grid-oracle value and (b) the seed sd, and lead with the
bin-free Wasserstein-1 (already in `jsd_summary.csv`; HuGaDB 164 frames = 0.55×
mean GT length, LARa 51 = 0.27×, BABEL-1 21 = 0.36×), which has no bin-width
parameter and no per-video sparsity problem.

### Not recommended

Further work on boundary detection (§4.2: latent-change AUC 0.53–0.69, barely
above the raw-feature baseline, and SMQ's existing boundaries already support
77–84 MoF). Mode filtering beyond the free HuGaDB win in §4.4.

---

## 6. Reproducing this report

```bash
PY=.../envs/smq/python.exe
for ds in hugadb lara babel1 babel2 babel3; do
  $PY script/repro/dump_predictions.py --dataset $ds \
      --ckpt models/pretrained/$ds.model --out results/preds/${ds}_pretrained.npz
  $PY script/repro/e0_metrics.py --preds results/preds/${ds}_pretrained.npz \
      --out results/e0/${ds}_pretrained.json
done
$PY script/repro/jsd_sweep.py                                  # §2 convention
$PY script/repro/e1_jsd.py --preds results/preds/*_pretrained.npz   # §2, figures
$PY script/repro/fig3_qualitative.py                           # §3
$PY script/repro/d_error_decomposition.py --preds results/preds/{hugadb,lara,babel1}_pretrained.npz
for ds in hugadb lara babel1; do
  $PY script/repro/d_boundary_vs_label.py  --dataset $ds --ckpt models/pretrained/$ds.model \
      --preds results/preds/${ds}_pretrained.npz
  $PY script/repro/d_what_codes_encode.py  --dataset $ds --ckpt models/pretrained/$ds.model \
      --preds results/preds/${ds}_pretrained.npz
done
```

### Deviations from the published configuration

Inherited from the working tree, flagged for the record:

* `main.py`: `torch.backends.cudnn.deterministic` set to `False` (was `True`);
  seed exposed as a CLI flag. Affects GPU float-rounding order only.
* `model.py`: optional gradient-accumulation (`--micro_batch_size`) and a
  heartbeat file. With the flag unset the training step is identical to the
  original; when set, the VQ EMA update runs once per micro-batch rather than
  once per batch, which is **not** equivalent — record whether it was used for
  any run whose numbers are reported.
* Nothing in `src/model/` was modified. All §1 numbers come from the released
  checkpoints through the repo's own `eval_utils`.
