# Phase 8a: gravity- and heading-aware relational tokens (BABEL-2/3, then LARa)

Revision 1 — 4 October 2026. Implemented in `script/week5/timescale_inspection/run_phase8a.py` (tokenizers:
`tokenizers8.py`, report: `report8a.py`, tests: `tests8a.py`); not launched. Projected **≤ 2.3 h**.
Parent: `PHASE7_LABEL_FREE_SCALE.md`. The deferred part of the original Phase 8 plan is `PHASE8B_GRAVITY_HEADING_EXTENSION.md`.

## Why

Phase 7 found that the result which holds up on BABEL-1 is the **relational tokenizer (T2)**: +12.3 F1@50 over raw
tokens at the same span (4 paired seeds), while the short-mask gain did not transfer. T2 describes each window by the
statistics of every inter-joint distance. Distances are invariant to every rotation and reflection of the body, so T2
discards:

1. **Heading and turning** (rotation about the vertical axis), including the direction of a turn.
2. **The gravity direction.** A pose and the same pose tilted or lying down give identical distances, as do an arm
   raised overhead and an arm held forward at the same joint distances.

T2's per-class losses (post-hoc; 3 seeds; global Hungarian mapping, MoF reproduces the saved values) fall on small
classes defined by exactly these properties, while it gains on the large classes:

| dataset | class | share of GT segments | recall T0 → T2 (span 8) |
|---|---|---|---|
| BABEL-1 | turn | 11% | 35 → **13** (span 2: 28) |
| BABEL-2 | kick | 5% | 32 → **15** |
| BABEL-2 | stand_up | 14% | 30 → 27 |
| BABEL-3 | wave | 2% | 43 → **8** (span 2: 23) |
| BABEL-1 | walk | 28% | 36 → 50 |
| BABEL-2 | run | 11% | 30 → 55 |
| LARa | Handling (up / down) | 11% / 9% | 16 → 57 / 17 → 56 |

Small classes are noisy under a one-to-one mapping, so this is motivation, not evidence.

**Data facts** (verified on the feature files, 4 Oct 2026):

- Positions are **already root-centred per frame** (BABEL joint 1, spine-mid; LARa segment 21, root: exactly zero).
  Global translation is absent for T0 and T2 alike; walking vs stepping in place is out of scope.
- **Axis 2 is vertical** (head-minus-feet points along axis 2 in 286/300 sampled BABEL-1 and 217/255 BABEL-2
  recordings; the rest are lying or horizontal poses). Trunk tilt exceeds 70° in 7% (BABEL-1) and 13% (BABEL-2) of
  sampled frames.
- Heading is roughly canonical across recordings (SD of the per-recording mean 25-41°) but varies within them (median
  range 93° on BABEL-1): turning is a within-recording signal.
- **BABEL joints** (NTU-25, checked geometrically): spine base 0, spine-mid 1 (root), neck 2, head 3, shoulders 4 / 8,
  hips 12 / 16, knees 13 / 17, feet 15 / 19.
- **LARa segments** follow the LARa annotation tool (`Behaviour/Annotation_Tool_LARa/.../src/data_management.py`,
  `body_segments`); each position is the segment's proximal end: head 0 (neck), head end 1, collars 2 / 12, elbows
  3 / 13, femurs 4 / 14 (hips), feet 5 / 15 (ankles), humerus 6 / 16 (shoulders), tibias 7 / 17 (knees), toes 8 / 18,
  wrists 9 / 19, wrist ends 10 / 20, lower back 11, root 21. Verified on 36,000 frames: median heights are in
  anatomical order, left/right pairs differ almost only horizontally, hip and shoulder axes agree to a median 5°.
  **The two collars are the same point**, so one T2 distance is identically zero and the R collar is dropped from the
  height block.

**Question.** Does adding yaw-invariant gravity channels and a heading-rate channel to T2 recover the gravity- and
heading-defined classes without losing T2's gains elsewhere, and does that raise F1@50?

## Tokenizers

Each keeps T2's distance block unchanged and appends blocks. Every block uses the window's own frames only (no
neighbour leakage), and none uses the absolute heading.

| key | blocks | features per window (BABEL / LARa) |
|---|---|---|
| T2 (reference, reused) | distances: mean, SD, mean \|Δ\| | 900 / 693 |
| `t5_gravity` (T5g) | + **vertical**: root-relative height of every distinct non-root joint, and trunk tilt; each by mean, SD, mean \|Δ\| | 975 / 756 |
| `t5_heading` (T5h) | + **heading rate**: mean ω, mean \|ω\|, SD ω | 903 / 696 |
| `t5_both` (T5gh) | + vertical + heading rate | 978 / 759 |

- **Heights:** axis-2 coordinates of the root-centred positions (BABEL 24 non-root joints; LARa 20: 21 non-root
  segments minus the duplicate R collar). Relative to the root, not the floor (no floor reference exists).
- **Trunk tilt:** angle between the trunk vector and +axis 2 (BABEL spine base 0 → neck 2; LARa root 21 → head 0).
- **Heading rate.** Per frame, u = (right hip − left hip) + (right shoulder − left shoulder) (BABEL 12 / 16 and
  4 / 8; LARa 4 / 14 and 6 / 16); ψ = atan2 of its horizontal part. A frame is valid if the horizontal part is at least
  30% of |u| (the body axis is within ~73° of horizontal); otherwise (for example lying on the back) it is undefined.
  Inside each window, ω = wrap(ψ_t − ψ_{t−1}) × fps over consecutive frames of that window that are both valid; the
  window's features are mean ω, mean |ω| and SD ω over those differences (zeros if there are none). The undefined
  share per dataset is reported. Signed ω restores the direction of a turn; its sign convention follows each dataset's
  axes and is never compared across datasets.

**Block weighting (declared).** After global standardization of every feature, each appended block *b* is multiplied
by s₁(distances) / s₁(b), with s₁ the block's first singular value on the fitting sample (multiple factor analysis
weighting, anchored on the distance block). The distance block is left exactly as in T2. Then PCA-64 and K-means-512 as
in T2. Known risk: the 3-feature heading block gets a large per-feature weight; the per-block share of PCA-64 variance
is reported.

## Fixed protocol

- Contextualizer: the Phase 5 reference (guard 0, ALiBi, 5,000 updates, validation-selected checkpoint) at **mask span
  2**; OT unchanged (current radius, outer cap 50, rungs 500/1,000/2,000/4,000); frame predictions saved for every
  decode.
- T2 span-2 references are read, not rebuilt: BABEL-2/3 from phase 5 (stage D), LARa from phase 6; their predictions
  from phase 7's S0 re-decodes.
- New preparations reuse the Phase 5 sampling, recording membership, split and window. `tokenizers.py`, the Phase 5-7
  runners and all existing results are not edited; the T5 keys are registered at runtime.

## Run matrix and the third-seed rule

| Stage | Work | Units |
|---|---|---|
| G | Gates G8a-c | — |
| A | BABEL-2 and BABEL-3: T5gh, T5h, T5g × seeds 111, 222 (`context`) | 12 |
| B | LARa: T5gh × seeds 111, 222 (`context`) | 2 |
| C | Third seed 1538574472, only for each (dataset, tokenizer) whose 2-seed mean F1@50 is **≥ 1.0 point above T2's mean on the same seeds** | 0-7 |

The third-seed decisions are computed from the saved units and written to `third_seed.json` (with the values) before
any third-seed job starts. "1.0" is an absolute F1@50 difference in points.

## Reading rules

- **Two-seed cells are screening, not evidence.** The declared bar (mean gain ≥ 3 F1@50 with every seed positive)
  applies only to cells with three seeds. The third seed is run *because* the first two looked promising, so its own
  value is the unbiased part; all three are reported.
- **Q1 (primary):** T5gh beats T2 on BABEL-2 (affected classes: stand_up + kick, 19% of segments). On BABEL-3 (wave: 2%)
  and LARa no gain is predicted; there the reading is **non-inferiority**: no loss ≥ 3 F1@50 or ≥ 3 MoF.
- **Q2 (class level, diagnostic):** recall of stand_up and kick (BABEL-2) and wave (BABEL-3) rises over T2 under T5g /
  T5gh, with the large classes reported alongside. Paired with T2 at the same seeds.
- **Q3 (descriptive):** the 2×2 {no, yes vertical} × {no, yes heading rate} on BABEL-2/3.
- All five metrics are reported for every comparison.
- **Not claimed:** choosing among T5g/T5h/T5gh here is development. Whichever is adopted needs a prospective test
  (Phase 8b's BABEL-1 runs are the first candidate, with the caveats already recorded for BABEL-1).

## Diagnostics

- Block weights, per-block share of PCA-64 variance, occupied codes (label-free).
- Heading-rate undefined share (label-free).
- Per-class recall/precision against T2 at the same seeds (uses labels; analysis only).
- Predicted vs ground-truth segments per recording.

## Gates (before any training)

- **G8a reduction:** the T5 code path with only the distance block (`t5_dist_only`) reproduces ts5_v1's T2 BABEL-2
  seed-111 preparation: codes, continuous features, mean, RMS, grouping and split exactly; K-means centres to 1e-4.
  A failure blocks every cell.
- **G8b geometry** (per dataset): the root is exactly zero in every sampled frame; axis 2 is the median head-minus-feet
  direction; median heights are in anatomical order (BABEL head > shoulders > hips > knees > feet; LARa head end > head >
  shoulders > lower back > hips > knees > feet); left/right hip and shoulder pairs differ mostly horizontally (median
  vertical share < 0.2) and their horizontal axes agree (median < 15°); on LARa the two collars coincide. A failure
  blocks that dataset.
- **G8c invariance:** on a real recording, T5gh features are unchanged (to 1e-4) by a rotation about axis 2, and a
  rotation growing at 1 rad/s adds 1 rad/s to mean ω in every window whose frames are all valid. A failure blocks every
  cell.

## Budget

Unit costs measured in Phases 5-7 (prep / train / `context` decode, minutes): BABEL-2 0.5 / 1.5 / 2.5; BABEL-3 0.5 /
1.6 / 2.1; LARa 2 / 2.2 / 14.4. G ~5 min; A ~52 min; B ~37 min; C at most ~45 min (all seven cells promising).
**Total ≤ 2.3 h.** Launch: `run_phase8a.ps1 -RunId ts5_v1 -Hours 4`. Output:
`results/week5/timescale_inspection/ts5_v1/phase8a/`.
