# Phase 8c: which invariance? A bounded representation diagnostic

Revision 2 — 4 October 2026 (review corrections applied; see the end). Proposal; not implemented. Projected **≤ 3.5 h**.
Parent: `PHASE8A_GRAVITY_HEADING_TOKENS.md` (results: `results/week5/timescale_inspection/ts5_v1/phase8a/REPORT.md`).
Phase 8b (BABEL-1 turn test) stays on hold until this is read.

## Why

Phase 8a was negative: no T5 tokenizer beat T2 (BABEL-2 T5gh −4.43, BABEL-3 −7.57 F1@50, two seeds each). Its declared
block weighting multiplied the three heading features by 9.6-13.0 and the vertical block by 2.6-3.4 (dataset dependent).
The distance coordinates then accounted for 37-63% of the retained PCA-64 variance. That share describes where retained
variance sits. It does **not** measure how much of T2's useful information survived. **The weighting is a credible
explanation for 8a's loss, but a hypothesis**, and 8a cannot separate it from the effect of the added channels.

A second hypothesis comes from the symmetry of the data. T2's distances are invariant to the full rigid group E(3):
every rotation, translation and reflection. The nuisance in these features is narrower. Translation is already removed
(positions are root-centred), heading is a real nuisance (median 93° of turning within a BABEL-1 recording), and
gravity and chirality may carry action information. The matching symmetry would be rotation about the vertical axis
only.

Phase 8c is a **bounded representation diagnostic** with three separate questions, each answered by a comparison that
changes one thing:

| question | comparison | the one thing that differs |
|---|---|---|
| Q1 Does removing absolute heading (keeping gravity and chirality) help? | T6 vs T0 | per-window rotation |
| Q2 With the same temporal summary, do canonical coordinates match distances? | T6s vs T2 | coordinates vs pairwise distances |
| Q3 Did 8a's weighting cause its loss? | T5gh-std vs T5gh (8a) | block weighting |

T6 vs T2 is also reported, but descriptively only: those two differ in invariance, temporal summary (flattened frames vs
window statistics), dimensional layout and the fitted PCA/codebook, so their gap cannot be assigned to any one factor.

A positive result identifies a better **input** for the later learned-token work (Phase 9). Canonicalization itself is
established practice (HumanML3D's facing-frame positions, Guo et al. 2022; Keypoint-MoSeq's egocentric alignment, Weinreb
et al. 2024) and is not claimed as a contribution.

## Tokenizers

### Heading canonicalization (shared by T6 and T6s)

- **Input:** root-centred joint positions only (BABEL 25 joints; LARa position channels 3-5, 22 segments). On BABEL this
  is exactly T0's input. On LARa, T0 also uses orientation channels 0-2, which cannot be rotated as vectors, so T6/T6s use
  positions only and T6 vs T0 is not read on LARa.
- **Facing direction:** per frame, u = (right hip − left hip) + (right shoulder − left shoulder), with phase 8a's indices
  and validity rule (BABEL 12/16, 4/8; LARa 4/14, 6/16; valid if the horizontal part is ≥ 30% of |u|). ψ_w is the heading
  at window *w*'s middle frame, or at the nearest valid frame inside the same window. If no frame in the window is valid,
  the window is not rotated, and this is counted.
- **Rotation:** every frame of window *w* is rotated about axis 2 by −ψ_w. One rotation per window: absolute heading is
  removed; gravity (axis 2 untouched) and chirality (no reflection) are kept; turning inside the window is kept (the joints
  rotate across the window's frames). ψ_w uses only the window's own frames.

### T6 `t6_heading_canonical`: T0's layout

The canonical window is flattened on the zero-padded grid exactly as T0, standardized per coordinate, then PCA-64 and
K-means-512. One homogeneous block: no weighting rule.

### T6s `t6s_heading_canonical_stats`: T2's temporal summary on canonical coordinates

Each canonical coordinate (BABEL 24 non-root joints × 3; LARa 21 × 3) is summarised over the window's own frames by mean,
SD and mean |frame-to-frame change|, exactly as T2 summarises each distance (`tokenizers.window_stats`). BABEL 216 features,
LARa 189. Then standardization, PCA-64 and K-means-512 as T2. One block: no weighting rule. T6s and T2 share the temporal
summary and quantizer, and differ in what is summarised: canonical coordinates (SO(2) invariant) vs pairwise distances
(E(3) invariant). They still differ in dimension and fitted codebook, which the report states.

### T5gh-std `t5_both_std`: 8a's T5gh without block weighting

Phase 8a's T5gh features (distances + vertical + heading rate) with plain global standardization only. Before PCA the
blocks carry variance in proportion to their feature counts (BABEL: distances 92.0%, vertical 7.7%, heading 0.3%). This
choice follows 8a's result and is labelled development; it is the parameter-free default, not a tuned value. With the
heading block at 0.3% of the pre-PCA variance, T5gh-std mainly tests the vertical channels.

## Fixed protocol (as Phase 8a)

- Contextualizer: the Phase 5 reference (guard 0, ALiBi, 5,000 updates, validation-selected checkpoint) at **mask span 2**.
  OT unchanged. Frame predictions are saved.
- References read, never rebuilt (span 2, `context`, seeds 111 / 222 / 1538574472):
  - **T2:** BABEL-2/3 (phase 5), LARa (phase 6).
  - **T0:** BABEL-2/3 (phase 6 S1a).
  - **T5gh (weighted):** phase 8a has seeds 111 and 222 only. Stage A adds seed 1538574472 for BABEL-2/3, so that Q3 is
    paired on three seeds.
- New code in new files (`tokenizers8c.py`, `run_phase8c.py`, `report8c.py`, `tests8c.py`). Earlier files and results are
  not edited.

## Run matrix: three fixed seeds, no screening

The 8a screening rule (two seeds, third only if ≥ 1.0 above T2) is **not** used here: it compared every candidate with T2,
so a tokenizer that answered Q1 or Q3 positively might never receive a third seed. Given the short runtime, every cell runs
all three seeds.

| Stage | Work | Units |
|---|---|---|
| G | Gates G8c-1 to G8c-4, plus G8b geometry inherited from phase 8a | — |
| A | BABEL-2, BABEL-3: T6, T6s, T5gh-std × 3 seeds; weighted T5gh seed 1538574472 | 20 |
| B | LARa: T6, T6s × 3 seeds | 6 |

Order: stage A by seed (all four new BABEL cells for seed 111, then 222, then 1538574472), so a budget stop leaves complete
seeds rather than complete tokenizers.

## Reading rules

Primary metric F1@50 at the last rung, paired by seed. The bar is a mean gain ≥ 3 with every seed in the same direction.
All five metrics are reported for every comparison.

- **Q1 (T6 vs T0), BABEL-2/3:** a pass means removing absolute heading per window, while keeping gravity and chirality,
  improves on raw coordinates.
- **Q2 (T6s vs T2), BABEL-2/3 and LARa:** a pass either way means, at the same temporal summary, one invariance gives better
  tokens than the other. Within the bar means **comparable performance**, not equivalent representations.
- **Q3 (T5gh-std vs weighted T5gh), BABEL-2/3:** a pass means the weighting caused at least part of 8a's loss. T5gh-std vs T2
  is also reported. Matching T2 within the bar means **comparable performance**, not that the added channels are inert.
- **Descriptive only:** T6 vs T2 (several factors differ); class-level recall for 8a's targeted classes (BABEL-2 stand_up and
  kick, BABEL-3 wave) and the large classes; per-block PCA variance shares; windows left unrotated.
- **LARa:** T6/T6s vs T2 as non-inferiority (no loss ≥ 3 F1@50 or MoF).

## What follows

- **Phase 9 input** (learned HuBERT-style tokens): T6s or T6 if Q2 favours canonical coordinates on both BABEL datasets with
  no LARa loss; otherwise T2. The choice is made on BABEL-2/3 and LARa, so Phase 9's comparisons must be prospective.
- **Phase 8b (BABEL-1 turn test):** warranted only if Q1 or Q2 favours canonical coordinates, since those keep within-window
  turning and are the representations a `turn` test would probe. If neither does, 8b is dropped.

## Gates

- **G8c-1 reduction (T6):** with the rotation disabled (ψ_w ≡ 0), the T6 path reproduces ts5_v1's T0 BABEL-2 seed-111
  preparation: codes, continuous features, mean, RMS, grouping and split exactly; K-means centres to 1e-4.
- **G8c-2 reduction (T6s):** with the rotation disabled, T6s window features equal `tokenizers.window_stats` applied to the
  root-centred coordinates (exact).
- **G8c-3 invariance:** on a real BABEL-2 and LARa recording, T6 and T6s features are unchanged (to 1e-3) by a constant
  rotation about axis 2. A rotation growing at 1 rad/s changes them in windows with valid frames (within-window turning
  kept). A mirror image (axis 0 negated) changes them (chirality kept).
- **G8c-4 reduction (T5gh-std):** with only the distance block and no weighting, the path reproduces T2's preparation, as
  G8a.

A G8c-1 or G8c-3 failure blocks T6 and T6s; a G8c-2 failure blocks T6s; a G8c-4 failure blocks T5gh-std.

## Budget

Unit costs measured in phases 5-8a (prep / train / `context` decode, minutes): BABEL-2 0.5 / 1.5 / 2.5; BABEL-3 0.5 / 1.6 /
2.1; LARa 2 / 2.2 / 14.4. G ~5 min; A 20 units ~90 min; B 6 units ~112 min (LARa decodes ~14 min each). **Total ≤ 3.5 h.** Launch (once implemented):
`run_phase8c.ps1 -RunId ts5_v1 -Hours 5`. Output: `results/week5/timescale_inspection/ts5_v1/phase8c/`.

## Revision 2: review corrections

1. The 37-63% figure is the distance coordinates' share of retained PCA variance, not the share of T2's information that
   survived. Weighting is stated as a hypothesis, tested by Q3. The weights are given per dataset (heading 9.6-13.0,
   vertical 2.6-3.4).
2. The third-seed trigger against T2 is removed; every cell runs three fixed seeds, and weighted T5gh gets its third seed so
   Q3 is paired on three.
3. T6 vs T2 is descriptive only, because the two differ in invariance, temporal summary, layout and fitted codebook. T6s is
   added to isolate the invariance at a fixed temporal summary (Q2).
4. Matching performance is read as comparable performance, not as inert channels or equivalent representations.
5. Canonicalization is framed as an input choice for Phase 9, not a contribution. Phase 8b runs only if Q1 or Q2 favours
   canonical coordinates.
