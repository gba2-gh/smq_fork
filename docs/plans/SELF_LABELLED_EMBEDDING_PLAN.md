# Self-labelled action embedding with cross-recording temporal OT (BABEL-2, stage 1)

Plan for implementation · 8 October 2026 · Code goes in `script/self_labelled/`; outputs in `results/self_labelled/<run_id>/`.

## 1. Why this, and what would count as success

Every test since Milestone 1 froze the representation and changed what surrounds it: inputs, decoders, post-hoc grouping. None moved the step where it fails. On BABEL-2 a label-chosen grouping of the 512 codes reaches about 73% window accuracy, while every label-free grouping reaches 48–55%, around the 51% majority baseline. The failures share one signature: the unsupervised structure organizes by recording, not by action.
- Temporal adjacency was no better than within-recording shuffled adjacency.
- Only 3.5–5.6% of the co-occurrence affinity goes beyond recording membership.
- The learned contextualizers raised recording NMI.

Competitive methods (SMQ, MASQ, HiST-VQ; ASOT and online-clustering methods for video) **learn the embedding with the segmentation objective**. Our matched B0 is only the decode half of ASOT. This plan adds the trainable half, with two levers aimed at the measured failure:
1. **Cross-recording equipartition.** Pseudo-labels come from one temporal OT problem per **batch of recordings**, with the relaxed state marginal applied to the whole batch instead of to each recording. A state cannot become "this recording", and individual recordings no longer have to use every state (B0's per-recording uniform marginal is unrealistic: a BABEL-2 recording contains about 2 classes).
2. **Exact 3D nuisance augmentation.** The two views in swapped prediction differ only in viewpoint, mirror and body scale, all applied to the 3D skeleton before MotionBERT. The embedding is pushed to discard them. This also targets the head-on failure: run recall was 0.50 vs 0.93 when the runner faced the camera.

**Success for this stage:** the learned embedding beats the frozen-MotionBERT baseline under the development rule (Section 6), and the levers explain the gain. **Success for the project** comes later: competitive results against SMQ and HiST-VQ on the standard benchmarks (Section 9). Novelty rests on the skeleton-specific anti-shortcut design and its controls; temporal OT with online clustering has video prior art (for example Kumar et al., CVPR 2022, to be verified in the related-work pass).

## 2. Data and protocol

| item | setting |
|---|---|
| Dataset | BABEL-2 only (development data; diagnosed with labels in earlier tests) |
| Base features | frozen MotionBERT `MB_release` (SHA `02a9ee01…`), 243-frame clips, mean over joints, 30 fps; window = 8 frames |
| Window input | the same window statistics as the baseline (mean, sd, mean \|Δ\|; 1536-d), standardized with fitting-recording statistics only |
| Folds | the 5 label-free recording folds of `pe_babel2_001` (`script/pretrained_encoder/run.py::make_folds`) |
| Seeds | 111, 222, 1538574472, 333, 444 (model initialization, batch order, augmentation draws) |
| Training population | fitting recordings of the fold only; the held-out fold never enters training, normalization or prototype fitting |
| Labels | evaluation and diagnostics only |
| Evaluation | matched B0 decode (Section 5), held-out fold, final endpoint, one Hungarian mapping per fold, mean over folds; MoF and F1@50 co-primary |

## 3. Augmentation bank (precomputed, label-free)

For every recording, precompute **8 augmented MotionBERT feature sequences**. Each variant draws, with a saved seed:
- a rotation about the vertical axis, uniform on [0°, 360°);
- a mirror with probability 0.5 (x flipped **and** left/right joints swapped, so the result is a valid body);
- a body scale, uniform on [0.85, 1.15], applied to positions before the camera transform.

The augmentations must preserve time exactly, so each window of one variant corresponds to the same window of every other. Speed augmentation is excluded for that reason.

The canonical (unaugmented) feature is the baseline's: 270° view, no mirror, scale 1. It is used for evaluation and normalization statistics.

Cost: 255 recordings × 8 variants at about 0.1–0.2 s each, a few minutes on the RTX 4090. Cache under `results/self_labelled/aug_bank/<checkpoint>_<bank_seed>/`.

## 4. Model and training

**Encoder:**
- linear 1536 → 256;
- two temporal convolution blocks: kernel 5, dilation 1, GELU (explicit erf, see the GELU bug note in memory), LayerNorm, residual;
- linear 256 → 128;
- L2 normalization.

The receptive field is ±4 windows (about ±1.07 s), deliberately short, given that MotionBERT's 4-second context already blurs short events. Padding is masked at recording ends.

**Prototypes:** C = 5 learnable unit vectors in 128 dimensions. They are initialized by k-means++ on the initial embeddings of fitting windows (label-free, seeded) and frozen for the first 300 updates.

**One update:**
1. Sample a batch of 16 fitting recordings and two distinct variants per recording (views a and b).
2. Embed both views. The cost is D = 1 − z·μ.
3. **Pseudo-labels (no gradient):** solve the **batched temporal OT** (Section 4.1) on each view's cost, with 25 accepted mirror steps from a cold start. Rows of T/p are the soft labels Q_a and Q_b.
4. **Swapped prediction:** loss = CE(Q_a, softmax(z_b·μ/τ)) + CE(Q_b, softmax(z_a·μ/τ)), with τ = 0.1, averaged over windows and weighted by real window length.
5. Update with AdamW (lr 1e-3, weight decay 1e-4, gradient clipping 1.0), then renormalize the prototypes.

**Budget:** 3,000 updates. No early stopping and no checkpoint selection by labels; the final model is the one evaluated.

### 4.1 Batched temporal OT

This is the existing objective (`script/action_transport/transport.py`, coefficients `SETTING_T`: a 0.7, β 0.3, λ 0.05, ε 0.07; kernel half-width `fps // window`), applied to the concatenation of a batch's recordings with:
- **row marginal:** each window's real frames divided by the total frames in the batch;
- **structural term:** block-diagonal. The kernel is applied separately within each recording and never across recording boundaries;
- **column marginal:** relaxed toward uniform q = 1/C over the **whole batch**. This is the equipartition lever.

It is implemented on GPU by reusing `gradient_log`, `objective_log` and `row_project_log`, with a block-diagonal `structural_grad`. **Parity:** with a batch of one recording, the solver must reproduce `transport.solve_inner` within float64 tolerance.

### 4.2 Collapse and shortcut monitoring

These are logged, not used for selection:
- the state-usage entropy of the pseudo-labels and the share of states below 1% of the mass, per update;
- the training loss;
- at the end: recording NMI of the decoded states, and the same-recording nearest-neighbour share of the embeddings (excluding overlapping windows), against the frozen baseline.

A collapse (one state above 90% of the mass at the final update) is reported as a failure of that cell. It is not tuned away.

## 5. Arms and evaluation

Every arm is evaluated with the **same matched B0 decoder**, fitted on fitting recordings and decoded on the held-out fold. Only the input changes.

| arm | input to the decoder | role |
|---|---|---|
| `F0` | frozen MotionBERT, K-means codes (`centre`, as in `pe_babel2_001`) | current baseline; existing cells reused for 3 seeds, 2 new seeds built |
| `F1` | frozen MotionBERT window features, PCA-64 continuous, no quantization | control: separates "learning" from "dropping quantization" |
| `L_full` | learned embedding: batch equipartition and 3D augmentation | **the method** |
| `L_noeq` | learned embedding: **per-recording** marginal (standard ASOT), with augmentation | equipartition ablation |
| `L_noaug` | learned embedding: batch equipartition; the two views are the canonical feature with independent feature dropout (p = 0.1) | augmentation ablation |

**Secondary (reported, not gated):** `L_full` decoded with its own learned prototypes (per-recording temporal OT at evaluation) instead of a refitted B0.

## 6. Decision rules (declared before any scored run)

All comparisons use means over the 5 folds, paired by seed over 5 seeds. **Development rule:** mean MoF and mean F1@50 each ≥ −1, and at least one ≥ +2 with at least 4 of 5 seeds positive.

| question | comparison | outcome |
|---|---|---|
| Is the learned embedding the new baseline? | `L_full` vs `F0` | **ADOPT** if the rule holds |
| Is the gain from learning, not from dropping quantization? | `L_full` vs `F1` | the "learning" claim needs the rule to hold |
| Does equipartition matter? | `L_full` vs `L_noeq` | the claim needs the rule to hold |
| Does 3D augmentation matter? | `L_full` vs `L_noaug` | the claim needs the rule to hold |
| Does it address the shortcut? | recording NMI and same-recording neighbour share, `L_full` vs `F0` | reported; a reduction is expected if the mechanism works |

**Progression:** if `L_full` is adopted **and** beats `F1` under the rule, run stage 2 (Section 9). Otherwise stop this method on BABEL-2. Do not tune τ, the update budget, the receptive field or the OT coefficients in response to scores; any such change is a new declared experiment.

The gate is INCONCLUSIVE unless every cell completed and the test suite's verification record matches the code hash.

## 7. Implementation (`script/self_labelled/`)

| module | content |
|---|---|
| `config.py` | everything declared above; a code hash that includes the reused modules |
| `augment.py` | rotation, mirror with joint swap, scale; bank builder (reuses `script/pretrained_encoder/encoder.py`) |
| `batched_ot.py` | the batched temporal OT of Section 4.1 |
| `model.py` | encoder and prototypes |
| `train.py` | the update loop, monitoring and checkpoint at the final update |
| `evaluate.py` | embedding extraction for all recordings; matched B0 through `script/compositional_dynamics` with run-time overrides as in `script/pretrained_encoder/run.py::apply_overrides`; continuous-input B0 for `F1` and the learned arms |
| `run.py`, `run.ps1` | preflight, smoke, dry-run, run; cells (arm × seed × fold) in fresh child processes; resume by completed-cell identity; REPORT.md with the rules |
| `tests.py` | the tests below; a pass writes `results/self_labelled/verification.json` |

**Tests (all must pass before a scored run):**
1. **Augmentation correctness:**
   - the rotation is a proper rotation about the vertical axis;
   - the mirror swaps left/right joints, so bone lengths are preserved;
   - the scale changes every pairwise distance by the same factor;
   - time alignment is exact.
2. **Batched OT:**
   - a batch of one reproduces `solve_inner`;
   - rows sum to p;
   - perturbing one recording's cost leaves another recording's structural gradient unchanged (no cross-recording kernel leakage);
   - the batch marginal, not the per-recording one, appears in the objective (checked on a synthetic batch where each recording uses one state: per-recording equipartition must spread it, batch equipartition need not).
3. **No evaluation leakage:** changing held-out recordings' features or labels cannot change the trained weights, normalization or prototypes.
4. **Pseudo-labels carry no gradient;** the swapped loss matches a hand computation.
5. **Gradient check:** float32 gradients match float64 for the encoder (guards the CPU GELU bug).
6. **Determinism:** the same seed gives identical weights after 20 updates on CPU.
7. **Collapse detector:** fires on a synthetic collapsed assignment.
8. **Decision rules:** every branch, with complete and incomplete cells.

**Smoke run** (under 10 minutes, throwaway run id containing "smoke"): seed 111, fold0, `L_full` and `F1`, 100 updates, a short solver ladder. It exercises the bank, training, monitoring, the B0 decode and the report.

**Pilot** (not a gate, user-launched, about 30–45 minutes): `L_full` and `F1`, seed 111, all 5 folds, full budget. Its purpose is to see whether training is stable (no collapse, falling loss) before the full grid is spent. Its cells are reused by the full run if identities match.

## 8. Cost

- Bank: a few minutes.
- One training (3,000 updates; batch OT of 25 steps on two views of about 16 × 40 windows, on GPU): estimated 3–6 minutes, to be measured in the smoke.
- B0 decodes: about 2 minutes per cell.
- Full grid: 3 learned arms × 5 seeds × 5 folds = 75 trainings, plus decode cells for 5 arms × 25 (with `F0`'s 15 existing cells reused). About 6–9 hours, inside one overnight cap.

Implementation: about one day, including tests and smoke.

## 9. Stage 2 (declared now; runs only after stage 1 passes)

Run the frozen configuration on the standard benchmarks with each dataset's standard protocol, against published SMQ, MASQ and HiST-VQ numbers and our SMQ retrains:
- **BABEL-1 and BABEL-3:** MotionBERT input. Both were touched by earlier phases, so this is cross-dataset evidence, not untouched confirmation.
- **LARa:** the method's input must change. LARa's skeleton needs a mapping to H36M, or T2 is used.
- **HuGaDB:** inertial sensors, so MotionBERT cannot be used. The method has to be input-agnostic there (T2-style or raw window features with augmentations suitable for inertial data).

This is a known gap in the current design. It is the first decision to make if stage 1 passes, and it should be declared before stage 2 runs.

## 10. Risks, stated up front

- **Collapse or trivial pseudo-labels:** self-labelling with few classes is prone to it. Prototype freezing, the relaxed batch marginal and the monitoring are the safeguards; there is no tuning loop.
- **Inherited bias:** pseudo-labels start from the frozen features' structure and may only reproduce it. The swapped views and the augmentations are what can move it.
- **Little transfer from augmentation:** MotionBERT's 2D lifting fails for back-facing views, so features of rotated views may be noisy rather than informative.
- **Possible null result:** batch equipartition may still allow sequence-level shortcuts within a recording. The shortcut diagnostics will show it.

## 11. Implementation notes (8 October 2026)

- Code: `script/self_labelled/` (`config.py`, `augment.py`, `batched_ot.py`, `model.py`, `train.py`, `evaluate.py`, `run.py`, `run.ps1`, `tests.py`). The decode reuses `script/compositional_dynamics` (ladder, scoring, association) and `script/action_transport` (continuous fit); `F0` cells come from `pe_babel2_001` for seeds 111/222/1538574472, and the two new seeds build their own via the `pretrained_encoder` stages.
- **Batched OT detail not in Section 4.1:** each recording's structural kernel is divided by that recording's frame share π_r. With the column marginal off, or applied per recording (`L_noeq`), the batch problem then equals the per-recording problems. This is tested. A batch of one reproduces `transport.solve_inner` to 1e-9.
- **Initialization of F1 and the learned arms' decoder:** length-weighted k-means++ (n_init 5, seeded) on the fitting windows' unit vectors. `F0` keeps the historical code-grouping initializer, so F0 vs F1 also differs in initializer. F1 vs L_* differ only in the input.
- **Tests: 13 pass** (augmentation geometry; feature/window alignment; batched OT parity, no kernel leakage, decoupling, equipartition behaviour; no evaluation leakage and determinism; swapped loss by hand; float32 vs float64 gradients; masking and receptive field ±4; collapse flag; layout and folds; decision rules; cache identity). A test found one real bug before any run: the bank builder passed `mirror=` to a function expecting a different keyword.
- **Smoke run (seed 111, fold0, 40 fit / 10 eval recordings, 100 updates; numbers are not evidence):** all four arms ran end to end. Training takes about 0.1 s per update. In the first 100 updates the pseudo-label state usage became uneven (`L_full`: one state at 51%, another near 0) but did not trip the 90% collapse flag; whether this persists over 3,000 updates is exactly what the pilot is for. No change to the loss, marginal weight or budget is licensed by the smoke.
- **Cost revision:** a learned cell is about 5 minutes of training plus about 4 minutes of decodes (the refit-B0 decode and the secondary own-prototype decode), so about 9 minutes, not the 3–6 minutes of Section 8. The full grid (75 learned cells plus F1 and the new-seed F0 cells) is about 12–13 hours. The driver runs arm-major (`F0`, `F1`, `L_full`, `L_noeq`, `L_noaug`) and resumes by completed-cell identity, so `L_full` finishes first and the run can be continued over two nights.
- Launch commands are in `run.ps1`: pilot (`-Arms F1,L_full -Seeds 111`, about 1 hour), then the full grid with `-Hours 12 -ReuseCompleted`.

## 12. Pre-run corrections after review (8 October 2026; no scored run existed)

A review of the implementation found one design flaw and four smaller issues. All were fixed before any pilot or grid.

**Design change: the training marginal weight.** Section 4.1 used the decoder's relaxed-marginal weight (λ = 0.05) for the training pseudo-labels. A label-free check (400 updates, 80 fitting recordings, only state usage measured) showed that at λ = 0.05 both `L_full` and `L_noeq` fell to about 2 effective states (usage entropy 0.70–0.72 against 1.61 for uniform; largest state 54–58% of the mass), so the equipartition lever was inert and the two arms were indistinguishable. At λ = 1 usage stayed balanced and the lever behaved as designed (`L_noeq` 0.195–0.205 per state; `L_full` 0.167–0.234, so recordings may differ while the batch stays balanced).
- **Rule now:** training pseudo-labels use **λ_train = 1.0**. Everything else in the objective (a, β, ε, kernel) and the whole decoding path keep `SETTING_T`, so decoding stays matched to the baseline.
- **Selection basis:** chosen by this state-usage criterion alone. No action label or score was involved. Re-check on the final code with `L_noaug` as well: min share 0.167, max 0.236, entropy 1.60.
- The earlier smoke numbers were produced at λ = 0.05 and are void.
- The equipartition unit test previously passed only because it used λ = 2. A new test fixes λ_train and requires more than ln 3 usage entropy in a short run.

**Smaller fixes and declarations**
1. **Collapse flag:** now computed from the mean over the last 50 updates, and also set when usage entropy is below ln 3 (fewer than three effective states), not only when one state exceeds 90% of the mass at the final update. It is reported, never used for selection.
2. **F1 builds its own preparation** (`pretrained_encoder.features.prepare`, a no-op when it exists), so F1 no longer depends on F0 having run first for seeds 333 and 444. Verified on seed 333.
3. **What a seed varies:** model initialization, batch order, which two variants form a pair, and (for the decode) the k-means++ initialization and the seed-specific tokenizer. The augmentation bank itself is fixed (`BANK_SEED`); seeds do not redraw augmentation parameters.
4. **GPU runs are not bit-reproducible** (the determinism test runs on CPU). Cells are reproducible only up to GPU floating-point effects, so a rerun of a cell may differ in the last digits.
5. **Runtime:** about 12 hours for the grid, more than one night; resume by completed-cell identity. The pilot (`-Arms F1,L_full -Seeds 111`, about 1.2 h) should be inspected first for loss, state usage and collapse flags.

## 13. Validity fixes after a second review (8 October 2026; still no scored run)

The objective and architecture are unchanged. These fixes concern the validity of the eventual verdict, the inputs of F1, and what is saved.

1. **Verdict only on the complete, current grid.** `decide` returns INCONCLUSIVE unless all of these hold:
   - the run covers exactly the declared five seeds, five folds and five arms;
   - every contributing cell is complete **and** its recorded identity equals the identity the current code assigns (stale, smoke, invalid or interrupted cells are not counted);
   - the test record matches the code hash.

   F0 cells are re-checked against the identity the current pretrained-encoder / Milestone 1 code would give them (preparation hash and code hash). All 15 existing F0 cells verify.
2. **Failed pseudo-label solves invalidate the cell.** If either view's batched OT returns `ok = False` or fewer than 25 accepted steps, training stops before the optimizer step, the cell is marked `invalid` with the failing update recorded, and it cannot contribute to a verdict.
3. **Collapse monitoring beyond soft mass.** Balanced soft assignments do not show distinct states: identical embeddings and prototypes give uniform soft usage, as a test demonstrates. Each update now also logs:
   - hard-assignment occupancy (effective number of states);
   - the effective rank of the batch embeddings;
   - the largest prototype–prototype cosine.

   A cell is flagged when any of these indicates collapse: soft usage, fewer than 3 effective hard states, embedding effective rank below 3, or two prototypes with cosine above 0.95. Flags use the mean of the last 50 updates. **Collapsed cells stay in the averages, so the method is not flattered, but any collapsed `L_full` cell blocks adoption** (verdict BLOCKED_BY_COLLAPSE). The full monitoring history is saved per cell.
4. **F1 is fitting-only, like the learned arms.** F1 now fits its scaler and PCA-64 on the fold's fitting windows only (a seeded sample of at most 10,000 windows) and projects every window. It previously reused the historical, globally fitted preparation, whose sample included held-out windows (1,884 of 9,779 for seed 111, fold 0). **F0 remains the one declared transductive baseline**, because its cells are reused from `pe_babel2_001`. F1 vs the learned arms now differ only in learning.
5. **Shortcut diagnostic at two exclusion radii, for every arm including F0.**
   - **Shared context** (±8 windows, the historical radius): neighbours may still share MotionBERT source frames.
   - **Support-disjoint:** excludes every same-recording window whose source support may overlap. MotionBERT clips are 121.5 source frames long and a window adds its own frames plus ±4 windows of the learned encoder, so the radius is ±42 windows.

   BABEL recordings are short, so at ±42 many queries have no eligible same-recording candidate. That share is therefore reported only over queries with one, together with how many there were. Neither statistic is gated.
6. **Saved artifacts.** For both the primary (refit B0) and secondary (own-prototype) decodes, each cell now saves:
   - the fitted decoder prototypes and their initialization;
   - evaluation costs;
   - raw window states at both endpoints;
   - raw and mapped frame predictions;
   - the fitting objective trace and per-recording solver status, steps and residual.

   Learned cells also save the full training log, the model, and float32 embeddings of every fitting and held-out window.
7. **Provenance of the augmentation bank.** The bank directory is now named by a hash of the augmentation and extraction code (`augment.py`, `encoder.py`, `features.py`, the tokenizer window statistics), the checkpoint and every bank setting. A change in any of them builds a new bank instead of reusing stale features. The earlier bank directory is obsolete.
8. **Scope of the anti-shortcut claim.** Batch equipartition relaxes the requirement that every recording use every state. It does **not** prevent recording-based states: several recordings can occupy different states while the batch stays balanced, as the synthetic test itself shows. Any anti-shortcut statement must rest on the diagnostics and the `L_noeq` ablation, not on the mechanism.

**Status:** tests: 22 pass. A smoke run of all arms plus a verified F0 cell completed. Bank rebuild under the new identity: a few minutes at the start of the pilot.
