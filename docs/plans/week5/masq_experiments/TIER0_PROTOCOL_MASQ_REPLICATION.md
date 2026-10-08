# Tier 0: label-free protocol and MASQ replication

Plan · 5 October 2026 · Version 1. Not implemented. Implementation instructions: `AGENT_INSTRUCTIONS.md` (this folder).

Tier 0 is the prerequisite for the architecture work (Tier 1: a closed-loop learned encoder; Tier 2: the compositional
dynamics decoder in `docs/plans/COMPOSITIONAL_ACTION_DYNAMICS_PROPOSAL.md`). It has two deliverables:

- **A. A label-free protocol.** Every later result must be interpretable without asking which settings were tuned on the
  evaluation labels.
- **B. A seeded replication of MASQ** (arXiv 2608.29891, 30 August 2026; single runs, not peer reviewed), on the SMQ code
  in this repository. MASQ is our main external comparison and the strongest candidate encoder for Tier 1. Its numbers
  are single runs with an unspecified validation protocol, so they cannot anchor decisions until they are reproduced.

Implementation about 1 day. Runs (user-launched): Stage 1 about 4.9 h, Stage 2 about 4.6 h.

## Part A: label-free protocol

### A.1 Where labels influenced the current configuration

| Setting | Current value | How it was chosen | Label use |
|---|---|---|---|
| Tokenizer | T2 relational (PCA-64, K-means-512) | Phase 5 stage A, compared by F1@50 | development selection |
| Token window | 15 / 12 / 8 frames (≈0.25 s) | inherited from `rkx_v2`; window sizes compared by F1 in Phase 5 stage E | development selection |
| Mask span | 2 tokens (≈0.5 s) | Phase 5 stage D on BABEL-2/3 by F1; its transfer to LARa was predicted from a ratio to the mean labelled action duration (Phase 6) | development selection; depends on label statistics |
| OT temporal radius | fps // window (≈1 s) | original rule, kept after the labelled Phase 1 screen | examined with labels |
| OT coefficients | a=0.7, b=0.3, λ=0.05, ε=0.07 | ASOT published training values | none (published) |
| Number of OT states / SMQ codes | C (number of classes) | benchmark convention | count of classes only, shared by all baselines |
| Contextualizer checkpoint | best validation loss over 5,000 updates | rule | none |
| SMQ checkpoint | final epoch (30) | published rule | none |
| MASQ dropout rate p | 0.25 / 0.20 / 0.50 (HuGaDB / LARa / BABEL) | "validation protocol", not described | unknown; probably labels |
| MASQ velocity weight | 0.001 | sweep, criterion not described | unknown |

All five datasets have informed earlier decisions. No untouched evaluation set exists.

### A.2 Rules from Tier 0 onward

- **P1 Freeze.** The current T2 recipe becomes the frozen *development reference*, with its history (A.1) disclosed. It is
  not tuned further. The mask span stays at 2 tokens everywhere; architectures that inherit it do not retune it. Span 8
  is reported once as a sensitivity.
- **P2 New settings.** Every new hyperparameter is one of three things, declared before any scored run:
  - a published value, with its source;
  - a fixed value in physical units, such as seconds;
  - a label-free rule.

  If several values are run, all are reported. The pre-declared value is the result; no winner is picked by evaluation
  scores.
- **P3 Checkpoints.** Either a fixed budget (final epoch), or a label-free validation criterion declared in advance.
  Never chosen by evaluation scores.
- **P4 Benchmark conventions.** Number of codes/states K = C and the dataset-global Hungarian evaluator
  (`src/model/eval_utils.evaluate_global_hungarian`) are shared by every baseline and declared as such. Per-sequence
  Hungarian scores are reported separately and never mixed with global ones.
- **P5 Seeds and reading.** Seeds 111, 222, 1538574472. The bar is a mean difference of at least 3 points with every seed
  in the same direction, read only with three seeds. **MoF and F1@50 are co-primary from now on.** We previously led with
  F1@50; the external comparisons lead with MoF, and choosing the metric that favours us is not acceptable. Edit and
  F1@10/25 are always reported.
- **P6 Evaluation settings.** The benchmark's transductive, pooled setting (train and score on all recordings) is primary,
  because it is what the published tables use. A subject-disjoint inductive setting on HuGaDB and LARa is secondary.
- **P7 Status labels.** A result is *development* until its code and configuration are frozen and it is run once
  afterwards (*prospective*). Reimplemented baselines are labelled as reimplementations.

## Part B: MASQ replication

### B.1 What MASQ changes over SMQ

From the paper (method section and Tables 3-6), checked against this repository's SMQ code:

| MASQ component | Already in SMQ here? | Change needed |
|---|---|---|
| Joint-disentangled two-stage TCN encoder/decoder, 1-s patch VQ, EMA codebook with dead-code replacement | yes (`src/model/smq.py`, `ms_tcn.py`, `motion_quantizer.py`) | none |
| Inter-joint distance-matrix reconstruction loss | **yes**: `--joint_distance_recons`, on by default | none |
| Commitment loss | yes | none |
| **Joint-Level Structured Dropout (JLSD)** | no | new |
| **Mask-aware velocity loss** | no | new |

MASQ is therefore SMQ plus two training-time changes. Inference is unchanged.

- **JLSD.** During training only, draw a Bernoulli mask over (sample, person, joint) with drop probability p. Zero that
  joint's encoder latent across all time steps and channels, scale the kept latents by 1/(1−p), then quantize. The code
  is chosen from the visible joints, and the decoder reconstructs every joint from it.
- **Velocity loss.** λ_vel times the mean over valid points of ‖(X̂_t − X̂_{t−1}) − (X_t − X_{t−1})‖² per joint. It is
  counted only for joints that are neither dropped nor padded, normalized by the number of valid points.

### B.2 Details the paper leaves open, and declared choices

| Open detail | Declared choice | Reason |
|---|---|---|
| Epochs, learning rate, batch size, patch size, latent size, codebook settings | SMQ defaults (30 epochs, lr 5e-4, batch 8 / 8 / 32, patch 60 / 50 / 30 frames = 1 s, latent 16 per joint) | MASQ states it keeps SMQ's backbone and inference |
| Velocity loss space | input feature space, all channels per joint | the formula uses X per joint |
| Velocity normalization | squared L2 norm over a joint's channels, averaged over valid (frame pair, joint) points; a pair is valid when both frames are unpadded and the joint is kept | equation 5: sum over t and v of ‖·‖², divided by N_vld valid spatio-temporal points |
| Dropout scope | the batch's mask per sample; also excluded from the velocity loss | "joints not dropped in the current batch" |
| Reconstruction loss under dropout | all joints, as in SMQ | the decoder must infer dropped joints |
| Checkpoint | final epoch (30) | SMQ rule; label-free (P3) |
| p and λ_vel | published values in arm M-pub; a uniform label-free p in arm M-fix | the published p were selected by an undescribed protocol |

### B.3 Arms

| Arm | JLSD | Velocity loss | Datasets | Stage |
|---|---|---|---|---|
| **S0** SMQ retrained (matched control) | — | — | all five | 1 |
| **M-pub** MASQ, published settings | p = 0.25 / 0.20 / 0.50 | λ = 0.001 | all five | 1 |
| **M-JL** JLSD only | published p | — | all five | 2 |
| **M-fix** MASQ with one label-free p | p = 0.25 everywhere | λ = 0.001 | LARa, BABEL-1/2/3 (identical to M-pub on HuGaDB) | 2 |

- **Why S0 is retrained** rather than borrowed from the existing `e0` seeds: the pairing must share code, machine and
  seeds. Its HuGaDB and LARa results are checked against the existing retrained spread (B.5, G5).
- **Why p = 0.25 for M-fix:** it is MASQ's published HuGaDB value and the frame-masking default in their Table 5.
  It is a fixed choice, not a selected one.
- **Not run:** frame masking (Table 5), codebook-size sweep (Table 6), velocity-weight sweep. These are secondary to
  "does the published configuration reproduce".

### B.4 Datasets, seeds and scoring

HuGaDB, LARa, BABEL-1, BABEL-2, BABEL-3, with the data and loaders the SMQ code already uses. Seeds 111, 222,
1538574472. Each arm is trained once per (dataset, seed) and scored at epoch 30 with the SMQ evaluator: global Hungarian
(primary, matches the published tables) and per-sequence Hungarian (reported separately). MoF, Edit, F1@10/25/50.

### B.5 Gates (before any scored run)

- **G1 Off switch.** With p = 0 and λ_vel = 0, the MASQ model and loss give identical outputs, losses and gradients to
  the unmodified SMQ model on fixed inputs (CPU, deterministic).
- **G2 JLSD.** On a synthetic batch:
  - dropped joints are zero at every time step and channel;
  - kept latents are scaled by 1/(1−p);
  - the empirical drop rate is within binomial tolerance of p;
  - the mask is off in `eval()`;
  - padded frames stay padded.
- **G3 Velocity loss.** On constructed tensors, the loss equals a hand-computed value, ignores dropped and padded
  joints, and is zero when the reconstruction equals the input.
- **G4 Evaluator.** Re-scoring the released SMQ checkpoints reproduces `results/e0/*_pretrained.json` exactly (HuGaDB and
  LARa; already shown once, re-run in minutes).
- **G5 Control sanity** (after Stage 1, before reading comparisons). S0 on HuGaDB and LARa falls within 2 SD of the
  earlier retrained spread (`results/REPORT_reproduction_and_diagnosis.md` §1.1: HuGaDB MoF 41.60 ± 1.13, F1@50
  21.50 ± 2.54; LARa MoF 36.26 ± 1.61, F1@50 15.75 ± 0.75). If not, the comparison is reported but flagged.
- **Smoke run** (≤10 min, throwaway run ID): one epoch per arm on BABEL-3. Checks losses are finite, the codebook is
  occupied and checkpoints are written, and times each step for the budget.

### B.6 Measures and diagnostics

- **Segmentation:** all five metrics, both mappings; predicted vs ground-truth segments per recording; code occupancy
  (K = C codes) and perplexity.
- **Identity diagnostics:** these answer whether JLSD reduces subject or recording dependence (the open question from
  Phase 9). Label-free unless stated.
  - **Codes.** Adjusted mutual information (AMI, chance-corrected because K is small) of native codes with subject (HuGaDB,
    LARa; subject parsed from the recording ID) and with recording (all datasets).
  - **Latents.** Same-recording share of nearest neighbours of pre-quantization latents, excluding a temporal
    neighbourhood of at least one patch, on a fixed sample of 10,000 frames per dataset.
  - **Per class** (uses labels; analysis only). Subject AMI within each action class, so that "different subjects
    perform different actions" is not read as identity encoding.
- **Training:** reconstruction, commitment and velocity losses per epoch; dead-code replacements per epoch.

### B.7 Declared comparisons and replication criteria

Per dataset, paired by seed, MoF and F1@50 co-primary:

- **R1 (main): M-pub vs S0.** Does MASQ's training change improve SMQ? Passing the bar on MoF counts as a reproduced
  gain.
- **R2: M-pub vs published MASQ.** Per dataset, *reproduced* if the 3-seed mean is within 3 points of the published
  value on both MoF and F1@50 (published: HuGaDB 52.3 / 39.5, LARa 47.2 / 22.2, BABEL-1 46.7 / 21.0, BABEL-2
  55.8 / 28.4, BABEL-3 47.7 / 24.5).
- **R3: S0 vs published SMQ.** Locates our baseline; retrained SMQ sat about 1 SD below published on F1 before.
- **R4 (Stage 2): M-pub vs M-JL**, the velocity loss's contribution; **M-fix vs M-pub**, whether the published per-dataset
  p matters (within the bar = the label-free p suffices).
- **R5 (descriptive): identity.** S0 vs M-pub on subject/recording AMI and the same-recording neighbour share.

**Overall verdict, declared now.**
- **Replicated:** R1 passes on MoF in at least 3 of 5 datasets, including LARa (the largest claimed gain and our largest
  MoF gap).
- **Partly replicated:** R1 passes on 1-2 datasets, or the MoF gain is real but R2 fails.
- **Not replicated:** R1 is within the bar everywhere.

### B.8 What the verdict decides for Tier 1

| Verdict | Tier 1 encoder | Reporting |
|---|---|---|
| Replicated | MASQ encoder; M-fix instead of M-pub if R4 shows them within the bar (label-free p preferred) | MASQ cited with our seeded numbers alongside |
| Partly replicated | MASQ encoder only where R1 passed; otherwise SMQ | discrepancy reported per dataset |
| Not replicated | SMQ encoder | our reimplementation and its result reported as such |

R5 sets Tier 2 priorities. If JLSD lowers subject AMI on LARa (all seeds lower), spatial masking becomes a declared
component of the Tier 1 encoder. If it doesn't, body-retargeting consistency (proposal §5) is the only identity
mechanism in the plan and moves earlier.

### B.9 Budget (RTX 4090)

Measured for SMQ (30 epochs): HuGaDB about 8 min, LARa about 13 min, BABEL-1 about 10 min. BABEL-2 and -3 have 255 and
183 recordings against BABEL-1's 1,960, so ≤3 min is budgeted for each. JLSD and the velocity term are element-wise, so
≤10% overhead is assumed, and the smoke run replaces these estimates. Scoring is budgeted at 2 min per checkpoint.

| Stage | Content | Estimate |
|---|---|---|
| Gates + smoke | G1-G4, one-epoch smoke | ≤20 min |
| Stage 1 | S0 and M-pub × 5 datasets × 3 seeds (30 trainings, 30 scorings) | about 4.9 h |
| Stage 2 | M-JL × 5 × 3, M-fix × 4 × 3 (27 trainings and scorings) | about 4.6 h |
| Diagnostics + report | identity measures, tables | ≤30 min |

Stage 1 alone decides R1-R3 and the verdict. Stage 2 can run later without blocking Tier 1.

### B.10 Implementation

- **New directory, no edits to shared files.** `script/week5/masq_experiments/`:
  - `model_masq.py`: `MASQModel(SMQModel)`, overriding `forward` to insert JLSD after the encoder;
  - `train_masq.py`: the SMQ training loop with the velocity term; same optimizer, schedule, checkpoint cadence and
    loss logging, plus the velocity loss;
  - `diagnostics.py`, `run.py` (stages, resumable), `report.py`, `tests.py`, `run.ps1`.

  `main.py`, `model.py` and `src/` are not modified, because earlier experiments depend on them.
- **Outputs:** checkpoints under `models/week5/masq_experiments/<run-id>/<arm>/<dataset>/seed<k>/`, results under
  `results/week5/masq_experiments/<run-id>/`. A manifest records the code hash, arm settings, seeds and timings.
- **Execution:** each (arm, dataset, seed) runs in its own process. Finished units are cached by identity and a crashed
  unit is retried, as in the Phase 9 driver.
- **Kept for Tier 1:** the epoch-30 checkpoint of every arm. Pre-quantization latents are extracted from those
  checkpoints in Tier 1, not here.

### B.11 Optional Stage 3: subject-disjoint check (P6, secondary)

S0 and M-pub on HuGaDB and LARa with the four subject folds from `script/e2_prepare_splits.py` (verify the fold
definitions before use): train on three folds' subjects, score the held-out subjects with the global evaluator restricted
to them. One seed per fold. About 2 × 2 × 4 trainings ≈ 2 h. Run only if Stage 1 is at least partly replicated, since
it asks whether the gain transfers to unseen people.

## What Tier 0 cannot show

- That MASQ's mechanism (joint inference) is why it works. The ablations show which component matters, not why.
- That any configuration is untouched-test performance (P7). Everything here is development.
- Anything about our OT decoder on MASQ latents. That combination is Tier 1.

## Deliverables

1. This protocol (Part A), adopted for all later plans.
2. `results/week5/masq_experiments/<run-id>/REPORT.md`: gates, the five-metric tables for every arm, R1-R5 with readings, the
   verdict, and the identity diagnostics.
3. A spreadsheet sheet in the standard 40-column layout, one row per (arm, dataset).
