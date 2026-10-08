# Phase 5: the tokenizer, then span

Revision 2 — 3 October 2026. Implemented (`script/week5/timescale_inspection/`, see its README); not launched. It is sized for a single
**10-hour** budgeted run: points 1–4 complete, span on the development set, and the rest on resume.
Changes from Revision 1: T4 `context` now uses guard 14 / span 16 (the SMQ encoder's receptive field would
otherwise leak); H6 is decided on F1@50 alone, with probes as a diagnostic; T3 is split into T3/T3b and the
LARa whitening slot became T3b; added the T4 confound and the Phase 4 velocity caveat. Parent plan:
`TIMESCALE_INSPECTION.md`.

## Why this phase exists

Phases 0–4 compared the contextualizer against no-contextualizer decoding (`pca`/`centre`), always on **one**
tokenizer: a flattened raw patch, standardized, PCA-64, then Euclidean K-means-512. On BABEL and LARa the
contextualizer lost to the direct arms, and I read that as "the contextualizer fails on skeleton data". That reading
was wrong. The contextualizer was never tested on any other tokenizer, and the evidence points upstream, at the
tokenizer:

| Evidence | BABEL-2 | LARa | HuGaDB (works) |
|---|---|---|---|
| PCA-64 retained variance | 0.998 | 0.978 | 0.712 |
| Variance in PCs 1–5 / between-recording share of it | 0.77 / **0.68** | 0.53 / 0.32 | 0.39 / 0.07 |
| Action η² of PCs 1–5 | 0.031 | 0.080 | 0.096 |
| 1-NN probe, PCA-64 unscaled (the metric K-means uses) → standardized (whitened) | 0.508 → **0.625** | 0.398 → 0.439 | 0.686 → 0.684 |
| Linear probe: raw patch → PCA-64 | 0.450 → 0.421 | **0.529 → 0.466** | 0.656 → 0.610 |
| Linear probe: raw positions → inter-joint distances (+ change) | 0.450 → **0.581** | 0.521 → **0.594** | n/a (IMU) |
| Code–action MI / H(action) | 0.66 | 0.30 | 0.69 |
| Code–subject vs code–action MI (nats) | n/a | **1.34 vs 0.52** | 1.63 vs 1.40 |

Source: the diagnostic `relational_probe.py` (scratchpad, 3 October), with Phase 0.4's split, caps and
classifiers, and the `rkx_v2` code diagnostics.

The mechanism differs by dataset, but it is the tokenizer in both cases:
- **BABEL:** PCA keeps the information, but the highest-variance directions are per-clip constants (orientation, body
  shape). Euclidean K-means organizes the codes along those directions.
- **LARa:** compressing a flat 1,584-dim patch to PCA-64 drops low-variance, action-relevant directions. The
  orientation channels add noise as well: positions alone give 0.551 on 1-NN against 0.468 for all six channels.

Two interventions that cleaned up the tokens helped the **contextualizer specifically** and left the direct arms flat:
- LARa per-recording normalization: context 5.1 → 7.1, direct arms 9.9/10.2 → 9.7/9.1.
- Velocity: context +3.1 to +3.2 on LARa and BABEL-2, direct arms ±1.

SMQ's own ablation (its Table 4) shows a disentangled per-joint encoder beating an entangled one by +4 to +6 F1@50.
Its entangled numbers (LARa 12.3, BABEL average 16.9) sit next to our best arms (about 10 and 17–21).

**Questions, in priority order:**
1. **Does a better tokenizer raise F1@50 on BABEL and LARa?**
2. **On a better tokenizer, does the contextualizer stop losing to the direct arms?** That is: was its failure
   conditional on the tokens?
3. **On the best tokenizer, does a shorter mask span help?** This is the span question that Phase 0's indirect
   evidence could not settle.

## Caveat on Phase 4, carried into this phase

Phase 4's velocity features compute each window's first-frame velocity from the previous window's last frame
(`vel[t] = frame[t] − frame[t−1]` before windowing). So a masked token's code depends slightly on the visible token
before it. This is far weaker than the encoder leak above: it is one frame's difference, and adjacent windows are
already strongly dependent through smooth motion. But it makes the contextualizer's task a little easier than the
position-only family's, so Phase 4's contextualizer gain (+3.1 to +3.2) is an upper-end estimate. If velocity is
reused, the first frame of each window should be computed within the window (zero), as for the first frame of a
recording. T2 is unaffected: its distances and their changes are computed within each window.

## Tokenizers (all label-free)

Each tokenizer is a new family, built per (dataset, seed) cell in the existing qic layout. The downstream pipeline is
unchanged: K-means-512, the frozen reference contextualizer (guard 0, span 8, 5,000 updates) for T1–T3, OT at the
current radius with outer cap 50, rungs 500/1,000/2,000/4,000, and the repository scorer.

**T4 is the one exception, and the reason matters.** SMQ's encoder is two TCN stages of three dilated layers
(dilations 1, 2, 4, kernel 3), so each frame's latent sees ±14 frames, the same 29-frame receptive field that
motivated the guard rule earlier. With guard 0, a visible neighbouring token's code already contains masked frames,
so the contextualizer can copy instead of predict, and its task is not comparable to T0–T3. So T4's `context` arm uses
**guard 14 with span 16** (the repository's `long_mask` configuration, `contextualizer_calibration.plan.CONFIGS`),
on every dataset. The reasons for span 16:
- A masked token only scores if its frame support is disjoint from every visible token's. That needs the token to be
  at least 5 tokens from any visible token on BABEL's 8-frame grid, at least 4 on LARa's 12-frame grid and at least 3 on
  HuGaDB's 15-frame grid. An isolated span therefore needs ≥ 9 tokens on BABEL (≥ 7 on LARa, ≥ 5 on HuGaDB) to score
  anything; adjacent random spans can merge into longer runs and score their interior.
- Measured with the repository's `masking.build_eligible_mask` (guard 14, fraction 0.30, 128-token chunks, 200
  random draws): the share of masked targets that score is

  | span | BABEL (8 frames) | LARa (12) | HuGaDB (15) |
  |---|---|---|---|
  | 2 | 0% | 1% | 5% |
  | 4 | 4% | 9% | 20% |
  | 8 (the reference) | **16%** | 35% | 56% |
  | 16 | **52%** | 63% | 75% |

- So span 8 under guard 14 is not impossible but is about 3× sparser on BABEL (about 6 scored targets per 128-token
  chunk, against about 20 for span 16), and short BABEL clips score somewhat fewer. Spans 2 and 4 score almost
  nothing on BABEL, so they cannot be used with guard 14. That is why stage D never selects T4.
- An earlier draft said span 8 would score no targets on BABEL; the measurement above corrects that.

Consequences, stated up front:
- T4 `context` is compared **within T4** (`context − centre`), never against T0's `context`. Different masking means a
  different training task.
- The T4 `centre` and `pca` arms are leak-free and are T4's primary result. They answer "does a joint-aware encoder
  make better tokens?" directly.
- T4 `context` does not count towards H6 or H7, and stage D (span) never selects T4.

| ID | Name | Input per window | Reduction before K-means | Datasets | Targets |
|---|---|---|---|---|---|
| T0 | raw (baseline, **not rebuilt**) | flattened raw patch | standardize → PCA-64 | all | — |
| T1 | `whiten` | flattened raw patch | standardize → **PCA-64, whitened** | BABEL-2/3, HuGaDB (not LARa) | the BABEL metric problem |
| T2 | `relational` | per-window **inter-joint distances**: mean, sd and mean \|Δ\| over the window's frames (position channels only) | standardize → PCA-64 | BABEL-2/3, LARa | rotation/translation nuisance; joint structure |
| T3 | `lara_pos256` | flattened **position channels only** (orientations dropped) | standardize → **PCA-256** | LARa | PCA-64 truncation (read against T3b) |
| T3b | `lara_pos64` | flattened **position channels only** (orientations dropped) | standardize → PCA-64 | LARa | orientation-channel noise (read against T0) |
| T4 | `smq` | **SMQ's released encoder latents** (pre-quantization, 16 dims per joint, joint-disentangled TCN), patched at our window | standardize → PCA-64 | BABEL-2/3, HuGaDB, LARa | whether SMQ's learned, joint-aware encoder makes better tokens (see the confound note) |

Notes:
- **T2 dimensions:** BABEL has 25 joints, so 300 distances × 3 = 900 dims. LARa has 22 joints, so 231 × 3 = 693.
  HuGaDB is excluded: its "joints" are IMU sensors, and distances between them are meaningless. The probe agrees,
  scoring distances 0.614 against 0.648 for the raw channels.
- **LARa position channels are 3–5.** The extent check found body-sized ranges there, and 0–2 are orientations. The
  preparation step asserts this per run (gate G3) rather than assuming it.
- **T4 source:** the checkpoints are `models/pretrained/{dataset}.model`, built with
  `script/repro/dump_predictions.build_model`. Latents come from `fsq_reconstruction.experiment.extract_recording`.
  Feeding raw `.npy` features unnormalized matches SMQ's own `batch_gen`. There is one SMQ checkpoint per dataset, so
  the three seeds vary only the K-means vocabulary and the contextualizer, never the encoder. The SMQ encoder was
  trained unsupervised on the full dataset; that is the same transductive setting as everything else here.
- **T3 and T3b together isolate two effects.** T3 differs from T3b only in PCA-256 vs PCA-64, so T3 − T3b is the
  truncation effect. T3b differs from T0 only in dropping the orientation channels, so T3b − T0 is the orientation
  effect. The original single T3 changed both at once and could not separate them.
- **T4 confounds three properties, not one:** per-joint (disentangled) encoding, about 1 s of temporal context from
  the receptive field, and a learned objective (code quantization plus inter-joint-distance reconstruction). A T4 win
  therefore means "SMQ's encoder helps", not specifically "joint structure helps". T2 is the cleaner test of joint
  relations. T4 also uses PCA-64; if T3 − T3b shows truncation hurts LARa, T4's LARa number is a lower bound for the
  encoder, and the report says so.
- **T2 drops orientation entirely,** including turning motion. That is fine for BABEL-2/3's classes (BABEL-2: sit,
  run, stand_up, kick; BABEL-3: jog, wave, dance, gesture). It would hurt BABEL-1's `turn`, which is out of scope.
- **T1 detail:** whitening requires K-means to be fit in the whitened space. So T1 gets its own population fit. It
  uses the identical `KMeans(k-means++, n_init=5, max_iter=300, tol=1e-4, random_state=seed)` and the same sample
  windows, with `PCA(whiten=True)`. `project_pca` then whitens automatically via `pca.transform`.

## Arms per cell

- **`context`:** train the contextualizer on the tokenizer's codes, then decode its embedding.
- **`centre`:** the K-means centre of each window's code, with no contextualizer. This is the fair counterpart of
  `context`: the same information, without context.
- **`pca`:** the tokenizer's continuous feature (PCA space), with no contextualizer. It runs on BABEL, and on T3 on
  LARa. On LARa it is otherwise dropped to save time, since `centre` and `pca` agreed within 0.3 F1@50 there in
  Phases 1 and 4. T3 needs it because `centre` goes through K-means, and `pca` shows what PCA-256 itself carries.
- **T4 `context`** runs with guard 14 and span 16 (see above); every other `context` arm uses guard 0, span 8.

Baselines are read, not rebuilt:
- T0 `context` from `rkx_v2`;
- T0 `pca`/`centre` from `ts1_v1` Phase 1 `inputs`;
- velocity results from `ts4_v1`.

All at the current radius, with the same 3 seeds (111, 222, 1538574472).

## Matrix, order and runtime (10-hour budget)

Unit costs were measured in `ts4_v1`, `ts1_v1` and `rkx_v2`. Decoding runs on CPU and training on GPU.

| Unit | BABEL-2 | BABEL-3 | LARa | HuGaDB |
|---|---|---|---|---|
| prep (raw-like / relational / SMQ extraction) | ≤ 0.5 min | ≤ 0.5 min | ≤ 2 min | ≤ 0.5 min |
| train (5,000 updates) | 1.5 min | 1.6 min | 2.2 min | 2.7 min |
| decode, `context` | 2.5 min | 2.1 min | 14.4 min | ~8 min |
| decode, `centre` / `pca` | 2.1 min | 1.5 min | 10.6 min | 4.6 min |

Stages run in this order. A budget stop always leaves every earlier stage complete. The order follows the stated
priority: points 1–4 (the tokenizers) first, point 5 (span, then window) after.

| Stage | Content | Units | Estimate | Cumulative |
|---|---|---|---|---|
| G | gates G1–G3 (below) | — | ~6 min | 0.1 h |
| A | **BABEL-2 and BABEL-3**: T1, T2, T4 × 3 seeds × {context, centre, pca} | 18 cells | ~143 min | 2.5 h |
| B | **HuGaDB control**, seed 111: T1, T4 × {context, centre} (T4 `context`: guard 14, span 16) | 2 cells | ~32 min | 3.0 h |
| C1 | **LARa**: T3 (with `pca`), T2, T4 × 3 seeds × {context, centre} | 9 cells | ~295 min | 7.9 h |
| P1 | per-token probes, seed 111, every new family | 10 cells | ~10 min | 8.1 h |
| C2 | **LARa**: T3b × 3 seeds × {context, centre} | 3 cells | ~88 min | 9.6 h |
| D | **span** on the selected tokenizer (below), spans {2, 4} × 3 seeds, `context` only: **BABEL-2 first**, then BABEL-3 | 12 trainings | ~46 min | 10.4 h |
| E | *(only if time remains)* window {4, 15 frames} on the selected tokenizer, BABEL-2/3 × 3 seeds × {context, centre} | 12 cells | ~74 min | 11.6 h |
| P2 | probes for the remaining seeds | 20 cells | ~20 min | 11.9 h |

How the estimates were built:
- **A:** (8.7 + 7.2 min per seed) × 3 seeds × 3 tokenizers.
- **C1:** 29.2 min per LARa cell × 9, plus T3's `pca` arm (3 × 10.6).
- **D:** 12 × 1.55 min training plus 12 × 2.3 min decoding.

**Expected outcome at 10 hours:**
- Points 1–4 complete: stages A, B, C1, P1 and C2 (T1 on BABEL and HuGaDB; T2, T3, T3b and T4 on their datasets).
- Span (D) on **BABEL-2, the development set, complete**: its six trainings and decodes take about 24 min. That is
  enough for the H2′ development half.
- BABEL-3's span confirmation, window (E) and the remaining probe seeds (P2) go to a `--resume`, about 1.5–2 h more.
- If you can give it **12 hours**, everything finishes.

Two ordering choices:
- **T1 (whitening) is not run on LARa.** Whitening is mainly a BABEL fix: LARa's top components are less
  nuisance-dominated, and the probe gain there is small (0.398 → 0.439). The slot goes to T3b, which isolates the
  orientation-channel effect that T3 would otherwise confound. LARa whitening can be added later if T1 helps on BABEL.
- **Span runs BABEL-2 before BABEL-3,** so a budget stop still leaves a complete development result.

Risk: the estimates assume the machine is otherwise idle. Phase 4 ran about 15% slower while another decode competed
for CPU. Launch with `-Hours 10` and nothing else running.

## Gates (run first; a failing gate blocks only what depends on it)

- **G1 — the generalized preparation reproduces the baseline.** The new tokenizer module, with spec "raw, PCA-64, not
  whitened", must reproduce `rkx_v2`'s `raw_kmeans` family for BABEL-2 seed 111: codes, continuous features, per-window
  starts and lengths, mean, RMS, split and grouping **exactly**, and the K-means centres (`lookup`) to 1e-4. The centres
  cannot be compared exactly: two runs of the original preparation on the same data differ by up to 7.6e-6 in the
  centres (measured, with identical codes and features), a thread-order effect in K-means. Preparation is pinned to 32
  threads, as in `rkx_v2`. If this fails, nothing runs.
- **G2 — the SMQ checkpoints are loaded correctly.** For each dataset used, the pretrained checkpoint must load
  verbatim, with tensors compared after `load_state_dict`. Its own per-frame codes (`model.indices`), scored with
  dataset-level Hungarian matching, must reproduce the saved re-score `results/e0/{dataset}_pretrained.json` →
  `global` to within 0.005 on every metric. The gate calls the exact producer of those files,
  `script/repro/e0_metrics.py`, rather than a reimplementation, so a small difference in patch alignment cannot fail
  the gate while the model is in fact loaded correctly. That is the same tolerance as `rkx_v2`'s label gate, since the scorer
  rounds to two decimals. If this fails, T4 is blocked for that dataset, and a wrong constructor can never silently
  produce junk latents.
- **G3 — LARa channel semantics.** The chosen position channels (3–5) must have a larger median across-joint extent
  than channels 0–2. The extents are recorded. If this fails, T2, T3 and T3b on LARa are blocked.

## Selection for stage D (declared now)

The span stage runs on the tokenizer with the **highest mean `context` F1@50 on BABEL-2 over 3 seeds**, among the
guard-0 tokenizers T0, T1 and T2 (T4's `context` uses a different masking and cannot be compared; and under guard 14,
spans 2 and 4 score under 5% of masked targets on BABEL, so they are unusable for it), read when stage A finishes. BABEL-2 is the development set. BABEL-3 is reported with the choice frozen.
If T0 (raw) wins, the span stage runs on raw tokens. The choice is written to `selection.json` before stage D starts,
so it cannot be influenced by D's own results. Span 8 comes from stage A; it is not retrained.

## Reading rules (declared now)

The primary number throughout is F1@50, selected checkpoint, last rung, mean of 3 seeds, paired by seed. The bar is
≥ 3 points on the mean, with all 3 seeds in the same direction, as in Phases 1–4.

- **H6 — the tokenizer limits skeleton datasets.** Supported if some new tokenizer beats T0 by the bar on BABEL-2 or
  LARa in the **`centre` arm** (the leak-free, same-information counterpart that every tokenizer has), or in `context`
  for T1–T3b (not T4). **F1@50 alone decides H6.** The per-token probes (Phase 0.4's split and caps) are run on every
  new family and reported as a mechanism check in three variants: linear, 1-NN, and 1-NN on whitened features. They
  are not part of the verdict. An earlier draft declared the whitened 1-NN probe as the deciding probe on the grounds
  that it "tracked F1@50 best", but that was never measured, and about ten data points could not establish it.
  Instead the report labels each tokenizer by how many of the three probes rise over T0 (all, some, none), so a
  disagreement among probes stays visible and is not hidden behind one chosen probe.
- **H7 — the contextualizer's failure was conditional on the tokens.** For each tokenizer in T1–T3b and dataset:
  - *helps:* `context − centre` ≥ +3, all seeds;
  - *neutral:* within ±3;
  - *hurts:* ≤ −3, all seeds.

  H7 is supported if a tokenizer that satisfies H6 moves `context − centre` up by ≥ 3 against T0 on that dataset.
  On T0 the gap is −4.6 on BABEL-2 and −5.1 on LARa. T4 is excluded from H7 because its `context` is trained with a
  different masking; its own `context − centre` is reported descriptively.
- **A caution on `context` verdicts.** The `context` arm is noisy: its seed spread on BABEL-2 is about 3.5–3.9
  points, against about 1 for `centre`. A 3-point bar with three seeds is therefore close to the noise floor for
  `context`, so its verdicts (H7, H2′) are weaker than the `centre`-based ones. Two more BABEL seeds would cost about
  95 minutes, which does not fit in 10 hours, but they are the first thing to add if the run is extended.
- **H2′ — span, on the selected tokenizer.** Supported if span 2 or 4 beats span 8 by the bar on BABEL-2 `context`,
  and BABEL-3 moves the same way. The masked-loss boundary split from Phase 0.4b is recomputed for each span as a
  diagnostic.
- **HuGaDB transfer check (descriptive only; one seed):** a tokenizer whose HuGaDB seed-111 `context` falls more than
  6 points below T0's 50.00 is flagged. Six points is the observed seed spread.
- **SMQ reference:** every table shows SMQ's local re-score next to our best arm: HuGaDB 24.3, LARa 16.4, BABEL-2
  27.5, BABEL-3 19.4. Beating SMQ is not a declared criterion. This phase asks what limits the pipeline, not whether
  it wins.

## Implementation notes

- **Package:** `script/week5/timescale_inspection/`, reusing Phase 4's runner pattern.
  - New `tokenizers.py`: a `TokenizerSpec(name, source ∈ {raw, raw_positions, relational, smq_latent}, pca_dim,
    whiten)` and one preparation path. With `include_velocity=False` and the T0 spec it reduces to Phase 4's
    `velocity_prepare`, and through it to `raw_kmeans_cross_dataset.prepare`; gate G1 checks this.
  - New `run_phase5.py`: stages, budget, resume and `selection.json`. New `report5.py`.
  - Shared modules stay untouched, so earlier run identities remain valid. That includes `quantized_input_comparison`
    config and fit, the solver and the contextualizer. T1's population fit is a local copy of `fit_population` with
    `whiten=True`; a test checks it equals `fit_population` when `whiten=False`.
- **Output:** `results/week5/timescale_inspection/<run_id>/phase5/`, with:
  - `prep/<tokenizer>/<cell>`, `jobs/<tokenizer>/<cell>`, `units/...`;
  - `probes.csv`, `gates.json`, `selection.json`, `REPORT.md`.
- **Spans (stage D):** a `JobSpec` with `mask_span` ∈ {2, 4}. The job identity already hashes the span through
  `effective_settings`.
- **Window (stage E):** a new preparation with `window` ∈ {4, 15}. The radius stays `current` (fps // window), so it
  changes with the grid. That is noted in the report as a confound.
- **Tests:**
  - G1-style equality on a small cap;
  - whitening off equals `fit_population`;
  - relational features: shape and translation/rotation invariance on a synthetic skeleton;
  - SMQ latent extraction shape;
  - the selection reads BABEL-2 only and never returns T4;
  - T4's `context` job is built with guard 14 and span 16, and the BABEL masks then score at least ~40% of masked
    targets (the test asserts the measured table's ordering: span 16 scores more than span 8, and spans 2 and 4
    score under 5%);
  - T3 and T3b differ only in PCA dimension, and T3b differs from T0 only in dropped channels;
  - the probe labelling (all / some / none of three probes rise) on synthetic numbers;
  - the stage order is respected under an expired budget;
  - resume reuses verified units.
- **Launcher:** `run_phase5.ps1 -RunId ts5_v1 -Hours 10`, with `-DryRun` printing the unit list, the stage order and
  the time projection.

## Out of scope

- Re-tuning the radius. Phase 1's screen is still pending; the radius stays `current` here, so Phase 5 is comparable
  with Phases 1–4.
- Training our own joint-disentangled encoder. T4 uses SMQ's released one.
- BABEL-1. Each of its decodes takes 15 minutes, and it is not needed for the declared rules.
- Combining tokenizers, for example relational plus velocity. That comes only if a single tokenizer passes H6.
