# Contextualizer search calibration

Revision 2 — 1 October 2026. Design only; no experiments launched.

## Objective

Calibrate the contextualizer-search procedure before comparing native VQ training at different patch durations and vocabulary sizes. A fixed decoder measures one operating point, not each tokenizer's optimized performance. [Task 2](../../results/quantized_input_comparison/task2/task2_v1/REPORT.md) showed inconsistent agreement between pretext loss and segmentation, a raw-token protocol gap, and 38/40 capped outer OT fits. Check training duration, checkpoint selection, masking, solver sensitivity and cost before the larger search.

## Data and pipelines

Reuse frozen [qic_v3 artifacts](../../results/quantized_input_comparison/qic_v3/manifest.json): HuGaDB, 15-frame / 0.25-second tokens, nominal K=512, existing recording train/validation split.

- **Native VQ quantized vectors:** `vq_native/context_quantized`, pilot002 epoch-30 tokenizer, guard radius 14 raw frames.
- **Raw K-means IDs:** `raw_kmeans/context_id`, primary guard radius 0. Non-overlapping raw patches have no encoder-induced overlap.
- An extra radius-14 raw reference reproduces Task 2 and isolates the guard change under that protocol. It cannot alone explain all historical differences.

HuGaDB is development data, including segmentation-based procedure selection. Preserve the masked-validation split; OT development evaluation is pooled. Reserve LARa for the later prospectively frozen cross-dataset transfer comparison, without contextualizer search on its new scores. LARa has already informed this project and is **not an untouched test set**. Pooled tokenizer/OT fitting remains transductive.

## Training matrix

Run each main configuration on both pipelines at seed 111:

| Configuration | Context | Nominal mask span | Training/inference stride | Effective batch |
|---|---:|---:|---:|---:|
| Reference | 128 tokens / 32 s | 8 tokens / 2 s | 64 | 64 |
| Short context | 64 tokens / 16 s | 8 tokens / 2 s | 32 | 128 |
| Long mask | 128 tokens / 32 s | 16 tokens / 4 s | 64 | 64 |
| Combined | 64 tokens / 16 s | 16 tokens / 4 s | 32 | 128 |

Keep mask fraction 0.30 and Task 2 architecture/optimization fixed. Main fits run 5,000 updates without actual early termination. Match 8,192 nominal token positions per full update; report actual valid tokens, frame weights, eligible targets, batch tails and skipped batches. Context changes both training overlap and inference support.

| Training item | Fits |
|---|---:|
| Main: two pipelines × four configurations | 8 |
| Guarded raw reproduction reference, seed 111 | 1 |
| Genuine 3,000-update-schedule reference, each primary pipeline | 2 |
| Second reference seed, 222, each primary pipeline | 2 |
| Matched-target-count native-VQ long-mask sensitivity | 1 |
| Optional matched-target-count native-VQ combined sensitivity | 1 optional |
| **Total** | **14 default; 15 with optional sensitivity** |

The target-count sensitivity keeps long-mask inputs masked but subsamples eligible training targets to the corresponding span-8 reference count on each effective batch. Isolate its RNG, explicitly handle insufficient candidates, and report residual differences in frame weights. This is a supervision-count sensitivity, not a change to the tokenizer or guard. Audit realized spans because the existing sampler trims masks to its target count.

## Reproduction and checkpoint policies

First reproduce Task 2's validation-selected checkpoint, seed 111, final rung: guarded raw IDs **36.08 F1@50** and native-VQ quantized vectors **38.81**. Compare all five metrics, selected steps and available numerical artifacts. Seek exact agreement at stored precision; investigate deviations rather than automatically blaming GPU nondeterminism.

Save fixed checkpoints at 1k/3k/5k where applicable and every strictly improving validation checkpoint. Validation runs every 100 updates. Replay patience 1,000 offline with strict improvement and earliest ties: select the best checkpoint available at the first simulated stop, or the best through the endpoint if no stop occurs. Replay must not change actual training.

Intermediate checkpoints measure progress under a 5k schedule. The genuine 3k references have cosine horizon 3,000 and warmup 500; they measure a shorter training budget. Compare fixed-final, full-history validation-selected and replay-selected policies on development segmentation. Do not select an arbitrary checkpoint by its highest evaluation score.

## OT jobs and count

Use Task 2's grouping, initialization, 10 states, coefficients and 50-outer/25-inner fitting budget. Final rungs continue transport with fixed fitted prototypes; they are not separate outer fits. Retain historical rung boundaries 500/1,000/2,000/4,000 for reproduction, reporting 500 and 4,000 as the primary endpoints.

| Item | Checkpoint policies | Maximum OT fits |
|---|---|---:|
| Main eight | 1k, 3k, 5k, validation-selected, early-stopping replay | 40 |
| Guarded raw reference | selected, replay | 2 |
| Two 3k references | selected, final | 4 |
| Two seed-222 references | selected, final, replay | 6 |
| Matched-target run(s) | selected, final | 2–4 |
| Outer sensitivity | Reference and combined, final 5k checkpoint, both primary pipelines; cap 150 | 4 |
| **Total** | Deduplicate identical checkpoint/evaluation identities | **≤58 default; ≤60 optional** |

The four extended fits use the same objective and initialization, with outer cap 150. Keep this comparison separate from final-solver continuation. Invalid solves are invalid; capped results remain budgeted outcomes.

## Runtime and decisions

The estimate is **6.5–7 hours sequentially**, not a hard upper bound. The 58–60 total already includes four extended fits; add only their incremental cost above ordinary fits. Re-estimate training, extraction and solver time from the first references. Require an explicit elapsed-time budget, incremental saving and reasons for unfinished cells; no silent substitutions or grid shrinking.

Add one separately accounted short width-256/eight-layer/context-256 memory and throughput probe, including validation and embedding extraction, with no OT. It is not a full training fit or proof that 5,000 updates suffice.

Use development F1@50 for procedure selection, reporting MoF, Edit, F1@10/25 and run statistics alongside it. A **1-point F1@50** change is a practical solver-sensitivity flag, not a significance or convergence threshold. Also flag ordering reversals. Reference seed 222 checks checkpoint-policy stability. Small or inconsistent policy differences remain unresolved rather than establishing a winner.

## Intended later search: declaration only

| Group | Candidate values |
|---|---|
| Position | ALiBi, sinusoidal, none |
| Context | 8 / 16 / 32 / 64 seconds; 32 / 64 / 128 / 256 tokens at this resolution |
| Masking | Fraction 0.15 / 0.30 / 0.50; nominal span 0.25 / 1 / 2 / 4 seconds |
| Capacity | Width 64 / 128 / 256; four heads; FF=4×width; layers 2 / 4 / 8; dropout 0.1 / 0.3 |
| Optimization | LR 3e-4 / 1e-3; pre/post LayerNorm; horizon and checkpoint policy determined by the pilot |

There are 10,368 nominal combinations before rounding and feasibility filtering. The declared initial search budget is **12 unique candidates per tokenizer**, including the frozen reference and 11 feasible candidates from a seed-111 permutation of the remaining definitions. Save label-free rejections and actual compute. Leading-candidate replication belongs to the later search plan, not this pilot.

Convert seconds using actual token duration, round up, record realized durations, and deduplicate effective configurations. Guarded isolated spans 1 and 4 at W=15/radius=14 have no eligible interior targets; merged spans/boundaries do not justify an effectively empty objective. Feasibility and adequate training are distinct. Positional mode `none` requires explicit implementation and tests. A largest-model probe does not certify all search configurations.

## Outputs and scope

Save source/checkpoint hashes, splits, complete planned matrices, checkpoints, embeddings, OT fits/predictions, metrics, masks/exposure, timings and missing-cell reasons under a new run directory. Preserve previous runs. No tokenizer retraining, FSQ expansion, LARa execution or future-grid execution in this pilot. Keep native VQ in scope; replacing it with encoder K-means would change the scientific question.

Implementation and verification instructions: [CONTEXTUALIZER_SEARCH_CALIBRATION_AGENT_INSTRUCTIONS.md](CONTEXTUALIZER_SEARCH_CALIBRATION_AGENT_INSTRUCTIONS.md).
