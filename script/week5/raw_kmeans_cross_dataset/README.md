# Raw K-means tokens across datasets (HuGaDB anchor, LARa, BABEL 1/2/3)

Implements [the plan](../../../docs/plans/week5/RAW_KMEANS_CROSS_DATASET/RAW_KMEANS_CROSS_DATASET.md): the frozen HuGaDB
configuration (raw patches -> global standardizer -> PCA-64 -> K-means-512 IDs -> masked contextualizer at guard 0, 5,000
updates -> contextual OT, outer cap 50, rungs 500-4000) on every dataset, with only the window (HuGaDB 15, LARa 12, BABEL 8
frames), frame rate and class count (10 / 8 / 5) changing. Nothing is tuned on LARa or BABEL.

Matrix: 18 cells = HuGaDB x 3 seeds, BABEL-1/2/3 x 3 seeds, LARa x {global, per-recording normalization} x 3 seeds; each cell
trains one contextualizer and decodes the validation-selected and the final checkpoint (36 OT fits). The seed drives both the
K-means vocabulary and the contextualizer; the train/validation split is fixed (seed 111). Order: HuGaDB, BABEL, then LARa.

## Files

| File | Role |
|---|---|
| `datasets.py` | dataset specs, label loading from `groundTruth` + `mapping` (scoring only) |
| `prepare.py` | per-cell raw K-means artifacts in the qic layout (read through the unchanged `artifacts.Loader`), code diagnostics |
| `decode.py` | dataset-aware embeddings -> OT -> ladder (fps / window / classes / labels come from the dataset) |
| `run_cross_dataset.py` / `.ps1` | gates, schedule, resume, deadline, aggregates, CLI |
| `report.py`, `tests.py` | report from saved artifacts (declared reading rule), 12 focused tests + smoke verification |

Reused unchanged: the calibration package's training (`train.py`), embeddings, checkpoint ledger and replay; `fit.py` /
`families.py` / `artifacts.py` from the quantized-input comparison; `action_transport` OT. One backward-compatible extension:
`contextualizer_calibration.evaluate.eval_complete` takes an optional `num_states` (default 10).

## Gates (run order)

1. **Label loading and scoring (every dataset, at start):** the loader's ground truth equals the saved SMQ ground truth and
   re-scoring `results/preds/*_pretrained.npz` reproduces `results/e0/*` "global" within the scorer's two-decimal rounding
   (0.005). A failing dataset is blocked; others continue. The local SMQ F1@50 used in the report comes from here.
2. **Tokenizer port (HuGaDB / global / seed 111, full runs):** the rebuilt raw family equals `qic_v3`: codes, features,
   starts/lengths, mean, RMS, grouping, occupancy, split, scaler and PCA identical. The K-means centers (`lookup`, not an input
   of the raw ID arm) may differ by float32 accumulation noise (<= 1e-5; measured 3.8e-6). Randomized PCA and K-means reduce in a
   thread-count-dependent order and `qic_v3` was built with 32 threads, so **token preparation is pinned to 32 threads**
   (`PREP_THREADS`, part of the preparation identity) while training and OT keep the launcher's 8-thread cap. Without the pin the
   launcher's cap changed the PCA components by up to 2.5e-5 (codes unchanged), which failed this gate in `rkx_v1`.
3. **Pipeline port (same cell):** the validation-selected decode must equal `ctx_cal_v1` job
   `raw__ctx128_span8_g0_h5000_seed111_tcnone` (step 4100, all five metrics at every rung, F1@50 50.00): exact metrics and step;
   trajectory, tensors and embeddings recorded.

A port gate failure stops the run before dependent cells (`-ContinueOnGateFailure` overrides) and writes
`gate_mismatch_environment.json`. Nothing is attributed to GPU nondeterminism without evidence, and no setting is changed to
recover a score. Resume refuses any change of sources, configuration, cell set or environment; an interrupted fit restarts from
initialization (logged in `restarts.jsonl`). Aggregates include only verified records (job, checkpoint tensors, identity and
array contents); stale records are removed. A failed training is not retried within an invocation, and a deterministic invalid solve is final. Gate outcomes carry across
resumes; reloading verified preparation artifacts ignores the deadline, so a resume that is short on time never drops
completed work.

## Commands (repository root)

```powershell
.\script\week5\raw_kmeans_cross_dataset\run_cross_dataset.ps1 -Preflight                       # 12 focused tests
.\script\week5\raw_kmeans_cross_dataset\run_cross_dataset.ps1 -DryRun -RunId rkx_dry          # plan + gate 3 only
.\script\week5\raw_kmeans_cross_dataset\run_cross_dataset.ps1 -Smoke  -RunId rkx_smoke        # bounded, not evidence
.\script\week5\raw_kmeans_cross_dataset\run_cross_dataset.ps1 -RunId rkx_v1 -Hours 8          # FULL RUN
.\script\week5\raw_kmeans_cross_dataset\run_cross_dataset.ps1 -RunId rkx_v1 -Hours 8 -Resume  # continue
```

Optional: `-Datasets hugadb babel1` (subset; the resume identity covers the subset), `-Cpu`, `-ReferenceRun`, `-QicRunId`.
`SMQ_SLOW_TESTS=1` additionally rebuilds the HuGaDB raw family and compares it with `qic_v3` (about 1 minute).

## Verification performed

* Preflight: 12/12 tests (plan and frozen configuration; real label files and saved SMQ scores for all five datasets; extraction
  equal to the existing raw extractor and per-recording normalization incl. the std floor; preparation reuse / rebuild / seeds /
  fixed split; training and decode on a synthetic dataset with 3 classes and a different window; failed training, stale records
  and gate stop; resume identity; reading-guide verdicts; a bounded-budget resume keeping verified results; gates persisting;
  the pipeline comparator against the real `ctx_cal_v1` artifacts) plus the slow tokenizer-port test, which rebuilds HuGaDB in a fresh process under the launcher's 8-thread settings.
* Smoke `rkx_smoke` (3 min 21 s; 12 / 6 / 60 recordings per HuGaDB / LARa / BABEL; 40 updates; 2 rungs): 18/18 cells, 36/36 fits
  (29 distinct decodes), 29/29 saved predictions rescored to their saved metrics; its resume under an expired budget kept all 36 records and all gate outcomes.
* Not verified on full data: gates 2 and 3 (they run first in your launch), and the LARa and BABEL runtimes.

## Limits

Three paired seeds; the outer OT fit is capped in almost every fit; BABEL clips are short (mostly padding in a 128-token context);
published numbers use different protocols; tokenizer and OT fitting are transductive; LARa is not untouched.
