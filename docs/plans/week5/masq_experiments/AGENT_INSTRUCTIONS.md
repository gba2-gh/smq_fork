# Agent instructions: Tier 0 MASQ replication

Version 1 · 5 October 2026. Implements Part B of `TIER0_PROTOCOL_MASQ_REPLICATION.md` (this folder) under the rules of
its Part A. Read that plan in full before starting. Where this document and the plan disagree, stop and ask.

**Your job:** implement, test, smoke-run and dry-run. **Not your job:** launching Stage 1, 2 or 3. The user launches every
run longer than 10 minutes.

---

## 1. Ground rules

### 1.1 Environment

- Python: `C:\Users\gzaz976\AppData\Local\anaconda3\envs\smq\python.exe` (`python` is not on PATH). It has torch and
  sklearn, but **not openpyxl**. The spreadsheet step uses the base interpreter
  `C:\Users\gzaz976\AppData\Local\anaconda3\python.exe`, which has openpyxl but must not import torch.
- Set `OPENBLAS_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OMP_NUM_THREADS=8`, `PYTHONIOENCODING=utf-8`, and `PYTHONPATH` = repo
  root, for every process.
- GPU: one RTX 4090 (24 GB). Use `cuda` when available.

### 1.2 Files you may and may not touch

- **Create only:** `script/week5/masq_experiments/` (code), `docs/plans/week5/masq_experiments/` (notes you add),
  `results/week5/masq_experiments/<run-id>/`, `models/week5/masq_experiments/<run-id>/`.
- **Do not modify:** `main.py`, `model.py`, `batch_gen.py`, anything in `src/`, `script/repro/`, `third_party/`, or any
  existing `results/` or `models/` content. Earlier experiments depend on them. Import from them; never copy-edit them in
  place.
- **Spreadsheet:** `outputs/all_runs/all_runs_results.xlsx` is changed only by `sheet.py`, and only by adding or replacing
  its own sheet. Back up the workbook first. If Excel holds a lock (`PermissionError`), write a staged copy and report it;
  never force.

### 1.3 Run discipline

- You may run unit tests, gates, a **smoke run of at most 10 minutes**, and dry runs. Smoke and dry runs use a
  **throwaway run ID** (for example `masq_smoke`), never the production ID.
- Before deleting anything under `results/` or `models/`, check that no python process is writing there
  (`tasklist | grep -i python`). Delete only throwaway run folders you created.
- Report outcomes faithfully. A failed gate is reported with its output, not worked around.

### 1.4 Part A rules, as they apply to this code (binding)

- **No label use in training or model code.** `model_masq.py`, `losses.py` and `train_masq.py` must not import or read
  `groundTruth/`, mapping files, or anything derived from them. Labels are read only in `score.py` (evaluation) and in the
  per-class analysis of `diagnostics.py`.
- **Number of codes K = C** is the benchmark convention. Compute it exactly as `main.py` does
  (`src.model.utils.get_num_actions(gt_path)`). This is the only permitted contact between training and the label files,
  and it is a count.
- **No tuning.** Every value in §3 is declared. Do not change any of them because of a score. If something fails
  numerically (NaN, OOM), stop and report; the allowed responses are listed in §8.
- **Checkpoint:** epoch 30, final. No other checkpoint is ever scored for a reported result.
- **Report everything that ran.** Incomplete units are listed as incomplete, never dropped silently.

---

## 2. What you are building

MASQ = SMQ + two training-time changes (plan §B.1). SMQ's inter-joint distance reconstruction is already on by default
in this repo.

1. **JLSD:** after the per-joint encoder, during training only, zero whole joint latent trajectories with probability p,
   and scale the rest by 1/(1−p).
2. **Mask-aware velocity loss:** match first-order differences of the reconstruction to those of the input, only on
   joints that are kept and frames that are unpadded.

Four arms (plan §B.3): S0 (SMQ), M-pub, M-JL, M-fix. **All four arms use the same new model and trainer code**; S0 is
simply p = 0, λ = 0. Gate G1 proves that this configuration equals SMQ.

---

## 3. Declared settings (copy these exactly into `config.py`)

### 3.1 Per dataset (from `main.py::get_dataset_defaults`)

| dataset | num_features (C) | num_joints (V) | num_person (M) | batch_size | patch_size |
|---|---|---|---|---|---|
| hugadb | 6 | 6 | 1 | 8 | 60 |
| lara | 6 | 22 | 1 | 8 | 50 |
| babel1 / babel2 / babel3 | 3 | 25 | 1 | 32 | 30 |

### 3.2 Shared (from `main.py` argument defaults)

epochs 30 · lr 5e-4 (Adam) · num_f_maps 128 · num_layers 3 · latent_dim 16 · sample_rate 1 · kmeans False ·
kmeans_metric "euclidean" · sampling_quantile 0.5 · replacement_strategy "representative" · decay 0.5 ·
dead_code_threshold 10 · tc_weight 0.0 · tc_beta 1.0 · mse_loss_weight 0.001 · commit_weight 1.0 ·
joint_distance_recons True · micro_batch_size None · grad_checkpoint False.

Data: `data/<dataset>/features/*.npy` (C, T, V, M), `data/<dataset>/groundTruth/*.txt`,
`data/<dataset>/mapping/mapping.txt`.

### 3.3 Arms

| arm | jlsd_p | vel_weight | datasets | stage |
|---|---|---|---|---|
| S0 | 0.0 | 0.0 | all five | 1 |
| M-pub | hugadb 0.25, lara 0.20, babel1/2/3 0.50 | 0.001 | all five | 1 |
| M-JL | as M-pub | 0.0 | all five | 2 |
| M-fix | 0.25 | 0.001 | lara, babel1, babel2, babel3 | 2 |

Seeds: 111, 222, 1538574472. Stage 3 (optional, plan §B.11) is specified in §6.8 and is **not implemented unless the user
asks**.

### 3.4 Published references (constants for the report)

- **MASQ published** (MoF / Edit / F1@10 / F1@25 / F1@50):

  | dataset | MoF | Edit | F1@10 | F1@25 | F1@50 |
  |---|---|---|---|---|---|
  | hugadb | 52.3 | 45.1 | 52.1 | 47.0 | 39.5 |
  | lara | 47.2 | 42.4 | 42.0 | 35.4 | 22.2 |
  | babel1 | 46.7 | 33.1 | 39.6 | 34.0 | 21.0 |
  | babel2 | 55.8 | 38.1 | 46.1 | 38.7 | 28.4 |
  | babel3 | 47.7 | 26.9 | 28.3 | 27.5 | 24.5 |

  Source: MASQ Tables 1-2. MASQ's Table 1 text extraction is shifted by one row; these values were read from the
  correct rows.
- **SMQ released checkpoints, re-scored:** `results/e0/<dataset>_pretrained.json` (global block).
- **SMQ retrained, 3 seeds, earlier spread:** `results/REPORT_reproduction_and_diagnosis.md` §1.1. HuGaDB MoF 41.60 ± 1.13,
  F1@50 21.50 ± 2.54; LARa MoF 36.26 ± 1.61, F1@50 15.75 ± 0.75. Per-seed files: `results/e0/{hugadb,lara}_seed*.json`.

---

## 4. Directory layout

```
script/week5/masq_experiments/
  __init__.py
  config.py        # §3 constants, unit list, identities
  model_masq.py    # MASQModel(SMQModel): JLSD
  losses.py        # velocity loss
  train_masq.py    # one training unit
  score.py         # one scoring unit (predictions + metrics)
  diagnostics.py   # identity diagnostics (one dataset × arm × seed)
  run.py           # driver: stages, process isolation, retries, caching, manifest, dry run
  report.py        # REPORT.md, scores.csv, verdict
  sheet.py         # spreadsheet sheet (base python, no torch)
  tests.py         # unit tests + G1-G3
  run.ps1          # PowerShell launcher
  README.md        # how to run, outputs, gates
```

Outputs for run ID `<rid>`:

```
models/week5/masq_experiments/<rid>/<arm>/<dataset>/seed<k>/
    epoch-30.model  config.json  losses.jsonl  training_info.json  heartbeat.txt
results/week5/masq_experiments/<rid>/
    manifest.json  gates.json
    units/<arm>__<dataset>__seed<k>/
        train.json  scores.json  predictions.npz  diagnostics.json
    REPORT.md  scores.csv
```

---

## 5. Module specifications

### 5.1 `config.py`

- `DATASETS`, `SHARED`, `ARMS`, `SEEDS`, `MASQ_PUBLISHED`, `SMQ_RETRAINED_SPREAD` from §3.
- `@dataclass(frozen=True) class Unit: arm: str; dataset: str; seed: int` with `unit_id = f"{arm}__{dataset}__seed{seed}"`.
- `stage_units(stage) -> list[Unit]`. Stage "1": S0 and M-pub × 5 datasets × 3 seeds. Stage "2": M-JL × 5 × 3, then M-fix
  × 4 × 3. **Order:** dataset-major, smallest dataset first (babel3, babel2, hugadb, babel1, lara), arms interleaved per
  seed. Then a partial stage still yields paired S0/M-pub cells.
- `settings(unit) -> dict`: every value that affects the result (§3.1, §3.2, the arm's p and λ, the seed).
- `code_hash()`: sha256 over this package's model/loss/train/score files plus the SMQ files they import:
  `src/model/smq.py`, `src/model/ms_tcn.py`, `src/model/motion_quantizer.py`, `src/model/utils.py`,
  `src/model/eval_utils.py`, `batch_gen.py`, `script/repro/e0_metrics.py`.
- `analysis_hash()`: sha256 of `report.py` and `diagnostics.py`. It is recorded at launch and compared when the report is
  written, as in Phase 9.
- `unit_identity(unit) = stable_hash(settings(unit), code_hash())`.

### 5.2 `model_masq.py`

```python
class MASQModel(SMQModel):
    def __init__(self, *args, jlsd_p: float = 0.0, **kwargs): ...
    def forward(self, x, mask): ...
```

- `forward` is a **verbatim copy** of `SMQModel.forward` (`src/model/smq.py`) with exactly one inserted block, delimited
  by `# --- MASQ: JLSD ---` comments, immediately after `self.latent = self.encoder(x, mask)`:
  - if `self.training and self.jlsd_p > 0`: draw `keep = (torch.rand(N*M*V, 1, 1, device=...) >= p).float()` from the
    global torch RNG, and set `latent_for_vq = self.latent * keep / (1 - p)`;
  - otherwise `keep = ones(N*M*V)` and `latent_for_vq = self.latent`.
  - Store `self.joint_keep = keep.view(N, M, V)` (detached) for the loss.
  - Continue the copied code using `latent_for_vq` where the original used `self.latent` for repacking. Keep
    `self.latent` as the undropped encoder output, so diagnostics read clean latents.
- **Row order:** the packed layout comes from `permute(0, 4, 3, 1, 2)`, so row r = n·M·V + m·V + v. `keep.view(N, M, V)`
  is therefore correct. Assert that shape.
- In `eval()` mode, JLSD is never applied (`self.training` is False).
- Record in a module docstring that JLSD acts on encoder latents per (sample, person, joint), broadcast over time and
  channels (MASQ eq. 1).

### 5.3 `losses.py`

```python
def velocity_loss(x, x_hat, mask, joint_keep) -> torch.Tensor:
    """x, x_hat, mask: (N, C, T, V, M); joint_keep: (N, M, V) in {0,1}."""
```

- `dx = x[:, :, 1:] - x[:, :, :-1]`; `dxh = x_hat[:, :, 1:] - x_hat[:, :, :-1]`.
- Frame-pair validity: `pv = mask[:, 0, 1:] * mask[:, 0, :-1]`, shape (N, T−1, V, M).
- Joint validity: `jk = joint_keep.permute(0, 2, 1)[:, None]`, shape (N, 1, V, M).
- `w = pv * jk`; `sq = ((dxh - dx) ** 2).sum(dim=1)`, which sums over channels to (N, T−1, V, M).
- `loss = (sq * w).sum() / w.sum().clamp_min(1.0)`.
- This is the declared normalization (plan §B.2): squared L2 over channels, averaged over valid (frame pair, joint) points.

Total loss per batch, exactly as `model.Trainer.train` plus one term:

```
rec    = mse_loss_weight * mean(MSE(distance_joints(x), distance_joints(x_hat)))   # unchanged SMQ
commit = commit_weight * model.commit_loss
tc     = model.vq.tc_loss                                                           # 0 here
vel    = vel_weight * velocity_loss(x, x_hat, mask, model.joint_keep)   # computed only if vel_weight > 0
loss   = rec + commit + tc + vel
```

The reconstruction loss covers all joints, including dropped ones (plan §B.2).

### 5.4 `train_masq.py`

`train_unit(unit, out_dir, device, epochs=30, max_batches_per_epoch=None) -> dict`

- **Seeding order must match `main.py`:**
  1. import `batch_gen` (which calls `random.seed(42)` at import);
  2. `random.seed(seed)`;
  3. `torch.manual_seed(seed)`;
  4. `torch.cuda.manual_seed_all(seed)`;
  5. `torch.backends.cudnn.deterministic = False` (as `main.py`; documented there).

  Then construct the model, then the `BatchGenerator`, then `read_data()`.
- **Model:** `MASQModel` with every SMQ constructor argument from §3 (`num_actions = get_num_actions(gt_path)`,
  `num_person = 1`, `commitment_cost` as SMQModel sets it), plus `jlsd_p` from the arm.
- **Loop:** a faithful copy of `model.Trainer.train` without micro-batching:
  - per batch: `zero_grad`, forward, losses as in §5.3, `backward`, `step`;
  - `batch_gen.reset()` per epoch;
  - log per-epoch means of rec, commit, tc and vel to `losses.jsonl`;
  - write `heartbeat.txt` each batch.
- **Dead-code count:** wrap `model.vq.replace` on the instance with a counter that calls the original unchanged. Record
  replacements per epoch.
- **Save:** only `epoch-30.model` (state_dict), plus `config.json` (settings + identity) and `training_info.json` (seconds,
  per-epoch losses, dead-code counts, peak GPU memory, final-epoch code occupancy).
- `max_batches_per_epoch` exists only for the smoke run. A smoke checkpoint is never scored as a result.

### 5.5 `score.py`

`score_unit(unit, ckpt_path, out_dir, device) -> dict`

- Build `MASQModel` with the unit's settings, load the state dict, call `eval()`.
- **Prediction loop:** identical to `script/week5/timescale_inspection/gates5.py::smq_global_scores`:
  - sequences sorted by `os.listdir`;
  - input `(1, C, T, V, M)` with an all-ones mask;
  - prediction = `model.indices.squeeze(0)` (int16);
  - GT via the mapping file.
- **Global:** `create_correspondences(concat(gt), concat(pred), mapping=True)`, map each sequence, then
  `script.repro.e0_metrics.score`.
- **Local:** per-sequence `create_correspondences`, then `e0_metrics.score`. Same as `e0_metrics.main`.
- Also record:
  - mean predicted and ground-truth segments per recording;
  - occupied codes and code perplexity over all frames.
- **Write:**
  - `scores.json` with `{global: {...}, local: {...}, segments: {...}, codes: {...}, n_sequences}`, using the metric keys
    of `results/e0/*.json` (`MoF`, `Edit`, `F1@10`, `F1@25`, `F1@50`);
  - `predictions.npz` with `names`, `gt`, `pred` as object arrays.
- **Patch latents (for diagnostics):** in the same forward pass, take the pre-quantization latent, reshape to
  (T, V·M·latent_dim), and average within each non-overlapping patch of `patch_size` frames. A final partial patch is
  averaged over its frames. Save as `patch_latents` (object array) in `predictions.npz`.

### 5.6 `diagnostics.py`

`diagnose_unit(unit, predictions_npz) -> dict`, written to `diagnostics.json`.

- **Subject:** `script.week5.raw_kmeans_cross_dataset.datasets.SPECS[dataset]` and `subject_of(spec, name)`. HuGaDB and
  LARa only; BABEL returns `null`.
- **Code AMI** (sklearn `adjusted_mutual_info_score`, arithmetic averaging):
  - frame codes vs subject;
  - frame codes vs recording.

  If over 300,000 frames, use an evenly spaced subsample (declared).
- **Per-class subject AMI** (uses labels; analysis only): for each GT class with at least 1,000 frames, the AMI of codes vs
  subject within that class. Report each class and the frame-weighted mean. Mark the block `"uses_labels": true`.
- **Latent neighbours:** same-recording share of nearest neighbours of `patch_latents`, using
  `script.week5.timescale_inspection.units9.nn_same_recording_share(feat_by_name, exclude_radius=1)`, so the query's
  immediately adjacent patches are excluded. Defaults for query and pool sizes, seed 0.
- No training, no GPU.

### 5.7 `run.py`

CLI:

```
python -m script.week5.masq_experiments.run --run-id RID --stage {gates,smoke,1,2} [--hours H] [--dry-run] [--cpu]
python -m script.week5.masq_experiments.run --run-id RID --unit UNIT_ID --step {train,score,diagnose}   # worker (internal)
```

- **Manifest** (`results/.../<rid>/manifest.json`): run ID, stage, code hash, analysis hash at launch, the settings of
  every unit in the stage, python and torch versions, GPU name, start time, status (`running`, `finished`,
  `budget_stop`, `failed`).
- **Resume:** if the manifest exists with a different code hash, refuse and tell the user to choose a new run ID. Never
  overwrite another run's results.
- **Unit caching:** a step is done when its output file exists and its stored identity equals `unit_identity(unit)`.
  - `train` → `training_info.json` + `epoch-30.model`;
  - `score` → `scores.json` + `predictions.npz`;
  - `diagnose` → `diagnostics.json`.

  Write files atomically (temp file + rename).
- **Process isolation:** each (unit, step) runs as a child process. A nonzero exit (native crashes leave no traceback;
  see Phase 9) is retried up to 3 attempts with a 15 s pause, printing the exit code in hex. A step failing 3 times is
  recorded in `manifest.not_run` with its last exit code, and the driver moves on.
- **Budget:** `--hours` is required for stages 1-2.
  - Unit cost estimates (minutes, train + score), to be replaced by smoke-run timings stored in `gates.json`: hugadb 10,
    lara 15, babel1 12, babel2 5, babel3 5.
  - Do not start a unit projected to cross the deadline; record it as `not_run: budget`.
- **Order per unit:** train → score → diagnose. Diagnostics may lag; they never block training.
- **Stage `gates`:** runs `tests.py`'s G1-G3 in-process, then G4 (§7) on GPU, writing `gates.json`. Before stages 1-2,
  the driver runs the gates stage itself if this run ID has no `gates.json` with G1-G4 passed under the current code
  hash (minutes). It refuses to train if any gate fails.
- **Stage `smoke`:** one unit per arm on babel3, seed 111:
  - `max_batches_per_epoch=3`, 2 epochs;
  - score and diagnose the smoke checkpoints (marked `smoke: true`; never reported as results);
  - store per-step timings in `gates.json`;
  - must finish within 10 minutes.
- **`--dry-run`:** print the unit list, the cached or pending status of each, the projected time per stage and in total,
  and exit without training.
- **Finalize** (after the last unit of a stage): run `report.py`, set the status, print the report path.

### 5.8 `report.py`

`write_report(run_dir) -> Path` writes `REPORT.md` and `scores.csv`.

- **Header:** run ID, status, code and analysis hashes (flag if the analysis hash changed since launch), gate summary,
  `not_run` list.
- **Cells:** one table per dataset. Rows are arms; columns are per-seed values and mean ± SD for all five global metrics.
  A second, separate table gives per-sequence (local) metrics. Also report segments per recording and code occupancy.
- **References:** MASQ published, SMQ released (e0), and the SMQ retrained spread, all shown next to the cells.
- **Comparisons**, paired by seed, MoF and F1@50 co-primary (Edit and F1@10/25 in an "every metric" table):
  - **R1:** M-pub − S0.
  - **R2:** M-pub mean vs MASQ published, within 3 points on both MoF and F1@50 → "reproduced".
  - **R3:** S0 vs SMQ released (e0) and the retrained spread.
  - **R4:** M-pub − M-JL; M-fix − M-pub. Stage 2 only; "not run" otherwise.
  - **R5:** S0 vs M-pub on subject AMI, recording AMI and the latent neighbour share, plus per-class subject AMI labelled
    "uses labels".
- **Reading function** (use exactly): with 3 paired seeds, "passes the bar" if mean ≥ 3 and all differences > 0;
  "reverse passes" if mean ≤ −3 and all < 0; otherwise "within the bar: comparable performance, not equivalence". With
  fewer than 3 seeds: "incomplete (n seeds)".
- **G5:** S0 mean on HuGaDB and LARa within 2 SD of the retrained spread, for MoF and F1@50 → passed. Otherwise flag
  every R1 reading on that dataset as "control outside expected spread".
- **Verdict** (plan §B.7, computed, not written by hand):
  - replicated = R1 passes on MoF in ≥3 of 5 datasets, including LARa;
  - partly = R1 passes on 1-2 datasets, or MoF gains pass but R2 fails everywhere;
  - not replicated = no R1 pass.

  Print the counts behind the verdict.
- **Caveat block** (verbatim from plan "What Tier 0 cannot show", plus P7): development results, published MASQ numbers
  are single runs, our arms are a reimplementation from the paper text.

### 5.9 `sheet.py` (base python; reads `scores.json` only)

- Adds or replaces sheet **"MASQ replication"** in `outputs/all_runs/all_runs_results.xlsx`. Back up first, to
  `results/week5/masq_experiments/<rid>/all_runs_results.backup.xlsx`.
- Copy header rows 1-3, styles, merged cells and column widths from sheet `Runs`. Then write one row per (arm, dataset)
  with seed means:

  | col | value |
  |---|---|
  | A | `<rid> (<arm>)` |
  | B | `results\week5\masq_experiments\<rid>\REPORT.md` |
  | C | HuGaDB / LARa / BABEL-1 / BABEL-2 / BABEL-3 |
  | D | seeds that completed |
  | E | `TCN joint-disentangled (SMQ backbone), trained` |
  | F | 29 |
  | G | `vq (EMA)` |
  | H | `k=c` |
  | I | S0: `joint-distance reconstruction + commit`; M-pub/M-fix: `+ JLSD p=<p> + mask-aware velocity λ=0.001`; M-JL: `+ JLSD p=<p>` |
  | J-M | `na` |
  | N | `vq codes (native)` |
  | O | C |
  | P | patch_size |
  | Q | `none` |
  | R-AB | `na` |
  | AC | `none` |
  | AD-AH | `na` |
  | AI-AM | global MoF, Edit, F1@10, F1@25, F1@50 (2 decimals) |
  | AN | notes |

- **Notes (AN):** arm and its role; "epoch 30 (final), global Hungarian, pooled/transductive, seed-mean shown"; incomplete
  seeds; "reimplementation of MASQ from the paper text" for M arms.
- **No comparison columns.** The sheet records configurations and scores only (user instruction).
- **On `PermissionError`:** write `results/.../<rid>/all_runs_results.staged.xlsx` and print a message asking the user to
  close Excel.

### 5.10 `tests.py`

Run with `python -m script.week5.masq_experiments.tests`. Each test prints PASS or FAIL and the run exits nonzero on any
failure. Required tests:

1. **G1 off-switch equivalence (CPU, float64, deterministic):** for hugadb and babel3 shapes, with a random input and a
   mask that includes padding:
   - `MASQModel(jlsd_p=0)` and `SMQModel`, built from the same seed and state dict, give identical outputs, `indices`,
     `commit_loss` and parameter gradients of the §5.3 total loss with `vel_weight=0` (`torch.equal`, or max abs diff
     ≤ 1e-12 in float64);
   - repeat in `train()` mode (EMA updates included, same RNG state) and in `eval()` mode.
2. **G2 JLSD mechanics:** with p = 0.5 on a synthetic latent:
   - dropped rows are exactly zero at every (t, channel);
   - kept rows equal input × 2;
   - over 10,000 draws the drop rate is within 4 binomial SDs of p;
   - `eval()` applies no drop;
   - `joint_keep.view(N, M, V)` maps row r = n·M·V + m·V + v (construct a case where only joint v=3 of sample n=1 is
     dropped and check that it lands there).
3. **G3 velocity loss:**
   - a hand-computed value on a 2×2×4×3×1 tensor;
   - zero when x_hat = x;
   - unchanged when dropped joints or padded frames are perturbed;
   - normalization = sum over channels / count of valid (pair, joint) points.
4. **Seeding:** two `train_unit` calls (CPU, `max_batches_per_epoch=1`, 1 epoch, same seed) see the same first batch file
   list and the same initial weights; different seeds see different ones.
5. **Identity/caching:** changing any setting or the code hash changes `unit_identity`; a step with a stale identity is
   not treated as done.
6. **Reading and verdict logic:** synthetic paired results covering pass, reverse, within, incomplete, and each verdict
   branch.
7. **No labels in training code:** a static check that `model_masq.py`, `losses.py` and `train_masq.py` do not reference
   `groundTruth`, `mapping`, `read_mapping_file`, `create_correspondences` or `gt_path`, except the single `get_num_actions`
   call in `train_masq.py`.

### 5.11 `run.ps1`

Model it on `script/week5/timescale_inspection/run_phase9.ps1`:
- parameters `-RunId`, `-Stage gates|smoke|1|2`, `-Hours`, `-DryRun`, `-Cpu`, `-PythonExe`;
- resolve the smq python;
- set the thread caps and `PYTHONPATH`;
- require `-Hours` for stages 1-2;
- call `python -m script.week5.masq_experiments.run ...`;
- exit with its code.

Recommend in README that the user pipes output to a log: `*>&1 | Tee-Object masq_<stage>.log`.

---

## 6. Order of work (do not reorder)

1. **Read:**
   - the plan;
   - `src/model/smq.py`, `src/model/motion_quantizer.py` (forward, `replace`), `model.py` (`Trainer.train`), `main.py`
     (defaults, seeding), `batch_gen.py`;
   - `script/repro/e0_metrics.py`, `gates5.py::smq_global_scores`;
   - `units9.nn_same_recording_share`, `raw_kmeans_cross_dataset/datasets.py` (`SPECS`, `subject_of`).
2. **Write** `config.py`, `model_masq.py`, `losses.py`. Write tests 1-3 and pass them before anything else.
3. **Write** `train_masq.py` and pass tests 4-5.
4. **Write** `score.py`. Run G4 (§7) on HuGaDB and LARa.
5. **Write** `diagnostics.py`. Unit-test it on a small synthetic `predictions.npz` (AMI = 0 for codes independent of
   subject, and positive for codes equal to subject).
6. **Write** `run.py`, `report.py` (with test 6), `run.ps1`, `README.md`. Pass test 7.
7. **Gates stage:** `run.ps1 -RunId masq_smoke -Stage gates`.
8. **Smoke stage:** `run.ps1 -RunId masq_smoke -Stage smoke` (≤10 min). Check finite losses, nonzero code occupancy, all
   files written, and REPORT.md generated, marked smoke.
9. **Dry runs:** `run.ps1 -RunId masq_dry -Stage 1 -DryRun` and `-Stage 2 -DryRun`, using the smoke timings (read from
   `masq_smoke/gates.json`). Always use the throwaway ID `masq_dry`, never `masq_v1`, and delete it afterwards.
10. **Write `sheet.py`.** Test it against the smoke run's `scores.json` writing to a **staged copy only**
    (`--out <scratch path>`), never the real workbook, until real results exist.
11. **Delete** throwaway run folders (`masq_smoke`, `masq_dry`) after checking no python process is using them.
12. **Hand off** (§9).

### 6.8 Stage 3 (specified for later; do not implement unless asked)

Subject-disjoint check: S0 and M-pub on HuGaDB and LARa with the four subject folds from `script/e2_prepare_splits.py`
(verify fold definitions first). Train on three folds, score the held-out subjects with the global evaluator restricted
to them, one seed per fold.

---

## 7. Gates

| gate | what | pass condition | when |
|---|---|---|---|
| G1 | MASQModel(p=0) ≡ SMQModel | tests.py 1 | before any training |
| G2 | JLSD mechanics | tests.py 2 | before any training |
| G3 | velocity loss | tests.py 3 | before any training |
| G4 | scorer parity | `score.py` on `models/pretrained/{hugadb,lara}.model` reproduces `results/e0/{hugadb,lara}_pretrained.json` (global and local) within 0.005 on every metric; the model is built as `MASQModel(jlsd_p=0)` and loads the released state dict with no missing or unexpected keys | before any training |
| G5 | control sanity | report.py check (§5.8) after Stage 1 | at report time |

G1-G4 results go to `gates.json` with the code hash. If any of G1-G4 fails, stop and report. Do not proceed to the smoke
run.

---

## 8. When to stop and ask

- G1 cannot be made exact without editing SMQ's files.
- G4 fails.
- GPU out-of-memory in the smoke run. **Allowed responses, in order:**
  1. `grad_checkpoint=True` for that dataset (results unchanged; `main.py` documents this);
  2. only if that fails, `micro_batch_size` for that dataset. This changes the EMA update granularity, so it applies to
     **all arms** of that dataset and is recorded as a deviation.

  Report either response before production.
- NaN or inf losses in any arm.
- Any setting in §3 appears to conflict with the SMQ code, for example a constructor argument that doesn't exist.
- Anything would require reading labels in training code.

---

## 9. Hand-off (what to report to the user)

1. Files created, with one line each.
2. Test output: all PASS lines, with any failure verbatim.
3. `gates.json` summary: G1-G4 status, and the G4 max differences.
4. Smoke run: per-arm step timings, losses after 2 epochs, code occupancy, and confirmation that it finished in under
   10 minutes.
5. Dry-run projections for Stage 1 and Stage 2, against the plan's estimates (about 4.9 h and 4.6 h). Flag a deviation
   above 25%.
6. Any deviation from this document, with the reason.
7. The launch commands, for the user to run:

   ```
   .\script\week5\masq_experiments\run.ps1 -RunId masq_v1 -Stage 1 -Hours 6 *>&1 | Tee-Object masq_stage1.log
   .\script\week5\masq_experiments\run.ps1 -RunId masq_v1 -Stage 2 -Hours 6 *>&1 | Tee-Object masq_stage2.log
   ```

   Stage 2 reuses the same run ID; Stage 1's units are cached.

Do not launch Stage 1 or 2. Do not edit the plan's declared settings. Do not write the spreadsheet sheet until real
Stage 1 results exist and the user asks.
