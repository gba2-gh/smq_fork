# Milestone 1: fixed-unit action dynamics (script/compositional_dynamics)

Implements `docs/plans/COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md` (v2.1; changes in `docs/plans/M1_AGENT_INSTRUCTIONS_CHANGES_v2.1.md`).
Question: does action-specific prediction of **ordered** local-unit sequences give better action evidence than occupancy and
order-agnostic prediction, and improve on a matched fixed-centroid decoder? LARa only, T2 relational K-means units (K=512, C=8),
seeds 111/222/1538574472, the existing byte-identical split (395 fitting / 44 evaluation recordings).

**Held-out decoder evaluation with transductive tokenization.** The tokenizer was fitted on all recordings. Prototypes, occupancy,
model gradients, calibration and every assignment refresh use the 395 fitting recordings only; the model is frozen before the 44
evaluation recordings are decoded. Seed spread is initialization, not split variation.

## Arms

| arm | what it is |
|---|---|
| `b0_centre` | fixed T2 centroid ASOT, refitted on the fitting recordings only (matched baseline, common initialization) |
| `m1_ordered` | occupancy + action-conditioned causal unit prediction (GRU over the 8 preceding codes, residual adapters per action) |
| `m1_bag` | same network, history canonically sorted by a fixed random key (order-agnostic control) |
| `occupancy` | action-specific categorical occupancy only |

`D_raw = -0.5 log theta[a,c_t] - 0.5 log p_a(c_t | previous 8 codes)`; scaled by a frozen median row range; 5,000 updates with a plan
refresh (25 accepted inner steps) every 500; final decode from a cold start, continuation ladder 500 -> 4,000 steps.

## Modules

| file | role |
|---|---|
| `config.py` | declared settings, scales (full, smoke), code hash |
| `data.py` | preparation access, source verification, window populations with finite histories |
| `model.py` | the history model (`gelu` is explicit; see the note below) |
| `train.py` | occupancy, costs, calibration, plans, matched B0, alternating fit (fitting recordings only) |
| `decode.py` | frozen final decoding ladder |
| `evaluate.py` | the only module that reads labels: scoring, assignment agreement, identity MI/NMI, assigned-state NLL |
| `report.py` | tables, the Section 10 gate, REPORT.md |
| `parity.py` | historical pooled B0 reproduction (separate from the experiment) |
| `run.py`, `run.ps1` | modes, cells in fresh child processes, estimates |
| `tests.py` | 18 scientific-contract tests |
| `run_overnight.ps1` | preflight, tests, parity, then primary + fold cells under one time cap |

## Commands (repo root, `smq` environment, threads capped at 8)

```
.\script\compositional_dynamics\run.ps1                         # no mode: prints help
.\script\compositional_dynamics\run.ps1 -Preflight              # source verification only; no training
.\script\compositional_dynamics\run.ps1 -DryRun                 # projected time per cell
.\script\compositional_dynamics\run.ps1 -Smoke -RunId m1_smoke_02 -Device cuda      # ~30 s, reduced population, throwaway
.\script\compositional_dynamics\run.ps1 -Parity                 # historical B0 reproduction, about 10 min
python -m script.compositional_dynamics.tests                   # writes results/compositional_dynamics/verification.json
.\script\compositional_dynamics\run.ps1 -Run -RunId m1_lara_001 -Arms b0_centre,m1_ordered,m1_bag,occupancy -Seeds 111,222,1538574472 -Device cuda -Hours 10
.\script\compositional_dynamics\run.ps1 -Run -RunId m1_lara_001 -Hours 10 -ReuseCompleted    # restart: completed cells are kept
```

`-Run` needs `-Hours` (a cap, not a runtime promise) and a run id without the word "smoke"; `-Smoke` needs one with it.

## Overnight (v2.2)

```
.\script\compositional_dynamicsun_overnight.ps1 -RunId m1_lara_001 -Hours 10 *>&1 | Tee-Object m1_overnight.log
```

Runs preflight, tests, parity, then the primary split followed by four subject-disjoint folds (`-Splits primary,fold0,...,fold3`),
with one elapsed-time cap. Fold cells are named `<arm>__seed<k>__fold<j>`. Multiplier decodes (0.5, 2) are stored in each primary
cell's `scores.json`. Secondary results are descriptive only; the gate is computed from the primary split.

## Outputs

`results/compositional_dynamics/<run_id>/`: `manifest.json`, `cells.csv`, `training.csv`, `transport.csv`, `gates.json`, `not_run.csv`,
`REPORT.md`, and per cell `cells/<arm>__seed<k>/`: `info.json` (status: running / complete / interrupted / invalid), `scores.json`,
`diagnostics.json`, `training.json`, `states.npz`, `frame_predictions_{fit,eval}.npz`, `costs_{fit,eval}.npz` (scaled cost, separate
occupancy and conditional NLL, and for neural arms the history representations), `final_plans.npz`, `model.pt`, `calibration_ids.npy`.
A cell is reusable only if its status is `complete` and its identity (code hash, scale, seed, arm, source hash, matched-B0 identity)
matches. Interrupted cells restart from their recorded initialization.

The gate (Section 10) is computed in `REPORT.md` / `gates.json`. It reports INCONCLUSIVE unless all three seeds completed for the
arms compared and `verification.json` records a pass for the current code hash.

## Environment note: torch 2.1.0 CPU `F.gelu` backward

In this environment the float32 backward of `F.gelu` returned gradients about 10^4 times too large when the incoming gradient was
non-contiguous (it was, after the adapter einsum), while float64 matched finite differences. Training then stalled at the uniform
predictor. `model.py` therefore uses an explicit erf GELU, and `tests.py::float32_gradients_match_float64` guards it. Other code in
this repository that uses `F.gelu` or `nn.GELU` in float32 on CPU has not been checked.
