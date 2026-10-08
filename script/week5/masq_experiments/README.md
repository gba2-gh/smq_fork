# Tier 0: MASQ replication (script/week5/masq_experiments)

Seeded replication of MASQ (arXiv 2608.29891) on this repository's SMQ code. MASQ = SMQ + joint-level structured dropout
(JLSD) + a mask-aware velocity loss; the inter-joint distance reconstruction is already SMQ's default.

Plan and protocol: `docs/plans/week5/masq_experiments/TIER0_PROTOCOL_MASQ_REPLICATION.md`. Implementation brief:
`docs/plans/week5/masq_experiments/AGENT_INSTRUCTIONS.md`. Declared settings: `config.py` (copied from the brief; never
changed because of a score).

## Arms

| arm | JLSD p | velocity λ | stage |
|---|---|---|---|
| S0 (SMQ retrained) | 0 | 0 | 1 |
| M-pub | 0.25 / 0.20 / 0.50 / 0.50 / 0.50 (HuGaDB / LARa / BABEL-1/2/3) | 0.001 | 1 |
| M-JL | as M-pub | 0 | 2 |
| M-fix | 0.25 (LARa, BABEL-1/2/3) | 0.001 | 2 |
| V1 | as M-pub's HuGaDB rate (0.25) | coordinate velocity loss, weight calibrated to be scale-free (rho 0.0016) | hu |
| V2a / V2b | 0.25 | latent smoothness prior on pre-quantization latents, calibrated (rho 0.01 / 0.1) | hu |

V1, V2a and V2b exist only for the HuGaDB-focused follow-up (`TIER0_RESULTS_AND_HUGADB_FOCUS.md`): M-JL plus one calibrated term
whose weight is set at initialization so that weight x term = rho x reconstruction (label-free, scale-free).

Seeds 111, 222, 1538574472; 30 epochs; the epoch-30 checkpoint is the only one ever scored. Global Hungarian mapping is
primary; per-sequence results are reported separately.

## Modules

| file | role |
|---|---|
| `config.py` | settings, units, identities, published references (torch-free) |
| `model_masq.py` | `MASQModel(SMQModel)`: JLSD after the encoder; `build_masq` |
| `losses.py` | mask-aware velocity loss; loss assembly |
| `train_masq.py` | one training unit (copy of `model.Trainer.train` + velocity term) |
| `losses_variants.py`, `train_variants.py` | the V1/V2 terms, their label-free calibration, and their training unit |
| `report_hu.py` | REPORT_HU.md: variants vs the masq_v1 references, with the declared stop rule |
| `score.py` | one scoring unit (the e0 / gates5 path) and gate G4 |
| `diagnostics.py` | subject / recording AMI of codes, latent same-recording neighbour share |
| `run.py` | driver: gates, smoke, stages 1-2, caching, retries, manifest, dry run |
| `report.py` | REPORT.md, scores.csv, R1-R5, G5, computed verdict |
| `sheet.py` | spreadsheet sheet (base python, no torch; only after real results exist) |
| `tests.py` | 10 tests; G1-G3 are tests 1-3, G6/G7 are tests 9-10 |
| `run_all.ps1` | chains Stage 1 then Stage 2 (waits for a live run with -WaitForRunning) |

## Running

All commands from the repo root. Python: the `smq` conda env (the launcher finds it). Thread caps are set to 8.

```
.\script\week5\masq_experiments\run.ps1 -RunId masq_smoke -Stage gates          # G1-G4, minutes
.\script\week5\masq_experiments\run.ps1 -RunId masq_smoke -Stage smoke          # <= 10 min, throwaway
.\script\week5\masq_experiments\run.ps1 -RunId masq_dry   -Stage 1 -DryRun      # projections only
.\script\week5\masq_experiments\run.ps1 -RunId masq_v1 -Stage 1 -Hours 6 *>&1 | Tee-Object masq_stage1.log
.\script\week5\masq_experiments\run.ps1 -RunId masq_v1 -Stage 2 -Hours 6 *>&1 | Tee-Object masq_stage2.log
.\script\week5\masq_experiments\run.ps1 -RunId masq_hu1 -Stage hu -Hours 2 *>&1 | Tee-Object masq_hu1.log   # HuGaDB variants, ~1 h
```

Per-unit minutes (config.UNIT_MINUTES) are the values measured in masq_v1; smoke timings are no longer used for projections.

Stage 2 reuses the same run ID; Stage 1's units are cached. Stages 1-2 run the gates first if the run ID has no passing
`gates.json` for the current code hash. A different code hash on an existing run ID is refused.

Tests: `python -m script.week5.masq_experiments.tests [--only N ...]`.

## Outputs

```
models/week5/masq_experiments/<run-id>/<arm>/<dataset>/seed<k>/   epoch-N.model, config.json, losses.jsonl, training_info.json
results/week5/masq_experiments/<run-id>/
    manifest.json  gates.json  REPORT.md  scores.csv
    units/<arm>__<dataset>__seed<k>/   scores.json  predictions.npz  diagnostics.json
```

## Gates

G1 MASQModel with p = 0 and no velocity loss equals SMQModel exactly (outputs, indices, losses, every gradient; train and
eval); G2 JLSD mechanics; G3 velocity loss; G4 re-scoring the released SMQ checkpoints (HuGaDB, LARa) through this scorer
reproduces `results/e0` (global and local) within 0.005; G5 (at report time) S0 within 2 SD of the earlier retrained spread.

G6 (test 9): the variant calibration is scale-free, satisfies weight x term = rho x reconstruction, and leaves the RNG stream and
the quantizer untouched; G7 (test 10): the latent smoothness loss matches a hand computation, is scale invariant, ignores dropped
joints and padded frames, and maps rows correctly. Variant code is hashed separately and only into variant identities, so every
masq_v1 identity is unchanged (checked: 52 of 52).

G1-G3 run on CPU in float32, not float64: SMQ's quantizer builds float32 one-hots, so it is not float64-clean. G1 still
requires bitwise equality, because both models perform identical operations.

## Robustness

Each (unit, step) is its own process. A nonzero exit (native crashes leave no traceback; see Phase 9) is retried up to 3
times; a step that fails every attempt is recorded in the manifest's `not_run` and the driver moves on. Output files are
written atomically and cached by identity (settings + code hash).
