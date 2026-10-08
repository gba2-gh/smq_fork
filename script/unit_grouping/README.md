# Unit-to-action grouping test (script/unit_grouping)

Implements `docs/plans/UNIT_GROUPING_TEST_PLAN.md` (v2). LARa, T2 units: fit the M = 24 state matched B0 on the 395 fitting
recordings, decode fitting and evaluation recordings afresh under the frozen prototypes, learn label-free groupings of the 24 states into
8 actions from the fitting plans only, relabel and score on the 44 evaluation recordings.

| file | role |
|---|---|
| `config.py` | declared settings (M = 24, h = 8, 100 random partitions, bar, floor, percentile) |
| `grouping.py` | affinity, spectral grouping, shuffle control, appearance grouping, size-matched random partitions, relabelling (plans in, groups out; no labels) |
| `run.py`, `run.ps1` | cells in fresh child processes, scoring and diagnostics, declared decision rule, REPORT.md |
| `tests.py` | 11 tests; a pass writes `results/unit_grouping/verification.json` for the current code hash |

The Milestone 1 fit/decode code is reused with `compositional_dynamics.config.NUM_STATES` set to 24 at run time. `decode_ladder` now also
returns the final soft plans (`responsibilities`); its other outputs are unchanged.

```
.\script\unit_grouping\run.ps1 -Preflight
.\script\unit_grouping\run.ps1 -DryRun -Splits primary,fold0,fold1,fold2,fold3
.\script\unit_grouping\run.ps1 -Smoke -RunId ug_smoke_02 -Splits primary,fold0
.\script\unit_grouping\run.ps1 -Run -RunId ug_lara_001 -Splits primary,fold0,fold1,fold2,fold3 -Hours 10 -ReuseCompleted *>&1 | Tee-Object ug.log
```

Outcome precedence when more than one condition fails: STOP (a fails), then PRIOR_MASS (d fails), then OVERCOMPLETE_ONLY (b or c fails),
else TEMPORAL_GROUPING. The verdict is INCONCLUSIVE unless all seeds completed and `verification.json` matches the code hash.
`b0_c` is read from the Milestone 1 run (`results/compositional_dynamics/m1_lara_001`), primary split and folds.
