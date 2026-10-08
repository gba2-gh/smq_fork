# Changes to the Milestone 1 instructions: v2.1 → v2.2

6 October 2026 · Applies to [COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md](COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md)

**Why:** the declared run is projected at about 1.5 h and an overnight budget of 10 h is available. v2.2 spends spare time on readouts that make the go/stop decision more trustworthy. It does not widen the experiment (no new dataset, arm, capacity or tuning).

**Unchanged:** the question, the four arms, the model, training schedule, transport settings, the primary split, and the Section 10 gate and stop rule.

## What was added (new Section 11)

| Addition | What it is | Why | Cost |
|---|---|---|---|
| Subject-disjoint folds | 4 folds of 4 held-out subjects (file-name subjects, seed-111 shuffle, round-robin); fit on the other 12 subjects; same arms and seeds; one Hungarian mapping per fold | Tests the recurring identity problem directly: every recording is scored once on unseen subjects, instead of 44 recordings that share subjects with the fitting set | about 5 h |
| Decode multipliers 0.5 and 2 | Re-decode the evaluation recordings of each primary cell with the frozen cost scaled | Shows whether the ranking depends on the declared cost scaling; decode only | about 0.5 h |
| Parity in the overnight launcher | The existing Section 7 check, run first | Required verification that was still unrun | about 0.2 h |

## Rules

- Secondary readouts are descriptive. They cannot rescue a failed gate or overturn a passed one; a disagreement is printed as a warning in the report.
- The tokenizer is fitted on all recordings, so the folds are not fully inductive. The report says so.
- Folds fit on the primary evaluation recordings of the non-held-out subjects, so their fitting sets differ from the primary split's.
- Execution order is primary split (all seeds) first, then folds. If time runs out, cells projected to overrun are not launched and secondary cells go first.

## Implementation

- `data.py`: `subject_folds`, `split_names`, `load_cell_data(..., split)`.
- `run.py`: `--split`/`--splits`, fold-aware cell directories and identities (a fold cell depends on its own fold's B0), multiplier decodes in primary cells, `dry-run --splits`.
- `report.py`: gate computed on the primary split only; secondary sections for multipliers and folds, with the disagreement warning.
- `run_overnight.ps1`: preflight, tests, parity, then `-Run -Splits primary,fold0..fold3 -ReuseCompleted` under one cap.
- Tests: 18 (adds fold disjointness/determinism/label-freedom and multiplier-equals-scaled-cost). Smoke run covering the primary split and fold0 passed.

## Projection

`run.py dry-run --splits primary,fold0,fold1,fold2,fold3`: 6.7 h for 3 seeds, plus about 0.2 h parity. This is a lower bound, because the new arms' final decode may need more solver steps than the historical B0 ladder it borrows. The 10 h cap leaves about 3 h of margin.
