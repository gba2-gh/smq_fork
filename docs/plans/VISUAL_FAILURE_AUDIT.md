# Visual failure audit across motion projects

The complete implementation and human-inspection plan is now [VISUAL_INSPECTION_PLAN.md](VISUAL_INSPECTION_PLAN.md). This document retains the initial audit scope and delivered LARa slice.

The LARa-only workspace is at [the inspection atlas](../../results/visual_audit/lara_001/index.html).

## Purpose

Use existing predictions and synchronized motion to establish what failures look like before assigning a mechanism to aggregate scores. This is retrospective diagnosis, not an additional training sweep. Findings on inspected labeled populations are development evidence, not untouched evaluation.

An excess of predicted segments establishes an aggregate count imbalance. It does not establish that all true actions are split, that missing transitions are absent, or that switches represent meaningful sub-actions. Visual inspection should support or correct each claim.

## First delivered slice

`results/visual_audit/lara_matched_001/index.html` compares the six recent matched LARa arms, three seeds, all 44 evaluation recordings. It uses the saved global evaluation mapping without per-recording rematching. All five pooled metrics are checked against saved scores. The viewer contains aligned GT/prediction timelines, raw states, an explicitly label-assisted many-to-one display, per-recording metrics, action recall and subject MoF. Exact per-GT-segment diagnostics and file hashes are exported. No fitting or transport solves are performed.

This is held-out decoder evaluation with transductive tokenization and overlapping subjects. It does not yet cover historical contextualizer runs, all datasets, or the other repositories, and does not include motion replay.

Rebuild with the `smq` Python environment:

```powershell
python script/visual_audit/build.py
```

## Common record contract

Every adapter must supply dataset, canonical recording identity, subject if known, FPS, exact frame timestamps or coverage intervals, GT labels, raw units/states, predicted labels, mapping and mapping-fit population, run/seed, protocol, and source hashes. Optional artifacts: token IDs, emissions before OT, final soft assignments, embeddings, and source motion.

Verify matching frame counts and timestamps; preserve absent/unavailable fields explicitly. Never interpolate labels to hide misalignment. Reproduce original metrics under their original background and matching rules before comparing methods. Distinguish pooled/transductive runs, held-out recordings, held-out subjects, and oracle-boundary experiments. Compare numerical gains only on matched populations and protocols; matched-recording pictures alone do not make protocols equivalent.

## Views to complete

1. **Aligned motion replay and timelines.** Select an interval, replay its skeleton or relevant sensor traces, and show GT, tokens, pre-OT decisions, and final predictions on the same time axis. Use physical seconds, fixed action colors, and label raw state IDs as model-local. A movement change inside a labeled action may be legitimate motion structure; annotation granularity needs inspection.
2. **Recording and action atlas.** Sort every recording by recall, segment F1, extra/missed boundaries, and change relative to a matched reference. Show per-action recall, precision, duration, and GT-segment error distributions. Include subject and acquisition metadata where available. Do not hide failed or missing recordings.
3. **Boundary audit.** Report exact timing and declared tolerances in seconds and tokens. Use one-to-one class-agnostic matching separately from label-aware segment F1. Plot boundary offsets and internal splits. Extra and missed counts can both arise from shifted transitions, so these names describe observations rather than causal categories.
4. **State-to-action audit.** Show frame counts and both row- and column-normalized contingency matrices. Keep majority-class occupancy visible. Compare the saved mapping with oracle many-to-one relabeling only as a diagnostic. Inspect whether multiple same-action states are reproducible movement phases, subject partitions, or unstable switches.
5. **Before/after decoder audit.** Where saved emissions permit, distinguish an incorrect local decision corrected by OT, an incorrect decision propagated by OT, a correct transition suppressed, and a state swap induced by global matching. Do not regenerate expensive transport plans just to populate a view without an explicit compute decision.
6. **Cross-run persistence.** On identical recordings, measure which GT segments consistently fail across seeds and methods, and which fail only under one representation or objective. Align intervals by recording and time, never by raw state number. Averages across seeds should not conceal unstable label assignments.

## Evidence and example policy

Include all-recording summaries. Select a fixed deterministic spread for narrative figures: worst, median, best, largest matched improvement, and largest matched deterioration. Label the selection criterion and avoid treating selected clips as population estimates. Display counts and denominators for subgroup claims.

Maintain an observation ledger: recording/time, actions involved, visible event, quantitative evidence, candidate explanation, alternatives, and whether raw motion was inspected. For example, "six switches within a Handling segment" is an observation; "the model discovered six reusable sub-actions" is a hypothesis.

## Project coverage

- **SMQ:** current LARa slice is ready. Connect representative released SMQ, categorical OT, contextual OT, Phase 9, and grouping runs, then extend to HuGaDB and BABEL. Historical single-run plots in `outputs/ot_pairings/` and temporal diagnostics in `script/repro/d_error_decomposition.py` are reusable references, not a cross-run audit.
- **BehaveMAE:** saved `experiments/fsq_exp3/runs/*/codes_eval.npz` artifacts exist. Validate their recording identities, token intervals, and evaluation protocol before displaying them alongside SMQ results. Code sequences alone are not action predictions.
- **LAPS:** `output/moveme_analysis/` and `output/segment_action_similarity/` contain analyses. Locate the underlying time-aligned predictions and source motion; summary reports alone cannot populate an aligned failure atlas.

A common viewer can cover all projects, but datasets and incompatible protocols must remain visibly separated. The next implementation work is adapters and synchronized motion inspection, not more model training.
