# Visual inspection plan: motion, units, context and action decoding

7 October 2026 | LARa implementation complete; human review and other datasets remain open

## 1. Objective

Establish **where, when and under which conditions the pipelines fail**, using saved predictions and synchronized source motion. Connect aggregate metrics to observable errors before choosing another architectural change.

This is a retrospective audit, not a training experiment. Do not retrain models, tune smoothing, select checkpoints or change label mappings to improve examples. Inspected labeled populations are development evidence, not new held-out confirmation data.

Questions:

1. Do excess predicted segments come from repeated switches inside actions, displaced boundaries, incorrect action identities, or a mixture?
2. Which actions, durations, transitions, recordings and subjects account for failures? Which recur across seeds and methods?
3. At what stage do errors emerge or change: local features, quantization, contextualization, transport, or state-to-action mapping?
4. When several states occupy one labeled action, do they correspond to repeatable movement phases, recording differences, or unstable switching?
5. Which observations support a particular intervention, and which explanations remain untested?

The audit must allow a finding of ambiguity. A temporal association or compelling clip is not proof of mechanism.

## 2. Scope and existing work

### Existing first slice

`results/visual_audit/lara_matched_001/` compares six matched LARa arms: T2 B0, body-scale B0, centered B0, ordered GRU, bag and occupancy. It covers three seeds and the same 44 evaluation recordings. All five pooled metrics reproduce the saved values across 18 cells.

The viewer contains action/state timelines, evaluation-label oracle relabeling, recording metrics, action recall and subject MoF. It lacks synchronized motion, historical pipeline comparisons and other repositories. Its static export was inspected; browser interaction verification remains outstanding.

The LARa-only implementation now lives at `results/visual_audit/lara_001/index.html`. It adds verified hip-relative joint views, all 44 aligned recording pages, saved local-cost and token layers, historical pooled run rows, an all-recording atlas, frozen inspection selection, observation ledger and claim audit. The grouped-state experiment has only aggregate scores: its evaluation predictions were not saved. The viewer was rendered in a browser and exercised with scripted control checks. Human observations remain pending.

Existing evidence: T2 B0 averages 42.1 extra and 17.3 missed boundaries per recording at a 12-frame tolerance (0.24 s), with one-to-one class-agnostic matching. Both occur in every recording/seed instance. Displaced transitions can produce both counts; they do not establish simultaneous causal splitting and merging.

### Target coverage

Inventory relevant runs, then register representative comparisons by experimental role, not favorable scores. Save exact run IDs, cells, checkpoints, seeds, populations and selection rationale **before selecting clips**.

| Project / dataset | Comparisons | Required distinction |
|---|---|---|
| SMQ / LARa | Released SMQ; raw/T2 B0; categorical/contextual OT; Phase 9; matched M1/input-representation arms; completed grouping arms | Earlier pooled runs and recent held-out decoder runs remain separate comparison groups. |
| SMQ / HuGaDB | Released SMQ; categorical/contextual OT; matched vocabulary-size and mask-guard comparisons; native VQ/FSQ versus K-means | Match representation, checkpoint, token duration and protocol before attributing a difference to one factor. |
| SMQ / BABEL-1/2/3 | Raw/relational inputs; matched long/short masks; contextualizer/no contextualizer; relevant Phase 9 arms | Each subset is a separate population. Recording identity is not subject identity. |
| BehaveMAE | Representative continuous/discrete and VQ/FSQ runs with their established controls | Tokens are not action predictions. Verify dataset, stride, timestamps and evaluation protocol independently. |
| LAPS | Token/vector transitions and saved segmentation/action-similarity analyses where aligned artifacts exist | Intrinsic movement segmentation and action-label prediction answer different questions. |

Include MASQ/histVQ only if aligned predictions and their protocols are available. A spreadsheet score cannot populate a visual comparison. Missing artifacts become coverage gaps, not instructions to train another model.

Inventory starting locations, not guarantees of artifact availability:

- SMQ: `results/preds/`, `results/action_transport_stage_c/`, `results/frozen_study/frozen_001/`, `results/quantized_input_comparison/`, `results/fsq_reconstruction/`, `results/week5/timescale_inspection/`, `results/compositional_dynamics/m1_lara_001/`, `results/input_representation/irt_lara_001/`, `results/unit_grouping/`.
- BehaveMAE: `C:/Users/gzaz976/Documents/repo/Behaviour/BehaveMAE/experiments/`, including `fsq_exp3/runs/*/codes_eval.npz`.
- LAPS: `C:/Users/gzaz976/Documents/repo/Behaviour/LAPS/output/moveme_analysis/` and `output/segment_action_similarity/`; locate their underlying aligned artifacts.
- Reusable tools: `script/repro/d_error_decomposition.py`, `outputs/ot_pairings/plot_ot_pairings.py`, `script/visual_audit/`.

## 3. Phase A — Inventory, alignment and comparability

Create a small adapter per artifact format. Keep shared metrics/rendering independent of project-specific loading. Preserve original files; write new implementation under SMQ's `script/visual_audit/` and outputs under `results/visual_audit/`.

Every registered cell must declare:

- Project, dataset/subset, run, arm, seed, checkpoint/prototype identity and source paths/hashes.
- Canonical recording ID, subject if supported, acquisition/session metadata if available, frame count, FPS and exact token coverage intervals.
- GT labels, raw states, final mapped labels, mapping method and mapping-fit population.
- Fitting/evaluation population, encoder/tokenizer exposure, background handling and scoring implementation.
- Available intermediate artifacts: input codes/vectors, contextual outputs, emissions, final plans, group memberships and source motion/sensors. Absence is allowed and displayed.

Verification gates:

1. Check exact recording and frame/timestamp alignment, including offsets, subsampling, padding and terminal windows. Never truncate or interpolate labels to conceal mismatches.
2. Reproduce saved pooled scores under the original protocol, within saved rounding precision. Quarantine discrepant cells from quantitative comparisons and record the reason.
3. Verify skeleton layout, joint names/edges, coordinate units and reference frame from loaders/metadata; do not guess topology. Verify sensor channels/units likewise.
4. Preserve each mapping's original scope. Never rematch the currently displayed recording. A historical local-mapping result cannot silently become comparable to global matching.
5. Declare comparison populations explicitly. An intersection-only diagnostic states its coverage and mapping treatment; it is not the original benchmark result.

Produce `compatibility.csv`: **matched comparison**, **descriptive side-by-side only**, or **unavailable**, with reasons. Models trained or selected differently are not controlled component ablations simply because their displayed recordings match.

## 4. Phase B — All-recording quantitative atlas

Calculate diagnostics over every available recording in each valid comparison group. Clips illustrate the findings; this atlas supplies population evidence.

| Level | Measurements / view |
|---|---|
| Recording | Original MoF/Edit/F1, GT/predicted run counts, duration distributions, occupied states, extra/missed boundaries, paired changes versus a declared reference. |
| Action | Support in frames/segments/recordings/subjects; precision/recall; segment F1 where supported; confusion; duration distribution; internal switches per GT segment and per second. |
| Transition | Directed GT action pairs, matched/missed boundaries, timing offsets and predicted identities on either side. Keep rare transitions with denominators. |
| Subject/session | Frame-weighted scores and recording distributions; show both frame and recording counts. Association alone does not establish a body-shape/style explanation. |
| State/group | State-to-action contingency counts, row/column normalization, state/class mass, saved mapping and oracle many-to-one diagnostic. |
| Cross-run | Paired changes on identical intervals; consistently missed GT segments versus method/seed-specific failures. Raw state numbers are never identities across independent fits. |

Boundary analysis is class-agnostic and separate from label-aware segment F1. Use maximum-cardinality one-to-one matching at declared tolerances: exact, 0.10 s, 0.25 s and 0.50 s, floored to integer frames. Also report one native token width. Preserve the existing 12-frame LARa diagnostic for continuity. These are descriptive sensitivities, not settings selected to favor a method.

Use the names **extra predicted boundaries** and **missed GT boundaries**. A displaced boundary can produce both. Do not match several predictions successfully to one GT boundary. For offset plots, use a deterministic minimum-total-absolute-offset tie break among maximum-cardinality matches.

Count predicted boundaries strictly inside each GT segment, and separately those more than one token from its edges. Report duration-normalized rates: longer actions have more opportunities for switches. Preserve background/None according to the source protocol and show it separately when useful.

Retain seed-level values and actual denominators. Frames/segments are dependent observations; seed variation is initialization sensitivity, not sampling uncertainty.

## 5. Phase C — Synchronized inspection workspace

Build one reusable offline viewer with project/dataset/protocol selection and lazy loading of recording details. Keep the existing LARa viewer usable as adapters are added. Do not embed all raw motion or embeddings in one giant HTML file.

### Main view: one shared cursor and time axis

1. **Source motion:** skeleton replay where verified coordinates exist; sensor traces for HuGaDB's IMU/EMG data. Do not manufacture skeletons from sensor features. Video is optional, only if already available and aligned.
2. **GT timeline**, including None/background and exact annotation intervals.
3. **Tokens/quantized vectors:** code-change events and IDs on demand, plus vector-change magnitude when saved. Avoid a 512-color strip.
4. **Context/local evidence:** saved contextual outputs or action-state emissions. Define confidence/margins and scaling explicitly; raw costs from different representations are not automatically comparable.
5. **Pre-OT/post-OT predictions**, using the same prototypes and a common mapping for the stage comparison. Independently remapping both outputs with labels does not isolate transport's effect. Standalone optimally mapped scores, if shown, occupy a separate labeled panel.
6. **Grouping/final actions**, alongside raw state IDs and an optional clearly marked evaluation-label oracle display.

Provide play/pause, speed, frame stepping, token/GT/predicted-boundary navigation, interval zoom and export. Use physical seconds and display token duration/receptive-field metadata. Action colors are stable within each dataset; raw state IDs remain cell-local.

For skeletons, use a fixed camera and consistent coordinate scale across matched clips. If centered/body-normalized display is offered, label it and retain the original view: automatic recentering/rescaling could hide the phenomenon. Sensor views likewise preserve original units alongside any explicitly normalized display.

### Supporting views

- Recording-by-method error table linked to intervals.
- Action confusion and duration/error plots with counts.
- State-to-action matrix linked to examples from different recordings/subjects.
- Other seeds on the identical interval.
- For grouping: raw M states, group memberships, fixed group IDs and final actions. Merging states alone does not establish meaningful motifs.

Embedding/UMAP plots are optional later; their appearance is not primary evidence of separability, identity or composition.

## 6. Phase D — Human inspection protocol

### Freeze the inspection list before watching motion

Choose one primary comparison group per dataset for the first pass. Seed 111 supplies the initial display, not the sole evidence. Save the selection algorithm and output:

- Five recording roles: worst, median and best reference F1@50; largest paired F1 improvement and deterioration for the declared contrast. Break ties by recording name; replace duplicates with the next eligible recording.
- Three additional recordings sampled uniformly without replacement from the remainder, seed 111. Inspect all if fewer exist and disclose the smaller count.
- Keep the same list across methods. Inspect whole-recording timelines, then flagged intervals with surrounding context.
- Add an action-stratified pass: up to three GT segments per class from distinct recordings where available, sampled with seed 111 independently of prediction errors. Record shortages. This exposes rare actions missed by score-based selection.

These are illustrative cases, not a prevalence sample. Use the full atlas for prevalence. Additional interesting clips are marked exploratory and cannot replace the frozen list.

### Procedure for each interval

1. Inspect source motion/sensors with GT first. Note annotation ambiguity, sensor anomalies and visible movement changes before seeing which model won.
2. Reveal anonymized method rows. Record observable boundary/identity differences before revealing model names.
3. Inspect available intermediate stages to locate where the difference emerges.
4. Revisit the identical interval under the other saved seeds. Retain instability; do not select the best seed.
5. Reveal identities and record candidate explanations, alternatives and missing evidence.

If two reviewers are available, independently annotate a small shared subset and retain disagreements. Otherwise mark single-reviewer judgments explicitly.

| Observable pattern | Checks before interpretation |
|---|---|
| Several switches inside one GT action | Duration, boundary proximity, visible movement, recurrence across recordings/seeds. |
| One prediction spans several GT actions | Whether transitions are present in inputs/emissions and suppressed by smoothing/grouping. |
| Correct-looking interval, wrong action | Saved global mapping, class confusion, and whether the state mixes actions elsewhere. |
| Consistent early/late boundary | Alignment, token grid, receptive field, annotation uncertainty. |
| Repeating states within an action | Cross-recording/subject recurrence and use of those states in other actions. Candidate structure, not proof of reusable movemes. |
| Subject/session concentration | Action composition, durations, acquisition and coverage confounds before an identity explanation. |
| Improvement after OT/grouping | Both corrected and harmed intervals; whether gains come from merging into a frequent class. |

## 7. Phase E — Evidence ledger and architectural decisions

Save `observations.csv`: dataset/protocol, run/seed, recording/subject, start/end frames, actions, selection reason, stage inspected, motion availability, direct observation, candidate explanation, alternatives, seed consistency, reviewer/status and exported figure/clip path.

Create `claims_audit.md` for existing claims: overfragmentation, missing transitions, identity shortcuts, insufficient local information, useful context, and state/action mismatch. For each, list supporting/contradicting cases, population measurements and one status:

- **Observed:** verified event or measured association.
- **Supported hypothesis:** recurring quantitative/visual pattern with relevant alternatives checked.
- **Unresolved:** missing data or competing explanations.
- **Contradicted:** examined evidence does not support the stated claim.

Link any next intervention to a concrete failure and a discriminating test. For example: correct local emissions suppressed by OT motivate decoder work; confusion already present before OT motivates representation/emission work; recurring states with inconsistent action assignment motivate grouping/context work. None follows from one clip or the absence of a visible distinction alone.

The audit cannot establish novelty or causal necessity. It should prevent expensive implementation based on the wrong diagnosis. Later architecture selection using this audit requires separately declared confirmation data/protocol.

## 8. Outputs and acceptance checks

New outputs: `results/visual_audit/cross_project_001/`. Preserve the first LARa slice.

- `run_registry.csv`, `compatibility.csv`, `missing_artifacts.csv`, `manifest.json`: identities, hashes, mapping scope, coverage and scoring policies.
- Seed-level recording/action/transition/segment diagnostics and denominators.
- Offline viewer, synchronized source replay, screenshots and short exported clips for reviewed cases.
- `inspection_selection.csv`, `observations.csv`, `claims_audit.md`, and `REPORT.md` separating observations, interpretations and next interventions.
- Minimal adapter documentation and one execution entry point that prepares audit artifacts, not models.

Acceptance:

1. Verify broadcasting with known token/GT boundaries and a short terminal token. Test boundary matching on exact, shifted, split and merged examples.
2. Reproduce saved metrics; ensure changing displayed recording/order/zoom never changes the mapping.
3. Check source/cursor alignment at the beginning, a middle transition and final frame for each artifact format. No silent FPS/topology assumptions.
4. Test dataset/seed/method selection, navigation, missing optional data, exports and browser rendering. Data verification alone does not complete UI QA.
5. Cover all registered populations quantitatively or list gaps. Complete the declared human sample and retain uncertainty rather than inventing explanations.

## 9. Implementation path and effort

1. **Inventory:** register exact cells across projects and validate a representative artifact per format.
2. **Complete LARa end to end:** add source replay, available intermediate layers, sampling and observation capture to the existing audit. Use this as the reference implementation.
3. **Extend adapters:** HuGaDB/BABEL and then compatible BehaveMAE/LAPS artifacts. Keep incompatible protocols separate; absent optional layers do not block other available views.
4. **Validate and inspect:** complete browser/alignment checks, conduct the frozen review and synthesize the evidence ledger.

Planning estimates, to update after inventory: 1–2 hours for registry/alignment discovery; 4–8 hours for the first complete LARa viewer and verification; 2–6 hours for remaining adapters depending on artifact consistency; 2–4 hours for the first human review. These are effort estimates, not a promise that complete cross-project coverage fits one overnight run. Time automatic preparation on one cell and cache results.

No GPU training is planned. Read source data lazily. If an essential intermediate is missing, record it and estimate targeted extraction separately. Do not silently run inference, transport solves or training to fill a view. Do not reduce coverage based on favorable outcomes; report partial coverage explicitly.
