# Stage A / Stage B implementation audit

26 September 2026. Review of the working tree, saved results and checkpoints; no experimental sweep launched or training implementation changed. Companion `verification.json` records source/result hashes, aggregates and the objective-check reproduction parameters. The existing experiment artifacts are preserved.

## Assessment

Stage A has a real outer-objective bookkeeping defect, but the evidence does not attribute its poor segmentation to a broken transport solver. Stage B's first training budget was binding; its longer rerun stopped by validation patience in every cell. The available optimization study demonstrates successful memorization and poor generalization of a simple copying task, not a diagnosed optimizer defect. The recent plan changes turn these observations into stronger causal and project-level conclusions than the measurements support.

The immediate problem is threefold: an incorrect Stage A stopping trace, incomplete Stage B experiment identity, and conflation of unsuccessful benchmarks with diagnosed mechanisms. There is no demonstrated single bug that explains both stages.

## 1. Confirmed implementation defects

### A1. Outer stopping evaluates a mixture of parameter states — high priority

Source: `script/action_transport/categorical.py:148–175`, used by both categorical and continuous fits.

Each iteration solves transport at the old parameters, adds those transport objectives, updates the parameters, and then adds the prior at the new parameters. For categorical fitting, the logged value is

`sum_r N_r F(T_new; theta_old) + prior(theta_new)`

instead of the declared objective of the returned state,

`sum_r N_r F(T_new; theta_new) + prior(theta_new)`.

For continuous fitting the prior is zero, but the logged objective still precedes the prototype update. Neither trace is the objective of the returned full iterate. This trace controls relative stopping, so the defect concerns stopping as well as reporting. Existing tests check the parameter update in isolation, not this integration.

**Reproduced:** a deterministic CPU example with two 30-window recordings, six codes, three states and one outer iteration logs **17.3050015091** while independently evaluating the returned state gives **7.0086905815**. The 10.2963109276 difference is not numerical roundoff. Full reproduction parameters are in `verification.json`.

**Repair:** after updating parameters, evaluate the objective at the saved transport states and updated parameters, including the updated prior. Alternatively, replace each old unary contribution with its new unary contribution; the other transport terms are unchanged. Test equality with an independent evaluation and nonincrease after both blocks. Version corrected runs separately.

**Scope of impact:** 43 of 52 saved fits reached the outer cap; only nine stopped by the current outer convergence rule. In a deterministic rerun, changing only a stopping diagnostic cannot change the updates of a fit that still executes all 50 iterations, but it can change the iteration at which other fits stop. This bug alone is not evidence that the main A1 score deficit will disappear. Saved artifacts omit the transport states needed to reconstruct the complete final fitting objective without solving again.

### B1. Changed validation masks are not reflected in run identity — high priority

Sources: `stageb_data.py:90`, `run_stage_b.py:30,123–140`.

The current `stable_seed` correctly uses CRC32. Its own change note documents that v1.0–v1.5 used Python's process-randomized string hash. Consequently, old masks were fixed within a process but not necessarily across independent runs. The old and new training comparisons therefore need not evaluate the same masked positions. Increasing the epoch limit was not a strictly matched comparison unless the process hash seed was also controlled and recorded.

The runner still declares version **1.5**, loads existing manifests without checking their compatibility, and skips completed cells on resume using only the cell key. It can mix old and new mask protocols under one run. Old checkpoints cannot establish the exact old mask realization unless its masks or process hash seed were separately preserved.

**Repair:** bump the protocol identity; persist mask realizations or a versioned deterministic mask recipe plus checksums, splits, tokenizer identity and complete model/training configuration. Refuse incompatible resume. Keep historical results as historical; reevaluating old checkpoints on new masks does not recreate their original checkpoint-selection process.

### B2. Optimization-study resume can mix different budgets — high priority before a sweep

Sources: `stageb_optstudy.py:149–205,223–237`.

`--smoke` runs the first named configuration on one cell for 300 steps. A full run uses the same configuration name, and `phase1_sweep` skips existing rows by `(cell, config)` only. Reusing a smoke study ID for a full study therefore silently accepts that short run as the full-budget cell. An existing `DESIGN.json` is preserved but not checked against the current configuration. Source changes under an unchanged name have the same risk. Selection checks row count, not the exact required set of unique cells.

**Repair:** separate smoke outputs; hash the effective configuration, budget, mask recipe, source and inputs; reject mismatches; require one valid row for every expected cell before selection. Persist per-fit learning curves and checkpoint/configuration artifacts, not only summary rows. No evidence was found that this contamination has already happened: the inspected study contains `DESIGN.json` and `phase0.json`, but no completed `sweep.csv` or `selection.json`.

### B3. Smaller implementation and operational issues

- `stageb_train.py:201`: floor division drops the final partial minibatch each epoch. HuGaDB uses 896/930 training chunks and LARa 2752/2765. Shuffling rotates the omitted chunks; this is not permanent exclusion. Include the final batch or explicitly document `drop_last`. This is too small, by itself, to diagnose the observed failures.
- `stageb_train.py:192,208`: training-time validation materializes all validation chunks and evaluates them in one forward pass, although the later diagnostic evaluator supports batches. It can exhaust memory on larger populations. No saved run here demonstrates such an OOM.
- `run_stage_b.py:141–146`: the runtime deadline is checked between cells, not within training or embedding extraction. It is not a hard elapsed-time limit.
- `stageb_optstudy.py:75`: the tiny-overfit docstring says fixed training masks, but `train` resamples training masks each epoch. The actual experiment tests memorization with varying masks and fixed validation masks; describe it accurately.
- `data.py:135–157`: tokenizer-cache reuse checks fitting indices and window counts, not recording identities or feature/checkpoint hashes. Same-shaped changed/reordered data could pass. The cache also omits fitted scaler/PCA objects. This is a reproducibility weakness, not evidence that current caches are mismatched.

## 2. What Stage A actually established

The current small-test suite passes **26/26**, including finite-difference gradients, frame masses, permutation restoration/equivariance, cosine convention, window broadcasting and parity with the local upstream ASOT reference. The maximum responsibility error in that parity check is **1.351e-11**. These are local correctness checks, not validation of the full learning procedure or its statistical assumptions.

The prior log-floor defect in v1.3 is addressed by the current log-domain implementation. Do not attribute v1.3.1's remaining poor scores to that already-corrected defect without new evidence. Final fixed-parameter convergence also does not establish outer alternating convergence or global optimality.

Primary T, pooled F1@50, mean over three seeds (SDs retained in `verification.json` and the original report):

| Dataset / normalization | A0 | A0 + filter | A1 | A1-KC | A-cont | Released SMQ |
|---|---:|---:|---:|---:|---:|---:|
| HuGaDB / off | 2.66 | 6.75 | 3.73 | 4.80 | 4.85 | 24.30 |
| HuGaDB / on | 2.55 | 5.67 | 6.74 | 3.17 | 4.93 | 24.30 |
| LARa / off | 6.08 | 13.05 | 9.86 | 11.09 | 17.21 | 16.39 |
| LARa / on | 4.77 | 13.84 | 7.78 | 8.36 | 17.05 | 16.39 |

The plan records HuGaDB as 10 classes, 18 subjects and inertial input, and LARa as 8 classes, 16 subjects and optical motion-capture input. These are separate dataset comparisons, not a controlled modality or subject-count experiment. All frozen-encoder results remain transductive.

**Observed symptom:** HuGaDB A1 predicts median runs of one window (0.25 s), with roughly 7.6–8.6 times the ground-truth run count. The segmentation deficit and fragmentation are real properties of the saved predictions. In the raw HuGaDB A1 cells, every final T recording solve reports convergence despite very poor F1; simply increasing the final solver cap is therefore not a supported explanation for those cells.

**Unresolved cause:** the plan's “Diagnosed mechanism” passage goes too far when it attributes fragmentation specifically to a weak temporal kernel and noisy emissions. E changes four coefficients together. In particular, `a/epsilon` is 10 under both T and E, while `beta/epsilon` rises from about 4.29 to 15 and `lambda/epsilon` falls from about 0.71 to 0.25. E is not an isolated test of one coefficient. Its large HuGaDB improvement occurs in the continuous arm; that does not establish the same mechanism for categorical A1.

A1's model is a code-to-state emission table plus local disagreement regularization and marginal/entropy terms. It has no learned ordered action grammar, transition matrix or duration model. Its failure limits this particular decoder at the declared settings; it does not test every way of learning action sequences from fine codes. Oracle-boundary information results also do not guarantee free-boundary segmentation.

## 3. What Stage B actually established

The current small-test suite passes **15/15**, covering masking, padding, stitching, reload inference, process-independent seeds, step limits and a small learning test. Passing them does not establish optimization quality on the real or synthetic task.

`stage_b_001` hit the 50-epoch cap in all 12 cells. Raising that cap was a reasonable optimization diagnostic. In `stage_b_002`, all 12 cells instead stopped by patience after **60–127 epochs**. Their best checkpoints were at zero-based epochs 51–118. Increasing the cap beyond 500 would not affect this run unless another stopping setting also changed.

Validation masked-code accuracy, three-seed means in percent:

| Dataset / normalization | Model | Neighbor-copy | Difference, points |
|---|---:|---:|---:|
| HuGaDB / off | 48.36 | 50.65 | -2.29 |
| HuGaDB / on | 24.18 | 26.64 | -2.46 |
| LARa / off | 43.57 | 52.87 | -9.30 |
| LARa / on | 25.85 | 29.94 | -4.09 |

One cell, raw HuGaDB seed 1538574472, passes the copy gate (50.79% versus 50.41%). Thus “every Stage B model failed” is not accurate for the longer rerun. The aggregate deficit remains. Baseline values can differ by seed because the KMeans vocabulary also changes.

Checkpoint histories show substantial loss reduction. For example, raw LARa seed 111 moves from training/validation CE 5.996/5.709 to 2.014/2.520 at the last epoch. This does not look like a completely disconnected loss or absent learning. Training losses use dropout and changing masks, while validation uses evaluation mode and fixed masks, so their gap is not a matched generalization measurement; the study's fixed-mask training probe is the appropriate additional diagnostic.

### The optimization study is incomplete and changes more than optimization

Available `optstudy_001/phase0.json`:

- All four variants memorize 16 chunks: 99.51–100% validation accuracy on those same chunks. This makes a universally broken forward/loss/backpropagation path unlikely; it does not establish generalization.
- On independent synthetic sequences with mean repeated-code run length 2, all four variants score about 0.02–0.24% against copy's 19.25%.
- At mean run length 4, the baseline scores 1.86% and the best tested variant 20.42%, against copy's 46.97%.
- All these synthetic fits stop through patience after 1100–2600 steps. The current source has a separate 15,000-step `phase 0c`, but its result artifact is absent. `--phase all` currently executes phases 0 and 1, not 0c.

This establishes difficulty learning a transferable local-copy rule under those settings. It does not distinguish insufficient optimization time, early stopping, positional inductive bias, code-identity learning and finite-data generalization. The missing longer-run diagnostic cannot be described as completed evidence.

The proposed 32-configuration real-data sweep changes learning rate/schedule, dropout, LayerNorm placement and output/input weight tying. The separately added ALiBi option changes positional inductive bias. These are architecture and regularization factors as well as optimizer settings. A successful variant would not, on its own, prove that the original optimizer was the cause. The internal tuning split correctly excludes the official contextualizer-validation recordings; preserve that separation.

### The acceptance gate changes the research question

The current plan says both that Stage C is the downstream usefulness test and that a model below neighbor-copy “cannot test whether context helps.” These statements conflict. Copy accuracy is a useful diagnostic of fine-code reconstruction; it is neither a necessary nor sufficient condition for recovering coarse action segments. Copying can reward retention of nuisance-specific code identity. Conversely, low code accuracy may reflect genuinely poor contextual learning. Segmentation has to be measured to distinguish those possibilities.

The copy gate was introduced after the first Stage B outcome, although before its rerun. Describe it as a revised gate, not an original pre-experiment requirement. No Stage C result was located in the reviewed artifacts, so the project has not yet measured failure of the complete contextual-segmentation pipeline.

## 4. Minimal corrective sequence

1. Fix the Stage A full-iterate objective and add its regression test. Retain the existing budgeted scores; flag their stopping traces. Check the effect on stopping and predictions before attributing the main score deficit to this bug.
2. Freeze Stage B's mask recipe, input/split hashes, configuration and resume guards under a new identity. Keep smoke runs separate. This is required before interpreting further comparisons as matched.
3. Complete one bounded diagnostic of the unchanged model on the synthetic task with a declared step budget and no premature patience stop; save learning curves, fixed-mask training probes, validation probes and gradient/update diagnostics. The source's longer-run experiment is relevant, but distinguish its unchanged-model result from tied-head/ALiBi variants. Avoid treating the full 32-by-4 sweep as the first diagnosis.
4. Treat copy accuracy as a reported diagnostic. Evaluate downstream segmentation through the already-planned Stage C comparison if proceeding with that scientific question; label any architecture-tuned variant as a new arm. Do not discard unfavorable seeds or configurations based on the copy gate and then report the surviving seeds as the original protocol.
5. Revise the plan's causal language: report fragmentation and reconstruction deficits as observations, and state competing explanations as open questions. Preserve the objective of testing coarse action structure learned from fine discrete sequences.

No expensive jobs, source-code fixes, protocol changes or replacements of prior reports were performed in this audit. The report identifies what should be repaired and what still needs a measurement; it does not claim that a particular repair will rescue the research result.
