# optstudy_002: code and results analysis

Reviewed 26 September 2026. No training launched or experiment source changed. Derived summaries are in [summary.json](summary.json); all original study artifacts remain unchanged.

## Finding

This completed sweep provides consistent evidence that the implemented symmetric ALiBi positional variant improves masked-code prediction over the sinusoidal variant under the tested training procedure. It narrows the earlier problem to a model/training-design issue more than a simple shortage of epochs. It does not establish that optimization is fully solved, that all remaining error is irreducible, or that contextual embeddings improve action segmentation.

The declared selection correctly chooses **postLN_cosine_wu500_lr0.0003_alibi**: post-LayerNorm, learning rate 0.0003, 500-step warmup followed by cosine decay, untied output head, dropout 0.1, symmetric relative-distance attention bias. Width 128, four layers, four heads and the masked-code objective are unchanged. The budget is at most 5000 optimizer steps, with tuning evaluation every 100 steps and patience 1000 steps.

## Study integrity and current implementation

- All **128/128** planned fits exist: 32 configurations across HuGaDB/LARa and raw/per-recording-normalized latents, clustering/training seed 111. Every expected cell has each configuration once; there are 128 learning curves.
- All three saved source hashes match current `stageb_train.py`, `stageb_model.py` and `stageb_data.py`. Every effective configuration hash matches the current grid. Recomputing selection exactly reproduces `selection.json`.
- Every curve's minimum tuning loss and corresponding best step agree with its CSV row, as does the final executed step. Copy accuracy is identical across all configurations within each dataset/normalization cell, as expected under fixed masks and code targets.
- The revised grid fixes dropout at 0.1 and varies position mode instead. The amendment is recorded in DESIGN.json. This is an architecture-and-training study, not solely a change in optimization settings.
- The earlier process-dependent mask seed, smoke/full-budget identity collision, dropped tail minibatch and unbatched training-time validation have been addressed in current code. Source/configuration guards are present. The 16 current Stage B CPU tests pass, including production-size train/eval consistency and positional sensitivity.
- The ALiBi implementation explicitly executes the encoder blocks with additive attention bias instead of the earlier encoder fast path. The earlier invalid synthetic ALiBi results are separately marked in `optstudy_001`. The new checks support correctness in the tested CPU environment; this audit did not rerun training or independently replay the historical GPU execution.
- Study fitting and tuning use whole recordings drawn from the official training recordings; official contextualizer-validation recordings are excluded. The frozen tokenizer is pooled, so these are not fully inductive unseen-data results.

## Selected model's tuning results

These are **single-seed tuning results used for selection**, not independent final validation or action-segmentation scores. "Raw" means no per-recording latent normalization; it does not mean raw sensor input.

| Cell | Matched sinusoidal accuracy | Selected ALiBi accuracy | Neighbor-copy | ALiBi minus copy, points | Selected CE, nats |
|---|---:|---:|---:|---:|---:|
| HuGaDB / raw | 25.42% | 29.31% | 23.90% | +5.41 | 2.7434 |
| LARa / raw | 43.06% | 48.48% | 49.77% | -1.29 | 2.1293 |
| HuGaDB / normalized | 18.10% | 24.28% | 19.07% | +5.21 | 3.1776 |
| LARa / normalized | 21.48% | 25.47% | 25.97% | -0.50 | 3.5235 |

The mean tuning CE is **2.8934604**. The next configuration scores **2.8951190**, a difference of only **0.001659 nats**. Respect the declared choice, but do not infer a robust advantage of this exact learning rate/schedule/normalization combination from one split and seed.

None of the 32 configurations beats copy on either LARa condition. On HuGaDB, 28/32 raw and 26/32 normalized configurations do. Thus `any_config_beats_copy_all_cells=false` is arithmetically correct. It is not equivalent to every model failing to learn useful context.

The original v1.5-default-shaped configuration, retrained within this study using current code and the step budget, scores 25.07%, 44.25%, 18.89% and 21.95%, respectively in the table's order. Comparing the selected configuration with it gives +4.24, +4.23, +5.38 and +3.52 accuracy points. The matched sinusoidal column isolates the position-mode factor more closely because the learning-rate schedule is also shared.

HuGaDB and LARa results are separate conditions: the plan records 10 classes/18 subjects/wearable IMU for HuGaDB and 8 classes/16 subjects/optical motion capture for LARa. These differences prevent a controlled cross-dataset inference about modality or subject count.

## What the factorial comparisons reveal

Pair each ALiBi fit with its sinusoidal counterpart at identical LayerNorm placement, head tying, learning rate and schedule. Across all **64 pairs**, ALiBi has both lower tuning CE and higher accuracy.

| Cell | Pairs | Mean accuracy gain, points | Mean CE change, nats |
|---|---:|---:|---:|
| HuGaDB / raw | 16 | +4.23 | -0.2851 |
| LARa / raw | 16 | +4.21 | -0.2015 |
| HuGaDB / normalized | 16 | +4.38 | -0.3227 |
| LARa / normalized | 16 | +3.44 | -0.2148 |

These are paired configuration contrasts, not 64 independent dataset replications. The implemented position factor changes both positional representation and the attention execution path; the numerical-equivalence tests reduce concern about an implementation artifact but do not make this an optimizer-only comparison. Best checkpoints and early-stopping steps can also differ.

Other average paired CE changes across the grid:

| Factor change | HuGaDB raw | LARa raw | HuGaDB norm | LARa norm |
|---|---:|---:|---:|---:|
| Tie the output/input embeddings | +0.0997 | +0.0766 | +0.1476 | +0.1793 |
| Post-LN to pre-LN | -0.0057 | -0.0142 | -0.0062 | -0.0250 |
| Learning rate 0.0003 to 0.001 | -0.0897 | -0.0877 | -0.0806 | -0.1391 |
| Constant LR to warmup + cosine | +0.0117 | +0.0152 | +0.0238 | +0.0384 |

Negative is lower CE. These averages include interactions and unequal effective training lengths; they do not override the selected configuration. In particular, the real-data evidence does not justify carrying all the synthetic-study modifications forward together: head tying worsens average tuning CE, and the selected head is untied. Warmup and cosine are bundled, so their individual effects are not identified.

## Where the LARa deficit occurs

Depth is the distance from a masked token to its nearest visible token. Depth 1 is at a mask edge; depth 4+ is at least four code positions from visible input. For the selected model:

| Cell | Depth 1 model / copy | Depth 4+ model / copy |
|---|---:|---:|
| HuGaDB / raw | 31.75% / 22.22% | 27.24% / 24.90% |
| LARa / raw | 62.27% / 69.40% | 39.13% / 36.72% |
| HuGaDB / normalized | 26.72% / 19.18% | 21.67% / 19.32% |
| LARa / normalized | 38.58% / 44.33% | 17.02% / 15.97% |

On LARa, the model loses 7.12 and 5.75 points at mask edges but gains 2.40 and 1.05 points at depth 4+. The aggregate shortfall therefore hides a tradeoff between adjacent-code copying and prediction farther inside masked spans. Depth 4+ is still only about one second or more from visible input at these code windows; it is not evidence of learned long action grammar. Nor does accuracy above copy identify which temporal dependencies were learned.

## Learning curves and optimization status

![Matched tuning learning curves](matched_learning_curves.png)

**73/128** fits reach the 5000-step cap; **55/128** stop by patience. In 24 fits, the best recorded checkpoint is at step 5000. It would be incorrect to call the whole sweep converged.

Selected-model details:

| Cell | Best step | Executed steps | Stop reason | Fixed-mask training-probe accuracy | Tuning accuracy |
|---|---:|---:|---|---:|---:|
| HuGaDB / raw | 3100 | 4100 | Patience | 44.44% | 29.31% |
| LARa / raw | 4300 | 5000 | Step cap | 51.26% | 48.48% |
| HuGaDB / normalized | 4500 | 5000 | Step cap | 30.56% | 24.28% |
| LARa / normalized | 5000 | 5000 | Step cap | 31.68% | 25.47% |

Normalized LARa's tuning CE still improves from 3.5365 at step 4000 to 3.5235 at 5000. Raw LARa is nearly flat by then: best 2.1293 at 4300 versus 2.1298 at 5000. This makes remaining budget sensitivity plausible, especially for normalized LARa, without establishing that more training will cross the copy baseline.

The fixed-mask training probes exceed copy in every selected-model cell. Their training-versus-tuning gaps also caution against treating all remaining error as underfitting. Raw HuGaDB's copy baseline itself differs strongly between training probe (37.41%) and tuning recordings (23.90%), indicating that the two populations differ in local predictability. Thus the accuracy gap is not a clean measurement of memorization alone. Probes are subsets with another mask seed, not the full training population.

Logged gradient norms are pre-clipping minibatch norms sampled at evaluations, not full-objective stationarity certificates. They cannot establish convergence. Historical `stage_b_002` used a different evaluation population, mask realization and budget; its absolute accuracies should not be subtracted from these tuning scores to estimate improvement.

## Remaining code and reporting issues

1. **The selected model is not connected to the normal execution path.** `stageb_optstudy.py` discards each returned model and saves only summary rows and learning curves. There are no study checkpoints/embeddings to pass to Stage C. `run_stage_b.py:120` calls `train` without supplying the selected configuration, so the default remains the sinusoidal model. Running that launcher unchanged does not implement this study's selected arm. A final fit needs an explicit selected configuration and a recorded final training/checkpoint-selection protocol.

2. **The study's stop rule is stale.** DESIGN.json and the optimization-study module still recommend stopping Stage B if no configuration beats copy everywhere. The current v1.6 plan explicitly removed copy accuracy as a Stage C gate. Preserve the historical design and report its boolean result, but do not silently let that stale rule replace the revised research question. This remains a code-reconstruction diagnostic, not an action-segmentation acceptance test.

3. **The selected positional bias is symmetric in time.** It uses `-slope * abs(i-j)` and removes absolute sinusoidal positions. Reversing a full unpadded chunk therefore reverses its outputs: the architecture cannot distinguish left from right purely through positional information. A CPU check gives maximum hidden-state reversal error 1.19e-6. This is not a coding defect; it is an important limitation when describing the module as learning ordered action sequences. It can represent distance-sensitive context, but not give different center representations to exactly time-reversed neighborhoods. Chunking and later processing must be considered separately from this encoder property.

4. **Provenance is improved but incomplete.** DESIGN hashes cover the three core training files, not the study orchestrator, tokenizer code, split contents, input cache checksums or device/library environment. `select` still counts rows rather than validating the precise unique cell set. Independent checks found complete, unique coverage here, so that weakness does not invalidate this selection. Future resume/selection should validate these identities and persist actual study-train/tune memberships. The curve files do not retain all depth-bin counts, best model weights or per-position predictions.

5. **Documentation overstates or lags the evidence.** README_STAGE_B still says the sweep has not run and describes copy accuracy as a synthetic ceiling. It is a baseline, not an established information-theoretic ceiling. DESIGN's motivating 0.464-versus-0.482 synthetic result is not reproduced by the saved `optstudy_001/phase0c.json`, whose mean-run-4 ALiBi result is 0.4042 versus 0.4697. That may refer to another diagnostic, but its artifact/protocol must be identified before citing it. The sweep's own matched comparisons stand independently of that missing provenance.

## What this leaves established

The sweep supports retaining the selected relative-distance contextualizer as a declared candidate for the planned downstream comparison. It does not support the earlier broad conclusion that Stage B cannot learn context. It also does not establish a unique optimizer failure: the strongest consistent contrast changes the positional architecture, and several fits remain budget-limited.

The next decisive measurement is the planned final validation and Stage C segmentation comparison with the selected configuration explicitly frozen and recorded. That is preferable to interpreting an all-cells copy threshold as the project's outcome or selecting further architectures from action scores. Any final three-seed result requires those actual fits; this study contains only seed 111.
