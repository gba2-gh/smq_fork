# Encoder sweep: evidence and project implications

28 September 2026. Read-only analysis of encoder_sweep_001, its code, source checkpoint identities and prior-pilot protocol. No training or implementation changes were performed.

## Decision

The epoch-30 SMQ VQ encoder should no longer be the default representation for the proposed pipeline. Raw-input tokenization outperforms it on all three matched downstream seeds in both contextual F1@50 and MoF. The raw and untrained controls should have been established before treating further development on the frozen SMQ encoder as the preferred research direction.

The earlier oracle-prototype gain establishes conditional headroom with a particular representation and decoder. It cannot establish that the representation is a good choice compared with raw inputs, that unsupervised training can realize that gain, or that a decoder modification is the highest-value next contribution.

This conclusion is specific to the tested HuGaDB VQ training trajectory and downstream pipeline. It is not a demonstration that every encoder, every reconstruction objective, or the released SMQ checkpoint is harmful.

## What was compared

Source: fsq_pilot_002, HuGaDB, autoencoder seed 111. Representations: raw input, the shared random encoder initialization, and VQ checkpoints at epochs 10, 20 and 30. No FSQ encoder checkpoint was included in this sweep.

All representations use 15-frame patches followed by standardization, PCA-64 and K-means-512, then either categorical OT or a fresh masked contextualizer plus contextual ASOT. Three downstream seeds are present, despite the script default listing five. Each seed changes the fitting-window sample, randomized PCA and K-means together, as well as contextualizer/OT initialization. The same seed uses matched recording/patch selection across representations.

There are 120 unique score rows: five representations x three seeds x two methods x four final-solver endpoints. All 15 job artifacts are present, not_run.csv is empty and no failure rows occur. All four saved checkpoint hashes match the source checkpoint files.

## Final contextual results

At the 4,000-step final-solver endpoint; mean and population SD over the three downstream seeds. These are initialization/subsampling sensitivities conditional on one trained encoder trajectory.

| Input to tokenization | F1@50 | MoF | Edit |
|---|---:|---:|---:|
| Raw input | 46.88 +/- 2.87 | 47.58 +/- 2.51 | 42.36 +/- 2.12 |
| Untrained encoder, epoch 0 | 45.32 +/- 3.75 | 49.28 +/- 1.95 | 44.75 +/- 2.23 |
| VQ encoder, epoch 10 | 47.48 +/- 1.26 | 48.17 +/- 1.73 | 44.22 +/- 1.84 |
| VQ encoder, epoch 20 | 41.93 +/- 1.05 | 45.81 +/- 1.34 | 40.92 +/- 1.53 |
| VQ encoder, epoch 30 | 39.61 +/- 3.07 | 44.15 +/- 1.21 | 39.70 +/- 2.27 |

Epoch-30 minus raw differences:

| Downstream seed | F1@50 | MoF |
|---|---:|---:|
| 111 | -6.94 | -4.71 |
| 222 | -7.61 | -3.95 |
| 1538574472 | -7.24 | -1.61 |
| Mean | -7.26 | -3.42 |

This is a consistent measured disadvantage at the selected training endpoint. Epoch 30 also loses 5.71 F1 points and 5.13 MoF points relative to the untrained encoder on the means.

Epoch 10 is only +0.60 F1 and +0.59 MoF above raw on the means, with mixed paired results. It is not evidence of a reliable advantage. Selecting epoch 10 now by benchmark action scores would be post-hoc tuning; a future checkpoint-selection rule needs independent development data or a validated label-free criterion.

The untrained encoder has slightly lower F1 but higher mean MoF and Edit than raw. Thus there is no total ordering of every representation on every metric. The strong conclusion concerns late training, not the mere presence of the TCN architecture.

## Information diagnostics qualify the mechanism

Mean empirical unit-action mutual information, in nats:

| Representation | I(unit; action) | I(unit; subject) | I(unit; subject given action) |
|---|---:|---:|---:|
| Raw | 1.3896 | 1.5985 | 1.6064 |
| Epoch 0 | 1.3861 | 1.2957 | 1.3215 |
| Epoch 10 | 1.4485 | 1.3894 | 1.4168 |
| Epoch 20 | 1.4403 | 1.4623 | 1.4865 |
| Epoch 30 | 1.4467 | 1.4164 | 1.4415 |

Each observation is a window with its majority action label; these are empirical plug-in diagnostics, without permutation references or temporal sampling uncertainty. They are not measures of boundary quality or all information in the continuous features.

The epoch-30 units have slightly higher measured action information than raw units despite worse segmentation. They also have lower measured subject information than raw. Consequently, the sweep does not support a simple explanation that the encoder necessarily erases action identity or that subject encoding alone causes the performance gap. Training may change temporal structure, geometry or the suitability of the tokens for the contextualizer and action-discovery procedure; this sweep does not isolate which mechanism is responsible.

PCA-64 retains about 71.2% of standardized raw variance, 79.9% of untrained-encoder variance and 97.9% at epoch 30. This describes different feature geometry, not a quality ranking. The raw pipeline wins despite lower explained variance.

## What remains useful in the pipeline

Raw tokens with categorical ASOT score 3.51 F1@50 and 34.75 MoF, versus 46.88 and 47.58 with the learned contextualizer and contextual ASOT. The high raw result is therefore **raw input -> K-means -> learned masked contextualizer -> ASOT**, not simply raw K-means plus unchanged categorical OT.

The contextualizer is itself a learned representation. The sweep supports removing the unjustified reconstruction-pretrained front end; it does not demonstrate that learned representations in general are detrimental. The contextual-versus-categorical comparison combines several changes and does not isolate temporal order, masking or model capacity individually.

## Novelty assessment

[HuBERT](https://arxiv.org/abs/2106.07447) uses offline clustering to provide masked-prediction targets while its model consumes continuous speech inputs; it does not implement ASOT action segmentation. Our token-input contextualizer is not an exact HuBERT reproduction, but the cluster-target/masked-prediction ingredients have clear precedent.

[ASOT](https://arxiv.org/abs/2404.01518) already supplies temporally consistent transport and uses transport-generated pseudo-labels for unsupervised learning. Adding an encoder, masked prediction, a quantizer, or end-to-end self-training does not on its own establish a new methodological contribution.

The current composition is a useful internal control and diagnostic platform, not the external performance or novelty target. These results do not yet establish the major architectural novelty or broad empirical findings needed for the intended paper.

Correction after reading the local D-CLOT paper (Bueno-Benito and Dimiccoli, 2026), Sections III and IV and Table II: [D-CLOT](https://arxiv.org/abs/2608.05877) is directly relevant prior work that the initial recommendation omitted. It builds on CLOT's frame/segment refinement, adds graph-constrained representation regularization, and explicitly refreshes action prototypes using either K-means or transport-weighted barycenters (D-CLOT_B). Representation-prototype consistency, prototype re-estimation and feedback between frame/segment learning and transport therefore cannot be treated as unoccupied novelty merely because our diagnostics reveal a discovery gap.

Table II reports activity-level F1 on 50Salads Eval of 53.6 for ASOT, 63.2 for CLOT and 72.1 for D-CLOT_B. These establish a stronger relevant comparator within that paper's protocol, not a directly transferable HuGaDB F1@50 target. The paper uses video features and per-video/activity-level matching; feature exposure, matching scope and metric definitions must be aligned before comparing to the motion pipeline. D-CLOT and D-CLOT_B belong in the external method comparison, alongside relevant skeleton methods for our selected datasets. A literature-wide novelty claim still requires a focused review of the proposed mechanism.

## Audit limits

- The sweep changes sampling and PCA randomness along with K-means across seeds. This differs from the plan's fixed fitting sample/PCA policy, but remains matched between representations within each seed. Report the actual procedure rather than calling the spread pure K-means sensitivity.
- Pilot results held K-means fixed across seeds; this sweep refits it. Do not treat the difference between the pilot's epoch-30 mean and this sweep's mean as a new training effect.
- Saved artifacts still lack predictions, contextualizer checkpoints and tokenizer transforms. Full independent rescoring is not possible from these files alone; the reported scores were audited for consistency and completeness, not rederived from predictions.
- At the final endpoint, 10,679 recording/method/seed solves are labeled converged and 241 capped. Outer-fit traces are absent. Epoch-30 seed 1538574472 loses 2.85 F1 points from 500 to 4,000 final steps; epoch-10 on that seed gains 0.71. Final-step sensitivity exists, but the raw-versus-epoch-30 F1 disadvantage is already present for all seeds at 500 steps too.
- One dataset, one encoder-training seed and one family of training objectives limit generalization. The released C-code encoder, MAE and contrastive encoders have not been compared in this sweep.
- Resume validates job completion by row count, not immutable artifact identities. Checkpoint hashes now provide useful provenance, but source/device/split identities and robust resume validation remain incomplete.

No observed implementation issue reverses the central matched comparison. These limitations should constrain the claims rather than serve as a reason to ignore the negative result.

## Reconsidered scope

1. Retire SMQ reconstruction pretraining as the default front end. Retain it as a historical or negative control; suspend further FSQ adapter work on this objective for now.
2. Freeze the raw-token contextual pipeline as an internal control that any added encoder must beat, preserving proper predictions, models and identities in the next run. It is not the external target for a paper: assess D-CLOT/D-CLOT_B and relevant skeleton methods under matched protocols. Do not redesign the internal control after inspecting evaluation labels.
3. Reopen representation learning as a primary research question. MAE or continuous-input masked cluster prediction would be meaningful controls, not automatic novel contributions. Avoid launching a broad architecture grid or combining new data, quantization, objective and decoder changes at once.
4. Choose one research hypothesis and its novelty before a major training commitment. An encoder-based method must demonstrate benefit over raw and untrained controls under the same decoder, including another dataset and a declared development protocol. If that benefit cannot be established, remove that component rather than preserving it for architectural appearance.
5. Keep two possible paper directions distinct: a new mechanism with replicated advantages, or a rigorous multi-method/multi-dataset investigation of reconstruction pretraining versus segmentation utility. The current one-dataset result is a starting observation for the latter, not yet a complete contribution. Neither path guarantees acceptance.

The appropriate decision is a scope reset, not another small repair to the same frozen-front-end pipeline. The existing experiments remain valuable baselines and evidence for rejecting an unsupported design choice.
