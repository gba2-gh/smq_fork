# Input-representation test (LARa)

6–7 October 2026 · Code: `script/input_representation/` · Follows the failed Milestone 1 gate ([results](../../results/compositional_dynamics/m1_lara_001/REPORT.md)).

## Question

Four decoders on fixed T2 codes give the same frame accuracy (MoF about 38–39 on LARa). Is the limit in **what the per-window input contains**, in **how the unsupervised decoder organises it**, or in the skeleton signal itself?

## Design (one dataset, declared before any scored run)

Three per-window inputs, each passed through the unchanged Phase 5 pipeline (standardize → PCA-64 → K-means-512, same seeds, same existing label-free primary split of 395 fitting / 44 evaluation recordings):

| input | content |
|---|---|
| `t2` | existing T2: per-window mean, sd, mean \|Δ\| of the 231 inter-joint distances. Reference; its B0 cells are read from the Milestone 1 run. |
| `t2_bodyscale` | T2 with every frame's distances divided by the recording's body-scale proxy (median over frames of the mean pairwise distance). Removes body size, a plausible identity carrier. |
| `t2_centered` | centroid-centred joint positions (divided by the same scale) plus centroid velocity; same window statistics. Restores orientation and global motion that distances discard. **Not heading-invariant**, so a win can come from orientation rather than posture. |

For each (input, seed 111/222/1538574472):

1. **Unsupervised decode:** the matched B0 of Milestone 1, fitted on the 395 fitting recordings and decoded on the 44 evaluation recordings, final-solver endpoint, global Hungarian mapping.
2. **Action probe (label-using upper bound, subject-grouped):** logistic regression from the window's PCA-64 feature to its majority action label, trained on 3 of 4 subject folds and tested on the held-out fold (mean over folds; length-weighted accuracy). Also a code-table probe (majority label per K-means code) and the majority-class baseline.
3. **Subject probe (identity left in the feature):** the same classifier predicting the subject, trained on a random half of the recordings and tested on the other half. Subject IDs are metadata, not action labels.

Probes use labels as diagnostics only. They never select a decoder setting or an input.

## Decision rules

Co-primary metrics are B0 MoF and F1@50 on the 44 evaluation recordings, paired by seed against `t2`.

1. **ADOPT_INPUT:** a new input beats `t2` by a mean of at least **+3 on both MoF and F1@50, with all three seeds positive on both**. It becomes the baseline any end-to-end encoder must beat. (If both pass, the larger MoF + F1@50 sum.)
2. Otherwise compute, for each input, the mean supervised action-probe accuracy minus its mean B0 MoF:
   - best gap **≥ 15 points → ORGANISATION_PROBLEM:** the information is present but unsupervised decoding does not organise it. An end-to-end encoder with an identity-breaking objective (subject-invariance, cross-subject positives, action-scale OT targets) is justified, evaluated against the frozen B0 baseline under a declared gate.
   - best gap **< 15 points → REPRESENTATION_LIMIT:** skeleton window features sit near what decoding recovers; the next step is a different information source (pretrained motion encoder, other modality), not another decoder.
3. Missing seeds or probes make the outcome INCONCLUSIVE.

The subject probe is reported without a rule. A new input that passes rule 1 while lowering the subject probe is evidence that identity was removed; one that passes with the subject probe unchanged would suggest orientation or global motion is the source.

## Limits

- The tokenizer is still fitted on all recordings (transductive); the evaluation recordings share subjects with the fitting set.
- The 15-point threshold is a declared judgement, not derived. Supervised and unsupervised accuracies are not the same quantity (Hungarian-mapped clusters versus a trained classifier), so the gap is an upper-bound comparison.
- Probe training is subsampled to 40,000 windows per fit with a fixed seed.
- Three seeds on one split are not confidence intervals.

## Run

```powershell
.\script\input_representation\run.ps1 -Preflight
.\script\input_representation\run.ps1 -Smoke -RunId irt_smoke_01
.\script\input_representation\run.ps1 -Run -RunId irt_lara_001 -Hours 3
```

Expected cost: new inputs need preparation (about 2–3 min) and a B0 decode (about 9 min) per seed, so about 1 h for 2 inputs × 3 seeds, plus probes (about 1–2 min each).
