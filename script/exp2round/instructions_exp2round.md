Brief: does a larger unit vocabulary make action identity recoverable?
Context

script/exp2round/a_unit_histogram.py already computes, on HuGaDB with the released checkpoint:

Arm	MoF	F1@50
SMQ as-is	41.99	24.30
GT boundaries + majority SMQ code	44.85	41.74
GT boundaries + SMQ-code histogram, re-clustered to C	46.42	43.98
Predicted boundaries + GT labels	78.78	79.17

The histogram arm inherits SMQ's vocabulary, so it only has K=C=10 bins and about 5 patches per segment. It is near-degenerate. This task replaces the vocabulary with an independently fitted one and sweeps its size. The question: with perfect boundaries, does a richer vocabulary of shorter units make the action recoverable, or is the information not in the representation?

Do not retrain SMQ. Do not modify existing scripts or result files. New code under script/exp2round/, new results under results/exp2round/.

Step 1 — Dump latents

Write script/exp2round/dump_latents.py, modelled on script/repro/dump_predictions.py.

For each sequence, save the frame-level encoder output before quantisation, shape (T, V, D), plus gt, the cached predicted code IDs, names, patch_size, num_actions.
Use the released checkpoint models/{dataset}/epoch-30.model, the same one behind the table above.
Gate 0. Re-form 1-second patches from the dumped latents, quantise them with that checkpoint's codebook, and compare against the cached predictions. Agreement must be ≥ 0.999. If not, stop and report the mismatch. Do not proceed.
Step 2 — Build vocabularies

For each window length W ∈ {patch, patch/2, patch/4} (HuGaDB: 60, 30, 15 frames):

Form one vector per non-overlapping window by flattening (W, V, D). Drop incomplete windows at sequence ends.
Standardise per feature, then PCA to 64 dimensions. Record the retained variance ratio.
Fit k-means for K_u ∈ {10, 20, 50, 100, 500, 1000}, n_init=5, three seeds. Use MiniBatchKMeans for K_u ≥ 500.
Also run K_u = 100 at W = patch without PCA, to confirm PCA isn't driving the result.
Record the mean number of units per GT segment for each W. This is the diagnostic that explains the curve's shape.
Step 3 — Score

For every (W, K_u, seed) cell, reuse the existing pipeline: histogram of unit IDs over each GT segment, relative counts, square root, k-means into C = num_actions clusters, global_map, score. Never Hungarian-match the units, only the final C clusters. Assert len(pred) == len(gt) per sequence.

Required controls, each scored the same way:

Majority unit per GT segment instead of the histogram. Separates a richer vocabulary from a richer readout.
Continuous reference: mean PCA-64 latent per GT segment, clustered into C. This is what the representation supports without discretisation.
Shuffled units: reassign each window a unit drawn from the empirical unit distribution, destroying within-segment structure.
Length-weighted segment clustering (sample_weight = segment length), at the best K_u only.
Step 4 — Nuisance check

Derive subject IDs from filenames (HuGaDB ..._01_00 → subject 01; LARa L02_S15_R04 → S15). BABEL has no participant IDs — skip it there and say so.

For each cell report NMI(units, action) and NMI(units, subject) at window level, and NMI(segment clusters, subject). If subject NMI is comparable to action NMI, the units are tracking who is moving.

Step 5 — Datasets and checkpoints
HuGaDB and LARa, released checkpoints. These are the primary runs.
If and only if Gate 2 below passes, repeat the two best K_u cells on the three T1 K=C seeds (models/exp/T1/{dataset}/K{K}_s{seed}/...; latents may already exist in results/exp/dumps/T1/). Recompute the D7 reference per seed rather than reusing the released one.
BABEL-1 last, released checkpoint only. Never mix retrained and released BABEL numbers in one table.
Gates
Gate 1. At W = patch, K_u = 10, HuGaDB must score within about 1.5 MoF of 46.42. If not, there's a wiring bug. Stop and report.
Gate 2. The best cell must beat the D7 majority-code reference (44.85 HuGaDB, 41.66 LARa) by at least 5 MoF on both datasets. If it doesn't, stop and report a negative result. Do not tune further, do not try more K_u values, do not change the distance or the weighting to rescue it.

A negative result here is a valid and useful outcome. Report it as such.

Outputs
results/exp2round/units_sweep.csv: one row per (dataset, checkpoint, seed, window, K_u, readout) with MoF, Edit, F1@10/25/50, units per segment, NMI action, NMI subject.
One JSON per dataset holding the D7 reference rows and the full sweep.
A plot of MoF against K_u, one line per window, with horizontal reference lines at the majority-code and GT-label numbers.
results/exp2round/REPORT.md: the table, the curve, control results, gate status, and separate lists of what is supported and what is not. Mark anything planned but not run.
Rules
CPU only. Cache latents to disk so the sweep doesn't re-run the encoder.
Fixed seeds, deterministic, log runtimes.
Report the whole curve. Do not pick a K_u by its score and present that as the headline.
No ground-truth labels anywhere except the GT segment boundaries, the final Hungarian matching and the NMI diagnostics.