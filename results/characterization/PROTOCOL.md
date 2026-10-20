# Frozen protocol: what do SMQ codes represent?

Recorded 21 September 2026, before the characterization outcomes. Scope: existing HuGaDB and LARa representations only; no retraining, model selection, new architecture, or downstream task.

## Models, data and exposure

Use all three T1 K=C epoch-30 checkpoints: seeds 1538574472, 111, 222; HuGaDB K10/W60 and LARa K8/W50. Checkpoint paths are `models/exp/T1/{dataset}/K{K}_s{seed}/{dataset}/epoch-30.model`; exports are `results/exp/dumps/T1/{dataset}/K{K}_s{seed}.npz`. Freeze paths, hashes, configuration and splits in `manifest.json`.

All these SMQ models trained on the full respective benchmark. “Evaluation” below means held out from this study's scaling and static-posture control fitting, **not unseen by SMQ**. No claim about genuinely training-unseen people will be made. Code IDs are specific to a checkpoint and will not be equated across seeds.

Each latent frame has a 29-frame temporal receptive field: two TCN stages, each with three kernel-3 dilated convolutions at dilations 1,2,4. Quantized patch p represents preprocessed frames [pW,(p+1)W), but assignment depends on [pW−14,(p+1)W+14). Select complete odd-indexed patches p=1,3,5,... with the entire context inside the recording. Their dependency intervals do not overlap. LARa native preprocessing strides original 200-Hz rows by four; preprocessed index t corresponds to original row 4t. HuGaDB uses native sensor rows at the pipeline's nominal 60 Hz.

LARa channels 3:6 are positional coordinates, already made relative to hip index 21; exclude the identically-zero root from distance calculations. Preserve the original six channels when computing model assignments. Do not rotate, mirror, rescale model input, or remove root orientation.

**User-approved scope adjustment:** HuGaDB supplies IMU signals, not joint positions. Its posture-controlled joint-movement test is unavailable. Run native signal stability and signal exemplars only; never present sensor derivatives as joint velocity or claim to control skeletal posture.

## Calibration and evaluation partition

Sort participant IDs by SHA256 of `smq-characterization-20260921/{dataset}/{participant}`. First floor(N/3) participants are calibration (5 LARa, 6 HuGaDB); the rest are evaluation. Fit descriptor scaling and posture k-means on calibration recordings only. All evaluation recordings contribute coverage and eligibility counts.

For primary inference, pair consecutive evaluation participants in this same fixed order into disjoint dyads (five LARa dyads); a final unpaired participant contributes stability/exemplars but not primary matching. Each person belongs to exactly one primary dyad. This makes the participant-dyad bootstrap account for dependence from both endpoints rather than pretending pair combinations are independent.

## Descriptors and metric checks

LARa posture P is the mean root-relative xyz configuration over the full one-second patch (63 coordinates excluding the root). Fit coordinate SD to calibration patch means; floor each SD at 5% of the median nonzero coordinate SD. Mean centering and scaling use calibration only. Posture distance is RMS standardized coordinate difference.

Ordered displacement is xyz(t)−xyz(0), divided by those same coordinate SDs. Velocity is forward frame difference ×50 fps, with the first difference repeated at frame zero, divided by the same SDs. Both retain temporal order, sign and articulated joint identity. Fixed-length primary distances are RMS displacement difference Dd and RMS velocity difference Dv over all frames and coordinates. Normalize each distance by its median over 5,000 fixed-RNG different-participant calibration patch pairs. Composite D = 0.5(Dd/median_d + Dv/median_v). Report both component distances and the composite separately.

Sensitivity: endpoint-constrained, forward-only DTW, Sakoe–Chiba band ±5 LARa frames (0.1 s), local cost 0.5(mean squared normalized displacement difference + mean squared normalized velocity difference). Report square root of mean cost along the minimum-total-cost path. No time reversal or rotation is allowed.

Before reading grouping effects, validate synthetic trajectories with equal mean posture: forward versus reverse movement, and the same movement under modest monotonic speed variation. Both distances must give identical motion zero and speed variation less distance than reversed direction. Failures block interpretation until a documented metric correction passes.

## Primary matched comparison

Use at most 12 evenly spaced eligible anchor patches per evaluation recording. Donor candidates are all eligible nonoverlapping patches from the other participant of its fixed dyad. Exclude already-used patches within each model/grouping/subset analysis. Traverse anchors in a fixed hash order independent of code or action values.

For anchor a, select the nearest-posture available same-code donor b, requiring posture RMS(a,b) ≤0.50. Then select a different-code control c **from the same donor recording as b**, requiring RMS(b,c) ≤0.25, RMS(a,c) ≤0.50, and |RMS(a,c)−RMS(a,b)| ≤0.05. Minimize RMS(b,c), breaking ties by recording/frame order. If the nearest same-code donor has no control, mark the anchor unmatched; do not search farther donors or relax calipers. Every accepted triple uses three distinct, nonoverlapping patches; no patch is reused within that analysis.

Primary effect per triple: D(a,c)−D(a,b); positive means greater same-code movement similarity. Report same-code distance, different-code control distance, paired difference, and component results. Report anchor and matched counts, unique recordings/people, exclusions, posture-balance diagnostics and per-code eligibility/coverage. Results describe the matched population, not every occurrence of a code.

Run the same prespecified matching procedure separately (1) without action restrictions, and (2) requiring both donor and control majority action labels to differ from the anchor. Majority action labels only restrict this secondary subset, not matching outcome or primary movement distance. Report action purity; do not hide low coverage.

Control: calibration-fitted MiniBatchKMeans on standardized mean posture, K=8, random states equal to the three model seeds, n_init=10, max_iter=200, batch_size=2048. Apply unchanged to evaluation patches and repeat all matching. This is a static-posture reference. Compare coverage as well as effect; SMQ and k-means matched populations may differ.

## Uncertainty and nuisance interpretation

Compute frequency-weighted (accepted triple weighted) and code-balanced (equal weight over represented codes) summaries, with per-code results. Resample the five disjoint participant dyads with replacement, 2,000 fixed-RNG bootstrap replicates, keeping all recordings/triples within a dyad together. Report percentile 95% intervals and the number of dyads contributing. Per-code inference supported by fewer than three dyads is explicitly weak. Do not bootstrap individual patches or all combinations independently. Report across-seed mean/SD separately from the within-seed intervals; do not average code IDs between models.

Recordings are nested within participant-dyad clusters. Fixed donor-recording matching controls participant and recording identity between the two alternatives. It does not control all unmeasured context. A cross-person result on training-seen people is not unseen-person generalization. Weak residual movement effects, posture-control effects, and perturbation sensitivity may coexist; do not force a single explanation.

## Stability

Use the same evaluation pool, selecting at most six evenly spaced eligible patches per recording, fixed before perturbation outcomes. Re-encode exact context windows with native preprocessing and frozen models in eval mode. Verify unperturbed crop assignments equal cached full-recording codes; do not proceed on a mismatch without investigation.

Temporal shift: +2 native preprocessed frames (40 ms LARa; nominal 33.3 ms HuGaDB). Shift represented raw interval and context together; compare code IDs without Hungarian remapping. Report descriptor changes both on complete relative-time windows and on their explicitly aligned common raw frames; overlapping raw descriptors should agree exactly. The shifted windows are paired perturbations, not new independent observations.

Pose noise (LARa): independent zero-mean Gaussian xyz perturbations before root subtraction at 0.1% and 0.5% of calibration RMS root-relative position scale; subtract the perturbed root, keep root zero and first three channels unchanged. This preserves native root-relative convention without changing orientations. HuGaDB: signal noise at 0.1% and 0.5% of calibration per-channel/sensor SD; explicitly not pose noise. Fixed random noise by recording/patch/amplitude, shared across model seeds. Report posture and ordered movement-descriptor changes (signal analogues for HuGaDB), code agreement, original code frequencies and per-code agreement. Participant-cluster bootstrap intervals (2,000 draws) and seed spread are separate. Noise changes velocity more than position; interpret sensitivity conditional on both changes.

## Exemplars and stopping rule

Export all codes of every selected checkpoint. Select up to six occurrences per code by fixed SHA256 order; first maximize different participants, then different recordings, then fill remaining slots. Use evaluation patches only. Record available/selected counts, IDs and frame bounds, including rare or absent codes. LARa displays full root-relative joint trajectories with temporal direction; HuGaDB displays complete ordered sensor traces, never fabricated skeletons. Descriptions of a fixed representative subset (code 0, most frequent, least frequent nonempty code of default seed) are qualitative, not validated semantic labels.

Deliver protocol+manifest, primary table for both datasets (HuGaDB marked unavailable), per-code/matching coverage tables, compact stability results, exemplar montage index, and a research verdict separating supported/unsupported claims. Stop after characterization; no larger K, loss changes, hierarchy or recognition experiments.
