Experiment Brief: Diagnostic Analysis of Discrete Codebooks for Unsupervised Skeleton Action Segmentation
Read this first

This brief is staged with hard stop conditions. Do not proceed to a later stage until the previous stage's result has been reported and reviewed. Each stage states explicitly what a negative result looks like and what to do when you get one.

The default action on an ambiguous or negative result is to stop and report, not to proceed anyway and not to tune until the result improves.

Several of these experiments are designed so that a negative result is as informative as a positive one. Do not treat a null result as a failure to be fixed.
Background and purpose

SMQ (Gökay et al., "Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation", ICCV 2025) learns a discrete codebook over skeleton sequences and evaluates it on unsupervised temporal action segmentation (TAS). HiST-VQ (Ahmed et al., 2026) extends it with a two-level hierarchical codebook and reports state of the art on the same benchmarks. HiST-VQ's code is not released; its acknowledgements state it builds on SMQ, whose code is public. SMQ is therefore our base.

Both papers evaluate with one protocol: global Hungarian matching between discovered codes and ground-truth action classes, then MoF, Edit score, and F1@{10,25,50}. That protocol answers exactly one question — does the discovered segmentation agree with this annotation, on this data, pooled over all subjects.

We are testing three things it cannot answer:

    Person invariance. Both datasets have multiple subjects (HuGaDB 18, LARa 14). The published protocol pools them. Nobody has held subjects out, so nobody knows whether a code means the same movement for two different people.
    Unit duration. A code corresponds to one patch, and the patch size is hand-set per dataset (60 / 50 / 30 frames for HuGaDB / LARa / BABEL — one second each). The metrics never inspect duration, so the relationship between a code and the actual duration structure of the data is unmeasured.
    Granularity provenance. Frame rate and patch size are fixed constants. Whether the recovered granularity is a property of the behaviour or of the preprocessing has never been manipulated.

Everything below uses labelled benchmarks, published metrics, and released code. No new evaluation theory is required.
Stage 0 — Reconnaissance

Do this first. Report before writing any new code.

    Locate and clone the SMQ repository. Read the README end to end.
    Produce a written map of the codebase covering:
        Data loading for HuGaDB, LARa, and BABEL, and the preprocessing each requires (frame rate, joint format, normalisation, any alignment step such as Procrustes).
        Where the encoder, the quantisation module, and the decoder are defined.
        Where the codebook update happens (EMA, commitment loss, or FSQ — establish which).
        The patch size and where it is set. Confirm whether it is specified in frames or in seconds, and whether it is per-dataset.
        Where evaluation happens. Confirm the protocol is global Hungarian matching followed by MoF, Edit, and F1@{10,25,50}.
        How train/test splits are constructed, and specifically whether subject identity is available in the dataloader. This determines whether E2 and E3 are possible without building new split logic. Flag clearly.
        Whether per-frame or per-patch code indices can be extracted from a trained model without modification. E1 and E3 both need this.
    Confirm the repo runs: start a training run on the smallest dataset (likely HuGaDB) and let it produce a checkpoint.

Report back: the codebase map, the patch-size mechanism, whether subject identity is accessible, and whether code extraction is available. Do not start E0 training runs until this is reviewed.

    erparameters to close the gap; if the published configuration does not reproduce, that is itself the finding.

E0 — Reproduce published numbers

Everything downstream is a delta from this baseline. An unverified baseline makes every later result meaningless.
Procedure

    Train SMQ as published on HuGaDB and LARa using the documented hyperparameters. Modify nothing.
    Evaluate with the repo's own evaluation code.
    Run three seeds per dataset. This is not optional — see below.

Target numbers

From HiST-VQ's Table 1, SMQ row:
Dataset 	MoF 	Edit 	F1@10 	F1@25 	F1@50
HuGaDB 	42.0 	36.1 	38.5 	31.5 	24.3
LARa 	37.4 	39.4 	34.7 	28.4 	16.4
Why three seeds

Neither paper reports seed variance. We need to know it before any later comparison is interpretable. If MoF varies by ±3 points across seeds, then a 2-point drop in E2 is noise and must not be reported as an effect.

Deliverable: mean and standard deviation of every metric, per dataset, across seeds. This standard deviation becomes the significance threshold for E2 and E4.
Stop condition

If reproduced numbers deviate from published by more than the seed variance you measured — treat

    3–5 MoF points as a flag if variance is small — stop and report the discrepancy. Do not proceed. Do not tune hyperparameters to close the gap; if the published configuration does not reproduce, that is itself the finding.

Time budget

Two weeks. If Stage 0 plus E0 exceeds this, report it — the overall project timeline depends on this number.

E1 — How many patches is an action?

Cheapest experiment here. Part A needs no trained model at all. Run Part A in parallel with E0.
Part A — Ground-truth duration structure

    For each dataset (HuGaDB, LARa, and BABEL if the dataloader supports it), iterate every annotated segment in the ground truth and record its length in frames.
    Divide each length by that dataset's patch size (60 / 50 / 30 frames respectively — confirm against Stage 0's finding).
    Produce, per dataset:
        A histogram of segment length in patches, log-x axis.
        A table of per-class mean, standard deviation, min, max, and the 5th/95th percentiles.
        The fraction of segments shorter than one patch.
        The fraction longer than five patches.

Part B — Discovered duration structure

    From each E0 checkpoint, extract the predicted segmentation (the run-length encoding of the predicted label sequence, post-Hungarian-matching).

    Histogram predicted segment lengths on the same axes as Part A. Overlay.

    Compute the Jensen–Shannon distance between predicted and ground-truth segment-length histograms. This is HiST-VQ's metric, borrowed from HVQ. Convention to match:
        Bin segment lengths at 20 frames, with an overflow bin for the tail.
        Normalise both histograms to sum to 1.
        Compute JS distance (the square root of the divergence), per video.
        Average over videos, then frame-weight across activities.
        Report ×100.

    Note that the 20-frame bin width is the metric's own parameter and is unrelated to patch size. Keep JSD in raw frames so it stays comparable to published values; the patch-normalised histogram in Part A is a separate output.

    Published values for reference — lower is better:
    	HuGaDB 	Lara Babel1
    SMQ 	87.1 	74.2    72.8
    HiST-VQ 	89.0 	73.8    74.4



Compute the null floor. This is required, not optional. Published JSD values all sit between roughly 65 and 95, compressed near the ceiling, and no paper reports what a perfect method would score. The reason is sparsity: each video contributes only a handful of ground-truth segments (Breakfast averages about six), so you are computing a distance between two histograms built from very few samples. Two independent draws from the same true length distribution score well above zero.

Estimate the floor by bootstrapping: within each activity, resample ground-truth segment lengths into two disjoint halves and compute JSD between them. Repeat and average. Report every JSD value relative to this floor as well as raw.

Also report a bin-free alternative. Compute the Wasserstein-1 distance (or a KS statistic) between pooled raw predicted and ground-truth segment lengths per activity. Same diagnostic, no bin-width parameter, no per-video sparsity problem. If the two metrics disagree in their ordering, that is worth knowing and worth reporting.

What this establishes

Part A answers "how many units combine to an action" by measurement. HiST-VQ's architecture assumes roughly two, via α=2 setting the subaction codebook to twice the action codebook. If the real distribution is broad, a fixed patch plus a fixed α cannot accommodate it, and that is shown rather than argued.

Part B answers whether the method reproduces the data's duration structure or imposes its own.
Stop condition / branch

    If ground-truth segments cluster tightly around one patch (say, most mass between 0.5 and 2 patches), the fixed-unit criticism is weaker than assumed. Report this prominently. It shifts priority toward E4.
    If the distribution is broad or multi-modal, this is the project's Figure 1. Proceed.

Deliverable

The overlaid histograms as publication-quality figures (vector format), the per-class table as CSV, and the JSD numbers.



E2 — Subject holdout

Requires that Stage 0 confirmed subject identity is accessible. If it is not, report the amount of work required to build subject-aware splits before starting.
Procedure

    Construct held-out-subject splits:
        HuGaDB: 14 train / 4 test.
        LARa: 11 train / 3 test.
    Generate at least four different random splits per dataset. With only 3–4 held-out subjects, a single split is dominated by whoever happens to be in it. Four is a minimum, more is better if compute allows.
    Retrain SMQ from scratch on each split, using E0's hyperparameters unchanged.
    Evaluate on the held-out subjects. Report MoF, Edit, F1@{10,25,50}.
    Compare to E0's pooled numbers.

What this establishes

The published protocol pools subjects, so cross-subject generalisation is entirely unmeasured. If the drop is large, the codebook is partly subject-specific — which would also partly explain the gap between unsupervised performance (LARa 45.9 MoF) and supervised (67.9).

A representation that does not transfer across people is not a representation of behaviour.
Interpretation rules — read before reporting

    Compare the drop against E0's seed standard deviation, not against zero. A drop smaller than seed variance is not a result.
    Report the spread across splits, not just the mean. High variance across splits is itself informative: it means some subjects are much harder than others, which is a finding.
    If one split is a dramatic outlier, investigate which subject drives it before drawing conclusions.

Negative result handling

If there is no drop beyond seed variance, report that plainly and stop this branch. "The codebook generalises across subjects" is a legitimate and useful finding. Do not search for a split that produces a drop.


E3 — Is subject identity decodable from code usage?

No retraining required. Uses E0's checkpoints. Run this before E2 — it is nearly free and it predicts E2's outcome.
Procedure

    From an E0 checkpoint, extract the code index sequence for every sequence in the dataset.
    For each subject, build a normalised histogram of code usage across all their frames. If sequences per subject are long, build one histogram per sequence instead so you have multiple samples per subject.
    Train a linear classifier (logistic regression) mapping code-usage histogram → subject identity. Cross-validate over held-out sequences from the same subjects. Report accuracy and chance level (1/18 for HuGaDB, 1/14 for LARa).
    Baseline: the identical classifier, but from the ground-truth action-label histogram instead of code usage.
    Control: scale-normalise the skeleton — divide coordinates by a body-intrinsic length such as mean limb length or torso length — re-extract codes (this requires re-running the encoder on normalised input; note if it requires retraining and flag the cost), and repeat step 3.

What this establishes

The action-label baseline should sit near chance: in HuGaDB everyone walks, runs, and climbs stairs, so what they did barely differs between people. Any gap between code-usage accuracy and that baseline is individual signal present in the codes and absent from the annotation.
Why the control is non-negotiable

LARa centres the skeleton at the root but does not normalise limb lengths. VQ allocates codebook capacity to reduce reconstruction error, which tracks displacement magnitude. A taller subject produces larger displacements for identical movement. So identity might be decodable from body size rather than movement style.

    Signal survives scale normalisation → the codes carry movement style. This is the interesting result.
    Signal vanishes → the codebook is partly encoding anthropometry. Still reportable, different claim. Do not present it as style.

Additional measurement while you are here

Compute pairwise Jensen–Shannon divergence between per-subject code-usage distributions. High divergence means the codebook has partitioned by person rather than by behaviour. Report as a subject × subject heatmap.


E4 — Does recovered granularity track the sampling rate or the data?

Most expensive experiment. Do not start until E1 has reported. Scope to HuGaDB first; add LARa only if time and compute allow.
Two sweeps

Sweep A — patch fixed in frames, real duration varies.

Downsample the input to several frame rates (e.g. the native rate, 1/2, 1/4). Keep patch size at 60 frames throughout. The patch's real duration therefore changes across conditions. Retrain at each rate.

Sweep B — patch fixed in seconds, real duration constant.

Same frame rates, but scale the patch size so it always spans one second of real time. Retrain at each rate.
Reporting for both sweeps

    MoF, Edit, F1@{10,25,50}.
    Discovered segment durations in seconds, not frames — this is the whole point and frames are not comparable across conditions.
    JSD against ground-truth segment lengths (also converted to a common time base).

What this establishes

This is the only experiment that manipulates rather than measures. It answers whether the recovered unit is a property of behaviour or of preprocessing.

    Sweep A isolates the patch's temporal extent. If discovered durations track it in seconds, the "discovered unit" is just the patch and nothing was discovered.
    Sweep B isolates input resolution with timescale held constant. If results also change here, the method is sensitive to sampling fineness independently of unit size.
    A dissociation — A moves results, B does not — is the cleanest outcome: temporal extent is what matters and frame rate is incidental, meaning every published number is a function of a hand-set constant.

Confound to control

Downsampling changes both the timescale and the number of frames per sequence, so it changes the effective amount of training data. Hold the number of training samples (patches seen during training) constant across conditions, or you will confound "coarser sampling" with "less data". State explicitly in your report how you did this.
Stop condition

If Sweep A shows discovered durations are flat in seconds across conditions — i.e. the method finds the same real timescale regardless of patch size — that contradicts the expected result and is a strong finding in the opposite direction. Report it and stop rather than searching for a configuration that behaves as predicted.





E5 — Segment-level code audit: what do the codes actually encode?

Runs on E0's checkpoints. No retraining. But it has a feasibility gate that must be cleared first — read E5.0 before writing any audit code.
Motivation

An independent audit of a related method (LAPS, Zhang et al.) built segment-level code histograms on Breakfast and asked whether instances of the same ground-truth action share a code fingerprint once video and activity are controlled for. The answer was no: same-action cross-video pairs scored 0.478 against a random baseline of 0.487, permutation p = 0.739. The positive control failed, RSA was near-zero, and a block-shuffle null reproduced almost the whole effect.

The one positive finding was a phase split: the first half of one action was more similar to the first half of a different action (0.391) than to the second half of the same action performed by someone else (0.356). The codes track motion phase — onset, sustained, offset — more strongly than action identity.

We are testing whether that replicates on a different tokenizer (SMQ's skeleton codebook rather than FSQ over keypoint tracks), different data, and a different task framing. If it does, it is a property of discrete motion codes rather than of one pipeline.
E5.0 — Feasibility gate (do this first, report before proceeding)

SMQ emits one code per patch, and a patch is one second. E1 Part A measured median segment lengths of 3.47 patches on HuGaDB, 2.20 on LARa, 1.20 on BABEL-1. So a segment-level histogram may contain only two to four codes over a codebook of hundreds.

A cosine similarity between two three-element histograms over a large vocabulary is close to meaningless, and this is the same sparsity problem that produced the varying JSD null floors in E1 Part B.

Measure before building:

    Distribution of codes per segment for HuGaDB and LARa, using E0's checkpoints and the ground-truth segmentation. Report median, quartiles, and the fraction of segments with fewer than 5, 10, and 20 codes.
    Compute the null floor for the audit itself. Take the ground-truth labels, shuffle them, and compute the D-vs-E gap. Repeat. This tells you the smallest gap the audit could detect at this sparsity. If the detectable effect size is larger than any plausible real effect, the audit cannot answer the question on this data and you should say so rather than run it.

Gate:

    Median codes per segment ≥ 10 on at least one dataset → proceed to E5.1 on that dataset.
    Median < 10 → do not proceed with per-segment histograms. Report the sparsity and stop. Three fallbacks are available, in order of preference; propose one and wait for review rather than picking unilaterally:
        Restrict to long segments. E1A found 29.1% of HuGaDB segments exceed 5 patches. Auditing only those is valid but biased toward long actions — state the bias.
        Use continuous pre-quantisation latents, mean-pooled per segment, instead of code histograms. This tests the representation rather than the codes, which is a different and weaker claim.
        Wait for E4. A sweep at finer patch sizes produces more codes per segment as a side effect.

E5.1 — Bucket construction

Breakfast's confound was dish (the activity a video depicts). HuGaDB and LARa have no dish-level grouping, so subject replaces dish as the controlled confound. This is an improvement, not a compromise — it makes E5 a direct test of the question E2 and E3 left open.

Build every pair of ground-truth action instances and assign it to exactly one bucket:
	Recording 	Subject 	Label 	Interpretation
A 	same 	same 	different 	context ceiling
B 	different 	same 	same 	subject-confounded match
C 	different 	same 	different 	subject-matched confusion
D 	different 	different 	same 	cleanest action signal
E 	different 	different 	different 	random baseline

Same-recording pairs are necessarily same-subject, which is why only five of the eight cells exist.

The headline comparison is D vs E. Both are cross-recording and cross-subject; the only thing that varies is whether the label matches. If codes encoded action identity, D sits well above E.

Include only labels with ≥3 instances across ≥3 recordings. Report how many labels and instances qualify, per dataset.

For LARa, which has scenarios (L01/L02/L03) as well as subjects, run the bucketing twice — once with subject as the confound, once with scenario — and report both.
E5.2 — Checks, in run order

Run all of these. Report each one whether or not it supports the hypothesis.

    Vocabulary check. Codes used out of the full codebook, perplexity, fraction of mass in the top 50 codes. This detects codebook collapse, which would invalidate everything downstream.
    TF-IDF weighting. Report weighted and unweighted results side by side throughout. If the direction of any result changes between them, say so prominently.
    Positive control. Pick the most static, most self-similar class in each dataset — likely standing on HuGaDB and Standing or None on LARa. It should be the easiest class: highest self-similarity, top-ranked. If the positive control fails, the pipeline is suspect and the negative results below cannot be trusted. Report this before interpreting anything else.
    Retrieval. Leave-one-out top-1 accuracy over instances, excluding same-recording neighbours. Report against chance (1/n_labels).
    Bootstrap CIs. 2,000 resamples over instances. Report whether D and E intervals overlap.
    Label permutation test. Shuffle labels 10,000 times, one-sided, and report the p-value for the D−E gap.
    Block-shuffle null. Scramble code order within each recording, preserving per-recording marginals, and recompute D−E. This separates genuine action signal from pure recording context. If the shuffled gap matches the real one, there is no residual signal above context.
    Conditional mutual information. I(code; subject | action), real versus a null. If codes cleanly separated action from subject this collapses toward zero. Compare directly against E3's identity decodability — these two measurements should agree, and if they don't, that needs explaining.
    Label × label RSA. Correlate the cross-recording similarity matrix against a semantic similarity structure over labels. On LARa the Handling (upwards / centred / downwards) family gives a natural shared-verb grouping. HuGaDB has no obvious verb family — use a static/dynamic or limb-involvement grouping instead, and state that it is weaker.
    Bigrams. Append the top-500 consecutive code pairs to each histogram and rerun steps 4–6. This tests whether flat pooling destroying local order is the bottleneck.

E5.3 — The phase split (the key experiment)

This is the result most worth replicating. Do not treat it as an afterthought.

    Split every qualifying segment into a first half and a second half.
    Build separate code histograms for each half.
    Compute two cross-recording similarities:
        Same action, opposite phase — first half of instance i vs second half of instance j, same label, different recording.
        Different action, same phase — first half of instance i vs first half of instance j, different labels, different recording.

If the second exceeds the first, phase dominates identity, and the LAPS finding replicates.

Reporting constraint, state it explicitly in the writeup: halving segments halves the codes per histogram, so both phase numbers sit structurally below the bucket D and E values. Only the within-experiment comparison between the two phase conditions is valid. Do not compare phase numbers to D or E.

Given the sparsity already flagged in E5.0, this sub-experiment is the most affected by it — halved segments on HuGaDB means roughly 1–2 codes per histogram. It may only be runnable on the long-segment subset. Check before running.
What each outcome means

    Phase dominates, replicating LAPS → the codes are sub-action units with internal onset / sustained / offset structure. This points directly at a multi-state left-to-right topology per unit with an explicit duration, which is the form Lee & Glass use for phones and for the same reason. It also means Hungarian matching to action classes does not reveal a correspondence, it manufactures one across a level gap.
    D significantly exceeds E → the codes do carry action identity and the LAPS result is specific to that pipeline. Report it; it weakens the level-gap argument.
    Positive control fails → stop, debug, report. Nothing else is interpretable.
    Sparsity floor exceeds any plausible effect → the audit cannot be run on per-second codes, which is itself a finding about the representation's granularity.

Deliverables

Bucket means with bootstrap CIs, all ten check results in run order, the label × label similarity heatmap (with and without the positive-control class), the phase-split pair, and the sparsity measurements from E5.0.























Global rules

Report negative results. Four of these five experiments have informative null outcomes. A null is a result, not a failure.

Never tune to produce an expected result. If a configuration does not behave as predicted, that is the finding. Report it.

Every comparison is against E0's seed variance, not against zero.

Do not silently change hyperparameters. Any deviation from the published SMQ configuration must be flagged explicitly in the report, with the reason.

Version everything. Record the exact commit of the SMQ repo, the exact config for every run, and the random seed. Store checkpoints so E3-style analyses can be rerun without retraining.
Deliverables per stage

Each report should contain:

    What was run — exact configs, seeds, commit hashes.
    The numbers, with variance.
    Figures where specified, in vector format.
    Any deviation from this brief and why.
    A one-paragraph plain-language statement of what the result means and whether the stop condition was triggered.
