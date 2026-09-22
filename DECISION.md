# DECISION — is a segmentation-first motion representation worth developing?

Written for: the project team deciding how to spend the remaining submission time.
Evidence: [RESULTS.md](RESULTS.md) (all numbers generated from saved outputs), [PROTOCOL.md](PROTOCOL.md) (frozen before the runs), [EXPERIMENT_MANIFEST.md](EXPERIMENT_MANIFEST.md).
Scope of the evidence: frozen released SMQ encoders (one checkpoint per dataset), LARa and BABEL-1/2/3, **transductive** setting (the encoder has seen every recording), k-means at K = number of action classes, 8-bin segment encoding.
BABEL subsets are related, not independent; BABEL has no participant IDs.

## Verdict

**Do not invest the remaining time in the tested recipe (label-free linear-fit DP segmentation → 8-bin segment encoding → k-means).** It failed the prespecified continuation rule on every dataset under the primary mapping.
**The oracle diagnostic does show headroom on LARa**, so the idea is not refuted; the failure is in getting label-free boundaries that recover that headroom. No replication with retraining and no Experiment 3 were launched, per the stopping rule.

## 1. Do meaningful boundaries improve representations beyond duration-matched random boundaries?

**Yes on retrieval, with a built-in qualification.** Continuous common-anchor precision@10, Oracle − Random (paired 95 % interval): LARa +0.251 [0.232, 0.271], BABEL-1 +0.132 [0.118, 0.147], BABEL-2 +0.084 [0.058, 0.109], BABEL-3 +0.099 [0.079, 0.124].
Oracle − Fixed: +0.121, +0.102, +0.083, +0.081 (all intervals exclude 0). Random boundaries are *worse* than the 1 s grid on LARa and BABEL-1, so this is not a pooling-duration/rate effect.
The oracle segments are label-pure by construction (the anchor's label equals its segment's label for 100 % of anchors, vs 91–96 % for the grid and 77–89 % for random), and among segments of ≥ 0.8 purity the oracle-vs-fixed continuous difference is ≈ 0 on LARa, BABEL-1 and BABEL-2.
So relative to the grid the gain is mostly avoiding transition-straddling segments; relative to random it holds on both endpoints. On BABEL the null is weak (33–81 % of recordings have < 4 annotated runs), though restricting to informative-null recordings leaves the BABEL-1 and BABEL-2 conclusions unchanged (BABEL-3 keeps only 35 recordings and its intervals include 0).

## 2. Is the main limitation the encoder, segmentation, or quantization?

**Segmentation is the clearest limitation; quantization is a second one; the encoder's role is unresolved.**
* The frozen features contain the oracle advantage (continuous), so this is not simply a weak encoder for LARa.
* Label-free DP segmentation recovers little of it (question 3).
* Quantization at K = C keeps 30–40 % (LARa) and 10–24 % (BABEL) of the continuous lift, and the oracle-vs-fixed quantized gain is +0.070 on LARa but ≤ 0.013 on BABEL (5–16 % of the continuous gain). This is a real loss, but K = C is a benchmark budget, not an optimum; a larger-vocabulary sensitivity was not run.
* The controlled recipe on fixed windows is **below original SMQ** on F1@50 by 2.4–8.8 points. One untested explanation is that the frozen latent is co-adapted to SMQ's jointly trained 1 s-patch quantizer; encoder versus quantizer effects cannot be separated with the present design.

## 3. Does label-free segmentation recover any of the oracle advantage?

**Little on LARa, none on BABEL.** DP − Fixed continuous common-anchor precision: LARa +0.016 [0.011, 0.022] (about 14 % of the oracle's +0.121; quantized +0.003, NMI +0.004 not distinguishable from 0); BABEL-1 −0.019, BABEL-2 +0.004, BABEL-3 −0.006.
Primary endpoint F1@50 (DP − Fixed, Hungarian mapping; target +2.0): LARa +0.1 [−1.3, 1.7], BABEL-1 −1.0 [−1.7, −0.4], BABEL-2 −2.0 [−4.0, 0.0], BABEL-3 −8.4 [−11.4, −5.5].
On LARa DP beats the count-and-duration-matched random partition (+1.5 [0.5, 2.6] F1@50), so its boundaries are not random, but its raw cuts are not better aligned to annotated boundaries than the grid, and its anchor–segment label agreement equals the grid's (93.8 %).
Under the train-fitted mapping LARa shows +2.1 [0.5, 3.8] F1@50 and would pass the rule, but that mapping reaches only ~3 of 8 labels (label collapse) and was not the prespecified readout; it is recorded, not used.
Rate matching worked (≈ 1.0 segments/s), the 10 s bound is almost never reached, 15–22 % of BABEL segments sit at the 0.25 s minimum (LARa 1.9 %); DP durations (median 0.6–0.8 s) are far shorter than annotated ones (1.2–2.2 s).

## 4. Is there sufficient evidence to invest the remaining submission time?

**No, not in this recipe.** Continuation rule (a: ≥ +2 F1@50 with interval excluding 0; b: Edit and MoF compatible; c: random does not explain it): no dataset advances under the primary mapping; BABEL-1/2/3 have negative point estimates; LARa's effect is null under the primary mapping and mapping-dependent otherwise. There is no "clear success on one dataset" to justify a targeted replication.
What the evidence does license is a bounded, separately decided question, not an automatic next step: the LARa oracle gap (+0.12 continuous, +0.07 quantized) is large and stable across k-means seeds, so whether *any* label-free boundary source can recover part of it is open. Per the stopping rule I have not added a second segmenter, losses, larger codebooks, a hierarchy, or other datasets, and I recommend against doing so without an explicit decision.
Not tested here and therefore not evidence for or against: H2 (discretization utility, Experiment 3), encoder-seed variability, a 4K vocabulary, inductive (held-out-recording) encoders. If the team wants Experiment 3 anyway, it needs causal segment extraction and held-out-recording encoders (four LARa subject-split encoders exist; BABEL needs retraining, ≈ 1–3 GPU-hours; see the manifest).

## 5. What exact claim can the current evidence support?

> With the frozen released SMQ encoders (transductive setting), an 8-bin encoding of latent trajectories over annotation-defined action intervals retrieves same-action anchors from other participants (LARa) or recordings (BABEL) better than the same encoding over duration-matched random intervals (continuous p@10 +0.08 to +0.25) and over one-second windows (+0.08 to +0.12); the advantage over one-second windows appears to come mostly from label-pure segments avoiding annotated transitions (among segments of ≥ 0.8 purity the continuous oracle-vs-fixed difference is ≈ 0 on LARa, BABEL-1 and BABEL-2). After k-means at K = number of classes the advantage survives on LARa (+0.07 over the grid) but is ≤ 0.013 on BABEL. A label-free, exact linear-fit dynamic-programming segmentation matched to one event per second recovers about 14 % of the continuous oracle gain on LARa and none on BABEL, and does not improve segmental F1@50 over fixed windows (−8.4 to +0.1 points under the primary mapping).

It does **not** support: discovery of action atoms, syllables or semantic primitives; a discretization advantage (H2 was not tested; continuous retrieval was better than quantized everywhere); generalization to unseen people or recordings (the encoder is transductive); independent replication across BABEL subsets; or that oracle boundaries indicate boundary discovery.

**No gain from a tested recipe is not proof that discrete motion representations cannot work; a gain in segmentation or prediction is not proof of discovered semantic atoms.**
