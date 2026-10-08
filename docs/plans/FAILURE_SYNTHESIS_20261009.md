# Why the recent architectures failed, and what that implies

9 October 2026 · Covers Milestone 1, the input-representation test, unit grouping, residual co-occurrence, frozen MotionBERT and the self-labelled embedding. Numbers come from the saved run outputs cited in each results document; the self-labelled figures are from `results/self_labelled/sl_babel2_001` (125 cells; `L_noaug` was finishing its last cells at the time of writing).

## 1. Each experiment: what it tested, what failed, and the measured reason

| # | experiment (dataset) | hypothesis | outcome | measured reason for the failure |
|---|---|---|---|---|
| 1 | Milestone 1, ordered unit dynamics (LARa) | the order of preceding units carries action identity | STOP: ordered −6.3 F1@50 vs B0, ≈ bag | order was learned (conditional NLL 1.9 vs 2.3 for bag), but shuffling the history changed F1@50 by < 1. At a 2 s horizon, order predicts **local continuity**, which is shared across actions. The dynamics costs also over-segmented (3.0–3.3× true segments). |
| 2 | input representation: body-scale, centred (LARa) | the per-window input lacks information or carries identity | no input passes; supervised probe 65–67 vs decode 39–42 | (a) codes keep the information (code-table probe 61–63). (b) Scale normalization did not remove identity: body **proportions** remain (subject probe 98.5%). (c) 15 MoF is lost to **one-to-one mapping**: 5 of 8 states fall in Handling(centred), which is 46% of frames. |
| 3 | unit grouping, 24 → 8 states (LARa) | over-complete states grouped by temporal co-occurrence form actions | STOP; temporal ≈ shuffled ≈ random | the 24 states are purer (oracle 58–59 vs 54), so the headroom exists. Temporal co-occurrence matched **same-recording co-occurrence**, and no label-free grouping matched the oracle grouping (ARI 0.07–0.20). |
| 4 | residual co-occurrence (LARa) | adjacency beyond recording membership groups states into actions | STOP; residual ≈ temporal | only **3.5–5.6%** of the temporal affinity is not explained by within-recording occupancy. With a ~2 s horizon and ~3.7 s segments, much of that small residual links *neighbouring actions* across boundaries. |
| 5 | frozen MotionBERT (BABEL-2) | a large pretrained encoder supplies missing information | STOP (MoF +6.4, F1@50 +1.4, probe −0.5) | no new linearly decodable information (probe 77 vs 77, purity ~64 for both). The MoF gain is **sit recall** (0.58 → 0.70; sit is 51% of frames). Short events were lost: the ~4 s context blurs segments under 1 s (stand_up 0.40 → 0.11). |
| 6 | self-labelled embedding with batch OT (BABEL-2) | learning the embedding with the segmentation objective, with cross-recording equipartition and 3D augmentation, organizes by action | fails clearly: `L_full` MoF 39.4 vs F0 54.5 and F1 46.2; F1@50 21.4 vs 24.5 / 21.6 | training did **not raise purity** (many-to-one 57.5 vs 58.8 for its frozen input F1). It under-segments (0.64× true segments) and loses sit recall (0.53 → 0.36). The batch-equipartition arm *raised* recording NMI (0.33 vs 0.29). Forcing every recording to use every state (`L_noeq`) removed recording association (NMI 0.04) and gave the **purest** states (61.2) but the **worst** MoF (33.6): each recording used 2.4× as many states as it has classes. |

## 2. What the failures have in common

### A. Label-free purity barely moves, whatever we change

Many-to-one purity (each decoded state mapped to its majority class) is the best a hard relabelling of the states could achieve. It stays in a narrow band across six interventions:

| | LARa | BABEL-2 |
|---|---|---|
| label-free decodes (all inputs, decoders, groupings, trained embeddings) | 49–55 (24-state oracle 58–59) | 57–64 |
| supervised linear probe | 65–67 | 77 |
| majority class | 46 | 51 |

Changing the input (T2 variants, MotionBERT), the decoder (dynamics), the post-hoc grouping (temporal, residual, appearance) or the embedding objective (self-labelling) moved purity by a few points at most. **None of the label-free signals we used changes *which* windows cluster together much.** All of them recover the same coarse structure, about 10–15 points of purity below what a supervised probe finds. Self-training, our attempt to change it, amplified the starting structure: Phase 9 raised recording NMI, `L_full` kept purity flat.

### B. Most MoF differences come from how states are allocated to classes, not from purity

With purity nearly fixed, MoF moves when the C states line up differently with the C classes under one-to-one Hungarian mapping:
- LARa: about 15 points are lost because several states share the 46% majority class (many-to-one 54 vs one-to-one 39).
- MotionBERT's +6 MoF is mostly sit recall.
- `L_noeq` has the purest states but the worst MoF, because the decoder's prior spreads each recording over all states.
- `L_full`'s loss against F0 is concentrated in sit.

The quantities that set this allocation are **decoder priors**:
- the per-recording relaxed uniform state marginal (q = 1/C per recording);
- the emission-to-temporal balance. Halving the emission weight gave +5 to +6 F1@50 for every arm on LARa, and took T2 on BABEL-2 from 19.7 to 25.3.
- the requirement of exactly C states.

None of these was ever the object of a test. They were held fixed as part of B0 in every experiment.

### C. The recording/identity signal is partly real action structure, so removing it does not help

I treated recording association as a shortcut to remove (residual co-occurrence, batch equipartition). The evidence does not support that framing on these datasets:
- Recordings genuinely differ in which actions they contain (about 2 classes per BABEL-2 recording).
- Removing recording structure removed action signal with it. The residual affinity left 4% of the signal and did not group actions. `L_noeq` drove recording NMI to 0.04 and accuracy to its lowest.
- Subject identity is carried by body proportions, which are not removable by scale normalization, but nothing showed that this identity is what blocks action organization.

**My earlier claims that identity was "the" bottleneck were not established and should not guide the next step.**

### D. Time scales are mismatched

BABEL-2 segments have a median of 1.4 s, and 30% are under 1 s; LARa segments average about 3.7 s. Our mechanisms operated at 2–7 s:
- the 2 s dynamics history;
- the 2 s co-occurrence horizon;
- MotionBERT's ~4 s clips (about 6.75 s of support after overlap averaging);
- the ±0.8 s decoder kernel on top.

At these scales the signal they capture is continuity and neighbouring actions, not action identity. That explains why order (1), co-occurrence (3, 4) and long context (5) all failed in the same way.

## 3. What this rules out, and what it leaves

**Ruled out as the main lever, at the scales and with the tests run:**
- more information in the input: probes did not rise with MotionBERT, and codes keep the information;
- label-free temporal co-occurrence or order at about 2 s;
- removing recording association as such;
- self-training from the frozen features' own clusters without a new external signal.

**Left, with direct evidence:**
1. **Decoder priors and allocation** (B). This is the only factor that moved results substantially and consistently (the multiplier and `L_noeq` vs `L_full` evidence), and it has never been varied deliberately. In particular:
   - a dataset-level instead of per-recording state marginal (training evidence: global vs per-recording equipartition, +5.8 MoF);
   - the emission/temporal balance;
   - allowing more states than classes in the decode, with a principled many-to-one allowance. The latter conflicts with the benchmark's one-to-one convention and would have to be reported separately.
2. **The purity gap** (A). Purity sits about 10–15 points below the supervised probe in every condition. What could close it is an external signal, not another label-free self-consistency signal derived from the same features. Two ways to find out which signal matters:
   - **Diagnose the published methods with the same instruments.** Measure purity and allocation for the released SMQ checkpoints, scored with our verified scorer. If SMQ's advantage on some metrics comes from allocation rather than purity, the field's progress is also mostly in the decoder; if it comes from purity, SMQ's joint training is doing something ours does not.
   - **Measure the label-efficiency curve.** Report purity and MoF with k labelled segments per class (k = 1, 3, 5). This quantifies how much external signal the gap needs.
3. **Time scale** (D). Any new mechanism should operate at the segment scale of the dataset (about 1–2 s on BABEL-2), not at 2–7 s.

## 4. Proposed order of next steps

1. **SMQ diagnostic and standing check (cheap, decisive for direction).** Partly done already: the released SMQ checkpoints were re-scored with many-to-one mapping in an earlier audit (`results/exp/analysis/pretrained/*.json`; `results/CONSOLIDATED_AUDIT.json`, `released_many_to_one_rescore`):

   | released SMQ | one-to-one MoF | purity (many-to-one MoF) |
   |---|---|---|
   | HuGaDB | 42.0 | 56.5 |
   | LARa | 37.4 | 48.1 |
   | BABEL-1 | 36.4 | 48.7 |

   SMQ loses 11–15 MoF to one-to-one allocation, like our decoders. On LARa its states are less pure than our T2 B0 (48.1 vs 54.3, on different evaluation populations), so its results there do not come from higher purity.

   **Remaining work:**
   - the same numbers for SMQ on BABEL-2/3;
   - per-class recall and segment-length breakdowns for all datasets;
   - our best pipeline (frozen MotionBERT codes with B0) scored under the same full-dataset protocol, so the comparison is matched.
2. **Decoder-prior test on the current best input (one dataset, declared rule).** On the frozen MotionBERT codes, BABEL-2, five folds, vary only the decoder prior:
   - per-recording vs dataset-level state marginal;
   - a declared emission/temporal balance, fixed by a label-free criterion, not chosen on labels.

   This is the factor the evidence points to, and it has never been tested.
3. **Only then a new representation mechanism,** targeted at the purity gap, operating at the 1–2 s segment scale, and justified by whichever of step 1's findings it addresses.

## 5. Process lessons

- I proposed each experiment with more confidence than the evidence supported, and several rationales (identity as the bottleneck, order, long context) were assumptions rather than measured findings at the time. Each plan should state the measured fact it rests on and the competing explanation it rules out.
- The B0 decoder was held fixed as "the baseline" in every test, so its priors were never examined although they drive most of the variation we observed. Holding a component fixed is a choice that should be revisited when results cluster.
- Purity, allocation, per-class recall and segment-length breakdowns explained every failure after the fact. They should be standard outputs of every run, not post-hoc analyses.
