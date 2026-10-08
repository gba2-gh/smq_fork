# Unit-to-action grouping test (LARa)

Version 2 · 7 October 2026 · Follows [INPUT_REPRESENTATION_RESULTS.md](INPUT_REPRESENTATION_RESULTS.md) (revised reading) and the failed Milestone 1 gate ([M1 report](../../results/compositional_dynamics/m1_lara_001/REPORT.md)). Changes from v1 are listed in Section 10.

## 1. Why this test, and what it can and cannot show

On LARa, the matched B0 decoder (C = 8 states) splits the dominant action into sub-states: with T2, 5 of 8 states are mostly Handling(centred), which is 46% of frames. One-to-one Hungarian mapping can credit only one of them, costing about 15 MoF (many-to-one 54.3 vs one-to-one 38.9). Rare actions get no state.

The test separates local state discovery from action discovery:

> 512 input tokens → M = 24 transport states → 8 action groups (label-free).

Question:

> Does grouping over-complete transport states into C actions by **local temporal association** beat the C-state decoder, beat grouping by appearance, and beat grouping by mere co-membership in the same recordings?

**Scope of the claim.**
- **Temporal grouping, not ordered composition.** The affinity is symmetric, so reversed sequences give the same groups. A positive result shows that local temporal association helps group states. It says nothing about order (which Milestone 1 already found adds no action evidence).
- **Transport states, not raw tokens.** States are grouped, and each state belongs to exactly one action. That is a limited form of hierarchy: it does not test whether one moveme can take different action meanings in different contexts.
- **Not novel by itself.** Co-occurrence grouping has prior art (for example SSCAP's co-occurrence-based action parsing). This test decides whether separating reusable motion states from action identity deserves the remaining implementation time. It is not a publishable contribution on its own.
- **The reference is B0.** On LARa and BABEL-2 the contextualized pipeline is not better than B0-centre (LARa 32.2/18.0 vs 38.5/18.4; BABEL-2 48.0/31.5 vs 57.4/30.6 MoF/F1@50), so B0 is the strongest available reference here. Integration with the contextualizer is a later question.

Input, tokenizer, transport and evaluation are unchanged; the grouping is the only new component.

## 2. Fixed settings

| Item | Setting |
|---|---|
| Dataset | LARa only. The confirmation dataset is declared in Section 7 and runs only after TEMPORAL_GROUPING. |
| Input | T2 relational units, the existing Phase 5 preparations for seeds 111, 222, 1538574472 |
| Population | existing primary split: 395 fitting / 44 evaluation recordings (`split.json` SHA-256 `04f8a605…`) |
| Over-complete decoder | the matched B0 of Milestone 1 (cosine cost on T2 code vectors, same transport coefficients, kernel and solver budgets), with **M = 3C = 24 states** |
| Initialization | the existing `initial_grouping` helper with M groups, fitted on fitting recordings only |
| Affinity horizon | h = 8 windows (about 1.9 s), Milestone 1's declared history length; not derived from labels, no sweep |
| Evaluation | unchanged: C = 8 output labels, one Hungarian mapping fitted globally over the 44 evaluation recordings, final solver endpoint, MoF and F1@50 co-primary |

M = 24 is declared, not tuned. No other state count is run.

## 3. The over-complete decode and its plans

1. Fit the M-state B0 on the fitting recordings (existing alternating fit).
2. **Freeze the final prototypes and decode every fitting recording afresh** with the final-inference ladder, from the cold start, up to the final endpoint. The resulting soft plans R_fit[t, i] (rows sum to one) are the only input to every grouping. Plans returned by the alternating fit are never used: they were solved before the last prototype update.
3. Decode the evaluation recordings with the same frozen prototypes and ladder. Their window labels are the argmax state, relabelled through a frozen grouping.

Every grouping arm shares steps 1–3. Arms differ only in which states share an action label, including group sizes.

## 4. Arms

All arms output C = 8 labels per window, so evaluation is identical to B0.

| Arm | Grouping of the 24 states | Purpose |
|---|---|---|
| `b0_c` | none; the existing Milestone 1 C-state cells, read, not re-run | reference |
| `group_temporal` | spectral grouping of the local temporal affinity (below) | **the hypothesis** |
| `group_shuffled` | identical procedure on R_fit with rows shuffled **within each fitting recording** (seeded) | same-recording co-membership with local adjacency destroyed |
| `group_appearance` | weighted KMeans(C) on the 24 prototypes (weights = state mass on fitting recordings) | grouping by appearance only, no temporal information |
| `group_random` | 100 seeded random partitions per seed, **matched to `group_temporal`'s group sizes** | prior-mass control (Section 4.1) |

**Temporal affinity and grouping** (label-free, fitting recordings only):

1. A[i, j] = Σ_recordings Σ_t Σ_{k=1..h} R[t, i] R[t+k, j], within recordings only.
2. A ← (A + Aᵀ)/2; diagonal set to zero.
3. Spectral clustering: normalized affinity D^-1/2 A D^-1/2, top C eigenvectors, rows normalized, KMeans(C) with k-means++, n_init = 10, cell seed.

`group_shuffled` repeats steps 1–3 on a within-recording row permutation of R_fit, then labels the original, unshuffled evaluation decode. Nothing from the evaluation recordings enters any grouping.

### 4.1 Why the random control is size-matched

With a uniform prior over 24 states, a group of six states carries six times the prior mass of a single-state group. A grouping can therefore gain over B0 by shifting prior mass toward the dominant class, not by representing sub-actions. Random partitions with the same group sizes as `group_temporal` keep that mass allocation and randomize membership. The learned grouping must beat them; see Section 5.

## 5. Decision rule (declared before any scored run)

Comparisons are paired by seed on the 44 evaluation recordings. **Bar**: mean ≥ +3 with all three seeds in the same direction. **Floor**: when a comparison is won on one metric, the other metric's mean may not fall by more than 2.

`group_temporal` must satisfy all of:

- **(a) vs B0:** passes the bar on MoF **and** F1@50 against `b0_c`.
- **(b) vs appearance:** passes the bar on MoF or F1@50 against `group_appearance`, with the floor on the other metric.
- **(c) vs shuffled:** passes the bar on MoF or F1@50 against `group_shuffled`, with the floor on the other metric.
- **(d) vs random:** in every seed, exceeds the 95th percentile of its 100 size-matched random groupings on MoF and on F1@50.

| Outcome | Condition | Meaning and next step |
|---|---|---|
| **TEMPORAL_GROUPING** | (a)–(d) all hold | Local temporal association organizes states into actions beyond appearance, recording co-membership and prior mass. Run the confirmation dataset (Section 7). If confirmed, integrate a grouping layer and then test whether it learns the state–action relationship across recordings. |
| **OVERCOMPLETE_ONLY** | (a) holds, and (b) or (c) fails | Over-complete decoding with grouping helps, but the temporal signal is not shown to be the reason. Report; do not build on temporal grouping specifically. |
| **PRIOR_MASS** | (a) holds but (d) fails | The gain is reproduced by size-matched random groupings: it comes from reallocating prior mass, not from grouping. Report as a decoder-prior finding. |
| **STOP** | (a) fails | **This** fixed-horizon spectral grouping of B0 states does not improve on the C-state decoder. It does not show that unit grouping is generally unpromising. Stop this specific method. |
| **INCONCLUSIVE** | missing seeds, invalid solves or failed verification | Report what is missing; no extra runs to fill gaps after inspection. |

Secondary readouts (Section 6) cannot rescue a failed gate or overturn a passed one; any disagreement is printed as a warning.

## 6. Diagnostics and secondary readouts

Diagnostics are label-using and for interpretation only.

- **Imbalance (reported prominently):**
  - balanced accuracy (mean per-class recall under the same Hungarian mapping);
  - per-class recall;
  - the number of classes that receive at least one group;
  - the predicted mass per group against class frequency.
- **Oracle relabelling bound:** many-to-one mapping of the 24 fixed states to labels on the evaluation recordings. It is an upper bound on MoF for hard relabellings of these states only, not on F1@50 or on any new decode. Each arm's share of the oracle's MoF gain over `b0_c` is reported.
- **Grouping agreement:** adjusted Rand index between each grouping and the oracle's state-to-class assignment.
- **Segmentation:** predicted ÷ true segments.
- **Smoothing sensitivity:** the 24-state decode at emission multipliers 0.5 and 2, relabelled by each grouping, to separate F1@50 differences from smoothing.
- **Identity:** recording and subject NMI of the final labels, read with the fragmentation caveat from the input-representation results.
- **Subject-disjoint folds (secondary):** the same 4 folds of 4 held-out subjects as Milestone 1 v2.2. Each fold has its own 24-state fit, fresh fitting plans and groupings estimated from that fold's fitting recordings only. Results are the mean over folds, with the decision rule applied descriptively. The folds are run after the primary split in the same launch, and cells projected to overrun the cap are skipped first.

## 7. Confirmation dataset (declared now, run only after TEMPORAL_GROUPING)

BABEL-2, T2, seeds 111/222/1538574472, its existing split, with every setting frozen from this plan: M = 3C = 15, h = 8 windows, the same arms and decision rule, run as a prospective test. LARa's evaluation recordings become development data once the LARa gate has been inspected.

## 8. Limits

- C is known (a benchmark convention), and M = 3C is a declared multiple of it.
- The tokenizer is fitted on all recordings (transductive). The evaluation recordings share subjects with the fitting set, except in the folds.
- h is declared, not swept. A different horizon could change the temporal grouping.
- LARa's 46% majority class favours methods that put mass on the dominant class. The size-matched random control, balanced accuracy and per-class recall are there to expose that.
- Three seeds on one split are not confidence intervals. The 95th-percentile random criterion is a per-seed screening threshold, not a significance test.

## 9. Implementation and cost

New code goes in `script/unit_grouping/` and reuses `script/compositional_dynamics`: the B0 cell with a state-count override, the decoding ladder and scoring. New code covers:
- the fresh fitting decode;
- the four groupings and the size-matched random partitions;
- the oracle and imbalance diagnostics;
- the report with the decision rule.

Required tests:
- every grouping is label-free and deterministic for a given seed;
- evaluation recordings cannot change any grouping;
- the affinity matrix is symmetric and invariant to sequence reversal (documenting the scope limit);
- the within-recording shuffle preserves each recording's state occupancy;
- relabelling preserves frame coverage;
- spectral grouping recovers planted groups on synthetic plans, and the shuffled control does not;
- size-matched random partitions reproduce the learned group sizes exactly;
- the fresh fitting plans come from the frozen final prototypes, not from the alternating fit.

Then a bounded smoke run.

Estimated implementation: half a day including tests and smoke.

Runtime: one 24-state fit plus two fresh decodes per seed. Transport cost grows roughly linearly with the number of states, so this is about 3× the C-state B0, about 25–30 min per seed, or about 1.5 h for the primary split. Groupings and random partitions take seconds. Folds add about 4–5 h. The total fits the overnight budget, with the primary split first. The dry run must report measured timings before launch.

## 10. Changes from version 1

| Change | Reason |
|---|---|
| Added `group_shuffled` (within-recording row shuffle) | Separates local temporal association from same-recording co-membership, the recurring identity confound. No extra transport fit. |
| Replaced the single random partition with 100 size-matched random partitions per seed, and added criterion (d) and outcome PRIOR_MASS | With a uniform prior over states, group size sets an action's prior mass. A gain over B0 could come from mass allocation instead of grouping. |
| Renamed COMPOSITION to TEMPORAL_GROUPING and narrowed the stated claim | The affinity is symmetric and reversal-invariant, so it cannot test order. |
| STOP now refers to this specific method | Failure of one fixed-horizon spectral grouping does not rule out unit grouping in general. |
| A random grouping close to B0 no longer makes the result inconclusive | That condition could void a clear learned improvement for an irrelevant reason. |
| Added a floor of −2 on the other metric when a control comparison is won on one metric | Prevents a gain on one metric from hiding a large loss on the other. |
| Groupings use fresh plans decoded under the frozen final prototypes | The alternating fit returns plans solved before the last prototype update. |
| Dropped the C-state re-decode variant | Minimum-over-states costs favour large groups, and a uniform C-state prior changes the hypothesis. |
| Dropped the M = 16 sensitivity | Adds little. The primary question uses only M = 24. |
| Oracle described as an MoF bound for hard relabellings only | It does not bound F1@50 or a new decode. |
| Balanced accuracy, per-class recall and group mass vs class frequency reported prominently | Class imbalance is central to interpreting any gain on LARa. MoF and F1@50 remain co-primary under the project protocol. |
| Stated scope: transport states (not tokens), fixed state-to-action assignment, B0 reference, novelty limits | So a positive result is not over-read. |

## 11. Implementation notes

- Outcome precedence when several conditions fail: STOP (a fails), then PRIOR_MASS (d fails), then OVERCOMPLETE_ONLY (b or c fails), else TEMPORAL_GROUPING.
- Code: `script/unit_grouping/` (README there). The Milestone 1 fit and decode are reused with their state count set to 24 at run time; `decode_ladder` additionally returns the final soft plans.
- The within-recording shuffle control is also exercised on planted data: when group identity lives only in recording membership the shuffled control recovers it, which is the confound it exists to expose.
- Smoke-tested (plumbing only). Projected cost from the dry run: about 32 min per primary cell, 7.0 h for the primary split plus four folds, unmeasured until a full cell completes.
