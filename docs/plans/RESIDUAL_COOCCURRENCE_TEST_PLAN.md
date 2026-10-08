# Recording-residual co-occurrence grouping (LARa, stage 1)

8 October 2026 · Follows [UNIT_GROUPING_RESULTS.md](UNIT_GROUPING_RESULTS.md) and [PRETRAINED_ENCODER_RESULTS.md](PRETRAINED_ENCODER_RESULTS.md) · Code: `script/unit_grouping/residual.py`, `residual_run.py`.

## 1. Why

Across the recent tests, the information needed for actions survives into the units: on BABEL-2 a supervised mapping of the 512 codes reaches about 73% against about 77% for a linear probe. The loss happens when units are grouped into actions without labels. The unit-grouping test showed why temporal grouping failed: its affinity was no better than one built after shuffling windows within each recording. In other words, temporal adjacency mostly measured **which states share recordings**, a recording and subject signal, not an action signal.

Hypothesis:

> Grouping states by the adjacency that recording membership does **not** explain organizes them into actions better than raw temporal adjacency, appearance, recording co-membership or random grouping.

## 2. Method (label-free, fitting recordings only)

- **A:** the temporal affinity of the unit-grouping test, unchanged. It sums, over windows up to h = 8 apart within a recording, the products of the fitting plans' state responsibilities, then symmetrizes A and sets the diagonal to zero.
- **E:** the exact expectation of A when each recording's windows are randomly permuted. This preserves each recording's state occupancy and destroys adjacency. Per recording with L windows, column sums S and Gram matrix G = RᵀR, every lag contributes (L − k) × (S Sᵀ − G) / (L(L − 1)). No sampling is involved.
- **Residual:** W = max(0, A − E), with a zero diagonal.
- **Grouping:** spectral grouping of W into C = 8 groups, with the same procedure and seed as the temporal arm.
- **Label-free diagnostic:** the residual share, Σ max(0, A − E) / Σ A.

## 3. Stage 1: LARa, reusing `ug_lara_001`

Everything except the grouping is reused from the unit-grouping run: the 24-state prototypes and the fresh fitting plans (saved as float16, rows renormalized), the 3 seeds, the primary split and the 4 subject-disjoint folds. Each cell:
1. re-decodes the evaluation recordings with the frozen prototypes;
2. recomputes the temporal, shuffled and appearance groupings;
3. adds the residual grouping and 100 random partitions size-matched to it;
4. scores everything with the original test's functions.

**Parity check:** the recomputed groupings and metrics must equal the saved ones. The smoke run already confirms identical groupings from the float16 plans. A metric mismatch is reported; it does not block the run.

## 4. Decision rule (declared before the run)

This is the unit-grouping v2 rule with `group_residual` as the hypothesis arm. The bar is a mean of at least +3 with all three seeds in the same direction; the floor is that the other metric's mean may not fall by more than 2.

`group_residual` must satisfy all of:
- **(a)** beat `b0_c` on both MoF and F1@50;
- **(b)** beat `group_appearance` on MoF or F1@50, with the floor on the other metric;
- **(c)** beat `group_shuffled` on MoF or F1@50, with the floor on the other metric;
- **(d)** in every seed, exceed the 95th percentile of its size-matched random partitions on both metrics.

| Outcome | Condition | Next step |
|---|---|---|
| **RESIDUAL_GROUPING** | (a)–(d) all hold | Stage 2 on BABEL-2 (Section 5). |
| **OVERCOMPLETE_ONLY** | (a) holds, (b) or (c) fails | Report; not a recording-invariance effect. |
| **PRIOR_MASS** | (a) holds, (d) fails | Report; the gain comes from group sizes. |
| **STOP** | (a) fails | This residual grouping does not fix the code-to-action step on LARa. |
| **INCONCLUSIVE** | missing cells or failed verification | Report. |

The outcomes are checked in this order: STOP, then PRIOR_MASS, then OVERCOMPLETE_ONLY.

**Reported, not gated:**
- residual minus temporal grouping;
- the subject-disjoint folds, as a descriptive readout;
- the residual share;
- ARI against the oracle;
- balanced accuracy;
- the number of classes that receive a group.

## 5. Stage 2: BABEL-2 (declared now; runs only after RESIDUAL_GROUPING)

The same method on BABEL-2 over-complete states (M = 3C = 15) for the current baseline input (MotionBERT, 243-frame clips) and for T2, with the 5 recording folds and 3 seeds, applying the same rule. BABEL-2 is development data, so this is a second-dataset check, not confirmation.

## 6. Limits

- Only adjacency **beyond within-recording occupancy** is kept. Any action signal carried by which actions a recording contains is removed along with the identity signal. That is intended, but it also discards some information.
- The smoke run gave a residual share of about 0.035: only about 3.5% of the affinity is not explained by recording membership. The method works with a small, possibly noisy signal.
- h = 8 and M = 24 are inherited, not tuned. The grouping assigns each state to exactly one action.

## 7. Run

```powershell
$env:PYTHONPATH = (Get-Location).Path
& "$env:USERPROFILE\AppData\Local\anaconda3\envs\smq\python.exe" -m script.unit_grouping.residual_run run --run-id rc_lara_001 --splits primary,fold0,fold1,fold2,fold3 *>&1 | Tee-Object rc.log
```

15 cells, each re-decoding about 44–120 held-out recordings: about 20–40 minutes. The tests (`python -m script.unit_grouping.tests`, 14 pass) write the verification record that the verdict requires.
