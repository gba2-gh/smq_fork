# LARa claim audit · before human motion review

| Claim | Current measurement | Status | Next visual check |
|---|---|---|---|
| Excess predicted boundaries | All matched arms have excess boundaries; counts and tolerances are in recordings.csv. | Observed count pattern | Check whether switches recur inside actions away from GT edges. |
| Missing boundaries | The same runs also miss GT boundaries under one-to-one matching. Displaced transitions can appear in both counts. | Observed count pattern | Inspect GT transitions with local cost and final OT rows. |
| Multiple states within one action | Saved confusion matrices and oracle relabeling expose mapping granularity. | Observed association; semantic meaning unresolved | Inspect repeatability of movement phases across recordings and seeds. |
| Subject/recording cues drive errors | Current viewer can sort by subject and recording. Association does not identify a physical cause. | Unresolved | Compare action mix, motion and acquisition across affected recordings. |
| Context corrects unit evidence | Historical contextual runs and the matched GRU run have different protocols/objectives; model-specific before/after costs exist only for matched arms. | Unresolved | Inspect paired local/final rows and compare matched contextual settings separately. |
| Grouping repairs action identity | Grouping gate failed; evaluation predictions were not saved. | Contradicted for tested spectral grouping at aggregate level; recording mechanism unavailable | Save evaluation states in any future grouping test before making a visual claim. |

These statuses describe saved measurements. Pending observation rows are not evidence of visible movement motifs.
