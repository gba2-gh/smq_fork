# Raw K-means tokens across datasets — rkx_v1

Manifest: `manifest.json` (status `finished_with_unfinished_units`, implementation `34647920d0acc6c9`). Cells with verified training: 0/18; unfinished units: 37 (`not_run.csv`).

Frozen HuGaDB configuration on every dataset (raw patches, K-means-512, reference contextualizer, guard 0, 5,000 updates, outer cap 50). Only window, frame rate and class count differ by dataset. Tokenizer and OT fitting pool all recordings, validation included, so decoding is transductive; HuGaDB settings were chosen on HuGaDB development scores; LARa is not untouched (it informed earlier choices).

## Gates

| dataset | labels = saved SMQ ground truth | max |metric diff| vs saved SMQ scores | passed |
|---|---|---|---|
| babel1 | True | 0.0026 | True |
| babel2 | True | 0.0043 | True |
| babel3 | True | 0.0020 | True |
| hugadb | True | 0.0049 | True |
| lara | True | 0.0041 | True |

**Gate 1 — tokenizer port vs qic_v3:** failed

**Gate 2 — pipeline port vs ctx_cal_v1:** failed

**A port-regression gate failed; dependent cells are not interpreted.** See `gates.json` and `gate_mismatch_environment.json`.

## F1@50 by dataset (mean ± sd over three paired seeds)

Validation-selected checkpoint (primary) and final 5,000-update checkpoint (secondary), last final-solve rung (4,000 in a full run). SMQ is the released checkpoint re-scored locally; the published numbers use different protocols (end-to-end training, 1-second patches, K = number of classes).

_no rows_

## Declared reading

**incomplete** (selected checkpoint, last rung, mean of three seeds).

- not every LARa normalization and BABEL subset has three verified seeds

The rule was fixed before scoring; an intermediate outcome is reported as mixed, and the normalization factor and checkpoint policy are shown as they came out.

## All metrics (mean over seeds, last rung)

_no rows_

## Per-cell diagnostics

_no rows_

The subject probe is a label-aware diagnostic of what the codes carry; it never entered selection.

## Missing cells and limits

| unit | status | reason |
|---|---|---|
| hugadb__global__seed111 | gate | tokenizer port regression failed |
| hugadb__global__seed111::selected | unavailable | cell did not train (gate: tokenizer port regression failed) |
| hugadb__global__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| hugadb__global__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| hugadb__global__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| hugadb__global__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| hugadb__global__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed111::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel1__global__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed111::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel2__global__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed111::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| babel3__global__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed111::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__global__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed111::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed111::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed222::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed222::final | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed1538574472::selected | not_run | a port-regression gate failed; dependent cells not run |
| lara__per_recording__seed1538574472::final | not_run | a port-regression gate failed; dependent cells not run |

Limits: three paired seeds (each drives the vocabulary and the contextualizer); the outer OT fit is budget-capped in essentially every fit; BABEL clips are short (mostly padding in a 128-token context); published numbers come from different protocols. This run does not tune anything on LARa or BABEL.
