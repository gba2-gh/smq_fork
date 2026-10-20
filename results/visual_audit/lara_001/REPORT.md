# LARa visual inspection: generated measurements

[Open the atlas](index.html). All 44 matched evaluation recordings have motion pages. Historical pooled recordings are displayed with their own global mapping and protocol labels.

## Matched held-out decoder cells

Means over the three seeds. Boundary columns average over 44 recordings and three seeds at ±0.24 s. These are descriptive counts; a displaced transition can contribute to both columns.

| Arm | MoF | F1@50 | Pooled pred./GT runs | Extra / rec. | Missed / rec. |
|---|---:|---:|---:|---:|---:|
| T2 B0 | 38.9 | 19.7 | 1.82 | 42.1 | 17.3 |
| Body-scale B0 | 40.7 | 11.6 | 2.96 | 72.8 | 13.7 |
| Centered B0 | 41.8 | 15.7 | 2.15 | 50.7 | 16.0 |
| Ordered GRU | 38.4 | 13.4 | 3.19 | 79.2 | 13.2 |
| Bag | 38.6 | 14.1 | 3.02 | 74.1 | 13.2 |
| Occupancy | 38.6 | 21.1 | 1.62 | 36.3 | 17.6 |

## Earlier pooled LARa runs: separate protocol

These runs were fitted and mapped over 439 recordings. Their scores are not paired differences from the 44-recording decoder cells.

| Run | Seeds | Mean MoF | Mean F1@50 |
|---|---|---:|---:|
| Released SMQ | released | 37.4 | 16.4 |
| Stage C context only | 111, 222, 1538574472 | 20.2 | 5.0 |
| Stage C contextual ASOT | 111, 222, 1538574472 | 21.3 | 7.5 |
| Short mask T2 | 111, 222, 1538574472 | 35.4 | 24.0 |
| Phase 9 continuous I1 | 111, 222, 1538574472 | 37.3 | 25.2 |

## T2 B0 action recall, seed 111

The large Handling(centred) class makes aggregate MoF sensitive to how its states are mapped. Recall remains a separate per-class readout.

| Action | GT frames | Recall |
|---|---:|---:|
| Standing | 32643 | 32.3% |
| Walking | 27222 | 39.0% |
| Cart | 27821 | 38.8% |
| Handling(upwards) | 17580 | 50.8% |
| Handling(centred) | 121619 | 35.1% |
| Handling(downwards) | 19449 | 76.1% |
| Synchronization | 4579 | 94.2% |
| None | 13087 | 9.7% |

## Inspection coverage

The frozen list has 8 whole recordings and 24 action-stratified segments. All observations are marked pending until motion is reviewed by a person.

The grouping run reports a failed gate, but its evaluation predictions and costs were not saved. Its recording-level failures are therefore absent from this workspace.

Source features were hip centered during preparation, so absolute translation cannot be inspected. The viewer shows verified position channels 3–5 at 25 displayed frames/s; action/timing arrays remain at 50 frames/s. No unverified joint connections are drawn.

The local decision is argmin of each saved window-cost row, mapped with the same final global Hungarian mapping. It does not represent a separate trained model without OT.

Original matched-cell metrics and window-code alignment were checked for every cell. Browser screenshots verified the main layout; the automatic control checks cover seed, method, display, frame navigation and atlas ordering. Human motion interpretation remains pending.
