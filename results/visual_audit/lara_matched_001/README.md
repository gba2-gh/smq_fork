# Matched LARa visual audit

18 cells: six arms, three seeds, the same 44 evaluation recordings.
Open index.html in a browser. All five pooled metrics reproduced the saved scores in every cell.

| Arm | Mean extra boundaries / recording | Mean missed boundaries / recording | Recordings with both (%) |
|---|---:|---:|---:|
| T2 B0 | 42.1 | 17.3 | 100.0 |
| Body-scale B0 | 72.8 | 13.7 | 100.0 |
| Centered B0 | 50.7 | 16.0 | 98.5 |
| Ordered GRU | 79.2 | 13.2 | 100.0 |
| Bag | 74.1 | 13.2 | 100.0 |
| Occupancy | 36.3 | 17.6 | 100.0 |

Boundary matching is class-agnostic, one-to-one, within 12 frames (0.24 s).
Extra and missed boundaries can both reflect displaced transitions; these are descriptive counts, not causal labels.
A high segment-count ratio alone does not establish that all errors are overfragmentation.

The oracle row uses evaluation-label many-to-one mapping; it is not an unsupervised result.
Colors identify actions consistently. Raw state IDs are local to each cell and cannot be compared across seeds.
Recordings are listed without filtering out failures. Default ordering is by name, not selected examples.
This first audit covers the latest matched LARa experiments, not all historical projects.
No synchronized skeleton replay is included yet; label timelines cannot establish a physical explanation.
The static first_recording.png export was visually inspected. Automated browser interaction verification was unavailable in this session.
