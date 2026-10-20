# Time-scale inspection, phase 0 — ts0_v1

Source run `rkx_v2` (snapshot of 33 verified decodes; the source may still be running). Implementation `05d9d98f016268f0`. Everything below is label-aware diagnostics; nothing here selects a setting.

## 0.1 Token-grid ceilings (perfect token labels)

| dataset | window (frames) | token (s) | current | MoF | Edit | F1@10 | F1@25 | F1@50 | identity mapping |
|---|---|---|---|---|---|---|---|---|---|
| hugadb | 15 | 0.25 | True | 98.85 | 100.0 | 100.0 | 100.0 | 100.0 | True |
| hugadb | 60 | 1.0 | False | 95.48 | 99.52 | 99.7 | 99.7 | 97.68 | True |
| babel1 | 4 | 0.1333 | False | 98.62 | 99.22 | 99.4 | 99.4 | 99.34 | True |
| babel1 | 8 | 0.2667 | True | 97.21 | 97.64 | 98.24 | 98.24 | 97.41 | True |
| babel1 | 15 | 0.5 | False | 94.88 | 93.64 | 95.43 | 95.41 | 92.63 | True |
| babel1 | 30 | 1.0 | False | 90.17 | 83.88 | 88.43 | 88.2 | 81.35 | True |
| babel2 | 4 | 0.1333 | False | 99.13 | 99.31 | 99.21 | 99.21 | 98.78 | True |
| babel2 | 8 | 0.2667 | True | 98.25 | 97.86 | 97.97 | 97.97 | 96.9 | True |
| babel2 | 15 | 0.5 | False | 96.82 | 96.38 | 95.97 | 95.86 | 94.23 | True |
| babel2 | 30 | 1.0 | False | 93.85 | 91.2 | 91.29 | 91.06 | 86.62 | True |
| babel3 | 4 | 0.1333 | False | 99.39 | 99.8 | 99.48 | 99.48 | 99.03 | True |
| babel3 | 8 | 0.2667 | True | 98.8 | 99.07 | 98.2 | 98.05 | 97.44 | True |
| babel3 | 15 | 0.5 | False | 97.79 | 96.64 | 95.85 | 95.54 | 93.23 | True |
| babel3 | 30 | 1.0 | False | 95.93 | 90.62 | 90.36 | 90.04 | 85.67 | True |
| lara | 12 | 0.24 | True | 98.45 | 99.9 | 99.96 | 99.96 | 99.87 | True |
| lara | 50 | 1.0 | False | 93.83 | 94.57 | 96.74 | 96.73 | 93.17 | True |

## 0.2 Smoothing ceilings (mode filter on perfect token labels, current grid)

`fixed` radii are in tokens (seconds shown); `relative` is floor(fraction × tokens in the recording), `relative_floor1` floors it at 1 token. `current` marks the radius the OT kernel uses today.

| dataset | rule | value | radius (s) | current | median radius (tokens) | radius 0 share | MoF | Edit | F1@50 |
|---|---|---|---|---|---|---|---|---|---|
| hugadb | none | 0 |  |  | n/a | n/a | 98.85 | 100.0 | 100.0 |
| hugadb | fixed | 1 | 0.25 | False | 1.0 | 0.00 | 98.85 | 100.0 | 100.0 |
| hugadb | fixed | 2 | 0.5 | False | 2.0 | 0.00 | 98.84 | 99.83 | 99.92 |
| hugadb | fixed | 3 | 0.75 | False | 3.0 | 0.00 | 98.83 | 99.72 | 99.82 |
| hugadb | fixed | 4 | 1.0 | True | 4.0 | 0.00 | 98.67 | 99.53 | 98.19 |
| hugadb | relative | 0.02 |  | False | 3.0 | 0.05 | 98.62 | 99.05 | 98.97 |
| hugadb | relative | 0.04 |  | False | 7.0 | 0.01 | 94.41 | 92.17 | 88.57 |
| hugadb | relative | 0.08 |  | False | 15.0 | 0.00 | 78.04 | 63.87 | 51.19 |
| hugadb | relative_floor1 | 0.02 |  | False | 3.0 | 0.00 | 98.62 | 99.05 | 98.97 |
| hugadb | relative_floor1 | 0.04 |  | False | 7.0 | 0.00 | 94.41 | 92.17 | 88.57 |
| hugadb | relative_floor1 | 0.08 |  | False | 15.0 | 0.00 | 78.04 | 63.87 | 51.19 |
| babel1 | none | 0 |  |  | n/a | n/a | 97.21 | 97.64 | 97.41 |
| babel1 | fixed | 1 | 0.267 | False | 1.0 | 0.00 | 97.1 | 96.34 | 95.99 |
| babel1 | fixed | 2 | 0.533 | False | 2.0 | 0.00 | 95.79 | 86.93 | 88.8 |
| babel1 | fixed | 3 | 0.8 | True | 3.0 | 0.00 | 93.77 | 81.36 | 79.18 |
| babel1 | fixed | 4 | 1.067 | False | 4.0 | 0.00 | 91.12 | 74.79 | 70.92 |
| babel1 | relative | 0.02 |  | False | 0.0 | 0.83 | 96.98 | 96.62 | 95.8 |
| babel1 | relative | 0.04 |  | False | 1.0 | 0.43 | 96.02 | 94.65 | 92.08 |
| babel1 | relative | 0.08 |  | False | 2.0 | 0.02 | 92.44 | 87.94 | 82.82 |
| babel1 | relative_floor1 | 0.02 |  | False | 1.0 | 0.00 | 96.92 | 95.78 | 95.01 |
| babel1 | relative_floor1 | 0.04 |  | False | 1.0 | 0.00 | 95.99 | 94.2 | 91.75 |
| babel1 | relative_floor1 | 0.08 |  | False | 2.0 | 0.00 | 92.44 | 87.89 | 82.8 |
| babel2 | none | 0 |  |  | n/a | n/a | 98.25 | 97.86 | 96.9 |
| babel2 | fixed | 1 | 0.267 | False | 1.0 | 0.00 | 98.13 | 96.35 | 94.75 |
| babel2 | fixed | 2 | 0.533 | False | 2.0 | 0.00 | 97.29 | 90.81 | 87.46 |
| babel2 | fixed | 3 | 0.8 | True | 3.0 | 0.00 | 96.23 | 88.81 | 81.46 |
| babel2 | fixed | 4 | 1.067 | False | 4.0 | 0.00 | 95.03 | 84.95 | 78.28 |
| babel2 | relative | 0.02 |  | False | 0.0 | 0.83 | 97.92 | 96.93 | 92.76 |
| babel2 | relative | 0.04 |  | False | 1.0 | 0.25 | 96.85 | 94.96 | 87.9 |
| babel2 | relative | 0.08 |  | False | 2.0 | 0.10 | 95.28 | 91.44 | 82.85 |
| babel2 | relative_floor1 | 0.02 |  | False | 1.0 | 0.00 | 97.88 | 95.86 | 92.05 |
| babel2 | relative_floor1 | 0.04 |  | False | 1.0 | 0.00 | 96.82 | 94.18 | 87.4 |
| babel2 | relative_floor1 | 0.08 |  | False | 2.0 | 0.00 | 95.26 | 90.66 | 82.32 |
| babel3 | none | 0 |  |  | n/a | n/a | 98.8 | 99.07 | 97.44 |
| babel3 | fixed | 1 | 0.267 | False | 1.0 | 0.00 | 98.7 | 97.29 | 94.93 |
| babel3 | fixed | 2 | 0.533 | False | 2.0 | 0.00 | 98.14 | 89.91 | 87.04 |
| babel3 | fixed | 3 | 0.8 | True | 3.0 | 0.00 | 97.29 | 87.45 | 78.89 |
| babel3 | fixed | 4 | 1.067 | False | 4.0 | 0.00 | 96.2 | 81.25 | 77.64 |
| babel3 | relative | 0.02 |  | False | 0.0 | 0.55 | 98.56 | 97.42 | 93.32 |
| babel3 | relative | 0.04 |  | False | 1.0 | 0.17 | 98.08 | 93.64 | 89.62 |
| babel3 | relative | 0.08 |  | False | 3.0 | 0.02 | 96.19 | 85.39 | 78.88 |
| babel3 | relative_floor1 | 0.02 |  | False | 1.0 | 0.00 | 98.51 | 96.43 | 92.01 |
| babel3 | relative_floor1 | 0.04 |  | False | 1.0 | 0.00 | 98.08 | 93.64 | 89.62 |
| babel3 | relative_floor1 | 0.08 |  | False | 3.0 | 0.00 | 96.19 | 85.39 | 78.88 |
| lara | none | 0 |  |  | n/a | n/a | 98.45 | 99.9 | 99.87 |
| lara | fixed | 1 | 0.24 | False | 1.0 | 0.00 | 98.44 | 99.35 | 99.51 |
| lara | fixed | 2 | 0.48 | False | 2.0 | 0.00 | 98.28 | 96.86 | 97.71 |
| lara | fixed | 3 | 0.72 | False | 3.0 | 0.00 | 97.72 | 91.74 | 93.76 |
| lara | fixed | 4 | 0.96 | True | 4.0 | 0.00 | 96.92 | 87.76 | 88.08 |
| lara | relative | 0.02 |  | False | 10.0 | 0.00 | 86.07 | 57.35 | 51.26 |
| lara | relative | 0.04 |  | False | 20.0 | 0.00 | 74.7 | 36.04 | 23.51 |
| lara | relative | 0.08 |  | False | 40.0 | 0.00 | 64.98 | 22.77 | 9.07 |
| lara | relative_floor1 | 0.02 |  | False | 10.0 | 0.00 | 86.07 | 57.35 | 51.26 |
| lara | relative_floor1 | 0.04 |  | False | 20.0 | 0.00 | 74.7 | 36.04 | 23.51 |
| lara | relative_floor1 | 0.08 |  | False | 40.0 | 0.00 | 64.98 | 22.77 | 9.07 |

## 0.3 Failure profile of the saved predictions

`F1@50` is recomputed from saved predictions and must equal the saved last-rung value. `no none` drops ground-truth `none` frames from both sequences before scoring (this splices segments, so read it as indicative).

| dataset | norm | seed | checkpoint | F1@50 | saved | GT segs/rec | pred segs/rec | GT median s | pred median s | GT <1 s | pred <1 s | errors involving none | F1@50 no none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hugadb | global | 111 | final | 48.64 | 48.64 | 10.5 | 9.4 | 3.47 | 3.75 | 0.04 | 0.08 | n/a |  |
| hugadb | global | 111 | selected | 50.0 | 50.0 | 10.5 | 9.3 | 3.47 | 3.75 | 0.04 | 0.08 | n/a |  |
| hugadb | global | 1538574472 | final | 43.85 | 43.85 | 10.5 | 9.0 | 3.47 | 3.75 | 0.04 | 0.05 | n/a |  |
| hugadb | global | 1538574472 | selected | 43.85 | 43.85 | 10.5 | 9.0 | 3.47 | 3.75 | 0.04 | 0.05 | n/a |  |
| hugadb | global | 222 | final | 44.71 | 44.71 | 10.5 | 8.6 | 3.47 | 3.75 | 0.04 | 0.05 | n/a |  |
| hugadb | global | 222 | selected | 44.83 | 44.83 | 10.5 | 8.7 | 3.47 | 3.75 | 0.04 | 0.05 | n/a |  |
| babel1 | global | 111 | final | 16.91 | 16.91 | 5.1 | 2.5 | 1.20 | 2.95 | 0.40 | 0.01 | 0.41 | 22.43 |
| babel1 | global | 111 | selected | 15.89 | 15.89 | 5.1 | 2.5 | 1.20 | 3.00 | 0.40 | 0.01 | 0.39 | 20.81 |
| babel1 | global | 1538574472 | final | 13.43 | 13.43 | 5.1 | 2.5 | 1.20 | 2.93 | 0.40 | 0.03 | 0.41 | 17.68 |
| babel1 | global | 1538574472 | selected | 13.64 | 13.64 | 5.1 | 2.5 | 1.20 | 2.93 | 0.40 | 0.02 | 0.41 | 18.0 |
| babel1 | global | 222 | final | 13.11 | 13.11 | 5.1 | 2.6 | 1.20 | 2.93 | 0.40 | 0.02 | 0.44 | 18.53 |
| babel1 | global | 222 | selected | 13.21 | 13.21 | 5.1 | 2.6 | 1.20 | 2.93 | 0.40 | 0.02 | 0.44 | 18.65 |
| babel2 | global | 111 | final | 18.54 | 18.54 | 3.7 | 2.9 | 1.70 | 2.77 | 0.29 | 0.04 | 0.43 | 11.03 |
| babel2 | global | 111 | selected | 16.68 | 16.68 | 3.7 | 2.3 | 1.70 | 3.47 | 0.29 | 0.04 | 0.43 | 13.91 |
| babel2 | global | 1538574472 | final | 16.01 | 16.01 | 3.7 | 3.0 | 1.70 | 2.67 | 0.29 | 0.05 | 0.53 | 19.26 |
| babel2 | global | 1538574472 | selected | 17.46 | 17.46 | 3.7 | 1.9 | 1.70 | 4.27 | 0.29 | 0.01 | 0.44 | 20.77 |
| babel2 | global | 222 | final | 18.18 | 18.18 | 3.7 | 2.9 | 1.70 | 2.77 | 0.29 | 0.03 | 0.50 | 15.95 |
| babel2 | global | 222 | selected | 11.12 | 11.12 | 3.7 | 1.7 | 1.70 | 4.80 | 0.29 | 0.00 | 0.45 | 19.85 |
| babel3 | global | 111 | final | 14.05 | 14.05 | 3.7 | 3.6 | 1.77 | 3.47 | 0.30 | 0.01 | 0.46 | 8.33 |
| babel3 | global | 111 | selected | 14.78 | 14.78 | 3.7 | 2.7 | 1.77 | 4.18 | 0.30 | 0.02 | 0.45 | 10.88 |
| babel3 | global | 1538574472 | final | 10.97 | 10.97 | 3.7 | 3.4 | 1.77 | 3.60 | 0.30 | 0.03 | 0.42 | 7.96 |
| babel3 | global | 1538574472 | selected | 15.15 | 15.15 | 3.7 | 2.8 | 1.77 | 4.00 | 0.30 | 0.02 | 0.40 | 12.48 |
| babel3 | global | 222 | final | 11.02 | 11.02 | 3.7 | 3.7 | 1.77 | 3.23 | 0.30 | 0.02 | 0.40 | 6.62 |
| babel3 | global | 222 | selected | 16.04 | 16.04 | 3.7 | 2.5 | 1.77 | 4.52 | 0.30 | 0.01 | 0.45 | 15.16 |
| lara | global | 111 | final | 5.03 | 5.03 | 32.1 | 15.1 | 2.20 | 5.52 | 0.14 | 0.00 | n/a |  |
| lara | global | 111 | selected | 5.03 | 5.03 | 32.1 | 15.0 | 2.20 | 5.28 | 0.14 | 0.00 | n/a |  |
| lara | global | 1538574472 | final | 5.71 | 5.71 | 32.1 | 14.9 | 2.20 | 5.28 | 0.14 | 0.00 | n/a |  |
| lara | global | 1538574472 | selected | 5.55 | 5.55 | 32.1 | 15.0 | 2.20 | 5.28 | 0.14 | 0.00 | n/a |  |
| lara | global | 222 | final | 4.67 | 4.67 | 32.1 | 14.4 | 2.20 | 5.52 | 0.14 | 0.00 | n/a |  |
| lara | global | 222 | selected | 4.6 | 4.6 | 32.1 | 14.3 | 2.20 | 5.52 | 0.14 | 0.00 | n/a |  |
| lara | per_recording | 111 | final | 7.32 | 7.32 | 32.1 | 18.4 | 2.20 | 4.80 | 0.14 | 0.00 | n/a |  |
| lara | per_recording | 111 | selected | 7.32 | 7.32 | 32.1 | 18.4 | 2.20 | 4.80 | 0.14 | 0.00 | n/a |  |
| lara | per_recording | 222 | selected | 7.41 | 7.41 | 32.1 | 18.6 | 2.20 | 4.80 | 0.14 | 0.00 | n/a |  |

### Ground-truth segments by outcome (selected checkpoint, mean over seeds)

`hit`: a same-label predicted segment with IoU ≥ 0.5. Otherwise, by the majority predicted label over the segment: `boundary` (own label, wrong extent), `merged` (a neighbour's label), `wrong_label` (any other).

| dataset | norm | GT length | segments | hit | boundary | merged | wrong label |
|---|---|---|---|---|---|---|---|
| hugadb | global | 1-2s | 660 | 0.21 | 0.06 | 0.48 | 0.26 |
| hugadb | global | <1s | 162 | 0.02 | 0.07 | 0.66 | 0.25 |
| hugadb | global | >=2s | 2999 | 0.50 | 0.06 | 0.06 | 0.37 |
| hugadb | global | all | 3821 | 0.43 | 0.06 | 0.16 | 0.35 |
| babel1 | global | 1-2s | 3236 | 0.12 | 0.22 | 0.27 | 0.39 |
| babel1 | global | <1s | 3968 | 0.01 | 0.20 | 0.41 | 0.38 |
| babel1 | global | >=2s | 2722 | 0.24 | 0.11 | 0.22 | 0.43 |
| babel1 | global | all | 9926 | 0.11 | 0.18 | 0.31 | 0.40 |
| babel2 | global | 1-2s | 282 | 0.06 | 0.20 | 0.24 | 0.50 |
| babel2 | global | <1s | 276 | 0.00 | 0.18 | 0.29 | 0.53 |
| babel2 | global | >=2s | 396 | 0.23 | 0.08 | 0.22 | 0.47 |
| babel2 | global | all | 954 | 0.12 | 0.15 | 0.24 | 0.49 |
| babel3 | global | 1-2s | 161 | 0.10 | 0.25 | 0.18 | 0.47 |
| babel3 | global | <1s | 206 | 0.01 | 0.16 | 0.28 | 0.55 |
| babel3 | global | >=2s | 310 | 0.23 | 0.13 | 0.17 | 0.47 |
| babel3 | global | all | 677 | 0.13 | 0.17 | 0.21 | 0.49 |
| lara | global | 1-2s | 4117 | 0.01 | 0.14 | 0.25 | 0.61 |
| lara | global | <1s | 2024 | 0.00 | 0.17 | 0.21 | 0.62 |
| lara | global | >=2s | 7964 | 0.06 | 0.11 | 0.22 | 0.61 |
| lara | global | all | 14105 | 0.04 | 0.13 | 0.22 | 0.61 |
| lara | per_recording | 1-2s | 4117 | 0.02 | 0.15 | 0.25 | 0.59 |
| lara | per_recording | <1s | 2024 | 0.00 | 0.18 | 0.23 | 0.59 |
| lara | per_recording | >=2s | 7964 | 0.09 | 0.11 | 0.21 | 0.58 |
| lara | per_recording | all | 14105 | 0.06 | 0.13 | 0.23 | 0.58 |

### Per class (selected checkpoint, mean over seeds)

| dataset | norm | class | frame share | median GT segment s | frame recall | frame precision | segment hit rate |
|---|---|---|---|---|---|---|---|
| hugadb | global | walking | 0.28 | 3.42 | 0.43 | 0.96 | 0.45 |
| hugadb | global | running | 0.06 | 13.23 | 0.78 | 0.58 | 0.80 |
| hugadb | global | going_up | 0.12 | 5.15 | 0.52 | 0.62 | 0.51 |
| hugadb | global | going_down | 0.11 | 4.20 | 0.47 | 0.65 | 0.49 |
| hugadb | global | sitting | 0.07 | 3.33 | 0.99 | 0.68 | 0.99 |
| hugadb | global | sitting_down | 0.02 | 1.08 | 0.00 | 0.00 | 0.00 |
| hugadb | global | standing_up | 0.02 | 1.15 | 0.03 | 0.01 | 0.00 |
| hugadb | global | standing | 0.20 | 2.70 | 0.64 | 0.52 | 0.45 |
| hugadb | global | up_by_elevator | 0.08 | 12.28 | 0.13 | 0.16 | 0.00 |
| hugadb | global | down_by_elevator | 0.06 | 11.52 | 0.09 | 0.09 | 0.02 |
| babel1 | global | walk | 0.42 | 1.87 | 0.36 | 0.51 | 0.21 |
| babel1 | global | stand | 0.25 | 1.03 | 0.35 | 0.33 | 0.08 |
| babel1 | global | turn | 0.07 | 1.07 | 0.35 | 0.15 | 0.05 |
| babel1 | global | jump | 0.03 | 1.47 | 0.15 | 0.05 | 0.06 |
| babel1 | global | none | 0.23 | 0.77 | 0.20 | 0.32 | 0.06 |
| babel2 | global | sit | 0.51 | 5.00 | 0.34 | 0.70 | 0.24 |
| babel2 | global | run | 0.09 | 1.50 | 0.30 | 0.16 | 0.09 |
| babel2 | global | stand_up | 0.10 | 1.42 | 0.30 | 0.14 | 0.05 |
| babel2 | global | kick | 0.04 | 1.47 | 0.32 | 0.06 | 0.06 |
| babel2 | global | none | 0.27 | 1.23 | 0.32 | 0.44 | 0.09 |
| babel3 | global | jog | 0.06 | 1.87 | 0.37 | 0.14 | 0.11 |
| babel3 | global | wave | 0.02 | 1.93 | 0.43 | 0.05 | 0.21 |
| babel3 | global | dance | 0.43 | 10.97 | 0.31 | 0.62 | 0.18 |
| babel3 | global | gesture | 0.26 | 4.47 | 0.25 | 0.29 | 0.09 |
| babel3 | global | none | 0.23 | 1.03 | 0.41 | 0.37 | 0.13 |
| lara | global | Standing | 0.11 | 1.24 | 0.20 | 0.15 | 0.02 |
| lara | global | Walking | 0.12 | 2.96 | 0.20 | 0.18 | 0.08 |
| lara | global | Cart | 0.12 | 6.04 | 0.15 | 0.17 | 0.06 |
| lara | global | Handling(upwards) | 0.08 | 2.20 | 0.16 | 0.10 | 0.03 |
| lara | global | Handling(centred) | 0.44 | 2.50 | 0.24 | 0.62 | 0.03 |
| lara | global | Handling(downwards) | 0.07 | 2.20 | 0.17 | 0.12 | 0.05 |
| lara | global | Synchronization | 0.02 | 2.06 | 0.09 | 0.02 | 0.03 |
| lara | global | None | 0.04 | 1.80 | 0.18 | 0.07 | 0.03 |
| lara | per_recording | Standing | 0.11 | 1.24 | 0.23 | 0.17 | 0.02 |
| lara | per_recording | Walking | 0.12 | 2.96 | 0.19 | 0.15 | 0.08 |
| lara | per_recording | Cart | 0.12 | 6.04 | 0.18 | 0.17 | 0.08 |
| lara | per_recording | Handling(upwards) | 0.08 | 2.20 | 0.20 | 0.10 | 0.05 |
| lara | per_recording | Handling(centred) | 0.44 | 2.50 | 0.25 | 0.64 | 0.06 |
| lara | per_recording | Handling(downwards) | 0.07 | 2.20 | 0.29 | 0.23 | 0.13 |
| lara | per_recording | Synchronization | 0.02 | 2.06 | 0.12 | 0.02 | 0.03 |
| lara | per_recording | None | 0.04 | 1.80 | 0.09 | 0.05 | 0.03 |

### By recording length tertile (selected checkpoint, mean over seeds)

| dataset | norm | tertile | median length s | F1@50 |
|---|---|---|---|---|
| hugadb | global | 1 | 28.8 | 50.34 |
| hugadb | global | 2 | 47.2 | 49.12 |
| hugadb | global | 3 | 79.0 | 40.99 |
| babel1 | global | 1 | 5.0 | 12.32 |
| babel1 | global | 2 | 7.0 | 20.66 |
| babel1 | global | 3 | 14.1 | 11.14 |
| babel2 | global | 1 | 4.7 | 11.87 |
| babel2 | global | 2 | 8.5 | 14.48 |
| babel2 | global | 3 | 13.6 | 17.08 |
| babel3 | global | 1 | 6.4 | 11.40 |
| babel3 | global | 2 | 11.4 | 16.78 |
| babel3 | global | 3 | 28.2 | 16.51 |
| lara | global | 1 | 120.0 | 5.06 |
| lara | per_recording | 1 | 120.0 | 7.37 |

## 0.4 Action information per token (balanced accuracy, grouped split, mean over seeds)

`pca`: the patch's PCA-64 feature; `centre`: its K-means centre; `onehot`: the code (the `nn` row is the per-code majority label); `context`: the selected checkpoint's contextual embedding.

| dataset | norm | probe | pca | centre | onehot | context | seeds |
|---|---|---|---|---|---|---|---|
| hugadb | global | linear | 0.609 | 0.577 | 0.550 | 0.634 | 3 |
| hugadb | global | nn | 0.682 | 0.504 | 0.582 | 0.651 | 3 |
| babel1 | global | linear | 0.455 | 0.436 | 0.455 | 0.460 | 3 |
| babel1 | global | nn | 0.515 | 0.377 | 0.463 | 0.513 | 3 |
| babel2 | global | linear | 0.422 | 0.413 | 0.418 | 0.367 | 3 |
| babel2 | global | nn | 0.520 | 0.477 | 0.478 | 0.461 | 3 |
| babel3 | global | linear | 0.379 | 0.355 | 0.345 | 0.323 | 3 |
| babel3 | global | nn | 0.420 | 0.333 | 0.380 | 0.380 | 3 |
| lara | global | linear | 0.464 | 0.355 | 0.330 | 0.271 | 3 |
| lara | global | nn | 0.445 | 0.323 | 0.343 | 0.338 | 3 |
| lara | per_recording | linear | 0.444 | 0.340 | 0.341 | 0.325 | 2 |
| lara | per_recording | nn | 0.422 | 0.304 | 0.359 | 0.365 | 2 |

## 0.4 Masked-prediction loss by boundary crossing (validation masks, selected checkpoint)

A target is `crossing` when the contiguous masked run containing it spans a ground-truth label change. `all` must match the saved validation loss of the selected step.

| dataset | norm | seed | step | share crossing | loss within | loss crossing | top-1 within | top-1 crossing | loss all | saved |
|---|---|---|---|---|---|---|---|---|---|---|
| hugadb | global | 111 | 4100 | 0.39 | 1.468 | 2.349 | 0.630 | 0.403 | 1.8097 | 1.8097 |
| hugadb | global | 1538574472 | 5000 | 0.39 | 1.446 | 2.438 | 0.623 | 0.400 | 1.8304 | 1.8304 |
| hugadb | global | 222 | 4900 | 0.39 | 1.438 | 2.412 | 0.631 | 0.388 | 1.8154 | 1.8154 |
| babel1 | global | 111 | 4800 | 0.53 | 3.031 | 3.139 | 0.322 | 0.259 | 3.0885 | 3.0885 |
| babel1 | global | 1538574472 | 4300 | 0.53 | 3.063 | 3.049 | 0.312 | 0.275 | 3.0557 | 3.0557 |
| babel1 | global | 222 | 4600 | 0.53 | 3.044 | 2.996 | 0.320 | 0.266 | 3.0186 | 3.0186 |
| babel2 | global | 111 | 1100 | 0.17 | 3.403 | 6.054 | 0.440 | 0.019 | 3.8561 | 3.8561 |
| babel2 | global | 1538574472 | 800 | 0.17 | 3.887 | 6.514 | 0.389 | 0.038 | 4.3361 | 4.3361 |
| babel2 | global | 222 | 800 | 0.17 | 3.299 | 5.938 | 0.440 | 0.019 | 3.7503 | 3.7503 |
| babel3 | global | 111 | 600 | 0.29 | 4.540 | 5.118 | 0.177 | 0.134 | 4.7104 | 4.7104 |
| babel3 | global | 1538574472 | 900 | 0.29 | 4.620 | 5.460 | 0.121 | 0.093 | 4.8680 | 4.8680 |
| babel3 | global | 222 | 700 | 0.29 | 4.735 | 4.846 | 0.155 | 0.144 | 4.7679 | 4.7679 |
| lara | global | 111 | 4700 | 0.43 | 1.526 | 2.296 | 0.647 | 0.441 | 1.8547 | 1.8547 |
| lara | global | 1538574472 | 4900 | 0.43 | 1.533 | 2.378 | 0.657 | 0.430 | 1.8934 | 1.8934 |
| lara | global | 222 | 4900 | 0.43 | 1.598 | 2.331 | 0.615 | 0.417 | 1.9107 | 1.9107 |
| lara | per_recording | 111 | 5000 | 0.43 | 2.190 | 2.832 | 0.502 | 0.358 | 2.4638 | 2.4638 |
| lara | per_recording | 222 | 4600 | 0.43 | 2.326 | 2.844 | 0.494 | 0.366 | 2.5467 | 2.5467 |
