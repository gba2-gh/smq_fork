# Retrieval examples — babel1, fold 0 evaluation set, continuous 8-bin representation

Query anchors: first 2 per class in SHA-256 order of (recording, timestamp) — not selected by outcome. Top-3 nearest gallery anchors' segments (Euclidean), gallery excludes the query's recording. ✔ = same action label as the query anchor.


## fixed segmentation

| query (recording, t) | label | query segment (s) | top-1 | top-2 | top-3 | hits |
|---|---|---|---|---|---|---|
| 8811, 7.5s | walk | 7.0–8.0 | 12936 [1.0–2.0s] walk ✔ | 12936 [2.0–3.0s] walk ✔ | 12570 [1.0–2.0s] walk ✔ | 3/3 |
| 10215, 3.2s | walk | 3.0–4.0 | 10449 [2.0–3.0s] none ✘ | 3524 [2.0–3.0s] none ✘ | 8187 [4.0–5.0s] walk ✔ | 1/3 |
| 7101, 18.0s | stand | 18.0–18.6 | 5020 [4.0–4.6s] stand ✔ | 1446 [6.0–6.6s] none ✘ | 7944 [3.0–3.5s] none ✘ | 1/3 |
| 5471, 14.6s | stand | 14.0–15.0 | 11014 [4.0–5.0s] none ✘ | 11754 [5.0–6.0s] stand ✔ | 11754 [16.0–17.0s] stand ✔ | 2/3 |
| 5717, 21.7s | turn | 21.0–22.0 | 4821 [3.0–4.0s] stand ✘ | 4821 [4.0–5.0s] none ✘ | 2403 [6.0–7.0s] stand ✘ | 0/3 |
| 6379, 2.2s | turn | 2.0–3.0 | 3839 [5.0–6.0s] stand ✘ | 5272 [2.0–3.0s] stand ✘ | 3839 [4.0–5.0s] stand ✘ | 0/3 |
| 9238, 3.6s | jump | 3.0–4.0 | 3936 [1.0–2.0s] stand ✘ | 6842 [20.0–21.0s] stand ✘ | 3168 [6.0–7.0s] walk ✘ | 0/3 |
| 3608, 0.8s | jump | 0.0–1.0 | 7595 [0.0–1.0s] walk ✘ | 8064 [0.0–1.0s] walk ✘ | 598 [0.0–1.0s] stand ✘ | 0/3 |
| 8786, 1.8s | none | 1.0–2.0 | 8254 [23.0–24.0s] walk ✘ | 8254 [24.0–25.0s] none ✔ | 8254 [22.0–23.0s] walk ✘ | 1/3 |
| 11289, 3.7s | none | 3.0–4.0 | 8562 [3.0–4.0s] turn ✘ | 12381 [4.0–5.0s] walk ✘ | 963 [4.0–5.0s] stand ✘ | 0/3 |

## oracle segmentation

| query (recording, t) | label | query segment (s) | top-1 | top-2 | top-3 | hits |
|---|---|---|---|---|---|---|
| 8811, 7.5s | walk | 1.5–16.2 | 5700 [5.2–6.8s] turn ✘ | 5700 [5.2–6.8s] turn ✘ | 8856 [4.2–5.5s] turn ✘ | 0/3 |
| 10215, 3.2s | walk | 3.0–10.7 | 8187 [3.0–9.3s] walk ✔ | 8187 [3.0–9.3s] walk ✔ | 8187 [3.0–9.3s] walk ✔ | 3/3 |
| 7101, 18.0s | stand | 17.7–18.6 | 5198 [6.2–7.1s] stand ✔ | 7681 [5.3–6.2s] stand ✔ | 11028 [5.5–6.2s] stand ✔ | 3/3 |
| 5471, 14.6s | stand | 11.7–15.6 | 8268 [3.0–28.6s] stand ✔ | 8268 [3.0–28.6s] stand ✔ | 8268 [3.0–28.6s] stand ✔ | 3/3 |
| 5717, 21.7s | turn | 20.8–22.0 | 4821 [3.3–4.6s] stand ✘ | 4821 [4.6–4.9s] none ✘ | 10507 [4.5–6.2s] stand ✘ | 0/3 |
| 6379, 2.2s | turn | 0.0–2.8 | 6842 [0.0–3.2s] stand ✘ | 6842 [0.0–3.2s] stand ✘ | 6842 [0.0–3.2s] stand ✘ | 0/3 |
| 9238, 3.6s | jump | 2.6–6.2 | 11006 [7.0–9.9s] stand ✘ | 11006 [7.0–9.9s] stand ✘ | 11006 [7.0–9.9s] stand ✘ | 0/3 |
| 3608, 0.8s | jump | 0.7–2.3 | 12936 [0.0–22.9s] walk ✘ | 12936 [0.0–22.9s] walk ✘ | 12936 [0.0–22.9s] walk ✘ | 0/3 |
| 8786, 1.8s | none | 0.5–2.6 | 9850 [1.2–2.9s] none ✔ | 9850 [1.2–2.9s] none ✔ | 3168 [0.4–8.8s] walk ✘ | 2/3 |
| 11289, 3.7s | none | 3.6–4.1 | 5539 [3.7–4.6s] walk ✘ | 4205 [4.9–5.8s] none ✔ | 7668 [4.9–6.7s] stand ✘ | 1/3 |