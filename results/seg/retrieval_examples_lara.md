# Retrieval examples — lara, fold 0 evaluation set, continuous 8-bin representation

Query anchors: first 2 per class in SHA-256 order of (recording, timestamp) — not selected by outcome. Top-3 nearest gallery anchors' segments (Euclidean), gallery excludes the query's participant. ✔ = same action label as the query anchor.


## fixed segmentation

| query (recording, t) | label | query segment (s) | top-1 | top-2 | top-3 | hits |
|---|---|---|---|---|---|---|
| L01_S01_R09, 29.8s | Standing | 29.0–30.0 | L03_S14_R17 [38.0–39.0s] Standing ✔ | L03_S14_R17 [39.0–40.0s] Standing ✔ | L03_S14_R17 [36.0–37.0s] Walking ✘ | 2/3 |
| L01_S03_R13, 24.6s | Standing | 24.0–25.0 | L01_S01_R28 [8.0–9.0s] Standing ✔ | L01_S01_R06 [47.0–48.0s] Walking ✘ | L01_S01_R05 [8.0–9.0s] Handling(centred) ✘ | 1/3 |
| L01_S03_R17, 116.0s | Walking | 115.0–116.0 | L02_S14_R08 [9.0–10.0s] Walking ✔ | L01_S01_R14 [66.0–67.0s] Cart ✘ | L03_S14_R20 [18.0–19.0s] Walking ✔ | 2/3 |
| L01_S03_R24, 100.9s | Walking | 100.0–101.0 | L01_S01_R08 [107.0–108.0s] Standing ✘ | L01_S01_R08 [109.0–110.0s] Standing ✘ | L02_S14_R07 [96.0–97.0s] Standing ✘ | 0/3 |
| L02_S14_R08, 15.9s | Cart | 15.0–16.0 | L01_S01_R04 [112.0–113.0s] Standing ✘ | L01_S01_R15 [99.0–100.0s] Cart ✔ | L01_S01_R18 [69.0–70.0s] Walking ✘ | 1/3 |
| L01_S01_R14, 68.4s | Cart | 68.0–69.0 | L02_S15_R04 [35.0–36.0s] Walking ✘ | L01_S03_R12 [31.0–32.0s] Cart ✔ | L02_S15_R02 [110.0–111.0s] Walking ✘ | 1/3 |
| L01_S01_R06, 45.0s | Handling(upwards) | 44.0–45.0 | L01_S03_R14 [52.0–53.0s] Handling(centred) ✘ | L02_S14_R14 [95.0–96.0s] Handling(centred) ✘ | L01_S03_R14 [59.0–60.0s] Standing ✘ | 0/3 |
| L01_S14_R01, 94.2s | Handling(upwards) | 94.0–95.0 | L01_S01_R09 [8.0–9.0s] Standing ✘ | L01_S01_R16 [106.0–107.0s] Cart ✘ | L01_S01_R26 [106.0–107.0s] Standing ✘ | 0/3 |
| L01_S03_R21, 21.2s | Handling(centred) | 21.0–22.0 | L01_S01_R26 [106.0–107.0s] Standing ✘ | L02_S14_R16 [91.0–92.0s] Walking ✘ | L02_S14_R13 [81.0–82.0s] Handling(centred) ✔ | 1/3 |
| L03_S15_R25, 109.3s | Handling(centred) | 109.0–110.0 | L03_S14_R27 [60.0–61.0s] Handling(centred) ✔ | L03_S14_R27 [106.0–107.0s] Handling(centred) ✔ | L03_S14_R30 [106.0–107.0s] Handling(centred) ✔ | 3/3 |
| L01_S03_R04, 98.2s | Handling(downwards) | 98.0–99.0 | L01_S01_R15 [43.0–44.0s] Handling(downwards) ✔ | L02_S14_R16 [71.0–72.0s] Standing ✘ | L02_S14_R09 [84.0–85.0s] Handling(downwards) ✔ | 2/3 |
| L02_S14_R16, 26.2s | Handling(downwards) | 26.0–27.0 | L01_S03_R15 [98.0–99.0s] Handling(downwards) ✔ | L01_S03_R15 [99.0–100.0s] Handling(downwards) ✔ | L01_S03_R15 [100.0–101.0s] Handling(downwards) ✔ | 3/3 |
| L01_S01_R18, 4.8s | Synchronization | 4.0–5.0 | L02_S14_R06 [2.0–3.0s] Synchronization ✔ | L02_S14_R14 [2.0–3.0s] Synchronization ✔ | L01_S03_R01 [4.0–5.0s] Synchronization ✔ | 3/3 |
| L03_S14_R20, 1.8s | Synchronization | 1.0–2.0 | L01_S03_R11 [1.0–2.0s] None ✘ | L02_S15_R13 [1.0–2.0s] Synchronization ✔ | L03_S15_R21 [96.0–97.0s] Handling(upwards) ✘ | 1/3 |
| L01_S03_R18, 10.4s | None | 10.0–11.0 | L01_S01_R16 [45.0–46.0s] Walking ✘ | L01_S01_R30 [86.0–87.0s] Walking ✘ | L01_S01_R05 [42.0–43.0s] Walking ✘ | 0/3 |
| L01_S03_R18, 48.4s | None | 48.0–49.0 | L01_S01_R08 [78.0–79.0s] Standing ✘ | L02_S14_R15 [94.0–95.0s] Handling(centred) ✘ | L01_S14_R02 [113.0–114.0s] Walking ✘ | 0/3 |

## oracle segmentation

| query (recording, t) | label | query segment (s) | top-1 | top-2 | top-3 | hits |
|---|---|---|---|---|---|---|
| L01_S01_R09, 29.8s | Standing | 28.9–32.4 | L03_S14_R17 [32.9–37.1s] Walking ✘ | L03_S14_R17 [32.9–37.1s] Walking ✘ | L03_S14_R17 [32.9–37.1s] Walking ✘ | 0/3 |
| L01_S03_R13, 24.6s | Standing | 24.5–24.9 | L01_S01_R18 [41.4–44.4s] Walking ✘ | L01_S01_R18 [41.4–44.4s] Walking ✘ | L01_S01_R18 [41.4–44.4s] Walking ✘ | 0/3 |
| L01_S03_R17, 116.0s | Walking | 113.4–120.0 | L02_S14_R08 [3.0–11.5s] Walking ✔ | L02_S14_R08 [3.0–11.5s] Walking ✔ | L02_S14_R08 [3.0–11.5s] Walking ✔ | 3/3 |
| L01_S03_R24, 100.9s | Walking | 99.1–102.2 | L01_S01_R14 [4.6–5.6s] Standing ✘ | L01_S01_R14 [4.6–5.6s] Standing ✘ | L01_S01_R08 [106.7–110.0s] Standing ✘ | 0/3 |
| L02_S14_R08, 15.9s | Cart | 12.2–22.4 | L01_S01_R04 [113.6–115.1s] Handling(centred) ✘ | L01_S01_R04 [113.6–115.1s] Handling(centred) ✘ | L01_S01_R15 [94.5–100.3s] Cart ✔ | 1/3 |
| L01_S01_R14, 68.4s | Cart | 64.7–72.0 | L01_S03_R12 [27.2–33.9s] Cart ✔ | L01_S03_R12 [27.2–33.9s] Cart ✔ | L01_S03_R12 [27.2–33.9s] Cart ✔ | 3/3 |
| L01_S01_R06, 45.0s | Handling(upwards) | 44.4–46.5 | L01_S03_R09 [26.8–28.3s] Standing ✘ | L01_S03_R09 [26.8–28.3s] Standing ✘ | L02_S14_R14 [112.6–114.2s] Standing ✘ | 0/3 |
| L01_S14_R01, 94.2s | Handling(upwards) | 92.7–94.3 | L01_S01_R28 [87.6–89.2s] Handling(upwards) ✔ | L01_S01_R28 [87.6–89.2s] Handling(upwards) ✔ | L01_S01_R28 [87.6–89.2s] Handling(upwards) ✔ | 3/3 |
| L01_S03_R21, 21.2s | Handling(centred) | 20.1–23.6 | L01_S01_R14 [72.0–73.0s] None ✘ | L01_S14_R01 [71.6–72.4s] Standing ✘ | L01_S01_R03 [53.3–59.6s] Cart ✘ | 0/3 |
| L03_S15_R25, 109.3s | Handling(centred) | 16.8–120.0 | L03_S14_R26 [40.0–40.7s] Standing ✘ | L03_S14_R23 [87.9–90.1s] Handling(centred) ✔ | L03_S14_R23 [87.9–90.1s] Handling(centred) ✔ | 2/3 |
| L01_S03_R04, 98.2s | Handling(downwards) | 93.1–99.1 | L02_S14_R16 [39.1–49.6s] Handling(centred) ✘ | L02_S14_R16 [39.1–49.6s] Handling(centred) ✘ | L02_S14_R16 [39.1–49.6s] Handling(centred) ✘ | 0/3 |
| L02_S14_R16, 26.2s | Handling(downwards) | 18.9–29.2 | L01_S03_R15 [95.8–102.1s] Handling(downwards) ✔ | L01_S03_R15 [95.8–102.1s] Handling(downwards) ✔ | L01_S03_R15 [95.8–102.1s] Handling(downwards) ✔ | 3/3 |
| L01_S01_R18, 4.8s | Synchronization | 3.5–5.7 | L02_S14_R06 [1.4–3.9s] Synchronization ✔ | L02_S14_R06 [1.4–3.9s] Synchronization ✔ | L02_S14_R06 [1.4–3.9s] Synchronization ✔ | 3/3 |
| L03_S14_R20, 1.8s | Synchronization | 1.2–3.1 | L01_S03_R02 [2.7–4.4s] Synchronization ✔ | L01_S03_R20 [0.6–2.7s] Synchronization ✔ | L01_S03_R20 [0.6–2.7s] Synchronization ✔ | 3/3 |
| L01_S03_R18, 10.4s | None | 2.7–120.0 | L01_S01_R14 [47.8–53.7s] Cart ✘ | L01_S01_R14 [47.8–53.7s] Cart ✘ | L01_S01_R14 [47.8–53.7s] Cart ✘ | 0/3 |
| L01_S03_R18, 48.4s | None | 2.7–120.0 | L01_S01_R14 [47.8–53.7s] Cart ✘ | L01_S01_R14 [47.8–53.7s] Cart ✘ | L01_S01_R14 [47.8–53.7s] Cart ✘ | 0/3 |