# Fitting-sample size check (token level)

Does the 10k-window fitting cap change the tokens? Token level only (no contextualizer, no OT).

| dataset | fitting windows | fit s (mean) | held-out relative error | seed ARI | codes used |
|---|---|---|---|---|---|
| hugadb | 10000 (10000) | 6 | 0.2033 | 0.966 | 332 |
| hugadb | 40000 (40000) | 23 | 0.1905 | 0.928 | 340 |
| hugadb | 67962 (all) | 39 | 0.1888 | 0.981 | 336 |
| lara | 10000 (10000) | 6 | 0.4126 | 0.633 | 492 |
| lara | 40000 (40000) | 28 | 0.3845 | 0.609 | 490 |
| lara | 197500 (all) | 124 | 0.3702 | 0.693 | 494 |

| dataset | pair | cross-size ARI | seed ARI of the smaller size | read |
|---|---|---|---|---|
| hugadb | 10000 vs 40000 | 0.948 | 0.966 | size moves the codes no more than seed noise |
| hugadb | 10000 vs all | 0.943 | 0.966 | size moves the codes more than seed noise |
| hugadb | 40000 vs all | 0.949 | 0.928 | size moves the codes no more than seed noise |
| lara | 10000 vs 40000 | 0.568 | 0.633 | size moves the codes more than seed noise |
| lara | 10000 vs all | 0.536 | 0.633 | size moves the codes more than seed noise |
| lara | 40000 vs all | 0.609 | 0.609 | size moves the codes no more than seed noise |
