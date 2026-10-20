# PCA solver check (token level)

For each anchor dataset and seed, on exactly the fitting sample the anchor uses (<= 10k complete windows, the cell's
seed, global standardization), PCA-64 is fitted with sklearn's `randomized` (the anchor), `full` (exact LAPACK SVD)
and `arpack` (exact truncated) solvers, at the anchor's whitening. Reported per cell:

| dataset | seed | fit s (rand / full / arpack) | variance captured (rand / full) | subspace cos min / mean (full) | codes ARI vs full | codes ARI vs arpack | K-means seed reference ARI |
|---|---|---|---|---|---|---|---|
| babel2 | 111 | 0.19 / 1.31 / 0.33 | 0.94073 / 0.94075 | 0.955115 / 0.999122 | 0.6937 | 0.6937 | 0.6257 |
| babel2 | 222 | 0.19 / 1.29 / 0.32 | 0.94072 / 0.94075 | 0.980906 / 0.999471 | 0.6218 | 0.6218 | 0.6657 |
| babel2 | 1538574472 | 0.20 / 1.28 / 0.32 | 0.94073 / 0.94075 | 0.983594 / 0.999614 | 0.6502 | 0.6478 | 0.6370 |
| babel3 | 111 | 0.14 / 0.73 / 0.23 | 0.99631 / 0.99632 | 0.966592 / 0.999255 | 0.5736 | 0.5736 | 0.5243 |
| babel3 | 222 | 0.14 / 0.82 / 0.25 | 0.99632 / 0.99633 | 0.975541 / 0.999384 | 0.5269 | 0.5269 | 0.5507 |
| babel3 | 1538574472 | 0.15 / 0.81 / 0.26 | 0.99631 / 0.99632 | 0.977945 / 0.999450 | 0.5417 | 0.5417 | 0.5497 |
| hugadb | 111 | 0.27 / 2.98 / 0.37 | 0.97866 / 0.97866 | 0.999398 / 0.999990 | 0.7614 | 0.7614 | 0.6804 |
| hugadb | 222 | 0.30 / 3.19 / 0.38 | 0.97861 / 0.97861 | 0.999776 / 0.999995 | 0.7251 | 0.7393 | 0.6834 |
| hugadb | 1538574472 | 0.26 / 3.20 / 0.34 | 0.97883 / 0.97883 | 0.999208 / 0.999987 | 0.7220 | 0.7220 | 0.8681 |
| lara | 111 | 0.21 / 0.90 / 0.31 | 0.91825 / 0.91829 | 0.948700 / 0.998879 | 0.3836 | 0.3836 | 0.3961 |
| lara | 222 | 0.21 / 0.91 / 0.27 | 0.92016 / 0.92023 | 0.901836 / 0.997998 | 0.4245 | 0.3962 | 0.3866 |
| lara | 1538574472 | 0.21 / 1.00 / 0.34 | 0.92100 / 0.92107 | 0.957153 / 0.998927 | 0.3888 | 0.3907 | 0.3850 |

| dataset | variance captured, randomized − full | mean subspace cos | codes ARI vs full | K-means seed reference ARI | full / randomized fit time |
|---|---|---|---|---|---|
| babel2 | -2.7e-05 | 0.99940 | 0.655 | 0.643 | 6.6x |
| babel3 | -5.6e-06 | 0.99936 | 0.547 | 0.542 | 5.6x |
| hugadb | -3.8e-07 | 0.99999 | 0.736 | 0.744 | 11.3x |
| lara | -5.6e-05 | 0.99860 | 0.399 | 0.389 | 4.5x |