**T07. Does AIM-DG beat the graph that keeps every edge? All scores, both horizons**

| Score | Days ahead | Mean difference | DM stat | p (DM) | p (Wilcoxon) | p (permutation) | Folds won |
|---|---|---|---|---|---|---|---|
| QLIKE | 1 | +0.021 | 2.75 | 0.006* | 0.055 | 0.007* | 1/8 |
| QLIKE | 5 | +0.015 | 2.37 | 0.018* | 0.039* | 0.010* | 1/8 |
| Brier | 1 | +0.00016 | 0.58 | 0.559 | 0.742 | 0.560 | 3/8 |
| Brier | 5 | -0.00032 | -0.37 | 0.713 | 0.844 | 0.724 | 5/8 |
| Log-loss | 1 | +0.00034 | 0.60 | 0.551 | 0.641 | 0.555 | 3/8 |
| Log-loss | 5 | -0.00068 | -0.38 | 0.706 | 0.844 | 0.724 | 5/8 |

*A negative mean difference says AIM-DG's average loss was lower than the static graph's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds AIM-DG came out ahead.*

Source: `phase4-aimdg-vs-static_gnn/significance.csv`
