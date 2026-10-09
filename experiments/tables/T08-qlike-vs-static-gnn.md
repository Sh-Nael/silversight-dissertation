**T08. Each model compared with the static graph on QLIKE**

| Model | Mean diff. (1-day) | p DM (1-day) | p Wilcoxon (1-day) | p permutation (1-day) | Folds won (1-day) | Mean diff. (5-day) | p DM (5-day) | p Wilcoxon (5-day) | p permutation (5-day) | Folds won (5-day) |
|---|---|---|---|---|---|---|---|---|---|---|
| climatology | +0.321 | <0.001* | 0.008* | <0.001* | 0/8 | +0.269 | <0.001* | 0.039* | <0.001* | 1/8 |
| ewma | +0.050 | <0.001* | 0.016* | <0.001* | 1/8 | +0.027 | 0.034* | 0.148 | 0.071 | 2/8 |
| ar_garch | +0.034 | 0.013* | 0.109 | 0.017* | 1/8 | +0.008 | 0.650 | 0.742 | 0.728 | 3/8 |
| linear | +0.011 | 0.384 | 0.461 | 0.382 | 3/8 | -0.000 | 0.987 | 1.000 | 0.996 | 4/8 |
| gbm | +0.029 | 0.024* | 0.195 | 0.031* | 3/8 | +0.005 | 0.698 | 0.945 | 0.765 | 4/8 |
| lstm | +0.024 | 0.037* | 0.078 | 0.051 | 2/8 | +0.007 | 0.563 | 0.312 | 0.653 | 2/8 |
| gru | +0.016 | 0.098 | 0.383 | 0.129 | 2/8 | +0.005 | 0.611 | 0.742 | 0.661 | 3/8 |
| aimdg | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |

*A negative mean difference says the model's average loss was lower than the reference's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds the model came out ahead.*

Source: `phase4-aimdg-vs-static_gnn/significance.csv`
