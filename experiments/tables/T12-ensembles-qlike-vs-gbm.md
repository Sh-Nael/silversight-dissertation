**T12. The averages and their members compared with gradient boosting on QLIKE**

| Model | Mean diff. (1-day) | p DM (1-day) | p Wilcoxon (1-day) | p permutation (1-day) | Folds won (1-day) | Mean diff. (5-day) | p DM (5-day) | p Wilcoxon (5-day) | p permutation (5-day) | Folds won (5-day) |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | -0.018 | 0.134 | 0.148 | 0.131 | 6/8 | -0.005 | 0.635 | 1.000 | 0.727 | 4/8 |
| static_gnn | -0.029 | 0.024* | 0.195 | 0.031* | 5/8 | -0.005 | 0.698 | 0.945 | 0.765 | 4/8 |
| ens_lin_gbm | -0.021 | 0.002* | 0.055 | 0.001* | 7/8 | -0.011 | 0.030* | 0.039* | 0.087 | 6/8 |
| ens_lin_gbm_sgnn | -0.034 | <0.001* | 0.008* | <0.001* | 8/8 | -0.018 | 0.010* | 0.039* | 0.034* | 6/8 |

*A negative mean difference says the model's average loss was lower than the reference's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds the model came out ahead.*

Source: `ensemble-6.2/significance.csv`
