**T12. QLIKE: ensembles and members against gradient boosting**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | -0.018 | 0.134 | 0.148 | 0.131 | 6/8 | -0.005 | 0.635 | 1.000 | 0.727 | 4/8 |
| static_gnn | -0.029 | 0.024* | 0.195 | 0.031* | 5/8 | -0.005 | 0.698 | 0.945 | 0.765 | 4/8 |
| ens_lin_gbm | -0.021 | 0.002* | 0.055 | 0.001* | 7/8 | -0.011 | 0.030* | 0.039* | 0.087 | 6/8 |
| ens_lin_gbm_sgnn | -0.034 | <0.001* | 0.008* | <0.001* | 8/8 | -0.018 | 0.010* | 0.039* | 0.034* | 6/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference.*

Source: `ensemble-6.2/significance.csv`
