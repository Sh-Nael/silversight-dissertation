**T08. QLIKE: every model against the static graph**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| climatology | +0.321 | <0.001* | 0.008* | <0.001* | 0/8 | +0.269 | <0.001* | 0.039* | <0.001* | 1/8 |
| ewma | +0.050 | <0.001* | 0.016* | <0.001* | 1/8 | +0.027 | 0.034* | 0.148 | 0.071 | 2/8 |
| ar_garch | +0.034 | 0.013* | 0.109 | 0.017* | 1/8 | +0.008 | 0.650 | 0.742 | 0.728 | 3/8 |
| linear | +0.011 | 0.384 | 0.461 | 0.382 | 3/8 | -0.000 | 0.987 | 1.000 | 0.996 | 4/8 |
| gbm | +0.029 | 0.024* | 0.195 | 0.031* | 3/8 | +0.005 | 0.698 | 0.945 | 0.765 | 4/8 |
| lstm | +0.024 | 0.037* | 0.078 | 0.051 | 2/8 | +0.007 | 0.563 | 0.312 | 0.653 | 2/8 |
| gru | +0.016 | 0.098 | 0.383 | 0.129 | 2/8 | +0.005 | 0.611 | 0.742 | 0.661 | 3/8 |
| aimdg | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference.*

Source: `phase4-aimdg-vs-static_gnn/significance.csv`
