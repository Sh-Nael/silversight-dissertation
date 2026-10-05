**T15. QLIKE: AIM-DG and its ablations against the static graph**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| aimdg | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |
| aimdg_replay | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |
| aimdg_A1 | +0.014 | 0.119 | 0.148 | 0.115 | 3/8 | +0.007 | 0.451 | 0.461 | 0.523 | 3/8 |
| aimdg_A2 | +0.023 | 0.005* | 0.055 | 0.006* | 1/8 | +0.018 | 0.004* | 0.008* | <0.001* | 0/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference.*

Source: `phase5-test-ablations/significance.csv`
