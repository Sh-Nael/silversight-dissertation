**T06. Log-loss: every model against climatology**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| ewma | +0.00000 | – | 1.000 | 1.000 | 0/8 | +0.00000 | – | 1.000 | 1.000 | 0/8 |
| ar_garch | -0.00025 | 0.701 | 0.844 | 0.699 | 5/8 | +0.00328 | 0.003* | 0.312 | 0.016* | 3/8 |
| linear | +0.00053 | 0.701 | 1.000 | 0.723 | 4/8 | +0.00265 | 0.282 | 0.195 | 0.362 | 1/8 |
| gbm | -0.00036 | 0.828 | 0.641 | 0.774 | 5/8 | +0.00324 | 0.220 | 0.195 | 0.303 | 2/8 |
| lstm | +0.00031 | 0.729 | 0.461 | 0.748 | 3/8 | +0.00076 | 0.768 | 0.461 | 0.830 | 2/8 |
| gru | +0.00118 | 0.172 | 0.312 | 0.198 | 3/8 | +0.00288 | 0.275 | 0.250 | 0.376 | 2/8 |
| static_gnn | +0.00112 | 0.197 | 0.195 | 0.269 | 2/8 | +0.00258 | 0.347 | 0.461 | 0.477 | 2/8 |
| aimdg | +0.00146 | 0.135 | 0.148 | 0.200 | 1/8 | +0.00190 | 0.509 | 0.383 | 0.609 | 3/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference.*

Source: `phase4-aimdg/significance.csv`
