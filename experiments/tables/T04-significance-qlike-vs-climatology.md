**T04. QLIKE: every model against climatology**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| ewma | -0.271 | <0.001* | 0.016* | <0.001* | 7/8 | -0.241 | <0.001* | 0.039* | <0.001* | 6/8 |
| ar_garch | -0.286 | <0.001* | 0.008* | <0.001* | 8/8 | -0.260 | <0.001* | 0.008* | <0.001* | 8/8 |
| linear | -0.309 | <0.001* | 0.039* | <0.001* | 6/8 | -0.269 | <0.001* | 0.039* | <0.001* | 6/8 |
| gbm | -0.292 | <0.001* | 0.039* | <0.001* | 6/8 | -0.264 | <0.001* | 0.023* | <0.001* | 7/8 |
| lstm | -0.297 | <0.001* | 0.008* | <0.001* | 8/8 | -0.262 | <0.001* | 0.023* | <0.001* | 7/8 |
| gru | -0.305 | <0.001* | 0.016* | <0.001* | 7/8 | -0.264 | <0.001* | 0.039* | <0.001* | 6/8 |
| static_gnn | -0.321 | <0.001* | 0.008* | <0.001* | 8/8 | -0.269 | <0.001* | 0.039* | <0.001* | 7/8 |
| aimdg | -0.300 | <0.001* | 0.023* | <0.001* | 7/8 | -0.254 | <0.001* | 0.039* | <0.001* | 7/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference.*

Source: `phase4-aimdg/significance.csv`
