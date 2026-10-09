**T04. Each model compared with climatology on QLIKE**

| Model | Mean diff. (1-day) | p DM (1-day) | p Wilcoxon (1-day) | p permutation (1-day) | Folds won (1-day) | Mean diff. (5-day) | p DM (5-day) | p Wilcoxon (5-day) | p permutation (5-day) | Folds won (5-day) |
|---|---|---|---|---|---|---|---|---|---|---|
| ewma | -0.271 | <0.001* | 0.016* | <0.001* | 7/8 | -0.241 | <0.001* | 0.039* | <0.001* | 6/8 |
| ar_garch | -0.286 | <0.001* | 0.008* | <0.001* | 8/8 | -0.260 | <0.001* | 0.008* | <0.001* | 8/8 |
| linear | -0.309 | <0.001* | 0.039* | <0.001* | 6/8 | -0.269 | <0.001* | 0.039* | <0.001* | 6/8 |
| gbm | -0.292 | <0.001* | 0.039* | <0.001* | 6/8 | -0.264 | <0.001* | 0.023* | <0.001* | 7/8 |
| lstm | -0.297 | <0.001* | 0.008* | <0.001* | 8/8 | -0.262 | <0.001* | 0.023* | <0.001* | 7/8 |
| gru | -0.305 | <0.001* | 0.016* | <0.001* | 7/8 | -0.264 | <0.001* | 0.039* | <0.001* | 6/8 |
| static_gnn | -0.321 | <0.001* | 0.008* | <0.001* | 8/8 | -0.269 | <0.001* | 0.039* | <0.001* | 7/8 |
| aimdg | -0.300 | <0.001* | 0.023* | <0.001* | 7/8 | -0.254 | <0.001* | 0.039* | <0.001* | 7/8 |

*A negative mean difference says the model's average loss was lower than the reference's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds the model came out ahead.*

Source: `phase4-aimdg/significance.csv`
