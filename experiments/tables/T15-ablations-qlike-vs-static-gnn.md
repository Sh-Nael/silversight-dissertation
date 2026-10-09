**T15. AIM-DG and its variants compared with the static graph on QLIKE**

| Model | Mean diff. (1-day) | p DM (1-day) | p Wilcoxon (1-day) | p permutation (1-day) | Folds won (1-day) | Mean diff. (5-day) | p DM (5-day) | p Wilcoxon (5-day) | p permutation (5-day) | Folds won (5-day) |
|---|---|---|---|---|---|---|---|---|---|---|
| aimdg | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |
| aimdg_replay | +0.021 | 0.006* | 0.055 | 0.007* | 1/8 | +0.015 | 0.018* | 0.039* | 0.010* | 1/8 |
| aimdg_A1 | +0.014 | 0.119 | 0.148 | 0.115 | 3/8 | +0.007 | 0.451 | 0.461 | 0.523 | 3/8 |
| aimdg_A2 | +0.023 | 0.005* | 0.055 | 0.006* | 1/8 | +0.018 | 0.004* | 0.008* | <0.001* | 0/8 |

*A negative mean difference says the model's average loss was lower than the reference's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds the model came out ahead.*

Source: `phase5-test-ablations/significance.csv`
