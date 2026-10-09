**T14. The variants compared with AIM-DG itself on QLIKE**

| Model | Mean diff. (1-day) | p DM (1-day) | p Wilcoxon (1-day) | p permutation (1-day) | Folds won (1-day) | Mean diff. (5-day) | p DM (5-day) | p Wilcoxon (5-day) | p permutation (5-day) | Folds won (5-day) |
|---|---|---|---|---|---|---|---|---|---|---|
| static_gnn | -0.021 | 0.006* | 0.055 | 0.007* | 7/8 | -0.015 | 0.018* | 0.039* | 0.010* | 7/8 |
| aimdg_replay | +0.000 | – | 1.000 | 1.000 | 0/8 | +0.000 | – | 1.000 | 1.000 | 0/8 |
| aimdg_A1 | -0.007 | 0.309 | 0.641 | 0.438 | 5/8 | -0.008 | 0.120 | 0.742 | 0.265 | 3/8 |
| aimdg_A2 | +0.002 | 0.128 | 1.000 | 0.305 | 2/8 | +0.003 | 0.024* | 1.000 | 0.206 | 2/8 |

*A negative mean difference says the model's average loss was lower than the reference's. p DM comes from the Diebold–Mariano test on all 4,141 daily losses, p Wilcoxon from a signed-rank test on the eight per-fold averages, and p permutation from flipping signs in blocks of 21 days. A star marks p below 0.05. 'Folds won' says in how many of the eight folds the model came out ahead. Caution for aimdg_A2: it picked different edges from AIM-DG in only 3 of the 200 months (266 of 4,141 forecast days differ), so its p-values depend on very few months.*

Source: `phase5-test-ablations-vs-aimdg/significance.csv`
