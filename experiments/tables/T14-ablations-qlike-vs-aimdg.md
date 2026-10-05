**T14. QLIKE: ablations against AIM-DG**

| Model | Diff 1d | DM p 1d | Wilcoxon p 1d | Perm. p 1d | Folds 1d | Diff 5d | DM p 5d | Wilcoxon p 5d | Perm. p 5d | Folds 5d |
|---|---|---|---|---|---|---|---|---|---|---|
| static_gnn | -0.021 | 0.006* | 0.055 | 0.007* | 7/8 | -0.015 | 0.018* | 0.039* | 0.010* | 7/8 |
| aimdg_replay | +0.000 | – | 1.000 | 1.000 | 0/8 | +0.000 | – | 1.000 | 1.000 | 0/8 |
| aimdg_A1 | -0.007 | 0.309 | 0.641 | 0.438 | 5/8 | -0.008 | 0.120 | 0.742 | 0.265 | 3/8 |
| aimdg_A2 | +0.002 | 0.128 | 1.000 | 0.305 | 2/8 | +0.003 | 0.024* | 1.000 | 0.206 | 2/8 |

*Difference: mean loss of the model minus the reference's; negative means the model is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which the model beat the reference. Read A2 with care: its final selection differs from AIM-DG's in 3 of the 200 months and its forecasts on 266 of the 4,141 days, so its p-values rest on few months.*

Source: `phase5-test-ablations-vs-aimdg/significance.csv`
