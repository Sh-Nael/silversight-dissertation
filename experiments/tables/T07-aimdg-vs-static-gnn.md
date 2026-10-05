**T07. AIM-DG against the static graph (the main hypothesis test)**

| Loss | Horizon | Difference | DM statistic | DM p | Wilcoxon p | Permutation p | Folds better |
|---|---|---|---|---|---|---|---|
| QLIKE | 1 day | +0.021 | 2.75 | 0.006* | 0.055 | 0.007* | 1/8 |
| QLIKE | 5 days | +0.015 | 2.37 | 0.018* | 0.039* | 0.010* | 1/8 |
| Brier | 1 day | +0.00016 | 0.58 | 0.559 | 0.742 | 0.560 | 3/8 |
| Brier | 5 days | -0.00032 | -0.37 | 0.713 | 0.844 | 0.724 | 5/8 |
| Log-loss | 1 day | +0.00034 | 0.60 | 0.551 | 0.641 | 0.555 | 3/8 |
| Log-loss | 5 days | -0.00068 | -0.38 | 0.706 | 0.844 | 0.724 | 5/8 |

*Difference: mean loss of AIM-DG minus the static graph's; negative means AIM-DG is better. DM: Diebold–Mariano on the 4,141 daily losses; Wilcoxon: signed-rank over the 8 fold means; Perm.: block permutation (21-day blocks). * p < 0.05. Folds: folds in which AIM-DG beat the static graph.*

Source: `phase4-aimdg-vs-static_gnn/significance.csv`
