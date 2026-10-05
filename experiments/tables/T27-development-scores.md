**T27. Development scores of the tuned models**

| Model | Joint | Log-loss 1d | Log-loss 5d | QLIKE 1d | QLIKE 5d |
|---|---|---|---|---|---|
| linear | -4.7583 | 0.6903 | 0.6939 | -6.9631 | -5.3219 |
| gbm | -4.7655 | 0.6880 | 0.6911 | -6.9655 | -5.3237 |
| lstm | -4.7508 | 0.6925 | 0.6924 | -6.9557 | -5.3158 |
| gru | -4.7514 | 0.6922 | 0.6951 | -6.9597 | -5.3175 |
| static_gnn | -4.7580 | 0.6935 | 0.6928 | -6.9690 | -5.3196 |
| aimdg | -4.7570 | 0.6927 | 0.6895 | -6.9619 | -5.3166 |

*Out-of-fold losses at the 17 yearly tuning points, mean. Used for design decisions only; they are the best of the searched trials and so optimistic (about 0.01 in the joint loss).*

Source: `phase4-aimdg/dev_scoreboard.csv`
