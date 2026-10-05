**T13. Ablations on the test folds: volatility**

| Model | QLIKE 1d | QLIKE 5d | RMSE 1d | RMSE 5d | MZ R² 1d | MZ R² 5d |
|---|---|---|---|---|---|---|
| static_gnn | **-6.974** | **-5.315** | **0.0163** | **0.0229** | 0.059 | **0.253** |
| aimdg | -6.953 | -5.300 | 0.0165 | 0.0240 | 0.052 | 0.199 |
| aimdg_replay | -6.953 | -5.300 | 0.0165 | 0.0240 | 0.052 | 0.199 |
| aimdg_A1 | -6.961 | -5.308 | **0.0163** | 0.0235 | **0.062** | 0.231 |
| aimdg_A2 | -6.951 | -5.297 | 0.0165 | 0.0241 | 0.052 | 0.198 |

*aimdg_replay: the official run reproduced from cached networks. A1: no evolution (top 3 edges by reliability). A2: no variance penalty in the reliability. Both variants were declared before the replays. Best value per column in bold.*

Source: `phase5-test-ablations/scoreboard.csv`
