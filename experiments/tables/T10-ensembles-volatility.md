**T10. Volatility: two simple averages of models next to the models they average**

| Model | QLIKE (1-day) | QLIKE (5-day) | RMSE (1-day) | RMSE (5-day) | MZ R² (1-day) | MZ R² (5-day) |
|---|---|---|---|---|---|---|
| linear | -6.963 | -5.315 | **0.0162** | 0.0237 | **0.081** | 0.165 |
| gbm | -6.945 | -5.310 | 0.0169 | 0.0246 | 0.028 | 0.111 |
| static_gnn | -6.974 | -5.315 | 0.0163 | **0.0229** | 0.059 | **0.253** |
| ens_lin_gbm | -6.966 | -5.321 | 0.0165 | 0.0240 | 0.049 | 0.139 |
| ens_lin_gbm_sgnn | **-6.980** | **-5.328** | 0.0163 | 0.0233 | 0.059 | 0.198 |

*Scored over the 4,141 trading days from 4 Jan 2010 to 23 Jun 2026 (eight folds, refitted monthly). Which models to average was decided on development scores. Bold marks the leader of each column.*

Source: `ensemble-6.2/scoreboard.csv`
