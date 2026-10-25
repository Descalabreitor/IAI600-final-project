# Archived notebooks

The project's notebooks before the unification of 19 Oct 2026. Their experiments were rerun on the base data (`data/btc_hourly.csv.gz`) and the common train / validation / test blocks in the four notebooks of the repo root. Use these only to trace where a result or a quoted text comes from.

**Their numbers are not comparable with the current ones.** Except for `model_test.ipynb`, they were run on the old `data/btc_hourly_windows.npz` (data from 2012, global scaler), and almost each one uses its own split.

| Notebook | Author | Content | Now in |
|---|---|---|---|
| `data-preparation.ipynb` | Adrián P. | 1-minute Kaggle data → hourly candles → 168 h input / 24 h target windows, chronological split, global `StandardScaler`, saved to `data/btc_hourly_windows.npz` + `data/scaler.joblib` | `hourly-base-data.ipynb` |
| `models-and-hyperparameters-ver1.ipynb` | Alberto | Early draft: Ridge, Lasso, KNN, RF, GB, MLP on scale-free features, data from 2012. All at or below the zero forecast; conclusions never written | `classical-models.ipynb` §3 |
| `bitcoin-models-new-data.ipynb` (version 1) | Alberto | 42 scale-free features, walk-forward CV, binomial test on independent days, three daily targets | `classical-models.ipynb` §2–3, §8 |
| `bitcoin-models-2018-2026.ipynb` (version 2) | Alberto | Version 1 with data from 2018 and three fixed blocks (train 2018–2022, val 2023–Jun 2024, test Jul 2024–Oct 2026) | `classical-models.ipynb` §3, §8 |
| `bitcoin-models-v3.ipynb` (version 3) | Alberto | Why returns look like noise, and 14 improvements: long format, hourly \|r\| target, quantile ranges, metric-matched selection, Diebold–Mariano, HAR / range features, ensembles, economic test, bootstrap (~45 min) | `data-analysis.ipynb` §4–5, `classical-models.ipynb` §4–11 |
| `v3-extensions.ipynb` | Filip | On top of version 3: Binance / ETH attributes, monthly refits with recency weights, conformal ranges, next-hour direction (~2 h) | `classical-models.ipynb` §5–7 |
| `course-methods.ipynb` | Filip | The project question with only the course-lab methods (I–VI): pipeline, regression, classification, SVM, K-Means, Keras MLP; daily forecasts, train 2018–2023, test 2024+ (~1 min) | `classical-models.ipynb` §9, `data-analysis.ipynb` §6 |
| `model_test.ipynb` | Adrián P. | RNN and LSTM in PyTorch on the 168 h windows, scaled and raw inputs, MSE loss | `neural-models.ipynb` §3 |
| `correlation_analyses.ipynb` | Adri "no chacho" | First version of the macro correlation notebook (now in `Correlation Analyses/`) | `data-analysis.ipynb` §8 |

## Main results, as archived

- **Data preparation.** Its last commit (13 Oct) moved the data start to 2017 with an 80/20 split; the model notebooks had been run on the earlier version (from 2012, test from Oct 2023). The global scaler does not remove the price trend: scaled test prices reach 7.6, far outside the training range.
- **Version 1.** No model beats the zero forecast on the 24 hourly returns (MAE skill −0.0003 to −0.0010). The 53.4% daily direction accuracy is "always up" (53.5% up days in the test, balanced accuracy 0.50). More model freedom always means worse validation. Aggregated targets: next-day return IC 0.094; **next-day volatility R² 0.45** (Gradient Boosting) against 0.09 for last week's average.
- **Version 2.** Same conclusions on a balanced test block (50.4% up days): best direction model AUC 0.544, p = 0.08; next-day volatility R² 0.47. Training from 2012 hurts the volatility model (R² 0.29 against 0.41 from 2017/2018).
- **Version 3.** Hourly returns: tiny reversal at lags 1, 2, 24 (≤ 0.4% of the variance); the size of the moves is autocorrelated (0.27 at lag 1) and follows the clock. **Hourly \|r\| R² 0.12** against 0.05 for the best rule; 80% ranges 36% narrower than fixed ones; return IC 0.01–0.02 (DM p ≥ 0.13); volatility targeting loses to buy & hold.
- **Version 3 extensions.** Monthly refits with a 12-month half-life: hourly \|r\| R² 0.124 → **0.148** (better in 28 of 29 months). Rolling conformal: 80.0% coverage. Binance / ETH add nothing. Next hour direction: AUC 0.537, 55% right on the 10% most confident calls.
- **Course methods.** Next-day move size: Linear Regression R² 0.48, SVR 0.46, RF 0.41, Keras MLP 0.37. Direction: 49–51% accuracy, AUC ≈ 0.50. K-Means: crash, rally, quiet and normal days.
- **RNN / LSTM.** All four variants equal the zero forecast (MSE ratio 1.000–1.002). The results table has a "GB" row that no cell produces (old kernel state).
- **Correlation (first version).** Data 2012–2026; the later version in `Correlation Analyses/` replaces it.

## Running them

Open and run them where they are: their first cell moves to the repo root, so `data/` and `utils/` are found. They need `data/btc_hourly_windows.npz` and `data/scaler.joblib`. `models-and-hyperparameters-ver1.ipynb` first runs `data-preparation.ipynb`, which downloads the **latest, unpinned** Kaggle data and overwrites both files.

The old helper scripts are now the `utils/` package: `btc_utils.py` → `utils/hourly.py`, `binance_features.py` → `utils/binance.py`, `shared_cells.py` → `utils/notebooks.py`, and the correlation scripts → `utils/macro.py`, `utils/correlation.py`, `utils/btc_data.py`.
