# ₿ Bitcoin Prediction Model

> Predicting Bitcoin's hourly price changes for the next day from the previous weeks, with classical machine-learning models and neural networks.

**Course project** · Dataset: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data), 1-minute Bitstamp data aggregated to hourly

---

## 📑 Contents

1. [Main ideas](#-main-ideas)
2. [Notebooks](#-notebooks)
3. [Results](#-results)
4. [Conclusions](#-conclusions)
5. [Repository structure](#-repository-structure)
6. [How to run](#-how-to-run)
7. [Use of LLMs / AI tools](#-use-of-llms--ai-tools)
8. [Project status](#-project-status)

---

## 💡 Main ideas

| | |
|---|---|
| **Assigned task** | Predict the 24 hourly price changes of the next day, `Close[h] / Close[h-1] − 1`, from the previous hours |
| **Data** | 76,851 consecutive hours, 1 Jan 2018 → 8 Oct 2026 (pinned Kaggle version 746). Earlier years are mostly carried-forward prices with no trades |
| **Blocks** | Chronological, by the date of the predicted hours. **Train** Jan 2018 → Dec 2022 · **Validation** Jan 2023 → Jun 2024 · **Test** Jul 2024 → Oct 2026, with a 192-hour purge before each border |
| **Forecast origin** | Every hour with 30 days of history and 24 hours after it. Independent test days: one forecast per day at 23:00 (829 days) |
| **Features** | Scale-free (returns, volatilities, ranges, volume ratios, calendar), so they mean the same in 2018 and in 2026 |
| **Evaluation** | Every choice is made on train + validation; the chosen model is refitted on both and scored **once** on test. Skill is measured against simple rules, with block-bootstrap confidence intervals |

Why predict *changes* instead of prices? The price went from about $3,000 to over $120,000 during the data: 72% of the test hours trade above the highest training price. Relative changes are comparable across all those years, absolute prices are not.

---

## 📓 Notebooks

Run them in this order, from the repo root. Each one reads the output of `hourly-base-data.ipynb` and nothing else from `data/`, except the Binance files.

| Notebook | Content | Run time |
|---|---|---|
| `hourly-base-data.ipynb` | Builds the base data `data/btc_hourly.csv.gz` | not measured (Kaggle download) |
| `data-analysis.ipynb` | What the data says before any model | not measured (macro downloads) |
| `classical-models.ipynb` | Every non-neural experiment of the project | ≈ 27 min (16 cores) |
| `neural-models.ipynb` | Every neural-network experiment, on the same rows and test days | ≈ 133 min (CPU) |

**`hourly-base-data.ipynb`.** Downloads the pinned Kaggle version and aggregates the minutes into hourly candles, without `dropna()` so the hourly grid stays continuous. It finds that the dataset hides its gaps: a minute without trades is a flat, zero-volume candle at the previous close (95% of 2012, 25–40% of 2013–2016, 1–4% since 2018). That is why the series starts on 1 Jan 2018. The 15 outages of an hour or more are flagged (`filled_minutes`, `no_trade`), not removed. No split and no scaling: those are model decisions.

**`data-analysis.ipynb`.** Overview of price and volume by year, the return distribution (excess kurtosis 41 hourly) and the largest moves, all real events. Autocorrelation, Ljung–Box and an AR(24) model show that returns are almost pure noise while their size is strongly autocorrelated. Calendar patterns and their drift (the 14 UTC volatility peak since the spot ETFs), volatility regimes and K-Means day clusters, Binance and ETH, and the macro correlations of `Correlation Analyses/`. Analyses that could steer a model use only the train block.

**`classical-models.ipynb`.** Brings together versions 1–3 of the models, `v3-extensions.ipynb` and `course-methods.ipynb`: the 24 hourly returns (wide and long format, metric-matched selection, Diebold–Mariano), the next-hour direction, the size of each hourly move (with monthly refits), 80% prediction ranges with conformal calibration, the three next-day targets (return, direction, volatility), the course-lab pipeline, an economic test and bootstrap intervals. Models: Ridge, Lasso, ElasticNet, KNN, SVR / SVC, Random Forest, Extra Trees, Gradient Boosting, ensembles.

**`neural-models.ipynb`.** MLPs (scikit-learn and PyTorch), RNN and LSTM (PyTorch) and the Keras MLP of the course, on the same origins, blocks and test days as the classical models, so every network is compared with a classical reference. The recurrent models read a 168-hour window of 9 window-normalized channels, with MSE and Huber losses. The PyTorch networks early-stop on the validation block, never on a random hold-out; final networks are averaged over several seeds. **Its written conclusions are still pending** (see the to-do list); the results below are from its last run.

---

## 📊 Results

Last run of `classical-models.ipynb` and `neural-models.ipynb`, test block Jul 2024 → Oct 2026. Every model was chosen without looking at the test block.

| Question | Best model | Test result | Simple rule | Better than the rule? |
|---|---|---|---|---|
| **24 hourly returns** (assigned task) | none beats zero | MAE skill −0.0005 to −0.0002 (networks −0.0017 to −0.0003), IC ≤ 0.018, Diebold–Mariano p ≥ 0.16 | zero forecast | no |
| **Direction of the next hour** | GB classifier · LSTM, every origin | AUC 0.540, 56.6% right on the 10% most confident calls · AUC 0.539, 59.5% | reversal of the last hour: AUC 0.528 | yes (p < 10⁻⁸), small and before costs |
| **Size of each of the next 24 hours** | GB, refitted monthly · ensemble GB + MLP + LSTM | R² 0.144 (fitted once 0.117) · R² 0.127 | last week × hour profile: R² 0.053 | yes |
| **80% range for each hour** | GB quantile + rolling conformal | coverage 79.9% (calm 78%, wild 82%), width 0.98% | fixed range: 87.6%, width 1.28% | yes |
| **Next-day return** | Ridge | R²_oos +0.001, IC 0.041 | zero forecast | no |
| **Next-day direction** | Random Forest | AUC 0.513, accuracy 50.3% | always up: 50.1% | no |
| **Next-day volatility** | Gradient Boosting | R² 0.477 (95% CI 0.42–0.53); MLP 0.460 | last week's average: R² 0.094 | yes |
| **Next-day move size** (course pipeline) | Linear Regression | R² 0.490 (SVR 0.478, RF 0.443, Keras MLP 0.396) | same as today: R² ≈ 0 | yes (not bootstrapped) |
| **Volatility targeting** (economic test) | position from the GB volatility forecast | Sharpe 0.23, max drawdown 67% | buy & hold: Sharpe 0.50, 53% | no, worse |

Networks against the classical reference (bootstrap, 95% CI of the difference):

- Hourly move size: the ensemble GB + MLP + LSTM (every origin) beats GB alone by +0.010 R² (+0.007 to +0.013). The MLP alone ties GB; the LSTM is worse (−0.012).
- Next-hour direction: LSTM − GB AUC −0.005 to +0.007, a tie.
- Next-day volatility: MLP − GB −0.036 to −0.000; the ensemble ties GB.
- Next-day move size: Keras MLP − Linear Regression −0.17 to −0.03.

---

## ✅ Conclusions

1. **Hourly returns are close to a random walk.** No model (linear, trees, KNN, SVM, MLP, RNN, LSTM) beats predicting 0% change, and the more freedom a model gets, the worse it does on validation. The only dependence is a tiny reversal (lags 1, 2 and 24 h, ≤ 0.4% of the variance).
2. **The only direction signal is the next hour**, mostly the reversal of the last hour: about 55–60 right out of 100 confident calls. Hours 2–24 and the next day are a coin flip.
3. **The size of the moves is predictable**, per hour (R² 0.12–0.14, more than twice the best rule) and per day (R² ≈ 0.48). Prediction ranges can be made both honest (80% coverage in calm and wild hours) and narrower than fixed ones.
4. **Keeping models up to date matters more than more data.** Monthly refits with recency weights are the largest single gain; Binance order flow and ETH add nothing, and the macro variables only move together with BTC: none leads it. Bitcoin's behaviour drifts (e.g. the 14 UTC volatility peak since the ETFs).
5. **Networks do not beat trees on these tables.** They match Gradient Boosting at best; a small gain comes only from ensembling model families trained on every origin. On the small daily table the linear model wins.
6. **A good forecast is not a good strategy.** Volatility targeting with an R² ≈ 0.48 forecast still lost to buy & hold, because of its design (target level, leverage, costs).

---

## 📁 Repository structure

| Path | Content |
|---|---|
| `hourly-base-data.ipynb`, `data-analysis.ipynb`, `classical-models.ipynb`, `neural-models.ipynb` | The four project notebooks (see above) |
| `data/btc_hourly.csv.gz` | Base data: `Open, High, Low, Close, Volume, Return, filled_minutes, no_trade` |
| `data/binance_BTCUSDT_1h.csv.gz`, `data/binance_ETHUSDT_1h.csv.gz` | Binance hourly candles (17 Aug 2017 → 18 Oct 2026), from `python -m utils.binance` |
| `data/btc_hourly_windows.npz`, `data/scaler.joblib` | Windows and scaler of the old data preparation, read only by the archived notebooks |
| `utils/` | Shared helpers imported by the notebooks |
| `old_notebooks/` | Archived notebooks, superseded by the four above. [Their summary](old_notebooks/README.md) |
| `Correlation Analyses/` | Macro correlation notebook (data from 2017), its README and results brief (`.docx`) |
| `report_notes.md` | Notes for the final report |

| Module | Content | Used by |
|---|---|---|
| `utils/hourly.py` | Features, targets, metrics, Ljung–Box, Diebold–Mariano, block bootstrap | data analysis, classical, neural |
| `utils/binance.py` | Binance download from data.binance.vision (SHA-256 checked), order-flow / ETH / ETH-vs-BTC attributes | data analysis, classical |
| `utils/macro.py` | FRED / Yahoo Finance downloads (cached in `.macro_cache/`), D / W / ME panels, hourly macro features without lookahead | data analysis, correlation |
| `utils/correlation.py` | Correlation tables with FDR, pruning, lead-lag, per-year stability, plots | data analysis, correlation |
| `utils/btc_data.py` | Daily / weekly / monthly BTC frames from the base data (`frames_from_hourly`) or the 1-minute file (`load_btc`) | data analysis, correlation |
| `utils/notebooks.py` | Runs the setup cells of version 3 inside another notebook | archived `v3-extensions.ipynb` |

---

## 🚀 How to run

1. **Environment.** Python 3.12 and `pip install -r requirements.txt`. A CUDA GPU speeds up `neural-models.ipynb`; PyTorch falls back to the CPU.
2. **Notebooks.** Run them from the repo root, in the order of the table above. `data/btc_hourly.csv.gz` is already in the repo; rebuilding it gives the same file, because the Kaggle version is pinned.
3. **Binance data.** Already in the repo. To extend it to the latest day: `python -m utils.binance`.
4. **Macro data.** Downloads are cached in `.macro_cache/` and `.btc_cache/`; delete them to force a fresh download.
5. **Archived notebooks.** See [`old_notebooks/README.md`](old_notebooks/README.md).

After changing a file in `utils/`, restart the notebook kernel: Python keeps imported modules in memory.

---

## 🧠 Use of LLMs / AI tools

*Pending: each member adds how they used AI tools.*

---

## 🧭 Project status

*Updated 19 Oct 2026, after unifying the notebooks on branch `feature/experiments-unification`.*

### To-do list

| # | Task | Status | Owner |
|---|---|---|---|
| 1 | One frozen dataset and one train / validation / test split for everybody | ✅ Done: `hourly-base-data.ipynb` + the version 2/3 blocks, used by all four notebooks. Any change to `data/` goes through a pull request | Adrián P. |
| 2 | Merge branches `filip` (PR #7) and `alb` (PR #8) into `main` | ✅ Done | Adrián P. |
| 3 | Archive the earlier notebooks, unify the helper scripts in `utils/`, stop tracking caches | ✅ Done (`old_notebooks/`, `utils/`) | Adrián P. |
| 4 | Rerun every experiment on the agreed dataset | ✅ Done in `classical-models.ipynb` and `neural-models.ipynb` | Adrián P. |
| 5 | Check the data: hidden gaps, outages, `no_trade` hours, extreme moves | ✅ Done in `hourly-base-data.ipynb` | Adrián P. |
| 6 | Normalize each window by itself for the recurrent models | ✅ Done (`neural-models.ipynb` §1.4) | Adrián P. |
| 7 | Rerun RNN/LSTM with normalized windows, a robust loss and the volatility targets | ✅ Done (`neural-models.ipynb` §3, §5), except monthly refits | Adrián P. |
| 8 | **Write the conclusions of `neural-models.ipynb`** (§3–§9 are marked *Pending*) and replace "Run time pending" in its header with ≈ 133 min | ⏳ Pending | *…* |
| 9 | **Review the "What we see" texts** of `data-analysis.ipynb` and `classical-models.ipynb`: they are quoted from the archived runs and some numbers no longer match the outputs (e.g. K-Means cluster sizes, course-methods results). Rewrite them on this run | ⏳ Pending | *…* |
| 10 | **Fix the Yahoo Finance downloads.** The last run of `data-analysis.ipynb` failed with `No module named 'yfinance'`: install it (it is in `requirements.txt`) and rerun §8, so S&P 500, VIX, DXY and gold are back in the lead-lag and stability analyses. HY_OAS covers only 34% of the daily panel | ⏳ Pending | *…* |
| 11 | Link the macro work to the models: Granger causality, and macro features in the next-day volatility model | ⏳ Pending | *…* |
| 12 | Add a GARCH(1,1) benchmark for volatility | ⏳ Pending (optional) | *…* |
| 13 | Decide what to do with `Correlation Analyses/`: `data-analysis.ipynb` §8 now repeats it on the base data (from 2018). Archive it or keep it as the reference for the brief | ⏳ Pending | *…* |
| 14 | Write the *Use of LLMs* section and the final report (from `report_notes.md` and this README) | ⏳ Pending | everybody |

### Possible improvements

**Models**

- **Monthly refits for the networks and the ensembles.** The largest classical gain (hourly \|r\| R² 0.117 → 0.144) has not been tried with networks, nor with the ensemble that already beats Gradient Boosting fitted once.
- **Monthly refit + rolling conformal for the quantile models.** `v3-extensions.ipynb` found the lowest pinball loss with the same 80% coverage; not rerun.
- **Use the `no_trade` / `filled_minutes` flags.** No model uses them: drop or down-weight the origins whose input or target hours contain an outage.
- **Rerun the version 3 parts left out**: volatility-scaled target with Huber loss and 1/σ² weights, permutation importance, the leave-one-out ablation, and the regime analysis of the models.
- **Other networks**: GRU, temporal convolutions or a small Transformer, an LSTM for next-day volatility, quantile networks for the ranges.

**Evaluation**

- **More than one test period.** Every conclusion rests on 2.25 years; a rolling-origin evaluation over 2023–2026 would show how stable the ranking of close models is.
- **Correct for multiple testing** (Holm or Benjamini–Hochberg) across the many direction and return tests, instead of reading single p-values.
- **Economic test with costs for the next-hour signal**, the only direction signal found; and a volatility-targeting design tuned on validation (target level from a recent window, leverage cap, rebalancing threshold to cut turnover).

**Code and reproducibility**

- **Move the shared setup to `utils/`.** `neural-models.ipynb` copies §1.1–1.4 and the helpers of `classical-models.ipynb`; one module would keep both notebooks identical by construction.
- **Update the `utils/hourly.py` docstring**: it still describes the old `.npz` windows, which the current notebooks do not use.
- **Headless runs.** A script that runs the four notebooks with `nbconvert` and records the run times, and a quick mode (fewer seeds, smaller grids) for `neural-models.ipynb`, which takes over 2 h on CPU.
- **Remove the old data files** (`btc_hourly_windows.npz`, `scaler.joblib`) from `data/` once nobody needs to rerun the archived notebooks.
