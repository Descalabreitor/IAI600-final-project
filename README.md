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
| **Data** | The base file has 76,851 consecutive hours, 1 Jan 2018 → 8 Oct 2026 (pinned Kaggle version 746; earlier years are mostly carried-forward prices with no trades). The analysis and the models use it from **1 Jan 2020** (`DATA_START` in `utils/config.py`): 59,331 hours |
| **Blocks** | Chronological, by the date of the predicted hours. **Train** Jan 2020 → Dec 2022 · **Validation** Jan 2023 → Jun 2024 · **Test** Jul 2024 → Oct 2026, with a 192-hour purge before each border. The dates are in `utils/config.py` |
| **Forecast origin** | Every hour with 30 days of history and 24 hours after it. Independent test days: one forecast per day at 23:00 (829 days) |
| **Features** | Scale-free (returns, volatilities, ranges, volume ratios, calendar), so they mean the same in 2020 and in 2026. The candidate features of the data analysis (volatility ratios, release hours, recent hour profile) are in `utils/features.py` |
| **Evaluation** | Every choice is made on train + validation; the chosen model is refitted on both and scored **once** on test. Skill is measured against simple rules, with block-bootstrap confidence intervals |

Why predict *changes* instead of prices? The price went from about $4,000 (March 2020) to over $126,000 during the data: 72% of the test hours trade above the highest training price. Relative changes are comparable across all those years, absolute prices are not.

---

## 📓 Notebooks

Run them in this order, from the repo root. Each one reads the base data `data/btc_hourly.csv.gz` written by `hourly-base-data.ipynb`; the period, the split dates and the release calendar come from `utils/config.py`, and the candidate features from `utils/features.py`.

| Notebook | Content | Run time |
|---|---|---|
| `hourly-base-data.ipynb` | Builds the base data `data/btc_hourly.csv.gz` | not measured (Kaggle download) |
| `data-analysis.ipynb` | What the data says before any model; builds the candidate features | not measured (macro downloads) |
| `classical-models.ipynb` | Every non-neural experiment of the project | ≈ 45 min (16 cores) |
| `neural-models.ipynb` | Every neural-network experiment, on the same rows and test days | ≈ 41 min (CUDA GPU, RTX 5060 Laptop) |

**`hourly-base-data.ipynb`.** Downloads the pinned Kaggle version and aggregates the minutes into hourly candles, without `dropna()` so the hourly grid stays continuous. It finds that the dataset hides its gaps: a minute without trades is a flat, zero-volume candle at the previous close (95% of 2012, 25–40% of 2013–2016, 1–4% since 2018). That is why the base series starts on 1 Jan 2018. The 15 outages of an hour or more are flagged (`filled_minutes`, `no_trade`), not removed. No split and no scaling: those are model decisions. The other notebooks cut the series at 1 Jan 2020, so the train block (2020–2022) is closer to the test period.

**`data-analysis.ipynb`.** Overview of price and volume by year, the return distribution (excess kurtosis 55 hourly) and the largest moves, all real events. Autocorrelation, Ljung–Box and an AR(24) model show that returns are almost pure noise while their size is strongly autocorrelated. Calendar patterns and their drift (the 14 UTC volatility peak grows from 2022 and jumps with the spot ETFs; weekends get calmer), volatility regimes and K-Means day clusters, Binance and ETH, the macro correlations of `Correlation Analyses/` and the US release days (CPI, payrolls, FOMC). Section 9 turns the findings into candidate features (`utils/features.py`), checks them on train and validation and saves them to `data/features_hourly.csv.gz`. Analyses that could steer a model use only the train block.

**`classical-models.ipynb`.** Brings together versions 1–3 of the models, `v3-extensions.ipynb` and `course-methods.ipynb`: the 24 hourly returns (wide and long format, metric-matched selection, Diebold–Mariano), the next-hour direction, the size of each hourly move (feature sets chosen on validation, monthly refits), 80% prediction ranges with conformal calibration, the three next-day targets (return, direction, volatility, the last one also refitted monthly), the course-lab pipeline, an economic test and bootstrap intervals. Models: Ridge, Lasso, ElasticNet, KNN, SVR / SVC, Random Forest, Extra Trees, Gradient Boosting, ensembles.

**`neural-models.ipynb`.** MLPs (scikit-learn and PyTorch) and LSTMs (PyTorch), on the same origins, blocks and test days as the classical models, so every network is compared with a classical reference (for the size of the hourly moves, Gradient Boosting fitted once and refitted every month). The LSTMs read a 168-hour window of 10 window-normalized channels, with MSE and Huber losses; a second LSTM also reads the clock, the recent hour profile and the release flags of each predicted hour. The MLP is also refitted every month, and the networks are averaged with Gradient Boosting. The PyTorch networks early-stop on the validation block, never on a random hold-out; final networks are averaged over several seeds.

---

## 📊 Results

Last run of `classical-models.ipynb` and `neural-models.ipynb` (data from 1 Jan 2020), test block Jul 2024 → Oct 2026. Every model was chosen without looking at the test block.

| Question | Best model | Test result | Simple rule | Better than the rule? |
|---|---|---|---|---|
| **24 hourly returns** (assigned task) | none beats zero | MAE skill −0.0004 to −0.0002 (networks −0.0020 to −0.0004), IC ≤ 0.021, Diebold–Mariano p ≥ 0.15 | zero forecast | no |
| **Direction of the next hour** | GB classifier (base + Binance) · LSTM, every origin | AUC 0.538, 56.0% right on the 10% most confident calls · AUC 0.538, 59.4% | reversal of the last hour: AUC 0.528 | yes (p < 10⁻⁷), small and before costs |
| **Size of each of the next 24 hours** | GB, long format, refitted monthly (12-month half-life) · LSTM + hour features | R² 0.148 (fitted once 0.131) · R² 0.132 | last week × hour profile: R² 0.056 (0.072 with the recent profile) | yes |
| **80% range for each hour** | GB quantile + rolling conformal | coverage 80.0% (calm 78%, wild 82%), width 0.98%; release hours only 56% | fixed range: 87.6%, width 1.28% | yes, except in release hours |
| **Next-day return** | Ridge | R²_oos −0.001, IC 0.051 | zero forecast | no |
| **Next-day direction** | Random Forest | AUC 0.509, accuracy 49.9% | always up: 50.1% | no |
| **Next-day volatility** | top-3 tree ensemble · GB refitted monthly · GB + MLP | R² 0.510 (95% CI 0.45–0.56) · 0.517 · 0.518 | last week's average: R² 0.094 | yes |
| **Next-day move size** (course pipeline) | Linear Regression | R² 0.514 (SVR 0.519, RF 0.464, Keras MLP 0.399) | same as today: R² ≈ 0 | yes (not bootstrapped) |
| **Volatility targeting** (economic test) | position from the next-day volatility forecast | Sharpe 0.27, max drawdown 64% | buy & hold: Sharpe 0.50, 53% | no, worse |

Monthly refits against fitting once (bootstrap, 95% CI of the difference):

- Size of each hourly move: +0.017 R² (+0.010 to +0.024), better in 20 of 29 test months.
- Next-day volatility: +0.015 R² (+0.004 to +0.025), expanding window (recency weights add nothing here).

Networks against the classical reference (bootstrap, 95% CI of the difference):

- Hourly move size: the LSTM with the features of the predicted hours, chosen on validation, ties GB fitted once (+0.003, −0.006 to +0.010) and is worse than GB refitted monthly (−0.016, −0.025 to −0.008). The MLP refitted monthly ties it (−0.003, −0.008 to +0.002). Averages of GB and the networks reach R² 0.157 on test, but validation did not choose them.
- Next-hour direction: LSTM − GB AUC +0.005 (−0.001 to +0.012), a tie.
- Next-day volatility: the ensemble GB + MLP beats GB by +0.013 (+0.002 to +0.024); the MLP alone ties.
- Next-day move size: Keras MLP − Linear Regression −0.115 (−0.209 to −0.034).

The candidate features of `data-analysis.ipynb` §9 in the models:

- Volatility ratios and minutes without trades (B4), release hours and US session (B5), recent hour profile: **no gain for the trees**. The feature set chosen on validation is base + per-hour columns, the same as before; with monthly refits all the new features together change the hourly R² by −0.0004 (−0.004 to +0.003). Release hours stay under-predicted (forecast ≈ 0.65× the actual size on test): they are too few for the leaves of the tuned Gradient Boosting.
- The recent hour profile improves the simple rule (0.056 → 0.072), and with the clock and the release flags of the predicted hours it lifts the LSTM from 0.098 to 0.132.
- B4 and the releases of the next day leave next-day volatility unchanged (±0.004 walk-forward R²).

---

## ✅ Conclusions

1. **Hourly returns are close to a random walk.** No model (linear, trees, KNN, SVM, MLP, RNN, LSTM) beats predicting 0% change, and the more freedom a model gets, the worse it does on validation. The only dependence is a tiny reversal (lags 1, 2 and 24 h, at most ≈ 0.5% of the variance), whose size changes from year to year.
2. **The only direction signal is the next hour**, mostly the reversal of the last hour: about 55–60 right out of 100 confident calls. Hours 2–24 and the next day are a coin flip.
3. **The size of the moves is predictable**, per hour (R² 0.13 fitted once, 0.148 refitted monthly, against 0.056 for the best rule) and per day (R² ≈ 0.51). Prediction ranges can be made both honest (80% coverage in calm and wild hours) and narrower than fixed ones, except around US macro releases.
4. **Keeping models up to date matters more than more data or more features.** Bitcoin's behaviour drifts (the 14 UTC volatility peak grows from 2022 and jumps in 2024, weekends get calmer), and monthly refits are the largest single gain for both volatility targets. The features suggested by the data analysis, Binance order flow and ETH add nothing to the trees; the macro variables only move together with BTC (none leads it), and the release hours, which do move about 3× more, are too rare for the trees to use.
5. **A network alone does not beat the best classical model.** For the size of the hourly moves the best LSTM ties Gradient Boosting fitted once and is reliably worse than Gradient Boosting refitted every month; the networks help only as a second model family in an ensemble (reliably for next-day volatility, +0.013). On the small daily table the linear model wins.
6. **A good forecast is not a good strategy.** Volatility targeting with an R² ≈ 0.5 forecast still lost to buy & hold, because of its design (target level, leverage, costs).

---

## 📁 Repository structure

| Path | Content |
|---|---|
| `hourly-base-data.ipynb`, `data-analysis.ipynb`, `classical-models.ipynb`, `neural-models.ipynb` | The four project notebooks (see above) |
| `data/btc_hourly.csv.gz` | Base data from 1 Jan 2018: `Open, High, Low, Close, Volume, Return, filled_minutes, no_trade` |
| `data/binance_BTCUSDT_1h.csv.gz`, `data/binance_ETHUSDT_1h.csv.gz` | Binance hourly candles (17 Aug 2017 → 18 Oct 2026), from `python -m utils.binance` |
| `data/macro_raw.csv.gz` | Frozen copy of the FRED / Yahoo Finance macro series (written by the first run of `data-analysis.ipynb` §8), so everybody uses the same macro data |
| `data/release_dates.csv` | US CPI, payrolls and FOMC release times 2018 → 2026 (BLS archives, Federal Reserve calendars), for the release-day analysis (`data-analysis.ipynb` §8.7 and §9) and the release features; rebuilt with `python -m utils.release_dates` |
| `data/features_hourly.csv.gz` | The 19 candidate features of `data-analysis.ipynb` §9, one row per hour from 1 Jan 2020: volatility (`rv_1d`, `rv_7d`, `rv_30d`, `rv_ratio_1d_30d`, `parkinson_1d`, `jump_ratio_1d`), minutes without trades, clock, hour profile of the train block and of the last 90 days (`next_hour_profile_recent`), release hours, hours to / since a release, VIX. Built by `utils/features.py` and checked for lookahead in §9.1; no targets |
| `data/btc_hourly_windows.npz`, `data/scaler.joblib` | Windows and scaler of the old data preparation, read only by the archived notebooks |
| `utils/` | Shared helpers imported by the notebooks |
| `old_notebooks/` | Archived notebooks, superseded by the four above. [Their summary](old_notebooks/README.md) |
| `Correlation Analyses/` | Macro correlation notebook (data from 2017), its README and results brief (`.docx`) |
| `report_notes.md` | Notes for the final report |

| Module | Content | Used by |
|---|---|---|
| `utils/config.py` | Period and split dates shared by every notebook (`DATA_START` = 1 Jan 2020, `VAL_START`, `TEST_START`, `TRAIN_END`, `VAL_END`) and the path of the release calendar | data analysis, classical, neural |
| `utils/features.py` | The features of `data-analysis.ipynb` §9 (`build_features`) and their parts used by the models: origin features (B4), features of the predicted hour (B5), recent hour profile, releases of the next day. Self-test: `python -m utils.features` | data analysis, classical, neural |
| `utils/hourly.py` | Features, targets, metrics, Ljung–Box, Diebold–Mariano, block bootstrap | data analysis, classical, neural |
| `utils/binance.py` | Binance download from data.binance.vision (SHA-256 checked), order-flow / ETH / ETH-vs-BTC attributes | data analysis, classical |
| `utils/macro.py` | FRED / Yahoo Finance downloads (cached in `.macro_cache/`) and the frozen copy in `data/` (`load_macro`), D / W / ME panels, level panels, hourly macro features without lookahead | data analysis, correlation |
| `utils/correlation.py` | Correlation tables with FDR, pruning, lead-lag, per-year stability, plots | data analysis, correlation |
| `utils/btc_data.py` | Daily / weekly / monthly BTC frames from the base data (`frames_from_hourly`) or the 1-minute file (`load_btc`) | data analysis, correlation |
| `utils/notebooks.py` | Runs the setup cells of version 3 inside another notebook | archived `v3-extensions.ipynb` |
| `utils/release_dates.py` | US CPI, payrolls and FOMC release times with their sources; writes `data/release_dates.csv` | data analysis |

---

## 🚀 How to run

1. **Environment.** Python 3.12 and `pip install -r requirements.txt`. A CUDA GPU speeds up `neural-models.ipynb` (`requirements.txt` shows how to install the CUDA build of PyTorch); PyTorch falls back to the CPU.
2. **Notebooks.** Run them from the repo root, in the order of the table above. `data/btc_hourly.csv.gz` is already in the repo; rebuilding it gives the same file, because the Kaggle version is pinned.
3. **Features.** `python -m utils.features` checks the feature module: no lookahead, invariants, and equality with `data/features_hourly.csv.gz`.
4. **Binance data.** Already in the repo. To extend it to the latest day: `python -m utils.binance`.
5. **Macro data.** `data-analysis.ipynb` reads the frozen copy `data/macro_raw.csv.gz`; if it is missing, the first run downloads every series and writes it. To download again set `REFRESH_MACRO = True` in its section 1 and commit the new file through a pull request. `data/` is in `.gitignore`, so new data files need `git add -f`. The `Correlation Analyses/` notebook still downloads through `.macro_cache/` and `.btc_cache/`.
6. **Archived notebooks.** See [`old_notebooks/README.md`](old_notebooks/README.md).

After changing a file in `utils/`, restart the notebook kernel: Python keeps imported modules in memory.

---

## 🧠 Use of LLMs / AI tools

*Pending: each member adds how they used AI tools.*

---

## 🧭 Project status

*Updated 25 Oct 2026, after rerunning the analysis and the models on the data from 2020 (branch `rerun-models`).*

### To-do list

| # | Task | Status | Owner |
|---|---|---|---|
| 1 | One frozen dataset and one train / validation / test split for everybody | ✅ Done: `hourly-base-data.ipynb` + the version 2/3 blocks from 2020, defined once in `utils/config.py` and used by every notebook. Any change to `data/` goes through a pull request | Adrián P. |
| 2 | Merge branches `filip` (PR #7) and `alb` (PR #8) into `main` | ✅ Done | Adrián P. |
| 3 | Archive the earlier notebooks, unify the helper scripts in `utils/`, stop tracking caches | ✅ Done (`old_notebooks/`, `utils/`) | Adrián P. |
| 4 | Rerun every experiment on the agreed dataset | ✅ Done, on the data from 2020, in `data-analysis.ipynb`, `classical-models.ipynb` and `neural-models.ipynb` | Adrián P. |
| 5 | Check the data: hidden gaps, outages, `no_trade` hours, extreme moves | ✅ Done in `hourly-base-data.ipynb` | Adrián P. |
| 6 | Normalize each window by itself for the recurrent models | ✅ Done (`neural-models.ipynb` §1.4) | Adrián P. |
| 7 | Rerun RNN/LSTM with normalized windows, a robust loss and the volatility targets | ✅ Done (`neural-models.ipynb` §3, §5), with monthly refits for the MLP; not for the LSTM (task 16) | Adrián P. |
| 8 | Write the conclusions of `neural-models.ipynb` and its run time | ✅ Done (§3–§8, ≈ 41 min on a GPU) | Adrián P. |
| 9 | Review the "What we see" texts of `data-analysis.ipynb` and `classical-models.ipynb` on this run | ✅ Done: every text rewritten on the run from 2020, and the numbers each notebook quotes from the others checked against their outputs | Adrián P. |
| 10 | Fix the Yahoo Finance downloads and rerun §8 of `data-analysis.ipynb` | ✅ Done | Adrián P. |
| 11 | Link the macro work to the models: Granger causality, and macro features in the volatility models | 🔄 In progress: the release hours and the releases of the next day were tested as features in the model notebooks (no gain); the VIX had no signal within the train block (`data-analysis.ipynb` §9.2) and was not used. Granger still pending | Adrián M. |
| 12 | Add a GARCH(1,1) benchmark for volatility | ⏳ Pending (optional) | *…* |
| 13 | Decide what to do with `Correlation Analyses/`: `data-analysis.ipynb` §8 now repeats it on the base data (from 2020). Archive it or keep it as the reference for the brief | ⏳ Pending | *…* |
| 14 | Write the *Use of LLMs* section and the final report (from `report_notes.md` and this README) | ⏳ Pending | everybody |
| 15 | Feature selection with candidate sets chosen on validation (`data-analysis.ipynb` §9 → `utils/features.py` → the model notebooks) | ✅ Done: `CHOSEN_SET` = base + B3 in both model notebooks | Adrián P. |
| 16 | **Monthly refits of the LSTM** (fine-tune last month's network for a few epochs) and a fairer validation for the networks (early stopping on its own block, or walk-forward folds), so they compete with the refitted Gradient Boosting on equal terms | ⏳ Pending | *…* |
| 17 | **Release hours**: a Gradient Boosting with smaller leaves, or a separate correction, so the models use the release flags (forecast ≈ 0.65× the actual size; 80% ranges cover 56% of those hours) | ⏳ Pending | *…* |

### Possible improvements

**Models**

- **Monthly refits for the ensembles.** GB + MLP, both refitted monthly, reach R² 0.154 on the hourly move size, and with the LSTM 0.157 (against 0.148 for GB alone), but validation did not choose them and they are not bootstrapped (task 16 would make the choice fair).
- **Monthly refit + rolling conformal for the quantile models.** `v3-extensions.ipynb` found the lowest pinball loss with the same 80% coverage; not rerun.
- **Use the `no_trade` / `filled_minutes` flags further.** They enter only as a feature (`filled_minutes_1d`, no gain) and as an LSTM channel: drop or down-weight the origins whose input or target hours contain an outage.
- **Rerun the version 3 parts left out**: volatility-scaled target with Huber loss and 1/σ² weights, permutation importance, the leave-one-out ablation, and the regime analysis of the models.
- **Other networks**: GRU, temporal convolutions or a small Transformer, an LSTM for next-day volatility, quantile networks for the ranges.

**Evaluation**

- **More than one test period.** Every conclusion rests on 2.25 years; a rolling-origin evaluation over 2023–2026 would show how stable the ranking of close models is.
- **Correct for multiple testing** (Holm or Benjamini–Hochberg) across the many direction and return tests, instead of reading single p-values.
- **Economic test with costs for the next-hour signal**, the only direction signal found; and a volatility-targeting design tuned on validation (target level from a recent window, leverage cap, rebalancing threshold to cut turnover).

**Code and reproducibility**

- **Move the shared setup to `utils/`.** `neural-models.ipynb` copies §1.1–1.4 and the helpers of `classical-models.ipynb`; one module would keep both notebooks identical by construction (`utils/config.py` and `utils/features.py` already do it for the period and the features).
- **Update the `utils/hourly.py` docstring**: it still describes the old `.npz` windows, which the current notebooks do not use.
- **Headless runs.** A script that runs the four notebooks with `nbconvert` and records the run times, and a quick mode (fewer seeds, smaller grids) for `neural-models.ipynb`, which takes 41 min on a GPU and over 2 h on a CPU.
- **Remove the old data files** (`btc_hourly_windows.npz`, `scaler.joblib`) from `data/` once nobody needs to rerun the archived notebooks.
