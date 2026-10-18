# ₿ Bitcoin Prediction Model

> Predicting Bitcoin's hourly price changes for the next day from the previous week, using basic machine-learning models.

**Course project** · Dataset: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data), 1-minute data aggregated to hourly

---

## 📑 Contents

0. [Project status: what has been tried and what is left](#-project-status-what-has-been-tried-and-what-is-left)
1. [Main ideas](#-main-ideas)
2. [Repository structure](#-repository-structure)
3. [Data processing and analysis](#-data-processing-and-analysis) *(to be completed)*
4. [Models and hyperparameters](#-models-and-hyperparameters)
5. [Results](#-results)
6. [Conclusions](#-conclusions)
7. [Extra model work](#-extra-model-work) *(to be completed)*
8. [Use of LLMs / AI tools](#-use-of-llms--ai-tools) *(to be completed)*
9. [How to run](#-how-to-run)
10. [Team and contributions](#-team-and-contributions)

---

## 🧭 Project status: what has been tried and what is left

*Written on 18 Oct 2026 after reviewing every notebook on every branch. Its purpose is to get the whole team on the same page. Authors are taken from the git history.*

### Summary of each piece of work

**Data preparation** (`data-preparation.ipynb`, Adrián P., branch `descalabreitor`).
This notebook downloads the 1-minute Kaggle data, aggregates it to hourly candles, computes the hourly return, builds windows (168 input hours → 24 target hours), splits them chronologically, fits a `StandardScaler` on train only and saves everything to `data/btc_hourly_windows.npz`. The latest commit (13 Oct) changed the dataset: it now **starts in 2017 and uses an 80/20 split (test from 26 Oct 2024)**. Before, it started in 2012 with the test from Oct 2023. Every model notebook except `model_test.ipynb` was run on the old version, so their numbers are not reproducible with the file now in the repo. The global scaler does not remove the price trend: scaled test prices have a mean of 3.3 and reach 7.6, far outside the training range.

**Data quality check** (done 18 Oct, not yet in a notebook).
Since 2017 the series is complete: no missing minutes in the raw CSV and no missing hours after aggregation. Only 0.2% of hours have exactly 0% change. There are 13 hours with moves above 10%, and all of them are real events (COVID crash of 12–13 Mar 2020, crash of 19 May 2021, the ETF rejection of Mar 2017, Musk's tweet of 29 Jan 2021), not data errors. Hourly returns have very fat tails (excess kurtosis 35.7). **The data is clean. The noise is the market itself, not a data-quality problem**, so it cannot be "cleaned away". It can only be handled with robust losses, volatility scaling and aggregated targets.

**Models, early draft** (`models-and-hyperparameters-ver1.ipynb`, Alberto).
This was the first model notebook: scale-free features, one validation split, data from 2012. Every model (Ridge, Lasso, KNN, Random Forest, Gradient Boosting, MLP) ended up equal to or slightly worse than predicting 0% change. Its conclusions were never filled in, and it is **superseded by version 1**.

**Models, version 1** (`bitcoin-models-new-data.ipynb`, Alberto).
This version introduced 42 scale-free features, walk-forward cross-validation with a purge gap, and a binomial test on independent days. On the 24 hourly returns no model beats the zero forecast (MAE skill between −0.0003 and −0.0010). The 53.4% daily direction accuracy is an illusion: 53.5% of test days were up days, so the models just say "always up" (balanced accuracy 0.50). Every model gets worse on validation the more freedom it is given, which is the signature of a target that is mostly noise. With aggregated targets some signal appears: the next-day total return has an IC of 0.094, and **next-day volatility is clearly predictable** (Gradient Boosting R² 0.45, against 0.09 for the best simple rule). The current *Results* and *Conclusions* sections of this README come from this notebook.

**Deep models, RNN and LSTM** (`model_test.ipynb`, Adrián P.).
Stacked RNNs and LSTMs in PyTorch read the 168-hour sequence directly, with both scaled and unscaled inputs. All four variants end up equal to the zero forecast (MSE ratio 1.000–1.002) with ~50% direction accuracy. The training loss barely moves from the first epoch. Two things may be holding them back: the inputs are global-scaled prices that lie outside the training range in the test period, and the MSE loss is dominated by the few extreme hours. This is the only notebook already run on the current dataset. Note: the results table has a "GB" row that no cell of the notebook produces (it comes from an old kernel state).

**Models, version 2** (`bitcoin-models-2018-2026.ipynb`, Alberto, **branch `alb`, not merged**).
This version repeats version 1 with data from 2018 only and three fixed blocks: train 2018–2022, validation 2023–Jun 2024, test Jul 2024–Oct 2026. The conclusions do not change. The return models still sit at the zero forecast, and the "always up" illusion disappears because the new test period is balanced (50.4% up days). The best direction model (Logistic Regression, AUC 0.544, 52.5% accuracy) is not significant (p = 0.08). Next-day volatility remains predictable (Gradient Boosting R² 0.47). Removing the early years makes no difference for returns but clearly helps for volatility (R² 0.29 when training from 2012, against 0.41 from 2017/2018).

**Models, version 3** (`bitcoin-models-v3.ipynb` + `btc_utils.py`, Alberto, **branch `alb`, not merged**, takes ~45 min to run).
This is the most complete analysis in the project. It first explains *why* returns are unpredictable: Ljung–Box tests and autocorrelations show only a tiny reversal at lags 1, 2 and 24 hours (≤0.4% of the variance), while the *size* of the moves is strongly autocorrelated (0.27 at lag 1) and follows the clock (1.5× between the busiest and the quietest hour). It then tests 14 improvements one by one and all together: one model for all 24 hours ("long format"), the size |r| of each hour as a new target, quantile ranges, metric-matched model selection with Diebold–Mariano tests, HAR and range-based volatility features, volatility-scaled targets with Huber loss, ensembles and an economic backtest. Results:
- Predicting the size of each of the next 24 hours reaches **R² 0.12 against 0.05 for a simple rule**, and the difference is significant.
- The 80% hourly ranges are 36% narrower than fixed ranges.
- Return direction stays unpredictable (IC 0.01–0.02, DM p ≥ 0.13).
- Volatility targeting with the forecast still loses to buy & hold.
- The extra 30-day and range-based features add nothing.

**Correlation with macro variables** (`Correlation Analyses/`, Adri "no chacho").
This analysis computes Spearman correlations between BTC returns and about 40 macro and market variables (FRED and Yahoo Finance) at daily, weekly and monthly frequency. It drops redundant variables, corrects for multiple testing (FDR), and adds lead-lag, rolling and per-year analyses. The strongest relationships are with credit and financial conditions:
- High-yield spread −0.26 daily, but only 31% data coverage.
- NFCI −0.19 weekly and −0.32 monthly.
- Broad dollar −0.15 daily.
- Net liquidity +0.29 monthly.

Almost nothing relates to the *size* of BTC moves. **All 16 Yahoo Finance series failed to download** (S&P 500, Nasdaq, VIX, DXY, gold, oil…), so 4 of the 7 shortlisted variables are missing from the lead-lag and stability analyses. These results measure co-movement, not prediction, and they are not yet connected to any model.

### Overall picture

All the work points to the same story. **Hourly Bitcoin returns are close to a random walk**: no model (linear, trees, KNN, MLP, RNN, LSTM) beats predicting 0% change, and more flexible models only overfit. **The size of the moves (volatility) is predictable**, both per hour and for the next day, and there tree models clearly beat simple rules. The numbers of versions 1, 2 and 3 are not directly comparable because each uses a different test period.

### To-do list

**Coordination and repository (do first)**

| # | Task | Owner |
|---|---|---|
| 1 | **Agree on one frozen dataset and one train/validation/test split for everybody.** Proposal: the version 2/3 blocks (train 2018–2022, validation 2023–Jun 2024, test Jul 2024–Oct 2026), since version 3 is the most advanced work. From then on, any change to `data/` is announced and goes through a pull request. | *…* |
| 2 | Merge branch `alb` (versions 2 and 3, `btc_utils.py`, `requirements.txt`) into `main`. Expect a conflict in `README.md`, because `alb` edited the old version of it. Also merge `requirements.txt` with the PyTorch dependency of `model_test.ipynb`. | *…* |
| 3 | Rerun every notebook on the agreed dataset, so that all numbers in the report come from the same data. | *…* |
| 4 | Clean the repo: archive or delete `models-and-hyperparameters-ver1.ipynb`, remove the committed `__pycache__/` folders and the `.btc_cache/*.pkl` file, and add them to `.gitignore`. | *…* |

**Data**

| # | Task | Owner |
|---|---|---|
| 5 | Change `data-preparation.ipynb` to the agreed split, and also export the continuous hourly series (version 3 already rebuilds it from the windows). | *…* |
| 6 | Normalize each window by itself (prices relative to the last close of the window, volume relative to the window mean) instead of the global scaler. This is needed for the RNN/LSTM inputs. | *…* |
| 7 | Move the data-quality check (gaps, zero-volume hours, extreme moves) into `data-preparation.ipynb` so it appears in the report. | *…* |
| 8 | Fix the Yahoo Finance downloads in `Correlation Analyses/` (16 series failed), then rerun the lead-lag and stability analyses with the full shortlist. Add a caveat on the 31% coverage of the high-yield spread. | *…* |

**Models and analysis**

| # | Task | Owner |
|---|---|---|
| 9 | Rerun RNN/LSTM with the normalized windows (task 6), a robust loss (Huber or MAE), and the volatility / hourly-size targets, where version 3 found real signal. Compare with version 3's tree models on the same test block. | *…* |
| 10 | Link the macro work to the models. Use Granger causality to check whether any macro variable leads BTC, and if so test it as a feature for the next-day volatility model. | *…* |
| 11 | Optional: add a GARCH(1,1) benchmark for volatility (version 3 has HAR features but no GARCH), and calibrated ranges with conformal prediction (version 3, idea 1). | *…* |

**Report and README**

| # | Task | Owner |
|---|---|---|
| 12 | Update *Main ideas* (the split dates there are out of date) and replace the version 1 tables in *Results* with the final, rerun numbers. | *…* |
| 13 | Write the empty sections: *Data processing and analysis*, *Additional model work* (versions 2–3, RNN/LSTM, correlation analysis) and *LLMs* (each member notes how they used AI tools). | *…* |
| 14 | Merge the correlation brief (`BTC_macro_correlation_results_brief.docx`) and `report_notes.md` into the README/report. | *…* |

---

## 💡 Main ideas

| | |
|---|---|
| **Target** | Price change of each hour relative to the previous hour: `Close[h] / Close[h-1] − 1` (range −1 → ∞; 0.004 = +0.4%) |
| **Features (hourly)** | Open, High, Low, Close, Volume, Return. *More datasets: …* |
| **Input** | 7 days of hourly data: a tensor of 168 hours × 6 features (7 days × 24 hours × 6) |
| **Output** | The 24 hourly changes of the next day |
| **Sample period** | 8 days per sample: 7 days of input + 1 day to predict |
| **Train / test split** | Chronological. Train: Jan 2012 → Oct 2023 · Test: Oct 2023 → Oct 2026 |

Why predict *changes* instead of prices? The price moved from a few dollars to over $100,000. Relative changes are comparable across all those years, while absolute prices are not.

---

## 📁 Repository structure

| File | Content | Owner |
|---|---|---|
| `data-preparation.ipynb` | Download, cleaning, hourly aggregation, windows, scaling | *…* |
| `btc_hourly_windows.npz` | Prepared windows: `X_train`, `X_test`, `y_train`, `y_test`, dates | *…* |
| `scaler.joblib` | `StandardScaler` fitted on the training windows | *…* |
| `bitcoin-models-new-data.ipynb` | Models, hyperparameter tuning, evaluation, alternative targets (**already executed: all results and charts are inside**) | *…* |
| *…* | *…* | *…* |

---

## 🧹 Data processing and analysis


---

## 🤖 Models and hyperparameters

### Step 1: From windows to model inputs

The standardized prices still contain the long-term trend: test-period prices are mostly **above** anything seen in training. Tree models cannot predict outside their training range, and linear models extrapolate it badly.

So each 7-day window is turned into **42 scale-free features**. These mean the same thing in 2017 and in 2026.

| Group | Features |
|---|---|
| Recent returns | the last 24 hourly returns |
| Trend and volatility | mean and standard deviation of returns over 24 h, 72 h and 168 h |
| Price position | 24 h return, 7-day return, position in the week's high–low range, distance to the week's average |
| Intra-hour range | average (High − Low) / Close over 24 h and 168 h |
| Volume | last 24 h and last hour compared with the week |
| Calendar | hour of day and day of week (sine/cosine) |

Other choices, each checked against the data:

- **Training starts in 2017.** In 2012 about half the hours had 0% change (almost no trading), and early volatility was 2–3× higher. An experiment confirmed that adding the early years does not help.
- **Every 6th window is kept for training.** Consecutive windows share 167 of their 168 hours, so this is ~6× faster and loses almost nothing.
- **The first 24 test windows are dropped**, because their targets overlap the last training targets.

### Step 2: Validation method

```
2017 ───────────────────────────────────── Oct 2023 │ Oct 2023 ───────── Oct 2026
Fold 1: [ train ][ val ]                            │
Fold 2: [   train   ][ val ]                        │       TEST
Fold 3: [     train     ][ val ]                    │   (used once per
Fold 4: [        train        ][ val ]              │    final model)
          walk-forward cross-validation             │
```

- **Walk-forward cross-validation** (4 folds) is used to choose hyperparameters. Each fold trains on the past and validates on the following ~16 months, with a gap so overlapping windows cannot leak. Random K-fold would let the model "see the future".
- The best setting is retrained on the full training period and evaluated **once** on the test period.
- Direction accuracy is checked on **~1,075 non-overlapping test days** with a binomial test, to separate skill from luck.

### Step 3: Metrics

| Metric | Meaning | "No skill" value |
|---|---|---|
| **MAE** | Average error per predicted hour, in % | ≈ 0.325% (predicting 0 every hour) |
| **MAE skill** | `1 − MAE_model / MAE_zero`: share of the zero forecast's error that is removed | 0 |
| **R²_oos** | Same idea with squared errors | 0 |
| **IC** | Correlation between predicted and actual changes | 0 |
| **Direction accuracy** | Share of days where up/down is predicted correctly | ≈ 50% |
| **Balanced accuracy** | Average accuracy on up days and on down days, so "always up" cannot cheat | 0.50 |

### Step 4: Models tried

All models are standard scikit-learn models; only their hyperparameters were tuned.

| Model | Hyperparameters tuned | Best setting |
|---|---|---|
| Ridge regression | `alpha` (0.1 → 10⁷) | alpha = 100,000 |
| Lasso regression | `alpha` (0.003 → 0.3) | alpha = 0.1 (removes all features) |
| k-Nearest Neighbors | `n_neighbors`, `weights` | k = 1000, uniform |
| Random Forest | `max_depth`, `min_samples_leaf`, `max_features` | depth 4, leaf 100, 30% of features |
| Gradient Boosting | `learning_rate`, `max_depth` | lr 0.03, depth 2 |
| Small MLP *(optional)* | `hidden_layer_sizes`, `alpha` | (32,), alpha = 100 |

Baselines for comparison: **zero** (no change), **mean per hour** (≈ "always up"), **persistence** (repeat today), and **last-hour reversal**.

---

## 📊 Results

*Test period: 25 Oct 2023 → 5 Oct 2026. Numbers will shift slightly if the data is refreshed.*

### Main target: the 24 hourly changes

| Model | MAE (%) | MAE skill | IC | Daily direction | Balanced acc. |
|---|---|---|---|---|---|
| Baseline: zero | **0.3251** | 0.0000 | none | 46.6% | 0.500 |
| Baseline: mean per hour | 0.3252 | −0.0004 | 0.000 | 53.4% | 0.500 |
| Ridge | 0.3252 | −0.0003 | 0.004 | 53.4% | 0.500 |
| Lasso | 0.3252 | −0.0004 | 0.000 | 53.4% | 0.500 |
| k-Nearest Neighbors | 0.3254 | −0.0010 | 0.001 | 50.3% | 0.485 |
| Random Forest | 0.3252 | −0.0003 | 0.003 | 53.4% | 0.500 |
| Gradient Boosting | 0.3252 | −0.0004 | 0.005 | 53.5% | 0.501 |
| MLP | 0.3254 | −0.0008 | 0.001 | 53.4% | 0.500 |
| Baseline: persistence | 0.4830 | −0.486 | −0.011 | 48.7% | 0.485 |

### Alternative targets (built from the same data)

| Target | Best model | Result | No-skill reference |
|---|---|---|---|
| **A. Next-day total return** (one number) | Ridge | IC = **0.094**, R²_oos = +0.009 | IC = 0, R²_oos = 0 |
| **B. Next-day up/down** (classification) | Logistic Regression | ROC AUC = **0.54** | AUC = 0.50 |
| **C. Next-day volatility** (size of the moves) | Gradient Boosting | R² = **0.45** | R² = 0.09 (last week's average) |

<details>
<summary>Volatility results in detail</summary>

| Model | MAE (log volatility) | R² | Correlation |
|---|---|---|---|
| Baseline: yesterday's volatility | 0.412 | −0.14 | 0.43 |
| Baseline: last week's average | 0.360 | 0.09 | 0.45 |
| Ridge | 0.313 | 0.34 | 0.60 |
| Random Forest | 0.292 | 0.43 | 0.66 |
| **Gradient Boosting** | **0.285** | **0.45** | **0.68** |

</details>

---

## ✅ Conclusions

1. **Hourly changes cannot be predicted better than "no change".** Every tuned model ends up within 0.0003 percentage points of the zero forecast. This matches the well-known behaviour of short-term prices, which are close to a random walk, so it is a correct result rather than a failure of the models.

2. **Beware of direction accuracy.** The models reach 53.4% daily direction accuracy (p ≈ 0.01), but only because 53.5% of test days were up days. They effectively predict "always up", and their balanced accuracy is exactly 0.50.

3. **Hyperparameters behave as theory predicts.** For every model, the more freedom it gets, the worse it does on new data.
   - Random Forest: depth 16 with small leaves reaches train skill +0.016 but validation skill −0.004.
   - Gradient Boosting: a higher learning rate and deeper trees overfit in the same way.
   - KNN: k = 10 gives skill −0.086, while k = 1000 gives −0.001.
   - Ridge, Lasso and MLP: the strongest regularization wins.

   This pattern is the signature of a target that is mostly noise.

4. **Old data does not help.** Training from 2012, 2014, 2017, 2019 or 2021 makes no meaningful difference, and including 2012 gave the worst correlation.

5. **Aggregating helps a little.** Predicting the next day's *total* return gives a weak but measurable signal (IC ≈ 0.09).

6. **Volatility is predictable.** The size of tomorrow's moves can be forecast well. Gradient Boosting explains about 45% of its variation, and here non-linear models clearly beat linear ones.

> **In one sentence:** hourly Bitcoin price changes are essentially unpredictable with basic ML, but how *much* the price will move tomorrow is predictable, and there Gradient Boosting performs best.

---

## 🧪 Aditional model work

---

## 🧠 LLMs


---
