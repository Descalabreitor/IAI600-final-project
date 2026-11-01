# Predicting Bitcoin's Hourly Price Changes

> The direction is noise, the size is not. We predict Bitcoin's hourly price changes for the next day from the previous weeks, with classical machine-learning models, neural networks and a pretrained time-series foundation model, and test what the forecasts are worth to a trader.

**IAI600 final project** · University West, Trollhättan · MSc AI

**Team:** Alberto Prats · Adrián Perera Moreno · Filip Botlo · Adrián Muñiz Dos Santos

**Report:** [`latex_report/main.pdf`](latex_report/main.pdf) (IEEE conference format, source in [`latex_report/`](latex_report/))

**Dataset:** [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data), 1-minute Bitstamp data aggregated to hourly

---

## Contents

1. [Summary](#summary)
2. [Setup of the problem](#setup-of-the-problem)
3. [Notebooks](#notebooks)
4. [Results](#results)
5. [Conclusions](#conclusions)
6. [Limitations](#limitations)
7. [How to run](#how-to-run)
8. [Repository structure](#repository-structure)
9. [Team contributions](#team-contributions)
10. [Use of generative AI](#use-of-generative-ai)

---

## Summary

- **Hourly returns are close to a random walk.** No model (linear, trees, KNN, SVM, MLP, LSTM, Timestral) beats predicting a 0% change for the 24 hours of the next day.
- **The size of the moves is predictable.** For each of the next 24 hours, R² 0.164 on test (Timestral zero-shot averaged with a monthly refitted Gradient Boosting) against 0.072 for the best simple rule; for the next day's volatility, R² ≈ 0.51.
- **Honest prediction ranges.** 80% ranges with rolling conformal calibration cover 80.0% of the test hours, in calm and wild periods, and are narrower than fixed ranges; calibrating the US macro release hours apart lifts their coverage from 56% to 82%.
- **Keeping models up to date matters more than more features.** Monthly refits are the largest single gain; the extra features from the data analysis, the Binance order flow, ETH and the macro series add nothing to the trees.
- **A good risk forecast makes trading safer, not richer.** With $10,000, risk-sized positions cut the worst day, the chance of a large drawdown and the number of liquidations under leverage, but no forecast earns more money than the naive rule.

---

## Setup of the problem

| | |
|---|---|
| **Task** | Predict the 24 hourly price changes of the next day, `Close[h] / Close[h-1] − 1`, from the previous hours |
| **Data** | 76,851 consecutive hours, 1 Jan 2018 → 8 Oct 2026 (pinned Kaggle version 746). Analysis and models use it from **1 Jan 2020** (`DATA_START` in `utils/config.py`): 59,331 hours |
| **Split** | Chronological, by the date of the predicted hours. **Train** Jan 2020 → Dec 2022 · **Validation** Jan 2023 → Jun 2024 · **Test** Jul 2024 → Oct 2026, with a 192-hour purge before each border (`utils/config.py`) |
| **Forecast origin** | Every hour with 30 days of history and 24 hours after it. Independent test days: one forecast per day at 23:00 (829 days) |
| **Features** | Scale-free (returns, volatilities, ranges, volume ratios, calendar), so they mean the same in 2020 and in 2026 (`utils/features.py`) |
| **Evaluation** | Every choice is made on train + validation; the chosen model is refitted on both and scored **once** on test. Skill is measured against simple rules, with block-bootstrap confidence intervals and Diebold–Mariano tests |

Why predict *changes* instead of prices? The price went from about $4,000 (March 2020) to over $126,000 during the data: 72% of the test hours trade above the highest training price. Relative changes are comparable across all those years, absolute prices are not.

---

## Notebooks

Run them from the repo root, in this order. Every notebook reads the base data `data/btc_hourly.csv.gz`; the period, the split dates and the release calendar come from `utils/config.py`, the features from `utils/features.py`.

| # | Notebook | Content | Run time |
|---|---|---|---|
| 1 | [`hourly-base-data.ipynb`](hourly-base-data.ipynb) | Builds the base data from the Kaggle 1-minute file | not measured (Kaggle download) |
| 2 | [`data-analysis.ipynb`](data-analysis.ipynb) | What the data says before any model; builds the candidate features | not measured |
| 3 | [`classical-models.ipynb`](classical-models.ipynb) | Every non-neural experiment | ≈ 45 min (16 cores) |
| 4 | [`neural-models.ipynb`](neural-models.ipynb) | Every neural-network experiment, on the same rows and test days | ≈ 33 min (RTX 5060 Laptop GPU) |
| 5 | [`timestral-zero-shot.ipynb`](timestral-zero-shot.ipynb) | Mistral's pretrained Timestral, used zero-shot, on the same rows and test days | < 2 h the first time (GPU), ≈ 3 min from the cache |
| 6 | [`trading-simulation.ipynb`](trading-simulation.ipynb) | $10,000 traded with and without the models' risk forecasts | not measured; Timestral forecasts of §5.1 ≈ 50 min without a GPU |
| – | [`loading-models.ipynb`](loading-models.ipynb) | How to load each saved model from `models/` and forecast with it, without training | about a minute |

**1. `hourly-base-data.ipynb`.** Downloads the pinned Kaggle version and aggregates the minutes into hourly candles, keeping the hourly grid continuous. The dataset hides its gaps: a minute without trades is a flat, zero-volume candle at the previous close (95% of 2012, 25–40% of 2013–2016, 1–4% since 2018), which is why the series starts in 2018. The 15 outages of an hour or more are flagged (`filled_minutes`, `no_trade`), not removed.

**2. `data-analysis.ipynb`.** Price and volume by year, the return distribution (excess kurtosis 55) and the largest moves. Autocorrelation, Ljung–Box and an AR(24) model show that returns are almost pure noise while their size is strongly autocorrelated. Calendar patterns and their drift (the 14 UTC volatility peak grows from 2022 and jumps with the spot ETFs; weekends get calmer), volatility regimes and K-Means day clusters, Binance and ETH, macro correlations and US release days (CPI, payrolls, FOMC). Section 9 turns the findings into candidate features. Analyses that could steer a model use only the train block.

**3. `classical-models.ipynb`.** The 24 hourly returns (wide and long format, metric-matched selection, Diebold–Mariano), the next-hour direction, the size of each hourly move (feature sets chosen on validation, monthly refits), 80% prediction ranges with conformal calibration, three next-day targets (return, direction, volatility), the course-lab pipeline, an economic test and bootstrap intervals. Models: Ridge, Lasso, ElasticNet, KNN, SVR / SVC, Random Forest, Extra Trees, Gradient Boosting, ensembles.

**4. `neural-models.ipynb`.** MLPs (scikit-learn and PyTorch) and LSTMs (PyTorch), each compared with a classical reference on the same rows. The LSTMs read a 168-hour window of 10 window-normalized channels (MSE and Huber losses); the best one also reads the clock and release flags of each predicted hour, attends to the week through a volume mask and is fine-tuned every month. Early stopping uses the last 6 months of train, never a random hold-out; final networks are averaged over several seeds.

**5. `timestral-zero-shot.ipynb`.** Timestral, the time-series foundation model Mistral released on 15 Oct 2026, used as it comes: its weights never see our data. For each origin it reads the last 30–180 days of the hourly series and returns 21 quantiles of each of the next 24 hours, optionally with covariates (clock, release calendar, past range and volume, ETH, Binance order flow, VIX, GARCH). Series, context, covariates and model size are chosen on validation. Also: averaging and stacking with Gradient Boosting, GARCH(1,1) baselines, checks against lookahead, and an evaluation of the whole forecast distribution (quantile reliability, CRPS). The 28M model is used; the 209M model added 0.003 R² for 7× the compute.

**6. `trading-simulation.ipynb`.** Four flows trade $10,000 with identical rules (trend filter, position sized to a 1.5% daily loss budget, hourly stop-loss, 0.10% fees) and differ only in their estimate of tomorrow's risk: last week's volatility, Gradient Boosting, Timestral, or their average. It backtests the loss limit (Kupiec, Christoffersen), tail losses, Monte Carlo drawdowns and liquidations under leverage, and tests whether Timestral predicts the direction.

---

## Results

Test block Jul 2024 → Oct 2026. Every model was chosen without looking at the test block.

| Question | Best model | Test result | Simple rule | Better than the rule? |
|---|---|---|---|---|
| **24 hourly returns** (assigned task) | none beats zero | MAE skill −0.0004 to −0.0002 (networks −0.0020 to −0.0004), IC ≤ 0.021, Diebold–Mariano p ≥ 0.15 | zero forecast | no |
| **Direction of the next hour** | GB classifier · LSTM | AUC 0.538, 56.0% right on the 10% most confident calls · AUC 0.538, 59.9% | reversal of the last hour: AUC 0.528 | yes (p < 10⁻⁷), small and before costs |
| **Size of each of the next 24 hours** | Timestral + GB refitted monthly · Timestral alone · LSTM fine-tuned monthly · GB refitted monthly | R² **0.164** · 0.157 · 0.156 · 0.148 (GB fitted once 0.131) | last week × recent (90-day) hour profile: R² 0.072 | yes |
| **80% range for each hour** | GB quantile + rolling Mondrian conformal · Timestral q10–q90, no calibration | coverage 80.0% (calm 78%, wild 82%, release hours 82%), width 0.98% · coverage 80.3%, width 1.00%, release hours 70% | fixed range: 87.6%, width 1.28% | yes |
| **Next-day return** | Ridge | R²_oos −0.001, IC 0.051 | zero forecast | no |
| **Next-day direction** | Random Forest | AUC 0.509, accuracy 49.9% | always up: 50.1% | no |
| **Next-day volatility** | top-3 tree ensemble · GB refitted monthly · GB + MLP | R² 0.510 (95% CI 0.45–0.56) · 0.517 · 0.516 | last week's average: R² 0.094 | yes |
| **Next-day move size** (course pipeline) | Linear Regression | R² 0.514 (SVR 0.519, RF 0.464, Keras MLP 0.399) | same as today: R² ≈ 0 | yes |
| **Trading $10,000** (risk-sized, trend filter) | GB + Timestral risk forecast | end capital $12,278 vs $12,388 for the naive flow; liquidations at ~13× leverage 3 vs 12; P(drawdown > 20% in a year) 20.8% vs 27.2% | last week's volatility | safer, not richer |

Selected comparisons (block bootstrap, 95% CI of the difference):

- **Monthly refits vs fitting once:** +0.017 R² (+0.010 to +0.024) for the hourly move size, better in 20 of 29 test months; +0.015 R² (+0.004 to +0.025) for next-day volatility.
- **LSTM vs GB refitted monthly** (hourly move size): +0.008 (+0.001 to +0.015). Each ingredient helps on test: hour features +0.031, volume attention +0.014, monthly fine-tuning +0.011. The MLP only ties the trees.
- **Timestral zero-shot vs GB refitted monthly** (hourly move size): +0.009 (+0.002 to +0.016) alone, +0.016 (+0.013 to +0.020) averaged with GB. More covariates (ETH, VIX, order flow, GARCH) and stacking into GB do not improve on the plain average.
- **Timestral's ranges** are calibrated as they come: every one of its 21 quantile levels is within 1.4 points of its nominal level on test, and its CRPS is the lowest of the ranges tried.
- **Next-day volatility:** the ensemble GB + MLP beats GB by +0.011 (+0.001 to +0.022).

---

## Conclusions

1. **Hourly returns are close to a random walk.** No model beats predicting 0% change, and the more freedom a model gets, the worse it does on validation. The only dependence is a tiny reversal (lags 1, 2 and 24 h, at most ≈ 0.5% of the variance), whose size changes from year to year.
2. **The only direction signal is the next hour**, mostly the reversal of the last hour: about 55–60 right out of 100 confident calls. Hours 2–24 and the next day are a coin flip, also for Timestral.
3. **The size of the moves is predictable**, per hour (R² up to 0.164) and per day (R² ≈ 0.51). Prediction ranges can be both honest (80% coverage in calm and wild hours) and narrower than fixed ones; around US macro releases they need to be calibrated apart (56% → 82% coverage).
4. **Keeping models up to date matters more than more data or more features.** Bitcoin's behaviour drifts, and monthly refits are the largest single gain for both volatility targets. The features suggested by the data analysis add nothing to the trees; the macro variables move together with BTC but none leads it.
5. **Networks win narrowly, and only with care.** An LSTM beats the refitted Gradient Boosting once it knows the hours it predicts, reads the week through a volume attention mask and is fine-tuned every month. On the small daily table the linear model wins.
6. **A pretrained model, used as it comes, does as well as anything we trained.** Timestral 28M, with nothing fitted on our data, beats the refitted Gradient Boosting and ties the fine-tuned LSTM; averaged with Gradient Boosting it gives the best result of the project. Its ranges need no calibration.
7. **A good forecast is not a good strategy.** A risk forecast makes positions and leverage much safer (fewer liquidations, smaller worst days, fewer large drawdowns), but knowing the size of a move without its direction does not earn money: no forecast ends with more money than the naive rule.

---

## Limitations

- **One test period** of 2.25 years; a rolling-origin evaluation over 2023–2026 would show how stable the ranking of close models is.
- **Timestral's training data is not public.** It was released after our test block; if Bitcoin 2023–2026 is in its training series, its validation and test results are optimistic.
- **Multiple testing.** Many direction and return tests were run; the p-values are not corrected (Holm or Benjamini–Hochberg).
- **Trading simulation:** two market periods with one price path each, stops executed at the hour's close, no slippage beyond the fee and no funding cost for leveraged positions.

---

## How to run

1. **Environment.** Python 3.12 and `pip install -r requirements.txt`, which also installs the `timestral` package from its GitHub repository (pinned commit). A CUDA GPU speeds up `neural-models.ipynb` and `timestral-zero-shot.ipynb` (`requirements.txt` shows how to install the CUDA build of PyTorch); PyTorch falls back to the CPU. The Timestral weights (28M, 111 MB) are downloaded to the Hugging Face cache on first use.
2. **Data.** Everything the notebooks need is already in `data/`: the base data, Binance BTC and ETH candles, a frozen copy of the macro series and the US release calendar. Rebuilding the base data gives the same file, because the Kaggle version is pinned.
3. **Notebooks.** Run them from the repo root, in the order of the [table above](#notebooks). Trained models are saved in `models/` and Timestral forecasts cached in `data/timestral_cache/` (both local only, rebuilt by the notebooks).
4. **Checks.** `python -m utils.features` checks the feature module: no lookahead, invariants, and equality with `data/features_hourly.csv.gz`.

---

## Repository structure

| Path | Content |
|---|---|
| `*.ipynb` (root) | The project notebooks (see [Notebooks](#notebooks)) |
| `latex_report/` | Final report (`main.pdf`) and its LaTeX source |
| `data/btc_hourly.csv.gz` | Base data from 1 Jan 2018: `Open, High, Low, Close, Volume, Return, filled_minutes, no_trade` |
| `data/binance_BTCUSDT_1h.csv.gz`, `data/binance_ETHUSDT_1h.csv.gz` | Binance hourly candles (Aug 2017 → Oct 2026) |
| `data/macro_raw.csv.gz` | Frozen copy of the FRED / Yahoo Finance macro series |
| `data/release_dates.csv` | US CPI, payrolls and FOMC release times 2018 → 2026 |
| `data/features_hourly.csv.gz` | The 19 candidate features of `data-analysis.ipynb` §9, one row per hour from 1 Jan 2020 |
| `utils/` | Shared helpers imported by the notebooks (table below) |
| `Correlation Analyses/` | Standalone macro correlation study (data from 2017) with its own README and results brief |
| `old_notebooks/` | Earlier notebooks, superseded by the ones above and kept for traceability ([summary](old_notebooks/README.md)); their numbers are not comparable with the current ones |

| Module | Content |
|---|---|
| `utils/config.py` | Period and split dates shared by every notebook, path of the release calendar |
| `utils/features.py` | The candidate features (`build_features`) and the parts used by the models |
| `utils/hourly.py` | Features, targets, metrics, Ljung–Box, Diebold–Mariano, block bootstrap |
| `utils/binance.py` | Binance download (SHA-256 checked), order-flow and ETH attributes |
| `utils/macro.py` | FRED / Yahoo Finance downloads, frozen copy, macro panels and hourly macro features without lookahead |
| `utils/correlation.py` | Correlation tables with FDR, lead-lag, per-year stability, plots |
| `utils/btc_data.py` | Daily / weekly / monthly BTC frames |
| `utils/saving.py` | Saving and loading of the trained models (`ModelSaver`, `load_model`) |
| `utils/release_dates.py` | US release times with their sources; writes `data/release_dates.csv` |
| `utils/notebooks.py` | Helper for the archived notebooks |

---

## Team contributions

| Member | Contribution |
|---|---|
| Alberto Prats | Classical models and hyperparameter tuning |
| Adrián Perera Moreno | Data pipeline, unification of the notebooks, neural networks and Timestral |
| Filip Botlo | Binance features, monthly refits, conformal ranges, course-method pipeline, trading simulation |
| Adrián Muñiz Dos Santos | Exploratory data analysis and macro correlations |

---

## Use of generative AI

Claude Code (Anthropic) was used for assistance in coding, debugging, refactoring, git operations, wording of the notebook texts and drafting of the report. No language model is part of the forecasting solution: Timestral is a time-series model, used zero-shot.
