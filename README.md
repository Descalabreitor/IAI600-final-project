# ₿ Bitcoin Prediction Model

> Predicting Bitcoin's hourly price changes for the next day from the previous week, using basic machine-learning models.

**Course project** · Dataset: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data), 1-minute data aggregated to hourly

---

## 📑 Contents

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
