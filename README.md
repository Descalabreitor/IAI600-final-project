**Bitcoin Prediction Model**

## Main Ideas

- target: Price change (in percentage) each hour (range -1 -> whatever)
- Features (hourly):
    - Volume Traded
    - Open Price
    - Close Price
    - Low Price
    - High Price
    - *More datasets*
- Input: 7 days tensor (7 days, 24 Hours, 6 Elements)
- Output: 1 day array of hourly target (24 hour elements)
---
- Test + train period = 8 days (7 test + 1 train)

---

## Models and hyperparameter tuning

All notebooks read the output of `data-preparation.ipynb` (`data/btc_hourly_windows.npz`, `data/scaler.joblib`) without modifying it. Install the packages with `pip install -r requirements.txt` (Python 3.12).

| File | Content |
|---|---|
| `bitcoin-models-new-data.ipynb` | **Version 1.** Scale-free features, walk-forward CV, Ridge / Lasso / KNN / Random Forest / Gradient Boosting / MLP on the 24 hourly returns, alternative daily targets (return, direction, volatility). |
| `bitcoin-models-2018-2026.ipynb` | **Version 2.** Only 2018–2026 data, fixed train / validation / test blocks (2018–2022 / 2023–Jun 2024 / Jul 2024–Oct 2026), test of whether the early years hurt. |
| `bitcoin-models-v3.ipynb` | **Version 3 (final).** Why the hourly forecasts look like noise, and 14 improvements tested one by one and together (≈45 min to run). |
| `btc_utils.py` | Shared helpers of version 3: rebuilding the continuous hourly series, features, targets, metrics, Ljung–Box, Diebold–Mariano and block-bootstrap. |

**Main results (test block Jul 2024 → Oct 2026):**
- Hourly returns are close to a random walk: no model beats the "0%" forecast in a statistically reliable way (best IC ≈ 0.02, R²_oos ≈ 0).
- The *size* of each of the next 24 hourly moves is predictable: one long-format Gradient Boosting / ensemble model reaches R² ≈ 0.12, 2.2× a simple volatility rule.
- Quantile models give 80% ranges for every hour that are 36% narrower than fixed ranges and adapt to calm and wild periods.
- Next-day volatility is predictable (R² ≈ 0.46–0.50); next-day direction is not (AUC 0.52, not significant).
