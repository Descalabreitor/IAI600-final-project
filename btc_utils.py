"""Helper functions for bitcoin-models-v3.ipynb.

Everything here is a model-side transformation of the files produced by the data-preparation
step (data/btc_hourly_windows.npz and data/scaler.joblib); those files are never modified.

Sections
    1. Data: rebuild the continuous hourly series from the overlapping windows
    2. Features: computed from the hourly series at each forecast origin
    3. Targets
    4. Metrics and statistical tests
"""
import numpy as np
import pandas as pd
import joblib
from scipy import stats

PCT = 100.0          # everything is expressed in percent
H_AHEAD = 24         # hours predicted after each origin
WEEK = 168           # hours in the input window of the data-preparation step
MONTH = 720          # 30 days, the longest lookback used by the extended features


# =====================================================================================
# 1. Data
# =====================================================================================
def load_hourly(path_npz="data/btc_hourly_windows.npz", path_scaler="data/scaler.joblib"):
    """Rebuild the continuous, unscaled hourly series from the overlapping windows.

    Window i covers 168 input hours; its last input hour is the forecast origin T_i, and
    dates[i] is the last of the 24 target hours, so T_i = dates[i] - 24 h. Consecutive windows
    move by one hour, so the first window plus the last row of every other window gives every
    hour exactly once.

    Returns
        hourly  : DataFrame indexed by hour with Open, High, Low, Close, Volume, Return (Return in %)
        origins : DatetimeIndex, the forecast origin T_i of every window (train and test joined)
        y       : array (n_windows, 24), the next 24 hourly returns after each origin, in %
        X, sc   : the joined (still scaled) windows and the scaler, for the verification cell
    """
    d = np.load(path_npz)
    sc = joblib.load(path_scaler)
    cols = list(d["feature_cols"])
    X = np.concatenate([d["X_train"], d["X_test"]])
    y = np.concatenate([d["y_train"], d["y_test"]]).astype(np.float64) * PCT
    dates = pd.DatetimeIndex(np.concatenate([d["dates_train"], d["dates_test"]]))
    origins = dates - pd.Timedelta(hours=H_AHEAD)

    rows = np.concatenate([X[0], X[1:, -1, :]]).astype(np.float64) * sc.scale_ + sc.mean_
    index = pd.date_range(origins[0] - pd.Timedelta(hours=WEEK - 1), origins[-1], freq="h")
    hourly = pd.DataFrame(rows, index=index, columns=cols)
    hourly["Return"] *= PCT
    return hourly, origins, y, X, sc


# =====================================================================================
# 2. Features
# =====================================================================================
def _rolling(s, w, fn="mean"):
    return getattr(s.rolling(w, min_periods=w), fn)()


def base_features(h):
    """The 42 scale-free features of version 1/2, computed at every hour of the series.

    Identical (up to float rounding) to make_features() of the earlier notebooks, which worked on
    one 168-hour window at a time; here the same quantities are rolling statistics of the series.
    """
    r, c, hi, lo = h["Return"], h["Close"], h["High"], h["Low"]
    vol = np.log1p(h["Volume"].clip(lower=0))
    f = {f"ret_lag{k}": r.shift(k - 1) for k in range(1, 25)}
    for w in (24, 72, 168):
        f[f"ret_mean_{w}h"] = _rolling(r, w)
        f[f"ret_std_{w}h"] = r.rolling(w, min_periods=w).std(ddof=0)
    week_hi, week_lo = _rolling(hi, WEEK, "max"), _rolling(lo, WEEK, "min")
    hl = (hi - lo) / c * PCT
    f["ret_24h"] = (c / c.shift(24) - 1) * PCT
    f["ret_7d"] = (c / c.shift(WEEK - 1) - 1) * PCT
    f["pos_in_week_range"] = (c - week_lo) / (week_hi - week_lo + 1e-12)
    f["dist_to_week_mean"] = (c / _rolling(c, WEEK) - 1) * PCT
    f["hl_range_24h"] = _rolling(hl, 24)
    f["hl_range_168h"] = _rolling(hl, WEEK)
    f["volume_24h_vs_week"] = _rolling(vol, 24) - _rolling(vol, WEEK)
    f["volume_last_vs_week"] = vol - _rolling(vol, WEEK)
    start = h.index + pd.Timedelta(hours=1)                    # first predicted hour
    f["hour_sin"] = np.sin(2 * np.pi * start.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * start.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * start.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * start.dayofweek / 7)
    return pd.DataFrame(f, index=h.index)


def long_lookback_features(h):
    """B1: features that need more than one week of history (up to 30 days).

    Realized volatility at three horizons (the HAR model of volatility research: yesterday,
    last week, last month), plus the 30-day trend.
    """
    r, c = h["Return"], h["Close"]
    lr2 = np.log1p(r / PCT) ** 2 * PCT ** 2                    # squared log return, in %^2
    f = {}
    for name, w in (("1d", 24), ("7d", WEEK), ("30d", MONTH)):
        f[f"log_rv_{name}"] = np.log(np.sqrt(_rolling(lr2, w, "sum") * 24 / w) + 1e-6)  # daily units
    f["rv_7d_vs_30d"] = f["log_rv_7d"] - f["log_rv_30d"]
    f["ret_30d"] = (c / c.shift(MONTH - 1) - 1) * PCT
    f["dist_to_30d_mean"] = (c / _rolling(c, MONTH) - 1) * PCT
    f["ret_std_720h"] = r.rolling(MONTH, min_periods=MONTH).std(ddof=0)
    return pd.DataFrame(f, index=h.index)


def range_vol_features(h):
    """B2: volatility estimated from the Open/High/Low/Close of each hour.

    Parkinson: uses only the high-low range. Garman-Klass: also uses open and close.
    Both are known to be more precise than squared returns for the same amount of data.
    """
    lhl = np.log(h["High"] / h["Low"]).clip(lower=0) * PCT
    lco = np.log(h["Close"] / h["Open"]) * PCT
    park = lhl ** 2 / (4 * np.log(2))
    gk = 0.5 * lhl ** 2 - (2 * np.log(2) - 1) * lco ** 2
    f = {}
    for w in (24, WEEK, MONTH):
        f[f"log_parkinson_{w}h"] = np.log(np.sqrt(_rolling(park, w).clip(lower=0)) + 1e-6)
        f[f"log_garman_klass_{w}h"] = np.log(np.sqrt(_rolling(gk, w).clip(lower=0)) + 1e-6)
    return pd.DataFrame(f, index=h.index)


# =====================================================================================
# 3. Targets (all from y, the 24 returns after each origin, in %)
# =====================================================================================
def day_log_return(y):
    return np.log1p(y / PCT).sum(axis=1) * PCT


def day_log_vol(y):
    lr = np.log1p(y / PCT) * PCT
    return np.log(np.sqrt((lr ** 2).sum(axis=1)) + 1e-6)


def compound(y):
    """24 hourly % returns -> one next-day % return (works on predictions too)."""
    return (np.prod(1 + y / PCT, axis=1) - 1) * PCT


# =====================================================================================
# 4. Metrics and statistical tests
# =====================================================================================
def safe_corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return np.nan if np.std(a) < 1e-12 else np.corrcoef(a, b)[0, 1]


def r2_oos(y, p):
    """Out-of-sample R^2 against the zero forecast (the finance convention for returns)."""
    y, p = np.ravel(y), np.ravel(p)
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum()


def r2(y, p):
    y, p = np.ravel(y), np.ravel(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pinball(y, q, tau):
    """Pinball (quantile) loss: the error measure that quantile tau minimizes."""
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def ljung_box(x, lags):
    """Ljung-Box test: are the first `lags` autocorrelations jointly zero? Returns (Q, p-value)."""
    x = np.asarray(x, float) - np.mean(x)
    n, den = len(x), (x ** 2).sum()
    rho = np.array([(x[k:] * x[:-k]).sum() / den for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)


def newey_west_se(d, lags):
    """Standard error of the mean of d, robust to autocorrelation up to `lags` (Newey-West)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = (d ** 2).sum() / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] * d[:-k]).sum() / n
    return np.sqrt(v / n)


def diebold_mariano(loss_ref, loss_model, lags=5):
    """Diebold-Mariano test of 'the model has a lower expected loss than the reference'.

    loss_ref, loss_model: one loss value per independent forecast (e.g. per day).
    Returns (mean loss difference, t statistic, one-sided p-value). A positive difference means the
    model is better.
    """
    d = np.asarray(loss_ref) - np.asarray(loss_model)
    t = d.mean() / newey_west_se(d, lags)
    return d.mean(), t, stats.t.sf(t, len(d) - 1)


def block_bootstrap(stat_fn, n, n_boot=1000, block=10, seed=0):
    """Moving-block bootstrap over n consecutive (daily) observations.

    stat_fn(idx) computes the statistic on the resampled positions idx. Blocks of `block`
    consecutive days keep the short-term dependence (e.g. volatility clustering) intact.
    Returns the array of bootstrap statistics.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        out[b] = stat_fn(idx)
    return out
