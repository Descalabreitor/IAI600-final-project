"""
btc_data.py - BTC OHLCV loading for the correlation analysis.

Only the three frames the analysis uses are built (daily, weekly, monthly). The 1-minute
file is read inside a function, so it is released as soon as the frames are built, and the
resampled result is cached on disk, so later runs do not re-read ~7M rows.

Dependencies: pandas, numpy (kagglehub only if the CSV is not found locally,
matplotlib only for plot_return_hists)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FILENAME = "btcusd_1-min_data.csv"
KAGGLE_DATASET = "mczielinski/bitcoin-historical-data"
OHLCV = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def locate_csv(filename: str = FILENAME, dataset: str = KAGGLE_DATASET) -> Path:
    """Return the local CSV path; if it is not in the working directory, download it."""
    local = Path.cwd() / filename
    if local.is_file():
        print("Using existing dataset:", local)
        return local
    import kagglehub

    src = Path(kagglehub.dataset_download(dataset)) / filename
    if not src.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {src}")
    shutil.copy2(src, local)
    print("Dataset copied to:", local)
    return local


def _resample_all(csv: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(csv, usecols=["Timestamp", *OHLCV])
    df.index = pd.to_datetime(df.pop("Timestamp"), unit="s")
    df.index.name = "Timestamp"
    daily = df.resample("D").agg(OHLCV).dropna()
    del df                                   # the 1-minute frame is not needed anymore
    # first/max/min/last/sum compose, so W and ME can be built from the daily frame
    # and give exactly the same result as resampling the 1-minute data directly
    return {
        "D": daily,
        "W": daily.resample("W").agg(OHLCV).dropna(),
        "ME": daily.resample("ME").agg(OHLCV).dropna(),
    }


def load_btc(csv: str | Path | None = None, cache_dir: str = ".btc_cache",
             refresh: bool = False) -> dict[str, pd.DataFrame]:
    """OHLCV frames {"D", "W", "ME"} (UTC, naive index, label = last day of the period).

    The cache is keyed on the CSV size and modification time, so it is rebuilt
    automatically if the file changes. refresh=True forces a rebuild.
    """
    csv = Path(csv) if csv else locate_csv()
    st = csv.stat()
    cache = Path(cache_dir)
    p = cache / f"btc_{st.st_size}_{int(st.st_mtime)}.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    frames = _resample_all(csv)
    cache.mkdir(exist_ok=True)
    for old in cache.glob("btc_*.pkl"):      # drop caches of previous versions of the file
        old.unlink()
    pd.to_pickle(frames, p)
    return frames


def return_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Descriptive statistics of BTC log returns (in %) for each timeframe."""
    rows = {}
    for f, df in frames.items():
        r = 100 * np.log(df["Close"]).diff().dropna()
        rows[f] = dict(start=df.index[0].date(), n=len(r), mean=r.mean(), std=r.std(),
                       skew=r.skew(), excess_kurtosis=r.kurt(), min=r.min(), max=r.max())
    return pd.DataFrame(rows).T


def plot_return_hists(frames: dict[str, pd.DataFrame], bins: int = 100):
    """Histogram of BTC log returns per timeframe, clipped to the 1st-99th percentile."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 3.5))
    for ax, (f, df) in zip(np.atleast_1d(axes), frames.items()):
        r = 100 * np.log(df["Close"]).diff().dropna()
        low, high = r.quantile([0.01, 0.99])
        r.hist(bins=bins, range=(low, high), ax=ax)
        ax.axvline(r[r.between(low, high)].mean(), color="red", linewidth=0.8, label="Mean")
        ax.set_title(f"{f}: log return (%)")
        ax.legend()
    fig.tight_layout()
    return fig
