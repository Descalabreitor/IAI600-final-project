"""
macro_panel.py - expanded macro variables and D / W / ME panels for correlation
analysis against BTC.

Differences compared with the old macro_features.py:
  * ~40 variables (rates, curve, liquidity, credit, FX, commodities, global
    equities, inflation, labour, activity).
  * Disk cache (.macro_cache/) and retries: not everything is re-downloaded on
    every run.
  * A series that fails is skipped with a warning; it does not break the whole build.
  * The panel is built at the analysis frequency (D, W, ME), not at the hourly one.
    Each variable is transformed (log return or difference) AFTER being aligned to
    that frequency, so a step function (a monthly value repeated 720 times) is never
    correlated with hourly returns.
  * pit=False -> alignment by observation date (contemporaneous analysis).
    pit=True  -> alignment by availability date (observation + publication lag),
    which is the correct one if you want to claim that X "leads" BTC.

Usage:
    macro = load_raw()                                   # download / read cache ONCE
    panels, meta = build_panels(close, macro)            # close = {"D": Series, "W": ..., "ME": ...}
    panels_pit, _ = build_panels(close, macro, pit=True, freqs=("D", "W"))

Dependencies: pandas, numpy, yfinance
"""
from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(".macro_cache")

# approximate number of days of each native / target frequency
NATIVE_DAYS = {"D": 1, "W": 7, "M": 30}
TARGET_DAYS = {"D": 1, "W": 7, "ME": 30}
# maximum age tolerated when forward-filling a value (prevents a discontinued
# series from staying frozen forever and generating false zeros)
STALE_TOL = {"D": pd.Timedelta(days=10), "W": pd.Timedelta(days=21), "M": pd.Timedelta(days=75)}

# ---------------------------------------------------------------------------
# FRED: name -> (series, publication_lag_days, native_frequency, transform, group)
#   transform: "logret" (log return, for positive levels) or "diff" (difference,
#   for rates, spreads and indices that can be ~0 or negative).
#   Lags: conservative approximations. For full rigour, use ALFRED (vintages),
#   because M2, CPI, PCE, payrolls, etc. are revised after publication.
# ---------------------------------------------------------------------------
FRED_SPECS = {
    # Rates and curve (H.15: published the afternoon of the next business day -> lag 2)
    "US2Y":            ("DGS2",          2, "D", "diff",   "rates"),
    "US10Y":           ("DGS10",         2, "D", "diff",   "rates"),
    "US30Y":           ("DGS30",         2, "D", "diff",   "rates"),
    "Curve_10Y2Y":     ("T10Y2Y",        2, "D", "diff",   "rates"),
    "Curve_10Y3M":     ("T10Y3M",        2, "D", "diff",   "rates"),
    "Real10Y":         ("DFII10",        2, "D", "diff",   "rates"),
    "Breakeven10Y":    ("T10YIE",        2, "D", "diff",   "rates"),
    "FedFunds":        ("DFF",           1, "D", "diff",   "rates"),
    # Credit (ICE BofA: FRED has been limiting the history of licensed series;
    # check the coverage the notebook prints)
    "HY_OAS":          ("BAMLH0A0HYM2",  2, "D", "diff",   "credit"),
    # Dollar and currencies (H.10 is published weekly -> lag 7)
    "USD_broad":       ("DTWEXBGS",      7, "D", "logret", "fx"),
    "USDJPY":          ("DEXJPUS",       7, "D", "logret", "fx"),
    "EURUSD":          ("DEXUSEU",       7, "D", "logret", "fx"),
    # Liquidity
    "FedBalanceSheet": ("WALCL",         2, "W", "logret", "liquidity"),
    "TGA":             ("WTREGEN",       2, "W", "logret", "liquidity"),
    "ON_RRP":          ("RRPONTSYD",     1, "D", "diff",   "liquidity"),
    "M2":              ("M2SL",         55, "M", "logret", "liquidity"),
    # Financial conditions and weekly activity
    "NFCI":            ("NFCI",          5, "W", "diff",   "conditions"),
    "Jobless_claims":  ("ICSA",          5, "W", "logret", "activity"),
    # Monthly (the index is dated on the 1st of the reference month)
    "CPI":             ("CPIAUCSL",     45, "M", "logret", "inflation"),
    "CoreCPI":         ("CPILFESL",     45, "M", "logret", "inflation"),
    "PCE":             ("PCEPI",        60, "M", "logret", "inflation"),
    "CorePCE":         ("PCEPILFE",     60, "M", "logret", "inflation"),
    "Unemployment":    ("UNRATE",       38, "M", "diff",   "activity"),
    "Payrolls":        ("PAYEMS",       38, "M", "logret", "activity"),
    "IndProd":         ("INDPRO",       48, "M", "logret", "activity"),
    "RetailSales":     ("RSAFS",        50, "M", "logret", "activity"),
    "ConsSentiment":   ("UMCSENT",      30, "M", "diff",   "activity"),
}

# ---------------------------------------------------------------------------
# Yahoo Finance (daily, adjusted close): name -> (ticker, group)
# Continuous futures (CL=F, NG=F...) have jumps at rolls and WTI traded negative
# in April 2020; the log return masks prices <= 0.
# ---------------------------------------------------------------------------
YF_SPECS = {
    "SP500":       ("^GSPC",     "equities"),
    "NASDAQ":      ("^IXIC",     "equities"),
    "Russell2000": ("^RUT",      "equities"),
    "EuroStoxx50": ("^STOXX50E", "equities"),
    "Nikkei":      ("^N225",     "equities"),
    "HangSeng":    ("^HSI",      "equities"),
    "VIX":         ("^VIX",      "risk"),
    "DXY":         ("DX-Y.NYB",  "fx"),
    "Gold":        ("GC=F",      "commodities"),
    "Silver":      ("SI=F",      "commodities"),
    "Copper":      ("HG=F",      "commodities"),
    "WTI":         ("CL=F",      "commodities"),
    "Brent":       ("BZ=F",      "commodities"),
    "NatGas":      ("NG=F",      "commodities"),
    "TLT":         ("TLT",       "bonds"),
    "HYG":         ("HYG",       "credit"),
}


# ------------------------------- downloads ---------------------------------
def _fresh(path: Path, max_age_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600


def load_fred(series_id: str, max_age_h: float = 24) -> pd.Series:
    """Public FRED CSV (no API key), with cache and retries."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"fred_{series_id}.csv"
    if not _fresh(p, max_age_h):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        ok, last = False, None
        for attempt in range(3):
            try:
                pd.read_csv(url, index_col=0, parse_dates=True, na_values=".").to_csv(p)
                ok = True
                break
            except Exception as e:  # network, 5xx, timeout...
                last = e
                time.sleep(2 * (attempt + 1))
        if not ok:
            if p.exists():
                warnings.warn(f"{series_id}: download failed, using old cache ({last})")
            else:
                raise RuntimeError(f"{series_id}: download failed ({last})") from last
    s = pd.read_csv(p, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna().astype(float)


def load_yf(tickers: list[str], start: str, max_age_h: float = 24) -> pd.DataFrame:
    """Daily closes from Yahoo Finance with cache. Index = date (naive, normalized)."""
    import yfinance as yf

    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5((",".join(sorted(tickers)) + start).encode()).hexdigest()[:10]
    p = CACHE / f"yf_{key}.pkl"
    if _fresh(p, max_age_h):
        return pd.read_pickle(p)
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    idx = pd.DatetimeIndex(px.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    px.index = idx.normalize()
    px.to_pickle(p)
    return px


# ------------------------------- alignment ---------------------------------
def _to_available(s: pd.Series, native: str, lag: int, pit: bool) -> pd.Series:
    """Reindex a series by the moment the value is available.

    pit=True : observation date + publication lag (what was known at the time).
    pit=False: end of the observation period (end of day; end of month if monthly).
    """
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if pit:
        idx = s.index + pd.Timedelta(days=lag)
    elif native == "M":
        idx = s.index + pd.offsets.MonthBegin(1)
    else:
        idx = s.index + pd.Timedelta(days=1)
    return pd.Series(s.values, index=idx).sort_index()


def _transform(x: pd.Series, how: str) -> pd.Series:
    if how == "logret":
        return np.log(x.where(x > 0)).diff()     # prices <= 0 (WTI 2020) -> NaN
    if how == "diff":
        return x.diff()
    raise ValueError(how)


# components of the derived NetLiquidity series
NET_LIQ_PARTS = ("FedBalanceSheet", "TGA", "ON_RRP")


@dataclass
class MacroData:
    """Raw downloaded series (indexed by observation date) and their metadata."""
    raw: dict
    meta: pd.DataFrame      # name, source, id, native, lag, transform, group
    failed: list


def load_raw(data_start: str = "2012-01-01", refresh_hours: float = 24,
             fred_specs=None, yf_specs=None) -> MacroData:
    """Download (or read from cache) every series once. Failed series are skipped."""
    fred_specs = FRED_SPECS if fred_specs is None else fred_specs
    yf_specs = YF_SPECS if yf_specs is None else yf_specs
    raw, meta, failed = {}, [], []

    for name, (sid, lag, native, tf, group) in fred_specs.items():
        try:
            raw[name] = load_fred(sid, refresh_hours)
        except Exception as e:
            failed.append((name, str(e)[:90]))
            continue
        meta.append(dict(name=name, source="FRED", id=sid, native=native, lag=lag,
                         transform=tf, group=group))

    try:
        px = load_yf([t for t, _ in yf_specs.values()], data_start, refresh_hours)
    except Exception as e:
        px = pd.DataFrame()
        failed.append(("yfinance", str(e)[:90]))
    for name, (tkr, group) in yf_specs.items():
        if tkr not in px.columns or px[tkr].dropna().empty:
            failed.append((name, f"no data for {tkr}"))
            continue
        raw[name] = px[tkr].dropna()
        meta.append(dict(name=name, source="Yahoo", id=tkr, native="D", lag=1,
                         transform="logret", group=group))

    # Net liquidity = Fed balance sheet - TGA - ON RRP (derived in align_levels).
    if all(k in raw for k in NET_LIQ_PARTS):
        meta.append(dict(name="NetLiquidity", source="derived", id="WALCL-WTREGEN-RRP",
                         native="W", lag=0, transform="logret", group="liquidity"))

    if failed:
        print("Skipped series:", *[f"  {n}: {m}" for n, m in failed], sep="\n")
    return MacroData(raw, pd.DataFrame(meta), failed)


def align_levels(macro: MacroData, pit: bool) -> dict:
    """Index every series by the moment it is available (see _to_available)."""
    spec = macro.meta.set_index("name")
    levels = {n: _to_available(s, spec.at[n, "native"], int(spec.at[n, "lag"]), pit)
              for n, s in macro.raw.items()}
    if "NetLiquidity" in spec.index:
        # Units on FRED: WALCL and WTREGEN in millions, RRPONTSYD in billions of USD
        # (check that the result is ~ 5,000-6,500 billion in 2022-2024).
        d = pd.concat({k: levels[k] for k in NET_LIQ_PARTS}, axis=1).sort_index().ffill()
        d["ON_RRP"] = d["ON_RRP"].fillna(0)      # before Sep-2013 the facility did not exist
        levels["NetLiquidity"] = (d["FedBalanceSheet"] / 1e3 - d["TGA"] / 1e3 - d["ON_RRP"]).dropna()
    return levels


def build_panels(btc_close: dict, macro: MacroData, pit: bool = False, freqs=None,
                 start: str = "2017-01-01", weekdays_only: bool = True, variables=None):
    """Panels (BTC + transformed macro variables) at frequency D, W and/or ME.

    btc_close : {"D": Series, "W": Series, "ME": Series} of BTC Close, already
                resampled (label = last calendar day of the period).
    macro     : result of load_raw().
    pit       : False = contemporaneous (by observation date);
                True  = by availability date (for lead-lag / "leads").
    freqs     : subset of btc_close to build (default: all).
    variables : optional list of macro variables to keep (default: all).
    weekdays_only (only freq='D'): drops Saturdays and Sundays, so that BTC's Monday
                return is Friday->Monday, just like the equity return.
                With a full calendar, equities are 0 on weekends while BTC moves,
                which artificially attenuates correlations.
    Variables whose native frequency is slower than the panel's are excluded (e.g. CPI
    is not in the daily or weekly panel).
    Returns ({freq: panel}, meta indexed by variable name).
    """
    freqs = list(btc_close) if freqs is None else list(freqs)
    levels = align_levels(macro, pit)
    meta = macro.meta
    if variables is not None:
        meta = meta[meta["name"].isin(variables)]

    panels, kept = {}, set()
    for freq in freqs:
        if freq not in TARGET_DAYS:
            raise ValueError("freq must be 'D', 'W' or 'ME'")
        btc = btc_close[freq].dropna()
        if freq == "D" and weekdays_only:
            btc = btc[btc.index.dayofweek < 5]
        # With resample D/W/ME the label is the last calendar day of the period, and
        # the period ends 1 day later at 00:00.
        end = btc.index + pd.Timedelta(days=1)
        logp = np.log(btc)
        cols = {"BTC_ret": logp.diff(), "BTC_absret": logp.diff().abs()}
        for m in meta.to_dict("records"):
            if NATIVE_DAYS[m["native"]] > TARGET_DAYS[freq]:
                continue
            x = levels[m["name"]].reindex(end, method="ffill", tolerance=STALE_TOL[m["native"]])
            x.index = btc.index
            cols[m["name"]] = _transform(x, m["transform"])
            kept.add(m["name"])
        panels[freq] = pd.DataFrame(cols).loc[pd.Timestamp(start):]

    return panels, macro.meta[macro.meta["name"].isin(kept)].set_index("name")
