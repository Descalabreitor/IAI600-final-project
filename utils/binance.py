"""Binance BTCUSDT / ETHUSDT data: download, hourly candles and the attributes built from them.

Download (run once from the repo root; the hourly files are already in the repo):
    python -m utils.binance                    # both symbols
    python -m utils.binance BTCUSDT            # one symbol
Source: https://data.binance.vision (public, no API key). Monthly zip files for finished months,
daily zip files for the current month, each checked against its published SHA-256. Raw 1-minute
files go to data/raw/binance/<SYMBOL>/ (not in git), hourly candles (UTC, start of the hour) to
data/binance_<SYMBOL>_1h.csv.gz.

Attributes: every attribute at hour t uses only hours up to and including t, and all of them are
scale-free. Groups:
    flow   BTC order flow: share of taker buys, net buying, number and size of trades
    eth    ETH's own moves, volatility, buy share and volume
    cross  ETH relative to BTC: relative moves, correlation, beta, ETH/BTC ratio position
"""
import hashlib
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- download
BASE = "https://data.binance.vision/data/spot"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START = pd.Timestamp("2017-08-01")         # first month Binance has data for
RAW_DIR = Path("data/raw/binance")
OUT_DIR = Path("data")

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]


def file_urls(symbol, today):
    # monthly files for finished months, daily files for the current month (up to yesterday)
    this_month = today.to_period("M")
    for m in pd.period_range(START, this_month - 1, freq="M"):
        yield f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{m.strftime('%Y-%m')}.zip"
    for d in pd.date_range(this_month.start_time, today - pd.Timedelta(days=1), freq="D"):
        yield f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{d.strftime('%Y-%m-%d')}.zip"


def download(url, folder):
    path = folder / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    try:
        content = urlopen(url, timeout=60).read()
        expected = urlopen(url + ".CHECKSUM", timeout=60).read().decode().split()[0]
    except HTTPError as e:
        if e.code == 404:          # e.g. yesterday's daily file not published yet
            print(f"  missing: {url}")
            return None
        raise
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"checksum mismatch: {url}")
    path.write_bytes(content)
    return path


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS)
    if not str(df.iloc[0, 0]).isdigit():   # some files have a header row
        df = df.iloc[1:]
    df = df.drop(columns=["close_time", "ignore"]).apply(pd.to_numeric)
    # Binance switched spot timestamps from milliseconds to microseconds in 2025
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit)
    return df


def to_hourly(minutes):
    hourly = minutes.set_index("open_time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum", "trades": "sum",
        "taker_buy_base_volume": "sum", "taker_buy_quote_volume": "sum",
    })
    hourly["minutes"] = minutes.set_index("open_time")["close"].resample("1h").count()
    hourly.index.name = "timestamp"
    return hourly[hourly["minutes"] > 0]       # drop hours with no trading at all (exchange downtime)


def main(symbols):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    for symbol in symbols:
        folder = RAW_DIR / symbol
        folder.mkdir(parents=True, exist_ok=True)
        urls = list(file_urls(symbol, today))
        print(f"{symbol}: {len(urls)} files")
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = [p for p in pool.map(lambda u: download(u, folder), urls) if p is not None]

        minutes = pd.concat([read_zip(p) for p in sorted(paths)], ignore_index=True)
        minutes = minutes.drop_duplicates("open_time").sort_values("open_time")
        hourly = to_hourly(minutes)

        out = OUT_DIR / f"binance_{symbol}_1h.csv.gz"
        hourly.to_csv(out)
        print(f"{symbol}: {len(minutes):,} minutes -> {len(hourly):,} hours "
              f"({hourly.index[0]} -> {hourly.index[-1]}), saved to {out}")


# ---------------------------------------------------------------- loading and attributes
def load_binance(symbol):
    """Hourly candles; hours with no data (127 in 2018-2019) keep the price, get volume 0 and buy share 0.5."""
    h = pd.read_csv(f"data/binance_{symbol}_1h.csv.gz", index_col=0, parse_dates=True)
    h = h.reindex(pd.date_range(h.index[0], h.index[-1], freq="1h"))
    h["close"] = h["close"].ffill()
    for c in ["volume", "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        h[c] = h[c].fillna(0)
    h["ret"] = h["close"].pct_change() * 100                          # hourly change in %
    h["buy_share"] = (h["taker_buy_base_volume"] / h["volume"]).where(h["volume"] > 0, 0.5)
    return h


def roll(s, w, how="mean"):
    return getattr(s.rolling(w, min_periods=w), how)()


def change(close, w):
    return (close / close.shift(w) - 1) * 100


def build_attributes(btc, eth):
    """Dict group name -> DataFrame of attributes, indexed by hour (the last input hour)."""
    groups = {}

    qv = btc["quote_volume"]
    flow = pd.DataFrame(index=btc.index)
    for k in range(1, 7):
        flow[f"buy_share_lag{k}"] = btc["buy_share"].shift(k - 1)
    for w in (6, 24, 168):
        flow[f"buy_share_{w}h"] = (roll(btc["taker_buy_quote_volume"], w, "sum") / roll(qv, w, "sum")).fillna(0.5)
    flow["buy_share_24h_vs_week"] = flow["buy_share_24h"] - flow["buy_share_168h"]
    net = (2 * btc["taker_buy_quote_volume"] - qv)                      # bought minus sold, in USDT
    week_avg_qv = roll(qv, 168).replace(0, np.nan)
    flow["net_flow_6h"] = roll(net, 6, "sum") / (6 * week_avg_qv)
    flow["net_flow_24h"] = roll(net, 24, "sum") / (24 * week_avg_qv)
    trades = np.log1p(btc["trades"])
    flow["trades_24h_vs_week"] = roll(trades, 24) - roll(trades, 168)
    size = np.log1p(qv / btc["trades"].replace(0, np.nan)).fillna(0)
    flow["trade_size_24h_vs_week"] = roll(size, 24) - roll(size, 168)
    groups["flow"] = flow

    e = pd.DataFrame(index=eth.index)
    for k in range(1, 7):
        e[f"eth_ret_lag{k}"] = eth["ret"].shift(k - 1)
    e["eth_ret_24h"] = change(eth["close"], 24)
    e["eth_ret_7d"] = change(eth["close"], 168)
    e["eth_std_24h"] = roll(eth["ret"], 24, "std")
    e["eth_std_168h"] = roll(eth["ret"], 168, "std")
    e["eth_buy_share_24h"] = (roll(eth["taker_buy_quote_volume"], 24, "sum")
                              / roll(eth["quote_volume"], 24, "sum")).fillna(0.5)
    lq = np.log1p(eth["quote_volume"])
    e["eth_volume_24h_vs_week"] = roll(lq, 24) - roll(lq, 168)
    groups["eth"] = e

    cross = pd.DataFrame(index=btc.index)
    for w in (1, 6, 24, 168):
        cross[f"eth_minus_btc_{w}h"] = change(eth["close"], w) - change(btc["close"], w)
    cross["eth_btc_corr_168h"] = btc["ret"].rolling(168, min_periods=168).corr(eth["ret"])
    cross["eth_beta_168h"] = (eth["ret"].rolling(168, min_periods=168).cov(btc["ret"])
                              / roll(btc["ret"], 168, "var"))
    ratio = eth["close"] / btc["close"]
    lo, hi = roll(ratio, 168, "min"), roll(ratio, 168, "max")
    cross["ethbtc_pos_in_week_range"] = (ratio - lo) / (hi - lo + 1e-12)
    cross["ethbtc_dist_to_week_mean"] = (ratio / roll(ratio, 168) - 1) * 100
    groups["cross"] = cross
    return groups


if __name__ == "__main__":
    main(sys.argv[1:] or SYMBOLS)
