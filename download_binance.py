"""Download Binance spot 1-minute klines (BTCUSDT, ETHUSDT) and aggregate them to hourly candles.

Source: https://data.binance.vision (public, no API key). Monthly zip files for finished months,
daily zip files for the current month. Every file is checked against its published SHA-256.

Output:
    data/raw/binance/<SYMBOL>/*.zip        raw 1-minute files (not in git, re-download with this script)
    data/binance_<SYMBOL>_1h.csv.gz        hourly candles, timestamps in UTC (start of the hour)

Usage:
    python download_binance.py                 # both symbols
    python download_binance.py BTCUSDT         # one symbol
"""
import hashlib
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

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


if __name__ == "__main__":
    main(sys.argv[1:] or SYMBOLS)
