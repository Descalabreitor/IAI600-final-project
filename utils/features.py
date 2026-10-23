"""Hourly features suggested by data-analysis.ipynb (section 9), shared by every notebook.

`hourly` is the frame of data/btc_hourly.csv.gz with Return already in % (Return *= hourly.PCT)
and the filled_minutes column. Two kinds of rows:

    origin rows  (origin_features, recent_hour_profile, next_day_release_features, build_features)
                 row t uses only the hours <= t; it is the information at the end of candle t
    target rows  (target_hour_features)
                 row h describes hour h itself (the predicted hour); only facts known in advance

Self-test (no lookahead, invariants, shapes, equality with data/features_hourly.csv.gz):

    python -m utils.features
"""
import os

import numpy as np
import pandas as pd

from utils.config import DATA_START, RELEASES_FILE, TRAIN_END
from utils.hourly import MONTH, PCT, WEEK

DAY = 24
ONE_HOUR = pd.Timedelta(hours=1)
EVENTS = ["CPI", "Payrolls", "FOMC"]

ORIGIN_FEATURES = ["rv_ratio_1d_30d", "jump_ratio_1d", "filled_minutes_1d"]
TARGET_HOUR_FEATURES = ["tgt_cpi", "tgt_payrolls", "tgt_fomc", "tgt_after_fomc", "tgt_us_session"]
NEXT_DAY_FEATURES = ["next_day_cpi", "next_day_payrolls", "next_day_fomc", "next_day_release"]


def load_releases(path=RELEASES_FILE):
    """The release calendar written by `python -m utils.release_dates`."""
    return pd.read_csv(path, parse_dates=["date", "timestamp_utc"])


def _release_hours(releases):
    """Candle (start hour) of every scheduled release, per event and all together."""
    sched = releases[releases["kind"] == "scheduled"]
    hours = {ev: pd.DatetimeIndex(g["timestamp_utc"].dt.floor("h")).sort_values() for ev, g in sched.groupby("event")}
    hours["all"] = pd.DatetimeIndex(sched["timestamp_utc"].dt.floor("h")).unique().sort_values()
    return hours


def origin_features(hourly):
    """Volatility shape of the last day, from the candles that end at t."""
    a = hourly["Return"].abs()
    rv_30d = a.rolling(MONTH).mean()
    return pd.DataFrame({
        "rv_ratio_1d_30d": a.rolling(DAY).mean() / rv_30d,
        "jump_ratio_1d": a.rolling(DAY).max() / rv_30d,
        "filled_minutes_1d": hourly["filled_minutes"].rolling(DAY).sum(),
    }, index=hourly.index)


def target_hour_features(index, releases):
    """Calendar facts of the predicted hour h (scheduled releases only: they are known in advance)."""
    index = pd.DatetimeIndex(index)
    rel = _release_hours(releases)
    f = pd.DataFrame(index=index)
    for ev in EVENTS:
        f[f"tgt_{ev.lower()}"] = index.isin(rel[ev]).astype(int)
    f["tgt_after_fomc"] = (index.isin(rel["FOMC"] + ONE_HOUR) | index.isin(rel["FOMC"] + 2 * ONE_HOUR)).astype(int)
    f["tgt_us_session"] = ((index.hour >= 13) & (index.hour <= 16)).astype(int)
    return f


def recent_hour_profile(hourly, days=90):
    """Mean |r| per hour of the day over the `days` days that end at t, divided by its mean.

    Returns an array (len(hourly), 24); column k is the hour k UTC. The window expands until it
    has `days` days; rows with less than 7 days of history are NaN.
    """
    a = hourly["Return"].abs()
    idx = hourly.index
    prof = np.full((len(idx), 24), np.nan)
    for k in range(24):
        ak = a[idx.hour == k]
        prof[:, k] = ak.rolling(f"{days}D").mean().reindex(idx, method="ffill").to_numpy()
    prof /= prof.mean(axis=1, keepdims=True)
    prof[(idx - idx[0]) < pd.Timedelta(days=7) - ONE_HOUR] = np.nan
    return prof


def next_day_release_features(hourly, releases):
    """1 if a scheduled release falls in one of the next 24 candles, t+1 .. t+24."""
    idx = hourly.index
    rel = _release_hours(releases)
    f = pd.DataFrame(index=idx)
    for ev in EVENTS:
        r = rel[ev].values
        pos = np.searchsorted(r, (idx + ONE_HOUR).values)
        f[f"next_day_{ev.lower()}"] = ((pos < len(r)) & (r[np.minimum(pos, len(r) - 1)] <= (idx + DAY * ONE_HOUR).values)).astype(int)
    f["next_day_release"] = f.max(axis=1)
    return f


def build_features(hourly, releases, macro=None, train_end=TRAIN_END):
    """Candidate features, one row per hourly candle t, using only what is known at the end of t.

    The target they are meant for is the size of candle t+1 (or of the next 24 candles). The VIX
    columns are added only when `macro` (utils.macro.load_macro) is given.
    """
    r = hourly["Return"]
    a = r.abs()
    idx = hourly.index
    nxt = idx + ONE_HOUR                                   # the candle being predicted
    orig = origin_features(hourly)
    tgt = target_hour_features(nxt, releases).set_axis(idx)
    f = pd.DataFrame(index=idx)

    # volatility: windows that end at candle t
    f["rv_1d"] = a.rolling(DAY).mean()
    f["rv_7d"] = a.rolling(WEEK).mean()
    f["rv_30d"] = a.rolling(MONTH).mean()
    f["rv_ratio_1d_30d"] = orig["rv_ratio_1d_30d"]
    hl2 = np.log(hourly["High"] / hourly["Low"]) ** 2
    f["parkinson_1d"] = np.sqrt(hl2.rolling(DAY).mean() / (4 * np.log(2))) * PCT
    f["jump_ratio_1d"] = orig["jump_ratio_1d"]
    f["filled_minutes_1d"] = orig["filled_minutes_1d"]

    # clock of the predicted candle (known in advance); fixed profile from the train block, recent one up to t
    f["next_weekend"] = (nxt.dayofweek >= 5).astype(int)
    f["next_us_session"] = tgt["tgt_us_session"]
    a_tr = a.loc[:train_end]
    profile = a_tr.groupby(a_tr.index.hour).mean()
    f["next_hour_profile"] = (profile / profile.mean()).reindex(nxt.hour).to_numpy()
    f["next_hour_profile_recent"] = recent_hour_profile(hourly)[np.arange(len(idx)), nxt.hour]

    # scheduled releases (known in advance)
    for ev in EVENTS:
        f[f"next_{ev.lower()}"] = tgt[f"tgt_{ev.lower()}"]
    f["after_fomc_1_2h"] = tgt["tgt_after_fomc"]
    all_rel = _release_hours(releases)["all"]
    pos = np.searchsorted(all_rel.values, nxt.values)        # first release candle at or after the predicted one
    to_next = np.where(pos < len(all_rel), (all_rel.values[np.minimum(pos, len(all_rel) - 1)] - nxt.values) / np.timedelta64(1, "h"), np.inf)
    since = np.where(pos > 0, (nxt.values - all_rel.values[np.maximum(pos - 1, 0)]) / np.timedelta64(1, "h"), np.inf)
    f["hours_to_release"] = np.minimum(to_next, WEEK)
    f["hours_since_release"] = np.minimum(since, WEEK)

    # market stress: VIX close that was public at the end of candle t (availability date)
    if macro is not None:
        from utils.macro import align_levels
        vix = align_levels(macro, pit=True)["VIX"]
        f["vix_level"] = vix.reindex(idx, method="ffill", tolerance=pd.Timedelta(days=10)).to_numpy()
        f["vix_above_20"] = (f["vix_level"] > 20).astype(float).where(f["vix_level"].notna())
    return f


# =====================================================================================
# Self-test
# =====================================================================================
def _load_hourly(path=os.path.join("data", "btc_hourly.csv.gz")):
    hourly = pd.read_csv(path, index_col="Timestamp", parse_dates=True).loc[DATA_START:]
    hourly["Return"] *= PCT
    return hourly


def _fake_future(hourly, cut, seed=42):
    """Copy of hourly with random prices, returns, volumes and filled minutes from `cut` onwards."""
    fake = hourly.copy()
    after = fake.index >= cut
    ohlc = ["Open", "High", "Low", "Close"]
    fake.loc[after, ohlc] *= np.random.default_rng(seed).uniform(0.5, 2.0, (after.sum(), 4))
    fake.loc[after, "High"] = fake.loc[after, ohlc].max(axis=1)
    fake.loc[after, "Low"] = fake.loc[after, ohlc].min(axis=1)
    fake.loc[after, "Return"] = np.random.default_rng(seed + 1).normal(0, 5, after.sum())
    fake.loc[after, "Volume"] *= 3
    fake.loc[after, "filled_minutes"] = np.random.default_rng(seed + 2).integers(0, 60, after.sum())
    return fake


def _selftest():
    from utils.macro import load_macro

    hourly = _load_hourly()
    rel = load_releases()
    macro = load_macro(os.path.join("data", "macro_raw.csv.gz"))
    idx, n = hourly.index, len(hourly)
    print(f"hourly: {n:,} hours {idx[0]} -> {idx[-1]}")

    funcs = {
        "origin_features": lambda h: origin_features(h).to_numpy(float),
        "target_hour_features": lambda h: target_hour_features(h.index, rel).to_numpy(float),
        "recent_hour_profile": lambda h: recent_hour_profile(h),
        "next_day_release_features": lambda h: next_day_release_features(h, rel).to_numpy(float),
        "build_features": lambda h: build_features(h, rel, macro).to_numpy(float),
    }
    out = {name: fn(hourly) for name, fn in funcs.items()}

    # shapes and NaN
    feats = build_features(hourly, rel, macro)
    expected = {"origin_features": 3, "target_hour_features": 5, "recent_hour_profile": 24,
                "next_day_release_features": 4, "build_features": 19}
    for name, k in expected.items():
        assert out[name].shape == (n, k), (name, out[name].shape)
    assert list(feats.columns[:10]) == ["rv_1d", "rv_7d", "rv_30d", "rv_ratio_1d_30d", "parkinson_1d", "jump_ratio_1d",
                                        "filled_minutes_1d", "next_weekend", "next_us_session", "next_hour_profile"]
    assert "vix_level" not in build_features(hourly.iloc[:2000], rel).columns
    nan_rows = {c: int(feats[c].isna().sum()) for c in feats.columns}
    assert nan_rows["rv_30d"] == nan_rows["rv_ratio_1d_30d"] == nan_rows["jump_ratio_1d"] == MONTH - 1
    assert nan_rows["rv_1d"] == nan_rows["parkinson_1d"] == nan_rows["filled_minutes_1d"] == DAY - 1
    assert nan_rows["next_hour_profile_recent"] == WEEK - 1
    assert np.isnan(out["recent_hour_profile"][:WEEK - 1]).all() and not np.isnan(out["recent_hour_profile"][WEEK - 1:]).any()
    assert np.allclose(out["recent_hour_profile"][WEEK - 1:].mean(axis=1), 1)
    for name in ("target_hour_features", "next_day_release_features"):
        assert not np.isnan(out[name]).any(), name
    assert feats.dropna().index[0] == idx[MONTH - 1]
    print(f"shapes and NaN: ok; build_features {feats.shape}, complete rows from {feats.dropna().index[0]}")
    print("  NaN rows per column:", {c: v for c, v in nan_rows.items() if v})

    # invariants next_* (row t) <-> tgt_* (row t+1)
    tgt = target_hour_features(idx, rel)
    pairs = {"next_cpi": "tgt_cpi", "next_payrolls": "tgt_payrolls", "next_fomc": "tgt_fomc",
             "after_fomc_1_2h": "tgt_after_fomc", "next_us_session": "tgt_us_session"}
    for nc, tc in pairs.items():
        assert (feats[nc].to_numpy()[:-1] == tgt[tc].to_numpy()[1:]).all(), nc
    prof = out["recent_hour_profile"]
    assert np.array_equal(feats["next_hour_profile_recent"].to_numpy(), prof[np.arange(n), (idx + ONE_HOUR).hour], equal_nan=True)
    nd = next_day_release_features(hourly, rel)
    full = slice(0, n - DAY)                                   # rows whose next 24 hours are in the index
    for ev in ("cpi", "payrolls", "fomc"):
        ref = tgt[f"tgt_{ev}"].rolling(DAY).max().shift(-DAY).to_numpy()[full]
        assert (nd[f"next_day_{ev}"].to_numpy()[full] == ref).all(), ev
    sched = _release_hours(rel)
    in_range = lambda h: ((h >= idx[0]) & (h <= idx[-1])).sum()
    for ev in EVENTS:
        assert tgt[f"tgt_{ev.lower()}"].sum() == in_range(sched[ev]), ev
    unscheduled = rel.loc[rel["kind"] == "unscheduled", "timestamp_utc"].dt.floor("h")
    assert tgt.loc[tgt.index.isin(unscheduled), ["tgt_cpi", "tgt_payrolls", "tgt_fomc"]].sum().sum() == 0
    counts = {c: int(tgt[c].sum()) for c in TARGET_HOUR_FEATURES[:4]}
    print(f"invariants next_* <-> tgt_*: ok; release hours {counts}, unscheduled excluded: {len(unscheduled)}")

    # no lookahead: random data from CUT onwards must not change any earlier row
    cut = pd.Timestamp("2025-01-01")
    fake = _fake_future(hourly, cut)
    before = idx < cut
    for name, fn in funcs.items():
        x, y = out[name], fn(fake)
        same = np.isclose(x[before], y[before], equal_nan=True).all()
        changed = (~np.isclose(x[~before], y[~before], equal_nan=True)).any(axis=0).sum()
        print(f"  {name:<26} rows before {cut.date()} unchanged: {same}; columns changed after: {changed}/{x.shape[1]}")
        assert same, f"{name} uses future data"
    print("no lookahead: ok")

    # equality with the file written by data-analysis.ipynb section 9
    path = os.path.join("data", "features_hourly.csv.gz")
    saved = pd.read_csv(path, index_col=0, parse_dates=True)
    assert saved.index.equals(idx), "index differs from features_hourly.csv.gz"
    ours = feats[saved.columns].to_numpy(float)
    ok = np.allclose(ours, saved.to_numpy(float), equal_nan=True)
    rel_err = np.nanmax(np.abs(ours - saved.to_numpy(float)) / np.maximum(np.abs(saved.to_numpy(float)), 1e-12))
    print(f"equal to {path} ({saved.shape[1]} columns, {len(saved):,} rows): {ok}; max relative difference {rel_err:.1e}")
    assert ok
    print("all checks passed")


if __name__ == "__main__":
    _selftest()
