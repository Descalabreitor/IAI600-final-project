"""Binance BTCUSDT / ETHUSDT hourly data (from download_binance.py) and the attributes built from it.

Every attribute at hour t uses only hours up to and including t, and all of them are scale-free.
Groups:
    flow   BTC order flow: share of taker buys, net buying, number and size of trades
    eth    ETH's own moves, volatility, buy share and volume
    cross  ETH relative to BTC: relative moves, correlation, beta, ETH/BTC ratio position
"""
import numpy as np
import pandas as pd


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
