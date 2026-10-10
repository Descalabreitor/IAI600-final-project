"""
corr_tools.py - correlation analysis utilities for the panels built by macro_panel.py

Dependencies: pandas, numpy, scipy, matplotlib
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

TARGETS = ("BTC_ret", "BTC_absret")


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment (FDR control) for multiple comparisons."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def corr_table(panel: pd.DataFrame, target: str = "BTC_ret", method: str = "spearman",
               min_obs: int = 40) -> pd.DataFrame:
    """Correlation of each variable with `target`: r, p, n and q (FDR-adjusted p).

    With ~40 variables, some will look "significant" by chance: look at q, not p.
    """
    rows = []
    for c in panel.columns:
        if c in TARGETS:
            continue
        d = panel[[target, c]].dropna()
        if len(d) < min_obs or d[c].nunique() < 3:
            continue
        r, p = (stats.spearmanr if method == "spearman" else stats.pearsonr)(d[target], d[c])
        rows.append((c, r, p, len(d)))
    out = pd.DataFrame(rows, columns=["var", "r", "p", "n"]).set_index("var")
    out["q_FDR"] = _bh(out["p"].values)
    return out.reindex(out["r"].abs().sort_values(ascending=False).index)


def lead_lag(panel: pd.DataFrame, target: str = "BTC_ret", max_lag: int = 5,
             method: str = "spearman", min_obs: int = 40) -> pd.DataFrame:
    """corr(target_t, X_{t-k}) for k in [-max_lag, max_lag].

    k > 0: X leads BTC by k periods (X "anticipates").
    k < 0: BTC leads X.
    For lead claims, use a panel built with pit=True.
    """
    feats = [c for c in panel.columns if c not in TARGETS]
    y = panel[target]
    res = {c: {k: y.corr(panel[c].shift(k), method=method, min_periods=min_obs)
               for k in range(-max_lag, max_lag + 1)} for c in feats}
    return pd.DataFrame(res).T


def corr_by_year(panel: pd.DataFrame, variables, target: str = "BTC_ret",
                 method: str = "spearman", min_obs: int = 20) -> pd.DataFrame:
    """Correlation with `target` per calendar year (rows = year). Shows (in)stability."""
    out = {}
    for y, g in panel.groupby(panel.index.year):
        out[y] = {v: g[target].corr(g[v], method=method, min_periods=min_obs) for v in variables}
    return pd.DataFrame(out).T


def cluster_order(corr: pd.DataFrame) -> list:
    """Hierarchical order (distance 1-|r|) so that redundant variables sit together."""
    d = (1 - corr.abs().fillna(0)).values
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    leaves = leaves_list(linkage(squareform(d, checks=False), "average"))
    return list(corr.index[leaves])


def heatmap(df: pd.DataFrame, title: str = "", vmax: float | None = None,
            annot: bool = True, figsize=None, xlabel: str = ""):
    vmax = vmax or float(np.nanmax(np.abs(df.values)))
    figsize = figsize or (max(6, 0.5 * df.shape[1] + 3), max(4, 0.32 * df.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if annot and df.size <= 700:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def rolling_corr_plot(panel: pd.DataFrame, variables, window: int = 180,
                      target: str = "BTC_ret", method: str = "pearson"):
    """Rolling correlation. (Rolling Spearman is not available in pandas: uses Pearson.)"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for v in variables:
        panel[target].rolling(window, min_periods=int(window * 0.8)).corr(panel[v]).plot(
            ax=ax, label=v, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling correlation with {target} (window = {window} periods)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig
