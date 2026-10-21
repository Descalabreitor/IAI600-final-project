"""
correlation.py - correlation analysis utilities for the panels built by utils.macro

Pipeline: prune_panels (drop redundant macro variables, independent of BTC)
          -> corr_summary (correlation with BTC) -> lead_lag / corr_by_year (focus list).

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


def corr_summary(panel: pd.DataFrame, method: str = "spearman", min_obs: int = 40) -> pd.DataFrame:
    """One table: correlation with BTC return (r_ret) and with |return| (r_abs).

    n     : observations used
    q_*   : p-value adjusted for multiple comparisons (Benjamini-Hochberg FDR), computed
            separately within each target. Read q, not p.
    Sorted by |r_ret|. |return| measures the size of the move, not its direction.
    """
    a = corr_table(panel, "BTC_ret", method, min_obs)
    b = corr_table(panel, "BTC_absret", method, min_obs)
    out = pd.DataFrame({"n": a["n"], "r_ret": a["r"], "q_ret": a["q_FDR"],
                        "r_abs": b["r"], "q_abs": b["q_FDR"]})
    return out.reindex(out["r_ret"].abs().sort_values(ascending=False).index)


def weekday_rank_corr(a: pd.Series, b: pd.Series):
    """Spearman correlation within weekdays: ranks minus their weekday mean, then Pearson.

    In the weekday-only daily panel BTC's Monday return covers Friday -> Monday (3 days), so
    |BTC| is larger on Mondays. Any variable whose size has a weekday pattern (e.g. a series
    published with a 7-day lag, which repeats a value on one weekday and doubles up on another)
    would correlate with |BTC| for that reason alone. Comparing within weekdays removes it.
    Returns (r, p, n).
    """
    d = pd.concat([a, b], axis=1).dropna()
    rk = d.rank()
    rk = rk - rk.groupby(d.index.dayofweek).transform("mean")
    r, p = stats.pearsonr(rk.iloc[:, 0], rk.iloc[:, 1])
    return r, p, len(d)


def size_summary(panel: pd.DataFrame, lag: int = 0, method: str = "spearman",
                 min_obs: int = 40, within_weekday: bool = False) -> pd.DataFrame:
    """Size of BTC's moves vs size of each variable's moves: corr(|BTC_t|, |X_{t-lag}|).

    corr_summary's r_abs correlates |BTC| with the SIGNED change of X, which only asks whether
    big BTC moves come with rises or with falls of X. This asks whether they come with big moves
    of X in either direction (co-volatility). lag=0: same period; lag=1: X's previous period
    (use a pit=True panel for that). within_weekday=True (daily panel): Spearman within weekdays,
    see weekday_rank_corr. q is Benjamini-Hochberg adjusted. Sorted by |r|.
    """
    y = panel["BTC_absret"]
    rows = []
    for c in panel.columns:
        if c in TARGETS:
            continue
        d = pd.concat([y, panel[c].abs().shift(lag)], axis=1).dropna()
        if len(d) < min_obs or d.iloc[:, 1].nunique() < 3:
            continue
        if within_weekday:
            r, p, _ = weekday_rank_corr(d.iloc[:, 0], d.iloc[:, 1])
        else:
            r, p = (stats.spearmanr if method == "spearman" else stats.pearsonr)(d.iloc[:, 0], d.iloc[:, 1])
        rows.append((c, len(d), r, p))
    out = pd.DataFrame(rows, columns=["var", "n", "r", "p"]).set_index("var")
    out["q"] = _bh(out["p"].values)
    return out.drop(columns="p").reindex(out["r"].abs().sort_values(ascending=False).index)


def prune_panels(panels: dict, threshold: float = 0.85, method: str = "spearman",
                 min_obs: int = 40, prefer=()):
    """Drop macro variables that are redundant with another macro variable.

    Uses ONLY the correlation between macro variables (never with BTC), so the pruning
    does not leak the result being interpreted later. Greedy: while two variables have
    |r| > threshold, drop the one with the higher mean |r| against all the others.
    Panels are processed from the finest to the coarsest frequency, and at each step only
    variables that first appear in that panel can be dropped (a daily variable is judged
    on the daily panel, a monthly-only one on the monthly panel).

    Returns (pruned panels, report). The report is indexed by dropped variable and gives
    the variable kept instead of it, their |r| and the panel where it was detected.

    prefer: variables to keep when they are redundant with another one (e.g. your FOCUS
    list). It only decides which of two redundant variables survives, so it is still
    independent of BTC.
    """
    dropped, seen = {}, set()
    for freq in ("D", "W", "ME"):
        if freq not in panels:
            continue
        cols = [c for c in panels[freq].columns if c not in TARGETS and c not in dropped]
        if len(cols) < 2:
            seen |= set(cols)
            continue
        new = np.array([c not in seen for c in cols])
        corr = panels[freq][cols].corr(method=method, min_periods=min_obs).abs().fillna(0).to_numpy(copy=True)
        np.fill_diagonal(corr, 0.0)
        alive = np.ones(len(cols), dtype=bool)
        while True:
            eligible = (new[:, None] | new[None, :]) & alive[:, None] & alive[None, :]
            masked = np.where(eligible, corr, 0.0)
            i, j = np.unravel_index(masked.argmax(), masked.shape)
            if masked[i, j] <= threshold:
                break
            # only variables new to this panel can be dropped
            if new[i] and new[j]:
                pi, pj = cols[i] in prefer, cols[j] in prefer
                mean = corr[:, alive].mean(axis=1)
                if pi != pj:                       # keep the preferred one
                    drop, keep = (j, i) if pi else (i, j)
                else:
                    drop, keep = (i, j) if mean[i] >= mean[j] else (j, i)
            else:
                drop, keep = (i, j) if new[i] else (j, i)
            dropped[cols[drop]] = dict(kept_instead=cols[keep], abs_r=corr[i, j], panel=freq)
            alive[drop] = False
        seen |= set(cols)

    pruned = {f: p.drop(columns=[c for c in dropped if c in p.columns]) for f, p in panels.items()}
    report = pd.DataFrame.from_dict(dropped, orient="index",
                                    columns=["kept_instead", "abs_r", "panel"])
    return pruned, report


def resolve_focus(focus, panels: dict, report: pd.DataFrame) -> list:
    """Map an a-priori shortlist onto the pruned data.

    A shortlisted variable that was pruned is replaced by the variable kept instead of it;
    variables that are not available at all (failed download) are reported and skipped.
    """
    available = {c for p in panels.values() for c in p.columns} - set(TARGETS)
    out = []
    for v in focus:
        while v in report.index:
            v = report.at[v, "kept_instead"]
        if v not in available:
            print(f"focus variable not available: {v}")
        elif v not in out:
            out.append(v)
    return out


def lead_lag(panel: pd.DataFrame, target: str = "BTC_ret", max_lag: int = 5,
             method: str = "spearman", min_obs: int = 40, variables=None) -> pd.DataFrame:
    """corr(target_t, X_{t-k}) for k in [-max_lag, max_lag].

    k > 0: X leads BTC by k periods (X "anticipates").
    k < 0: BTC leads X.
    For lead claims, use a panel built with pit=True.
    """
    feats = [c for c in panel.columns if c not in TARGETS] if variables is None else list(variables)
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


def rolling_corr(panel: pd.DataFrame, variable: str, window: int = 180, target: str = "BTC_ret",
                 method: str = "pearson", min_frac: float = 0.8) -> pd.Series:
    """Rolling correlation of `variable` with `target` over the last `window` rows.

    method="spearman" ranks the values inside each window (rank correlation), so a single
    extreme day cannot move the result much; "pearson" uses the raw values. A window needs
    at least min_frac * window complete pairs, otherwise the value is NaN.
    """
    d = panel[[target, variable]]
    if method == "pearson":
        return d[target].rolling(window, min_periods=int(window * min_frac)).corr(d[variable])
    y, x = d[target].to_numpy(float), d[variable].to_numpy(float)
    out = np.full(len(d), np.nan)
    for i in range(window, len(d) + 1):
        a, b = y[i - window:i], x[i - window:i]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() >= window * min_frac:
            out[i - 1] = stats.spearmanr(a[ok], b[ok])[0]
    return pd.Series(out, index=d.index, name=variable)


def rolling_corr_plot(panel: pd.DataFrame, variables, window: int = 180,
                      target: str = "BTC_ret", method: str = "pearson",
                      absolute: bool = False, background: pd.Series | None = None,
                      background_label: str = ""):
    """Plot rolling correlations (signed by default, or absolute magnitudes).

    background: optional series drawn in grey on a second axis (e.g. the VIX level), to
    compare the correlation with market conditions. Returns (fig, {variable: series}).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    series = {}
    if background is not None:
        ax2 = ax.twinx()
        ax2.fill_between(background.index, background.to_numpy(float), color="#d9d8d3", alpha=0.6, lw=0)
        ax2.set_ylabel(background_label, color="#7a7974")
        ax2.tick_params(axis="y", colors="#7a7974")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    for v in variables:
        corr = rolling_corr(panel, v, window, target, method)
        series[v] = corr
        shown = corr.abs() if absolute else corr
        ax.plot(shown.index, shown.to_numpy(float), label=v, linewidth=1.2)
    if not absolute:
        ax.axhline(0, color="black", linewidth=0.8)
    kind = "Absolute rolling" if absolute else "Rolling"
    ax.set_title(f"{kind} {method.capitalize()} correlation with {target} (window = {window} periods)")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, series
