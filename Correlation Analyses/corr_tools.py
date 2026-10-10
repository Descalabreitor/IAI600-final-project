"""
corr_tools.py - correlation analysis utilities for the panels built by macro_panel.py

Pipeline: prune_panels (drop redundant macro variables, independent of BTC)
          -> corr_summary (correlation with BTC) -> lead_lag / corr_by_year (focus list).

Dependencies: pandas, numpy, scipy, matplotlib
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
                      target: str = "BTC_ret", method: str = "pearson",
                      absolute: bool = False):
    """Plot rolling correlation, optionally as absolute magnitudes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for v in variables:
        corr = panel[target].rolling(window, min_periods=int(window * 0.8)).corr(panel[v])
        if absolute:
            corr = corr.abs()
        corr.plot(ax=ax, label=v, linewidth=1.2)
    if not absolute:
        ax.axhline(0, color="black", linewidth=0.8)
    title = "Absolute rolling correlation" if absolute else "Rolling correlation"
    ax.set_title(f"{title} with {target} (window = {window} periods)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig
