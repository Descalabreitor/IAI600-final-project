"""
macro_features.py - macro variables for MODELLING on BTC's hourly index.

For CORRELATION ANALYSIS use macro_panel.build_panel (D / W / ME panels); do not
correlate macro series repeated every hour with hourly returns.

This version fixes the previous one:
  * Reuses macro_panel (more variables, cache, retries, failed series skipped).
  * Alignment is always by availability date (observation + publication lag).
    Daily FRED series use lag 2 (before: 1, ~20 h of lookahead).
  * The transformation (difference / log return) is computed on the native
    observations, already indexed by availability, and then forward-filled.
  * The ffill has a maximum tolerance per frequency: a discontinued series stops
    being propagated (NaN) instead of staying frozen.
  * Levels and year-over-year changes (which provided regime context in the
    previous version) are optional and named *_lvl and *_yoy.

Dependencies: pandas, numpy, yfinance (via macro_panel)
"""
import pandas as pd

from macro_panel import STALE_TOL, _transform, build_levels

DEFAULT_LEVELS = ("US10Y", "FedFunds", "Curve_10Y2Y", "VIX", "HY_OAS")
DEFAULT_YOY = ("M2", "CPI", "CorePCE")


def build_macro_hourly(hourly_index: pd.DatetimeIndex,
                       add_levels=DEFAULT_LEVELS, add_yoy=DEFAULT_YOY, **kw) -> pd.DataFrame:
    """Macro features aligned to a naive UTC hourly index, without lookahead.

    add_levels: variables whose LEVEL is added as <name>_lvl (regime).
    add_yoy:    monthly variables whose year-over-year change is added as <name>_yoy.
    **kw is passed to macro_panel.build_levels (data_start, refresh_hours, ...).
    """
    levels, meta, _ = build_levels(pit=True, **kw)
    idx = pd.DatetimeIndex(hourly_index)
    cols = {}
    for m in meta.to_dict("records"):
        name, lv, tol = m["name"], levels[m["name"]], STALE_TOL[m["native"]]
        cols[name] = _transform(lv, m["transform"]).dropna().reindex(
            idx, method="ffill", tolerance=tol)
        if name in add_levels:
            cols[f"{name}_lvl"] = lv.reindex(idx, method="ffill", tolerance=tol)
        if name in add_yoy and m["native"] == "M":
            yoy = (lv / lv.shift(12) - 1) * 100       # 12 monthly observations
            cols[f"{name}_yoy"] = yoy.dropna().reindex(idx, method="ffill", tolerance=tol)
    out = pd.DataFrame(cols, index=hourly_index)
    return out

# Usage in the notebook:
#   from macro_features import build_macro_hourly
#   macro = build_macro_hourly(df_BTC_hourly.index)
#   data = df_BTC_hourly.join(macro)
