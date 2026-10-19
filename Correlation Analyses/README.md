# BTC vs macro variables - correlation analysis

Files:
- `correlation_analyses.ipynb` - the report (all settings are in the first code cell)
- `../utils/btc_data.py`    - BTC loading and resampling to D / W / ME, with disk cache (`.btc_cache/`)
- `../utils/macro.py`       - FRED / Yahoo downloads (cache in `.macro_cache/`) and panel construction
- `../utils/correlation.py` - pruning, correlation tables, lead-lag, stability, plots

The notebook adds the repo root to `sys.path` to import `utils`, so run it from this folder.
If you change these files, restart the notebook kernel before running
(Python keeps imported modules in memory).

To force a fresh download: delete `.macro_cache/` and `.btc_cache/`.
