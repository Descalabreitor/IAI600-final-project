# BTC vs macro variables - correlation analysis

Files (keep them in the same folder as the notebook):
- `correlation_analyses.ipynb` - the report (all settings are in the first code cell)
- `btc_data.py`    - BTC loading and resampling to D / W / ME, with disk cache (`.btc_cache/`)
- `macro_panel.py` - FRED / Yahoo downloads (cache in `.macro_cache/`) and panel construction
- `corr_tools.py`  - pruning, correlation tables, lead-lag, stability, plots

If you replace older copies of these files, restart the notebook kernel before running
(Python keeps imported modules in memory).

To force a fresh download: delete `.macro_cache/` and `.btc_cache/`.
