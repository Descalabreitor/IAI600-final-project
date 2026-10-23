"""Shared helpers of the project. Import the submodule you need, e.g. `from utils import hourly as U`.

    hourly       continuous hourly BTC series, features, targets, metrics and statistical tests
    binance      Binance BTCUSDT / ETHUSDT download (`python -m utils.binance`) and order-flow / ETH attributes
    macro        FRED / Yahoo macro variables: D / W / ME correlation panels and hourly modelling features
    btc_data     BTC daily / weekly / monthly OHLCV frames from the Kaggle 1-minute file, with disk cache
    correlation  correlation tables, pruning, lead-lag, stability and plots for the macro panels
    notebooks    runs the setup cells of an earlier notebook inside another one
    release_dates  US CPI, payrolls and FOMC release times (`python -m utils.release_dates`)
"""
