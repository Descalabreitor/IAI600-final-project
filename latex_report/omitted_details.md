# Details left out of the compact report

The compact version of the report (30 Oct 2026) is exactly 4 pages, so about one page (about 50–55 lines over both columns) is free under the 5-page limit. These are the details removed in that pass, with an estimated cost in lines and their value. Sources: `data-analysis.ipynb`, `classical-models.ipynb`, `neural-models.ipynb`, `timestral-zero-shot.ipynb`, `trading-simulation.ipynb`.

## Data and analysis (§II)

| # | Detail | ~Lines | Value |
|---|---|---|---|
| 1 | Bitstamp and Binance hourly returns correlate at 0.993 (evidence the data is clean) | 1 | Low |
| 2 | Scaling fitted on training data only; the LSTM and Timestral normalise their own windows | 2 | Medium (leakage hygiene) |
| 3 | AR(24) on \|r\| explains 10% of validation variance (a model-free sign that size is predictable) | 1 | Medium |
| 4 | Robust autocorrelation band: only the lag-24 reversal stays significant (plus the Fig. 1 caption note "too narrow under volatility clustering") | 2 | Medium (honesty about Fig. 1) |
| 5 | Exact hour-profile numbers: 14 UTC peak 1.28× → 1.67× (before/after ETFs), weekends 0.8× → 0.6× | 1 | Low (the figure shows it) |
| 6 | Volatility regimes persist: next day 0.32%/h after calm weeks vs 0.71% after wild ones | 2 | Medium (supports refits and the persistence of size) |
| 7 | Macro details: Spearman with FDR correction (Benjamini–Hochberg reference), S&P 500 0.33 daily / 0.53 monthly, links unstable over time | 2 | Medium (the macro analysis is one person's whole contribution) |
| 8 | Release effect details: release hour 2.2–2.5× nearby days, FOMC up to 3.6×, Wilcoxon p < 0.001 on train and validation | 1 | Low |
| 9 | Candidate features built from the analysis (volatility ratios, minutes without trades, release flags, 90-day hour profile), checked for look-ahead and screened on train and validation | 2 | Medium (closes the analysis → features loop) |

## Models and setup (§II–III)

| # | Detail | ~Lines | Value |
|---|---|---|---|
| 10 | Timestral's expected \|r\| is the mean of its quantiles, because R² rewards the mean, not the median | 1 | Low |
| 11 | Settings table: MLP row (hidden (256, 128), weight decay 0.01); LSTM Adam 10⁻³ and weight decay 10⁻⁴; GB next-day 100 rows per leaf; GB depth 3–6 within 0.005 on validation | 2 | Low (documentation) |
| 12 | Block sizes in hours (26,304 / 13,128 / 19,899); every 6th origin for train/validation, every origin for test; purge = 168-h LSTM window + 24 target hours | 2 | Low |
| 13 | Walk-forward folds show ±0.08 R² between periods; grids extended when the best value is on an edge | 2 | Medium (tuning method, Alberto's part) |
| 14 | Early stopping on a random hold-out used up to 2.4× too many boosting trees, so the number of trees is tuned directly | 1 | Medium (tuning insight) |
| 15 | Networks averaged over 5 seeds | 0.5 | Low |
| 16 | R²_oos metric definition (R² against the zero forecast); ranges also scored by CRPS (Gneiting–Raftery reference) | 1 | Low |
| 17 | Trading setup details: decision at 00:00; loss limit L = c·σ̂ with c recalibrated on the last 90 known days; stop 1.5L below the day's open, checked hourly; 10-point rebalancing band; daily GB built for the simulation (fixed settings, 12-month half-life); Timestral without ETH/VIX; validation slightly favourable to the model flows | 4 | Low–medium |
| 18 | Trading measurement methods: Kupiec/Christoffersen VaR backtests, block Monte Carlo of drawdowns, 5,000 resampled years (+2 references) | 2 | Medium if #28 comes back |

## Results (§IV)

| # | Detail | ~Lines | Value |
|---|---|---|---|
| 19 | Returns: every model within 0.0001 pp of the zero forecast's MAE (0.3193%); Diebold–Mariano p ≥ 0.15; networks' MAE skill down to −0.0020; IC 0.003 → 0.021 when chosen by squared error, interval [+0.006, +0.035] | 2 | Low |
| 20 | Next hour: the reversal rule is right 54.6% of the time on its confident calls (p = 2·10⁻⁵); an LSTM ties GB only when trained on every origin, which helps GB equally (0.541), so "the data, not the network" makes the difference | 3 | Medium (a clear insight) |
| 21 | Hourly size: GB vs rule interval [+0.057, +0.084]; refit better in 20 of 29 months; release hours forecast at 0.65× their size (544 training rows < one 1,000-row leaf); a correction factor learnt from past errors over-shot on test | 3 | Medium |
| 22 | LSTM step-by-step gains: raw 0.100; +0.031 hour features; +0.014 volume attention; +0.011 monthly fine-tuning, all reliable; final 0.156, +0.008 [+0.001, +0.015] over GB; the MLP only ties | 2 | Medium (shows what made the network work) |
| 23 | Timestral details:<br>• reading \|r\| beats r (validation 0.108 vs 0.067)<br>• context 30/90/180 days gives 0.088/0.105/0.108<br>• release calendar +0.008; the clock adds nothing (found in the context)<br>• 209M +0.003 for 7× the compute<br>• ETH/VIX pass validation by < 0.001 but add nothing on test<br>• stacking into GB worse (0.155–0.157) because the trees learned on wilder 2020–22 hours<br>• Timestral alone +0.009 [+0.002, +0.016] over GB | 4 | Medium–high (understanding) |
| 24 | Ranges:<br>• raw GB 0.92% wide vs fixed 1.28% (28% narrower)<br>• split conformal 77.3%<br>• rolling 80.0% at 0.98% width<br>• fixed range covers 95% calm / 80% wild hours<br>• Mondrian ranges 2.08% wide in release hours<br>• Timestral's 21 quantiles within 1.4 points; lowest CRPS (0.236 vs 0.240)<br>• Timestral +13.8 points [+7.9, +18.9] over a single rolling factor in release hours | 3 | Low–medium |
| 25 | Next-day: GB + MLP average +0.011 [+0.001, +0.022] over GB; refit from 0.503 to 0.517 (+0.015 [+0.004, +0.025]); Linear Regression (0.514) and SVR (0.519) beat Random Forest (0.464) because tomorrow's volatility is almost linear in today's | 3 | Medium (the "linear wins" insight) |
| 26 | First economic test: volatility targeting lost to buy-and-hold (Sharpe 0.27 vs 0.50, not significant) because of its design (target level, leverage, turnover) | 2 | Low (superseded by the trading simulation) |
| 27 | Trading numbers:<br>• bias-removed R² of next-day log volatility 0.54 (GB) / 0.55 (Timestral) / 0.57 (average) vs 0.17 (naive)<br>• end capital $11,813–12,388 on test, naive highest<br>• model flows ahead in ≤ 3 of 12 other budget/stop settings<br>• liquidation gains vanish at 21–23× (50–71 liquidations for every flow) | 3 | Medium |
| 28 | VaR backtests: the average and naive flows pass both tests in both periods; GB borderline optimistic on validation (12.4% breaches, Kupiec p = 0.07); Timestral's breaches cluster on test (Christoffersen p = 0.01) | 2 | Medium |
| 29 | Timestral daily direction AUC 0.48–0.51; most of the drawdown gain comes from the rules themselves (11.9% with naive estimates); buy-and-hold maximum drawdown 53% | 1.5 | Medium |
| 30 | Separate Discussion subsection: model families converge within 0.01 R², plus the two caveats (single test period, possible Timestral training overlap) as their own paragraph | 3 | Low (now partly in §IV-B and the conclusion) |

## Introduction and conclusion

| # | Detail | ~Lines | Value |
|---|---|---|---|
| 31 | Why forecast changes, not prices: $4,000 (March 2020) → $126,000, 72% of test hours above the training range (now only in Table I) | 2 | Low |
| 32 | Explicit contributions list (i)–(v) | 4 | Low |
| 33 | Conclusion: Binance helped the next hour a little; features, ETH and macro added nothing; Holm/BH corrections as future work | 2 | Low |

## Suggested if nothing else is decided

Put back the items that do most for the teacher's criteria (understanding, and methodology from the analysis to the features), about 30 lines:

- **Understanding:** #20, #22, #23, #25.
- **Methodology and the analysis-to-features loop:** #6, #9, #13, #14.
- **Trading (optional):** #28, which also needs #18.
