# Repot Notes
Write here your notes and later we make the report using this file as the base.

## Adrian (no chacho)

- Analisys of BTC data
- Analisys of correlations between BTC data and other macroeconomic variables

## Rerun on the data from 2020 (Oct 2026)

Every notebook now uses the hours from 1 Jan 2020 (`utils/config.py`): train 2020 → 2022, validation Jan 2023 → Jun 2024, test Jul 2024 → Oct 2026. The numbers below are test results unless marked otherwise, from `data-analysis.ipynb`, `classical-models.ipynb`, `neural-models.ipynb` and `timestral-zero-shot.ipynb`.

### New features: what helped and what did not

- `data-analysis.ipynb` §9 proposed candidate features from the analysis and checked them one by one (Spearman correlation on train and validation, also relative to last week's volatility). Carrying information on both blocks: the volatility ratios `rv_ratio_1d_30d` and `jump_ratio_1d`, the minutes without trades, the release-hour flags (CPI, payrolls, FOMC: about 3× larger moves) and the recent hour profile. No signal: the VIX level and the VIX-above-20 flag (within the train block), the hours to / since a release. Redundant: Parkinson volatility, `rv_30d`.
- In the models none of them helps the trees. With the Gradient Boosting fixed, the hourly move size has validation R² 0.095 with base + per-hour columns (B3), 0.088 adding B4 (volatility ratios, minutes without trades), 0.094 adding B5 (release hours, US session), 0.089 with both and 0.092 with the recent profile too. The set chosen on validation is base + B3, the same as before the new features.
- Release hours stay under-predicted (forecast 0.39× the actual size on validation, 0.65× on test) and the 80% ranges cover only 56% of them. The train block has 544 rows whose predicted hour is a release hour, fewer than one leaf of the tuned Gradient Boosting (1,000 rows), so no tree can single them out.
- For next-day volatility, B4 and the releases of the next 24 hours change the walk-forward R² by ±0.004 (no gain).
- The networks are the exception: telling the LSTM the clock, the recent hour profile and the release flags of each predicted hour lifts its R² from 0.100 to 0.131 (+0.031, 95% CI +0.024 to +0.039). This is what the per-hour columns of the long format already do for the trees.
- Timestral (zero-shot) also uses the release calendar: as a known-future covariate it adds +0.008 validation R², the largest gain of any covariate there; the clock adds nothing, because the model finds the hour profile in its context.
- Lesson for the report: a feature that carries information on its own is not a feature that improves a model. Recent volatility and the clock already carry most of it, and a feature needs enough examples for the model to use it.

### The recent hour profile

- The daily rhythm of volatility drifts: 14 UTC moved 1.17× the average hour in 2020 and 1.70× in 2024–2025 (growing from 2022, jumping in 2024); 00 UTC faded; weekends went from about 0.8× to 0.6× of weekdays. Releases do not explain the 14 UTC peak: it is the US trading session.
- A profile of the last 90 days (`next_hour_profile_recent`) correlates 0.17 with the next hour's move on validation, against 0.12 for the profile learned on train.
- As a simple rule it helps: last week's |r| × the recent profile has test R² 0.072, against 0.056 with the train profile.
- In a model fitted once, the set with the recent profile is the best on test (0.138 against 0.130), but not on validation, so it was not chosen. Once the model is refitted every month the difference disappears: −0.0004 (95% CI −0.004 to +0.003). A model refitted every month already follows the drift.

### Monthly refits

- Size of each hourly move: Gradient Boosting refitted every month with a 12-month half-life (chosen on validation) has R² 0.148 against 0.131 fitted once: +0.017 (95% CI +0.010 to +0.024), better in 20 of 29 test months. Smaller and less regular than with data from 2018 (+0.022, 27 of 29 months). A 6-month half-life is worse than fitting once (too few effective rows from 2020).
- Next-day volatility (new): every monthly variant beats fitting once on validation (0.442–0.449 against 0.408); the expanding window is chosen. Test R² 0.517 against 0.503: +0.015 (95% CI +0.004 to +0.025), better in 18 of 27 months. Recency weights add nothing here: with one row per day and data from 2020, down-weighting the early years removes more information than drift.
- The learning curves agree: volatility targets do better with recent data (next-day volatility validation R² 0.438 with the last 6 months against 0.417 with all 36), return targets need all the history.
- Refitting every month is the largest single gain of the project, larger than any feature, model family or ensemble.

### Networks against the best classical reference

- Size of each hourly move: the reference is Gradient Boosting refitted every month (R² 0.148). The network chosen on validation, the LSTM with hour features and a volume attention mask, fine-tuned every month, has 0.156: +0.008 (CI +0.001 to +0.015). Each ingredient is reliable: hour features +0.031, volume attention +0.014, monthly fine-tuning +0.011. The MLP only ties the trees. The average of Gradient Boosting, MLP and LSTM, all refitted monthly, reaches 0.161, but its validation R² was 0.001 below the LSTM's. With early stopping on the last 6 months of train the networks' validation R² is no longer inflated.
- Release hours: the LSTMs with hour features over-predict them on test (1.3–1.6× the actual size), Gradient Boosting under-predicts them (0.65×), the ensembles land in between (0.90–0.97×).
- Next-day volatility: the average of Gradient Boosting and an MLP is reliably better than Gradient Boosting alone (0.516 against 0.505, +0.011, CI +0.001 to +0.022); the MLP alone ties (0.505).
- Next-hour direction: LSTM on every origin AUC 0.538 against 0.533 for Gradient Boosting, a tie (CI −0.002 to +0.012). Training on every origin, not the network, is what helps (Gradient Boosting on every origin: 0.541).
- 24 hourly returns: every network stays at the zero forecast (MAE skill −0.0020 to −0.0004).
- Course pipeline: the Keras MLP stays behind Linear Regression (R² 0.399 against 0.514).
- Verdict: a network beats the best classical model only for the size of the hourly moves, narrowly, and only with hour features, volume attention and monthly fine-tuning; elsewhere it helps as a second model family in an ensemble or ties.

### Timestral, a pretrained foundation model, zero-shot

- Timestral (Mistral, released 15 Oct 2026) is pretrained on mostly synthetic series and used as it comes: nothing is fitted on our data. On validation we choose what it reads: the series (r or |r|), the context (30, 90 or 180 days), covariate groups one step at a time, and the model size.
- Chosen: the 28M model reading |r| over 180 days, with the clock (as categories), the release calendar (known-future) and the range and volume of the past hours. Validation R² 0.120, against 0.101 for Gradient Boosting refitted monthly and 0.121 for the best LSTM.
- Size of each hourly move, test: Timestral alone 0.157, +0.009 over Gradient Boosting refitted monthly (CI +0.003 to +0.016), level with the fine-tuned LSTM (0.156). The average with that Gradient Boosting, chosen on validation, reaches 0.164 (+0.016, CI +0.013 to +0.020): the best result of the project for this question.
- What it reads matters more than its size: |r| instead of r lifts validation R² from 0.067 to 0.108; the context still helps at 180 days (0.088 → 0.105 → 0.108); the release calendar +0.008, range and volume +0.004, the clock nothing. The 209M model gave +0.003 on test for 7× the compute (3.2 h for the test block on a laptop GPU), so the 28M is kept.
- 80% ranges: its own q10–q90 cover 80.3% of the test hours without any calibration (Gradient Boosting quantile regression needs the rolling conformal step to reach 79.9%), with the same width (0.99% against 0.98%) and even coverage across calm and wild hours. In release hours its ranges are wider (1.50% against 1.18%) and cover 70% against 56%.
- Returns: the median of its forecast does not beat 0% (MAE skill −0.0015), like every model.
- Caveat for the report: Timestral was released after our test block and Mistral does not list its real training series. If Bitcoin 2023–2026 is among them, its validation and test results are optimistic; this cannot be checked.
- Lesson for the report: a foundation model can match careful feature engineering and refitting on a target with real signal (the size of the moves), but cannot create signal where there is none (the returns).
