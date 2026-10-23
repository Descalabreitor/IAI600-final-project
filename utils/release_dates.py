"""
release_dates.py - US macro release times (CPI, payrolls, FOMC) for the release-day analysis.

Builds data/release_dates.csv, used by data-analysis.ipynb (sections 8.7 and 9):

    python -m utils.release_dates                 # writes data/release_dates.csv
    from utils.release_dates import build_releases
    releases = build_releases()                    # same table as a DataFrame

Sources (dates checked on 20 Oct 2026):
  CPI:      https://www.bls.gov/bls/news-release/cpi.htm     (BLS archive; dates from the file names)
  Payrolls: https://www.bls.gov/bls/news-release/empsit.htm  (Employment Situation; same archive)
  FOMC:     https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (2021 ->) and
            https://www.federalreserve.gov/monetarypolicy/fomchistorical2018.htm, ...2019.htm, ...2020.htm

Times (New York time, converted to UTC with daylight saving):
  CPI and Employment Situation: 08:30 (12:30 or 13:30 UTC)
  FOMC statement: 14:00 on the last meeting day (18:00 or 19:00 UTC)
  Unscheduled 2020 cuts: 3 Mar 10:00 and Sunday 15 Mar 17:00, kind = "unscheduled"
  (not known in advance, so they must not be used as forecast features).

Not included: Fed notation votes, the 4 Oct 2019 FOMC call (no policy announcement), and the
October 2025 CPI and payrolls reports (not published during the US government shutdown).

To extend the table, add the new dates to the lists below (from the same pages) and run the module.

Columns: event (CPI / Payrolls / FOMC), date, time_new_york, kind (scheduled / unscheduled),
timestamp_utc (naive UTC, like the BTC index).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUTPUT = Path("data") / "release_dates.csv"

# BLS Consumer Price Index, release dates (one row per year)
CPI = """
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14
"""

# BLS Employment Situation (payrolls), release dates
PAYROLLS = """
2018-01-05 2018-02-02 2018-03-09 2018-04-06 2018-05-04 2018-06-01 2018-07-06 2018-08-03 2018-09-07 2018-10-05 2018-11-02 2018-12-07
2019-01-04 2019-02-01 2019-03-08 2019-04-05 2019-05-03 2019-06-07 2019-07-05 2019-08-02 2019-09-06 2019-10-04 2019-11-01 2019-12-06
2020-01-10 2020-02-07 2020-03-06 2020-04-03 2020-05-08 2020-06-05 2020-07-02 2020-08-07 2020-09-04 2020-10-02 2020-11-06 2020-12-04
2021-01-08 2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04 2021-07-02 2021-08-06 2021-09-03 2021-10-08 2021-11-05 2021-12-03
2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02
"""

# FOMC: last day of each scheduled meeting (the statement day)
FOMC = """
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16
"""

# FOMC emergency cuts of March 2020: (date, New York time of the announcement)
FOMC_UNSCHEDULED = [("2020-03-03", "10:00"), ("2020-03-15", "17:00")]

TIMES = {"CPI": "08:30", "Payrolls": "08:30", "FOMC": "14:00"}


def build_releases() -> pd.DataFrame:
    """The release table, sorted by time, with a few sanity checks."""
    rows = [(event, d, TIMES[event], "scheduled")
            for event, block in [("CPI", CPI), ("Payrolls", PAYROLLS), ("FOMC", FOMC)]
            for d in block.split()]
    rows += [("FOMC", d, t, "unscheduled") for d, t in FOMC_UNSCHEDULED]

    df = pd.DataFrame(rows, columns=["event", "date", "time_new_york", "kind"])
    local = pd.to_datetime(df["date"] + " " + df["time_new_york"]).dt.tz_localize("America/New_York")
    df["timestamp_utc"] = local.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # checks: no duplicates; every release on a weekday (except the Sunday cut of 15 Mar 2020)
    assert not df.duplicated(["event", "date"]).any(), "duplicated release"
    weekday = pd.to_datetime(df["date"]).dt.dayofweek
    assert ((weekday < 5) | (df["date"] == "2020-03-15")).all(), df[weekday >= 5]
    return df


def main(path=OUTPUT) -> None:
    df = build_releases()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} releases {df.groupby('event').size().to_dict()}, "
          f"{df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUTPUT)
