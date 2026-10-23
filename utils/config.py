"""Period, split dates and data files shared by every notebook.

The blocks are defined by the date of the predicted hour; the notebooks purge 192 h before each border.
"""
import os

import pandas as pd

DATA_START = pd.Timestamp("2020-01-01")
VAL_START = pd.Timestamp("2023-01-01")      # first predicted hour of validation
TEST_START = pd.Timestamp("2024-07-01")     # first predicted hour of test
TRAIN_END = VAL_START - pd.Timedelta(hours=1)
VAL_END = TEST_START - pd.Timedelta(hours=1)
RELEASES_FILE = os.path.join("data", "release_dates.csv")
