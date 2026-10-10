"""Run the setup cells of Alberto's notebooks (data, features, splits, helpers) in another notebook.

Usage in a notebook:
    from shared_cells import run_shared_cells
    run_shared_cells(globals())            # version 1: windows, 42 features, walk-forward folds
    run_shared_cells(globals(), "v3")      # version 3: hourly series, train/val/test blocks, long format
"""
import json

import pandas as pd
from IPython.utils.capture import capture_output

# Each cell is found by a piece of code it contains
SOURCES = {
    "v1": ("bitcoin-models-new-data.ipynb", [
        "data = np.load(",          # imports + loading the windows
        "def unscale(",             # unscale() + data checks
        "def make_features(",       # features, Ftr/Ytr, Fte/Yte
        "def metrics(",             # metrics
        "folds = list(",            # walk-forward folds
        "def cv_tune(",             # cv_tune, best_params, final_report
    ]),
    "v3": ("bitcoin-models-v3.ipynb", [
        "U.load_hourly()",          # imports + continuous hourly series
        "pos_all = ",               # position of every origin in the series (+ data checks)
        "F_base = U.base_features", # FEAT with the BASE, B1, B2 feature groups
        "DATA_START, VAL_START",    # train / val / test blocks: P, Y, D, Xw()
        "def tune(",                # tune(), RESULTS, record()
        "def Xl(",                  # long format Xl(), yl(), B3 per-hour features, baselines (needs tr_hours)
    ]),
}


def _cells(notebook):
    with open(notebook) as f:
        return ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]


def run_shared_cells(namespace, version="v1"):
    notebook, markers = SOURCES[version]
    cells = _cells(notebook)
    with capture_output():      # their printouts and plots are already in the source notebook
        for marker in markers:
            matches = [c for c in cells if marker in c]
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one cell containing {marker!r} in {notebook}, "
                                   f"found {len(matches)}")
            if marker == "def Xl(":     # tr_hours comes from Alberto's analysis cell, which also draws plots
                n = namespace
                n["tr_hours"] = n["hourly"].loc[n["D"]["train"][0]: n["D"]["train"][-1] + pd.Timedelta(hours=24)]
            exec(matches[0], namespace)
    n = namespace
    if version == "v1":
        print(f"{n['Ftr'].shape[1]} features | train {len(n['Ftr'])} windows "
              f"({n['Dtr'][0].date()} -> {n['Dtr'][-1].date()}) | test {len(n['Fte'])} windows "
              f"({n['Dte'][0].date()} -> {n['Dte'][-1].date()}) | {len(n['folds'])} CV folds")
    else:
        print(f"hourly series {n['hourly'].index[0]} -> {n['hourly'].index[-1]} | features: "
              f"base {len(n['BASE'])}, B1 {len(n['B1'])}, B2 {len(n['B2'])}, B3 {len(n['B3'])} (per hour)")
        for k in ("train", "val", "test"):
            print(f"  {k:<5} {len(n['P'][k]):>6} origins  {n['D'][k][0].date()} -> {n['D'][k][-1].date()}")
