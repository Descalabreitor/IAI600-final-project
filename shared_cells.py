"""Run the setup cells of Alberto's notebook (data, features, CV folds, metrics, helpers) in another notebook.

Usage in a notebook:
    from shared_cells import run_shared_cells
    run_shared_cells(globals())
"""
import json

from IPython.utils.capture import capture_output

SOURCE_NB = "bitcoin-models-new-data.ipynb"

# Each cell is found by a piece of code it contains
SHARED_CELLS = [
    "data = np.load(",          # imports + loading the windows
    "def unscale(",             # unscale() + data checks
    "def make_features(",       # features, Ftr/Ytr, Fte/Yte
    "def metrics(",             # metrics
    "folds = list(",            # walk-forward folds
    "def cv_tune(",             # cv_tune, best_params, final_report
]


def run_shared_cells(namespace, source_nb=SOURCE_NB):
    with open(source_nb) as f:
        cells = ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    with capture_output():      # their printouts and plots are already in the source notebook
        for marker in SHARED_CELLS:
            matches = [c for c in cells if marker in c]
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one cell containing {marker!r} in {source_nb}, "
                                   f"found {len(matches)}")
            exec(matches[0], namespace)
    n = namespace
    print(f"{n['Ftr'].shape[1]} features | train {len(n['Ftr'])} windows "
          f"({n['Dtr'][0].date()} -> {n['Dtr'][-1].date()}) | test {len(n['Fte'])} windows "
          f"({n['Dte'][0].date()} -> {n['Dte'][-1].date()}) | {len(n['folds'])} CV folds")
