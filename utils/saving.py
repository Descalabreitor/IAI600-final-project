"""Save the models a notebook trains in models/<notebook>/, to load them again without training.

Usage in a notebook:
    from utils.saving import ModelSaver, load_model
    save = ModelSaver("classical-models", enabled=SAVE_MODELS)
    m = save("Ridge {'alpha': 1000}", make_ridge(1000).fit(X, y))     # returns the model, so calls can be chained
    m = load_model("classical-models", "Ridge {'alpha': 1000}")

scikit-learn objects (or dicts / lists of them) go to <file>.joblib. PyTorch networks and state_dicts (or lists of
them, one per seed) go to <file>.pt, as state_dicts on the CPU: the network classes live in the notebooks, so a
network is rebuilt with its class and then `net.load_state_dict(state)`. models/<notebook>/index.json lists every
model: its name, file, save time and the extra information passed to save().
"""
import hashlib
import json
import re
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def model_file(name):
    # readable file name + a short hash of the full name (two names never share a file)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80]
    return f"{slug}-{hashlib.md5(name.encode()).hexdigest()[:6]}"


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _is_torch(obj, torch):
    if isinstance(obj, (torch.nn.Module, torch.Tensor)):
        return True
    if isinstance(obj, dict):
        return any(_is_torch(v, torch) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_is_torch(v, torch) for v in obj)
    return False


def _cpu_states(obj, torch):
    # networks -> state_dicts on the CPU; lists are kept (one entry per seed)
    if isinstance(obj, torch.nn.Module):
        return {k: v.detach().cpu().clone() for k, v in obj.state_dict().items()}
    if isinstance(obj, (list, tuple)):
        return [_cpu_states(o, torch) for o in obj]
    return obj


class ModelSaver:
    def __init__(self, notebook, enabled=True, root=MODELS_DIR):
        self.dir, self.enabled = Path(root) / notebook, enabled

    def _index(self):
        path = self.dir / "index.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def __call__(self, name, model, **info):
        if not self.enabled:
            return model
        self.dir.mkdir(parents=True, exist_ok=True)
        torch = _torch()
        if torch is not None and _is_torch(model, torch):
            file = model_file(name) + ".pt"
            torch.save(_cpu_states(model, torch), self.dir / file)
        else:
            file = model_file(name) + ".joblib"
            joblib.dump(model, self.dir / file)
        index = self._index()
        index[name] = {"file": file, "saved": pd.Timestamp.now().isoformat(timespec="seconds"), **info}
        (self.dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return model

    def names(self):
        return list(self._index())

    def load(self, name):
        file = self._index()[name]["file"]
        if file.endswith(".pt"):
            return _torch().load(self.dir / file, map_location="cpu")
        return joblib.load(self.dir / file)


def load_model(notebook, name, root=MODELS_DIR):
    return ModelSaver(notebook, root=root).load(name)
