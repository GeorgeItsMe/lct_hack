"""Immutable operational bundles; historical replay keeps its original models."""

import hashlib
import json
import re
from functools import lru_cache

import numpy as np
from catboost import CatBoostClassifier, Pool

from moscollector.paths import ARTIFACTS
from moscollector.train import CATEGORICAL, calibrated, model_input


def manifest_version(manifest):
    payload = {k: v for k, v in manifest.items() if k != "version"}
    return "op-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def active_version():
    path = ARTIFACTS / "active_model.json"
    return json.loads(path.read_text())["version"] if path.exists() else "legacy"


@lru_cache(maxsize=16)
def load_bundle(version):
    if not re.fullmatch(r"op-[a-f0-9]{12}", version):
        raise ValueError("Unknown operational model version")
    folder = ARTIFACTS / "operational" / version
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("version") != version or manifest_version(manifest) != version:
        raise ValueError("Model manifest integrity check failed")
    heads = {}
    for kind, meta in manifest["models"].items():
        members = []
        for member in meta["members"]:
            name = member["file"]
            if not re.fullmatch(r"[a-z_]+\.cbm", name):
                raise ValueError("Invalid model member filename")
            file = folder / name
            if hashlib.sha256(file.read_bytes()).hexdigest() != member["sha256"]:
                raise ValueError("Model weights integrity check failed")
            model = CatBoostClassifier()
            model.load_model(str(file))
            if model.feature_names_ != meta["features"]:
                raise ValueError("Model feature schema mismatch")
            members.append((float(member["weight"]), model))
        if not members or any(w <= 0 for w, _ in members) or not np.isclose(sum(w for w, _ in members), 1):
            raise ValueError("Invalid ensemble weights")
        heads[kind] = OperationalHead(meta, members)
    return heads


class OperationalHead:
    def __init__(self, meta, members):
        self.meta, self.members = meta, members

    def probability(self, frame):
        x = model_input(frame, self.meta["features"])
        raw = sum(w * m.predict(x, prediction_type="RawFormulaVal") for w, m in self.members)
        return calibrated(raw, self.meta["calibration"])

    def shap(self, frame):
        x = Pool(model_input(frame, self.meta["features"]), cat_features=CATEGORICAL)
        return sum(w * m.get_feature_importance(x, type="ShapValues") for w, m in self.members)


def activate(version):
    load_bundle(version)  # Validate every model before publishing the pointer.
    path = ARTIFACTS / "active_model.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": version}, indent=2))
    temporary.replace(path)
