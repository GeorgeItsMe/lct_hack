"""Immutable operational bundles; historical replay keeps its original models."""

import hashlib
import json
import re
from functools import lru_cache

import numpy as np
from catboost import CatBoostClassifier, Pool

from moscollector.inference import CATEGORICAL, calibrated, model_input
from moscollector.paths import ARTIFACTS


def manifest_version(manifest):
    payload = {k: v for k, v in manifest.items() if k != "version"}
    return "op-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def active_version():
    path = ARTIFACTS / "active_model.json"
    return json.loads(path.read_text(encoding="utf-8"))["version"] if path.exists() else "legacy"


@lru_cache(maxsize=16)
def load_bundle(version):
    if not re.fullmatch(r"op-[a-f0-9]{12}", version):
        raise ValueError("Unknown operational model version")
    folder = ARTIFACTS / "operational" / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
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
            # A probability blend keeps each member's own features and calibration.
            features = member.get("features", meta["features"])
            if model.feature_names_ != features:
                raise ValueError("Model feature schema mismatch")
            members.append((float(member["weight"]), model, features, member.get("calibration")))
        weights = [m[0] for m in members]
        if not members or any(w <= 0 for w in weights) or not np.isclose(sum(weights), 1):
            raise ValueError("Invalid ensemble weights")
        if meta.get("variant") == "probability_blend" and any(m[3] is None for m in members):
            raise ValueError("A probability blend needs a calibration per member")
        heads[kind] = OperationalHead(meta, members)
    return heads


class OperationalHead:
    """A released head: weighted raw scores with one calibration, or, for a probability
    blend, a weighted mean of members' own calibrated probabilities."""

    def __init__(self, meta, members):
        self.meta, self.members = meta, members
        self.blend = meta.get("variant") == "probability_blend"

    def _raw(self, frame):
        x = model_input(frame, self.meta["features"])
        return sum(w * m.predict(x, prediction_type="RawFormulaVal") for w, m, _, _ in self.members)

    def probability(self, frame):
        if self.blend:
            return sum(
                w * calibrated(m.predict(model_input(frame, f), prediction_type="RawFormulaVal"), c)
                for w, m, f, c in self.members
            )
        return calibrated(self._raw(frame), self.meta["calibration"])

    def shap(self, frame):
        """Contributions over ``meta["features"]`` plus the bias column; a blend weights each
        member's contributions by its share and its calibration slope (calibrated log-odds)."""
        if not self.blend:
            x = Pool(model_input(frame, self.meta["features"]), cat_features=CATEGORICAL)
            return sum(w * m.get_feature_importance(x, type="ShapValues") for w, m, _, _ in self.members)
        names = self.meta["features"]
        total = np.zeros((len(frame), len(names) + 1))
        for w, m, f, c in self.members:
            w = w * c.get("slope", 1.0)
            values = m.get_feature_importance(
                Pool(model_input(frame, f), cat_features=CATEGORICAL), type="ShapValues"
            )
            total[:, [names.index(n) for n in f]] += w * values[:, :-1]
            total[:, -1] += w * values[:, -1]
        return total

    def expected_count(self, frame):
        if "rate_scale" not in self.meta or self.blend:
            return None
        return np.exp(np.clip(self._raw(frame), -20, 20)) * self.meta["rate_scale"]


def activate(version):
    load_bundle(version)  # Validate every model before publishing the pointer.
    path = ARTIFACTS / "active_model.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
    temporary.replace(path)
