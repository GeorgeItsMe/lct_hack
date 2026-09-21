import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from moscollector import model_registry as registry
from moscollector.importing import ImportManager
from moscollector.train import CATEGORICAL, calibrated, model_input


def test_versioned_ensemble_and_shap_use_the_same_weights_and_reject_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ARTIFACTS", tmp_path)
    registry.load_bundle.cache_clear()
    data = pd.DataFrame(
        {"object_id": [1, 2] * 10, "parent_id": [1] * 20, "object_kind": ["node"] * 20, "signal": range(20)}
    )
    columns = list(data.columns)
    x = model_input(data, columns)
    models, members = [], []
    work = tmp_path / "work"
    work.mkdir()
    for i, cutoff in enumerate((7, 12)):
        model = CatBoostClassifier(
            iterations=8,
            depth=2,
            cat_features=CATEGORICAL,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(x, (data.signal > cutoff).astype(int))
        name = f"{'first' if i == 0 else 'second'}.cbm"
        model.save_model(str(work / name))
        models.append(model)
        members.append(
            {"file": name, "sha256": hashlib.sha256((work / name).read_bytes()).hexdigest(), "weight": 0.5}
        )
    calibration = {"slope": 1.2, "intercept": -0.3}
    manifest = {"models": {"fire": {"features": columns, "calibration": calibration, "members": members}}}
    version = registry.manifest_version(manifest)
    folder = tmp_path / "operational" / version
    folder.parent.mkdir()
    work.rename(folder)
    manifest["version"] = version
    (folder / "manifest.json").write_text(json.dumps(manifest))
    registry.activate(version)
    assert registry.active_version() == version
    head = registry.load_bundle(version)["fire"]
    expected_raw = sum(model.predict(x, prediction_type="RawFormulaVal") for model in models) / 2
    np.testing.assert_allclose(head.probability(data), calibrated(expected_raw, calibration))
    # SHAP including expected value reconstructs this exact ensemble, not one member.
    np.testing.assert_allclose(head.shap(data).sum(axis=1), expected_raw, atol=1e-12)
    registry.load_bundle.cache_clear()
    (folder / "first.cbm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        registry.load_bundle(version)
    with pytest.raises(ValueError, match="Unknown"):
        registry.load_bundle("../../other")
    registry.load_bundle.cache_clear()


def test_new_model_creates_new_snapshot_but_old_snapshot_remains_addressable(tmp_path, monkeypatch):
    version = ["legacy"]
    monkeypatch.setattr(registry, "active_version", lambda: version[0])
    manager = ImportManager.__new__(ImportManager)
    manager.root, manager.lock, manager.slots = tmp_path, threading.Lock(), threading.BoundedSemaphore(4)
    manager.pool = SimpleNamespace(submit=lambda *args: None)
    frame = pd.DataFrame({"signal": [1]})
    cutoff = pd.Timestamp("2026-06-15T12:00")
    old = manager.submit_frame(frame, cutoff, {"accepted_rows": 1}, "same-input", "json", 1)
    version[0] = "op-123456789abc"
    new = manager.submit_frame(frame, cutoff, {"accepted_rows": 1}, "same-input", "json", 1)
    replay = manager.submit_frame(frame, cutoff, {"accepted_rows": 1}, "same-input", "json", 1)
    assert old["id"] != new["id"] == replay["id"]
    assert manager.get(old["id"])["model_version"] == "legacy"
    assert manager.get(new["id"])["model_version"] == version[0]
    assert manager.find("same-input", model_version="legacy")["id"] == old["id"]
