"""The serving environment must reject stale or incomplete optional-GPU evidence."""

import json

import pytest

from moscollector.experiments.event_sequence_verification import verified_evidence
from moscollector.prepare import sha256, write_json


def make_proof(root):
    model = root / "fire/screen_1/event_gru"
    model.mkdir(parents=True)
    write_json(
        root / "plan.json", {"device": "mps", "torch": "2.14.0", "source_hashes": {}, "code_hashes": {}}
    )
    write_json(root / "report.json", {"fire": "test fixture"})
    write_json(root / "selection.json", {"fire": False})
    write_json(model.parent / "result.json", {})
    write_json(model / "fit.json", {})
    (model / "model.pt").write_bytes(b"synthetic weights; no Torch dependency")
    proof = {
        "device": "mps",
        "torch": "2.14.0",
        "report": {"fire": "test fixture"},
        "periods": {str(model.parent / "result.json"): {}},
        "models": {str(model / "fit.json"): {}},
        "prediction_and_frontier_hashes": {},
        "training_arrays_and_vocabularies_recomputed": True,
        "raw_scores_calibration_policies_and_metrics_recomputed": True,
        "all_archived_controls_exact_parity": True,
        "source_hashes": {str(p): sha256(p) for p in root.rglob("*") if p.is_file()},
        "code_hashes": {},
    }
    write_json(root / "weight-replay.json", proof)
    return model, proof


def test_replay_reader_rejects_weight_change_without_torch(tmp_path):
    model, proof = make_proof(tmp_path)
    assert verified_evidence(tmp_path) == json.loads(json.dumps(proof))
    (model / "model.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Changed raw-onset replay evidence"):
        verified_evidence(tmp_path)


def test_replay_reader_requires_coverage_and_hashes(tmp_path):
    model, proof = make_proof(tmp_path)
    extra = model.parent / "current_mlp"
    extra.mkdir()
    write_json(extra / "fit.json", {})
    with pytest.raises(ValueError, match="does not cover all neural weights"):
        verified_evidence(tmp_path)
    (extra / "fit.json").unlink()
    del proof["source_hashes"][str(model / "model.pt")]
    write_json(tmp_path / "weight-replay.json", proof)
    with pytest.raises(ValueError, match="omits required evidence hashes"):
        verified_evidence(tmp_path)
