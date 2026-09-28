import json

import pandas as pd
import pytest

from moscollector.count_research import fit_one
from moscollector.experiments.count_extension_research import validate_opportunities


def test_count_checkpoint_cannot_silently_reuse_different_training(tmp_path):
    old = {"kind": "fire", "best_iteration": 63}
    (tmp_path / "fire.json").write_text(json.dumps(old))
    assert fit_one(None, None, None, tmp_path, "fire", "2026-02-01") == old
    with pytest.raises(ValueError, match="different training configuration"):
        fit_one(None, None, None, tmp_path, "fire", "2026-02-01", depth=4)


def test_missing_evaluation_month_rejected_before_fitting():
    dates = pd.to_datetime(["2025-11-20", "2025-12-10"])
    frame = pd.DataFrame({"object_id": [1, 1], "as_of": dates, "eligible": True})
    periods = {
        "policy": (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")),
        "test": (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")),
    }
    with pytest.raises(ValueError, match="Missing 1 reference opportunities in test"):
        validate_opportunities(frame, frame.iloc[:1], periods)
    validate_opportunities(frame, frame, periods)
