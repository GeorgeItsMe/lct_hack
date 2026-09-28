import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.numeric_channel_features import COLUMNS, KEYS
from moscollector.experiments.numeric_count_research import attach, screen


def rows():
    return pd.DataFrame(
        {
            "object_id": [2, 1],
            "as_of": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "count_target": [0, 1],
            "eligible": [True, True],
        }
    )


def context(frame):
    return pd.concat([frame[KEYS], pd.DataFrame(np.nan, index=frame.index, columns=COLUMNS)], axis=1)


def test_absent_numeric_coverage_preserves_negative_opportunities_and_targets():
    original = rows()
    out = attach(original, context(original).iloc[::-1])
    pd.testing.assert_frame_equal(out[original.columns], original, check_exact=True)
    assert out[COLUMNS].isna().all().all()


def test_missing_query_does_not_become_missing_coverage():
    original = rows()
    with pytest.raises(ValueError, match="Missing numeric query"):
        attach(original, context(original).iloc[[1]])


def test_label_column_cannot_enter_numeric_context():
    original = rows()
    with pytest.raises(ValueError, match="schema"):
        attach(original, context(original).assign(target_fire=1))


def score(tp, alerts, events=100):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def period(candidate, control, historical, recent):
    return {
        "arms": {"numeric": {"scores": candidate}, "control": {"scores": control}},
        "references": {"historical": historical, "recent": recent},
    }


def test_gate_requires_every_stronger_reference():
    row = period(score(70, 100), score(50, 90), score(55, 100), score(75, 100))
    selected = screen([row, row])
    assert not selected["passed_screen"]
    assert selected["scores"]["numeric"]["true_alerts"] == 140


def test_primary_gain_does_not_override_f1_loss():
    row = period(score(52, 100), score(90, 180), score(40, 100), score(40, 100))
    assert not screen([row])["passed_screen"]


def test_clear_improvement_over_all_references_passes_research_gate():
    row = period(score(70, 90), score(50, 90), score(50, 100), score(55, 100))
    selected = screen([row, row])
    assert selected["passed_screen"]
    assert not selected["scores"]["numeric"]["both_90"]


def test_reference_cannot_silently_disappear_from_one_period():
    a = period(score(70, 90), score(50, 90), score(50, 100), score(55, 100))
    b = period(score(70, 90), score(50, 90), score(50, 100), score(55, 100))
    b["references"].pop("recent")
    with pytest.raises(ValueError, match="omit a reference"):
        screen([a, b])
    with pytest.raises(ValueError, match="No numeric"):
        screen([])
