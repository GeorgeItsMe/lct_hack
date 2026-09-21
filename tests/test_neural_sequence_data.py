import numpy as np
import pandas as pd
import pytest

from moscollector.neural_sequence_data import encode, fit_codec, history_batch, history_positions


def test_history_excludes_present_future_other_objects_and_old_rows():
    history = pd.DataFrame(
        {
            "object_id": [1, 1, 2, 1, 1],
            "as_of": pd.to_datetime(
                ["2025-01-01", "2025-01-03", "2025-01-03 01:00", "2025-01-03 03:00", "2025-01-03 06:00"],
                format="mixed",
            ),
            "eligible": [False] * 5,
        }
    )
    now = pd.DataFrame({"object_id": [1, 2, 3], "as_of": pd.to_datetime(["2025-01-03 03:00"] * 3)})
    ids, ages = history_positions(history, now, steps=3)
    np.testing.assert_array_equal(ids, [[1, -1, -1], [2, -1, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(ages, [[3, 0, 0], [2, 0, 0], [0, 0, 0]])
    newer = pd.concat(
        [
            history,
            pd.DataFrame({"object_id": [1], "as_of": [pd.Timestamp("2025-02-01")], "eligible": [True]}),
        ],
        ignore_index=True,
    )
    np.testing.assert_array_equal(history_positions(newer, now, steps=3)[0], ids)
    history["eligible"] = True
    np.testing.assert_array_equal(history_positions(history, now, steps=3)[0], ids)


def test_history_boundary_order_and_right_padding_preserve_row_identity():
    h = pd.DataFrame(
        {
            "object_id": [1] * 4,
            "as_of": pd.to_datetime(
                ["2025-01-02", "2025-01-01", "2025-01-02 03:00", "2025-01-03"], format="mixed"
            ),
        }
    )
    current = pd.DataFrame(
        {"object_id": [1, 1], "as_of": pd.to_datetime(["2025-01-03", "2025-01-02"], format="mixed")}
    )
    ids, ages = history_positions(h, current, steps=4)
    np.testing.assert_array_equal(ids, [[1, 0, 2, -1], [1, -1, -1, -1]])
    np.testing.assert_array_equal(ages, [[48, 24, 21, 0], [24, 0, 0, 0]])
    tokens, lengths = history_batch(np.arange(8, dtype=np.float32).reshape(4, 2), ids, ages)
    np.testing.assert_array_equal(lengths, [3, 1])
    assert not tokens[0, 3].any() and not tokens[1, 1:].any()
    with pytest.raises(ValueError):
        history_positions(pd.concat([h, h.iloc[:1]]), current)


def test_codec_uses_only_training_features_and_marks_missing_unknown():
    train = pd.DataFrame(
        {
            "object_id": [1, 2],
            "parent_id": [1, 1],
            "object_kind": ["a", "b"],
            "x": [0, 3],
            "empty": [np.nan, np.inf],
            "target_access": [0, 1],
        }
    )
    columns = ["object_id", "parent_id", "object_kind", "x", "empty"]
    codec = fit_codec(train, columns)
    x, _ = encode(train, codec)
    np.testing.assert_allclose(x[:, 0], [-1, 1])
    np.testing.assert_array_equal(x[:, 1:], [[0, 0, 1], [0, 0, 1]])
    future = train.iloc[:1].assign(object_id=999, object_kind="new", x=1000000)
    values, cats = encode(future, codec)
    assert values[0, 0] == 8 and cats[0, 0] == cats[0, 2] == 0
    train["target_access"] = [999, 999]
    assert fit_codec(train, columns) == codec
    with pytest.raises(ValueError):
        fit_codec(train, [*columns, "target_access"])
