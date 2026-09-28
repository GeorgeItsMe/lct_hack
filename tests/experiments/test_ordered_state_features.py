import duckdb
import pandas as pd

from moscollector.experiments.ordered_state_features import MAX_AGE_HOURS, canonical_transitions, transform


def test_raw_transitions_conflicts_numeric_recovery_and_year_carry():
    c = duckdb.connect()
    c.execute("""CREATE VIEW source AS SELECT * FROM (VALUES
        (1,TIMESTAMP '2025-01-01 00:00:00','1',1.0,false),
        (1,TIMESTAMP '2025-01-01 00:01:00','2',2.0,false),
        (1,TIMESTAMP '2025-01-01 00:02:00','Неисправен',NULL,true),
        (1,TIMESTAMP '2025-01-01 00:03:00','3',3.0,false),
        (2,TIMESTAMP '2025-01-01 00:00:00','Открыт',NULL,true),
        (2,TIMESTAMP '2025-01-01 00:00:00','Закрыт',NULL,false),
        (2,TIMESTAMP '2025-01-01 00:01:00','Закрыт',NULL,false),
        (2,TIMESTAMP '2025-01-01 00:01:00','Закрыт',NULL,false))
        AS v(channel_id,ts,value,numeric_value,alarm)""")
    c.execute("""CREATE VIEW seed AS SELECT 1 channel_id,TIMESTAMP '2024-12-31' ts,
                 '__NUMERIC__|alarm=false' state""")
    canonical_transitions(c)
    result = c.sql("SELECT * FROM changes ORDER BY channel_id,ts").df()
    assert len(result) == 4
    assert result.state.tolist() == [
        "Неисправен|alarm=true",
        "__NUMERIC__|alarm=false",
        "__CONFLICT__|alarm=true",
        "Закрыт|alarm=false",
    ]
    assert result.previous.iloc[0] == "__NUMERIC__|alarm=false"
    assert result.previous.iloc[2] == "__INITIAL_OBSERVATION__"
    c.close()


def test_ordered_features_strict_boundary_future_invariance_and_siblings():
    at = pd.Timestamp("2025-01-01 12:00")
    frame = pd.DataFrame({"object_id": [1, 2, 3, 4], "parent_id": [10, 10, None, None], "as_of": at})
    history = pd.DataFrame(
        {
            "channel_id": [11, 22, 12, 21, 41],
            "object_id": [1, 2, 1, 2, 4],
            "parent_id": [10, 10, 10, 10, None],
            "ts": [
                at - pd.Timedelta(hours=1),
                at - pd.Timedelta(hours=2),
                at,
                at + pd.Timedelta(hours=1),
                at - pd.Timedelta(hours=MAX_AGE_HOURS + 1),
            ],
            "previous": "normal",
            "state": "alarm",
        }
    )
    result = transform(frame, history)
    assert result.order_own_1_channel.iloc[0] == "11"
    assert result.order_sibling_1_channel.iloc[0] == "22"
    assert result.order_sibling_1_age_h.iloc[0] == 2
    assert result.order_sibling_1_channel.iloc[1] == "11"
    assert result.order_sibling_1_channel.iloc[2] == "__NONE__"
    assert result.order_own_1_channel.iloc[3] == "__NONE__"
    pd.testing.assert_frame_equal(result, transform(frame, history[history.ts.lt(at)]))
    pd.testing.assert_frame_equal(
        result, transform(frame.assign(parent_id=["10", "10", None, None]), history)
    )


def test_simultaneous_changes_have_stable_id_order_and_lag_limit():
    at = pd.Timestamp("2025-01-01 12:00")
    frame = pd.DataFrame({"object_id": [1], "parent_id": [10], "as_of": [at]})
    history = pd.DataFrame(
        {
            "channel_id": [5, 2, 4, 1, 3],
            "object_id": 1,
            "parent_id": 10,
            "ts": at - pd.Timedelta(seconds=1),
            "previous": "a",
            "state": "b",
        }
    )
    result = transform(frame, history)
    assert [result[f"order_own_{i}_channel"].iloc[0] for i in range(1, 5)] == ["5", "4", "3", "2"]
    pd.testing.assert_frame_equal(result, transform(frame, history.iloc[::-1]))


def test_per_object_hour_pruning_preserves_own_and_sibling_history():
    at = pd.Timestamp("2025-01-02 00:00")
    frame = pd.DataFrame({"object_id": [1, 2], "parent_id": [10, 10], "as_of": at})
    history = pd.DataFrame(
        {
            "channel_id": range(1, 21),
            "object_id": [1, 2] * 10,
            "parent_id": 10,
            "ts": pd.date_range("2025-01-01 23:05", periods=20, freq="3min"),
            "previous": "normal",
            "state": "alarm",
        }
    )
    pruned = (
        history.assign(hour=history.ts.dt.floor("h"))
        .sort_values(["ts", "channel_id"])
        .groupby(["object_id", "hour"])
        .tail(4)
    )
    pd.testing.assert_frame_equal(transform(frame, history), transform(frame, pruned))


def test_dynamic_categories_are_not_silently_coerced_to_numeric():
    from moscollector.experiments.ordered_state_features import AGES, CATS
    from moscollector.experiments.ordered_state_research import ordered_input
    from moscollector.train import CATEGORICAL

    values = {key: ["known", "other"] for key in CATEGORICAL + CATS}
    values.update({key: [1.0, 2.0] for key in AGES})
    frame = pd.DataFrame(values)
    for key in CATS:
        frame[key] = frame[key].astype("category")
    result = ordered_input(frame, list(frame))
    assert not result.isna().any().any()
    assert result[CATS].iloc[0].eq("known").all()
    assert result[CATS].iloc[1].eq("other").all()
