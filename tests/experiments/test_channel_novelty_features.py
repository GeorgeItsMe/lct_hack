import duckdb
import numpy as np
import pandas as pd

from moscollector.experiments.channel_novelty_features import extraction_query, transform


def test_onsets_are_real_family_entries_not_initial_states_or_alarm_toggles():
    c = duckdb.connect()
    c.execute("""CREATE VIEW typed AS SELECT * FROM (VALUES
        (1,10,TIMESTAMP '2025-01-01','Неисправен|alarm=true','__INITIAL_OBSERVATION__','Датчик дыма'),
        (1,10,TIMESTAMP '2025-01-02','Неисправен|alarm=true','Неисправен|alarm=false','Датчик дыма'),
        (1,10,TIMESTAMP '2025-01-03','Неисправен|alarm=true','Норма|alarm=false','Датчик дыма'),
        (2,10,TIMESTAMP '2025-01-03','Обнаружен дым|alarm=false','Норма|alarm=false','Датчик дыма'),
        (3,10,TIMESTAMP '2025-01-03','Не замкнут|alarm=false','Норма|alarm=false','КД Дверь'),
        (3,10,TIMESTAMP '2025-01-04','Не замкнут|alarm=true','Не замкнут|alarm=false','КД Дверь'),
        (4,10,TIMESTAMP '2025-01-03','xxCONFLICTyy|alarm=true','Норма|alarm=false','КД Дверь'),
        (4,10,TIMESTAMP '2025-01-04','__CONFLICT__|alarm=true','Норма|alarm=false','КД Дверь'))
        AS v(channel_id,object_id,ts,state,previous,sensor_type)""")
    result = c.sql(extraction_query()).df()
    assert result.signal.tolist() == ["fault", "fire", "access", "conflict"]
    assert len(result) == 4
    c.close()


def test_channel_relative_rates_strict_windows_and_future_invariance():
    at = pd.Timestamp("2025-02-15 12:00")
    frame = pd.DataFrame({"object_id": [1, 2, 3], "as_of": at})
    initial = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "channel_id": [10, 11, 20],
            "first_ts": [at - pd.Timedelta(days=40), at - pd.Timedelta(days=40), at - pd.Timedelta(days=1)],
        }
    )
    onsets = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 1, 2],
            "channel_id": [10, 10, 11, 11, 20],
            "ts": [
                at - pd.Timedelta(days=10),
                at - pd.Timedelta(hours=1),
                at - pd.Timedelta(hours=2),
                at,
                at - pd.Timedelta(hours=6),
            ],
            "signal": "unknown",
        }
    )
    result = transform(frame, onsets, initial)
    assert result.nov_unknown_channels_6h.tolist() == [2, 1, 0]
    assert result.nov_unknown_new_channels_24h.tolist() == [1, 0, 0]
    assert result.nov_observed_channels.tolist() == [2, 1, 0]
    assert result.nov_established_channels.tolist() == [2, 0, 0]
    assert result.nov_unknown_concentration_6h.tolist() == [0.5, 1, 0]
    assert np.isclose(result.nov_unknown_max_log_ratio_6h.iloc[0], np.log(11))
    assert result.nov_unknown_max_log_ratio_6h.iloc[1] == 0
    assert result.nov_fault_channels_6h.eq(0).all()
    past_only = transform(frame, onsets[onsets.ts.lt(at)], initial)
    pd.testing.assert_frame_equal(result, past_only)
    future = pd.DataFrame(
        {"object_id": [1], "channel_id": [12], "ts": [at + pd.Timedelta(days=1)], "signal": "unknown"}
    )
    future_initial = future.rename(columns={"ts": "first_ts"}).drop(columns="signal")
    pd.testing.assert_frame_equal(
        result, transform(frame, pd.concat([onsets, future]), pd.concat([initial, future_initial]))
    )


def test_disjoint_baseline_excludes_recent_window_and_preserves_object_order():
    at = pd.Timestamp("2025-02-15 12:00")
    frame = pd.DataFrame({"object_id": [1, 2], "as_of": at})
    initial = pd.DataFrame(
        {"object_id": [1, 2], "channel_id": [10, 20], "first_ts": at - pd.Timedelta(days=40)}
    )
    onsets = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "channel_id": [10, 10, 20],
            "ts": [at - pd.Timedelta(hours=6), at - pd.Timedelta(hours=720), at - pd.Timedelta(hours=1)],
            "signal": "fault",
        }
    )
    result = transform(frame, onsets, initial)
    assert np.isclose(result.nov_fault_max_excess_6h.iloc[0], 1 - 6 / 714)
    assert np.isclose(result.nov_fault_max_log_ratio_6h.iloc[0], np.log1p(1 / (6 / 714 + 0.1)))
    assert result.nov_fault_new_channels_24h.tolist() == [0, 1]
    pd.testing.assert_frame_equal(
        result.iloc[::-1].reset_index(drop=True), transform(frame.iloc[::-1], onsets, initial)
    )
