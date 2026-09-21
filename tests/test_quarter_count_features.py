import duckdb
import numpy as np
import pandas as pd
import pytest

from moscollector.features import COUNT_COLUMNS
from moscollector.quarter_count_features import aggregate_counts, carry_features, refresh_counts


def test_raw_quarter_counts_preserve_canonical_conflicts_and_signal_types():
    origin = pd.Timestamp("2025-10-01")
    source = pd.DataFrame(
        {
            "channel_id": [1, 1, 2, 2, 3, 4, 1, 2],
            "ts": [origin] * 6 + [origin + pd.Timedelta(minutes=15)] * 2,
            "value": [
                "Обнаружен дым",
                "Обнаружен дым",
                "Не замкнут",
                "Замкнут",
                "-100",
                "Включен",
                "Обнаружен дым",
                "Не замкнут",
            ],
            "alarm": [False, True, True, False, False, False, True, True],
            "numeric_value": [None, None, None, None, -100.0, None, None, None],
        }
    )
    channels = pd.DataFrame(
        {
            "channel_id": [1, 2, 3, 4],
            "object_id": [10, 10, 10, 20],
            "sensor_type": ["Датчик дыма", "КД Дверь", "Датчик температуры", "Состояние насоса"],
        }
    )
    with duckdb.connect() as con:
        con.register("source", source)
        con.register("channels", channels)
        actual = aggregate_counts(con).df()
    first = actual[(actual.object_id == 10) & actual.bucket.eq(origin)].iloc[0]
    assert (
        first.events,
        first.alarms,
        first.smoke_reports,
        first.ambiguous,
        first.access_reports,
        first.technical_codes,
    ) == (3, 2, 1, 1, 0, 1)
    following = actual[(actual.object_id == 10) & actual.bucket.gt(origin)].iloc[0]
    assert (following.events, following.smoke_reports, following.access_reports) == (2, 1, 1)
    assert actual[actual.object_id.eq(20)].pump_switches.tolist() == [1]


def test_count_refresh_exact_window_boundaries_and_future_invariance():
    origin = pd.Timestamp("2025-10-01")
    bucket = pd.to_datetime(
        [
            origin - pd.Timedelta(hours=168),
            origin - pd.Timedelta(hours=24),
            origin - pd.Timedelta(hours=6),
            origin - pd.Timedelta(hours=1),
            origin - pd.Timedelta(minutes=15),
            origin,
            origin + pd.Timedelta(minutes=15),
        ]
    )
    counts = pd.DataFrame({"object_id": 1, "bucket": bucket})
    for name in COUNT_COLUMNS:
        counts[name] = np.arange(1, 8)
    keys = pd.DataFrame(
        {"object_id": [1, 1, 2], "as_of": [origin, origin + pd.Timedelta(minutes=15), origin]}
    )
    frame = keys.assign(temperature=123.0)
    for hours in (1, 6, 24, 168):
        for name in COUNT_COLUMNS:
            frame[f"{name}_{hours}h"] = np.float32(0)
    frame["fault_reports_burst"] = np.float32(0)
    result = refresh_counts(frame.sample(frac=1, random_state=4).reset_index(drop=True), counts)
    for row in result.itertuples():
        for hours in (1, 6, 24, 168):
            matching = counts[
                (counts.object_id == row.object_id)
                & counts.bucket.ge(row.as_of - pd.Timedelta(hours=hours))
                & counts.bucket.lt(row.as_of)
            ]
            for name in COUNT_COLUMNS:
                assert getattr(row, f"{name}_{hours}h") == matching[name].sum()
        assert row.temperature == 123
    before = refresh_counts(frame.iloc[:1], counts)
    changed = counts.copy()
    changed.loc[changed.bucket.ge(origin), COUNT_COLUMNS] = 10000
    pd.testing.assert_frame_equal(before, refresh_counts(frame.iloc[:1], changed))
    assert before.events_1h.iloc[0] == 9
    assert before.events_168h.iloc[0] == 15
    np.testing.assert_allclose(before.fault_reports_burst.iloc[0], 12 / (1 + 15 / 28), rtol=1e-6)


def test_all_conflicting_values_remain_unknown_in_raw_counts_and_zero_in_model_counts():
    origin = pd.Timestamp("2025-10-01")
    source = pd.DataFrame(
        {
            "channel_id": [1, 1],
            "ts": [origin, origin],
            "value": ["Неисправен", "Исправен"],
            "alarm": [True, True],
            "numeric_value": [None, None],
        }
    )
    channels = pd.DataFrame({"channel_id": [1], "object_id": [10], "sensor_type": ["Датчик дыма"]})
    with duckdb.connect() as con:
        con.register("source", source)
        con.register("channels", channels)
        counts = aggregate_counts(con).df()
    assert counts.fault_reports.isna().all()
    assert counts.technical_codes.isna().all()
    assert counts.ambiguous.tolist() == [1]
    frame = pd.DataFrame(
        {
            "object_id": [10],
            "as_of": [origin + pd.Timedelta(minutes=15)],
            "fault_reports_1h": [np.float32(99)],
            "ambiguous_1h": [np.float32(99)],
        }
    )
    features = refresh_counts(frame, counts)
    assert features.fault_reports_1h.tolist() == [0]
    assert features.ambiguous_1h.tolist() == [1]


def test_feature_carry_is_past_only_preserves_missing_values_and_rejects_gaps():
    origin = pd.Timestamp("2025-10-01")
    hourly = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "as_of": [origin, origin + pd.Timedelta(hours=1), origin],
            "temperature": [np.nan, 200.0, -10.0],
            "parent_id": ["p", "p", "q"],
        }
    )
    slots = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "as_of": [origin + pd.Timedelta(minutes=15), origin + pd.Timedelta(minutes=45), origin],
        }
    )
    carried = carry_features(hourly, slots, ["object_id", "temperature", "parent_id"])
    assert carried[carried.object_id.eq(1)].temperature.isna().all()
    assert carried[carried.object_id.eq(2)].temperature.tolist() == [-10]
    assert carried.source_time.eq(origin).all()
    assert carried.parent_id.tolist() == ["p", "p", "q"]
    missing = pd.DataFrame({"object_id": [2], "as_of": [origin + pd.Timedelta(hours=1)]})
    with pytest.raises(ValueError, match="No whole-hour"):
        carry_features(hourly, missing, ["temperature"])
    with pytest.raises(ValueError, match="Duplicate"):
        carry_features(hourly, pd.concat([slots, slots]), ["temperature"])
