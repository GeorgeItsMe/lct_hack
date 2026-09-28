import duckdb
import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.numeric_channel_features import COLUMNS, create_hourly, transform

START = pd.Timestamp("2024-07-01")


def hourly(records):
    events = pd.DataFrame(records, columns=["channel_id", "hours", "value", "numeric_value"])
    events["ts"] = START + pd.to_timedelta(events.pop("hours"), unit="h")
    events["alarm"] = False
    channels = pd.DataFrame(
        {
            "channel_id": [1, 2, 3],
            "object_id": [10, 10, 20],
            "sensor_type": ["Датчик температуры", "Датчик температуры", "Газовый датчик"],
        }
    )
    with duckdb.connect() as c:
        c.register("events", events)
        c.register("channels", channels)
        create_hourly(c)
        return c.execute("SELECT * FROM numeric_hourly").fetchdf()


def query(hours, objects=None):
    return pd.DataFrame(
        {
            "object_id": objects if objects is not None else [10] * len(hours),
            "as_of": START + pd.to_timedelta(hours, unit="h"),
        }
    )


def test_conflict_and_later_invalid_report_do_not_resurrect_old_value():
    h = hourly(
        [
            (1, 0.1, "10", 10),
            (1, 0.1, "10", 10),
            (1, 0.2, "12", 12),
            (1, 0.2, "13", 13),
            (1, 1.1, "14", 14),
            (1, 1.2, "Неисправен", None),
        ]
    )
    assert h.n.tolist() == [1, 1]
    assert h.last_value.isna().all()
    r = transform(query([1, 2]), h)
    assert r.num_temperature_numeric_channels_168h.tolist() == [0, 0]
    assert r.num_temperature_logged_channels_168h.tolist() == [1, 1]
    assert r.num_temperature_last_mean.isna().all()
    assert r.num_temperature_seen_channels_24h.tolist() == [1, 1]


def test_exact_time_and_future_rows_are_excluded():
    first = [(1, 0.25, "10", 10)]
    a = transform(query([1]), hourly(first))
    b = transform(query([1]), hourly([*first, (1, 1, "90", 90), (2, 25, "99", 99)]))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.num_temperature_last_mean.item() == 10
    assert a.num_temperature_last_age_mean_h.item() == 0.75


def test_unknown_objects_families_and_original_query_order():
    h = hourly([(1, 0.25, "10", 10), (3, 0.25, "0.2", 0.2)])
    q = query([2, 1, 1], [20, 99, 10])
    result = transform(q, h)
    assert list(result) == COLUMNS
    assert result.num_temperature_numeric_channels_168h.tolist() == [0, 0, 1]
    assert result.num_gas_numeric_channels_168h.tolist() == [1, 0, 0]
    assert result.loc[1, "num_temperature_logged_channels_168h"] == 0
    assert np.isnan(result.loc[1, "num_temperature_last_mean"])


def test_stale_numeric_values_expire_even_without_an_invalid_report():
    r = transform(query([168, 169, 170]), hourly([(1, 0.25, "10", 10)]))
    assert r.num_temperature_numeric_channels_168h.tolist() == [1, 0, 0]
    assert r.num_temperature_seen_channels_24h.tolist() == [0, 0, 0]
    assert r.num_temperature_last_mean.iloc[1:].isna().all()


def test_same_channel_shift_preserves_offsetting_changes():
    h = hourly([(1, 0.25, "10", 10), (2, 0.25, "10", 10), (1, 24.25, "15", 15), (2, 24.25, "5", 5)])
    r = transform(query([1, 25]), h)
    assert r.num_temperature_last_mean.tolist() == [10, 10]
    assert r.num_temperature_shift_supported_channels.tolist() == [0, 2]
    assert r.num_temperature_relative_shift_24h_max.iloc[1] == pytest.approx(5 / 11)
    assert r.num_temperature_relative_shift_24h_min.iloc[1] == pytest.approx(-5 / 11)
    assert r.num_temperature_relative_shift_24h_abs_mean.iloc[1] == pytest.approx(5 / 11)


def test_relative_ranges_use_channel_observations_and_exact_windows():
    h = hourly([(1, 0.25, "10", 10), (1, 0.75, "20", 20), (1, 5.25, "30", 30), (2, 5.25, "5", 5)])
    r = transform(query([6, 7, 25]), h)
    assert r.num_temperature_relative_range_6h_max.iloc[0] == pytest.approx(20 / 21)
    assert r.num_temperature_relative_range_6h_mean.iloc[0] == pytest.approx(10 / 21)
    assert r.num_temperature_relative_range_6h_max.iloc[1] == 0
    assert r.num_temperature_relative_range_24h_max.iloc[2] == 0
    assert r.num_temperature_seen_channels_24h.iloc[2] == 2


@pytest.mark.parametrize("bad", ["duplicate", "fractional", "future_bucket"])
def test_unavailable_or_ambiguous_queries_are_rejected(bad):
    h = hourly([(1, 0.25, "10", 10)])
    q = query([1])
    if bad == "duplicate":
        q = pd.concat([q, q], ignore_index=True)
    elif bad == "fractional":
        q.loc[0, "as_of"] += pd.Timedelta(minutes=15)
    else:
        h.loc[0, "last_ts"] = h.loc[0, "as_of"]
    with pytest.raises(ValueError):
        transform(q, h)


def test_numeric_ranges_keep_existing_cleaning_rules():
    h = hourly(
        [
            (1, 0.1, "-100", -100),
            (1, 0.2, "-50", -50),
            (1, 0.3, "100", 100),
            (3, 0.1, "-1", -1),
            (3, 0.2, "0", 0),
            (3, 0.3, "100", 100),
        ]
    )
    assert h.n.tolist() == [2, 2]
    assert h.lo.tolist() == [-50, 0]
    assert h.hi.tolist() == [100, 100]


def test_channel_values_match_independent_raw_history_calculation():
    rng = np.random.default_rng(52)
    records = []
    for channel in (1, 2):
        for hour in sorted(rng.choice(240, 35, replace=False)):
            x = float(rng.integers(-10, 31))
            records.append((channel, hour + 0.25, str(x), x))
    q = query([250, 1, 12, 25, 72, 169, 240])
    out = transform(q, hourly(records))
    for i, t in enumerate([250, 1, 12, 25, 72, 169, 240]):
        values, ages, shifts, recent = [], [], [], []
        for channel in (1, 2):
            history = sorted((at, x) for c, at, _, x in records if c == channel and at < t)
            recent.append(any(t - 24 <= at < t for at, _ in history))
            if history and t - history[-1][0] <= 168:
                at, x = history[-1]
                values.append(x)
                ages.append(t - at)
                old = [(when, value) for when, value in history if when < t - 24]
                if old and t - 24 - old[-1][0] <= 168:
                    shifts.append((x - old[-1][1]) / (1 + abs(old[-1][1])))
        row = out.iloc[i]
        assert row.num_temperature_seen_channels_24h == sum(recent)
        assert row.num_temperature_numeric_channels_168h == len(values)
        assert row.num_temperature_shift_supported_channels == len(shifts)
        if values:
            for field, value in (
                ("mean", np.mean(values)),
                ("min", min(values)),
                ("max", max(values)),
                ("std", np.std(values)),
            ):
                assert row[f"num_temperature_last_{field}"] == pytest.approx(value, abs=1e-6)
            assert row.num_temperature_last_age_mean_h == pytest.approx(np.mean(ages))
        if shifts:
            assert row.num_temperature_relative_shift_24h_max == pytest.approx(max(shifts))
            assert row.num_temperature_relative_shift_24h_min == pytest.approx(min(shifts))
            assert row.num_temperature_relative_shift_24h_abs_mean == pytest.approx(np.abs(shifts).mean())
