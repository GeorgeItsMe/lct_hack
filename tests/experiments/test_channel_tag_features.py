import duckdb
import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.channel_tag_features import (
    COLUMNS,
    SIGNALS,
    STATS,
    catalog_groups,
    extra_query,
    transform,
)

T = pd.Timestamp("2025-01-03")


def catalog():
    return pd.DataFrame(
        {
            "channel_id": [1, 2, 3, 4, 5, 6],
            "object_id": [10, 10, 10, 20, 10, 10],
            "sensor_type": ["Датчик дыма", "ИБП", "КД Дверь", "ИБП", "ИБП", "ИБП"],
            "system_tag": ["1-2.3.1.", "1-2.3.2.", "1-2.30.1.", "1-2.3.1.", None, "1-2.3.."],
        }
    )


def events(records):
    c = catalog().set_index("channel_id")
    return pd.DataFrame(
        [
            {
                "channel_id": ch,
                "object_id": int(c.loc[ch, "object_id"]),
                "ts": T + pd.Timedelta(hours=hours),
                "signal": signal,
            }
            for ch, hours, signal in records
        ],
        columns=["channel_id", "object_id", "ts", "signal"],
    )


def query(hours=(0,), objects=None):
    return pd.DataFrame(
        {
            "object_id": [10] * len(hours) if objects is None else objects,
            "as_of": T + pd.to_timedelta(hours, unit="h"),
        }
    )


def test_groups_use_exact_segments_and_isolate_unknown_syntax():
    g = catalog_groups(catalog()).set_index("channel_id")
    assert g.loc[1, "tag_group"] == g.loc[2, "tag_group"]
    assert g.loc[1, "tag_group"] != g.loc[3, "tag_group"]
    assert g.loc[1, "group_size"] == 2
    assert g.loc[4, "group_size"] == 1
    assert g.loc[5, "tag_group"] != g.loc[6, "tag_group"]
    assert not g.loc[[5, 6], "tag_valid"].any()


def test_peer_activity_does_not_count_the_same_channel_twice():
    onset = events([(1, -1, "fire"), (1, -1, "fault"), (2, -2, "power"), (3, -1, "access")])
    r = transform(query(), onset, catalog()).iloc[0]
    assert r.tag_fire_max_other_channels_6h == 1
    assert r.tag_fire_max_signal_types_6h == 3
    assert r.tag_fire_max_fraction_6h == 0.5
    assert r.tag_access_max_other_channels_6h == 0
    assert r.tag_power_max_other_channels_6h == 1


def test_other_objects_cannot_supply_peers_or_change_group_denominator():
    a = transform(query(), events([(1, -1, "fire")]), catalog())
    b = transform(query(), events([(1, -1, "fire"), (4, -1, "power")]), catalog())
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_exact_time_and_future_onsets_are_unavailable():
    a = transform(query(), events([(1, -1, "fire")]), catalog())
    b = transform(query(), events([(1, -1, "fire"), (2, 0, "power"), (3, 4, "access")]), catalog())
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_window_lower_bounds_and_distinct_channel_counts():
    onset = events([(1, -24, "fire"), (1, -6, "fire"), (1, -3, "fire"), (2, -6, "fire")])
    r = transform(query(), onset, catalog()).iloc[0]
    assert r.tag_fire_groups_6h == 1
    assert r.tag_fire_max_channels_6h == 2
    assert r.tag_fire_max_fraction_6h == 1
    assert r.tag_fire_max_channels_24h == 2
    assert r.tag_fire_max_other_channels_6h == 0


def test_empty_history_and_unknown_object_preserve_query_rows():
    q = query([0, -1, 2], [10, 99, 20])
    r = transform(q, events([]), catalog())
    assert len(r) == 3 and list(r) == COLUMNS
    assert (r.to_numpy() == 0).all()


@pytest.mark.parametrize("bad", ["duplicate", "unknown_signal", "wrong_object", "missing_time"])
def test_invalid_onset_identity_is_rejected(bad):
    onset = events([(1, -1, "fire")])
    if bad == "duplicate":
        onset = pd.concat([onset, onset], ignore_index=True)
    elif bad == "unknown_signal":
        onset.loc[0, "signal"] = "invented"
    elif bad == "wrong_object":
        onset.loc[0, "object_id"] = 999
    else:
        onset.loc[0, "ts"] = pd.NaT
    with pytest.raises(ValueError):
        transform(query(), onset, catalog())


def test_water_and_pump_entries_exclude_initial_and_alarm_only_pump_changes():
    rows = [
        (1, "Датчик затопления", "Не замкнут|alarm=true", "Не замкнут|alarm=false"),
        (2, "Состояние насоса", "Затоплен|alarm=false", "Выключен|alarm=false"),
        (3, "Состояние насоса", "Включен|alarm=true", "Включен|alarm=false"),
        (4, "Состояние насоса", "Выключен|alarm=false", "Включен|alarm=false"),
        (5, "Состояние насоса", "Затоплен|alarm=true", "__INITIAL_OBSERVATION__"),
        (6, "Состояние насоса", "Работают все насосы в АНС|alarm=false", "Неисправен|alarm=true"),
    ]
    typed = pd.DataFrame(rows, columns=["channel_id", "sensor_type", "state", "previous"]).assign(
        object_id=10, ts=T
    )
    with duckdb.connect() as c:
        c.register("typed", typed)
        r = c.execute(extra_query()).fetchdf()
    assert sorted(zip(r.channel_id, r.signal, strict=True)) == [
        (1, "flood"),
        (2, "flood"),
        (4, "pump"),
        (6, "pump"),
    ]


def test_all_summaries_match_direct_set_operations():
    rng = np.random.default_rng(6)
    records = []
    for channel in range(1, 7):
        for h in sorted(rng.choice(50, 8, replace=False)):
            records.append((channel, float(h - 30), SIGNALS[int(rng.integers(len(SIGNALS)))]))
    onset = events(records)
    q = query([4, -5, 0, 3], [10, 20, 10, 99])
    actual = transform(q, onset, catalog())
    memberships = {10: [[1, 2], [3], [5], [6]], 20: [[4]], 99: []}
    for i, row in q.iterrows():
        expected = {f"tag_{signal}_{stat}": 0.0 for signal in SIGNALS for stat in STATS}
        for members in memberships[row.object_id]:
            windows = {
                h: onset.loc[
                    onset.channel_id.isin(members)
                    & onset.ts.ge(row.as_of - pd.Timedelta(hours=h))
                    & onset.ts.lt(row.as_of)
                ]
                for h in (6, 24)
            }
            all6 = set(windows[6].channel_id)
            types6 = windows[6].signal.nunique()
            for signal in SIGNALS:
                active = {h: set(windows[h].loc[windows[h].signal.eq(signal), "channel_id"]) for h in (6, 24)}
                for h in (6, 24):
                    expected[f"tag_{signal}_groups_{h}h"] += bool(active[h])
                    for stat, v in (
                        (f"max_channels_{h}h", len(active[h])),
                        (f"max_fraction_{h}h", len(active[h]) / len(members)),
                    ):
                        key = f"tag_{signal}_{stat}"
                        expected[key] = max(expected[key], v)
                if active[6]:
                    for stat, v in (
                        ("max_other_channels_6h", len(all6 - active[6])),
                        ("max_signal_types_6h", types6),
                    ):
                        key = f"tag_{signal}_{stat}"
                        expected[key] = max(expected[key], v)
        np.testing.assert_allclose(actual.iloc[i], [expected[c] for c in COLUMNS], rtol=1e-7, atol=1e-7)
