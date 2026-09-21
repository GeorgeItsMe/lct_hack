import numpy as np
import pandas as pd

from moscollector.features import aggregate_year, future_counts, group_episodes
from moscollector.train import alert_metrics, choose_policy, split_mask


def test_future_window_has_exact_exclusive_horizon():
    counts = np.zeros(30, dtype=int)
    counts[0] = 1
    counts[23] = 2
    counts[24] = 4
    assert future_counts(counts, 24)[0] == 3
    assert future_counts(counts, 24)[1] == 6
    assert future_counts(counts, 24)[24] == 4


def test_split_purges_future_confirmation_across_boundary():
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(
                ["2026-02-27 22:00", "2026-02-27 23:00", "2026-02-28 12:00", "2026-03-01"], format="mixed"
            ),
            "eligible": True,
        }
    )
    assert split_mask(frame, "train").tolist() == [True, False, False, False]


def episode_rows(times):
    return pd.DataFrame(
        [
            {
                "object_id": 1,
                "kind": "fault",
                "start_ts": pd.Timestamp(t),
                "end_ts": pd.Timestamp(t) + pd.Timedelta(hours=2),
                "channel_id": i + 1,
                "sensor_name": f"Датчик {i}",
                "duration_seconds": 7200.0,
                "right_censored": False,
                "event_id": i + 1,
            }
            for i, t in enumerate(times)
        ]
    )


def test_cascade_window_cannot_chain_for_hours():
    frame = episode_rows(["2026-01-01 12:00", "2026-01-01 12:08", "2026-01-01 12:16"])
    groups = group_episodes(frame)
    assert len(groups) == 2
    assert groups.iloc[0].channel_count == 2
    assert groups.iloc[1].start_ts == pd.Timestamp("2026-01-01 12:16")


def test_one_warning_cannot_claim_multiple_incidents():
    predictions = pd.DataFrame(
        {
            "object_id": [1] * 3,
            "as_of": pd.to_datetime(["2026-06-10 00:00", "2026-06-10 03:00", "2026-06-10 06:00"]),
            "probability": [0.9, 0.9, 0.9],
        }
    )
    episodes = pd.DataFrame(
        {
            "object_id": [1, 1],
            "start_ts": pd.to_datetime(["2026-06-10 04:00", "2026-06-10 07:00"]),
            "episode_id": ["a", "b"],
        }
    )
    m = alert_metrics(predictions, episodes, 0.5)
    assert m["alerts"] == 1
    assert m["true_alerts"] == 1
    assert m["eligible_episodes"] == 2
    assert m["recall"] == 0.5
    assert m["median_lead_hours"] == 4


def test_episode_at_horizon_is_not_a_hit():
    predictions = pd.DataFrame(
        {"object_id": [1], "as_of": pd.to_datetime(["2026-06-10"]), "probability": [0.9]}
    )
    episodes = pd.DataFrame(
        {"object_id": [1], "start_ts": pd.to_datetime(["2026-06-11"]), "episode_id": ["a"]}
    )
    m = alert_metrics(predictions, episodes, 0.5)
    assert m["true_alerts"] == 0
    assert m["false_alerts"] == 1
    assert m["eligible_episodes"] == 0


def test_sparse_policy_period_disables_automatic_warnings():
    predictions = pd.DataFrame(
        {
            "object_id": [1] * 2,
            "as_of": pd.to_datetime(["2026-06-10", "2026-06-11"]),
            "probability": [0.9, 0.9],
        }
    )
    episodes = pd.DataFrame(
        {"object_id": [1], "start_ts": pd.to_datetime(["2026-06-10 04:00"]), "episode_id": ["a"]}
    )
    best, _ = choose_policy(predictions, episodes)
    assert best["threshold"] == 1.01
    assert best["status"] == "insufficient_policy_events"


def test_episode_extraction_requires_prior_state_and_sustained_fault(tmp_path, monkeypatch):
    from moscollector import features

    monkeypatch.setattr(features, "PROCESSED", tmp_path)
    monkeypatch.setattr(features, "ARTIFACTS", tmp_path)
    records = []
    sequences = {
        1: [("00:00:00", "Норма"), ("01:00:00", "Неисправен"), ("02:01:00", "Норма")],
        2: [("00:00:00", "Неисправен"), ("04:00:00", "Норма")],
        3: [("00:00:00", "Норма"), ("01:00:00", "Неисправен"), ("01:00:00", "Норма"), ("04:00:00", "Норма")],
        4: [("00:00:00", "Норма"), ("01:00:00", "Неисправен"), ("01:00:05", "Норма")],
        5: [
            ("00:00:00", "Норма"),
            ("01:00:00", "Неисправен"),
            ("01:15:00", "Неисправен"),
            ("03:00:00", "Норма"),
        ],
    }
    for channel, seq in sequences.items():
        for at, value in seq:
            records.append(
                {
                    "event_id": len(records) + 1,
                    "channel_id": channel,
                    "ts": pd.Timestamp("2027-01-01 " + at),
                    "value": value,
                    "numeric_value": float("nan"),
                    "alarm": value == "Неисправен",
                }
            )
    pd.DataFrame(records).to_parquet(tmp_path / "events-2027.parquet")
    pd.DataFrame(
        [
            {"channel_id": i, "object_id": 1, "sensor_type": "Датчик дыма", "sensor_name": f"Датчик {i}"}
            for i in sequences
        ]
    ).to_parquet(tmp_path / "channels.parquet")
    aggregate_year(2027)
    actual = pd.read_parquet(tmp_path / "episodes-2027.parquet")
    assert set(actual.channel_id) == {1, 5}
    assert set(actual.duration_seconds) == {3660.0, 7200.0}
