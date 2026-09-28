import pandas as pd

from moscollector.experiments import device_features as device


def test_device_states_canonical_conflicts_and_strict_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(device, "PROCESSED", tmp_path)
    channels = pd.DataFrame(
        {"channel_id": [1, 2], "object_id": [10, 10], "sensor_type": ["Состояние охраны", "ИБП"]}
    )
    channels.to_parquet(tmp_path / "channels.parquet")
    entries = [
        (1, "2025-01-01 00:10", "На охране"),
        (2, "2025-01-01 00:20", "Питание от батарей"),
        (1, "2025-01-01 01:00", "Снято с охраны"),
        (2, "2025-01-01 01:30", "Питание от сети"),
        (2, "2025-01-01 02:10", "Неисправен"),
        (2, "2025-01-01 02:10", "Норма"),
        (1, "2025-06-01 00:00", "На охране"),
    ]
    events = pd.DataFrame(entries, columns=["channel_id", "ts", "value"])
    events["ts"] = pd.to_datetime(events.ts)
    events["alarm"] = False
    events.to_parquet(tmp_path / "events-2025.parquet")
    hourly = pd.read_parquet(device.aggregate(2025, "2025-06-01"))
    assert hourly.hour.max() < pd.Timestamp("2025-06-01")
    at = pd.date_range("2025-01-01 01:00", periods=3, freq="h")
    result = device.transform(at, hourly)
    assert result.dev_last_reported_armed.tolist() == [1, 0, 0]
    assert result.dev_last_reported_disarmed.tolist() == [0, 1, 1]
    assert result.dev_last_reported_ups_battery.tolist() == [1, 0, 0]
    assert result.dev_last_reported_fault.tolist() == [0, 0, 0]
    altered = hourly.copy()
    altered.loc[altered.hour.ge(at[0]), [f"delta_{k}" for k in device.PREDICATES]] = 999
    pd.testing.assert_frame_equal(result.iloc[[0]], device.transform(at, altered).iloc[[0]])
