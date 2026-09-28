from datetime import datetime, timedelta

import pytest

from moscollector import workorders
from moscollector.domain import data_recommendations, is_fault_state, risk_level
from moscollector.workorders import archive_outcome, emulated_status, suggested_label


@pytest.mark.parametrize(
    ("probability", "threshold", "level"),
    [
        (0.5, 0.2, "critical"),
        (0.3, 0.2, "high"),
        (0.12, 0.2, "watch"),
        (0.05, 0.2, "low"),
        (0.97, 0.6, "critical"),
    ],
)
def test_risk_levels_are_relative_to_the_threshold(probability, threshold, level):
    assert risk_level(probability, threshold) == level


def test_disabled_kind_never_becomes_a_warning_level():
    assert risk_level(0.99, 1.01) == "watch"
    assert risk_level(0.2, 1.01) == "low"


def test_recommendations_follow_recorded_signals_then_generic_guidance():
    events = [
        {"channel_id": i, "value": "Неисправен", "sensor_type": "Датчик дыма", "alarm": True}
        for i in range(6)
    ] + [{"channel_id": 99, "value": "Питание от батарей", "sensor_type": "ИБП", "alarm": True}]
    result = data_recommendations("fault", events)
    signals = [r for r in result if r["source"] == "signals"]
    # Six faulty smoke channels plus the UPS on battery: seven channels in fault state.
    assert "7 каналах" in signals[0]["text"] and "шлейф" in signals[0]["text"]
    assert any("АКБ" in r["text"] for r in signals)
    assert all(r["basis"] for r in result if r["source"] == "signals")
    assert result[-1]["source"] == "general"


def test_no_signals_leaves_only_generic_guidance():
    result = data_recommendations("access", [])
    assert result and {r["source"] for r in result} == {"general"}


def test_sentinel_dates_and_battery_states_count_as_faults():
    assert is_fault_state("01.01.1970 03:00:00") and is_fault_state("Обесточен")
    assert not is_fault_state("Норма") and not is_fault_state(None)


def test_emulated_work_order_progresses_with_time(monkeypatch):
    monkeypatch.setattr(workorders, "EMULATOR_STEP_SECONDS", 10.0)
    start = datetime(2026, 9, 28, 12, 0, 0)
    assert emulated_status(None) == "draft"
    assert [emulated_status(start, start + timedelta(seconds=s)) for s in (0, 11, 25, 45, 500)] == [
        "submitted",
        "accepted",
        "in_progress",
        "done",
        "done",
    ]


def test_outcome_and_labels_come_from_evidence_not_chance():
    assert archive_outcome("fire", True, False) == "confirmed"
    assert archive_outcome("fire", False, True) == "sensor_fault"
    assert archive_outcome("fire", False, False) == "not_confirmed"
    assert archive_outcome("fire", None, False) is None
    assert suggested_label("fault", "dispatch", "sensor_fault", False)[0] == 1
    assert suggested_label("fire", "dispatch", "sensor_fault", True)[0] == 0
    assert suggested_label("fire", "false_alarm", None, True)[0] == 0
    assert suggested_label("fire", "monitor", None, True)[0] == 1
    assert suggested_label("fire", "monitor", None, None)[0] is None
