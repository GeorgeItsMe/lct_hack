"""The 24-hour warning rule: one warning per object and kind per day, everywhere."""

import pandas as pd
import pytest

from moscollector import daily_rule
from moscollector.database import make_database
from moscollector.warning_storage import decide_stream_warnings


def hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


def test_repeat_warning_waits_24_hours():
    times = hours("2026-06-01", 72)
    frame = pd.DataFrame({"object_id": 1, "as_of": times, "probability": 0.9})
    issued = daily_rule.schedule(frame, 0.5)[1]
    assert list(pd.to_datetime(issued)) == [times[0], times[24], times[48]]
    assert len(daily_rule.schedule(frame, 0.95)[1]) == 0


def test_incident_day_counts_a_burst_of_episodes_once():
    times = hours("2026-06-01", 72)
    probability = [0.9 if t == times[10] else 0.1 for t in times]
    frame = pd.DataFrame({"object_id": 1, "as_of": times, "probability": probability})
    # Three episodes within one day form one incident day; the warning at 10:00 caught it.
    burst = pd.to_datetime(["2026-06-01 14:00", "2026-06-01 15:00", "2026-06-01 20:00"])
    # A later day nobody warned about.
    missed = pd.to_datetime(["2026-06-03 05:00"])
    episodes = pd.DataFrame({"object_id": 1, "start_ts": burst.append(missed)})
    result = daily_rule.metrics(frame, episodes, 0.5)
    assert result["alerts"] == result["true_alerts"] == 1
    assert result["precision"] == 1.0
    assert result["incident_days"] == 2 and result["caught_incident_days"] == 1
    assert result["recall"] == 0.5
    # One-to-one: one warning can match only one of the four episodes.
    assert result["recall_strict"] == pytest.approx(0.25)
    assert result["median_lead_hours"] == pytest.approx(4.0)
    # The per-warning table tells the same story: one warning, 4 hours before the day began.
    rows = daily_rule.Evaluator(frame, episodes).warnings(0.5)
    assert rows.to_dict("records") == [
        {"object_id": 1, "as_of": times[10], "probability": 0.9, "hit": True, "lead_hours": 4.0}
    ]


def test_warning_inside_a_running_incident_day_forecasts_nothing_new():
    times = hours("2026-06-01", 48)
    probability = [0.9 if t == times[15] else 0.1 for t in times]
    frame = pd.DataFrame({"object_id": 1, "as_of": times, "probability": probability})
    episodes = pd.DataFrame(
        {"object_id": 1, "start_ts": pd.to_datetime(["2026-06-01 14:00", "2026-06-01 18:00"])}
    )
    result = daily_rule.metrics(frame, episodes, 0.5)
    # Issued at 15:00 inside the day that began at 14:00: the 18:00 episode belongs to the
    # same incident day, so precision and recall share one unit and both are zero.
    assert result["precision"] == 0.0 and result["recall"] == 0.0
    assert result["precision_any_episode"] == 1.0 and result["alerts_in_running_incident"] == 1


def test_warning_inside_a_running_day_is_correct_when_the_next_day_starts_in_time():
    times = hours("2026-06-01", 72)
    probability = [0.9 if t == times[15] else 0.1 for t in times]
    frame = pd.DataFrame({"object_id": 1, "as_of": times, "probability": probability})
    episodes = pd.DataFrame(
        {"object_id": 1, "start_ts": pd.to_datetime(["2026-06-01 14:00", "2026-06-02 14:30"])}
    )
    result = daily_rule.metrics(frame, episodes, 0.5)
    # The second episode is 24.5 h after the first, so it starts a new incident day 23.5 h after
    # the 15:00 warning: that day is caught and the warning is correct.
    assert result["incident_days"] == 2 and result["caught_incident_days"] == 1
    assert result["precision"] == 1.0 and result["median_lead_hours"] == 23.5


def test_selection_prefers_recall_among_thresholds_with_precision_07():
    rows, episodes = [], []
    for obj in range(30):
        times = hours("2026-05-15", 48)
        good = obj < 20
        for t in times:
            p = 0.8 if (t == times[2] and good) else 0.6 if t == times[3] else 0.05
            rows.append({"object_id": obj, "as_of": t, "probability": p})
        if good:
            episodes.append({"object_id": obj, "start_ts": times[6]})
    chosen = daily_rule.select_threshold(pd.DataFrame(rows), pd.DataFrame(episodes))
    assert chosen["rule"] == "max_recall_at_precision_0.7"
    assert chosen["precision"] >= 0.7 and chosen["recall"] == 1.0


def stream_call(factory, key, at, probability):
    forecasts = [{"object_id": 1, "kind": "fire", "probability": probability, "threshold": 0.3}]
    episodes = pd.DataFrame({"object_id": [], "kind": [], "start_ts": pd.to_datetime([])})
    policies = {"fire": {"type": "daily", "cooldown_hours": 24}}
    return decide_stream_warnings(factory, key, at, "v1", forecasts, episodes, policies)[0]


def test_stream_issues_once_a_day_and_remembers_it_after_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'daily.db'}"
    engine, factory = make_database(url)
    first = stream_call(factory, "a", "2026-06-15T12:00:00", 0.9)
    assert first["notification_due"] and first["warning_active"]
    assert first["warning_issued_at"] == "2026-06-15T12:00:00"
    engine.dispose()
    engine, factory = make_database(url)
    # Still risky an hour later: the same warning stays active, no second notification.
    later = stream_call(factory, "b", "2026-06-15T13:00:00", 0.95)
    assert not later["notification_due"] and later["warning_active"]
    assert later["warning_issued_at"] == "2026-06-15T12:00:00"
    # Between hourly checks nothing is issued, the warning remains active.
    between = stream_call(factory, "c", "2026-06-15T13:05:00", 0.95)
    assert between["notification_status"] == "between_hourly_checks" and between["warning_active"]
    # Below the threshold the warning still covers its 24 hours.
    calm = stream_call(factory, "d", "2026-06-16T11:00:00", 0.1)
    assert calm["warning_active"] and not calm["notification_due"]
    expired = stream_call(factory, "e", "2026-06-16T12:00:00", 0.1)
    assert not expired["warning_active"] and expired["warning_issued_at"] is None
    again = stream_call(factory, "f", "2026-06-16T13:00:00", 0.9)
    assert again["notification_due"] and again["warning_issued_at"] == "2026-06-16T13:00:00"
    engine.dispose()


def test_stream_rejects_non_finite_probability_without_partial_state(tmp_path):
    engine, factory = make_database(f"sqlite:///{tmp_path / 'nan.db'}")
    with pytest.raises(ValueError, match="Non-finite"):
        stream_call(factory, "bad", "2026-06-15T12:00:00", float("nan"))
    assert stream_call(factory, "good", "2026-06-15T12:00:00", 0.9)["notification_due"]
    engine.dispose()


def test_published_june_table_matches_the_release_report():
    """The analyst's "warning and fact" table is the released rule on June, warning by warning."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "artifacts" / "daily_rule_evaluation.json").read_text(encoding="utf-8"))
    for folder in (root / "artifacts", root / "vercel_runtime" / "artifacts"):
        table = json.loads((folder / "daily_rule_matches.json").read_text(encoding="utf-8"))
        assert table["version"] == report["version"]
        assert set(table["kinds"]) == set(report["kinds"])
        for kind, item in table["kinds"].items():
            june = report["kinds"][kind]["june"]
            rows = item["rows"]
            assert len(rows) == june["alerts"]
            assert sum(r["hit"] for r in rows) == june["true_alerts"]
            leads = sorted(r["lead_hours"] for r in rows if r["hit"])
            assert all(r["lead_hours"] is None for r in rows if not r["hit"])
            assert pd.Series(leads).median() == pytest.approx(june["median_lead_hours"], abs=0.01)
