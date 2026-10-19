"""Registry labels and situation checks of the polygon demo world (committed bundle)."""

import pandas as pd
import pytest

from moscollector import polygon_view


@pytest.fixture(scope="module", autouse=True)
def bundle():
    if not polygon_view.available():
        pytest.skip("Polygon results are not present")


def test_hot_work_alarm_is_labelled_with_its_permit():
    rows = [
        r
        for r in polygon_view.queue("2026-07-01T00:00:00", 24 * 31)["rows"]
        if r["scenario"] == "hot_work_smoke"
    ]
    assert rows
    for row in rows:
        labels = [t["label"] for t in row["tags"] if t["kind"] == "hot_work"]
        assert labels and labels[0].split()[-1] in row["system"]


def test_intrusion_without_permit_is_labelled_access_without_permit():
    rows = polygon_view.queue("2026-07-01T00:00:00", 24 * 31)["rows"]
    intrusions = [r for r in rows if r["scenario"] == "intrusion" and "наряда нет" in r["truth"]]
    assert intrusions
    for row in intrusions:
        assert any(t["kind"] == "no_permit" for t in row["tags"])
        assert polygon_view.active_permits(row["object_id"], row["time"]).empty


def test_queue_window_and_situation_follow_the_moment():
    day = polygon_view.queue("2026-06-15T12:00:00", 24)
    assert all(day["begin"] < r["time"] <= day["end"] for r in day["rows"])
    row = polygon_view.queue("2026-07-01T00:00:00", 24 * 31)["rows"][0]
    situation = polygon_view.situation(row["object_id"], row["time"])
    at = pd.Timestamp(row["time"])
    for p in situation["permits"]:
        assert pd.Timestamp(p["end"]) >= at - pd.Timedelta(hours=12)
        assert pd.Timestamp(p["start"]) <= at + pd.Timedelta(hours=12)
        assert p["active"] == (pd.Timestamp(p["start"]) <= at <= pd.Timestamp(p["end"]))


def test_unit_warnings_come_from_pump_and_fan_scenarios():
    units = polygon_view.units("2026-06-15T12:00:00")
    assert units["pumps"] == 180 and units["fans"] == 354
    assert units["warnings"] and {w["unit"] for w in units["warnings"]} <= {"Насос", "Вентилятор"}
