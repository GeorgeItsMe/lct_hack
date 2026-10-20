"""Polygon: generated incidents sit on real channels, and the service shows the results."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from moscollector import daily_rule, polygon

BUNDLE = Path(__file__).resolve().parents[1] / "vercel_runtime"
PROCESSED = BUNDLE / "data" / "processed"


@pytest.fixture(scope="module")
def catalog():
    if not (PROCESSED / "events-2026.parquet").exists():
        pytest.skip("Runtime bundle is not present")
    return polygon.Catalog(PROCESSED)


@pytest.fixture(scope="module")
def world(catalog):
    return polygon.generate(7, processed=PROCESSED, catalog=catalog)


def test_same_seed_gives_the_same_world(catalog, world):
    again = polygon.generate(7, processed=PROCESSED, catalog=catalog)
    pd.testing.assert_frame_equal(world.events, again.events)
    assert world.truth.incident_id.tolist() == again.truth.incident_id.tolist()


def test_incident_counts_follow_the_agreed_rates(world):
    counts = world.truth.scenario.value_counts()
    assert counts["intrusion"] == polygon.planned(polygon.RATES["intrusions_per_week"])
    assert counts["real_fire"] == polygon.planned(polygon.RATES["real_fires_per_week"])
    assert counts["cascade"] == polygon.planned(polygon.RATES["cascades_per_week"])


def test_injected_records_use_real_channels_of_the_right_type(catalog, world):
    types = catalog.channels.set_index("channel_id").group
    events = world.events.assign(group=world.events.channel_id.map(types))
    assert events.group.notna().all()
    scenario = events.incident_id.map(world.truth.set_index("incident_id").scenario)
    smoke = events[scenario.eq("real_fire") & events.value.eq("Обнаружен дым")]
    assert len(smoke) and smoke.group.eq("smoke").all() and smoke.alarm.all()
    doors = events[scenario.eq("intrusion") & events.value.eq("Не замкнут")]
    assert len(doors) and doors.group.eq("door").all()
    pumps = events[scenario.eq("pump_failure")]
    assert set(pumps.group) == {"pump"}


def test_intrusions_never_overlap_a_permit(world):
    for inc in world.truth[world.truth.scenario.eq("intrusion")].itertuples():
        assert polygon.permit_at(world.permits, inc.object_id, inc.start) is None


def test_hard_worlds_hide_some_hot_works_from_the_registry(catalog):
    hard = polygon.generate(7, processed=PROCESSED, catalog=catalog, rates=polygon.HARD_RATES)
    hot = hard.truth[hard.truth.scenario.eq("hot_work_smoke")]
    assert (~hot.registered.astype(bool)).any()
    assert hot[~hot.registered.astype(bool)].permit_id.isna().all()


def test_warning_list_matches_the_metrics():
    times = pd.date_range("2026-06-01", periods=96, freq="h")
    predictions = pd.DataFrame(
        {"object_id": 1, "as_of": times, "probability": [0.9 if t.hour == 8 else 0.1 for t in times]}
    )
    episodes = pd.DataFrame(
        {"object_id": 1, "start_ts": pd.to_datetime(["2026-06-01 12:00", "2026-06-03 20:00"])}
    )
    evaluator = daily_rule.Evaluator(predictions, episodes)
    warnings = evaluator.warnings(0.5)
    result = evaluator.evaluate(0.5)
    assert len(warnings) == result["alerts"] == 4
    assert int(warnings.hit.sum()) == result["true_alerts"] == 2


def test_polygon_results_and_dataset_are_served(monkeypatch):
    from moscollector import polygon_view

    if not polygon_view.available():
        pytest.skip("Polygon results are not prepared")
    report = polygon_view.report()
    assert {"scenarios", "hard", "effect", "design"} <= set(report)
    feed = polygon_view.feed()
    assert feed and {"time", "truth", "system", "outcome"} <= set(feed[0])
    with zipfile.ZipFile(io.BytesIO(polygon_view.dataset_zip())) as z:
        assert {"permits.csv", "truth.csv", "injected.csv", "README.md"} <= set(z.namelist())
    monkeypatch.setattr(polygon_view, "PROCESSED", PROCESSED)
    day = pd.Timestamp(feed[0]["time"]).strftime("%Y-%m-%d")
    journal = pd.read_csv(io.StringIO(polygon_view.journal_day(day)))
    assert list(journal.columns) == polygon_view.JOURNAL_COLUMNS
    injected = pd.read_parquet(polygon_view.folder() / "injected.parquet")
    own = injected[injected.ts.dt.strftime("%Y-%m-%d").eq(day)]
    assert set(own.event_id) <= set(journal["ид_события"])
    with pytest.raises(ValueError, match="июня"):
        polygon_view.journal_day("2026-03-01")


def test_nearest_channels_do_not_depend_on_row_order():
    """Several sensors share a picket; the pick must not depend on how rows or ties were ordered."""
    import numpy as np

    frame = pd.DataFrame({"channel_id": [30, 10, 20, 40, 50], "pk": [5.0, 5.0, 7.0, 5.0, 9.0]})
    picks = set()
    for seed in range(20):
        shuffled = frame.sample(frac=1, random_state=seed)
        chosen = polygon.nearest(shuffled, 5.0, np.random.default_rng(0), k=3)
        picks.add(tuple(chosen.channel_id))
    assert picks == {(10, 30, 40)}
