"""Checks against the committed demo runtime bundle (real archive slice, frozen models)."""

from pathlib import Path

import pytest

from moscollector import service as service_module

BUNDLE = Path(__file__).resolve().parents[1] / "vercel_runtime"


@pytest.fixture(scope="module")
def service():
    if not (BUNDLE / "artifacts" / "evaluation_report.json").exists():
        pytest.skip("Runtime bundle is not present")
    patch = pytest.MonkeyPatch()
    patch.setattr(service_module, "ARTIFACTS", BUNDLE / "artifacts")
    patch.setattr(service_module, "PROCESSED", BUNDLE / "data" / "processed")
    try:
        yield service_module.AnalyticsService()
    finally:
        patch.undo()


@pytest.mark.parametrize("kind", ["fault", "fire", "access"])
def test_threshold_preview_reproduces_the_stored_policy_period(service, kind):
    stored = service.meta[kind]["policy_period"]
    preview = service.threshold_preview(kind, service.meta[kind]["threshold"])["current"]
    for key in ("alerts", "true_alerts", "eligible_episodes", "precision", "recall"):
        assert preview[key] == pytest.approx(stored[key]), key


def test_higher_threshold_never_issues_more_warnings(service):
    low = service.threshold_preview("access", 0.2)["proposed"]
    high = service.threshold_preview("access", 0.5)["proposed"]
    assert high["alerts"] <= low["alerts"] and high["warnings_now"] <= low["warnings_now"]


def test_summary_and_equipment_are_consistent_with_forecasts(service):
    forecasts = service.forecast_rows(None)
    summary = service.summary(None)
    assert summary["totals"]["warnings"] == sum(r["above_threshold"] for r in forecasts)
    assert sum(n["warnings"] for n in summary["nodes"]) == summary["totals"]["warnings"]
    assert {s["year"] for s in summary["seasonality"]} >= {2019, 2026}
    equipment = service.equipment(None)
    assert len(equipment["objects"]) == sum(r["kind"] == "fault" for r in forecasts)
    assert all(c["ts"] < forecasts[0]["as_of"] for o in equipment["objects"] for c in o["channels"])


def test_card_context_uses_only_the_past(service):
    detail = service.detail(3215, "access")
    assert detail["calendar"]["object_last_episode"] < detail["as_of"]
    assert {s["id"] for s in detail["verification"]["external_sources"]} == {
        "cameras",
        "planned_works",
        "access_permits",
    }
    assert detail["recommendation_details"][-1]["source"] == "general"
