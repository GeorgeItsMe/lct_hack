"""Checks against the committed demo runtime bundle (real archive slice, frozen models)."""

from pathlib import Path

import pandas as pd
import pytest

from moscollector import service as service_module
from moscollector.service import prediction_id

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
def test_threshold_preview_reproduces_the_released_selection_period(service, kind):
    """The preview replays the scores the active 24-hour-rule bundle was selected on."""
    stored = service.rule[kind]["selection"]
    preview = service.threshold_preview(kind, service.rule[kind]["threshold"])["current"]
    assert preview["threshold"] == service.thresholds()[kind] == service.rule[kind]["threshold"]
    for key in ("precision", "recall", "recall_strict", "alerts_per_day", "false_alerts_per_day"):
        assert preview[key] == pytest.approx(stored[key]), key


def test_archive_queue_issues_at_most_one_warning_per_object_and_kind_a_day(service):
    """Warnings keep their id for 24 hours, so a decision stays with the warning."""
    times = [t for t in service.times if pd.Timestamp("2026-06-14") <= t < pd.Timestamp("2026-06-17")]
    seen = {}
    for t in times:
        for row in service.forecast_rows(t):
            if not row["above_threshold"]:
                assert row["warning_issued_at"] is None
                continue
            issued = pd.Timestamp(row["warning_issued_at"])
            assert issued <= t < issued + pd.Timedelta(hours=24)
            assert row["id"] == prediction_id(row["object_id"], row["kind"], issued)
            assert row["valid_until"] == (issued + pd.Timedelta(hours=24)).isoformat()
            seen.setdefault((row["object_id"], row["kind"]), set()).add(issued)
    assert seen, "the replay window should contain warnings"
    for issued in seen.values():
        ordered = sorted(issued)
        assert all(b - a >= pd.Timedelta(hours=24) for a, b in zip(ordered, ordered[1:], strict=False))


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


def test_management_reports_render_in_both_formats(service):
    from io import BytesIO

    from openpyxl import load_workbook

    from moscollector.reports import build_report, to_pdf, to_xlsx

    summary = service.summary(None)
    forecasts = service.forecast_rows(None)
    decision = {
        "object_name": "объект Йота",
        "kind": "access",
        "forecast_at": summary["as_of"],
        "action": "false_alarm",
        "action_label": "Ложное срабатывание",
        "reason": "false_signal",
        "comment": "Проверено по камерам",
        "work_order": None,
    }
    report = build_report(summary, forecasts, [decision], [])
    assert report["workload"]["false_alarm_share"] == 1
    book = load_workbook(BytesIO(to_xlsx(report)))
    assert book.sheetnames[:3] == ["Сводка", "По узлам", "Предупреждения"]
    assert book["Предупреждения"].max_row - 1 == summary["totals"]["warnings"]
    pdf = to_pdf(report)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
