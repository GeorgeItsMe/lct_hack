"""Headline numbers in the docs match the machine reports they are taken from."""

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    name: (ROOT / name).read_text(encoding="utf-8")
    for name in (
        "README.md",
        "docs/MODEL_CARD.md",
        "docs/ACCEPTANCE.md",
        "docs/POLYGON.md",
        "docs/PRESENTATION_CONTENT.md",
    )
}
RULE = json.loads((ROOT / "artifacts" / "daily_rule_evaluation.json").read_text(encoding="utf-8"))
POLYGON = json.loads((ROOT / "artifacts" / "polygon_report.json").read_text(encoding="utf-8"))


def pct1(v):
    return f"{v * 100:.1f}%".replace(".", ",")


def pct0(v):
    return f"{round(v * 100)}%"


def dec(v, digits=3):
    return f"{v:.{digits}f}".replace(".", ",")


def half_up(v, digits=1):
    """The interface rounds with Math.round, so the docs quote the same rounding."""
    return f"{math.floor(v * 10**digits + 0.5) / 10**digits:.{digits}f}".replace(".", ",")


def hours(v):
    return f"{v:.1f}".replace(".", ",") + " ч"


@pytest.mark.parametrize(
    "kind,label,bold",
    [
        ("access", "Охранный сигнал", True),
        ("fire", "Пожарный сигнал", True),
        ("fault", "Неисправность", False),
    ],
)
def test_june_rule_metrics_in_every_document(kind, label, bold):
    june = RULE["kinds"][kind]["june"]
    p, r = june["precision"], june["recall"]
    cell = (lambda v: f"**{pct1(v)}**") if bold else pct1
    assert f"{cell(p)} | {cell(r)}" in DOCS["README.md"]
    assert f"{dec(p)} | {dec(r)} | {dec(june['recall_strict'])}" in DOCS["docs/MODEL_CARD.md"]
    acceptance = DOCS["docs/ACCEPTANCE.md"]
    assert f"{dec(p)} / {dec(r)}" in acceptance or f"P {dec(p)} / R {dec(r)}" in acceptance
    assert f"| {pct0(p)} | {pct0(r)} |" in DOCS["docs/PRESENTATION_CONTENT.md"]


def test_selection_and_bundle_are_the_released_ones():
    access = RULE["kinds"]["access"]["selection"]
    assert f"{dec(access['precision'])} при полноте {dec(access['recall'])}" in DOCS["README.md"]
    for name in ("README.md", "docs/MODEL_CARD.md", "docs/ACCEPTANCE.md"):
        assert f"`{RULE['version']}`" in DOCS[name], name


def test_polygon_results_in_readme_and_polygon_doc():
    s = POLYGON["scenarios"]
    pump, fan, fire, gas = s["pump_forecast"], s["fan_forecast"], s["fire_forecast"], s["gas_forecast"]
    expected = [
        f"{s['fire_triage']['caught']} из {s['fire_triage']['incidents']} возгораний",
        f"{s['access_triage']['caught']} из {s['access_triage']['incidents']} проникновений",
        f"{fire['caught_incident_days']} из {fire['incident_days']} заранее",
        f"{pump['caught_incident_days']} из {pump['incident_days']}",
        f"{fan['caught_incident_days']} из {fan['incident_days']}, за {hours(fan['median_lead_hours'])}",
        f"точность {pct0(pump['precision'])}",
        f"точность {dec(fan['precision'] * 100, 1)}%",
        f"{gas['fan_caused']['caught_incident_days']} из {gas['fan_caused']['incident_days']}",
        f"за {hours(gas['fan_caused']['median_lead_hours'])}",
        f"{s['cascades']['caught']} из {s['cascades']['incidents']}",
    ]
    for text in expected:
        assert text in DOCS["README.md"], text
        assert text in DOCS["docs/POLYGON.md"] or text.endswith("проникновений"), text
    assert f"за {hours(fire['median_lead_hours'])}" in DOCS["README.md"]
    assert f"за {hours(pump['median_lead_hours'])}" in DOCS["README.md"]


def test_shift_averages_in_the_talk():
    from moscollector import service as service_module

    bundle = ROOT / "vercel_runtime"
    patch = pytest.MonkeyPatch()
    patch.setattr(service_module, "ARTIFACTS", bundle / "artifacts")
    patch.setattr(service_module, "PROCESSED", bundle / "data" / "processed")
    try:
        average = service_module.AnalyticsService().shift()["average"]["per_shift"]
    finally:
        patch.undo()
    text = (
        f"в среднем {half_up(average['warnings'])} предупреждения за смену, из них {half_up(average['false'])} "
        f"ложных; заранее пойманы {half_up(average['caught'])} из {half_up(average['incident_days'])}"
    )
    assert text in DOCS["docs/PRESENTATION_CONTENT.md"]
