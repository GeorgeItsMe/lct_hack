import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.temporal_count_diagnostics import (
    late_confirmation_audit,
    paired,
    policy_lookup,
    warning_audit,
)


def calculate(hours, event_hours, alerts):
    start = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame({"object_id": 1, "as_of": start + pd.to_timedelta(hours, unit="h"), "alert": alerts})
    eps = pd.DataFrame({"object_id": 1, "start_ts": start + pd.to_timedelta(event_hours, unit="h")})
    evaluator = evaluator_for(rows, eps, 0.25, 3)
    expected = evaluator.evaluate(rows.alert, 0.5, 0.25)
    return warning_audit(rows, eps, expected, 0.25, 3)


def test_error_categories_preserve_all_true_false_and_missed_events():
    result = calculate([0, 1, 2, 30], [0.5, 12, 23], [1, 1, 1, 1])
    assert result["stats"]["true_alerts"] == 3
    assert result["stats"]["false_empty"] == 1
    assert result["stats"]["false_redundant"] == 0
    result = calculate([0, 1, 2, 30], [0.5, 12], [1, 1, 1, 1])
    assert result["stats"]["true_alerts"] == 2
    assert result["stats"]["false_empty"] == 1
    assert result["stats"]["false_redundant"] == 1


def test_duplicate_episode_starts_are_not_collapsed():
    result = calculate([0, 0.25, 0.5], [1, 1], [1, 1, 1])
    assert result["stats"]["eligible_episodes"] == 2
    assert result["stats"]["true_alerts"] == 2
    assert result["stats"]["false_redundant"] == 1
    assert set(result["matches"]) == {(1, 0), (1, 1)}


def test_paired_leads_keep_lost_events_and_reject_different_cohorts():
    a = calculate([0, 0.25, 0.5], [1, 3], [1, 0, 0])
    b = calculate([0, 0.25, 0.5], [1, 3], [0, 1, 1])
    compare = paired(a, b)
    assert compare["both_find"] == 1
    assert compare["candidate_only"] == 0
    assert compare["reference_only"] == 1
    assert compare["candidate_earlier"] == 1
    assert compare["median_paired_lead_difference_hours"] == 0.25
    with pytest.raises(ValueError, match="cohort"):
        paired(a, calculate([0, 0.25, 0.5], [1], [1, 0, 0]))


def test_policy_lookup_never_invents_missing_grid_results():
    chosen = {"capacity": 1, "margin": 0.5, "floor": 0.75}
    assert policy_lookup([{**chosen, "precision": 0.7}], chosen)["precision"] == 0.7
    assert policy_lookup([], {**chosen, "capacity": 0}) is None
    with pytest.raises(ValueError, match="missing"):
        policy_lookup([], chosen)
    with pytest.raises(ValueError, match="duplicated"):
        policy_lookup([chosen, chosen], chosen)


def test_warning_audit_rejects_changed_results_and_invalid_flags():
    rows = pd.DataFrame({"object_id": [1], "as_of": [pd.Timestamp("2025-01-01")], "alert": [np.nan]})
    eps = pd.DataFrame({"object_id": [1], "start_ts": [pd.Timestamp("2025-01-01 01:00")]})
    with pytest.raises(ValueError, match="binary"):
        warning_audit(rows, eps, {}, 0.25, 1)
    with pytest.raises(ValueError, match="archived"):
        warning_audit(rows.assign(alert=1), eps, {}, 0.25, 1)


def late(hours, event_hours, warning_hours):
    start = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame(
        {
            "object_id": 1,
            "as_of": start + pd.to_timedelta(hours, unit="h"),
            "alert": np.isin(hours, warning_hours).astype(float),
        }
    )
    eps = pd.DataFrame({"object_id": 1, "start_ts": start + pd.to_timedelta(event_hours, unit="h")})
    return late_confirmation_audit(rows, eps, 0.25, 3)


def test_late_confirmation_can_wrongly_clear_a_younger_warning():
    result = late([0, 2, 23, 24, 25], [23.5, 25.5], [0, 2, 25])
    totals = result["totals"]
    assert totals["confirmed_events"] == 1
    assert totals["wrong_warning_resolutions"] == 1
    assert totals["warnings_issued_while_under_reserved"] == 1
    assert totals["under_reserved_false_warnings"] == 1
    assert totals["under_reserved_true_warnings"] == 0
    assert result["ownership_differences"][0]["old_warning_at"] == "2025-01-01 02:00:00"
    assert result["ownership_differences"][0]["original_warning_at"] == "2025-01-01 00:00:00"


def test_a_real_gap_before_confirmation_preserves_legacy_resolution_first():
    result = late([0, 2, 23, 25], [23.5, 25.5], [0, 2])
    assert result["totals"]["wrong_warning_resolutions"] == 0
    assert result["totals"]["expired_original_warning_resolutions"] == 1


def test_expired_warning_can_lose_confirmation_without_a_younger_owner():
    result = late(np.arange(29), [23.5], [0])
    assert result["totals"]["lost_confirmations"] == 1
    assert result["totals"]["wrong_warning_resolutions"] == 0
    assert result["totals"]["confirmed_events_with_original_owner"] == 1


def test_future_unreleased_events_cannot_change_the_confirmed_prefix():
    a = late([0, 2, 23, 24, 25], [23.5, 25.5], [0, 2, 25])
    b = late([0, 2, 23, 24, 25], [23.5, 27.5, 50], [0, 2, 25])
    assert a["ownership_differences"] == b["ownership_differences"]
    # Retrospective truth of an issued warning may change with future events;
    # the causal ownership verification and known state must not.
    for key in (
        "confirmed_events",
        "wrong_warning_resolutions",
        "lost_confirmations",
        "warnings_issued_while_under_reserved",
    ):
        assert a["totals"][key] == b["totals"][key]


def test_retained_ledger_matches_duplicate_events_and_discards_safely_after_long_gap():
    result = late([0, 0.25, 1, 2, 3, 100], [1, 1, 99], [0, 0.25])
    assert result["confirmed_prefix_ownership_verified"]
    assert result["totals"]["confirmed_events_with_original_owner"] == 2
    assert result["totals"]["wrong_warning_resolutions"] == 0
    assert result["totals"]["lost_confirmations"] == 0
    delayed = late([0, 1, 2, 100], [23.5], [0])
    assert delayed["totals"]["confirmed_events_with_original_owner"] == 1
    assert delayed["totals"]["expired_original_warning_resolutions"] == 1
    assert delayed["totals"]["lost_confirmations"] == 0
