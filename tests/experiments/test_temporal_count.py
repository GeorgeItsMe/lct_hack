import numpy as np
import pandas as pd
import pytest

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.experiments.fine_cadence_research import FinePendingSimulator
from moscollector.experiments.temporal_count import (
    RATES,
    DeadlineSimulator,
    bin_counts,
    reserved_mass,
    select_deadline,
    training_table,
)
from moscollector.train import CATEGORICAL


def example():
    times = pd.date_range("2025-01-01", periods=220, freq="15min")
    rows = pd.DataFrame({"object_id": 1, "as_of": times})
    eps = pd.DataFrame(
        {"object_id": [1] * 5, "start_ts": times[0] + pd.to_timedelta([0, 6, 12, 18, 24], unit="h")}
    )
    return rows, eps


def test_four_targets_cover_original_window_without_boundary_duplicates():
    rows, eps = example()
    queries = pd.concat([rows.iloc[[0, 1]], rows.iloc[[0]].assign(object_id=2)], ignore_index=True)
    actual = bin_counts(queries, eps)
    np.testing.assert_array_equal(actual, [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    later = pd.concat([eps, eps.iloc[[0]].assign(start_ts=pd.Timestamp("2026-01-01"))])
    np.testing.assert_array_equal(bin_counts(queries, later), actual)


def test_bin_targets_match_direct_sets_for_shuffled_objects_and_times():
    rows, eps = example()
    rows = pd.concat([rows, rows.assign(object_id=2)], ignore_index=True).sample(frac=1, random_state=4)
    eps = pd.concat(
        [
            eps,
            eps.assign(object_id=2, start_ts=pd.to_datetime(eps.start_ts.astype("int64") + 19 * 60 * 10**9)),
        ]
    )
    direct = [
        [
            sum(
                row.as_of.value + 6 * b * HOUR_NS <= t.value < row.as_of.value + 6 * (b + 1) * HOUR_NS
                for t in eps.loc[eps.object_id.eq(row.object_id), "start_ts"]
            )
            for b in range(4)
        ]
        for row in rows.itertuples()
    ]
    np.testing.assert_array_equal(bin_counts(rows, eps), direct)


def test_long_training_keeps_negatives_original_inputs_and_anchor_mass():
    rows = pd.DataFrame({c: ["a", "b"] for c in CATEGORICAL}).assign(value=[3.0, 5.0])
    original = rows.copy(deep=True)
    labels = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
    x, y, weight = training_table(rows, list(rows), labels)
    pd.testing.assert_frame_equal(rows, original)
    assert len(x) == 8
    np.testing.assert_array_equal(y, [0, 1, 0, 2, 0, 3, 0, 4])
    np.testing.assert_array_equal(x.forecast_bin, [0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(weight.reshape(4, 2).sum(axis=0), [1, 1])
    with pytest.raises(ValueError):
        training_table(rows, list(rows), -labels)


def test_deadline_mass_accounts_for_late_events_and_preserves_bounds():
    assert reserved_mass([0, 0, 0, 1], [1, 2]) == 0
    assert reserved_mass([1, 0, 0, 0], [6, 12]) == 1
    assert reserved_mass([3, 0, 0, 0], [6, 12]) == 2
    assert reserved_mass([0.25] * 4, [6]) == 0.25
    assert reserved_mass([1, 2, 3, 4], []) == 0
    assert reserved_mass([1, 2, 3, 4], [24]) == 1
    for profile, deadlines in [([-1, 0, 0, 0], [1]), ([1] * 4, [25]), ([np.nan] * 4, [1])]:
        with pytest.raises(ValueError):
            reserved_mass(profile, deadlines)


def direct_alerts(rows, eps, profiles, probability, multiplier, margin, floor):
    """Slow reference visits EVERY original tick; no pruning shortcuts."""
    result = np.zeros(len(rows))
    for obj, group in rows.reset_index(drop=True).groupby("object_id"):
        events = sorted(pd.Timestamp(t).value for t in eps.loc[eps.object_id.eq(obj), "start_ts"])
        pending = []
        seen = set()
        for row in group.sort_values("as_of").itertuples():
            now = row.as_of.value
            for j, event in enumerate(events):
                release = ((event + 70 * 60 * 10**9) // HOUR_NS + 1) * HOUR_NS
                if release <= now and j not in seen:
                    seen.add(j)
                    for k, issued in enumerate(pending):
                        if issued <= event < issued + 24 * HOUR_NS:
                            pending.pop(k)
                            break
            pending = [issued for issued in pending if now - issued < 24 * HOUR_NS]
            profile = profiles[row.Index] * multiplier
            reserve = reserved_mass(profile, [(issued + 24 * HOUR_NS - now) / HOUR_NS for issued in pending])
            if probability[row.Index] >= floor and profiles[row.Index].sum() * multiplier - reserve >= margin:
                result[row.Index] = 1
                pending.append(now)
    return result


@pytest.mark.parametrize("seed", [8, 31, 57])
def test_pruned_simulator_matches_full_ticks_with_gaps_delays_and_multiple_objects(seed):
    rng = np.random.default_rng(seed)
    rows, eps = example()
    rows = pd.concat([rows, rows.assign(object_id=2)], ignore_index=True)
    rows = rows.loc[rng.random(len(rows)) > 0.13].sample(frac=1, random_state=seed).reset_index(drop=True)
    extra = pd.DataFrame(
        {
            "object_id": rng.integers(1, 3, 30),
            "start_ts": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(-30, 80, 30), unit="h"),
        }
    )
    eps = pd.concat([eps, extra], ignore_index=True)
    profiles = rng.gamma(0.3, 1.2, (len(rows), 4))
    probability = rng.random(len(rows))
    for multiplier, margin, floor in [(0, 1, 0), (0.5, 0.25, 0), (1, 1, 0.75), (2, 3, 0.95)]:
        actual = DeadlineSimulator(rows, eps).alerts(profiles, probability, multiplier, margin, floor)
        expected = direct_alerts(rows, eps, profiles, probability, multiplier, margin, floor)
        np.testing.assert_array_equal(actual, expected)


def test_future_outcomes_cannot_change_unreleased_prefix():
    rows, eps = example()
    altered = pd.concat(
        [eps, pd.DataFrame({"object_id": [1], "start_ts": [pd.Timestamp("2025-01-02 01:12")]})]
    )
    profiles = np.tile([0.2, 0.1, 0.3, 0.4], (len(rows), 1))
    a = DeadlineSimulator(rows, eps).alerts(profiles, np.ones(len(rows)), 1, 0.5, 0)
    b = DeadlineSimulator(rows, altered).alerts(profiles, np.ones(len(rows)), 1, 0.5, 0)
    prefix = rows.as_of.lt(pd.Timestamp("2025-01-02 03:00"))
    np.testing.assert_array_equal(a[prefix], b[prefix])


def test_early_mass_reproduces_full_pending_when_every_deadline_covers_it():
    rows, eps = example()
    rows = rows.iloc[:20]
    profiles = np.tile([8.0, 0, 0, 0], (len(rows), 1))
    p = np.ones(len(rows))
    actual = DeadlineSimulator(rows, eps).alerts(profiles, p, 1, 1, 0)
    old = FinePendingSimulator(rows, eps).alerts(profiles.sum(axis=1), p, 1, 0)
    np.testing.assert_array_equal(actual, old)


def test_invalid_probabilities_profiles_or_duplicate_slots_are_rejected():
    rows, eps = example()
    sim = DeadlineSimulator(rows, eps)
    with pytest.raises(ValueError):
        sim.alerts(np.ones((len(rows), 4)), np.full(len(rows), 2), 1, 1, 0)
    with pytest.raises(ValueError):
        sim.alerts(np.ones((len(rows), 3)), np.ones(len(rows)), 1, 1, 0)
    with pytest.raises(ValueError):
        DeadlineSimulator(pd.concat([rows, rows.iloc[[0]]]), eps)


@pytest.mark.parametrize("with_expiry_tick", [True, False])
def test_skipped_expiry_ticks_preserve_original_confirmation_order(with_expiry_tick):
    hours = [0, 2, 23, 24, 25] if with_expiry_tick else [0, 2, 23, 25]
    rows = pd.DataFrame(
        {"object_id": 1, "as_of": pd.Timestamp("2025-01-01") + pd.to_timedelta(hours, unit="h")}
    )
    eps = pd.DataFrame({"object_id": [1], "start_ts": [pd.Timestamp("2025-01-01 23:30")]})
    profiles = np.zeros((len(rows), 4))
    profiles[:, 0] = 1.1
    profiles[1, 0] = 3
    p = np.array([float(h in (0, 2, 25)) for h in hours])
    values = DeadlineSimulator(rows, eps).alerts(profiles, p, 1, 1, 0.5)
    np.testing.assert_array_equal(values, direct_alerts(rows, eps, profiles, p, 1, 1, 0.5))
    assert values[-1] == int(with_expiry_tick)


def test_training_rejects_future_label_as_a_feature():
    rows = pd.DataFrame({c: ["x"] for c in CATEGORICAL}).assign(target_fire=1)
    with pytest.raises(ValueError, match="schema"):
        training_table(rows, list(rows), np.ones((1, 4)))


def test_all_policy_options_equal_direct_simulation_and_keep_original_exposure():
    from moscollector.experiments.fine_cadence_research import evaluator_for

    rows, eps = example()
    rows = rows.iloc[:8].copy()
    rows[RATES] = np.tile([1.0, 0.5, 0.25, 0.1], (len(rows), 1))
    rows["probability"] = np.linspace(0, 1, len(rows))
    exposure = 2 / 24
    evaluator = evaluator_for(rows, eps, 0.25, exposure)
    chosen, frontier = select_deadline(rows, eps, 0.25, exposure)
    assert len(frontier) == 456
    for entry in frontier:
        values = direct_alerts(
            rows,
            eps,
            rows[RATES].to_numpy(),
            rows.probability.to_numpy(),
            entry["capacity"],
            entry["margin"],
            entry["floor"],
        )
        expected = evaluator.evaluate(values, 0.5, 0.25)
        assert all(entry[k] == v for k, v in expected.items())
    assert chosen["false_alerts_per_object_day"] <= 0.25
    with pytest.raises(ValueError, match="probabilities"):
        select_deadline(rows.assign(probability=np.nan), eps, 0.25, exposure)
