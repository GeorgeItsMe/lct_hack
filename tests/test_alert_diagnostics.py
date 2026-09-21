import numpy as np
import pandas as pd
import pytest

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.train import alert_metrics


@pytest.mark.parametrize("cooldown", [3, 6, 12, 24])
def test_fast_evaluator_matches_frozen_event_protocol(cooldown):
    rng = np.random.default_rng(51)
    times = pd.date_range("2026-01-01", periods=160, freq="3h")
    pred = pd.DataFrame(
        {
            "object_id": np.repeat([1, 2], len(times)),
            "as_of": np.tile(times, 2),
            "probability": rng.random(2 * len(times)),
        }
    )
    event_times = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(-50, 30000, 180), unit="m")
    eps = pd.DataFrame(
        {
            "object_id": rng.choice([1, 2, 3], 180),
            "start_ts": event_times,
            "episode_id": [f"e{i}" for i in range(180)],
        }
    )
    # Include both boundary cases and simultaneous distinct episodes.
    eps.loc[0, ["object_id", "start_ts"]] = [1, times[0]]
    eps.loc[1, ["object_id", "start_ts"]] = [1, times[0] + pd.Timedelta(hours=24)]
    pred = pred.sample(frac=1, random_state=4)
    evaluator = EventEvaluator(pred, eps)
    for threshold in [0.1, 0.5, 0.9, 1.01]:
        reference = alert_metrics(pred, eps, threshold, cooldown)
        actual = evaluator.evaluate(pred.probability.to_numpy(), threshold, cooldown)
        for key in actual:
            assert (
                actual[key] == pytest.approx(reference[key])
                if isinstance(actual[key], float)
                else actual[key] == reference[key]
            )


def test_oracle_is_clearly_separate_and_has_no_false_alarms():
    pred = pd.DataFrame({"object_id": [1] * 20, "as_of": pd.date_range("2026-01-01", periods=20, freq="3h")})
    eps = pd.DataFrame(
        {
            "object_id": [1] * 3,
            "start_ts": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-02 12:00"]),
        }
    )
    evaluator = EventEvaluator(pred, eps)
    assert evaluator.clairvoyant_schedule(24)["true_alerts"] == 2
    assert evaluator.clairvoyant_schedule(3)["true_alerts"] == 3
    assert evaluator.clairvoyant_schedule(3)["false_alerts"] == 0
