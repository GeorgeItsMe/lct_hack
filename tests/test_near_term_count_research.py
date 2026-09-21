import numpy as np
import pandas as pd
import pytest

from moscollector.count_research import episode_counts
from moscollector.near_term_count_research import horizon_counts


def test_short_count_uses_same_episodes_without_including_horizon_boundary():
    at = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame({"object_id": [1, 2], "as_of": [at, at]})
    episodes = pd.DataFrame(
        {"object_id": [1, 1, 1, 2], "start_ts": at + pd.to_timedelta([0, 6, 24, 5], unit="h")}
    )
    np.testing.assert_array_equal(horizon_counts(frame, episodes, 6), [1, 1])
    np.testing.assert_array_equal(horizon_counts(frame, episodes, 24), episode_counts(frame, episodes))
    assert (horizon_counts(frame, episodes, 6) <= episode_counts(frame, episodes)).all()
    with pytest.raises(ValueError):
        horizon_counts(frame, episodes, 0)
