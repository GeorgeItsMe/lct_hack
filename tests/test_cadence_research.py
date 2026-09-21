import numpy as np
import pandas as pd
import pytest

from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort


def frame(objects, hours):
    return pd.DataFrame(
        {"object_id": objects, "as_of": pd.Timestamp("2026-01-01") + pd.to_timedelta(hours, unit="h")}
    )


def test_alignment_preserves_objects_boundaries_and_eligibility_gaps():
    reference = frame([1, 1, 1, 1, 2], [0, 3, 9, 12, 6]).sample(frac=1, random_state=3)
    dense = frame(np.repeat([1, 2, 3], 15), np.tile(np.arange(-1, 14), 3))
    dense = dense.sample(frac=1, random_state=2)
    result = align_opportunities(dense, reference)
    expected = frame([1] * 8 + [2], [0, 1, 2, 3, 9, 10, 11, 12, 6])
    keys = ["object_id", "as_of"]
    pd.testing.assert_frame_equal(
        result.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
    )
    # New interpolation points cannot change the union of [t,t+24h) windows.
    episodes = frame(np.repeat([1, 2, 3], 40), np.tile(np.arange(-1, 39), 3)).rename(
        columns={"as_of": "start_ts"}
    )
    assert_same_episode_cohort(result, reference, episodes)


def test_same_count_is_not_sufficient_for_episode_cohort_equality():
    reference = frame([1], [0])
    dense = frame([1], [24])
    episodes = frame([1, 1], [1, 25]).rename(columns={"as_of": "start_ts"})
    with pytest.raises(ValueError, match="different eligible episodes"):
        assert_same_episode_cohort(dense, reference, episodes)
