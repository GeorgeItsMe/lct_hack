from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from moscollector.database import make_database
from moscollector.warning_storage import decide_stream_warnings


def data():
    return (
        [{"object_id": 1, "kind": "access", "probability": 0.9, "expected_episodes": 0.9}],
        pd.DataFrame({"object_id": [], "kind": [], "start_ts": pd.to_datetime([])}),
        {"access": {"type": "pending_count", "margin": 0.75, "probability_floor": 0.5}},
    )


def test_decisions_and_pending_state_survive_restart_and_retry(tmp_path):
    url = f"sqlite:///{tmp_path / 'warnings.db'}"
    engine, factory = make_database(url)
    f, eps, policies = data()

    def call(key, at):
        return decide_stream_warnings(factory, key, at, "v1", f, eps, policies)

    first = call("a", "2026-01-01T00:00:00")
    assert first[0]["notification_due"]
    assert call("a", "2026-01-01T00:00:00") == first
    engine.dispose()
    engine, factory = make_database(url)
    assert call("a", "2026-01-01T00:00:00") == first
    assert not call("b", "2026-01-01T00:00:00")[0]["notification_due"]
    assert not call("c", "2026-01-01T00:05:00")[0]["notification_due"]
    assert not call("d", "2026-01-01T01:00:00")[0]["notification_due"]
    eps = pd.DataFrame(
        {"object_id": [1], "kind": ["access"], "start_ts": pd.to_datetime(["2026-01-01T00:30:00"])}
    )
    assert call("e", "2026-01-01T02:00:00")[0]["notification_due"]
    assert not call("f", "2026-01-01T03:00:00")[0]["notification_due"]
    with pytest.raises(ValueError, match="identity changed"):
        call("a", "2026-01-02T00:00:00")
    engine.dispose()


def test_concurrent_snapshots_issue_once_and_rollback_on_error(tmp_path):
    engine, factory = make_database(f"sqlite:///{tmp_path / 'concurrent.db'}")
    f, eps, policies = data()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda key: decide_stream_warnings(factory, key, "2026-01-01", "v1", f, eps, policies),
                ["a", "b"],
            )
        )
    assert sum(r[0]["notification_due"] for r in results) == 1
    bad = f + [{"object_id": 2, "kind": "access", "probability": float("nan"), "expected_episodes": 1}]
    with pytest.raises(ValueError, match="Non-finite"):
        decide_stream_warnings(factory, "failed", "2026-01-02T01:00:00", "v1", bad, eps, policies)
    result = decide_stream_warnings(factory, "good", "2026-01-02T01:00:00", "v1", f, eps, policies)
    assert result[0]["notification_due"]
    engine.dispose()
