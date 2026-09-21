"""Exact offline/incremental parity, including serialization across a restart."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.count_research import pending_alerts
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.warning_policy import PendingWarningState, replay_pending

episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
report = {"scope": "exact_implementation_parity_not_new_validation", "cases": []}
for path in sorted(Path("artifacts/research-v9").glob("*/*.json")):
    meta = json.loads(path.read_text())
    if "policies" not in meta:
        continue
    eps = episodes[episodes.kind.eq(meta["kind"])]
    policy = meta["policies"]["pending"]
    for period in ("policy", "test"):
        source = path.with_name(f"{meta['kind']}-{period}.parquet")
        pred = pd.read_parquet(source)
        expected = pending_alerts(pred, eps, policy["margin"], policy["probability_floor"])
        if "pending_alert" in pred:
            np.testing.assert_array_equal(expected, pred.pending_alert)
        middle = sorted(pred.as_of.unique())[pred.as_of.nunique() // 2]
        first, second = pred[pred.as_of.lt(middle)], pred[pred.as_of.ge(middle)]
        a, states = replay_pending(first, eps, policy["margin"], policy["probability_floor"])
        states = {
            k: PendingWarningState.from_dict(json.loads(json.dumps(v.to_dict()))) for k, v in states.items()
        }
        b, _ = replay_pending(second, eps, policy["margin"], policy["probability_floor"], states)
        actual = pd.Series(np.r_[a, b], index=list(first.index) + list(second.index)).sort_index().to_numpy()
        np.testing.assert_array_equal(actual, expected)
        report["cases"].append(
            {
                "source": str(source),
                "sha256": sha256(source),
                "rows": len(pred),
                "alerts": int(actual.sum()),
                "exact_parity_after_restart": True,
            }
        )
        print(source, len(pred), int(actual.sum()), "exact", flush=True)
report["total_rows"] = sum(r["rows"] for r in report["cases"])
write_json(Path("artifacts/warning_policy_parity.json"), report)
