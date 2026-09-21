"""Causal report-volume context excluding the forecast object and its own group."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

SIGNALS = (
    "events",
    "alarms",
    "fault_reports",
    "power_reports",
    "unknown_reports",
    "smoke_reports",
    "access_reports",
    "technical_codes",
)
WINDOWS = (1, 6, 24)
SCOPES = ("parent", "other_groups")
COLUMNS = [f"peer_{scope}_{signal}_{hours}h" for hours in WINDOWS for signal in SIGNALS for scope in SCOPES]
COLUMNS += [f"peer_{scope}_{signal}_active_objects_6h" for signal in SIGNALS for scope in SCOPES]
FOLDER = PROCESSED / "peer-context-v30"


def peer_context(frame):
    """Use only same-time strictly-past count summaries, never eligibility/labels.

    Parent scope excludes self. Other-groups scope excludes the entire parent.
    No observation-count denominator is inferred from future grid availability.
    Missing peer windows propagate as missing; zero reports do not mean healthy.
    """
    keys = frame[["object_id", "parent_id", "as_of"]].reset_index(drop=True)
    if keys.isna().any().any() or keys.duplicated(["object_id", "as_of"]).any():
        raise ValueError("Missing or duplicate peer identities/timestamps")
    if keys.groupby("object_id").parent_id.nunique().gt(1).any():
        raise ValueError("Changing parent membership requires an explicit historical catalog")
    local_keys, global_keys = [keys.as_of, keys.parent_id], keys.as_of
    extra = {}
    for hours in WINDOWS:
        columns = [f"{signal}_{hours}h" for signal in SIGNALS]
        raw = frame[columns].reset_index(drop=True).astype(np.float64)
        values = raw.to_numpy()
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
            raise ValueError("Invalid past report counts")
        clean, missing = raw.fillna(0), raw.isna().astype(np.int32)
        local = clean.groupby(local_keys, sort=False).transform("sum")
        total = clean.groupby(global_keys, sort=False).transform("sum")
        local_missing = missing.groupby(local_keys, sort=False).transform("sum")
        total_missing = missing.groupby(global_keys, sort=False).transform("sum")
        summaries = {
            "parent": (local - clean).mask((local_missing - missing).gt(0)),
            "other_groups": (total - local).mask((total_missing - local_missing).gt(0)),
        }
        for signal, column in zip(SIGNALS, columns, strict=True):
            for scope in SCOPES:
                extra[f"peer_{scope}_{signal}_{hours}h"] = summaries[scope][column].to_numpy()
        if hours == 6:
            active = raw.gt(0).astype(np.int32)
            local_active = active.groupby(local_keys, sort=False).transform("sum")
            total_active = active.groupby(global_keys, sort=False).transform("sum")
            active_summaries = {
                "parent": (local_active - active).astype(float).mask((local_missing - missing).gt(0)),
                "other_groups": (total_active - local_active)
                .astype(float)
                .mask((total_missing - local_missing).gt(0)),
            }
            for signal, column in zip(SIGNALS, columns, strict=True):
                for scope in SCOPES:
                    extra[f"peer_{scope}_{signal}_active_objects_6h"] = active_summaries[scope][
                        column
                    ].to_numpy()
    result = pd.DataFrame(extra, index=frame.index)[COLUMNS]
    if np.isinf(result.to_numpy()).any() or result.lt(0).any().any():
        raise ValueError("Invalid peer aggregation")
    return result


def build(folder=FOLDER):
    inputs = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
    ]
    code = Path(__file__)
    signature = {
        "inputs": {str(p): sha256(p) for p in inputs},
        "code_hashes": {str(code): sha256(code)},
        "columns": COLUMNS,
        "definition": "64 same-time cross-object raw report features:8 signals x1/6/24h x2 disjoint scopes, plus8 active-object counts over6h x2 scopes. Parent excludes self; other_groups excludes own parent. Uses all original rows before eligibility masking. Missing peer windows remain missing; absent raw reports are not health labels. Static original catalog membership only, no historical inventory claim.",
        "difference_from_earlier_context": "V4 classifier used same-parent24/168h rates inside a larger feature bundle; v17 had ordered neighboring transitions. This study separately adds short report volume, active-object counts and outside-parent activity to current count heads.",
    }
    manifest = folder / "build.json"
    if manifest.exists():
        before = read(manifest)
        if any(before[k] != v for k, v in signature.items()):
            raise ValueError("Peer feature inputs/code changed; use another folder")
        for path, digest in before["outputs"].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed peer feature output: {path}")
        return before
    folder.mkdir(parents=True, exist_ok=True)
    outputs, tables, summaries = {}, [], {}
    for label, source in zip(("original", "dense"), inputs, strict=True):
        frame = pd.read_parquet(source)
        if not frame.as_of.lt(pd.Timestamp("2026-06-01")).all():
            raise ValueError("Peer inputs include post-May rows")
        extra = peer_context(frame)
        enriched = pd.concat([frame, extra], axis=1)
        pd.testing.assert_frame_equal(enriched[frame.columns], frame)
        output = folder / f"{label}.parquet"
        enriched.to_parquet(output, index=False, compression="zstd")
        outputs[str(output)] = sha256(output)
        tables.append(enriched[["object_id", "as_of", *COLUMNS]])
        summaries[label] = {
            "rows": len(frame),
            "all_original_columns_unchanged": True,
            "rows_with_any_missing_peer_window": int(extra.isna().any(axis=1).sum()),
            "features": len(COLUMNS),
        }
        print("BUILT peer", label, summaries[label], flush=True)
        del frame, enriched, extra
    keys = ["object_id", "as_of"]
    shared = tables[0].merge(tables[1], on=keys, suffixes=("_original", "_dense"), validate="one_to_one")
    np.testing.assert_allclose(
        shared[[f"{c}_original" for c in COLUMNS]],
        shared[[f"{c}_dense" for c in COLUMNS]],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    result = {**signature, "outputs": outputs, "tables": summaries, "exact_shared_rows": len(shared)}
    write_json(manifest, result)
    print("PEER shared-row parity", len(shared), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    build(parser.parse_args().folder)
