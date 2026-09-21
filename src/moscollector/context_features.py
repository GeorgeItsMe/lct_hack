"""Causal, snapshot-local context features for the second research round.

Input rows already contain strictly historical summaries. No labels, later rows,
or fitted statistics enter this transformation. Neighbor summaries use only
objects from the same parent at exactly the same forecast time.
"""

import numpy as np
import pandas as pd


def enrich_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(["as_of", "object_id"]).any():
        raise ValueError("Context requires one row per object and forecast time")
    extra = {}
    channel_count = frame.channel_count.clip(lower=1)
    groups = frame.groupby(["as_of", "parent_id"], sort=False)
    neighbor_channels = (groups.channel_count.transform("sum") - frame.channel_count).clip(lower=1)
    for signal in (
        "fault_reports",
        "power_reports",
        "unknown_reports",
        "smoke_reports",
        "flood_reports",
        "access_reports",
        "ambiguous",
        "technical_codes",
    ):
        for window in (6, 24, 168):
            column = f"{signal}_{window}h"
            extra[f"ctx_{signal}_per_channel_{window}h"] = frame[column] / channel_count
        extra[f"ctx_{signal}_share_24h"] = frame[f"{signal}_24h"] / (1 + frame.events_24h)
        extra[f"ctx_{signal}_acceleration"] = frame[f"{signal}_6h"] / (
            1 + (frame[f"{signal}_24h"] - frame[f"{signal}_6h"]).clip(lower=0) / 3
        )
        for window in (24, 168):
            column = f"{signal}_{window}h"
            neighbor = (groups[column].transform("sum") - frame[column]).clip(lower=0)
            extra[f"ctx_neighbor_{signal}_{window}h"] = neighbor / neighbor_channels
    for kind in ("fault", "fire", "flood", "access"):
        # Exponentially decaying recurrence signals inspired by point processes;
        # these are engineered features, not a fitted Hawkes process.
        recency = frame[f"past_{kind}_recency_h"]
        for hours in (24, 168, 720):
            extra[f"ctx_{kind}_decay_{hours}h"] = np.exp(-recency / hours)
        extra[f"ctx_{kind}_recurrence_growth"] = frame[f"past_{kind}_episodes_168h"] / (
            1
            + (frame[f"past_{kind}_episodes_720h"] - frame[f"past_{kind}_episodes_168h"]).clip(lower=0)
            * 168
            / (720 - 168)
        )
        for hours in (168, 720):
            column = f"past_{kind}_episodes_{hours}h"
            extra[f"ctx_neighbor_{kind}_episodes_{hours}h"] = (
                groups[column].transform("sum") - frame[column]
            ).clip(lower=0)
    for column, period in (("hour", 24), ("day_of_week", 7), ("month", 12)):
        extra[f"ctx_{column}_sin"] = np.sin(2 * np.pi * frame[column] / period)
        extra[f"ctx_{column}_cos"] = np.cos(2 * np.pi * frame[column] / period)
    extra["ctx_reporting_fraction"] = frame.reporting_channels_24h / channel_count
    return pd.concat([frame, pd.DataFrame(extra, index=frame.index).astype(np.float32)], axis=1)
