"""Small inference helpers shared by the API and training code.

Keeping these helpers outside ``train`` prevents the web process from importing
scikit-learn merely to format CatBoost inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["object_id", "parent_id", "object_kind"]


def model_input(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in CATEGORICAL:
        result[column] = result[column].fillna("unknown").astype(str)
    for column in set(columns) - set(CATEGORICAL):
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return result


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def calibrated(raw, calibration):
    return sigmoid(np.asarray(raw) * calibration["slope"] + calibration["intercept"])
