"""Importable numeric transforms so persisted research pipelines remain loadable."""

import numpy as np


def signed_log(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))
