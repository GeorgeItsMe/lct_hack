"""Separate weighted Poisson error into count shape and a global scale error.

This is a validation diagnostic, not an event precision/recall metric. The
optimal scale is derived on these targets and must not be used on this same
period to claim independently calibrated forecast quality.
"""

import numpy as np


def profile_count_loss(raw, target, weight):
    raw, target, weight = (np.asarray(x, dtype=np.float64) for x in (raw, target, weight))
    if (
        raw.ndim != 1
        or not len(raw)
        or target.shape != raw.shape
        or weight.shape != raw.shape
        or any(not np.isfinite(x).all() for x in (raw, target, weight))
        or np.any(target < 0)
        or np.any(weight <= 0)
    ):
        raise ValueError("Invalid weighted count validation inputs")
    observed = float(np.average(target, weights=weight))
    maximum = float(raw.max())
    log_mean = maximum + float(np.log(np.average(np.exp(raw - maximum), weights=weight)))
    if log_mean >= np.log(np.finfo(np.float64).max):
        raise ValueError("Predicted mean is not representable")
    predicted = float(np.exp(log_mean))
    dot = float(np.average(target * raw, weights=weight))
    raw_loss = predicted - dot
    if observed == 0:
        return {
            "observed_mean": 0.0,
            "predicted_mean": predicted,
            "raw_loss": raw_loss,
            "profiled_loss": 0.0,
            "constant_optimum_loss": 0.0,
            "shape_gain_over_constant": 0.0,
            "scale_penalty": predicted,
            "optimal_log_scale": None,
            "support": "no_positive_count",
        }
    log_observed = float(np.log(observed))
    log_scale = log_observed - log_mean
    constant = observed * (1 - log_observed)
    shape_gain = dot - observed * log_mean
    profiled = constant - shape_gain
    penalty = predicted - observed + observed * log_scale
    # D(c)=c*mean(exp(raw))-mean(target*raw)-mean(target)*log(c)
    # has its minimum at c=mean(target)/mean(exp(raw)).
    np.testing.assert_allclose(raw_loss, profiled + penalty, rtol=1e-11, atol=1e-11)
    if penalty < -1e-10:
        raise ValueError("Negative scale penalty")
    return {
        "observed_mean": observed,
        "predicted_mean": predicted,
        "raw_loss": raw_loss,
        "profiled_loss": profiled,
        "constant_optimum_loss": constant,
        "shape_gain_over_constant": shape_gain,
        "scale_penalty": max(0.0, penalty),
        "optimal_log_scale": log_scale,
        "support": "positive_count",
    }
