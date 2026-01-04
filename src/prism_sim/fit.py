"""Likelihood utilities for fitting the prism adaptation model."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from prism_sim.model import ModelParams


def loglik_subject(errors: Iterable[float], params: ModelParams, r_measure: float) -> float:
    """Kalman-filter log-likelihood for a single subject's observed errors."""

    a = params.A
    q = params.Q
    h = -1.0
    c = params.m + params.b

    x = 0.0
    p = 1.0
    ll = 0.0

    for y in errors:
        x_pred = a * x
        p_pred = a * p * a + q
        y_pred = h * x_pred + c
        s = h * p_pred * h + r_measure

        v = y - y_pred
        ll += -0.5 * (math.log(2.0 * math.pi * s) + (v * v) / s)

        k = p_pred * h / s
        x = x_pred + k * v
        p = (1.0 - k * h) * p_pred

    return ll


def stack_errors(trials_df, subject_col: str = "subject") -> dict[int, np.ndarray]:
    grouped = trials_df.groupby(subject_col)["error"]
    result = {}
    for subject, group in grouped:
        try:
            key = int(subject)
        except (TypeError, ValueError):
            key = subject
        result[key] = group.values
    return result
