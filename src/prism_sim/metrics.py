"""Utility metrics for simulation summaries."""

from __future__ import annotations

import numpy as np


def bias(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    return float(np.mean(estimate - truth))


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))
