"""Simulation routines for the scalar state-space adaptation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np

from prism_sim.model import ModelParams, compute_r


@dataclass(frozen=True)
class SubjectConfig:
    subject_id: int
    r_post1: float
    delta_pi: float
    model: str
    beta: float
    lam: float
    group: str = "all"


def simulate_subject(
    params: ModelParams,
    r_measure: float,
    n_trials: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate a single subject's trajectory under a fixed R."""

    x = np.zeros(n_trials)
    e_obs = np.zeros(n_trials)
    e_true = np.zeros(n_trials)
    k = np.zeros(n_trials)
    p = np.zeros(n_trials)

    p_prev = 1.0
    x_prev = 0.0

    for t in range(n_trials):
        p_pred = params.A * p_prev * params.A + params.Q
        k_t = p_pred / (p_pred + r_measure)

        v_t = rng.normal(0.0, math.sqrt(r_measure))
        e_t = params.m - x_prev + v_t
        e_obs_t = e_t + params.b
        x_t = params.A * x_prev + k_t * e_t

        p_t = (1.0 - k_t) * p_pred

        x[t] = x_t
        e_obs[t] = e_obs_t
        e_true[t] = e_t
        k[t] = k_t
        p[t] = p_t

        x_prev = x_t
        p_prev = p_t

    return {"x": x, "e_obs": e_obs, "e_true": e_true, "k": k, "p": p}


def simulate_cohort(
    subjects: List[SubjectConfig],
    params: ModelParams,
    n_trials: int,
    seed: int | None = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[float]]:
    rng = np.random.default_rng(seed)
    outputs: List[Dict[str, np.ndarray]] = []
    r_values: List[float] = []

    for cfg in subjects:
        r_measure = compute_r(cfg.model, cfg.r_post1, cfg.delta_pi, cfg.beta, cfg.lam)
        r_values.append(r_measure)
        outputs.append(simulate_subject(params, r_measure, n_trials, rng))

    return outputs, r_values


def sample_correlated_normals(
    n: int,
    rho: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = max(-1.0, min(1.0, rho))
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = z1
    y = rho * z1 + math.sqrt(1.0 - rho**2) * z2
    return x, y
