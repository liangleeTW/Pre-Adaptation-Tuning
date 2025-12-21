"""Model definitions and R mappings for prism adaptation simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelParams:
    """Shared scalar state-space parameters."""

    A: float = 1.0
    Q: float = 1e-4
    m: float = -12.1
    b: float = 0.0


def compute_r_m0(r_post1: float) -> float:
    return max(r_post1, 1e-8)


def compute_r_m1(r_post1: float, beta: float, delta_pi: float) -> float:
    return max(r_post1 + beta * delta_pi, 1e-8)


def compute_r_m2(r_post1: float, lam: float, delta_pi: float) -> float:
    lam = max(-1.0, min(1.0, lam))
    scale = 1.0 - lam * math.tanh(delta_pi)
    return max(r_post1 * scale, 1e-8)


def compute_r(model: str, r_post1: float, delta_pi: float, beta: float, lam: float) -> float:
    model = model.upper()
    if model == "M0":
        return compute_r_m0(r_post1)
    if model == "M1":
        return compute_r_m1(r_post1, beta, delta_pi)
    if model == "M2":
        return compute_r_m2(r_post1, lam, delta_pi)
    raise ValueError(f"Unknown model '{model}'. Use M0, M1, or M2.")
