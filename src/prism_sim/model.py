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


def compute_r_m1exp(r_post1: float, beta: float, delta_pi: float) -> float:
    """Multiplicative log-link: R = R_post1 * exp(beta * delta_pi)."""
    r = r_post1 * math.exp(beta * delta_pi)
    return max(r, 1e-8)


def compute_r_m1asym(r_post1: float, beta_pos: float, beta_neg: float, delta_pi: float) -> float:
    """Signed-linear modulation: separate slopes for positive and negative delta_pi."""
    delta_pos = max(delta_pi, 0.0)
    delta_neg = min(delta_pi, 0.0)
    scale = 1.0 + beta_pos * delta_pos + beta_neg * delta_neg
    return max(r_post1 * scale, 1e-8)


def compute_r_m2(r_post1: float, lam: float, delta_pi: float) -> float:
    lam = max(-1.0, min(1.0, lam))
    scale = 1.0 - lam * math.tanh(delta_pi)
    return max(r_post1 * scale, 1e-8)


def compute_q_mq(q_baseline: float, beta_q: float, delta_pi: float) -> float:
    """Process noise modulation: Q = Q_baseline * exp(beta_q * delta_pi)."""
    q = q_baseline * math.exp(beta_q * delta_pi)
    return max(q, 1e-8)


def compute_rq_hybrid(
    r_post1: float, q_baseline: float, beta_r: float, beta_q: float, delta_pi: float
) -> tuple[float, float]:
    """Joint R+Q modulation: R and Q both exp-modulated by delta_pi."""
    r = r_post1 * math.exp(beta_r * delta_pi)
    q = q_baseline * math.exp(beta_q * delta_pi)
    return max(r, 1e-8), max(q, 1e-8)


def compute_r(model: str, r_post1: float, delta_pi: float, beta: float, lam: float) -> float:
    model = model.upper()
    if model == "M0":
        return compute_r_m0(r_post1)
    if model == "M1":
        return compute_r_m1(r_post1, beta, delta_pi)
    if model in {"M1EXP", "M1-EXP", "M1LOG"}:
        return compute_r_m1exp(r_post1, beta, delta_pi)
    if model in {"M1ASYM", "M1-ASYM"}:
        beta_neg = beta if lam is None or (isinstance(lam, float) and math.isnan(lam)) else lam
        return compute_r_m1asym(r_post1, beta, beta_neg, delta_pi)
    if model == "M2":
        return compute_r_m2(r_post1, lam, delta_pi)
    raise ValueError(f"Unknown model '{model}'. Use M0, M1, or M2.")
