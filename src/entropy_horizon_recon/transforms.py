from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def log_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    out = -np.logaddexp(0.0, -np.asarray(x, dtype=float))
    if np.isscalar(x):
        return float(out)
    return out


def log1m_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    out = -np.logaddexp(0.0, np.asarray(x, dtype=float))
    if np.isscalar(x):
        return float(out)
    return out


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p_arr = np.asarray(p, dtype=float)
    out = np.log(p_arr) - np.log1p(-p_arr)
    if np.isscalar(p):
        return float(out)
    return out


def bounded_from_unbounded(u: np.ndarray | float, lo: float, hi: float) -> tuple[np.ndarray | float, np.ndarray | float]:
    u_arr = np.asarray(u, dtype=float)
    lo = float(lo)
    hi = float(hi)
    if not (hi > lo):
        raise ValueError("Invalid bounds for transform: require hi > lo.")
    s = sigmoid(u_arr)
    x = lo + (hi - lo) * s
    log_jac = np.log(hi - lo) + log_sigmoid(u_arr) + log1m_sigmoid(u_arr)
    if np.isscalar(u):
        return float(x), float(log_jac)
    return x, log_jac


def unbounded_from_bounded(x: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    lo = float(lo)
    hi = float(hi)
    if not (hi > lo):
        raise ValueError("Invalid bounds for transform: require hi > lo.")
    t = (x_arr - lo) / (hi - lo)
    if np.any(t <= 0.0) or np.any(t >= 1.0):
        raise ValueError("x must be strictly inside (lo, hi) for inverse transform.")
    u = logit(t)
    if np.isscalar(x):
        return float(u)
    return u

