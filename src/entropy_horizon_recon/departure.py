from __future__ import annotations

import numpy as np


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Fit y ≈ a + b x with weights w (returns a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("x,y,w shape mismatch")
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("w must be positive and finite")
    X = np.column_stack([np.ones_like(x), x])
    XtW = (X.T * w)
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return float(beta[0]), float(beta[1])


def compute_departure_stats(
    *,
    logA_grid: np.ndarray,
    logmu_samples: np.ndarray,
    eps: float = 0.01,
    slope_eps: float | None = 0.01,
    weight_mode: str = "variance",
) -> dict:
    """Model-agnostic scalar summaries of departure from BH (logμ=0).

    Definitions
    -----------
    Let w(logA) ∝ 1/Var[logμ(logA)] and normalized to integrate to 1 over the grid.

    - m = ∫ w(logA) logμ(logA) dlogA
    - s = weighted least-squares slope of logμ vs (logA - <logA>_w)
    """
    logA_grid = np.asarray(logA_grid, dtype=float)
    draws = np.asarray(logmu_samples, dtype=float)
    if draws.ndim != 2:
        raise ValueError("logmu_samples must be (n_draws, n_grid)")
    if draws.shape[1] != logA_grid.size:
        raise ValueError("Grid mismatch.")

    mean = np.mean(draws, axis=0)
    var = np.var(draws, axis=0, ddof=1)
    if weight_mode == "variance":
        w = 1.0 / np.clip(var, 1e-12, np.inf)
    elif weight_mode == "uniform":
        w = np.ones_like(var)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")
    w = w / np.trapezoid(w, x=logA_grid)
    logA0 = float(np.average(logA_grid, weights=w))
    x = logA_grid - logA0

    m_draw = np.trapezoid(draws * w[None, :], x=logA_grid, axis=1)
    s_draw = np.empty(draws.shape[0])
    for i in range(draws.shape[0]):
        _, b = _weighted_linear_fit(x, draws[i], w)
        s_draw[i] = b

    def _summ(arr: np.ndarray) -> dict[str, float]:
        arr = np.asarray(arr, dtype=float)
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return {"mean": float(np.mean(arr)), "std": std}

    out = {
        "grid": {"logA_min": float(logA_grid.min()), "logA_max": float(logA_grid.max()), "n": int(logA_grid.size)},
        "weights": {"logA0_w": logA0},
        "m": {
            **_summ(m_draw),
            "p_gt0": float(np.mean(m_draw > 0.0)),
            "p_abs_lt_eps": float(np.mean(np.abs(m_draw) < float(eps))),
            "eps": float(eps),
        },
        "slope": {
            **_summ(s_draw),
            "p_gt0": float(np.mean(s_draw > 0.0)),
        },
    }
    if slope_eps is not None:
        out["slope"]["p_abs_lt_eps"] = float(np.mean(np.abs(s_draw) < float(slope_eps)))
        out["slope"]["eps"] = float(slope_eps)
    return out
