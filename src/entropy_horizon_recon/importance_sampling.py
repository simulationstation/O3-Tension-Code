from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _ess_from_weights(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return 0.0
    w = np.where(np.isfinite(w), w, 0.0)
    s1 = float(np.sum(w))
    if not (np.isfinite(s1) and s1 > 0.0):
        return 0.0
    s2 = float(np.sum(w * w))
    if not (np.isfinite(s2) and s2 > 0.0):
        return 0.0
    return float((s1 * s1) / s2)


@dataclass(frozen=True)
class SmoothedLogWeights:
    """Result of stabilizing/smoothing log-weights for importance sampling."""

    method: Literal["none", "truncate", "psis"]
    logw: np.ndarray  # same shape as input
    ess_raw: float
    ess_smooth: float
    clip_frac: float | None = None
    pareto_k: float | None = None
    n_tail: int | None = None


def truncate_logweights(
    logw: np.ndarray,
    *,
    tau: float | None = None,
) -> SmoothedLogWeights:
    """Truncated importance sampling (TIS) for stabilizing heavy-tailed weights.

    Uses the Ionides-style cap:

      w_i <- min(w_i, tau * mean(w)),
      with tau defaulting to sqrt(n).

    This is intentionally cheap (O(n)) and suitable for per-H0-grid evaluations.
    """
    logw = np.asarray(logw, dtype=float)
    if logw.ndim != 1:
        raise ValueError("truncate_logweights expects a 1D logw array.")

    m = np.isfinite(logw)
    if not np.any(m):
        return SmoothedLogWeights(method="truncate", logw=logw, ess_raw=0.0, ess_smooth=0.0, clip_frac=0.0)

    lw = np.asarray(logw[m], dtype=float)
    lw_max = float(np.max(lw))
    # Normalize for stability; keep a floor so weights remain strictly positive for logs.
    w = np.exp(np.clip(lw - lw_max, -700.0, 50.0))
    n = int(w.size)

    ess_raw = _ess_from_weights(w)

    mean_w = float(np.mean(w))
    if not (np.isfinite(mean_w) and mean_w > 0.0):
        return SmoothedLogWeights(method="truncate", logw=logw, ess_raw=ess_raw, ess_smooth=0.0, clip_frac=1.0)

    tau_eff = float(np.sqrt(n)) if tau is None or float(tau) <= 0.0 else float(tau)
    if not np.isfinite(tau_eff) or tau_eff <= 0.0:
        raise ValueError("tau must be finite and positive when provided.")

    w_cap = tau_eff * mean_w
    w_clip = np.minimum(w, w_cap)
    clip_frac = float(np.mean(w > w_cap))

    ess_smooth = _ess_from_weights(w_clip)

    lw_clip = np.log(np.clip(w_clip, 1e-300, np.inf)) + lw_max
    out = np.full_like(logw, -np.inf, dtype=float)
    out[m] = lw_clip
    return SmoothedLogWeights(method="truncate", logw=out, ess_raw=ess_raw, ess_smooth=ess_smooth, clip_frac=clip_frac)


def psis_smooth_logweights(
    logw: np.ndarray,
    *,
    tail_fraction: float = 0.2,
    min_tail: int = 20,
) -> SmoothedLogWeights:
    """Pareto-smoothed importance sampling (PSIS) on log-weights.

    Notes:
    - This is O(n + m log m) per call (m=tail size), so it is *not* suitable for millions
      of per-H0 evaluations (use `truncate` for that). It is intended for diagnostics or
      smaller workloads.
    - We fit a generalized Pareto distribution (GPD) to the tail exceedances of the
      normalized weights w = exp(logw - max(logw)), then replace the tail weights with
      the expected order statistics under the fitted tail.
    """
    logw = np.asarray(logw, dtype=float)
    if logw.ndim != 1:
        raise ValueError("psis_smooth_logweights expects a 1D logw array.")

    m = np.isfinite(logw)
    if not np.any(m):
        return SmoothedLogWeights(method="psis", logw=logw, ess_raw=0.0, ess_smooth=0.0, pareto_k=float("nan"), n_tail=0)

    lw = np.asarray(logw[m], dtype=float)
    lw_max = float(np.max(lw))
    w = np.exp(np.clip(lw - lw_max, -700.0, 50.0))
    n = int(w.size)
    ess_raw = _ess_from_weights(w)

    if n < 2 * int(min_tail):
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=float("nan"), n_tail=0)

    tf = float(tail_fraction)
    if not (np.isfinite(tf) and 0.01 <= tf <= 0.5):
        raise ValueError("tail_fraction must be in [0.01, 0.5] for PSIS.")

    m_tail = int(max(int(min_tail), int(np.ceil(tf * n))))
    m_tail = min(m_tail, n - 1)
    if m_tail < 5:
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=float("nan"), n_tail=m_tail)

    # Indices of the m_tail largest weights (O(n)).
    idx_tail = np.argpartition(w, n - m_tail)[n - m_tail :]
    tail = np.asarray(w[idx_tail], dtype=float)

    # Threshold is the smallest tail weight.
    t = float(np.min(tail))
    excess = tail - t
    if not np.any(excess > 0.0):
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=0.0, n_tail=m_tail)

    # Fit GPD to exceedances with loc fixed at 0.
    try:
        from scipy.stats import genpareto  # type: ignore

        c_hat, _loc, scale_hat = genpareto.fit(excess, floc=0.0)
        k_hat = float(c_hat)
        scale_hat = float(scale_hat)
    except Exception:
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=float("nan"), n_tail=m_tail)

    if not (np.isfinite(k_hat) and np.isfinite(scale_hat) and scale_hat > 0.0):
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=float("nan"), n_tail=m_tail)

    # Replace tail weights by expected order statistics (rank-based smoothing).
    order = np.argsort(tail)
    probs = (np.arange(1, m_tail + 1, dtype=float) - 0.5) / float(m_tail)
    try:
        sm_excess = genpareto.ppf(probs, c=k_hat, loc=0.0, scale=scale_hat)
        sm_excess = np.asarray(sm_excess, dtype=float)
    except Exception:
        out = np.full_like(logw, -np.inf, dtype=float)
        out[m] = lw
        return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_raw, pareto_k=k_hat, n_tail=m_tail)

    sm_excess = np.where(np.isfinite(sm_excess), sm_excess, 0.0)
    sm_tail = t + np.maximum(sm_excess, 0.0)

    w_smooth = np.array(w, copy=True)
    w_smooth[idx_tail[order]] = sm_tail
    ess_smooth = _ess_from_weights(w_smooth)

    lw_smooth = np.log(np.clip(w_smooth, 1e-300, np.inf)) + lw_max
    out = np.full_like(logw, -np.inf, dtype=float)
    out[m] = lw_smooth
    return SmoothedLogWeights(method="psis", logw=out, ess_raw=ess_raw, ess_smooth=ess_smooth, pareto_k=k_hat, n_tail=m_tail)


def smooth_logweights(
    logw: np.ndarray,
    *,
    method: Literal["none", "truncate", "psis"],
    truncate_tau: float | None = None,
    psis_tail_fraction: float = 0.2,
    psis_min_tail: int = 20,
) -> SmoothedLogWeights:
    """Dispatch to a log-weight stabilization method."""
    method = str(method)
    if method == "none":
        logw = np.asarray(logw, dtype=float)
        m = np.isfinite(logw)
        lw = np.asarray(logw[m], dtype=float)
        lw_max = float(np.max(lw)) if lw.size else 0.0
        w = np.exp(np.clip(lw - lw_max, -700.0, 50.0))
        ess = _ess_from_weights(w)
        return SmoothedLogWeights(method="none", logw=logw, ess_raw=ess, ess_smooth=ess)
    if method == "truncate":
        return truncate_logweights(logw, tau=truncate_tau)
    if method == "psis":
        return psis_smooth_logweights(logw, tail_fraction=float(psis_tail_fraction), min_tail=int(psis_min_tail))
    raise ValueError("Unknown importance sampling smoothing method (expected: none|truncate|psis).")

