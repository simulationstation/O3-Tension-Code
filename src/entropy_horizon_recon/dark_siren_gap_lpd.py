from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Sequence

import numpy as np


def _logsumexp(x: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(y, axis=axis) if axis is not None else np.squeeze(y)


def _logmeanexp(x: np.ndarray, *, axis: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(y, axis=axis)


@dataclass(frozen=True)
class BetaPrior:
    """Convenience wrapper for a Beta(mean, kappa) prior.

    Parameterization:
      alpha = mean * kappa
      beta  = (1-mean) * kappa
    """

    mean: float
    kappa: float

    @property
    def alpha(self) -> float:
        return float(self.mean) * float(self.kappa)

    @property
    def beta(self) -> float:
        return float(1.0 - float(self.mean)) * float(self.kappa)

    def logpdf(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        a = float(self.alpha)
        b = float(self.beta)
        if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
            raise ValueError("Invalid Beta prior parameters (alpha/beta must be finite and >0).")
        logB = lgamma(a) + lgamma(b) - lgamma(a + b)
        return (a - 1.0) * np.log(f) + (b - 1.0) * np.log1p(-f) - float(logB)


@dataclass(frozen=True)
class MarginalizedFMissResult:
    lpd_mu_total: float
    lpd_gr_total: float
    lpd_mu_total_data: float
    lpd_gr_total_data: float
    lpd_mu_total_sel: float
    lpd_gr_total_sel: float


def _stack_event_draws(arrs: Sequence[np.ndarray] | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(arrs, np.ndarray):
        x = np.asarray(arrs, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"{name} must be a (n_event, n_draw) array when provided as a numpy array.")
        return x
    rows: list[np.ndarray] = []
    for a in arrs:
        v = np.asarray(a, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"{name} must be a sequence of 1D arrays (one per event).")
        rows.append(v)
    if not rows:
        raise ValueError(f"{name} is empty.")
    n_draw = int(rows[0].size)
    if any(int(r.size) != n_draw for r in rows):
        raise ValueError(f"{name}: inconsistent draw counts across events.")
    return np.stack(rows, axis=0)


def marginalize_f_miss_global(
    *,
    logL_cat_mu_by_event: Sequence[np.ndarray] | np.ndarray,
    logL_cat_gr_by_event: Sequence[np.ndarray] | np.ndarray,
    logL_missing_mu_by_event: Sequence[np.ndarray] | np.ndarray,
    logL_missing_gr_by_event: Sequence[np.ndarray] | np.ndarray,
    log_alpha_mu: np.ndarray | float | None,
    log_alpha_gr: np.ndarray | float | None,
    prior: BetaPrior,
    n_f: int = 401,
    eps: float = 1e-6,
) -> MarginalizedFMissResult:
    """Marginalize a global missing-host fraction f_miss shared across events.

    This is a lightweight, dependency-minimal implementation used by the
    smoking-gun / reproduction scripts.

    Inputs are per-event log-likelihood vectors over posterior draws:
      logL_cat_*      : catalog term
      logL_missing_*  : missing-host term

    The mixture model per event is:
      logL = log( (1-f) * exp(logL_cat) + f * exp(logL_missing) )

    We integrate f over a fixed grid with trapezoid weights, using a Beta(mean,kappa) prior.
    Selection normalization enters as a per-draw subtraction of n_events * log_alpha(draw).
    """
    n_f = int(n_f)
    if n_f < 21:
        raise ValueError("n_f too small (use >=21).")
    eps = float(eps)
    if not (0.0 < eps < 0.1):
        raise ValueError("eps must be in (0, 0.1).")

    cat_mu = _stack_event_draws(logL_cat_mu_by_event, name="logL_cat_mu_by_event")
    cat_gr = _stack_event_draws(logL_cat_gr_by_event, name="logL_cat_gr_by_event")
    miss_mu = _stack_event_draws(logL_missing_mu_by_event, name="logL_missing_mu_by_event")
    miss_gr = _stack_event_draws(logL_missing_gr_by_event, name="logL_missing_gr_by_event")

    if not (cat_mu.shape == cat_gr.shape == miss_mu.shape == miss_gr.shape):
        raise ValueError("logL arrays must all have the same shape (n_event, n_draw).")

    n_ev, n_draw = (int(cat_mu.shape[0]), int(cat_mu.shape[1]))
    if n_ev <= 0 or n_draw <= 0:
        raise ValueError("Need at least one event and one draw.")

    # f-grid and trapezoid weights.
    f_grid = np.linspace(eps, 1.0 - eps, n_f, dtype=float)
    if not np.all(np.isfinite(f_grid)):
        raise ValueError("Invalid f_grid constructed.")
    w = np.ones_like(f_grid)
    w[0] = 0.5
    w[-1] = 0.5
    w = w * float(f_grid[1] - f_grid[0])
    logw = np.log(w)

    log_prior_f = prior.logpdf(f_grid)
    logf = np.log(f_grid)
    log1mf = np.log1p(-f_grid)

    # Selection normalization terms (per draw).
    if log_alpha_mu is None or log_alpha_gr is None:
        log_alpha_mu_v = None
        log_alpha_gr_v = None
    else:
        log_alpha_mu_v = np.asarray(log_alpha_mu, dtype=float).reshape((-1,))
        log_alpha_gr_v = np.asarray(log_alpha_gr, dtype=float).reshape((-1,))
        if log_alpha_mu_v.size not in (1, n_draw) or log_alpha_gr_v.size not in (1, n_draw):
            raise ValueError("log_alpha_mu/log_alpha_gr must be scalars or 1D arrays of length n_draw.")
        if log_alpha_mu_v.size == 1:
            log_alpha_mu_v = np.full((n_draw,), float(log_alpha_mu_v[0]), dtype=float)
        if log_alpha_gr_v.size == 1:
            log_alpha_gr_v = np.full((n_draw,), float(log_alpha_gr_v[0]), dtype=float)

    # Build f-grid logL matrices: shape (n_f, n_draw).
    logL_mu_fd = np.zeros((n_f, n_draw), dtype=float)
    logL_gr_fd = np.zeros((n_f, n_draw), dtype=float)
    for i in range(n_f):
        lf = float(logf[i])
        l1 = float(log1mf[i])
        # Broadcast over events/draws; then sum over events.
        ev_mu = np.logaddexp(l1 + cat_mu, lf + miss_mu)
        ev_gr = np.logaddexp(l1 + cat_gr, lf + miss_gr)
        logL_mu_fd[i] = np.sum(ev_mu, axis=0)
        logL_gr_fd[i] = np.sum(ev_gr, axis=0)

    # Selection-corrected totals.
    if log_alpha_mu_v is not None and log_alpha_gr_v is not None:
        logL_mu_fd_sel = logL_mu_fd - float(n_ev) * log_alpha_mu_v.reshape((1, -1))
        logL_gr_fd_sel = logL_gr_fd - float(n_ev) * log_alpha_gr_v.reshape((1, -1))
    else:
        logL_mu_fd_sel = logL_mu_fd
        logL_gr_fd_sel = logL_gr_fd

    lpd_mu_f = _logmeanexp(logL_mu_fd_sel, axis=1)
    lpd_gr_f = _logmeanexp(logL_gr_fd_sel, axis=1)
    lpd_mu_total = float(_logsumexp(log_prior_f + lpd_mu_f + logw))
    lpd_gr_total = float(_logsumexp(log_prior_f + lpd_gr_f + logw))

    # Data-only (no selection correction) totals.
    lpd_mu_f_data = _logmeanexp(logL_mu_fd, axis=1)
    lpd_gr_f_data = _logmeanexp(logL_gr_fd, axis=1)
    lpd_mu_total_data = float(_logsumexp(log_prior_f + lpd_mu_f_data + logw))
    lpd_gr_total_data = float(_logsumexp(log_prior_f + lpd_gr_f_data + logw))

    return MarginalizedFMissResult(
        lpd_mu_total=float(lpd_mu_total),
        lpd_gr_total=float(lpd_gr_total),
        lpd_mu_total_data=float(lpd_mu_total_data),
        lpd_gr_total_data=float(lpd_gr_total_data),
        lpd_mu_total_sel=float(lpd_mu_total - lpd_mu_total_data),
        lpd_gr_total_sel=float(lpd_gr_total - lpd_gr_total_data),
    )

