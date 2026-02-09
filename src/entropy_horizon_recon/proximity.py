from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from .entropy_models import (
    BarrowParams,
    KaniadakisParams,
    TsallisParams,
    log_mu_barrow,
    log_mu_kaniadakis,
    log_mu_tsallis,
)


def weighted_L2_distance(
    x: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    *,
    w: np.ndarray,
) -> float:
    """Compute ∫ w(x) [f(x)-g(x)]^2 dx using trapezoidal rule."""
    return float(np.trapezoid(w * (f - g) ** 2, x=x))


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Fit y ≈ a + b x with weights w."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("x,y,w shape mismatch")
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("w must be positive and finite")
    X = np.column_stack([np.ones_like(x), x])
    # Weighted normal equations
    XtW = (X.T * w)
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    a, b = float(beta[0]), float(beta[1])
    return a, b


def fit_tsallis(
    logA: np.ndarray,
    logmu: np.ndarray,
    *,
    w: np.ndarray,
    logA0: float | None = None,
) -> TsallisParams:
    """Weighted least-squares fit of logμ = logμ0 - (δ-1)(logA-logA0)."""
    logA = np.asarray(logA, dtype=float)
    logmu = np.asarray(logmu, dtype=float)
    w = np.asarray(w, dtype=float)
    if logA0 is None:
        logA0 = float(np.average(logA, weights=w))
    x = logA - logA0
    a, b = _weighted_linear_fit(x, logmu, w)
    # logmu = a + b (logA-logA0) = logmu0 - (δ-1)(logA-logA0)
    delta = 1.0 - b
    return TsallisParams(delta=float(delta), log_mu0=float(a), logA0=float(logA0))


def fit_barrow(
    logA: np.ndarray,
    logmu: np.ndarray,
    *,
    w: np.ndarray,
    logA0: float | None = None,
) -> BarrowParams:
    """Weighted fit; Barrow slope corresponds to Δ = -2 * slope in logμ vs logA."""
    logA = np.asarray(logA, dtype=float)
    logmu = np.asarray(logmu, dtype=float)
    w = np.asarray(w, dtype=float)
    if logA0 is None:
        logA0 = float(np.average(logA, weights=w))
    x = logA - logA0
    a, b = _weighted_linear_fit(x, logmu, w)
    Delta = -2.0 * b
    return BarrowParams(Delta=float(Delta), log_mu0=float(a), logA0=float(logA0))


def fit_kaniadakis(
    logA: np.ndarray,
    logmu: np.ndarray,
    *,
    w: np.ndarray,
    A_ref: float | None = None,
) -> KaniadakisParams:
    """Fit logμ = logμ0 - logcosh(beta_tilde * A/A_ref)."""
    logA = np.asarray(logA, dtype=float)
    logmu = np.asarray(logmu, dtype=float)
    w = np.asarray(w, dtype=float)
    if A_ref is None:
        A_ref = float(np.exp(np.median(logA)))

    def obj(p: np.ndarray) -> float:
        log_mu0, log_beta = float(p[0]), float(p[1])
        beta = float(np.exp(log_beta))
        params = KaniadakisParams(beta_tilde=beta, log_mu0=log_mu0, A_ref=A_ref)
        pred = log_mu_kaniadakis(logA, params)
        return float(np.sum(w * (logmu - pred) ** 2))

    p0 = np.array([float(np.average(logmu, weights=w)), np.log(1e-3)], dtype=float)
    res = minimize(obj, p0, method="Nelder-Mead")
    log_mu0, log_beta = float(res.x[0]), float(res.x[1])
    return KaniadakisParams(beta_tilde=float(np.exp(log_beta)), log_mu0=log_mu0, A_ref=A_ref)


def proximity_summary(
    *,
    logA_grid: np.ndarray,
    logmu_samples: np.ndarray,
) -> dict:
    """Compute proximity metrics and projection-posteriors for Tsallis/Barrow/Kaniadakis."""
    logA_grid = np.asarray(logA_grid, dtype=float)
    draws = np.asarray(logmu_samples, dtype=float)
    if draws.ndim != 2:
        raise ValueError("logmu_samples must be (n_draws, n_grid)")
    if draws.shape[1] != logA_grid.size:
        raise ValueError("Grid mismatch.")

    mean = np.mean(draws, axis=0)
    var = np.var(draws, axis=0, ddof=1)
    if not np.all(np.isfinite(var)) or np.all(var <= 0):
        w = np.ones_like(logA_grid)
    else:
        w = 1.0 / np.clip(var, 1e-8, np.inf)
        norm = np.trapezoid(w, x=logA_grid)
        if not np.isfinite(norm) or norm <= 0:
            w = np.ones_like(logA_grid)
        else:
            w = w / norm
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        w = np.ones_like(logA_grid)

    # Fit to posterior mean
    ts = fit_tsallis(logA_grid, mean, w=w)
    ba = fit_barrow(logA_grid, mean, w=w)
    ka = fit_kaniadakis(logA_grid, mean, w=w)

    model_mean = {
        "tsallis": log_mu_tsallis(logA_grid, ts),
        "barrow": log_mu_barrow(logA_grid, ba),
        "kaniadakis": log_mu_kaniadakis(logA_grid, ka),
        "bh": np.zeros_like(logA_grid),
    }
    D2_mean = {
        name: weighted_L2_distance(logA_grid, mean, f, w=w) for name, f in model_mean.items()
    }

    # Projection posterior: fit each draw
    ts_d, ba_d, ka_d = [], [], []
    D2_ts, D2_ba, D2_ka, D2_bh = [], [], [], []
    for d in draws:
        ts_i = fit_tsallis(logA_grid, d, w=w)
        ba_i = fit_barrow(logA_grid, d, w=w)
        ka_i = fit_kaniadakis(logA_grid, d, w=w)
        ts_d.append(ts_i)
        ba_d.append(ba_i)
        ka_d.append(ka_i)
        D2_ts.append(weighted_L2_distance(logA_grid, d, log_mu_tsallis(logA_grid, ts_i), w=w))
        D2_ba.append(weighted_L2_distance(logA_grid, d, log_mu_barrow(logA_grid, ba_i), w=w))
        D2_ka.append(weighted_L2_distance(logA_grid, d, log_mu_kaniadakis(logA_grid, ka_i), w=w))
        D2_bh.append(weighted_L2_distance(logA_grid, d, np.zeros_like(d), w=w))

    return {
        "fit_to_mean": {
            "tsallis": ts.__dict__,
            "barrow": ba.__dict__,
            "kaniadakis": ka.__dict__,
        },
        "D2_mean": D2_mean,
        "projection": {
            "tsallis": {
                "delta": np.array([p.delta for p in ts_d]),
                "log_mu0": np.array([p.log_mu0 for p in ts_d]),
                "D2": np.array(D2_ts),
            },
            "barrow": {
                "Delta": np.array([p.Delta for p in ba_d]),
                "log_mu0": np.array([p.log_mu0 for p in ba_d]),
                "D2": np.array(D2_ba),
            },
            "kaniadakis": {
                "beta_tilde": np.array([p.beta_tilde for p in ka_d]),
                "log_mu0": np.array([p.log_mu0 for p in ka_d]),
                "D2": np.array(D2_ka),
            },
            "bh": {"D2": np.array(D2_bh)},
        },
    }


def gp_log_evidence(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    kernel: str,
    amp: float,
    ell: float,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if kernel == "rbf":
        K = (amp**2) * np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)
    elif kernel == "matern32":
        r = np.abs(x[:, None] - x[None, :]) / ell
        K = (amp**2) * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    elif kernel == "matern52":
        r = np.abs(x[:, None] - x[None, :]) / ell
        K = (amp**2) * (1.0 + np.sqrt(5.0) * r + 5.0 * r**2 / 3.0) * np.exp(-np.sqrt(5.0) * r)
    else:
        raise ValueError("kernel must be one of: rbf, matern32, matern52")

    C = K + np.diag(yerr**2) + 1e-10 * np.eye(x.size)
    cho = cho_factor(C, lower=True, check_finite=False)
    alpha = cho_solve(cho, y, check_finite=False)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    return -0.5 * (float(y @ alpha) + logdet + x.size * np.log(2.0 * np.pi))


def fit_gp_hyperparams(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    kernel: str,
) -> dict:
    """Empirical-Bayes hyperparameter fit maximizing GP marginal likelihood."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    def obj(p: np.ndarray) -> float:
        log_amp, log_ell = float(p[0]), float(p[1])
        return -gp_log_evidence(x, y, yerr, kernel=kernel, amp=float(np.exp(log_amp)), ell=float(np.exp(log_ell)))

    p0 = np.array([np.log(np.std(y) + 1e-3), np.log(0.3)], dtype=float)
    res = minimize(obj, p0, method="Nelder-Mead")
    log_amp, log_ell = float(res.x[0]), float(res.x[1])
    amp, ell = float(np.exp(log_amp)), float(np.exp(log_ell))
    return {"amp": amp, "ell": ell, "logZ": -float(res.fun), "success": bool(res.success)}


def log_evidence_parametric(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    model: str,
    n_mc: int = 20_000,
    seed: int = 0,
) -> dict:
    """Monte-Carlo evidence for a parametric model against Gaussian pseudo-data y±yerr."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    inv_var = 1.0 / (yerr**2)
    log_norm = -0.5 * float(np.sum(np.log(2.0 * np.pi * yerr**2)))

    logA0 = float(np.median(x))
    A_ref = float(np.exp(logA0))

    if model == "tsallis":
        delta = rng.uniform(0.0, 2.0, size=n_mc)
        log_mu0 = rng.uniform(-5.0, 5.0, size=n_mc)
        ll = np.empty(n_mc)
        for i in range(n_mc):
            params = TsallisParams(delta=float(delta[i]), log_mu0=float(log_mu0[i]), logA0=logA0)
            pred = log_mu_tsallis(x, params)
            ll[i] = log_norm - 0.5 * float(np.sum((y - pred) ** 2 * inv_var))
        logZ = float(np.log(np.mean(np.exp(ll - ll.max()))) + ll.max())
        return {"logZ": logZ, "priors": {"delta": [0.0, 2.0], "log_mu0": [-5.0, 5.0], "logA0": logA0}}

    if model == "barrow":
        Delta = rng.uniform(-1.0, 1.0, size=n_mc)
        log_mu0 = rng.uniform(-5.0, 5.0, size=n_mc)
        ll = np.empty(n_mc)
        for i in range(n_mc):
            params = BarrowParams(Delta=float(Delta[i]), log_mu0=float(log_mu0[i]), logA0=logA0)
            pred = log_mu_barrow(x, params)
            ll[i] = log_norm - 0.5 * float(np.sum((y - pred) ** 2 * inv_var))
        logZ = float(np.log(np.mean(np.exp(ll - ll.max()))) + ll.max())
        return {"logZ": logZ, "priors": {"Delta": [-1.0, 1.0], "log_mu0": [-5.0, 5.0], "logA0": logA0}}

    if model == "kaniadakis":
        beta_tilde = np.exp(rng.uniform(np.log(1e-6), np.log(50.0), size=n_mc))
        log_mu0 = rng.uniform(-5.0, 5.0, size=n_mc)
        ll = np.empty(n_mc)
        for i in range(n_mc):
            params = KaniadakisParams(
                beta_tilde=float(beta_tilde[i]),
                log_mu0=float(log_mu0[i]),
                A_ref=A_ref,
            )
            pred = log_mu_kaniadakis(x, params)
            ll[i] = log_norm - 0.5 * float(np.sum((y - pred) ** 2 * inv_var))
        logZ = float(np.log(np.mean(np.exp(ll - ll.max()))) + ll.max())
        return {
            "logZ": logZ,
            "priors": {
                "beta_tilde_log": [float(np.log(1e-6)), float(np.log(50.0))],
                "log_mu0": [-5.0, 5.0],
                "A_ref": A_ref,
            },
        }

    if model == "bh":
        pred = np.zeros_like(y)
        ll = log_norm - 0.5 * float(np.sum((y - pred) ** 2 * inv_var))
        return {"logZ": float(ll), "priors": {}}

    raise ValueError("model must be one of: tsallis, barrow, kaniadakis, bh")
