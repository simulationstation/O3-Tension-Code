from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import PhysicalConstants
from .cosmology import build_background_from_logH_knots, spline_logH_from_knots
from .likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike


@dataclass(frozen=True)
class GPPosterior:
    z_knots: np.ndarray
    z_grid: np.ndarray
    H_samples: np.ndarray
    dH_dz_samples: np.ndarray
    hyper_samples: dict[str, np.ndarray]
    meta: dict


# For multiprocessing+emcee: pool.map must pickle the log-prob callable. Our GP
# log-prob is defined as a closure inside reconstruct_H_gp, so we stash it in a
# global and pass a module-level wrapper (requires "fork" start method).
_GLOBAL_LOG_PROB = None


def _log_prob_wrapper(theta: np.ndarray) -> float:
    fn = _GLOBAL_LOG_PROB
    if fn is None:
        raise RuntimeError("Global log-prob function not set (requires fork start method).")
    return float(fn(theta))


def _rbf_kernel(x: np.ndarray, y: np.ndarray, *, amp: float, ell: float) -> np.ndarray:
    dx = x[:, None] - y[None, :]
    return (amp**2) * np.exp(-0.5 * (dx / ell) ** 2)


def _matern32_kernel(x: np.ndarray, y: np.ndarray, *, amp: float, ell: float) -> np.ndarray:
    r = np.abs(x[:, None] - y[None, :]) / ell
    return (amp**2) * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


def _matern52_kernel(x: np.ndarray, y: np.ndarray, *, amp: float, ell: float) -> np.ndarray:
    r = np.abs(x[:, None] - y[None, :]) / ell
    return (amp**2) * (1.0 + np.sqrt(5.0) * r + 5.0 * r**2 / 3.0) * np.exp(-np.sqrt(5.0) * r)


def reconstruct_H_gp(
    *,
    z_knots: np.ndarray,
    sn_like: SNLogLike,
    cc_like: ChronometerLogLike,
    bao_likes: list[BaoLogLike],
    constants: PhysicalConstants,
    z_grid: np.ndarray,
    z_max_background: float,
    kernel: str = "matern32",
    n_walkers: int = 64,
    n_steps: int = 1500,
    n_burn: int = 500,
    seed: int = 0,
    n_processes: int | None = None,
    r_d_prior: tuple[float, float] = (120.0, 170.0),
) -> GPPosterior:
    """GP-prior reconstruction of H(z) using a spline representation for log H at knots.

    Parameters
    ----------
    z_knots:
        Knot locations for spline representation of log H(z).
    z_grid:
        Output grid where posterior samples are reported.
    z_max_background:
        Upper limit for distance integration grid (should be >= max(z) in all datasets).
    kernel:
        "rbf", "matern32", or "matern52".
    """
    if kernel not in {"rbf", "matern32", "matern52"}:
        raise ValueError("kernel must be one of: rbf, matern32, matern52")
    if np.any(np.diff(z_knots) <= 0):
        raise ValueError("z_knots must be strictly increasing.")

    rng = np.random.default_rng(seed)
    x_knots = np.log1p(z_knots)
    mean_logH = np.log(70.0)  # loose centering only

    kernel_fn = {"rbf": _rbf_kernel, "matern32": _matern32_kernel, "matern52": _matern52_kernel}[kernel]

    # Parameter vector: [logH_knots..., log_amp, log_ell, log_r_d]
    ndim = len(z_knots) + 3
    if n_walkers < 2 * ndim:
        raise ValueError(f"n_walkers must be >= 2*ndim = {2*ndim} (got {n_walkers}).")

    def log_prior(theta: np.ndarray) -> float:
        logH = theta[: len(z_knots)]
        log_amp, log_ell, log_r_d = theta[len(z_knots) :]

        # Weak box priors to keep numerics stable
        if not (np.log(10.0) < logH.min() < logH.max() < np.log(400.0)):
            return -np.inf
        if not (-6.0 < log_amp < 2.0):
            return -np.inf
        if not (-6.0 < log_ell < 2.0):
            return -np.inf
        r_d = float(np.exp(log_r_d))
        if not (r_d_prior[0] <= r_d <= r_d_prior[1]):
            return -np.inf

        amp = float(np.exp(log_amp))
        ell = float(np.exp(log_ell))
        K = kernel_fn(x_knots, x_knots, amp=amp, ell=ell)
        K[np.diag_indices_from(K)] += 1e-6
        cho = cho_factor(K, lower=True, check_finite=False)
        r = logH - mean_logH
        quad = float(r @ cho_solve(cho, r, check_finite=False))
        logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        return -0.5 * (quad + logdet + len(z_knots) * np.log(2.0 * np.pi))

    def log_likelihood(theta: np.ndarray) -> tuple[float, dict]:
        logH = theta[: len(z_knots)]
        log_r_d = theta[-1]
        r_d = float(np.exp(log_r_d))

        bg = build_background_from_logH_knots(
            z_knots,
            logH,
            z_max=z_max_background,
            n_grid=800,
            constants=constants,
        )

        # Chronometers
        H_cc = bg.H(cc_like.z)
        ll_cc = cc_like.loglike(H_cc)

        # BAO
        ll_bao = 0.0
        for bl in bao_likes:
            y_model = bl.predict(bg, r_d_Mpc=r_d)
            ll_bao += bl.loglike(y_model)

        # SN: model mu0 = 5 log10(D_L/Mpc)
        Dl = bg.Dl(sn_like.z)
        if np.any(Dl <= 0):
            return -np.inf, {}
        mu0 = 5.0 * np.log10(Dl)
        ll_sn, M_hat = sn_like.loglike_marginalized_M(mu0)

        return ll_cc + ll_bao + ll_sn, {"r_d": r_d, "M_hat": M_hat}

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll, _ = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # Initialize walkers around a smooth guess from chronometers
    z0 = z_knots
    H_guess = np.interp(z0, cc_like.z, cc_like.H, left=cc_like.H[0], right=cc_like.H[-1])
    init_logH = np.log(np.clip(H_guess, 30.0, 200.0))
    p0 = np.zeros((n_walkers, ndim))
    for i in range(n_walkers):
        p0[i, : len(z_knots)] = init_logH + rng.normal(scale=0.05, size=len(z_knots))
        p0[i, len(z_knots)] = np.log(0.5) + rng.normal(scale=0.2)  # log_amp
        p0[i, len(z_knots) + 1] = np.log(0.3) + rng.normal(scale=0.2)  # log_ell
        p0[i, len(z_knots) + 2] = np.log(147.0) + rng.normal(scale=0.05)  # log_r_d

    import emcee

    if n_processes is None or n_processes <= 1:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(p0, n_steps, progress=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        global _GLOBAL_LOG_PROB
        _GLOBAL_LOG_PROB = log_prob
        try:
            with ctx.Pool(processes=n_processes) as pool:
                sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_prob_wrapper, pool=pool)
                sampler.run_mcmc(p0, n_steps, progress=True)
        finally:
            _GLOBAL_LOG_PROB = None

    chain = sampler.get_chain(discard=n_burn, flat=True)
    if chain.shape[0] < 10:
        raise RuntimeError("Too few posterior samples; reduce burn-in or increase n_steps.")

    # Thin for output sampling
    n_draws = min(500, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=n_draws, replace=False)
    draws = chain[idx]

    H_samples = np.empty((n_draws, len(z_grid)))
    dH_dz_samples = np.empty((n_draws, len(z_grid)))
    r_d_s = np.empty(n_draws)
    M_hat_s = np.empty(n_draws)
    log_amp_s = np.empty(n_draws)
    log_ell_s = np.empty(n_draws)

    for j, th in enumerate(draws):
        logH_k = th[: len(z_knots)]
        log_amp_s[j] = th[len(z_knots)]
        log_ell_s[j] = th[len(z_knots) + 1]
        r_d_s[j] = float(np.exp(th[-1]))

        spline = spline_logH_from_knots(z_knots, logH_k)
        logH_g = spline(z_grid)
        H_g = np.exp(logH_g)
        dlogH_dz = spline(z_grid, 1)
        dH_dz = H_g * dlogH_dz

        H_samples[j] = H_g
        dH_dz_samples[j] = dH_dz

        ll, extra = log_likelihood(th)
        M_hat_s[j] = float(extra.get("M_hat", np.nan)) if np.isfinite(ll) else np.nan

    meta = {
        "kernel": kernel,
        "n_walkers": int(n_walkers),
        "n_steps": int(n_steps),
        "n_burn": int(n_burn),
        "n_draws": int(n_draws),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    return GPPosterior(
        z_knots=z_knots,
        z_grid=z_grid,
        H_samples=H_samples,
        dH_dz_samples=dH_dz_samples,
        hyper_samples={"log_amp": log_amp_s, "log_ell": log_ell_s, "r_d_Mpc": r_d_s, "M_hat": M_hat_s},
        meta=meta,
    )
