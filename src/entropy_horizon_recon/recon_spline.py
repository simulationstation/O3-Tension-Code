from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid
from .likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike


@dataclass(frozen=True)
class SplinePosterior:
    z_knots: np.ndarray
    z_grid: np.ndarray
    H_samples: np.ndarray
    dH_dz_samples: np.ndarray
    meta: dict


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # Stable softplus: log(1+exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def reconstruct_H_spline(
    *,
    z_knots: np.ndarray,
    sn_like: SNLogLike,
    cc_like: ChronometerLogLike,
    bao_likes: list[BaoLogLike],
    constants: PhysicalConstants,
    z_grid: np.ndarray,
    z_max_background: float,
    smooth_lambda: float = 10.0,
    n_bootstrap: int = 100,
    seed: int = 0,
    r_d_init: float = 147.0,
    monotone: bool = True,
    monotone_tol: float = 1e-3,
) -> SplinePosterior:
    """Penalized spline reconstruction of H(z) with bootstrap uncertainties.

    This is a *cross-check* reconstruction (not used for μ(A) inference). We enforce
    physical plausibility by optionally constraining log H(z) to be non-decreasing.
    """
    rng = np.random.default_rng(seed)
    K = len(z_knots)
    z_knots = np.asarray(z_knots, dtype=float)

    # Use a dense grid for distance integration (shared across evaluations).
    z_bg = np.linspace(0.0, float(z_max_background), 800)

    def _bg_from_logH_knots(logH: np.ndarray) -> tuple[object, PchipInterpolator]:
        # Use a shape-preserving interpolator for logH to avoid cubic overshoot.
        sp = PchipInterpolator(z_knots, np.asarray(logH, dtype=float), extrapolate=True)
        logH_bg = sp(z_bg)
        H_bg = np.exp(logH_bg)
        bg = build_background_from_H_grid(z_bg, H_bg, constants=constants)
        return bg, sp

    def loglike_from(logH: np.ndarray, r_d: float) -> tuple[float, float]:
        bg, _ = _bg_from_logH_knots(logH)
        ll_cc = cc_like.loglike(bg.H(cc_like.z))
        ll_bao = 0.0
        for bl in bao_likes:
            ll_bao += bl.loglike(bl.predict(bg, r_d_Mpc=r_d))
        Dl = bg.Dl(sn_like.z)
        if np.any(Dl <= 0):
            return -np.inf, np.nan
        mu0 = 5.0 * np.log10(Dl)
        ll_sn, M_hat = sn_like.loglike_marginalized_M(mu0)
        return ll_cc + ll_bao + ll_sn, M_hat

    def decode_theta(theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (K + 1,):
            raise ValueError("theta shape mismatch")
        if not monotone:
            logH = theta[:K]
        else:
            logH0 = float(theta[0])
            du = theta[1:K]
            deltas = _softplus(du)
            logH = np.empty(K, dtype=float)
            logH[0] = logH0
            logH[1:] = logH0 + np.cumsum(deltas)
        r_d = float(np.exp(theta[K]))
        return logH, r_d

    def objective(theta: np.ndarray) -> float:
        logH, r_d = decode_theta(theta)
        # Basic physical bounds to keep optimization stable.
        if np.any(~np.isfinite(logH)):
            return 1e50
        if logH.min() < np.log(10.0) or logH.max() > np.log(400.0):
            return 1e50
        ll, _ = loglike_from(logH, r_d)
        if not np.isfinite(ll):
            return 1e50
        # quadratic penalty on discrete second differences of logH
        d2 = np.diff(logH, n=2)
        penalty = 0.5 * smooth_lambda * float(np.sum(d2**2))
        return -ll + penalty

    # Initial guess: interpolate chronometers to knots
    H_guess = np.interp(z_knots, cc_like.z, cc_like.H, left=cc_like.H[0], right=cc_like.H[-1])
    logH0 = np.log(np.clip(H_guess, 30.0, 200.0))
    if monotone:
        # Enforce a monotone starting point; use cumulative max (fast, deterministic).
        logH_iso = np.maximum.accumulate(logH0)
        d = np.diff(logH_iso)
        d = np.clip(d, 1e-6, None)
        du0 = np.log(np.expm1(d))
        theta0 = np.concatenate([[logH_iso[0]], du0, [np.log(r_d_init)]])
    else:
        theta0 = np.concatenate([logH0, [np.log(r_d_init)]])

    if monotone:
        # logH0 + (K-1) unconstrained increments + log(r_d)
        bounds = [(np.log(10.0), np.log(400.0))] + [(-12.0, 3.0)] * (K - 1) + [(np.log(120.0), np.log(170.0))]
    else:
        bounds = [(np.log(10.0), np.log(400.0))] * K + [(np.log(120.0), np.log(170.0))]
    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Spline optimization failed: {res.message}")

    theta_hat = res.x

    def evaluate_samples(theta_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H_s = np.empty((theta_vecs.shape[0], len(z_grid)))
        dH_s = np.empty_like(H_s)
        r_d_s = np.empty(theta_vecs.shape[0])
        for i, th in enumerate(theta_vecs):
            logH, r_d = decode_theta(th)
            r_d_s[i] = float(r_d)
            sp = PchipInterpolator(z_knots, logH, extrapolate=True)
            logH_g = sp(z_grid)
            H_g = np.exp(logH_g)
            dlogH_dz = sp(z_grid, 1)
            H_s[i] = H_g
            dH_s[i] = H_g * dlogH_dz
        return H_s, dH_s, r_d_s

    # Bootstrap by sampling data realizations
    H_samples = np.empty((n_bootstrap, len(z_grid)))
    dH_dz_samples = np.empty_like(H_samples)
    r_d_boot = np.empty(n_bootstrap)
    bootstrap_success = np.zeros(n_bootstrap, dtype=bool)

    # Precompute Cholesky factors for correlated datasets
    sn_cho, sn_lower = sn_like.cho
    sn_L = np.tril(sn_cho) if sn_lower else np.triu(sn_cho).T
    bao_Ls = []
    for bl in bao_likes:
        cho, lower = bl.cov_cho
        L = np.tril(cho) if lower else np.triu(cho).T
        bao_Ls.append(L)

    for b in range(n_bootstrap):
        # sample data
        cc_H = cc_like.H + cc_like.sigma_H * rng.normal(size=cc_like.H.shape)
        sn_m = sn_like.m + sn_L @ rng.normal(size=sn_like.m.shape)
        bao_ys = []
        for bl, L in zip(bao_likes, bao_Ls, strict=True):
            bao_ys.append(bl.y + L @ rng.normal(size=bl.y.shape))

        # local objective using bootstrapped data
        def loglike_boot(logH: np.ndarray, r_d: float) -> float:
            bg, _ = _bg_from_logH_knots(logH)
            ll = -0.5 * float(np.sum(((cc_H - bg.H(cc_like.z)) / cc_like.sigma_H) ** 2))
            for bl, yb in zip(bao_likes, bao_ys, strict=True):
                y_model = bl.predict(bg, r_d_Mpc=r_d)
                r = yb - y_model
                # use stored covariance cholesky
                from scipy.linalg import cho_solve

                chi2 = float(r @ cho_solve(bl.cov_cho, r, check_finite=False))
                ll += -0.5 * (chi2 + bl.logdet)
            Dl = bg.Dl(sn_like.z)
            mu0 = 5.0 * np.log10(Dl)
            # analytic M with same covariance
            r = sn_m - mu0
            from scipy.linalg import cho_solve

            cinv_r = cho_solve(sn_like.cho, r, check_finite=False)
            b = float(np.ones_like(r) @ cinv_r)
            M_hat = b / sn_like.ones_cinv_ones
            r2 = r - M_hat
            cinv_r2 = cho_solve(sn_like.cho, r2, check_finite=False)
            chi2 = float(r2 @ cinv_r2)
            ll += -0.5 * (chi2 + sn_like.logdet + np.log(sn_like.ones_cinv_ones))
            return ll

        def objective_boot(th: np.ndarray) -> float:
            logH, r_d = decode_theta(th)
            if logH.min() < np.log(10.0) or logH.max() > np.log(400.0):
                return 1e50
            if not (120.0 <= r_d <= 170.0):
                return 1e50
            ll = loglike_boot(logH, r_d)
            if not np.isfinite(ll):
                return 1e50
            d2 = np.diff(logH, n=2)
            return -ll + 0.5 * smooth_lambda * float(np.sum(d2**2))

        res_b = minimize(objective_boot, theta_hat, method="L-BFGS-B", bounds=bounds)
        th_b = res_b.x if res_b.success else theta_hat
        bootstrap_success[b] = bool(res_b.success)
        H_b, dH_b, r_d_b = evaluate_samples(th_b[None, :])
        H_samples[b] = H_b[0]
        dH_dz_samples[b] = dH_b[0]
        r_d_boot[b] = r_d_b[0]

    # Diagnostics for physical plausibility.
    dH = np.diff(H_samples, axis=1)
    monotone_mask = np.all(dH >= -float(monotone_tol), axis=1) if dH.size else np.zeros(n_bootstrap, dtype=bool)
    monotone_fraction = float(np.mean(monotone_mask)) if monotone_mask.size else float("nan")
    # Roughness metric: |Δ^2 H| / |Δ H| (dimensionless)
    d2H = np.diff(H_samples, n=2, axis=1)
    roughness = np.mean(np.abs(d2H), axis=1) / (np.mean(np.abs(dH), axis=1) + 1e-12) if dH.size else np.full(n_bootstrap, np.nan)
    rough_med = float(np.median(roughness)) if roughness.size else float("nan")
    rough_p90 = float(np.percentile(roughness, 90.0)) if roughness.size else float("nan")

    meta = {
        "smooth_lambda": float(smooth_lambda),
        "n_bootstrap": int(n_bootstrap),
        "monotone": bool(monotone),
        "monotone_tol": float(monotone_tol),
        "monotone_fraction": monotone_fraction,
        "roughness_median": rough_med,
        "roughness_p90": rough_p90,
        "bootstrap_success_fraction": float(np.mean(bootstrap_success)) if bootstrap_success.size else float("nan"),
        "r_d_boot_mean": float(np.mean(r_d_boot)),
        "r_d_boot_std": float(np.std(r_d_boot)),
    }
    return SplinePosterior(
        z_knots=z_knots,
        z_grid=z_grid,
        H_samples=H_samples,
        dH_dz_samples=dH_dz_samples,
        meta=meta,
    )
