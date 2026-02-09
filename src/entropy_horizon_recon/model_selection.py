from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid
from .inversion import ForwardMuPosterior


@dataclass(frozen=True)
class WaicResult:
    waic: float
    lppd: float
    p_waic: float
    n_points: int


def waic_from_loglik(loglik: np.ndarray) -> WaicResult:
    """Compute a (block) WAIC from per-point log-likelihood values.

    Notes
    -----
    WAIC assumes conditionally independent 'points'. For correlated data (e.g. SN/BAO
    with full covariance), treat each correlated block as one 'point' and interpret
    WAIC as a rough model-selection heuristic (not a strict evidence).
    """
    ll = np.asarray(loglik, dtype=float)
    if ll.ndim != 2:
        raise ValueError("loglik must have shape (n_draws, n_points).")
    if ll.shape[0] < 5:
        raise ValueError("Need at least 5 posterior draws for WAIC.")
    # Numerically stable log-mean-exp per point.
    m = np.max(ll, axis=0)
    lppd = float(np.sum(m + np.log(np.mean(np.exp(ll - m), axis=0))))
    p_waic = float(np.sum(np.var(ll, axis=0, ddof=1)))
    waic = float(-2.0 * (lppd - p_waic))
    return WaicResult(waic=waic, lppd=lppd, p_waic=p_waic, n_points=int(ll.shape[1]))


def loglik_blocks_for_forward_posterior(
    post: ForwardMuPosterior,
    *,
    sn_z: np.ndarray,
    sn_m: np.ndarray,
    sn_cov: np.ndarray,
    cc_z: np.ndarray,
    cc_H: np.ndarray,
    cc_sigma_H: np.ndarray,
    bao_likes,
    constants: PhysicalConstants,
) -> tuple[np.ndarray, list[str]]:
    """Compute a blockwise log-likelihood matrix for a forward-model posterior.

    Returns
    -------
    loglik:
        Array of shape (n_draws, n_blocks) where blocks are:
          - SN (full covariance, M marginalized) as one block
          - each CC point as one block
          - each BAO dataset-like object as one block
    names:
        Block names matching the columns of `loglik`.
    """
    sn_z = np.asarray(sn_z, dtype=float)
    sn_m = np.asarray(sn_m, dtype=float)
    sn_cov = np.asarray(sn_cov, dtype=float)
    cc_z = np.asarray(cc_z, dtype=float)
    cc_H = np.asarray(cc_H, dtype=float)
    cc_sigma_H = np.asarray(cc_sigma_H, dtype=float)
    if sn_cov.shape != (sn_z.size, sn_z.size):
        raise ValueError("sn_cov shape mismatch.")
    if sn_m.shape != sn_z.shape:
        raise ValueError("sn_m shape mismatch.")
    if cc_z.shape != cc_H.shape or cc_z.shape != cc_sigma_H.shape:
        raise ValueError("CC arrays shape mismatch.")

    # Precompute SN eigen decomposition for diagonal-jitter loglike.
    evals, evecs = np.linalg.eigh(sn_cov)
    if np.any(evals <= 0):
        raise ValueError("SN covariance must be positive definite.")
    ones_sn = np.ones_like(sn_m)
    Qt_ones = evecs.T @ ones_sn

    def sn_loglike(mu0: np.ndarray, sigma_jit: float) -> float:
        if sigma_jit < 0 or not np.isfinite(sigma_jit):
            return -np.inf
        r = sn_m - mu0
        lam = evals + sigma_jit**2
        inv_lam = 1.0 / lam
        Qt_r = evecs.T @ r
        cinv_r = evecs @ (inv_lam * Qt_r)
        cinv_ones = evecs @ (inv_lam * Qt_ones)
        ones_cinv_ones = float(ones_sn @ cinv_ones)
        if ones_cinv_ones <= 0:
            return -np.inf
        b = float(ones_sn @ cinv_r)
        M_hat = b / ones_cinv_ones
        r2 = r - M_hat
        Qt_r2 = evecs.T @ r2
        cinv_r2 = evecs @ (inv_lam * Qt_r2)
        chi2 = float(r2 @ cinv_r2)
        logdet = float(np.sum(np.log(lam)))
        return float(-0.5 * (chi2 + logdet + np.log(ones_cinv_ones)))

    n_draws = int(post.logmu_x_samples.shape[0])
    n_cc = int(cc_z.size)
    n_bao = int(len(bao_likes))
    n_blocks = 1 + n_cc + n_bao
    out = np.empty((n_draws, n_blocks), dtype=float)
    names = ["SN_margM"] + [f"CC[{i}]" for i in range(n_cc)] + [f"BAO[{i}]" for i in range(n_bao)]

    # Pull per-draw nuisance parameters.
    sig_cc = np.asarray(post.params.get("sigma_cc_jit"), dtype=float)
    sig_sn = np.asarray(post.params.get("sigma_sn_jit"), dtype=float)
    r_d = np.asarray(post.params.get("r_d_Mpc"), dtype=float)
    if sig_cc.shape != (n_draws,) or sig_sn.shape != (n_draws,) or r_d.shape != (n_draws,):
        raise ValueError("Posterior params missing or shape mismatch for WAIC.")

    for j in range(n_draws):
        bg = build_background_from_H_grid(post.z_grid, post.H_samples[j], constants=constants)

        Dl = bg.Dl(sn_z)
        if np.any(Dl <= 0) or not np.all(np.isfinite(Dl)):
            out[j] = -np.inf
            continue
        mu0 = 5.0 * np.log10(Dl)
        ll_sn = sn_loglike(mu0, float(sig_sn[j]))

        H_cc = bg.H(cc_z)
        sig_eff = np.sqrt(cc_sigma_H**2 + float(sig_cc[j]) ** 2)
        ll_cc = -0.5 * (((cc_H - H_cc) / sig_eff) ** 2 + 2.0 * np.log(sig_eff))

        ll_bao = []
        for bl in bao_likes:
            y_model = bl.predict(bg, r_d_Mpc=float(r_d[j]))
            ll_bao.append(float(bl.loglike(y_model)))

        out[j, 0] = float(ll_sn)
        out[j, 1 : 1 + n_cc] = ll_cc
        out[j, 1 + n_cc :] = ll_bao

    return out, names

