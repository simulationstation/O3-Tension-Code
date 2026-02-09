from __future__ import annotations

import numpy as np

from ..constants import PhysicalConstants


def delta_mu_from_lensing(delta_dl_over_dl: np.ndarray) -> np.ndarray:
    return (5.0 / np.log(10.0)) * delta_dl_over_dl


def delta_dl_over_dl(kappa: np.ndarray, gamma2: np.ndarray | None = None) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=float)
    if gamma2 is None:
        return -kappa
    g2 = np.asarray(gamma2, dtype=float)
    return -kappa - 0.5 * g2


def inject_mu(mu: np.ndarray, kappa: np.ndarray, gamma2: np.ndarray | None = None) -> np.ndarray:
    d = delta_dl_over_dl(kappa, gamma2)
    return mu + delta_mu_from_lensing(d)


def estimate_delta_h0_over_h0(mu_obs: np.ndarray, mu_model: np.ndarray, weights: np.ndarray) -> float:
    mu_obs = np.asarray(mu_obs, dtype=float)
    mu_model = np.asarray(mu_model, dtype=float)
    w = np.asarray(weights, dtype=float)
    if mu_obs.shape != mu_model.shape or mu_obs.shape != w.shape:
        raise ValueError("mu_obs, mu_model, weights must have same shape")
    resid = mu_obs - mu_model
    # dmu/dlnH0 = -5/ln10
    dlnH0 = - (np.log(10.0) / 5.0) * np.sum(w * resid)
    return float(dlnH0)
