from __future__ import annotations

import numpy as np


def inverse_variance_weights(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    w = np.zeros_like(sigma)
    good = sigma > 0
    w[good] = 1.0 / (sigma[good] ** 2)
    if np.sum(w) <= 0:
        raise ValueError("No valid weights.")
    return w / np.sum(w)


def h0_estimator_weights(
    z: np.ndarray,
    sigma_mu: np.ndarray,
    *,
    z_min: float = 0.023,
    z_max: float = 0.15,
) -> np.ndarray:
    """Weights for a low-z H0 estimator based on distance modulus sensitivity.

    For a simple H0-only fit, dmu/dlnH0 = -5/ln(10) constant, so weights reduce
    to inverse-variance with a redshift cut.
    """
    z = np.asarray(z, dtype=float)
    sigma_mu = np.asarray(sigma_mu, dtype=float)
    sel = (z >= z_min) & (z <= z_max)
    w = np.zeros_like(z)
    w[sel] = 1.0 / (sigma_mu[sel] ** 2)
    if np.sum(w) <= 0:
        raise ValueError("No valid weights after redshift cut.")
    return w / np.sum(w)
