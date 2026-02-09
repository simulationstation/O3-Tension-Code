from __future__ import annotations

import numpy as np

from ..constants import PhysicalConstants
from ..cosmology import build_background_from_H_grid


def fiducial_mu(
    z: np.ndarray,
    *,
    H0: float = 70.0,
    omega_m0: float = 0.3,
    n_grid: int = 600,
) -> np.ndarray:
    """Compute fiducial distance modulus for flat LCDM.

    Parameters
    ----------
    z: redshift array
    H0: Hubble constant [km/s/Mpc]
    omega_m0: matter density
    n_grid: grid size for distance integration
    """
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    if zmax <= 0:
        return np.zeros_like(z)
    zg = np.linspace(0.0, zmax, int(n_grid))
    H = H0 * np.sqrt(omega_m0 * (1.0 + zg) ** 3 + (1.0 - omega_m0))
    bg = build_background_from_H_grid(zg, H, constants=PhysicalConstants())
    Dl = bg.Dl(z)
    return 5.0 * np.log10(Dl)


def residuals_mu(
    mu_obs: np.ndarray,
    z: np.ndarray,
    *,
    H0: float = 70.0,
    omega_m0: float = 0.3,
) -> np.ndarray:
    mu_obs = np.asarray(mu_obs, dtype=float)
    mu0 = fiducial_mu(z, H0=H0, omega_m0=omega_m0)
    return mu_obs - mu0


def detrend_residuals(
    z: np.ndarray,
    residuals: np.ndarray,
    *,
    mode: str = "none",
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if mode == "none":
        return r
    if mode not in {"poly1", "poly2"}:
        raise ValueError(f"Unknown detrend mode '{mode}'.")
    deg = 1 if mode == "poly1" else 2
    coeff = np.polyfit(z, r, deg=deg)
    trend = np.polyval(coeff, z)
    return r - trend
