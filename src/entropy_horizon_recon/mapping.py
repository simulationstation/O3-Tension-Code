from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import PhysicalConstants


@dataclass(frozen=True)
class HorizonMapping:
    z: np.ndarray
    A: np.ndarray
    mu: np.ndarray


def H_to_area(H_km_s_Mpc: np.ndarray, *, constants: PhysicalConstants) -> np.ndarray:
    """Apparent-horizon area in a flat FLRW universe: A = 4π (c/H)^2."""
    return 4.0 * np.pi * (constants.c_km_s / H_km_s_Mpc) ** 2


def mu_from_H_and_dHdz(
    H_km_s_Mpc: np.ndarray,
    dH_dz_km_s_Mpc: np.ndarray,
    z: np.ndarray,
    *,
    H0_km_s_Mpc: float,
    omega_m0: float,
) -> np.ndarray:
    """Phenomenological slope modification μ(A) from Cai–Kim-style Clausius mapping.

    Using μ = (dS/dA)_BH / (dS/dA) and, assuming (ρ+p) is matter-dominated,

      d(H^2)/dz = 3 H0^2 Ω_m0 (1+z)^2 μ .
    """
    H2 = H_km_s_Mpc**2
    dH2_dz = 2.0 * H_km_s_Mpc * dH_dz_km_s_Mpc
    denom = 3.0 * (H0_km_s_Mpc**2) * omega_m0 * (1.0 + z) ** 2
    return dH2_dz / denom


def mapping_from_H_draw(
    z: np.ndarray,
    H_km_s_Mpc: np.ndarray,
    dH_dz_km_s_Mpc: np.ndarray,
    *,
    omega_m0: float,
    constants: PhysicalConstants,
) -> HorizonMapping:
    """Compute (A(z), μ(z)) for one reconstructed H(z) draw."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("z must be 1D.")
    if z.size != H_km_s_Mpc.size or z.size != dH_dz_km_s_Mpc.size:
        raise ValueError("z, H_km_s_Mpc, and dH_dz_km_s_Mpc must have the same length.")
    if z[0] != 0.0:
        raise ValueError("z must start at 0 to define H0 consistently.")
    H0 = float(H_km_s_Mpc[0])
    A = H_to_area(H_km_s_Mpc, constants=constants)
    mu = mu_from_H_and_dHdz(
        H_km_s_Mpc,
        dH_dz_km_s_Mpc,
        z,
        H0_km_s_Mpc=H0,
        omega_m0=omega_m0,
    )
    return HorizonMapping(z=z, A=A, mu=mu)


def forward_H_from_muA(
    z_grid: np.ndarray,
    *,
    mu_of_A,
    H0_km_s_Mpc: float,
    omega_m0: float,
    constants: PhysicalConstants,
    omega_k0: float = 0.0,
    residual_of_z=None,
) -> np.ndarray:
    """Forward model H(z) from μ(A) via du/dz = 3 H0^2 Ωm0 (1+z)^2 μ(A) + R(z).

    The apparent-horizon area mapping supports an optional curvature nuisance Ω_k0:

      r_A(z) = c / sqrt(H(z)^2 - H0^2 Ω_k0 (1+z)^2)
      A(z) = 4π r_A(z)^2

    Parameters
    ----------
    omega_k0:
        Present-day curvature density parameter (Ω_k0). Default is 0 (flat).
    residual_of_z:
        Optional callable returning the residual term R(z) in the same units as d(H^2)/dz.
        This is used as a controlled 'closure error' term in mapping sensitivity studies.
    """
    z = np.asarray(z_grid, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_grid must be a 1D array with at least 2 points.")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z_grid must be strictly increasing.")
    if z[0] != 0.0:
        raise ValueError("z_grid must start at z=0 for the forward integration.")
    if not (0.0 < omega_m0 < 1.0):
        raise ValueError("omega_m0 must be in (0,1).")
    if H0_km_s_Mpc <= 0:
        raise ValueError("H0_km_s_Mpc must be positive.")
    if not np.isfinite(float(omega_k0)):
        raise ValueError("omega_k0 must be finite.")

    H = np.empty_like(z)
    u = float(H0_km_s_Mpc**2)
    H[0] = H0_km_s_Mpc

    const_A = 4.0 * np.pi * (constants.c_km_s**2)

    def rhs(zv: float, uv: float) -> float:
        denom = uv - (H0_km_s_Mpc**2) * float(omega_k0) * (1.0 + zv) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Apparent-horizon mapping produced invalid denominator during integration.")
        A = const_A / denom
        mu = float(mu_of_A(A))
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu_of_A returned non-positive or non-finite value during integration.")
        du = 3.0 * (H0_km_s_Mpc**2) * omega_m0 * (1.0 + zv) ** 2 * mu
        if residual_of_z is not None:
            du = du + float(residual_of_z(float(zv)))
        return float(du)

    # RK4 over provided grid
    for i in range(1, len(z)):
        z0 = float(z[i - 1])
        z1 = float(z[i])
        dz = z1 - z0

        k1 = rhs(z0, u)
        k2 = rhs(z0 + 0.5 * dz, u + 0.5 * dz * k1)
        k3 = rhs(z0 + 0.5 * dz, u + 0.5 * dz * k2)
        k4 = rhs(z0 + dz, u + dz * k3)
        u = u + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if u <= 0 or not np.isfinite(u):
            raise ValueError("Forward integration produced invalid H^2.")
        H[i] = np.sqrt(u)

    return H


def fit_logmu_spline(
    logA: np.ndarray,
    logmu: np.ndarray,
    *,
    smooth: float = 0.0,
) -> CubicSpline:
    """Fit a cubic spline to logμ(logA) for a single draw.

    `smooth` is currently a no-op placeholder for future regularized splines; the
    returned spline is an interpolating natural cubic spline.
    """
    _ = smooth
    order = np.argsort(logA)
    x = np.asarray(logA, dtype=float)[order]
    y = np.asarray(logmu, dtype=float)[order]
    # drop non-finite points
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        raise ValueError("Not enough finite points to fit spline.")
    # remove duplicate x values (keep first)
    uniq, idx = np.unique(x, return_index=True)
    x, y = uniq, y[idx]
    return CubicSpline(x, y, bc_type="natural", extrapolate=True)
