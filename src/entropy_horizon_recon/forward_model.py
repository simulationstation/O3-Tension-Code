from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import PhysicalConstants
from .mapping import forward_H_from_muA


def mu_of_A(logA: np.ndarray, theta: dict) -> np.ndarray:
    """Evaluate μ(A) from a spline parameterization in logA.

    Parameters
    ----------
    logA:
        log horizon area values.
    theta:
        Must contain:
          - "x_knots": array of x=log(A/A0) knots (ascending)
          - "logmu_knots": array of logμ values at those knots
          - "logA0": reference logA0 corresponding to z=0 (A0 = 4π(c/H0)^2)
    """
    x_knots = np.asarray(theta["x_knots"], dtype=float)
    y_knots = np.asarray(theta["logmu_knots"], dtype=float)
    logA0 = float(theta["logA0"])
    x = np.asarray(logA, dtype=float) - logA0
    spline = make_spline(x_knots, y_knots)
    return np.exp(spline(clamp(x, x_knots[0], x_knots[-1])))


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.minimum(np.maximum(x, lo), hi)


def make_spline(x_knots: np.ndarray, y_knots: np.ndarray) -> CubicSpline:
    x_knots = np.asarray(x_knots, dtype=float)
    y_knots = np.asarray(y_knots, dtype=float)
    if x_knots.ndim != 1 or y_knots.ndim != 1 or x_knots.shape != y_knots.shape:
        raise ValueError("x_knots and y_knots must be 1D arrays with matching shape.")
    if np.any(np.diff(x_knots) <= 0):
        raise ValueError("x_knots must be strictly increasing.")
    return CubicSpline(x_knots, y_knots, bc_type="natural", extrapolate=True)


def H_of_z_from_mu(
    z_grid: np.ndarray,
    mu_grid: np.ndarray,
    *,
    H0_km_s_Mpc: float,
    omega_m0: float,
    z0: float = 0.0,
) -> np.ndarray:
    """Integrate H(z) from μ(z) using the integral formulation.

    d(H^2)/dz = 3 H0^2 Ωm0 (1+z)^2 μ(z)

    This function is used for diagnostics; the primary forward model in this
    repository uses μ(A) and integrates the ODE consistently with A(H).
    """
    z = np.asarray(z_grid, dtype=float)
    mu = np.asarray(mu_grid, dtype=float)
    if z.ndim != 1 or mu.ndim != 1 or z.shape != mu.shape:
        raise ValueError("z_grid and mu_grid must be 1D arrays with the same shape.")
    if z[0] != z0:
        raise ValueError("z_grid must start at z0.")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z_grid must be strictly increasing.")
    if H0_km_s_Mpc <= 0:
        raise ValueError("H0 must be positive.")
    if not (0.0 < omega_m0 < 1.0):
        raise ValueError("omega_m0 must be in (0,1).")
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
        raise ValueError("mu_grid must be positive and finite.")

    K = 3.0 * (H0_km_s_Mpc**2) * omega_m0 * (1.0 + z) ** 2
    # integrate u(z)=H^2(z)
    u = np.empty_like(z)
    u[0] = H0_km_s_Mpc**2
    dz = np.diff(z)
    u[1:] = u[0] + np.cumsum(0.5 * dz * (K[:-1] * mu[:-1] + K[1:] * mu[1:]))
    if np.any(u <= 0):
        raise ValueError("Integrated H^2 became non-positive.")
    return np.sqrt(u)


@dataclass(frozen=True)
class ForwardModel:
    constants: PhysicalConstants
    x_knots: np.ndarray

    def solve_H_from_logmu_knots(
        self,
        z_grid: np.ndarray,
        *,
        logmu_knots: np.ndarray,
        H0_km_s_Mpc: float,
        omega_m0: float,
        omega_k0: float = 0.0,
        residual_of_z=None,
    ) -> np.ndarray:
        """Solve H(z) using μ(A) defined by logμ(x) with x=log(A/A0)."""
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.ndim != 1 or z_grid.size < 2:
            raise ValueError("z_grid must be 1D with at least 2 points.")
        if z_grid[0] != 0.0:
            raise ValueError("z_grid must start at 0 for H0.")

        const_A = 4.0 * np.pi * (self.constants.c_km_s**2)
        denom0 = (H0_km_s_Mpc**2) * (1.0 - float(omega_k0))
        if denom0 <= 0 or not np.isfinite(denom0):
            raise ValueError("Invalid A0 mapping: require 1 - omega_k0 > 0.")
        A0 = const_A / denom0
        logA0 = float(np.log(A0))
        spline = make_spline(self.x_knots, np.asarray(logmu_knots, dtype=float))

        def mu_of_A(A: float) -> float:
            logA = np.log(A)
            x = float(np.clip(logA - logA0, self.x_knots[0], self.x_knots[-1]))
            return float(np.exp(spline(x)))

        return forward_H_from_muA(
            z_grid,
            mu_of_A=mu_of_A,
            H0_km_s_Mpc=H0_km_s_Mpc,
            omega_m0=omega_m0,
            constants=self.constants,
            omega_k0=float(omega_k0),
            residual_of_z=residual_of_z,
        )
