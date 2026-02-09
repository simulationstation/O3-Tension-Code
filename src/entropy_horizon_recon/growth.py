from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrowthSolution:
    """Growth-factor solution on an ln(a) grid."""

    x_grid: np.ndarray  # ln a, increasing (x<=0)
    D: np.ndarray  # normalized so D(x=0)=1
    Dp: np.ndarray  # dD/dx, normalized consistently with D

    @property
    def f(self) -> np.ndarray:
        """Growth rate f = d ln D / d ln a."""
        return self.Dp / np.clip(self.D, 1e-300, np.inf)


def _lcdm_extension(z: np.ndarray, *, H0: float, omega_m0: float, omega_k0: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (H(z), d ln H / d ln a) using a minimal LCDM-like extension.

    This is used only to set growth initial conditions at high z. We set
    Omega_de0 = max(1 - Omega_m0 - Omega_k0, 0) to avoid negative E^2 when
    priors allow Omega_m0 + Omega_k0 > 1.
    """
    z = np.asarray(z, dtype=float)
    om = float(omega_m0)
    ok = float(omega_k0)
    ode = max(1.0 - om - ok, 0.0)
    zp1 = 1.0 + z
    E2 = om * zp1**3 + ok * zp1**2 + ode
    if np.any(E2 <= 0) or not np.all(np.isfinite(E2)):
        raise ValueError("Non-positive E(z)^2 in LCDM extension.")
    H = float(H0) * np.sqrt(E2)
    dE2_dz = 3.0 * om * zp1**2 + 2.0 * ok * zp1
    dlnH_dz = 0.5 * dE2_dz / E2
    dlnH_dx = -(1.0 + z) * dlnH_dz  # x=ln a = -ln(1+z)
    return H, dlnH_dx


def solve_growth_ode(
    *,
    z_grid: np.ndarray,
    H_grid: np.ndarray,
    H0: float,
    omega_m0: float,
    omega_k0: float = 0.0,
    z_start: float = 50.0,
    n_ext: int = 220,
) -> GrowthSolution:
    """Solve GR linear growth using the inferred background H(z) (background-only).

    We solve for D(x) where x=ln a, using the standard GR growth ODE:

        D'' + [2 + d ln H / d ln a] D' - (3/2) Omega_m(a) D = 0

    with initial conditions at high redshift where D≈a:
        D(x_start) = a_start,  D'(x_start) = D(x_start).

    To avoid relying on H(z) extrapolation beyond the inference domain, we
    prepend an LCDM-like extension only for z>z_max to set initial conditions.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    H_grid = np.asarray(H_grid, dtype=float)
    if z_grid.ndim != 1 or H_grid.ndim != 1 or z_grid.shape != H_grid.shape:
        raise ValueError("z_grid and H_grid must be 1D arrays with matching shape.")
    if z_grid.size < 10 or z_grid[0] != 0.0 or np.any(np.diff(z_grid) <= 0):
        raise ValueError("z_grid must start at 0 and be strictly increasing.")
    if np.any(H_grid <= 0) or not np.all(np.isfinite(H_grid)):
        raise ValueError("H_grid must be positive and finite.")
    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be positive and finite.")
    if not (0.0 < omega_m0 < 1.0) or not np.isfinite(omega_m0):
        raise ValueError("omega_m0 must be in (0,1).")
    if not np.isfinite(omega_k0):
        raise ValueError("omega_k0 must be finite.")

    z_max = float(z_grid[-1])
    z_start_eff = float(max(z_max, float(z_start)))

    # Work in x=ln a, increasing from x_start to 0.
    x_data = -np.log1p(z_grid)[::-1]  # from z_max -> 0 (increasing)
    logH_data = np.log(H_grid[::-1])
    dlogH_dx_data = np.gradient(logH_data, x_data)

    if z_start_eff > z_max and n_ext > 1:
        x_start = -np.log1p(z_start_eff)
        x_switch = float(x_data[0])
        # Exclude endpoint to avoid duplicating x_switch from the data grid.
        x_ext = np.linspace(x_start, x_switch, int(n_ext), endpoint=False)
        z_ext = np.expm1(-x_ext)
        H_ext, dlnH_dx_ext = _lcdm_extension(z_ext, H0=float(H0), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
        x = np.concatenate([x_ext, x_data])
        H = np.concatenate([H_ext, H_grid[::-1]])
        dlnH_dx = np.concatenate([dlnH_dx_ext, dlogH_dx_data])
    else:
        x = x_data
        H = H_grid[::-1]
        dlnH_dx = dlogH_dx_data

    if np.any(np.diff(x) <= 0) or x[-1] != 0.0:
        raise ValueError("Internal growth grid must be strictly increasing and end at x=0.")

    # Initial conditions at high z: D≈a.
    a0 = float(np.exp(x[0]))
    y0 = np.array([a0, a0], dtype=float)  # [D, D']
    y = np.empty((x.size, 2), dtype=float)
    y[0] = y0

    def rhs(xv: float, Dv: float, Dpv: float, Hv: float, dlnH_dxv: float) -> tuple[float, float]:
        a = float(np.exp(xv))
        Om_a = float(omega_m0) * (a**-3) * (float(H0) / float(Hv)) ** 2
        dD = Dpv
        dDp = 1.5 * Om_a * Dv - (2.0 + dlnH_dxv) * Dpv
        return dD, dDp

    # RK4 integrate across the fixed grid (fast, deterministic).
    for i in range(x.size - 1):
        x0 = float(x[i])
        x1 = float(x[i + 1])
        h = x1 - x0
        if not (h > 0 and np.isfinite(h)):
            raise ValueError("Invalid growth step.")
        xm = x0 + 0.5 * h
        H0i = float(H[i])
        H1i = float(H[i + 1])
        Hm = 0.5 * (H0i + H1i)
        d0 = float(dlnH_dx[i])
        d1 = float(dlnH_dx[i + 1])
        dm = 0.5 * (d0 + d1)

        D, Dp = float(y[i, 0]), float(y[i, 1])
        k1 = np.array(rhs(x0, D, Dp, H0i, d0), dtype=float)
        k2 = np.array(rhs(xm, D + 0.5 * h * k1[0], Dp + 0.5 * h * k1[1], Hm, dm), dtype=float)
        k3 = np.array(rhs(xm, D + 0.5 * h * k2[0], Dp + 0.5 * h * k2[1], Hm, dm), dtype=float)
        k4 = np.array(rhs(x1, D + h * k3[0], Dp + h * k3[1], H1i, d1), dtype=float)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    D0 = float(y[-1, 0])
    if not np.isfinite(D0) or D0 <= 0:
        raise ValueError("Non-physical growth normalization at z=0.")
    D = y[:, 0] / D0
    Dp = y[:, 1] / D0
    return GrowthSolution(x_grid=x, D=D, Dp=Dp)


def solve_growth_ode_muP(
    *,
    z_grid: np.ndarray,
    H_grid: np.ndarray,
    H0: float,
    omega_m0: float,
    muP_grid: np.ndarray,
    omega_k0: float = 0.0,
    z_start: float = 50.0,
    n_ext: int = 220,
    muP_highz: float = 1.0,
) -> GrowthSolution:
    """Solve linear growth with a scale-independent modified Poisson strength muP(z).

    This is the minimal extension used by the "void prism" / E_G style tests.

    We solve for D(x) where x=ln a using:

        D'' + [2 + d ln H / d ln a] D' - (3/2) Omega_m(a) * muP(a) * D = 0

    Inputs:
      - (z_grid, H_grid): inferred background grid (must start at z=0).
      - muP_grid: values of muP(z) on the same z_grid. muP is treated as a multiplicative
        rescaling of the GR source term in the growth ODE.

    High-z handling:
      Growth initial conditions require extending above the inference z-range. For z > z_max we
      set muP=muP_highz (default 1) so the extension does not assume unknown early-time physics.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    H_grid = np.asarray(H_grid, dtype=float)
    muP_grid = np.asarray(muP_grid, dtype=float)
    if z_grid.ndim != 1 or H_grid.ndim != 1 or muP_grid.ndim != 1:
        raise ValueError("z_grid, H_grid, muP_grid must be 1D arrays.")
    if z_grid.shape != H_grid.shape or z_grid.shape != muP_grid.shape:
        raise ValueError("z_grid, H_grid, muP_grid must have matching shapes.")
    if z_grid.size < 10 or z_grid[0] != 0.0 or np.any(np.diff(z_grid) <= 0):
        raise ValueError("z_grid must start at 0 and be strictly increasing.")
    if np.any(H_grid <= 0) or not np.all(np.isfinite(H_grid)):
        raise ValueError("H_grid must be positive and finite.")
    if not np.all(np.isfinite(muP_grid)) or np.any(muP_grid <= 0):
        raise ValueError("muP_grid must be positive and finite.")
    if not np.isfinite(muP_highz) or muP_highz <= 0:
        raise ValueError("muP_highz must be positive and finite.")
    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be positive and finite.")
    if not (0.0 < omega_m0 < 1.0) or not np.isfinite(omega_m0):
        raise ValueError("omega_m0 must be in (0,1).")
    if not np.isfinite(omega_k0):
        raise ValueError("omega_k0 must be finite.")

    z_max = float(z_grid[-1])
    z_start_eff = float(max(z_max, float(z_start)))

    # Work in x=ln a, increasing from x_start to 0.
    x_data = -np.log1p(z_grid)[::-1]  # from z_max -> 0 (increasing)
    logH_data = np.log(H_grid[::-1])
    dlogH_dx_data = np.gradient(logH_data, x_data)
    muP_data = muP_grid[::-1]

    if z_start_eff > z_max and n_ext > 1:
        x_start = -np.log1p(z_start_eff)
        x_switch = float(x_data[0])
        # Exclude endpoint to avoid duplicating x_switch from the data grid.
        x_ext = np.linspace(x_start, x_switch, int(n_ext), endpoint=False)
        z_ext = np.expm1(-x_ext)
        H_ext, dlnH_dx_ext = _lcdm_extension(z_ext, H0=float(H0), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
        muP_ext = np.full_like(x_ext, float(muP_highz))

        x = np.concatenate([x_ext, x_data])
        H = np.concatenate([H_ext, H_grid[::-1]])
        dlnH_dx = np.concatenate([dlnH_dx_ext, dlogH_dx_data])
        muP = np.concatenate([muP_ext, muP_data])
    else:
        x = x_data
        H = H_grid[::-1]
        dlnH_dx = dlogH_dx_data
        muP = muP_data

    if np.any(np.diff(x) <= 0) or x[-1] != 0.0:
        raise ValueError("Internal growth grid must be strictly increasing and end at x=0.")

    # Initial conditions at high z: D≈a.
    a0 = float(np.exp(x[0]))
    y0 = np.array([a0, a0], dtype=float)  # [D, D']
    y = np.empty((x.size, 2), dtype=float)
    y[0] = y0

    def rhs(xv: float, Dv: float, Dpv: float, Hv: float, dlnH_dxv: float, muPv: float) -> tuple[float, float]:
        a = float(np.exp(xv))
        Om_a = float(omega_m0) * (a**-3) * (float(H0) / float(Hv)) ** 2
        dD = Dpv
        dDp = 1.5 * Om_a * float(muPv) * Dv - (2.0 + dlnH_dxv) * Dpv
        return dD, dDp

    # RK4 integrate across the fixed grid (fast, deterministic).
    for i in range(x.size - 1):
        x0 = float(x[i])
        x1 = float(x[i + 1])
        h = x1 - x0
        if not (h > 0 and np.isfinite(h)):
            raise ValueError("Invalid growth step.")
        xm = x0 + 0.5 * h
        H0i = float(H[i])
        H1i = float(H[i + 1])
        Hm = 0.5 * (H0i + H1i)
        d0 = float(dlnH_dx[i])
        d1 = float(dlnH_dx[i + 1])
        dm = 0.5 * (d0 + d1)
        mu0 = float(muP[i])
        mu1 = float(muP[i + 1])
        mum = 0.5 * (mu0 + mu1)

        D, Dp = float(y[i, 0]), float(y[i, 1])
        k1 = np.array(rhs(x0, D, Dp, H0i, d0, mu0), dtype=float)
        k2 = np.array(rhs(xm, D + 0.5 * h * k1[0], Dp + 0.5 * h * k1[1], Hm, dm, mum), dtype=float)
        k3 = np.array(rhs(xm, D + 0.5 * h * k2[0], Dp + 0.5 * h * k2[1], Hm, dm, mum), dtype=float)
        k4 = np.array(rhs(x1, D + h * k3[0], Dp + h * k3[1], H1i, d1, mu1), dtype=float)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    D0 = float(y[-1, 0])
    if not np.isfinite(D0) or D0 <= 0:
        raise ValueError("Non-physical growth normalization at z=0.")
    D = y[:, 0] / D0
    Dp = y[:, 1] / D0
    return GrowthSolution(x_grid=x, D=D, Dp=Dp)


def predict_fsigma8(
    *,
    z_eval: np.ndarray,
    z_grid: np.ndarray,
    H_grid: np.ndarray,
    H0: float,
    omega_m0: float,
    sigma8_0: float,
    omega_k0: float = 0.0,
) -> np.ndarray:
    """Predict fσ8(z) in GR using the reconstructed background H(z)."""
    z_eval = np.asarray(z_eval, dtype=float)
    if z_eval.ndim != 1:
        raise ValueError("z_eval must be 1D.")
    if not np.isfinite(sigma8_0) or sigma8_0 <= 0:
        raise ValueError("sigma8_0 must be positive and finite.")

    sol = solve_growth_ode(z_grid=z_grid, H_grid=H_grid, H0=H0, omega_m0=omega_m0, omega_k0=omega_k0)
    x_eval = -np.log1p(z_eval)
    D_eval = np.interp(x_eval, sol.x_grid, sol.D)
    f_eval = np.interp(x_eval, sol.x_grid, sol.f)
    return f_eval * float(sigma8_0) * D_eval


def S8_from_sigma8_omega_m0(*, sigma8_0: float, omega_m0: float) -> float:
    """Return the common late-time clustering summary S8."""
    return float(sigma8_0) * float(np.sqrt(float(omega_m0) / 0.3))
