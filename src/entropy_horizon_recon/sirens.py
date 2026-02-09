from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid


@dataclass(frozen=True)
class MuForwardPosterior:
    """Posterior artifacts saved by run_realdata_recon.py for EM-only inference.

    This file is produced by the pipeline for finished runs as:
      <run_dir>/samples/mu_forward_posterior.npz
    """

    x_grid: np.ndarray
    logmu_x_samples: np.ndarray  # (n_draws, n_x)
    z_grid: np.ndarray
    H_samples: np.ndarray  # (n_draws, n_z)
    H0: np.ndarray  # (n_draws,)
    omega_m0: np.ndarray  # (n_draws,)
    omega_k0: np.ndarray  # (n_draws,)
    sigma8_0: np.ndarray | None = None  # (n_draws,) if present


def load_mu_forward_posterior(run_dir: str | Path) -> MuForwardPosterior:
    run_dir = Path(run_dir)
    npz_path = run_dir / "samples" / "mu_forward_posterior.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing posterior artifact: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as d:
        sigma8_0 = np.asarray(d["sigma8_0"], dtype=float) if "sigma8_0" in d.files else None
        return MuForwardPosterior(
            x_grid=np.asarray(d["x_grid"], dtype=float),
            logmu_x_samples=np.asarray(d["logmu_x_samples"], dtype=float),
            z_grid=np.asarray(d["z_grid"], dtype=float),
            H_samples=np.asarray(d["H_samples"], dtype=float),
            H0=np.asarray(d["H0"], dtype=float),
            omega_m0=np.asarray(d["omega_m0"], dtype=float),
            omega_k0=np.asarray(d["omega_k0"], dtype=float),
            sigma8_0=sigma8_0,
        )


def x_of_z_from_H(
    z: np.ndarray,
    H: np.ndarray,
    *,
    H0: float,
    omega_k0: float,
) -> np.ndarray:
    """Compute x(z)=log(A(z)/A0) using the pipeline's apparent-horizon area mapping.

    Using A(z) = 4*pi*c^2 / (H(z)^2 - Ok H0^2 (1+z)^2), the constant 4*pi*c^2 cancels in A/A0.
    """
    z = np.asarray(z, dtype=float)
    H = np.asarray(H, dtype=float)
    if z.shape != H.shape:
        raise ValueError("z and H must have the same shape.")
    denom0 = H0**2 * (1.0 - omega_k0)
    denom = H**2 - omega_k0 * H0**2 * (1.0 + z) ** 2
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("Non-physical horizon area mapping: denom <= 0.")
    if denom0 <= 0.0 or not np.isfinite(denom0):
        raise ValueError("Non-physical horizon area mapping at z=0: denom0 <= 0.")
    return np.log(denom0 / denom)


def _interp_logmu_rows(
    *,
    x_eval: np.ndarray,  # (n_draws, n_eval)
    x_grid: np.ndarray,  # (n_x,)
    logmu_x_samples: np.ndarray,  # (n_draws, n_x)
    allow_extrapolation: bool,
) -> np.ndarray:
    n_draws, n_eval = x_eval.shape
    out = np.empty((n_draws, n_eval), dtype=float)
    xmin, xmax = float(x_grid[0]), float(x_grid[-1])
    tol = 1e-12 * max(1.0, abs(xmin), abs(xmax))
    for j in range(n_draws):
        xj = x_eval[j]
        if not allow_extrapolation and (np.min(xj) < xmin - tol or np.max(xj) > xmax + tol):
            raise ValueError(
                "Requested z-grid maps outside inferred x-domain. "
                f"x_eval in [{np.min(xj):.3g},{np.max(xj):.3g}], "
                f"x_grid in [{xmin:.3g},{xmax:.3g}]."
            )
        if not allow_extrapolation:
            xj = np.clip(xj, xmin, xmax)
        out[j] = np.interp(xj, x_grid, logmu_x_samples[j])
    return out


def predict_r_gw_em(
    post: MuForwardPosterior,
    *,
    z_eval: np.ndarray | None = None,
    convention: Literal["A", "B"] = "A",
    allow_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict R_GW/EM(z) = dL_GW/dL_EM from a mu(A) posterior.

    Conventions (see siren_test.md):
      - "A" (default): R = sqrt(mu(z)/mu(0))
      - "B":           R = sqrt(mu(0)/mu(z))
    """
    if convention not in ("A", "B"):
        raise ValueError("convention must be 'A' or 'B'.")

    if z_eval is None:
        z_eval = post.z_grid
        H_eval = post.H_samples
    else:
        z_eval = np.asarray(z_eval, dtype=float)
        if z_eval.ndim != 1:
            raise ValueError("z_eval must be a 1D array.")
        if np.any(np.diff(z_eval) <= 0):
            raise ValueError("z_eval must be strictly increasing.")
        if not allow_extrapolation and (z_eval[0] < post.z_grid[0] or z_eval[-1] > post.z_grid[-1]):
            raise ValueError(
                "z_eval outside posterior z_grid range. "
                f"z_eval in [{z_eval[0]:.3g},{z_eval[-1]:.3g}], "
                f"z_grid in [{post.z_grid[0]:.3g},{post.z_grid[-1]:.3g}]."
            )
        # Interpolate each draw onto the requested z_eval grid.
        H_eval = np.vstack([np.interp(z_eval, post.z_grid, post.H_samples[j]) for j in range(post.H_samples.shape[0])])

    n_draws = post.H_samples.shape[0]
    z_eval = np.asarray(z_eval, dtype=float)
    H0 = post.H0.reshape((n_draws, 1))
    ok = post.omega_k0.reshape((n_draws, 1))

    denom0 = H0**2 * (1.0 - ok)
    denom = H_eval**2 - ok * H0**2 * (1.0 + z_eval.reshape((1, -1))) ** 2
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("Non-physical horizon area mapping: denom <= 0.")
    if not np.all(np.isfinite(denom0)) or np.any(denom0 <= 0.0):
        raise ValueError("Non-physical horizon area mapping at z=0: denom0 <= 0.")
    x_eval = np.log(denom0 / denom)

    logmu_eval = _interp_logmu_rows(
        x_eval=x_eval,
        x_grid=post.x_grid,
        logmu_x_samples=post.logmu_x_samples,
        allow_extrapolation=allow_extrapolation,
    )
    mu_eval = np.exp(logmu_eval)

    # Prefer exact grid point if present; otherwise fall back to interpolation.
    if np.isclose(post.x_grid[-1], 0.0):
        mu0 = np.exp(post.logmu_x_samples[:, -1])
    else:
        mu0 = np.exp(np.array([np.interp(0.0, post.x_grid, post.logmu_x_samples[j]) for j in range(n_draws)]))

    if convention == "A":
        R = np.sqrt(mu_eval / mu0.reshape((n_draws, 1)))
    else:
        R = np.sqrt(mu0.reshape((n_draws, 1)) / mu_eval)
    return z_eval, R


def predict_dL_em(
    post: MuForwardPosterior,
    *,
    z_eval: np.ndarray,
    constants: PhysicalConstants | None = None,
) -> np.ndarray:
    """Compute EM luminosity distance draws from the posterior H(z).

    Uses standard FRW distances with curvature handled via omega_k0 when nonzero.
    """
    constants = constants or PhysicalConstants()
    z_eval = np.asarray(z_eval, dtype=float)
    n_draws = post.H_samples.shape[0]
    out = np.empty((n_draws, z_eval.size), dtype=float)
    for j in range(n_draws):
        bg = build_background_from_H_grid(post.z_grid, post.H_samples[j], constants=constants)
        Dc = bg.Dc(z_eval)
        ok = float(post.omega_k0[j])
        H0 = float(post.H0[j])
        if ok == 0.0:
            Dm = Dc
        elif ok > 0.0:
            sk = np.sqrt(ok) * (H0 * Dc / constants.c_km_s)
            Dm = (constants.c_km_s / (H0 * np.sqrt(ok))) * np.sinh(sk)
        else:
            sk = np.sqrt(abs(ok)) * (H0 * Dc / constants.c_km_s)
            Dm = (constants.c_km_s / (H0 * np.sqrt(abs(ok)))) * np.sin(sk)
        out[j] = (1.0 + z_eval) * Dm
    return out


def predict_dL_gw(
    post: MuForwardPosterior,
    *,
    z_eval: np.ndarray,
    convention: Literal["A", "B"] = "A",
    constants: PhysicalConstants | None = None,
    allow_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GW luminosity distance draws and R(z) draws."""
    z_eval, R = predict_r_gw_em(
        post,
        z_eval=z_eval,
        convention=convention,
        allow_extrapolation=allow_extrapolation,
    )
    dL_em = predict_dL_em(post, z_eval=z_eval, constants=constants)
    return dL_em * R, R
