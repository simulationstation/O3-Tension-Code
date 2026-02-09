from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .growth import solve_growth_ode_muP
from .growth import solve_growth_ode
from .sirens import MuForwardPosterior, predict_r_gw_em


@dataclass(frozen=True)
class VoidPrismMeasurement:
    """A void-conditioned E_G-like measurement.

    This is intentionally flexible: you can provide either an E_G(ell) vector (with covariance)
    or a single scalar summary (ell omitted).

    The measurement is assumed to already include any geometric normalization factors so that the
    model prediction can be treated as a dimensionless E_G(z_eff) (possibly repeated across ell).
    """

    name: str
    z_eff: float
    ell: np.ndarray  # (n_ell,)
    eg_obs: np.ndarray  # (n_ell,)
    cov: np.ndarray  # (n_ell, n_ell)
    env: dict[str, Any] | None = None
    notes: str | None = None
    source: dict[str, Any] | None = None


def load_void_prism_measurements(path: str | Path) -> list[VoidPrismMeasurement]:
    """Load measurements from a small JSON file.

    Schema (per measurement):
      - name: str
      - z_eff: float
      - eg_obs: float | list[float]
      - eg_sigma: float | list[float]           (diagonal covariance)
        OR
        cov: list[list[float]]                  (full covariance)
      - ell: int | list[int] (optional; if missing, ell=[0,1,2,...])
      - env: dict (optional, for screening tomography bookkeeping)
      - notes/source: optional metadata
    """
    p = Path(path)
    data = json.loads(p.read_text())
    items = data["measurements"] if isinstance(data, dict) and "measurements" in data else data
    if not isinstance(items, list):
        raise ValueError("Expected a list of measurements or {'measurements': [...]} JSON.")

    out: list[VoidPrismMeasurement] = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("Each measurement must be a JSON object.")
        eg_obs = np.atleast_1d(np.asarray(it["eg_obs"], dtype=float))
        if "cov" in it and it["cov"] is not None:
            cov = np.asarray(it["cov"], dtype=float)
        else:
            eg_sig = np.atleast_1d(np.asarray(it.get("eg_sigma"), dtype=float))
            if eg_sig.shape != eg_obs.shape:
                raise ValueError("If cov is omitted, eg_sigma must match eg_obs shape.")
            cov = np.diag(eg_sig**2)

        if cov.shape != (eg_obs.size, eg_obs.size):
            raise ValueError("cov must have shape (n,n) matching eg_obs length.")
        if not np.all(np.isfinite(cov)):
            raise ValueError("cov must be finite.")

        if "ell" in it and it["ell"] is not None:
            ell = np.atleast_1d(np.asarray(it["ell"], dtype=int))
            if ell.size == 1 and eg_obs.size > 1:
                ell = np.arange(eg_obs.size, dtype=int)
        else:
            ell = np.arange(eg_obs.size, dtype=int)

        if ell.shape != eg_obs.shape:
            raise ValueError("ell must match eg_obs length (or be omitted).")

        out.append(
            VoidPrismMeasurement(
                name=str(it["name"]),
                z_eff=float(it["z_eff"]),
                ell=ell,
                eg_obs=eg_obs,
                cov=cov,
                env=dict(it["env"]) if "env" in it and it["env"] is not None else None,
                notes=str(it["notes"]) if "notes" in it and it["notes"] is not None else None,
                source=dict(it["source"]) if "source" in it and it["source"] is not None else None,
            )
        )
    return out


def omega_m_of_z(*, z: float, H: float, H0: float, omega_m0: float) -> float:
    """Return Omega_m(z) from background quantities.

    Uses Omega_m(z) = Omega_m0 * (1+z)^3 * (H0/H(z))^2.
    """
    z = float(z)
    H = float(H)
    H0 = float(H0)
    om0 = float(omega_m0)
    if not (np.isfinite(z) and np.isfinite(H) and np.isfinite(H0) and np.isfinite(om0)):
        raise ValueError("Non-finite inputs.")
    if H <= 0 or H0 <= 0:
        raise ValueError("H and H0 must be positive.")
    if not (0.0 < om0 < 1.0):
        raise ValueError("omega_m0 must be in (0,1).")
    return om0 * (1.0 + z) ** 3 * (H0 / H) ** 2


def _eta_of_z(z: np.ndarray, *, eta0: float, eta1: float) -> np.ndarray:
    """Simple slip / lensing-strength modifier eta(z) >= 0.

    Parameterization: eta(z) = eta0 + eta1 * z/(1+z).
    """
    z = np.asarray(z, dtype=float)
    if not np.isfinite(eta0) or not np.isfinite(eta1):
        raise ValueError("eta0/eta1 must be finite.")
    t = np.clip(z / (1.0 + z), 0.0, 1.0)
    eta = float(eta0) + float(eta1) * t
    return np.clip(eta, 1e-6, np.inf)


def predict_EG_void_from_mu(
    post: MuForwardPosterior,
    *,
    z_eff: float,
    ell: np.ndarray | None = None,
    convention: Literal["A", "B"] = "A",
    embedding: Literal["minimal", "slip_allowed", "screening_allowed"] = "minimal",
    # Slip parameters: Sigma(z) = mu_ratio(z) * eta(z). muP(z) stays at mu_ratio(z).
    eta0: float = 1.0,
    eta1: float = 0.0,
    # Screening parameters: multiply both muP and Sigma by exp(alpha * env_proxy)
    env_proxy: float = 0.0,
    env_alpha: float = 0.0,
    muP_highz: float = 1.0,
    max_draws: int | None = 5000,
) -> np.ndarray:
    """Predict a draw-wise E_G^void(z_eff) under a chosen embedding.

    Returns:
      EG_draws with shape (n_draws, n_ell). If ell is omitted, n_ell=1.
    """
    z_eff = float(z_eff)
    if not np.isfinite(z_eff) or z_eff < 0:
        raise ValueError("z_eff must be finite and >= 0.")

    # Subsample draws to keep the post-processing lightweight.
    n_draws_full = int(post.H_samples.shape[0])
    if max_draws is not None and n_draws_full > int(max_draws):
        idx = np.linspace(0, n_draws_full - 1, int(max_draws), dtype=int)
        post = MuForwardPosterior(
            x_grid=post.x_grid,
            logmu_x_samples=post.logmu_x_samples[idx],
            z_grid=post.z_grid,
            H_samples=post.H_samples[idx],
            H0=post.H0[idx],
            omega_m0=post.omega_m0[idx],
            omega_k0=post.omega_k0[idx],
            sigma8_0=post.sigma8_0[idx] if post.sigma8_0 is not None else None,
        )

    # Compute mu(z)/mu(0) (or its inverse) on the posterior z_grid for each draw.
    _, R = predict_r_gw_em(post, z_eval=post.z_grid, convention=convention, allow_extrapolation=False)
    if convention == "A":
        mu_ratio = R**2
    else:
        mu_ratio = 1.0 / (R**2)

    if embedding == "minimal":
        muP = mu_ratio
        Sigma = mu_ratio
    elif embedding == "slip_allowed":
        eta = _eta_of_z(post.z_grid.reshape((1, -1)), eta0=float(eta0), eta1=float(eta1))
        muP = mu_ratio
        Sigma = mu_ratio * eta
    elif embedding == "screening_allowed":
        if not np.isfinite(env_proxy) or not np.isfinite(env_alpha):
            raise ValueError("env_proxy/env_alpha must be finite.")
        scale = float(np.exp(float(env_alpha) * float(env_proxy)))
        muP = mu_ratio * scale
        Sigma = mu_ratio * scale
    else:
        raise ValueError("embedding must be one of: minimal, slip_allowed, screening_allowed.")

    # Clamp to avoid numerical blowups in growth integration.
    muP = np.clip(muP, 1e-3, 1e3)
    Sigma = np.clip(Sigma, 1e-3, 1e3)

    # Evaluate EG per draw.
    n_draws = int(post.H_samples.shape[0])
    eg = np.empty(n_draws, dtype=float)
    for j in range(n_draws):
        H0 = float(post.H0[j])
        om0 = float(post.omega_m0[j])
        ok0 = float(post.omega_k0[j])
        H_z = float(np.interp(z_eff, post.z_grid, post.H_samples[j]))
        omz = omega_m_of_z(z=z_eff, H=H_z, H0=H0, omega_m0=om0)

        # Growth with muP.
        sol = solve_growth_ode_muP(
            z_grid=post.z_grid,
            H_grid=post.H_samples[j],
            H0=H0,
            omega_m0=om0,
            omega_k0=ok0,
            muP_grid=muP[j],
            muP_highz=float(muP_highz),
        )
        x_eff = -np.log1p(z_eff)
        f_eff = float(np.interp(x_eff, sol.x_grid, sol.f))
        if not np.isfinite(f_eff) or f_eff <= 0:
            raise ValueError("Non-physical growth rate encountered in EG prediction.")
        Sigma_eff = float(np.interp(z_eff, post.z_grid, Sigma[j]))
        eg[j] = omz * Sigma_eff / f_eff

    ell = np.atleast_1d(np.asarray(ell if ell is not None else np.array([0], dtype=int), dtype=int))
    out = np.repeat(eg.reshape((-1, 1)), ell.size, axis=1)
    return out


def predict_EG_void_concat_from_mu(
    post: MuForwardPosterior,
    *,
    blocks: list[tuple[float, np.ndarray]],
    convention: Literal["A", "B"] = "A",
    embedding: Literal["minimal", "slip_allowed", "screening_allowed"] = "minimal",
    eta0: float = 1.0,
    eta1: float = 0.0,
    env_proxy: float = 0.0,
    env_alpha: float = 0.0,
    muP_highz: float = 1.0,
    max_draws: int | None = 5000,
) -> np.ndarray:
    """Predict a concatenated EG^void vector for multiple (z_eff, ell) blocks.

    This is a speed-optimized version of repeatedly calling predict_EG_void_from_mu(): we solve the
    growth ODE at most once per posterior draw and then evaluate f(z_eff) for each block.

    Args:
      blocks: list of (z_eff, ell_array) in the order that matches the measurement concatenation.
    """
    if not blocks:
        raise ValueError("blocks must be non-empty.")

    # Subsample draws to keep post-processing lightweight.
    n_draws_full = int(post.H_samples.shape[0])
    if max_draws is not None and n_draws_full > int(max_draws):
        idx = np.linspace(0, n_draws_full - 1, int(max_draws), dtype=int)
        post = MuForwardPosterior(
            x_grid=post.x_grid,
            logmu_x_samples=post.logmu_x_samples[idx],
            z_grid=post.z_grid,
            H_samples=post.H_samples[idx],
            H0=post.H0[idx],
            omega_m0=post.omega_m0[idx],
            omega_k0=post.omega_k0[idx],
            sigma8_0=post.sigma8_0[idx] if post.sigma8_0 is not None else None,
        )

    z_grid = np.asarray(post.z_grid, dtype=float)
    z_eff = np.array([float(z) for z, _ell in blocks], dtype=float)
    if np.any(~np.isfinite(z_eff)) or np.any(z_eff < 0):
        raise ValueError("All block z_eff must be finite and >= 0.")

    # mu(z)/mu(0) (or its inverse) on the posterior z_grid for each draw.
    _, R = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)
    if convention == "A":
        mu_ratio = R**2
    else:
        mu_ratio = 1.0 / (R**2)

    if embedding == "minimal":
        muP = mu_ratio
        Sigma = mu_ratio
    elif embedding == "slip_allowed":
        eta = _eta_of_z(z_grid.reshape((1, -1)), eta0=float(eta0), eta1=float(eta1))
        muP = mu_ratio
        Sigma = mu_ratio * eta
    elif embedding == "screening_allowed":
        if not np.isfinite(env_proxy) or not np.isfinite(env_alpha):
            raise ValueError("env_proxy/env_alpha must be finite.")
        scale = float(np.exp(float(env_alpha) * float(env_proxy)))
        muP = mu_ratio * scale
        Sigma = mu_ratio * scale
    else:
        raise ValueError("embedding must be one of: minimal, slip_allowed, screening_allowed.")

    muP = np.clip(muP, 1e-3, 1e3)
    Sigma = np.clip(Sigma, 1e-3, 1e3)

    # Precompute which z_eff -> x_eff (ln a) values we need.
    x_eff = -np.log1p(z_eff)
    ell_sizes = [int(np.atleast_1d(np.asarray(ell, dtype=int)).size) for _z, ell in blocks]
    out_dim = int(np.sum(ell_sizes))

    n_draws = int(post.H_samples.shape[0])
    out = np.empty((n_draws, out_dim), dtype=float)
    for j in range(n_draws):
        H0 = float(post.H0[j])
        om0 = float(post.omega_m0[j])
        ok0 = float(post.omega_k0[j])

        # One growth solve per draw.
        sol = solve_growth_ode_muP(
            z_grid=z_grid,
            H_grid=post.H_samples[j],
            H0=H0,
            omega_m0=om0,
            omega_k0=ok0,
            muP_grid=muP[j],
            muP_highz=float(muP_highz),
        )
        f_eff = np.interp(x_eff, sol.x_grid, sol.f).astype(float)
        if np.any(~np.isfinite(f_eff)) or np.any(f_eff <= 0):
            raise ValueError("Non-physical growth rate encountered in EG prediction.")

        # Evaluate Omega_m(z_eff) and Sigma(z_eff) per block.
        H_eff = np.interp(z_eff, z_grid, post.H_samples[j]).astype(float)
        omz = np.array([omega_m_of_z(z=float(z), H=float(H), H0=H0, omega_m0=om0) for z, H in zip(z_eff, H_eff, strict=False)], dtype=float)
        Sigma_eff = np.interp(z_eff, z_grid, Sigma[j]).astype(float)

        eg_eff = omz * Sigma_eff / f_eff

        # Expand to the concatenated (block,ell) vector.
        k = 0
        for eg_val, n_ell in zip(eg_eff, ell_sizes, strict=False):
            out[j, k : k + n_ell] = float(eg_val)
            k += n_ell

    return out


def eg_gr_baseline_from_background(
    post: MuForwardPosterior,
    *,
    z_eff: float,
    ell: np.ndarray | None = None,
    max_draws: int | None = 5000,
) -> np.ndarray:
    """Compute an internal GR baseline E_G(z_eff) using the same background draws.

    This is a useful diagnostic baseline for embedding tests: Sigma=1 and muP=1, so
    E_G ~ Omega_m(z)/f(z) given the draw's inferred H(z).
    """
    z_eff = float(z_eff)
    if not np.isfinite(z_eff) or z_eff < 0:
        raise ValueError("z_eff must be finite and >= 0.")

    n_draws_full = int(post.H_samples.shape[0])
    if max_draws is not None and n_draws_full > int(max_draws):
        idx = np.linspace(0, n_draws_full - 1, int(max_draws), dtype=int)
        post = MuForwardPosterior(
            x_grid=post.x_grid,
            logmu_x_samples=post.logmu_x_samples[idx],
            z_grid=post.z_grid,
            H_samples=post.H_samples[idx],
            H0=post.H0[idx],
            omega_m0=post.omega_m0[idx],
            omega_k0=post.omega_k0[idx],
            sigma8_0=post.sigma8_0[idx] if post.sigma8_0 is not None else None,
        )

    n_draws = int(post.H_samples.shape[0])
    eg = np.empty(n_draws, dtype=float)
    for j in range(n_draws):
        H0 = float(post.H0[j])
        om0 = float(post.omega_m0[j])
        ok0 = float(post.omega_k0[j])
        H_z = float(np.interp(z_eff, post.z_grid, post.H_samples[j]))
        omz = omega_m_of_z(z=z_eff, H=H_z, H0=H0, omega_m0=om0)
        sol = solve_growth_ode(z_grid=post.z_grid, H_grid=post.H_samples[j], H0=H0, omega_m0=om0, omega_k0=ok0)
        x_eff = -np.log1p(z_eff)
        f_eff = float(np.interp(x_eff, sol.x_grid, sol.f))
        eg[j] = omz / f_eff

    ell = np.atleast_1d(np.asarray(ell if ell is not None else np.array([0], dtype=int), dtype=int))
    return np.repeat(eg.reshape((-1, 1)), ell.size, axis=1)


def eg_gr_baseline_concat_from_background(
    post: MuForwardPosterior,
    *,
    blocks: list[tuple[float, np.ndarray]],
    max_draws: int | None = 5000,
) -> np.ndarray:
    """GR baseline EG(z_eff) concatenated across blocks, solving growth once per draw."""
    if not blocks:
        raise ValueError("blocks must be non-empty.")

    n_draws_full = int(post.H_samples.shape[0])
    if max_draws is not None and n_draws_full > int(max_draws):
        idx = np.linspace(0, n_draws_full - 1, int(max_draws), dtype=int)
        post = MuForwardPosterior(
            x_grid=post.x_grid,
            logmu_x_samples=post.logmu_x_samples[idx],
            z_grid=post.z_grid,
            H_samples=post.H_samples[idx],
            H0=post.H0[idx],
            omega_m0=post.omega_m0[idx],
            omega_k0=post.omega_k0[idx],
            sigma8_0=post.sigma8_0[idx] if post.sigma8_0 is not None else None,
        )

    z_grid = np.asarray(post.z_grid, dtype=float)
    z_eff = np.array([float(z) for z, _ell in blocks], dtype=float)
    x_eff = -np.log1p(z_eff)
    ell_sizes = [int(np.atleast_1d(np.asarray(ell, dtype=int)).size) for _z, ell in blocks]
    out_dim = int(np.sum(ell_sizes))

    n_draws = int(post.H_samples.shape[0])
    out = np.empty((n_draws, out_dim), dtype=float)
    for j in range(n_draws):
        H0 = float(post.H0[j])
        om0 = float(post.omega_m0[j])
        ok0 = float(post.omega_k0[j])

        sol = solve_growth_ode(z_grid=z_grid, H_grid=post.H_samples[j], H0=H0, omega_m0=om0, omega_k0=ok0)
        f_eff = np.interp(x_eff, sol.x_grid, sol.f).astype(float)
        H_eff = np.interp(z_eff, z_grid, post.H_samples[j]).astype(float)
        omz = np.array([omega_m_of_z(z=float(z), H=float(H), H0=H0, omega_m0=om0) for z, H in zip(z_eff, H_eff, strict=False)], dtype=float)
        eg_eff = omz / f_eff

        k = 0
        for eg_val, n_ell in zip(eg_eff, ell_sizes, strict=False):
            out[j, k : k + n_ell] = float(eg_val)
            k += n_ell

    return out
