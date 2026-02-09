from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid
from .forward_model import ForwardModel
from .sirens import MuForwardPosterior, predict_r_gw_em


@dataclass(frozen=True)
class VoidAmpMeasurement:
    """Amplitude-only void lensing measurement.

    This is intentionally minimal: the "Tier 1" test only compares an observed amplitude
    A_obs +/- A_sigma to a posterior-predicted amplitude A_pred derived from mu(A).
    """

    name: str
    kind: str
    z_min: float
    z_max: float
    A_obs: float
    A_sigma: float
    # Optional metadata controlling how we compute the z-averaged prediction.
    weight: str | None = None
    kernel: str | None = None  # e.g. "cmb_kappa"
    chi_source_mpc: float | None = None  # e.g. ~1.4e4 for CMB last-scattering distance
    nz: dict[str, Any] | None = None  # optional {"z":[...], "w":[...]} to approximate the sample n(z)
    notes: str | None = None
    source: dict[str, Any] | None = None


def load_void_amp_measurements(path: str | Path) -> list[VoidAmpMeasurement]:
    p = Path(path)
    data = json.loads(p.read_text())
    items = data["measurements"] if isinstance(data, dict) and "measurements" in data else data
    if not isinstance(items, list):
        raise ValueError("Expected a list of measurements or {'measurements': [...]} JSON.")
    out: list[VoidAmpMeasurement] = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("Each measurement must be a JSON object.")
        out.append(
            VoidAmpMeasurement(
                name=str(it["name"]),
                kind=str(it.get("kind", "kappa_amp")),
                z_min=float(it["z_min"]),
                z_max=float(it["z_max"]),
                A_obs=float(it["A_obs"]),
                A_sigma=float(it["A_sigma"]),
                weight=str(it["weight"]) if "weight" in it and it["weight"] is not None else None,
                kernel=str(it["kernel"]) if "kernel" in it and it["kernel"] is not None else None,
                chi_source_mpc=float(it["chi_source_mpc"]) if "chi_source_mpc" in it and it["chi_source_mpc"] is not None else None,
                nz=dict(it["nz"]) if "nz" in it and it["nz"] is not None else None,
                notes=str(it["notes"]) if "notes" in it and it["notes"] is not None else None,
                source=dict(it["source"]) if "source" in it and it["source"] is not None else None,
            )
        )
    return out


def _interp_nz_weight(nz: dict[str, Any], z_eval: np.ndarray) -> np.ndarray:
    z = np.asarray(nz.get("z"), dtype=float)
    w = np.asarray(nz.get("w"), dtype=float)
    if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape or z.size < 2:
        raise ValueError("nz must contain 1D arrays 'z' and 'w' of equal length >= 2.")
    if np.any(np.diff(z) <= 0):
        raise ValueError("nz['z'] must be strictly increasing.")
    w_eval = np.interp(z_eval, z, w)
    return np.clip(w_eval, 0.0, None)


def _dm_from_dc(
    Dc: np.ndarray,
    *,
    H0: float,
    omega_k0: float,
    constants: PhysicalConstants,
) -> np.ndarray:
    """Transverse comoving distance D_M from line-of-sight comoving distance D_C."""
    if omega_k0 == 0.0:
        return Dc
    if omega_k0 > 0.0:
        sk = np.sqrt(omega_k0) * (H0 * Dc / constants.c_km_s)
        return (constants.c_km_s / (H0 * np.sqrt(omega_k0))) * np.sinh(sk)
    sk = np.sqrt(abs(omega_k0)) * (H0 * Dc / constants.c_km_s)
    return (constants.c_km_s / (H0 * np.sqrt(abs(omega_k0)))) * np.sin(sk)


def predict_void_kappa_amp_from_mu(
    post: MuForwardPosterior,
    *,
    z_min: float,
    z_max: float,
    z_n: int = 200,
    convention: Literal["A", "B"] = "A",
    weight: Literal["uniform", "z", "z2", "cmb_kappa"] = "cmb_kappa",
    chi_source_mpc: float = 14_000.0,
    nz: dict[str, Any] | None = None,
    allow_extrapolation: bool = False,
    extend_H_to_zmax: bool = False,
    mapping_variant: str = "M0",
    constants: PhysicalConstants | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict an effective amplitude scaling for void lensing from mu(A).

    We compute a draw-wise amplitude:

      A_pred(draw) = average_{z in [z_min,z_max]} [ mu(z)/mu(0) ].

    Under the selected alpha_M-only embedding, a crude lensing-amplitude scaling is proportional
    to mu(z)/mu(0) (since 1/M_*^2 ∝ mu). This does *not* model nonlinear void profiles; it is a
    fast consistency check.

    Returns:
      (z_eval, A_draws)
    """
    if z_n < 2:
        raise ValueError("z_n must be >= 2.")
    z_min = float(z_min)
    z_max = float(z_max)
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min):
        raise ValueError("Invalid z_min/z_max.")

    exceeds_grid = z_max > float(post.z_grid[-1])
    if exceeds_grid and not (allow_extrapolation or extend_H_to_zmax):
        raise ValueError(
            "Requested z-range outside posterior z_grid range. "
            f"requested=[{z_min:.3g},{z_max:.3g}] vs z_grid=[{float(post.z_grid[0]):.3g},{float(post.z_grid[-1]):.3g}]"
        )
    z_eval = np.linspace(z_min, z_max, int(z_n))

    constants = constants or PhysicalConstants()

    # If the measurement z-range extends beyond the stored posterior H(z) grid, optionally re-solve
    # H(z) from mu(A) out to z_max (model-based prediction) for each draw. This avoids the
    # non-physical np.interp endpoint extrapolation.
    if extend_H_to_zmax and exceeds_grid:
        mv = str(mapping_variant)
        if mv not in ("M0", "M2"):
            raise ValueError(f"extend_H_to_zmax only supports mapping_variant in {{M0,M2}} (got {mv!r}).")

        fm = ForwardModel(constants=constants, x_knots=post.x_grid)
        n_draws = post.logmu_x_samples.shape[0]
        H_eval = np.empty((n_draws, z_eval.size), dtype=float)
        # The forward solver requires z_grid[0] == 0 to anchor H0. If the measurement starts at
        # z_min > 0, solve on a grid that includes z=0 and then interpolate down to z_eval.
        if float(z_eval[0]) != 0.0:
            z_solve = np.concatenate(([0.0], z_eval))
            interp_back = True
        else:
            z_solve = z_eval
            interp_back = False
        for j in range(n_draws):
            H_sol = fm.solve_H_from_logmu_knots(
                z_solve,
                logmu_knots=post.logmu_x_samples[j],
                H0_km_s_Mpc=float(post.H0[j]),
                omega_m0=float(post.omega_m0[j]),
                omega_k0=float(post.omega_k0[j]),
                residual_of_z=None,
            )
            H_eval[j] = np.interp(z_eval, z_solve, H_sol) if interp_back else H_sol

        # Compute mu(z)/mu(0) directly on the extended grid. We intentionally clamp x outside the
        # inferred mu(x) domain to the endpoint value (same behavior as np.interp + the forward
        # solver's own knot clamping).
        H0 = post.H0.reshape((n_draws, 1))
        ok = post.omega_k0.reshape((n_draws, 1))
        denom0 = H0**2 * (1.0 - ok)
        denom = H_eval**2 - ok * H0**2 * (1.0 + z_eval.reshape((1, -1))) ** 2
        if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
            raise ValueError("Non-physical horizon area mapping: denom <= 0.")
        if not np.all(np.isfinite(denom0)) or np.any(denom0 <= 0.0):
            raise ValueError("Non-physical horizon area mapping at z=0: denom0 <= 0.")
        x_eval = np.log(denom0 / denom)
        xmin, xmax = float(post.x_grid[0]), float(post.x_grid[-1])
        x_eval = np.clip(x_eval, xmin, xmax)
        logmu_eval = np.empty((n_draws, z_eval.size), dtype=float)
        for j in range(n_draws):
            logmu_eval[j] = np.interp(x_eval[j], post.x_grid, post.logmu_x_samples[j])
        mu_eval = np.exp(logmu_eval)
        if np.isclose(post.x_grid[-1], 0.0):
            mu0 = np.exp(post.logmu_x_samples[:, -1])
        else:
            mu0 = np.exp(np.array([np.interp(0.0, post.x_grid, post.logmu_x_samples[j]) for j in range(n_draws)]))
        mu_ratio = mu_eval / mu0.reshape((n_draws, 1))

        z_bg = z_eval
        H_bg = H_eval
    else:
        z_eval, R = predict_r_gw_em(
            post,
            z_eval=z_eval,
            convention=convention,
            allow_extrapolation=allow_extrapolation,
        )

        # Under convention A, R^2 = mu(z)/mu0. Under convention B, 1/R^2 = mu(z)/mu0.
        if convention == "A":
            mu_ratio = R**2
        else:
            mu_ratio = 1.0 / (R**2)

        z_bg = post.z_grid
        H_bg = post.H_samples

    # Optional n(z) factor (same for all draws).
    nz_w = _interp_nz_weight(nz, z_eval) if nz is not None else np.ones_like(z_eval)

    if weight in ("uniform", "z", "z2"):
        if weight == "uniform":
            w0 = np.ones_like(z_eval)
        elif weight == "z":
            w0 = np.clip(z_eval, 0.0, None)
        else:
            w0 = np.clip(z_eval, 0.0, None) ** 2
        w0 = w0 * nz_w
        wsum = float(np.sum(w0))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise ValueError("Non-positive weight normalization.")
        A_draws = np.sum(mu_ratio * w0.reshape((1, -1)), axis=1) / wsum
        return z_eval, A_draws

    if weight == "cmb_kappa":
        # Use a standard single-source lensing efficiency weight with a very high source plane
        # (CMB last scattering). For low z, (chi_s - chi)/chi_s ~ 1, so the weight is close to
        # (1+z)*D_M(z). We keep the full factor for clarity.
        chi_s = float(chi_source_mpc)
        if not np.isfinite(chi_s) or chi_s <= 0.0:
            raise ValueError("chi_source_mpc must be positive and finite.")

        n_draws = post.H_samples.shape[0]
        w = np.empty((n_draws, z_eval.size), dtype=float)
        for j in range(n_draws):
            bg = build_background_from_H_grid(z_bg, H_bg[j], constants=constants)
            Dc = bg.Dc(z_eval)
            Dm = _dm_from_dc(Dc, H0=float(post.H0[j]), omega_k0=float(post.omega_k0[j]), constants=constants)
            wj = (1.0 + z_eval) * Dm * np.clip(chi_s - Dm, 0.0, None) / chi_s
            w[j] = wj * nz_w

        wsum = np.sum(w, axis=1)
        if not np.all(np.isfinite(wsum)) or np.any(wsum <= 0.0):
            raise ValueError("Non-positive weight normalization for cmb_kappa weights.")
        A_draws = np.sum(mu_ratio * w, axis=1) / wsum
        return z_eval, A_draws

    raise ValueError("weight must be one of: 'uniform', 'z', 'z2', 'cmb_kappa'.")
