from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .constants import PhysicalConstants


@dataclass(frozen=True)
class GWDistancePrior:
    """Distance prior model for converting GW posterior densities into a proxy likelihood.

    Public 3D sky maps provide (an approximation of) the posterior density
      p(Ω, dL | data) ∝ L(data | Ω, dL) π(Ω, dL).

    For dark-siren scoring we need something proportional to the likelihood, so we divide out an
    assumed distance prior:
      L_proxy(data | Ω, dL) ∝ p(Ω, dL | data) / π(dL),

    up to a model-independent constant (sky prior is typically uniform and cancels).
    """

    mode: Literal["none", "dL_powerlaw", "comoving_lcdm", "comoving_lcdm_sourceframe"] = "dL_powerlaw"
    # dL_powerlaw: π(dL) ∝ dL^k
    powerlaw_k: float = 2.0
    # comoving_lcdm*: π(dL) induced by a fixed LCDM cosmology (used only as a *prior to divide out*).
    h0_ref: float = 67.7
    omega_m0: float = 0.31
    omega_k0: float = 0.0
    z_max: float = 10.0
    n_grid: int = 50_000

    # Cached interpolation arrays for comoving modes (built lazily).
    _dL_grid_mpc: np.ndarray | None = None
    _logpi_grid: np.ndarray | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "powerlaw_k": float(self.powerlaw_k),
            "h0_ref": float(self.h0_ref),
            "omega_m0": float(self.omega_m0),
            "omega_k0": float(self.omega_k0),
            "z_max": float(self.z_max),
            "n_grid": int(self.n_grid),
        }

    def _ensure_comoving_cache(self) -> tuple[np.ndarray, np.ndarray]:
        if self._dL_grid_mpc is not None and self._logpi_grid is not None:
            return self._dL_grid_mpc, self._logpi_grid

        z_max = float(self.z_max)
        n = int(self.n_grid)
        if not (np.isfinite(z_max) and z_max > 0.0):
            raise ValueError("GWDistancePrior.z_max must be finite and positive.")
        if n < 500:
            raise ValueError("GWDistancePrior.n_grid too small.")

        om = float(self.omega_m0)
        ok = float(self.omega_k0)
        ol = 1.0 - om - ok
        if not np.isfinite(ol):
            raise ValueError("Non-finite omega_L inferred from omega_m0/omega_k0.")

        z = np.linspace(0.0, z_max, n)
        Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
        Ez2 = np.clip(Ez2, 1e-15, np.inf)
        Ez = np.sqrt(Ez2)
        invE = 1.0 / Ez

        dz = np.diff(z)
        chi = np.empty_like(z)
        chi[0] = 0.0
        chi[1:] = np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))

        if ok == 0.0:
            Sk = chi
        elif ok > 0.0:
            rk = np.sqrt(ok)
            Sk = np.sinh(rk * chi) / rk
        else:
            rk = np.sqrt(abs(ok))
            Sk = np.sin(rk * chi) / rk

        Dm_dimless = Sk  # D_M = (c/H0) * Dm_dimless
        f = (1.0 + z) * Dm_dimless  # dL = (c/H0) * f

        const = PhysicalConstants()
        dL = (const.c_km_s / float(self.h0_ref)) * f

        ddL_dz = np.gradient(dL, z)
        ddL_dz = np.clip(ddL_dz, 1e-12, np.inf)

        # p(z) ∝ dV/dz/dΩ (up to constant), with dV/dz/dΩ ∝ D_M^2 / E(z).
        base = (Dm_dimless**2) / Ez
        if self.mode == "comoving_lcdm_sourceframe":
            # Constant rate in source-frame time => divide by (1+z).
            base = base / np.clip(1.0 + z, 1e-12, np.inf)

        # Transform to a density in dL: p(dL) = p(z) * dz/ddL.
        pi = base / ddL_dz
        pi = np.clip(pi, 1e-300, np.inf)
        logpi = np.log(pi)

        # Drop z=0 point (dL=0) to keep monotone interpolation stable.
        dL = dL[1:]
        logpi = logpi[1:]
        if not np.all(np.isfinite(dL)) or np.any(np.diff(dL) <= 0.0):
            raise ValueError("Non-monotone dL(z) while building comoving distance prior cache.")

        object.__setattr__(self, "_dL_grid_mpc", dL)
        object.__setattr__(self, "_logpi_grid", logpi)
        return dL, logpi

    def log_pi_dL(self, dL_mpc: np.ndarray) -> np.ndarray:
        """Evaluate log π(dL) up to an additive constant."""
        dL = np.asarray(dL_mpc, dtype=float)
        if self.mode == "none":
            return np.zeros_like(dL, dtype=float)

        if self.mode == "dL_powerlaw":
            k = float(self.powerlaw_k)
            return k * np.log(np.clip(dL, 1e-12, np.inf))

        if self.mode in ("comoving_lcdm", "comoving_lcdm_sourceframe"):
            grid_dL, grid_logpi = self._ensure_comoving_cache()
            flat = dL.reshape(-1)
            flat = np.clip(flat, float(grid_dL[0]), float(grid_dL[-1]))
            out = np.interp(flat, grid_dL, grid_logpi)
            return out.reshape(dL.shape)

        raise ValueError(f"Unknown GWDistancePrior.mode {self.mode!r}.")

