from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .growth import predict_fsigma8


@dataclass(frozen=True)
class RsdFs8LogLike:
    """Gaussian likelihood for fσ8(z) points (diagonal errors)."""

    z: np.ndarray
    fs8: np.ndarray
    sigma_fs8: np.ndarray
    meta: dict

    @classmethod
    def from_data(cls, *, z: np.ndarray, fs8: np.ndarray, sigma_fs8: np.ndarray, meta: dict | None = None) -> "RsdFs8LogLike":
        z = np.asarray(z, dtype=float)
        fs8 = np.asarray(fs8, dtype=float)
        sigma = np.asarray(sigma_fs8, dtype=float)
        if z.shape != fs8.shape or z.shape != sigma.shape:
            raise ValueError("RSD arrays shape mismatch.")
        if np.any(~np.isfinite(z)) or np.any(~np.isfinite(fs8)) or np.any(~np.isfinite(sigma)):
            raise ValueError("RSD arrays contain non-finite values.")
        if np.any(sigma <= 0):
            raise ValueError("sigma_fs8 must be positive.")
        order = np.argsort(z)
        return cls(z=z[order], fs8=fs8[order], sigma_fs8=sigma[order], meta=dict(meta or {}))

    def predict(
        self,
        *,
        z_grid: np.ndarray,
        H_grid: np.ndarray,
        H0: float,
        omega_m0: float,
        omega_k0: float,
        sigma8_0: float,
    ) -> np.ndarray:
        return predict_fsigma8(
            z_eval=self.z,
            z_grid=z_grid,
            H_grid=H_grid,
            H0=H0,
            omega_m0=omega_m0,
            omega_k0=omega_k0,
            sigma8_0=sigma8_0,
        )

    def loglike(self, model: np.ndarray) -> float:
        model = np.asarray(model, dtype=float)
        if model.shape != self.fs8.shape:
            raise ValueError("Model shape mismatch for RSD loglike.")
        if np.any(~np.isfinite(model)):
            return -np.inf
        r = self.fs8 - model
        inv = 1.0 / self.sigma_fs8
        chi2 = float(np.sum((r * inv) ** 2))
        # Include normalization for transparency (constant wrt theta except log sigma, but sigma is fixed here).
        ll = -0.5 * (chi2 + float(np.sum(2.0 * np.log(self.sigma_fs8))) + self.fs8.size * np.log(2.0 * np.pi))
        return float(ll)

