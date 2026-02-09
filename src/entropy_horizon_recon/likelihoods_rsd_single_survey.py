from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .growth import predict_fsigma8


@dataclass(frozen=True)
class RsdFs8CovLogLike:
    """Gaussian likelihood for fσ8(z) with full covariance."""

    z: np.ndarray
    fs8: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    meta: dict

    @classmethod
    def from_data(cls, *, z: np.ndarray, fs8: np.ndarray, cov: np.ndarray, meta: dict | None = None) -> "RsdFs8CovLogLike":
        z = np.asarray(z, dtype=float)
        fs8 = np.asarray(fs8, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if z.shape != fs8.shape:
            raise ValueError("RSD arrays shape mismatch.")
        if cov.shape != (z.size, z.size):
            raise ValueError("RSD covariance shape mismatch.")
        if np.any(~np.isfinite(z)) or np.any(~np.isfinite(fs8)) or np.any(~np.isfinite(cov)):
            raise ValueError("RSD arrays contain non-finite values.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("RSD covariance is not symmetric.")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("RSD covariance is singular.") from exc
        order = np.argsort(z)
        z = z[order]
        fs8 = fs8[order]
        cov = cov[np.ix_(order, order)]
        cov_inv = cov_inv[np.ix_(order, order)]
        return cls(z=z, fs8=fs8, cov=cov, cov_inv=cov_inv, meta=dict(meta or {}))

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
        chi2 = float(r.T @ self.cov_inv @ r)
        # Include normalization constant for transparency.
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            return -np.inf
        ll = -0.5 * (chi2 + logdet + self.fs8.size * np.log(2.0 * np.pi))
        return float(ll)
