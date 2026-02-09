from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camb_utils import camb_clpp


@dataclass(frozen=True)
class PlanckLensingClppLogLike:
    """Gaussian likelihood for Planck 2018 lensing bandpowers using CAMB C_l^{phi phi}."""

    ell_eff: np.ndarray
    clpp: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    meta: dict

    @classmethod
    def from_data(
        cls,
        *,
        ell_eff: np.ndarray,
        clpp: np.ndarray,
        cov: np.ndarray,
        meta: dict | None = None,
    ) -> "PlanckLensingClppLogLike":
        ell_eff = np.asarray(ell_eff, dtype=int)
        clpp = np.asarray(clpp, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (clpp.size, clpp.size):
            raise ValueError("Lensing covariance shape mismatch.")
        if ell_eff.shape != clpp.shape:
            raise ValueError("ell_eff shape mismatch.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Lensing covariance not symmetric.")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Lensing covariance is singular.") from exc
        return cls(ell_eff=ell_eff, clpp=clpp, cov=cov, cov_inv=cov_inv, meta=dict(meta or {}))

    def predict(self, *, H0: float, omega_m0: float, omega_k0: float, sigma8_0: float) -> np.ndarray:
        return camb_clpp(H0=H0, omega_m0=omega_m0, omega_k0=omega_k0, sigma8_0=sigma8_0, ell=self.ell_eff)

    def loglike(self, model: np.ndarray) -> float:
        model = np.asarray(model, dtype=float)
        if model.shape != self.clpp.shape:
            raise ValueError("Model shape mismatch for lensing bandpower loglike.")
        if np.any(~np.isfinite(model)):
            return -np.inf
        r = self.clpp - model
        chi2 = float(r.T @ self.cov_inv @ r)
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            return -np.inf
        ll = -0.5 * (chi2 + logdet + self.clpp.size * np.log(2.0 * np.pi))
        return float(ll)
