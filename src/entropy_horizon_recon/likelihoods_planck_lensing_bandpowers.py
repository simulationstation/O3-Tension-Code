from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanckLensingBandpowerLogLike:
    """Gaussian likelihood for Planck 2018 lensing bandpowers.

    Model is a scaled template: C_L^pp(model) = A_scale * template_clpp.
    The scale A_scale is derived from a proxy (sigma8 * Omega_m^alpha)^2.
    """

    clpp: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    template_clpp: np.ndarray
    alpha: float
    s8om_fid: float
    meta: dict

    @classmethod
    def from_data(
        cls,
        *,
        clpp: np.ndarray,
        cov: np.ndarray,
        template_clpp: np.ndarray,
        alpha: float = 0.25,
        s8om_fid: float = 0.589,
        meta: dict | None = None,
    ) -> "PlanckLensingBandpowerLogLike":
        clpp = np.asarray(clpp, dtype=float)
        cov = np.asarray(cov, dtype=float)
        template = np.asarray(template_clpp, dtype=float)
        if cov.shape != (clpp.size, clpp.size):
            raise ValueError("Lensing covariance shape mismatch.")
        if template.shape != clpp.shape:
            raise ValueError("Template shape mismatch.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Lensing covariance not symmetric.")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Lensing covariance is singular.") from exc
        return cls(
            clpp=clpp,
            cov=cov,
            cov_inv=cov_inv,
            template_clpp=template,
            alpha=float(alpha),
            s8om_fid=float(s8om_fid),
            meta=dict(meta or {}),
        )

    def predict(self, *, omega_m0: float, sigma8_0: float, **_kwargs) -> np.ndarray:
        s8om = float(sigma8_0) * float(omega_m0) ** float(self.alpha)
        A_scale = (s8om / self.s8om_fid) ** 2
        return self.template_clpp * A_scale

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
