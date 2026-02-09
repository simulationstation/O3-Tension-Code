from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .camb_utils import camb_pk_linear


@dataclass(frozen=True)
class FullShapePkLogLike:
    """Gaussian likelihood for P(k) monopole with covariance."""

    k: np.ndarray
    pk: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    z_eff: float
    meta: dict

    @classmethod
    def from_data(cls, *, k: np.ndarray, pk: np.ndarray, cov: np.ndarray, z_eff: float, meta: dict | None = None) -> "FullShapePkLogLike":
        k = np.asarray(k, dtype=float)
        pk = np.asarray(pk, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if k.shape != pk.shape:
            raise ValueError("Pk arrays shape mismatch.")
        if cov.shape != (k.size, k.size):
            raise ValueError("Pk covariance shape mismatch.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Pk covariance not symmetric.")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Pk covariance is singular.") from exc
        return cls(k=k, pk=pk, cov=cov, cov_inv=cov_inv, z_eff=float(z_eff), meta=dict(meta or {}))

    def predict(
        self,
        *,
        H0: float,
        omega_m0: float,
        omega_k0: float,
        sigma8_0: float,
        b1: float,
        pshot: float,
    ) -> np.ndarray:
        p_lin = camb_pk_linear(
            H0=H0,
            omega_m0=omega_m0,
            omega_k0=omega_k0,
            sigma8_0=sigma8_0,
            z_eff=self.z_eff,
            k=self.k,
        )
        return (b1 ** 2) * p_lin + pshot

    def loglike(self, model: np.ndarray) -> float:
        model = np.asarray(model, dtype=float)
        if model.shape != self.pk.shape:
            raise ValueError("Model shape mismatch for P(k) loglike.")
        if np.any(~np.isfinite(model)):
            return -np.inf
        r = self.pk - model
        chi2 = float(r.T @ self.cov_inv @ r)
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            return -np.inf
        ll = -0.5 * (chi2 + logdet + self.pk.size * np.log(2.0 * np.pi))
        return float(ll)
