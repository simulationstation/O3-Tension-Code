from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanckLensingProxyLogLike:
    """Gaussian proxy likelihood for Planck CMB lensing reconstruction.

    Default quantity: sigma8 * Omega_m0^0.25 (Planck lensing reconstruction).
    """

    mean: float
    sigma: float
    meta: dict
    exponent: float = 0.25

    def predict(self, *, omega_m0: float, sigma8_0: float, **_kwargs) -> float:
        om = float(omega_m0)
        s8 = float(sigma8_0)
        if not np.isfinite(om) or om <= 0:
            return np.nan
        if not np.isfinite(s8) or s8 <= 0:
            return np.nan
        return float(s8 * (om ** float(self.exponent)))

    def loglike(self, model: float) -> float:
        if not np.isfinite(model):
            return -np.inf
        r = (float(model) - float(self.mean)) / float(self.sigma)
        return float(-0.5 * (r * r + np.log(2.0 * np.pi * float(self.sigma) ** 2)))
