from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import PhysicalConstants
from .cosmology import BackgroundCosmology
from .growth import predict_fsigma8
from .ingest_fsbao import FsBaoMeasurements


FSBAO_SPECS: dict[str, dict] = {
    # Alam et al. 2016 DR12 full-shape consensus; values stored with rs_fid scaling.
    "sdss_dr12_consensus_fs": {
        "rs_fid": 147.78,  # Mpc
        "predict": "dr12_consensus_fs",
    },
    # eBOSS DR16 BAO+FSBAO (LRG/QSO) stored directly as D_M/rs and D_H/rs, plus fσ8.
    "sdss_dr16_lrg_fsbao_dmdhfs8": {"predict": "dm_dh_over_rd_fs8"},
    "sdss_dr16_qso_fsbao_dmdhfs8": {"predict": "dm_dh_over_rd_fs8"},
}


@dataclass(frozen=True)
class FsBaoLogLike:
    """Joint BAO geometry + RSD (fσ8) likelihood with full covariance."""

    dataset: str
    z: np.ndarray
    y: np.ndarray
    obs: np.ndarray
    cov: np.ndarray
    cov_cho: tuple[np.ndarray, bool]
    logdet: float
    constants: PhysicalConstants

    @classmethod
    def from_data(
        cls,
        fsbao: FsBaoMeasurements,
        *,
        dataset: str,
        constants: PhysicalConstants,
        z_min: float | None = None,
        z_max: float | None = None,
        diag_cov: bool = False,
    ) -> "FsBaoLogLike":
        if dataset not in FSBAO_SPECS:
            raise ValueError(f"Unknown FSBAO dataset spec '{dataset}'.")
        mask = np.ones_like(fsbao.z, dtype=bool)
        if z_min is not None:
            mask &= fsbao.z >= z_min
        if z_max is not None:
            mask &= fsbao.z <= z_max
        z = fsbao.z[mask]
        y = fsbao.value[mask]
        obs = fsbao.obs[mask].astype("U")
        cov = fsbao.cov[np.ix_(mask, mask)]
        if bool(diag_cov):
            cov = np.diag(np.diag(cov))
        cho = cho_factor(cov, lower=True, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        return cls(
            dataset=dataset,
            z=z,
            y=y,
            obs=obs,
            cov=cov,
            cov_cho=cho,
            logdet=logdet,
            constants=constants,
        )

    def ordering(self) -> list[tuple[str, float, str]]:
        """Return the exact y-vector ordering as (dataset, z, obs) tuples."""
        return [(self.dataset, float(z), str(obs)) for z, obs in zip(self.z, self.obs, strict=True)]

    def predict(
        self,
        bg: BackgroundCosmology,
        *,
        z_grid: np.ndarray,
        H_grid: np.ndarray,
        H0: float,
        omega_m0: float,
        omega_k0: float,
        r_d_Mpc: float,
        sigma8_0: float,
    ) -> np.ndarray:
        """Return the model prediction vector in this dataset's ordering."""
        spec = FSBAO_SPECS[self.dataset]
        mode = str(spec["predict"])
        pred = np.empty_like(self.y)

        # Precompute fσ8(z) for the unique z points present in the measurement vector.
        zs_fs8 = sorted({float(z) for z, o in zip(self.z, self.obs, strict=True) if str(o) == "f_sigma8"})
        fs8_map: dict[float, float] = {}
        if zs_fs8:
            fs8_vals = predict_fsigma8(
                z_eval=np.asarray(zs_fs8, dtype=float),
                z_grid=np.asarray(z_grid, dtype=float),
                H_grid=np.asarray(H_grid, dtype=float),
                H0=float(H0),
                omega_m0=float(omega_m0),
                omega_k0=float(omega_k0),
                sigma8_0=float(sigma8_0),
            )
            fs8_map = {float(z): float(v) for z, v in zip(zs_fs8, fs8_vals, strict=True)}

        if mode == "dr12_consensus_fs":
            rs_fid = float(spec["rs_fid"])
            for i, (z, obs) in enumerate(zip(self.z, self.obs, strict=True)):
                zf = float(z)
                o = str(obs)
                if o == "DM_over_rs":
                    pred[i] = bg.Dm(np.array([zf]))[0] * (rs_fid / float(r_d_Mpc))
                elif o == "bao_Hz_rs":
                    pred[i] = bg.H(np.array([zf]))[0] * (float(r_d_Mpc) / rs_fid)
                elif o == "f_sigma8":
                    pred[i] = fs8_map[zf]
                else:
                    raise ValueError(f"Unsupported FSBAO observable '{o}' for {self.dataset}.")
            return pred

        if mode == "dm_dh_over_rd_fs8":
            for i, (z, obs) in enumerate(zip(self.z, self.obs, strict=True)):
                zf = float(z)
                o = str(obs)
                if o == "DM_over_rs":
                    pred[i] = bg.Dm(np.array([zf]))[0] / float(r_d_Mpc)
                elif o == "DH_over_rs":
                    pred[i] = bg.Dh(np.array([zf]))[0] / float(r_d_Mpc)
                elif o == "f_sigma8":
                    pred[i] = fs8_map[zf]
                else:
                    raise ValueError(f"Unsupported FSBAO observable '{o}' for {self.dataset}.")
            return pred

        raise ValueError(f"Unknown FSBAO prediction mode '{mode}'.")

    def loglike(self, y_model: np.ndarray) -> float:
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        chi2 = float(r @ cho_solve(self.cov_cho, r, check_finite=False))
        return -0.5 * (chi2 + self.logdet)

    def chi2(self, y_model: np.ndarray) -> float:
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        return float(r @ cho_solve(self.cov_cho, r, check_finite=False))

    def chi2_diag(self, y_model: np.ndarray) -> float:
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        d = np.diag(self.cov)
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError("Non-positive or non-finite diagonal in FSBAO covariance.")
        return float(np.sum((r * r) / d))

    def max_abs_offdiag_cov(self) -> float:
        if self.cov.size == 0:
            return 0.0
        off = self.cov - np.diag(np.diag(self.cov))
        return float(np.max(np.abs(off)))

