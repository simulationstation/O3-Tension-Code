from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import PhysicalConstants
from .cosmology import BackgroundCosmology
from .ingest import BaoMeasurements, Chronometers, PantheonPlus


@dataclass(frozen=True)
class SNLogLike:
    z: np.ndarray
    m: np.ndarray
    cov: np.ndarray
    cho: tuple[np.ndarray, bool]
    logdet: float
    cinv_ones: np.ndarray
    ones_cinv_ones: float

    @classmethod
    def from_pantheon(
        cls,
        sn: PantheonPlus,
        *,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> "SNLogLike":
        z = sn.z
        mask = np.ones_like(z, dtype=bool)
        if z_min is not None:
            mask &= z >= z_min
        if z_max is not None:
            mask &= z <= z_max

        z_f = z[mask]
        m_f = sn.m_b_corr[mask]
        cov_f = sn.cov[np.ix_(mask, mask)]

        cho = cho_factor(cov_f, lower=True, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        ones = np.ones_like(m_f)
        cinv_ones = cho_solve(cho, ones, check_finite=False)
        ones_cinv_ones = float(ones @ cinv_ones)
        if ones_cinv_ones <= 0:
            raise ValueError("SN covariance produced non-positive ones^T C^{-1} ones.")

        return cls(
            z=z_f,
            m=m_f,
            cov=cov_f,
            cho=cho,
            logdet=logdet,
            cinv_ones=cinv_ones,
            ones_cinv_ones=ones_cinv_ones,
        )

    @classmethod
    def from_arrays(cls, *, z: np.ndarray, m: np.ndarray, cov: np.ndarray) -> "SNLogLike":
        z = np.asarray(z, dtype=float)
        m = np.asarray(m, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if z.ndim != 1 or m.ndim != 1:
            raise ValueError("z and m must be 1D.")
        if z.shape != m.shape:
            raise ValueError("z and m must have the same shape.")
        if cov.shape != (z.size, z.size):
            raise ValueError("cov shape mismatch.")

        cho = cho_factor(cov, lower=True, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        ones = np.ones_like(m)
        cinv_ones = cho_solve(cho, ones, check_finite=False)
        ones_cinv_ones = float(ones @ cinv_ones)
        if ones_cinv_ones <= 0:
            raise ValueError("SN covariance produced non-positive ones^T C^{-1} ones.")
        return cls(
            z=z,
            m=m,
            cov=cov,
            cho=cho,
            logdet=logdet,
            cinv_ones=cinv_ones,
            ones_cinv_ones=ones_cinv_ones,
        )

    def loglike_marginalized_M(self, mu0: np.ndarray) -> tuple[float, float]:
        """Log-likelihood with analytic marginalization over intercept M (flat prior).

        Model: m_model = mu0 + M
        """
        if mu0.shape != self.m.shape:
            raise ValueError("mu0 shape mismatch.")
        r = self.m - mu0
        cinv_r = cho_solve(self.cho, r, check_finite=False)
        b = float(np.ones_like(self.m) @ cinv_r)
        M_hat = b / self.ones_cinv_ones
        r2 = r - M_hat
        cinv_r2 = cho_solve(self.cho, r2, check_finite=False)
        chi2 = float(r2 @ cinv_r2)
        # drop additive constants (N log 2π); keep logdet and marginalization term
        loglike = -0.5 * (chi2 + self.logdet + np.log(self.ones_cinv_ones))
        return loglike, M_hat


def bin_sn_loglike(
    sn: SNLogLike,
    *,
    z_edges: np.ndarray,
    min_per_bin: int = 3,
) -> SNLogLike:
    """Compress SN data into redshift bins using BLUE weights with the full covariance.

    This returns an exact linear compression:
      m_bin = W m
      C_bin = W C W^T
    where each row of W is the generalized least-squares (BLUE) estimator of the bin mean.
    """
    z_edges = np.asarray(z_edges, dtype=float)
    if z_edges.ndim != 1 or z_edges.size < 3:
        raise ValueError("z_edges must be 1D with at least 3 edges.")
    if np.any(np.diff(z_edges) <= 0):
        raise ValueError("z_edges must be strictly increasing.")

    z = sn.z
    m = sn.m
    C = sn.cov

    # Build bins
    bin_indices: list[np.ndarray] = []
    for i in range(len(z_edges) - 1):
        lo, hi = float(z_edges[i]), float(z_edges[i + 1])
        if i < len(z_edges) - 2:
            idx = np.where((z >= lo) & (z < hi))[0]
        else:
            idx = np.where((z >= lo) & (z <= hi))[0]
        if idx.size >= min_per_bin:
            bin_indices.append(idx)

    if len(bin_indices) < 3:
        raise ValueError("Too few populated SN bins; increase z range or bin width.")

    weights: list[np.ndarray] = []
    z_bin = np.empty(len(bin_indices))
    m_bin = np.empty(len(bin_indices))

    for b, idx in enumerate(bin_indices):
        Cbb = C[np.ix_(idx, idx)]
        ones = np.ones(idx.size)
        cho = cho_factor(Cbb, lower=True, check_finite=False)
        cinv_ones = cho_solve(cho, ones, check_finite=False)
        norm = float(ones @ cinv_ones)
        if norm <= 0:
            raise ValueError("Non-positive BLUE normalization in SN binning.")
        w = cinv_ones / norm
        weights.append(w)
        z_bin[b] = float(w @ z[idx])
        m_bin[b] = float(w @ m[idx])

    # Covariance between bins
    nb = len(bin_indices)
    C_bin = np.empty((nb, nb))
    for i in range(nb):
        idx_i = bin_indices[i]
        w_i = weights[i]
        for j in range(i, nb):
            idx_j = bin_indices[j]
            w_j = weights[j]
            Cij = C[np.ix_(idx_i, idx_j)]
            cov_ij = float(w_i @ (Cij @ w_j))
            C_bin[i, j] = cov_ij
            C_bin[j, i] = cov_ij

    return SNLogLike.from_arrays(z=z_bin, m=m_bin, cov=C_bin)


@dataclass(frozen=True)
class ChronometerLogLike:
    z: np.ndarray
    H: np.ndarray
    sigma_H: np.ndarray

    @classmethod
    def from_data(
        cls,
        cc: Chronometers,
        *,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> "ChronometerLogLike":
        mask = np.ones_like(cc.z, dtype=bool)
        if z_min is not None:
            mask &= cc.z >= z_min
        if z_max is not None:
            mask &= cc.z <= z_max
        return cls(z=cc.z[mask], H=cc.H[mask], sigma_H=cc.sigma_H[mask])

    def loglike(self, H_model: np.ndarray) -> float:
        if H_model.shape != self.H.shape:
            raise ValueError("H_model shape mismatch.")
        chi2 = float(np.sum(((self.H - H_model) / self.sigma_H) ** 2))
        return -0.5 * chi2

    @classmethod
    def from_arrays(cls, *, z: np.ndarray, H: np.ndarray, sigma_H: np.ndarray) -> "ChronometerLogLike":
        z = np.asarray(z, dtype=float)
        H = np.asarray(H, dtype=float)
        sigma_H = np.asarray(sigma_H, dtype=float)
        if z.shape != H.shape or z.shape != sigma_H.shape:
            raise ValueError("z,H,sigma_H shape mismatch")
        return cls(z=z, H=H, sigma_H=sigma_H)


BAO_SPECS: dict[str, dict] = {
    # Alam et al. 2016 DR12 BAO-only consensus; values stored with rs_fid scaling.
    "sdss_dr12_consensus_bao": {
        "rs_fid": 147.78,  # Mpc
        "predict": "dr12_consensus_bao",
    },
    # eBOSS DR16 (LRG/QSO) typically stored directly as D_M/rs and D_H/rs.
    "sdss_dr16_lrg_bao_dmdh": {"predict": "dm_dh_over_rd"},
    "sdss_dr16_qso_bao_dmdh": {"predict": "dm_dh_over_rd"},
    "desi_2024_bao_all": {"predict": "desi_2024_all"},
}


@dataclass(frozen=True)
class BaoLogLike:
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
        bao: BaoMeasurements,
        *,
        dataset: str,
        constants: PhysicalConstants,
        z_min: float | None = None,
        z_max: float | None = None,
        diag_cov: bool = False,
    ) -> "BaoLogLike":
        if dataset not in BAO_SPECS:
            raise ValueError(f"Unknown BAO dataset spec '{dataset}'.")
        mask = np.ones_like(bao.z, dtype=bool)
        if z_min is not None:
            mask &= bao.z >= z_min
        if z_max is not None:
            mask &= bao.z <= z_max
        z = bao.z[mask]
        y = bao.value[mask]
        obs = bao.obs[mask].astype("U")
        cov = bao.cov[np.ix_(mask, mask)]
        if bool(diag_cov):
            cov = np.diag(np.diag(cov))
        if z.size == 0:
            raise ValueError(f"No BAO points in selected z-range for dataset '{dataset}'.")
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

    @classmethod
    def from_arrays(
        cls,
        *,
        dataset: str,
        z: np.ndarray,
        y: np.ndarray,
        obs: np.ndarray,
        cov: np.ndarray,
        constants: PhysicalConstants,
        diag_cov: bool = False,
    ) -> "BaoLogLike":
        if dataset not in BAO_SPECS:
            raise ValueError(f"Unknown BAO dataset spec '{dataset}'.")
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        obs = np.asarray(obs).astype("U")
        cov = np.asarray(cov, dtype=float)
        if bool(diag_cov):
            cov = np.diag(np.diag(cov))
        if z.ndim != 1 or y.ndim != 1 or obs.ndim != 1:
            raise ValueError("z,y,obs must be 1D arrays.")
        if not (z.size == y.size == obs.size):
            raise ValueError("z,y,obs lengths mismatch.")
        if cov.shape != (z.size, z.size):
            raise ValueError("cov shape mismatch.")
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

    def predict(self, bg: BackgroundCosmology, *, r_d_Mpc: float) -> np.ndarray:
        spec = BAO_SPECS[self.dataset]
        mode = spec["predict"]
        if mode == "dr12_consensus_bao":
            rs_fid = float(spec["rs_fid"])
            pred = np.empty_like(self.y)
            for i, (z, obs) in enumerate(zip(self.z, self.obs, strict=True)):
                if obs == "DM_over_rs":
                    pred[i] = bg.Dm(np.array([z]))[0] * (rs_fid / r_d_Mpc)
                elif obs == "bao_Hz_rs":
                    pred[i] = bg.H(np.array([z]))[0] * (r_d_Mpc / rs_fid)
                else:
                    raise ValueError(f"Unsupported BAO observable '{obs}' for {self.dataset}.")
            return pred
        if mode == "dm_dh_over_rd":
            pred = np.empty_like(self.y)
            for i, (z, obs) in enumerate(zip(self.z, self.obs, strict=True)):
                if obs == "DM_over_rs":
                    pred[i] = bg.Dm(np.array([z]))[0] / r_d_Mpc
                elif obs == "DH_over_rs":
                    pred[i] = bg.Dh(np.array([z]))[0] / r_d_Mpc
                else:
                    raise ValueError(f"Unsupported BAO observable '{obs}' for {self.dataset}.")
            return pred
        if mode == "desi_2024_all":
            pred = np.empty_like(self.y)
            for i, (z, obs) in enumerate(zip(self.z, self.obs, strict=True)):
                if obs == "DM_over_rs":
                    pred[i] = bg.Dm(np.array([z]))[0] / r_d_Mpc
                elif obs == "DH_over_rs":
                    pred[i] = bg.Dh(np.array([z]))[0] / r_d_Mpc
                elif obs == "DV_over_rs":
                    pred[i] = bg.Dv(np.array([z]))[0] / r_d_Mpc
                else:
                    raise ValueError(f"Unsupported BAO observable '{obs}' for {self.dataset}.")
            return pred
        raise ValueError(f"Unknown BAO prediction mode '{mode}'.")

    def ordering(self) -> list[tuple[str, float, str]]:
        """Return the exact y-vector ordering as (dataset, z, obs) tuples."""
        return [(self.dataset, float(z), str(obs)) for z, obs in zip(self.z, self.obs, strict=True)]

    def loglike(self, y_model: np.ndarray) -> float:
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        chi2 = float(r @ cho_solve(self.cov_cho, r, check_finite=False))
        return -0.5 * (chi2 + self.logdet)

    def chi2(self, y_model: np.ndarray) -> float:
        """Return chi^2 = r^T C^{-1} r for the current covariance."""
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        return float(r @ cho_solve(self.cov_cho, r, check_finite=False))

    def chi2_diag(self, y_model: np.ndarray) -> float:
        """Return chi^2 using only diag(C), i.e. r^T diag(C)^{-1} r."""
        if y_model.shape != self.y.shape:
            raise ValueError("y_model shape mismatch.")
        r = self.y - y_model
        d = np.diag(self.cov)
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError("Non-positive or non-finite diagonal in BAO covariance.")
        return float(np.sum((r * r) / d))

    def max_abs_offdiag_cov(self) -> float:
        if self.cov.size == 0:
            return 0.0
        off = self.cov - np.diag(np.diag(self.cov))
        return float(np.max(np.abs(off)))
