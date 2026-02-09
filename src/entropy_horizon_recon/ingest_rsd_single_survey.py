from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DataPaths
from .ingest_fsbao import load_fsbao


@dataclass(frozen=True)
class RsdSingleSurvey:
    """Single-survey RSD fσ8(z) measurements with full covariance."""

    z: np.ndarray
    fs8: np.ndarray
    cov: np.ndarray
    meta: dict


def load_rsd_single_survey(*, paths: DataPaths, dataset: str = "sdss_dr16_lrg_fsbao_dmdhfs8") -> RsdSingleSurvey:
    """Load a single-survey RSD fσ8(z) dataset from an FSBAO source.

    Uses the fσ8 elements and corresponding covariance submatrix.
    """
    fsbao = load_fsbao(paths=paths, dataset=dataset)
    obs = np.asarray(fsbao.obs, dtype="U")
    mask = obs == "f_sigma8"
    if not np.any(mask):
        raise ValueError(f"No f_sigma8 entries found in FSBAO dataset '{dataset}'.")
    z = np.asarray(fsbao.z, dtype=float)[mask]
    fs8 = np.asarray(fsbao.value, dtype=float)[mask]
    cov = np.asarray(fsbao.cov, dtype=float)
    cov_sub = cov[np.ix_(mask, mask)]
    if cov_sub.shape[0] != z.size:
        raise ValueError("RSD covariance submatrix shape mismatch.")
    if not np.allclose(cov_sub, cov_sub.T, atol=1e-10):
        raise ValueError("RSD covariance submatrix is not symmetric.")
    meta = dict(fsbao.meta)
    meta.update({
        "source": "fsbao",
        "rsd_dataset": dataset,
        "n_points": int(z.size),
        "obs": "f_sigma8",
    })
    return RsdSingleSurvey(z=z, fs8=fs8, cov=cov_sub, meta=meta)
