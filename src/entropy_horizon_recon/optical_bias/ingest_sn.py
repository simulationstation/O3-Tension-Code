from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..cache import DataPaths, make_pooch

# NOTE: Update SHA256 after first successful retrieval.
# Zenodo record: 10.5281/zenodo.16365279 (Pantheon_SH0ES.dat).
PANTHEONP_BASE_URL = "https://zenodo.org/records/16365279/files/"
PANTHEONP_FILES = {
    "Pantheon_SH0ES.dat": "sha256:1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
}


@dataclass(frozen=True)
class SNDataset:
    z: np.ndarray
    mu: np.ndarray
    sigma_mu: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    meta: dict[str, str]


def _select_column(df: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of the columns found: {names}")


def _load_pantheon_file(path: Path) -> SNDataset:
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    z_col = _select_column(df, ["zHD", "zCMB", "z", "zcmb"])
    mu_col = _select_column(df, ["MU_SH0ES", "MU", "mu", "mB", "mucorr", "muobs", "m_b_corr"])
    sig_col = _select_column(df, ["MU_SH0ES_ERR_DIAG", "MUERR", "muerr", "sigmu", "MUERR_RAW", "m_b_corr_err_DIAG", "dmb"])
    ra_col = _select_column(df, ["RA", "ra", "RAdeg", "ra_deg", "RAJ2000", "ALPHA_J2000"])
    dec_col = _select_column(df, ["DEC", "dec", "DECdeg", "dec_deg", "DECJ2000", "DELTA_J2000"])

    z = df[z_col].to_numpy(dtype=float)
    mu = df[mu_col].to_numpy(dtype=float)
    sigma = df[sig_col].to_numpy(dtype=float)
    ra = df[ra_col].to_numpy(dtype=float)
    dec = df[dec_col].to_numpy(dtype=float)

    return SNDataset(z=z, mu=mu, sigma_mu=sigma, ra_deg=ra, dec_deg=dec, meta={"source": path.name})


def fetch_pantheon_plus(*, paths: DataPaths, allow_unverified: bool = False) -> Path:
    registry = dict(PANTHEONP_FILES)
    if not allow_unverified:
        for k, v in registry.items():
            if "TODO" in v:
                raise RuntimeError(
                    "SHA256 not set for Pantheon+ file. Update PANTHEONP_FILES after first download."
                )
    pooch = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=PANTHEONP_BASE_URL, registry=registry)
    return Path(pooch.fetch("Pantheon_SH0ES.dat"))


def load_sn_dataset(*, paths: DataPaths, allow_unverified: bool = False, local_path: Path | None = None) -> SNDataset:
    if local_path is not None:
        return _load_pantheon_file(Path(local_path))
    path = fetch_pantheon_plus(paths=paths, allow_unverified=allow_unverified)
    return _load_pantheon_file(path)
