from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

from .cache import DataPaths, make_pooch


@dataclass(frozen=True)
class FsBaoMeasurements:
    """Full-shape BAO + RSD (fσ8) compressed measurements with full covariance."""

    z: np.ndarray
    value: np.ndarray
    obs: np.ndarray
    cov: np.ndarray
    meta: dict


def load_fsbao(*, paths: DataPaths, dataset: str = "sdss_dr16_lrg_fsbao_dmdhfs8") -> FsBaoMeasurements:
    """Load a full-shape BAO+RSD (FSBAO) dataset with full covariance.

    Notes
    -----
    We use the same provenance-tracked CobayaSampler/bao_data repository as for BAO-only
    data. These datasets provide correlated constraints on distance measures and fσ8(z),
    which helps break Ωm0–μ(A) degeneracies more strongly than heterogeneous diagonal
    fσ8 compilations.
    """
    supported = {
        # DR12 full-shape consensus: (D_M * rs_fid/rs, H * rs/rs_fid, fσ8) at z=0.38/0.51/0.61.
        "sdss_dr12_consensus_fs": (
            "sdss_DR12Consensus_FS.dat",
            "FS_consensus_covtot_dM_Hz_fsig.txt",
        ),
        # eBOSS DR16 BAO+FSBAO (LRG): (D_M/rs, D_H/rs, fσ8) at z=0.38/0.51/0.698.
        "sdss_dr16_lrg_fsbao_dmdhfs8": (
            "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat",
            "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt",
        ),
        # eBOSS DR16 BAO+FSBAO (QSO): (D_M/rs, D_H/rs, fσ8) at z=1.48.
        "sdss_dr16_qso_fsbao_dmdhfs8": (
            "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat",
            "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt",
        ),
    }
    if dataset not in supported:
        raise ValueError(f"Unsupported FSBAO dataset '{dataset}'. Supported: {sorted(supported)}")

    processed_dir = paths.processed_dir / "fsbao"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"fsbao_{dataset}.npz"
    processed_meta = processed_dir / f"fsbao_{dataset}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        npz = np.load(processed_npz, allow_pickle=False)
        try:
            obs = np.asarray(npz["obs"]).astype("U")
        except ValueError:
            # Backward-compatible: allow object arrays.
            npz = np.load(processed_npz, allow_pickle=True)
            obs = np.asarray(npz["obs"]).astype("U")
        return FsBaoMeasurements(z=npz["z"], value=npz["value"], obs=obs, cov=npz["cov"], meta=meta)

    commit = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
    base_url = f"https://raw.githubusercontent.com/CobayaSampler/bao_data/{commit}/"
    registry = {
        "sdss_DR12Consensus_FS.dat": "sha256:16c28eb4cee0c0a8c5e645b381f25f39c469477e44194e3ba5dcdb78b213ddc0",
        "FS_consensus_covtot_dM_Hz_fsig.txt": "sha256:a7e5d4a757b39591eb3db56414786cfaddbd322ae9178f00c7bf6be63ecb68a6",
        "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat": "sha256:a098ea4df320ac1c18a9404237a75ae26953e16403a20294beb1d9573be33c56",
        "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt": "sha256:409cabbf4ccf6993053427f5a34d52e6557f2429c17777267459471180e72f96",
        "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat": "sha256:cddd6cbbca7dadc910a5e8742f1f2144c066cb347b8ba03ae0bd4876fa06d8ed",
        "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt": "sha256:88f844447fb546792769cdf09b4df7b7a7f77a948f02ef371f54a6f7dddb3d41",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    meas_file, cov_file = supported[dataset]
    downloader = pooch.HTTPDownloader(timeout=120)
    meas_path = Path(p.fetch(meas_file, downloader=downloader))
    cov_path = Path(p.fetch(cov_file, downloader=downloader))

    meas = pd.read_csv(meas_path, sep=r"\s+", header=None, names=["z", "value", "obs"], comment="#")
    cov = np.loadtxt(cov_path, dtype=float)
    if cov.ndim == 1:
        n = int(np.sqrt(cov.size))
        cov = cov.reshape((n, n))
    if cov.shape[0] != len(meas):
        raise ValueError(f"FSBAO covariance shape {cov.shape} does not match N={len(meas)}")

    z = meas["z"].to_numpy(dtype=float)
    obs = meas["obs"].astype(str).to_numpy(dtype="U")
    val = meas["value"].to_numpy(dtype=float)

    meta = {
        "source": "CobayaSampler/bao_data",
        "commit": commit,
        "dataset": dataset,
        "measurement_file": meas_file,
        "cov_file": cov_file,
        "n_total": int(len(meas)),
        "obs": sorted(set(obs.tolist())),
    }
    np.savez_compressed(processed_npz, z=z, value=val, obs=np.asarray(obs, dtype="U"), cov=cov)
    pd.Series(meta).to_json(processed_meta, indent=2)
    return FsBaoMeasurements(z=z, value=val, obs=obs, cov=cov, meta=meta)

