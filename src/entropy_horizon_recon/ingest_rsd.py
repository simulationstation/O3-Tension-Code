from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pooch

from .cache import DataPaths, make_pooch


@dataclass(frozen=True)
class RsdFs8:
    z: np.ndarray
    fs8: np.ndarray
    sigma_fs8: np.ndarray
    omega_m0_fid: np.ndarray
    meta: dict


def load_rsd_fs8(*, paths: DataPaths, compilation: str = "kazantzidis2018") -> RsdFs8:
    """Load an fσ8(z) compilation (diagonal errors).

    Notes
    -----
    This is used as an *external* late-time growth constraint to help break the
    Ωm0–μ(A) degeneracy. We start with diagonal errors because full covariances
    are not consistently available across heterogeneous surveys in simple public
    compendia.
    """
    if compilation != "kazantzidis2018":
        raise ValueError("Only compilation='kazantzidis2018' is currently supported.")

    processed_dir = paths.processed_dir / "rsd"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / "rsd_fs8_kazantzidis2018.npz"
    processed_meta = processed_dir / "rsd_fs8_kazantzidis2018.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        npz = np.load(processed_npz)
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        return RsdFs8(
            z=npz["z"],
            fs8=npz["fs8"],
            sigma_fs8=npz["sigma_fs8"],
            omega_m0_fid=npz["omega_m0_fid"],
            meta=meta,
        )

    # Source: Lavrentios Kazantzidis & Leandros Perivolaropoulos (2018) "Evolution of the
    # fσ8 tension with the Planck15-ΛCDM determination and implications for modified gravity theories"
    # Table II (as provided in the accompanying "growth-tomography" code/data release).
    commit = "cbec97c3641adc2d409b2772bc07bcaa3b64079e"
    base_url = f"https://raw.githubusercontent.com/lkazantzi/growth-tomography/{commit}/"
    registry = {
        "growth_tomography.zip": "sha256:89a50e949578d06ded60c5020eb3ea96da2e20400f30fbbaaf7a20b44c8ba8ce",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    downloader = pooch.HTTPDownloader(timeout=600)
    zip_path = Path(p.fetch("growth_tomography.zip", downloader=downloader))

    # Extract the table from the zip without writing temporary files.
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("data/Growth_tableII.txt").decode("utf-8", errors="replace")
    arr = np.loadtxt(raw.splitlines(), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Unexpected Growth_tableII.txt format; expected >=3 columns (z, fs8, sigma[, Om_fid]).")
    if arr.shape[1] == 3:
        z, fs8, sig = arr[:, 0], arr[:, 1], arr[:, 2]
        om_fid = np.full_like(z, np.nan, dtype=float)
    else:
        z, fs8, sig, om_fid = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    order = np.argsort(z)
    z, fs8, sig, om_fid = z[order], fs8[order], sig[order], om_fid[order]

    meta = {
        "source": "lkazantzi/growth-tomography",
        "commit": commit,
        "file": "growth_tomography.zip:data/Growth_tableII.txt",
        "compilation": "kazantzidis2018_tableII",
        "columns": ["z", "fs8", "sigma_fs8", "omega_m0_fid"],
        "n": int(z.size),
        "covariance": "diagonal (heterogeneous compilation; no unified full covariance provided)",
    }

    # Also write a convenient CSV+meta into data/cache/ for inspection.
    cache_csv = paths.pooch_cache_dir / "rsd_fs8.csv"
    cache_meta = paths.pooch_cache_dir / "rsd_fs8.meta.json"
    df = pd.DataFrame({"z": z, "fs8": fs8, "sigma_fs8": sig, "omega_m0_fid": om_fid})
    df.to_csv(cache_csv, index=False)
    pd.Series(meta).to_json(cache_meta, indent=2)

    np.savez_compressed(processed_npz, z=z, fs8=fs8, sigma_fs8=sig, omega_m0_fid=om_fid)
    pd.Series(meta).to_json(processed_meta, indent=2)
    return RsdFs8(z=z, fs8=fs8, sigma_fs8=sig, omega_m0_fid=om_fid, meta=meta)

