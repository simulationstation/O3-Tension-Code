from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

from .cache import DataPaths, make_pooch


@dataclass(frozen=True)
class PlanckLensingBandpowers:
    ell_min: np.ndarray
    ell_max: np.ndarray
    ell_eff: np.ndarray
    clpp: np.ndarray
    cov: np.ndarray
    template_clpp: np.ndarray
    meta: dict


def load_planck_lensing_bandpowers(*, paths: DataPaths, dataset: str = "consext8") -> PlanckLensingBandpowers:
    """Load Planck 2018 lensing bandpowers and covariance from CobayaSampler data repo.

    The template_clpp uses PP/Ahat from the bandpowers file (fiducial spectrum proxy).
    """
    supported = {
        "consext8": (
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_bandpowers.dat",
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_cov.dat",
        ),
        "agr2": (
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2_bandpowers.dat",
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2_cov.dat",
        ),
    }
    if dataset not in supported:
        raise ValueError(f"Unsupported lensing bandpower dataset '{dataset}'.")

    processed_dir = paths.processed_dir / "planck_lensing_bandpowers"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"planck_lensing_{dataset}.npz"
    processed_meta = processed_dir / f"planck_lensing_{dataset}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        npz = np.load(processed_npz, allow_pickle=False)
        return PlanckLensingBandpowers(
            ell_min=npz["ell_min"],
            ell_max=npz["ell_max"],
            ell_eff=npz["ell_eff"],
            clpp=npz["clpp"],
            cov=npz["cov"],
            template_clpp=npz["template_clpp"],
            meta=meta,
        )

    commit = "4c160c735f2f3741c2429160743b1faca7a29474"
    base_url = f"https://raw.githubusercontent.com/CobayaSampler/planck_supp_data_and_covmats/{commit}/lensing/2018/"
    registry = {
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_bandpowers.dat": "sha256:0113871c95b026dbf544c21f3c0cd667bea25ad146dddb93db4189cff660a6f0",
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_cov.dat": "sha256:fdd19b43dacd3c65a3d092442c291401a3497cc4fddf9ce08bb098d5a428efc0",
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2_bandpowers.dat": "sha256:022ecd45a4dc0d8ea42643b084811e5ff295040cbc3fead7d4e7ecf6c0c2254e",
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2_cov.dat": "sha256:b9e25a798e6b30a6503981e71b0d1f963d040d909236c2b45f6c3d97224170cd",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    bp_file, cov_file = supported[dataset]
    downloader = pooch.HTTPDownloader(timeout=120)
    bp_path = Path(p.fetch(bp_file, downloader=downloader))
    cov_path = Path(p.fetch(cov_file, downloader=downloader))

    cols = ["bin", "ell_min", "ell_max", "ell_eff", "clpp", "error", "ahat"]
    df = pd.read_csv(bp_path, sep=r"\s+", comment="#", header=None, names=cols)
    cov = np.loadtxt(cov_path)
    if cov.ndim == 1:
        n = int(np.sqrt(cov.size))
        cov = cov.reshape((n, n))
    if cov.shape[0] != len(df):
        raise ValueError("Lensing covariance shape mismatch.")

    template = df["clpp"].to_numpy(dtype=float) / df["ahat"].to_numpy(dtype=float)

    meta = {
        "source": "CobayaSampler/planck_supp_data_and_covmats",
        "commit": commit,
        "dataset": dataset,
        "bandpowers_file": bp_file,
        "cov_file": cov_file,
        "n_bins": int(len(df)),
    }

    np.savez_compressed(
        processed_npz,
        ell_min=df["ell_min"].to_numpy(dtype=float),
        ell_max=df["ell_max"].to_numpy(dtype=float),
        ell_eff=df["ell_eff"].to_numpy(dtype=float),
        clpp=df["clpp"].to_numpy(dtype=float),
        cov=cov,
        template_clpp=template,
    )
    pd.Series(meta).to_json(processed_meta, indent=2)
    return PlanckLensingBandpowers(
        ell_min=df["ell_min"].to_numpy(dtype=float),
        ell_max=df["ell_max"].to_numpy(dtype=float),
        ell_eff=df["ell_eff"].to_numpy(dtype=float),
        clpp=df["clpp"].to_numpy(dtype=float),
        cov=cov,
        template_clpp=template,
        meta=meta,
    )
