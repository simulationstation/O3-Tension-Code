from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

from .cache import DataPaths, make_pooch


@dataclass(frozen=True)
class PantheonPlus:
    z: np.ndarray
    m_b_corr: np.ndarray
    cov: np.ndarray
    meta: dict


@dataclass(frozen=True)
class PantheonPlusSky:
    """Pantheon+ subset with sky positions (for anisotropy tests).

    Notes
    -----
    Pantheon+ provides RA/DEC in the public release table. We keep them in degrees.
    """

    z: np.ndarray
    m_b_corr: np.ndarray
    cov: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    idsurvey: np.ndarray
    meta: dict


def _load_cov_stream(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    n = int(first)
    flat = np.loadtxt(path, skiprows=1, dtype=float)
    if flat.size != n * n:
        raise ValueError(f"Covariance file has {flat.size} entries but expected {n*n}.")
    return flat.reshape((n, n))


def load_pantheon_plus(
    *,
    paths: DataPaths,
    cov_kind: str = "stat+sys",
    subset: str = "cosmology",
    z_column: str = "zHD",
) -> PantheonPlus:
    """Load Pantheon+ data from the official PantheonPlusSH0ES release.

    Parameters
    ----------
    cov_kind:
        Either "stat+sys" or "statonly".
    subset:
        - "cosmology": exclude Cepheid calibrators (IS_CALIBRATOR == 0). This is the
          default late-time Hubble diagram sample.
        - "shoes_hubble_flow": the low-z SH0ES Hubble-flow subset (USED_IN_SH0ES_HF == 1).
        - "all": keep all entries (including calibrators).
    z_column:
        One of "zHD", "zCMB", "zHEL". Default: "zHD" (recommended for the Hubble diagram).
    """
    if cov_kind not in {"stat+sys", "statonly"}:
        raise ValueError("cov_kind must be 'stat+sys' or 'statonly'.")
    if subset not in {"cosmology", "all", "shoes_hubble_flow"}:
        raise ValueError("subset must be 'cosmology', 'all', or 'shoes_hubble_flow'.")
    if z_column not in {"zHD", "zCMB", "zHEL"}:
        raise ValueError("z_column must be one of: zHD, zCMB, zHEL.")

    processed_dir = paths.processed_dir / "pantheon_plus"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"pantheon_plus_{subset}_{cov_kind}_{z_column}.npz"
    processed_meta = processed_dir / f"pantheon_plus_{subset}_{cov_kind}_{z_column}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        npz = np.load(processed_npz)
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        return PantheonPlus(
            z=npz["z"] if "z" in npz else npz["z_cmb"],
            m_b_corr=npz["m_b_corr"],
            cov=npz["cov"],
            meta=meta,
        )

    commit = "c447f0fea703fcd0fff57de5000947b5ca81286b"
    base_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
        f"{commit}/Pantheon+_Data/4_DISTANCES_AND_COVAR/"
    )
    cov_file = (
        "Pantheon+SH0ES_STAT+SYS.cov"
        if cov_kind == "stat+sys"
        else "Pantheon+SH0ES_STATONLY.cov"
    )
    registry = {
        "Pantheon+SH0ES.dat": "sha256:1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
        "Pantheon+SH0ES_STAT+SYS.cov": "sha256:abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
        "Pantheon+SH0ES_STATONLY.cov": "sha256:9f177129a332735d3637affd20054080d5260815f3ca0809120c05b2c902297f",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    downloader = pooch.HTTPDownloader(timeout=600)
    dat_path = Path(p.fetch("Pantheon+SH0ES.dat", downloader=downloader))
    cov_path = Path(p.fetch(cov_file, downloader=downloader))

    df = pd.read_csv(dat_path, sep=r"\s+", comment="#")
    if subset == "cosmology":
        mask = df["IS_CALIBRATOR"].to_numpy(dtype=int) == 0
    elif subset == "shoes_hubble_flow":
        mask = df["USED_IN_SH0ES_HF"].to_numpy(dtype=int) == 1
    else:
        mask = np.ones(len(df), dtype=bool)

    cov_full = _load_cov_stream(cov_path)
    if cov_full.shape[0] != len(df):
        raise ValueError(
            f"Pantheon+ covariance size {cov_full.shape} does not match N={len(df)} rows."
        )
    cov = cov_full[np.ix_(mask, mask)]

    z = df.loc[mask, z_column].to_numpy(dtype=float)
    m_b_corr = df.loc[mask, "m_b_corr"].to_numpy(dtype=float)

    meta = {
        "source": "PantheonPlusSH0ES/DataRelease",
        "commit": commit,
        "cov_kind": cov_kind,
        "subset": subset,
        "z_column": z_column,
        "n": int(mask.sum()),
        "n_total": int(len(df)),
        "columns": list(df.columns),
    }
    np.savez_compressed(processed_npz, z=z, m_b_corr=m_b_corr, cov=cov)
    pd.Series(meta).to_json(processed_meta, indent=2)
    return PantheonPlus(z=z, m_b_corr=m_b_corr, cov=cov, meta=meta)


def load_pantheon_plus_sky(
    *,
    paths: DataPaths,
    cov_kind: str = "stat+sys",
    subset: str = "cosmology",
    z_column: str = "zHD",
) -> PantheonPlusSky:
    """Load Pantheon+ including sky coordinates (RA/DEC) and survey ID.

    This mirrors load_pantheon_plus() but additionally returns:
    - ra_deg, dec_deg (degrees)
    - idsurvey (integer survey ID from the release)
    """
    if cov_kind not in {"stat+sys", "statonly"}:
        raise ValueError("cov_kind must be 'stat+sys' or 'statonly'.")
    if subset not in {"cosmology", "all", "shoes_hubble_flow"}:
        raise ValueError("subset must be 'cosmology', 'all', or 'shoes_hubble_flow'.")
    if z_column not in {"zHD", "zCMB", "zHEL"}:
        raise ValueError("z_column must be one of: zHD, zCMB, zHEL.")

    processed_dir = paths.processed_dir / "pantheon_plus"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"pantheon_plus_sky_{subset}_{cov_kind}_{z_column}.npz"
    processed_meta = processed_dir / f"pantheon_plus_sky_{subset}_{cov_kind}_{z_column}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        npz = np.load(processed_npz)
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        return PantheonPlusSky(
            z=npz["z"],
            m_b_corr=npz["m_b_corr"],
            cov=npz["cov"],
            ra_deg=npz["ra_deg"],
            dec_deg=npz["dec_deg"],
            idsurvey=npz["idsurvey"],
            meta=meta,
        )

    # Keep provenance identical to load_pantheon_plus().
    commit = "c447f0fea703fcd0fff57de5000947b5ca81286b"
    base_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
        f"{commit}/Pantheon+_Data/4_DISTANCES_AND_COVAR/"
    )
    cov_file = (
        "Pantheon+SH0ES_STAT+SYS.cov"
        if cov_kind == "stat+sys"
        else "Pantheon+SH0ES_STATONLY.cov"
    )
    registry = {
        "Pantheon+SH0ES.dat": "sha256:1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
        "Pantheon+SH0ES_STAT+SYS.cov": "sha256:abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
        "Pantheon+SH0ES_STATONLY.cov": "sha256:9f177129a332735d3637affd20054080d5260815f3ca0809120c05b2c902297f",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    downloader = pooch.HTTPDownloader(timeout=600)
    dat_path = Path(p.fetch("Pantheon+SH0ES.dat", downloader=downloader))
    cov_path = Path(p.fetch(cov_file, downloader=downloader))

    df = pd.read_csv(dat_path, sep=r"\s+", comment="#")
    if subset == "cosmology":
        mask = df["IS_CALIBRATOR"].to_numpy(dtype=int) == 0
    elif subset == "shoes_hubble_flow":
        mask = df["USED_IN_SH0ES_HF"].to_numpy(dtype=int) == 1
    else:
        mask = np.ones(len(df), dtype=bool)

    cov_full = _load_cov_stream(cov_path)
    if cov_full.shape[0] != len(df):
        raise ValueError(
            f"Pantheon+ covariance size {cov_full.shape} does not match N={len(df)} rows."
        )
    cov = cov_full[np.ix_(mask, mask)]

    z = df.loc[mask, z_column].to_numpy(dtype=float)
    m_b_corr = df.loc[mask, "m_b_corr"].to_numpy(dtype=float)
    ra_deg = df.loc[mask, "RA"].to_numpy(dtype=float)
    dec_deg = df.loc[mask, "DEC"].to_numpy(dtype=float)
    idsurvey = df.loc[mask, "IDSURVEY"].to_numpy(dtype=int)

    meta = {
        "source": "PantheonPlusSH0ES/DataRelease",
        "commit": commit,
        "cov_kind": cov_kind,
        "subset": subset,
        "z_column": z_column,
        "n": int(mask.sum()),
        "n_total": int(len(df)),
        "columns": list(df.columns),
    }
    np.savez_compressed(
        processed_npz,
        z=z,
        m_b_corr=m_b_corr,
        cov=cov,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        idsurvey=idsurvey,
    )
    pd.Series(meta).to_json(processed_meta, indent=2)
    return PantheonPlusSky(
        z=z,
        m_b_corr=m_b_corr,
        cov=cov,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        idsurvey=idsurvey,
        meta=meta,
    )


@dataclass(frozen=True)
class Chronometers:
    z: np.ndarray
    H: np.ndarray
    sigma_H: np.ndarray
    meta: dict


def load_chronometers(
    *,
    paths: DataPaths,
    variant: str = "BC03_all",
) -> Chronometers:
    if variant not in {"BC03_all", "BC03", "MaStro"}:
        raise ValueError("variant must be one of: BC03_all, BC03, MaStro")

    processed_dir = paths.processed_dir / "chronometers"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"chronometers_{variant}.npz"
    processed_meta = processed_dir / f"chronometers_{variant}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        npz = np.load(processed_npz)
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        return Chronometers(z=npz["z"], H=npz["H"], sigma_H=npz["sigma_H"], meta=meta)

    base_url = "https://raw.githubusercontent.com/baudren/montepython_public/2.2/data/cosmic_clocks/"
    file_map = {"BC03_all": "Hz_BC03_all.dat", "BC03": "Hz_BC03.dat", "MaStro": "Hz_MaStro.dat"}
    registry = {
        "Hz_BC03_all.dat": "sha256:8168b48ada7b5a66009c2ad71b0b7136b2fcf26ef50c9f59703f840dedf14692",
        "Hz_BC03.dat": "sha256:c8d44b0d769b783bfc3c7822459be7a263160ec337d27b809a8f2104a8a00f71",
        "Hz_MaStro.dat": "sha256:2748d5949bad1af2f44bf0f587580357c4e660fe6c43446bb124e195d6a0da8a",
        "Readme.txt": None,
    }

    # pooch requires hashes for registry entries; omit Readme if not pinned.
    reg = {k: v for k, v in registry.items() if v is not None}
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=reg)
    downloader = pooch.HTTPDownloader(timeout=120)
    data_path = Path(p.fetch(file_map[variant], downloader=downloader))

    df = pd.read_csv(data_path, comment="#", sep=r"\s+", names=["z", "H", "sigma_H"])
    df = df.dropna()
    z = df["z"].to_numpy(dtype=float)
    H = df["H"].to_numpy(dtype=float)
    sigma_H = df["sigma_H"].to_numpy(dtype=float)
    order = np.argsort(z)
    z, H, sigma_H = z[order], H[order], sigma_H[order]

    meta = {
        "source": "baudren/montepython_public (tag 2.2)",
        "variant": variant,
        "file": file_map[variant],
        "n": int(len(z)),
    }
    np.savez_compressed(processed_npz, z=z, H=H, sigma_H=sigma_H)
    pd.Series(meta).to_json(processed_meta, indent=2)
    return Chronometers(z=z, H=H, sigma_H=sigma_H, meta=meta)


@dataclass(frozen=True)
class BaoMeasurements:
    z: np.ndarray
    value: np.ndarray
    obs: np.ndarray
    cov: np.ndarray
    meta: dict


def load_bao(
    *,
    paths: DataPaths,
    dataset: str = "sdss_dr12_consensus_bao",
) -> BaoMeasurements:
    """Load a BAO dataset with a full covariance matrix (where provided).

    This loader uses the CobayaSampler/bao_data public repository (pinned by commit)
    as a provenance-tracked source of curated BAO compilation files.
    """
    supported = {
        "sdss_dr12_consensus_bao": (
            "sdss_DR12Consensus_bao.dat",
            "BAO_consensus_covtot_dM_Hz.txt",
        ),
        "sdss_dr16_lrg_bao_dmdh": (
            "sdss_DR16_LRG_BAO_DMDH.dat",
            "sdss_DR16_LRG_BAO_DMDH_covtot.txt",
        ),
        "sdss_dr16_qso_bao_dmdh": (
            "sdss_DR16_QSO_BAO_DMDH.txt",
            "sdss_DR16_QSO_BAO_DMDH_covtot.txt",
        ),
        "desi_2024_bao_all": (
            "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
            "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
        ),
    }
    if dataset not in supported:
        raise ValueError(f"Unsupported BAO dataset '{dataset}'. Supported: {sorted(supported)}")

    processed_dir = paths.processed_dir / "bao"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"bao_{dataset}.npz"
    processed_meta = processed_dir / f"bao_{dataset}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        npz = np.load(processed_npz, allow_pickle=False)
        try:
            obs = np.asarray(npz["obs"]).astype("U")
        except ValueError:
            # Backward-compatible load for earlier cached files that stored strings as object arrays.
            npz = np.load(processed_npz, allow_pickle=True)
            obs = np.asarray(npz["obs"]).astype("U")
        return BaoMeasurements(z=npz["z"], value=npz["value"], obs=obs, cov=npz["cov"], meta=meta)

    commit = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
    base_url = f"https://raw.githubusercontent.com/CobayaSampler/bao_data/{commit}/"
    registry = {
        "sdss_DR12Consensus_bao.dat": "sha256:fc43f1cd9c815bb58b09f4d8d1d272d2c4ec57e05e4893e2121c20dc08f4f862",
        "BAO_consensus_covtot_dM_Hz.txt": "sha256:05c04829c8edc117870efe809494593a23de6c35547f8b66760a5250804b65cf",
        "sdss_DR16_LRG_BAO_DMDH.dat": "sha256:b3317e7590799fad71a9a707023d0743c14d87399d6bb4129965d6a5732d91be",
        "sdss_DR16_LRG_BAO_DMDH_covtot.txt": "sha256:1a45e106f8e2bbf8742a6c3d4a9c11bdc288801fc6824e0db8cfbab4290f6160",
        "sdss_DR16_QSO_BAO_DMDH.txt": "sha256:9d3a43515d009d5c836728d4af1f1d02887fcdd874aba098c597f1f47693bbe6",
        "sdss_DR16_QSO_BAO_DMDH_covtot.txt": "sha256:c0d8bab47132045139c5bbd0ebfd8464434e1354371ceeeca70bb90ecbcee383",
        "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt": "sha256:dd2873a0b88459a491af3c0c0307ba059f62df9211d5b976760f310565a1be68",
        "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt": "sha256:bbafa9074b51cf1a45e0d10e4f37db8c0e80a5d1d1788857abb7fc49fb21abcc",
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
        raise ValueError(f"BAO covariance shape {cov.shape} does not match N={len(meas)}")

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
    return BaoMeasurements(z=z, value=val, obs=obs, cov=cov, meta=meta)
