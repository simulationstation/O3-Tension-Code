from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import hashlib

import numpy as np
import pandas as pd
import requests

from .cache import DataPaths


@dataclass(frozen=True)
class FullShapePk:
    k: np.ndarray
    pk: np.ndarray
    cov: np.ndarray
    z_eff: float
    meta: dict


def _download_gdrive(file_id: str, dest: Path) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    resp = session.get(url, stream=True)
    if 'text/html' in resp.headers.get('Content-Type', ''):
        text = resp.text
        token = None
        for part in text.split('confirm=')[1:]:
            token = part.split('&', 1)[0]
            if token:
                break
        if token:
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            resp = session.get(url, stream=True)
    resp.raise_for_status()
    with dest.open('wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def load_fullshape_pk(*, paths: DataPaths, dataset: str = "shapefit_lrgz1_ngc_mono") -> FullShapePk:
    """Load full-shape P(k) monopole from Shapefit BOSS DR12 release.

    Uses LRG z1 NGC monopole with covariance submatrix (monopole only).
    """
    supported = {
        "shapefit_lrgz1_ngc_mono": {
            "file_id": "12G2Snz5U1PzyNnJAX4oUQGnwF8Zb2ZQZ",
            "sha256": "90c1ebe1c4285002e493fcf2d00ce3f6cf153463ad96dfdf40917fa99467b576",
            "tar_name": "Pk.tar.gz",
            "data_file": "LRGz1/Monopole_LRGz1_NGC.txt",
            "cov_file": "LRGz1/cov/CovariancePk_LRGz1_NGC.cov",
            "z_eff": 0.51,
        }
    }
    if dataset not in supported:
        raise ValueError(f"Unsupported full-shape pk dataset '{dataset}'.")

    info = supported[dataset]
    cache_dir = paths.pooch_cache_dir / "shapefit_pk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / info["tar_name"]

    if not tar_path.exists():
        _download_gdrive(info["file_id"], tar_path)
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    if sha != info["sha256"]:
        raise ValueError("Full-shape pk archive SHA256 mismatch.")

    processed_dir = paths.processed_dir / "fullshape_pk"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_npz = processed_dir / f"{dataset}.npz"
    processed_meta = processed_dir / f"{dataset}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        meta = pd.read_json(processed_meta, typ="series").to_dict()
        npz = np.load(processed_npz, allow_pickle=False)
        return FullShapePk(k=npz["k"], pk=npz["pk"], cov=npz["cov"], z_eff=float(npz["z_eff"]), meta=meta)

    with tarfile.open(tar_path, "r:gz") as tf:
        data_member = tf.extractfile(info["data_file"])
        cov_member = tf.extractfile(info["cov_file"])
        if data_member is None or cov_member is None:
            raise ValueError("Required data/cov file missing in archive.")
        data_text = data_member.read().decode("utf-8")
        cov_text = cov_member.read().decode("utf-8")

    data_rows = [ln for ln in data_text.splitlines() if ln.strip() and not ln.startswith("#")]
    k = []
    pk = []
    for ln in data_rows:
        parts = ln.split()
        k.append(float(parts[0]))
        pk.append(float(parts[1]))
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)

    cov = np.loadtxt([ln for ln in cov_text.splitlines() if ln.strip()])
    # Cov includes multipoles; use monopole block (top-left).
    n = k.size
    if cov.shape[0] < n:
        raise ValueError("Covariance smaller than data length.")
    cov = cov[:n, :n]

    meta = {
        "source": "shapefit BOSS DR12",
        "dataset": dataset,
        "data_file": info["data_file"],
        "cov_file": info["cov_file"],
        "z_eff": info["z_eff"],
        "n_points": int(k.size),
    }

    np.savez_compressed(processed_npz, k=k, pk=pk, cov=cov, z_eff=info["z_eff"])
    pd.Series(meta).to_json(processed_meta, indent=2)
    return FullShapePk(k=k, pk=pk, cov=cov, z_eff=float(info["z_eff"]), meta=meta)
