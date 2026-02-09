from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pooch

from .cache import DataPaths


# -----------------------------------------------------------------------------
# Public 2MRS NeuralNet density + velocity fields (Lilow+ 2024, arXiv:2404.02278).
#
# The GitHub repo provides a Dropbox folder link with numpy arrays. Dropbox serves
# the folder as a zip download; we pin the zip SHA256 once fetched to ensure
# reproducibility.
# -----------------------------------------------------------------------------

TWO_MRS_NN_ZIP_FNAME = "2mrs_neuralnet_fields_lilow2024.zip"
TWO_MRS_NN_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/wb8iyg113hyin4ni7srkg/h"
    "?rlkey=bfry3x0s612qtnmgb6n82rnny&dl=1"
)

# Pinned after first successful retrieval on 2026-01-30.
TWO_MRS_NN_ZIP_SHA256 = "60ee42950f95720fe6d0c9491ccd617e93c3b08af5b892a84ff97b51d5f9595c"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_2mrs_neuralnet_zip(*, paths: DataPaths, allow_unverified: bool = False) -> Path:
    base = Path(paths.pooch_cache_dir) / "void_prism" / "2mrs_neuralnet"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / TWO_MRS_NN_ZIP_FNAME

    if dest.exists():
        if allow_unverified or "TODO" in TWO_MRS_NN_ZIP_SHA256:
            return dest
        got = _sha256(dest)
        if got != TWO_MRS_NN_ZIP_SHA256:
            raise ValueError("2MRS NeuralNet zip SHA256 mismatch.")
        return dest

    if "TODO" in TWO_MRS_NN_ZIP_SHA256 and not allow_unverified:
        raise RuntimeError(
            "TWO_MRS_NN_ZIP_SHA256 is not pinned yet. Download once with --allow-unverified, "
            "compute sha256, then pin TWO_MRS_NN_ZIP_SHA256."
        )

    if "TODO" in TWO_MRS_NN_ZIP_SHA256 and allow_unverified:
        out = pooch.retrieve(url=TWO_MRS_NN_ZIP_URL, known_hash=None, path=base, fname=TWO_MRS_NN_ZIP_FNAME, progressbar=False)
        return Path(out)

    out = pooch.retrieve(
        url=TWO_MRS_NN_ZIP_URL,
        known_hash=f"sha256:{TWO_MRS_NN_ZIP_SHA256}",
        path=base,
        fname=TWO_MRS_NN_ZIP_FNAME,
        progressbar=False,
    )
    return Path(out)


def _extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        return list(out_dir.rglob("*"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return list(out_dir.rglob("*"))


@dataclass(frozen=True)
class TwoMRSNeuralNetFields:
    density_1pdelta: np.ndarray
    vx_kms: np.ndarray
    vy_kms: np.ndarray
    vz_kms: np.ndarray
    density_err: np.ndarray | None
    vx_err: np.ndarray | None
    vy_err: np.ndarray | None
    vz_err: np.ndarray | None
    meta: dict[str, float | str]


def load_2mrs_neuralnet_fields(
    *,
    paths: DataPaths,
    allow_unverified: bool = False,
    include_errors: bool = True,
) -> TwoMRSNeuralNetFields:
    """Load the Lilow+ 2024 2MRS NeuralNet density/velocity grid from the pinned zip."""
    zip_path = fetch_2mrs_neuralnet_zip(paths=paths, allow_unverified=allow_unverified)
    base = zip_path.parent
    extract_dir = base / "unzipped"
    files = _extract_zip(zip_path, extract_dir)
    by_name = {p.name: p for p in files if p.is_file()}

    def _need(name: str) -> Path:
        p = by_name.get(name)
        if p is None:
            raise FileNotFoundError(f"Missing {name!r} in extracted 2MRS NeuralNet zip.")
        return p

    density = np.load(_need("density.npy"))
    vx = np.load(_need("xVelocity.npy"))
    vy = np.load(_need("yVelocity.npy"))
    vz = np.load(_need("zVelocity.npy"))

    dens_err = vx_err = vy_err = vz_err = None
    if include_errors:
        # Errors are optional; if absent we leave them None.
        dens_err = np.load(by_name["density_error.npy"]) if "density_error.npy" in by_name else None
        vx_err = np.load(by_name["xVelocity_error.npy"]) if "xVelocity_error.npy" in by_name else None
        vy_err = np.load(by_name["yVelocity_error.npy"]) if "yVelocity_error.npy" in by_name else None
        vz_err = np.load(by_name["zVelocity_error.npy"]) if "zVelocity_error.npy" in by_name else None

    meta = {
        "source": "Lilow+2024 2MRS NeuralNet (Dropbox zip)",
        "zip_name": zip_path.name,
        "grid_n": int(density.shape[0]),
        "grid_spacing_mpc_h": 3.125,
        "grid_side_mpc_h": 400.0,
        "valid_radius_mpc_h": 200.0,
    }
    return TwoMRSNeuralNetFields(
        density_1pdelta=np.asarray(density, dtype=float),
        vx_kms=np.asarray(vx, dtype=float),
        vy_kms=np.asarray(vy, dtype=float),
        vz_kms=np.asarray(vz, dtype=float),
        density_err=np.asarray(dens_err, dtype=float) if dens_err is not None else None,
        vx_err=np.asarray(vx_err, dtype=float) if vx_err is not None else None,
        vy_err=np.asarray(vy_err, dtype=float) if vy_err is not None else None,
        vz_err=np.asarray(vz_err, dtype=float) if vz_err is not None else None,
        meta=meta,
    )
