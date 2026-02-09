from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import DataPaths, make_pooch


@dataclass(frozen=True)
class ShearCatalog:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    weight: np.ndarray
    zbin: np.ndarray
    meta: dict[str, str]


def _select_column(cols: dict[str, str], names: list[str]) -> str:
    for name in names:
        if name in cols:
            return cols[name]
    raise KeyError(f"Missing required column (tried: {names})")


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".fits", ".fz"}:
        try:
            from astropy.table import Table
        except Exception as exc:  # pragma: no cover - runtime guard
            raise ImportError("astropy is required to read FITS shear catalogs.") from exc
        tab = Table.read(path)
        return tab.to_pandas()
    return pd.read_csv(path)


# NOTE: Update SHA256 after first successful retrieval for the full catalog.
# Zenodo record: 10.5281/zenodo.16366035 (KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits + kids_subset_safe.fits)
KIDS_BASE_URL = "https://zenodo.org/records/16366035/files/"
KIDS_FILES = {
    "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits": "sha256:TODO",
    "kids_subset_safe.fits": "sha256:caa5a5e5db898ed1d425718fbcc21facd5187f8762ab52dc6c8ddfcf77494b16",
}


def fetch_kids1000(*, paths: DataPaths, allow_unverified: bool = False, use_subset: bool = False) -> Path:
    registry = dict(KIDS_FILES)
    if not allow_unverified:
        for v in registry.values():
            if "TODO" in v:
                raise RuntimeError("SHA256 not set for KiDS file. Update KIDS_FILES after first download.")
    pooch = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=KIDS_BASE_URL, registry=registry)
    name = "kids_subset_safe.fits" if use_subset else "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
    return Path(pooch.fetch(name))


def load_kids1000_catalog(path: Path) -> ShearCatalog:
    """Load a KiDS-1000 shear catalog from a local file.

    Expected columns (case-insensitive):
      - RA/DEC (RAJ2000, ALPHA_J2000, RA; DECJ2000, DELTA_J2000, DEC)
      - e1/e2 (e1/e2, g1/g2)
      - weight (weight, w)
      - tomographic bin (tomo_bin, zbin) OR photo-z (z_b) to be binned.
    """
    df = _load_table(Path(path))
    cols = {c.lower(): c for c in df.columns}
    ra = df[_select_column(cols, ["ra", "raj2000", "alpha_j2000"])].to_numpy(dtype=float)
    dec = df[_select_column(cols, ["dec", "decj2000", "delta_j2000"])].to_numpy(dtype=float)
    e1 = df[_select_column(cols, ["e1", "g1"])].to_numpy(dtype=float)
    e2 = df[_select_column(cols, ["e2", "g2"])].to_numpy(dtype=float)
    w = df[_select_column(cols, ["weight", "w"])].to_numpy(dtype=float)
    if "zbin" in cols or "tomo_bin" in cols or "tomobin" in cols:
        zbin = df[_select_column(cols, ["zbin", "tomo_bin", "tomobin"])].to_numpy(dtype=int)
    elif "z_b" in cols:
        z_b = df[cols["z_b"]].to_numpy(dtype=float)
        # Bin into 5 tomographic bins by quantiles as a fallback.
        edges = np.quantile(z_b[np.isfinite(z_b)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        zbin = np.digitize(z_b, edges[1:-1], right=False)
    else:
        raise KeyError("No tomographic bin or photo-z column found for KiDS catalog.")
    return ShearCatalog(ra_deg=ra, dec_deg=dec, e1=e1, e2=e2, weight=w, zbin=zbin, meta={"source": path.name})


def load_kids1000(
    *, paths: DataPaths, allow_unverified: bool = False, local_path: Path | None = None, use_subset: bool = False
) -> ShearCatalog:
    if local_path is not None:
        return load_kids1000_catalog(Path(local_path))
    path = fetch_kids1000(paths=paths, allow_unverified=allow_unverified, use_subset=use_subset)
    return load_kids1000_catalog(path)
