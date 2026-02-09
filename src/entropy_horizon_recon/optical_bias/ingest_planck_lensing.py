from __future__ import annotations

import tarfile
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pooch

from ..cache import DataPaths
from .maps import _require_healpy

# Planck 2018 lensing (PLA file id COM_Lensing_4096_R3.00.tgz).
# Direct link works without cookies as of 2026-01-27.
PLANCK_LENSING_FNAME = "COM_Lensing_4096_R3.00.tgz"
PLANCK_LENSING_URL = (
    "https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.FILE_ID=COM_Lensing_4096_R3.00.tgz"
)
# NOTE: Fill this in after first successful retrieval.
PLANCK_LENSING_SHA256 = "d3d99e50979bb6ae84e350d050feebf0714e7626d35d8c772ea5551309875835"


@dataclass(frozen=True)
class PlanckLensing:
    kappa_map: np.ndarray
    mask: np.ndarray | None
    nside: int
    meta: dict[str, str]


def _extract_tar(tar_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Avoid re-extracting on every run (the archive is large).
    if any(out_dir.iterdir()):
        return list(out_dir.rglob("*"))
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    return list(out_dir.rglob("*"))


def _find_first(paths: list[Path], patterns: list[str]) -> Path | None:
    for pat in patterns:
        for p in paths:
            if p.name.lower().endswith(pat.lower()):
                return p
    return None


def fetch_planck_tar(*, paths: DataPaths, allow_unverified: bool = False) -> Path:
    dest = paths.pooch_cache_dir / PLANCK_LENSING_FNAME
    if dest.exists():
        if allow_unverified or "TODO" in PLANCK_LENSING_SHA256:
            return dest
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha != PLANCK_LENSING_SHA256:
            raise ValueError("Planck lensing archive SHA256 mismatch.")
        return dest

    if "TODO" in PLANCK_LENSING_SHA256 and not allow_unverified:
        raise RuntimeError(
            "SHA256 not set for Planck lensing file. Download once with allow_unverified "
            "and then pin PLANCK_LENSING_SHA256."
        )

    if "TODO" in PLANCK_LENSING_SHA256 and allow_unverified:
        # Download without hash checking (first-time bootstrap).
        out = pooch.retrieve(url=PLANCK_LENSING_URL, known_hash=None, path=paths.pooch_cache_dir, fname=PLANCK_LENSING_FNAME)
        return Path(out)

    out = pooch.retrieve(
        url=PLANCK_LENSING_URL,
        known_hash=f"sha256:{PLANCK_LENSING_SHA256}",
        path=paths.pooch_cache_dir,
        fname=PLANCK_LENSING_FNAME,
        progressbar=False,
    )
    return Path(out)


def load_planck_kappa(*, paths: DataPaths, nside_out: int | None = None, allow_unverified: bool = False) -> PlanckLensing:
    hp = _require_healpy()

    # Cache a downgraded kappa map on disk so repeated experiments don't need to read/convert the
    # full high-lmax alm product every time (which can take minutes).
    if nside_out is not None:
        cache_dir = paths.processed_dir / "planck_lensing"
        cache_dir.mkdir(parents=True, exist_ok=True)
        kappa_cache = cache_dir / f"planck2018_kappa_nside{int(nside_out)}.fits"
        mask_cache = cache_dir / f"planck2018_mask_nside{int(nside_out)}.fits"
        if kappa_cache.exists():
            kappa_map = hp.read_map(str(kappa_cache), verbose=False)
            mask = hp.read_map(str(mask_cache), verbose=False) if mask_cache.exists() else None
            return PlanckLensing(
                kappa_map=np.asarray(kappa_map, dtype=float),
                mask=np.asarray(mask, dtype=float) if mask is not None else None,
                nside=int(hp.get_nside(kappa_map)),
                meta={"source": "cached", "kappa_cache": kappa_cache.name},
            )

    tar_path = fetch_planck_tar(paths=paths, allow_unverified=allow_unverified)
    extract_dir = paths.pooch_cache_dir / "planck_lensing"
    files = _extract_tar(tar_path, extract_dir)

    kappa_path = _find_first(files, ["_kappa.fits", "_kappa_map.fits", "_kappa.fits.gz"])
    if kappa_path is None:
        # Prefer kappa alm (klm) if present (Planck delivers klm in COM_Lensing_4096_R3.00.tgz).
        def _prefer_path(suffix: str) -> Path | None:
            suf = suffix.lower()
            for p in files:
                if str(p).lower().endswith(suf):
                    return p
            return None

        klm_path = _prefer_path("/mv/dat_klm.fits") or _prefer_path("/dat_klm.fits")
        mf_path = _prefer_path("/mv/mf_klm.fits") or _prefer_path("/mf_klm.fits")
        if klm_path is not None:
            alm_kappa = hp.read_alm(str(klm_path))
            if mf_path is not None:
                alm_kappa = alm_kappa - hp.read_alm(str(mf_path))
        else:
            # Fallback: lensing potential alm (plm/phi), convert to kappa.
            phi_path = _prefer_path("/mv/dat_plm.fits") or _prefer_path("/dat_plm.fits") or _prefer_path("/phi.fits")
            if phi_path is None:
                raise FileNotFoundError("Could not locate kappa map/klm or phi alm product in Planck lensing tarball.")
            alm_phi = hp.read_alm(str(phi_path))
            lmax = hp.Alm.getlmax(len(alm_phi))
            ell = np.arange(lmax + 1)
            factor = 0.5 * ell * (ell + 1.0)
            alm_kappa = hp.almxfl(alm_phi, factor)

        nside_target = int(nside_out) if nside_out is not None else 4096
        lmax_src = int(hp.Alm.getlmax(len(alm_kappa)))
        lmax_target = min(int(lmax_src), int(3 * nside_target - 1))

        if lmax_target < lmax_src:
            # Truncate alm to the smaller lmax expected by healpy.
            alm_trunc = np.zeros(hp.Alm.getsize(lmax_target), dtype=np.complex128)
            for m in range(lmax_target + 1):
                l_arr = np.arange(m, lmax_target + 1, dtype=int)
                idx_src = hp.Alm.getidx(lmax_src, l_arr, m)
                idx_tgt = hp.Alm.getidx(lmax_target, l_arr, m)
                alm_trunc[idx_tgt] = alm_kappa[idx_src]
            alm_kappa = alm_trunc
        kappa_map = hp.alm2map(alm_kappa, nside=nside_target, lmax=lmax_target, pol=False, verbose=False)
    else:
        kappa_map = hp.read_map(str(kappa_path), verbose=False)

    mask_path = _find_first(files, ["mask.fits", "mask_2048.fits", "mask.fits.gz"])
    mask = hp.read_map(str(mask_path), verbose=False) if mask_path is not None else None

    nside = hp.get_nside(kappa_map)
    if mask is not None and hp.get_nside(mask) != nside:
        # Planck delivers a 2048 mask; ensure it matches the chosen kappa nside.
        mask = hp.ud_grade(mask, nside)
    if nside_out is not None and nside_out != nside:
        kappa_map = hp.ud_grade(kappa_map, nside_out)
        if mask is not None:
            mask = hp.ud_grade(mask, nside_out)
        nside = int(nside_out)

    # Persist downgraded products (small nsides only) for future runs.
    if nside_out is not None:
        cache_dir = paths.processed_dir / "planck_lensing"
        cache_dir.mkdir(parents=True, exist_ok=True)
        kappa_cache = cache_dir / f"planck2018_kappa_nside{int(nside)}.fits"
        mask_cache = cache_dir / f"planck2018_mask_nside{int(nside)}.fits"
        if not kappa_cache.exists():
            hp.write_map(str(kappa_cache), np.asarray(kappa_map, dtype=float), overwrite=True, dtype=np.float64)
        if mask is not None and not mask_cache.exists():
            hp.write_map(str(mask_cache), np.asarray(mask, dtype=float), overwrite=True, dtype=np.float64)

    return PlanckLensing(kappa_map=kappa_map, mask=mask, nside=nside, meta={"source": tar_path.name})
