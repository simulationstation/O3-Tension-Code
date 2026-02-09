from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import DataPaths


# -----------------------------------------------------------------------------
# BOSS DR12 void catalog (Mao+ 2017 ApJ 835 161) via CDS/VizieR.
#
# We use the HTTP endpoint (not HTTPS) because some environments reject the CDS
# SSL chain. Integrity is enforced via pinned SHA256.
# -----------------------------------------------------------------------------

BOSS_VOID_BASE_URL = "http://cdsarc.u-strasbg.fr/ftp/cats/J_ApJ/835/161/"
BOSS_VOID_README_FNAME = "ReadMe"
BOSS_VOID_TABLE_FNAME = "table1.dat"

# Pinned from CDS files as of 2026-01-30.
BOSS_VOID_README_SHA256 = "298a0fb70094d0e99aab06718d88ae66778a1c08ec887aac817a3833d5847896"
BOSS_VOID_TABLE_SHA256 = "d579cfc1dc0bcb8886dab0645221c3788e8b784ba7bea1d2be5f096545971883"


@dataclass(frozen=True)
class BossDr12VoidRow:
    sample: str
    void_id: int
    ra_deg: float
    dec_deg: float
    z: float
    ngal: int
    volume_mpc3_h3: float
    reff_mpc_h: float
    nmin_h3_mpc3: float
    delmin: float
    r_ratio: float
    prob: float
    dbound_mpc_h: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_sha(*, url: str, dest: Path, sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        got = _sha256(dest)
        if got != sha256:
            raise ValueError(f"SHA256 mismatch for cached file {dest} (got {got}, expected {sha256}).")
        return dest

    # Avoid adding requests as a hard dependency; use urllib.
    from urllib.request import urlopen  # noqa: S310 (controlled URL + pinned hash)

    with urlopen(url) as r, dest.open("wb") as w:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            w.write(b)

    got = _sha256(dest)
    if got != sha256:
        raise ValueError(f"SHA256 mismatch for downloaded file {dest} (got {got}, expected {sha256}).")
    return dest


def fetch_boss_dr12_void_catalog(*, paths: DataPaths) -> dict[str, Path]:
    """Download/pin the BOSS DR12 void catalog (quality-cut version)."""
    base = Path(paths.pooch_cache_dir) / "void_prism" / "boss_dr12_voids_mao2017"
    readme = _download_with_sha(
        url=BOSS_VOID_BASE_URL + BOSS_VOID_README_FNAME,
        dest=base / BOSS_VOID_README_FNAME,
        sha256=BOSS_VOID_README_SHA256,
    )
    table = _download_with_sha(
        url=BOSS_VOID_BASE_URL + BOSS_VOID_TABLE_FNAME,
        dest=base / BOSS_VOID_TABLE_FNAME,
        sha256=BOSS_VOID_TABLE_SHA256,
    )
    return {"readme": readme, "table": table}


def load_boss_dr12_void_rows(*, paths: DataPaths) -> list[BossDr12VoidRow]:
    """Parse table1.dat into structured rows.

    Parsing uses the byte-by-byte description in the CDS ReadMe.
    """
    files = fetch_boss_dr12_void_catalog(paths=paths)
    lines = files["table"].read_text().splitlines()
    out: list[BossDr12VoidRow] = []
    for ln in lines:
        if not ln.strip():
            continue
        # Byte ranges (1-indexed in ReadMe) mapped to 0-index slices.
        sample = ln[0:11].strip()
        void_id = int(ln[12:17])
        ra_deg = float(ln[18:25])
        dec_deg = float(ln[26:32])
        z = float(ln[33:38])
        ngal = int(ln[39:45])
        V = float(ln[46:55])
        reff = float(ln[56:63])
        nmin = float(ln[64:73])
        delmin = float(ln[74:80])
        r_ratio = float(ln[81:86])
        prob = float(ln[87:96])
        dbound = float(ln[97:104])
        out.append(
            BossDr12VoidRow(
                sample=sample,
                void_id=void_id,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                z=z,
                ngal=ngal,
                volume_mpc3_h3=V,
                reff_mpc_h=reff,
                nmin_h3_mpc3=nmin,
                delmin=delmin,
                r_ratio=r_ratio,
                prob=prob,
                dbound_mpc_h=dbound,
            )
        )
    return out


def load_boss_dr12_void_catalog_arrays(
    *,
    paths: DataPaths,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, np.ndarray]:
    """Return arrays for the BOSS DR12 void catalog (with optional z cuts)."""
    rows = load_boss_dr12_void_rows(paths=paths)
    ra = np.array([r.ra_deg for r in rows], dtype=float)
    dec = np.array([r.dec_deg for r in rows], dtype=float)
    z = np.array([r.z for r in rows], dtype=float)
    reff = np.array([r.reff_mpc_h for r in rows], dtype=float)
    w = np.array([r.ngal for r in rows], dtype=float)

    m = np.ones_like(z, dtype=bool)
    if z_min is not None:
        m &= z >= float(z_min)
    if z_max is not None:
        m &= z <= float(z_max)

    return {"ra_deg": ra[m], "dec_deg": dec[m], "z": z[m], "Rv_mpc_h": reff[m], "weight": w[m]}


# -----------------------------------------------------------------------------
# Planck 2018 SMICA-noSZ CMB temperature map (R3.00).
# -----------------------------------------------------------------------------

PLANCK_CMB_FNAME = "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
PLANCK_CMB_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/"
    "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
)

# NOTE: Fill this in after first successful retrieval in a given environment (large file).
# Pinned after first successful retrieval on 2026-01-30.
PLANCK_CMB_SHA256 = "8dca51299f2b0e53187810db6edead9c369757b519a66fe7c41288fa45b4cd98"


def fetch_planck_smica_nosz_fits(*, paths: DataPaths, allow_unverified: bool = False) -> Path:
    dest = Path(paths.pooch_cache_dir) / "planck_cmb" / PLANCK_CMB_FNAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if allow_unverified or "TODO" in PLANCK_CMB_SHA256:
            return dest
        got = _sha256(dest)
        if got != PLANCK_CMB_SHA256:
            raise ValueError("Planck CMB file SHA256 mismatch.")
        return dest

    if "TODO" in PLANCK_CMB_SHA256 and not allow_unverified:
        raise RuntimeError(
            "PLANCK_CMB_SHA256 is not pinned yet. Download once with allow_unverified=True, "
            "compute sha256, and then pin PLANCK_CMB_SHA256."
        )

    # Download with urllib (no extra deps). For first bootstrap we allow unverified.
    from urllib.request import urlopen  # noqa: S310 (public URL)

    with urlopen(PLANCK_CMB_URL) as r, dest.open("wb") as w:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            w.write(b)

    if not allow_unverified and "TODO" not in PLANCK_CMB_SHA256:
        got = _sha256(dest)
        if got != PLANCK_CMB_SHA256:
            raise ValueError("Planck CMB file SHA256 mismatch.")
    return dest
