from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Frame = Literal["icrs", "galactic"]


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise ImportError("healpy is required for HEALPix map operations.") from exc
    return hp


def _require_astropy():
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception as exc:  # pragma: no cover - runtime guard
        raise ImportError("astropy is required for frame transformations.") from exc
    return SkyCoord, u


def _to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    SkyCoord, u = _require_astropy()
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g = c.galactic
    return np.asarray(g.l.deg), np.asarray(g.b.deg)


def radec_to_healpix(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    nside: int,
    frame: Frame = "icrs",
    nest: bool = False,
) -> np.ndarray:
    """Convert RA/Dec to HEALPix pixel indices.

    Parameters
    ----------
    frame: "icrs" or "galactic". For "galactic", input is still RA/Dec and is
    internally converted to l/b.
    """
    hp = _require_healpy()
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if frame == "galactic":
        lon, lat = _to_galactic(ra, dec)
    else:
        lon, lat = ra, dec
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    return hp.ang2pix(nside, theta, phi, nest=nest)


def healpix_to_radec(
    pix: np.ndarray,
    *,
    nside: int,
    frame: Frame = "icrs",
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    hp = _require_healpy()
    pix = np.asarray(pix, dtype=int)
    theta, phi = hp.pix2ang(nside, pix, nest=nest)
    lat = 90.0 - np.rad2deg(theta)
    lon = np.rad2deg(phi)
    if frame == "galactic":
        # Convert l/b to RA/Dec
        SkyCoord, u = _require_astropy()
        c = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic")
        icrs = c.icrs
        return np.asarray(icrs.ra.deg), np.asarray(icrs.dec.deg)
    return lon, lat


@dataclass(frozen=True)
class HealpixMap:
    data: np.ndarray
    nside: int
    nest: bool = False


def downgrade_map(map_in: np.ndarray, nside_out: int, *, power: float = 0.0, nest: bool = False) -> np.ndarray:
    hp = _require_healpy()
    order = "NESTED" if nest else "RING"
    return hp.ud_grade(map_in, nside_out, order_in=order, order_out=order, power=power)


def smooth_map(map_in: np.ndarray, *, fwhm_deg: float) -> np.ndarray:
    hp = _require_healpy()
    if fwhm_deg <= 0:
        return map_in
    return hp.smoothing(map_in, fwhm=np.deg2rad(fwhm_deg), verbose=False)


def apply_mask(map_in: np.ndarray, mask: np.ndarray, *, fill_value: float = np.nan) -> np.ndarray:
    out = np.array(map_in, copy=True)
    m = np.asarray(mask, dtype=float)
    bad = m <= 0
    out[bad] = fill_value
    return out


def extract_at_positions(
    map_in: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    nside: int,
    frame: Frame = "icrs",
    nest: bool = False,
) -> np.ndarray:
    hp = _require_healpy()
    pix = radec_to_healpix(ra_deg, dec_deg, nside=nside, frame=frame, nest=nest)
    return map_in[pix]
