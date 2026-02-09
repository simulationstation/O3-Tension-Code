from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .optical_bias.estimators import cross_cl_pseudo
from .optical_bias.maps import _require_healpy, radec_to_healpix


Frame = Literal["icrs", "galactic"]


@dataclass(frozen=True)
class VoidCatalog:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    z: np.ndarray
    Rv: np.ndarray | None = None
    weight: np.ndarray | None = None


def load_void_catalog_csv(
    path: str | Path,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    z_col: str = "z",
    Rv_col: str | None = None,
    weight_col: str | None = None,
    delimiter: str = ",",
    skiprows: int = 0,
) -> VoidCatalog:
    """Load a minimal void catalog from a CSV.

    This is deliberately generic: "void definition" is treated as an input choice; the prism test
    should be run across multiple catalogs/definitions as a systematics axis.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # Load header to locate columns.
    header = path.read_text().splitlines()[0]
    cols = [c.strip() for c in header.split(delimiter)]
    col_to_idx = {c: i for i, c in enumerate(cols)}
    for c in (ra_col, dec_col, z_col):
        if c not in col_to_idx:
            raise ValueError(f"Missing required column {c!r} in {path.name}.")

    usecols = [col_to_idx[ra_col], col_to_idx[dec_col], col_to_idx[z_col]]
    extra: list[tuple[str, int]] = []
    if Rv_col is not None:
        if Rv_col not in col_to_idx:
            raise ValueError(f"Missing Rv column {Rv_col!r} in {path.name}.")
        extra.append(("Rv", col_to_idx[Rv_col]))
        usecols.append(col_to_idx[Rv_col])
    if weight_col is not None:
        if weight_col not in col_to_idx:
            raise ValueError(f"Missing weight column {weight_col!r} in {path.name}.")
        extra.append(("weight", col_to_idx[weight_col]))
        usecols.append(col_to_idx[weight_col])

    data = np.loadtxt(path, delimiter=delimiter, skiprows=1 + int(skiprows), usecols=usecols, ndmin=2)
    ra = data[:, 0]
    dec = data[:, 1]
    z = data[:, 2]
    col = 3
    Rv = None
    w = None
    for name, _idx in extra:
        if name == "Rv":
            Rv = data[:, col]
        elif name == "weight":
            w = data[:, col]
        col += 1
    return VoidCatalog(ra_deg=ra, dec_deg=dec, z=z, Rv=Rv, weight=w)


def select_voids_by_z(cat: VoidCatalog, *, z_min: float, z_max: float) -> VoidCatalog:
    z = np.asarray(cat.z, dtype=float)
    m = (z >= float(z_min)) & (z <= float(z_max)) & np.isfinite(z)
    if not np.any(m):
        raise ValueError("No voids in selected z range.")
    return VoidCatalog(
        ra_deg=np.asarray(cat.ra_deg, dtype=float)[m],
        dec_deg=np.asarray(cat.dec_deg, dtype=float)[m],
        z=z[m],
        Rv=np.asarray(cat.Rv, dtype=float)[m] if cat.Rv is not None else None,
        weight=np.asarray(cat.weight, dtype=float)[m] if cat.weight is not None else None,
    )


def select_voids_by_Rv(cat: VoidCatalog, *, Rv_min: float, Rv_max: float) -> VoidCatalog:
    """Select voids by radius (if available)."""
    if cat.Rv is None:
        raise ValueError("VoidCatalog has no Rv column; cannot select by radius.")
    Rv = np.asarray(cat.Rv, dtype=float)
    m = (Rv >= float(Rv_min)) & (Rv <= float(Rv_max)) & np.isfinite(Rv)
    if not np.any(m):
        raise ValueError("No voids in selected Rv range.")
    return VoidCatalog(
        ra_deg=np.asarray(cat.ra_deg, dtype=float)[m],
        dec_deg=np.asarray(cat.dec_deg, dtype=float)[m],
        z=np.asarray(cat.z, dtype=float)[m],
        Rv=Rv[m],
        weight=np.asarray(cat.weight, dtype=float)[m] if cat.weight is not None else None,
    )


def void_overdensity_map(
    cat: VoidCatalog,
    *,
    nside: int,
    mask: np.ndarray | None = None,
    frame: Frame = "icrs",
    nest: bool = False,
) -> np.ndarray:
    """Build a simple HEALPix overdensity field from void centers.

    Output is a mean-subtracted, mask-aware map suitable for pseudo-C_ell cross spectra.
    """
    hp = _require_healpy()
    nside = int(nside)
    npix = int(hp.nside2npix(nside))
    pix = radec_to_healpix(cat.ra_deg, cat.dec_deg, nside=nside, frame=frame, nest=nest)

    w = np.ones_like(pix, dtype=float) if cat.weight is None else np.asarray(cat.weight, dtype=float)
    if w.shape != pix.shape:
        raise ValueError("Void weights must match void positions.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite void weights.")

    m = np.ones(npix, dtype=float) if mask is None else np.asarray(mask, dtype=float)
    if m.shape != (npix,):
        raise ValueError("Mask shape mismatch for nside.")
    good = m > 0

    counts = np.zeros(npix, dtype=float)
    np.add.at(counts, pix, w)
    mean = float(np.mean(counts[good])) if np.any(good) else float(np.mean(counts))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Non-positive void map mean; check mask/catalog.")
    delta = counts / mean - 1.0
    delta[~good] = 0.0
    return delta


def void_overdensity_map_from_pix(
    pix: np.ndarray,
    *,
    nside: int,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    """Build a simple HEALPix overdensity field from precomputed pixel indices.

    This is a performance helper for jackknife/suite runs: computing RA/Dec->HEALPix with astropy
    inside each resample can be expensive, so we precompute pixels once and reuse them.
    """
    hp = _require_healpy()
    nside = int(nside)
    npix = int(hp.nside2npix(nside))
    pix = np.asarray(pix, dtype=int)
    if pix.ndim != 1:
        raise ValueError("pix must be 1D.")
    if np.any((pix < 0) | (pix >= npix)):
        raise ValueError("pix contains out-of-range indices for nside.")

    w = np.ones_like(pix, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != pix.shape:
        raise ValueError("weights must match pix shape.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite weights.")

    m = np.ones(npix, dtype=float) if mask is None else np.asarray(mask, dtype=float)
    if m.shape != (npix,):
        raise ValueError("Mask shape mismatch for nside.")
    good = m > 0

    counts = np.zeros(npix, dtype=float)
    np.add.at(counts, pix, w)
    mean = float(np.mean(counts[good])) if np.any(good) else float(np.mean(counts))
    if not np.isfinite(mean) or mean <= 0:
        # This can happen in aggressive null tests (e.g. rotating voids under a small footprint mask)
        # when *all* rotated void centers land outside the analysis region. In such cases, returning
        # an identically-zero overdensity map is a reasonable "no signal" placeholder.
        if allow_empty:
            delta = np.zeros(npix, dtype=float)
            delta[~good] = 0.0
            return delta
        raise ValueError("Non-positive void map mean; check mask/pix.")
    delta = counts / mean - 1.0
    delta[~good] = 0.0
    return delta


def bin_cl(
    ell: np.ndarray,
    cl: np.ndarray,
    *,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ell = np.asarray(ell, dtype=int)
    cl = np.asarray(cl, dtype=float)
    edges = np.asarray(bin_edges, dtype=int)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing 1D array of length >= 2.")
    if ell.ndim != 1 or cl.ndim != 1 or ell.shape != cl.shape:
        raise ValueError("ell and cl must be 1D arrays with matching shapes.")

    ell_centers: list[float] = []
    cl_binned: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (ell >= lo) & (ell < hi)
        if not np.any(m):
            continue
        ell_centers.append(float(np.mean(ell[m])))
        cl_binned.append(float(np.mean(cl[m])))
    return np.asarray(ell_centers), np.asarray(cl_binned)


def measure_void_prism_spectra(
    *,
    kappa_map: np.ndarray,
    theta_map: np.ndarray,
    void_delta_map: np.ndarray,
    mask: np.ndarray,
    lmax: int | None = None,
    bin_edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the two cross spectra needed for a void prism estimator.

    This does *not* compute a covariance; it returns pseudo-C_ell cross spectra.
    """
    ell, cl_kv = cross_cl_pseudo(kappa_map, void_delta_map, mask, lmax=lmax)
    ell2, cl_tv = cross_cl_pseudo(theta_map, void_delta_map, mask, lmax=lmax)
    if ell2.shape != ell.shape:
        raise ValueError("Internal ell mismatch.")

    if bin_edges is not None:
        ell_b, cl_kv_b = bin_cl(ell, cl_kv, bin_edges=bin_edges)
        ell_b2, cl_tv_b = bin_cl(ell, cl_tv, bin_edges=bin_edges)
        if not np.allclose(ell_b, ell_b2):
            raise ValueError("Binned ell mismatch.")
        return {"ell": ell_b, "cl_kappa_void": cl_kv_b, "cl_theta_void": cl_tv_b}

    return {"ell": ell, "cl_kappa_void": cl_kv, "cl_theta_void": cl_tv}


def eg_void_from_spectra(
    *,
    cl_kappa_void: np.ndarray,
    cl_theta_void: np.ndarray,
    prefactor: float = 1.0,
    denom_floor: float = 1e-30,
) -> np.ndarray:
    """Form a simple void-prism E_G-like ratio from the two cross spectra.

    This is intentionally a *schematic* ratio:

        E_G^void(ell) = prefactor * C_{kappa,v}(ell) / C_{theta,v}(ell)

    The appropriate prefactor depends on the exact kSZ velocity proxy definition and the
    projection conventions used to build the theta map. For now we treat this as a user-supplied
    normalization constant.
    """
    num = np.asarray(cl_kappa_void, dtype=float)
    den = np.asarray(cl_theta_void, dtype=float)
    if num.shape != den.shape:
        raise ValueError("cl_kappa_void and cl_theta_void must have matching shapes.")
    if not np.isfinite(prefactor):
        raise ValueError("prefactor must be finite.")
    # Avoid zero sign -> zero denom: use a strict +/- floor with 0 treated as +.
    sgn = np.where(den >= 0.0, 1.0, -1.0)
    den2 = np.where(np.abs(den) < float(denom_floor), sgn * float(denom_floor), den)
    return float(prefactor) * (num / den2)


def jackknife_covariance(
    samples: np.ndarray,
) -> np.ndarray:
    """Return jackknife covariance for samples with shape (n_jk, n_dim)."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("samples must have shape (n_jk, n_dim) with n_jk>=2.")
    n = x.shape[0]
    mean = np.mean(x, axis=0, keepdims=True)
    d = x - mean
    # Standard jackknife covariance prefactor.
    return ((n - 1) / n) * (d.T @ d)


def jackknife_region_index(*, nside: int, nside_jk: int, nest: bool = False) -> np.ndarray:
    """Return region_id per pixel for leave-one-region-out jackknife.

    Each high-res pixel is assigned the index of the corresponding low-res (nside_jk) pixel.
    """
    hp = _require_healpy()
    nside = int(nside)
    nside_jk = int(nside_jk)
    if nside_jk <= 0 or nside <= 0:
        raise ValueError("nside and nside_jk must be positive.")
    if nside_jk > nside:
        raise ValueError("Require nside_jk <= nside.")
    npix = int(hp.nside2npix(nside))
    ipix = np.arange(npix, dtype=int)
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    return hp.ang2pix(nside_jk, theta, phi, nest=nest).astype(np.int32)
