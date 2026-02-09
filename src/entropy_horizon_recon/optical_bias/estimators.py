from __future__ import annotations

import math
import numpy as np

from .maps import extract_at_positions, radec_to_healpix, _require_healpy


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.size != y.size or x.size != w.size:
        raise ValueError("x, y, w must have same length")
    sw = np.sum(w)
    if sw <= 0:
        raise ValueError("Sum of weights <= 0")
    xbar = np.sum(w * x) / sw
    ybar = np.sum(w * y) / sw
    sxx = np.sum(w * (x - xbar) ** 2)
    sxy = np.sum(w * (x - xbar) * (y - ybar))
    if sxx <= 0:
        raise ValueError("Degenerate x for regression")
    b = sxy / sxx
    a = ybar - b * xbar
    # Weighted residual variance (crude; adequate for diagnostics/smoke).
    resid = y - (a + b * x)
    s2 = np.sum(w * resid ** 2) / sw
    var_b = s2 / sxx
    z = float(b / np.sqrt(var_b)) if var_b > 0 else np.nan
    # Two-sided Normal approx p-value (avoid SciPy dependency).
    p = float(math.erfc(abs(z) / math.sqrt(2.0))) if np.isfinite(z) else np.nan
    return {
        "a": float(a),
        "b": float(b),
        "b_err": float(np.sqrt(var_b)) if var_b >= 0 else np.nan,
        "z": z,
        "p_two_sided_norm": p,
    }


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.size != y.size or x.size != w.size:
        raise ValueError("x, y, w must have same length")
    sw = float(np.sum(w))
    if sw <= 0:
        raise ValueError("Sum of weights <= 0")
    xbar = float(np.sum(w * x) / sw)
    ybar = float(np.sum(w * y) / sw)
    cov = float(np.sum(w * (x - xbar) * (y - ybar)) / sw)
    vx = float(np.sum(w * (x - xbar) ** 2) / sw)
    vy = float(np.sum(w * (y - ybar) ** 2) / sw)
    denom = math.sqrt(vx * vy)
    if denom <= 0:
        return np.nan
    return float(cov / denom)


def residual_map_from_samples(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    nside: int,
    frame: str = "icrs",
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin residuals into a HEALPix map (weighted mean) and a hit-count map."""
    hp = _require_healpy()
    pix = radec_to_healpix(ra_deg, dec_deg, nside=nside, frame=frame, nest=nest)
    npix = hp.nside2npix(nside)
    num = np.zeros(npix)
    den = np.zeros(npix)
    for p, r, w in zip(pix, residuals, weights, strict=False):
        if not np.isfinite(r) or not np.isfinite(w):
            continue
        num[p] += w * r
        den[p] += w
    m = np.full(npix, np.nan)
    good = den > 0
    m[good] = num[good] / den[good]
    return m, den


def cross_cl_pseudo(
    map_a: np.ndarray,
    map_b: np.ndarray,
    mask: np.ndarray,
    *,
    lmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple pseudo-C_ell cross-spectrum using healpy.anafast."""
    hp = _require_healpy()
    m = np.asarray(mask, dtype=float)
    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    # anafast does not accept NaNs; also ensure we only keep values on the masked sky.
    a = np.where((m > 0) & np.isfinite(a), a, 0.0)
    b = np.where((m > 0) & np.isfinite(b), b, 0.0)
    fsky = float(np.mean(m > 0))
    if fsky <= 0:
        raise ValueError("Mask has zero sky fraction")
    cl = hp.anafast(a, b, lmax=lmax)
    ell = np.arange(cl.size)
    return ell, cl / fsky


def evaluate_kappa_at_sn(
    kappa_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    nside: int,
    frame: str = "icrs",
    nest: bool = False,
) -> np.ndarray:
    return extract_at_positions(kappa_map, ra_deg, dec_deg, nside=nside, frame=frame, nest=nest)
