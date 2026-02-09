from __future__ import annotations

import numpy as np

from .maps import radec_to_healpix, _require_healpy


def build_shear_maps(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    weight: np.ndarray,
    zbin: np.ndarray,
    *,
    nside: int,
    n_zbin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build gamma1/gamma2 maps per z-bin.

    Returns gamma1[zbin, pix], gamma2[zbin, pix], wmap[zbin, pix].
    """
    hp = _require_healpy()
    npix = hp.nside2npix(nside)
    g1 = np.zeros((n_zbin, npix))
    g2 = np.zeros((n_zbin, npix))
    wmap = np.zeros((n_zbin, npix))

    pix = radec_to_healpix(ra_deg, dec_deg, nside=nside)
    for p, ee1, ee2, w, zb in zip(pix, e1, e2, weight, zbin, strict=False):
        if zb < 0 or zb >= n_zbin:
            continue
        if not np.isfinite(ee1) or not np.isfinite(ee2) or w <= 0:
            continue
        g1[zb, p] += w * ee1
        g2[zb, p] += w * ee2
        wmap[zb, p] += w

    for zb in range(n_zbin):
        good = wmap[zb] > 0
        g1[zb, good] /= wmap[zb, good]
        g2[zb, good] /= wmap[zb, good]
    return g1, g2, wmap


def gamma2_map(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return g1**2 + g2**2


def debias_shape_noise_random(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    weight: np.ndarray,
    zbin: np.ndarray,
    *,
    nside: int,
    n_zbin: int,
    n_rot: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """Estimate noise bias using random rotations."""
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_rot):
        phi = rng.uniform(0, 2 * np.pi, size=len(e1))
        c = np.cos(2 * phi)
        s = np.sin(2 * phi)
        e1r = e1 * c - e2 * s
        e2r = e1 * s + e2 * c
        g1, g2, _ = build_shear_maps(ra_deg, dec_deg, e1r, e2r, weight, zbin, nside=nside, n_zbin=n_zbin)
        g2map = gamma2_map(g1, g2)
        acc = g2map if acc is None else acc + g2map
    return acc / float(n_rot)


def zbin_weights_from_sn(z_sn: np.ndarray, zbin_edges: np.ndarray) -> np.ndarray:
    z_sn = np.asarray(z_sn, dtype=float)
    zb = np.digitize(z_sn, zbin_edges) - 1
    n_zbin = len(zbin_edges) - 1
    w = np.zeros(n_zbin)
    for i in range(n_zbin):
        w[i] = np.sum(zb == i)
    if np.sum(w) > 0:
        w /= np.sum(w)
    return w


def effective_gamma2_at_sn(
    sn_pix: np.ndarray,
    gamma2_by_z: np.ndarray,
    zbin_weights: np.ndarray,
) -> np.ndarray:
    # Weighted combination of z-bin gamma^2 maps at SN pixels
    g2 = np.zeros(sn_pix.size)
    for zb, w in enumerate(zbin_weights):
        if w <= 0:
            continue
        g2 += w * gamma2_by_z[zb, sn_pix]
    return g2
