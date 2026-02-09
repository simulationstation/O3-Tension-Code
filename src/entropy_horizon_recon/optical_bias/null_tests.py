from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .maps import _require_healpy, radec_to_healpix


def rotate_map_random(map_in: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    hp = _require_healpy()
    rng = np.random.default_rng(seed)
    # Approximate uniform random rotation: Haar measure ~ dphi dpsi sin(theta) dtheta.
    phi = rng.uniform(0.0, 360.0)
    psi = rng.uniform(0.0, 360.0)
    cos_theta = rng.uniform(-1.0, 1.0)
    theta = float(np.rad2deg(np.arccos(cos_theta)))
    rot = hp.Rotator(rot=[phi, theta, psi], deg=True)
    return rot.rotate_map_pixel(map_in)


def shuffle_positions(ra_deg: np.ndarray, dec_deg: np.ndarray, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ra_deg))
    return np.asarray(ra_deg)[idx], np.asarray(dec_deg)[idx]


def rotate_radec_random(ra_deg: np.ndarray, dec_deg: np.ndarray, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ICRS RA/Dec by a uniform random SO(3) rotation."""
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    # Shoemake (1992) uniform random unit quaternion.
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    # Rotation matrix from quaternion (x,y,z,w).
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )

    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    v = np.vstack([x, y, z])
    v2 = R @ v
    x2, y2, z2 = v2[0], v2[1], v2[2]
    dec2 = np.rad2deg(np.arcsin(np.clip(z2, -1.0, 1.0)))
    ra2 = np.rad2deg(np.arctan2(y2, x2)) % 360.0
    return ra2, dec2


def permutation_pvalue(null_stats: np.ndarray, obs_stat: float) -> float:
    null_stats = np.asarray(null_stats, dtype=float)
    if null_stats.size == 0:
        return np.nan
    return float((np.sum(null_stats >= obs_stat) + 1) / (null_stats.size + 1))


def permutation_pvalue_two_sided(null_stats: np.ndarray, obs_stat: float) -> float:
    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if null_stats.size == 0 or not np.isfinite(obs_stat):
        return np.nan
    return float((np.sum(np.abs(null_stats) >= abs(obs_stat)) + 1) / (null_stats.size + 1))


def parallel_nulls(func, seeds: list[int], n_jobs: int) -> np.ndarray:
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(func)(s) for s in seeds)
    return np.asarray(out, dtype=float)
