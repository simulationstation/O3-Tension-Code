from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import DataPaths
from .optical_bias.maps import _require_healpy
from .void_prism_inputs import fetch_planck_smica_nosz_fits


@dataclass(frozen=True)
class PlanckSmicaNoSz:
    T_map: np.ndarray
    nside: int
    meta: dict[str, str]


def load_planck_smica_nosz_T(
    *,
    paths: DataPaths,
    nside_out: int | None = None,
    field: int = 0,
    allow_unverified: bool = False,
) -> PlanckSmicaNoSz:
    """Load Planck 2018 SMICA-noSZ temperature map.

    This is a large public product (~400MB). It is used here as a *kSZ-friendly temperature map*
    for building a velocity/temperature proxy. For a true kSZ velocity reconstruction, additional
    processing and external LSS data are required.
    """
    hp = _require_healpy()
    fits_path = fetch_planck_smica_nosz_fits(paths=paths, allow_unverified=allow_unverified)
    T = hp.read_map(str(fits_path), field=field, verbose=False)
    nside = int(hp.get_nside(T))
    if nside_out is not None and int(nside_out) != nside:
        T = hp.ud_grade(T, int(nside_out))
        nside = int(nside_out)
    return PlanckSmicaNoSz(T_map=np.asarray(T, dtype=float), nside=nside, meta={"source": Path(fits_path).name})


def bandpass_filter_map(
    map_in: np.ndarray,
    *,
    nside: int,
    lmin: int = 300,
    lmax: int = 1500,
    remove_monopole_dipole: bool = True,
    nest: bool = False,
) -> np.ndarray:
    """Apply a hard bandpass filter in harmonic space.

    This is a simple, reproducible filter intended for building a *kSZ temperature proxy* map.
    """
    hp = _require_healpy()
    nside = int(nside)
    lmax_hard = int(3 * nside - 1)
    lmin = int(lmin)
    lmax = int(lmax)
    if lmin < 0 or lmax <= lmin:
        raise ValueError("Require 0 <= lmin < lmax.")
    lmax = min(lmax, lmax_hard)

    m = np.asarray(map_in, dtype=float)
    if remove_monopole_dipole:
        # healpy removes only monopole/dipole by default. This helps reduce large-scale leakage.
        m = hp.remove_dipole(m, fitval=False, verbose=False)

    alm = hp.map2alm(m, lmax=lmax_hard, iter=0, pol=False, use_weights=False)
    ell = np.arange(lmax_hard + 1, dtype=float)
    fl = np.zeros_like(ell)
    fl[(ell >= lmin) & (ell <= lmax)] = 1.0
    alm_f = hp.almxfl(alm, fl)
    out = hp.alm2map(alm_f, nside=nside, lmax=lmax_hard, pol=False, verbose=False)
    return np.asarray(out, dtype=float)

