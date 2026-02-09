from __future__ import annotations

import numpy as np
import pytest

# These tests are optional: they require healpy + ligo.skymap.
pytest.importorskip("healpy")
pytest.importorskip("ligo.skymap")

from entropy_horizon_recon.dark_sirens import SkyMap3D, credible_region_pixels, score_dark_siren_event  # noqa: E402
from entropy_horizon_recon.sirens import MuForwardPosterior  # noqa: E402


def test_dark_siren_score_smoke_mu_equals_gr() -> None:
    # Minimal posterior: mu=1 so dL_GW == dL_EM and Delta_LPD should be ~0.
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((4, x_grid.size), dtype=float)  # 4 draws, mu=1
    z_grid = np.array([0.0, 0.1, 0.2], dtype=float)
    H = np.tile(np.array([70.0, 75.0, 80.0], dtype=float), (4, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((4,), 70.0),
        omega_m0=np.full((4,), 0.3),
        omega_k0=np.zeros((4,)),
        sigma8_0=None,
    )

    nside = 4
    npix = 12 * nside * nside
    prob = np.full((npix,), 1.0 / float(npix))
    sky = SkyMap3D(
        path="<mem>",
        nside=nside,
        nest=True,
        prob=prob,
        distmu=np.full((npix,), 200.0),
        distsigma=np.full((npix,), 50.0),
        distnorm=np.full((npix,), 1.0),
    )

    ra = np.array([10.0, 120.0, 250.0], dtype=float)
    dec = np.array([-10.0, 20.0, 45.0], dtype=float)
    z = np.array([0.01, 0.05, 0.10], dtype=float)
    w = np.ones_like(z)

    sc = score_dark_siren_event(
        event="TEST",
        sky=sky,
        sky_area_deg2=1.0,
        post=post,
        ra_deg=ra,
        dec_deg=dec,
        z=z,
        w=w,
        convention="A",
        max_draws=4,
    )
    assert np.isfinite(sc.delta_lpd)
    assert abs(sc.delta_lpd) < 1e-10


def test_credible_region_pixels_uniform() -> None:
    nside = 8
    npix = 12 * nside * nside
    prob = np.full((npix,), 1.0 / float(npix))
    sky = SkyMap3D(
        path="<mem>",
        nside=nside,
        nest=True,
        prob=prob,
        distmu=np.full((npix,), 200.0),
        distsigma=np.full((npix,), 50.0),
        distnorm=np.full((npix,), 1.0),
    )

    sel, area = credible_region_pixels(sky, nside_out=4, p_credible=0.5)
    # Uniform map: should take about half the coarse pixels.
    npix_out = 12 * 4 * 4
    assert sel.size in {npix_out // 2, npix_out // 2 + 1}
    assert area > 0.0


def test_credible_region_pixels_uniform_upsample() -> None:
    # Ensure low-nside skymaps can be expanded to a higher nside_out deterministically.
    nside_in = 4
    npix = 12 * nside_in * nside_in
    prob = np.full((npix,), 1.0 / float(npix))
    sky = SkyMap3D(
        path="<mem>",
        nside=nside_in,
        nest=True,
        prob=prob,
        distmu=np.full((npix,), 200.0),
        distsigma=np.full((npix,), 50.0),
        distnorm=np.full((npix,), 1.0),
    )

    nside_out = 8
    sel, area = credible_region_pixels(sky, nside_out=nside_out, p_credible=0.5)
    npix_out = 12 * nside_out * nside_out
    assert sel.size in {npix_out // 2, npix_out // 2 + 1}
    assert area > 0.0
