from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("healpy")

from entropy_horizon_recon.dark_sirens_pe import (  # noqa: E402
    PePixelDistanceHistogram,
    compute_dark_siren_logL_draws_from_pe_hist,
)
from entropy_horizon_recon.sirens import MuForwardPosterior  # noqa: E402


def test_pe_prior_only_distance_null_mu_equals_gr() -> None:
    # Construct a toy posterior with mu != 1 (so dL_GW != dL_EM) but verify that
    # distance_mode=prior_only kills distance information and yields logL_mu == logL_gr.
    x_grid = np.array([-2.0, 0.0], dtype=float)
    logmu = np.tile(np.array([np.log(1.5), 0.0], dtype=float), (3, 1))  # 3 draws
    z_grid = np.array([0.0, 0.2], dtype=float)
    H = np.tile(np.array([70.0, 190.0], dtype=float), (3, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((3,), 70.0),
        omega_m0=np.full((3,), 0.3),
        omega_k0=np.zeros((3,)),
        sigma8_0=None,
    )

    pe = PePixelDistanceHistogram(
        nside=1,
        nest=True,
        p_credible=0.9,
        pix_sel=np.array([0], dtype=np.int64),
        prob_pix=np.array([1.0], dtype=float),
        dL_edges=np.array([1.0, 100.0, 1_000.0, 10_000.0], dtype=float),
        pdf_bins=np.full((1, 3), 1e-6, dtype=float),
    )

    # A few galaxies all in the selected pixel.
    z = np.array([0.05, 0.10, 0.15], dtype=float)
    w = np.ones_like(z)
    ipix = np.zeros_like(z, dtype=np.int64)

    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
        event="TEST",
        pe=pe,
        post=post,
        z_gal=z,
        w_gal=w,
        ipix_gal=ipix,
        convention="A",
        distance_mode="prior_only",
        gal_chunk_size=10,
    )

    assert np.all(np.isfinite(logL_mu))
    assert np.all(np.isfinite(logL_gr))
    assert np.allclose(logL_mu, logL_gr, rtol=0.0, atol=1e-12)

