from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("healpy")

from entropy_horizon_recon.dark_sirens_incompleteness import (  # noqa: E402
    MissingHostPriorPrecompute,
    compute_missing_host_logL_draws_from_histogram,
)
from entropy_horizon_recon.dark_sirens_pe import (  # noqa: E402
    PePixelDistanceHistogram,
    compute_dark_siren_logL_draws_from_pe_hist,
)
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.sirens import MuForwardPosterior  # noqa: E402


def test_pe_sky_only_null_mu_equals_gr_even_if_distances_outside_hist() -> None:
    # Construct a toy posterior with strong mu != 1 and a tiny dL histogram support. The sky_only
    # mode must ignore distance entirely and produce logL_mu == logL_gr by construction.
    x_grid = np.array([-4.0, 0.0], dtype=float)
    logmu = np.tile(np.array([np.log(20.0), 0.0], dtype=float), (2, 1))  # 2 draws
    z_grid = np.array([0.0, 1.0], dtype=float)
    H = np.tile(np.array([70.0, 500.0], dtype=float), (2, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((2,), 70.0),
        omega_m0=np.full((2,), 0.3),
        omega_k0=np.zeros((2,)),
        sigma8_0=None,
    )

    pe = PePixelDistanceHistogram(
        nside=1,
        nest=True,
        p_credible=0.9,
        pix_sel=np.array([0], dtype=np.int64),
        prob_pix=np.array([1.0], dtype=float),
        dL_edges=np.array([10.0, 20.0, 30.0], dtype=float),  # intentionally tiny support
        pdf_bins=np.full((1, 2), 1e-6, dtype=float),
    )

    z = np.array([0.5, 0.8, 0.9], dtype=float)
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
        distance_mode="sky_only",
        gal_chunk_size=10,
    )

    assert np.all(np.isfinite(logL_mu))
    assert np.all(np.isfinite(logL_gr))
    assert np.allclose(logL_mu, logL_gr, rtol=0.0, atol=0.0)


def test_missing_host_sky_only_null_mu_equals_gr() -> None:
    pre = MissingHostPriorPrecompute(
        z_grid=np.array([0.0, 0.2], dtype=float),
        dL_em=np.ones((3, 2), dtype=float),
        dL_gw=np.ones((3, 2), dtype=float),
        base_z=np.ones((3, 2), dtype=float),
        ddLdz_em=np.ones((3, 2), dtype=float),
        ddLdz_gw=np.ones((3, 2), dtype=float),
    )

    prob_pix = np.array([0.25, 0.75], dtype=float)
    pdf_bins = np.full((2, 2), 1e-6, dtype=float)
    dL_edges = np.array([1.0, 10.0, 100.0], dtype=float)

    logL_mu, logL_gr = compute_missing_host_logL_draws_from_histogram(
        prob_pix=prob_pix,
        pdf_bins=pdf_bins,
        dL_edges=dL_edges,
        pre=pre,
        gw_distance_prior=GWDistancePrior(),
        distance_mode="sky_only",
        pixel_chunk_size=10,
    )

    assert logL_mu.shape == (3,)
    assert logL_gr.shape == (3,)
    assert np.all(np.isfinite(logL_mu))
    assert np.all(np.isfinite(logL_gr))
    assert np.allclose(logL_mu, logL_gr, rtol=0.0, atol=0.0)
