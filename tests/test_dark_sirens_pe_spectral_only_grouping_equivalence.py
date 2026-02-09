from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

pytest.importorskip("healpy")

from entropy_horizon_recon.dark_sirens_pe import (  # noqa: E402
    PePixelDistanceHistogram,
    compute_dark_siren_logL_draws_from_pe_hist,
)
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em  # noqa: E402


def test_spectral_only_matches_explicit_per_galaxy_sum() -> None:
    # Small toy posterior with mu != 1 so that dL_GW != dL_EM.
    x_grid = np.array([-2.0, 0.0], dtype=float)
    logmu = np.tile(np.array([np.log(1.2), 0.0], dtype=float), (3, 1))  # 3 draws
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
        pdf_bins=np.array([[0.2, 0.5, 0.3]], dtype=float),
    )

    # A few galaxies all in the selected pixel, with repeated z values.
    z = np.array([0.05, 0.05, 0.10, 0.10, 0.10], dtype=float)
    w = np.array([1.0, 2.0, 1.0, 1.0, 3.0], dtype=float)
    ipix = np.zeros_like(z, dtype=np.int64)

    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
        event="TEST",
        pe=pe,
        post=post,
        z_gal=z,
        w_gal=w,
        ipix_gal=ipix,
        convention="A",
        distance_mode="spectral_only",
        gal_chunk_size=10,
    )

    # Reference implementation: explicit per-galaxy sum for spectral_only.
    prior = GWDistancePrior()
    edges = np.asarray(pe.dL_edges, dtype=float)
    nb = int(edges.size - 1)
    widths = np.diff(edges)
    pdf_1d = np.asarray(pe.pdf_bins[0], dtype=float)
    pdf_1d = np.clip(pdf_1d, 0.0, np.inf)
    pdf_1d = pdf_1d / float(np.sum(pdf_1d * widths))

    z_u = np.unique(z)
    dL_em_u = predict_dL_em(post, z_eval=z_u)
    _, R_u = predict_r_gw_em(post, z_eval=z_u, convention="A", allow_extrapolation=False)
    inv = np.searchsorted(z_u, z, side="left")
    dL_em = dL_em_u[:, inv]
    R = R_u[:, inv]
    dL_gw = dL_em * R

    def _ref(dL: np.ndarray) -> np.ndarray:
        bin_idx = np.searchsorted(edges, dL, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
        pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
        pdf = np.where(valid, pdf, 0.0)
        logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
        logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
        logw = np.log(np.clip(w, 1e-30, np.inf))[None, :]
        logprob = 0.0  # prob_pix is 1 everywhere in this toy setup.
        logterm = logw + logprob + logpdf - logprior
        logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
        return logsumexp(logterm, axis=1)

    ref_mu = _ref(dL_gw)
    ref_gr = _ref(dL_em)

    assert np.all(np.isfinite(logL_mu))
    assert np.all(np.isfinite(logL_gr))
    assert np.allclose(logL_mu, ref_mu, rtol=0.0, atol=1e-12)
    assert np.allclose(logL_gr, ref_gr, rtol=0.0, atol=1e-12)
