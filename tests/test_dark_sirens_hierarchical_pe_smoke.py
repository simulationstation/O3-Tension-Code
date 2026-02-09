from __future__ import annotations

import numpy as np

from entropy_horizon_recon.dark_sirens_hierarchical_pe import (
    GWTCPeHierarchicalSamples,
    compute_hierarchical_pe_logL_draws,
)
from entropy_horizon_recon.sirens import MuForwardPosterior, predict_dL_em


def test_hierarchical_pe_smoke_mu_equals_gr() -> None:
    # If mu=1 then dL_GW == dL_EM and the hierarchical log-likelihoods should match.
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((2, x_grid.size), dtype=float)  # 2 draws, mu=1
    z_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    H = np.tile(np.array([70.0, 72.0, 75.0, 78.0], dtype=float), (2, 1))
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

    rng = np.random.default_rng(123)
    z_s = rng.uniform(0.051, 0.149, size=2000)
    dL_s = predict_dL_em(post, z_eval=z_s)[0]

    pe = GWTCPeHierarchicalSamples(
        file="<mem>",
        analysis="TEST",
        n_total=int(dL_s.size),
        n_used=int(dL_s.size),
        dL_mpc=np.asarray(dL_s, dtype=float),
        chirp_mass_det=np.full((dL_s.size,), 30.0, dtype=float),
        mass_ratio=np.full((dL_s.size,), 0.8, dtype=float),
        log_pi_dL=np.zeros((dL_s.size,), dtype=float),
        log_pi_chirp_mass=np.zeros((dL_s.size,), dtype=float),
        log_pi_mass_ratio=np.zeros((dL_s.size,), dtype=float),
        prior_spec={},
    )

    logL_mu, logL_gr = compute_hierarchical_pe_logL_draws(
        pe=pe,
        post=post,
        z_max=0.15,
        pop_z_mode="none",
        pop_mass_mode="none",
    )

    assert np.all(np.isfinite(logL_mu))
    assert np.all(np.isfinite(logL_gr))
    assert np.allclose(logL_mu, logL_gr, rtol=0.0, atol=1e-12)

