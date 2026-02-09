from __future__ import annotations

import numpy as np

from entropy_horizon_recon.dark_siren_h0 import (
    _build_lcdm_distance_cache,
    compute_gr_h0_posterior_grid_hierarchical_pe,
)
from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from entropy_horizon_recon.constants import PhysicalConstants


def test_gr_h0_hierarchical_pe_smoke_posterior_normalizes() -> None:
    rng = np.random.default_rng(0)
    constants = PhysicalConstants()

    omega_m0 = 0.31
    omega_k0 = 0.0
    z_max = 0.25
    H0_true = 70.0

    dist_cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=omega_m0, omega_k0=omega_k0)
    z_s = rng.uniform(0.01, 0.2, size=2000)
    dL_s = (constants.c_km_s / H0_true) * dist_cache.f(z_s)

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

    H0_grid = np.linspace(60.0, 80.0, 21)
    res = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event={"GWTEST": pe},
        H0_grid=H0_grid,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        z_max=z_max,
        cache_dir=None,
        injections=None,
        ifar_threshold_yr=1.0,
        det_model="snr_binned",
        snr_threshold=None,
        snr_binned_nbins=50,
        weight_mode="none",
        pop_z_mode="none",
        pop_z_powerlaw_k=0.0,
        pop_mass_mode="none",
        pop_m1_alpha=2.3,
        pop_m_min=5.0,
        pop_m_max=80.0,
        pop_q_beta=0.0,
    )

    p = np.asarray(res["posterior"], dtype=float)
    assert p.shape == H0_grid.shape
    assert np.all(np.isfinite(p))
    assert np.isclose(float(np.sum(p)), 1.0, rtol=0.0, atol=1e-12)
    assert float(res["summary"]["p16"]) >= float(H0_grid[0])
    assert float(res["summary"]["p84"]) <= float(H0_grid[-1])

