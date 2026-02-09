from __future__ import annotations

import numpy as np

from entropy_horizon_recon.dark_sirens_selection import O3InjectionSet
from entropy_horizon_recon.dark_sirens_selection import compute_selection_alpha_from_injections
from entropy_horizon_recon.mu_injection_recovery import (
    build_linear_mu_forward_posterior,
    generate_synthetic_detected_events_from_injections_for_post,
    score_mu_vs_gr_hierarchical_pe,
    synthesize_hierarchical_pe_by_event,
)
from entropy_horizon_recon.siren_injection_recovery import InjectionRecoveryConfig


def _toy_injections(seed: int = 0) -> O3InjectionSet:
    rng = np.random.default_rng(int(seed))
    n = 200
    z = rng.uniform(0.02, 0.6, size=n).astype(float)
    dL_fid = rng.uniform(500.0, 3000.0, size=n).astype(float)
    snr = np.linspace(6.0, 20.0, n).astype(float)
    found = snr > 12.0
    return O3InjectionSet(
        path="<toy>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL_fid,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=np.full_like(z, 30.0),
        m2_source=np.full_like(z, 20.0),
        total_generated=int(n),
        analysis_time_s=1.0,
    )


def test_mu_vs_gr_injection_smoke_sign() -> None:
    injections = _toy_injections(seed=1)
    cfg = InjectionRecoveryConfig(
        h0_true=67.7,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        det_model="threshold",
        weight_mode="none",
        pop_z_mode="comoving_powerlaw",
        pop_z_k=-8.0,
        pop_mass_mode="none",
        pe_obs_mode="truth",
        pe_n_samples=1_000,
        pe_synth_mode="likelihood_resample",
        pe_prior_resample_n_candidates=20_000,
        pe_seed=0,
    )

    # Score model: strong deviation to make the sign robust.
    post_score = build_linear_mu_forward_posterior(
        h0=67.7,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        mu_slope_x=-4.0,
        n_draws=1,
    )
    selection_alpha = compute_selection_alpha_from_injections(
        post=post_score,
        injections=injections,
        convention="A",
        z_max=0.62,
        det_model="threshold",
        weight_mode="none",
        mu_det_distance="gw",
        pop_z_mode="comoving_powerlaw",
        pop_z_powerlaw_k=-8.0,
        pop_mass_mode="none",
    )

    # Truth: mu
    post_truth_mu = build_linear_mu_forward_posterior(
        h0=67.7,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        mu_slope_x=-4.0,
        n_draws=1,
    )
    truths_mu = generate_synthetic_detected_events_from_injections_for_post(
        injections=injections,
        cfg=cfg,
        post_truth=post_truth_mu,
        convention="A",
        mu_det_distance="gw",
        n_events=6,
        seed=123,
    )
    pe_mu = synthesize_hierarchical_pe_by_event(truths=truths_mu, cfg=cfg, seed=123)
    score_mu = score_mu_vs_gr_hierarchical_pe(
        pe_by_event=pe_mu,
        post_score=post_score,
        selection_alpha=selection_alpha,
        cfg=cfg,
        convention="A",
        z_max=0.62,
    )
    assert score_mu.delta_lpd_total > 0.0

    # Truth: GR
    post_truth_gr = build_linear_mu_forward_posterior(
        h0=67.7,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        mu_slope_x=0.0,
        n_draws=1,
    )
    truths_gr = generate_synthetic_detected_events_from_injections_for_post(
        injections=injections,
        cfg=cfg,
        post_truth=post_truth_gr,
        convention="A",
        mu_det_distance="gw",
        n_events=6,
        seed=456,
    )
    pe_gr = synthesize_hierarchical_pe_by_event(truths=truths_gr, cfg=cfg, seed=456)
    score_gr = score_mu_vs_gr_hierarchical_pe(
        pe_by_event=pe_gr,
        post_score=post_score,
        selection_alpha=selection_alpha,
        cfg=cfg,
        convention="A",
        z_max=0.62,
    )
    assert score_gr.delta_lpd_total < 0.0
