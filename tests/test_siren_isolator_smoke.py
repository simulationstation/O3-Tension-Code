from __future__ import annotations

import numpy as np
import pytest

from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from entropy_horizon_recon.siren_isolator import apply_hierarchical_pe_scramble


def _toy_pe(n: int = 2000) -> GWTCPeHierarchicalSamples:
    dL = np.linspace(100.0, 2000.0, n, dtype=float)
    mc = np.linspace(10.0, 50.0, n, dtype=float)
    # Deterministic (mc,q) pairing so we can check whether a scramble preserves the joint mass pairing.
    q = 0.1 * mc + 0.01

    # Define toy "priors" that are explicit functions of each coordinate.
    log_pi_dL = -dL
    log_pi_mc = -mc
    log_pi_q = -q

    return GWTCPeHierarchicalSamples(
        file="<mem>",
        analysis="TEST",
        n_total=int(n),
        n_used=int(n),
        dL_mpc=dL,
        chirp_mass_det=mc,
        mass_ratio=q,
        log_pi_dL=log_pi_dL,
        log_pi_chirp_mass=log_pi_mc,
        log_pi_mass_ratio=log_pi_q,
        prior_spec={},
    )


def test_scramble_shuffle_dL_preserves_dL_prior_pairing_and_keeps_mass_pairing() -> None:
    pe0 = _toy_pe()
    pe = apply_hierarchical_pe_scramble(pe0, mode="shuffle_dL", seed=123, tag="GWTEST")
    assert np.allclose(pe.log_pi_dL, -pe.dL_mpc)
    assert np.allclose(pe.log_pi_chirp_mass, -pe.chirp_mass_det)
    assert np.allclose(pe.log_pi_mass_ratio, -pe.mass_ratio)
    assert np.allclose(pe.mass_ratio, 0.1 * pe.chirp_mass_det + 0.01)


def test_scramble_shuffle_mass_preserves_joint_mass_pairing_and_keeps_dL() -> None:
    pe0 = _toy_pe()
    pe = apply_hierarchical_pe_scramble(pe0, mode="shuffle_mass", seed=123, tag="GWTEST")
    # dL should be unchanged.
    assert np.allclose(pe.dL_mpc, pe0.dL_mpc)
    assert np.allclose(pe.log_pi_dL, -pe.dL_mpc)
    # (mc,q) pairing should be preserved under the joint shuffle.
    assert np.allclose(pe.mass_ratio, 0.1 * pe.chirp_mass_det + 0.01)
    assert np.allclose(pe.log_pi_chirp_mass, -pe.chirp_mass_det)
    assert np.allclose(pe.log_pi_mass_ratio, -pe.mass_ratio)


def test_scramble_shuffle_mc_breaks_joint_mass_pairing() -> None:
    pe0 = _toy_pe()
    pe = apply_hierarchical_pe_scramble(pe0, mode="shuffle_mc", seed=123, tag="GWTEST")
    assert np.allclose(pe.log_pi_chirp_mass, -pe.chirp_mass_det)
    # mc is shuffled but q is not; pairing should not remain deterministic.
    assert not np.allclose(pe.mass_ratio, 0.1 * pe.chirp_mass_det + 0.01)


def test_scramble_is_deterministic_for_same_seed_and_tag() -> None:
    pe0 = _toy_pe()
    a = apply_hierarchical_pe_scramble(pe0, mode="shuffle_dL_mass", seed=7, tag="GWDET")
    b = apply_hierarchical_pe_scramble(pe0, mode="shuffle_dL_mass", seed=7, tag="GWDET")
    assert np.allclose(a.dL_mpc, b.dL_mpc)
    assert np.allclose(a.chirp_mass_det, b.chirp_mass_det)
    assert np.allclose(a.mass_ratio, b.mass_ratio)


def test_scramble_prior_dL_requires_analytic_distance_prior() -> None:
    pe0 = _toy_pe()
    with pytest.raises(ValueError):
        apply_hierarchical_pe_scramble(pe0, mode="prior_dL", seed=0, tag="GWTEST")
