from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_gwtc3_pe_distance_prior_is_dl2_for_known_event() -> None:
    pytest.importorskip("h5py")

    # This is a *smoke* guardrail for the "analytic prior removal" plumbing:
    # if GWTC-3 PEDataRelease distance priors change format or stop being ~dL^2,
    # any code path that assumes π(dL) ∝ dL^2 becomes unsafe.
    pe_file = Path("data/cache/gw/zenodo/5546663/IGWN-GWTC3p0-v1-GW200220_061928_PEDataRelease_mixed_nocosmo.h5")
    if not pe_file.exists():
        pytest.skip(f"Missing PEDataRelease file: {pe_file}")

    # Import from the *real* package under src/ (the repo-root shim ensures this works).
    from entropy_horizon_recon.dark_sirens_hierarchical_pe import load_gwtc_pe_hierarchical_samples
    from entropy_horizon_recon.gwtc_pe_priors import load_gwtc_pe_analytic_priors

    # Load samples + cached log π_PE(dL).
    pe = load_gwtc_pe_hierarchical_samples(path=pe_file, analysis="C01:IMRPhenomXPHM", max_samples=20_000, seed=0)
    assert pe.dL_mpc.size == pe.log_pi_dL.size
    assert np.all(np.isfinite(pe.dL_mpc))
    assert np.all(pe.dL_mpc > 0.0)
    assert np.all(np.isfinite(pe.log_pi_dL))

    # Re-load the analytic prior spec to verify it is PowerLaw(alpha=2) and matches the cached log π.
    pri = load_gwtc_pe_analytic_priors(path=pe_file, analysis=pe.analysis, parameters=["luminosity_distance"])
    spec, prior = pri["luminosity_distance"]
    assert spec.class_name == "PowerLaw"
    assert abs(float(spec.kwargs["alpha"]) - 2.0) < 1e-12

    log_pi = prior.logpdf(pe.dL_mpc)
    assert np.allclose(log_pi, pe.log_pi_dL, rtol=0.0, atol=0.0)

    # For a PowerLaw(alpha=2) prior, log π(dL) - 2 log dL should be (approximately) constant.
    delta = log_pi - 2.0 * np.log(pe.dL_mpc)
    # Numerical noise should be tiny (all operations are analytic + vectorized).
    assert float(np.nanstd(delta)) < 1e-10

