import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.inversion import infer_logmu_forward
from entropy_horizon_recon.likelihoods import BaoLogLike


def test_mapping_variants_run_and_are_sane():
    """Synthetic smoke/consistency test across mapping variants.

    This is not a precision calibration test; it ensures the mapping-variant code paths
    (Ωk nuisance, residual R(z) term) run and do not produce pathological μ(A) under BH truth.
    """
    constants = PhysicalConstants()
    H0_true = 70.0
    omega_m0_true = 0.3
    z_max = 0.6
    z_grid = np.linspace(0.0, z_max, 80)
    H_true = H0_true * np.sqrt(omega_m0_true * (1.0 + z_grid) ** 3 + (1.0 - omega_m0_true))
    bg = build_background_from_H_grid(z_grid, H_true, constants=constants)

    rng = np.random.default_rng(0)

    # SN (diagonal covariance)
    sn_z = np.linspace(0.02, z_max, 18)
    m_true = 5.0 * np.log10(bg.Dl(sn_z)) + 0.0
    sigma_m = 0.08
    sn_cov = (sigma_m**2) * np.eye(sn_z.size)
    sn_m = m_true + sigma_m * rng.normal(size=m_true.shape)

    # Chronometers
    cc_z = np.linspace(0.1, z_max, 5)
    cc_sigma = np.full_like(cc_z, 8.0)
    cc_H = bg.H(cc_z) + cc_sigma * rng.normal(size=cc_z.shape)

    # Minimal BAO block
    bao = BaoLogLike.from_arrays(
        dataset="desi_2024_bao_all",
        z=np.array([0.35]),
        y=np.array([bg.Dm(np.array([0.35]))[0] / 147.0]) + np.array([0.05]) * rng.normal(size=1),
        obs=np.array(["DM_over_rs"]),
        cov=np.array([[0.05**2]]),
        constants=constants,
    )

    H_zmax = float(H_true[-1])
    x_min = float(2.0 * np.log(H0_true / H_zmax))
    x_knots = np.linspace(1.25 * x_min, 0.0, 6)
    x_grid = np.linspace(x_min, 0.0, 50)

    base = dict(
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_z=sn_z,
        sn_m=sn_m,
        sn_cov=sn_cov,
        cc_z=cc_z,
        cc_H=cc_H,
        cc_sigma_H=cc_sigma,
        bao_likes=[bao],
        constants=constants,
        n_steps=260,
        n_burn=90,
        n_draws=160,
        progress=False,
        sigma_cc_jit_scale=0.5,
        sigma_sn_jit_scale=0.02,
        sigma_d2_scale=0.185,
    )

    post0 = infer_logmu_forward(**base, n_walkers=32, seed=1, n_processes=1)
    postk = infer_logmu_forward(**base, n_walkers=36, seed=2, n_processes=1, omega_k0_prior=(-0.02, 0.02))
    postr = infer_logmu_forward(**base, n_walkers=48, seed=3, n_processes=1, use_residual=True)

    med0 = np.median(post0.logmu_x_samples, axis=0)
    medk = np.median(postk.logmu_x_samples, axis=0)
    medr = np.median(postr.logmu_x_samples, axis=0)

    # Sane: no huge excursions in the supported domain.
    assert float(np.max(np.abs(med0))) < 1.0
    assert float(np.max(np.abs(medk))) < 1.0
    assert float(np.max(np.abs(medr))) < 1.0

    # Variants shouldn't wildly disagree in this small-noise BH closure.
    assert float(np.mean(np.abs(medk - med0))) < 0.4
    assert float(np.mean(np.abs(medr - med0))) < 0.6

    # Residual knots should remain small under BH truth.
    assert "residual_r_knots" in postr.params
    assert float(np.median(np.abs(postr.params["residual_r_knots"]))) < 0.2

