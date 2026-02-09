import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import BackgroundCosmology
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike
from entropy_horizon_recon.recon_spline import reconstruct_H_spline


def test_small_synthetic_closure_mu_near_one():
    constants = PhysicalConstants()
    H0 = 70.0
    omega_m0 = 0.3
    z_max = 0.6
    z_grid = np.linspace(0.0, z_max, 200)

    H_true = H0 * np.sqrt(omega_m0 * (1.0 + z_grid) ** 3 + (1.0 - omega_m0))
    invH = 1.0 / H_true
    Dc = np.empty_like(z_grid)
    Dc[0] = 0.0
    dz = np.diff(z_grid)
    Dc[1:] = constants.c_km_s * np.cumsum(0.5 * dz * (invH[:-1] + invH[1:]))
    bg = BackgroundCosmology(z_grid=z_grid, H_grid=H_true, Dc_grid=Dc, constants=constants)

    rng = np.random.default_rng(0)

    # Small SN dataset with diagonal covariance
    z_sn = np.linspace(0.02, z_max, 25)
    M_true = -3.0
    m_true = 5.0 * np.log10(bg.Dl(z_sn)) + M_true
    sigma_m = 0.08
    cov_sn = (sigma_m**2) * np.eye(z_sn.size)
    m_obs = m_true + sigma_m * rng.normal(size=z_sn.shape)
    sn_like = SNLogLike.from_arrays(z=z_sn, m=m_obs, cov=cov_sn)

    # Small chronometer dataset
    z_cc = np.array([0.1, 0.2, 0.35, 0.5])
    H_cc_true = bg.H(z_cc)
    sigma_H = np.full_like(z_cc, 8.0)
    H_cc = H_cc_true + sigma_H * rng.normal(size=z_cc.shape)
    cc_like = ChronometerLogLike.from_arrays(z=z_cc, H=H_cc, sigma_H=sigma_H)

    # Minimal BAO dataset (DM/rs and DH/rs) with diagonal covariance
    z_bao = np.array([0.35, 0.5])
    y = np.array([bg.Dm(z_bao[0]) / 147.0, bg.Dh(z_bao[1]) / 147.0])
    obs = np.array(["DM_over_rs", "DH_over_rs"])
    cov_bao = np.diag([0.2**2, 0.2**2])
    bao_like = BaoLogLike.from_arrays(
        dataset="desi_2024_bao_all",
        z=z_bao,
        y=y + np.sqrt(np.diag(cov_bao)) * rng.normal(size=y.shape),
        obs=obs,
        cov=cov_bao,
        constants=constants,
    )

    z_knots = np.linspace(0.0, z_max, 10)
    post = reconstruct_H_spline(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=[bao_like],
        constants=constants,
        z_grid=z_grid,
        z_max_background=z_max,
        smooth_lambda=5.0,
        n_bootstrap=8,
        seed=1,
    )

    H_med = np.median(post.H_samples, axis=0)
    rel = np.abs(H_med - H_true) / H_true
    # A very small synthetic dataset + bootstrap is intentionally noisy; this is a smoke-level closure check.
    assert np.median(rel[(z_grid > 0.05) & (z_grid < 0.55)]) < 0.25
