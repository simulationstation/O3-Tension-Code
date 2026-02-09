import numpy as np

from entropy_horizon_recon.siren_gap import SirenEvent, score_siren_events
from entropy_horizon_recon.sirens import MuForwardPosterior, predict_dL_em, predict_dL_gw


def _fake_posterior(*, logmu_x: np.ndarray) -> MuForwardPosterior:
    """Build a tiny, deterministic MuForwardPosterior for unit tests."""
    z_grid = np.linspace(0.0, 1.0, 50)
    H0 = 70.0
    # Smooth monotonic H(z) that stays physical for Ok=0.
    H = H0 * (1.0 + 0.5 * z_grid)

    n_draws = 128
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu_x = np.asarray(logmu_x, dtype=float)
    if logmu_x.shape != x_grid.shape:
        raise ValueError("logmu_x must match x_grid shape.")

    return MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=np.repeat(logmu_x.reshape((1, -1)), n_draws, axis=0),
        z_grid=z_grid,
        H_samples=np.repeat(H.reshape((1, -1)), n_draws, axis=0),
        H0=np.full((n_draws,), H0),
        omega_m0=np.full((n_draws,), 0.3),
        omega_k0=np.zeros((n_draws,)),
        sigma8_0=None,
    )


def test_siren_gap_mu_equals_gr_gives_zero_delta_lpd():
    # mu(A)=1 everywhere => R(z)=1 => GW distance prediction equals EM distance prediction.
    post = _fake_posterior(logmu_x=np.array([0.0, 0.0]))
    z = 0.5
    dL_em = float(predict_dL_em(post, z_eval=np.array([z]))[0, 0])
    ev = SirenEvent(name="toy", z=z, dist="normal", dL_Mpc=dL_em, dL_sigma_Mpc=0.05 * dL_em)

    summ, scores = score_siren_events(run_label="toy", post=post, events=[ev], convention="A")
    assert len(scores) == 1
    assert abs(scores[0].delta_lpd) < 1e-10
    assert abs(summ.delta_lpd_total) < 1e-10


def test_siren_gap_nontrivial_mu_can_prefer_mu_model():
    # Make mu increase with redshift (x<0): set logmu(x=-1)=+0.2, logmu(0)=0 => mu(z)/mu0 > 1.
    post = _fake_posterior(logmu_x=np.array([0.2, 0.0]))
    z = 0.5
    dL_gw = float(predict_dL_gw(post, z_eval=np.array([z]), convention="A")[0][0, 0])
    dL_em = float(predict_dL_em(post, z_eval=np.array([z]))[0, 0])
    assert dL_gw > dL_em  # convention A with mu(z)>mu0 implies GW distance larger

    # Event centered on the mu-model prediction.
    ev = SirenEvent(name="toy", z=z, dist="normal", dL_Mpc=dL_gw, dL_sigma_Mpc=0.01 * dL_em)
    summ, scores = score_siren_events(run_label="toy", post=post, events=[ev], convention="A")
    assert scores[0].delta_lpd > 0.0
    assert summ.delta_lpd_total > 0.0

