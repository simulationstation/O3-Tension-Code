import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.forward_model import ForwardModel


def test_forward_model_bh_matches_lcdm():
    constants = PhysicalConstants()
    z = np.linspace(0.0, 2.0, 800)
    H0 = 70.0
    omega_m0 = 0.3

    x_knots = np.linspace(-3.0, 0.0, 9)
    logmu_knots = np.zeros_like(x_knots)
    fm = ForwardModel(constants=constants, x_knots=x_knots)
    H = fm.solve_H_from_logmu_knots(z, logmu_knots=logmu_knots, H0_km_s_Mpc=H0, omega_m0=omega_m0)

    H2_ref = (H0**2) * (omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0))
    rel_err = np.max(np.abs(H**2 / H2_ref - 1.0))
    assert rel_err < 1e-6
    assert np.all(np.diff(H) > 0)
