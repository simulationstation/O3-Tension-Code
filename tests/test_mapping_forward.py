import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.mapping import forward_H_from_muA, mu_from_H_and_dHdz


def test_forward_mu_one_recovers_lcdm_form():
    constants = PhysicalConstants()
    z = np.linspace(0.0, 1.2, 200)
    H0 = 70.0
    omega_m0 = 0.3

    def mu_of_A(_A):
        return 1.0

    H = forward_H_from_muA(
        z,
        mu_of_A=mu_of_A,
        H0_km_s_Mpc=H0,
        omega_m0=omega_m0,
        constants=constants,
    )
    H2 = H**2
    H2_ref = (H0**2) * (omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0))
    assert np.max(np.abs(H2 / H2_ref - 1.0)) < 5e-4


def test_mu_from_lcdm_is_unity():
    z = np.linspace(0.0, 1.2, 400)
    H0 = 70.0
    omega_m0 = 0.3
    H = H0 * np.sqrt(omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0))
    # analytic derivative
    dH2_dz = 3.0 * (H0**2) * omega_m0 * (1.0 + z) ** 2
    dH_dz = 0.5 * dH2_dz / H
    mu = mu_from_H_and_dHdz(H, dH_dz, z, H0_km_s_Mpc=H0, omega_m0=omega_m0)
    assert np.max(np.abs(mu - 1.0)) < 1e-10

