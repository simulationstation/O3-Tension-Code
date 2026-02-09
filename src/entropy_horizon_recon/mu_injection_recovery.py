from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples, compute_hierarchical_pe_logL_draws
from .dark_sirens_selection import O3InjectionSet, SelectionAlphaResult, calibrate_snr_threshold_match_count
from .siren_injection_recovery import InjectionRecoveryConfig, SyntheticEventTruth, synthesize_pe_posterior_samples
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em, x_of_z_from_H


def _logmeanexp_1d(logw: np.ndarray) -> float:
    logw = np.asarray(logw, dtype=float)
    if logw.ndim != 1:
        raise ValueError("_logmeanexp_1d expects a 1D array")
    if logw.size == 0:
        return float("-inf")
    m_finite = np.isfinite(logw)
    if not np.any(m_finite):
        return float("-inf")
    lw = np.asarray(logw[m_finite], dtype=float)
    m = float(np.max(lw))
    # Guard against the "all -inf" case which otherwise produces NaN via (-inf) - (-inf).
    if not np.isfinite(m):
        return float("-inf")
    exp_term = np.exp(np.clip(lw - m, -700.0, 50.0))
    mean_exp = float(np.mean(exp_term))
    if not (np.isfinite(mean_exp) and mean_exp > 0.0):
        return float("-inf")
    return float(m + np.log(mean_exp))


def build_linear_mu_forward_posterior(
    *,
    h0: float,
    omega_m0: float,
    omega_k0: float = 0.0,
    z_max: float,
    mu_slope_x: float,
    n_draws: int = 1,
    n_z: int = 501,
    n_x: int = 401,
) -> MuForwardPosterior:
    """Build a simple synthetic MuForwardPosterior for μ-vs-GR injection tests.

    Background: flat/curved LCDM with analytic H(z) on a grid.
    Deformation: log μ(x) = (mu_slope_x) * x, with μ(0)=1.
    """
    h0 = float(h0)
    omega_m0 = float(omega_m0)
    omega_k0 = float(omega_k0)
    z_max = float(z_max)
    mu_slope_x = float(mu_slope_x)

    if not (np.isfinite(h0) and h0 > 0.0):
        raise ValueError("h0 must be finite and positive.")
    if not (np.isfinite(omega_m0) and 0.0 <= omega_m0 <= 2.0):
        raise ValueError("omega_m0 must be finite and in a sane range.")
    if not (np.isfinite(omega_k0) and -2.0 <= omega_k0 <= 2.0):
        raise ValueError("omega_k0 must be finite and in a sane range.")
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")
    if int(n_draws) <= 0:
        raise ValueError("n_draws must be positive.")
    if int(n_z) < 10:
        raise ValueError("n_z too small.")
    if int(n_x) < 10:
        raise ValueError("n_x too small.")

    z_grid = np.linspace(0.0, z_max, int(n_z), dtype=float)
    zp1 = 1.0 + z_grid
    omega_l = 1.0 - omega_m0 - omega_k0
    E2 = omega_m0 * zp1**3 + omega_k0 * zp1**2 + omega_l
    if np.any(E2 <= 0.0) or np.any(~np.isfinite(E2)):
        raise ValueError("Non-physical E(z)^2 encountered for the requested Ω parameters.")
    H = h0 * np.sqrt(E2)

    # Ensure the x-grid covers the mapped x(z) domain (x=0 at z=0).
    x_z = x_of_z_from_H(z_grid, H, H0=h0, omega_k0=omega_k0)
    x_min = float(np.min(x_z))
    # Force the last grid point to be exactly 0 so mu0 is handled robustly in predict_r_gw_em.
    x_grid = np.linspace(x_min, 0.0, int(n_x), dtype=float)
    x_grid[-1] = 0.0

    logmu = mu_slope_x * x_grid

    H_samples = np.tile(H.reshape((1, -1)), (int(n_draws), 1))
    logmu_x_samples = np.tile(logmu.reshape((1, -1)), (int(n_draws), 1))
    return MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu_x_samples,
        z_grid=z_grid,
        H_samples=H_samples,
        H0=np.full((int(n_draws),), h0, dtype=float),
        omega_m0=np.full((int(n_draws),), omega_m0, dtype=float),
        omega_k0=np.full((int(n_draws),), omega_k0, dtype=float),
        sigma8_0=None,
    )


def _chirp_mass_from_m1_m2(m1: float, m2: float) -> float:
    m1 = float(m1)
    m2 = float(m2)
    mt = m1 + m2
    return float((m1 * m2) ** (3.0 / 5.0) / (mt ** (1.0 / 5.0)))


def _pe_sigmas_from_cfg_and_snr(*, cfg: InjectionRecoveryConfig, snr_net_opt_true: float) -> tuple[float, float, float]:
    snr = float(max(float(snr_net_opt_true), 1e-6))
    if str(cfg.dl_sigma_mode) == "constant":
        dl_log_sigma = float(cfg.dl_frac_sigma0)
    else:
        dl_log_sigma = float(cfg.dl_frac_sigma0) / snr
        dl_log_sigma = max(dl_log_sigma, float(cfg.dl_frac_sigma_floor))
    dl_log_sigma = float(np.clip(dl_log_sigma, 1e-4, 2.0))

    mc_log_sigma = float(np.clip(float(cfg.mc_frac_sigma0) / snr, 1e-4, 0.5))
    q_sigma = float(np.clip(float(cfg.q_sigma0), 1e-4, 1.0))
    return dl_log_sigma, mc_log_sigma, q_sigma


def _sample_truncated_normal(
    rng: np.random.Generator,
    *,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    size: int,
    max_iter: int = 50,
) -> np.ndarray:
    mu = float(mu)
    sigma = float(sigma)
    lo = float(lo)
    hi = float(hi)
    n = int(size)
    if n <= 0:
        return np.zeros((0,), dtype=float)
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("Invalid truncated normal parameters.")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("Invalid truncation bounds.")

    out = np.empty((n,), dtype=float)
    filled = 0
    for _ in range(int(max_iter)):
        m = n - filled
        if m <= 0:
            break
        x = rng.normal(loc=mu, scale=sigma, size=m).astype(float)
        keep = x[(x >= lo) & (x <= hi)]
        k = int(min(keep.size, m))
        if k > 0:
            out[filled : filled + k] = keep[:k]
            filled += k
    if filled < n:
        out[filled:] = np.clip(rng.normal(loc=mu, scale=sigma, size=n - filled).astype(float), lo, hi)
    return out


def _build_snr_binned_pdet(
    *,
    snr: np.ndarray,
    found_ifar: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    snr = np.asarray(snr, dtype=float)
    found = np.asarray(found_ifar, dtype=bool)
    if snr.shape != found.shape:
        raise ValueError("snr and found_ifar must match.")
    nb = int(nbins)
    if nb < 20:
        raise ValueError("snr_binned_nbins too small.")
    edges = np.quantile(snr, np.linspace(0.0, 1.0, nb + 1))
    edges = np.unique(edges)
    if edges.size < 10:
        raise ValueError("Too few unique SNR edges for snr_binned.")
    bin_idx = np.clip(np.digitize(snr, edges) - 1, 0, edges.size - 2)
    p = np.zeros(edges.size - 1, dtype=float)
    for i in range(p.size):
        m_i = bin_idx == i
        if not np.any(m_i):
            p[i] = p[i - 1] if i > 0 else 0.0
            continue
        p[i] = float(np.mean(found[m_i].astype(float)))
    p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
    return edges, p


def generate_synthetic_detected_events_from_injections_for_post(
    *,
    injections: O3InjectionSet,
    cfg: InjectionRecoveryConfig,
    post_truth: MuForwardPosterior,
    convention: Literal["A", "B"] = "A",
    mu_det_distance: Literal["gw", "em"] = "gw",
    n_events: int,
    seed: int,
) -> list[SyntheticEventTruth]:
    """Closed-loop detected-event generator using an injection set and a truth MuForwardPosterior.

    This mirrors the selection proxy logic in `compute_selection_alpha_from_injections`:
      - calibrate a p_det(snr) curve from fiducial injection SNRs + found_ifar
      - rescale SNR by dL_fid / dL_det_truth(z) for each injection
      - sample detected events from w(z,m) * p_det

    The injected GW "measured distance" is interpreted as dL_GW (i.e., the propagation distance),
    with dL_det_truth = dL_GW when mu_det_distance='gw'.
    """
    n_events = int(n_events)
    if n_events <= 0:
        raise ValueError("n_events must be positive.")
    rng = np.random.default_rng(int(seed))

    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be one of {'gw','em'}.")

    # Injection arrays.
    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr_fid = np.asarray(injections.snr_net_opt, dtype=float)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)
    found = np.asarray(injections.found_ifar, dtype=bool)

    z_hi = float(min(float(cfg.z_max), float(np.nanmax(z)), float(post_truth.z_grid[-1])))
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr_fid)
        & (snr_fid > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError("No usable injections after basic cuts.")
    z = z[m]
    dL_fid = dL_fid[m]
    snr_fid = snr_fid[m]
    m1 = m1[m]
    m2 = m2[m]
    found = found[m]

    # Population-vs-injection importance weights (matches siren_injection_recovery generator).
    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("injections.mixture_weight must match injections.z shape.")
    w = w * mw
    if str(cfg.weight_mode) == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite/non-positive values; cannot sample events.")
        w = w / pdf

    if str(cfg.pop_mass_mode) != "none" and str(cfg.weight_mode) == "inv_sampling_pdf":
        if str(cfg.inj_mass_pdf_coords) == "m1m2":
            w = w / np.clip(m1, 1e-300, np.inf)
        elif str(cfg.inj_mass_pdf_coords) == "m1q":
            pass
        else:
            raise ValueError("Unknown inj_mass_pdf_coords.")

    if str(cfg.pop_z_mode) != "none":
        # Fixed-LCDM approximation (Planck-ish), matching selection proxy conventions.
        H0_ref = 67.7
        om0 = 0.31
        c = 299792.458
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        dc = (c / H0_ref) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        dVdz = (c / (H0_ref * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        base = dVdz / (1.0 + z)
        if str(cfg.pop_z_mode) == "comoving_uniform":
            w = w * base
        else:
            w = w * base * (1.0 + z) ** float(cfg.pop_z_k)

    if str(cfg.pop_mass_mode) != "none":
        alpha = float(cfg.pop_m1_alpha)
        mmin = float(cfg.pop_m_min)
        mmax = float(cfg.pop_m_max)
        beta_q = float(cfg.pop_q_beta)
        q = np.clip(m2 / m1, 1e-6, 1.0)
        if str(cfg.pop_mass_mode) == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        else:
            delta = float(cfg.pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for smooth mass modes.")

            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 1e-300, 1.0)
            log_q = beta_q * np.log(np.clip(q, 1e-300, np.inf))
            log_taper = np.log(taper)
            log_pl = -alpha * np.log(np.clip(m1, 1e-300, np.inf)) + log_q + log_taper

            if str(cfg.pop_mass_mode) == "powerlaw_q_smooth":
                log_mass = log_pl
            elif str(cfg.pop_mass_mode) == "powerlaw_peak_q_smooth":
                mp = float(cfg.pop_m_peak)
                sig = float(cfg.pop_m_peak_sigma)
                f_peak = float(cfg.pop_m_peak_frac)
                log_peak = -0.5 * ((m1 - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper
                if f_peak <= 0.0:
                    log_mass = log_pl
                elif f_peak >= 1.0:
                    log_mass = log_peak
                else:
                    log_mass = np.logaddexp(np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak)
            else:
                raise ValueError("Unknown pop_mass_mode.")

            m_ok = np.isfinite(log_mass)
            if not np.any(m_ok):
                raise ValueError("All mass weights non-finite for injection generation.")
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights; check configuration.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr_fid = snr_fid[good_w]
        m1 = m1[good_w]
        m2 = m2[good_w]
        found = found[good_w]
        w = w[good_w]

    # Truth distances from post_truth (interpolate from a dense grid).
    z_post = np.asarray(post_truth.z_grid, dtype=float)
    dL_em_grid = np.asarray(predict_dL_em(post_truth, z_eval=z_post), dtype=float)[0]
    _, R_grid = predict_r_gw_em(post_truth, z_eval=None, convention=convention, allow_extrapolation=False)
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)[0]
    dL_em = np.interp(z, z_post, dL_em_grid)
    dL_gw = np.interp(z, z_post, dL_gw_grid)
    dL_det = dL_gw if mu_det_distance == "gw" else dL_em
    dL_det = np.clip(dL_det, 1e-6, np.inf)

    snr_true = snr_fid * (dL_fid / dL_det)

    thresh = float(calibrate_snr_threshold_match_count(snr_net_opt=snr_fid, found_ifar=found))
    if str(cfg.det_model) == "threshold":
        pdet = (snr_true > thresh).astype(float)
    else:
        edges, pvals = _build_snr_binned_pdet(snr=snr_fid, found_ifar=found, nbins=int(cfg.snr_binned_nbins))
        idx_p = np.clip(np.digitize(snr_true, edges) - 1, 0, pvals.size - 1)
        pdet = np.asarray(pvals[idx_p], dtype=float)

    w_det = np.clip(w * np.clip(pdet, 0.0, 1.0), 0.0, np.inf)
    good_det = np.isfinite(w_det) & (w_det > 0.0)
    if not np.any(good_det):
        raise ValueError("No injections have positive detected weight w*p_det.")
    if not np.all(good_det):
        z = z[good_det]
        dL_det = dL_det[good_det]
        snr_true = snr_true[good_det]
        pdet = pdet[good_det]
        m1 = m1[good_det]
        m2 = m2[good_det]
        w_det = w_det[good_det]
        dL_fid = dL_fid[good_det]
        snr_fid = snr_fid[good_det]

    prob = w_det / float(np.sum(w_det))
    n_avail = int(prob.size)
    replace = bool(n_events > n_avail)
    idx = rng.choice(n_avail, size=n_events, replace=replace, p=prob)

    out: list[SyntheticEventTruth] = []
    for i, j in enumerate(idx.tolist()):
        zt = float(z[j])
        dL_true_j = float(dL_det[j])  # interpret PE distance as dL_GW (or dL_EM if mu_det_distance='em')
        snr_true_j = float(snr_true[j])
        m1s = float(m1[j])
        m2s = float(m2[j])
        q_true = float(m2s / m1s)
        mc_src = _chirp_mass_from_m1_m2(m1s, m2s)
        mc_det = float(mc_src * (1.0 + zt))

        if str(cfg.pe_obs_mode) == "truth":
            dL_obs = dL_true_j
            mc_det_obs = mc_det
            q_obs = q_true
        elif str(cfg.pe_obs_mode) == "noisy":
            dl_log_sigma, mc_log_sigma, q_sigma = _pe_sigmas_from_cfg_and_snr(cfg=cfg, snr_net_opt_true=snr_true_j)
            dL_obs = float(np.exp(rng.normal(loc=np.log(dL_true_j), scale=dl_log_sigma)))
            mc_det_obs = float(np.exp(rng.normal(loc=np.log(mc_det), scale=mc_log_sigma)))
            q_obs = float(_sample_truncated_normal(rng, mu=q_true, sigma=q_sigma, lo=0.05, hi=1.0, size=1)[0])
        else:
            raise ValueError("Unknown pe_obs_mode.")

        out.append(
            SyntheticEventTruth(
                event=f"SYNTH_{i+1:05d}",
                z=zt,
                dL_mpc_true=dL_true_j,
                dL_mpc_obs=float(dL_obs),
                m1_source=m1s,
                m2_source=m2s,
                chirp_mass_det_true=mc_det,
                chirp_mass_det_obs=float(mc_det_obs),
                mass_ratio_true=q_true,
                mass_ratio_obs=float(q_obs),
                snr_net_opt_fid=float(snr_fid[j]),
                dL_mpc_fid=float(dL_fid[j]),
                snr_net_opt_true=snr_true_j,
                p_det_true=float(pdet[j]),
            )
        )
    return out


@dataclass(frozen=True)
class DetectedInjectionSampler:
    """Precomputed detected-injection sampling distribution for fast multi-rep injection suites."""

    z: np.ndarray
    dL_det: np.ndarray
    snr_true: np.ndarray
    pdet: np.ndarray
    dL_fid: np.ndarray
    snr_fid: np.ndarray
    m1_source: np.ndarray
    m2_source: np.ndarray
    prob: np.ndarray  # normalized sampling probabilities (sum=1)
    z_hi: float
    convention: str
    mu_det_distance: str
    meta: dict[str, Any]

    def sample_truths(self, *, cfg: InjectionRecoveryConfig, n_events: int, seed: int) -> list[SyntheticEventTruth]:
        n_events = int(n_events)
        if n_events <= 0:
            raise ValueError("n_events must be positive.")
        rng = np.random.default_rng(int(seed))
        prob = np.asarray(self.prob, dtype=float)
        n_avail = int(prob.size)
        replace = bool(n_events > n_avail)
        idx = rng.choice(n_avail, size=n_events, replace=replace, p=prob)

        out: list[SyntheticEventTruth] = []
        for i, j in enumerate(idx.tolist()):
            zt = float(self.z[j])
            dL_true_j = float(self.dL_det[j])
            snr_true_j = float(self.snr_true[j])
            m1s = float(self.m1_source[j])
            m2s = float(self.m2_source[j])
            q_true = float(m2s / m1s)
            mc_src = _chirp_mass_from_m1_m2(m1s, m2s)
            mc_det = float(mc_src * (1.0 + zt))

            if str(cfg.pe_obs_mode) == "truth":
                dL_obs = dL_true_j
                mc_det_obs = mc_det
                q_obs = q_true
            elif str(cfg.pe_obs_mode) == "noisy":
                dl_log_sigma, mc_log_sigma, q_sigma = _pe_sigmas_from_cfg_and_snr(cfg=cfg, snr_net_opt_true=snr_true_j)
                dL_obs = float(np.exp(rng.normal(loc=np.log(dL_true_j), scale=dl_log_sigma)))
                mc_det_obs = float(np.exp(rng.normal(loc=np.log(mc_det), scale=mc_log_sigma)))
                q_obs = float(_sample_truncated_normal(rng, mu=q_true, sigma=q_sigma, lo=0.05, hi=1.0, size=1)[0])
            else:
                raise ValueError("Unknown pe_obs_mode.")

            out.append(
                SyntheticEventTruth(
                    event=f"SYNTH_{i+1:05d}",
                    z=zt,
                    dL_mpc_true=dL_true_j,
                    dL_mpc_obs=float(dL_obs),
                    m1_source=m1s,
                    m2_source=m2s,
                    chirp_mass_det_true=mc_det,
                    chirp_mass_det_obs=float(mc_det_obs),
                    mass_ratio_true=q_true,
                    mass_ratio_obs=float(q_obs),
                    snr_net_opt_fid=float(self.snr_fid[j]),
                    dL_mpc_fid=float(self.dL_fid[j]),
                    snr_net_opt_true=snr_true_j,
                    p_det_true=float(self.pdet[j]),
                )
            )
        return out


def build_detected_injection_sampler_for_post(
    *,
    injections: O3InjectionSet,
    cfg: InjectionRecoveryConfig,
    post_truth: MuForwardPosterior,
    convention: Literal["A", "B"] = "A",
    mu_det_distance: Literal["gw", "em"] = "gw",
) -> DetectedInjectionSampler:
    """Precompute the detected sampling distribution w*p_det for a fixed truth model."""
    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be one of {'gw','em'}.")

    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr_fid = np.asarray(injections.snr_net_opt, dtype=float)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)
    found = np.asarray(injections.found_ifar, dtype=bool)

    z_hi = float(min(float(cfg.z_max), float(np.nanmax(z)), float(post_truth.z_grid[-1])))
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr_fid)
        & (snr_fid > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError("No usable injections after basic cuts.")
    z = z[m]
    dL_fid = dL_fid[m]
    snr_fid = snr_fid[m]
    m1 = m1[m]
    m2 = m2[m]
    found = found[m]

    # Population-vs-injection importance weights.
    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("injections.mixture_weight must match injections.z shape.")
    w = w * mw
    if str(cfg.weight_mode) == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite/non-positive values; cannot build sampler.")
        w = w / pdf

    if str(cfg.pop_mass_mode) != "none" and str(cfg.weight_mode) == "inv_sampling_pdf":
        if str(cfg.inj_mass_pdf_coords) == "m1m2":
            w = w / np.clip(m1, 1e-300, np.inf)
        elif str(cfg.inj_mass_pdf_coords) == "m1q":
            pass
        else:
            raise ValueError("Unknown inj_mass_pdf_coords.")

    if str(cfg.pop_z_mode) != "none":
        H0_ref = 67.7
        om0 = 0.31
        c = 299792.458
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        dc = (c / H0_ref) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        dVdz = (c / (H0_ref * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        base = dVdz / (1.0 + z)
        if str(cfg.pop_z_mode) == "comoving_uniform":
            w = w * base
        else:
            w = w * base * (1.0 + z) ** float(cfg.pop_z_k)

    if str(cfg.pop_mass_mode) != "none":
        alpha = float(cfg.pop_m1_alpha)
        mmin = float(cfg.pop_m_min)
        mmax = float(cfg.pop_m_max)
        beta_q = float(cfg.pop_q_beta)
        q = np.clip(m2 / m1, 1e-6, 1.0)
        if str(cfg.pop_mass_mode) == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        else:
            delta = float(cfg.pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for smooth mass modes.")

            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 1e-300, 1.0)
            log_q = beta_q * np.log(np.clip(q, 1e-300, np.inf))
            log_taper = np.log(taper)
            log_pl = -alpha * np.log(np.clip(m1, 1e-300, np.inf)) + log_q + log_taper

            if str(cfg.pop_mass_mode) == "powerlaw_q_smooth":
                log_mass = log_pl
            elif str(cfg.pop_mass_mode) == "powerlaw_peak_q_smooth":
                mp = float(cfg.pop_m_peak)
                sig = float(cfg.pop_m_peak_sigma)
                f_peak = float(cfg.pop_m_peak_frac)
                log_peak = -0.5 * ((m1 - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper
                if f_peak <= 0.0:
                    log_mass = log_pl
                elif f_peak >= 1.0:
                    log_mass = log_peak
                else:
                    log_mass = np.logaddexp(np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak)
            else:
                raise ValueError("Unknown pop_mass_mode.")

            m_ok = np.isfinite(log_mass)
            if not np.any(m_ok):
                raise ValueError("All mass weights non-finite for injection sampler.")
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights; check configuration.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr_fid = snr_fid[good_w]
        m1 = m1[good_w]
        m2 = m2[good_w]
        found = found[good_w]
        w = w[good_w]

    # Truth distances from post_truth (interpolate from a dense grid).
    z_post = np.asarray(post_truth.z_grid, dtype=float)
    dL_em_grid = np.asarray(predict_dL_em(post_truth, z_eval=z_post), dtype=float)[0]
    _, R_grid = predict_r_gw_em(post_truth, z_eval=None, convention=convention, allow_extrapolation=False)
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)[0]
    dL_em = np.interp(z, z_post, dL_em_grid)
    dL_gw = np.interp(z, z_post, dL_gw_grid)
    dL_det = dL_gw if mu_det_distance == "gw" else dL_em
    dL_det = np.clip(dL_det, 1e-6, np.inf)

    snr_true = snr_fid * (dL_fid / dL_det)

    meta: dict[str, Any] = {"det_model": str(cfg.det_model), "z_hi": float(z_hi)}
    thresh = float(calibrate_snr_threshold_match_count(snr_net_opt=snr_fid, found_ifar=found))
    meta["snr_threshold"] = float(thresh)
    if str(cfg.det_model) == "threshold":
        pdet = (snr_true > thresh).astype(float)
    else:
        edges, pvals = _build_snr_binned_pdet(snr=snr_fid, found_ifar=found, nbins=int(cfg.snr_binned_nbins))
        meta["pdet_edges_n"] = int(edges.size)
        idx_p = np.clip(np.digitize(snr_true, edges) - 1, 0, pvals.size - 1)
        pdet = np.asarray(pvals[idx_p], dtype=float)

    w_det = np.clip(w * np.clip(pdet, 0.0, 1.0), 0.0, np.inf)
    good_det = np.isfinite(w_det) & (w_det > 0.0)
    if not np.any(good_det):
        raise ValueError("No injections have positive detected weight w*p_det.")
    if not np.all(good_det):
        z = z[good_det]
        dL_det = dL_det[good_det]
        snr_true = snr_true[good_det]
        pdet = pdet[good_det]
        m1 = m1[good_det]
        m2 = m2[good_det]
        w_det = w_det[good_det]
        dL_fid = dL_fid[good_det]
        snr_fid = snr_fid[good_det]

    prob = w_det / float(np.sum(w_det))
    return DetectedInjectionSampler(
        z=np.asarray(z, dtype=float),
        dL_det=np.asarray(dL_det, dtype=float),
        snr_true=np.asarray(snr_true, dtype=float),
        pdet=np.asarray(pdet, dtype=float),
        dL_fid=np.asarray(dL_fid, dtype=float),
        snr_fid=np.asarray(snr_fid, dtype=float),
        m1_source=np.asarray(m1, dtype=float),
        m2_source=np.asarray(m2, dtype=float),
        prob=np.asarray(prob, dtype=float),
        z_hi=float(z_hi),
        convention=str(convention),
        mu_det_distance=str(mu_det_distance),
        meta=meta,
    )


@dataclass(frozen=True)
class MuVsGrScore:
    delta_lpd_data: float
    delta_lpd_total: float
    lpd_mu_data: float
    lpd_gr_data: float
    lpd_mu_total: float
    lpd_gr_total: float
    n_events: int
    selection_alpha_mu: float | None
    selection_alpha_gr: float | None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def score_mu_vs_gr_hierarchical_pe(
    *,
    pe_by_event: dict[str, GWTCPeHierarchicalSamples],
    post_score: MuForwardPosterior,
    selection_alpha: SelectionAlphaResult | None,
    cfg: InjectionRecoveryConfig,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    importance_smoothing: Literal["none", "truncate", "psis"] = "none",
    importance_truncate_tau: float | None = None,
) -> MuVsGrScore:
    """Compute ΔLPD for a set of events under a fixed μ model (post_score) vs its GR baseline."""
    if not pe_by_event:
        raise ValueError("pe_by_event is empty.")

    # Per-event logL(draw) vectors.
    n_draws = int(post_score.H_samples.shape[0])
    sum_mu = np.zeros((n_draws,), dtype=float)
    sum_gr = np.zeros((n_draws,), dtype=float)
    for pe in pe_by_event.values():
        logL_mu, logL_gr = compute_hierarchical_pe_logL_draws(
            pe=pe,
            post=post_score,
            convention=convention,
            z_max=float(z_max),
            pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
            pop_z_k=float(cfg.pop_z_k),
            pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(cfg.pop_m1_alpha),
            pop_m_min=float(cfg.pop_m_min),
            pop_m_max=float(cfg.pop_m_max),
            pop_q_beta=float(cfg.pop_q_beta),
            pop_m_taper_delta=float(cfg.pop_m_taper_delta),
            pop_m_peak=float(cfg.pop_m_peak),
            pop_m_peak_sigma=float(cfg.pop_m_peak_sigma),
            pop_m_peak_frac=float(cfg.pop_m_peak_frac),
            importance_smoothing=importance_smoothing,
            importance_truncate_tau=importance_truncate_tau,
            return_diagnostics=False,
        )
        sum_mu += np.asarray(logL_mu, dtype=float)
        sum_gr += np.asarray(logL_gr, dtype=float)

    lpd_mu_data = float(_logmeanexp_1d(sum_mu))
    lpd_gr_data = float(_logmeanexp_1d(sum_gr))
    delta_lpd_data = float(lpd_mu_data - lpd_gr_data)

    if selection_alpha is None:
        return MuVsGrScore(
            delta_lpd_data=delta_lpd_data,
            delta_lpd_total=delta_lpd_data,
            lpd_mu_data=lpd_mu_data,
            lpd_gr_data=lpd_gr_data,
            lpd_mu_total=lpd_mu_data,
            lpd_gr_total=lpd_gr_data,
            n_events=int(len(pe_by_event)),
            selection_alpha_mu=None,
            selection_alpha_gr=None,
        )

    n_ev = int(len(pe_by_event))
    log_alpha_mu = np.log(np.clip(np.asarray(selection_alpha.alpha_mu, dtype=float), 1e-300, np.inf))
    log_alpha_gr = np.log(np.clip(np.asarray(selection_alpha.alpha_gr, dtype=float), 1e-300, np.inf))

    lpd_mu_total = float(_logmeanexp_1d(sum_mu - float(n_ev) * log_alpha_mu))
    lpd_gr_total = float(_logmeanexp_1d(sum_gr - float(n_ev) * log_alpha_gr))
    delta_lpd_total = float(lpd_mu_total - lpd_gr_total)

    return MuVsGrScore(
        delta_lpd_data=delta_lpd_data,
        delta_lpd_total=delta_lpd_total,
        lpd_mu_data=lpd_mu_data,
        lpd_gr_data=lpd_gr_data,
        lpd_mu_total=lpd_mu_total,
        lpd_gr_total=lpd_gr_total,
        n_events=n_ev,
        selection_alpha_mu=float(np.nanmean(selection_alpha.alpha_mu)),
        selection_alpha_gr=float(np.nanmean(selection_alpha.alpha_gr)),
    )


def synthesize_hierarchical_pe_by_event(
    *,
    truths: list[SyntheticEventTruth],
    cfg: InjectionRecoveryConfig,
    seed: int,
) -> dict[str, GWTCPeHierarchicalSamples]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(cfg.pe_seed)]))
    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = {}
    for t in truths:
        pe_by_event[str(t.event)] = synthesize_pe_posterior_samples(truth=t, cfg=cfg, rng=rng)
    return pe_by_event


def write_pe_cache_npz(out_dir: str | Path, *, pe_by_event: dict[str, GWTCPeHierarchicalSamples], truth_meta: dict[str, Any]) -> None:
    out_path = Path(out_dir)
    pe_dir = out_path / "pe_cache"
    pe_dir.mkdir(parents=True, exist_ok=True)
    for ev, pe in pe_by_event.items():
        meta = dict(truth_meta)
        meta.update(
            {
                "event": str(ev),
                "pe_file": str(pe.file),
                "pe_analysis": str(pe.analysis),
                "pe_analysis_chosen": str(pe.analysis),
                "n_total": int(pe.n_total),
                "n_used": int(pe.n_used),
                "prior_spec_json": json.dumps(pe.prior_spec, sort_keys=True),
            }
        )
        np.savez(
            pe_dir / f"pe_hier_{ev}.npz",
            meta=json.dumps(meta, sort_keys=True),
            dL_mpc=np.asarray(pe.dL_mpc, dtype=np.float64),
            chirp_mass_det=np.asarray(pe.chirp_mass_det, dtype=np.float64),
            mass_ratio=np.asarray(pe.mass_ratio, dtype=np.float64),
            log_pi_dL=np.asarray(pe.log_pi_dL, dtype=np.float64),
            log_pi_chirp_mass=np.asarray(pe.log_pi_chirp_mass, dtype=np.float64),
            log_pi_mass_ratio=np.asarray(pe.log_pi_mass_ratio, dtype=np.float64),
        )
