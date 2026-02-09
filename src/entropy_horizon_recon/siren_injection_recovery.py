from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .constants import PhysicalConstants
from .dark_siren_h0 import (  # noqa: SLF001
    _calibrate_detection_model_from_snr_and_found,
    _alpha_h0_grid_from_injections,
    _build_lcdm_distance_cache,
    _injection_weights,
    compute_gr_h0_posterior_grid_hierarchical_pe,
)
from .dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from .dark_sirens_selection import (
    O3InjectionSet,
    calibrate_snr_threshold_match_count,
    load_o3_injections,
)
from .gwtc_pe_priors import parse_gwtc_analytic_prior


@dataclass(frozen=True)
class SyntheticEventTruth:
    event: str
    z: float
    dL_mpc_true: float
    dL_mpc_obs: float
    m1_source: float
    m2_source: float
    chirp_mass_det_true: float
    chirp_mass_det_obs: float
    mass_ratio_true: float
    mass_ratio_obs: float
    snr_net_opt_fid: float
    dL_mpc_fid: float
    snr_net_opt_true: float
    p_det_true: float

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InjectionRecoveryConfig:
    """Configuration for a synthetic injection-recovery run (GR H0 control)."""

    # Truth cosmology for generating dL(z) (GR).
    h0_true: float
    omega_m0: float = 0.31
    omega_k0: float = 0.0
    z_max: float = 0.62

    # Detectability proxy for generating detections (must match inference alpha proxy choice).
    det_model: Literal["threshold", "snr_binned"] = "threshold"
    snr_binned_nbins: int = 200
    selection_ifar_thresh_yr: float = 1.0

    # Population model used both for sampling events and for inference weighting.
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform"
    pop_z_k: float = 0.0
    pop_z_include_h0_volume_scaling: bool = False
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"] = "powerlaw_peak_q_smooth"
    pop_m1_alpha: float = 2.3
    pop_m_min: float = 5.0
    pop_m_max: float = 80.0
    pop_q_beta: float = 0.0
    pop_m_taper_delta: float = 3.0
    pop_m_peak: float = 35.0
    pop_m_peak_sigma: float = 5.0
    pop_m_peak_frac: float = 0.1

    # Injection importance weighting used inside alpha (and for sampling from the injection set).
    weight_mode: Literal["none", "inv_sampling_pdf"] = "inv_sampling_pdf"

    # Coordinate convention for the injection `sampling_pdf` mass variables.
    #
    # Our population mass models are parameterized in (m1_source, q=m2/m1). If the injection
    # `sampling_pdf` is in component-mass coordinates (m1_source, m2_source), then converting to
    # (m1,q) introduces a Jacobian dm2 = m1 dq, i.e. p_inj(m1,q) = p_inj(m1,m2)*m1. In that case,
    # importance weights that target a population defined in (m1,q) must include an extra 1/m1.
    inj_mass_pdf_coords: Literal["m1m2", "m1q"] = "m1m2"

    # Whether to include p_det(dL, masses) inside the hierarchical PE event term.
    #
    # The standard detected-event hierarchical likelihood uses a selection correction alpha(Λ)
    # and *does not* require p_det in the numerator. This knob exists as an audit/diagnostic
    # option for exploring selection bookkeeping and PE-conditioning assumptions.
    include_pdet_in_event_term: bool = False

    # Gate-2 selection normalization convention (affects only the GR(H0) control / audit).
    # If True, use a xi-style normalization by including an H0^{-3} factor in the selection term.
    selection_include_h0_volume_scaling: bool = False

    # Synthetic PE posterior sampling.
    pe_obs_mode: Literal["truth", "noisy"] = "noisy"
    pe_n_samples: int = 10_000
    pe_synth_mode: Literal["naive_gaussian", "prior_resample", "likelihood_resample"] = "likelihood_resample"
    pe_prior_resample_n_candidates: int = 200_000
    pe_seed: int = 0
    dl_frac_sigma0: float = 0.25
    dl_frac_sigma_floor: float = 0.05
    dl_sigma_mode: Literal["constant", "snr"] = "snr"
    mc_frac_sigma0: float = 0.02
    q_sigma0: float = 0.08

    # Synthetic PE priors (analytic strings parsed by gwtc_pe_priors.parse_gwtc_analytic_prior).
    pe_prior_dL_expr: str = "PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)"
    pe_prior_chirp_mass_expr: str = "UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)"
    pe_prior_mass_ratio_expr: str = "UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)"

    # Event quality control (mirrors Gate-2 GR H0 runner semantics).
    event_qc_mode: Literal["fail", "skip"] = "skip"
    event_min_finite_frac: float = 0.0

    # Importance-sampling stabilization for the hierarchical PE reweighting.
    #
    # This primarily matters when the PE posterior is prior-dominated (very broad likelihood),
    # where p_pop/π_PE weights can become heavy-tailed and degrade Monte Carlo stability.
    importance_smoothing: Literal["none", "truncate", "psis"] = "none"
    importance_truncate_tau: float | None = None


def infer_z_max_for_h0_grid_closed_loop(
    *,
    omega_m0: float,
    omega_k0: float,
    z_gen_max: float,
    h0_true: float,
    h0_eval: float,
    z_cap: float = 5.0,
) -> float:
    """Closed-loop lower bound on z_max to avoid artificial support truncation at high H0.

    If events are generated with truth redshifts z<=z_gen_max under H0_true, their *true* distances obey:

      dL_true(z) = (c/H0_true) f(z),

    where f(z) is the dimensionless FRW distance factor (fixed Ωm0, Ωk0).

    When evaluating a larger H0 on the same distance, the implied redshift is:

      z_eval = f^{-1}((H0_eval/H0_true) f(z)).

    If a Gate‑2 control uses a hard population cutoff z_max < z_eval, it will artificially kill support at
    high H0 and can induce QC-driven biases (e.g. by skipping “partial support” events).

    This helper computes the bound at z=z_gen_max using a temporary inversion cache up to z_cap.
    """
    z_cap = float(z_cap)
    if not (np.isfinite(z_cap) and z_cap > 0.0):
        raise ValueError("z_cap must be finite and positive.")

    z_gen_max = float(z_gen_max)
    if not (np.isfinite(z_gen_max) and z_gen_max > 0.0):
        raise ValueError("z_gen_max must be finite and positive.")

    h0_true = float(h0_true)
    h0_eval = float(h0_eval)
    if not (np.isfinite(h0_true) and h0_true > 0.0 and np.isfinite(h0_eval) and h0_eval > 0.0):
        raise ValueError("Invalid H0 values.")

    cache = _build_lcdm_distance_cache(z_max=float(z_cap), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    z_grid = np.asarray(cache.z_grid, dtype=float)
    f_grid = np.asarray(cache.f_grid, dtype=float)

    f_z = float(np.interp(z_gen_max, z_grid, f_grid, left=np.nan, right=np.nan))
    if not np.isfinite(f_z) or f_z <= 0.0:
        raise ValueError("Failed to evaluate f(z_gen_max); increase z_cap.")

    f_eval = (h0_eval / h0_true) * f_z
    z_eval = float(np.interp(f_eval, f_grid, z_grid, left=np.nan, right=np.nan))
    if not np.isfinite(z_eval):
        return float(z_cap)
    return float(z_eval)


def compute_selection_alpha_h0_grid_for_cfg(
    *,
    injections: O3InjectionSet,
    cfg: InjectionRecoveryConfig,
    h0_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute selection alpha(H0) for the given config on an H0 grid.

    This is intentionally separated so suite runners can compute alpha(H0) **once** and reuse it
    across many injection-recovery replicates.
    """
    h0_grid = np.asarray(h0_grid, dtype=float)
    dist_cache = _build_lcdm_distance_cache(z_max=float(cfg.z_max), omega_m0=float(cfg.omega_m0), omega_k0=float(cfg.omega_k0))  # noqa: SLF001
    alpha, meta = _alpha_h0_grid_from_injections(
        injections=injections,
        H0_grid=h0_grid,
        dist_cache=dist_cache,
        z_max=float(cfg.z_max),
        det_model=str(cfg.det_model),  # type: ignore[arg-type]
        snr_threshold=None,
        snr_binned_nbins=int(cfg.snr_binned_nbins),
        weight_mode=str(cfg.weight_mode),  # type: ignore[arg-type]
        pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(cfg.pop_z_k),
        pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(cfg.pop_m1_alpha),
        pop_m_min=float(cfg.pop_m_min),
        pop_m_max=float(cfg.pop_m_max),
        pop_q_beta=float(cfg.pop_q_beta),
        pop_m_taper_delta=float(cfg.pop_m_taper_delta),
        pop_m_peak=float(cfg.pop_m_peak),
        pop_m_peak_sigma=float(cfg.pop_m_peak_sigma),
        pop_m_peak_frac=float(cfg.pop_m_peak_frac),
        inj_mass_pdf_coords=str(cfg.inj_mass_pdf_coords),  # type: ignore[arg-type]
    )
    return np.asarray(alpha, dtype=float), dict(meta)


def _h0_grid_posterior_from_logL_rel(H0_grid: np.ndarray, logL_rel: np.ndarray) -> dict[str, Any]:
    """Compute posterior + summary from a relative log-likelihood on an H0 grid."""
    H0_grid = np.asarray(H0_grid, dtype=float)
    logL_rel = np.asarray(logL_rel, dtype=float)
    if H0_grid.shape != logL_rel.shape:
        raise ValueError("H0_grid and logL_rel must have the same shape.")

    m = np.isfinite(logL_rel)
    if not np.any(m):
        raise ValueError("logL_rel has no finite entries.")
    log_post = logL_rel - float(np.nanmax(logL_rel[m]))
    p = np.exp(np.clip(log_post, -700.0, 50.0))
    p = np.where(m, p, 0.0)
    psum = float(np.sum(p))
    if not (np.isfinite(psum) and psum > 0.0):
        raise ValueError("Posterior normalization failed (non-finite or zero).")
    p = p / psum

    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))

    mean = float(np.sum(p * H0_grid))
    sd = float(np.sqrt(np.sum(p * (H0_grid - mean) ** 2)))
    i_map = int(np.argmax(p))
    H0_map = float(H0_grid[i_map])
    return {
        "H0_map": float(H0_map),
        "H0_map_index": int(i_map),
        "H0_map_at_edge": bool(i_map == 0 or i_map == (H0_grid.size - 1)),
        "logL_H0_rel": [float(x) for x in log_post.tolist()],
        "posterior": [float(x) for x in p.tolist()],
        "summary": {"mean": mean, "sd": sd, "p50": q50, "p16": q16, "p84": q84},
    }


def _chirp_mass_from_m1_m2(m1: float, m2: float) -> float:
    m1 = float(m1)
    m2 = float(m2)
    mt = m1 + m2
    return float((m1 * m2) ** (3.0 / 5.0) / (mt ** (1.0 / 5.0)))


def _pe_sigmas_from_cfg_and_snr(*, cfg: InjectionRecoveryConfig, snr_net_opt_true: float) -> tuple[float, float, float]:
    """Return (sigma_log_dL, sigma_log_Mc_det, sigma_q) for synthetic PE generation."""
    snr = float(max(float(snr_net_opt_true), 1e-6))
    if cfg.dl_sigma_mode == "constant":
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
    """Cheap truncated normal via rejection; OK for modest truncation + moderate size."""
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
        # Fallback: clamp remaining values.
        out[filled:] = np.clip(rng.normal(loc=mu, scale=sigma, size=n - filled).astype(float), lo, hi)
    return out


def _build_snr_binned_pdet(
    *,
    snr: np.ndarray,
    found_ifar: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the simple unweighted p_det curve used in compute_selection_alpha_from_injections."""
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


def generate_synthetic_detected_events_from_injections(
    *,
    injections: O3InjectionSet,
    cfg: InjectionRecoveryConfig,
    n_events: int,
    seed: int,
) -> list[SyntheticEventTruth]:
    """Generate synthetic detected events by sampling the injection set + applying p_det at H0_true.

    This is a *closed-loop* generator: it uses the same injectables (z, masses, snr_fid) and the same
    proxy detectability mapping as alpha(model), evaluated at a chosen GR truth H0.
    """
    n_events = int(n_events)
    if n_events <= 0:
        raise ValueError("n_events must be positive.")
    rng = np.random.default_rng(int(seed))

    # Build injection importance weights consistent with the selection proxy, then sample from the detected
    # population distribution w*p_det at H0_true.
    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr_fid = np.asarray(injections.snr_net_opt, dtype=float)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)
    found = np.asarray(injections.found_ifar, dtype=bool)

    z_hi = float(min(cfg.z_max, float(np.nanmax(z))))
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
    z = z[m]
    dL_fid = dL_fid[m]
    snr_fid = snr_fid[m]
    m1 = m1[m]
    m2 = m2[m]
    found = found[m]

    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    w = w * mw
    if cfg.weight_mode == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite/non-positive values; cannot sample events.")
        w = w / pdf

    if cfg.pop_mass_mode != "none" and cfg.weight_mode == "inv_sampling_pdf":
        if cfg.inj_mass_pdf_coords == "m1m2":
            w = w / np.clip(m1, 1e-300, np.inf)
        elif cfg.inj_mass_pdf_coords == "m1q":
            pass
        else:
            raise ValueError("Unknown inj_mass_pdf_coords (expected 'm1m2' or 'm1q').")

    if cfg.pop_z_mode != "none":
        # Use the same fixed-LCDM approximation as the selection proxy (relative weighting only).
        H0_ref = 67.7
        om0 = 0.31
        c = 299792.458
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        dc = (c / H0_ref) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        dVdz = (c / (H0_ref * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        base = dVdz / (1.0 + z)
        if cfg.pop_z_mode == "comoving_uniform":
            w = w * base
        else:
            w = w * base * (1.0 + z) ** float(cfg.pop_z_k)

    if cfg.pop_mass_mode != "none":
        alpha = float(cfg.pop_m1_alpha)
        mmin = float(cfg.pop_m_min)
        mmax = float(cfg.pop_m_max)
        beta_q = float(cfg.pop_q_beta)
        q = np.clip(m2 / m1, 1e-6, 1.0)
        if cfg.pop_mass_mode == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        else:
            # Smooth taper; optionally with a peak in m1.
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

            if cfg.pop_mass_mode == "powerlaw_q_smooth":
                log_mass = log_pl
            elif cfg.pop_mass_mode == "powerlaw_peak_q_smooth":
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
                raise ValueError("All mass weights non-finite for injection recovery.")
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)

    # Drop injections with zero/invalid weights (outside population support, etc.).
    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights; check population/injection weighting configuration.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr_fid = snr_fid[good_w]
        m1 = m1[good_w]
        m2 = m2[good_w]
        found = found[good_w]
        w = w[good_w]

    # Compute proxy detectability p_det at the truth H0, mirroring the selection-alpha construction:
    #   - calibrate the p_det curve using fiducial SNRs
    #   - evaluate p_det at scaled SNRs for the truth distances
    dist_cache = _build_lcdm_distance_cache(z_max=float(cfg.z_max), omega_m0=float(cfg.omega_m0), omega_k0=float(cfg.omega_k0))  # noqa: SLF001
    const = PhysicalConstants()
    dL_true = (const.c_km_s / float(cfg.h0_true)) * dist_cache.f(z)
    dL_true = np.clip(dL_true, 1e-6, np.inf)
    snr_true = snr_fid * (dL_fid / dL_true)

    thresh = float(calibrate_snr_threshold_match_count(snr_net_opt=snr_fid, found_ifar=found))
    if cfg.det_model == "threshold":
        pdet = (snr_true > thresh).astype(float)
    else:
        pdet_edges, pdet_vals = _build_snr_binned_pdet(snr=snr_fid, found_ifar=found, nbins=int(cfg.snr_binned_nbins))
        idx_p = np.clip(np.digitize(snr_true, pdet_edges) - 1, 0, pdet_vals.size - 1)
        pdet = np.asarray(pdet_vals[idx_p], dtype=float)

    w_det = np.clip(w * np.clip(pdet, 0.0, 1.0), 0.0, np.inf)
    good_det = np.isfinite(w_det) & (w_det > 0.0)
    if not np.any(good_det):
        raise ValueError("No injections have positive detected weight w*p_det.")
    if not np.all(good_det):
        z = z[good_det]
        dL_fid = dL_fid[good_det]
        snr_fid = snr_fid[good_det]
        m1 = m1[good_det]
        m2 = m2[good_det]
        dL_true = dL_true[good_det]
        snr_true = snr_true[good_det]
        pdet = pdet[good_det]
        w_det = w_det[good_det]

    prob = w_det / float(np.sum(w_det))
    n_avail = int(prob.size)
    replace = bool(n_events > n_avail)
    idx = rng.choice(n_avail, size=n_events, replace=replace, p=prob)

    out: list[SyntheticEventTruth] = []
    for i, j in enumerate(idx.tolist()):
        zt = float(z[j])
        dL_true_j = float(dL_true[j])
        snr_true_j = float(snr_true[j])
        m1s = float(m1[j])
        m2s = float(m2[j])
        q = float(m2s / m1s)
        mc_src = _chirp_mass_from_m1_m2(m1s, m2s)
        mc_det = float(mc_src * (1.0 + zt))

        if cfg.pe_obs_mode == "truth":
            dL_obs = dL_true_j
            mc_det_obs = mc_det
            q_obs = q
        elif cfg.pe_obs_mode == "noisy":
            dl_log_sigma, mc_log_sigma, q_sigma = _pe_sigmas_from_cfg_and_snr(cfg=cfg, snr_net_opt_true=snr_true_j)
            dL_obs = float(np.exp(rng.normal(loc=np.log(dL_true_j), scale=dl_log_sigma)))
            mc_det_obs = float(np.exp(rng.normal(loc=np.log(mc_det), scale=mc_log_sigma)))
            q_obs = float(_sample_truncated_normal(rng, mu=q, sigma=q_sigma, lo=0.05, hi=1.0, size=1)[0])
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
                mass_ratio_true=q,
                mass_ratio_obs=float(q_obs),
                snr_net_opt_fid=float(snr_fid[j]),
                dL_mpc_fid=float(dL_fid[j]),
                snr_net_opt_true=snr_true_j,
                p_det_true=float(pdet[j]),
            )
        )

    _ = thresh  # Keep in scope for debugging/repro (written in truths via p_det_true).
    return out


def synthesize_pe_posterior_samples(
    *,
    truth: SyntheticEventTruth,
    cfg: InjectionRecoveryConfig,
    rng: np.random.Generator,
) -> GWTCPeHierarchicalSamples:
    """Create synthetic PE posterior samples consistent with posterior ∝ L × π_PE.

    This is intentionally a lightweight stand-in for full PE:
      - choose a simple factorized likelihood in (log dL, log Mc_det, q) around an event-level noisy observation
      - sample from the PE prior π_PE
      - importance-resample candidates with weights ∝ likelihood
    """
    n = int(cfg.pe_n_samples)
    if n < 1000:
        raise ValueError("pe_n_samples must be >= 1000 for hierarchical reweighting stability.")

    # PE priors: parse analytic expressions and evaluate log π on the samples.
    spec_dL, prior_dL = parse_gwtc_analytic_prior(str(cfg.pe_prior_dL_expr))
    spec_mc, prior_mc = parse_gwtc_analytic_prior(str(cfg.pe_prior_chirp_mass_expr))
    spec_q, prior_q = parse_gwtc_analytic_prior(str(cfg.pe_prior_mass_ratio_expr))

    prior_spec = {
        "luminosity_distance": {"expr": spec_dL.expr, "class_name": spec_dL.class_name, "kwargs": spec_dL.kwargs},
        "chirp_mass": {"expr": spec_mc.expr, "class_name": spec_mc.class_name, "kwargs": spec_mc.kwargs},
        "mass_ratio": {"expr": spec_q.expr, "class_name": spec_q.class_name, "kwargs": spec_q.kwargs},
    }

    # Likelihood widths.
    dl_log_sigma, mc_log_sigma, q_sigma = _pe_sigmas_from_cfg_and_snr(cfg=cfg, snr_net_opt_true=float(truth.snr_net_opt_true))

    # Likelihood centers: use the event-level "observed" summaries.
    dL_mu = float(truth.dL_mpc_obs)
    mc_mu = float(truth.chirp_mass_det_obs)
    q_mu = float(truth.mass_ratio_obs)

    def _loglike_std_norm(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        t = (x - float(mu)) / float(sigma)
        return -0.5 * t**2

    if cfg.pe_synth_mode == "naive_gaussian":
        # Backward-compatible: sample directly around the truth and attach π_PE values.
        log_dL = rng.normal(loc=np.log(dL_mu), scale=dl_log_sigma, size=n).astype(float)
        dL_s = np.exp(log_dL)
        log_mc = rng.normal(loc=np.log(mc_mu), scale=mc_log_sigma, size=n).astype(float)
        mc_det = np.exp(log_mc)
        q = _sample_truncated_normal(rng, mu=q_mu, sigma=q_sigma, lo=0.05, hi=1.0, size=n)
    elif cfg.pe_synth_mode == "prior_resample":
        n_cand = int(cfg.pe_prior_resample_n_candidates)
        if n_cand < max(20_000, 2 * n):
            raise ValueError("pe_prior_resample_n_candidates too small for stable resampling.")

        dL_c = prior_dL.sample(rng, size=n_cand)
        mc_c = prior_mc.sample(rng, size=n_cand)
        q_c = prior_q.sample(rng, size=n_cand)

        # Factorized likelihood in (log dL, log Mc_det, q).
        loglike = _loglike_std_norm(np.log(np.clip(dL_c, 1e-300, np.inf)), np.log(dL_mu), dl_log_sigma)
        loglike = loglike + _loglike_std_norm(np.log(np.clip(mc_c, 1e-300, np.inf)), np.log(mc_mu), mc_log_sigma)
        loglike = loglike + _loglike_std_norm(q_c, q_mu, q_sigma)

        m_ok = np.isfinite(loglike)
        if not np.any(m_ok):
            raise ValueError("All synthetic PE candidate likelihood weights are non-finite.")
        loglike = np.where(m_ok, loglike, -np.inf)

        # Resample candidates ∝ likelihood to obtain posterior draws.
        loglike = loglike - float(np.nanmax(loglike[m_ok]))
        w = np.exp(np.clip(loglike, -700.0, 50.0))
        wsum = float(np.sum(w))
        if not (np.isfinite(wsum) and wsum > 0.0):
            raise ValueError("Synthetic PE candidate weights sum to non-positive/non-finite; increase candidates or loosen likelihood widths.")
        prob = w / wsum
        idx = rng.choice(int(prob.size), size=n, replace=True, p=prob)
        dL_s = np.asarray(dL_c[idx], dtype=float)
        mc_det = np.asarray(mc_c[idx], dtype=float)
        q = np.asarray(q_c[idx], dtype=float)
    elif cfg.pe_synth_mode == "likelihood_resample":
        # Efficient alternative for narrow likelihoods:
        #   - draw candidates from the (factorized) likelihood proposal in (log dL, log Mc_det, q)
        #   - importance-resample candidates with weights ∝ π_PE × J, where J corrects the
        #     log-variable proposal to the dL/Mc_det measures:
        #       q(log x) ~ Normal(...)  =>  q(x) ∝ Normal(log x)/x  so L(x)/q(x) ∝ x.
        n_cand = int(cfg.pe_prior_resample_n_candidates)
        if n_cand < max(20_000, 2 * n):
            raise ValueError("pe_prior_resample_n_candidates too small for likelihood_resample.")

        dL_min = float(spec_dL.kwargs.get("minimum", 1e-6))
        dL_max = float(spec_dL.kwargs.get("maximum", np.inf))
        mc_min = float(spec_mc.kwargs.get("minimum", 1e-6))
        mc_max = float(spec_mc.kwargs.get("maximum", np.inf))
        q_min = float(spec_q.kwargs.get("minimum", 0.05))
        q_max = float(spec_q.kwargs.get("maximum", 1.0))

        log_dL_c = rng.normal(loc=np.log(dL_mu), scale=dl_log_sigma, size=n_cand).astype(float)
        dL_c = np.exp(log_dL_c)
        if np.isfinite(dL_min) or np.isfinite(dL_max):
            dL_c = np.clip(dL_c, dL_min, dL_max)
            log_dL_c = np.log(np.clip(dL_c, 1e-300, np.inf))

        log_mc_c = rng.normal(loc=np.log(mc_mu), scale=mc_log_sigma, size=n_cand).astype(float)
        mc_c = np.exp(log_mc_c)
        if np.isfinite(mc_min) or np.isfinite(mc_max):
            mc_c = np.clip(mc_c, mc_min, mc_max)
            log_mc_c = np.log(np.clip(mc_c, 1e-300, np.inf))

        q_c = _sample_truncated_normal(rng, mu=q_mu, sigma=q_sigma, lo=q_min, hi=q_max, size=n_cand)

        # Importance weights for posterior resampling: prior × Jacobian corrections.
        logw = prior_dL.logpdf(dL_c) + prior_mc.logpdf(mc_c) + prior_q.logpdf(q_c) + log_dL_c + log_mc_c
        m_ok = np.isfinite(logw)
        if not np.any(m_ok):
            raise ValueError("All likelihood_resample weights are non-finite; check prior bounds and likelihood widths.")
        logw = np.where(m_ok, logw, -np.inf)
        logw = logw - float(np.nanmax(logw[m_ok]))
        w = np.exp(np.clip(logw, -700.0, 50.0))
        wsum = float(np.sum(w))
        if not (np.isfinite(wsum) and wsum > 0.0):
            raise ValueError("likelihood_resample weights sum to non-positive/non-finite.")
        prob = w / wsum
        idx = rng.choice(int(prob.size), size=n, replace=True, p=prob)
        dL_s = np.asarray(dL_c[idx], dtype=float)
        mc_det = np.asarray(mc_c[idx], dtype=float)
        q = np.asarray(q_c[idx], dtype=float)
    else:
        raise ValueError("Unknown pe_synth_mode.")

    log_pi_dL = prior_dL.logpdf(dL_s)
    log_pi_mc = prior_mc.logpdf(mc_det)
    log_pi_q = prior_q.logpdf(q)
    if not (np.all(np.isfinite(log_pi_dL)) and np.all(np.isfinite(log_pi_mc)) and np.all(np.isfinite(log_pi_q))):
        raise ValueError("Synthetic PE priors produced non-finite log π values; check prior expressions and sample ranges.")

    return GWTCPeHierarchicalSamples(
        file="<synthetic>",
        analysis="SYNTH",
        n_total=n,
        n_used=n,
        dL_mpc=np.asarray(dL_s, dtype=float),
        chirp_mass_det=np.asarray(mc_det, dtype=float),
        mass_ratio=np.asarray(q, dtype=float),
        log_pi_dL=np.asarray(log_pi_dL, dtype=float),
        log_pi_chirp_mass=np.asarray(log_pi_mc, dtype=float),
        log_pi_mass_ratio=np.asarray(log_pi_q, dtype=float),
        prior_spec=prior_spec,
        snr_net_opt_ref=float(truth.snr_net_opt_fid),
        dL_mpc_ref=float(truth.dL_mpc_fid),
    )


def run_injection_recovery_gr_h0(
    *,
    injections: O3InjectionSet,
    cfg: InjectionRecoveryConfig,
    n_events: int,
    h0_grid: np.ndarray,
    seed: int,
    selection_alpha_h0_grid: np.ndarray | None = None,
    selection_alpha_meta: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end synthetic injection-recovery for the GR H0 selection-on control.

    Generates N detected synthetic events from the injection set under H0_true, builds synthetic PE samples
    for each, then runs GR H0 inference using the same selection model (alpha(H0)) and population knobs.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(cfg.pe_seed)]))
    truths = generate_synthetic_detected_events_from_injections(injections=injections, cfg=cfg, n_events=int(n_events), seed=int(seed))
    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = {}
    for t in truths:
        pe_by_event[t.event] = synthesize_pe_posterior_samples(truth=t, cfg=cfg, rng=rng)

    # PE posterior calibration check: percentile of truth within the synthetic PE posterior.
    # Under a correct synthetic PE generator (data ~ likelihood around truth; posterior ∝ L×π_PE),
    # these percentiles should be approximately Uniform[0,1] across injected events.
    def _pp(u: np.ndarray) -> dict[str, Any]:
        u = np.asarray(u, dtype=float)
        u = u[np.isfinite(u)]
        n_u = int(u.size)
        if n_u == 0:
            return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ks_d": float("nan")}
        u = np.clip(u, 0.0, 1.0)
        u_sorted = np.sort(u)
        i = np.arange(1, n_u + 1, dtype=float)
        # One-sample KS distance vs Uniform[0,1].
        d_plus = float(np.max(i / n_u - u_sorted))
        d_minus = float(np.max(u_sorted - (i - 1.0) / n_u))
        return {"n": n_u, "mean": float(np.mean(u_sorted)), "sd": float(np.std(u_sorted, ddof=0)), "ks_d": float(max(d_plus, d_minus))}

    pp_by_event: dict[str, dict[str, float]] = {}
    pp_dL: list[float] = []
    pp_mc: list[float] = []
    pp_q: list[float] = []
    for t in truths:
        pe = pe_by_event[t.event]
        u_dL = float(np.mean(np.asarray(pe.dL_mpc, dtype=float) <= float(t.dL_mpc_true)))
        u_mc = float(np.mean(np.asarray(pe.chirp_mass_det, dtype=float) <= float(t.chirp_mass_det_true)))
        u_q = float(np.mean(np.asarray(pe.mass_ratio, dtype=float) <= float(t.mass_ratio_true)))
        pp_by_event[str(t.event)] = {"u_dL": u_dL, "u_chirp_mass_det": u_mc, "u_mass_ratio": u_q}
        pp_dL.append(u_dL)
        pp_mc.append(u_mc)
        pp_q.append(u_q)

    pp_summary = {
        "dL": _pp(np.asarray(pp_dL, dtype=float)),
        "chirp_mass_det": _pp(np.asarray(pp_mc, dtype=float)),
        "mass_ratio": _pp(np.asarray(pp_q, dtype=float)),
    }

    # Calibrate a p_det(snr) curve on the same filtered injection set used for alpha(H0),
    # but only if the event term explicitly includes p_det.
    det_model_obj = None
    if bool(cfg.include_pdet_in_event_term):
        _z, _dL_fid, _snr, _found, _w = _injection_weights(
            injections,  # type: ignore[arg-type]
            weight_mode=str(cfg.weight_mode),  # type: ignore[arg-type]
            pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
            pop_z_powerlaw_k=float(cfg.pop_z_k),
            pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(cfg.pop_m1_alpha),
            pop_m_min=float(cfg.pop_m_min),
            pop_m_max=float(cfg.pop_m_max),
            pop_q_beta=float(cfg.pop_q_beta),
            pop_m_taper_delta=float(cfg.pop_m_taper_delta),
            pop_m_peak=float(cfg.pop_m_peak),
            pop_m_peak_sigma=float(cfg.pop_m_peak_sigma),
            pop_m_peak_frac=float(cfg.pop_m_peak_frac),
            z_max=float(cfg.z_max),
            inj_mass_pdf_coords=str(cfg.inj_mass_pdf_coords),  # type: ignore[arg-type]
        )
        det_model_obj = _calibrate_detection_model_from_snr_and_found(
            snr_net_opt=_snr,
            found_ifar=_found,
            det_model=str(cfg.det_model),  # type: ignore[arg-type]
            snr_threshold=None,
            snr_binned_nbins=int(cfg.snr_binned_nbins),
        )

    res_off = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event=pe_by_event,
        H0_grid=np.asarray(h0_grid, dtype=float),
        omega_m0=float(cfg.omega_m0),
        omega_k0=float(cfg.omega_k0),
        z_max=float(cfg.z_max),
        cache_dir=None,
        include_pdet_in_event_term=bool(cfg.include_pdet_in_event_term),
        pdet_model=det_model_obj if bool(cfg.include_pdet_in_event_term) else None,
        pop_z_include_h0_volume_scaling=bool(cfg.pop_z_include_h0_volume_scaling),
        importance_smoothing=str(cfg.importance_smoothing),  # type: ignore[arg-type]
        importance_truncate_tau=cfg.importance_truncate_tau,
        injections=None,
        ifar_threshold_yr=float(cfg.selection_ifar_thresh_yr),
        det_model=str(cfg.det_model),  # type: ignore[arg-type]
        snr_threshold=None,
        snr_binned_nbins=int(cfg.snr_binned_nbins),
        weight_mode=str(cfg.weight_mode),  # type: ignore[arg-type]
        pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(cfg.pop_z_k),
        pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(cfg.pop_m1_alpha),
        pop_m_min=float(cfg.pop_m_min),
        pop_m_max=float(cfg.pop_m_max),
        pop_q_beta=float(cfg.pop_q_beta),
        pop_m_taper_delta=float(cfg.pop_m_taper_delta),
        pop_m_peak=float(cfg.pop_m_peak),
        pop_m_peak_sigma=float(cfg.pop_m_peak_sigma),
        pop_m_peak_frac=float(cfg.pop_m_peak_frac),
        event_qc_mode=str(cfg.event_qc_mode),  # type: ignore[arg-type]
        event_min_finite_frac=float(cfg.event_min_finite_frac),
        prior="uniform",
    )

    H0_grid = np.asarray(h0_grid, dtype=float)
    alpha_grid = selection_alpha_h0_grid
    alpha_meta = selection_alpha_meta
    if alpha_grid is None:
        alpha_grid, alpha_meta = compute_selection_alpha_h0_grid_for_cfg(injections=injections, cfg=cfg, h0_grid=H0_grid)
    else:
        alpha_grid = np.asarray(alpha_grid, dtype=float)
        if alpha_grid.shape != H0_grid.shape:
            raise ValueError("selection_alpha_h0_grid must have the same shape as h0_grid.")
        if alpha_meta is None:
            alpha_meta = {"note": "selection_alpha_h0_grid provided without selection_alpha_meta."}
        alpha_meta = dict(alpha_meta)

    log_alpha_grid = np.log(np.clip(alpha_grid, 1e-300, np.inf))
    if bool(cfg.selection_include_h0_volume_scaling):
        # Optional xi-style normalization for the GR(H0) control: include the (c/H0)^3 scaling
        # (dropping the constant c^3). This is audit-only and can be useful when exploring
        # selection/volume bookkeeping pathologies.
        log_alpha_grid = log_alpha_grid - 3.0 * np.log(np.clip(H0_grid, 1e-12, np.inf))
    n_used = int(res_off.get("n_events", 0))
    logL_sum_events_rel = np.asarray(res_off.get("logL_sum_events_rel", []), dtype=float)
    if logL_sum_events_rel.shape != H0_grid.shape:
        raise ValueError("Selection-off result missing/invalid logL_sum_events_rel; realize selection-on from it.")

    post_on = _h0_grid_posterior_from_logL_rel(H0_grid, logL_sum_events_rel - float(n_used) * log_alpha_grid)
    res_on = dict(res_off)
    res_on.update(
        {
            "log_alpha_grid": [float(x) for x in log_alpha_grid.tolist()],
            "logL_H0_rel": list(post_on["logL_H0_rel"]),
            "posterior": list(post_on["posterior"]),
            "H0_map": float(post_on["H0_map"]),
            "H0_map_index": int(post_on["H0_map_index"]),
            "H0_map_at_edge": bool(post_on["H0_map_at_edge"]),
            "summary": dict(post_on["summary"]),
            "selection_alpha": alpha_meta,
            "selection_alpha_grid": [float(x) for x in alpha_grid.tolist()],
            "selection_ifar_threshold_yr": float(cfg.selection_ifar_thresh_yr),
            "gate2_pass": bool((not post_on["H0_map_at_edge"]) and (int(res_off.get("n_events_skipped", 0)) == 0)),
        }
    )

    summary = {
        "h0_true": float(cfg.h0_true),
        "n_events_truth": int(len(truths)),
        "n_events_used_selection_off": int(res_off.get("n_events", -1)),
        "n_events_skipped_selection_off": int(res_off.get("n_events_skipped", -1)),
        "n_events_used_selection_on": int(res_on.get("n_events", -1)),
        "n_events_skipped_selection_on": int(res_on.get("n_events_skipped", -1)),
        "bias_map_selection_on": float(res_on["H0_map"]) - float(cfg.h0_true),
        "bias_p50_selection_on": float(res_on["summary"]["p50"]) - float(cfg.h0_true),
        "selection_off": {"H0_map": float(res_off["H0_map"]), "summary": dict(res_off["summary"])},
        "selection_on": {"H0_map": float(res_on["H0_map"]), "summary": dict(res_on["summary"])},
        "pe_pp": pp_summary,
    }

    out: dict[str, Any] = {
        "manifest": {"seed": int(seed), "config": asdict(cfg), "h0_grid": [float(x) for x in np.asarray(h0_grid, dtype=float).tolist()]},
        "truths": [t.to_jsonable() for t in truths],
        "summary": summary,
        "gr_h0_selection_off": res_off,
        "gr_h0_selection_on": res_on,
        "pe_pp_by_event": pp_by_event,
    }

    if out_dir is not None:
        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "manifest.json").write_text(json.dumps(out["manifest"], indent=2, sort_keys=True) + "\n")
        (out_path / "truths.json").write_text(json.dumps(out["truths"], indent=2, sort_keys=True) + "\n")
        (out_path / "summary.json").write_text(json.dumps(out["summary"], indent=2, sort_keys=True) + "\n")
        (out_path / "gr_h0_selection_off.json").write_text(json.dumps(out["gr_h0_selection_off"], indent=2, sort_keys=True) + "\n")
        (out_path / "gr_h0_selection_on.json").write_text(json.dumps(out["gr_h0_selection_on"], indent=2, sort_keys=True) + "\n")

        # Also write a PE cache compatible with Gate-2 runner for debugging.
        pe_dir = out_path / "pe_cache"
        pe_dir.mkdir(parents=True, exist_ok=True)
        for ev, pe in pe_by_event.items():
            meta = {
                "event": str(ev),
                "pe_file": str(pe.file),
                "pe_analysis": str(pe.analysis),
                "pe_analysis_chosen": str(pe.analysis),
                "n_total": int(pe.n_total),
                "n_used": int(pe.n_used),
                "prior_spec_json": json.dumps(pe.prior_spec, sort_keys=True),
                "mode": "synthetic_injection_recovery",
                "h0_true": float(cfg.h0_true),
            }
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

    return out


def load_injections_for_recovery(path: str | Path, *, ifar_threshold_yr: float) -> O3InjectionSet:
    return load_o3_injections(Path(path).expanduser().resolve(), ifar_threshold_yr=float(ifar_threshold_yr))
