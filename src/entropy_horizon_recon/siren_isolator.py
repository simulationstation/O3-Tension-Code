from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples, compute_hierarchical_pe_logL_draws
from .gwtc_pe_priors import parse_gwtc_analytic_prior
from .sirens import MuForwardPosterior


ScrambleMode = Literal[
    "none",
    "shuffle_dL",
    "shuffle_mass",
    "shuffle_dL_mass",
    "shuffle_mc",
    "shuffle_q",
    "prior_dL",
]


def stable_int_seed(s: str) -> int:
    """Deterministic 32-bit-ish seed derived from a string."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("logmeanexp expects a 1D array")
    if not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def apply_hierarchical_pe_scramble(
    pe0: GWTCPeHierarchicalSamples,
    *,
    mode: ScrambleMode,
    seed: int,
    tag: str = "",
) -> GWTCPeHierarchicalSamples:
    """Return a scrambled copy of hierarchical PE samples.

    The intent is to preserve the *marginals* of the scrambled coordinates while killing specific
    correlations in the joint posterior, without breaking the required PE prior-division pairing:
    each permuted coordinate is permuted alongside its corresponding log π_PE array.
    """
    mode = str(mode)
    if mode == "none":
        return pe0

    n = int(pe0.dL_mpc.size)
    if n <= 0:
        raise ValueError("pe0 has no samples.")

    rng = np.random.default_rng(stable_int_seed(f"siren_isolator:{int(seed)}:{mode}:{tag}"))

    dL = np.asarray(pe0.dL_mpc, dtype=float)
    mc = np.asarray(pe0.chirp_mass_det, dtype=float)
    q = np.asarray(pe0.mass_ratio, dtype=float)
    log_pi_dL = np.asarray(pe0.log_pi_dL, dtype=float)
    log_pi_mc = np.asarray(pe0.log_pi_chirp_mass, dtype=float)
    log_pi_q = np.asarray(pe0.log_pi_mass_ratio, dtype=float)

    if mode == "prior_dL":
        spec = dict(getattr(pe0, "prior_spec", {}))
        pri = spec.get("luminosity_distance", {})
        expr = str(pri.get("expr", "")).strip()
        if not expr:
            raise ValueError("prior_dL requires pe0.prior_spec['luminosity_distance']['expr'].")
        _, prior = parse_gwtc_analytic_prior(expr)
        dL = prior.sample(rng, size=n)
        log_pi_dL = prior.logpdf(dL)
        return GWTCPeHierarchicalSamples(
            file=str(pe0.file),
            analysis=str(pe0.analysis),
            n_total=int(pe0.n_total),
            n_used=int(pe0.n_used),
            dL_mpc=np.asarray(dL, dtype=float),
            chirp_mass_det=mc,
            mass_ratio=q,
            log_pi_dL=np.asarray(log_pi_dL, dtype=float),
            log_pi_chirp_mass=log_pi_mc,
            log_pi_mass_ratio=log_pi_q,
            prior_spec=dict(pe0.prior_spec),
        )

    if mode in ("shuffle_dL", "shuffle_dL_mass"):
        p = rng.permutation(n)
        dL = dL[p]
        log_pi_dL = log_pi_dL[p]

    if mode in ("shuffle_mass", "shuffle_dL_mass"):
        # Preserve the (Mc, q) pairing but break its correlation with dL.
        p = rng.permutation(n)
        mc = mc[p]
        q = q[p]
        log_pi_mc = log_pi_mc[p]
        log_pi_q = log_pi_q[p]

    if mode == "shuffle_mc":
        p = rng.permutation(n)
        mc = mc[p]
        log_pi_mc = log_pi_mc[p]

    if mode == "shuffle_q":
        p = rng.permutation(n)
        q = q[p]
        log_pi_q = log_pi_q[p]

    return GWTCPeHierarchicalSamples(
        file=str(pe0.file),
        analysis=str(pe0.analysis),
        n_total=int(pe0.n_total),
        n_used=int(pe0.n_used),
        dL_mpc=dL,
        chirp_mass_det=mc,
        mass_ratio=q,
        log_pi_dL=log_pi_dL,
        log_pi_chirp_mass=log_pi_mc,
        log_pi_mass_ratio=log_pi_q,
        prior_spec=dict(pe0.prior_spec),
    )


@dataclass(frozen=True)
class IsolatorEventScore:
    mode: str
    event: str
    n_pe_samples: int
    z_max: float
    lpd_mu_data: float
    lpd_gr_data: float
    delta_lpd_data: float
    lpd_mu: float
    lpd_gr: float
    delta_lpd: float
    lpd_mu_sel: float
    lpd_gr_sel: float
    delta_lpd_sel: float
    pe_weight_ess_mu_p50: float
    pe_weight_ess_gr_p50: float
    pe_weight_ess_mu_min: float
    pe_weight_ess_gr_min: float
    pe_good_frac_mu_p50: float
    pe_good_frac_gr_p50: float

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def score_event_hierarchical_pe(
    *,
    event: str,
    mode_label: str,
    pe: GWTCPeHierarchicalSamples,
    post: MuForwardPosterior,
    z_max: float,
    convention: Literal["A", "B"] = "A",
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "none",
    pop_z_k: float = 0.0,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"] = "none",
    pop_m1_alpha: float = 2.3,
    pop_m_min: float = 5.0,
    pop_m_max: float = 80.0,
    pop_q_beta: float = 0.0,
    pop_m_taper_delta: float = 0.0,
    pop_m_peak: float = 35.0,
    pop_m_peak_sigma: float = 5.0,
    pop_m_peak_frac: float = 0.1,
    log_alpha_mu: np.ndarray | None = None,
    log_alpha_gr: np.ndarray | None = None,
) -> tuple[IsolatorEventScore, np.ndarray, np.ndarray]:
    """Compute per-event hierarchical PE LPD pieces for mu vs GR.

    Returns (event_score, logL_mu_draws, logL_gr_draws).
    """
    res = compute_hierarchical_pe_logL_draws(
        pe=pe,
        post=post,
        convention=convention,
        z_max=float(z_max),
        pop_z_mode=pop_z_mode,
        pop_z_k=float(pop_z_k),
        pop_mass_mode=pop_mass_mode,
        pop_m1_alpha=float(pop_m1_alpha),
        pop_m_min=float(pop_m_min),
        pop_m_max=float(pop_m_max),
        pop_q_beta=float(pop_q_beta),
        pop_m_taper_delta=float(pop_m_taper_delta),
        pop_m_peak=float(pop_m_peak),
        pop_m_peak_sigma=float(pop_m_peak_sigma),
        pop_m_peak_frac=float(pop_m_peak_frac),
        return_diagnostics=True,
    )

    logL_mu = np.asarray(getattr(res, "logL_mu"), dtype=float)
    logL_gr = np.asarray(getattr(res, "logL_gr"), dtype=float)
    if logL_mu.ndim != 1 or logL_gr.ndim != 1 or logL_mu.shape != logL_gr.shape:
        raise ValueError("Hierarchical PE logL arrays must be 1D and have matching shapes.")

    lpd_mu_data = float(logmeanexp(logL_mu))
    lpd_gr_data = float(logmeanexp(logL_gr))

    if log_alpha_mu is not None and log_alpha_gr is not None:
        log_alpha_mu = np.asarray(log_alpha_mu, dtype=float)
        log_alpha_gr = np.asarray(log_alpha_gr, dtype=float)
        if log_alpha_mu.shape != logL_mu.shape or log_alpha_gr.shape != logL_gr.shape:
            raise ValueError("log_alpha arrays must match the shape of logL_* draws.")
        lpd_mu = float(logmeanexp(logL_mu - log_alpha_mu))
        lpd_gr = float(logmeanexp(logL_gr - log_alpha_gr))
    else:
        lpd_mu = float(lpd_mu_data)
        lpd_gr = float(lpd_gr_data)

    ess_mu = np.asarray(getattr(res, "ess_mu"), dtype=float)
    ess_gr = np.asarray(getattr(res, "ess_gr"), dtype=float)
    n_good_mu = np.asarray(getattr(res, "n_good_mu"), dtype=float)
    n_good_gr = np.asarray(getattr(res, "n_good_gr"), dtype=float)
    n_samp = float(getattr(res, "n_samples"))

    ess_mu_p50 = float(np.nanmedian(ess_mu))
    ess_gr_p50 = float(np.nanmedian(ess_gr))
    ess_mu_min = float(np.nanmin(ess_mu))
    ess_gr_min = float(np.nanmin(ess_gr))
    good_mu = float(np.nanmedian(n_good_mu / max(n_samp, 1.0)))
    good_gr = float(np.nanmedian(n_good_gr / max(n_samp, 1.0)))

    delta_lpd_data = float(lpd_mu_data - lpd_gr_data) if np.isfinite(lpd_mu_data) and np.isfinite(lpd_gr_data) else float("nan")
    delta_lpd = float(lpd_mu - lpd_gr) if np.isfinite(lpd_mu) and np.isfinite(lpd_gr) else float("nan")
    lpd_mu_sel = float(lpd_mu - lpd_mu_data) if np.isfinite(lpd_mu) and np.isfinite(lpd_mu_data) else float("nan")
    lpd_gr_sel = float(lpd_gr - lpd_gr_data) if np.isfinite(lpd_gr) and np.isfinite(lpd_gr_data) else float("nan")
    delta_lpd_sel = float(delta_lpd - delta_lpd_data) if np.isfinite(delta_lpd) and np.isfinite(delta_lpd_data) else float("nan")

    score = IsolatorEventScore(
        mode=str(mode_label),
        event=str(event),
        n_pe_samples=int(getattr(pe, "n_used", 0)),
        z_max=float(z_max),
        lpd_mu_data=float(lpd_mu_data),
        lpd_gr_data=float(lpd_gr_data),
        delta_lpd_data=float(delta_lpd_data),
        lpd_mu=float(lpd_mu),
        lpd_gr=float(lpd_gr),
        delta_lpd=float(delta_lpd),
        lpd_mu_sel=float(lpd_mu_sel),
        lpd_gr_sel=float(lpd_gr_sel),
        delta_lpd_sel=float(delta_lpd_sel),
        pe_weight_ess_mu_p50=float(ess_mu_p50),
        pe_weight_ess_gr_p50=float(ess_gr_p50),
        pe_weight_ess_mu_min=float(ess_mu_min),
        pe_weight_ess_gr_min=float(ess_gr_min),
        pe_good_frac_mu_p50=float(good_mu),
        pe_good_frac_gr_p50=float(good_gr),
    )
    return score, logL_mu, logL_gr
