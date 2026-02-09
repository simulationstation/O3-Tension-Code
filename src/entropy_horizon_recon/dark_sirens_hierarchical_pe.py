from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from .gwtc_pe_priors import load_gwtc_pe_analytic_priors, select_gwtc_pe_analysis_with_analytic_priors
from .importance_sampling import smooth_logweights
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class GWTCPeHierarchicalSamples:
    """Minimal PE sample set for hierarchical (population) reweighting.

    Uses GWTC PEDataRelease posterior samples with the associated bilby analytic priors.
    """

    file: str
    analysis: str
    n_total: int
    n_used: int
    # Samples
    dL_mpc: np.ndarray  # (n_samples,)
    chirp_mass_det: np.ndarray  # (n_samples,) Msun (detector-frame)
    mass_ratio: np.ndarray  # (n_samples,) q in (0,1]
    # Precomputed PE prior log-density for the reweighted parameters.
    log_pi_dL: np.ndarray  # (n_samples,)
    log_pi_chirp_mass: np.ndarray  # (n_samples,)
    log_pi_mass_ratio: np.ndarray  # (n_samples,)
    # Prior metadata for transparency/debugging.
    prior_spec: dict[str, dict[str, Any]]
    # Optional detectability proxy metadata for selection-aware hierarchical tests.
    #
    # If set, these define an event-specific "SNR normalization" for a simple proxy:
    #   snr(dL) = (snr_net_opt_ref * dL_mpc_ref) / dL.
    #
    # This is used by the siren audit / injection-recovery controls to evaluate a p_det(snr)
    # curve on PE posterior samples. For real GWTC PE samples these fields are typically unset.
    snr_net_opt_ref: float | None = None
    dL_mpc_ref: float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out = asdict(self)
        # Avoid dumping the full arrays.
        for k in ("dL_mpc", "chirp_mass_det", "mass_ratio", "log_pi_dL", "log_pi_chirp_mass", "log_pi_mass_ratio"):
            out.pop(k, None)
        return out


@dataclass(frozen=True)
class HierarchicalPeLogLResult:
    """Per-draw hierarchical log-likelihoods plus importance-weight diagnostics."""

    logL_mu: np.ndarray  # (n_draws,)
    logL_gr: np.ndarray  # (n_draws,)
    ess_mu: np.ndarray  # (n_draws,)
    ess_gr: np.ndarray  # (n_draws,)
    n_good_mu: np.ndarray  # (n_draws,)
    n_good_gr: np.ndarray  # (n_draws,)
    n_samples: int

    def summary_jsonable(self) -> dict[str, Any]:
        def _summ(a: np.ndarray) -> dict[str, float]:
            a = np.asarray(a, dtype=float)
            if a.size == 0 or not np.any(np.isfinite(a)):
                return {"min": float("nan"), "p50": float("nan")}
            return {"min": float(np.nanmin(a)), "p50": float(np.nanmedian(a))}

        return {
            "n_samples": int(self.n_samples),
            "logL_mu": _summ(self.logL_mu),
            "logL_gr": _summ(self.logL_gr),
            "ess_mu": _summ(self.ess_mu),
            "ess_gr": _summ(self.ess_gr),
            "n_good_mu": _summ(self.n_good_mu),
            "n_good_gr": _summ(self.n_good_gr),
        }


def _default_analysis_prefer_for_hierarchical(path: Path) -> list[str]:
    # For GWTC-3 `.h5` files, only IMRPhenomXPHM consistently includes analytic priors.
    if path.suffix == ".h5":
        return ["C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"]
    # GWTC-4.0 `.hdf5` convention.
    if path.suffix == ".hdf5":
        return [
            "C00:Mixed",
            "C00:Mixed+XO4a",
            "C00:IMRPhenomXPHM-SpinTaylor",
            "C00:SEOBNRv5PHM",
            "C00:NRSur7dq4",
            "C00:IMRPhenomXO4a",
        ]
    return []


def load_gwtc_pe_hierarchical_samples(
    *,
    path: str | Path,
    analysis: str | None = None,
    analysis_prefer: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> GWTCPeHierarchicalSamples:
    """Load PE samples and the matching analytic priors needed for hierarchical reweighting.

    This loader:
      - chooses an analysis group that provides analytic priors for (dL, chirp_mass, q),
      - loads posterior samples for those parameters,
      - loads the corresponding bilby analytic priors and precomputes log π_PE for each sample.
    """
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")

    require = ["luminosity_distance", "chirp_mass", "mass_ratio"]

    with h5py.File(path, "r") as f:
        if analysis is None:
            prefer = list(analysis_prefer) if analysis_prefer is not None else _default_analysis_prefer_for_hierarchical(path)
            analysis = select_gwtc_pe_analysis_with_analytic_priors(path=path, prefer=prefer, require_parameters=require)
        if analysis not in f:
            keys = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
            raise KeyError(f"{path}: analysis group '{analysis}' not found. Available: {keys}")

        dset = f[analysis]["posterior_samples"]
        dL = np.asarray(dset["luminosity_distance"], dtype=float)
        mc = np.asarray(dset["chirp_mass"], dtype=float)
        q = np.asarray(dset["mass_ratio"], dtype=float)

    m = np.isfinite(dL) & (dL > 0.0) & np.isfinite(mc) & (mc > 0.0) & np.isfinite(q) & (q > 0.0) & (q <= 1.0)
    if not np.any(m):
        raise ValueError(f"{path}: no finite samples for (dL, chirp_mass, q) in '{analysis}'.")
    dL = dL[m]
    mc = mc[m]
    q = q[m]

    n_total = int(dL.size)
    if n_total < 1000:
        raise ValueError(f"{path}: too few usable posterior samples in '{analysis}' ({n_total}).")

    if max_samples is not None and dL.size > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(dL.size, size=int(max_samples), replace=False)
        dL = dL[idx]
        mc = mc[idx]
        q = q[idx]

    pri = load_gwtc_pe_analytic_priors(path=path, analysis=str(analysis), parameters=require)
    spec: dict[str, dict[str, Any]] = {}
    log_pi_dL = np.zeros((dL.size,), dtype=float)
    log_pi_mc = np.zeros((dL.size,), dtype=float)
    log_pi_q = np.zeros((dL.size,), dtype=float)
    for name, (sp, prior) in pri.items():
        spec[name] = sp.to_jsonable()
        if name == "luminosity_distance":
            log_pi_dL = prior.logpdf(dL)
        elif name == "chirp_mass":
            log_pi_mc = prior.logpdf(mc)
        elif name == "mass_ratio":
            log_pi_q = prior.logpdf(q)
        else:  # pragma: no cover
            raise RuntimeError("Internal error: unexpected prior name.")

    if not np.all(np.isfinite(log_pi_dL)) or not np.all(np.isfinite(log_pi_mc)) or not np.all(np.isfinite(log_pi_q)):
        raise ValueError(f"{path}: non-finite log π_PE encountered; check prior parsing for '{analysis}'.")

    return GWTCPeHierarchicalSamples(
        file=str(path),
        analysis=str(analysis),
        n_total=n_total,
        n_used=int(dL.size),
        dL_mpc=dL,
        chirp_mass_det=mc,
        mass_ratio=q,
        log_pi_dL=log_pi_dL,
        log_pi_chirp_mass=log_pi_mc,
        log_pi_mass_ratio=log_pi_q,
        prior_spec=spec,
    )


def _log_pop_weight_z_fixed_lcdm(
    z: np.ndarray,
    *,
    mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    k: float,
) -> np.ndarray:
    """Unnormalized log population weight for redshift (fixed LCDM; matches selection proxy)."""
    z = np.asarray(z, dtype=float)
    if mode == "none":
        return np.zeros_like(z, dtype=float)

    # Match `dark_sirens_selection.compute_selection_alpha_from_injections` (Planck-ish).
    H0 = 67.7  # km/s/Mpc
    om0 = 0.31
    c = 299792.458  # km/s

    zmax = float(np.max(z)) if z.size else 0.0
    zmax = max(zmax, 1e-6)
    z_grid = np.linspace(0.0, zmax, 5001)
    Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
    invEz = 1.0 / Ez
    # Dc(z) = c/H0 * ∫ dz/E(z) via cumulative trapezoid.
    dc = (c / H0) * np.cumsum(np.concatenate([[0.0], 0.5 * (invEz[1:] + invEz[:-1]) * np.diff(z_grid)]))
    Dc = np.interp(z, z_grid, dc)
    Ez_z = np.interp(z, z_grid, Ez)

    # dV/dz/dΩ = c/H(z) * D_C^2, with H(z)=H0 E(z).
    dVdz = (c / (H0 * Ez_z)) * (Dc**2)
    base = dVdz / np.clip(1.0 + z, 1e-12, np.inf)  # source-frame time dilation
    if mode == "comoving_uniform":
        w = base
    elif mode == "comoving_powerlaw":
        w = base * np.clip(1.0 + z, 1e-12, np.inf) ** float(k)
    else:  # pragma: no cover
        raise ValueError("Unknown z population mode.")
    return np.log(np.clip(w, 1e-300, np.inf))


def _log_pop_weight_mass_source_powerlaw_q(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
) -> np.ndarray:
    """Unnormalized log population weight for source-frame masses under a simple powerlaw-q model."""
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & (m2 > 0.0)
    good &= (m1 >= float(m_min)) & (m1 <= float(m_max)) & (m2 >= float(m_min)) & (m2 <= m1)
    out = np.full_like(m1, -np.inf, dtype=float)
    if np.any(good):
        out[good] = -float(alpha) * np.log(np.clip(m1[good], 1e-300, np.inf)) + float(q_beta) * np.log(np.clip(q[good], 1e-300, np.inf))
    return out


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _log_pop_weight_mass_source_powerlaw_q_smooth(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
    m_taper_delta: float,
) -> np.ndarray:
    """Powerlaw-q model with smooth tapers at (m_min, m_max) for both component masses."""
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    delta = float(m_taper_delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("m_taper_delta must be finite and > 0 for powerlaw_q_smooth.")

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & np.isfinite(m2) & (m2 > 0.0)
    out = np.full_like(m1, -np.inf, dtype=float)
    if not np.any(good):
        return out

    m1g = m1[good]
    qg = q[good]
    m2g = m2[good]

    t1 = _sigmoid_stable((m1g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m1g) / delta)
    t2 = _sigmoid_stable((m2g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m2g) / delta)
    taper = np.clip(t1 * t2, 1e-300, 1.0)

    out[good] = (
        -float(alpha) * np.log(np.clip(m1g, 1e-300, np.inf))
        + float(q_beta) * np.log(np.clip(qg, 1e-300, np.inf))
        + np.log(taper)
    )
    return out


def _log_pop_weight_mass_source_powerlaw_peak_q_smooth(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
    m_taper_delta: float,
    m_peak: float,
    m_peak_sigma: float,
    m_peak_frac: float,
) -> np.ndarray:
    """Powerlaw+Gaussian-peak model with smooth mass tapers for both component masses."""
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    delta = float(m_taper_delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("m_taper_delta must be finite and > 0 for powerlaw_peak_q_smooth.")

    f_peak = float(m_peak_frac)
    if not (np.isfinite(f_peak) and 0.0 <= f_peak <= 1.0):
        raise ValueError("m_peak_frac must be finite and in [0,1].")

    mp = float(m_peak)
    sig = float(m_peak_sigma)
    if not (np.isfinite(mp) and mp > 0.0 and np.isfinite(sig) and sig > 0.0):
        raise ValueError("m_peak and m_peak_sigma must be finite and positive for powerlaw_peak_q_smooth.")

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & np.isfinite(m2) & (m2 > 0.0)
    out = np.full_like(m1, -np.inf, dtype=float)
    if not np.any(good):
        return out

    m1g = m1[good]
    qg = q[good]
    m2g = m2[good]

    t1 = _sigmoid_stable((m1g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m1g) / delta)
    t2 = _sigmoid_stable((m2g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m2g) / delta)
    taper = np.clip(t1 * t2, 1e-300, 1.0)

    log_q = float(q_beta) * np.log(np.clip(qg, 1e-300, np.inf))
    log_taper = np.log(taper)

    log_pl = -float(alpha) * np.log(np.clip(m1g, 1e-300, np.inf)) + log_q + log_taper
    log_peak = -0.5 * ((m1g - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper

    if f_peak <= 0.0:
        out[good] = log_pl
        return out
    if f_peak >= 1.0:
        out[good] = log_peak
        return out

    out[good] = logsumexp(
        np.stack([np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak], axis=0),
        axis=0,
    )
    return out


def _m1_source_from_chirp_mass_and_q(
    Mc_source: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute primary source-frame mass m1 from (chirp_mass_source, q=m2/m1)."""
    Mc = np.asarray(Mc_source, dtype=float)
    q = np.asarray(q, dtype=float)
    # m1 = Mc * (1+q)^(1/5) / q^(3/5)
    return Mc * np.clip(1.0 + q, 1e-12, np.inf) ** (1.0 / 5.0) / np.clip(q, 1e-12, np.inf) ** (3.0 / 5.0)


def _ensure_strictly_increasing(x: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError(f"{name} must be 1D with >=3 points.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite.")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return x


def _invert_monotone(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute x(y) for a monotone increasing y_grid(x_grid) using linear interpolation."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid and y_grid must have the same shape.")
    if np.any(np.diff(y_grid) <= 0.0):
        raise ValueError("y_grid must be strictly increasing for inversion.")
    return np.interp(y, y_grid, x_grid, left=np.nan, right=np.nan)


def compute_hierarchical_pe_logL_draws(
    *,
    pe: GWTCPeHierarchicalSamples,
    post: MuForwardPosterior,
    convention: Literal["A", "B"] = "A",
    z_max: float,
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
    importance_smoothing: Literal["none", "truncate", "psis"] = "none",
    importance_truncate_tau: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | HierarchicalPeLogLResult:
    """Compute (logL_mu, logL_gr) per EM posterior draw from PE samples via hierarchical reweighting.

    This implements the standard posterior-sample reweighting identity:

      p(d | hyper) ∝ ⟨ p_pop(theta | hyper) / π_PE(theta) ⟩_{theta ~ p(theta|d)}

    where θ uses PE posterior samples and π_PE is the full PE sampling prior for the reweighted
    parameters (here: dL, chirp_mass, q).

    We treat redshift as the latent variable inferred from each draw's distance–redshift mapping:

      z_s = z(dL_s ; model, draw),
      Mc_source = Mc_det / (1+z_s),

    and include the Jacobian dz/ddL as required when the population prior is expressed in z.
    """
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")
    z_grid_full = _ensure_strictly_increasing(np.asarray(post.z_grid, dtype=float), name="post.z_grid")
    # Avoid z=0 exactly to keep dL(z) strictly positive/monotone for inversion.
    m_z = (z_grid_full > 0.0) & (z_grid_full <= z_max)
    if not np.any(m_z):
        raise ValueError("z_max is below the posterior z_grid minimum.")
    z_grid = np.asarray(z_grid_full[m_z], dtype=float)
    if z_grid.size < 3:
        raise ValueError("Need >=3 z grid points within z_max for inversion.")

    # Model dL(z) curves on this shared z_grid.
    dL_em = predict_dL_em(post, z_eval=z_grid)  # (n_draws, n_z)
    _, R = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)  # (n_draws, n_z)
    dL_mu = dL_em * np.asarray(R, dtype=float)

    dL_s = np.asarray(pe.dL_mpc, dtype=float)
    mc_det = np.asarray(pe.chirp_mass_det, dtype=float)
    q = np.asarray(pe.mass_ratio, dtype=float)
    log_pi_dL = np.asarray(pe.log_pi_dL, dtype=float)
    log_pi_mc = np.asarray(pe.log_pi_chirp_mass, dtype=float)
    log_pi_q = np.asarray(pe.log_pi_mass_ratio, dtype=float)
    if not (
        dL_s.ndim == mc_det.ndim == q.ndim == log_pi_dL.ndim == log_pi_mc.ndim == log_pi_q.ndim == 1
        and dL_s.size == mc_det.size == q.size == log_pi_dL.size == log_pi_mc.size == log_pi_q.size
    ):
        raise ValueError("PE samples must be 1D arrays with matching sizes.")
    n_samp = int(dL_s.size)
    if n_samp < 1000:
        raise ValueError("Too few PE samples for hierarchical reweighting.")

    # Precompute z-prior weights on demand (fixed LCDM, matches selection proxy).
    def _logw_z(z: np.ndarray) -> np.ndarray:
        return _log_pop_weight_z_fixed_lcdm(z, mode=pop_z_mode, k=float(pop_z_k))

    def _logL_for_mapping(dL_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dL_grid = np.asarray(dL_grid, dtype=float)
        if dL_grid.ndim != 2:
            raise ValueError("dL_grid must be (n_draws, n_z).")
        n_draws, n_z = dL_grid.shape
        if n_z != z_grid.size:
            raise ValueError("dL_grid z dimension mismatch.")

        out = np.full((n_draws,), -np.inf, dtype=float)
        ess = np.zeros((n_draws,), dtype=float)
        n_good = np.zeros((n_draws,), dtype=float)
        for j in range(n_draws):
            dL_j = np.asarray(dL_grid[j], dtype=float)
            if np.any(~np.isfinite(dL_j)) or np.any(dL_j <= 0.0) or np.any(np.diff(dL_j) <= 0.0):
                raise ValueError("Non-physical/non-monotone dL(z) encountered; cannot invert.")

            z_samp = _invert_monotone(x_grid=z_grid, y_grid=dL_j, y=dL_s)

            # Compute dz/ddL via ddL/dz on the grid.
            ddLdz = np.gradient(dL_j, z_grid)
            if np.any(~np.isfinite(ddLdz)) or np.any(ddLdz <= 0.0):
                raise ValueError("Non-positive ddL/dz encountered; cannot build dz/ddL Jacobian.")
            ddLdz_s = np.interp(z_samp, z_grid, ddLdz, left=np.nan, right=np.nan)

            good = np.isfinite(z_samp) & (z_samp > 0.0) & (z_samp <= z_max) & np.isfinite(ddLdz_s) & (ddLdz_s > 0.0)
            n_good[j] = float(np.count_nonzero(good))
            if not np.any(good):
                out[j] = -np.inf
                continue

            log_jac = -np.log(ddLdz_s[good])  # dz/ddL
            log_z = _logw_z(z_samp[good])

            # Mass model: convert detector-frame chirp mass to source frame using z_samp.
            q_g = q[good]
            z_g = z_samp[good]
            log_m = np.zeros_like(log_z)
            log_mass_coord_jac = np.zeros_like(log_z)
            log_pi_denom = log_pi_dL[good]
            if pop_mass_mode != "none":
                # When reweighting masses, divide out the PE mass priors and apply a source-frame mass model.
                log_pi_denom = log_pi_denom + log_pi_mc[good] + log_pi_q[good]

                # Convert detector-frame chirp mass to source frame: Mc_src = Mc_det / (1+z).
                Mc_src = mc_det[good] / (1.0 + z_g)
                m1_src = _m1_source_from_chirp_mass_and_q(Mc_source=Mc_src, q=q_g)

                if pop_mass_mode == "powerlaw_q":
                    log_m = _log_pop_weight_mass_source_powerlaw_q(
                        m1_src,
                        q_g,
                        alpha=float(pop_m1_alpha),
                        m_min=float(pop_m_min),
                        m_max=float(pop_m_max),
                        q_beta=float(pop_q_beta),
                    )
                elif pop_mass_mode == "powerlaw_q_smooth":
                    log_m = _log_pop_weight_mass_source_powerlaw_q_smooth(
                        m1_src,
                        q_g,
                        alpha=float(pop_m1_alpha),
                        m_min=float(pop_m_min),
                        m_max=float(pop_m_max),
                        q_beta=float(pop_q_beta),
                        m_taper_delta=float(pop_m_taper_delta),
                    )
                elif pop_mass_mode == "powerlaw_peak_q_smooth":
                    log_m = _log_pop_weight_mass_source_powerlaw_peak_q_smooth(
                        m1_src,
                        q_g,
                        alpha=float(pop_m1_alpha),
                        m_min=float(pop_m_min),
                        m_max=float(pop_m_max),
                        q_beta=float(pop_q_beta),
                        m_taper_delta=float(pop_m_taper_delta),
                        m_peak=float(pop_m_peak),
                        m_peak_sigma=float(pop_m_peak_sigma),
                        m_peak_frac=float(pop_m_peak_frac),
                    )
                else:
                    raise ValueError("Unknown pop_mass_mode.")

                # Coordinate Jacobian to express p(m1_source, q) in terms of (Mc_det, q):
                #   m1_source = Mc_det * (1+q)^(1/5) / q^(3/5) / (1+z)
                # => dm1_source/dMc_det = (1+q)^(1/5) / q^(3/5) / (1+z)
            log_mass_coord_jac = (1.0 / 5.0) * np.log1p(q_g) - (3.0 / 5.0) * np.log(np.clip(q_g, 1e-300, np.inf)) - np.log1p(z_g)

            logw = log_z + log_jac + log_m + log_mass_coord_jac - log_pi_denom
            sm = smooth_logweights(logw, method=importance_smoothing, truncate_tau=importance_truncate_tau)
            # Expectation over PE posterior samples: mean(w) = sum(w)/N.
            log_sum = float(logsumexp(sm.logw))
            out[j] = float(log_sum - np.log(float(n_samp)))

            # Importance-weight effective sample size (ESS) diagnostic:
            #   ESS = (sum u)^2 / sum(u^2), with u=exp(logw). Defined as 0 when all weights are 0.
            ess[j] = float(sm.ess_smooth)
        return out, ess, n_good

    logL_gr_draw, ess_gr, n_good_gr = _logL_for_mapping(dL_em)
    logL_mu_draw, ess_mu, n_good_mu = _logL_for_mapping(dL_mu)

    logL_mu_draw = np.asarray(logL_mu_draw, dtype=float)
    logL_gr_draw = np.asarray(logL_gr_draw, dtype=float)
    if not return_diagnostics:
        return logL_mu_draw, logL_gr_draw

    return HierarchicalPeLogLResult(
        logL_mu=logL_mu_draw,
        logL_gr=logL_gr_draw,
        ess_mu=np.asarray(ess_mu, dtype=float),
        ess_gr=np.asarray(ess_gr, dtype=float),
        n_good_mu=np.asarray(n_good_mu, dtype=float),
        n_good_gr=np.asarray(n_good_gr, dtype=float),
        n_samples=int(n_samp),
    )
