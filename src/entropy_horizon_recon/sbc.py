from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chisquare, kstest

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid
from .forward_model import ForwardModel, make_spline
from .inversion import infer_logmu_forward
from .likelihoods import BaoLogLike


@dataclass(frozen=True)
class SBCPriorTruth:
    logmu_knots: np.ndarray
    H0: float
    omega_m0: float
    r_d_Mpc: float
    sigma_cc_jit: float
    sigma_sn_jit: float
    sigma_d2: float


def _sample_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(float(lo), float(hi)))


def _sample_halfnormal(rng: np.random.Generator, scale: float) -> float:
    if scale <= 0:
        raise ValueError("HalfNormal scale must be positive.")
    return float(np.abs(rng.normal(scale=scale)))


def sample_sbc_prior_truth(
    rng: np.random.Generator,
    *,
    x_knots: np.ndarray,
    omega_m0_prior: tuple[float, float],
    H0_prior: tuple[float, float],
    r_d_prior: tuple[float, float],
    sigma_cc_jit_scale: float,
    sigma_sn_jit_scale: float,
    logmu_knot_scale: float,
    log_sigma_d2_prior: tuple[float, float],
    sigma_d2_scale: float,
    mu_truth_mode: str = "prior",
) -> SBCPriorTruth:
    """Sample a 'prior-truth' parameter draw consistent with infer_logmu_forward priors.

    This samples:
      - H0, Ωm0, r_d from the same uniform priors
      - σ_cc_jit, σ_sn_jit, σ_d2 from the same half-normal priors
      - logμ knots from the implied Gaussian prior combining:
          (i) weak amplitude regularization around BH (logμ=0), and
          (ii) curvature (2nd-difference) smoothness prior with hyperparameter σ_d2.
    """
    x_knots = np.asarray(x_knots, dtype=float)
    n_mu = int(x_knots.size)
    if n_mu < 4:
        raise ValueError("Need at least 4 μ knots for a curvature prior.")

    H0 = _sample_uniform(rng, H0_prior[0], H0_prior[1])
    omega_m0 = _sample_uniform(rng, omega_m0_prior[0], omega_m0_prior[1])
    r_d_Mpc = _sample_uniform(rng, r_d_prior[0], r_d_prior[1])

    sigma_cc_jit = _sample_halfnormal(rng, sigma_cc_jit_scale)
    sigma_sn_jit = _sample_halfnormal(rng, sigma_sn_jit_scale)

    # σ_d2 has a (very wide) log prior truncation in infer_logmu_forward; apply the same bounds.
    sig_d2 = _sample_halfnormal(rng, sigma_d2_scale)
    sig_d2 = float(np.clip(sig_d2, np.exp(float(log_sigma_d2_prior[0])), np.exp(float(log_sigma_d2_prior[1]))))

    dx = np.diff(x_knots)
    x_step = float(np.mean(dx))
    if not np.isfinite(x_step) or x_step <= 0:
        raise ValueError("Invalid x_knots spacing.")

    # Prior for logμ knots: multivariate Gaussian with precision
    #   Q = I / s0^2 + (D2^T D2) / (σ_d2^2 x_step^4)
    # where D2 applies the discrete second difference.
    s0 = float(logmu_knot_scale)
    if s0 <= 0:
        raise ValueError("logmu_knot_scale must be positive.")

    D2 = np.zeros((n_mu - 2, n_mu), dtype=float)
    for i in range(n_mu - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0

    Q = (1.0 / (s0**2)) * np.eye(n_mu) + (D2.T @ D2) / ((sig_d2**2) * (x_step**4))
    # Sample from N(0, Q^{-1}) using Cholesky of Q (precision).
    L = np.linalg.cholesky(Q)
    z = rng.normal(size=n_mu)
    # Solve L y = z, then L^T x = y  => x = Q^{-1/2} z
    y = np.linalg.solve(L, z)
    if mu_truth_mode == "prior":
        logmu_knots = np.linalg.solve(L.T, y)
    elif mu_truth_mode == "bh":
        # BH-null truth: μ(A)=1 => logμ=0 on all knots.
        logmu_knots = np.zeros(n_mu, dtype=float)
    else:
        raise ValueError(f"Unsupported mu_truth_mode: {mu_truth_mode}")

    return SBCPriorTruth(
        logmu_knots=logmu_knots,
        H0=H0,
        omega_m0=omega_m0,
        r_d_Mpc=r_d_Mpc,
        sigma_cc_jit=sigma_cc_jit,
        sigma_sn_jit=sigma_sn_jit,
        sigma_d2=sig_d2,
    )


def _theta_summaries_from_logmu(
    *,
    x_grid: np.ndarray,
    logmu_x: np.ndarray,
    x_pivot: float,
) -> dict[str, float]:
    x_grid = np.asarray(x_grid, dtype=float)
    logmu_x = np.asarray(logmu_x, dtype=float)
    if x_grid.shape != logmu_x.shape:
        raise ValueError("x_grid/logmu_x shape mismatch.")
    mean = float(np.mean(logmu_x))
    slope = float(np.polyfit(x_grid, logmu_x, 1)[0])
    pivot = float(np.interp(float(x_pivot), x_grid, logmu_x))
    return {"logmu_mean": mean, "logmu_slope": slope, "logmu_pivot": pivot}


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Fit y ≈ a + b x with weights w (returns a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("x,y,w shape mismatch")
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("w must be positive and finite")
    X = np.column_stack([np.ones_like(x), x])
    XtW = (X.T * w)
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return float(beta[0]), float(beta[1])


def _scar_ms_from_grid(
    *,
    x_grid: np.ndarray,
    logmu_truth: np.ndarray,
    logmu_draws: np.ndarray,
    weight_mode: str = "variance",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute the paper-style scar statistics (m,s) on an x-grid.

    Notes
    -----
    Production uses logA = log(A) and compares logmu(logA). For M0/M1 (flat horizon mapping),
    logA differs from x=log(A/A0) only by an additive constant; (m,s) are invariant to this shift.

    We deliberately compute weights from the posterior draws (as in compute_departure_stats), then
    evaluate both truth and posterior draws using those weights.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    logmu_truth = np.asarray(logmu_truth, dtype=float)
    draws = np.asarray(logmu_draws, dtype=float)
    if x_grid.ndim != 1 or logmu_truth.shape != x_grid.shape:
        raise ValueError("x_grid/logmu_truth shape mismatch.")
    if draws.ndim != 2 or draws.shape[1] != x_grid.size:
        raise ValueError("logmu_draws must have shape (n_draws, x_grid.size).")
    if weight_mode == "variance":
        var = np.var(draws, axis=0, ddof=1)
        w = 1.0 / np.clip(var, 1e-12, np.inf)
    elif weight_mode == "uniform":
        w = np.ones(x_grid.size, dtype=float)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")
    w = w / np.trapezoid(w, x=x_grid)

    x0 = float(np.average(x_grid, weights=w))
    x = x_grid - x0

    m_true = float(np.trapezoid(logmu_truth * w, x=x_grid))
    m_draw = np.trapezoid(draws * w[None, :], x=x_grid, axis=1)

    _, s_true = _weighted_linear_fit(x, logmu_truth, w)
    s_draw = np.empty(draws.shape[0], dtype=float)
    for i in range(draws.shape[0]):
        _, b = _weighted_linear_fit(x, draws[i], w)
        s_draw[i] = b
    return m_true, float(s_true), np.asarray(m_draw, dtype=float), np.asarray(s_draw, dtype=float)


def _interp_scalar_on_rows(x: float, x_grid: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y(x) for a scalar x when y is shaped (n_draws, n_x)."""
    x_grid = np.asarray(x_grid, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != x_grid.size:
        raise ValueError("y must have shape (n_draws, x_grid.size).")
    x = float(x)
    if x <= float(x_grid[0]):
        return y[:, 0]
    if x >= float(x_grid[-1]):
        return y[:, -1]
    j = int(np.searchsorted(x_grid, x))
    x0 = float(x_grid[j - 1])
    x1 = float(x_grid[j])
    w = (x - x0) / (x1 - x0)
    return (1.0 - w) * y[:, j - 1] + w * y[:, j]


def compute_sbc_ranks(
    *,
    truths: list[dict[str, float]],
    post_draws: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Compute SBC ranks for scalar summaries.

    Parameters
    ----------
    truths:
        List (len N) of dicts mapping summary name -> truth scalar.
    post_draws:
        List (len N) of dicts mapping summary name -> array of posterior draws.

    Returns
    -------
    ranks:
        Dict mapping summary name -> integer ranks array of shape (N,).
    """
    if len(truths) != len(post_draws):
        raise ValueError("truths and post_draws must have the same length.")
    N = len(truths)
    if N == 0:
        raise ValueError("Need at least one replicate.")

    keys = sorted(set(truths[0].keys()))
    for t in truths:
        if sorted(t.keys()) != keys:
            raise ValueError("Truth summaries keys mismatch across replicates.")
    for d in post_draws:
        if sorted(d.keys()) != keys:
            raise ValueError("Posterior summary keys mismatch across replicates.")

    ranks: dict[str, np.ndarray] = {}
    for k in keys:
        r = np.empty(N, dtype=int)
        for i in range(N):
            draws = np.asarray(post_draws[i][k], dtype=float)
            truth = float(truths[i][k])
            r[i] = int(np.sum(draws < truth))
        ranks[k] = r
    return ranks


def uniformity_pvalues(ranks: np.ndarray, *, n_draws: int, n_bins: int = 20) -> dict[str, float]:
    """Compute basic SBC rank-uniformity p-values for a single scalar summary.

    Notes
    -----
    The raw discrete ranks live on {0,1,...,n_draws}. If N (replicates) is not large compared to
    (n_draws+1), a chi-square test on the full discrete histogram will be meaningless due to
    tiny expected counts per bin. We therefore compute chi2 on a *binned* histogram by default.
    """
    r = np.asarray(ranks, dtype=int)
    if r.ndim != 1:
        raise ValueError("ranks must be 1D.")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    if np.any((r < 0) | (r > n_draws)):
        raise ValueError("Rank out of bounds.")

    # Binned chi2 test: partition {0..n_draws} into n_bins contiguous groups.
    # Keep at most (n_draws+1) bins and at least 2.
    n_bins = int(n_bins)
    n_bins = max(2, min(int(n_bins), int(n_draws + 1)))
    parts = np.array_split(np.arange(n_draws + 1), n_bins)
    # Histogram edges are half-open intervals [edge_i, edge_{i+1}), over integer ranks.
    edges = np.array([int(p[0]) for p in parts] + [int(parts[-1][-1]) + 1], dtype=int)
    counts, _ = np.histogram(r, bins=edges)
    widths = np.diff(edges).astype(float)
    expected = float(r.size) * widths / float(n_draws + 1)
    chi2 = chisquare(counts, expected)

    # Approximate KS on normalized ranks in [0,1]; discrete -> conservative.
    u = (r.astype(float) + 0.5) / float(n_draws + 1)
    ks = kstest(u, "uniform")

    return {
        "chi2_p": float(chi2.pvalue),
        "chi2_bins": int(n_bins),
        "ks_p": float(ks.pvalue),
    }


_SBC_PRIOR_CTX: dict | None = None


def _sbc_prior_worker(i: int) -> tuple[int, dict[str, float], dict[str, np.ndarray], dict]:
    """Top-level worker for prior-truth SBC (picklable for multiprocessing)."""
    if _SBC_PRIOR_CTX is None:
        raise RuntimeError("SBC prior-truth context not initialized.")
    ctx = _SBC_PRIOR_CTX

    seed = int(ctx["seed"])
    rng_i = np.random.default_rng(seed + 1000 + int(i))

    truth = sample_sbc_prior_truth(
        rng_i,
        x_knots=ctx["x_knots"],
        omega_m0_prior=ctx["omega_m0_prior"],
        H0_prior=ctx["H0_prior"],
        r_d_prior=ctx["r_d_prior"],
        sigma_cc_jit_scale=ctx["sigma_cc_jit_scale"],
        sigma_sn_jit_scale=ctx["sigma_sn_jit_scale"],
        logmu_knot_scale=ctx["logmu_knot_scale"],
        log_sigma_d2_prior=ctx["log_sigma_d2_prior"],
        sigma_d2_scale=ctx["sigma_d2_scale_truth"],
        mu_truth_mode=str(ctx.get("mu_truth_mode", "prior")),
    )

    fm: ForwardModel = ctx["forward_model"]
    z_grid = ctx["z_grid"]
    constants: PhysicalConstants = ctx["constants"]

    H_true = fm.solve_H_from_logmu_knots(
        z_grid,
        logmu_knots=truth.logmu_knots,
        H0_km_s_Mpc=truth.H0,
        omega_m0=truth.omega_m0,
    )
    bg = build_background_from_H_grid(z_grid, H_true, constants=constants)

    # SN
    sn_z = ctx["sn_z"]
    sn_cov = ctx["sn_cov"]
    Dl = bg.Dl(sn_z)
    mu0 = 5.0 * np.log10(Dl)
    m_true = mu0 + 0.0
    cov_tot = sn_cov + (truth.sigma_sn_jit**2) * np.eye(sn_z.size)
    L = np.linalg.cholesky(cov_tot)
    sn_m = m_true + L @ rng_i.normal(size=sn_z.size)

    # Chronometers
    cc_z = ctx["cc_z"]
    cc_sigma_H = ctx["cc_sigma_H"]
    H_cc_true = bg.H(cc_z)
    sig_eff = np.sqrt(cc_sigma_H**2 + truth.sigma_cc_jit**2)
    cc_H = H_cc_true + sig_eff * rng_i.normal(size=cc_z.size)

    # BAO
    bao_templates = ctx["bao_templates"]
    bao_likes = []
    for bl in bao_templates:
        y_true = bl.predict(bg, r_d_Mpc=truth.r_d_Mpc)
        c, lower = bl.cov_cho
        Lb = c if lower else c.T
        y_obs = y_true + Lb @ rng_i.normal(size=y_true.shape)
        bao_likes.append(
            BaoLogLike.from_arrays(
                dataset=bl.dataset,
                z=bl.z,
                y=y_obs,
                obs=bl.obs,
                cov=bl.cov,
                constants=constants,
            )
        )

    post = infer_logmu_forward(
        z_grid=z_grid,
        x_knots=ctx["x_knots"],
        x_grid=ctx["x_grid"],
        sn_z=sn_z,
        sn_m=sn_m,
        sn_cov=sn_cov,
        cc_z=cc_z,
        cc_H=cc_H,
        cc_sigma_H=cc_sigma_H,
        bao_likes=bao_likes,
        constants=constants,
        sampler_kind=str(ctx.get("sampler_kind", "emcee")),
        pt_ntemps=int(ctx.get("pt_ntemps", 8)),
        pt_tmax=ctx.get("pt_tmax", None),
        n_walkers=int(ctx["n_walkers"]),
        n_steps=int(ctx["n_steps"]),
        n_burn=int(ctx["n_burn"]),
        seed=seed + 2000 + int(i),
        n_processes=1,
        n_draws=int(ctx["n_draws"]),
        progress=False,
        max_rss_mb=float(ctx["max_rss_mb"]) if ctx.get("max_rss_mb") is not None else None,
        omega_m0_prior=ctx["omega_m0_prior"],
        H0_prior=ctx["H0_prior"],
        r_d_prior=ctx["r_d_prior"],
        sigma_cc_jit_scale=float(ctx["sigma_cc_jit_scale"]),
        sigma_sn_jit_scale=float(ctx["sigma_sn_jit_scale"]),
        logmu_knot_scale=float(ctx["logmu_knot_scale"]),
        log_sigma_d2_prior=ctx["log_sigma_d2_prior"],
        sigma_d2_scale=float(ctx["sigma_d2_scale_infer"]),
        debug_log_path=ctx.get("debug_log_path"),
    )

    x_knots = ctx["x_knots"]
    x_grid = ctx["x_grid"]
    V = ctx["V"]
    x_pivot = float(ctx["x_pivot"])

    truth_spline = make_spline(x_knots, truth.logmu_knots)
    logmu_true_x = truth_spline(np.clip(x_grid, x_knots[0], x_knots[-1]))

    t = _theta_summaries_from_logmu(x_grid=x_grid, logmu_x=logmu_true_x, x_pivot=x_pivot)
    # "Paper-style" scar statistics (m,s) on the same grid.
    m_true, s_true, m_draw, s_draw = _scar_ms_from_grid(
        x_grid=x_grid,
        logmu_truth=logmu_true_x,
        logmu_draws=np.asarray(post.logmu_x_samples, dtype=float),
        weight_mode="variance",
    )
    t.update(
        {
            "H0": float(truth.H0),
            "omega_m0": float(truth.omega_m0),
            "r_d_Mpc": float(truth.r_d_Mpc),
            "sigma_cc_jit": float(truth.sigma_cc_jit),
            "sigma_sn_jit": float(truth.sigma_sn_jit),
            "scar_m": float(m_true),
            "scar_s": float(s_true),
        }
    )
    for j in range(V.shape[0]):
        t[f"proj{j}"] = float(V[j] @ logmu_true_x)

    logmu_samps = np.asarray(post.logmu_x_samples, dtype=float)
    draws: dict[str, np.ndarray] = {
        "H0": np.asarray(post.params["H0"], dtype=float),
        "omega_m0": np.asarray(post.params["omega_m0"], dtype=float),
        "r_d_Mpc": np.asarray(post.params["r_d_Mpc"], dtype=float),
        "sigma_cc_jit": np.asarray(post.params["sigma_cc_jit"], dtype=float),
        "sigma_sn_jit": np.asarray(post.params["sigma_sn_jit"], dtype=float),
        "logmu_mean": np.mean(logmu_samps, axis=1),
        "logmu_slope": np.polyfit(x_grid, logmu_samps.T, 1)[0],
        "logmu_pivot": _interp_scalar_on_rows(x_pivot, x_grid, logmu_samps),
        "scar_m": m_draw,
        "scar_s": s_draw,
    }
    for j in range(V.shape[0]):
        draws[f"proj{j}"] = logmu_samps @ V[j]

    meta = {
        "acceptance_fraction_mean": float(post.meta.get("acceptance_fraction_mean", np.nan)),
        "logprob": post.meta.get("logprob", {}),
    }
    return int(i), t, draws, meta


def run_sbc_prior_truth(
    *,
    seed: int,
    N: int,
    z_grid: np.ndarray,
    x_knots: np.ndarray,
    x_grid: np.ndarray,
    sn_z: np.ndarray,
    sn_cov: np.ndarray,
    cc_z: np.ndarray,
    cc_sigma_H: np.ndarray,
    bao_templates: list[BaoLogLike],
    constants: PhysicalConstants,
    # Inference settings (kept explicit to ensure SBC uses the same priors/hyperpriors)
    sampler_kind: str = "emcee",
    pt_ntemps: int = 8,
    pt_tmax: float | None = None,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    n_draws: int,
    n_processes: int = 1,
    max_rss_mb: float | None = None,
    omega_m0_prior: tuple[float, float] = (0.15, 0.5),
    H0_prior: tuple[float, float] = (40.0, 100.0),
    r_d_prior: tuple[float, float] = (120.0, 170.0),
    sigma_cc_jit_scale: float = 10.0,
    sigma_sn_jit_scale: float = 0.05,
    logmu_knot_scale: float = 1.0,
    log_sigma_d2_prior: tuple[float, float] = (-12.0, 3.0),
    sigma_d2_scale: float = 0.23,
    sigma_d2_scale_infer: float | None = None,
    mu_truth_mode: str = "prior",
    rank_bins: int = 20,
    debug_log_path: str | Path | None = None,
    progress: bool = False,
    progress_path: str | Path | None = None,
    progress_every: int = 1,
) -> dict:
    """Run proper prior-truth SBC for the forward μ(A) inference.

    Returns a JSON-serializable dict with ranks, p-values, and metadata.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    z_grid = np.asarray(z_grid, dtype=float)
    x_knots = np.asarray(x_knots, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    sn_z = np.asarray(sn_z, dtype=float)
    sn_cov = np.asarray(sn_cov, dtype=float)
    cc_z = np.asarray(cc_z, dtype=float)
    cc_sigma_H = np.asarray(cc_sigma_H, dtype=float)
    if sn_cov.shape != (sn_z.size, sn_z.size):
        raise ValueError("sn_cov shape mismatch.")
    if cc_z.shape != cc_sigma_H.shape:
        raise ValueError("cc arrays shape mismatch.")

    rng = np.random.default_rng(int(seed))
    fm = ForwardModel(constants=constants, x_knots=x_knots)

    # Choose fixed projection vectors (reproducible) for random-projection ranks.
    proj_rng = np.random.default_rng(int(seed) + 12345)
    n_proj = 3
    V = proj_rng.normal(size=(n_proj, x_grid.size))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    x_pivot = float(np.median(x_grid))

    if n_processes is None or n_processes <= 0:
        n_processes = 1
    # Initialize global context for multiprocessing workers.
    global _SBC_PRIOR_CTX
    _SBC_PRIOR_CTX = {
        "seed": int(seed),
        "constants": constants,
        "z_grid": z_grid,
        "x_knots": x_knots,
        "x_grid": x_grid,
        "sn_z": sn_z,
        "sn_cov": sn_cov,
        "cc_z": cc_z,
        "cc_sigma_H": cc_sigma_H,
        "bao_templates": bao_templates,
        "forward_model": fm,
        "sampler_kind": str(sampler_kind),
        "pt_ntemps": int(pt_ntemps),
        "pt_tmax": float(pt_tmax) if pt_tmax is not None else None,
        "omega_m0_prior": omega_m0_prior,
        "H0_prior": H0_prior,
        "r_d_prior": r_d_prior,
        "sigma_cc_jit_scale": float(sigma_cc_jit_scale),
        "sigma_sn_jit_scale": float(sigma_sn_jit_scale),
        "logmu_knot_scale": float(logmu_knot_scale),
        "log_sigma_d2_prior": log_sigma_d2_prior,
        "sigma_d2_scale_truth": float(sigma_d2_scale),
        "sigma_d2_scale_infer": float(sigma_d2_scale_infer if sigma_d2_scale_infer is not None else sigma_d2_scale),
        "mu_truth_mode": str(mu_truth_mode),
        "n_walkers": int(n_walkers),
        "n_steps": int(n_steps),
        "n_burn": int(n_burn),
        "n_draws": int(n_draws),
        "max_rss_mb": float(max_rss_mb) if max_rss_mb is not None else None,
        "V": V,
        "x_pivot": float(x_pivot),
        "debug_log_path": str(debug_log_path) if debug_log_path is not None else None,
    }

    # Run SBC replicates.
    truths: list[dict[str, float] | None] = [None] * int(N)
    post_draws: list[dict[str, np.ndarray] | None] = [None] * int(N)
    metas: list[dict | None] = [None] * int(N)

    progress = bool(progress)
    progress_every = int(progress_every)
    progress_every = max(1, progress_every)
    progress_path = Path(progress_path) if progress_path is not None else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")
    t0 = time.time()
    done = 0

    def _report_progress(i: int, meta: dict) -> None:
        nonlocal done
        done += 1
        if not progress and progress_path is None:
            return
        elapsed = float(time.time() - t0)
        rate = float(done) / max(elapsed, 1e-9)
        eta = float(int(N) - done) / max(rate, 1e-9)
        acc = float(meta.get("acceptance_fraction_mean", np.nan))

        if progress and (done % progress_every == 0 or done == int(N)):
            print(
                f"[sbc] done={done}/{int(N)} (i={int(i)}) elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m acc={acc:.3f}",
                flush=True,
            )
        if progress_path is not None:
            rec = {
                "done": int(done),
                "N": int(N),
                "i": int(i),
                "elapsed_s": elapsed,
                "eta_s": eta,
                "acceptance_fraction_mean": acc,
            }
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()

    if n_processes > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(processes=int(n_processes), maxtasksperchild=1) as pool:
            for i, t, d, m in pool.imap_unordered(_sbc_prior_worker, list(range(int(N))), chunksize=1):
                truths[int(i)] = t
                post_draws[int(i)] = d
                metas[int(i)] = m
                _report_progress(int(i), m)
    else:
        for i in range(int(N)):
            i, t, d, m = _sbc_prior_worker(int(i))
            truths[int(i)] = t
            post_draws[int(i)] = d
            metas[int(i)] = m
            _report_progress(int(i), m)

    _SBC_PRIOR_CTX = None

    if any(t is None for t in truths) or any(d is None for d in post_draws) or any(m is None for m in metas):
        raise RuntimeError("SBC replicate missing result (worker crash or early exit).")
    truths_f = [t for t in truths if t is not None]
    post_draws_f = [d for d in post_draws if d is not None]
    metas_f = [m for m in metas if m is not None]

    ranks = compute_sbc_ranks(truths=truths_f, post_draws=post_draws_f)
    pvals = {k: uniformity_pvalues(v, n_draws=int(n_draws), n_bins=int(rank_bins)) for k, v in ranks.items()}
    accept = [float(m.get("acceptance_fraction_mean", np.nan)) for m in metas_f]
    logprob_tot = 0
    logprob_bad = 0
    reason_counts: dict[str, int] = {}
    for m in metas_f:
        lp = m.get("logprob", {}) or {}
        if not lp.get("counts_valid", False):
            continue
        tot = int(lp.get("total_calls") or 0)
        bad = int(lp.get("invalid_calls") or 0)
        logprob_tot += tot
        logprob_bad += bad
        for k, v in (lp.get("invalid_reason_counts") or {}).items():
            reason_counts[str(k)] = int(reason_counts.get(str(k), 0)) + int(v)

    # Replicate-level summaries for diagnostics and null-injection false-positive rates.
    rep_keys = ["scar_m", "scar_s", "H0", "omega_m0", "r_d_Mpc"]
    replicates = []
    cov: dict[str, dict[str, float]] = {}
    for k in rep_keys:
        cov[k] = {"cover_68": 0.0, "cover_95": 0.0}
    for i in range(N):
        rep: dict[str, float | int] = {"i": int(i)}
        rep["acceptance_fraction_mean"] = float(metas_f[i].get("acceptance_fraction_mean", np.nan))
        for k in rep_keys:
            if k not in truths_f[i] or k not in post_draws_f[i]:
                continue
            truth_k = float(truths_f[i][k])
            draws_k = np.asarray(post_draws_f[i][k], dtype=float)
            q025, q16, q50, q84, q975 = np.percentile(draws_k, [2.5, 16, 50, 84, 97.5])
            rep[f"{k}_truth"] = truth_k
            rep[f"{k}_post_mean"] = float(np.mean(draws_k))
            rep[f"{k}_post_std"] = float(np.std(draws_k, ddof=1)) if draws_k.size > 1 else 0.0
            rep[f"{k}_post_q025"] = float(q025)
            rep[f"{k}_post_q16"] = float(q16)
            rep[f"{k}_post_q50"] = float(q50)
            rep[f"{k}_post_q84"] = float(q84)
            rep[f"{k}_post_q975"] = float(q975)
            if k in ("scar_m", "scar_s"):
                rep[f"{k}_post_p_gt0"] = float(np.mean(draws_k > 0.0))
                rep[f"{k}_post_p_lt0"] = float(np.mean(draws_k < 0.0))
            cov[k]["cover_68"] += 1.0 if (truth_k >= float(q16) and truth_k <= float(q84)) else 0.0
            cov[k]["cover_95"] += 1.0 if (truth_k >= float(q025) and truth_k <= float(q975)) else 0.0
        replicates.append(rep)
    for k in rep_keys:
        cov[k]["cover_68"] /= float(N)
        cov[k]["cover_95"] /= float(N)

    return {
        "N": int(N),
        "n_draws": int(n_draws),
        "ranks": {k: v.astype(int).tolist() for k, v in ranks.items()},
        "pvalues": pvals,
        "coverage": cov,
        "replicates": replicates,
        "meta": {
            "seed": int(seed),
            "mu_truth_mode": str(mu_truth_mode),
            "rank_bins": int(rank_bins),
            "sampler_kind": str(sampler_kind),
            "pt_ntemps": int(pt_ntemps),
            "pt_tmax": float(pt_tmax) if pt_tmax is not None else None,
            "n_walkers": int(n_walkers),
            "n_steps": int(n_steps),
            "n_burn": int(n_burn),
            "n_processes": int(n_processes),
            "acceptance_fraction_mean_mean": float(np.nanmean(accept)),
            "logprob": {
                "total_calls": int(logprob_tot),
                "invalid_calls": int(logprob_bad),
                "invalid_rate": float(logprob_bad) / float(logprob_tot) if logprob_tot > 0 else None,
                "invalid_reason_counts": reason_counts,
            },
        },
    }
