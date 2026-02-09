from __future__ import annotations

import os

# Avoid nested parallelism (BLAS/OpenMP) when using multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.entropy_models import KaniadakisParams, TsallisParams, log_mu_kaniadakis, log_mu_tsallis
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.inversion import infer_logmu_forward, reconstruct_logmu_of_logA
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike
from entropy_horizon_recon.mapping import H_to_area, forward_H_from_muA
from entropy_horizon_recon.proximity import proximity_summary
from entropy_horizon_recon.recon_gp import reconstruct_H_gp
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty

R_D_TRUTH_MPC = 147.0


def _band(samples: np.ndarray, *, level: float = 0.68) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    med = np.median(samples, axis=0)
    q = (1.0 - level) / 2.0
    lo = np.percentile(samples, 100.0 * q, axis=0)
    hi = np.percentile(samples, 100.0 * (1.0 - q), axis=0)
    return med, lo, hi


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _mean_std(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return {"mean": float("nan"), "std": float("nan")}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std}


@dataclass(frozen=True)
class SyntheticTruth:
    z_grid: np.ndarray
    H: np.ndarray
    omega_m0: float
    H0: float
    logA: np.ndarray
    logmu: np.ndarray


_SBC_CTX: dict | None = None


def _apply_cpu_affinity(n_cores: int) -> None:
    """Best-effort CPU affinity limiter (Linux only)."""
    if n_cores is None or n_cores <= 0:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            if hasattr(os, "sched_getaffinity"):
                allowed = sorted(os.sched_getaffinity(0))
                if not allowed:
                    return
                use = min(int(n_cores), len(allowed))
                os.sched_setaffinity(0, set(allowed[:use]))
                return
            total = os.cpu_count() or n_cores
            use = min(int(n_cores), int(total))
            os.sched_setaffinity(0, set(range(use)))
    except Exception:
        # If unsupported / permission denied, continue without affinity limiting.
        return


def _resolve_cpu_cores(requested: int | None) -> int:
    total = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = len(os.sched_getaffinity(0))
            if allowed > 0:
                total = min(int(total), int(allowed))
        except Exception:
            pass
    if requested is None or requested <= 0:
        return int(total)
    return int(min(int(requested), int(total)))


def _resolve_procs(requested: int | None, *, n_walkers: int, cpu_cores: int) -> int:
    if requested is None or requested <= 0:
        req = int(cpu_cores)
    else:
        req = int(requested)
    if cpu_cores and cpu_cores > 0:
        req = min(req, int(cpu_cores))
    if n_walkers and n_walkers > 0:
        req = min(req, int(n_walkers))
    return max(1, int(req))


def _sbc_worker(i: int) -> dict:
    """Multiprocessing-friendly SBC worker (BH truth)."""
    if _SBC_CTX is None:
        raise RuntimeError("SBC context not initialized.")
    ctx = _SBC_CTX
    _apply_cpu_affinity(int(ctx.get("cpu_cores", 0) or 0))
    seed_base = int(ctx["seed_base"])
    seed_i = int(seed_base + 1000 + i)
    rng_i = np.random.default_rng(seed_i)
    s = float(rng_i.uniform(float(ctx["noise_low"]), float(ctx["noise_high"])))

    m_true_bin = ctx["m_true_bin"]
    sn_L_bin = ctx["sn_L_bin"]
    sn_cov_base = ctx["sn_cov_base"]
    cc_z = ctx["cc_z"]
    cc_H_true = ctx["cc_H_true"]
    cc_sigma = ctx["cc_sigma"]
    bao_templates = ctx["bao_templates"]
    bao_diag_cov = bool(ctx.get("bao_diag_cov", False))
    bg = ctx["bg"]
    constants = ctx["constants"]

    # Sample binned SN directly (consistent with inference approximation)
    m_obs_bin_i = m_true_bin + (s * sn_L_bin) @ rng_i.normal(size=m_true_bin.shape)
    sn_cov_i = (s**2) * sn_cov_base

    # Chronometers
    cc_H_i = cc_H_true + (s * cc_sigma) * rng_i.normal(size=cc_z.shape)
    cc_sig_i = s * cc_sigma

    # BAO
    bao_likes_i = []
    for bl_real in bao_templates:
        y_true_i = bl_real.predict(bg, r_d_Mpc=R_D_TRUTH_MPC)
        # SciPy's cho_factor stores the factor in only one triangle; zero the other.
        cho_i, lower_i = bl_real.cov_cho
        L_i = np.tril(cho_i) if lower_i else np.triu(cho_i).T
        y_obs_i = y_true_i + (s * L_i) @ rng_i.normal(size=y_true_i.shape)
        bao_likes_i.append(
            BaoLogLike.from_arrays(
                dataset=bl_real.dataset,
                z=bl_real.z,
                y=y_obs_i,
                obs=bl_real.obs,
                cov=(s**2) * bl_real.cov,
                constants=constants,
                diag_cov=bao_diag_cov,
            )
        )

    infer_kwargs = dict(
        z_grid=ctx["z_grid"],
        x_knots=ctx["x_knots"],
        x_grid=ctx["x_grid"],
        sn_z=ctx["sn_z_bin"],
        sn_m=m_obs_bin_i,
        sn_cov=sn_cov_i,
        cc_z=cc_z,
        cc_H=cc_H_i,
        cc_sigma_H=cc_sig_i,
        bao_likes=bao_likes_i,
        constants=constants,
        n_walkers=int(ctx["n_walkers"]),
        n_steps=int(ctx["n_steps"]),
        n_burn=int(ctx["n_burn"]),
        n_processes=1,
        seed=seed_i,
        n_draws=int(ctx["n_draws"]),
        max_rss_mb=float(ctx["max_rss_mb"]) if ctx.get("max_rss_mb") is not None else None,
        progress=False,
        debug_log_path=ctx.get("debug_log_path"),
    )
    if ctx.get("omega_m0_prior") is not None:
        infer_kwargs["omega_m0_prior"] = (
            float(ctx["omega_m0_prior"][0]),
            float(ctx["omega_m0_prior"][1]),
        )
    if ctx.get("sigma_d2_scale") is not None:
        infer_kwargs["sigma_d2_scale"] = float(ctx["sigma_d2_scale"])
    if ctx.get("logmu_knot_scale") is not None:
        infer_kwargs["logmu_knot_scale"] = float(ctx["logmu_knot_scale"])
    if ctx.get("sigma_cc_jit_scale") is not None:
        infer_kwargs["sigma_cc_jit_scale"] = float(ctx["sigma_cc_jit_scale"])
    if ctx.get("sigma_sn_jit_scale") is not None:
        infer_kwargs["sigma_sn_jit_scale"] = float(ctx["sigma_sn_jit_scale"])
    if ctx.get("r_d_fixed") is not None:
        infer_kwargs["r_d_fixed"] = float(ctx["r_d_fixed"])

    post_i = infer_logmu_forward(**infer_kwargs)

    truth_H = ctx["truth_H"]
    x_grid = ctx["x_grid"]
    x_mask = ctx.get("x_mask", np.ones_like(x_grid, dtype=bool))
    z_mask = ctx.get("z_mask", np.ones_like(truth_H, dtype=bool))
    z_grid = ctx["z_grid"]
    z_eval = np.asarray(z_grid, dtype=float)[z_mask]
    H_truth_eval = np.asarray(truth_H, dtype=float)[z_mask]
    # Optional absolute-area grid for coverage in logμ(logA) space.
    logA_grid = ctx.get("logA_grid", None)
    logA_mask = ctx.get("logA_mask", None)
    out = {}
    out["logprob"] = post_i.meta.get("logprob", {})
    out["acceptance_fraction_mean"] = post_i.meta.get("acceptance_fraction_mean", None)
    out["ess_min"] = post_i.meta.get("ess_min", None)
    out["tau_by"] = post_i.meta.get("tau_by", None)
    for lev in (0.68, 0.95):
        q = (1.0 - lev) / 2.0
        lo = np.percentile(post_i.logmu_x_samples, 100.0 * q, axis=0)
        hi = np.percentile(post_i.logmu_x_samples, 100.0 * (1.0 - q), axis=0)
        out[f"cov_logmu_{int(100*lev)}"] = float(np.mean((0.0 >= lo[x_mask]) & (0.0 <= hi[x_mask])))
        # Simultaneous (max-deviation) credible band using standardized sup-norm.
        center = np.median(post_i.logmu_x_samples, axis=0)
        scale = np.std(post_i.logmu_x_samples, axis=0, ddof=1) + 1e-12
        t = np.max(np.abs((post_i.logmu_x_samples - center) / scale)[:, x_mask], axis=1)
        thr = float(np.quantile(t, lev))
        lo_sim = center - thr * scale
        hi_sim = center + thr * scale
        out[f"covsim_logmu_{int(100*lev)}"] = float(bool(np.all((0.0 >= lo_sim[x_mask]) & (0.0 <= hi_sim[x_mask]))))
        if logA_grid is not None:
            # Transform posterior draws from logμ(x) to logμ(logA) using each draw's inferred H0 (A0 shift).
            H0_s = np.asarray(post_i.H_samples[:, 0], dtype=float)
            logA0_s = np.log(4.0 * np.pi * (constants.c_km_s / H0_s) ** 2)
            xj = np.clip(np.asarray(logA_grid, dtype=float)[None, :] - logA0_s[:, None], x_grid[0], x_grid[-1])
            logmu_logA = np.empty((post_i.logmu_x_samples.shape[0], xj.shape[1]))
            for j in range(post_i.logmu_x_samples.shape[0]):
                logmu_logA[j] = np.interp(xj[j], x_grid, post_i.logmu_x_samples[j])
            loA = np.percentile(logmu_logA, 100.0 * q, axis=0)
            hiA = np.percentile(logmu_logA, 100.0 * (1.0 - q), axis=0)
            mA = logA_mask if logA_mask is not None else np.ones_like(loA, dtype=bool)
            out[f"cov_logmu_logA_{int(100*lev)}"] = float(np.mean((0.0 >= loA[mA]) & (0.0 <= hiA[mA])))
        loH = np.percentile(post_i.H_samples, 100.0 * q, axis=0)
        hiH = np.percentile(post_i.H_samples, 100.0 * (1.0 - q), axis=0)
        out[f"cov_H_{int(100*lev)}"] = float(np.mean((H_truth_eval >= loH[z_mask]) & (H_truth_eval <= hiH[z_mask])))
        if bool(ctx.get("debug_h_eval", False)) and int(i) == 0 and lev == 0.68:
            debug_dir = Path(str(ctx.get("out_dir", "."))) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            lo68 = np.percentile(post_i.H_samples, 16.0, axis=0)[z_mask]
            hi68 = np.percentile(post_i.H_samples, 84.0, axis=0)[z_mask]
            lo95 = np.percentile(post_i.H_samples, 2.5, axis=0)[z_mask]
            hi95 = np.percentile(post_i.H_samples, 97.5, axis=0)[z_mask]
            np.savez_compressed(
                debug_dir / "H_eval_arrays.npz",
                z_eval=z_eval,
                H_truth_eval=H_truth_eval,
                H_p16_eval=lo68,
                H_p84_eval=hi68,
                H_p2_5_eval=lo95,
                H_p97_5_eval=hi95,
            )
            outside68 = int(np.sum((H_truth_eval < lo68) | (H_truth_eval > hi68)))
            outside95 = int(np.sum((H_truth_eval < lo95) | (H_truth_eval > hi95)))
            print(f"DEBUG H coverage r0: outside68={outside68} outside95={outside95} n={int(H_truth_eval.size)}")
        centerH = np.median(post_i.H_samples, axis=0)
        scaleH = np.std(post_i.H_samples, axis=0, ddof=1) + 1e-12
        tH = np.max(np.abs((post_i.H_samples - centerH) / scaleH)[:, z_mask], axis=1)
        thrH = float(np.quantile(tH, lev))
        loH_sim = centerH - thrH * scaleH
        hiH_sim = centerH + thrH * scaleH
        out[f"covsim_H_{int(100*lev)}"] = float(bool(np.all((H_truth_eval >= loH_sim[z_mask]) & (H_truth_eval <= hiH_sim[z_mask]))))
    out["noise_scale"] = s
    out["seed"] = seed_i
    return out


def _coverage_pointwise(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if truth.shape != lo.shape or truth.shape != hi.shape:
        raise ValueError("truth/lo/hi shape mismatch.")
    return float(np.mean((truth >= lo) & (truth <= hi)))


def _dense_domain_zmax(
    z: np.ndarray,
    *,
    z_min: float,
    z_max_cap: float,
    bin_width: float,
    min_per_bin: int,
) -> float:
    """Choose a conservative z_max where SN sampling remains dense."""
    z = np.asarray(z, dtype=float)
    z = z[(z >= z_min) & (z <= z_max_cap)]
    if z.size == 0:
        raise ValueError("No SN redshifts in requested range.")
    edges = np.arange(z_min, z_max_cap + bin_width, bin_width)
    counts, _ = np.histogram(z, bins=edges)
    ok = counts >= min_per_bin
    if not np.any(ok):
        return float(z_min + bin_width)
    last_good = int(np.where(ok)[0].max())
    return float(edges[last_good + 1])


def _plot_band_with_truth(
    x: np.ndarray,
    y_med: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    y_truth: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_med, color="C0", lw=2, label="posterior median")
    ax.fill_between(x, y_lo, y_hi, color="C0", alpha=0.25, linewidth=0, label="credible band")
    ax.plot(x, y_truth, color="k", lw=1.5, ls="--", label="truth")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_truth(
    *,
    model: str,
    z_grid: np.ndarray,
    H0: float,
    omega_m0: float,
    constants: PhysicalConstants,
    seed: int,
) -> SyntheticTruth:
    rng = np.random.default_rng(seed)
    # Reference area at z=0 under H0
    A0 = 4.0 * np.pi * (constants.c_km_s / H0) ** 2
    logA0 = float(np.log(A0))

    if model == "bh":
        def mu_of_A(_A):
            return 1.0
    elif model == "tsallis":
        delta = 1.05
        params = TsallisParams(delta=delta, log_mu0=0.0, logA0=logA0)

        def mu_of_A(A):
            return float(np.exp(log_mu_tsallis(np.array([np.log(A)]), params)[0]))
    elif model == "kaniadakis":
        beta = 0.5
        params = KaniadakisParams(beta_tilde=beta, log_mu0=0.0, A_ref=A0)

        def mu_of_A(A):
            return float(np.exp(log_mu_kaniadakis(np.array([np.log(A)]), params)[0]))
    else:
        raise ValueError("model must be one of: bh, tsallis, kaniadakis")

    H = forward_H_from_muA(
        z_grid,
        mu_of_A=mu_of_A,
        H0_km_s_Mpc=H0,
        omega_m0=omega_m0,
        constants=constants,
    )
    A = 4.0 * np.pi * (constants.c_km_s / H) ** 2
    # Compute logμ on the same grid
    logmu = np.zeros_like(A)
    for i, Ai in enumerate(A):
        logmu[i] = np.log(mu_of_A(float(Ai)))
    return SyntheticTruth(z_grid=z_grid, H=H, omega_m0=omega_m0, H0=H0, logA=np.log(A), logmu=logmu)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic closure tests for the entropy-horizon pipeline.")
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_closure"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="bh", choices=["bh", "tsallis", "kaniadakis"])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use low-cost settings for quick debugging (small chains, fewer μ knots, limited processes).",
    )
    parser.add_argument("--z-max", type=float, default=1.2, help="Upper cap for z-domain selection.")
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--sn-bin-width", type=float, default=0.05)
    parser.add_argument("--sn-min-per-bin", type=int, default=20)
    parser.add_argument("--mu-knots", type=int, default=8)
    parser.add_argument("--mu-grid", type=int, default=120)
    parser.add_argument(
        "--sigma-d2-scale",
        type=float,
        default=0.185,
        help="Half-normal scale for curvature hyperprior (second differences) of logμ(x).",
    )
    parser.add_argument(
        "--logmu-knot-scale",
        type=float,
        default=None,
        help="Gaussian prior scale for logμ knot amplitudes (centered at 0).",
    )
    parser.add_argument(
        "--sigma-cc-jit-scale",
        type=float,
        default=0.5,
        help="Half-normal scale for chronometer jitter (km/s/Mpc).",
    )
    parser.add_argument(
        "--sigma-sn-jit-scale",
        type=float,
        default=0.02,
        help="Half-normal scale for SN-magnitude jitter (mag).",
    )
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--burn", type=int, default=400)
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--procs", type=int, default=0)
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=0,
        help="Limit this run to the first N CPU cores (best-effort, 0=all).",
    )
    parser.add_argument(
        "--omega-m0-prior",
        type=float,
        nargs=2,
        default=(0.2, 0.4),
        metavar=("LOW", "HIGH"),
        help="Uniform prior range for Ωm0 in forward inference (default: model default).",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=1536.0,
        help="Per-process RSS watchdog limit (MB). Set <=0 to disable.",
    )
    parser.add_argument(
        "--skip-derivative-ablation",
        action="store_true",
        help="Skip the expensive derivative-inversion (GP) ablation path.",
    )
    parser.add_argument("--sbc-n", type=int, default=0, help="If >0, run coverage test over N realizations.")
    parser.add_argument("--sbc-procs", type=int, default=0, help="Parallel processes for SBC (0=auto).")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Noise rescaling for the single displayed realization (1.0 = nominal).",
    )
    parser.add_argument(
        "--noise-scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Noise-scale range for SBC draws (uniform). Overrides --noise-scale-low/high.",
    )
    parser.add_argument("--noise-scale-low", type=float, default=0.8, help="(Deprecated) SBC noise-scale low.")
    parser.add_argument("--noise-scale-high", type=float, default=1.2, help="(Deprecated) SBC noise-scale high.")
    parser.add_argument("--gp-kernel", type=str, default="matern32", choices=["rbf", "matern32", "matern52"])
    parser.add_argument("--gp-walkers", type=int, default=64)
    parser.add_argument("--gp-steps", type=int, default=600)
    parser.add_argument("--gp-burn", type=int, default=200)
    parser.add_argument("--disable-sn", action="store_true", help="Disable SN likelihood (use CC+BAO only).")
    parser.add_argument("--disable-cc", action="store_true", help="Disable chronometers (use SN+BAO only).")
    parser.add_argument("--disable-bao", action="store_true", help="Disable BAO likelihood (use SN+CC only).")
    parser.add_argument(
        "--bao-diag-cov",
        action="store_true",
        help="Use diagonal BAO covariance in both synthetic generation and likelihood.",
    )
    parser.add_argument(
        "--debug-h-eval",
        action="store_true",
        help="Write H(z) eval arrays for SBC replicate 0 and print out-of-band counts.",
    )
    parser.add_argument(
        "--debug-bao-cov",
        action="store_true",
        help="Print BAO ordering + covariance diagnostics for the one-shot synthetic realization.",
    )
    # Default: keep the expensive derivative ablation off unless explicitly requested.
    parser.set_defaults(skip_derivative_ablation=True)
    parser.add_argument(
        "--run-derivative-ablation",
        dest="skip_derivative_ablation",
        action="store_false",
        help="Run the expensive derivative-inversion (GP) ablation path.",
    )
    args = parser.parse_args()
    cpu_cores = _resolve_cpu_cores(args.cpu_cores)
    _apply_cpu_affinity(cpu_cores)
    args.cpu_cores = cpu_cores

    use_sn = not args.disable_sn
    use_cc = not args.disable_cc
    use_bao = not args.disable_bao

    if args.noise_scale_range is not None:
        args.noise_scale_low = float(args.noise_scale_range[0])
        args.noise_scale_high = float(args.noise_scale_range[1])

    if args.fast:
        # Forward μ inference
        args.steps = 150
        args.burn = 50
        args.draws = 100
        args.walkers = 24
        args.procs = 1
        # Keep parameter count low so that emcee's n_walkers >= 2*ndim holds.
        args.mu_knots = min(int(args.mu_knots), 6)
        args.mu_grid = min(int(args.mu_grid), 80)
        # Derivative-inversion ablation (GP H(z))
        args.gp_steps = 50
        args.gp_burn = 20
        args.gp_walkers = min(int(args.gp_walkers), 32)
        # SBC processes: avoid oversubscribing
        if args.sbc_n and args.sbc_n > 0:
            args.sbc_procs = min(int(args.sbc_n), max(1, int(args.cpu_cores)))
        # The GP-based derivative ablation is not part of the minimal ladder.
        args.skip_derivative_ablation = True
    else:
        args.procs = _resolve_procs(args.procs, n_walkers=int(args.walkers), cpu_cores=cpu_cores)
        if args.sbc_n and args.sbc_n > 0:
            if args.sbc_procs and args.sbc_procs > 0:
                sbc_req = int(args.sbc_procs)
            else:
                sbc_req = min(int(args.sbc_n), int(cpu_cores))
            args.sbc_procs = max(1, min(int(sbc_req), int(cpu_cores), int(args.sbc_n)))

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = git_head_sha(repo_root=repo_root) or "unknown"
    git_dirty = git_is_dirty(repo_root=repo_root)
    cmd = command_str()
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()
    rng = np.random.default_rng(args.seed)

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = report_paths.out_dir / "debug_invalid_logprob.txt"
    if debug_log_path.exists():
        try:
            debug_log_path.unlink()
        except Exception:
            pass

    # Load real-data redshift sampling + covariances, but replace values with synthetic truth.
    sn_real = load_pantheon_plus(paths=paths, cov_kind="statonly", subset="cosmology", z_column="zHD")
    cc_real = load_chronometers(paths=paths, variant="BC03_all")
    bao12_real = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    desi24_real = load_bao(paths=paths, dataset="desi_2024_bao_all")

    z_min = 0.02
    z_max_cap = float(args.z_max)
    z_max = _dense_domain_zmax(
        sn_real.z,
        z_min=z_min,
        z_max_cap=z_max_cap,
        bin_width=float(args.sn_bin_width),
        min_per_bin=int(args.sn_min_per_bin),
    )
    z_grid = np.linspace(0.0, z_max, args.n_grid)

    truth = build_truth(
        model=args.model,
        z_grid=z_grid,
        H0=70.0,
        omega_m0=0.3,
        constants=constants,
        seed=args.seed,
    )

    bg = build_background_from_H_grid(z_grid, truth.H, constants=constants)

    # Synthetic chronometers at real z points
    if use_cc:
        cc_mask = (cc_real.z >= z_min) & (cc_real.z <= z_max)
        cc_z = cc_real.z[cc_mask]
        cc_sigma = cc_real.sigma_H[cc_mask]
        cc_H_true = bg.H(cc_z)
    else:
        cc_z = np.zeros((0,), dtype=float)
        cc_sigma = np.zeros((0,), dtype=float)
        cc_H_true = np.zeros((0,), dtype=float)
    # For diagnostics/coverage, treat the domain as "anchored" only where we actually have
    # at least one absolute H(z) measurement (chronometers). This avoids counting the
    # near-z=0 region where SN-only information is (by design) insensitive to H0.
    z_support_min = float(z_min)
    if cc_z.size > 0:
        z_support_min = float(max(z_support_min, float(np.min(cc_z))))

    # Synthetic SN magnitudes with full covariance
    M_true = -3.5  # arbitrary intercept
    if use_sn:
        sn_like_real = SNLogLike.from_pantheon(sn_real, z_min=z_min, z_max=z_max)
        Dl = bg.Dl(sn_like_real.z)
        m_true = 5.0 * np.log10(Dl) + M_true
        cho_sn, lower_sn = sn_like_real.cho
        sn_L = np.tril(cho_sn) if lower_sn else np.triu(cho_sn).T
        # Bin SN once to make forward inference cheap (keep full covariance in compression)
        z_edges = np.arange(z_min, z_max + args.sn_bin_width, args.sn_bin_width)
        sn_bin_template = bin_sn_loglike(sn_like_real, z_edges=z_edges, min_per_bin=args.sn_min_per_bin)
        Dl_bin = bg.Dl(sn_bin_template.z)
        m_true_bin = 5.0 * np.log10(Dl_bin) + M_true
        cho_sn_bin, lower_sn_bin = sn_bin_template.cho
        sn_L_bin = np.tril(cho_sn_bin) if lower_sn_bin else np.triu(cho_sn_bin).T
        sn_cov_base = sn_bin_template.cov
        sn_z_bin = sn_bin_template.z
    else:
        sn_like_real = None
        m_true = np.zeros((0,), dtype=float)
        sn_L = np.zeros((0, 0), dtype=float)
        sn_bin_template = None
        m_true_bin = np.zeros((0,), dtype=float)
        sn_L_bin = np.zeros((0, 0), dtype=float)
        sn_cov_base = np.zeros((0, 0), dtype=float)
        sn_z_bin = np.zeros((0,), dtype=float)

    # Synthetic BAO from two datasets (keep full cov)
    bao_templates: list[BaoLogLike] = []
    if use_bao:
        bao_diag_cov = bool(args.bao_diag_cov)
        for dataset, bao_real in [
            ("sdss_dr12_consensus_bao", bao12_real),
            ("desi_2024_bao_all", desi24_real),
        ]:
            try:
                bl_real = BaoLogLike.from_data(
                    bao_real,
                    dataset=dataset,
                    constants=constants,
                    z_min=z_min,
                    z_max=z_max,
                    diag_cov=bao_diag_cov,
                )
            except ValueError as e:
                print(f"Skipping BAO dataset {dataset}: {e}")
                continue
            y_true = bl_real.predict(bg, r_d_Mpc=R_D_TRUTH_MPC)
            bao_templates.append(bl_real)
    else:
        bao_diag_cov = False

    # --- One synthetic realization (for plots + ablation) ---
    noise_scale = float(args.noise_scale)
    if use_cc:
        cc_H = cc_H_true + (noise_scale * cc_sigma) * rng.normal(size=cc_z.shape)
        cc_like = ChronometerLogLike.from_arrays(z=cc_z, H=cc_H, sigma_H=noise_scale * cc_sigma)
        cc_H_obs = cc_like.H
        cc_sig_obs = cc_like.sigma_H
    else:
        cc_like = None
        cc_H_obs = np.zeros((0,), dtype=float)
        cc_sig_obs = np.zeros((0,), dtype=float)

    if use_sn:
        m_obs_bin = m_true_bin + (noise_scale * sn_L_bin) @ rng.normal(size=m_true_bin.shape)
        sn_like_bin = SNLogLike.from_arrays(
            z=sn_bin_template.z,
            m=m_obs_bin,
            cov=(noise_scale**2) * sn_bin_template.cov,
        )
        sn_z_obs = sn_like_bin.z
        sn_m_obs = sn_like_bin.m
        sn_cov_obs = sn_like_bin.cov
    else:
        sn_like_bin = None
        sn_z_obs = np.zeros((0,), dtype=float)
        sn_m_obs = np.zeros((0,), dtype=float)
        sn_cov_obs = np.zeros((0, 0), dtype=float)

    bao_likes = []
    if use_bao:
        for bl_real in bao_templates:
            y_true = bl_real.predict(bg, r_d_Mpc=R_D_TRUTH_MPC)
            cho_b, lower_b = bl_real.cov_cho
            L = np.tril(cho_b) if lower_b else np.triu(cho_b).T
            y_obs = y_true + (noise_scale * L) @ rng.normal(size=y_true.shape)
            bao_likes.append(
                BaoLogLike.from_arrays(
                    dataset=bl_real.dataset,
                    z=bl_real.z,
                    y=y_obs,
                    obs=bl_real.obs,
                    cov=(noise_scale**2) * bl_real.cov,
                    constants=constants,
                    diag_cov=bao_diag_cov,
                )
            )
        if args.debug_bao_cov:
            print("BAO covariance debug (one-shot):")
            for bl in bao_likes:
                y_model_truth = bl.predict(bg, r_d_Mpc=R_D_TRUTH_MPC)
                diag = np.diag(bl.cov)
                print(f"- dataset={bl.dataset} n={len(bl.y)}")
                print(f"  ordering={bl.ordering()}")
                print(f"  diag(C)={diag.tolist()}")
                print(f"  max_abs_offdiag={bl.max_abs_offdiag_cov():.6g}")
                print(f"  chi2_full_at_truth={bl.chi2(y_model_truth):.6g}")
                print(f"  chi2_diag_at_truth={bl.chi2_diag(y_model_truth):.6g}")

    # Choose x = log(A/A0) domain based on a BH baseline (broad enough for z_max).
    H0_guess = 70.0
    omega_m0_guess = 0.3
    H_zmin_guess = H0_guess * np.sqrt(omega_m0_guess * (1.0 + z_support_min) ** 3 + (1.0 - omega_m0_guess))
    x_max_data = float(2.0 * np.log(H0_guess / H_zmin_guess))  # slightly negative (z=z_min)
    H_zmax_guess = H0_guess * np.sqrt(omega_m0_guess * (1.0 + z_max) ** 3 + (1.0 - omega_m0_guess))
    x_min = float(2.0 * np.log(H0_guess / H_zmax_guess))  # negative
    x_knots = np.linspace(1.25 * x_min, 0.0, args.mu_knots)
    x_grid = np.linspace(x_min, 0.0, args.mu_grid)

    # Forward inference of logμ(x) with jitter and smoothness hyperprior.
    infer_kwargs = dict(
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_z=sn_z_obs,
        sn_m=sn_m_obs,
        sn_cov=sn_cov_obs,
        cc_z=cc_z,
        cc_H=cc_H_obs,
        cc_sigma_H=cc_sig_obs,
        bao_likes=bao_likes,
        constants=constants,
        n_walkers=max(2 * (int(args.mu_knots) + 6), int(args.walkers)),
        n_steps=args.steps,
        n_burn=args.burn,
        n_processes=args.procs,
        seed=args.seed,
        n_draws=args.draws,
        max_rss_mb=float(args.max_rss_mb) if args.max_rss_mb is not None else None,
        progress=True,
        debug_log_path=debug_log_path,
    )
    if args.omega_m0_prior is not None:
        infer_kwargs["omega_m0_prior"] = (float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1]))
    if args.sigma_d2_scale is not None:
        infer_kwargs["sigma_d2_scale"] = float(args.sigma_d2_scale)
    if args.logmu_knot_scale is not None:
        infer_kwargs["logmu_knot_scale"] = float(args.logmu_knot_scale)
    if args.sigma_cc_jit_scale is not None:
        infer_kwargs["sigma_cc_jit_scale"] = float(args.sigma_cc_jit_scale)
    if args.sigma_sn_jit_scale is not None:
        infer_kwargs["sigma_sn_jit_scale"] = float(args.sigma_sn_jit_scale)
    if args.model == "bh":
        infer_kwargs["r_d_fixed"] = float(R_D_TRUTH_MPC)
    fwd_post = infer_logmu_forward(**infer_kwargs)

    # Bands and plots (forward model)
    H_med, H_lo68, H_hi68 = _band(fwd_post.H_samples, level=0.68)
    dH = np.diff(fwd_post.H_samples, axis=1)
    mono_tol = 1e-3
    monotone_fraction = float(np.mean(np.all(dH >= -mono_tol, axis=1))) if dH.size else float("nan")
    monotone_median = bool(np.all(np.diff(H_med) >= -mono_tol)) if H_med.size else False
    d2H = np.diff(fwd_post.H_samples, n=2, axis=1)
    roughness = np.mean(np.abs(d2H), axis=1) / (np.mean(np.abs(dH), axis=1) + 1e-12) if dH.size else np.array([np.nan])
    rough_med = float(np.median(roughness))
    rough_p90 = float(np.percentile(roughness, 90)) if roughness.size else float("nan")
    smooth_ok = bool((rough_med < 0.5) and (rough_p90 < 1.0))
    _plot_band_with_truth(
        z_grid,
        H_med,
        H_lo68,
        H_hi68,
        truth.H,
        xlabel="z",
        ylabel="H(z) [km/s/Mpc]",
        title=f"Forward-model H(z) posterior ({args.model})",
        path=report_paths.figures_dir / "Hz_forward.png",
    )

    logmu_med, logmu_lo68, logmu_hi68 = _band(fwd_post.logmu_x_samples, level=0.68)
    logmu_mean_draw = np.mean(fwd_post.logmu_x_samples, axis=1)
    logmu_slope_draw = np.polyfit(x_grid, fwd_post.logmu_x_samples.T, 1)[0]
    post_summary = {
        "logmu_mean": _mean_std(logmu_mean_draw),
        "logmu_slope": _mean_std(logmu_slope_draw),
        "H0": _mean_std(fwd_post.params["H0"]),
        "omega_m0": _mean_std(fwd_post.params["omega_m0"]),
        "sigma_cc_jit": _mean_std(fwd_post.params["sigma_cc_jit"]),
        "sigma_sn_jit": _mean_std(fwd_post.params["sigma_sn_jit"]),
    }
    truth_logmu_x = np.zeros_like(x_grid)
    if args.model != "bh":
        x_true = truth.logA - float(truth.logA[0])
        truth_logmu_x = np.interp(x_grid, x_true[::-1], truth.logmu[::-1])
    _plot_band_with_truth(
        x_grid,
        logmu_med,
        logmu_lo68,
        logmu_hi68,
        truth_logmu_x,
        xlabel="x = log(A/A0)",
        ylabel="log μ",
        title="Forward-model log μ(x) posterior",
        path=report_paths.figures_dir / "logmu_x.png",
    )

    # Convert logμ(x) draws to logμ(logA) on a fixed logA grid using each draw's H0 (A0 shift).
    A_draws = H_to_area(fwd_post.H_samples, constants=constants)
    logA_draws = np.log(A_draws)
    logA_min = float(np.max(np.percentile(logA_draws, 2, axis=1)))
    logA_max = float(np.min(np.percentile(logA_draws, 98, axis=1)))
    logA_grid = np.linspace(logA_min, logA_max, 140)
    logmu_logA = np.empty((fwd_post.logmu_x_samples.shape[0], logA_grid.size))
    for j in range(fwd_post.logmu_x_samples.shape[0]):
        H0_j = float(fwd_post.H_samples[j, 0])
        logA0_j = float(np.log(4.0 * np.pi * (constants.c_km_s / H0_j) ** 2))
        xj = np.clip(logA_grid - logA0_j, x_grid[0], x_grid[-1])
        logmu_logA[j] = np.interp(xj, x_grid, fwd_post.logmu_x_samples[j])

    logmuA_med, logmuA_lo68, logmuA_hi68 = _band(logmu_logA, level=0.68)
    truth_logmuA = np.zeros_like(logA_grid)
    if args.model != "bh":
        truth_logmuA = np.interp(logA_grid, truth.logA[::-1], truth.logmu[::-1])
    _plot_band_with_truth(
        logA_grid,
        logmuA_med,
        logmuA_lo68,
        logmuA_hi68,
        truth_logmuA,
        xlabel="log A",
        ylabel="log μ(A)",
        title="Forward-model log μ(A) posterior",
        path=report_paths.figures_dir / "logmu_logA_forward.png",
    )

    # Coverage over the data-supported domain only (exclude z<z_min, which is unconstrained and prior-dominated).
    x_mask = x_grid <= x_max_data
    z_mask = z_grid >= z_support_min
    z_eval = z_grid[z_mask]
    H_truth_eval = truth.H[z_mask]
    if args.model == "bh":
        np.savez_compressed(
            report_paths.tables_dir / "H_truth_eval.npz",
            z_eval=z_eval,
            H_truth_eval=H_truth_eval,
        )

    # Pointwise coverage within this realization (diagnostic; not frequentist calibration)
    cov68_logmu = _coverage_pointwise(truth_logmu_x[x_mask], logmu_lo68[x_mask], logmu_hi68[x_mask])
    logmu_lo95 = np.percentile(fwd_post.logmu_x_samples, 2.5, axis=0)
    logmu_hi95 = np.percentile(fwd_post.logmu_x_samples, 97.5, axis=0)
    cov95_logmu = _coverage_pointwise(truth_logmu_x[x_mask], logmu_lo95[x_mask], logmu_hi95[x_mask])
    H_lo95 = np.percentile(fwd_post.H_samples, 2.5, axis=0)
    H_hi95 = np.percentile(fwd_post.H_samples, 97.5, axis=0)
    cov68_H = _coverage_pointwise(H_truth_eval, H_lo68[z_mask], H_hi68[z_mask])
    cov95_H = _coverage_pointwise(H_truth_eval, H_lo95[z_mask], H_hi95[z_mask])

    prox = proximity_summary(logA_grid=logA_grid, logmu_samples=logmu_logA)

    # --- Ablation: old derivative inversion (GP H(z) + divide dH^2/dz) ---
    gp_cov68_logmu = np.nan
    try:
        if args.skip_derivative_ablation or not use_sn:
            raise RuntimeError("skipped")
        # Build unbinned SN for GP recon (original code path)
        m_obs = m_true + (noise_scale * sn_L) @ rng.normal(size=m_true.shape)
        sn_like = SNLogLike.from_arrays(
            z=sn_like_real.z,
            m=m_obs,
            cov=(noise_scale**2) * sn_like_real.cov,
        )
        gp_post = reconstruct_H_gp(
            z_knots=np.linspace(0.0, z_max, 16),
            sn_like=sn_like,
            cc_like=cc_like,
            bao_likes=bao_likes,
            constants=constants,
            z_grid=z_grid,
            z_max_background=z_max,
            kernel=args.gp_kernel,
            n_walkers=args.gp_walkers,
            n_steps=args.gp_steps,
            n_burn=args.gp_burn,
            seed=args.seed,
            n_processes=1,
        )
        omega_m0 = np.full(gp_post.H_samples.shape[0], truth.omega_m0)
        mu_post = reconstruct_logmu_of_logA(
            z=z_grid,
            H_samples=gp_post.H_samples,
            dH_dz_samples=gp_post.dH_dz_samples,
            constants=constants,
            omega_m0_samples=omega_m0,
            logA_grid=logA_grid,
        )
        gp_logmu_med, gp_logmu_lo68, gp_logmu_hi68 = _band(mu_post.logmu_samples, level=0.68)
        gp_cov68_logmu = _coverage_pointwise(
            np.interp(logA_grid, truth.logA[::-1], truth.logmu[::-1]),
            gp_logmu_lo68,
            gp_logmu_hi68,
        )

        # Overlay plot: derivative inversion vs forward inversion on logA grid
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(logA_grid, logmuA_med, color="C0", lw=2, label="forward median")
        ax.fill_between(logA_grid, logmuA_lo68, logmuA_hi68, color="C0", alpha=0.25, linewidth=0)
        ax.plot(logA_grid, gp_logmu_med, color="C1", lw=2, label="derivative median")
        ax.fill_between(logA_grid, gp_logmu_lo68, gp_logmu_hi68, color="C1", alpha=0.18, linewidth=0)
        ax.plot(logA_grid, truth_logmuA, color="k", lw=1.5, ls="--", label="truth")
        ax.set(xlabel="log A", ylabel="log μ(A)", title="Ablation: forward vs derivative inversion")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(report_paths.figures_dir / "ablation_forward_vs_derivative.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        if str(e) != "skipped":
            print(f"Derivative-inversion ablation failed: {e}")

    # --- Optional coverage test over repeated synthetic realizations ---
    sbc_summary = {}
    if args.sbc_n and args.sbc_n > 0 and args.model == "bh":
        import multiprocessing as mp

        global _SBC_CTX
        _SBC_CTX = {
            "seed_base": int(args.seed),
            "noise_low": float(args.noise_scale_low),
            "noise_high": float(args.noise_scale_high),
            "z_grid": z_grid,
            "z_mask": z_mask,
            "out_dir": str(report_paths.out_dir),
            "debug_h_eval": bool(args.debug_h_eval),
            "x_knots": x_knots,
            "x_grid": x_grid,
            "x_mask": x_mask,
            "logA_grid": float(truth.logA[0]) + x_grid,
            "logA_mask": x_mask,
            "sn_z_bin": sn_z_bin,
            "m_true_bin": m_true_bin,
            "sn_L_bin": sn_L_bin,
            "sn_cov_base": sn_cov_base,
            "cc_z": cc_z,
            "cc_H_true": cc_H_true,
            "cc_sigma": cc_sigma,
            "bao_templates": bao_templates,
            "bao_diag_cov": bool(args.bao_diag_cov),
            "bg": bg,
            "constants": constants,
            "truth_H": truth.H,
            "cpu_cores": int(args.cpu_cores),
            # Use the requested chain settings (no hidden minimums).
            "n_walkers": max(2 * (args.mu_knots + 6), int(args.walkers)),
            "n_steps": int(args.steps),
            "n_burn": int(min(args.burn, max(0, args.steps - 1))),
            "n_draws": int(args.draws),
            "max_rss_mb": float(args.max_rss_mb) if args.max_rss_mb is not None else None,
            "omega_m0_prior": (
                [float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])]
                if args.omega_m0_prior is not None
                else None
            ),
            "sigma_d2_scale": float(args.sigma_d2_scale) if args.sigma_d2_scale is not None else None,
            "logmu_knot_scale": float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            "sigma_cc_jit_scale": float(args.sigma_cc_jit_scale) if args.sigma_cc_jit_scale is not None else None,
            "sigma_sn_jit_scale": float(args.sigma_sn_jit_scale) if args.sigma_sn_jit_scale is not None else None,
            "debug_log_path": str(debug_log_path),
            "r_d_fixed": float(R_D_TRUTH_MPC),
        }

        n_jobs = int(args.sbc_n)
        if args.sbc_procs:
            n_procs = int(args.sbc_procs)
        else:
            # Safety-first default: serial SBC unless explicitly requested.
            n_procs = 1
        if args.cpu_cores and args.cpu_cores > 0:
            n_procs = min(n_procs, int(args.cpu_cores))
        if n_procs == 1:
            results = [_sbc_worker(i) for i in range(n_jobs)]
        else:
            # Use maxtasksperchild=1 to avoid long-lived worker RSS growth (important for
            # iterative debugging on shared machines).
            with mp.get_context("fork").Pool(processes=n_procs, maxtasksperchild=1) as pool:
                results = pool.map(_sbc_worker, list(range(n_jobs)))

        cov_logmu_68 = float(np.mean([r["cov_logmu_68"] for r in results]))
        cov_logmu_95 = float(np.mean([r["cov_logmu_95"] for r in results]))
        covsim_logmu_68 = float(np.mean([r["covsim_logmu_68"] for r in results]))
        covsim_logmu_95 = float(np.mean([r["covsim_logmu_95"] for r in results]))
        cov_logmu_logA_68 = float(np.mean([r.get("cov_logmu_logA_68", np.nan) for r in results]))
        cov_logmu_logA_95 = float(np.mean([r.get("cov_logmu_logA_95", np.nan) for r in results]))
        cov_H_68 = float(np.mean([r["cov_H_68"] for r in results]))
        cov_H_95 = float(np.mean([r["cov_H_95"] for r in results]))
        covsim_H_68 = float(np.mean([r["covsim_H_68"] for r in results]))
        covsim_H_95 = float(np.mean([r["covsim_H_95"] for r in results]))
        logprob_tot = 0
        logprob_bad = 0
        logprob_rates = []
        acc_fracs = []
        ess_mins = []
        tau_H0 = []
        tau_mean_knot = []
        reason_counts: dict[str, int] = {}
        for r in results:
            lp = r.get("logprob", {}) or {}
            if not lp.get("counts_valid", False):
                continue
            tot = int(lp.get("total_calls") or 0)
            bad = int(lp.get("invalid_calls") or 0)
            logprob_tot += tot
            logprob_bad += bad
            if tot > 0:
                logprob_rates.append(float(bad) / float(tot))
            for k, v in (lp.get("invalid_reason_counts") or {}).items():
                reason_counts[str(k)] = int(reason_counts.get(str(k), 0)) + int(v)
            af = r.get("acceptance_fraction_mean", None)
            try:
                if af is not None and np.isfinite(float(af)):
                    acc_fracs.append(float(af))
            except Exception:
                pass
            em = r.get("ess_min", None)
            try:
                if em is not None and np.isfinite(float(em)):
                    ess_mins.append(float(em))
            except Exception:
                pass
            tb = r.get("tau_by", None) or {}
            try:
                tH0 = tb.get("H0", None)
                if tH0 is not None and np.isfinite(float(tH0)):
                    tau_H0.append(float(tH0))
            except Exception:
                pass
            try:
                tmk = tb.get("mean_knot", None)
                if tmk is not None and np.isfinite(float(tmk)):
                    tau_mean_knot.append(float(tmk))
            except Exception:
                pass

        invalid_p90 = float(np.percentile(logprob_rates, 90)) if logprob_rates else None
        invalid_max = float(np.max(logprob_rates)) if logprob_rates else None
        acc_summary = None
        if acc_fracs:
            acc_summary = {
                "mean": float(np.mean(acc_fracs)),
                "p10": float(np.percentile(acc_fracs, 10)),
                "p50": float(np.percentile(acc_fracs, 50)),
                "p90": float(np.percentile(acc_fracs, 90)),
                "min": float(np.min(acc_fracs)),
                "max": float(np.max(acc_fracs)),
            }
        sbc_summary = {
            "N": n_jobs,
            "noise_scale_range": [float(args.noise_scale_low), float(args.noise_scale_high)],
            "coverage": {
                "logmu_68": cov_logmu_68,
                "logmu_95": cov_logmu_95,
                "logmu_sim_68": covsim_logmu_68,
                "logmu_sim_95": covsim_logmu_95,
                "logmu_logA_68": cov_logmu_logA_68,
                "logmu_logA_95": cov_logmu_logA_95,
                "H_68": cov_H_68,
                "H_95": cov_H_95,
                "H_sim_68": covsim_H_68,
                "H_sim_95": covsim_H_95,
            },
            "domain": {
                "z_min": float(z_support_min),
                "z_max": float(z_max),
                "x_min": float(x_min),
                "x_max_data": float(x_max_data),
            },
            "logprob": {
                "total_calls": int(logprob_tot),
                "invalid_calls": int(logprob_bad),
                "invalid_rate": float(logprob_bad) / float(logprob_tot) if logprob_tot > 0 else None,
                "invalid_rate_per_rep": logprob_rates,
                "invalid_rate_p90": invalid_p90,
                "invalid_rate_max": invalid_max,
                "invalid_reason_counts": reason_counts,
            },
            "acceptance_fraction": acc_summary,
            "autocorr": {
                "ess_min": {
                    "p10": float(np.percentile(ess_mins, 10.0)) if ess_mins else None,
                    "p50": float(np.percentile(ess_mins, 50.0)) if ess_mins else None,
                    "p90": float(np.percentile(ess_mins, 90.0)) if ess_mins else None,
                },
                "tau": {
                    "H0": {
                        "p50": float(np.percentile(tau_H0, 50.0)) if tau_H0 else None,
                        "p90": float(np.percentile(tau_H0, 90.0)) if tau_H0 else None,
                    },
                    "mean_knot": {
                        "p50": float(np.percentile(tau_mean_knot, 50.0)) if tau_mean_knot else None,
                        "p90": float(np.percentile(tau_mean_knot, 90.0)) if tau_mean_knot else None,
                    },
                },
            },
        }
        _SBC_CTX = None

        # Simple coverage plot (nominal vs empirical for 68/95)
        fig, ax = plt.subplots(figsize=(5, 4))
        nominal = np.array([0.68, 0.95])
        emp = np.array([cov_logmu_68, cov_logmu_95])
        ax.plot([0, 1], [0, 1], color="k", lw=1, alpha=0.5)
        ax.plot(nominal, emp, marker="o", color="C0", label="logμ(x)")
        emp_sim = np.array([covsim_logmu_68, covsim_logmu_95])
        ax.plot(nominal, emp_sim, marker="s", color="C0", alpha=0.75, label="logμ(x) simultaneous")
        if np.isfinite(cov_logmu_logA_68) and np.isfinite(cov_logmu_logA_95):
            empA = np.array([cov_logmu_logA_68, cov_logmu_logA_95])
            ax.plot(nominal, empA, marker="o", color="C2", label="logμ(logA)")
        empH = np.array([cov_H_68, cov_H_95])
        ax.plot(nominal, empH, marker="o", color="C1", label="H(z)")
        empH_sim = np.array([covsim_H_68, covsim_H_95])
        ax.plot(nominal, empH_sim, marker="s", color="C1", alpha=0.75, label="H(z) simultaneous")
        ax.set(xlabel="Nominal credible level", ylabel="Empirical coverage", title="Coverage (BH truth)")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(report_paths.figures_dir / "coverage_points.png", dpi=200)
        plt.close(fig)

    summary = {
        "git": {"sha": git_sha, "dirty": git_dirty},
        "command": cmd,
        "truth": {"model": args.model, "H0": truth.H0, "omega_m0": truth.omega_m0},
        "data": {
            "sn_bins": int(sn_z_obs.size),
            "cc_n": int(cc_z.size),
            "bao_n": int(sum(bl.z.size for bl in bao_likes)),
            "z_max": z_max,
            "z_max_cap": z_max_cap,
            "coverage_domain": {
                "z_min": float(z_support_min),
                "z_max": float(z_max),
                "x_min": float(x_min),
                "x_max_data": float(x_max_data),
            },
        },
        "runtime": {"max_rss_mb": float(args.max_rss_mb)},
        "diagnostics_one_run": {
            "cov68_logmu_x": cov68_logmu,
            "cov95_logmu_x": cov95_logmu,
            "cov68_H": cov68_H,
            "cov95_H": cov95_H,
            "derivative_cov68_logmu_logA": gp_cov68_logmu,
            "fwd_meta": fwd_post.meta,
            "logprob": fwd_post.meta.get("logprob", {}),
            "posterior_summary": post_summary,
            "H_monotone_smooth": {
                "monotone_fraction": monotone_fraction,
                "monotone_median": monotone_median,
                "smooth_ok": smooth_ok,
                "roughness_median": rough_med,
                "roughness_p90": rough_p90,
                "monotone_tol": mono_tol,
            },
            "fwd_param_summary": {
                "sigma_cc_jit": {
                    "p16": float(np.percentile(fwd_post.params["sigma_cc_jit"], 16)),
                    "p50": float(np.percentile(fwd_post.params["sigma_cc_jit"], 50)),
                    "p84": float(np.percentile(fwd_post.params["sigma_cc_jit"], 84)),
                },
                "sigma_sn_jit": {
                    "p16": float(np.percentile(fwd_post.params["sigma_sn_jit"], 16)),
                    "p50": float(np.percentile(fwd_post.params["sigma_sn_jit"], 50)),
                    "p84": float(np.percentile(fwd_post.params["sigma_sn_jit"], 84)),
                },
                "sigma_d2": {
                    "p16": float(np.percentile(fwd_post.params["sigma_d2"], 16)),
                    "p50": float(np.percentile(fwd_post.params["sigma_d2"], 50)),
                    "p84": float(np.percentile(fwd_post.params["sigma_d2"], 84)),
                },
            },
        },
        "proximity": prox,
        "sbc": sbc_summary,
    }

    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    (report_paths.tables_dir / "summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")

    md: list[str] = []
    md.append(f"# Synthetic closure test ({args.model})\n")
    md.append(f"_git: {git_sha} (dirty={git_dirty})_\n")
    md.append(f"_command: `{cmd}`_\n")
    md.append("## Setup\n")
    md.append(
        format_table(
            rows=[
                ["seed", args.seed],
                ["z_max_cap", f"{z_max_cap:.2f}"],
                ["z_max_used (density rule)", f"{z_max:.2f}"],
                ["SN bins", int(sn_z_obs.size)],
                ["CC points", int(cc_z.size)],
                ["BAO points", int(sum(bl.z.size for bl in bao_likes))],
            ],
            headers=["item", "value"],
        )
        + "\n"
    )
    md.append("## One-run diagnostics\n")
    md.append(
        format_table(
            rows=[
                ["logμ(x) 68% band coverage (pointwise)", f"{cov68_logmu:.2f}"],
                ["logμ(x) 95% band coverage (pointwise)", f"{cov95_logmu:.2f}"],
                ["H(z) 68% band coverage (pointwise)", f"{cov68_H:.2f}"],
                ["H(z) 95% band coverage (pointwise)", f"{cov95_H:.2f}"],
                ["Derivative inversion logμ(A) 68% coverage (pointwise)", f"{gp_cov68_logmu:.2f}"],
            ],
            headers=["metric", "value"],
        )
        + "\n"
    )
    if sbc_summary:
        md.append("## Frequentist coverage (BH truth)\n")
        cov = sbc_summary["coverage"]
        cov_logA_68 = float(cov.get("logmu_logA_68", np.nan))
        cov_logA_95 = float(cov.get("logmu_logA_95", np.nan))
        logA_68_s = f"{cov_logA_68:.2f}" if np.isfinite(cov_logA_68) else "NA"
        logA_95_s = f"{cov_logA_95:.2f}" if np.isfinite(cov_logA_95) else "NA"
        md.append(
            format_table(
                rows=[
                    [
                        "68%",
                        f"{cov['logmu_68']:.2f}",
                        f"{cov.get('logmu_sim_68', np.nan):.2f}",
                        logA_68_s,
                        f"{cov['H_68']:.2f}",
                        f"{cov.get('H_sim_68', np.nan):.2f}",
                    ],
                    [
                        "95%",
                        f"{cov['logmu_95']:.2f}",
                        f"{cov.get('logmu_sim_95', np.nan):.2f}",
                        logA_95_s,
                        f"{cov['H_95']:.2f}",
                        f"{cov.get('H_sim_95', np.nan):.2f}",
                    ],
                ],
                headers=[
                    "nominal",
                    "logμ(x) pointwise",
                    "logμ(x) simultaneous",
                    "logμ(logA) pointwise",
                    "H(z) pointwise",
                    "H(z) simultaneous",
                ],
            )
            + "\n"
        )
        md.append("### Sampler health\n")
        lp = sbc_summary.get("logprob", {}) or {}
        acc = sbc_summary.get("acceptance_fraction", None)
        inv_rate = lp.get("invalid_rate", None)
        inv_p90 = lp.get("invalid_rate_p90", None)
        inv_max = lp.get("invalid_rate_max", None)
        md.append(
            format_table(
                rows=[
                    ["acceptance_fraction_mean (one-run)", f"{float(fwd_post.meta.get('acceptance_fraction_mean', float('nan'))):.3f}"],
                    [
                        "acceptance_fraction_mean (SBC reps)",
                        (
                            f"mean={acc['mean']:.3f} p10={acc['p10']:.3f} p50={acc['p50']:.3f} p90={acc['p90']:.3f} min={acc['min']:.3f} max={acc['max']:.3f}"
                            if isinstance(acc, dict)
                            else "NA"
                        ),
                    ],
                    [
                        "invalid_logprob_rate (SBC overall)",
                        (
                            f"{float(inv_rate):.4g} (p90={float(inv_p90):.4g}, max={float(inv_max):.4g})"
                            if inv_rate is not None and inv_p90 is not None and inv_max is not None
                            else "NA"
                        ),
                    ],
                    [
                        "invalid_logprob_reasons (top 5)",
                        (
                            ", ".join(
                                f"{k}={v}"
                                for k, v in sorted(
                                    (lp.get("invalid_reason_counts") or {}).items(),
                                    key=lambda kv: kv[1],
                                    reverse=True,
                                )[:5]
                            )
                            or "NA"
                        ),
                    ],
                ],
                headers=["metric", "value"],
            )
            + "\n"
        )
    md.append("## Plots\n")
    md.append("![Forward H(z)](figures/Hz_forward.png)\n")
    md.append("![Forward logμ(x)](figures/logmu_x.png)\n")
    md.append("![Forward logμ(A)](figures/logmu_logA_forward.png)\n")
    if (report_paths.figures_dir / "ablation_forward_vs_derivative.png").exists():
        md.append("![Ablation](figures/ablation_forward_vs_derivative.png)\n")
    if sbc_summary:
        md.append("![Coverage](figures/coverage_points.png)\n")

    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    print(f"Wrote {report_paths.report_md}")
    sigcc = fwd_post.params["sigma_cc_jit"]
    sigsn = fwd_post.params["sigma_sn_jit"]
    sigcc_p50 = float(np.percentile(sigcc, 50))
    sigsn_p50 = float(np.percentile(sigsn, 50))
    if sbc_summary:
        cov = sbc_summary["coverage"]
        logmuA68 = float(cov.get("logmu_logA_68", np.nan))
        logmuA95 = float(cov.get("logmu_logA_95", np.nan))
        logmuA_part = ""
        if np.isfinite(logmuA68) and np.isfinite(logmuA95):
            logmuA_part = f" cov_logmu_logA(68,95)=({logmuA68:.3f},{logmuA95:.3f})"
        print(
            f"SUMMARY z_max={z_max:.3f} SBC_N={sbc_summary['N']} noise_range=[{args.noise_scale_low:.3f},{args.noise_scale_high:.3f}] "
            f"cov_logmu(68,95)=({cov['logmu_68']:.3f},{cov['logmu_95']:.3f}) covsim_logmu(68,95)=({cov.get('logmu_sim_68', np.nan):.3f},{cov.get('logmu_sim_95', np.nan):.3f})"
            f"{logmuA_part} cov_H(68,95)=({cov['H_68']:.3f},{cov['H_95']:.3f}) "
            f"covsim_H(68,95)=({cov.get('H_sim_68', np.nan):.3f},{cov.get('H_sim_95', np.nan):.3f}) "
            f"sigma_jit_p50(cc,sn)=({sigcc_p50:.3g},{sigsn_p50:.3g})"
        )
    else:
        print(
            f"SUMMARY z_max={z_max:.3f} noise_scale={float(args.noise_scale):.3f} "
            f"sigma_jit_p50(cc,sn)=({sigcc_p50:.3g},{sigsn_p50:.3g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
