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
from pathlib import Path
import atexit
import signal
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.ingest_fsbao import load_fsbao
from entropy_horizon_recon.ingest_lensing import load_planck_lensing_proxy
from entropy_horizon_recon.ingest_planck_lensing_bandpowers import load_planck_lensing_bandpowers
from entropy_horizon_recon.ingest_rsd import load_rsd_fs8
from entropy_horizon_recon.ingest_rsd_single_survey import load_rsd_single_survey
from entropy_horizon_recon.ingest_fullshape_pk import load_fullshape_pk
from entropy_horizon_recon.departure import compute_departure_stats
from entropy_horizon_recon.inversion import infer_logmu_forward
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike
from entropy_horizon_recon.likelihoods_fsbao import FsBaoLogLike
from entropy_horizon_recon.likelihoods_lensing import PlanckLensingProxyLogLike
from entropy_horizon_recon.likelihoods_planck_lensing_bandpowers import PlanckLensingBandpowerLogLike
from entropy_horizon_recon.likelihoods_planck_lensing_clpp import PlanckLensingClppLogLike
from entropy_horizon_recon.likelihoods_rsd import RsdFs8LogLike
from entropy_horizon_recon.likelihoods_rsd_single_survey import RsdFs8CovLogLike
from entropy_horizon_recon.likelihoods_fullshape_pk import FullShapePkLogLike
from entropy_horizon_recon.mapping import H_to_area
from entropy_horizon_recon.proximity import (
    fit_gp_hyperparams,
    log_evidence_parametric,
    proximity_summary,
)
from entropy_horizon_recon.recon_gp import reconstruct_H_gp
from entropy_horizon_recon.recon_spline import reconstruct_H_spline
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty
from entropy_horizon_recon.viz import save_band_plot


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


def _dense_domain_zmax(
    z: np.ndarray,
    *,
    z_min: float,
    z_max_cap: float,
    bin_width: float,
    min_per_bin: int,
) -> float:
    z = np.asarray(z, dtype=float)
    z = z[(z >= z_min) & (z <= z_max_cap)]
    if z.size == 0:
        raise ValueError("No SN redshifts in requested range.")
    edges = np.arange(z_min, z_max_cap + bin_width, bin_width)
    counts, _ = np.histogram(z, bins=edges)
    # require all bins up to z_max to be dense
    ok = counts >= min_per_bin
    if not np.any(ok):
        return float(z_min + bin_width)
    last_good = int(np.where(ok)[0].max())
    return float(edges[last_good + 1])


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-data entropy-horizon reconstruction.")
    parser.add_argument("--out", type=Path, default=Path("outputs/realdata"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max-cap", type=float, default=1.2)
    parser.add_argument("--sn-bin-width", type=float, default=0.1)
    parser.add_argument("--sn-min-per-bin", type=int, default=60)
    parser.add_argument(
        "--sn-cov-kind",
        type=str,
        default="stat+sys",
        choices=["stat+sys", "statonly"],
        help="Pantheon+ covariance choice (default: stat+sys).",
    )
    parser.add_argument(
        "--sn-z-column",
        type=str,
        default="zHD",
        choices=["zHD", "zCMB", "zHEL"],
        help="Pantheon+ redshift column (default: zHD).",
    )
    parser.add_argument(
        "--sn-marg",
        type=str,
        default="M",
        choices=["M", "Mz", "Mstep"],
        help=(
            "SN linear nuisance marginalization model: "
            "M (intercept only), Mz (intercept + linear drift in z), "
            "Mstep (intercept + step at --sn-mstep-z)."
        ),
    )
    parser.add_argument("--sn-mstep-z", type=float, default=0.15, help="Step redshift for --sn-marg Mstep.")
    parser.add_argument("--sn-like-bin-width", type=float, default=0.05, help="Bin width for SN compression in forward inference.")
    parser.add_argument("--sn-like-min-per-bin", type=int, default=20, help="Min SNe per bin for SN compression.")
    parser.add_argument("--mu-knots", type=int, default=8)
    parser.add_argument("--mu-grid", type=int, default=120)
    parser.add_argument(
        "--mu-fixed",
        type=str,
        default=None,
        help="Fix logμ spline knots (e.g. 'bh' for μ=1) instead of sampling them.",
    )
    parser.add_argument("--mu-walkers", type=int, default=64)
    parser.add_argument("--mu-steps", type=int, default=1500)
    parser.add_argument("--mu-burn", type=int, default=500)
    parser.add_argument("--mu-draws", type=int, default=800)
    parser.add_argument("--mu-procs", type=int, default=0, help="Worker processes for μ inference (0=auto).")
    parser.add_argument("--mu-sampler", type=str, default="emcee", choices=["emcee", "ptemcee"])
    parser.add_argument("--pt-ntemps", type=int, default=8, help="Number of temperatures for ptemcee (if enabled).")
    parser.add_argument("--pt-tmax", type=float, default=None, help="Maximum temperature for ptemcee (if enabled).")
    parser.add_argument("--save-chain", type=Path, default=None, help="Optional path to save sampler chain (npz).")
    parser.add_argument(
        "--mu-init-seed",
        type=int,
        default=None,
        help="Optional separate RNG seed used ONLY to initialize the MCMC walkers (p0).",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=1536.0,
        help="Per-process RSS watchdog limit (MB). Set <=0 to disable.",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=0,
        help="Limit this run to the first N CPU cores (best-effort, 0=all).",
    )
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
        default=None,
        help="Half-normal scale for chronometer jitter (km/s/Mpc).",
    )
    parser.add_argument(
        "--sigma-sn-jit-scale",
        type=float,
        default=None,
        help="Half-normal scale for SN-magnitude jitter (mag).",
    )
    parser.add_argument(
        "--mapping-variant",
        type=str,
        default="M0",
        choices=["M0", "M1", "M2"],
        help=(
            "Mapping scenario: M0=baseline; M1=regularized closure residual R(z); "
            "M2=curved-horizon mapping with Ωk0 nuisance."
        ),
    )
    parser.add_argument(
        "--run-mapping-variants",
        action="store_true",
        help="Run mapping sensitivity variants (Ωm0 fixed, residual R(z), and Ωk0 nuisance).",
    )
    parser.add_argument(
        "--omega-k0-prior",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Uniform prior range for Ωk0 (enables curved-horizon mapping).",
    )
    parser.add_argument("--n-knots", type=int, default=16)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--gp-kernel", type=str, default="matern32", choices=["rbf", "matern32", "matern52"])
    parser.add_argument("--gp-walkers", type=int, default=64)
    parser.add_argument("--gp-steps", type=int, default=1200)
    parser.add_argument("--gp-burn", type=int, default=400)
    parser.add_argument("--gp-procs", type=int, default=0, help="Worker processes for GP recon (0=auto).")
    parser.add_argument("--skip-ablations", action="store_true", help="Skip small robustness ablations.")
    parser.add_argument("--ablation-steps", type=int, default=350)
    parser.add_argument("--ablation-burn", type=int, default=120)
    parser.add_argument("--spline-lambda", type=float, default=8.0)
    parser.add_argument("--spline-bootstrap", type=int, default=120)
    parser.add_argument("--omega-m0-min", type=float, default=0.2)
    parser.add_argument("--omega-m0-max", type=float, default=0.4)
    parser.add_argument("--n-logA", type=int, default=140)
    parser.add_argument("--m-weight-mode", type=str, default="variance", choices=["variance", "uniform"])
    parser.add_argument("--skip-hz-recon", action="store_true", help="Skip the GP/spline H(z) recon cross-checks.")
    parser.add_argument("--include-rsd", action="store_true", help="Include an external RSD fσ8(z) compilation likelihood.")
    parser.add_argument(
        "--rsd-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "diag_compilation",
            "dr12_consensus_fs",
            "dr16_lrg_fsbao",
            "dr12+dr16_fsbao",
            "dr16_lrg_fs8",
        ],
        help=(
            "Growth constraint source. FSBAO modes include correlated distance+fσ8 vectors and will "
            "automatically drop overlapping BAO-only datasets to avoid double counting."
        ),
    )
    parser.add_argument(
        "--fsbao-diag-cov",
        action="store_true",
        help="Replace FSBAO covariance with diag(C) (diagnostic only; default keeps full covariance).",
    )
    parser.add_argument(
        "--bao-diag-cov",
        action="store_true",
        help="Use diagonal BAO covariance for BAO-only datasets (diagnostic only).",
    )
    parser.add_argument(
        "--drop-bao",
        type=str,
        nargs="*",
        default=[],
        choices=["dr12", "dr16", "desi"],
        help="Drop BAO-only dataset families (dr12, dr16, desi).",
    )
    parser.add_argument(
        "--include-lensing",
        action="store_true",
        help="Include a compressed Planck CMB lensing proxy likelihood (no TT/TE/EE distance priors).",
    )
    parser.add_argument(
        "--include-planck-lensing-clpp",
        action="store_true",
        help="Include Planck 2018 lensing bandpower likelihood (scaled template).",
    )
    parser.add_argument("--clpp-alpha", type=float, default=0.25, help="Exponent alpha in sigma8*Omega_m^alpha scaling.")
    parser.add_argument(
        "--clpp-backend",
        type=str,
        default="scaled",
        choices=["scaled", "camb"],
        help="Planck lensing bandpower prediction backend.",
    )
    parser.add_argument(
        "--include-fullshape-pk",
        action="store_true",
        help="Include full-shape P(k) monopole likelihood (Shapefit BOSS DR12).",
    )
    parser.add_argument(
        "--pk-dataset",
        type=str,
        default="shapefit_lrgz1_ngc_mono",
        choices=["shapefit_lrgz1_ngc_mono"],
        help="Full-shape P(k) dataset key.",
    )
    parser.add_argument(
        "--pk-bias-prior",
        type=float,
        nargs=2,
        default=[0.5, 4.0],
        metavar=("LOW", "HIGH"),
        help="Prior for linear bias b1.",
    )
    parser.add_argument(
        "--pk-noise-prior",
        type=float,
        nargs=2,
        default=[0.0, 1.0e5],
        metavar=("LOW", "HIGH"),
        help="Prior for shot-noise term P_shot.",
    )
    parser.add_argument(
        "--clpp-dataset",
        type=str,
        default="consext8",
        choices=["consext8", "agr2"],
        help="Planck lensing bandpower dataset variant.",
    )
    parser.add_argument("--lensing-exponent", type=float, default=0.25, help="Exponent for sigma8*Omega_m^alpha proxy.")
    parser.add_argument("--lensing-sigma-scale", type=float, default=1.0, help="Scale factor for lensing proxy sigma.")
    parser.add_argument("--growth-mode", type=str, default="ode", choices=["ode", "gamma"])
    parser.add_argument("--sigma8-prior", type=float, nargs=2, default=[0.6, 1.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--lensing-mode", type=str, default="gaussian_s8", choices=["gaussian_s8", "full_clpp"])
    parser.add_argument("--timing-log", type=Path, default=None, help="Write timing diagnostics JSONL to this path.")
    parser.add_argument("--timing-every", type=int, default=200, help="Flush timing stats every N likelihood calls.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Checkpoint sampling state every N steps (ptemcee only; 0 disables).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Directory for checkpoint files (defaults to <out>/samples/mu_checkpoint if enabled).",
    )
    args = parser.parse_args()
    cpu_cores = _resolve_cpu_cores(args.cpu_cores)
    _apply_cpu_affinity(cpu_cores)
    args.cpu_cores = cpu_cores
    args.mu_procs = _resolve_procs(args.mu_procs, n_walkers=int(args.mu_walkers), cpu_cores=cpu_cores)
    args.gp_procs = _resolve_procs(args.gp_procs, n_walkers=int(args.gp_walkers), cpu_cores=cpu_cores)

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = git_head_sha(repo_root=repo_root) or "unknown"
    git_dirty = git_is_dirty(repo_root=repo_root)
    cmd = command_str()
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()
    rng = np.random.default_rng(args.seed)

    const_A = 4.0 * np.pi * (constants.c_km_s**2)

    def _logA0_from_params(H0_s: np.ndarray, omega_k0_s: np.ndarray) -> np.ndarray:
        denom0 = (H0_s**2) * (1.0 - omega_k0_s)
        if np.any(denom0 <= 0):
            raise ValueError("Invalid A0 mapping: require 1 - omega_k0 > 0 for all draws.")
        return np.log(const_A / denom0)

    def _area_from_H_samples(H_samples: np.ndarray, z: np.ndarray, H0_s: np.ndarray, omega_k0_s: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        u = np.asarray(H_samples, dtype=float) ** 2
        denom = u - (H0_s[:, None] ** 2) * omega_k0_s[:, None] * (1.0 + z[None, :]) ** 2
        if np.any(denom <= 0):
            raise ValueError("Invalid apparent-horizon area denominator for some draws.")
        return const_A / denom

    def _stable_logA_domain(logA_draws: np.ndarray) -> tuple[float, float, dict]:
        logA_draws = np.asarray(logA_draws, dtype=float)
        n_total = int(logA_draws.shape[0])
        finite_rows = np.all(np.isfinite(logA_draws), axis=1)
        logA_ok = logA_draws[finite_rows]
        n_valid = int(logA_ok.shape[0])
        if n_valid == 0:
            raise RuntimeError("No finite logA samples to determine domain.")

        lo = np.percentile(logA_ok, 2, axis=1)
        hi = np.percentile(logA_ok, 98, axis=1)
        logA_min = float(np.max(lo))
        logA_max = float(np.min(hi))
        method = "strict"

        if not np.isfinite(logA_min) or not np.isfinite(logA_max) or logA_max <= logA_min:
            # Fallback: drop outlier samples and take a robust overlap.
            if lo.size >= 5:
                logA_min = float(np.percentile(lo, 95))
                logA_max = float(np.percentile(hi, 5))
                method = "robust"
            if not np.isfinite(logA_min) or not np.isfinite(logA_max) or logA_max <= logA_min:
                # Final fallback: global percentiles across all finite logA values.
                vals = logA_ok[np.isfinite(logA_ok)]
                if vals.size:
                    logA_min = float(np.percentile(vals, 2))
                    logA_max = float(np.percentile(vals, 98))
                    method = "global"

        if not np.isfinite(logA_min) or not np.isfinite(logA_max) or logA_max <= logA_min:
            raise RuntimeError("Failed to determine a stable logA domain.")
        meta = {
            "method": method,
            "n_total": n_total,
            "n_valid": n_valid,
            "fallback_used": method != "strict",
        }
        if meta["fallback_used"]:
            print(f"[warn] logA domain fallback used: {meta}", flush=True)
        return logA_min, logA_max, meta

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = report_paths.out_dir / "debug_invalid_logprob.txt"
    runtime_log_path = report_paths.out_dir / "runtime.log"
    fatal_log_path = report_paths.out_dir / "fatal.log"

    def _runtime_log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            with open(runtime_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    def _fatal_log(exc: BaseException) -> None:
        try:
            with open(fatal_log_path, "a", encoding="utf-8") as f:
                f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass

    _runtime_log(f"START pid={os.getpid()} argv={sys.argv}")

    def _on_exit() -> None:
        _runtime_log("EXIT normal")

    atexit.register(_on_exit)

    def _on_signal(signum: int, _frame) -> None:
        _runtime_log(f"SIGNAL {signum}")
        raise SystemExit(128 + int(signum))

    _signals = [signal.SIGTERM, signal.SIGINT, signal.SIGQUIT]
    # Helpful for diagnosing unexpected exits when launched via nohup/remote sessions.
    if hasattr(signal, "SIGHUP"):
        _signals.append(signal.SIGHUP)
    for _sig in _signals:
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass

    try:
        import faulthandler

        _fh = open(fatal_log_path, "a", encoding="utf-8")
        faulthandler.enable(file=_fh)
        _runtime_log("faulthandler enabled")
    except Exception:
        _runtime_log("faulthandler enable failed")

    def _excepthook(exc_type, exc, tb) -> None:
        _runtime_log(f"EXCEPTION {exc_type.__name__}: {exc}")
        try:
            with open(fatal_log_path, "a", encoding="utf-8") as f:
                f.write("".join(traceback.format_exception(exc_type, exc, tb)))
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    _runtime_log("loading real data")
    # --- Load real data (cached) ---
    sn = load_pantheon_plus(
        paths=paths,
        cov_kind=str(args.sn_cov_kind),
        subset="cosmology",
        z_column=str(args.sn_z_column),
    )
    cc = load_chronometers(paths=paths, variant="BC03_all")
    bao12 = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    bao16 = load_bao(paths=paths, dataset="sdss_dr16_lrg_bao_dmdh")
    desi24 = load_bao(paths=paths, dataset="desi_2024_bao_all")

    _runtime_log("building SN/CC likelihoods")
    z_max = _dense_domain_zmax(
        sn.z,
        z_min=args.z_min,
        z_max_cap=args.z_max_cap,
        bin_width=args.sn_bin_width,
        min_per_bin=args.sn_min_per_bin,
    )
    z_min = float(args.z_min)

    sn_like = SNLogLike.from_pantheon(sn, z_min=z_min, z_max=z_max)
    cc_like = ChronometerLogLike.from_data(cc, z_min=z_min, z_max=z_max)
    _runtime_log("SN/CC likelihoods built")

    # --- Optional growth (RSD / FSBAO) + lensing constraints ---
    _runtime_log("building growth/lensing likelihoods")
    rsd_mode = str(args.rsd_mode)
    if args.include_rsd and rsd_mode == "none":
        rsd_mode = "diag_compilation"

    rsd_like = None
    fsbao_likes: list[FsBaoLogLike] = []
    skip_bao_datasets: set[str] = set()

    if rsd_mode == "diag_compilation":
        if args.growth_mode != "ode":
            raise ValueError("Only --growth-mode ode is implemented currently.")
        rsd = load_rsd_fs8(paths=paths)
        m = (rsd.z >= z_min) & (rsd.z <= z_max)
        rsd_like = RsdFs8LogLike.from_data(
            z=rsd.z[m],
            fs8=rsd.fs8[m],
            sigma_fs8=rsd.sigma_fs8[m],
            meta={"source": rsd.meta},
        )
    elif rsd_mode == "dr16_lrg_fs8":
        rsd_single = load_rsd_single_survey(paths=paths, dataset="sdss_dr16_lrg_fsbao_dmdhfs8")
        m = (rsd_single.z >= z_min) & (rsd_single.z <= z_max)
        rsd_like = RsdFs8CovLogLike.from_data(
            z=rsd_single.z[m],
            fs8=rsd_single.fs8[m],
            cov=rsd_single.cov[np.ix_(m, m)],
            meta=rsd_single.meta,
        )
    elif rsd_mode in {"dr12_consensus_fs", "dr12+dr16_fsbao"}:
        fs = load_fsbao(paths=paths, dataset="sdss_dr12_consensus_fs")
        fsbao_likes.append(
            FsBaoLogLike.from_data(
                fs,
                dataset="sdss_dr12_consensus_fs",
                constants=constants,
                z_min=z_min,
                z_max=z_max,
                diag_cov=bool(args.fsbao_diag_cov),
            )
        )
        skip_bao_datasets.add("sdss_dr12_consensus_bao")
    if rsd_mode in {"dr16_lrg_fsbao", "dr12+dr16_fsbao"}:
        fs = load_fsbao(paths=paths, dataset="sdss_dr16_lrg_fsbao_dmdhfs8")
        fsbao_likes.append(
            FsBaoLogLike.from_data(
                fs,
                dataset="sdss_dr16_lrg_fsbao_dmdhfs8",
                constants=constants,
                z_min=z_min,
                z_max=z_max,
                diag_cov=bool(args.fsbao_diag_cov),
            )
        )
        skip_bao_datasets.add("sdss_dr16_lrg_bao_dmdh")
    elif rsd_mode not in {"none", "diag_compilation", "dr12_consensus_fs", "dr16_lrg_fsbao", "dr12+dr16_fsbao", "dr16_lrg_fs8"}:
        raise ValueError(f"Unsupported --rsd-mode '{rsd_mode}'.")

    # BAO-only datasets (drop overlaps if FSBAO is enabled to avoid double-counting).
    _runtime_log("building BAO likelihoods")
    bao_likes = []
    for key in args.drop_bao:
        if key == "dr12":
            skip_bao_datasets.add("sdss_dr12_consensus_bao")
        elif key == "dr16":
            skip_bao_datasets.add("sdss_dr16_lrg_bao_dmdh")
        elif key == "desi":
            skip_bao_datasets.add("desi_2024_bao_all")
    for dataset, bao in [
        ("sdss_dr12_consensus_bao", bao12),
        ("sdss_dr16_lrg_bao_dmdh", bao16),
        ("desi_2024_bao_all", desi24),
    ]:
        if dataset in skip_bao_datasets:
            continue
        try:
            bl = BaoLogLike.from_data(bao, dataset=dataset, constants=constants, z_min=z_min, z_max=z_max)
            if args.bao_diag_cov:
                cov = np.diag(np.diag(bl.cov))
                bl = BaoLogLike.from_arrays(
                    dataset=bl.dataset,
                    z=bl.z,
                    y=bl.y,
                    obs=bl.obs,
                    cov=cov,
                    constants=constants,
                )
            bao_likes.append(bl)
        except ValueError as e:
            print(f"Skipping BAO dataset {dataset}: {e}")

    lens_like = None
    if args.include_planck_lensing_clpp and args.include_lensing:
        raise ValueError("Use either --include-planck-lensing-clpp or --include-lensing (proxy), not both.")
    if args.include_planck_lensing_clpp:
        pl = load_planck_lensing_bandpowers(paths=paths, dataset=str(args.clpp_dataset))
        if str(args.clpp_backend) == "camb":
            lens_like = PlanckLensingClppLogLike.from_data(
                ell_eff=pl.ell_eff,
                clpp=pl.clpp,
                cov=pl.cov,
                meta=pl.meta,
            )
        else:
            lens_like = PlanckLensingBandpowerLogLike.from_data(
                clpp=pl.clpp,
                cov=pl.cov,
                template_clpp=pl.template_clpp,
                alpha=float(args.clpp_alpha),
                s8om_fid=0.589,
                meta=pl.meta,
            )
    elif args.include_lensing:
        if args.lensing_mode != "gaussian_s8":
            raise ValueError("Only --lensing-mode gaussian_s8 is implemented currently.")
        pl = load_planck_lensing_proxy(paths=paths)
        lens_sigma = float(pl.sigma8_om025_sigma) * float(args.lensing_sigma_scale)
        lens_like = PlanckLensingProxyLogLike(
            mean=pl.sigma8_om025_mean,
            sigma=lens_sigma,
            meta=pl.meta,
            exponent=float(args.lensing_exponent),
        )
    _runtime_log("growth/lensing likelihoods built")

    pk_like = None
    if args.include_fullshape_pk:
        pk = load_fullshape_pk(paths=paths, dataset=str(args.pk_dataset))
        pk_like = FullShapePkLogLike.from_data(
            k=pk.k,
            pk=pk.pk,
            cov=pk.cov,
            z_eff=pk.z_eff,
            meta=pk.meta,
        )
    _runtime_log("fullshape pk likelihood built" if pk_like is not None else "fullshape pk disabled")

    # --- Common grids ---
    _runtime_log("building grids")
    z_knots = np.linspace(0.0, z_max, args.n_knots)
    z_grid = np.linspace(0.0, z_max, args.n_grid)

    def band(samples: np.ndarray, *, level: float = 0.68) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0,1).")
        med = np.median(samples, axis=0)
        q = (1.0 - level) / 2.0
        lo = np.percentile(samples, 100.0 * q, axis=0)
        hi = np.percentile(samples, 100.0 * (1.0 - q), axis=0)
        return med, lo, hi

    # --- Forward-model inference: log μ(x) with x = log(A/A0) ---
    z_edges = np.arange(z_min, z_max + args.sn_like_bin_width, args.sn_like_bin_width)
    sn_like_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=args.sn_like_min_per_bin)

    # Choose x-domain conservatively from a BH baseline guess.
    H0_guess = 70.0
    omega_m0_guess = 0.3
    H_zmax_guess = H0_guess * np.sqrt(omega_m0_guess * (1.0 + z_max) ** 3 + (1.0 - omega_m0_guess))
    x_min_guess = float(2.0 * np.log(H0_guess / H_zmax_guess))  # negative
    x_min = float(2.0 * x_min_guess)  # extra margin for safety
    x_knots = np.linspace(1.25 * x_min, 0.0, args.mu_knots)
    x_grid = np.linspace(x_min, 0.0, args.mu_grid)

    _runtime_log("starting inference")
    infer_kwargs = dict(
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_z=sn_like_bin.z,
        sn_m=sn_like_bin.m,
        sn_cov=sn_like_bin.cov,
        sn_marg=str(args.sn_marg),
        sn_mstep_z=float(args.sn_mstep_z),
        cc_z=cc_like.z,
        cc_H=cc_like.H,
        cc_sigma_H=cc_like.sigma_H,
        bao_likes=bao_likes,
        fsbao_likes=fsbao_likes,
        rsd_like=rsd_like,
        lensing_like=lens_like,
        pk_like=pk_like,
        constants=constants,
        sampler_kind=str(args.mu_sampler),
        pt_ntemps=int(args.pt_ntemps),
        pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
        n_walkers=max(2 * (int(args.mu_knots) + 6), int(args.mu_walkers)),
        n_steps=args.mu_steps,
        n_burn=args.mu_burn,
        seed=args.seed,
        init_seed=args.mu_init_seed,
        n_processes=args.mu_procs,
        n_draws=args.mu_draws,
        omega_m0_prior=(float(args.omega_m0_min), float(args.omega_m0_max)),
        sigma8_prior=(float(args.sigma8_prior[0]), float(args.sigma8_prior[1])),
        pk_bias_prior=(float(args.pk_bias_prior[0]), float(args.pk_bias_prior[1])),
        pk_noise_prior=(float(args.pk_noise_prior[0]), float(args.pk_noise_prior[1])),
        max_rss_mb=float(args.max_rss_mb) if args.max_rss_mb is not None else None,
        progress=True,
        debug_log_path=debug_log_path,
        timing_log_path=str(args.timing_log) if args.timing_log is not None else None,
        timing_every=int(args.timing_every),
        save_chain_path=str(args.save_chain) if args.save_chain is not None else None,
        checkpoint_every=int(args.checkpoint_every),
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path is not None else None,
    )
    # Mapping variants are implemented as separate *scenarios*; keep priors fixed unless explicitly enabled.
    if args.mapping_variant == "M1":
        infer_kwargs["use_residual"] = True
    elif args.mapping_variant == "M2":
        ok_prior = (
            tuple(float(x) for x in args.omega_k0_prior)
            if args.omega_k0_prior is not None
            else (-0.2, 0.2)
        )
        infer_kwargs["omega_k0_prior"] = ok_prior
    if args.sigma_d2_scale is not None:
        infer_kwargs["sigma_d2_scale"] = float(args.sigma_d2_scale)
    if args.logmu_knot_scale is not None:
        infer_kwargs["logmu_knot_scale"] = float(args.logmu_knot_scale)
    if args.sigma_cc_jit_scale is not None:
        infer_kwargs["sigma_cc_jit_scale"] = float(args.sigma_cc_jit_scale)
    if args.sigma_sn_jit_scale is not None:
        infer_kwargs["sigma_sn_jit_scale"] = float(args.sigma_sn_jit_scale)
    if args.checkpoint_every and (args.checkpoint_path is None):
        infer_kwargs["checkpoint_path"] = str(report_paths.samples_dir / "mu_checkpoint")
    if args.mu_fixed is not None:
        infer_kwargs["fixed_logmu_knots"] = str(args.mu_fixed)
    mu_post = infer_logmu_forward(**infer_kwargs)
    _runtime_log("inference done")

    # --- Bands and plots (forward model) ---
    H_fwd_med, H_fwd_lo, H_fwd_hi = band(mu_post.H_samples, level=0.68)
    save_band_plot(
        z_grid,
        H_fwd_med,
        H_fwd_lo,
        H_fwd_hi,
        xlabel="z",
        ylabel="H(z) [km/s/Mpc]",
        title="H(z) from forward μ(A) inference",
        path=report_paths.figures_dir / "Hz_forward.png",
    )

    logmu_x_med, logmu_x_lo, logmu_x_hi = band(mu_post.logmu_x_samples, level=0.68)
    save_band_plot(
        x_grid,
        logmu_x_med,
        logmu_x_lo,
        logmu_x_hi,
        xlabel="x = log(A/A0)",
        ylabel="log μ(x)",
        title="Reconstructed log μ(x) (posterior band)",
        path=report_paths.figures_dir / "logmu_x.png",
    )

    H0_s = np.asarray(mu_post.params["H0"], dtype=float)
    Ok_s = np.asarray(mu_post.params.get("omega_k0", np.zeros_like(H0_s)), dtype=float)
    A_draws = _area_from_H_samples(mu_post.H_samples, z_grid, H0_s, Ok_s)
    logA_samples = np.log(A_draws)
    logA_med_z, logA_lo_z, logA_hi_z = band(logA_samples, level=0.68)
    save_band_plot(
        z_grid,
        logA_med_z,
        logA_lo_z,
        logA_hi_z,
        xlabel="z",
        ylabel="log A(z) [log(Mpc^2)]",
        title="Apparent-horizon area A(z) from forward inference",
        path=report_paths.figures_dir / "Az_log.png",
    )

    # Convert logμ(x) posterior to logμ(logA) on a fixed logA grid.
    logA_draws = logA_samples
    logA_min, logA_max, logA_domain_meta = _stable_logA_domain(logA_draws)
    logA_grid = np.linspace(logA_min, logA_max, args.n_logA)

    logmu_logA_samples = np.empty((mu_post.logmu_x_samples.shape[0], logA_grid.size))
    for j in range(mu_post.logmu_x_samples.shape[0]):
        H0_j = float(H0_s[j])
        logA0_j = float(_logA0_from_params(np.array([H0_j]), np.array([float(Ok_s[j])]))[0])
        xj = np.clip(logA_grid - logA0_j, x_grid[0], x_grid[-1])
        logmu_logA_samples[j] = np.interp(xj, x_grid, mu_post.logmu_x_samples[j])

    logmu_med, logmu_lo, logmu_hi = band(logmu_logA_samples, level=0.68)
    save_band_plot(
        logA_grid,
        logmu_med,
        logmu_lo,
        logmu_hi,
        xlabel="log A",
        ylabel="log μ(A)",
        title="Reconstructed log μ(A) (posterior band)",
        path=report_paths.figures_dir / "logmu_logA.png",
    )

    prox = proximity_summary(logA_grid=logA_grid, logmu_samples=logmu_logA_samples)
    departure = compute_departure_stats(
        logA_grid=logA_grid, logmu_samples=logmu_logA_samples, weight_mode=str(args.m_weight_mode)
    )
    y = np.mean(logmu_logA_samples, axis=0)
    yerr = np.std(logmu_logA_samples, axis=0, ddof=1)
    gp_fit = fit_gp_hyperparams(logA_grid, y, yerr, kernel="matern32")
    logZ_gp = gp_fit["logZ"]
    logZ = {
        "bh": log_evidence_parametric(logA_grid, y, yerr, model="bh", n_mc=1, seed=args.seed)["logZ"],
        "tsallis": log_evidence_parametric(logA_grid, y, yerr, model="tsallis", seed=args.seed)["logZ"],
        "barrow": log_evidence_parametric(logA_grid, y, yerr, model="barrow", seed=args.seed)["logZ"],
        "kaniadakis": log_evidence_parametric(logA_grid, y, yerr, model="kaniadakis", seed=args.seed)["logZ"],
    }

    # --- Mapping sensitivity variants (optional; expensive) ---
    mapping_variants = {}
    mapping_sensitivity = None
    if args.run_mapping_variants:
        mapping_variants["V1_free"] = mu_post

        # Variant 0: Ωm0 fixed (tests sensitivity to assumed matter sector)
        v0_kwargs = dict(infer_kwargs)
        v0_kwargs["omega_m0_fixed"] = 0.3
        v0_kwargs["seed"] = int(args.seed) + 101
        v0_kwargs["progress"] = False
        v0_kwargs["n_processes"] = 1
        mapping_variants["V0_fixedOm"] = infer_logmu_forward(**v0_kwargs)

        # Variant 2: add a tightly-regularized residual closure term R(z)
        v2_kwargs = dict(infer_kwargs)
        v2_kwargs["use_residual"] = True
        v2_kwargs["seed"] = int(args.seed) + 202
        v2_kwargs["progress"] = False
        v2_kwargs["n_processes"] = 1
        mapping_variants["V2_residual"] = infer_logmu_forward(**v2_kwargs)

        # Curvature nuisance (Ωk0) in the apparent-horizon mapping
        ok_prior = tuple(float(x) for x in args.omega_k0_prior) if args.omega_k0_prior is not None else (-0.02, 0.02)
        vk_kwargs = dict(infer_kwargs)
        vk_kwargs["omega_k0_prior"] = ok_prior
        vk_kwargs["seed"] = int(args.seed) + 303
        vk_kwargs["progress"] = False
        vk_kwargs["n_processes"] = 1
        mapping_variants["V1_curved"] = infer_logmu_forward(**vk_kwargs)

        # Common logA grid on the overlap region across variants.
        ranges = []
        for post in mapping_variants.values():
            H0_v = np.asarray(post.params["H0"], dtype=float)
            Ok_v = np.asarray(post.params.get("omega_k0", np.zeros_like(H0_v)), dtype=float)
            logA_v = np.log(_area_from_H_samples(post.H_samples, z_grid, H0_v, Ok_v))
            lo = float(np.max(np.percentile(logA_v, 2, axis=1)))
            hi = float(np.min(np.percentile(logA_v, 98, axis=1)))
            ranges.append((lo, hi))
        logA_min_v = float(max(lo for lo, _ in ranges))
        logA_max_v = float(min(hi for _, hi in ranges))
        if not np.isfinite(logA_min_v) or not np.isfinite(logA_max_v) or logA_max_v <= logA_min_v:
            raise RuntimeError("Failed to determine overlap logA domain for mapping variants.")
        logA_grid_v = np.linspace(logA_min_v, logA_max_v, int(args.n_logA))

        variant_logmu = {}
        for name, post in mapping_variants.items():
            H0_v = np.asarray(post.params["H0"], dtype=float)
            Ok_v = np.asarray(post.params.get("omega_k0", np.zeros_like(H0_v)), dtype=float)
            logA0_v = _logA0_from_params(H0_v, Ok_v)
            xg = np.asarray(post.x_grid, dtype=float)
            xj = np.clip(logA_grid_v[None, :] - logA0_v[:, None], xg[0], xg[-1])
            samp = np.empty((post.logmu_x_samples.shape[0], logA_grid_v.size))
            for j in range(samp.shape[0]):
                samp[j] = np.interp(xj[j], xg, post.logmu_x_samples[j])
            variant_logmu[name] = samp

        base = variant_logmu["V1_free"]
        base_med = np.median(base, axis=0)
        base_std = np.std(base, axis=0, ddof=1) + 1e-12
        rows = []
        for name, samp in variant_logmu.items():
            med = np.median(samp, axis=0)
            delta_sigma = float(np.sqrt(np.mean(((med - base_med) / base_std) ** 2)))
            max_abs = float(np.max(np.abs(med - base_med)))
            rows.append([name, delta_sigma, max_abs])
        rows.sort(key=lambda r: r[0])
        mapping_sensitivity = {"logA_min": logA_min_v, "logA_max": logA_max_v, "deltas": rows}

        # Overlay plot of logμ(A) under mapping variants.
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"V1_free": "C0", "V0_fixedOm": "C1", "V2_residual": "C2", "V1_curved": "C3"}
        for name, samp in variant_logmu.items():
            med = np.median(samp, axis=0)
            lo = np.percentile(samp, 16.0, axis=0)
            hi = np.percentile(samp, 84.0, axis=0)
            c = colors.get(name, None)
            if name == "V1_free":
                ax.fill_between(logA_grid_v, lo, hi, color=c, alpha=0.20, linewidth=0)
                ax.plot(logA_grid_v, med, color=c, lw=2.2, label=name)
            else:
                ax.plot(logA_grid_v, med, color=c, lw=1.8, alpha=0.95, label=name)
        ax.set(xlabel="log A", ylabel="log μ(A)", title="Mapping sensitivity (log μ(A))")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(report_paths.figures_dir / "logmu_logA_variants.png", dpi=200)
        plt.close(fig)

    # --- Optional H(z) recon cross-checks (not used for μ inference) ---
    gp_post = None
    spline_post = None
    if not args.skip_hz_recon:
        gp_post = reconstruct_H_gp(
            z_knots=z_knots,
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
            n_processes=args.gp_procs,
        )
        spline_post = reconstruct_H_spline(
            z_knots=z_knots,
            sn_like=sn_like,
            cc_like=cc_like,
            bao_likes=bao_likes,
            constants=constants,
            z_grid=z_grid,
            z_max_background=z_max,
            smooth_lambda=args.spline_lambda,
            n_bootstrap=args.spline_bootstrap,
            seed=args.seed + 1,
        )

        H_gp_med, H_gp_lo, H_gp_hi = band(gp_post.H_samples, level=0.68)
        H_sp_med, H_sp_lo, H_sp_hi = band(spline_post.H_samples, level=0.68)
        save_band_plot(
            z_grid,
            H_gp_med,
            H_gp_lo,
            H_gp_hi,
            xlabel="z",
            ylabel="H(z) [km/s/Mpc]",
            title="Reconstructed H(z) (GP cross-check)",
            path=report_paths.figures_dir / "Hz_gp.png",
        )
        save_band_plot(
            z_grid,
            H_sp_med,
            H_sp_lo,
            H_sp_hi,
            xlabel="z",
            ylabel="H(z) [km/s/Mpc]",
            title="Reconstructed H(z) (Spline cross-check)",
            path=report_paths.figures_dir / "Hz_spline.png",
        )

    # --- Minimal ablations (prior/kernel/covariance sensitivity) ---
    ablations = []
    if not args.skip_ablations:
        print("Running ablation: sn_diagonal_cov (forward μ inference)")
        sn_cov_diag = np.diag(np.diag(sn_like_bin.cov))
        ab_kwargs = dict(
            z_grid=z_grid,
            x_knots=x_knots,
            x_grid=x_grid,
            sn_z=sn_like_bin.z,
            sn_m=sn_like_bin.m,
            sn_cov=sn_cov_diag,
            cc_z=cc_like.z,
            cc_H=cc_like.H,
            cc_sigma_H=cc_like.sigma_H,
            bao_likes=bao_likes,
            fsbao_likes=fsbao_likes,
            rsd_like=rsd_like,
            lensing_like=lens_like,
            constants=constants,
            sampler_kind=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            n_walkers=args.mu_walkers,
            n_steps=args.ablation_steps,
            n_burn=args.ablation_burn,
            seed=args.seed + 10,
            n_processes=1,
            n_draws=min(400, args.mu_draws),
            progress=True,
            debug_log_path=debug_log_path,
        )
        if args.sigma_d2_scale is not None:
            ab_kwargs["sigma_d2_scale"] = float(args.sigma_d2_scale)
        if args.logmu_knot_scale is not None:
            ab_kwargs["logmu_knot_scale"] = float(args.logmu_knot_scale)
        if args.sigma_cc_jit_scale is not None:
            ab_kwargs["sigma_cc_jit_scale"] = float(args.sigma_cc_jit_scale)
        if args.sigma_sn_jit_scale is not None:
            ab_kwargs["sigma_sn_jit_scale"] = float(args.sigma_sn_jit_scale)
        mu_ab = infer_logmu_forward(**ab_kwargs)
        logmu_logA_ab = np.empty((mu_ab.logmu_x_samples.shape[0], logA_grid.size))
        for j in range(mu_ab.logmu_x_samples.shape[0]):
            H0_j = float(mu_ab.H_samples[j, 0])
            logA0_j = float(np.log(4.0 * np.pi * (constants.c_km_s / H0_j) ** 2))
            xj = np.clip(logA_grid - logA0_j, x_grid[0], x_grid[-1])
            logmu_logA_ab[j] = np.interp(xj, x_grid, mu_ab.logmu_x_samples[j])
        prox_ab = proximity_summary(logA_grid=logA_grid, logmu_samples=logmu_logA_ab)
        ablations.append({"name": "sn_diagonal_cov", "D2_mean": prox_ab["D2_mean"]})

    # --- Save machine-readable results ---
    (report_paths.out_dir / "samples").mkdir(parents=True, exist_ok=True)
    sample_npz = {
        "x_grid": x_grid,
        "logmu_x_samples": mu_post.logmu_x_samples,
        "z_grid": z_grid,
        "H_samples": mu_post.H_samples,
        "H0": mu_post.params["H0"],
        "omega_m0": mu_post.params["omega_m0"],
        "omega_k0": mu_post.params.get("omega_k0"),
        "r_d_Mpc": mu_post.params["r_d_Mpc"],
        "sigma_cc_jit": mu_post.params["sigma_cc_jit"],
        "sigma_sn_jit": mu_post.params["sigma_sn_jit"],
        "sigma_d2": mu_post.params["sigma_d2"],
    }
    if "sigma8_0" in mu_post.params:
        sample_npz["sigma8_0"] = mu_post.params["sigma8_0"]
    if "S8" in mu_post.params:
        sample_npz["S8"] = mu_post.params["S8"]
    np.savez_compressed(report_paths.out_dir / "samples" / "mu_forward_posterior.npz", **sample_npz)
    (report_paths.out_dir / "samples" / "mu_forward_meta.json").write_text(
        json.dumps({"meta": mu_post.meta}, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        report_paths.out_dir / "samples" / "logmu_logA_samples.npz",
        logA=logA_grid,
        logmu_samples=logmu_logA_samples,
    )
    if gp_post is not None:
        np.savez_compressed(
            report_paths.out_dir / "samples" / "Hz_gp_samples.npz",
            z=z_grid,
            H_samples=gp_post.H_samples,
            dH_dz_samples=gp_post.dH_dz_samples,
        )
        (report_paths.out_dir / "samples" / "Hz_gp_meta.json").write_text(
            json.dumps({"meta": gp_post.meta, "hyper_samples": {k: v.tolist() for k, v in gp_post.hyper_samples.items()}}, indent=2),
            encoding="utf-8",
        )
    if spline_post is not None:
        np.savez_compressed(
            report_paths.out_dir / "samples" / "Hz_spline_samples.npz",
            z=z_grid,
            H_samples=spline_post.H_samples,
            dH_dz_samples=spline_post.dH_dz_samples,
        )
        (report_paths.out_dir / "samples" / "Hz_spline_meta.json").write_text(
            json.dumps({"meta": spline_post.meta}, indent=2),
            encoding="utf-8",
        )
    (report_paths.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_paths.out_dir / "tables" / "proximity.json").write_text(
        json.dumps(
            _jsonify(
                {
                    "git": {"sha": git_sha, "dirty": git_dirty},
                    "command": cmd,
                    "prox": prox,
                    "departure": departure,
                    "logZ_gp": logZ_gp,
                    "logZ_models": logZ,
                    "gp_fit": gp_fit,
                    "ablations": ablations,
                    "mapping_sensitivity": mapping_sensitivity,
                    "mapping_variants_meta": (
                        {k: v.meta for k, v in mapping_variants.items()} if mapping_variants else None
                    ),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_paths.out_dir / "tables" / "departure_stats.json").write_text(
        json.dumps(_jsonify(departure), indent=2),
        encoding="utf-8",
    )

    # Lightweight summary for reproducibility (full details in samples/ + tables/proximity.json).
    summary = {
        "git": {"sha": git_sha, "dirty": git_dirty},
        "command": cmd,
        "seed": int(args.seed),
        "z_domain": {"z_min": float(z_min), "z_max": float(z_max)},
        "logA_domain": {
            "logA_min": float(logA_min),
            "logA_max": float(logA_max),
            "method": logA_domain_meta.get("method"),
            "n_total": logA_domain_meta.get("n_total"),
            "n_valid": logA_domain_meta.get("n_valid"),
            "fallback_used": logA_domain_meta.get("fallback_used"),
        },
        "settings": {
            "mapping_variant": str(args.mapping_variant),
            "sn_cov_kind": str(args.sn_cov_kind),
            "sn_z_column": str(args.sn_z_column),
            "sn_marg": str(args.sn_marg),
            "sn_mstep_z": float(args.sn_mstep_z),
            "mu_knots": int(args.mu_knots),
            "mu_grid": int(args.mu_grid),
            "mu_walkers": int(mu_post.meta.get("n_walkers", args.mu_walkers)),
            "mu_steps": int(mu_post.meta.get("n_steps", args.mu_steps)),
            "mu_burn": int(mu_post.meta.get("n_burn", args.mu_burn)),
            "mu_draws": int(mu_post.meta.get("draws", args.mu_draws)),
            "mu_procs": int(args.mu_procs),
            "mu_sampler": str(args.mu_sampler),
            "pt_ntemps": int(args.pt_ntemps),
            "pt_tmax": float(args.pt_tmax) if args.pt_tmax is not None else None,
            "mu_init_seed": int(args.mu_init_seed) if args.mu_init_seed is not None else None,
            "checkpoint_every": int(args.checkpoint_every),
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else None,
            "include_rsd": bool(args.include_rsd),
            "rsd_mode": str(rsd_mode),
            "fsbao_diag_cov": bool(args.fsbao_diag_cov),
            "bao_diag_cov": bool(args.bao_diag_cov),
            "drop_bao": list(args.drop_bao),
            "include_lensing": bool(args.include_lensing),
            "growth_mode": str(args.growth_mode),
            "sigma8_prior": [float(args.sigma8_prior[0]), float(args.sigma8_prior[1])],
            "lensing_mode": str(args.lensing_mode),
            "lensing_exponent": float(args.lensing_exponent),
            "lensing_sigma_scale": float(args.lensing_sigma_scale),
            "m_weight_mode": str(args.m_weight_mode),
            "gp_kernel": str(args.gp_kernel),
            "gp_procs": int(args.gp_procs),
            "spline_lambda": float(args.spline_lambda),
            "spline_bootstrap": int(args.spline_bootstrap),
        },
        "data_counts": {
            "sn_n": int(sn_like.z.size),
            "sn_binned_n": int(sn_like_bin.z.size),
            "cc_n": int(cc_like.z.size),
            "bao_n": int(sum(int(bl.z.size) for bl in bao_likes)),
            "bao_datasets": [bl.dataset for bl in bao_likes],
            "bao_datasets_dropped_for_fsbao": sorted(skip_bao_datasets) if skip_bao_datasets else [],
            "fsbao_n": int(sum(int(fl.z.size) for fl in fsbao_likes)),
            "fsbao_datasets": [fl.dataset for fl in fsbao_likes],
            "rsd_n": int(rsd_like.z.size) if rsd_like is not None else 0,
            "lensing_enabled": bool(lens_like is not None),
        },
        "posterior": {
            "H0": {
                "p16": float(np.percentile(mu_post.params["H0"], 16.0)),
                "p50": float(np.percentile(mu_post.params["H0"], 50.0)),
                "p84": float(np.percentile(mu_post.params["H0"], 84.0)),
            },
            "omega_m0": {
                "p16": float(np.percentile(mu_post.params["omega_m0"], 16.0)),
                "p50": float(np.percentile(mu_post.params["omega_m0"], 50.0)),
                "p84": float(np.percentile(mu_post.params["omega_m0"], 84.0)),
            },
            "r_d_Mpc": {
                "p16": float(np.percentile(mu_post.params["r_d_Mpc"], 16.0)),
                "p50": float(np.percentile(mu_post.params["r_d_Mpc"], 50.0)),
                "p84": float(np.percentile(mu_post.params["r_d_Mpc"], 84.0)),
            },
            "sigma_cc_jit": {
                "p16": float(np.percentile(mu_post.params["sigma_cc_jit"], 16.0)),
                "p50": float(np.percentile(mu_post.params["sigma_cc_jit"], 50.0)),
                "p84": float(np.percentile(mu_post.params["sigma_cc_jit"], 84.0)),
            },
            "sigma_sn_jit": {
                "p16": float(np.percentile(mu_post.params["sigma_sn_jit"], 16.0)),
                "p50": float(np.percentile(mu_post.params["sigma_sn_jit"], 50.0)),
                "p84": float(np.percentile(mu_post.params["sigma_sn_jit"], 84.0)),
            },
            "sigma8_0": (
                {
                    "p16": float(np.percentile(mu_post.params["sigma8_0"], 16.0)),
                    "p50": float(np.percentile(mu_post.params["sigma8_0"], 50.0)),
                    "p84": float(np.percentile(mu_post.params["sigma8_0"], 84.0)),
                }
                if "sigma8_0" in mu_post.params
                else None
            ),
            "S8": (
                {
                    "p16": float(np.percentile(mu_post.params["S8"], 16.0)),
                    "p50": float(np.percentile(mu_post.params["S8"], 50.0)),
                    "p84": float(np.percentile(mu_post.params["S8"], 84.0)),
                }
                if "S8" in mu_post.params
                else None
            ),
        },
        "departure": departure,
        "mu_sampler": {
            "acceptance_fraction_mean": mu_post.meta.get("acceptance_fraction_mean"),
            "ess_min": mu_post.meta.get("ess_min"),
            "tau_by": mu_post.meta.get("tau_by"),
            "ess_by": mu_post.meta.get("ess_by"),
            "logprob": mu_post.meta.get("logprob"),
        },
        "hz_crosscheck": {
            "gp_meta": gp_post.meta if gp_post is not None else None,
            "spline_meta": spline_post.meta if spline_post is not None else None,
        },
    }
    (report_paths.out_dir / "tables" / "summary.json").write_text(
        json.dumps(_jsonify(summary), indent=2),
        encoding="utf-8",
    )

    # --- Report markdown ---
    data_rows = [
        [
            "Pantheon+ (cosmology subset)",
            len(sn_like.z),
            f"[{sn_like.z.min():.3f}, {sn_like.z.max():.3f}]",
            "full cov (input)",
        ],
        [
            "Pantheon+ (binned for forward inference)",
            len(sn_like_bin.z),
            f"[{sn_like_bin.z.min():.3f}, {sn_like_bin.z.max():.3f}]",
            "binned (from full cov)",
        ],
        ["Cosmic chronometers (BC03_all)", len(cc_like.z), f"[{cc_like.z.min():.3f}, {cc_like.z.max():.3f}]", "diag + jitter"],
    ]
    if rsd_like is not None and rsd_like.z.size > 0:
        data_rows.append(
            [
                "RSD fσ8(z) compilation",
                int(rsd_like.z.size),
                f"[{rsd_like.z.min():.3f}, {rsd_like.z.max():.3f}]",
                "diag",
            ]
        )
    if fsbao_likes:
        for fl in fsbao_likes:
            data_rows.append(
                [
                    f"FSBAO {fl.dataset}",
                    int(fl.z.size),
                    f"[{fl.z.min():.3f}, {fl.z.max():.3f}]",
                    "full cov" if not bool(args.fsbao_diag_cov) else "diag (diagnostic)",
                ]
            )
    if lens_like is not None:
        data_rows.append(["Planck CMB lensing proxy", 1, "-", "Gaussian (compressed)"])
    bao_rows = []
    for bl in bao_likes:
        bao_rows.append([bl.dataset, len(bl.z), f"[{bl.z.min():.3f}, {bl.z.max():.3f}]", "full cov"])

    prox_rows = [
        ["BH (μ=1)", "-", "-", f"{prox['D2_mean']['bh']:.3g}", f"{logZ['bh'] - logZ_gp:.2f}"],
        [
            "Tsallis",
            f"{prox['fit_to_mean']['tsallis']['delta']:.3f}",
            "-",
            f"{prox['D2_mean']['tsallis']:.3g}",
            f"{logZ['tsallis'] - logZ_gp:.2f}",
        ],
        [
            "Barrow",
            "-",
            f"{prox['fit_to_mean']['barrow']['Delta']:.3f}",
            f"{prox['D2_mean']['barrow']:.3g}",
            f"{logZ['barrow'] - logZ_gp:.2f}",
        ],
        [
            "Kaniadakis",
            "-",
            f"β̃={prox['fit_to_mean']['kaniadakis']['beta_tilde']:.3g}",
            f"{prox['D2_mean']['kaniadakis']:.3g}",
            f"{logZ['kaniadakis'] - logZ_gp:.2f}",
        ],
    ]

    md = []
    md.append("# Nonparametric reconstruction of the horizon entropy slope μ(A)\n")
    md.append(f"_git: {git_sha} (dirty={git_dirty})_\n")
    md.append(f"_command: `{cmd}`_\n")
    md.append(f"_mapping_variant: {args.mapping_variant}_\n")
    md.append("## Lay summary\n")
    md.append(
        "We reconstruct how a putative *horizon entropy law* would need to vary with horizon area "
        "to be consistent with late-time expansion data, *given* a specific horizon-thermodynamics mapping. "
        "The reconstruction is nonparametric: we do not assume Tsallis/Barrow/Kaniadakis forms a priori; "
        "those are tested only after reconstruction.\n"
    )
    md.append("## Data\n")
    md.append(f"Selected domain: z ∈ [{z_min:.2f}, {z_max:.2f}] (automatic SN density rule)\n")
    md.append(format_table(data_rows + bao_rows, headers=["Dataset", "N", "z-range", "Covariance"]))
    md.append("\n\n## Method\n")
    if args.mapping_variant == "M2":
        area_eq = r"A(z)=4\pi\frac{c^2}{H(z)^2 - H_0^2\,\Omega_{k0}(1+z)^2}"
    else:
        area_eq = r"A(z)=4\pi\left(\frac{c}{H(z)}\right)^2"
    if args.mapping_variant == "M1":
        ode_eq = r"\frac{dH^2}{dz} = 3 H_0^2 \Omega_{m0} (1+z)^2\,\mu(A) + R(z)"
    else:
        ode_eq = r"\frac{dH^2}{dz} = 3 H_0^2 \Omega_{m0} (1+z)^2\,\mu(A)"
    md.append(
        "### Thermodynamic mapping\n"
        "We use an apparent-horizon Clausius relation (Cai–Kim style) and reconstruct the *entropy-slope "
        "modification* μ(A) ≡ (dS/dA)_BH / (dS/dA). Assuming that (ρ+p) is matter-dominated at late times, "
        "the mapping implies:\n\n"
        f"$${ode_eq},\\qquad {area_eq}.$$"
        "\n\n"
        "Standard GR with Bekenstein–Hawking entropy corresponds to μ(A)=1.\n\n"
        "Critically, we *do not* infer μ(A) by differentiating a noisy reconstruction of H(z). Instead, we "
        "treat μ(A) as a latent function and solve this equation forward as an ODE for H(z), propagating "
        "posterior uncertainty through to observables.\n"
    )
    if args.mapping_variant == "M1":
        md.append("Mapping variant note: `M1` includes a tightly-regularized closure-error term R(z).\n\n")
    if args.mapping_variant == "M2":
        md.append("Mapping variant note: `M2` uses a curved-horizon apparent-area mapping with Ωk0 sampled.\n\n")
    md.append(
        "### Forward-model inference of μ(A)\n"
        "- We parameterize log μ(x) with a natural cubic spline in x ≡ log(A/A0), with μ>0 enforced by exponentiation.\n"
        "- Priors: a weak Gaussian amplitude prior on spline knot values around log μ=0, and a smoothness prior on second differences "
        "with a log-uniform hyperprior on the curvature scale.\n"
        "- Likelihood: SN (binned using the full covariance), cosmic chronometers, and BAO. Optional external constraints can be "
        "enabled for growth (fσ8) and CMB lensing (compressed proxy).\n"
        "- We include additive jitter terms σ_cc,jit and σ_sn,jit to absorb mild unmodeled dispersion.\n"
    )
    if fsbao_likes:
        md.append(
            f"\nFSBAO enabled (`--rsd-mode {rsd_mode}`): using correlated distance+fσ8 vectors; "
            f"dropped overlapping BAO-only datasets: {sorted(skip_bao_datasets) if skip_bao_datasets else 'none'}.\n"
        )
    if rsd_like is not None and rsd_like.z.size > 0:
        md.append("\nRSD compilation enabled: diagonal fσ8(z) likelihood.\n")
    if lens_like is not None:
        md.append("\nPlanck lensing proxy enabled: Gaussian constraint on σ8 Ωm0^0.25.\n")
    if gp_post is not None and spline_post is not None:
        md.append(
            "### H(z) cross-check reconstructions\n"
            "- GP reconstruction of log H(z) at spline knots (hyperparameters marginalized).\n"
            "- Penalized spline reconstruction with bootstrap.\n"
            "These are used only as method cross-checks; μ(A) inference is done via the forward model above.\n"
        )
    md.append("\n## Results\n")
    md.append("![H(z) forward](figures/Hz_forward.png)\n\n")
    if gp_post is not None:
        md.append("![H(z) GP](figures/Hz_gp.png)\n\n")
        md.append("![H(z) spline](figures/Hz_spline.png)\n\n")
    md.append(f"![log A(z)](figures/Az_log.png)\n\n")
    md.append("![log μ(x)](figures/logmu_x.png)\n\n")
    md.append("![log μ(log A)](figures/logmu_logA.png)\n\n")
    md.append("## Mapping sensitivity\n")
    if args.run_mapping_variants and mapping_sensitivity is not None:
        md.append("![Mapping variants](figures/logmu_logA_variants.png)\n\n")
        ms_rows = []
        for name, delta_sigma, max_abs in mapping_sensitivity["deltas"]:
            ms_rows.append([name, f"{float(delta_sigma):.2f}", f"{float(max_abs):.3g}"])
        md.append(format_table(ms_rows, headers=["Variant", "Δμ / σ (rms)", "max |Δ logμ|"]) + "\n\n")
        md.append(
            "If the inferred μ(A) changes materially under these controlled mapping variants, proximity to any "
            "parametric entropy family should not be interpreted as robust.\n\n"
        )
    else:
        md.append("Not run. Re-run with `--run-mapping-variants` to quantify mapping sensitivity.\n\n")
    md.append("## Proximity tests (post hoc)\n")
    md.append(
        "Distances are computed in function space as a weighted L2 distance in logA, with weights ∝ 1/Var[log μ]. "
        "Bayes factors here are *approximate* and computed against a GP baseline in (logA, logμ) using Gaussian pseudo-data.\n"
    )
    md.append(format_table(prox_rows, headers=["Model", "δ", "Δ / β̃", "D² (mean)", "ΔlogZ vs GP"]))
    md.append("\n\n## Diagnostics\n")
    def q16_50_84(x: np.ndarray) -> tuple[float, float, float]:
        q = np.percentile(np.asarray(x, dtype=float), [16.0, 50.0, 84.0])
        return float(q[0]), float(q[1]), float(q[2])

    H0_q = q16_50_84(mu_post.params["H0"])
    Om_q = q16_50_84(mu_post.params["omega_m0"])
    rd_q = q16_50_84(mu_post.params["r_d_Mpc"])
    sigcc_q = q16_50_84(mu_post.params["sigma_cc_jit"])
    sigsn_q = q16_50_84(mu_post.params["sigma_sn_jit"])
    sigma8_q = q16_50_84(mu_post.params["sigma8_0"]) if "sigma8_0" in mu_post.params else None
    S8_q = q16_50_84(mu_post.params["S8"]) if "S8" in mu_post.params else None
    diag_lines = [
        f"- Forward inference acceptance fraction (mean): {mu_post.meta['acceptance_fraction_mean']:.3f}",
        f"- H0 [km/s/Mpc] (16/50/84): [{H0_q[0]:.2f}, {H0_q[1]:.2f}, {H0_q[2]:.2f}]",
        f"- Ωm0 (16/50/84): [{Om_q[0]:.3f}, {Om_q[1]:.3f}, {Om_q[2]:.3f}]",
        f"- r_d [Mpc] (16/50/84): [{rd_q[0]:.2f}, {rd_q[1]:.2f}, {rd_q[2]:.2f}]",
    ]
    if sigma8_q is not None:
        diag_lines.append(f"- σ8,0 (16/50/84): [{sigma8_q[0]:.3f}, {sigma8_q[1]:.3f}, {sigma8_q[2]:.3f}]")
    if S8_q is not None:
        diag_lines.append(f"- S8 (16/50/84): [{S8_q[0]:.3f}, {S8_q[1]:.3f}, {S8_q[2]:.3f}]")
    diag_lines += [
        f"- σ_cc,jit [km/s/Mpc] (16/50/84): [{sigcc_q[0]:.3g}, {sigcc_q[1]:.3g}, {sigcc_q[2]:.3g}]",
        f"- σ_sn,jit [mag] (16/50/84): [{sigsn_q[0]:.3g}, {sigsn_q[1]:.3g}, {sigsn_q[2]:.3g}]",
        f"- GP baseline evidence kernel: matern32 (amp={gp_fit['amp']:.3g}, ell={gp_fit['ell']:.3g})",
    ]
    md.append("\n".join(diag_lines) + "\n")
    md.append("\n## Notes and limitations\n")
    md.append(
        "- The μ(A) inversion uses the late-time matter-dominance approximation for (ρ+p). Results should be interpreted as "
        "\"the entropy-slope modification required under this mapping\".\n"
        "- Full prior/kernel ablations and synthetic closure tests are available via `scripts/run_ablation_suite.py` and "
        "`scripts/run_synthetic_closure.py`.\n"
    )

    if ablations:
        md.append("\n## Ablations (quick)\n")
        ab_rows = []
        for a in ablations:
            d2 = a["D2_mean"]
            ab_rows.append(
                [
                    a["name"],
                    f"{d2['bh']:.3g}",
                    f"{d2['tsallis']:.3g}",
                    f"{d2['barrow']:.3g}",
                    f"{d2['kaniadakis']:.3g}",
                ]
            )
        md.append(format_table(ab_rows, headers=["Case", "D² BH", "D² Tsallis", "D² Barrow", "D² Kaniadakis"]))

    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    print(f"Wrote {report_paths.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
