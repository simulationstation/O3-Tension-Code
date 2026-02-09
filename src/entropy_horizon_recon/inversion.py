from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np

from .constants import PhysicalConstants
from .cosmology import build_background_from_H_grid
from .forward_model import ForwardModel, make_spline
from .mapping import fit_logmu_spline, mapping_from_H_draw
from .transforms import bounded_from_unbounded, sigmoid, unbounded_from_bounded


# For multiprocessing+emcee: pool.map must pickle the log-prob callable. Our log-prob
# is defined as a closure inside infer_logmu_forward, so we stash it in a global and
# pass a module-level wrapper (works with the "fork" start method).
_GLOBAL_LOG_PROB = None
_GLOBAL_LOG_LIKE = None
_GLOBAL_LOG_PRIOR = None


def _log_prob_wrapper(theta: np.ndarray) -> float:
    fn = _GLOBAL_LOG_PROB
    if fn is None:
        raise RuntimeError("Global log-prob function not set (requires fork start method).")
    return float(fn(theta))


def _log_like_wrapper(theta: np.ndarray) -> float:
    fn = _GLOBAL_LOG_LIKE
    if fn is None:
        raise RuntimeError("Global log-like function not set (requires fork start method).")
    return float(fn(theta))


def _log_prior_wrapper(theta: np.ndarray) -> float:
    fn = _GLOBAL_LOG_PRIOR
    if fn is None:
        raise RuntimeError("Global log-prior function not set (requires fork start method).")
    return float(fn(theta))


@dataclass(frozen=True)
class ForwardMuPosterior:
    x_grid: np.ndarray
    logmu_x_samples: np.ndarray
    z_grid: np.ndarray
    H_samples: np.ndarray
    params: dict[str, np.ndarray]
    meta: dict

    @property
    def mu_x_samples(self) -> np.ndarray:
        return np.exp(self.logmu_x_samples)


@dataclass(frozen=True)
class MuAPosterior:
    logA_grid: np.ndarray
    logmu_samples: np.ndarray
    omega_m0_samples: np.ndarray

    @property
    def mu_samples(self) -> np.ndarray:
        return np.exp(self.logmu_samples)


def reconstruct_logmu_of_logA(
    *,
    z: np.ndarray,
    H_samples: np.ndarray,
    dH_dz_samples: np.ndarray,
    constants: PhysicalConstants,
    omega_m0_samples: np.ndarray,
    logA_grid: np.ndarray,
) -> MuAPosterior:
    """Second-stage inversion: map H(z) posterior draws to a posterior over logμ(logA).

    Each draw is smoothed by fitting an interpolating spline in (logA, logμ).
    """
    if H_samples.shape != dH_dz_samples.shape:
        raise ValueError("H_samples and dH_dz_samples must have the same shape.")
    n_draws, n_z = H_samples.shape
    if z.shape != (n_z,):
        raise ValueError("z must have shape (n_z,).")
    if omega_m0_samples.shape != (n_draws,):
        raise ValueError("omega_m0_samples must have shape (n_draws,).")
    if logA_grid.ndim != 1:
        raise ValueError("logA_grid must be 1D.")

    logmu_grid = np.empty((n_draws, len(logA_grid)))
    for i in range(n_draws):
        m = mapping_from_H_draw(
            z,
            H_samples[i],
            dH_dz_samples[i],
            omega_m0=float(omega_m0_samples[i]),
            constants=constants,
        )
        mu = m.mu
        A = m.A
        ok = (mu > 0) & np.isfinite(mu) & np.isfinite(A) & (A > 0)
        if ok.sum() < 6:
            raise RuntimeError("Too few physical μ(A) points in a draw; check priors or smoothing.")
        logA = np.log(A[ok])
        logmu = np.log(mu[ok])
        spline = fit_logmu_spline(logA, logmu)
        logmu_grid[i] = spline(logA_grid)

    return MuAPosterior(
        logA_grid=np.asarray(logA_grid, dtype=float),
        logmu_samples=logmu_grid,
        omega_m0_samples=np.asarray(omega_m0_samples, dtype=float),
    )


def infer_logmu_forward(
    *,
    z_grid: np.ndarray,
    x_knots: np.ndarray,
    x_grid: np.ndarray,
    sn_z: np.ndarray,
    sn_m: np.ndarray,
    sn_cov: np.ndarray,
    sn_marg: str = "M",
    sn_mstep_z: float = 0.15,
    cc_z: np.ndarray,
    cc_H: np.ndarray,
    cc_sigma_H: np.ndarray,
    bao_likes,
    fsbao_likes=None,
    rsd_like=None,
    lensing_like=None,
    pk_like=None,
    constants: PhysicalConstants,
    sampler_kind: str = "emcee",
    pt_ntemps: int = 8,
    pt_tmax: float | None = None,
    n_walkers: int = 64,
    n_steps: int = 1500,
    n_burn: int = 500,
    seed: int = 0,
    init_seed: int | None = None,
    n_processes: int = 1,
    omega_m0_prior: tuple[float, float] = (0.15, 0.5),
    omega_m0_fixed: float | None = None,
    omega_k0_prior: tuple[float, float] | None = None,
    r_d_fixed: float | None = None,
    use_residual: bool = False,
    H0_prior: tuple[float, float] = (40.0, 100.0),
    r_d_prior: tuple[float, float] = (120.0, 170.0),
    sigma8_prior: tuple[float, float] = (0.6, 1.0),
    pk_bias_prior: tuple[float, float] = (0.5, 4.0),
    pk_noise_prior: tuple[float, float] = (0.0, 1.0e5),
    sigma_cc_jit_scale: float = 10.0,
    sigma_sn_jit_scale: float = 0.05,
    logmu_knot_scale: float = 1.0,
    log_sigma_d2_prior: tuple[float, float] = (-12.0, 3.0),
    sigma_d2_scale: float = 0.23,
    fixed_logmu_knots: np.ndarray | str | None = None,
    n_draws: int = 800,
    progress: bool = True,
    max_rss_mb: float | None = None,
    maxrss_check_every: int = 50,
    debug_log_path: str | Path | None = None,
    timing_log_path: str | Path | None = None,
    timing_every: int = 200,
    save_chain_path: str | Path | None = None,
    checkpoint_every: int = 0,
    checkpoint_path: str | Path | None = None,
) -> ForwardMuPosterior:
    """Infer log μ(A) by forward-modeling H(z) via integral/ODE mapping.

    This samples a spline representation of logμ(x) with x = log(A/A0) (A0 at z=0).
    """
    import emcee
    import resource
    import sys

    from scipy.interpolate import CubicSpline

    def _safe_log(msg: str) -> None:
        try:
            print(str(msg), flush=True)
        except BrokenPipeError:
            # If stdout is a broken pipe (e.g. a closed `tee`), keep the run alive.
            return

    # emcee uses tqdm for the progress bar. When stdout/stderr is a pipe (e.g. `... | tee`),
    # the reader can disappear, which raises BrokenPipeError and kills the run unless handled.
    progress = bool(progress)
    try:
        if progress and hasattr(sys.stderr, "isatty") and not sys.stderr.isatty():
            progress = False
    except Exception:
        # If isatty() fails for any reason, just keep the user-provided setting.
        pass

    rng = np.random.default_rng(seed)
    rng_init = rng if init_seed is None else np.random.default_rng(int(init_seed))

    z_grid = np.asarray(z_grid, dtype=float)
    if z_grid.ndim != 1 or z_grid.size < 10 or z_grid[0] != 0.0 or np.any(np.diff(z_grid) <= 0):
        raise ValueError("z_grid must be 1D, start at 0, and be strictly increasing.")
    x_knots = np.asarray(x_knots, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if np.any(np.diff(x_knots) <= 0):
        raise ValueError("x_knots must be strictly increasing.")
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be 1D.")

    # Precompute SN eigen decomposition for efficient diagonal-jitter evaluation
    sn_z = np.asarray(sn_z, dtype=float)
    sn_m = np.asarray(sn_m, dtype=float)
    sn_cov = np.asarray(sn_cov, dtype=float)
    if sn_cov.shape != (sn_z.size, sn_z.size):
        raise ValueError("sn_cov shape mismatch.")
    sn_enabled = sn_z.size > 0
    if sn_enabled:
        sn_cov_diag = np.diag(sn_cov)
        sn_cov_diag_ok = bool(np.all(np.isfinite(sn_cov_diag)) and np.all(sn_cov_diag > 0.0))
        # eigendecomposition (small if using binned SN)
        try:
            evals, evecs = np.linalg.eigh(sn_cov)
            sn_cov_pd = bool(np.all(np.isfinite(evals)) and np.all(evals > 0.0))
        except np.linalg.LinAlgError:
            evals = np.full(sn_z.size, np.nan)
            evecs = np.eye(sn_z.size)
            sn_cov_pd = False
        ones_sn = np.ones_like(sn_m)
        Qt_ones = evecs.T @ ones_sn

        sn_marg_norm = str(sn_marg).strip()
        if sn_marg_norm not in {"M", "Mz", "Mstep"}:
            raise ValueError("sn_marg must be one of: M, Mz, Mstep")
        cols = [ones_sn]
        col_names = ["M0"]
        if sn_marg_norm == "Mz":
            zc = sn_z - float(np.mean(sn_z))
            cols.append(zc)
            col_names.append("Mz_slope")
        elif sn_marg_norm == "Mstep":
            z0 = float(sn_mstep_z)
            if not np.isfinite(z0):
                raise ValueError("sn_mstep_z must be finite.")
            step = (sn_z >= z0).astype(float)
            # If the step never turns on/off in the selected z-range, the design is singular.
            if np.all(step == step[0]):
                raise ValueError("sn_marg=Mstep but step column is constant over sn_z; adjust sn_mstep_z or z-range.")
            cols.append(step)
            col_names.append(f"Mstep_ge_{z0:g}")
        X_sn = np.column_stack(cols)
        Qt_X_sn = evecs.T @ X_sn
    else:
        sn_cov_diag_ok = True
        sn_cov_pd = True
        evals = np.zeros((0,), dtype=float)
        evecs = np.zeros((0, 0), dtype=float)
        ones_sn = np.zeros((0,), dtype=float)
        Qt_ones = np.zeros((0,), dtype=float)
        sn_marg_norm = "M"
        col_names = []
        Qt_X_sn = np.zeros((0, 0), dtype=float)

    # Chronometers
    cc_z = np.asarray(cc_z, dtype=float)
    cc_H = np.asarray(cc_H, dtype=float)
    cc_sigma_H = np.asarray(cc_sigma_H, dtype=float)
    if cc_z.shape != cc_H.shape or cc_z.shape != cc_sigma_H.shape:
        raise ValueError("Chronometer arrays shape mismatch.")
    cc_sigma_ok = bool(np.all(np.isfinite(cc_sigma_H)) and np.all(cc_sigma_H > 0.0))

    forward = ForwardModel(constants=constants, x_knots=x_knots)

    bao_cov_diag_ok = True
    bao_chol_ok = True
    for bl in bao_likes:
        diag = np.diag(bl.cov)
        if not (np.all(np.isfinite(diag)) and np.all(diag > 0.0)):
            bao_cov_diag_ok = False
        cho_diag = np.diag(bl.cov_cho[0])
        if not (np.all(np.isfinite(cho_diag)) and np.all(cho_diag > 0.0)):
            bao_chol_ok = False

    fsbao_cov_diag_ok = True
    fsbao_chol_ok = True
    if fsbao_likes is None:
        fsbao_likes = []
    for fl in fsbao_likes:
        diag = np.diag(fl.cov)
        if not (np.all(np.isfinite(diag)) and np.all(diag > 0.0)):
            fsbao_cov_diag_ok = False
        cho_diag = np.diag(fl.cov_cho[0])
        if not (np.all(np.isfinite(cho_diag)) and np.all(cho_diag > 0.0)):
            fsbao_chol_ok = False

    n_mu_knots = int(len(x_knots))
    fixed_logmu = None
    if fixed_logmu_knots is not None:
        if isinstance(fixed_logmu_knots, str):
            mode = fixed_logmu_knots.strip().lower()
            if mode in {"bh", "bekenstein-hawking", "mu=1", "mu1", "mu_1"}:
                fixed_logmu = np.zeros(int(n_mu_knots), dtype=float)
            else:
                raise ValueError(f"Unsupported fixed_logmu_knots mode: {fixed_logmu_knots!r}")
        else:
            arr = np.asarray(fixed_logmu_knots, dtype=float)
            if arr.shape != (int(n_mu_knots),):
                raise ValueError(f"fixed_logmu_knots must have shape ({int(n_mu_knots)},), got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError("fixed_logmu_knots must be finite.")
            fixed_logmu = arr
    mu_is_fixed = fixed_logmu is not None
    n_mu_params = 0 if mu_is_fixed else int(n_mu_knots)
    if omega_m0_fixed is not None:
        omega_m0_fixed = float(omega_m0_fixed)
        if not (0.0 < omega_m0_fixed < 1.0):
            raise ValueError("omega_m0_fixed must be in (0,1).")
    free_omega_m0 = omega_m0_fixed is None
    free_omega_k0 = omega_k0_prior is not None
    if r_d_fixed is not None:
        r_d_fixed = float(r_d_fixed)
        if not np.isfinite(r_d_fixed) or r_d_fixed <= 0.0:
            raise ValueError("r_d_fixed must be finite and positive.")
        if not (r_d_prior[0] <= r_d_fixed <= r_d_prior[1]):
            raise ValueError("r_d_fixed must lie within r_d_prior bounds.")
    free_r_d = r_d_fixed is None
    # Full-shape P(k) also requires sigma8_0 to be a sampled parameter (for CAMB).
    include_sigma8 = bool(
        (rsd_like is not None)
        or (lensing_like is not None)
        or (fsbao_likes is not None and len(fsbao_likes) > 0)
        or (pk_like is not None)
    )

    if pk_like is None:
        pk_likes = []
    elif isinstance(pk_like, (list, tuple)):
        pk_likes = list(pk_like)
    else:
        pk_likes = [pk_like]

    # Optional residual closure term R(z) parameterization.
    use_residual = bool(use_residual)
    if use_residual:
        # Fixed knot count to enforce long length-scale.
        z_r_knots = np.linspace(0.0, float(z_grid[-1]), 5)
        n_r = int(z_r_knots.size)
        z_step = float(np.mean(np.diff(z_r_knots)))
        if not np.isfinite(z_step) or z_step <= 0:
            raise ValueError("Invalid residual knot spacing.")
        sigma_r0 = 0.05
        sigma_r_d2 = 0.02
    else:
        z_r_knots = None
        n_r = 0
        z_step = np.nan
        sigma_r0 = np.nan
        sigma_r_d2 = np.nan

    if not (H0_prior[1] > H0_prior[0]):
        raise ValueError("H0_prior must satisfy hi > lo.")
    if not (r_d_prior[1] > r_d_prior[0]):
        raise ValueError("r_d_prior must satisfy hi > lo.")
    if include_sigma8:
        if not (sigma8_prior[1] > sigma8_prior[0]):
            raise ValueError("sigma8_prior must satisfy hi > lo.")
        if not (np.isfinite(sigma8_prior[0]) and np.isfinite(sigma8_prior[1]) and sigma8_prior[0] > 0):
            raise ValueError("sigma8_prior bounds must be finite and positive.")
    if free_omega_m0:
        if not (omega_m0_prior[1] > omega_m0_prior[0]):
            raise ValueError("omega_m0_prior must satisfy hi > lo.")
        if not (omega_m0_prior[0] < 1.0 and omega_m0_prior[1] > 0.0):
            raise ValueError("omega_m0_prior must overlap (0,1).")
        om_lo_eff = max(float(omega_m0_prior[0]), 1e-6)
        om_hi_eff = min(float(omega_m0_prior[1]), 1.0 - 1e-6)
        if not (om_hi_eff > om_lo_eff):
            raise ValueError("omega_m0_prior must have non-empty intersection with (0,1).")
    else:
        om_lo_eff = np.nan
        om_hi_eff = np.nan
    if free_omega_k0:
        if not (omega_k0_prior[1] > omega_k0_prior[0]):
            raise ValueError("omega_k0_prior must satisfy hi > lo.")
        ok_lo_eff = float(omega_k0_prior[0])
        ok_hi_eff = min(float(omega_k0_prior[1]), 1.0 - 1e-6)
        if not (ok_hi_eff > ok_lo_eff):
            raise ValueError("omega_k0_prior upper bound must be < 1.")
    else:
        ok_lo_eff = np.nan
        ok_hi_eff = np.nan

    # Parameter layout (keep explicit for diagnostics).
    pos = 0
    sl_logmu = slice(pos, pos + n_mu_params)
    pos += n_mu_params
    idx_u_H0 = pos
    pos += 1
    idx_u_omega_m0 = None
    if free_omega_m0:
        idx_u_omega_m0 = pos
        pos += 1
    idx_u_omega_k0 = None
    if free_omega_k0:
        idx_u_omega_k0 = pos
        pos += 1
    idx_u_r_d = None
    if free_r_d:
        idx_u_r_d = pos
        pos += 1
    idx_u_sigma8 = None
    if include_sigma8:
        idx_u_sigma8 = pos
        pos += 1

    idx_u_pk_b1: list[int] = []
    idx_u_pk_noise: list[int] = []
    if pk_likes:
        for _ in pk_likes:
            idx_u_pk_b1.append(pos)
            pos += 1
            idx_u_pk_noise.append(pos)
            pos += 1
    idx_log_sig_cc = pos
    pos += 1
    idx_log_sig_sn = pos
    pos += 1
    idx_log_sig_d2 = pos
    pos += 1
    sl_r_knots = None
    if use_residual:
        sl_r_knots = slice(pos, pos + n_r)
        pos += n_r
    ndim = int(pos)
    param_names: list[str] = []
    if not mu_is_fixed:
        param_names += [f"logmu_{i}" for i in range(n_mu_knots)]
    param_names.append("u_H0")
    if idx_u_omega_m0 is not None:
        param_names.append("u_omega_m0")
    if idx_u_omega_k0 is not None:
        param_names.append("u_omega_k0")
    if idx_u_r_d is not None:
        param_names.append("u_r_d")
    if idx_u_sigma8 is not None:
        param_names.append("u_sigma8")
    if idx_u_pk_b1:
        for i in range(len(idx_u_pk_b1)):
            param_names.append(f"u_pk_b1_{i}")
            param_names.append(f"u_pk_noise_{i}")
    param_names += ["log_sig_cc", "log_sig_sn", "log_sig_d2"]
    if sl_r_knots is not None:
        param_names += [f"r_{i}" for i in range(n_r)]
    dx = np.diff(x_knots)
    if np.any(dx <= 0):
        raise ValueError("x_knots must be strictly increasing.")
    # For our default linspace knots this is constant; use the mean for robustness.
    x_step = float(np.mean(dx))
    if not np.isfinite(x_step) or x_step <= 0:
        raise ValueError("Invalid x_knots spacing.")

    def maxrss_mb() -> float:
        """Return the process max RSS in MB (portable across Linux/macOS)."""
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux: KB, macOS: bytes
        if rss > 1e9:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    def maybe_check_maxrss() -> None:
        if max_rss_mb is None:
            return
        if max_rss_mb <= 0:
            return
        rss_now = maxrss_mb()
        if rss_now > float(max_rss_mb):
            raise MemoryError(f"RSS {rss_now:.1f} MB exceeded limit {float(max_rss_mb):.1f} MB.")

    # Hard memory cap (best-effort) to avoid host OOM / kernel resets.
    old_rlimit_as: tuple[int, int] | None = None
    if max_rss_mb is not None and max_rss_mb > 0:
        try:
            old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
            old_rlimit_as = (int(old_soft), int(old_hard))
            limit_bytes = int(float(max_rss_mb) * 1024.0 * 1024.0)
            # RLIMIT_AS caps *virtual* memory, which can be several GB after importing
            # numpy/scipy due to large shared library mappings. If the requested cap is
            # below the current VmSize, setting it would make essentially all future
            # allocations fail. In that case, skip RLIMIT_AS and rely on the RSS watchdog.
            vm_size_bytes = None
            try:
                with open("/proc/self/status", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmSize:"):
                            vm_size_bytes = int(line.split()[1]) * 1024
                            break
            except Exception:
                vm_size_bytes = None

            if vm_size_bytes is not None and int(vm_size_bytes) >= int(limit_bytes):
                old_rlimit_as = None
            else:
                # RLIMIT_AS caps total virtual memory (soft limit). Do not lower the hard limit
                # so we can restore the original values when infer_logmu_forward returns.
                new_soft = limit_bytes
                if old_hard != resource.RLIM_INFINITY:
                    new_soft = min(int(limit_bytes), int(old_hard))
                resource.setrlimit(resource.RLIMIT_AS, (int(new_soft), int(old_hard)))
        except Exception:
            # If not supported (or permission denied), fall back to the soft RSS checks above.
            old_rlimit_as = None

    debug_log_limit = 20
    debug_log_disabled = False
    last_detail: dict | None = None
    if debug_log_path is not None:
        debug_log_path = Path(debug_log_path)
    if timing_log_path is not None:
        timing_log_path = Path(timing_log_path)
        timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        timing_log_path.write_text("", encoding="utf-8")

    def log_invalid(theta: np.ndarray, reason: str, detail: dict | None = None) -> None:
        nonlocal debug_log_disabled
        if debug_log_path is None or debug_log_disabled:
            return
        try:
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a+", encoding="utf-8") as f:
                try:
                    import fcntl

                    fcntl.flock(f, fcntl.LOCK_EX)
                except Exception:
                    pass
                f.seek(0)
                count = 0
                for _ in f:
                    count += 1
                    if count >= debug_log_limit:
                        break
                if count >= debug_log_limit:
                    debug_log_disabled = True
                    return
                entry = {"reason": str(reason), "theta": np.asarray(theta, dtype=float).tolist()}
                if detail is not None:
                    entry["detail"] = detail
                f.write(json.dumps(entry) + "\n")
        except Exception:
            return

    logprob_counts_valid = bool(n_processes is None or n_processes <= 1)
    logprob_total_calls = 0
    logprob_invalid_calls = 0
    invalid_reason_counts: dict[str, int] = {}

    timing_enabled = timing_log_path is not None
    timing_calls = 0
    timing_acc = {
        "total": 0.0,
        "cc": 0.0,
        "bao": 0.0,
        "fsbao": 0.0,
        "sn": 0.0,
        "rsd": 0.0,
        "lensing": 0.0,
        "pk": 0.0,
    }

    def _timing_flush() -> None:
        nonlocal timing_calls
        if not timing_enabled or timing_log_path is None:
            return
        if timing_calls <= 0:
            return
        payload = {"calls": int(timing_calls)}
        for k, v in timing_acc.items():
            payload[f"{k}_avg_s"] = float(v / timing_calls)
        with open(timing_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        timing_calls = 0
        for k in timing_acc:
            timing_acc[k] = 0.0

    def record_invalid(reason: str) -> None:
        nonlocal logprob_invalid_calls
        if not logprob_counts_valid:
            return
        logprob_invalid_calls += 1
        key = str(reason) if reason else "unknown"
        invalid_reason_counts[key] = int(invalid_reason_counts.get(key, 0)) + 1

    def log_halfnormal(sigma: float, scale: float) -> float:
        if sigma < 0 or not np.isfinite(sigma):
            return -np.inf
        return -0.5 * (sigma / scale) ** 2 - np.log(scale) - 0.5 * np.log(np.pi / 2.0)

    def bounded_from_u(u: float, lo: float, hi: float) -> tuple[float, float]:
        """Return bounded x and log prior (uniform in x) induced by u."""
        if not np.isfinite(u):
            return np.nan, -np.inf
        x, log_jac = bounded_from_unbounded(float(u), float(lo), float(hi))
        if not (np.isfinite(x) and np.isfinite(log_jac)):
            return np.nan, -np.inf
        logp = float(log_jac - np.log(float(hi) - float(lo)))
        return float(x), float(logp)

    def sn_loglike(mu0: np.ndarray, sigma_jit: float) -> tuple[float, float, np.ndarray]:
        """SN loglike with linear nuisance marginalization using eigen-decomp for C + sigma_jit^2 I.

        Model: m_model = mu0 + X beta, where X columns are controlled by sn_marg:
        - M:    X = [1]
        - Mz:   X = [1, z - mean(z)]
        - Mstep:X = [1, 1(z>=z_step)]

        We analytically marginalize over beta with an (improper) flat prior, yielding:
          logL = -0.5 (chi2 + logdet(C) + logdet(X^T C^{-1} X))
        """
        if not sn_enabled:
            return 0.0, np.nan, np.zeros((0,), dtype=float)
        if sigma_jit < 0 or not np.isfinite(sigma_jit):
            return -np.inf, np.nan, np.full((0,), np.nan)
        r = sn_m - mu0
        lam = evals + sigma_jit**2
        if np.any(~np.isfinite(lam)) or np.any(lam <= 0.0):
            return -np.inf, np.nan, np.full((Qt_X_sn.shape[1],), np.nan)
        inv_lam = 1.0 / lam
        Qt_r = evecs.T @ r

        # GLS projection for arbitrary small design matrix X (in eigen basis).
        XtCinvX = Qt_X_sn.T @ (inv_lam[:, None] * Qt_X_sn)
        XtCinvR = Qt_X_sn.T @ (inv_lam * Qt_r)
        try:
            chol = np.linalg.cholesky(XtCinvX)
            beta_hat = np.linalg.solve(chol.T, np.linalg.solve(chol, XtCinvR))
            logdet_xtcinvx = 2.0 * float(np.sum(np.log(np.diag(chol))))
        except np.linalg.LinAlgError:
            return -np.inf, np.nan, np.full((Qt_X_sn.shape[1],), np.nan)

        Qt_r2 = Qt_r - Qt_X_sn @ beta_hat
        chi2 = float(np.sum((Qt_r2**2) * inv_lam))
        logdet = float(np.sum(np.log(lam)))
        ll = -0.5 * (chi2 + logdet + logdet_xtcinvx)
        M_hat = float(beta_hat[0]) if beta_hat.size > 0 else np.nan
        return ll, M_hat, np.asarray(beta_hat, dtype=float)

    def log_prior(theta: np.ndarray) -> tuple[float, str | None]:
        if not np.all(np.isfinite(theta)):
            return -np.inf, "theta_nonfinite"
        logmu_knots = fixed_logmu if mu_is_fixed else theta[sl_logmu]
        u_H0 = float(theta[idx_u_H0])
        u_omega_m0 = float(theta[idx_u_omega_m0]) if idx_u_omega_m0 is not None else np.nan
        u_omega_k0 = float(theta[idx_u_omega_k0]) if idx_u_omega_k0 is not None else np.nan
        u_r_d = float(theta[idx_u_r_d]) if idx_u_r_d is not None else np.nan
        u_sigma8 = float(theta[idx_u_sigma8]) if idx_u_sigma8 is not None else np.nan
        u_pk_b1 = [float(theta[i]) for i in idx_u_pk_b1] if idx_u_pk_b1 else []
        u_pk_noise = [float(theta[i]) for i in idx_u_pk_noise] if idx_u_pk_noise else []
        log_sig_cc = float(theta[idx_log_sig_cc])
        log_sig_sn = float(theta[idx_log_sig_sn])
        log_sig_d2 = float(theta[idx_log_sig_d2])
        r_knots = theta[sl_r_knots] if sl_r_knots is not None else None

        if not np.all(np.isfinite(logmu_knots)):
            return -np.inf, "logmu_nonfinite"
        if r_knots is not None and not np.all(np.isfinite(r_knots)):
            return -np.inf, "residual_knots_nonfinite"

        H0, lp_H0 = bounded_from_u(u_H0, H0_prior[0], H0_prior[1])
        if not np.isfinite(H0):
            return -np.inf, "H0_nonfinite"
        lp = 0.0
        lp += lp_H0

        if idx_u_omega_m0 is not None:
            omega_m0, lp_om = bounded_from_u(u_omega_m0, om_lo_eff, om_hi_eff)
            if not np.isfinite(omega_m0):
                return -np.inf, "omega_m0_nonfinite"
            lp += lp_om
        else:
            omega_m0 = float(omega_m0_fixed)
        if not (0.0 < omega_m0 < 1.0):
            return -np.inf, "omega_m0_unit_interval"

        if idx_u_omega_k0 is not None:
            omega_k0, lp_ok = bounded_from_u(u_omega_k0, ok_lo_eff, ok_hi_eff)
            if not np.isfinite(omega_k0):
                return -np.inf, "omega_k0_nonfinite"
            lp += lp_ok
        else:
            omega_k0 = 0.0
        # require A0 mapping denominator positive: 1 - omega_k0 > 0
        if not (omega_k0 < 1.0):
            return -np.inf, "omega_k0_lt1"

        if idx_u_r_d is not None:
            r_d, lp_rd = bounded_from_u(u_r_d, r_d_prior[0], r_d_prior[1])
            if not np.isfinite(r_d):
                return -np.inf, "r_d_nonfinite"
            lp += lp_rd
        else:
            r_d = float(r_d_fixed) if r_d_fixed is not None else np.nan
            if not np.isfinite(r_d) or r_d <= 0.0:
                return -np.inf, "r_d_nonfinite"

        if idx_u_sigma8 is not None:
            sigma8_0, lp_s8 = bounded_from_u(u_sigma8, sigma8_prior[0], sigma8_prior[1])
            if not np.isfinite(sigma8_0):
                return -np.inf, "sigma8_nonfinite"
            lp += lp_s8
        else:
            sigma8_0 = np.nan

        pk_b1 = []
        pk_noise = []
        for ub1, un in zip(u_pk_b1, u_pk_noise):
            b1, lp_b1 = bounded_from_u(ub1, pk_bias_prior[0], pk_bias_prior[1])
            pn, lp_pn = bounded_from_u(un, pk_noise_prior[0], pk_noise_prior[1])
            if not (np.isfinite(b1) and np.isfinite(pn)):
                return -np.inf, "pk_params_nonfinite"
            pk_b1.append(b1)
            pk_noise.append(pn)
            lp += lp_b1 + lp_pn

        sig_cc = float(np.exp(log_sig_cc))
        sig_sn = float(np.exp(log_sig_sn))
        sig_d2 = float(np.exp(log_sig_d2))
        if not (np.isfinite(sig_cc) and np.isfinite(sig_sn) and np.isfinite(sig_d2)):
            return -np.inf, "sigma_nonfinite"

        lp += log_halfnormal(sig_cc, sigma_cc_jit_scale) + log_sig_cc  # Jacobian
        lp += log_halfnormal(sig_sn, sigma_sn_jit_scale) + log_sig_sn  # Jacobian
        if not (log_sigma_d2_prior[0] <= log_sig_d2 <= log_sigma_d2_prior[1]):
            return -np.inf, "log_sigma_d2_bounds"
        lp += log_halfnormal(sig_d2, sigma_d2_scale) + log_sig_d2  # Jacobian

        if not mu_is_fixed:
            # Weak amplitude prior around BH (logμ=0), broad.
            lp += -0.5 * float(np.sum((logmu_knots / logmu_knot_scale) ** 2)) - n_mu_knots * np.log(
                logmu_knot_scale * np.sqrt(2.0 * np.pi)
            )

            # Smoothness prior on curvature (discrete 2nd derivative).
            #
            # IMPORTANT: scale by x_step**2 so the prior is approximately invariant to the
            # number of knots. Without this, adding knots weakens the curvature penalty and
            # can yield over-flexible (wiggly) μ(A) posteriors.
            d2_raw = np.diff(logmu_knots, n=2)
            d2_scaled = d2_raw / (x_step**2)
            lp += -0.5 * float(np.sum((d2_scaled / sig_d2) ** 2)) - (n_mu_knots - 2) * np.log(
                sig_d2 * np.sqrt(2.0 * np.pi)
            )

        # Optional residual R(z) knots with strong long-lengthscale prior.
        if r_knots is not None:
            lp += -0.5 * float(np.sum((r_knots / sigma_r0) ** 2)) - n_r * np.log(
                sigma_r0 * np.sqrt(2.0 * np.pi)
            )
            d2r = np.diff(r_knots, n=2)
            d2r_scaled = d2r / (z_step**2)
            lp += -0.5 * float(np.sum((d2r_scaled / sigma_r_d2) ** 2)) - (n_r - 2) * np.log(
                sigma_r_d2 * np.sqrt(2.0 * np.pi)
            )

        if not np.isfinite(lp):
            return -np.inf, "prior_nonfinite"
        return lp, None

    def log_likelihood(theta: np.ndarray, timing_acc: dict[str, float] | None = None) -> tuple[float, dict, str | None]:
        nonlocal last_detail
        if not np.all(np.isfinite(theta)):
            return -np.inf, {}, "theta_nonfinite"
        if not sn_cov_diag_ok:
            return -np.inf, {}, "sn_cov_diag_nonpositive"
        if not sn_cov_pd:
            return -np.inf, {}, "sn_cov_not_pd"
        if not cc_sigma_ok:
            return -np.inf, {}, "cc_sigma_nonpositive"
        if not bao_cov_diag_ok:
            return -np.inf, {}, "bao_cov_diag_nonpositive"
        if not bao_chol_ok:
            return -np.inf, {}, "bao_cholesky_invalid"
        if not fsbao_cov_diag_ok:
            return -np.inf, {}, "fsbao_cov_diag_nonpositive"
        if not fsbao_chol_ok:
            return -np.inf, {}, "fsbao_cholesky_invalid"
        logmu_knots = fixed_logmu if mu_is_fixed else theta[sl_logmu]
        u_H0 = float(theta[idx_u_H0])
        u_omega_m0 = float(theta[idx_u_omega_m0]) if idx_u_omega_m0 is not None else np.nan
        u_omega_k0 = float(theta[idx_u_omega_k0]) if idx_u_omega_k0 is not None else np.nan
        u_r_d = float(theta[idx_u_r_d]) if idx_u_r_d is not None else np.nan
        u_sigma8 = float(theta[idx_u_sigma8]) if idx_u_sigma8 is not None else np.nan
        u_pk_b1 = [float(theta[i]) for i in idx_u_pk_b1] if idx_u_pk_b1 else []
        u_pk_noise = [float(theta[i]) for i in idx_u_pk_noise] if idx_u_pk_noise else []
        log_sig_cc = float(theta[idx_log_sig_cc])
        log_sig_sn = float(theta[idx_log_sig_sn])
        _log_sig_d2 = float(theta[idx_log_sig_d2])
        r_knots = theta[sl_r_knots] if sl_r_knots is not None else None
        H0, _ = bounded_from_u(u_H0, H0_prior[0], H0_prior[1])
        if idx_u_omega_m0 is not None:
            omega_m0, _ = bounded_from_u(u_omega_m0, om_lo_eff, om_hi_eff)
        else:
            omega_m0 = float(omega_m0_fixed)
        if idx_u_omega_k0 is not None:
            omega_k0, _ = bounded_from_u(u_omega_k0, ok_lo_eff, ok_hi_eff)
        else:
            omega_k0 = 0.0
        if idx_u_r_d is not None:
            r_d, _ = bounded_from_u(u_r_d, r_d_prior[0], r_d_prior[1])
        else:
            r_d = float(r_d_fixed) if r_d_fixed is not None else np.nan
        if idx_u_sigma8 is not None:
            sigma8_0, _ = bounded_from_u(u_sigma8, sigma8_prior[0], sigma8_prior[1])
        else:
            sigma8_0 = np.nan

        pk_b1 = []
        pk_noise = []
        for ub1, un in zip(u_pk_b1, u_pk_noise):
            b1, _ = bounded_from_u(ub1, pk_bias_prior[0], pk_bias_prior[1])
            pn, _ = bounded_from_u(un, pk_noise_prior[0], pk_noise_prior[1])
            pk_b1.append(b1)
            pk_noise.append(pn)
        sig_cc = float(np.exp(log_sig_cc))
        sig_sn = float(np.exp(log_sig_sn))
        if not np.all(np.isfinite(logmu_knots)):
            return -np.inf, {}, "logmu_nonfinite"
        if r_knots is not None and not np.all(np.isfinite(r_knots)):
            return -np.inf, {}, "residual_knots_nonfinite"
        if not np.isfinite(H0) or H0 <= 0.0:
            return -np.inf, {}, "H0_nonfinite"
        if not np.isfinite(omega_m0):
            return -np.inf, {}, "omega_m0_nonfinite"
        if not (0.0 < omega_m0 < 1.0):
            return -np.inf, {}, "omega_m0_unit_interval"
        if not np.isfinite(omega_k0):
            return -np.inf, {}, "omega_k0_nonfinite"
        if not (omega_k0 < 1.0):
            return -np.inf, {}, "omega_k0_lt1"
        if not np.isfinite(r_d) or r_d <= 0.0:
            return -np.inf, {}, "r_d_nonfinite"
        if idx_u_sigma8 is not None and (not np.isfinite(sigma8_0) or sigma8_0 <= 0.0):
            return -np.inf, {}, "sigma8_nonfinite"
        if not np.isfinite(sig_cc) or sig_cc < 0.0:
            return -np.inf, {}, "sigma_cc_nonfinite"
        if not np.isfinite(sig_sn) or sig_sn < 0.0:
            return -np.inf, {}, "sigma_sn_nonfinite"

        residual_of_z = None
        if r_knots is not None:
            sp = CubicSpline(z_r_knots, np.asarray(r_knots, dtype=float), bc_type="natural", extrapolate=True)

            def residual_of_z(zv: float) -> float:
                return float((H0**2) * sp(float(zv)))

        try:
            H_grid = forward.solve_H_from_logmu_knots(
                z_grid,
                logmu_knots=logmu_knots,
                H0_km_s_Mpc=H0,
                omega_m0=float(omega_m0),
                omega_k0=float(omega_k0),
                residual_of_z=residual_of_z,
            )
        except Exception:
            return -np.inf, {}, "solve_H_failed"
        if not np.all(np.isfinite(H_grid)):
            return -np.inf, {}, "H_grid_nonfinite"
        if np.any(H_grid <= 0.0):
            return -np.inf, {}, "H_grid_nonpositive"

        try:
            bg = build_background_from_H_grid(z_grid, H_grid, constants=constants)
        except Exception:
            return -np.inf, {}, "background_build_failed"

        # Chronometers with jitter
        t0 = time.perf_counter() if timing_acc is not None else None
        H_cc = bg.H(cc_z)
        if not np.all(np.isfinite(H_cc)):
            return -np.inf, {}, "H_cc_nonfinite"
        sig_eff = np.sqrt(cc_sigma_H**2 + sig_cc**2)
        if not np.all(np.isfinite(sig_eff)) or np.any(sig_eff <= 0.0):
            return -np.inf, {}, "cc_sigma_eff_nonpositive"
        ll_cc = -0.5 * float(np.sum(((cc_H - H_cc) / sig_eff) ** 2 + 2.0 * np.log(sig_eff)))
        if not np.isfinite(ll_cc):
            return -np.inf, {}, "ll_cc_nonfinite"
        if timing_acc is not None and t0 is not None:
            timing_acc["cc"] += float(time.perf_counter() - t0)

        # BAO
        t0 = time.perf_counter() if timing_acc is not None else None
        ll_bao = 0.0
        for bl in bao_likes:
            try:
                y_model = bl.predict(bg, r_d_Mpc=r_d)
            except Exception:
                return -np.inf, {}, f"bao_predict_failed:{bl.dataset}"
            if not np.all(np.isfinite(y_model)):
                return -np.inf, {}, f"bao_predict_nonfinite:{bl.dataset}"
            try:
                ll_bao += bl.loglike(y_model)
            except np.linalg.LinAlgError:
                return -np.inf, {}, f"bao_loglike_linalg:{bl.dataset}"
            except Exception:
                return -np.inf, {}, f"bao_loglike_failed:{bl.dataset}"
        if not np.isfinite(ll_bao):
            return -np.inf, {}, "ll_bao_nonfinite"
        if timing_acc is not None and t0 is not None:
            timing_acc["bao"] += float(time.perf_counter() - t0)

        # FSBAO (joint BAO geometry + fσ8) with full covariance (optional).
        t0 = time.perf_counter() if timing_acc is not None else None
        ll_fsbao = 0.0
        for fl in fsbao_likes:
            try:
                y_model = fl.predict(
                    bg,
                    z_grid=z_grid,
                    H_grid=H_grid,
                    H0=H0,
                    omega_m0=float(omega_m0),
                    omega_k0=float(omega_k0),
                    r_d_Mpc=r_d,
                    sigma8_0=float(sigma8_0),
                )
            except Exception:
                return -np.inf, {}, f"fsbao_predict_failed:{fl.dataset}"
            if not np.all(np.isfinite(y_model)):
                return -np.inf, {}, f"fsbao_predict_nonfinite:{fl.dataset}"
            try:
                ll_fsbao += fl.loglike(y_model)
            except np.linalg.LinAlgError:
                return -np.inf, {}, f"fsbao_loglike_linalg:{fl.dataset}"
            except Exception:
                return -np.inf, {}, f"fsbao_loglike_failed:{fl.dataset}"
        if not np.isfinite(ll_fsbao):
            return -np.inf, {}, "ll_fsbao_nonfinite"
        if timing_acc is not None and t0 is not None:
            timing_acc["fsbao"] += float(time.perf_counter() - t0)

        # SN
        t0 = time.perf_counter() if timing_acc is not None else None
        if sn_enabled:
            Dl = bg.Dl(sn_z)
            if np.any(~np.isfinite(Dl)) or np.any(Dl <= 0.0):
                return -np.inf, {}, "Dl_nonpositive"
            mu0 = 5.0 * np.log10(Dl)
            ll_sn, M_hat, _sn_beta_hat = sn_loglike(mu0, sig_sn)
            if not np.isfinite(ll_sn):
                return -np.inf, {}, "ll_sn_nonfinite"
        else:
            ll_sn, M_hat, _sn_beta_hat = 0.0, np.nan, np.zeros((0,), dtype=float)
        if timing_acc is not None and t0 is not None:
            timing_acc["sn"] += float(time.perf_counter() - t0)

        ll = ll_cc + ll_bao + ll_fsbao + ll_sn
        combined_done = False

        if lensing_like is not None and pk_likes:
            try:
                from .likelihoods_fullshape_pk import FullShapePkLogLike
                from .likelihoods_planck_lensing_clpp import PlanckLensingClppLogLike
            except Exception:
                FullShapePkLogLike = None
                PlanckLensingClppLogLike = None
            if (
                PlanckLensingClppLogLike is not None
                and FullShapePkLogLike is not None
                and isinstance(lensing_like, PlanckLensingClppLogLike)
                and len(pk_likes) == 1
                and isinstance(pk_likes[0], FullShapePkLogLike)
            ):
                t0 = time.perf_counter() if timing_acc is not None else None
                try:
                    from .camb_utils import camb_clpp_and_pk

                    pk_like = pk_likes[0]
                    clpp_model, pk_lin = camb_clpp_and_pk(
                        H0=float(H0),
                        omega_m0=float(omega_m0),
                        omega_k0=float(omega_k0),
                        sigma8_0=float(sigma8_0),
                        ell=lensing_like.ell_eff,
                        z_eff=float(pk_like.z_eff),
                        k=pk_like.k,
                    )
                    ll_lens = float(lensing_like.loglike(clpp_model))
                    if not np.isfinite(ll_lens):
                        return -np.inf, {}, "ll_lensing_nonfinite"
                    b1 = float(pk_b1[0]) if pk_b1 else 0.0
                    pshot = float(pk_noise[0]) if pk_noise else 0.0
                    model_pk = (b1**2) * np.asarray(pk_lin, dtype=float) + pshot
                    ll_pk = float(pk_like.loglike(model_pk))
                    if not np.isfinite(ll_pk):
                        return -np.inf, {}, "ll_pk_nonfinite:0"
                except Exception as exc:
                    try:
                        import traceback

                        ell_vals = np.asarray(lensing_like.ell_eff)
                        k_vals = np.asarray(pk_like.k)
                        context = {
                            "H0": float(H0),
                            "omega_m0": float(omega_m0),
                            "omega_k0": float(omega_k0),
                            "sigma8_0": float(sigma8_0),
                            "b1": float(pk_b1[0]) if pk_b1 else 0.0,
                            "pshot": float(pk_noise[0]) if pk_noise else 0.0,
                            "ell_size": int(ell_vals.size),
                            "ell_max": int(np.max(ell_vals)) if ell_vals.size else None,
                            "k_size": int(k_vals.size),
                            "k_min": float(np.min(k_vals)) if k_vals.size else None,
                            "k_max": float(np.max(k_vals)) if k_vals.size else None,
                            "z_eff": float(pk_like.z_eff),
                        }
                        last_detail = {
                            "reason": "lensing_pk_failed",
                            "exception": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                            "context": context,
                        }
                    except Exception:
                        last_detail = {
                            "reason": "lensing_pk_failed",
                            "exception": type(exc).__name__,
                            "message": str(exc),
                        }
                    return -np.inf, {}, "lensing_pk_failed"
                ll += ll_lens + ll_pk
                combined_done = True
                if timing_acc is not None and t0 is not None:
                    dt = float(time.perf_counter() - t0)
                    timing_acc["lensing"] += 0.5 * dt
                    timing_acc["pk"] += 0.5 * dt

        # RSD fσ8(z) growth constraint (optional).
        if rsd_like is not None:
            t0 = time.perf_counter() if timing_acc is not None else None
            try:
                fs8_model = rsd_like.predict(
                    z_grid=z_grid,
                    H_grid=H_grid,
                    H0=H0,
                    omega_m0=float(omega_m0),
                    omega_k0=float(omega_k0),
                    sigma8_0=float(sigma8_0),
                )
                ll_rsd = float(rsd_like.loglike(fs8_model))
            except Exception:
                return -np.inf, {}, "rsd_failed"
            if not np.isfinite(ll_rsd):
                return -np.inf, {}, "ll_rsd_nonfinite"
            ll += ll_rsd
            if timing_acc is not None and t0 is not None:
                timing_acc["rsd"] += float(time.perf_counter() - t0)

        # Planck lensing proxy constraint (optional).
        if lensing_like is not None and not combined_done:
            t0 = time.perf_counter() if timing_acc is not None else None
            try:
                lens_model = lensing_like.predict(
                    H0=float(H0),
                    omega_m0=float(omega_m0),
                    omega_k0=float(omega_k0),
                    sigma8_0=float(sigma8_0),
                )
                try:
                    lens_model = float(lens_model)
                except (TypeError, ValueError):
                    lens_model = np.asarray(lens_model, dtype=float)
                ll_lens = float(lensing_like.loglike(lens_model))
            except Exception:
                return -np.inf, {}, "lensing_failed"
            if not np.isfinite(ll_lens):
                return -np.inf, {}, "ll_lensing_nonfinite"
            ll += ll_lens
            if timing_acc is not None and t0 is not None:
                timing_acc["lensing"] += float(time.perf_counter() - t0)

        if pk_likes and not combined_done:
            for i, pk_like in enumerate(pk_likes):
                t0 = time.perf_counter() if timing_acc is not None else None
                try:
                    model_pk = pk_like.predict(
                        H0=H0,
                        omega_m0=float(omega_m0),
                        omega_k0=float(omega_k0),
                        sigma8_0=float(sigma8_0),
                        b1=float(pk_b1[i]),
                        pshot=float(pk_noise[i]),
                    )
                    ll_pk = float(pk_like.loglike(model_pk))
                except Exception:
                    return -np.inf, {}, f"pk_failed:{i}"
                if not np.isfinite(ll_pk):
                    return -np.inf, {}, f"ll_pk_nonfinite:{i}"
                ll += ll_pk
                if timing_acc is not None and t0 is not None:
                    timing_acc["pk"] += float(time.perf_counter() - t0)
        if not np.isfinite(ll):
            return -np.inf, {}, "ll_nonfinite"
        return (
            ll,
            {
                "H0": H0,
                "omega_m0": omega_m0,
                "omega_k0": omega_k0,
                "r_d": r_d,
                "sigma8_0": sigma8_0,
                "M_hat": M_hat,
                "pk_b1": np.asarray(pk_b1, dtype=float) if pk_b1 else np.zeros((0,), dtype=float),
                "pk_noise": np.asarray(pk_noise, dtype=float) if pk_noise else np.zeros((0,), dtype=float),
            },
            None,
        )

    def log_prob(theta: np.ndarray) -> float:
        nonlocal logprob_total_calls, timing_calls, last_detail
        if logprob_counts_valid:
            logprob_total_calls += 1
        lp, lp_reason = log_prior(theta)
        if not np.isfinite(lp):
            reason = f"prior:{lp_reason or 'nonfinite'}"
            record_invalid(reason)
            log_invalid(theta, reason)
            return -np.inf
        last_detail = None
        t0 = time.perf_counter() if timing_enabled else None
        ll, _, ll_reason = log_likelihood(theta, timing_acc=timing_acc if timing_enabled else None)
        if not np.isfinite(ll):
            reason = f"like:{ll_reason or 'nonfinite'}"
            record_invalid(reason)
            detail = last_detail
            last_detail = None
            log_invalid(theta, reason, detail=detail)
            return -np.inf
        total = lp + ll
        if timing_enabled and t0 is not None:
            timing_acc["total"] += float(time.perf_counter() - t0)
            nonlocal timing_calls
            timing_calls += 1
            if timing_every > 0 and timing_calls >= int(timing_every):
                _timing_flush()
        if not np.isfinite(total):
            record_invalid("posterior:nonfinite")
            log_invalid(theta, "posterior:nonfinite")
            return -np.inf
        return total

    if n_walkers < 2 * ndim:
        raise ValueError(f"n_walkers must be >= 2*ndim = {2*ndim} (got {n_walkers}).")

    def _init_row(rng: np.random.RandomState) -> np.ndarray:
        row = np.zeros((ndim,), dtype=float)
        if not mu_is_fixed:
            row[sl_logmu] = rng.normal(scale=0.05, size=n_mu_knots)  # logmu knots near 0
        H0_init = 0.5 * (H0_prior[0] + H0_prior[1])
        u_H0_0 = unbounded_from_bounded(H0_init, H0_prior[0], H0_prior[1])
        row[idx_u_H0] = u_H0_0 + rng.normal(scale=0.2)
        if idx_u_omega_m0 is not None:
            u_om_0 = unbounded_from_bounded(0.5 * (om_lo_eff + om_hi_eff), om_lo_eff, om_hi_eff)
            row[idx_u_omega_m0] = u_om_0 + rng.normal(scale=0.2)
        if idx_u_omega_k0 is not None:
            u_ok_0 = unbounded_from_bounded(0.5 * (ok_lo_eff + ok_hi_eff), ok_lo_eff, ok_hi_eff)
            row[idx_u_omega_k0] = u_ok_0 + rng.normal(scale=0.2)
        if idx_u_r_d is not None:
            r_d_init = 0.5 * (r_d_prior[0] + r_d_prior[1])
            u_rd_0 = unbounded_from_bounded(r_d_init, r_d_prior[0], r_d_prior[1])
            row[idx_u_r_d] = u_rd_0 + rng.normal(scale=0.2)
        if idx_u_sigma8 is not None:
            sigma8_init = 0.5 * (sigma8_prior[0] + sigma8_prior[1])
            u_s8_0 = unbounded_from_bounded(sigma8_init, sigma8_prior[0], sigma8_prior[1])
            row[idx_u_sigma8] = u_s8_0 + rng.normal(scale=0.2)
        if idx_u_pk_b1:
            for ib1, inois in zip(idx_u_pk_b1, idx_u_pk_noise):
                b1_init = 0.5 * (pk_bias_prior[0] + pk_bias_prior[1])
                n_init = 0.5 * (pk_noise_prior[0] + pk_noise_prior[1])
                row[ib1] = unbounded_from_bounded(b1_init, pk_bias_prior[0], pk_bias_prior[1]) + rng.normal(scale=0.2)
                row[inois] = unbounded_from_bounded(n_init, pk_noise_prior[0], pk_noise_prior[1]) + rng.normal(scale=0.2)
        row[idx_log_sig_cc] = np.log(1e-3) + rng.normal(scale=0.2)  # log sigma_cc_jit
        row[idx_log_sig_sn] = np.log(1e-3) + rng.normal(scale=0.2)  # log sigma_sn_jit
        row[idx_log_sig_d2] = np.log(0.1) + rng.normal(scale=0.5)  # log sigma_d2
        if sl_r_knots is not None:
            row[sl_r_knots] = rng.normal(scale=0.01, size=n_r)  # residual knots near 0
        return row

    # Init around BH with small perturbations, reject invalid starts.
    p0 = np.zeros((n_walkers, ndim), dtype=float)
    max_init_tries = 200
    for i in range(n_walkers):
        _safe_log(f"[init] walker {i+1}/{n_walkers} starting")
        for attempt in range(max_init_tries):
            row = _init_row(rng_init)
            if np.isfinite(log_prob(row)):
                p0[i] = row
                _safe_log(f"[init] walker {i+1}/{n_walkers} ok (tries={attempt+1})")
                break
        else:
            raise RuntimeError(
                "Failed to find a valid initial position after "
                f"{max_init_tries} attempts; check priors/likelihoods."
            )
    _safe_log(f"[init] initialized {n_walkers} walkers (ndim={ndim})")

    # Make emcee's internal RNG deterministic when `seed` is fixed.
    initial_state = emcee.State(p0, random_state=np.random.RandomState(seed))

    maybe_check_maxrss()

    def run_sampler(sampler: emcee.EnsembleSampler, *, state: emcee.State, n_steps_to_run: int, show_progress: bool) -> emcee.State:
        """Run emcee in chunks to allow RSS watchdog checks."""
        if n_steps_to_run <= 0:
            return state
        if maxrss_check_every <= 0:
            try:
                return sampler.run_mcmc(state, n_steps_to_run, progress=show_progress)
            except BrokenPipeError:
                # Progress bar output can fail if stdout/stderr is a broken pipe.
                return sampler.run_mcmc(state, n_steps_to_run, progress=False)
        step = 0
        chunk = int(maxrss_check_every)
        while step < n_steps_to_run:
            n_chunk = min(chunk, n_steps_to_run - step)
            try:
                state = sampler.run_mcmc(state, n_chunk, progress=show_progress and step == 0)
            except BrokenPipeError:
                # Disable progress for the rest of the run if the output stream broke.
                state = sampler.run_mcmc(state, n_chunk, progress=False)
                show_progress = False
            step += n_chunk
            maybe_check_maxrss()
        return state

    def compute_min_ess_from_chain(
        chain_3d: np.ndarray,
    ) -> tuple[float, dict[str, float], dict[str, float]] | None:
        """Estimate a minimum ESS across a few scalar projections for convergence checks.

        Parameters
        ----------
        chain_3d:
            Shape (n_steps, n_walkers, ndim) *after* burn-in discard.
        """
        chain_3d = np.asarray(chain_3d, dtype=float)
        if chain_3d.ndim != 3:
            return None
        n_t = int(chain_3d.shape[0])
        if n_t < 100:
            return None

        # Derived scalars to monitor (cheap, avoids forward-model recomputation).
        u_H0 = chain_3d[:, :, idx_u_H0]
        log_sig_cc = chain_3d[:, :, idx_log_sig_cc]
        log_sig_sn = chain_3d[:, :, idx_log_sig_sn]
        log_sig_d2 = chain_3d[:, :, idx_log_sig_d2]

        mean_knot = None
        slope_knot = None
        if not mu_is_fixed:
            logmu_knots = chain_3d[:, :, :n_mu_knots]
            mean_knot = np.mean(logmu_knots, axis=2)
            x_span = float(x_knots[-1] - x_knots[0])
            if not np.isfinite(x_span) or x_span <= 0:
                return None
            slope_knot = (logmu_knots[:, :, -1] - logmu_knots[:, :, 0]) / x_span

        H0_vals = H0_prior[0] + (H0_prior[1] - H0_prior[0]) * sigmoid(u_H0)
        parts = [H0_vals]
        names = ["H0"]
        if idx_u_omega_m0 is not None:
            om_vals = om_lo_eff + (om_hi_eff - om_lo_eff) * sigmoid(chain_3d[:, :, idx_u_omega_m0])
            parts.append(om_vals)
            names.append("omega_m0")
        if idx_u_omega_k0 is not None:
            ok_vals = ok_lo_eff + (ok_hi_eff - ok_lo_eff) * sigmoid(chain_3d[:, :, idx_u_omega_k0])
            parts.append(ok_vals)
            names.append("omega_k0")
        if idx_u_sigma8 is not None:
            s8_vals = sigma8_prior[0] + (sigma8_prior[1] - sigma8_prior[0]) * sigmoid(chain_3d[:, :, idx_u_sigma8])
            parts.append(s8_vals)
            names.append("sigma8_0")
        if idx_u_pk_b1:
            for i, (ib1, inois) in enumerate(zip(idx_u_pk_b1, idx_u_pk_noise)):
                b1_vals = pk_bias_prior[0] + (pk_bias_prior[1] - pk_bias_prior[0]) * sigmoid(chain_3d[:, :, ib1])
                n_vals = pk_noise_prior[0] + (pk_noise_prior[1] - pk_noise_prior[0]) * sigmoid(chain_3d[:, :, inois])
                parts.append(b1_vals)
                parts.append(n_vals)
                names.append(f"pk_b1_{i}")
                names.append(f"pk_noise_{i}")
        parts += [log_sig_cc, log_sig_sn, log_sig_d2]
        names += ["log_sig_cc", "log_sig_sn", "log_sig_d2"]
        if mean_knot is not None and slope_knot is not None:
            parts += [mean_knot, slope_knot]
            names += ["mean_knot", "slope_knot"]
        scalars = np.stack(parts, axis=-1)
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau = emcee.autocorr.integrated_time(scalars, quiet=True)
        except Exception:
            return None
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or tau.size != scalars.shape[-1]:
            return None
        if not np.all(np.isfinite(tau)):
            return None
        # Avoid nonsensical / tiny tau; treat tau<1 as 1 step.
        tau = np.maximum(tau, 1.0)
        ess = (n_t * n_walkers) / tau
        ess_by = {name: float(ess[i]) for i, name in enumerate(names)}
        tau_by = {name: float(tau[i]) for i, name in enumerate(names)}
        return float(np.min(ess)), ess_by, tau_by

    try:
        sampler_kind_norm = str(sampler_kind).strip().lower()
        if sampler_kind_norm not in {"emcee", "ptemcee"}:
            raise ValueError("sampler_kind must be one of: emcee, ptemcee")

        total_steps = int(n_steps)
        ess_min = np.nan
        ess_by: dict[str, float] | None = None
        tau_by: dict[str, float] | None = None
        acceptance_fraction_mean = np.nan
        sampler_extra: dict[str, object] | None = None
        chain_flat: np.ndarray

        if sampler_kind_norm == "emcee":
            _safe_log(f"[emcee] start run: steps={int(n_steps)} procs={int(n_processes)} walkers={int(n_walkers)}")
            if n_processes and n_processes > 1:
                import multiprocessing as mp

                # NOTE: emcee parallelization pickles the log-prob callable. Since our
                # log-prob is a local closure, we must route through a module-level wrapper
                # and rely on "fork" so workers inherit the closure via memory.
                ctx = mp.get_context("fork")
                global _GLOBAL_LOG_PROB
                _GLOBAL_LOG_PROB = log_prob
                try:
                    with ctx.Pool(processes=n_processes) as pool:
                        sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_prob_wrapper, pool=pool)
                        state = run_sampler(
                            sampler, state=initial_state, n_steps_to_run=int(n_steps), show_progress=bool(progress)
                        )
                finally:
                    _GLOBAL_LOG_PROB = None
            else:
                sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
                state = run_sampler(
                    sampler, state=initial_state, n_steps_to_run=int(n_steps), show_progress=bool(progress)
                )

            acceptance_fraction_mean = float(np.mean(sampler.acceptance_fraction))
            _safe_log(f"[emcee] done run: acceptance_mean={acceptance_fraction_mean:.4f}")

            def _current_chain_3d() -> np.ndarray | None:
                try:
                    return sampler.get_chain(discard=n_burn, flat=False)
                except Exception:
                    return None

            # Always attempt a lightweight tau/ESS estimate (if chain is long enough).
            c0 = _current_chain_3d()
            out0 = compute_min_ess_from_chain(c0) if c0 is not None else None
            if out0 is not None:
                ess_min, ess_by, tau_by = out0
            # Convergence enforcement: only for moderately long runs (avoid "fast mode" surprises).
            if total_steps >= 400 and not (n_processes and n_processes > 1):
                max_total_steps = int(min(8000, max(total_steps, 3 * total_steps)))
                target_ess = 200.0
                for _ in range(2):
                    c = _current_chain_3d()
                    out = compute_min_ess_from_chain(c) if c is not None else None
                    if out is None:
                        break
                    ess_min, ess_by, tau_by = out
                    if ess_min >= target_ess:
                        break
                    extra = min(total_steps, max_total_steps - total_steps)
                    if extra <= 0:
                        break
                    state = run_sampler(sampler, state=state, n_steps_to_run=int(extra), show_progress=False)
                    total_steps += int(extra)
                if ess_by is None:
                    c = _current_chain_3d()
                    out = compute_min_ess_from_chain(c) if c is not None else None
                    if out is not None:
                        ess_min, ess_by, tau_by = out

            chain_flat = sampler.get_chain(discard=n_burn, flat=True)
            if save_chain_path is not None:
                import numpy as _np
                from pathlib import Path as _Path

                save_path = _Path(save_chain_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                chain3d_post = sampler.get_chain(discard=n_burn, flat=False)
                _np.savez(
                    save_path,
                    chain=chain3d_post,
                    param_names=_np.asarray(param_names, dtype=object),
                    x_knots=_np.asarray(x_knots, dtype=float),
                    x_grid=_np.asarray(x_grid, dtype=float),
                )
        else:
            # ptemcee currently relies on deprecated NumPy aliases; patch for NumPy>=2 at runtime.
            if not hasattr(np, "float"):
                np.float = float  # type: ignore[attr-defined]
            if not hasattr(np, "int"):
                np.int = int  # type: ignore[attr-defined]
            import ptemcee  # type: ignore
            from pathlib import Path as _Path

            if int(pt_ntemps) < 2:
                raise ValueError("pt_ntemps must be >= 2 for ptemcee.")

            def logp_pt(theta: np.ndarray) -> float:
                lp, _reason = log_prior(np.asarray(theta, dtype=float))
                return float(lp)

            def logl_pt(theta: np.ndarray) -> float:
                nonlocal logprob_total_calls, timing_calls
                if logprob_counts_valid:
                    logprob_total_calls += 1
                # Early-exit on prior invalid (likelihood irrelevant).
                lp, lp_reason = log_prior(np.asarray(theta, dtype=float))
                if not np.isfinite(lp):
                    reason = f"prior:{lp_reason or 'nonfinite'}"
                    record_invalid(reason)
                    log_invalid(np.asarray(theta, dtype=float), reason)
                    return -np.inf
                t0 = time.perf_counter() if timing_enabled else None
                ll, _params, ll_reason = log_likelihood(
                    np.asarray(theta, dtype=float), timing_acc=timing_acc if timing_enabled else None
                )
                if not np.isfinite(ll):
                    reason = f"like:{ll_reason or 'nonfinite'}"
                    record_invalid(reason)
                    log_invalid(np.asarray(theta, dtype=float), reason)
                    return -np.inf
                if timing_enabled and t0 is not None:
                    timing_acc["total"] += float(time.perf_counter() - t0)
                    timing_calls += 1
                    if timing_every > 0 and timing_calls >= int(timing_every):
                        _timing_flush()
                return float(ll)

            # Initialize all temperatures near the same p0 with small temperature-dependent scatter.
            ntemps = int(pt_ntemps)
            p0_pt = np.repeat(p0[None, :, :], repeats=ntemps, axis=0)
            for t in range(ntemps):
                p0_pt[t] = p0 + rng_init.normal(scale=0.02 * (1.0 + 0.25 * t), size=p0.shape)

            checkpoint_every = int(checkpoint_every or 0)
            checkpoint_dir = _Path(checkpoint_path) if checkpoint_path is not None else None
            if checkpoint_every <= 0 or checkpoint_dir is None:
                checkpoint_every = 0
                checkpoint_dir = None

            chain_chunks: list[np.ndarray] = []
            steps_done = 0
            p_current = p0_pt
            rng_state = None

            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                state_path = checkpoint_dir / "state.npz"
                state_steps = None
                if state_path.exists():
                    try:
                        with np.load(state_path, allow_pickle=True) as npz:
                            steps_done = int(npz.get("steps_done", 0))
                            state_steps = steps_done
                            p_current = npz["p_current"]
                            rng_state = npz.get("rng_state", None)
                            if rng_state is not None:
                                rng_state = rng_state.tolist()
                            ntemps_ck = int(npz.get("ntemps", ntemps))
                            nwalkers_ck = int(npz.get("nwalkers", n_walkers))
                            ndim_ck = int(npz.get("ndim", ndim))
                        if (ntemps_ck, nwalkers_ck, ndim_ck) != (ntemps, n_walkers, ndim):
                            raise ValueError("Checkpoint shape mismatch; refusing to resume.")
                    except Exception:
                        steps_done = 0
                        p_current = p0_pt
                        rng_state = None

                # Load existing chain chunks (if any) to preserve progress.
                try:
                    chunk_paths = sorted(checkpoint_dir.glob("chain_chunk_*.npz"))
                    loaded_steps = 0
                    for cp in chunk_paths:
                        with np.load(cp, allow_pickle=False) as npz:
                            ch = np.asarray(npz["chain"], dtype=float)
                        if ch.ndim != 3 or ch.shape[1] != n_walkers or ch.shape[2] != ndim:
                            continue
                        chain_chunks.append(ch)
                        loaded_steps += int(ch.shape[0])
                    if loaded_steps > 0 and state_steps is None:
                        steps_done = loaded_steps
                except Exception:
                    pass

            maybe_check_maxrss()
            if n_processes and n_processes > 1:
                import multiprocessing as mp

                ctx = mp.get_context("fork")
                global _GLOBAL_LOG_LIKE, _GLOBAL_LOG_PRIOR
                _GLOBAL_LOG_LIKE = logl_pt
                _GLOBAL_LOG_PRIOR = logp_pt
                try:
                    with ctx.Pool(processes=n_processes) as pool:
                        sampler_pt = ptemcee.Sampler(
                            n_walkers,
                            ndim,
                            _log_like_wrapper,
                            _log_prior_wrapper,
                            ntemps=ntemps,
                            Tmax=float(pt_tmax) if pt_tmax is not None else None,
                            pool=pool,
                            random=np.random.RandomState(seed),
                        )
                        if rng_state is not None:
                            try:
                                sampler_pt._random.set_state(rng_state)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        remaining = max(0, int(n_steps) - int(steps_done))
                        if checkpoint_every and checkpoint_dir is not None:
                            saved_chain_steps = sampler_pt.chain.shape[2] if sampler_pt.chain is not None else 0
                            while remaining > 0:
                                n_chunk = min(int(checkpoint_every), int(remaining))
                                last_p = None
                                for last_p, _logpost, _logl in sampler_pt.sample(p0=p_current, iterations=n_chunk):
                                    pass
                                if last_p is None:
                                    raise RuntimeError("ptemcee produced no samples.")
                                new_steps = sampler_pt.chain.shape[2] - saved_chain_steps
                                chunk = sampler_pt.chain[0, :, saved_chain_steps : saved_chain_steps + new_steps, :]
                                chunk = np.transpose(chunk, (1, 0, 2))
                                chain_chunks.append(np.asarray(chunk, dtype=float))
                                if checkpoint_dir is not None:
                                    start = steps_done
                                    end = steps_done + new_steps
                                    np.savez_compressed(
                                        checkpoint_dir / f"chain_chunk_{start:06d}_{end:06d}.npz",
                                        chain=np.asarray(chunk, dtype=float),
                                    )
                                    np.savez_compressed(
                                        checkpoint_dir / "state.npz",
                                        steps_done=int(end),
                                        p_current=np.asarray(last_p, dtype=float),
                                        ntemps=int(ntemps),
                                        nwalkers=int(n_walkers),
                                        ndim=int(ndim),
                                        rng_state=np.asarray(sampler_pt._random.get_state(), dtype=object),
                                    )
                                steps_done += new_steps
                                saved_chain_steps += new_steps
                                remaining -= n_chunk
                                p_current = last_p
                        else:
                            sampler_pt.run_mcmc(p_current, int(remaining))
                finally:
                    _GLOBAL_LOG_LIKE = None
                    _GLOBAL_LOG_PRIOR = None
            else:
                sampler_pt = ptemcee.Sampler(
                    n_walkers,
                    ndim,
                    logl_pt,
                    logp_pt,
                    ntemps=ntemps,
                    Tmax=float(pt_tmax) if pt_tmax is not None else None,
                    random=np.random.RandomState(seed),
                )
                if rng_state is not None:
                    try:
                        sampler_pt._random.set_state(rng_state)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                remaining = max(0, int(n_steps) - int(steps_done))
                if checkpoint_every and checkpoint_dir is not None:
                    saved_chain_steps = sampler_pt.chain.shape[2] if sampler_pt.chain is not None else 0
                    while remaining > 0:
                        n_chunk = min(int(checkpoint_every), int(remaining))
                        last_p = None
                        for last_p, _logpost, _logl in sampler_pt.sample(p0=p_current, iterations=n_chunk):
                            pass
                        if last_p is None:
                            raise RuntimeError("ptemcee produced no samples.")
                        new_steps = sampler_pt.chain.shape[2] - saved_chain_steps
                        chunk = sampler_pt.chain[0, :, saved_chain_steps : saved_chain_steps + new_steps, :]
                        chunk = np.transpose(chunk, (1, 0, 2))
                        chain_chunks.append(np.asarray(chunk, dtype=float))
                        if checkpoint_dir is not None:
                            start = steps_done
                            end = steps_done + new_steps
                            np.savez_compressed(
                                checkpoint_dir / f"chain_chunk_{start:06d}_{end:06d}.npz",
                                chain=np.asarray(chunk, dtype=float),
                            )
                            np.savez_compressed(
                                checkpoint_dir / "state.npz",
                                steps_done=int(end),
                                p_current=np.asarray(last_p, dtype=float),
                                ntemps=int(ntemps),
                                nwalkers=int(n_walkers),
                                ndim=int(ndim),
                                rng_state=np.asarray(sampler_pt._random.get_state(), dtype=object),
                            )
                        steps_done += new_steps
                        saved_chain_steps += new_steps
                        remaining -= n_chunk
                        p_current = last_p
                else:
                    sampler_pt.run_mcmc(p_current, int(remaining))
            maybe_check_maxrss()

            acceptance_fraction_mean = float(np.mean(sampler_pt.acceptance_fraction[0]))
            sampler_extra = {
                "ptemcee": {
                    "ntemps": int(ntemps),
                    "Tmax": float(pt_tmax) if pt_tmax is not None else None,
                    "acceptance_fraction_by_temp_mean": [float(np.mean(a)) for a in sampler_pt.acceptance_fraction],
                    "tswap_acceptance_fraction": [float(x) for x in sampler_pt.tswap_acceptance_fraction],
                }
            }

            # Cold chain only.
            if checkpoint_every and checkpoint_dir is not None and chain_chunks:
                chain3d = np.concatenate(chain_chunks, axis=0)
            else:
                chain0 = np.asarray(sampler_pt.chain[0], dtype=float)  # (n_walkers, n_steps, ndim)
                chain3d = np.transpose(chain0, (1, 0, 2))  # (n_steps, n_walkers, ndim)
            if not (0 <= int(n_burn) < chain3d.shape[0]):
                raise ValueError("n_burn must be in [0, n_steps).")
            chain3d_post = chain3d[int(n_burn) :, :, :]

            out0 = compute_min_ess_from_chain(chain3d_post)
            if out0 is not None:
                ess_min, ess_by, tau_by = out0

            chain_flat = chain3d_post.reshape((-1, ndim))
            if save_chain_path is not None:
                import numpy as _np
                from pathlib import Path as _Path

                save_path = _Path(save_chain_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                _np.savez(
                    save_path,
                    chain=chain3d_post,
                    param_names=_np.asarray(param_names, dtype=object),
                    x_knots=_np.asarray(x_knots, dtype=float),
                    x_grid=_np.asarray(x_grid, dtype=float),
                )

        if chain_flat.shape[0] < 50:
            raise RuntimeError("Too few posterior samples; increase n_steps or reduce burn-in.")
        draw_n = min(int(n_draws), int(chain_flat.shape[0]))
        idx = rng.choice(chain_flat.shape[0], size=draw_n, replace=False)
        draws = chain_flat[idx]

        # Evaluate outputs for draws
        logmu_x = np.empty((draw_n, x_grid.size))
        H_s = np.empty((draw_n, z_grid.size))
        H0_s = np.empty(draw_n)
        Om_s = np.empty(draw_n)
        Ok_s = np.empty(draw_n)
        rd_s = np.empty(draw_n)
        sigma8_s = np.empty(draw_n) if include_sigma8 else None
        S8_s = np.empty(draw_n) if include_sigma8 else None
        sigcc_s = np.empty(draw_n)
        sigsn_s = np.empty(draw_n)
        sigd2_s = np.empty(draw_n)
        r_s = np.empty((draw_n, n_r)) if use_residual else None
        pk_b1_s = np.empty((draw_n, len(idx_u_pk_b1))) if idx_u_pk_b1 else None
        pk_noise_s = np.empty((draw_n, len(idx_u_pk_noise))) if idx_u_pk_noise else None

        fixed_logmu_x = None
        if mu_is_fixed:
            fixed_spline = make_spline(x_knots, np.asarray(fixed_logmu, dtype=float))
            fixed_logmu_x = fixed_spline(np.clip(x_grid, x_knots[0], x_knots[-1]))

        for j, th in enumerate(draws):
            logmu_knots = fixed_logmu if mu_is_fixed else th[sl_logmu]
            u_H0 = float(th[idx_u_H0])
            u_omega_m0 = float(th[idx_u_omega_m0]) if idx_u_omega_m0 is not None else np.nan
            u_omega_k0 = float(th[idx_u_omega_k0]) if idx_u_omega_k0 is not None else np.nan
            u_r_d = float(th[idx_u_r_d]) if idx_u_r_d is not None else np.nan
            u_sigma8 = float(th[idx_u_sigma8]) if idx_u_sigma8 is not None else np.nan
            u_pk_b1 = [float(th[i]) for i in idx_u_pk_b1] if idx_u_pk_b1 else []
            u_pk_noise = [float(th[i]) for i in idx_u_pk_noise] if idx_u_pk_noise else []
            log_sig_cc = float(th[idx_log_sig_cc])
            log_sig_sn = float(th[idx_log_sig_sn])
            log_sig_d2 = float(th[idx_log_sig_d2])
            r_knots = th[sl_r_knots] if sl_r_knots is not None else None

            H0, _ = bounded_from_u(u_H0, H0_prior[0], H0_prior[1])
            if idx_u_omega_m0 is not None:
                omega_m0, _ = bounded_from_u(u_omega_m0, om_lo_eff, om_hi_eff)
            else:
                omega_m0 = float(omega_m0_fixed)
            if idx_u_omega_k0 is not None:
                omega_k0, _ = bounded_from_u(u_omega_k0, ok_lo_eff, ok_hi_eff)
            else:
                omega_k0 = 0.0
            if idx_u_sigma8 is not None:
                sigma8_0, _ = bounded_from_u(u_sigma8, sigma8_prior[0], sigma8_prior[1])
            else:
                sigma8_0 = np.nan
            if pk_b1_s is not None and pk_noise_s is not None:
                for i, (ub1, un) in enumerate(zip(u_pk_b1, u_pk_noise)):
                    pk_b1_s[j, i] = float(bounded_from_u(ub1, pk_bias_prior[0], pk_bias_prior[1])[0])
                    pk_noise_s[j, i] = float(bounded_from_u(un, pk_noise_prior[0], pk_noise_prior[1])[0])
            if fixed_logmu_x is not None:
                logmu_x[j] = fixed_logmu_x
            else:
                spline = make_spline(x_knots, logmu_knots)
                logmu_x[j] = spline(np.clip(x_grid, x_knots[0], x_knots[-1]))

            residual_of_z = None
            if r_knots is not None:
                sp = CubicSpline(z_r_knots, np.asarray(r_knots, dtype=float), bc_type="natural", extrapolate=True)

                def residual_of_z(zv: float) -> float:
                    return float((H0**2) * sp(float(zv)))

            H_grid = forward.solve_H_from_logmu_knots(
                z_grid,
                logmu_knots=logmu_knots,
                H0_km_s_Mpc=H0,
                omega_m0=float(omega_m0),
                omega_k0=float(omega_k0),
                residual_of_z=residual_of_z,
            )
            H_s[j] = H_grid
            H0_s[j] = H0
            Om_s[j] = float(omega_m0)
            Ok_s[j] = float(omega_k0)
            if sigma8_s is not None and S8_s is not None:
                sigma8_s[j] = float(sigma8_0)
                S8_s[j] = float(sigma8_0) * float(np.sqrt(float(omega_m0) / 0.3))
            if idx_u_r_d is not None:
                rd_s[j] = float(bounded_from_u(u_r_d, r_d_prior[0], r_d_prior[1])[0])
            else:
                rd_s[j] = float(r_d_fixed) if r_d_fixed is not None else np.nan
            sigcc_s[j] = float(np.exp(log_sig_cc))
            sigsn_s[j] = float(np.exp(log_sig_sn))
            sigd2_s[j] = float(np.exp(log_sig_d2))
            if r_knots is not None and r_s is not None:
                r_s[j] = np.asarray(r_knots, dtype=float)

        logprob_meta = {
            "counts_valid": bool(logprob_counts_valid),
            "total_calls": int(logprob_total_calls) if logprob_counts_valid else None,
            "invalid_calls": int(logprob_invalid_calls) if logprob_counts_valid else None,
            "invalid_rate": (
                float(logprob_invalid_calls) / float(logprob_total_calls)
                if logprob_counts_valid and logprob_total_calls > 0
                else None
            ),
            "invalid_reason_counts": invalid_reason_counts if logprob_counts_valid else None,
        }

        meta = {
            "sampler_kind": str(sampler_kind_norm),
            "sampler_extra": sampler_extra,
            "acceptance_fraction_mean": float(acceptance_fraction_mean),
            "n_walkers": int(n_walkers),
            "n_steps": int(total_steps),
            "n_burn": int(n_burn),
            "draws": int(draw_n),
            "seed": int(seed),
            "init_seed": int(init_seed) if init_seed is not None else None,
            "n_mu_knots": int(n_mu_knots),
            "mu_fixed": bool(mu_is_fixed),
            "fixed_logmu_knots": (
                str(fixed_logmu_knots)
                if isinstance(fixed_logmu_knots, str)
                else ("array" if fixed_logmu_knots is not None else None)
            ),
            "n_residual_knots": int(n_r),
            "ndim": int(ndim),
            "param_names": list(param_names),
            "ess_min": float(ess_min) if np.isfinite(ess_min) else None,
            "ess_by": ess_by,
            "tau_by": tau_by,
            "logprob": logprob_meta,
            "priors": {
                "omega_m0_prior": tuple(float(x) for x in omega_m0_prior),
                "omega_m0_fixed": float(omega_m0_fixed) if omega_m0_fixed is not None else None,
                "omega_k0_prior": tuple(float(x) for x in omega_k0_prior) if omega_k0_prior is not None else None,
                "H0_prior": tuple(float(x) for x in H0_prior),
                "r_d_prior": tuple(float(x) for x in r_d_prior),
                "r_d_fixed": float(r_d_fixed) if r_d_fixed is not None else None,
                "sigma8_prior": tuple(float(x) for x in sigma8_prior) if include_sigma8 else None,
                "sigma_cc_jit_scale": float(sigma_cc_jit_scale),
                "sigma_sn_jit_scale": float(sigma_sn_jit_scale),
                "logmu_knot_scale": float(logmu_knot_scale),
                "log_sigma_d2_prior": tuple(float(x) for x in log_sigma_d2_prior),
                "sigma_d2_scale": float(sigma_d2_scale),
                "sn_marginalization": {"mode": sn_marg_norm, "mstep_z": float(sn_mstep_z), "columns": col_names},
                "residual": {
                    "enabled": bool(use_residual),
                    "sigma_r0": float(sigma_r0) if np.isfinite(sigma_r0) else None,
                    "sigma_r_d2": float(sigma_r_d2) if np.isfinite(sigma_r_d2) else None,
                },
            },
        }
        if timing_enabled:
            _timing_flush()
        return ForwardMuPosterior(
            x_grid=x_grid,
            logmu_x_samples=logmu_x,
            z_grid=z_grid,
            H_samples=H_s,
            params={
                "H0": H0_s,
                "omega_m0": Om_s,
                "omega_k0": Ok_s,
                "r_d_Mpc": rd_s,
                **({"sigma8_0": sigma8_s, "S8": S8_s} if include_sigma8 else {}),
                "sigma_cc_jit": sigcc_s,
                "sigma_sn_jit": sigsn_s,
                "sigma_d2": sigd2_s,
                **(
                    {"pk_b1": pk_b1_s, "pk_noise": pk_noise_s}
                    if pk_b1_s is not None and pk_noise_s is not None
                    else {}
                ),
                **({"residual_r_knots": r_s} if use_residual else {}),
            },
            meta=meta,
        )
    finally:
        if old_rlimit_as is not None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_rlimit_as)
            except Exception:
                pass
