#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.strong_lens_time_delay import H0LiCOW6Likelihood, fetch_h0licow_distance_catalog

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_GLOBAL_LIKE: H0LiCOW6Likelihood | None = None
_GLOBAL_BASELINE_LIKE: H0LiCOW6Likelihood | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_optional_int_list(text: str) -> list[int | None]:
    out: list[int | None] = []
    for tok in [x.strip().lower() for x in str(text).split(",") if x.strip()]:
        if tok in ("none", "null"):
            out.append(None)
        else:
            out.append(int(tok))
    return out


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _worker_eval(params: tuple[float, float, float]) -> tuple[float, str | None]:
    like = _GLOBAL_LIKE
    if like is None:
        raise RuntimeError("Global strong-lens likelihood not initialized.")
    H0, om, ok = params
    try:
        ll = float(like.loglike(float(H0), float(om), float(ok)))
        if not np.isfinite(ll):
            return float("-inf"), "nonfinite_loglike"
        return ll, None
    except Exception as exc:
        return float("-inf"), f"{type(exc).__name__}: {exc}"


def _baseline_log_prob(theta: np.ndarray) -> float:
    like = _GLOBAL_BASELINE_LIKE
    if like is None:
        return float("-inf")
    H0, om = float(theta[0]), float(theta[1])
    if not (40.0 <= H0 <= 110.0 and 0.05 <= om <= 0.5):
        return float("-inf")
    try:
        ll = float(like.loglike(H0, om, 0.0))
    except Exception:
        return float("-inf")
    if not np.isfinite(ll):
        return float("-inf")
    return ll


def _run_sampler_loop(
    *,
    sampler: emcee.EnsembleSampler,
    backend: emcee.backends.HDFBackend,
    p0: np.ndarray | None,
    n_steps: int,
    current_iter: int,
    heartbeat_sec: float,
    partial_write_min_sec: float,
    out_tables: Path,
) -> None:
    t0 = time.time()
    t_last_hb = 0.0
    t_last_partial = 0.0
    remaining = max(0, int(n_steps) - int(current_iter))
    if remaining > 0:
        print(
            f"[calib-baseline] resume_iter={current_iter} target={n_steps} remaining={remaining}",
            flush=True,
        )
        for _state in sampler.sample(p0, iterations=remaining, progress=False):
            now = time.time()
            it = int(backend.iteration)
            if now - t_last_hb >= float(heartbeat_sec) or it >= int(n_steps):
                rate = max(it, 1) / max(now - t0, 1e-9)
                rem = max(0, int(n_steps) - it)
                eta = rem / max(rate, 1e-9)
                print(
                    f"[calib-baseline] iter={it}/{n_steps} ({100.0*it/max(1,n_steps):.1f}%) "
                    f"rate={rate:.2f}/s eta_min={eta/60.0:.1f}",
                    flush=True,
                )
                t_last_hb = now
            if now - t_last_partial >= float(partial_write_min_sec) or it >= int(n_steps):
                _write_json_atomic(
                    out_tables / "baseline_partial.json",
                    {
                        "updated_utc": _utc_now(),
                        "iteration": int(it),
                        "n_steps_target": int(n_steps),
                        "progress_pct": float(100.0 * it / max(1, n_steps)),
                    },
                )
                t_last_partial = now
    else:
        print(f"[calib-baseline] already complete at iter={current_iter}/{n_steps}", flush=True)


def _run_baseline_mcmc(
    *,
    out_tables: Path,
    out_figures: Path,
    like: H0LiCOW6Likelihood,
    n_walkers: int,
    n_steps: int,
    burn: int,
    thin: int,
    seed: int,
    procs: int,
    heartbeat_sec: float,
    partial_write_min_sec: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    backend_path = out_tables / "baseline_chain.h5"
    backend = emcee.backends.HDFBackend(str(backend_path))
    ndim = 2  # flat LCDM: [H0, Om]
    try:
        current_iter = int(backend.iteration)
    except (OSError, FileNotFoundError):
        current_iter = 0
    need_init = current_iter <= 0
    if need_init:
        backend.reset(int(n_walkers), ndim)
        current_iter = 0
        p0 = np.column_stack(
            [
                np.clip(73.0 + 5.0 * rng.normal(size=n_walkers), 40.0, 110.0),
                np.clip(0.30 + 0.06 * rng.normal(size=n_walkers), 0.05, 0.5),
            ]
        )
    else:
        p0 = None

    def _log_prob(theta: np.ndarray) -> float:
        H0, om = float(theta[0]), float(theta[1])
        if not (40.0 <= H0 <= 110.0 and 0.05 <= om <= 0.5):
            return float("-inf")
        ll = like.loglike(H0, om, 0.0)
        if not np.isfinite(ll):
            return float("-inf")
        return float(ll)

    baseline_procs = max(1, int(procs))
    if baseline_procs <= 1:
        sampler = emcee.EnsembleSampler(int(n_walkers), ndim, _log_prob, backend=backend)
        _run_sampler_loop(
            sampler=sampler,
            backend=backend,
            p0=p0,
            n_steps=int(n_steps),
            current_iter=int(current_iter),
            heartbeat_sec=float(heartbeat_sec),
            partial_write_min_sec=float(partial_write_min_sec),
            out_tables=out_tables,
        )
    else:
        global _GLOBAL_BASELINE_LIKE
        _GLOBAL_BASELINE_LIKE = like
        try:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(processes=baseline_procs) as pool:
                sampler = emcee.EnsembleSampler(
                    int(n_walkers),
                    ndim,
                    _baseline_log_prob,
                    backend=backend,
                    pool=pool,
                )
                _run_sampler_loop(
                    sampler=sampler,
                    backend=backend,
                    p0=p0,
                    n_steps=int(n_steps),
                    current_iter=int(current_iter),
                    heartbeat_sec=float(heartbeat_sec),
                    partial_write_min_sec=float(partial_write_min_sec),
                    out_tables=out_tables,
                )
        finally:
            _GLOBAL_BASELINE_LIKE = None

    flat = backend.get_chain(discard=int(burn), thin=max(1, int(thin)), flat=True)
    if flat.size == 0:
        raise RuntimeError("Baseline chain empty after burn/thin.")
    h0 = flat[:, 0]
    om = flat[:, 1]
    target_h0 = 73.3
    target_sigma = 1.8
    summary = {
        "model": "FlatLCDM",
        "n_walkers": int(n_walkers),
        "n_steps_target": int(n_steps),
        "n_steps_done": int(backend.iteration),
        "burn": int(burn),
        "thin": int(thin),
        "n_posterior_samples": int(flat.shape[0]),
        "H0": {
            "mean": float(np.mean(h0)),
            "p16": float(np.percentile(h0, 16.0)),
            "p50": float(np.percentile(h0, 50.0)),
            "p84": float(np.percentile(h0, 84.0)),
        },
        "Omega_m0": {
            "mean": float(np.mean(om)),
            "p16": float(np.percentile(om, 16.0)),
            "p50": float(np.percentile(om, 50.0)),
            "p84": float(np.percentile(om, 84.0)),
        },
        "reference_h0licow_wong2019": {
            "h0": target_h0,
            "sigma": target_sigma,
            "zscore_median": float((np.percentile(h0, 50.0) - target_h0) / target_sigma),
        },
    }
    _write_json_atomic(out_tables / "baseline_reproduction.json", summary)

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(h0, bins=60, density=True, alpha=0.8, label="Pipeline baseline posterior")
    grid = np.linspace(np.min(h0), np.max(h0), 400)
    ref = np.exp(-0.5 * ((grid - target_h0) / target_sigma) ** 2) / (target_sigma * np.sqrt(2.0 * np.pi))
    plt.plot(grid, ref, lw=2.0, label="Wong+2019 reference Gaussian")
    plt.xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    plt.ylabel("Density")
    plt.title("Strong-lens baseline reproduction")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_figures / "baseline_h0_reproduction.png", dpi=180)
    plt.close()
    return summary


@dataclass(frozen=True)
class SweepConfig:
    bandwidth_1d: float
    bandwidth_2d: float
    bins_1d: int
    bins_2d: int
    wfi_dt_max: float
    j1206_bins_2d: int | None

    def key(self) -> str:
        jv = "none" if self.j1206_bins_2d is None else str(self.j1206_bins_2d)
        return (
            f"b1d{self.bandwidth_1d:g}_b2d{self.bandwidth_2d:g}_"
            f"n1d{self.bins_1d}_n2d{self.bins_2d}_wfi{self.wfi_dt_max:g}_j{jv}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandwidth_1d": float(self.bandwidth_1d),
            "bandwidth_2d": float(self.bandwidth_2d),
            "bins_1d": int(self.bins_1d),
            "bins_2d": int(self.bins_2d),
            "wfi_dt_max": float(self.wfi_dt_max),
            "j1206_bins_2d": None if self.j1206_bins_2d is None else int(self.j1206_bins_2d),
        }


def _build_sweep_configs(
    *,
    bw1d: list[float],
    bw2d: list[float],
    bins1d: list[int],
    bins2d: list[int],
    wfi: list[float],
    j1206: list[int | None],
    max_configs: int,
) -> list[SweepConfig]:
    if not (bw1d and bw2d and bins1d and bins2d and wfi and j1206):
        raise ValueError("All sweep grids must be non-empty.")
    base = SweepConfig(
        bandwidth_1d=float(bw1d[0]),
        bandwidth_2d=float(bw2d[0]),
        bins_1d=int(bins1d[0]),
        bins_2d=int(bins2d[0]),
        wfi_dt_max=float(wfi[0]),
        j1206_bins_2d=j1206[0],
    )
    seen: dict[str, SweepConfig] = {base.key(): base}
    for v in bw1d[1:]:
        c = SweepConfig(v, base.bandwidth_2d, base.bins_1d, base.bins_2d, base.wfi_dt_max, base.j1206_bins_2d)
        seen[c.key()] = c
    for v in bw2d[1:]:
        c = SweepConfig(base.bandwidth_1d, v, base.bins_1d, base.bins_2d, base.wfi_dt_max, base.j1206_bins_2d)
        seen[c.key()] = c
    for v in bins1d[1:]:
        c = SweepConfig(base.bandwidth_1d, base.bandwidth_2d, v, base.bins_2d, base.wfi_dt_max, base.j1206_bins_2d)
        seen[c.key()] = c
    for v in bins2d[1:]:
        c = SweepConfig(base.bandwidth_1d, base.bandwidth_2d, base.bins_1d, v, base.wfi_dt_max, base.j1206_bins_2d)
        seen[c.key()] = c
    for v in wfi[1:]:
        c = SweepConfig(base.bandwidth_1d, base.bandwidth_2d, base.bins_1d, base.bins_2d, v, base.j1206_bins_2d)
        seen[c.key()] = c
    for v in j1206[1:]:
        c = SweepConfig(base.bandwidth_1d, base.bandwidth_2d, base.bins_1d, base.bins_2d, base.wfi_dt_max, v)
        seen[c.key()] = c

    # Add a few coupled extrema.
    extreme_low = SweepConfig(min(bw1d), min(bw2d), min(bins1d), min(bins2d), min(wfi), j1206[0])
    extreme_hi = SweepConfig(max(bw1d), max(bw2d), max(bins1d), max(bins2d), max(wfi), j1206[-1])
    seen[extreme_low.key()] = extreme_low
    seen[extreme_hi.key()] = extreme_hi

    out = list(seen.values())
    out.sort(key=lambda x: x.key())
    if len(out) > int(max_configs):
        out = out[: int(max_configs)]
    return out


def _select_params(post, n_eval: int, seed: int) -> list[tuple[float, float, float]]:
    H0 = np.asarray(post.H0, dtype=float)
    om = np.asarray(post.omega_m0, dtype=float)
    ok = np.asarray(post.omega_k0, dtype=float)
    n = int(H0.size)
    if n == 0:
        raise RuntimeError("No posterior draws.")
    m = int(n_eval) if int(n_eval) > 0 else n
    m = min(m, n)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    return [(float(H0[i]), float(om[i]), float(ok[i])) for i in idx]


def _eval_params(
    *,
    like: H0LiCOW6Likelihood,
    params: list[tuple[float, float, float]],
    procs: int,
    label: str,
    out_tables: Path,
    heartbeat_sec: float,
    partial_write_min_sec: float,
) -> tuple[np.ndarray, list[str]]:
    global _GLOBAL_LIKE
    _GLOBAL_LIKE = like
    t0 = time.time()
    t_last_hb = 0.0
    t_last_partial = 0.0
    lls: list[float] = []
    errs: list[str] = []
    finite_count = 0
    lse_anchor = 0.0
    lse_sum = 0.0

    def _update_logmeanexp_state(v: float) -> None:
        nonlocal finite_count, lse_anchor, lse_sum
        if not np.isfinite(v):
            return
        if finite_count == 0:
            finite_count = 1
            lse_anchor = float(v)
            lse_sum = 1.0
            return
        if v > lse_anchor:
            lse_sum = lse_sum * np.exp(lse_anchor - v) + 1.0
            lse_anchor = float(v)
        else:
            lse_sum += float(np.exp(v - lse_anchor))
        finite_count += 1

    def _partial_lpd() -> float:
        if finite_count <= 0:
            return float("-inf")
        return float(lse_anchor + np.log(lse_sum / float(finite_count)))

    try:
        if int(procs) <= 1:
            for i, p in enumerate(params, start=1):
                ll, err = _worker_eval(p)
                lls.append(ll)
                _update_logmeanexp_state(ll)
                if err:
                    errs.append(err)
                now = time.time()
                if now - t_last_hb >= float(heartbeat_sec) or i == len(params):
                    rate = i / max(now - t0, 1e-9)
                    eta = (len(params) - i) / max(rate, 1e-9)
                    print(f"[calib-sweep] {label} {i}/{len(params)} rate={rate:.2f}/s eta_min={eta/60.0:.1f}", flush=True)
                    t_last_hb = now
                if now - t_last_partial >= float(partial_write_min_sec) or i == len(params):
                    _write_json_atomic(
                        out_tables / f"{label}_partial.json",
                        {
                            "updated_utc": _utc_now(),
                            "label": label,
                            "done": int(i),
                            "total": int(len(params)),
                            "progress_pct": float(100.0 * i / len(params)),
                            "lpd_partial": _partial_lpd(),
                            "invalid_count": int(len(errs)),
                        },
                    )
                    t_last_partial = now
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(processes=int(procs)) as pool:
                chunksize = max(1, int(len(params) / max(1, int(procs) * 8)))
                it = pool.imap_unordered(_worker_eval, params, chunksize=chunksize)
                for i, (ll, err) in enumerate(it, start=1):
                    lls.append(ll)
                    _update_logmeanexp_state(ll)
                    if err:
                        errs.append(err)
                    now = time.time()
                    if now - t_last_hb >= float(heartbeat_sec) or i == len(params):
                        rate = i / max(now - t0, 1e-9)
                        eta = (len(params) - i) / max(rate, 1e-9)
                        print(f"[calib-sweep] {label} {i}/{len(params)} rate={rate:.2f}/s eta_min={eta/60.0:.1f}", flush=True)
                        t_last_hb = now
                    if now - t_last_partial >= float(partial_write_min_sec) or i == len(params):
                        _write_json_atomic(
                            out_tables / f"{label}_partial.json",
                            {
                                "updated_utc": _utc_now(),
                                "label": label,
                                "done": int(i),
                                "total": int(len(params)),
                                "progress_pct": float(100.0 * i / len(params)),
                                "lpd_partial": _partial_lpd(),
                                "invalid_count": int(len(errs)),
                            },
                        )
                        t_last_partial = now
    finally:
        _GLOBAL_LIKE = None
    return np.asarray(lls, dtype=float), errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Strong-lens calibration suite: baseline reproduction + KDE nuisance sweep.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-dir-mg", required=True)
    ap.add_argument("--run-dir-bh", required=True)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--partial-write-min-sec", type=float, default=30.0)

    ap.add_argument("--baseline-walkers", type=int, default=40)
    ap.add_argument("--baseline-steps", type=int, default=1800)
    ap.add_argument("--baseline-burn", type=int, default=600)
    ap.add_argument("--baseline-thin", type=int, default=2)

    ap.add_argument("--grid-bandwidth1d", type=str, default="20,30,40")
    ap.add_argument("--grid-bandwidth2d", type=str, default="20,30,40")
    ap.add_argument("--grid-bins1d", type=str, default="300,400,600")
    ap.add_argument("--grid-bins2d", type=str, default="60,80,120")
    ap.add_argument("--grid-wfi-dt-max", type=str, default="7000,8000,9000")
    ap.add_argument("--grid-j1206-bins2d", type=str, default="none,120")
    ap.add_argument("--max-configs", type=int, default=24)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"strong_lens_calibration_suite_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SUTC')}"
    tab = out_dir / "tables"
    fig = out_dir / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        out_dir / "manifest.json",
        {
            "created_utc": _utc_now(),
            "argv": [str(x) for x in os.sys.argv],
            "run_dir_mg": str(args.run_dir_mg),
            "run_dir_bh": str(args.run_dir_bh),
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    files = fetch_h0licow_distance_catalog(paths)

    # Phase 1: baseline reproduction
    base_like = H0LiCOW6Likelihood(files)
    baseline_summary = _run_baseline_mcmc(
        out_tables=tab,
        out_figures=fig,
        like=base_like,
        n_walkers=int(args.baseline_walkers),
        n_steps=int(args.baseline_steps),
        burn=int(args.baseline_burn),
        thin=int(args.baseline_thin),
        seed=int(args.seed),
        procs=max(1, min(int(args.procs) if int(args.procs) > 0 else int(os.cpu_count() or 1), int(args.baseline_walkers))),
        heartbeat_sec=float(args.heartbeat_sec),
        partial_write_min_sec=float(args.partial_write_min_sec),
    )

    # Phase 2: MG vs BH sensitivity to KDE nuisance settings
    mg_post = load_mu_forward_posterior(Path(args.run_dir_mg))
    bh_post = load_mu_forward_posterior(Path(args.run_dir_bh))
    mg_params = _select_params(mg_post, int(args.n_eval), int(args.seed) + 11)
    bh_params = _select_params(bh_post, int(args.n_eval), int(args.seed) + 29)

    bw1d = _parse_float_list(args.grid_bandwidth1d)
    bw2d = _parse_float_list(args.grid_bandwidth2d)
    bins1d = _parse_int_list(args.grid_bins1d)
    bins2d = _parse_int_list(args.grid_bins2d)
    wfi = _parse_float_list(args.grid_wfi_dt_max)
    j12 = _parse_optional_int_list(args.grid_j1206_bins2d)
    cfgs = _build_sweep_configs(
        bw1d=bw1d,
        bw2d=bw2d,
        bins1d=bins1d,
        bins2d=bins2d,
        wfi=wfi,
        j1206=j12,
        max_configs=int(args.max_configs),
    )
    procs = int(args.procs) if int(args.procs) > 0 else int(os.cpu_count() or 1)
    procs = max(1, min(procs, len(mg_params)))

    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(cfgs, start=1):
        label = f"cfg{i:03d}_{cfg.key()}"
        print(f"[calib-sweep] start {i}/{len(cfgs)} {label}", flush=True)
        like = H0LiCOW6Likelihood(files, **cfg.to_dict())
        ll_mg, err_mg = _eval_params(
            like=like,
            params=mg_params,
            procs=procs,
            label=f"{label}__mg",
            out_tables=tab,
            heartbeat_sec=float(args.heartbeat_sec),
            partial_write_min_sec=float(args.partial_write_min_sec),
        )
        ll_bh, err_bh = _eval_params(
            like=like,
            params=bh_params,
            procs=procs,
            label=f"{label}__bh",
            out_tables=tab,
            heartbeat_sec=float(args.heartbeat_sec),
            partial_write_min_sec=float(args.partial_write_min_sec),
        )
        mg_fin = ll_mg[np.isfinite(ll_mg)]
        bh_fin = ll_bh[np.isfinite(ll_bh)]
        if mg_fin.size == 0 or bh_fin.size == 0:
            raise RuntimeError(f"No finite likelihoods for {label}.")
        lpd_mg = _logmeanexp(mg_fin)
        lpd_bh = _logmeanexp(bh_fin)
        row = {
            "config_label": label,
            "config": cfg.to_dict(),
            "n_eval_mg": int(len(ll_mg)),
            "n_eval_bh": int(len(ll_bh)),
            "invalid_frac_mg": float(np.mean(~np.isfinite(ll_mg))),
            "invalid_frac_bh": float(np.mean(~np.isfinite(ll_bh))),
            "lpd_mg": float(lpd_mg),
            "lpd_bh": float(lpd_bh),
            "delta_lpd_mg_minus_bh": float(lpd_mg - lpd_bh),
            "errors_mg_preview": err_mg[:10],
            "errors_bh_preview": err_bh[:10],
        }
        rows.append(row)
        _write_json_atomic(tab / f"{label}_result.json", row)
        _write_json_atomic(
            tab / "sweep_partial.json",
            {
                "updated_utc": _utc_now(),
                "done": int(i),
                "total": int(len(cfgs)),
                "progress_pct": float(100.0 * i / len(cfgs)),
                "latest_label": label,
                "latest_delta_lpd_mg_minus_bh": float(row["delta_lpd_mg_minus_bh"]),
            },
        )

    deltas = np.asarray([r["delta_lpd_mg_minus_bh"] for r in rows], dtype=float)
    sweep_summary = {
        "n_configs": int(len(rows)),
        "delta_lpd_mg_minus_bh": {
            "mean": float(np.mean(deltas)),
            "sd": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "p_gt_0": float(np.mean(deltas > 0.0)),
        },
        "best_for_mg": max(rows, key=lambda r: r["delta_lpd_mg_minus_bh"]),
        "best_for_bh": min(rows, key=lambda r: r["delta_lpd_mg_minus_bh"]),
    }

    order = np.argsort(deltas)
    plt.figure(figsize=(9.0, 4.5))
    plt.plot(np.arange(len(deltas)), deltas[order], marker="o", lw=1.2)
    plt.axhline(0.0, color="k", ls="--", lw=1.0)
    plt.xlabel("KDE config rank")
    plt.ylabel("Delta LPD (MG - BH)")
    plt.title("Strong-lens KDE nuisance sweep")
    plt.tight_layout()
    plt.savefig(fig / "kde_nuisance_delta_lpd_ranked.png", dpi=180)
    plt.close()

    final = {
        "created_utc": _utc_now(),
        "catalog": "H0LiCOW_6_lens_distance_likelihood",
        "baseline_reproduction": baseline_summary,
        "kde_sweep_summary": sweep_summary,
        "rows": rows,
    }
    _write_json_atomic(tab / "calibration_suite_results.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
