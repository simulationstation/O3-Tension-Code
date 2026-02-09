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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.strong_lens_time_delay import (
    H0LiCOW6Likelihood,
    fetch_h0licow_distance_catalog,
    fetch_tdcosmo2025_chain_release,
    fetch_tdcosmo_sample_posteriors,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_GLOBAL_LIKE: H0LiCOW6Likelihood | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _worker_eval(params: tuple[float, float, float]) -> tuple[float, str | None]:
    like = _GLOBAL_LIKE
    if like is None:
        raise RuntimeError("Worker likelihood not initialized.")
    H0, om, ok = params
    try:
        ll = like.loglike(float(H0), float(om), float(ok))
        if not np.isfinite(ll):
            return float("-inf"), "nonfinite_loglike"
        return float(ll), None
    except Exception as exc:
        return float("-inf"), f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: str
    run: str
    n_eval: int
    invalid_frac: float
    lpd: float
    ll_mean: float
    ll_p16: float
    ll_p50: float
    ll_p84: float


def _run_id(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    if "outputs" in parts:
        i = parts.index("outputs")
        tail = parts[i + 1 :]
    else:
        tail = parts[-4:]
    text = "__".join(tail)
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)


def _evaluate_run(
    *,
    run_dir: Path,
    out_tables: Path,
    like: H0LiCOW6Likelihood,
    n_eval: int,
    seed: int,
    procs: int,
    heartbeat_sec: float,
    partial_write_min_sec: float,
) -> RunResult:
    post = load_mu_forward_posterior(run_dir)
    H0 = np.asarray(post.H0, dtype=float)
    om = np.asarray(post.omega_m0, dtype=float)
    ok = np.asarray(post.omega_k0, dtype=float)
    n_draws = int(H0.size)
    if n_draws == 0:
        raise RuntimeError(f"No posterior draws in {run_dir}")
    if not (om.size == n_draws and ok.size == n_draws):
        raise RuntimeError(f"Inconsistent posterior arrays in {run_dir}")

    n = int(n_eval) if int(n_eval) > 0 else n_draws
    n = min(n, n_draws)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_draws, size=n, replace=False) if n < n_draws else np.arange(n_draws)
    params = [(float(H0[i]), float(om[i]), float(ok[i])) for i in idx]

    run_id = _run_id(run_dir)
    partial_path = out_tables / f"{run_id}_partial.json"
    errors: list[str] = []
    ll_vals: list[float] = []

    t0 = time.time()
    t_last_hb = 0.0
    t_last_partial = 0.0
    print(
        f"[strong-lens] start run={run_id} n_eval={len(params)} procs={procs} at={datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )

    global _GLOBAL_LIKE
    _GLOBAL_LIKE = like
    try:
        if int(procs) <= 1:
            iterator = enumerate(map(_worker_eval, params), start=1)
            for done, (ll, err) in iterator:
                ll_vals.append(ll)
                if err:
                    errors.append(err)
                now = time.time()
                if now - t_last_hb >= float(heartbeat_sec) or done == len(params):
                    rate = done / max(now - t0, 1e-9)
                    eta = (len(params) - done) / max(rate, 1e-9)
                    print(
                        f"[strong-lens] {run_id} {done}/{len(params)} ({100.0*done/len(params):.1f}%) "
                        f"rate={rate:.3f}/s eta_min={eta/60.0:.1f}",
                        flush=True,
                    )
                    t_last_hb = now
                if now - t_last_partial >= float(partial_write_min_sec) or done == len(params):
                    finite = np.asarray([x for x in ll_vals if np.isfinite(x)], dtype=float)
                    lpd_est = _logmeanexp(finite) if finite.size else float("-inf")
                    _write_json_atomic(
                        partial_path,
                        {
                            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run": run_id,
                            "done": int(done),
                            "total": int(len(params)),
                            "progress_pct": float(100.0 * done / len(params)),
                            "lpd_partial": float(lpd_est),
                            "invalid_count": int(len(errors)),
                            "invalid_frac": float(len(errors) / done),
                        },
                    )
                    t_last_partial = now
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(processes=int(procs)) as pool:
                it = pool.imap_unordered(_worker_eval, params, chunksize=1)
                for done, (ll, err) in enumerate(it, start=1):
                    ll_vals.append(ll)
                    if err:
                        errors.append(err)
                    now = time.time()
                    if now - t_last_hb >= float(heartbeat_sec) or done == len(params):
                        rate = done / max(now - t0, 1e-9)
                        eta = (len(params) - done) / max(rate, 1e-9)
                        print(
                            f"[strong-lens] {run_id} {done}/{len(params)} ({100.0*done/len(params):.1f}%) "
                            f"rate={rate:.3f}/s eta_min={eta/60.0:.1f}",
                            flush=True,
                        )
                        t_last_hb = now
                    if now - t_last_partial >= float(partial_write_min_sec) or done == len(params):
                        finite = np.asarray([x for x in ll_vals if np.isfinite(x)], dtype=float)
                        lpd_est = _logmeanexp(finite) if finite.size else float("-inf")
                        _write_json_atomic(
                            partial_path,
                            {
                                "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "run_dir": str(run_dir),
                                "run": run_id,
                                "done": int(done),
                                "total": int(len(params)),
                                "progress_pct": float(100.0 * done / len(params)),
                                "lpd_partial": float(lpd_est),
                                "invalid_count": int(len(errors)),
                                "invalid_frac": float(len(errors) / done),
                            },
                        )
                        t_last_partial = now
    finally:
        _GLOBAL_LIKE = None

    ll = np.asarray(ll_vals, dtype=float)
    finite = ll[np.isfinite(ll)]
    if finite.size == 0:
        raise RuntimeError(f"All strong-lens likelihood evaluations failed for {run_dir}")
    out = RunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        run=run_id,
        n_eval=int(len(params)),
        invalid_frac=float(np.mean(~np.isfinite(ll))),
        lpd=float(_logmeanexp(finite)),
        ll_mean=float(np.mean(finite)),
        ll_p16=float(np.percentile(finite, 16.0)),
        ll_p50=float(np.percentile(finite, 50.0)),
        ll_p84=float(np.percentile(finite, 84.0)),
    )
    _write_json_atomic(
        out_tables / f"{run_id}_result.json",
        {
            **out.__dict__,
            "errors_preview": errors[:20],
            "error_count": int(len(errors)),
        },
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Real strong-lens (H0LiCOW 6-lens) catalog external test.")
    ap.add_argument("--run-dir", action="append", required=True, help="Run directory with samples/mu_forward_posterior.npz")
    ap.add_argument("--out", default=None, help="Output dir (default outputs/strong_lens_real_catalog_<UTC>)")
    ap.add_argument("--n-eval", type=int, default=256, help="Posterior draws to evaluate per run (0=all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=0, help="Workers (0=cpu_count)")
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--partial-write-min-sec", type=float, default=30.0)
    ap.add_argument("--fetch-tdcosmo2025", action="store_true", help="Also download full TDCOSMO2025 chain release for local completeness.")
    ap.add_argument("--fetch-tdcosmo-sample", action="store_true", help="Also download per-lens TDCOSMO sample posterior release.")
    ap.add_argument("--raw-kde", action="store_true", help="Disable histogram compression and use full released chain samples in KDEs.")
    ap.add_argument("--bandwidth-1d", type=float, default=20.0)
    ap.add_argument("--bandwidth-2d", type=float, default=20.0)
    ap.add_argument("--bins-1d", type=int, default=400)
    ap.add_argument("--bins-2d", type=int, default=80)
    ap.add_argument("--j1206-bins-2d", type=int, default=-1, help="-1 means full released samples.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"strong_lens_real_catalog_{_utc_stamp()}"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    h0licow_files = fetch_h0licow_distance_catalog(paths)
    if args.fetch_tdcosmo2025:
        fetch_tdcosmo2025_chain_release(paths)
    if args.fetch_tdcosmo_sample:
        fetch_tdcosmo_sample_posteriors(paths)

    bins_1d = None if bool(args.raw_kde) else int(args.bins_1d)
    bins_2d = None if bool(args.raw_kde) else int(args.bins_2d)
    j1206_bins = None if int(args.j1206_bins_2d) < 0 else int(args.j1206_bins_2d)
    if bool(args.raw_kde):
        j1206_bins = None

    like = H0LiCOW6Likelihood(
        h0licow_files,
        bandwidth_1d=float(args.bandwidth_1d),
        bandwidth_2d=float(args.bandwidth_2d),
        bins_1d=bins_1d,
        bins_2d=bins_2d,
        j1206_bins_2d=j1206_bins,
    )
    procs = int(args.procs) if int(args.procs) > 0 else int(os.cpu_count() or 1)
    procs = max(1, procs)

    rows: list[RunResult] = []
    for i, run in enumerate(args.run_dir):
        rr = _evaluate_run(
            run_dir=Path(run),
            out_tables=tab_dir,
            like=like,
            n_eval=int(args.n_eval),
            seed=int(args.seed) + i,
            procs=int(procs),
            heartbeat_sec=float(args.heartbeat_sec),
            partial_write_min_sec=float(args.partial_write_min_sec),
        )
        rows.append(rr)

    deltas: list[dict[str, Any]] = []
    if len(rows) >= 2:
        by_id = {r.run_id: r for r in rows}
        ids = list(by_id.keys())
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                deltas.append(
                    {
                        "run_a": a,
                        "run_b": b,
                        "delta_lpd_a_minus_b": float(by_id[a].lpd - by_id[b].lpd),
                    }
                )

    results = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "catalog": "H0LiCOW_6_lens_distance_likelihood",
        "likelihood_config": like.config(),
        "files": {k: str(v) for k, v in h0licow_files.items()},
        "n_runs": int(len(rows)),
        "rows": [r.__dict__ for r in rows],
        "deltas": deltas,
    }
    _write_json_atomic(tab_dir / "results.json", results)

    if rows:
        plt.figure(figsize=(8.0, 4.2))
        x = np.arange(len(rows))
        y = np.array([r.lpd for r in rows], dtype=float)
        lbl = [r.run for r in rows]
        plt.bar(x, y, alpha=0.85)
        plt.xticks(x, lbl, rotation=20, ha="right")
        plt.ylabel("Strong-lens LPD")
        plt.title("Real strong-lens catalog score by run")
        plt.tight_layout()
        plt.savefig(fig_dir / "strong_lens_lpd_by_run.png", dpi=180)
        plt.close()

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
