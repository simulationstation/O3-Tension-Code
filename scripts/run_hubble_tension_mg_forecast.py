#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _q16_50_84(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 16.0)), float(np.percentile(x, 50.0)), float(np.percentile(x, 84.0))


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    p16, p50, p84 = _q16_50_84(x)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _parse_z_anchors(text: str) -> np.ndarray:
    vals: list[float] = []
    for tok in [t.strip() for t in str(text).split(",") if t.strip()]:
        vals.append(float(tok))
    if not vals:
        raise ValueError("No z anchors parsed.")
    out = np.array(sorted(set(vals)), dtype=float)
    if np.any(out <= 0.0):
        raise ValueError("All z anchors must be > 0.")
    return out


@dataclass(frozen=True)
class AnchorResult:
    z_anchor: float
    h0_highz_gr: dict[str, float]
    h0_highz_mg: dict[str, float]
    h0_local_obs: dict[str, float]
    delta_local_minus_highz_gr: dict[str, float]
    delta_local_minus_highz_mg: dict[str, float]
    local_minus_highz_gr_sigma: float
    local_minus_highz_mg_sigma: float
    apparent_gr_bias_vs_true: dict[str, float]
    p_gr_below_planck_ref: float
    p_gr_above_local_ref: float
    p_gr_below_local_ref: float


def _simulate_anchor(
    *,
    rng: np.random.Generator,
    z_anchor: float,
    h0_true_draws: np.ndarray,
    om_true_draws: np.ndarray,
    h_anchor_true_draws: np.ndarray,
    n_rep: int,
    sigma_highz_frac: float,
    highz_bias_frac: float,
    local_mode: str,
    local_ref: float,
    local_sigma: float,
    local_bias: float,
    gr_omega_mode: str,
    gr_omega_fixed: float,
    planck_ref: float,
) -> AnchorResult:
    n_draws = int(h0_true_draws.size)
    if n_draws <= 0:
        raise ValueError("No posterior draws available.")
    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")
    if sigma_highz_frac <= 0.0:
        raise ValueError("sigma_highz_frac must be positive.")
    if local_sigma <= 0.0:
        raise ValueError("local_sigma must be positive.")

    idx = rng.integers(0, n_draws, size=int(n_rep))
    h0_true = h0_true_draws[idx]
    om_true = om_true_draws[idx]
    h_anchor_true = h_anchor_true_draws[idx]

    h_anchor_obs = h_anchor_true * (
        1.0 + float(highz_bias_frac) + float(sigma_highz_frac) * rng.normal(size=n_rep)
    )
    if local_mode == "external":
        h0_local_obs = (
            float(local_ref)
            + float(local_bias)
            + float(local_sigma) * rng.normal(size=n_rep)
        )
    else:
        h0_local_obs = h0_true + float(local_bias) + float(local_sigma) * rng.normal(size=n_rep)

    e_true = h_anchor_true / h0_true
    h0_highz_mg = h_anchor_obs / e_true

    if gr_omega_mode == "sample":
        om_gr = om_true
    else:
        om_gr = np.full(n_rep, float(gr_omega_fixed), dtype=float)
    e_gr = np.sqrt(om_gr * (1.0 + float(z_anchor)) ** 3 + (1.0 - om_gr))
    h0_highz_gr = h_anchor_obs / e_gr

    delta_gr = h0_local_obs - h0_highz_gr
    delta_mg = h0_local_obs - h0_highz_mg
    bias_gr = h0_highz_gr - h0_true

    gr_sd = float(np.std(h0_highz_gr, ddof=1)) if n_rep > 1 else 0.0
    mg_sd = float(np.std(h0_highz_mg, ddof=1)) if n_rep > 1 else 0.0
    sigma_gap_gr = float((np.mean(h0_local_obs) - np.mean(h0_highz_gr)) / np.sqrt(local_sigma**2 + gr_sd**2 + 1e-12))
    sigma_gap_mg = float((np.mean(h0_local_obs) - np.mean(h0_highz_mg)) / np.sqrt(local_sigma**2 + mg_sd**2 + 1e-12))

    return AnchorResult(
        z_anchor=float(z_anchor),
        h0_highz_gr=_stats_1d(h0_highz_gr),
        h0_highz_mg=_stats_1d(h0_highz_mg),
        h0_local_obs=_stats_1d(h0_local_obs),
        delta_local_minus_highz_gr=_stats_1d(delta_gr),
        delta_local_minus_highz_mg=_stats_1d(delta_mg),
        local_minus_highz_gr_sigma=float(sigma_gap_gr),
        local_minus_highz_mg_sigma=float(sigma_gap_mg),
        apparent_gr_bias_vs_true=_stats_1d(bias_gr),
        p_gr_below_planck_ref=float(np.mean(h0_highz_gr < float(planck_ref))),
        p_gr_above_local_ref=float(np.mean(h0_highz_gr > float(local_ref))),
        p_gr_below_local_ref=float(np.mean(h0_highz_gr < float(local_ref))),
    )


def _plot_hz_ratio(*, z_eval: np.ndarray, ratio_q16: np.ndarray, ratio_q50: np.ndarray, ratio_q84: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7.6, 4.6))
    plt.fill_between(z_eval, ratio_q16, ratio_q84, alpha=0.25, linewidth=0.0, label="68% band")
    plt.plot(z_eval, ratio_q50, linewidth=2.0, label="median")
    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.5)
    plt.xlabel("z")
    plt.ylabel("H_MG(z) / H_LCDM,Planck(z)")
    plt.title("Expansion-ratio forecast under MG posterior truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_gr_bias_vs_z(*, z_eval: np.ndarray, bias_q16: np.ndarray, bias_q50: np.ndarray, bias_q84: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7.6, 4.6))
    plt.fill_between(z_eval, bias_q16, bias_q84, alpha=0.25, linewidth=0.0, label="68% band")
    plt.plot(z_eval, bias_q50, linewidth=2.0, label="median")
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    plt.xlabel("Anchor redshift z")
    plt.ylabel("Apparent H0 bias (GR-fit high-z anchor - true H0) [km/s/Mpc]")
    plt.title("Model-interpreted H0 bias if truth follows MG posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_anchor_inference(
    *,
    anchor_results: list[AnchorResult],
    local_ref: float,
    planck_ref: float,
    out: Path,
) -> None:
    z = np.array([r.z_anchor for r in anchor_results], dtype=float)
    gr_m = np.array([r.h0_highz_gr["mean"] for r in anchor_results], dtype=float)
    gr_s = np.array([r.h0_highz_gr["sd"] for r in anchor_results], dtype=float)
    mg_m = np.array([r.h0_highz_mg["mean"] for r in anchor_results], dtype=float)
    mg_s = np.array([r.h0_highz_mg["sd"] for r in anchor_results], dtype=float)

    plt.figure(figsize=(7.8, 4.8))
    plt.errorbar(z, gr_m, yerr=gr_s, fmt="o-", capsize=3, label="High-z inferred H0 (GR interpretation)")
    plt.errorbar(z, mg_m, yerr=mg_s, fmt="s--", capsize=3, label="High-z inferred H0 (MG interpretation)")
    plt.axhline(float(local_ref), color="tab:red", linewidth=1.2, alpha=0.7, label=f"Local reference ({local_ref:.2f})")
    plt.axhline(float(planck_ref), color="tab:green", linewidth=1.2, alpha=0.7, label=f"Planck reference ({planck_ref:.2f})")
    plt.xlabel("Anchor redshift z")
    plt.ylabel("Inferred H0 [km/s/Mpc]")
    plt.title("Synthetic high-z anchor inference under MG truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Forecast Hubble-tension signatures if MG posterior from a run_dir is treated as truth."
    )
    ap.add_argument("--run-dir", required=True, help="Finished mu-forward run directory containing samples/mu_forward_posterior.npz.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/hubble_tension_mg_forecast_<UTCSTAMP>).")
    ap.add_argument("--draws", type=int, default=4096, help="Number of posterior draws to use (subsampled if needed).")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--z-max", type=float, default=1.2, help="Max redshift for expansion-profile plots (must be <= posterior z_max).")
    ap.add_argument("--z-n", type=int, default=240, help="Number of points in z grid for profile plots.")
    ap.add_argument("--z-anchors", type=str, default="0.2,0.35,0.5,0.8,1.0,1.2", help="Comma list of high-z anchor points.")
    ap.add_argument("--n-rep", type=int, default=5000, help="MC replicates per anchor.")
    ap.add_argument("--sigma-highz-frac", type=float, default=0.01, help="Fractional uncertainty for synthetic H(z_anchor) observations.")
    ap.add_argument(
        "--highz-bias-frac",
        type=float,
        default=0.0,
        help="Deterministic fractional bias applied to synthetic high-z H observations.",
    )
    ap.add_argument("--local-mode", choices=["external", "truth"], default="external", help="How local H0 observations are generated.")
    ap.add_argument("--h0-local-ref", type=float, default=73.0, help="Local reference H0 (used when local_mode=external).")
    ap.add_argument("--h0-local-sigma", type=float, default=1.0, help="1-sigma uncertainty for local observations.")
    ap.add_argument(
        "--local-bias",
        type=float,
        default=0.0,
        help="Deterministic additive bias [km/s/Mpc] applied to local observations.",
    )
    ap.add_argument("--h0-planck-ref", type=float, default=67.4, help="Planck-like reference H0.")
    ap.add_argument("--h0-planck-sigma", type=float, default=0.5, help="Planck-like H0 uncertainty.")
    ap.add_argument("--omega-m-planck", type=float, default=0.315, help="Planck-like Omega_m for LCDM baseline profile.")
    ap.add_argument("--gr-omega-mode", choices=["sample", "fixed"], default="sample", help="Omega_m assumption when converting high-z anchors to H0 in GR interpretation.")
    ap.add_argument("--gr-omega-fixed", type=float, default=0.315, help="Fixed Omega_m if --gr-omega-mode=fixed.")
    ap.add_argument("--heartbeat-sec", type=float, default=60.0, help="Progress heartbeat interval.")
    ap.add_argument("--resume", action="store_true", help="Resume anchor results if per-anchor files already exist.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"hubble_tension_mg_forecast_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    post = load_mu_forward_posterior(args.run_dir)
    n_all = int(post.H_samples.shape[0])
    if n_all <= 0:
        raise ValueError("Posterior has zero draws.")
    n_use = int(min(max(1, int(args.draws)), n_all))

    rng_main = np.random.default_rng(int(args.seed))
    if n_use < n_all:
        keep = np.sort(rng_main.choice(n_all, size=n_use, replace=False))
    else:
        keep = np.arange(n_all, dtype=int)

    h0 = np.asarray(post.H0[keep], dtype=float)
    om = np.asarray(post.omega_m0[keep], dtype=float)
    hz = np.asarray(post.H_samples[keep], dtype=float)

    z_max_post = float(post.z_grid[-1])
    z_max = float(args.z_max)
    if not (0.0 < z_max <= z_max_post):
        raise ValueError(f"--z-max must be in (0, {z_max_post:.4g}]")
    z_eval = np.linspace(0.0, z_max, int(args.z_n))
    if z_eval.size < 4:
        raise ValueError("--z-n must be >= 4.")

    z_anchors = _parse_z_anchors(args.z_anchors)
    if float(np.max(z_anchors)) > z_max + 1e-12:
        raise ValueError("All z anchors must be <= z_max.")

    h_interp = np.empty((n_use, z_eval.size), dtype=float)
    for j in range(n_use):
        h_interp[j] = np.interp(z_eval, post.z_grid, hz[j])
    h_planck = float(args.h0_planck_ref) * np.sqrt(float(args.omega_m_planck) * (1.0 + z_eval) ** 3 + (1.0 - float(args.omega_m_planck)))
    h_ratio = h_interp / h_planck.reshape((1, -1))
    ratio_q16 = np.percentile(h_ratio, 16.0, axis=0)
    ratio_q50 = np.percentile(h_ratio, 50.0, axis=0)
    ratio_q84 = np.percentile(h_ratio, 84.0, axis=0)
    _plot_hz_ratio(
        z_eval=z_eval,
        ratio_q16=ratio_q16,
        ratio_q50=ratio_q50,
        ratio_q84=ratio_q84,
        out=fig_dir / "h_ratio_vs_planck.png",
    )

    if str(args.gr_omega_mode) == "sample":
        e_gr_grid = np.sqrt(om.reshape((-1, 1)) * (1.0 + z_eval.reshape((1, -1))) ** 3 + (1.0 - om.reshape((-1, 1))))
    else:
        e_gr_grid = np.sqrt(float(args.gr_omega_fixed) * (1.0 + z_eval.reshape((1, -1))) ** 3 + (1.0 - float(args.gr_omega_fixed)))
    h0_gr_from_true = h_interp / e_gr_grid
    bias_grid = h0_gr_from_true - h0.reshape((-1, 1))
    bias_q16 = np.percentile(bias_grid, 16.0, axis=0)
    bias_q50 = np.percentile(bias_grid, 50.0, axis=0)
    bias_q84 = np.percentile(bias_grid, 84.0, axis=0)
    _plot_gr_bias_vs_z(
        z_eval=z_eval,
        bias_q16=bias_q16,
        bias_q50=bias_q50,
        bias_q84=bias_q84,
        out=fig_dir / "h0_apparent_gr_bias_vs_z.png",
    )

    anchor_results: list[AnchorResult] = []
    last_hb = 0.0
    progress_path = tab_dir / "progress.json"
    for i, z_anchor in enumerate(z_anchors):
        label = f"{float(z_anchor):.3f}".replace(".", "p")
        anchor_path = tab_dir / f"anchor_z{label}.json"
        if bool(args.resume) and anchor_path.exists():
            data = json.loads(anchor_path.read_text(encoding="utf-8"))
            anchor_results.append(AnchorResult(**data))
        else:
            h_anchor = np.array([np.interp(float(z_anchor), z_eval, h_interp[j]) for j in range(n_use)], dtype=float)
            anchor_rng = np.random.default_rng(int(args.seed) + 10000 + int(i))
            result = _simulate_anchor(
                rng=anchor_rng,
                z_anchor=float(z_anchor),
                h0_true_draws=h0,
                om_true_draws=om,
                h_anchor_true_draws=h_anchor,
                n_rep=int(args.n_rep),
                sigma_highz_frac=float(args.sigma_highz_frac),
                highz_bias_frac=float(args.highz_bias_frac),
                local_mode=str(args.local_mode),
                local_ref=float(args.h0_local_ref),
                local_sigma=float(args.h0_local_sigma),
                local_bias=float(args.local_bias),
                gr_omega_mode=str(args.gr_omega_mode),
                gr_omega_fixed=float(args.gr_omega_fixed),
                planck_ref=float(args.h0_planck_ref),
            )
            anchor_results.append(result)
            _write_json_atomic(anchor_path, asdict(result))

        now = time.time()
        if (now - last_hb) >= float(args.heartbeat_sec) or i == (len(z_anchors) - 1):
            _write_json_atomic(
                progress_path,
                {
                    "updated_utc": _utc_stamp(),
                    "anchors_done": int(i + 1),
                    "anchors_total": int(len(z_anchors)),
                    "anchors_pct": float(100.0 * (i + 1) / max(1, len(z_anchors))),
                    "elapsed_sec": float(now - t_start),
                    "current_anchor_z": float(z_anchor),
                    "n_rep_per_anchor": int(args.n_rep),
                    "draws_used": int(n_use),
                },
            )
            print(
                f"[heartbeat] anchors_done={int(i + 1)}/{int(len(z_anchors))} "
                f"({100.0 * (i + 1) / max(1, len(z_anchors)):.1f}%) "
                f"current_z={float(z_anchor):.3f} elapsed_sec={now - t_start:.1f}",
                flush=True,
            )
            last_hb = now

    anchor_results = sorted(anchor_results, key=lambda r: float(r.z_anchor))
    _plot_anchor_inference(
        anchor_results=anchor_results,
        local_ref=float(args.h0_local_ref),
        planck_ref=float(args.h0_planck_ref),
        out=fig_dir / "anchor_h0_inference.png",
    )

    h0_stats = _stats_1d(h0)
    local_ref = float(args.h0_local_ref)
    local_sigma = float(args.h0_local_sigma)
    planck_ref = float(args.h0_planck_ref)
    planck_sigma = float(args.h0_planck_sigma)
    baseline_gap = abs(local_ref - planck_ref)
    mg_gap = abs(local_ref - float(h0_stats["p50"]))
    relief = 1.0 - (mg_gap / baseline_gap if baseline_gap > 0.0 else np.nan)

    anchor_gr_means = np.asarray([float(r.h0_highz_gr["mean"]) for r in anchor_results], dtype=float)
    anchor_gr_sds = np.asarray([float(r.h0_highz_gr["sd"]) for r in anchor_results], dtype=float)
    anchor_mg_means = np.asarray([float(r.h0_highz_mg["mean"]) for r in anchor_results], dtype=float)
    anchor_mg_sds = np.asarray([float(r.h0_highz_mg["sd"]) for r in anchor_results], dtype=float)
    anchor_local_means = np.asarray([float(r.h0_local_obs["mean"]) for r in anchor_results], dtype=float)
    anchor_local_sds = np.asarray([float(r.h0_local_obs["sd"]) for r in anchor_results], dtype=float)

    h0_gr_anchor_mean = float(np.mean(anchor_gr_means))
    h0_mg_anchor_mean = float(np.mean(anchor_mg_means))
    h0_local_anchor_mean = float(np.mean(anchor_local_means))
    h0_gr_anchor_internal_sd = float(np.sqrt(np.mean(anchor_gr_sds**2)))
    h0_mg_anchor_internal_sd = float(np.sqrt(np.mean(anchor_mg_sds**2)))
    h0_local_anchor_internal_sd = float(np.sqrt(np.mean(anchor_local_sds**2)))
    h0_gr_anchor_between_z_sd = float(np.std(anchor_gr_means, ddof=1)) if anchor_gr_means.size > 1 else 0.0
    h0_mg_anchor_between_z_sd = float(np.std(anchor_mg_means, ddof=1)) if anchor_mg_means.size > 1 else 0.0

    gap_local_vs_gr_anchor = float(h0_local_anchor_mean - h0_gr_anchor_mean)
    gap_local_vs_mg_anchor = float(h0_local_anchor_mean - h0_mg_anchor_mean)
    gap_sigma_gr_anchor = float(
        gap_local_vs_gr_anchor / np.sqrt(h0_local_anchor_internal_sd**2 + h0_gr_anchor_internal_sd**2 + 1e-12)
    )
    gap_sigma_mg_anchor = float(
        gap_local_vs_mg_anchor / np.sqrt(h0_local_anchor_internal_sd**2 + h0_mg_anchor_internal_sd**2 + 1e-12)
    )
    relief_anchor_gr = 1.0 - (abs(local_ref - h0_gr_anchor_mean) / baseline_gap if baseline_gap > 0.0 else np.nan)
    relief_anchor_mg = 1.0 - (abs(local_ref - h0_mg_anchor_mean) / baseline_gap if baseline_gap > 0.0 else np.nan)

    summary = {
        "timestamp_utc": _utc_stamp(),
        "run_dir": str(Path(args.run_dir).resolve()),
        "draws_used": int(n_use),
        "draws_total": int(n_all),
        "z_max": float(z_max),
        "z_n": int(args.z_n),
        "z_anchors": [float(z) for z in z_anchors.tolist()],
        "n_rep_per_anchor": int(args.n_rep),
        "h0_posterior_mg_truth": h0_stats,
        "references": {
            "h0_local_ref": float(args.h0_local_ref),
            "h0_local_sigma": float(args.h0_local_sigma),
            "local_bias": float(args.local_bias),
            "h0_planck_ref": float(args.h0_planck_ref),
            "h0_planck_sigma": float(args.h0_planck_sigma),
            "omega_m_planck": float(args.omega_m_planck),
            "local_mode": str(args.local_mode),
            "gr_omega_mode": str(args.gr_omega_mode),
            "gr_omega_fixed": float(args.gr_omega_fixed),
            "sigma_highz_frac": float(args.sigma_highz_frac),
            "highz_bias_frac": float(args.highz_bias_frac),
        },
        "h0_tension_projection": {
            "baseline_local_minus_planck": float(local_ref - planck_ref),
            "mg_p50_minus_planck": float(h0_stats["p50"] - planck_ref),
            "mg_p50_minus_local": float(h0_stats["p50"] - local_ref),
            "mg_sigma_vs_planck": float((h0_stats["p50"] - planck_ref) / np.sqrt(planck_sigma**2 + h0_stats["sd"] ** 2 + 1e-12)),
            "mg_sigma_vs_local": float((h0_stats["p50"] - local_ref) / np.sqrt(local_sigma**2 + h0_stats["sd"] ** 2 + 1e-12)),
            "tension_relief_fraction_vs_planck_local_baseline": float(relief),
            "anchor_h0_gr_mean": float(h0_gr_anchor_mean),
            "anchor_h0_gr_internal_sd": float(h0_gr_anchor_internal_sd),
            "anchor_h0_gr_between_z_sd": float(h0_gr_anchor_between_z_sd),
            "anchor_h0_mg_mean": float(h0_mg_anchor_mean),
            "anchor_h0_mg_internal_sd": float(h0_mg_anchor_internal_sd),
            "anchor_h0_mg_between_z_sd": float(h0_mg_anchor_between_z_sd),
            "anchor_h0_local_mean": float(h0_local_anchor_mean),
            "anchor_h0_local_internal_sd": float(h0_local_anchor_internal_sd),
            "anchor_gap_local_minus_gr": float(gap_local_vs_gr_anchor),
            "anchor_gap_local_minus_mg": float(gap_local_vs_mg_anchor),
            "anchor_gap_local_minus_gr_sigma": float(gap_sigma_gr_anchor),
            "anchor_gap_local_minus_mg_sigma": float(gap_sigma_mg_anchor),
            "tension_relief_fraction_anchor_gr": float(relief_anchor_gr),
            "tension_relief_fraction_anchor_mg": float(relief_anchor_mg),
        },
        "anchor_results": [asdict(r) for r in anchor_results],
    }
    _write_json_atomic(tab_dir / "summary.json", summary)
    _write_json_atomic(tab_dir / "expansion_profile_quantiles.json", {
        "z": [float(z) for z in z_eval.tolist()],
        "h_ratio_planck_q16": [float(x) for x in ratio_q16.tolist()],
        "h_ratio_planck_q50": [float(x) for x in ratio_q50.tolist()],
        "h_ratio_planck_q84": [float(x) for x in ratio_q84.tolist()],
        "h0_apparent_gr_bias_q16": [float(x) for x in bias_q16.tolist()],
        "h0_apparent_gr_bias_q50": [float(x) for x in bias_q50.tolist()],
        "h0_apparent_gr_bias_q84": [float(x) for x in bias_q84.tolist()],
    })

    lines = [
        "# Hubble-tension forecast under MG-posterior truth",
        "",
        f"- Run dir: `{Path(args.run_dir).resolve()}`",
        f"- Draws used: `{int(n_use)}` / `{int(n_all)}`",
        f"- Anchor replicates per z: `{int(args.n_rep)}`",
        f"- H0 (MG truth posterior, p16/p50/p84): `{h0_stats['p16']:.3f}`, `{h0_stats['p50']:.3f}`, `{h0_stats['p84']:.3f}` km/s/Mpc",
        f"- Local reference: `{float(args.h0_local_ref):.3f} ± {float(args.h0_local_sigma):.3f}` km/s/Mpc",
        f"- Injected local bias: `{float(args.local_bias):.4f}` km/s/Mpc",
        f"- Planck reference: `{float(args.h0_planck_ref):.3f} ± {float(args.h0_planck_sigma):.3f}` km/s/Mpc",
        f"- Injected high-z fractional bias: `{float(args.highz_bias_frac):.4f}`",
        f"- Tension-relief fraction vs local-planck baseline: `{float(relief):.3f}`",
        f"- Anchor-based relief (GR-interpreted high-z): `{float(relief_anchor_gr):.3f}`",
        f"- Anchor local-vs-high-z gap sigma (GR): `{float(gap_sigma_gr_anchor):.3f}`",
        "",
        "## Anchor highlights (GR interpretation)",
        "",
    ]
    for r in anchor_results:
        lines.append(
            f"- z={r.z_anchor:.3f}: inferred H0_GR p50={r.h0_highz_gr['p50']:.3f}, "
            f"local-highz gap sigma={r.local_minus_highz_gr_sigma:.3f}, "
            f"apparent bias p50={r.apparent_gr_bias_vs_true['p50']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/expansion_profile_quantiles.json`",
            "- `figures/h_ratio_vs_planck.png`",
            "- `figures/h0_apparent_gr_bias_vs_z.png`",
            "- `figures/anchor_h0_inference.png`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote forecast outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
