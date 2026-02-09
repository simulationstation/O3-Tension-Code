#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, x))


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    w = weights / np.sum(weights)
    mean = float(np.sum(w * values))
    var = float(np.sum(w * (values - mean) ** 2))
    return {
        "mean": mean,
        "sd": float(np.sqrt(max(0.0, var))),
        "p16": _weighted_quantile(values, w, 0.16),
        "p50": _weighted_quantile(values, w, 0.50),
        "p84": _weighted_quantile(values, w, 0.84),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _row_weights(df: pd.DataFrame, sigma_hz: float, sigma_local: float) -> np.ndarray:
    hb = df["highz_bias_frac"].to_numpy(dtype=float)
    lb = df["local_bias"].to_numpy(dtype=float)
    wh = np.exp(-0.5 * (hb / float(sigma_hz)) ** 2)
    wl = np.exp(-0.5 * (lb / float(sigma_local)) ** 2)
    w = wh * wl
    w = w / np.sum(w)
    return w


def _estimate_mc_sigma(run_a: pd.DataFrame, run_b: pd.DataFrame) -> dict[str, float]:
    key_cols = ["run_dir", "highz_bias_frac", "local_bias"]
    merged = run_a.merge(run_b, on=key_cols, suffixes=("_a", "_b"))
    if merged.empty:
        return {
            "n_pairs": 0,
            "anchor_gr_relief_sigma_mc": float("nan"),
            "anchor_gr_relief_diff_sd": float("nan"),
            "anchor_gr_relief_diff_mean": float("nan"),
        }
    d = merged["anchor_gr_relief_b"].to_numpy(dtype=float) - merged["anchor_gr_relief_a"].to_numpy(dtype=float)
    sd_diff = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    sigma_mc = sd_diff / np.sqrt(2.0)
    return {
        "n_pairs": int(d.size),
        "anchor_gr_relief_sigma_mc": float(sigma_mc),
        "anchor_gr_relief_diff_sd": float(sd_diff),
        "anchor_gr_relief_diff_mean": float(np.mean(d)),
    }


def _plot_relief_vs_highz_bias(
    *,
    pilot_df: pd.DataFrame,
    constrained_df: pd.DataFrame,
    fit_intercept: float,
    fit_slope: float,
    out_path: Path,
) -> None:
    g_pilot = pilot_df.groupby("highz_bias_frac")["anchor_gr_relief"].agg(["mean", "min", "max"]).reset_index()
    g_cons = constrained_df.groupby("highz_bias_frac")["anchor_gr_relief"].agg(["mean", "min", "max"]).reset_index()

    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(
        g_pilot["highz_bias_frac"],
        g_pilot["mean"],
        yerr=[g_pilot["mean"] - g_pilot["min"], g_pilot["max"] - g_pilot["mean"]],
        fmt="o",
        capsize=3,
        label="Pilot sweep means (range)",
    )
    plt.errorbar(
        g_cons["highz_bias_frac"],
        g_cons["mean"],
        yerr=[g_cons["mean"] - g_cons["min"], g_cons["max"] - g_cons["mean"]],
        fmt="s",
        capsize=3,
        label="Constrained sweep means (range)",
    )
    x = np.linspace(min(g_pilot["highz_bias_frac"]) - 0.001, max(g_pilot["highz_bias_frac"]) + 0.001, 200)
    y = fit_intercept + fit_slope * x
    plt.plot(x, y, "-", label="Linear fit (pilot means)")
    plt.axhline(0.10, color="tab:red", alpha=0.6, linewidth=1.0, label="10% relief threshold")
    plt.axhline(0.40, color="tab:purple", alpha=0.6, linewidth=1.0, label="40% relief threshold")
    plt.xlabel("Injected high-z calibration bias fraction")
    plt.ylabel("Anchor-based relief fraction")
    plt.title("Relief sensitivity to high-z calibration bias")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute final relief posterior and bias thresholds from constrained + pilot sweeps."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--constrained-csvs",
        required=True,
        help="Comma list of constrained sweep_results.csv paths (e.g., baseline+repeat).",
    )
    ap.add_argument("--pilot-csv", required=True, help="Pilot sweep_results.csv path for threshold fit.")
    ap.add_argument("--highz-prior-sigma", type=float, default=0.003)
    ap.add_argument("--local-prior-sigma", type=float, default=0.25)
    ap.add_argument("--mc-samples", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    constrained_paths = [Path(p.strip()).resolve() for p in str(args.constrained_csvs).split(",") if p.strip()]
    pilot_path = Path(args.pilot_csv).resolve()

    for p in constrained_paths + [pilot_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV: {p}")

    constrained_runs: list[pd.DataFrame] = [pd.read_csv(p) for p in constrained_paths]
    constrained_all = pd.concat(constrained_runs, ignore_index=True)
    pilot_df = pd.read_csv(pilot_path)

    # Weighted posterior over constrained cases (priors on high-z/local biases)
    w = _row_weights(constrained_all, sigma_hz=float(args.highz_prior_sigma), sigma_local=float(args.local_prior_sigma))
    relief = constrained_all["anchor_gr_relief"].to_numpy(dtype=float)
    base_stats = _weighted_stats(relief, w)

    # Finite-MC calibration from matched reruns (first two constrained runs, if present)
    if len(constrained_runs) >= 2:
        mc_est = _estimate_mc_sigma(constrained_runs[0], constrained_runs[1])
    else:
        mc_est = {
            "n_pairs": 0,
            "anchor_gr_relief_sigma_mc": 0.0,
            "anchor_gr_relief_diff_sd": 0.0,
            "anchor_gr_relief_diff_mean": 0.0,
        }

    sigma_mc = float(mc_est.get("anchor_gr_relief_sigma_mc", 0.0))
    rng = np.random.default_rng(int(args.seed))
    n = int(max(10000, args.mc_samples))
    idx = rng.choice(np.arange(relief.size), size=n, replace=True, p=w)
    draws = relief[idx] + sigma_mc * rng.normal(size=n)
    draw_stats = {
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "p16": float(np.percentile(draws, 16.0)),
        "p50": float(np.percentile(draws, 50.0)),
        "p84": float(np.percentile(draws, 84.0)),
        "min": float(np.min(draws)),
        "max": float(np.max(draws)),
    }

    # Simple uncertainty-budget slices
    g_seed = constrained_all.groupby("run_dir")["anchor_gr_relief"].mean()
    g_hb = constrained_all.groupby("highz_bias_frac")["anchor_gr_relief"].mean()
    g_lb = constrained_all.groupby("local_bias")["anchor_gr_relief"].mean()

    budget = {
        "seed_mean_sd": float(np.std(g_seed.to_numpy(dtype=float), ddof=1)),
        "highz_bias_mean_range": float(np.max(g_hb.to_numpy(dtype=float)) - np.min(g_hb.to_numpy(dtype=float))),
        "local_bias_mean_range": float(np.max(g_lb.to_numpy(dtype=float)) - np.min(g_lb.to_numpy(dtype=float))),
        "mc_sigma_from_repeatability": float(sigma_mc),
    }

    # Threshold fit using pilot means vs high-z bias
    gp = pilot_df.groupby("highz_bias_frac")["anchor_gr_relief"].mean().reset_index()
    x = gp["highz_bias_frac"].to_numpy(dtype=float)
    y = gp["anchor_gr_relief"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    hb_for_10 = float((0.10 - intercept) / slope)
    hb_for_40 = float((0.40 - intercept) / slope)

    _plot_relief_vs_highz_bias(
        pilot_df=pilot_df,
        constrained_df=constrained_all,
        fit_intercept=float(intercept),
        fit_slope=float(slope),
        out_path=out_dir / "relief_vs_highz_bias.png",
    )

    summary = {
        "created_utc": _utc_now(),
        "inputs": {
            "constrained_csvs": [str(p) for p in constrained_paths],
            "pilot_csv": str(pilot_path),
            "highz_prior_sigma": float(args.highz_prior_sigma),
            "local_prior_sigma": float(args.local_prior_sigma),
            "mc_samples": int(n),
        },
        "posterior_weighted_no_mc": base_stats,
        "posterior_with_mc_calibration": draw_stats,
        "mc_repeatability_estimate": mc_est,
        "uncertainty_budget": budget,
        "highz_bias_thresholds_linear_fit": {
            "fit_intercept": float(intercept),
            "fit_slope": float(slope),
            "highz_bias_for_10pct_relief": hb_for_10,
            "highz_bias_for_40pct_relief": hb_for_40,
            "note": "Linear fit on pilot mean relief vs high-z bias; thresholds are extrapolated if outside fit domain.",
        },
    }
    _write_json_atomic(out_dir / "final_relief_posterior_summary.json", summary)

    md = [
        "# Final Relief Posterior (Constrained + Repeatability Calibrated)",
        "",
        f"- Generated: `{summary['created_utc']}`",
        f"- High-z prior sigma: `{float(args.highz_prior_sigma):.4f}`",
        f"- Local-bias prior sigma: `{float(args.local_prior_sigma):.3f}` km/s/Mpc",
        "",
        "## Posterior (anchor-based relief fraction)",
        "",
        "- Weighted (no MC correction): "
        f"`mean={base_stats['mean']:.4f}`, `p16/p50/p84={base_stats['p16']:.4f}/{base_stats['p50']:.4f}/{base_stats['p84']:.4f}`",
        "- MC-calibrated (using repeatability sigma): "
        f"`mean={draw_stats['mean']:.4f}`, `p16/p50/p84={draw_stats['p16']:.4f}/{draw_stats['p50']:.4f}/{draw_stats['p84']:.4f}`",
        "",
        "## Uncertainty budget",
        "",
        f"- Seed spread (sd of per-seed means): `{budget['seed_mean_sd']:.4f}`",
        f"- High-z bias leverage (range of bias-level means): `{budget['highz_bias_mean_range']:.4f}`",
        f"- Local-bias leverage (range of bias-level means): `{budget['local_bias_mean_range']:.4f}`",
        f"- Finite-MC sigma from repeatability: `{budget['mc_sigma_from_repeatability']:.4f}`",
        "",
        "## Falsification thresholds (linearized, pilot fit)",
        "",
        f"- High-z bias needed for 10% relief: `{hb_for_10:+.4f}` (~`{100.0*hb_for_10:+.2f}%`)",
        f"- High-z bias needed for 40% relief: `{hb_for_40:+.4f}` (~`{100.0*hb_for_40:+.2f}%`)",
        "",
        "## Artifacts",
        "",
        "- `final_relief_posterior_summary.json`",
        "- `relief_vs_highz_bias.png`",
    ]
    (out_dir / "final_relief_posterior_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'final_relief_posterior_summary.json'}")
    print(f"[done] wrote {out_dir / 'final_relief_posterior_summary.md'}")
    print(f"[done] wrote {out_dir / 'relief_vs_highz_bias.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
