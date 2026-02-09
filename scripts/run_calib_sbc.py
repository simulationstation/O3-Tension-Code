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

import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report
from entropy_horizon_recon.sbc import run_sbc_prior_truth


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


def _mem_available_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            txt = f.read().splitlines()
        for line in txt:
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    except Exception:
        return None
    return None


def _safe_procs(requested: int, *, max_rss_mb: float | None, cpu_cores: int | None) -> int:
    if requested <= 0:
        requested = 1
    if cpu_cores and cpu_cores > 0:
        requested = min(int(requested), int(cpu_cores))
    if max_rss_mb is None or max_rss_mb <= 0:
        return max(1, min(int(requested), 2))
    avail = _mem_available_mb()
    if avail is None:
        return max(1, min(int(requested), 2))
    # Keep generous headroom: do not allocate more than 50% of available RAM.
    per = float(max_rss_mb) * 1.3
    safe = int(max(1, np.floor(0.5 * float(avail) / per)))
    return max(1, min(int(requested), safe))


def _plot_rank_hist(ranks: np.ndarray, *, n_draws: int, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = np.asarray(ranks, dtype=int)
    bins = np.arange(n_draws + 2)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(r, bins=bins, color="C0", alpha=0.85)
    ax.set(title=title, xlabel="rank", ylabel="count")
    ax.set_xlim(0, n_draws + 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    ok = counts >= min_per_bin
    if not np.any(ok):
        return float(z_min + bin_width)
    last_good = int(np.where(ok)[0].max())
    return float(edges[last_good + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Prior-truth SBC for forward μ(A) inference.")
    parser.add_argument("--out", type=Path, default=Path("outputs/calib_sbc"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sbc-n", type=int, default=20)
    parser.add_argument("--sbc-procs", type=int, default=0)
    parser.add_argument("--cpu-cores", type=int, default=0)
    parser.add_argument("--max-rss-mb", type=float, default=1536.0)
    parser.add_argument("--fast", action="store_true", help="Use low-cost chain settings (debug only).")
    parser.add_argument("--sampler-kind", choices=["emcee", "ptemcee"], default="emcee")
    parser.add_argument("--pt-ntemps", type=int, default=8)
    parser.add_argument("--pt-tmax", type=float, default=50.0)
    parser.add_argument("--walkers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burn", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--truth-mu", choices=["prior", "bh"], default="prior", help="Truth μ(A) mode for SBC.")
    parser.add_argument("--rank-bins", type=int, default=20, help="Number of bins used for chi2 rank test.")
    parser.add_argument("--detect-alpha", type=float, default=0.05, help="Decision alpha for BH-null false-positive rate.")
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max-cap", type=float, default=0.8)
    parser.add_argument("--sn-bin-width", type=float, default=0.05)
    parser.add_argument("--sn-min-per-bin", type=int, default=20)
    parser.add_argument("--n-grid", type=int, default=120)
    parser.add_argument("--mu-knots", type=int, default=8)
    parser.add_argument("--mu-grid", type=int, default=120)
    args = parser.parse_args()

    cpu_cores = _resolve_cpu_cores(args.cpu_cores)
    _apply_cpu_affinity(cpu_cores)
    args.cpu_cores = cpu_cores

    # Chain settings (kept small by default to avoid resource blowups).
    if args.fast:
        n_walkers = 24
        n_steps = 240
        n_burn = 80
        n_draws = 120
    else:
        n_walkers = 32
        n_steps = 400
        n_burn = 140
        n_draws = 200

    # Explicit overrides.
    if args.walkers is not None and int(args.walkers) > 0:
        n_walkers = int(args.walkers)
    if args.steps is not None and int(args.steps) > 0:
        n_steps = int(args.steps)
    if args.burn is not None and int(args.burn) >= 0:
        n_burn = int(args.burn)
    if args.draws is not None and int(args.draws) > 0:
        n_draws = int(args.draws)
    if not (0 <= int(n_burn) < int(n_steps)):
        raise ValueError("--burn must satisfy 0 <= burn < steps.")

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Load real data (cached) to define realistic templates ---
    sn = load_pantheon_plus(paths=paths, cov_kind="stat+sys", subset="cosmology", z_column="zHD")
    cc = load_chronometers(paths=paths, variant="BC03_all")
    bao12 = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    bao16 = load_bao(paths=paths, dataset="sdss_dr16_lrg_bao_dmdh")
    desi24 = load_bao(paths=paths, dataset="desi_2024_bao_all")

    z_max = _dense_domain_zmax(
        sn.z,
        z_min=args.z_min,
        z_max_cap=args.z_max_cap,
        bin_width=args.sn_bin_width,
        min_per_bin=args.sn_min_per_bin,
    )
    z_grid = np.linspace(0.0, float(z_max), int(args.n_grid))

    sn_like = SNLogLike.from_pantheon(sn, z_min=float(args.z_min), z_max=float(z_max))
    # Compress SN for speed (critical for SBC).
    z_edges = np.arange(float(args.z_min), float(z_max) + float(args.sn_bin_width), float(args.sn_bin_width))
    sn_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=int(args.sn_min_per_bin))

    cc_like = ChronometerLogLike.from_data(cc, z_min=float(args.z_min), z_max=float(z_max))
    bao_templates: list[BaoLogLike] = []
    for dataset, bao in [
        ("sdss_dr12_consensus_bao", bao12),
        ("sdss_dr16_lrg_bao_dmdh", bao16),
        ("desi_2024_bao_all", desi24),
    ]:
        try:
            bao_templates.append(BaoLogLike.from_data(bao, dataset=dataset, constants=constants, z_min=float(args.z_min), z_max=float(z_max)))
        except ValueError as e:
            print(f"Skipping BAO dataset {dataset}: {e}")

    # x domain for μ(A): x = log(A/A0) ~ 2 log(H0/H(z)).
    H0_guess = 70.0
    omega_m0_guess = 0.3
    H_zmax_guess = H0_guess * np.sqrt(omega_m0_guess * (1.0 + float(z_max)) ** 3 + (1.0 - omega_m0_guess))
    x_min = float(2.0 * np.log(H0_guess / H_zmax_guess))
    x_knots = np.linspace(1.25 * x_min, 0.0, int(args.mu_knots))
    x_grid = np.linspace(x_min, 0.0, int(args.mu_grid))

    if args.sbc_procs and args.sbc_procs > 0:
        req = int(args.sbc_procs)
    else:
        req = min(int(args.sbc_n), int(args.cpu_cores))
    procs = _safe_procs(req, max_rss_mb=float(args.max_rss_mb), cpu_cores=int(args.cpu_cores))
    if procs != req:
        print(f"Adjusted sbc-procs {req} -> {procs} for safety (max_rss_mb={args.max_rss_mb}).")

    res = run_sbc_prior_truth(
        seed=int(args.seed),
        N=int(args.sbc_n),
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_z=sn_bin.z,
        sn_cov=sn_bin.cov,
        cc_z=cc_like.z,
        cc_sigma_H=cc_like.sigma_H,
        bao_templates=bao_templates,
        constants=constants,
        sampler_kind=str(args.sampler_kind),
        pt_ntemps=int(args.pt_ntemps),
        pt_tmax=float(args.pt_tmax) if str(args.sampler_kind) == "ptemcee" else None,
        n_walkers=int(max(2 * (int(args.mu_knots) + 6), n_walkers)),
        n_steps=int(n_steps),
        n_burn=int(min(n_burn, max(0, n_steps - 1))),
        n_draws=int(n_draws),
        n_processes=int(procs),
        max_rss_mb=float(args.max_rss_mb) if args.max_rss_mb is not None else None,
        omega_m0_prior=(0.2, 0.4),
        # Use tighter jitter priors here to keep SBC focused on the forward model;
        # these must match the inference priors used in run_sbc_prior_truth.
        sigma_cc_jit_scale=0.5,
        sigma_sn_jit_scale=0.02,
        sigma_d2_scale=0.185,
        logmu_knot_scale=1.0,
        mu_truth_mode=str(args.truth_mu),
        rank_bins=int(args.rank_bins),
        debug_log_path=report_paths.out_dir / "debug_invalid_logprob.txt",
        progress=True,
        progress_path=report_paths.tables_dir / "progress.jsonl",
    )

    # Write tables.
    (report_paths.tables_dir / "ranks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    reps = res.get("replicates", []) or []
    alpha = float(args.detect_alpha)
    fpr = None
    if str(args.truth_mu) == "bh" and reps:
        p_gt0 = [float(r.get("scar_s_post_p_gt0", np.nan)) for r in reps]
        p_gt0 = [x for x in p_gt0 if np.isfinite(x)]
        fpr = float(np.mean(np.array(p_gt0) < alpha)) if p_gt0 else None
    summary = {
        "N": res["N"],
        "n_draws": res["n_draws"],
        "z_max": float(z_max),
        "sn_bins": int(sn_bin.z.size),
        "cc_points": int(cc_like.z.size),
        "bao_points": int(sum(int(bl.z.size) for bl in bao_templates)),
        "pvalues": res["pvalues"],
        "coverage": res.get("coverage", {}),
        "bh_null": {"alpha": alpha, "fpr_s_neg": fpr} if str(args.truth_mu) == "bh" else None,
        "meta": res["meta"],
        "logprob": res.get("meta", {}).get("logprob", {}),
    }
    (report_paths.tables_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots.
    for key, r in res["ranks"].items():
        _plot_rank_hist(np.asarray(r, dtype=int), n_draws=int(res["n_draws"]), title=f"SBC rank histogram: {key}", path=report_paths.figures_dir / f"rank_{key}.png")

    # Report.
    rows = []
    for k, pv in res["pvalues"].items():
        rows.append([k, f"{pv['chi2_p']:.3g}", f"{pv['ks_p']:.3g}"])
    rows.sort(key=lambda r: r[0])

    md: list[str] = []
    md.append("# Prior-truth SBC (forward μ(A) inference)\n")
    md.append("## Setup\n")
    md.append(
        format_table(
            rows=[
                ["seed", int(args.seed)],
                ["N", int(args.sbc_n)],
                ["z_max_used", f"{float(z_max):.3f}"],
                ["SN bins", int(sn_bin.z.size)],
                ["CC points", int(cc_like.z.size)],
                ["BAO points", int(sum(int(bl.z.size) for bl in bao_templates))],
                ["sampler", str(args.sampler_kind)],
                ["pt_ntemps", int(args.pt_ntemps) if str(args.sampler_kind) == "ptemcee" else "NA"],
                ["pt_Tmax", float(args.pt_tmax) if str(args.sampler_kind) == "ptemcee" else "NA"],
                ["walkers", int(max(2 * (int(args.mu_knots) + 6), n_walkers))],
                ["steps", int(n_steps)],
                ["burn", int(min(n_burn, max(0, n_steps - 1)))],
                ["posterior draws per replicate", int(res["n_draws"])],
                ["replicate procs", int(procs)],
                ["max_rss_mb (per process)", float(args.max_rss_mb)],
            ],
            headers=["item", "value"],
        )
        + "\n"
    )
    md.append("## Rank uniformity p-values\n")
    md.append(format_table(rows=rows, headers=["summary", "chi2_p", "ks_p"]) + "\n")
    if str(args.truth_mu) == "bh":
        md.append("## BH-null false-positive rate (scar_s)\n")
        md.append(
            format_table(
                rows=[
                    ["alpha", f"{alpha:.3g}"],
                    ["FPR(P(s>0)<alpha)", "NA" if fpr is None else f"{float(fpr):.3g}"],
                ],
                headers=["item", "value"],
            )
            + "\n"
        )
    md.append("## Rank histograms\n")
    for k in sorted(res["ranks"].keys()):
        md.append(f"### {k}\n")
        md.append(f"![rank_{k}](figures/rank_{k}.png)\n")

    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    print(f"Wrote {report_paths.report_md}")

    # One-line status summary for logs.
    bad = []
    for k, pv in res["pvalues"].items():
        if not (0.0 <= float(pv["chi2_p"]) <= 1.0) or not np.isfinite(float(pv["chi2_p"])):
            bad.append(k)
    status = "PASS" if not bad else f"FAIL ({len(bad)} bad p-values)"
    print(f"SUMMARY SBC {status} N={int(args.sbc_n)} z_max={float(z_max):.3f} procs={int(procs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
