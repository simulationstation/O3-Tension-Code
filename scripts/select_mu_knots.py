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

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.inversion import infer_logmu_forward
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike
from entropy_horizon_recon.model_selection import loglik_blocks_for_forward_posterior, waic_from_loglik
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report


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
    parser = argparse.ArgumentParser(description="Select μ(A) knot count using block-WAIC on a BH-truth synthetic dataset.")
    parser.add_argument("--out", type=Path, default=Path("outputs/mu_knots_selection"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max-cap", type=float, default=0.8)
    parser.add_argument("--sn-bin-width", type=float, default=0.05)
    parser.add_argument("--sn-min-per-bin", type=int, default=20)
    parser.add_argument("--n-grid", type=int, default=140)
    parser.add_argument("--candidates", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--delta-waic", type=float, default=10.0, help="Minimum WAIC improvement to justify a larger K.")
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--burn", type=int, default=140)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--max-rss-mb", type=float, default=1536.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()
    rng = np.random.default_rng(int(args.seed))

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = report_paths.out_dir / "debug_invalid_logprob.txt"

    # --- Load real data (cached) to define realistic templates ---
    sn = load_pantheon_plus(paths=paths, cov_kind="stat+sys", subset="cosmology", z_column="zHD")
    cc = load_chronometers(paths=paths, variant="BC03_all")
    bao12 = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    bao16 = load_bao(paths=paths, dataset="sdss_dr16_lrg_bao_dmdh")
    desi24 = load_bao(paths=paths, dataset="desi_2024_bao_all")

    z_max = _dense_domain_zmax(
        sn.z,
        z_min=float(args.z_min),
        z_max_cap=float(args.z_max_cap),
        bin_width=float(args.sn_bin_width),
        min_per_bin=int(args.sn_min_per_bin),
    )
    z_grid = np.linspace(0.0, float(z_max), int(args.n_grid))

    sn_like = SNLogLike.from_pantheon(sn, z_min=float(args.z_min), z_max=float(z_max))
    z_edges = np.arange(float(args.z_min), float(z_max) + float(args.sn_bin_width), float(args.sn_bin_width))
    sn_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=int(args.sn_min_per_bin))

    cc_like = ChronometerLogLike.from_data(cc, z_min=float(args.z_min), z_max=float(z_max))
    bao_likes = []
    for dataset, bao in [
        ("sdss_dr12_consensus_bao", bao12),
        ("sdss_dr16_lrg_bao_dmdh", bao16),
        ("desi_2024_bao_all", desi24),
    ]:
        try:
            bao_likes.append(BaoLogLike.from_data(bao, dataset=dataset, constants=constants, z_min=float(args.z_min), z_max=float(z_max)))
        except ValueError as e:
            print(f"Skipping BAO dataset {dataset}: {e}")

    # --- BH truth ---
    H0_true = 70.0
    omega_m0_true = 0.3
    H_true = H0_true * np.sqrt(omega_m0_true * (1.0 + z_grid) ** 3 + (1.0 - omega_m0_true))
    bg = build_background_from_H_grid(z_grid, H_true, constants=constants)

    # Synthetic observations at template locations (no extra jitter beyond quoted covariances).
    Dl = bg.Dl(sn_bin.z)
    m_true = 5.0 * np.log10(Dl) + 0.0
    Lsn = np.linalg.cholesky(sn_bin.cov)
    sn_m = m_true + Lsn @ rng.normal(size=m_true.shape)

    H_cc_true = bg.H(cc_like.z)
    cc_H = H_cc_true + cc_like.sigma_H * rng.normal(size=H_cc_true.shape)

    bao_obs = []
    for bl in bao_likes:
        y_true = bl.predict(bg, r_d_Mpc=147.0)
        c, lower = bl.cov_cho
        Lb = c if lower else c.T
        y_obs = y_true + Lb @ rng.normal(size=y_true.shape)
        bao_obs.append(
            BaoLogLike.from_arrays(
                dataset=bl.dataset,
                z=bl.z,
                y=y_obs,
                obs=bl.obs,
                cov=bl.cov,
                constants=constants,
            )
        )

    # Evaluate candidate knot counts by block-WAIC.
    results = []
    for K in [int(k) for k in args.candidates]:
        # x domain for μ(A): x = log(A/A0) ~ 2 log(H0/H(z)).
        H_zmax = float(H_true[-1])
        x_min = float(2.0 * np.log(H0_true / H_zmax))
        x_knots = np.linspace(1.25 * x_min, 0.0, K)
        x_grid = np.linspace(x_min, 0.0, 120)

        post = infer_logmu_forward(
            z_grid=z_grid,
            x_knots=x_knots,
            x_grid=x_grid,
            sn_z=sn_bin.z,
            sn_m=sn_m,
            sn_cov=sn_bin.cov,
            cc_z=cc_like.z,
            cc_H=cc_H,
            cc_sigma_H=cc_like.sigma_H,
            bao_likes=bao_obs,
            constants=constants,
            n_walkers=max(2 * (K + 6), int(args.walkers)),
            n_steps=int(args.steps),
            n_burn=int(min(args.burn, max(0, args.steps - 1))),
            seed=int(args.seed) + 999 + K,
            n_processes=1,
            n_draws=int(args.draws),
            progress=False,
            max_rss_mb=float(args.max_rss_mb) if args.max_rss_mb is not None else None,
            omega_m0_prior=(0.2, 0.4),
            sigma_cc_jit_scale=0.5,
            sigma_sn_jit_scale=0.02,
            sigma_d2_scale=0.185,
            debug_log_path=debug_log_path,
        )

        ll, names = loglik_blocks_for_forward_posterior(
            post,
            sn_z=sn_bin.z,
            sn_m=sn_m,
            sn_cov=sn_bin.cov,
            cc_z=cc_like.z,
            cc_H=cc_H,
            cc_sigma_H=cc_like.sigma_H,
            bao_likes=bao_obs,
            constants=constants,
        )
        waic = waic_from_loglik(ll)
        results.append(
            {
                "K": int(K),
                "waic": float(waic.waic),
                "lppd": float(waic.lppd),
                "p_waic": float(waic.p_waic),
                "n_blocks": int(waic.n_points),
                "block_names": names,
                "acceptance_fraction_mean": float(post.meta.get("acceptance_fraction_mean", np.nan)),
                "ess_min": post.meta.get("ess_min"),
            }
        )

    results.sort(key=lambda d: d["K"])
    chosen = results[0]["K"]
    for r in results[1:]:
        if float(r["waic"]) < float(results[0]["waic"]) - float(args.delta_waic):
            chosen = int(r["K"])

    out = {"chosen_K": int(chosen), "results": results, "z_max": float(z_max)}
    (report_paths.tables_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = []
    md.append("# μ-knot selection (block-WAIC)\n")
    md.append("## Setup\n")
    md.append(
        format_table(
            rows=[
                ["seed", int(args.seed)],
                ["z_max_used", f"{float(z_max):.3f}"],
                ["SN bins", int(sn_bin.z.size)],
                ["CC points", int(cc_like.z.size)],
                ["BAO points", int(sum(int(bl.z.size) for bl in bao_obs))],
                ["candidates", " ".join(str(int(k)) for k in args.candidates)],
                ["delta_waic", float(args.delta_waic)],
                ["chosen_K", int(chosen)],
            ],
            headers=["item", "value"],
        )
        + "\n"
    )
    rows = []
    for r in results:
        rows.append([r["K"], f"{r['waic']:.2f}", f"{r['p_waic']:.2f}", f"{r['lppd']:.2f}"])
    md.append("## Results\n")
    md.append(format_table(rows=rows, headers=["K", "WAIC", "p_WAIC", "lppd"]) + "\n")
    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    print(f"Wrote {report_paths.report_md}")
    print(f"SUMMARY chosen_K={int(chosen)} z_max={float(z_max):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
