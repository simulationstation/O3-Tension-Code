#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, marginalize_f_miss_global  # noqa: E402
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty  # noqa: E402
from entropy_horizon_recon.report import format_table  # noqa: E402


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _try_build_paper(*, out_papers_dir: Path) -> dict[str, Any]:
    import shutil as _shutil

    pdflatex = _shutil.which("pdflatex")
    if not pdflatex:
        return {"enabled": False, "reason": "pdflatex_not_found"}

    paper_dir = REPO_ROOT / "CQG_PAPER"
    if not paper_dir.exists():
        return {"enabled": False, "reason": "CQG_PAPER_missing"}

    built: list[str] = []
    try:
        for tex in ("dark_siren_cqg.tex", "dark_siren_cqg_iopjournal.tex"):
            # Two passes for references.
            _run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex], cwd=paper_dir)
            _run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex], cwd=paper_dir)
            pdf = tex.replace(".tex", ".pdf")
            src_pdf = paper_dir / pdf
            if src_pdf.exists():
                out_papers_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_pdf, out_papers_dir / pdf)
                built.append(pdf)
        return {"enabled": True, "built": built}
    except subprocess.CalledProcessError as e:
        return {"enabled": False, "reason": "pdflatex_failed", "returncode": int(e.returncode)}


def _spectral_only_baseline_from_cached_terms() -> dict[str, Any]:
    try:
        import numpy as np
    except Exception:
        return {"ok": False, "reason": "numpy_not_installed"}

    base = _read_json(REPO_ROOT / "artifacts/o3/summary_M0_start101.json")
    mix = (base.get("mixture") or {}).get("f_miss_meta") or {}
    prior_meta = mix.get("prior") or {}
    grid_meta = mix.get("grid") or {}
    prior = BetaPrior(mean=float(prior_meta["mean"]), kappa=float(prior_meta["kappa"]))
    n_f = int(grid_meta.get("n", 401))
    eps = float(grid_meta.get("eps", 1e-6))

    alpha_npz = REPO_ROOT / "artifacts/o3/selection_alpha_M0_start101.npz"
    terms_npz = REPO_ROOT / "artifacts/o3/smoking_gun_inputs/spectral_only_cached_terms.npz"
    if not alpha_npz.exists():
        return {"ok": False, "reason": "missing_selection_alpha_npz", "path": str(alpha_npz)}
    if not terms_npz.exists():
        return {"ok": False, "reason": "missing_spectral_terms_npz", "path": str(terms_npz)}

    with np.load(alpha_npz, allow_pickle=False) as d:
        log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
        log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)

    with np.load(terms_npz, allow_pickle=False) as d:
        events = [str(x) for x in d["events"].tolist()]
        logL_cat_mu = np.asarray(d["logL_cat_mu"], dtype=float)
        logL_cat_gr = np.asarray(d["logL_cat_gr"], dtype=float)
        logL_missing_mu = np.asarray(d["logL_missing_mu"], dtype=float)
        logL_missing_gr = np.asarray(d["logL_missing_gr"], dtype=float)

    res = marginalize_f_miss_global(
        logL_cat_mu_by_event=logL_cat_mu,
        logL_cat_gr_by_event=logL_cat_gr,
        logL_missing_mu_by_event=logL_missing_mu,
        logL_missing_gr_by_event=logL_missing_gr,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=n_f,
        eps=eps,
    )

    return {
        "ok": True,
        "n_events": int(len(events)),
        "n_draws": int(logL_cat_mu.shape[1]),
        "prior": {"mean": float(prior.mean), "kappa": float(prior.kappa), "alpha": float(prior.alpha), "beta": float(prior.beta)},
        "grid": {"n_f": int(n_f), "eps": float(eps)},
        "lpd_mu_total": float(res.lpd_mu_total),
        "lpd_gr_total": float(res.lpd_gr_total),
        "delta_lpd_total": float(res.lpd_mu_total - res.lpd_gr_total),
        "lpd_mu_total_data": float(res.lpd_mu_total_data),
        "lpd_gr_total_data": float(res.lpd_gr_total_data),
        "delta_lpd_total_data": float(res.lpd_mu_total_data - res.lpd_gr_total_data),
        "delta_lpd_total_sel": float((res.lpd_mu_total - res.lpd_mu_total_data) - (res.lpd_gr_total - res.lpd_gr_total_data)),
        "inputs": {"selection_alpha_npz": str(alpha_npz), "spectral_only_terms_npz": str(terms_npz)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed-replication driver: generate a one-folder report of CQG headline numbers from in-repo artifacts.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/seed_reproduce_<UTC timestamp>).")
    ap.add_argument("--build-paper", action="store_true", help="Attempt to rebuild CQG_PAPER PDFs with pdflatex and copy into output.")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve() if args.out else (REPO_ROOT / "outputs" / f"seed_reproduce_{_utc_now_compact()}")
    fig_dir = out_dir / "figures"
    papers_dir = out_dir / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Meta / provenance.
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": command_str(),
        "git_head_sha": git_head_sha(repo_root=REPO_ROOT),
        "git_dirty": git_is_dirty(repo_root=REPO_ROOT),
        "python": sys.version.replace("\n", " "),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS") if k in os.environ},
    }

    # Core artifacts (full pipeline headline numbers).
    baseline = _read_json(REPO_ROOT / "artifacts/o3/summary_M0_start101.json")
    inj_summary = _read_json(REPO_ROOT / "artifacts/o3/catalog_injection_summary.json")
    inj_deltas = _read_json(REPO_ROOT / "artifacts/o3/catalog_injection_deltas_n512.json")
    power_grid = _read_json(REPO_ROOT / "artifacts/o3/fixed_power_grid_summary.json")
    syst = _read_json(REPO_ROOT / "artifacts/o3/systematics_matrix_summary.json")
    jackknife = json.loads((REPO_ROOT / "artifacts/o3/jackknife_M0_start101.json").read_text(encoding="utf-8"))

    observed = float(baseline["delta_lpd_total"])
    # Injection p-value: P(ΔLPD >= observed | GR truth), estimated from 512 null replicates.
    rows = inj_deltas.get("rows", [])
    vals = [float(r["delta_lpd_total"]) for r in rows if "delta_lpd_total" in r]
    n = int(len(vals))
    n_ge = int(sum(1 for v in vals if v >= observed))
    p_ge_naive = float(n_ge / n) if n > 0 else float("nan")
    # Add-one smoothing to avoid p=0 in finite ensembles.
    p_ge = float((n_ge + 1) / (n + 1)) if n > 0 else float("nan")
    # One-sided Z (Gaussian equivalent).
    z_ge = float(NormalDist().inv_cdf(1.0 - p_ge)) if (n > 0 and 0.0 < p_ge < 1.0) else float("inf")

    # Spectral-only baseline (fast; uses cached terms + selection-alpha arrays).
    spectral_only = _spectral_only_baseline_from_cached_terms()

    # Build paper PDFs (optional).
    paper_build = _try_build_paper(out_papers_dir=papers_dir) if bool(args.build_paper) else {"enabled": False}

    # Copy headline figures into output.
    figures_to_copy = [
        REPO_ROOT / "artifacts/o3/fig_delta_lpd_total_hist.png",
        REPO_ROOT / "artifacts/o3/fig_delta_lpd_components_hist.png",
        REPO_ROOT / "artifacts/o3/fig_fixed_power_grid.png",
        REPO_ROOT / "artifacts/o3/fig_systematics_matrix.png",
        REPO_ROOT / "artifacts/o3/delta_lpd_by_event_M0_start101.png",
        REPO_ROOT / "CQG_PAPER/figures/fig_robustness_summary.png",
        REPO_ROOT / "CQG_PAPER/figures/fig_dlresid_o3.png",
    ]
    copied_figs: list[str] = []
    for p in figures_to_copy:
        if p.exists():
            shutil.copy2(p, fig_dir / p.name)
            copied_figs.append(p.name)

    # Checksums for key inputs (helps offline verification).
    key_inputs = [
        REPO_ROOT / "artifacts/o3/summary_M0_start101.json",
        REPO_ROOT / "artifacts/o3/catalog_injection_summary.json",
        REPO_ROOT / "artifacts/o3/catalog_injection_deltas_n512.json",
        REPO_ROOT / "artifacts/o3/fixed_power_grid_summary.json",
        REPO_ROOT / "artifacts/o3/systematics_matrix_summary.json",
        REPO_ROOT / "artifacts/o3/jackknife_M0_start101.json",
        REPO_ROOT / "artifacts/o3/selection_alpha_M0_start101.npz",
        REPO_ROOT / "artifacts/o3/smoking_gun_inputs/spectral_only_cached_terms.npz",
    ]
    input_hashes = []
    for p in key_inputs:
        if p.exists():
            input_hashes.append({"path": str(p.relative_to(REPO_ROOT)), "sha256": _sha256_file(p)})

    summary = {
        "meta": meta,
        "baseline_full": {
            "run": baseline.get("run"),
            "convention": baseline.get("convention"),
            "n_events": int(baseline.get("n_events", 0)),
            "n_draws": int(baseline.get("n_draws", 0)),
            "delta_lpd_total": float(baseline["delta_lpd_total"]),
            "delta_lpd_total_data": float(baseline["delta_lpd_total_data"]),
            "delta_lpd_total_sel": float(baseline["delta_lpd_total_sel"]),
            "bayes_factor_proxy_exp_delta": float(__import__("math").exp(float(baseline["delta_lpd_total"]))),
        },
        "baseline_spectral_only": spectral_only,
        "injection_null": {
            "n": n,
            "n_ge_observed": n_ge,
            "p_ge_observed_naive": p_ge_naive,
            "p_ge_observed_addone": p_ge,
            "z_ge_observed_addone": z_ge,
            "summary": inj_summary,
        },
        "fixed_power_grid": power_grid,
        "systematics_matrix": syst,
        "jackknife": {"n": int(len(jackknife)), "rows": jackknife},
        "paper_build": paper_build,
        "figures_copied": copied_figs,
        "input_hashes": input_hashes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Report.md (one-page-ish).
    lines: list[str] = []
    lines.append("# Seed Replication Report\n")
    lines.append(f"- Created (UTC): `{meta['created_utc']}`\n")
    if meta.get("git_head_sha"):
        lines.append(f"- Git HEAD: `{meta['git_head_sha']}` (dirty={meta['git_dirty']})\n")
    lines.append(f"- Command: `{meta['command']}`\n")
    lines.append("\n## Headline numbers\n")
    lines.append(
        f"- Full baseline: ΔLPD_total=`{baseline['delta_lpd_total']:+.3f}` (data=`{baseline['delta_lpd_total_data']:+.3f}`, sel=`{baseline['delta_lpd_total_sel']:+.3f}`; exp(Δ)≈`{summary['baseline_full']['bayes_factor_proxy_exp_delta']:.1f}`)\n"
    )
    if bool(spectral_only.get("ok", False)):
        lines.append(
            f"- Spectral-only baseline (cached): ΔLPD_total=`{spectral_only['delta_lpd_total']:+.3f}` (data=`{spectral_only['delta_lpd_total_data']:+.3f}`, sel≈`{spectral_only['delta_lpd_total_sel']:+.3f}`)\n"
        )
    else:
        lines.append(f"- Spectral-only baseline (cached): unavailable (`{spectral_only.get('reason')}`)\n")

    lines.append("\n## GR-truth injection calibration\n")
    lines.append(f"- N={n} GR-truth catalog injections; observed ΔLPD_total=`{observed:+.3f}`\n")
    lines.append(f"- Tail prob P(Δ>=obs): naive=`{p_ge_naive:.4g}`, add-one=`{p_ge:.4g}` → Z≈`{z_ge:.2f}` (one-sided)\n")
    lines.append("\n![](figures/fig_delta_lpd_total_hist.png)\n")

    # Fixed-power and systematics summaries.
    try:
        pg_rows = [
            [
                f"{r['scale']:.2g}",
                f"{r['mean_total']:+.3f}",
                f"{r['sd_total']:.3f}",
                f"{r['max_total']:+.3f}",
            ]
            for r in power_grid.get("rows", [])
        ]
        lines.append("\n## Fixed-power injection grid (summary)\n")
        lines.append(format_table(pg_rows, headers=["scale", "mean_total", "sd_total", "max_total"]) + "\n")
        lines.append("![](figures/fig_fixed_power_grid.png)\n")
    except Exception:
        pass

    try:
        sm_rows = [
            [
                str(r.get("name", "")),
                f"{float(r.get('mean_total', float('nan'))):+.3f}",
                f"{float(r.get('sd_total', float('nan'))):.3f}",
                f"{float(r.get('max_total', float('nan'))):+.3f}",
            ]
            for r in syst.get("rows", [])
        ]
        lines.append("\n## GR-systematics matrix (summary)\n")
        lines.append(format_table(sm_rows, headers=["variant", "mean_total", "sd_total", "max_total"]) + "\n")
        lines.append("![](figures/fig_systematics_matrix.png)\n")
    except Exception:
        pass

    # Jackknife top influences.
    try:
        jk = sorted(jackknife, key=lambda r: abs(float(r.get("influence", 0.0))), reverse=True)[:10]
        jk_rows = [[r["event"], f"{float(r['influence']):+.3f}", f"{float(r['delta_lpd_total_leave_one_out']):+.3f}"] for r in jk]
        lines.append("\n## Jackknife (top |influence|)\n")
        lines.append(format_table(jk_rows, headers=["event", "influence", "ΔLPD_total (LOO)"]) + "\n")
        lines.append("![](figures/delta_lpd_by_event_M0_start101.png)\n")
    except Exception:
        pass

    # Remaining paper figures copied for convenience.
    if (fig_dir / "fig_robustness_summary.png").exists():
        lines.append("\n## Robustness summary figure\n\n![](figures/fig_robustness_summary.png)\n")
    if (fig_dir / "fig_dlresid_o3.png").exists():
        lines.append("\n## Residual diagnostic figure\n\n![](figures/fig_dlresid_o3.png)\n")

    lines.append("\n## Input checksums\n")
    hash_rows = [[h["path"], h["sha256"][:16] + "…"] for h in input_hashes]
    lines.append(format_table(hash_rows, headers=["path", "sha256"]) + "\n")

    if bool(paper_build.get("enabled", False)):
        lines.append("\n## Paper build\n")
        lines.append(f"- Built PDFs: {', '.join(paper_build.get('built', []))}\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")

    print(f"[seed-reproduce] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

