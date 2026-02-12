from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.void_prism import eg_gr_baseline_concat_from_background, predict_EG_void_concat_from_mu

# Avoid OpenMP oversubscription / weird thread behavior in BLAS on large machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


def _logpdf_mvnormal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if x.shape != mu.shape or cov.shape != (x.size, x.size):
        raise ValueError("Shape mismatch for multivariate normal.")
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-12 * np.trace(cov) / max(1, x.size)
    cov_j = cov + np.eye(x.size) * jitter
    L = np.linalg.cholesky(cov_j)
    r = x - mu
    y = np.linalg.solve(L, r)
    maha = float(np.dot(y, y))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (x.size * np.log(2.0 * np.pi) + logdet + maha)


def _bestfit_amplitude(pred: np.ndarray, obs: np.ndarray, cov: np.ndarray) -> float:
    """GLS best-fit scalar amplitude A in obs ~ A * pred (no prior)."""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-12 * np.trace(cov) / max(1, obs.size)
    cov_j = cov + np.eye(obs.size) * jitter
    L = np.linalg.cholesky(cov_j)

    def _invC(v: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, y)

    iCy = _invC(obs)
    iCp = _invC(pred)
    denom = float(np.dot(pred, iCp))
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        raise ValueError("Degenerate amplitude fit (pred^T C^-1 pred ~ 0).")
    return float(np.dot(pred, iCy) / denom)


def _lpd_from_draws(pred: np.ndarray, obs: np.ndarray, cov: np.ndarray, *, fit_amplitude: bool) -> tuple[float, np.ndarray | None]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if pred.ndim != 2:
        raise ValueError("pred must be 2D (n_draws, n_dim).")
    if not fit_amplitude:
        logp = np.array([_logpdf_mvnormal(obs, pred[j], cov) for j in range(pred.shape[0])], dtype=float)
        return float(_logmeanexp(logp)), None

    A = np.empty(pred.shape[0], dtype=float)
    logp = np.empty(pred.shape[0], dtype=float)
    for j in range(pred.shape[0]):
        Aj = _bestfit_amplitude(pred[j], obs, cov)
        A[j] = Aj
        logp[j] = _logpdf_mvnormal(obs, Aj * pred[j], cov)
    return float(_logmeanexp(logp)), A


def _detect_mapping_variant(run_dir: Path) -> str:
    summary = run_dir / "tables" / "summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text())
            mv = (d.get("settings") or {}).get("mapping_variant")
            if mv:
                return str(mv)
        except Exception:
            pass
    name = run_dir.name
    if "M2" in name:
        return "M2"
    if "M1" in name:
        return "M1"
    return "M0"


def _load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    d = json.loads(path.read_text())
    meta = d.get("meta") or {}
    blocks = d.get("blocks")
    y_obs = d.get("y_obs")
    cov = d.get("cov")
    if not isinstance(blocks, list) or y_obs is None:
        raise ValueError("suite_joint.json missing required keys: blocks, y_obs.")
    y = np.asarray(y_obs, dtype=float)
    if cov is None:
        raise ValueError("suite_joint.json has cov=null; rerun measurement with --jackknife-nside (or implement an alternative covariance).")
    C = np.asarray(cov, dtype=float)
    if C.shape != (y.size, y.size):
        raise ValueError("Covariance shape mismatch.")
    return meta, blocks, y, C


@dataclass(frozen=True)
class JointRow:
    run: str
    mapping_variant: str
    embedding: str
    convention: str
    eta0: float
    eta1: float
    env_proxy: float
    env_alpha: float
    muP_highz: float
    fit_amplitude: bool
    report_both_amplitudes: bool
    max_draws: int | None
    lpd: float
    lpd_gr: float
    delta_lpd_vs_gr: float
    lpd_no_amp: float
    lpd_gr_no_amp: float
    delta_lpd_vs_gr_no_amp: float
    lpd_fit_amp: float
    lpd_gr_fit_amp: float
    delta_lpd_vs_gr_fit_amp: float
    amp_mean: float | None
    amp_std: float | None
    amp_q16: float | None
    amp_q50: float | None
    amp_q84: float | None


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint scoring of a void-prism suite (multiple z/Rv bins) vs mu(A) posteriors.")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--suite-json", required=True, help="suite_joint.json written by measure_void_prism_eg_suite_jackknife.py")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/void_prism_eg_joint_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="mu->coupling convention (A: mu(z)/mu0).")
    ap.add_argument(
        "--embedding",
        action="append",
        choices=["minimal", "slip_allowed", "screening_allowed"],
        default=None,
        help=(
            "Embedding(s) to score (repeatable). "
            "Must be provided explicitly unless --allow-implicit-minimal is set."
        ),
    )
    ap.add_argument(
        "--allow-implicit-minimal",
        action="store_true",
        help=(
            "Legacy behavior: if --embedding is omitted, score only 'minimal'. "
            "By default this is disallowed to prevent silently GR-like runs."
        ),
    )
    ap.add_argument("--max-draws", type=int, default=5000, help="Subsample posterior draws for speed (default 5000).")
    ap.add_argument(
        "--fit-amplitude",
        action="store_true",
        help=(
            "Choose amplitude-fitted score as primary output field (`lpd`/`delta_lpd_vs_gr`). "
            "Both fixed-amplitude and fitted-amplitude scores are always recorded."
        ),
    )
    ap.add_argument(
        "--no-report-both-amplitudes",
        action="store_true",
        help=(
            "If set, compute/store only the primary mode selected by --fit-amplitude. "
            "Default behavior computes and stores both amplitude modes."
        ),
    )
    ap.add_argument("--eta0", type=float, default=1.0, help="Slip model eta0 (used for slip_allowed).")
    ap.add_argument("--eta1", type=float, default=0.0, help="Slip model eta1 (used for slip_allowed).")
    ap.add_argument("--env-proxy", type=float, default=0.0, help="Screening env proxy value (used for screening_allowed).")
    ap.add_argument("--env-alpha", type=float, default=0.0, help="Screening alpha (used for screening_allowed).")
    ap.add_argument("--muP-highz", type=float, default=1.0, help="High-z muP value for growth extension (default 1).")
    ap.add_argument(
        "--progress-every-block",
        type=int,
        default=1,
        help="Print progress every N blocks while building the concatenated prediction (default 1).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_prism_eg_joint_{_utc_stamp()}"
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    suite_path = Path(args.suite_json)
    meta, blocks, y_obs, cov = _load_suite(suite_path)
    block_tuples = [(float(b["z_eff"]), np.asarray(b["ell"], dtype=int)) for b in blocks]

    if args.embedding:
        embeddings = list(args.embedding)
    elif args.allow_implicit_minimal:
        embeddings = ["minimal"]
        print(
            "[void_prism_joint] WARNING: using implicit legacy embedding=['minimal']; "
            "pass --embedding explicitly to avoid GR-like defaults.",
            flush=True,
        )
    else:
        raise ValueError(
            "No --embedding provided. Pass one or more --embedding values "
            "(minimal, slip_allowed, screening_allowed), or set --allow-implicit-minimal for legacy behavior."
        )

    report_both_amplitudes = not bool(args.no_report_both_amplitudes)

    # Warn if selected embeddings collapse to minimal behavior under current parameters.
    if "slip_allowed" in embeddings and float(args.eta0) == 1.0 and float(args.eta1) == 0.0:
        print(
            "[void_prism_joint] WARNING: slip_allowed with eta0=1 and eta1=0 is effectively identical to minimal.",
            flush=True,
        )
    if "screening_allowed" in embeddings and float(args.env_alpha) == 0.0:
        print(
            "[void_prism_joint] WARNING: screening_allowed with env_alpha=0 is effectively identical to minimal.",
            flush=True,
        )

    print(
        f"[void_prism_joint] suite={suite_path}  blocks={len(blocks)}  y_dim={y_obs.size}  embeddings={embeddings}  "
        f"report_both_amplitudes={report_both_amplitudes}",
        flush=True,
    )
    for i, b in enumerate(blocks):
        if i < 10:
            print(f"[void_prism_joint] block {i+1}/{len(blocks)} name={b.get('name')} z_eff={b.get('z_eff')}", flush=True)
    if len(blocks) > 10:
        print(f"[void_prism_joint] (showing first 10 blocks; total={len(blocks)})", flush=True)

    rows: list[dict[str, Any]] = []
    for rd in args.run_dir:
        run_path = Path(rd)
        print(f"[void_prism_joint] loading posterior: {run_path}", flush=True)
        post = load_mu_forward_posterior(run_path)
        run_label = run_path.name
        mapping_variant = _detect_mapping_variant(run_path)

        print(f"[void_prism_joint] {run_label} predicting GR baseline (single growth solve per draw)...", flush=True)
        eg_gr = eg_gr_baseline_concat_from_background(post, blocks=block_tuples, max_draws=int(args.max_draws) if args.max_draws else None)
        if report_both_amplitudes:
            lpd_gr_no_amp, _ = _lpd_from_draws(eg_gr, y_obs, cov, fit_amplitude=False)
            lpd_gr_fit_amp, _ = _lpd_from_draws(eg_gr, y_obs, cov, fit_amplitude=True)
        elif bool(args.fit_amplitude):
            lpd_gr_fit_amp, _ = _lpd_from_draws(eg_gr, y_obs, cov, fit_amplitude=True)
            lpd_gr_no_amp = float("nan")
        else:
            lpd_gr_no_amp, _ = _lpd_from_draws(eg_gr, y_obs, cov, fit_amplitude=False)
            lpd_gr_fit_amp = float("nan")

        lpd_gr = lpd_gr_fit_amp if bool(args.fit_amplitude) else lpd_gr_no_amp
        if report_both_amplitudes:
            print(
                f"[void_prism_joint] {run_label} GR lpd_primary={lpd_gr:.3g}  "
                f"lpd_no_amp={lpd_gr_no_amp:.3g}  lpd_fit_amp={lpd_gr_fit_amp:.3g}",
                flush=True,
            )
        else:
            print(f"[void_prism_joint] {run_label} GR lpd_primary={lpd_gr:.3g}", flush=True)

        for emb in embeddings:
            print(f"[void_prism_joint] {run_label} predicting emb={emb} (single growth solve per draw)...", flush=True)
            eg = predict_EG_void_concat_from_mu(
                post,
                blocks=block_tuples,
                convention=args.convention,  # type: ignore[arg-type]
                embedding=str(emb),  # type: ignore[arg-type]
                eta0=float(args.eta0),
                eta1=float(args.eta1),
                env_proxy=float(args.env_proxy),
                env_alpha=float(args.env_alpha),
                muP_highz=float(args.muP_highz),
                max_draws=int(args.max_draws) if args.max_draws else None,
            )
            if report_both_amplitudes:
                lpd_no_amp, _ = _lpd_from_draws(eg, y_obs, cov, fit_amplitude=False)
                lpd_fit_amp, A_fit = _lpd_from_draws(eg, y_obs, cov, fit_amplitude=True)
            elif bool(args.fit_amplitude):
                lpd_fit_amp, A_fit = _lpd_from_draws(eg, y_obs, cov, fit_amplitude=True)
                lpd_no_amp = float("nan")
            else:
                lpd_no_amp, _ = _lpd_from_draws(eg, y_obs, cov, fit_amplitude=False)
                lpd_fit_amp = float("nan")
                A_fit = None

            # Respect legacy "primary score" behavior while storing both modes.
            if bool(args.fit_amplitude):
                lpd = lpd_fit_amp
                A_primary = A_fit
            else:
                lpd = lpd_no_amp
                A_primary = None

            amp_stats = {"mean": None, "std": None, "q16": None, "q50": None, "q84": None}
            if A_primary is not None and A_primary.size > 0:
                amp_stats = {
                    "mean": float(np.mean(A_primary)),
                    "std": float(np.std(A_primary, ddof=1)) if A_primary.size > 1 else float("nan"),
                    "q16": float(np.percentile(A_primary, 16)),
                    "q50": float(np.percentile(A_primary, 50)),
                    "q84": float(np.percentile(A_primary, 84)),
                }

            if report_both_amplitudes:
                delta_no_amp = float(lpd_no_amp - lpd_gr_no_amp)
                delta_fit_amp = float(lpd_fit_amp - lpd_gr_fit_amp)
            else:
                if bool(args.fit_amplitude):
                    delta_no_amp = float("nan")
                    delta_fit_amp = float(lpd_fit_amp - lpd_gr_fit_amp)
                    lpd_no_amp = float("nan")
                    lpd_gr_no_amp = float("nan")
                else:
                    delta_no_amp = float(lpd_no_amp - lpd_gr_no_amp)
                    delta_fit_amp = float("nan")
                    lpd_fit_amp = float("nan")
                    lpd_gr_fit_amp = float("nan")

            row = JointRow(
                run=run_label,
                mapping_variant=str(mapping_variant),
                embedding=str(emb),
                convention=str(args.convention),
                eta0=float(args.eta0),
                eta1=float(args.eta1),
                env_proxy=float(args.env_proxy),
                env_alpha=float(args.env_alpha),
                muP_highz=float(args.muP_highz),
                fit_amplitude=bool(args.fit_amplitude),
                report_both_amplitudes=bool(report_both_amplitudes),
                max_draws=int(args.max_draws) if args.max_draws else None,
                lpd=float(lpd),
                lpd_gr=float(lpd_gr),
                delta_lpd_vs_gr=float(lpd - lpd_gr),
                lpd_no_amp=float(lpd_no_amp),
                lpd_gr_no_amp=float(lpd_gr_no_amp),
                delta_lpd_vs_gr_no_amp=float(delta_no_amp),
                lpd_fit_amp=float(lpd_fit_amp),
                lpd_gr_fit_amp=float(lpd_gr_fit_amp),
                delta_lpd_vs_gr_fit_amp=float(delta_fit_amp),
                amp_mean=amp_stats["mean"],
                amp_std=amp_stats["std"],
                amp_q16=amp_stats["q16"],
                amp_q50=amp_stats["q50"],
                amp_q84=amp_stats["q84"],
            )
            rows.append({**asdict(row), "status": "ok", "suite_meta": meta})
            _atomic_write_text(tab_dir / "results_partial.json", json.dumps(rows, indent=2))
            print(
                f"[void_prism_joint] {run_label} emb={emb} "
                f"ΔLPD_primary={(lpd-lpd_gr):+.3g} "
                f"ΔLPD_no_amp={delta_no_amp:+.3g} "
                f"ΔLPD_fit_amp={delta_fit_amp:+.3g} "
                f"amp_mean={amp_stats['mean']}",
                flush=True,
            )

    _atomic_write_text(tab_dir / "results.json", json.dumps(rows, indent=2))
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
