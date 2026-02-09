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
    fit_amplitude: bool
    max_draws: int | None
    lpd: float
    lpd_gr: float
    delta_lpd_vs_gr: float
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
        help="Embedding(s) to score (repeatable). Default: minimal only.",
    )
    ap.add_argument("--max-draws", type=int, default=5000, help="Subsample posterior draws for speed (default 5000).")
    ap.add_argument("--fit-amplitude", action="store_true", help="Fit a per-draw scalar amplitude before scoring (shape-only).")
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

    embeddings = args.embedding if args.embedding else ["minimal"]

    print(f"[void_prism_joint] suite={suite_path}  blocks={len(blocks)}  y_dim={y_obs.size}  embeddings={embeddings}", flush=True)
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
        lpd_gr, A_gr = _lpd_from_draws(eg_gr, y_obs, cov, fit_amplitude=bool(args.fit_amplitude))
        print(f"[void_prism_joint] {run_label} GR lpd={lpd_gr:.3g}", flush=True)

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
            lpd, A = _lpd_from_draws(eg, y_obs, cov, fit_amplitude=bool(args.fit_amplitude))
            amp_stats = {"mean": None, "std": None, "q16": None, "q50": None, "q84": None}
            if A is not None and A.size > 0:
                amp_stats = {
                    "mean": float(np.mean(A)),
                    "std": float(np.std(A, ddof=1)) if A.size > 1 else float("nan"),
                    "q16": float(np.percentile(A, 16)),
                    "q50": float(np.percentile(A, 50)),
                    "q84": float(np.percentile(A, 84)),
                }

            row = JointRow(
                run=run_label,
                mapping_variant=str(mapping_variant),
                embedding=str(emb),
                convention=str(args.convention),
                fit_amplitude=bool(args.fit_amplitude),
                max_draws=int(args.max_draws) if args.max_draws else None,
                lpd=float(lpd),
                lpd_gr=float(lpd_gr),
                delta_lpd_vs_gr=float(lpd - lpd_gr),
                amp_mean=amp_stats["mean"],
                amp_std=amp_stats["std"],
                amp_q16=amp_stats["q16"],
                amp_q50=amp_stats["q50"],
                amp_q84=amp_stats["q84"],
            )
            rows.append({**asdict(row), "status": "ok", "suite_meta": meta})
            (tab_dir / "results_partial.json").write_text(json.dumps(rows, indent=2))
            print(
                f"[void_prism_joint] {run_label} emb={emb} lpd={lpd:.3g}  Δlpd_vs_GR={(lpd-lpd_gr):.3g}  amp_mean={amp_stats['mean']}",
                flush=True,
            )

    (tab_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
