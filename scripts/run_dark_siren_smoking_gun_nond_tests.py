#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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

# Ensure we import the *local* package from this repository.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

import healpy as hp  # noqa: E402

from astroquery.ned import Ned  # noqa: E402

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, MarginalizedFMissResult, marginalize_f_miss_global  # noqa: E402
from entropy_horizon_recon.dark_sirens import GalaxyIndex, gather_galaxies_from_pixels, load_gladeplus_index  # noqa: E402
from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram  # noqa: E402
from entropy_horizon_recon.dark_sirens_pe_fast import compute_dark_siren_logL_draws_from_pe_hist_fast  # noqa: E402
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior, predict_dL_em, predict_r_gw_em  # noqa: E402


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _set_thread_env(n: int) -> None:
    n = int(max(1, n))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(n)


def _pe_sky_marginal_pdf_1d(pe: PePixelDistanceHistogram) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(pe.dL_edges, dtype=float)
    widths = np.diff(edges)
    prob_pix = np.asarray(pe.prob_pix, dtype=float)
    pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
    p_sum = float(np.sum(prob_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError("Invalid prob_pix normalization while building sky-marginal pdf.")
    if int(pdf_bins.shape[0]) == int(prob_pix.size):
        pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    elif int(pdf_bins.shape[0]) == 1:
        pdf_1d = np.asarray(pdf_bins[0], dtype=float)
    else:
        raise ValueError("Incompatible pdf_bins shape for sky-marginal pdf.")
    pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
    norm = float(np.sum(pdf_1d * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Invalid sky-marginal pdf normalization.")
    pdf_1d = pdf_1d / norm
    return edges, pdf_1d


def _build_fiducial_dL_of_z(*, h0: float, omega_m0: float, z_max: float, n: int = 5001) -> tuple[np.ndarray, np.ndarray]:
    # Flat LCDM approximation used only for dominant-host ranking diagnostics.
    c = 299792.458
    z_grid = np.linspace(0.0, float(z_max), int(n))
    Ez = np.sqrt(float(omega_m0) * (1.0 + z_grid) ** 3 + (1.0 - float(omega_m0)))
    invE = 1.0 / np.clip(Ez, 1e-12, np.inf)
    dz = np.diff(z_grid)
    dc = np.empty_like(z_grid)
    dc[0] = 0.0
    dc[1:] = (c / float(h0)) * np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))
    dL = (1.0 + z_grid) * dc
    return z_grid, dL


def _dominant_indices_by_weight(
    *,
    weight: np.ndarray,
    thresholds: list[float],
    initial_topk: int,
    max_topk: int,
) -> tuple[np.ndarray, dict[float, int], dict[float, float]]:
    w = np.asarray(weight, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    tot = float(np.sum(w))
    if not (np.isfinite(tot) and tot > 0.0):
        return np.zeros((0,), dtype=np.int64), {t: 0 for t in thresholds}, {t: 0.0 for t in thresholds}

    thresholds = [float(t) for t in thresholds]
    tmax = float(max(thresholds))
    target = tmax * tot

    k = int(min(max(1, initial_topk), w.size))
    # Grow k until we cover the largest threshold.
    while True:
        idx_k = np.argpartition(w, -k)[-k:]
        s_k = float(np.sum(w[idx_k]))
        if s_k >= target or k >= w.size or k >= int(max_topk):
            break
        k = int(min(w.size, max_topk, max(k * 2, k + 1)))

    # Sort the retained set and compute cumulative coverage.
    idx_k = np.argpartition(w, -k)[-k:]
    order = np.argsort(w[idx_k])[::-1]
    idx_sorted = idx_k[order].astype(np.int64, copy=False)
    cw = np.cumsum(w[idx_sorted])
    frac = cw / tot

    n_for: dict[float, int] = {}
    frac_for: dict[float, float] = {}
    for t in thresholds:
        j = int(np.searchsorted(frac, t, side="left"))
        j = min(max(j, 0), int(frac.size - 1)) if frac.size else 0
        n_for[t] = int(j + 1) if frac.size else 0
        frac_for[t] = float(frac[j]) if frac.size else 0.0

    # Return the indices needed to cover up to the max threshold.
    need = int(n_for[tmax])
    return idx_sorted[:need], n_for, frac_for


@dataclass(frozen=True)
class BaselineCache:
    gap_root: Path
    run_label: str
    events: list[str]
    log_alpha_mu: np.ndarray
    log_alpha_gr: np.ndarray
    prior: BetaPrior
    n_f: int
    eps: float
    draw_idx: list[int]


def _load_baseline_cache(*, gap_root: Path, run_label: str) -> BaselineCache:
    gap_root = gap_root.expanduser().resolve()
    summary = _read_json(gap_root / f"summary_{run_label}.json")
    mix = summary.get("mixture", {})
    mix_meta = mix.get("f_miss_meta", {})
    prior_meta = (mix_meta.get("prior") or {})
    grid_meta = (mix_meta.get("grid") or {})
    prior = BetaPrior(mean=float(prior_meta["mean"]), kappa=float(prior_meta["kappa"]))
    n_f = int(grid_meta.get("n", 401))
    eps = float(grid_meta.get("eps", 1e-6))
    draw_idx = [int(i) for i in summary.get("draw_idx", [])]
    if not draw_idx:
        raise ValueError("baseline summary missing draw_idx")

    with np.load(gap_root / "tables" / f"selection_alpha_{run_label}.npz", allow_pickle=True) as d:
        log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
        log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)

    events = []
    for p in sorted((gap_root / "cache").glob("event_*.npz")):
        name = p.stem
        if name.startswith("event_"):
            events.append(name[len("event_") :])
    if not events:
        raise FileNotFoundError("No event_*.npz found under gap_root/cache")

    return BaselineCache(
        gap_root=gap_root,
        run_label=str(run_label),
        events=events,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=n_f,
        eps=eps,
        draw_idx=draw_idx,
    )


def _subset_mu_posterior(post: MuForwardPosterior, idx: list[int]) -> MuForwardPosterior:
    ii = np.asarray(idx, dtype=int)
    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=post.logmu_x_samples[ii],
        z_grid=post.z_grid,
        H_samples=post.H_samples[ii],
        H0=post.H0[ii],
        omega_m0=post.omega_m0[ii],
        omega_k0=post.omega_k0[ii],
        sigma8_0=post.sigma8_0[ii] if post.sigma8_0 is not None else None,
    )


def _load_event_cache(ev_npz: Path, *, pe_nside: int, p_credible: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PePixelDistanceHistogram]:
    with np.load(ev_npz, allow_pickle=False) as d:
        z = np.asarray(d["z"], dtype=float)
        w = np.asarray(d["w"], dtype=float)
        ipix = np.asarray(d["ipix"], dtype=np.int64)
        hpix_sel = np.asarray(d["hpix_sel"], dtype=np.int64)
        pe = PePixelDistanceHistogram(
            nside=int(pe_nside),
            nest=True,
            p_credible=float(p_credible),
            pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
            prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
            dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
            pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
        )
    return z, w, ipix, hpix_sel, pe


def _select_top_events(*, gap_root: Path, run_label: str, leverage_target: float, min_events: int, max_events: int) -> list[str]:
    ev_scores_path = gap_root / "tables" / f"event_scores_{run_label}.json"
    ev_scores = json.loads(ev_scores_path.read_text(encoding="utf-8"))
    ev_scores = [e for e in ev_scores if isinstance(e, dict) and isinstance(e.get("event"), str)]
    ev_scores.sort(key=lambda e: float(e.get("delta_lpd", 0.0)), reverse=True)
    dsum = float(np.sum([float(e.get("delta_lpd", 0.0)) for e in ev_scores]))
    out: list[str] = []
    acc = 0.0
    for e in ev_scores:
        if not np.isfinite(dsum) or dsum <= 0.0:
            break
        out.append(str(e["event"]))
        acc += float(e.get("delta_lpd", 0.0))
        if (acc / dsum) >= float(leverage_target) and len(out) >= int(min_events):
            break
    out = out[: max(int(min_events), min(int(max_events), len(out)))]
    return out


def _ned_query_specz(*, ra_deg: float, dec_deg: float, radius_arcsec: float) -> dict[str, Any]:
    # Keep raw table content (as dict of columns) for auditability.
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    tab = Ned.query_region(coord, radius=float(radius_arcsec) * u.arcsec)
    # Serialize as row dicts with masked values -> None (robust across dtypes).
    rows: list[dict[str, Any]] = []
    for r in tab:
        d: dict[str, Any] = {}
        for c in tab.colnames:
            v = r[c]
            if np.ma.is_masked(v):
                d[c] = None
                continue
            # Numpy scalars -> Python scalars.
            if isinstance(v, np.generic):
                v = v.item()
            # Bytes -> str.
            if isinstance(v, (bytes, np.bytes_)):
                try:
                    v = v.decode("utf-8", errors="replace")
                except Exception:
                    v = str(v)
            d[c] = v
        rows.append(d)

    # Heuristic pick: closest match if separation exists; else first with finite redshift.
    z_best = None
    pick = None
    if "Redshift" in tab.colnames and len(rows) > 0:
        z = np.asarray([np.nan if r.get("Redshift") is None else float(r["Redshift"]) for r in rows], dtype=float)
        ok = np.isfinite(z)
        if np.any(ok):
            if "Separation" in tab.colnames:
                sep = np.asarray([np.inf if r.get("Separation") is None else float(r["Separation"]) for r in rows], dtype=float)
                j = int(np.argmin(np.where(ok, sep, np.inf)))
            else:
                j = int(np.argmax(ok))
            z_best = float(z[j])
            pick = int(j)
    return {
        "service": "ned",
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "radius_arcsec": float(radius_arcsec),
        "picked_row": pick,
        "z_best": z_best,
        "table_rows": rows,
        "n_matches": int(len(rows)),
    }


def _interp_photoz_grid(b0: np.ndarray, b1: np.ndarray, grid: dict[tuple[float, float], float]) -> np.ndarray:
    # Bilinear interpolation on a rectilinear (b0,b1) grid; values outside are clipped.
    b0_u = sorted({float(k[0]) for k in grid.keys()})
    b1_u = sorted({float(k[1]) for k in grid.keys()})
    if len(b0_u) < 2 or len(b1_u) < 2:
        raise ValueError("photo-z grid too small for interpolation")
    b0 = np.asarray(b0, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b0c = np.clip(b0, b0_u[0], b0_u[-1])
    b1c = np.clip(b1, b1_u[0], b1_u[-1])

    # Indices of left grid points.
    i0 = np.searchsorted(b0_u, b0c, side="right") - 1
    j0 = np.searchsorted(b1_u, b1c, side="right") - 1
    i0 = np.clip(i0, 0, len(b0_u) - 2)
    j0 = np.clip(j0, 0, len(b1_u) - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    x0 = np.asarray([b0_u[int(i)] for i in i0], dtype=float)
    x1 = np.asarray([b0_u[int(i)] for i in i1], dtype=float)
    y0 = np.asarray([b1_u[int(j)] for j in j0], dtype=float)
    y1 = np.asarray([b1_u[int(j)] for j in j1], dtype=float)
    tx = np.where(x1 > x0, (b0c - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (b1c - y0) / (y1 - y0), 0.0)

    def v(x: float, y: float) -> float:
        return float(grid[(float(x), float(y))])

    v00 = np.asarray([v(x0[k], y0[k]) for k in range(b0c.size)], dtype=float)
    v10 = np.asarray([v(x1[k], y0[k]) for k in range(b0c.size)], dtype=float)
    v01 = np.asarray([v(x0[k], y1[k]) for k in range(b0c.size)], dtype=float)
    v11 = np.asarray([v(x1[k], y1[k]) for k in range(b0c.size)], dtype=float)
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11


def _spectral_only_logL_from_weight_hist(
    *,
    pe: PePixelDistanceHistogram,
    z_cent: np.ndarray,
    weight_hist: np.ndarray,
    z_grid_post: np.ndarray,
    dL_em_grid: np.ndarray,
    R_grid: np.ndarray,
    gw_prior: GWDistancePrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate spectral-only catalog logL vectors using a binned-z representation.

    This is a controlled approximation to `compute_dark_siren_logL_draws_from_pe_hist_fast(..., distance_mode='spectral_only')`
    that replaces the galaxy-level sum over (potentially millions) of entries with a sum over z bins (typically O(10^2)).
    """
    z_cent = np.asarray(z_cent, dtype=float)
    weight_hist = np.asarray(weight_hist, dtype=float)
    if z_cent.ndim != 1 or weight_hist.ndim != 1 or z_cent.size != weight_hist.size:
        raise ValueError("z_cent and weight_hist must be 1D and same length.")
    m = np.isfinite(z_cent) & (z_cent > 0.0) & np.isfinite(weight_hist) & (weight_hist > 0.0)
    if not np.any(m):
        n_draws = int(dL_em_grid.shape[0])
        return np.full((n_draws,), -np.inf, dtype=float), np.full((n_draws,), -np.inf, dtype=float)
    z_cent = z_cent[m]
    w = weight_hist[m]

    edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
    nb = int(edges.size - 1)

    n_draws = int(dL_em_grid.shape[0])
    out_mu = np.full((n_draws,), -np.inf, dtype=float)
    out_gr = np.full((n_draws,), -np.inf, dtype=float)
    for j in range(n_draws):
        dL_em = np.interp(z_cent, z_grid_post, np.asarray(dL_em_grid[j], dtype=float))
        R = np.interp(z_cent, z_grid_post, np.asarray(R_grid[j], dtype=float))
        dL_gw = dL_em * R

        def _logL(dL: np.ndarray) -> float:
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
            if not np.any(valid):
                return float("-inf")
            pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
            pdf = np.where(valid, pdf, 0.0)
            inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL, 1e-6, np.inf)))
            t = w * pdf * inv_pi
            s = float(np.sum(t))
            return float(np.log(max(s, 1e-300)))

        out_gr[j] = _logL(dL_em)
        out_mu[j] = _logL(dL_gw)
    return out_mu, out_gr


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-O4b non-strain smoking-gun suite (spec-z override + catalog weight swap + photo-z prior marginalization).")
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--outdir", default=None, help="Optional output directory under outputs/.")
    ap.add_argument("--offline", action="store_true", help="Do not perform any online spec-z lookups; emit query pack only.")
    args = ap.parse_args()

    _set_thread_env(int(args.threads))

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _read_json(cfg_path)

    out_root = Path(args.outdir).expanduser().resolve() if args.outdir else (REPO_ROOT / "outputs" / f"dark_siren_smoking_gun_nond_{_utc_now_compact()}")
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    (out_root / "configs" / cfg_path.name).write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_cfg = cfg["baseline"]
    gap_root = Path(base_cfg["gap_run_root"]).expanduser().resolve()
    run_label = str(base_cfg["run_label"])
    recon_run_dir = Path(base_cfg["recon_run_dir"]).expanduser().resolve()

    baseline = _load_baseline_cache(gap_root=gap_root, run_label=run_label)
    manifest = _read_json(gap_root / "manifest.json")
    pe_nside = int(manifest.get("pe_nside", 64))
    p_credible = float(manifest.get("p_credible", 0.9))

    # Load mu forward posterior and match draw set.
    post_full = load_mu_forward_posterior(str(recon_run_dir))
    post = _subset_mu_posterior(post_full, baseline.draw_idx)
    z_grid_post = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = np.asarray(predict_dL_em(post, z_eval=z_grid_post), dtype=float)
    _, R_grid = predict_r_gw_em(post, z_eval=z_grid_post, convention=str(manifest.get("convention", "A")))
    R_grid = np.asarray(R_grid, dtype=float)

    # Baseline spectral-only terms from an existing hardening run (fast path).
    spec_terms_npz = Path(base_cfg["spectral_only_terms_npz"]).expanduser().resolve()
    with np.load(spec_terms_npz, allow_pickle=False) as d:
        events = [str(x) for x in d["events"].tolist()]
        base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
        base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
        base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
        base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]
    ev_to_idx = {e: i for i, e in enumerate(events)}

    # Baseline score (spectral-only, with selection normalization).
    base_res = marginalize_f_miss_global(
        logL_cat_mu_by_event=base_cat_mu,
        logL_cat_gr_by_event=base_cat_gr,
        logL_missing_mu_by_event=base_miss_mu,
        logL_missing_gr_by_event=base_miss_gr,
        log_alpha_mu=baseline.log_alpha_mu,
        log_alpha_gr=baseline.log_alpha_gr,
        prior=baseline.prior,
        n_f=baseline.n_f,
        eps=baseline.eps,
    )
    base_delta = float(base_res.lpd_mu_total - base_res.lpd_gr_total)
    base_delta_data = float(base_res.lpd_mu_total_data - base_res.lpd_gr_total_data)

    # Select top events using the same leverage proxy as the existing hardening suite.
    sel_cfg = cfg.get("event_selection", {})
    top_events = _select_top_events(
        gap_root=gap_root,
        run_label=run_label,
        leverage_target=float(sel_cfg.get("leverage_target_fraction", 0.8)),
        min_events=int(sel_cfg.get("min_events", 5)),
        max_events=int(sel_cfg.get("max_events", 12)),
    )

    # Load galaxy indices (lumB and uniform).
    gl_cfg = cfg["glade"]
    cat_lumB = load_gladeplus_index(gl_cfg["index_lumB"])
    cat_uni = load_gladeplus_index(gl_cfg["index_uniform"])

    # Dominant-host lists and optional spec-z lookup.
    dom_cfg = cfg.get("dominant_hosts", {})
    thresholds = [float(x) for x in dom_cfg.get("cumulative_weight_thresholds", [0.5, 0.8])]
    initial_topk = int(dom_cfg.get("initial_topk", 20000))
    max_topk = int(dom_cfg.get("max_topk", 2_000_000))

    specz_cfg = cfg.get("specz_lookup", {})
    specz_enabled = bool(specz_cfg.get("enabled", False)) and not bool(args.offline)
    radius_arcsec = float(specz_cfg.get("radius_arcsec", 3.0))
    sleep_sec = float(specz_cfg.get("sleep_sec", 0.5))
    max_q = int(specz_cfg.get("max_total_queries", 400))
    q_count = 0

    scoring_cfg = cfg.get("scoring", {})
    distance_mode = str(scoring_cfg.get("distance_mode", "spectral_only"))
    gal_chunk_size = int(scoring_cfg.get("gal_chunk_size", 50000))
    z_hist_nbins = int(scoring_cfg.get("z_hist_nbins", 400))
    if distance_mode != "spectral_only":
        raise ValueError("This suite currently supports only distance_mode='spectral_only' for the binned-z scoring path.")

    gw_prior = GWDistancePrior(
        mode="dL_powerlaw",
        powerlaw_k=float(manifest.get("gw_distance_prior_power", 2.0)),
        h0_ref=float(manifest.get("gw_distance_prior_h0_ref", 67.7)),
        omega_m0=float(manifest.get("gw_distance_prior_omega_m0", 0.31)),
        omega_k0=float(manifest.get("gw_distance_prior_omega_k0", 0.0)),
        z_max=float(manifest.get("gw_distance_prior_zmax", 10.0)),
        n_grid=50_000,
    )

    # Fiducial dL(z) for dominant-host ranking.
    ppc_cfg = cfg.get("ppc_residuals", {})
    zmax_rank = float(cfg.get("specz_override", {}).get("z_max", 0.3))
    z_grid_fid, dL_grid_fid = _build_fiducial_dL_of_z(
        h0=float(ppc_cfg.get("h0_ref", gw_prior.h0_ref)),
        omega_m0=float(ppc_cfg.get("omega_m0", gw_prior.omega_m0)),
        z_max=zmax_rank,
        n=5001,
    )

    dominant_rows_summary: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    overrides_by_event: dict[str, dict[float, list[dict[str, Any]]]] = {}

    query_pack_rows: list[dict[str, Any]] = []

    for ev in top_events:
        ev_npz = gap_root / "cache" / f"event_{ev}.npz"
        if not ev_npz.exists():
            continue
        z_cat, w_cat, ipix_cat, hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)
        edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
        widths = np.diff(edges)

        # Re-gather galaxies with coordinates from the lumB index and confirm alignment.
        ra, dec, z2, w2 = gather_galaxies_from_pixels(cat_lumB, hpix_sel, z_max=zmax_rank)
        if z2.size != z_cat.size or w2.size != w_cat.size:
            raise RuntimeError(f"{ev}: mismatch between cache galaxy arrays and gathered glade index arrays.")
        # Spot-check agreement on z/w (avoid full scan).
        rng = np.random.default_rng(0)
        check_n = min(2048, int(z_cat.size))
        ii = rng.integers(0, int(z_cat.size), size=check_n, endpoint=False)
        if not (np.allclose(z2[ii], z_cat[ii], rtol=0.0, atol=0.0) and np.allclose(w2[ii], w_cat[ii], rtol=0.0, atol=0.0)):
            raise RuntimeError(f"{ev}: cache arrays do not match glade index order; cannot attach coordinates reliably.")

        # Host-weight proxy (GR fiducial): w * prob_pix * pdf_1d(dL(z)) / pi(dL(z)).
        dL_em = np.interp(np.clip(z_cat, 0.0, z_grid_fid[-1]), z_grid_fid, dL_grid_fid)
        bin_idx = np.searchsorted(edges, dL_em, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < widths.size) & np.isfinite(dL_em) & (dL_em > 0.0)
        pdf = np.zeros_like(dL_em, dtype=float)
        pdf[valid] = pdf_1d[bin_idx[valid]]
        inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em, 1e-6, np.inf)))

        npix = int(12 * int(pe.nside) * int(pe.nside))
        pix_to_row = np.full((npix,), -1, dtype=np.int32)
        pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
        row = pix_to_row[ipix_cat]
        good = (row >= 0) & np.isfinite(z_cat) & (z_cat > 0.0) & np.isfinite(w_cat) & (w_cat > 0.0) & (pdf > 0.0) & np.isfinite(inv_pi)
        prob = np.zeros_like(z_cat, dtype=float)
        prob[good] = np.asarray(pe.prob_pix, dtype=float)[row[good]]
        weight = np.zeros_like(z_cat, dtype=float)
        weight[good] = w_cat[good] * prob[good] * pdf[good] * inv_pi[good]

        idx_dom, n_for, frac_for = _dominant_indices_by_weight(weight=weight, thresholds=thresholds, initial_topk=initial_topk, max_topk=max_topk)
        # Sort the returned dominant set by weight for reporting.
        idx_dom = np.asarray(idx_dom, dtype=np.int64)
        order = np.argsort(weight[idx_dom])[::-1]
        idx_dom = idx_dom[order]
        cw = np.cumsum(weight[idx_dom])
        tot = float(np.sum(weight))
        cum = cw / tot if tot > 0 else np.zeros_like(cw)

        # Per-event host candidate table up to the max threshold.
        rows = []
        for r, gi in enumerate(idx_dom.tolist(), start=1):
            rows.append(
                {
                    "event": ev,
                    "rank": int(r),
                    "gal_index_in_event": int(gi),
                    "ra_deg": float(ra[gi]),
                    "dec_deg": float(dec[gi]),
                    "z_cat": float(z_cat[gi]),
                    "weight": float(weight[gi]),
                    "cum_weight_frac": float(cum[r - 1]) if cum.size else 0.0,
                }
            )
        _write_csv(tab_dir / f"dominant_hosts_{ev}.csv", rows, fieldnames=["event", "rank", "gal_index_in_event", "ra_deg", "dec_deg", "z_cat", "weight", "cum_weight_frac"])

        dominant_rows_summary.append(
            {
                "event": ev,
                "n_gal": int(z_cat.size),
                "n_dom_for_0p5": int(n_for.get(0.5, 0)),
                "n_dom_for_0p8": int(n_for.get(0.8, 0)),
                "cum_frac_for_0p5": float(frac_for.get(0.5, 0.0)),
                "cum_frac_for_0p8": float(frac_for.get(0.8, 0.0)),
            }
        )

        overrides_by_event[ev] = {}
        for thr in [float(x) for x in cfg.get("specz_override", {}).get("override_thresholds", thresholds)]:
            n_need = int(n_for.get(float(thr), 0))
            idx_thr = idx_dom[:n_need]
            cand = []
            for gi in idx_thr.tolist():
                cand.append({"gal_index_in_event": int(gi), "ra_deg": float(ra[gi]), "dec_deg": float(dec[gi]), "z_cat": float(z_cat[gi]), "weight": float(weight[gi])})
                query_pack_rows.append({"event": ev, "gal_index_in_event": int(gi), "ra_deg": float(ra[gi]), "dec_deg": float(dec[gi]), "z_cat": float(z_cat[gi]), "weight": float(weight[gi]), "threshold": float(thr)})
            overrides_by_event[ev][thr] = cand

    # Always emit the query pack (offline reproducibility).
    _write_csv(tab_dir / "specz_query_pack.csv", query_pack_rows, fieldnames=["event", "threshold", "gal_index_in_event", "ra_deg", "dec_deg", "z_cat", "weight"])

    # Perform spec-z lookups (NED) for a *bounded* set of candidates. We prioritise by
    # host-weight proxy to maximise coverage per query, and we do not emit per-candidate
    # "skipped" files beyond the cap (the query pack already records the full request set).
    specz_results: dict[str, dict[int, dict[str, Any]]] = {}
    if bool(cfg.get("specz_override", {}).get("enabled", False)) and specz_enabled:
        global_cache_dir = REPO_ROOT / "data" / "cache" / "specz_ned_raw"
        global_cache_dir.mkdir(parents=True, exist_ok=True)

        # Build unique candidates up to the maximum threshold per event.
        all_cand = []
        for ev, by_thr in overrides_by_event.items():
            specz_results.setdefault(ev, {})
            tmax = float(max(by_thr.keys())) if by_thr else None
            if tmax is None:
                continue
            for c in by_thr[tmax]:
                all_cand.append(
                    {
                        "event": ev,
                        "gal_index_in_event": int(c["gal_index_in_event"]),
                        "ra_deg": float(c["ra_deg"]),
                        "dec_deg": float(c["dec_deg"]),
                        "weight": float(c.get("weight", 0.0)),
                    }
                )

        # Deduplicate and sort by weight descending.
        seen = set()
        uniq = []
        for c in all_cand:
            key = (c["event"], c["gal_index_in_event"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        uniq.sort(key=lambda r: float(r.get("weight", 0.0)), reverse=True)

        for c in uniq:
            if q_count >= max_q:
                break
            ev = str(c["event"])
            gi = int(c["gal_index_in_event"])
            if gi in specz_results[ev]:
                continue
            cache_key = f"{ev}__{gi}"
            raw_path = raw_dir / "specz_ned_raw" / f"{cache_key}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            global_path = global_cache_dir / f"{cache_key}.json"
            need_query = True
            rec: dict[str, Any] | None = None
            if global_path.exists():
                try:
                    rec = json.loads(global_path.read_text(encoding="utf-8"))
                except Exception:
                    rec = None
                # Refresh cached error records from older runs.
                if isinstance(rec, dict) and not rec.get("error"):
                    need_query = False

            did_query = False
            if need_query:
                try:
                    rec = _ned_query_specz(ra_deg=float(c["ra_deg"]), dec_deg=float(c["dec_deg"]), radius_arcsec=radius_arcsec)
                except Exception as e:
                    rec = {"service": "ned", "error": str(e), "ra_deg": float(c["ra_deg"]), "dec_deg": float(c["dec_deg"]), "radius_arcsec": float(radius_arcsec)}
                global_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                did_query = True

            if not isinstance(rec, dict):
                rec = {"service": "ned", "error": "cache_read_failed", "ra_deg": float(c["ra_deg"]), "dec_deg": float(c["dec_deg"]), "radius_arcsec": float(radius_arcsec)}
            # Copy into this run's audit directory for self-contained artifacts.
            raw_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            specz_results[ev][gi] = rec
            q_count += 1
            if did_query:
                time.sleep(max(0.0, sleep_sec))

    # Helper: compute Delta LPD with replacements to catalog terms (spectral-only base).
    def _score_with_cat_replacements(repl: dict[str, tuple[np.ndarray, np.ndarray]]) -> MarginalizedFMissResult:
        cat_mu = list(base_cat_mu)
        cat_gr = list(base_cat_gr)
        for ev, (mu, gr) in repl.items():
            j = ev_to_idx[ev]
            cat_mu[j] = np.asarray(mu, dtype=float)
            cat_gr[j] = np.asarray(gr, dtype=float)
        return marginalize_f_miss_global(
            logL_cat_mu_by_event=cat_mu,
            logL_cat_gr_by_event=cat_gr,
            logL_missing_mu_by_event=base_miss_mu,
            logL_missing_gr_by_event=base_miss_gr,
            log_alpha_mu=baseline.log_alpha_mu,
            log_alpha_gr=baseline.log_alpha_gr,
            prior=baseline.prior,
            n_f=baseline.n_f,
            eps=baseline.eps,
        )

    # ---- SPEC-Z OVERRIDE SCORING
    specz_override_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("specz_override", {}).get("enabled", False)):
        z_max = float(cfg["specz_override"].get("z_max", 0.3))
        mode = str(cfg["specz_override"].get("out_of_support_mode", "drop_weight"))
        thr_list = [float(x) for x in cfg["specz_override"].get("override_thresholds", thresholds)]

        rows = []
        for thr in thr_list:
            repl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            coverage = []
            for ev in top_events:
                ev_npz = gap_root / "cache" / f"event_{ev}.npz"
                z_cat, w_cat, ipix_cat, _hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)
                z_new = np.asarray(z_cat, dtype=float).copy()
                w_new = np.asarray(w_cat, dtype=float).copy()

                # Precompute w*prob weights and a baseline z histogram for this event.
                npix = int(12 * int(pe.nside) * int(pe.nside))
                pix_to_row = np.full((npix,), -1, dtype=np.int32)
                pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
                row = pix_to_row[ipix_cat]
                good = (row >= 0) & np.isfinite(z_cat) & (z_cat > 0.0) & np.isfinite(w_cat) & (w_cat > 0.0)
                wprob = np.zeros_like(z_cat, dtype=float)
                if np.any(good):
                    wprob[good] = np.asarray(w_cat[good], dtype=float) * np.asarray(pe.prob_pix, dtype=float)[row[good]]
                z_edges = np.linspace(0.0, z_max, z_hist_nbins + 1)
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
                hist, _ = np.histogram(np.clip(z_cat[good], 0.0, z_max), bins=z_edges, weights=wprob[good])
                hist = np.asarray(hist, dtype=float)

                cand = overrides_by_event.get(ev, {}).get(thr, [])
                wt_total = float(np.sum([float(c.get("weight", 0.0)) for c in cand]))
                wt_over = 0.0
                n_spec = 0
                for c in cand:
                    gi = int(c["gal_index_in_event"])
                    rec = specz_results.get(ev, {}).get(gi, {})
                    z_spec = rec.get("z_best")
                    if z_spec is None or not np.isfinite(float(z_spec)):
                        continue
                    z_spec_f = float(z_spec)
                    # Update the binned-z weight histogram by moving w*prob mass between bins.
                    wprob_gi = float(wprob[gi])
                    if not (np.isfinite(wprob_gi) and wprob_gi > 0.0):
                        continue
                    # Remove from old bin.
                    b_old = int(np.searchsorted(z_edges, float(z_cat[gi]), side="right") - 1)
                    if 0 <= b_old < hist.size:
                        hist[b_old] = max(0.0, float(hist[b_old] - wprob_gi))
                    # Add to new bin if in support.
                    if z_spec_f <= 0.0 or z_spec_f > z_max:
                        if mode != "drop_weight":
                            z_spec_f = float(np.clip(z_spec_f, 1e-6, z_max))
                            b_new = int(np.searchsorted(z_edges, z_spec_f, side="right") - 1)
                            if 0 <= b_new < hist.size:
                                hist[b_new] = float(hist[b_new] + wprob_gi)
                    else:
                        b_new = int(np.searchsorted(z_edges, z_spec_f, side="right") - 1)
                        if 0 <= b_new < hist.size:
                            hist[b_new] = float(hist[b_new] + wprob_gi)
                    wt_over += float(c.get("weight", 0.0))
                    n_spec += 1

                # Recompute catalog logL terms for this event using the binned-z approximation.
                logL_mu, logL_gr = _spectral_only_logL_from_weight_hist(
                    pe=pe,
                    z_cent=z_cent,
                    weight_hist=hist,
                    z_grid_post=z_grid_post,
                    dL_em_grid=dL_em_grid,
                    R_grid=R_grid,
                    gw_prior=gw_prior,
                )
                repl[ev] = (logL_mu, logL_gr)
                coverage.append({"event": ev, "threshold": float(thr), "n_candidates": int(len(cand)), "n_specz_used": int(n_spec), "weight_covered": float(wt_over), "weight_target": float(wt_total)})

            res = _score_with_cat_replacements(repl)
            delta = float(res.lpd_mu_total - res.lpd_gr_total)
            delta_data = float(res.lpd_mu_total_data - res.lpd_gr_total_data)
            rows.append({"threshold": float(thr), "delta_lpd_total": float(delta), "delta_lpd_data": float(delta_data)})
            for c in coverage:
                c.update({"delta_lpd_total": float(delta), "delta_lpd_data": float(delta_data)})
                override_rows.append(c)

        _write_csv(tab_dir / "specz_override_score_rows.csv", rows, fieldnames=["threshold", "delta_lpd_total", "delta_lpd_data"])
        _write_csv(
            tab_dir / "specz_override_coverage_rows.csv",
            override_rows,
            fieldnames=["event", "threshold", "n_candidates", "n_specz_used", "weight_target", "weight_covered", "delta_lpd_total", "delta_lpd_data"],
        )
        specz_override_summary = {"enabled": True, "rows": rows, "query_count": int(q_count), "max_total_queries": int(max_q), "radius_arcsec": float(radius_arcsec)}

    # ---- CATALOG WEIGHT SWAP (uniform weights)
    cat_swap_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("catalog_weight_swap", {}).get("enabled", False)):
        repl = {}
        rows = []
        for ev in top_events:
            ev_npz = gap_root / "cache" / f"event_{ev}.npz"
            _z_cat, _w_cat, _ipix_cat, hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)

            # "Catalog swap" within GLADE+: use the uniform-weight index, which includes galaxies
            # omitted by the luminosity-weighted index (e.g., missing M_B). This is not an
            # independent catalog, but it is a meaningful stress test of host-weight modelling.
            ra_u, dec_u, z_u, w_u = gather_galaxies_from_pixels(cat_uni, hpix_sel, z_max=zmax_rank)
            if z_u.size == 0:
                continue
            theta = np.deg2rad(90.0 - np.asarray(dec_u, dtype=float))
            phi = np.deg2rad(np.asarray(ra_u, dtype=float))
            ipix_u = hp.ang2pix(int(pe.nside), theta, phi, nest=True).astype(np.int64, copy=False)

            npix = int(12 * int(pe.nside) * int(pe.nside))
            pix_to_row = np.full((npix,), -1, dtype=np.int32)
            pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
            row = pix_to_row[ipix_u]
            good = (row >= 0) & np.isfinite(z_u) & (z_u > 0.0) & np.isfinite(w_u) & (w_u > 0.0)
            wprob = np.zeros_like(z_u, dtype=float)
            if np.any(good):
                wprob[good] = np.asarray(w_u[good], dtype=float) * np.asarray(pe.prob_pix, dtype=float)[row[good]]
            z_edges = np.linspace(0.0, zmax_rank, z_hist_nbins + 1)
            z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
            hist, _ = np.histogram(np.clip(z_u[good], 0.0, zmax_rank), bins=z_edges, weights=wprob[good])
            hist = np.asarray(hist, dtype=float)
            logL_mu, logL_gr = _spectral_only_logL_from_weight_hist(
                pe=pe,
                z_cent=z_cent,
                weight_hist=hist,
                z_grid_post=z_grid_post,
                dL_em_grid=dL_em_grid,
                R_grid=R_grid,
                gw_prior=gw_prior,
            )
            repl[ev] = (logL_mu, logL_gr)
            rows.append({"event": ev, "note": "uniform_glade_index_variant"})

        res = _score_with_cat_replacements(repl)
        cat_swap_summary = {
            "enabled": True,
            "top_events": top_events,
            "delta_lpd_total": float(res.lpd_mu_total - res.lpd_gr_total),
            "delta_lpd_data": float(res.lpd_mu_total_data - res.lpd_gr_total_data),
        }

    # ---- PHOTO-Z PRIOR MARGINALISATION (using existing grid; fast)
    pzprior_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("photoz_prior_marginalization", {}).get("enabled", False)):
        pzcfg = cfg["photoz_prior_marginalization"]
        b0_sig = float(pzcfg.get("prior_b0_sigma", 0.01))
        b1_sig = float(pzcfg.get("prior_b1_sigma", 0.1))
        n_samp = int(pzcfg.get("n_samples", 200000))
        seed = int(pzcfg.get("seed", 123))

        grid_path = Path(base_cfg["photoz_grid_csv"]).expanduser().resolve()
        rows = []
        for r in csv.DictReader(grid_path.read_text(encoding="utf-8").splitlines()):
            rows.append(r)
        grid = {(float(r["b0"]), float(r["b1"])): float(r["delta_lpd_total"]) for r in rows}
        rng = np.random.default_rng(seed)
        b0 = rng.normal(0.0, b0_sig, size=n_samp)
        b1 = rng.normal(0.0, b1_sig, size=n_samp)
        dlp = _interp_photoz_grid(b0, b1, grid)
        p_lt1 = float(np.mean(dlp < 1.0))
        p_lt0 = float(np.mean(dlp < 0.0))
        p_ge_base = float(np.mean(dlp >= base_delta))
        pzprior_summary = {
            "enabled": True,
            "grid_csv": str(grid_path),
            "prior_b0_sigma": float(b0_sig),
            "prior_b1_sigma": float(b1_sig),
            "n_samples": int(n_samp),
            "seed": int(seed),
            "delta_lpd_base_spectral_only": float(base_delta),
            "delta_lpd_mean_under_prior": float(np.mean(dlp)),
            "delta_lpd_p16_p50_p84": [float(np.quantile(dlp, q)) for q in (0.16, 0.5, 0.84)],
            "p_delta_lpd_lt_1": p_lt1,
            "p_delta_lpd_lt_0": p_lt0,
            "p_delta_lpd_ge_base": p_ge_base,
            "notes": "Fast marginalisation uses bilinear interpolation on the existing (b0,b1) stress grid; values are clipped to grid bounds.",
        }
        # Plot distribution.
        try:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.hist(dlp[np.isfinite(dlp)], bins=60, color="C0", alpha=0.85)
            ax.axvline(base_delta, color="C3", lw=2.0, label="baseline")
            ax.axvline(1.0, color="k", lw=1.2, linestyle="--", label="ΔLPD=1")
            ax.set(xlabel="ΔLPD (spectral-only total)", ylabel="count", title="Photo-z prior marginalisation (bias-only)")
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "photoz_prior_marginal_hist.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    # ---- SIMPLE MECHANISM-ALIGNMENT RESIDUAL PLOTS (supporting diagnostic)
    ppc_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("ppc_residuals", {}).get("enabled", False)):
        # Use event-level PE distance median and GR host-weight z_eff (same proxy used in hardening suite).
        rows = []
        for ev in events:
            ev_npz = gap_root / "cache" / f"event_{ev}.npz"
            if not ev_npz.exists():
                continue
            z_cat, w_cat, ipix_cat, _hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)
            # PE distance median from sky-marginal histogram.
            edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
            widths = np.diff(edges)
            dmid = 0.5 * (edges[:-1] + edges[1:])
            cdf = np.cumsum(pdf_1d * widths)
            j = int(np.searchsorted(cdf, 0.5, side="left"))
            j = min(max(j, 0), int(widths.size - 1))
            dL_med = float(dmid[j])

            dL_em = np.interp(np.clip(z_cat, 0.0, z_grid_fid[-1]), z_grid_fid, dL_grid_fid)
            bin_idx = np.searchsorted(edges, dL_em, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < widths.size) & np.isfinite(dL_em) & (dL_em > 0.0)
            pdf = np.zeros_like(dL_em, dtype=float)
            pdf[valid] = pdf_1d[bin_idx[valid]]
            inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em, 1e-6, np.inf)))

            npix = int(12 * int(pe.nside) * int(pe.nside))
            pix_to_row = np.full((npix,), -1, dtype=np.int32)
            pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
            rowm = pix_to_row[ipix_cat]
            good = (rowm >= 0) & np.isfinite(z_cat) & (z_cat > 0.0) & np.isfinite(w_cat) & (w_cat > 0.0) & (pdf > 0.0) & np.isfinite(inv_pi)
            prob = np.zeros_like(z_cat, dtype=float)
            prob[good] = np.asarray(pe.prob_pix, dtype=float)[rowm[good]]
            wt = np.zeros_like(z_cat, dtype=float)
            wt[good] = w_cat[good] * prob[good] * pdf[good] * inv_pi[good]
            if float(np.sum(wt)) <= 0.0:
                continue
            # Weighted median z_eff.
            order = np.argsort(z_cat, kind="mergesort")
            z_s = z_cat[order]
            wt_s = wt[order]
            cw = np.cumsum(wt_s)
            z_eff = float(z_s[int(np.searchsorted(cw, 0.5 * float(cw[-1]), side="left"))])

            dL_em_eff = float(np.interp(z_eff, z_grid_fid, dL_grid_fid))
            # MG prediction uses posterior mean R(z_eff) from the mu-forward draws.
            _, R_eff = predict_r_gw_em(post, z_eval=np.asarray([z_eff], dtype=float), convention=str(manifest.get("convention", "A")))
            R_mean = float(np.mean(np.asarray(R_eff, dtype=float)[:, 0]))
            dL_mu_eff = float(dL_em_eff * R_mean)
            rows.append(
                {
                    "event": ev,
                    "dL_med_mpc": float(dL_med),
                    "z_eff_gr": float(z_eff),
                    "dL_em_pred_mpc": float(dL_em_eff),
                    "dL_mu_pred_mpc": float(dL_mu_eff),
                    "R_mean": float(R_mean),
                    "log_resid_gr": float(np.log(dL_med / max(dL_em_eff, 1e-9))),
                    "log_resid_mu": float(np.log(dL_med / max(dL_mu_eff, 1e-9))),
                }
            )
        if rows:
            _write_csv(tab_dir / "ppc_distance_residuals.csv", rows, fieldnames=list(rows[0].keys()))
            try:
                z = np.asarray([r["z_eff_gr"] for r in rows], dtype=float)
                rg = np.asarray([r["log_resid_gr"] for r in rows], dtype=float)
                rm = np.asarray([r["log_resid_mu"] for r in rows], dtype=float)
                fig, ax = plt.subplots(figsize=(6.4, 3.8))
                ax.scatter(z, rg, s=18, alpha=0.8, label="GR residual", color="C3")
                ax.scatter(z, rm, s=18, alpha=0.8, label="MG residual (mean R)", color="C0")
                ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
                ax.set(xlabel="z_eff (GR-weight proxy)", ylabel="log(dL_med / dL_pred)", title="Distance residual proxy vs z_eff")
                ax.grid(alpha=0.25, linestyle=":")
                ax.legend(loc="best", frameon=False)
                fig.tight_layout()
                fig.savefig(fig_dir / "ppc_distance_residuals_vs_z.png", dpi=180)
                plt.close(fig)
            except Exception:
                pass
            ppc_summary = {"enabled": True, "n_events": int(len(rows))}

    # Write summaries and a short report.
    _write_csv(
        tab_dir / "dominant_hosts_summary.csv",
        dominant_rows_summary,
        fieldnames=["event", "n_gal", "n_dom_for_0p5", "n_dom_for_0p8", "cum_frac_for_0p5", "cum_frac_for_0p8"],
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(cfg_path),
        "baseline": {
            "gap_run_root": str(gap_root),
            "run_label": str(run_label),
            "recon_run_dir": str(recon_run_dir),
            "spectral_only_terms_npz": str(spec_terms_npz),
            "delta_lpd_total_spectral_only": float(base_delta),
            "delta_lpd_data_spectral_only": float(base_delta_data),
        },
        "top_events": top_events,
        "specz_override": specz_override_summary,
        "catalog_weight_swap": cat_swap_summary,
        "photoz_prior_marginalization": pzprior_summary,
        "ppc_residuals": ppc_summary,
        "offline": bool(args.offline),
    }
    _write_json(out_root / "summary.json", summary)

    # Minimal markdown report.
    lines = []
    lines.append("# Dark-Siren Non-Strain Smoking-Gun Suite (Pre-O4b)\n")
    lines.append(f"- Generated: `{summary['generated_utc']}`\n")
    lines.append(f"- Baseline spectral-only ΔLPD (total): `{base_delta:+.3f}` (data-only: `{base_delta_data:+.3f}`)\n")
    lines.append(f"- Top events (leverage proxy): {', '.join(top_events)}\n")
    lines.append("\n## Dominant Host Candidates (GR-weight proxy)\n")
    lines.append(f"- Summary CSV: `tables/dominant_hosts_summary.csv`\n")
    lines.append(f"- Per-event CSVs: `tables/dominant_hosts_<EVENT>.csv`\n")
    lines.append("\n## Spec-z Override\n")
    if specz_override_summary.get("enabled"):
        lines.append(f"- Queries executed: `{specz_override_summary.get('query_count', 0)}` (cap `{specz_override_summary.get('max_total_queries')}`)\n")
        lines.append("- Score rows: `tables/specz_override_score_rows.csv`\n")
        lines.append("- Coverage rows: `tables/specz_override_coverage_rows.csv`\n")
    else:
        lines.append("- Disabled.\n")
    lines.append("\n## Catalog Weight Swap (Uniform w)\n")
    if cat_swap_summary.get("enabled"):
        lines.append(f"- ΔLPD (total): `{cat_swap_summary['delta_lpd_total']:+.3f}` (data-only: `{cat_swap_summary['delta_lpd_data']:+.3f}`)\n")
    else:
        lines.append("- Disabled.\n")
    lines.append("\n## Photo-z Prior Marginalisation (Bias-only)\n")
    if pzprior_summary.get("enabled"):
        lines.append(f"- Prior σ(b0)={pzprior_summary['prior_b0_sigma']}, σ(b1)={pzprior_summary['prior_b1_sigma']}\n")
        lines.append(f"- Mean ΔLPD under prior: `{pzprior_summary['delta_lpd_mean_under_prior']:+.3f}`\n")
        lines.append(f"- P(ΔLPD<1): `{pzprior_summary['p_delta_lpd_lt_1']:.4f}`, P(ΔLPD<0): `{pzprior_summary['p_delta_lpd_lt_0']:.4f}`\n")
        lines.append("- Plot: `figures/photoz_prior_marginal_hist.png`\n")
    else:
        lines.append("- Disabled.\n")
    lines.append("\n## Mechanism Residual Diagnostic (Supporting)\n")
    if ppc_summary.get("enabled"):
        lines.append("- Table: `tables/ppc_distance_residuals.csv`\n")
        lines.append("- Plot: `figures/ppc_distance_residuals_vs_z.png`\n")
    else:
        lines.append("- Disabled.\n")

    (out_root / "report.md").write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
