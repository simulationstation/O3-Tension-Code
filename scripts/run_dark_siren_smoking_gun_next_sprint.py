#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
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

from astroquery.vizier import Vizier  # noqa: E402
from scipy.special import erf  # noqa: E402

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, MarginalizedFMissResult, marginalize_f_miss_global  # noqa: E402
from entropy_horizon_recon.dark_sirens import GalaxyIndex, gather_galaxies_from_pixels, load_gladeplus_index  # noqa: E402
from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram  # noqa: E402
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


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(float(np.mean(np.exp(x - m)))))


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # Standard normal CDF using erf.
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return float("nan")
    x = x[m]
    w = w[m]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    c = np.cumsum(w)
    t = float(q) * float(c[-1])
    j = int(np.searchsorted(c, t, side="left"))
    j = min(max(j, 0), int(x.size - 1))
    return float(x[j])


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
    # Flat LCDM approximation used only for host-weight proxy ranking diagnostics.
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


def _gather_galaxies_with_global_idx(
    cat: GalaxyIndex,
    hpix: np.ndarray,
    *,
    z_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hpix = np.asarray(hpix, dtype=np.int64)
    hpix = np.unique(hpix)
    if hpix.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    offs = cat.hpix_offsets
    n = int(np.sum(offs[hpix + 1] - offs[hpix]))
    ra = np.empty(n, dtype=np.float32)
    dec = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    w = np.empty(n, dtype=np.float32)
    idx = np.empty(n, dtype=np.int64)

    pos = 0
    for p in hpix.tolist():
        a = int(offs[p])
        b = int(offs[p + 1])
        if b <= a:
            continue
        m = slice(pos, pos + (b - a))
        ra[m] = cat.ra_deg[a:b]
        dec[m] = cat.dec_deg[a:b]
        z[m] = cat.z[a:b]
        w[m] = cat.w[a:b]
        idx[m] = np.arange(a, b, dtype=np.int64)
        pos += b - a

    ra = ra[:pos]
    dec = dec[:pos]
    z = z[:pos]
    w = w[:pos]
    idx = idx[:pos]

    if z_max is not None:
        m = np.isfinite(z) & (z > 0.0) & (z <= float(z_max))
        ra, dec, z, w, idx = ra[m], dec[m], z[m], w[m], idx[m]
    return ra, dec, z, w, idx


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
    """Approximate spectral-only catalog logL vectors using a binned-z representation."""
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


@dataclass(frozen=True)
class SpecZCatalog:
    name: str
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    z: np.ndarray
    z_err: np.ndarray | None
    quality: np.ndarray | None
    source_meta: dict[str, Any]


def _as_str_array(col: Any) -> np.ndarray:
    """Convert an astropy table column / numpy array to a clean string array.

    Masked values become empty strings. Bytes are decoded as UTF-8 with replacement.
    """
    arr = np.asarray(col)
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr.tolist()):
        if v is None:
            out[i] = ""
            continue
        try:
            if np.ma.is_masked(v):
                out[i] = ""
                continue
        except Exception:
            pass
        if isinstance(v, (bytes, np.bytes_)):
            try:
                v = v.decode("utf-8", errors="replace")
            except Exception:
                v = str(v)
        out[i] = str(v).strip()
    return np.asarray(out, dtype=object)


def _expand_packed_sexagesimal(vals: np.ndarray, *, is_ra: bool) -> np.ndarray:
    """Expand packed sexagesimal strings (e.g. 123456.7) into '12 34 56.7'.

    For Dec, preserves leading sign if present.
    """
    vals = np.asarray(vals, dtype=object)
    out = vals.copy()
    for i, s in enumerate(vals.tolist()):
        if not s:
            continue
        s0 = str(s).strip()
        if not s0:
            continue
        sign = ""
        core = s0
        if not is_ra and (core.startswith("+") or core.startswith("-")):
            sign = core[0]
            core = core[1:]
        # Only consider purely numeric packed values with at least HHMMSS.
        core2 = core
        if "." in core2:
            core2 = core2.split(".", 1)[0]
        if not (core2.isdigit() and len(core2) >= 6):
            continue
        hh = core[:2]
        mm = core[2:4]
        ss = core[4:]
        out[i] = f"{sign}{hh} {mm} {ss}"
    return out


def _detect_coord_mode(ra: Any, dec: Any) -> str:
    """Heuristic coordinate-mode detection: 'deg' or 'hourangle'."""
    ra_arr = np.asarray(ra)
    dec_arr = np.asarray(dec)
    if np.issubdtype(ra_arr.dtype, np.number) and np.issubdtype(dec_arr.dtype, np.number):
        return "deg"
    ra_s = _as_str_array(ra_arr)
    # Sample up to 100 non-empty entries.
    samp = [str(x) for x in ra_s.tolist() if str(x).strip()][:100]
    if not samp:
        return "deg"
    has_sep = any((":" in s) or (" " in s) for s in samp)
    if has_sep:
        # Sexagesimal-like strings in RA are almost always hourangle in these catalogs.
        return "hourangle"
    # No obvious separators: try numeric parse.
    ok = 0
    vals = []
    for s in samp:
        try:
            vals.append(float(s))
            ok += 1
        except Exception:
            pass
    if ok >= max(5, int(0.8 * len(samp))):
        vmax = float(np.nanmax(np.asarray(vals, dtype=float))) if vals else float("nan")
        if np.isfinite(vmax) and vmax <= 24.0:
            # Could be hours; ambiguous, but hourangle is the safer default if <=24.
            return "hourangle"
        return "deg"
    # If numeric parsing fails but strings are packed, treat as hourangle.
    ra_p = _expand_packed_sexagesimal(ra_s[: min(100, ra_s.size)], is_ra=True)
    packed_hit = any(
        (str(s).strip() and str(s).strip().replace(" ", "").replace(".", "").isdigit() and len(str(s).strip().split(" ")[-1]) >= 2) for s in ra_p.tolist()
    )
    return "hourangle" if packed_hit else "deg"


def _parse_ra_dec_to_deg(
    *,
    ra: Any,
    dec: Any,
    coord_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse RA/Dec columns into degrees with robust sexagesimal handling."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ra_arr = np.asarray(ra)
    dec_arr = np.asarray(dec)
    qc: dict[str, Any] = {
        "coord_mode_config": str(coord_mode),
        "coord_mode_used": None,
        "n_rows": int(ra_arr.size),
        "n_coord_present": 0,
        "n_coord_parsed": 0,
        "n_coord_failed": 0,
    }

    # Numeric fast path.
    if coord_mode == "deg" and np.issubdtype(ra_arr.dtype, np.number) and np.issubdtype(dec_arr.dtype, np.number):
        ra_deg = np.asarray(ra_arr, dtype=float)
        dec_deg = np.asarray(dec_arr, dtype=float)
        m = np.isfinite(ra_deg) & np.isfinite(dec_deg)
        qc["coord_mode_used"] = "deg"
        qc["n_coord_present"] = int(np.count_nonzero(m))
        qc["n_coord_parsed"] = int(np.count_nonzero(m))
        qc["n_coord_failed"] = int(ra_deg.size - qc["n_coord_parsed"])
        return ra_deg, dec_deg, qc

    # String path (including sexagesimal).
    ra_s = _as_str_array(ra_arr)
    dec_s = _as_str_array(dec_arr)
    # Expand packed sexagesimal patterns to space-separated triples.
    ra_s = _expand_packed_sexagesimal(ra_s, is_ra=True)
    dec_s = _expand_packed_sexagesimal(dec_s, is_ra=False)

    present = np.asarray([(str(a).strip() != "" and str(b).strip() != "") for a, b in zip(ra_s.tolist(), dec_s.tolist(), strict=True)], dtype=bool)
    qc["n_coord_present"] = int(np.count_nonzero(present))
    ra_deg = np.full((ra_s.size,), np.nan, dtype=float)
    dec_deg = np.full((dec_s.size,), np.nan, dtype=float)
    if not np.any(present):
        qc["coord_mode_used"] = str(coord_mode)
        qc["n_coord_parsed"] = 0
        qc["n_coord_failed"] = int(ra_s.size)
        return ra_deg, dec_deg, qc

    mode = str(coord_mode)
    if mode == "auto":
        mode = _detect_coord_mode(ra_s[present], dec_s[present])
    qc["coord_mode_used"] = mode

    # Normalise separators for robust parsing.
    ra_use = np.asarray([str(x).replace(":", " ").strip() for x in ra_s[present].tolist()], dtype=object)
    dec_use = np.asarray([str(x).replace(":", " ").strip() for x in dec_s[present].tolist()], dtype=object)

    # Vector parse, then fall back to row-wise only if needed.
    try:
        if mode == "hourangle":
            c = SkyCoord(ra=ra_use, dec=dec_use, unit=(u.hourangle, u.deg), frame="icrs")
        else:
            # 'deg' mode for string columns.
            c = SkyCoord(ra=ra_use, dec=dec_use, unit=(u.deg, u.deg), frame="icrs")
        ra_deg[present] = np.asarray(c.ra.deg, dtype=float)
        dec_deg[present] = np.asarray(c.dec.deg, dtype=float)
    except Exception:
        # Slow fallback to identify parse failures without killing the ingest.
        ok = np.zeros((int(np.count_nonzero(present)),), dtype=bool)
        ra_v = np.zeros_like(ok, dtype=float)
        dec_v = np.zeros_like(ok, dtype=float)
        for i, (a, b) in enumerate(zip(ra_use.tolist(), dec_use.tolist(), strict=True)):
            try:
                if mode == "hourangle":
                    c = SkyCoord(ra=str(a), dec=str(b), unit=(u.hourangle, u.deg), frame="icrs")
                else:
                    c = SkyCoord(ra=str(a), dec=str(b), unit=(u.deg, u.deg), frame="icrs")
                ra_v[i] = float(c.ra.deg)
                dec_v[i] = float(c.dec.deg)
                ok[i] = True
            except Exception:
                ok[i] = False
        idx = np.flatnonzero(present)
        ra_deg[idx[ok]] = ra_v[ok]
        dec_deg[idx[ok]] = dec_v[ok]

    parsed = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    qc["n_coord_parsed"] = int(np.count_nonzero(parsed))
    qc["n_coord_failed"] = int(qc["n_coord_present"] - qc["n_coord_parsed"])
    if qc["n_coord_parsed"] > 0:
        qc["ra_min_deg"] = float(np.nanmin(ra_deg[parsed]))
        qc["ra_max_deg"] = float(np.nanmax(ra_deg[parsed]))
        qc["dec_min_deg"] = float(np.nanmin(dec_deg[parsed]))
        qc["dec_max_deg"] = float(np.nanmax(dec_deg[parsed]))
    return ra_deg, dec_deg, qc


def _shift_ra_dec(*, ra_deg: np.ndarray, dec_deg: np.ndarray, dra_deg: float, ddec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a simple sky shift for false-match controls (wrap RA, clamp Dec)."""
    ra = (np.asarray(ra_deg, dtype=float) + float(dra_deg)) % 360.0
    dec = np.asarray(dec_deg, dtype=float) + float(ddec_deg)
    dec = np.clip(dec, -90.0, 90.0)
    return ra, dec


def _load_or_download_specz_catalogs(
    *,
    cfg: dict[str, Any],
    offline: bool,
    out_raw: Path,
) -> tuple[list[SpecZCatalog], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load spec-z catalogs from local cache, or download via VizieR if allowed.

    Returns (catalogs, download_manifest_rows, ingest_qc_rows).
    """
    spec_cfg = cfg.get("specz_catalogs", {})
    enabled = bool(spec_cfg.get("enabled", False))
    if not enabled:
        return [], []

    cache_dir = (REPO_ROOT / str(spec_cfg.get("cache_dir", "data/cache/specz_catalogs"))).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_if_missing = bool(spec_cfg.get("download_if_missing", False))
    sources = list(spec_cfg.get("sources", []))

    catalogs: list[SpecZCatalog] = []
    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    Vizier.TIMEOUT = 120
    for s in sources:
        name = str(s.get("name") or "specz")
        stype = str(s.get("type") or "vizier")
        if stype != "vizier":
            manifest_rows.append({"name": name, "type": stype, "status": "unsupported_type"})
            continue

        vizier_id = str(s.get("vizier_id") or "")
        if not vizier_id:
            manifest_rows.append({"name": name, "type": stype, "status": "missing_vizier_id"})
            continue

        cache_path = cache_dir / f"{name.replace(' ', '_')}.fits"
        clean_path = cache_dir / f"{name.replace(' ', '_')}_clean.npz"
        tab = None
        loaded_from = None
        if clean_path.exists():
            try:
                with np.load(clean_path, allow_pickle=True) as d:
                    ra = np.asarray(d["ra_deg"], dtype=float)
                    dec = np.asarray(d["dec_deg"], dtype=float)
                    z = np.asarray(d["z"], dtype=float)
                    z_err = np.asarray(d["z_err"], dtype=float) if "z_err" in d.files else None
                    q = np.asarray(d["quality"], dtype=float) if "quality" in d.files else None
                    meta = json.loads(str(d["source_meta"].tolist())) if "source_meta" in d.files else {}
                catalogs.append(
                    SpecZCatalog(
                        name=name,
                        ra_deg=ra,
                        dec_deg=dec,
                        z=z,
                        z_err=z_err,
                        quality=q,
                        source_meta=meta,
                    )
                )
                manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "loaded_clean_cache", "n_rows": int(ra.size), "cache_path": str(clean_path)})
                qc_rows.append({"name": name, "vizier_id": vizier_id, "status": "loaded_clean_cache", "n_rows": int(ra.size), "clean_path": str(clean_path)})
                continue
            except Exception as e:
                manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "clean_cache_read_failed", "error": str(e), "clean_path": str(clean_path)})

        if cache_path.exists():
            try:
                from astropy.table import Table

                tab = Table.read(str(cache_path))
                loaded_from = str(cache_path)
            except Exception as e:
                manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "cache_read_failed", "error": str(e), "cache_path": str(cache_path)})
                tab = None

        if tab is None:
            if offline or not download_if_missing:
                manifest_rows.append(
                    {
                        "name": name,
                        "vizier_id": vizier_id,
                        "status": "missing_local_cache",
                        "cache_path": str(cache_path),
                        "note": "offline or download disabled; manual download required",
                    }
                )
                continue
            try:
                v = Vizier(row_limit=-1)
                tabs = v.get_catalogs(vizier_id)
                if not tabs:
                    raise RuntimeError("no tables returned")
                tab = tabs[0]
                loaded_from = f"vizier:{vizier_id}"
                # Cache to disk for repeatability.
                try:
                    tab.write(str(cache_path), overwrite=True)
                except Exception:
                    # Cache failures are not fatal.
                    pass
            except Exception as e:
                manifest_rows.append(
                    {
                        "name": name,
                        "vizier_id": vizier_id,
                        "status": "download_failed",
                        "error": str(e),
                        "cache_path": str(cache_path),
                    }
                )
                continue

        ra_col = str(s.get("ra_col") or "RAJ2000")
        dec_col = str(s.get("dec_col") or "DEJ2000")
        coord_mode = str(s.get("coord_mode") or "auto")
        cz_col = s.get("cz_col")
        z_col = s.get("z_col")
        cz_err_col = s.get("cz_err_col")
        q_col = s.get("quality_col")

        try:
            ra_deg, dec_deg, qc = _parse_ra_dec_to_deg(ra=tab[ra_col], dec=tab[dec_col], coord_mode=coord_mode)
        except Exception as e:
            manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "missing_ra_dec_cols", "error": str(e), "ra_col": ra_col, "dec_col": dec_col})
            continue

        z = None
        z_err = None
        if z_col:
            try:
                z = np.asarray(tab[str(z_col)], dtype=float)
            except Exception:
                z = None
        if z is None and cz_col:
            try:
                cz = np.asarray(tab[str(cz_col)], dtype=float)
                z = cz / 299792.458
            except Exception:
                z = None
        if z is None:
            manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "missing_z_cols", "cz_col": cz_col, "z_col": z_col})
            continue
        if cz_err_col:
            try:
                cz_err = np.asarray(tab[str(cz_err_col)], dtype=float)
                z_err = cz_err / 299792.458
            except Exception:
                z_err = None

        q = None
        if q_col:
            try:
                q = np.asarray(tab[str(q_col)], dtype=float)
            except Exception:
                q = None

        m = np.isfinite(ra_deg) & np.isfinite(dec_deg) & np.isfinite(z)
        ra_deg = ra_deg[m]
        dec_deg = dec_deg[m]
        z = z[m]
        if z_err is not None:
            z_err = np.asarray(z_err, dtype=float)[m]
        if q is not None:
            q = np.asarray(q, dtype=float)[m]

        qc.update(
            {
                "name": name,
                "vizier_id": vizier_id,
                "loaded_from": loaded_from,
                "cache_path": str(cache_path),
                "clean_path": str(clean_path),
                "ra_col": ra_col,
                "dec_col": dec_col,
                "coord_mode_used": str(qc.get("coord_mode_used")),
                "coord_mode_config": str(qc.get("coord_mode_config")),
                "n_rows_after_z_filter": int(ra_deg.size),
                "z_min": float(np.nanmin(z)) if z.size else float("nan"),
                "z_max": float(np.nanmax(z)) if z.size else float("nan"),
            }
        )
        qc_rows.append(qc)

        # Cache a cleaned numeric version for repeatability.
        try:
            meta = {
                "type": stype,
                "vizier_id": vizier_id,
                "loaded_from": loaded_from,
                "cache_path": str(cache_path),
                "clean_path": str(clean_path),
                "ra_col": ra_col,
                "dec_col": dec_col,
                "coord_mode": str(coord_mode),
                "cz_col": cz_col,
                "cz_err_col": cz_err_col,
                "z_col": z_col,
                "quality_col": q_col,
                "n_rows": int(ra_deg.size),
            }
            z_err_arr = np.asarray(z_err, dtype=float) if z_err is not None else np.full_like(z, np.nan, dtype=float)
            q_arr = np.asarray(q, dtype=float) if q is not None else np.full_like(z, np.nan, dtype=float)
            np.savez_compressed(clean_path, ra_deg=ra_deg, dec_deg=dec_deg, z=z, z_err=z_err_arr, quality=q_arr, source_meta=json.dumps(meta))
        except Exception:
            pass

        catalogs.append(
            SpecZCatalog(
                name=name,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                z=z,
                z_err=z_err,
                quality=q,
                source_meta={
                    "type": stype,
                    "vizier_id": vizier_id,
                    "loaded_from": loaded_from,
                    "cache_path": str(cache_path),
                    "clean_path": str(clean_path),
                    "ra_col": ra_col,
                    "dec_col": dec_col,
                    "coord_mode": str(coord_mode),
                    "cz_col": cz_col,
                    "cz_err_col": cz_err_col,
                    "z_col": z_col,
                    "quality_col": q_col,
                    "n_rows": int(ra_deg.size),
                },
            )
        )
        manifest_rows.append({"name": name, "vizier_id": vizier_id, "status": "loaded", "n_rows": int(ra_deg.size), "loaded_from": loaded_from, "cache_path": str(cache_path), "clean_path": str(clean_path)})

    _write_json(out_raw / "specz_catalog_manifest.json", {"rows": manifest_rows})
    return catalogs, manifest_rows, qc_rows


def _xmatch_query_vizier(
    *,
    cand_ra_deg: np.ndarray,
    cand_dec_deg: np.ndarray,
    cat2: str,
    radius_max_arcsec: float,
    col_ra2: str | None,
    col_dec2: str | None,
    cache_path: Path,
    offline: bool,
    chunk_size: int = 5000,
    sleep_sec: float = 0.0,
) -> dict[str, Any]:
    """Bulk crossmatch candidate coordinates against a remote catalog via CDS XMatch.

    Returns a dict with keys:
      - "status": "loaded_cache" | "queried" | "missing_cache_offline" | "query_failed"
      - "cache_path"
      - "n_rows_raw"
      - "n_chunks"
      - "error" (optional)
      - "table" (astropy Table; only if status in {"loaded_cache","queried"})
    """
    from astropy.table import Table, vstack
    from astropy import units as u
    from astroquery.xmatch import XMatch
    import time

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            tab = Table.read(str(cache_path))
            return {"status": "loaded_cache", "cache_path": str(cache_path), "n_rows_raw": int(len(tab)), "n_chunks": 0, "table": tab}
        except Exception as e:
            # Fall through to query; cache is corrupt/incompatible.
            pass

    if offline:
        return {"status": "missing_cache_offline", "cache_path": str(cache_path), "n_rows_raw": 0, "n_chunks": 0}

    ra = np.asarray(cand_ra_deg, dtype=float)
    dec = np.asarray(cand_dec_deg, dtype=float)
    n = int(ra.size)
    if n == 0:
        t0 = Table()
        return {"status": "queried", "cache_path": str(cache_path), "n_rows_raw": 0, "n_chunks": 0, "table": t0}

    chunk_size = int(max(1, chunk_size))
    out_tabs = []
    n_chunks = 0
    try:
        for a in range(0, n, chunk_size):
            b = min(n, a + chunk_size)
            t = Table()
            # cid indexes into the per-event candidate list (0..kmax-1).
            t["cid"] = np.arange(a, b, dtype=np.int64)
            t["ra"] = ra[a:b]
            t["dec"] = dec[a:b]
            res = XMatch.query(
                cat1=t,
                cat2=str(cat2),
                max_distance=float(radius_max_arcsec) * u.arcsec,
                colRA1="ra",
                colDec1="dec",
                **(
                    {}
                    if str(cat2).lower().startswith("vizier:")
                    else {"colRA2": str(col_ra2), "colDec2": str(col_dec2)}
                    if (col_ra2 and col_dec2)
                    else {}
                ),
            )
            out_tabs.append(res)
            n_chunks += 1
            if sleep_sec and (b < n):
                time.sleep(float(sleep_sec))

        tab = vstack(out_tabs, metadata_conflicts="silent") if out_tabs else Table()
        # Cache the raw match table for reproducibility/offline reruns.
        try:
            tab.write(str(cache_path), overwrite=True)
        except Exception:
            pass
        return {"status": "queried", "cache_path": str(cache_path), "n_rows_raw": int(len(tab)), "n_chunks": int(n_chunks), "table": tab}
    except Exception as e:
        return {"status": "query_failed", "cache_path": str(cache_path), "n_rows_raw": 0, "n_chunks": int(n_chunks), "error": str(e)}


def _alts_from_xmatch_table(
    *,
    tab: Any,
    n_cand: int,
    source_name: str,
    z_col: str,
    z_err_col: str | None,
    quality_col: str | None,
    require_int_col_eq: dict[str, int] | None,
    max_alternatives: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Convert an XMatch result table into per-candidate alternatives, with optional quality filtering."""
    import numpy as _np

    out: list[list[dict[str, Any]]] = [[] for _ in range(int(n_cand))]
    qc: dict[str, Any] = {
        "source": str(source_name),
        "n_rows_in": int(len(tab)) if hasattr(tab, "__len__") else None,
        "n_rows_used": 0,
        "n_rows_dropped_quality": 0,
        "n_rows_missing_cols": 0,
    }
    if tab is None or int(n_cand) == 0:
        return out, qc
    try:
        cid = _np.asarray(tab["cid"], dtype=int)
        ang = _np.asarray(tab["angDist"], dtype=float)  # arcsec
        z = _np.asarray(tab[str(z_col)], dtype=float)
    except Exception:
        qc["n_rows_missing_cols"] = int(len(tab)) if hasattr(tab, "__len__") else 0
        return out, qc

    zerr = None
    if z_err_col:
        try:
            zerr = _np.asarray(tab[str(z_err_col)], dtype=float)
        except Exception:
            zerr = None

    qv = None
    if quality_col:
        try:
            qv = _np.asarray(tab[str(quality_col)], dtype=float)
        except Exception:
            qv = None

    # Quality / warning filters.
    m = _np.isfinite(ang) & _np.isfinite(cid) & (cid >= 0) & (cid < int(n_cand)) & _np.isfinite(z)
    if require_int_col_eq:
        for k, v in require_int_col_eq.items():
            try:
                col = _np.asarray(tab[str(k)])
                # Handle masked / strings by float cast.
                col_i = _np.asarray(col, dtype=float)
                m = m & _np.isfinite(col_i) & (_np.asarray(col_i, dtype=int) == int(v))
            except Exception:
                # If the column doesn't exist, drop all rows rather than silently accept.
                m = m & False
    dropped = int(_np.count_nonzero(_np.isfinite(cid))) - int(_np.count_nonzero(m))
    qc["n_rows_dropped_quality"] = int(max(0, dropped))

    cid = cid[m]
    ang = ang[m]
    z = z[m]
    if zerr is not None:
        zerr = _np.asarray(zerr, dtype=float)[m]
    if qv is not None:
        qv = _np.asarray(qv, dtype=float)[m]

    # Group by cid; keep best few by (quality desc, sep asc).
    # If qv is absent, fall back to sep only.
    order = _np.argsort(cid, kind="mergesort")
    cid = cid[order]
    ang = ang[order]
    z = z[order]
    if zerr is not None:
        zerr = zerr[order]
    if qv is not None:
        qv = qv[order]

    for i in range(cid.size):
        c = int(cid[i])
        entry = {
            "source": str(source_name),
            "z": float(z[i]),
            "z_err": None if (zerr is None or not _np.isfinite(float(zerr[i]))) else float(zerr[i]),
            "quality": None if (qv is None or not _np.isfinite(float(qv[i]))) else float(qv[i]),
            "sep_arcsec": float(ang[i]),
        }
        out[c].append(entry)

    # Sort/truncate.
    for i in range(len(out)):
        alts = out[i]
        if not alts:
            continue

        def key(m: dict[str, Any]) -> tuple[float, float]:
            q = m.get("quality")
            q = float(q) if q is not None and _np.isfinite(float(q)) else -1.0
            sep = float(m.get("sep_arcsec", _np.inf))
            return (-q, sep)

        alts.sort(key=key)
        out[i] = alts[: int(max(1, max_alternatives))]

    qc["n_rows_used"] = int(cid.size)
    return out, qc


def _crossmatch_candidates(
    *,
    cand_ra_deg: np.ndarray,
    cand_dec_deg: np.ndarray,
    catalogs: list[SpecZCatalog],
    radius_max_arcsec: float,
    max_alternatives: int,
) -> list[list[dict[str, Any]]]:
    """Return per-candidate match alternatives within radius_max_arcsec across all catalogs."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    cand_ra_deg = np.asarray(cand_ra_deg, dtype=float)
    cand_dec_deg = np.asarray(cand_dec_deg, dtype=float)
    cand_coords = SkyCoord(ra=cand_ra_deg * u.deg, dec=cand_dec_deg * u.deg, frame="icrs")
    out: list[list[dict[str, Any]]] = [[] for _ in range(int(cand_coords.size))]
    if not catalogs or cand_coords.size == 0:
        return out

    rmax = float(radius_max_arcsec) * u.arcsec
    for cat in catalogs:
        cat_coords = SkyCoord(ra=np.asarray(cat.ra_deg, dtype=float) * u.deg, dec=np.asarray(cat.dec_deg, dtype=float) * u.deg, frame="icrs")
        # SkyCoord.search_around_sky returns (idx_other, idx_self, sep2d, sep3d).
        idx_g, idx_c, sep2d, _ = cand_coords.search_around_sky(cat_coords, seplimit=rmax)
        if len(idx_c) == 0:
            continue
        idx_c = np.asarray(idx_c, dtype=np.int64)
        idx_g = np.asarray(idx_g, dtype=np.int64)
        sep_arcsec = np.asarray(sep2d.to_value(u.arcsec), dtype=float)
        z = np.asarray(cat.z, dtype=float)[idx_g]
        if cat.quality is not None:
            q = np.asarray(cat.quality, dtype=float)[idx_g]
        else:
            q = np.zeros_like(z)
        if cat.z_err is not None:
            zerr = np.asarray(cat.z_err, dtype=float)[idx_g]
        else:
            zerr = np.full_like(z, np.nan)

        order = np.argsort(idx_c, kind="mergesort")
        idx_c = idx_c[order]
        idx_g = idx_g[order]
        sep_arcsec = sep_arcsec[order]
        z = z[order]
        q = q[order]
        zerr = zerr[order]

        # Append matches; we will sort per-candidate later.
        for ic, ig, s, zz, qq, ze in zip(idx_c.tolist(), idx_g.tolist(), sep_arcsec.tolist(), z.tolist(), q.tolist(), zerr.tolist(), strict=True):
            out[int(ic)].append(
                {
                    "source": cat.name,
                    "z": float(zz),
                    "z_err": None if not np.isfinite(float(ze)) else float(ze),
                    "quality": None if not np.isfinite(float(qq)) else float(qq),
                    "sep_arcsec": float(s),
                }
            )

    # Sort and truncate to top alternatives per candidate.
    for i in range(len(out)):
        alts = out[i]
        if not alts:
            continue

        def key(m: dict[str, Any]) -> tuple[float, float]:
            qv = m.get("quality")
            qv = float(qv) if qv is not None and np.isfinite(float(qv)) else -1.0
            sep = float(m.get("sep_arcsec", np.inf))
            return (-qv, sep)

        alts.sort(key=key)
        out[i] = alts[: int(max(1, max_alternatives))]
    return out


def _pick_best_within_radius(alts: list[dict[str, Any]], radius_arcsec: float) -> dict[str, Any] | None:
    if not alts:
        return None
    r = float(radius_arcsec)
    best = None
    for m in alts:
        s = float(m.get("sep_arcsec", np.inf))
        if not (np.isfinite(s) and s <= r):
            continue
        qv = m.get("quality")
        qv = float(qv) if qv is not None and np.isfinite(float(qv)) else -1.0
        if best is None:
            best = (qv, s, m)
            continue
        if qv > best[0] or (qv == best[0] and s < best[1]):
            best = (qv, s, m)
    return None if best is None else dict(best[2])


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-O4b non-strain smoking-gun sprint: bulk spec-z crossmatch + watertight photo-z prior + catalog-weight family.")
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--outdir", default=None, help="Optional output directory under outputs/.")
    ap.add_argument("--offline", action="store_true", help="Do not download/query external spec-z catalogs.")
    args = ap.parse_args()

    _set_thread_env(int(args.threads))

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _read_json(cfg_path)

    out_root = Path(args.outdir).expanduser().resolve() if args.outdir else (REPO_ROOT / "outputs" / f"dark_siren_smoking_gun_next_{_utc_now_compact()}")
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    raw_dir = out_root / "raw"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
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

    # Baseline spectral-only terms.
    spec_terms_npz = Path(base_cfg["spectral_only_terms_npz"]).expanduser().resolve()
    with np.load(spec_terms_npz, allow_pickle=False) as d:
        events = [str(x) for x in d["events"].tolist()]
        base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
        base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
        base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
        base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]
    ev_to_idx = {e: i for i, e in enumerate(events)}

    # Baseline score.
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
    base_delta_sel = base_delta - base_delta_data

    top_events = [str(e) for e in cfg.get("top_events", [])]
    for ev in top_events:
        if ev not in ev_to_idx:
            raise KeyError(f"Top event {ev} missing from spectral-only cache.")

    scoring_cfg = cfg.get("scoring", {})
    z_hist_nbins = int(scoring_cfg.get("z_hist_nbins", 400))
    gal_chunk_size = int(scoring_cfg.get("gal_chunk_size", 50_000))
    if str(scoring_cfg.get("distance_mode", "spectral_only")) != "spectral_only":
        raise ValueError("This sprint currently supports only distance_mode='spectral_only'.")

    gw_prior = GWDistancePrior(
        mode="dL_powerlaw",
        powerlaw_k=float(manifest.get("gw_distance_prior_power", 2.0)),
        h0_ref=float(manifest.get("gw_distance_prior_h0_ref", 67.7)),
        omega_m0=float(manifest.get("gw_distance_prior_omega_m0", 0.31)),
        omega_k0=float(manifest.get("gw_distance_prior_omega_k0", 0.0)),
        z_max=float(manifest.get("gw_distance_prior_zmax", 10.0)),
        n_grid=50_000,
    )

    # Fiducial dL(z) for host-weight proxy ranking.
    ppc_cfg = cfg.get("ppc_residuals", {})
    zmax_rank = float(cfg.get("task1_specz_override", {}).get("z_max", 0.3))
    z_grid_fid, dL_grid_fid = _build_fiducial_dL_of_z(
        h0=float(ppc_cfg.get("h0_ref", gw_prior.h0_ref)),
        omega_m0=float(ppc_cfg.get("omega_m0", gw_prior.omega_m0)),
        z_max=zmax_rank,
        n=5001,
    )

    # Helper: score with replacements to catalog terms.
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

    # Load catalogs.
    gl_cfg = cfg["glade"]
    cat_lumB = load_gladeplus_index(gl_cfg["index_lumB"])
    cat_uni = load_gladeplus_index(gl_cfg["index_uniform"])

    # ==========================
    # TASK 1: BULK SPEC-Z OVERRIDE
    # ==========================
    t1_cfg = cfg.get("task1_specz_override", {})
    specz_override_summary: dict[str, Any] = {"enabled": bool(t1_cfg.get("enabled", False))}
    specz_score_rows: list[dict[str, Any]] = []
    specz_cov_rows: list[dict[str, Any]] = []
    host_capture_rows: list[dict[str, Any]] = []

    # Per-event candidate cache.
    cand_by_event: dict[str, dict[str, Any]] = {}

    if specz_override_summary["enabled"]:
        k_tiers = [int(k) for k in t1_cfg.get("k_tiers", [200, 1000, 5000, 20000])]
        k_tiers = sorted({int(max(1, k)) for k in k_tiers})
        k_max = int(max(k_tiers))
        radii = [float(r) for r in t1_cfg.get("radii_arcsec", [3.0, 10.0, 30.0])]
        radii = sorted({float(r) for r in radii})
        radius_max = float(t1_cfg.get("radius_max_arcsec", max(radii) if radii else 30.0))
        z_max = float(t1_cfg.get("z_max", 0.3))
        mode = str(t1_cfg.get("out_of_support_mode", "drop_weight"))
        max_alts = int(t1_cfg.get("max_alternatives", 3))

        # Optional: bulk crossmatch against large remote spec-z catalogs via CDS XMatch.
        xm_cfg = cfg.get("specz_xmatch", {}) or {}
        xm_enabled = bool(xm_cfg.get("enabled", False)) and not bool(args.offline)
        xm_cache_dir = (REPO_ROOT / str(xm_cfg.get("cache_dir", "data/cache/specz_xmatch"))).resolve()
        xm_chunk = int(xm_cfg.get("chunk_size", 5000))
        xm_sleep = float(xm_cfg.get("sleep_sec", 0.0))
        xm_sources = list(xm_cfg.get("sources", [])) if xm_enabled else []
        xm_manifest_rows: list[dict[str, Any]] = []
        xm_qc_rows: list[dict[str, Any]] = []

        def _merge_alts(a: list[list[dict[str, Any]]], b: list[list[dict[str, Any]]], *, max_keep: int) -> list[list[dict[str, Any]]]:
            if not b:
                return a
            if len(a) != len(b):
                raise ValueError("alternative lists must have the same length")
            for i in range(len(a)):
                if b[i]:
                    a[i].extend(b[i])
            # Sort/truncate to preserve best options.
            for i in range(len(a)):
                alts = a[i]
                if not alts:
                    continue

                def key(m: dict[str, Any]) -> tuple[float, float]:
                    qv = m.get("quality")
                    qv = float(qv) if qv is not None and np.isfinite(float(qv)) else -1.0
                    sep = float(m.get("sep_arcsec", np.inf))
                    return (-qv, sep)

                alts.sort(key=key)
                a[i] = alts[: int(max(1, max_keep))]
            return a

        # False-match controls (shifted sky): required if any radius > 30".
        fm_cfg = t1_cfg.get("false_match_control", {}) or {}
        fm_enabled = bool(fm_cfg.get("enabled", any(float(r) > 30.0 for r in radii)))
        shift_ra_deg = float(fm_cfg.get("shift_ra_deg", 0.5))
        shift_dec_deg = float(fm_cfg.get("shift_dec_deg", 0.5))
        shift_ratio_max = float(fm_cfg.get("shift_ratio_max", 0.1))

        # Build candidates for each top event.
        for ev in top_events:
            ev_npz = gap_root / "cache" / f"event_{ev}.npz"
            z_cat, w_cat, ipix_cat, hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)
            ra, dec, z2, w2, idx_global = _gather_galaxies_with_global_idx(cat_lumB, hpix_sel, z_max=z_max)
            if z2.size != z_cat.size or w2.size != w_cat.size:
                raise RuntimeError(f"{ev}: mismatch between cache galaxy arrays and gathered lumB index arrays.")
            # Spot-check agreement on z/w.
            rng = np.random.default_rng(0)
            check_n = min(2048, int(z_cat.size))
            ii = rng.integers(0, int(z_cat.size), size=check_n, endpoint=False)
            if not (np.allclose(z2[ii], z_cat[ii], rtol=0.0, atol=0.0) and np.allclose(w2[ii], w_cat[ii], rtol=0.0, atol=0.0)):
                raise RuntimeError(f"{ev}: cache arrays do not match lumB index order; cannot attach coordinates reliably.")

            edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
            widths = np.diff(edges)

            # Sky probability per galaxy.
            npix = int(hp.nside2npix(int(pe.nside)))
            pix_to_row = np.full((npix,), -1, dtype=np.int32)
            pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
            row = pix_to_row[ipix_cat]
            good = (row >= 0) & np.isfinite(z_cat) & (z_cat > 0.0) & np.isfinite(w_cat) & (w_cat > 0.0)
            prob = np.zeros_like(z_cat, dtype=float)
            prob[good] = np.asarray(pe.prob_pix, dtype=float)[row[good]]

            # Host-weight proxy for ranking: w * prob * pdf_1d(dL(z)) / pi(dL(z)).
            dL_em = np.interp(np.clip(z_cat, 0.0, z_grid_fid[-1]), z_grid_fid, dL_grid_fid)
            bin_idx = np.searchsorted(edges, dL_em, side="right") - 1
            valid = good & (bin_idx >= 0) & (bin_idx < widths.size) & np.isfinite(dL_em) & (dL_em > 0.0)
            pdf = np.zeros_like(dL_em, dtype=float)
            pdf[valid] = pdf_1d[bin_idx[valid]]
            inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em, 1e-6, np.inf)))
            weight = np.zeros_like(z_cat, dtype=float)
            weight[valid] = w_cat[valid] * prob[valid] * pdf[valid] * inv_pi[valid]

            tot_w = float(np.sum(weight))
            if not (np.isfinite(tot_w) and tot_w > 0.0):
                raise RuntimeError(f"{ev}: invalid host-weight proxy normalization.")

            k_use = min(int(k_max), int(weight.size))
            if k_use <= 0:
                raise RuntimeError(f"{ev}: no galaxies available for ranking.")
            if k_use == int(weight.size):
                idx_top = np.argsort(weight)[::-1].astype(np.int64, copy=False)
            else:
                idx_top = np.argpartition(weight, -k_use)[-k_use:]
                idx_top = idx_top[np.argsort(weight[idx_top])[::-1]].astype(np.int64, copy=False)

            w_sorted = weight[idx_top]
            cw = np.cumsum(w_sorted)
            cw_frac = cw / tot_w

            # Export full top-Kmax candidate set.
            cand_rows = []
            for rank, gi in enumerate(idx_top.tolist(), start=1):
                cand_rows.append(
                    {
                        "event": ev,
                        "rank": int(rank),
                        "gal_index_in_event": int(gi),
                        "catalog_id": int(idx_global[gi]),
                        "ra_deg": float(ra[gi]),
                        "dec_deg": float(dec[gi]),
                        "catalog_z": float(z_cat[gi]),
                        "catalog_z_err": float("nan"),
                        "weight_proxy": float(weight[gi]),
                        "cumulative_weight_proxy": float(cw[rank - 1]),
                        "cumulative_weight_frac": float(cw_frac[rank - 1]),
                    }
                )
            _write_csv(
                tab_dir / f"host_candidates_{ev}.csv",
                cand_rows,
                fieldnames=[
                    "event",
                    "rank",
                    "gal_index_in_event",
                    "catalog_id",
                    "ra_deg",
                    "dec_deg",
                    "catalog_z",
                    "catalog_z_err",
                    "weight_proxy",
                    "cumulative_weight_proxy",
                    "cumulative_weight_frac",
                ],
            )

            for k in k_tiers:
                kk = min(int(k), int(idx_top.size))
                host_capture_rows.append(
                    {
                        "event": ev,
                        "k": int(k),
                        "k_used": int(kk),
                        "cum_weight_frac_at_k": float(cw_frac[kk - 1]) if kk > 0 else 0.0,
                        "n_gal_total": int(weight.size),
                        "total_weight_proxy": float(tot_w),
                    }
                )

            cand_by_event[ev] = {
                "idx_top": idx_top,
                "weight_proxy": weight,
                "prob": prob,
                "z_cat": z_cat,
                "w_cat": w_cat,
                "ipix": ipix_cat,
                "pe": pe,
                "ra_top": ra[idx_top],
                "dec_top": dec[idx_top],
                "weight_top": w_sorted,
                "cum_frac_top": cw_frac,
                "z_max": float(z_max),
            }

        _write_csv(tab_dir / "host_candidates_capture_summary.csv", host_capture_rows, fieldnames=list(host_capture_rows[0].keys()) if host_capture_rows else [])

        # Load/download spec-z catalogs.
        catalogs, specz_manifest_rows, specz_qc_rows = _load_or_download_specz_catalogs(cfg=cfg, offline=bool(args.offline), out_raw=raw_dir)
        specz_override_summary["specz_catalog_manifest_rows"] = specz_manifest_rows
        specz_override_summary["n_specz_catalogs_loaded"] = int(len(catalogs))
        # Write ingest QC for auditability.
        _write_json(tab_dir / "specz_ingest_qc.json", {"rows": specz_qc_rows})
        if xm_enabled:
            (raw_dir / "specz_xmatch").mkdir(parents=True, exist_ok=True)

        # Perform crossmatch and rescore for each (radius, K).
        z_edges = np.linspace(0.0, z_max, z_hist_nbins + 1)
        z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

        per_event_match_rows: list[dict[str, Any]] = []
        match_cache: dict[str, Any] = {}
        match_cache_shift: dict[str, dict[str, Any]] = {}
        for ev, data in cand_by_event.items():
            idx_top = np.asarray(data["idx_top"], dtype=np.int64)
            ra_top = np.asarray(data["ra_top"], dtype=float)
            dec_top = np.asarray(data["dec_top"], dtype=float)
            alts = _crossmatch_candidates(cand_ra_deg=ra_top, cand_dec_deg=dec_top, catalogs=catalogs, radius_max_arcsec=radius_max, max_alternatives=max_alts)

            # Remote XMatch augmentation (SDSS/DESI etc): add match alternatives without downloading full catalogs.
            if xm_sources:
                for src in xm_sources:
                    sname = str(src.get("name") or "xmatch")
                    cat2 = str(src.get("cat2") or "")
                    if not cat2:
                        xm_manifest_rows.append({"event": ev, "source": sname, "status": "missing_cat2"})
                        continue
                    # Avoid repeated failing queries for tables not mirrored on the XMatch server.
                    try:
                        from astroquery.xmatch import XMatch

                        if not bool(XMatch.is_table_available(cat2)):
                            xm_manifest_rows.append({"event": ev, "source": sname, "cat2": cat2, "status": "table_unavailable"})
                            continue
                    except Exception:
                        pass
                    src_rmax = float(src.get("radius_max_arcsec", radius_max))
                    src_rmax = float(min(float(radius_max), max(0.1, src_rmax)))
                    col_ra2 = str(src.get("ra2_col") or src.get("col_ra2") or "RAJ2000")
                    col_dec2 = str(src.get("dec2_col") or src.get("col_dec2") or "DEJ2000")
                    z_col = str(src.get("z_col") or "z")
                    z_err_col = src.get("z_err_col")
                    q_col = src.get("quality_col")
                    req = src.get("require_int_col_eq") or {}
                    req = {str(k): int(v) for k, v in dict(req).items()}
                    cache_path = xm_cache_dir / f"{sname}_{ev}_true_rmax{int(round(src_rmax))}_k{int(k_max)}.fits"
                    q = _xmatch_query_vizier(
                        cand_ra_deg=ra_top,
                        cand_dec_deg=dec_top,
                        cat2=cat2,
                        radius_max_arcsec=src_rmax,
                        col_ra2=col_ra2,
                        col_dec2=col_dec2,
                        cache_path=cache_path,
                        offline=bool(args.offline),
                        chunk_size=xm_chunk,
                        sleep_sec=xm_sleep,
                    )
                    xm_manifest_rows.append(
                        {
                            "event": ev,
                            "source": sname,
                            "cat2": cat2,
                            "radius_max_arcsec": float(src_rmax),
                            "status": q.get("status"),
                            "cache_path": q.get("cache_path"),
                            "n_rows_raw": int(q.get("n_rows_raw", 0)),
                            "n_chunks": int(q.get("n_chunks", 0)),
                            "error": q.get("error", ""),
                        }
                    )
                    if q.get("status") not in {"loaded_cache", "queried"}:
                        continue
                    alts_x, qc = _alts_from_xmatch_table(
                        tab=q.get("table"),
                        n_cand=int(ra_top.size),
                        source_name=sname,
                        z_col=z_col,
                        z_err_col=str(z_err_col) if z_err_col else None,
                        quality_col=str(q_col) if q_col else None,
                        require_int_col_eq=req if req else None,
                        max_alternatives=max_alts,
                    )
                    qc.update({"event": ev, "source": sname, "cat2": cat2, "status": q.get("status")})
                    xm_qc_rows.append(qc)
                    alts = _merge_alts(alts, alts_x, max_keep=max_alts)

            match_cache[ev] = {"alternatives": alts}

            # Shifted-sky false-match controls (optional but required if radius>30").
            if fm_enabled and float(radius_max) > 0.0:
                shift_kinds = {
                    "shift_ra": (shift_ra_deg, 0.0),
                    "shift_dec": (0.0, shift_dec_deg),
                }
                for sk, (dra, ddec) in shift_kinds.items():
                    ra_s, dec_s = _shift_ra_dec(ra_deg=ra_top, dec_deg=dec_top, dra_deg=dra, ddec_deg=ddec)
                    alts_s = _crossmatch_candidates(cand_ra_deg=ra_s, cand_dec_deg=dec_s, catalogs=catalogs, radius_max_arcsec=radius_max, max_alternatives=max_alts)
                    if xm_sources:
                        for src in xm_sources:
                            sname = str(src.get("name") or "xmatch")
                            cat2 = str(src.get("cat2") or "")
                            if not cat2:
                                continue
                            try:
                                from astroquery.xmatch import XMatch

                                if not bool(XMatch.is_table_available(cat2)):
                                    xm_manifest_rows.append({"event": ev, "source": sname, "cat2": cat2, "shift_kind": sk, "status": "table_unavailable"})
                                    continue
                            except Exception:
                                pass
                            src_rmax = float(src.get("radius_max_arcsec", radius_max))
                            src_rmax = float(min(float(radius_max), max(0.1, src_rmax)))
                            col_ra2 = str(src.get("ra2_col") or src.get("col_ra2") or "RAJ2000")
                            col_dec2 = str(src.get("dec2_col") or src.get("col_dec2") or "DEJ2000")
                            z_col = str(src.get("z_col") or "z")
                            z_err_col = src.get("z_err_col")
                            q_col = src.get("quality_col")
                            req = src.get("require_int_col_eq") or {}
                            req = {str(k): int(v) for k, v in dict(req).items()}
                            cache_path = xm_cache_dir / f"{sname}_{ev}_{sk}_rmax{int(round(src_rmax))}_k{int(k_max)}.fits"
                            q = _xmatch_query_vizier(
                                cand_ra_deg=ra_s,
                                cand_dec_deg=dec_s,
                                cat2=cat2,
                                radius_max_arcsec=src_rmax,
                                col_ra2=col_ra2,
                                col_dec2=col_dec2,
                                cache_path=cache_path,
                                offline=bool(args.offline),
                                chunk_size=xm_chunk,
                                sleep_sec=xm_sleep,
                            )
                            xm_manifest_rows.append(
                                {
                                    "event": ev,
                                    "source": sname,
                                    "cat2": cat2,
                                    "radius_max_arcsec": float(src_rmax),
                                    "shift_kind": sk,
                                    "status": q.get("status"),
                                    "cache_path": q.get("cache_path"),
                                    "n_rows_raw": int(q.get("n_rows_raw", 0)),
                                    "n_chunks": int(q.get("n_chunks", 0)),
                                    "error": q.get("error", ""),
                                }
                            )
                            if q.get("status") not in {"loaded_cache", "queried"}:
                                continue
                            alts_x, qc = _alts_from_xmatch_table(
                                tab=q.get("table"),
                                n_cand=int(ra_s.size),
                                source_name=sname,
                                z_col=z_col,
                                z_err_col=str(z_err_col) if z_err_col else None,
                                quality_col=str(q_col) if q_col else None,
                                require_int_col_eq=req if req else None,
                                max_alternatives=max_alts,
                            )
                            qc.update({"event": ev, "source": sname, "cat2": cat2, "status": q.get("status"), "shift_kind": sk})
                            xm_qc_rows.append(qc)
                            alts_s = _merge_alts(alts_s, alts_x, max_keep=max_alts)
                    match_cache_shift.setdefault(sk, {})[ev] = {"alternatives": alts_s}

            # Emit per-candidate best matches (by radius) for audit.
            for i, gi in enumerate(idx_top.tolist(), start=1):
                row = {
                    "event": ev,
                    "rank": int(i),
                    "gal_index_in_event": int(gi),
                    "ra_deg": float(ra_top[i - 1]),
                    "dec_deg": float(dec_top[i - 1]),
                }
                row["n_alternatives_maxr"] = int(len(alts[i - 1]))
                for r in radii:
                    best = _pick_best_within_radius(alts[i - 1], radius_arcsec=r)
                    row[f"best_z_r{int(r)}"] = "" if best is None else float(best["z"])
                    row[f"best_source_r{int(r)}"] = "" if best is None else str(best["source"])
                    row[f"best_sep_arcsec_r{int(r)}"] = "" if best is None else float(best["sep_arcsec"])
                    row[f"best_quality_r{int(r)}"] = "" if best is None else ("" if best.get("quality") is None else float(best["quality"]))
                per_event_match_rows.append(row)

        if per_event_match_rows:
            fieldnames = list(per_event_match_rows[0].keys())
            _write_csv(tab_dir / "specz_candidate_matches.csv", per_event_match_rows, fieldnames=fieldnames)
        _write_json(raw_dir / "specz_candidate_match_alternatives.json", match_cache)
        if match_cache_shift:
            _write_json(raw_dir / "specz_candidate_match_alternatives_shifted.json", match_cache_shift)
        if xm_manifest_rows:
            _write_json(raw_dir / "specz_xmatch_manifest.json", {"rows": xm_manifest_rows})
        if xm_qc_rows:
            _write_csv(tab_dir / "specz_match_quality_summary.csv", xm_qc_rows, fieldnames=list(xm_qc_rows[0].keys()))

        # Precompute baseline per-event z histograms (w*prob weights).
        base_hist_by_event: dict[str, np.ndarray] = {}
        wprob_by_event: dict[str, np.ndarray] = {}
        for ev, data in cand_by_event.items():
            z_cat = np.asarray(data["z_cat"], dtype=float)
            w_cat = np.asarray(data["w_cat"], dtype=float)
            prob = np.asarray(data["prob"], dtype=float)
            wprob = np.asarray(w_cat, dtype=float) * np.asarray(prob, dtype=float)
            good = np.isfinite(z_cat) & (z_cat > 0.0) & (z_cat <= z_max) & np.isfinite(wprob) & (wprob > 0.0)
            hist, _ = np.histogram(np.clip(z_cat[good], 0.0, z_max), bins=z_edges, weights=wprob[good])
            base_hist_by_event[ev] = np.asarray(hist, dtype=float)
            wprob_by_event[ev] = wprob

        # Coverage bookkeeping for true and shifted-sky controls (no re-scoring).
        def _append_cov_rows(*, control: str, per_ev: dict[str, Any]) -> None:
            for r in radii:
                for k in k_tiers:
                    for ev, data in cand_by_event.items():
                        idx_top = np.asarray(data["idx_top"], dtype=np.int64)
                        alts = per_ev.get(ev, {}).get("alternatives") if per_ev else None
                        if not alts:
                            continue
                        z_cat = np.asarray(data["z_cat"], dtype=float)
                        wprob = np.asarray(wprob_by_event[ev], dtype=float)
                        kk = min(int(k), int(idx_top.size))
                        use = idx_top[:kk]
                        wt_total = float(np.sum(np.asarray(data["weight_proxy"], dtype=float)[use]))
                        wt_over = 0.0
                        n_match = 0
                        for j, gi in enumerate(use.tolist()):
                            best = _pick_best_within_radius(alts[j], radius_arcsec=r)
                            if best is None:
                                continue
                            z_spec = float(best["z"])
                            if not np.isfinite(z_spec):
                                continue
                            wprob_gi = float(wprob[int(gi)])
                            if not (np.isfinite(wprob_gi) and wprob_gi > 0.0):
                                continue
                            wt_over += float(np.asarray(data["weight_proxy"], dtype=float)[int(gi)])
                            n_match += 1
                        wt_all = float(np.sum(np.asarray(data["weight_proxy"], dtype=float)))
                        specz_cov_rows.append(
                            {
                                "event": ev,
                                "radius_arcsec": float(r),
                                "k": int(k),
                                "k_used": int(kk),
                                "n_matched": int(n_match),
                                "frac_gal_matched": float(n_match) / float(max(1, kk)),
                                "weight_matched": float(wt_over),
                                "weight_topk": float(wt_total),
                                "weight_total": float(wt_all),
                                "frac_weight_matched_topk": float(wt_over / wt_total) if wt_total > 0 else float("nan"),
                                "frac_weight_matched_total": float(wt_over / wt_all) if wt_all > 0 else float("nan"),
                                "control": str(control),
                            }
                        )

        # True-sky coverage.
        _append_cov_rows(control="true", per_ev=match_cache)
        # Shifted-sky coverage (false-match controls).
        if fm_enabled and match_cache_shift:
            for sk, per_ev in match_cache_shift.items():
                _append_cov_rows(control=str(sk), per_ev=per_ev)

        # Apply overrides and score.
        # If we requested radii > 30", evaluate false-match controls at K=max(k_tiers)
        # and skip scoring radii that look dominated by random coincidences.
        radii_to_score = list(radii)
        if fm_enabled and match_cache_shift and any(float(rr) > 30.0 for rr in radii):
            try:
                k_gate = int(max(k_tiers))
                # Compare true vs shifted (average of shift_ra/shift_dec) at this K.
                ratios = {}
                for rr in radii:
                    if float(rr) <= 30.0:
                        continue
                    f_true = []
                    f_shift = []
                    for ev in top_events:
                        rows_true = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "true" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        rows_sr = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "shift_ra" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        rows_sd = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "shift_dec" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        if rows_true:
                            f_true.append(float(rows_true[0].get("frac_weight_matched_total", 0.0)))
                        if rows_sr and rows_sd:
                            f_shift.append(0.5 * (float(rows_sr[0].get("frac_weight_matched_total", 0.0)) + float(rows_sd[0].get("frac_weight_matched_total", 0.0))))
                    if f_true and f_shift:
                        ft = float(np.nanmedian(np.asarray(f_true, dtype=float)))
                        fs = float(np.nanmedian(np.asarray(f_shift, dtype=float)))
                        ratios[float(rr)] = (ft, fs, float(fs / ft) if ft > 0 else float("inf"))
                bad = [rr for rr, (_, _, rat) in ratios.items() if np.isfinite(rat) and rat > shift_ratio_max]
                if bad:
                    radii_to_score = [rr for rr in radii_to_score if float(rr) not in set(bad)]
                    specz_override_summary["false_match_control_skipped_radii"] = [{"radius_arcsec": float(rr), "shift_ratio_median": float(ratios[float(rr)][2])} for rr in bad]
            except Exception:
                pass

        specz_override_summary["radii_requested_arcsec"] = [float(x) for x in radii]
        specz_override_summary["radii_scored_arcsec"] = [float(x) for x in radii_to_score]

        # Score only the radii that pass false-match control (if enabled).
        for r in radii_to_score:
            for k in k_tiers:
                # Override maps used for scoring (audit/reproducibility).
                override_maps: dict[str, list[dict[str, Any]]] = {}
                repl = {}
                for ev, data in cand_by_event.items():
                    pe = data["pe"]
                    idx_top = np.asarray(data["idx_top"], dtype=np.int64)
                    alts = match_cache[ev]["alternatives"]
                    z_cat = np.asarray(data["z_cat"], dtype=float)
                    wprob = np.asarray(wprob_by_event[ev], dtype=float)
                    hist = np.asarray(base_hist_by_event[ev], dtype=float).copy()

                    kk = min(int(k), int(idx_top.size))
                    use = idx_top[:kk]
                    # Coverage bookkeeping.
                    wt_total = float(np.sum(np.asarray(data["weight_proxy"], dtype=float)[use]))
                    wt_over = 0.0
                    n_match = 0
                    applied: list[dict[str, Any]] = []

                    for j, gi in enumerate(use.tolist()):
                        best = _pick_best_within_radius(alts[j], radius_arcsec=r)
                        if best is None:
                            continue
                        z_spec = float(best["z"])
                        if not np.isfinite(z_spec):
                            continue
                        wprob_gi = float(wprob[int(gi)])
                        if not (np.isfinite(wprob_gi) and wprob_gi > 0.0):
                            continue
                        # Remove from old bin.
                        b_old = int(np.searchsorted(z_edges, float(z_cat[int(gi)]), side="right") - 1)
                        if 0 <= b_old < hist.size:
                            hist[b_old] = max(0.0, float(hist[b_old] - wprob_gi))
                        # Add to new bin (or drop).
                        if z_spec <= 0.0 or z_spec > z_max:
                            if mode != "drop_weight":
                                z_spec = float(np.clip(z_spec, 1e-6, z_max))
                                b_new = int(np.searchsorted(z_edges, z_spec, side="right") - 1)
                                if 0 <= b_new < hist.size:
                                    hist[b_new] = float(hist[b_new] + wprob_gi)
                        else:
                            b_new = int(np.searchsorted(z_edges, z_spec, side="right") - 1)
                            if 0 <= b_new < hist.size:
                                hist[b_new] = float(hist[b_new] + wprob_gi)
                        wt_over += float(np.asarray(data["weight_proxy"], dtype=float)[int(gi)])
                        n_match += 1
                        applied.append(
                            {
                                "gal_index_in_event": int(gi),
                                "z_spec": float(z_spec),
                                "source": str(best.get("source", "")),
                                "sep_arcsec": float(best.get("sep_arcsec", float("nan"))),
                            }
                        )

                    # Recompute logL with binned-z approximation.
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
                    override_maps[ev] = applied

                res = _score_with_cat_replacements(repl)
                dlp_total = float(res.lpd_mu_total - res.lpd_gr_total)
                dlp_data = float(res.lpd_mu_total_data - res.lpd_gr_total_data)
                # Add a quick coverage summary for this condition (median across top events).
                cov_rows = [x for x in specz_cov_rows if x.get("control") == "true" and float(x.get("radius_arcsec")) == float(r) and int(x.get("k")) == int(k)]
                fwt = [float(x.get("frac_weight_matched_total", float("nan"))) for x in cov_rows]
                cov_median = float(np.nanmedian(np.asarray(fwt, dtype=float))) if fwt else float("nan")
                specz_score_rows.append(
                    {
                        "radius_arcsec": float(r),
                        "k": int(k),
                        "delta_lpd_total": float(dlp_total),
                        "delta_lpd_data": float(dlp_data),
                        "median_frac_weight_matched_total": cov_median,
                    }
                )

                # Save the applied override map for this condition.
                try:
                    out_map = {
                        "radius_arcsec": float(r),
                        "k": int(k),
                        "top_events": list(top_events),
                        "notes": "Applied spec-z overrides used in the binned-z scoring for this condition (only entries with finite z_spec and nonzero w*prob mass).",
                        "overrides_by_event": override_maps,
                    }
                    p = raw_dir / "specz_override_maps" / f"specz_overrides_r{int(round(float(r)))}_k{int(k)}.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(out_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                except Exception:
                    pass

        if specz_score_rows:
            _write_csv(tab_dir / "specz_override_score_rows.csv", specz_score_rows, fieldnames=list(specz_score_rows[0].keys()))
        if specz_cov_rows:
            _write_csv(tab_dir / "specz_coverage_summary.csv", specz_cov_rows, fieldnames=list(specz_cov_rows[0].keys()))

        # Coverage figure: fraction of total host-weight proxy covered, vs K (per event; radius=30").
        try:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for ev in top_events:
                rows = [r for r in specz_cov_rows if r.get("control", "true") == "true" and r["event"] == ev and float(r["radius_arcsec"]) == float(max(radii))]
                rows.sort(key=lambda x: int(x["k"]))
                if not rows:
                    continue
                ks = np.asarray([int(r["k_used"]) for r in rows], dtype=float)
                fw = np.asarray([float(r["frac_weight_matched_total"]) for r in rows], dtype=float)
                ax.plot(ks, fw, marker="o", lw=1.6, label=ev)
            ax.set_xscale("log")
            ax.set(xlabel="Top-K host candidates (by weight proxy)", ylabel="Matched spec-z weight fraction (of total)", title="Spec-z crossmatch coverage (radius=max)")
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "specz_weight_coverage_byK.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        # Coverage vs radius: true vs shifted controls at K=max(k_tiers).
        if fm_enabled and match_cache_shift:
            try:
                k_gate = int(max(k_tiers))
                fig, ax = plt.subplots(figsize=(6.8, 4.2))
                for ev in top_events:
                    y_true = []
                    y_sr = []
                    y_sd = []
                    for rr in radii:
                        rt = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "true" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        rsr = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "shift_ra" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        rsd = [x for x in specz_cov_rows if x.get("event") == ev and x.get("control") == "shift_dec" and float(x.get("radius_arcsec")) == float(rr) and int(x.get("k")) == k_gate]
                        y_true.append(float(rt[0].get("frac_weight_matched_total", 0.0)) if rt else float("nan"))
                        y_sr.append(float(rsr[0].get("frac_weight_matched_total", 0.0)) if rsr else float("nan"))
                        y_sd.append(float(rsd[0].get("frac_weight_matched_total", 0.0)) if rsd else float("nan"))
                    ax.plot(radii, y_true, marker="o", lw=1.6, label=f"{ev} true")
                    ax.plot(radii, y_sr, marker=".", lw=1.0, alpha=0.6, linestyle="--", label=f"{ev} shift_ra")
                    ax.plot(radii, y_sd, marker=".", lw=1.0, alpha=0.6, linestyle=":", label=f"{ev} shift_dec")
                ax.set(xlabel="Match radius (arcsec)", ylabel="Matched spec-z weight fraction (of total)", title=f"False-match control (K={k_gate})")
                ax.grid(alpha=0.25, linestyle=":")
                ax.legend(loc="best", fontsize=6, frameon=False, ncol=2)
                fig.tight_layout()
                fig.savefig(fig_dir / "specz_coverage_vs_radius_true_vs_shifted.png", dpi=180)
                plt.close(fig)
            except Exception:
                pass

        # ΔLPD vs coverage scatter.
        try:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            x = np.asarray([float(r.get("median_frac_weight_matched_total", np.nan)) for r in specz_score_rows], dtype=float)
            y = np.asarray([float(r.get("delta_lpd_total", np.nan)) for r in specz_score_rows], dtype=float)
            ax.scatter(x, y, s=40, alpha=0.85)
            for r in specz_score_rows:
                ax.annotate(f"r={int(round(float(r['radius_arcsec'])))} K={int(r['k'])}", (float(r.get("median_frac_weight_matched_total", np.nan)), float(r.get("delta_lpd_total", np.nan))), fontsize=7, alpha=0.8)
            ax.axhline(base_delta, color="C3", lw=1.8, label="baseline")
            ax.set(xlabel="Median matched weight fraction (total; top events)", ylabel="ΔLPD_total", title="Spec-z override: ΔLPD vs coverage")
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "specz_override_deltalpd_vs_coverage.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    # ==========================
    # TASK 2: PHOTO-Z PRIOR (NO INTERPOLATION MIRAGE)
    # ==========================
    t2_cfg = cfg.get("task2_photoz_prior", {})
    photoz_summary: dict[str, Any] = {"enabled": bool(t2_cfg.get("enabled", False))}
    if photoz_summary["enabled"]:
        grid_path = Path(base_cfg["photoz_grid_csv"]).expanduser().resolve()
        rows_grid = []
        for r in csv.DictReader(grid_path.read_text(encoding="utf-8").splitlines()):
            rows_grid.append({"b0": float(r["b0"]), "b1": float(r["b1"]), "delta": float(r["delta_lpd_total"])})
        b0_u = sorted({float(r["b0"]) for r in rows_grid})
        b1_u = sorted({float(r["b1"]) for r in rows_grid})
        surf = np.full((len(b0_u), len(b1_u)), np.nan, dtype=float)
        lookup = {(float(r["b0"]), float(r["b1"])): float(r["delta"]) for r in rows_grid}
        for i, b0 in enumerate(b0_u):
            for j, b1 in enumerate(b1_u):
                surf[i, j] = float(lookup[(float(b0), float(b1))])

        b0_sig = float(t2_cfg.get("prior_b0_sigma", 0.01))
        b1_sig = float(t2_cfg.get("prior_b1_sigma", 0.1))
        n_samp = int(t2_cfg.get("n_samples", 200000))
        seed = int(t2_cfg.get("seed", 123))

        # Cell edges for grid quadrature.
        def edges_from_centres(x: list[float]) -> np.ndarray:
            x = [float(v) for v in x]
            if len(x) < 2:
                raise ValueError("need at least 2 grid points to define cell edges")
            x = sorted(x)
            e = np.empty((len(x) + 1,), dtype=float)
            e[1:-1] = 0.5 * (np.asarray(x[:-1]) + np.asarray(x[1:]))
            e[0] = float(x[0]) - 0.5 * float(x[1] - x[0])
            e[-1] = float(x[-1]) + 0.5 * float(x[-1] - x[-2])
            return e

        b0_edges = edges_from_centres(b0_u)
        b1_edges = edges_from_centres(b1_u)
        mass_b0 = float(_norm_cdf(np.asarray([b0_edges[-1] / b0_sig])) - _norm_cdf(np.asarray([b0_edges[0] / b0_sig])))
        mass_b1 = float(_norm_cdf(np.asarray([b1_edges[-1] / b1_sig])) - _norm_cdf(np.asarray([b1_edges[0] / b1_sig])))
        mass_in_grid = float(mass_b0 * mass_b1)
        clip_fraction = float(max(0.0, 1.0 - mass_in_grid))

        # Estimator A: grid-cell quadrature (no interpolation; conditional on in-grid mass).
        cell_w = np.zeros_like(surf, dtype=float)
        for i in range(len(b0_u)):
            lo = float(b0_edges[i])
            hi = float(b0_edges[i + 1])
            wx = float(_norm_cdf(np.asarray([hi / b0_sig])) - _norm_cdf(np.asarray([lo / b0_sig])))
            for j in range(len(b1_u)):
                lo2 = float(b1_edges[j])
                hi2 = float(b1_edges[j + 1])
                wy = float(_norm_cdf(np.asarray([hi2 / b1_sig])) - _norm_cdf(np.asarray([lo2 / b1_sig])))
                cell_w[i, j] = wx * wy

        xA = surf.reshape(-1)
        wA = cell_w.reshape(-1)
        mA = np.isfinite(xA) & np.isfinite(wA) & (wA > 0.0)
        xA = xA[mA]
        wA = wA[mA]
        wA_sum = float(np.sum(wA))
        if wA_sum <= 0.0:
            raise RuntimeError("photo-z quadrature weights invalid")
        wA = wA / wA_sum
        estA = {
            "estimator": "grid_quadrature",
            "mean": float(np.sum(wA * xA)),
            "median": _weighted_quantile(xA, wA, 0.5),
            "p16": _weighted_quantile(xA, wA, 0.16),
            "p84": _weighted_quantile(xA, wA, 0.84),
            "p_lt_1": float(np.sum(wA * (xA < 1.0))),
            "p_lt_0": float(np.sum(wA * (xA < 0.0))),
            "clip_fraction": float(clip_fraction),
            "notes": "Conditional on in-grid prior mass; no interpolation; no clipping to edges.",
        }

        # Estimator B: nearest-neighbour Monte Carlo, rejecting out-of-grid samples (no clipping).
        rng = np.random.default_rng(seed)
        b0_s = rng.normal(0.0, b0_sig, size=n_samp)
        b1_s = rng.normal(0.0, b1_sig, size=n_samp)
        in_grid = (b0_s >= b0_edges[0]) & (b0_s <= b0_edges[-1]) & (b1_s >= b1_edges[0]) & (b1_s <= b1_edges[-1])
        clip_frac_mc = float(1.0 - float(np.mean(in_grid)))
        b0_s = b0_s[in_grid]
        b1_s = b1_s[in_grid]
        # Map to nearest grid point in each dimension.
        b0_u_arr = np.asarray(b0_u, dtype=float)
        b1_u_arr = np.asarray(b1_u, dtype=float)

        def nearest_idx(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
            # grid sorted; choose between neighbours.
            j = np.searchsorted(grid, x, side="left")
            j = np.clip(j, 0, grid.size - 1)
            j0 = np.clip(j - 1, 0, grid.size - 1)
            j1 = j
            d0 = np.abs(x - grid[j0])
            d1 = np.abs(x - grid[j1])
            return np.where(d1 < d0, j1, j0).astype(np.int64, copy=False)

        i0 = nearest_idx(b0_s, b0_u_arr)
        i1 = nearest_idx(b1_s, b1_u_arr)
        dB = surf[i0, i1]
        dB = np.asarray(dB, dtype=float)
        dB = dB[np.isfinite(dB)]
        estB = {
            "estimator": "nearest_neighbour_mc",
            "n_samples": int(dB.size),
            "mean": float(np.mean(dB)) if dB.size else float("nan"),
            "median": float(np.quantile(dB, 0.5)) if dB.size else float("nan"),
            "p16": float(np.quantile(dB, 0.16)) if dB.size else float("nan"),
            "p84": float(np.quantile(dB, 0.84)) if dB.size else float("nan"),
            "p_lt_1": float(np.mean(dB < 1.0)) if dB.size else float("nan"),
            "p_lt_0": float(np.mean(dB < 0.0)) if dB.size else float("nan"),
            "clip_fraction": float(clip_frac_mc),
            "notes": "Rejects out-of-grid prior draws; snaps remaining samples to nearest grid point.",
        }

        # Optional comparison: previous-style bilinear interpolation + edge clipping.
        estC = None
        if bool(t2_cfg.get("report_interpolated_clipped_for_comparison", False)):
            # Bilinear interpolation with explicit clipping to grid centres (mirrors older behaviour).
            b0_c = np.clip(rng.normal(0.0, b0_sig, size=n_samp), b0_u_arr[0], b0_u_arr[-1])
            b1_c = np.clip(rng.normal(0.0, b1_sig, size=n_samp), b1_u_arr[0], b1_u_arr[-1])
            # Find bracket indices.
            iL = np.searchsorted(b0_u_arr, b0_c, side="right") - 1
            jL = np.searchsorted(b1_u_arr, b1_c, side="right") - 1
            iL = np.clip(iL, 0, len(b0_u_arr) - 2)
            jL = np.clip(jL, 0, len(b1_u_arr) - 2)
            iH = iL + 1
            jH = jL + 1
            x0 = b0_u_arr[iL]
            x1 = b0_u_arr[iH]
            y0 = b1_u_arr[jL]
            y1 = b1_u_arr[jH]
            tx = np.where(x1 > x0, (b0_c - x0) / (x1 - x0), 0.0)
            ty = np.where(y1 > y0, (b1_c - y0) / (y1 - y0), 0.0)
            v00 = surf[iL, jL]
            v10 = surf[iH, jL]
            v01 = surf[iL, jH]
            v11 = surf[iH, jH]
            dC = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11
            dC = np.asarray(dC, dtype=float)
            dC = dC[np.isfinite(dC)]
            estC = {
                "estimator": "bilinear_interp_clipped",
                "n_samples": int(dC.size),
                "mean": float(np.mean(dC)) if dC.size else float("nan"),
                "median": float(np.quantile(dC, 0.5)) if dC.size else float("nan"),
                "p16": float(np.quantile(dC, 0.16)) if dC.size else float("nan"),
                "p84": float(np.quantile(dC, 0.84)) if dC.size else float("nan"),
                "p_lt_1": float(np.mean(dC < 1.0)) if dC.size else float("nan"),
                "p_lt_0": float(np.mean(dC < 0.0)) if dC.size else float("nan"),
                "clip_fraction": float("nan"),
                "notes": "For comparison only: bilinear interpolation with explicit edge clipping.",
            }

        # Write summary table.
        out_rows = [estA, estB] + ([] if estC is None else [estC])
        _write_csv(tab_dir / "photoz_prior_marginal_summary.csv", out_rows, fieldnames=list(out_rows[0].keys()))
        photoz_summary.update({"grid_csv": str(grid_path), "b0_sigma": float(b0_sig), "b1_sigma": float(b1_sig), "estimators": out_rows})

        # Surface plot with prior ellipses.
        try:
            B0, B1 = np.meshgrid(np.asarray(b1_u, dtype=float), np.asarray(b0_u, dtype=float))
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            im = ax.pcolormesh(B1, B0, surf, shading="nearest", cmap="viridis")
            fig.colorbar(im, ax=ax, label="ΔLPD_total")
            ax.scatter([0.0], [0.0], c="w", s=50, marker="x", label="baseline (0,0)")
            # Prior 1σ and 2σ ellipses.
            t = np.linspace(0.0, 2.0 * np.pi, 400)
            for nsig, ls in [(1.0, "-"), (2.0, "--")]:
                ax.plot(nsig * b1_sig * np.cos(t), nsig * b0_sig * np.sin(t), color="w", lw=1.2, linestyle=ls, alpha=0.9, label=f"{int(nsig)}σ prior" if nsig == 1.0 else None)
            ax.set(xlabel="b1", ylabel="b0", title="Photo-z stress grid ΔLPD(b0,b1) with prior overlay")
            ax.grid(alpha=0.2, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "photoz_surface_with_prior.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        # Histograms per estimator.
        try:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.hist(xA, bins=30, weights=wA, alpha=0.75, color="C0", label="grid quadrature")
            ax.axvline(base_delta, color="C3", lw=2.0, label="baseline")
            ax.set(xlabel="ΔLPD_total", ylabel="prior-weight", title="Photo-z prior marginalisation (grid quadrature; conditional)")
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "photoz_prior_marginal_hist_grid.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass
        try:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.hist(dB, bins=30, alpha=0.75, color="C1", label="NN MC (in-grid)")
            ax.axvline(base_delta, color="C3", lw=2.0, label="baseline")
            ax.set(xlabel="ΔLPD_total", ylabel="count", title="Photo-z prior marginalisation (nearest-neighbour MC; in-grid)")
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "photoz_prior_marginal_hist_nn.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    # ==========================
    # TASK 3: CATALOG WEIGHTING FAMILY
    # ==========================
    t3_cfg = cfg.get("task3_catalog_variants", {})
    catvar_summary: dict[str, Any] = {"enabled": bool(t3_cfg.get("enabled", False))}
    catvar_score_rows: list[dict[str, Any]] = []
    catvar_event_rows: list[dict[str, Any]] = []
    if catvar_summary["enabled"]:
        z_max = float(t3_cfg.get("z_max", 0.3))
        z_edges = np.linspace(0.0, z_max, z_hist_nbins + 1)
        z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

        variants = list(t3_cfg.get("variants", []))
        for v in variants:
            vid = str(v.get("id") or "variant")
            kind = str(v.get("kind") or "baseline")
            repl: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            for ev in top_events:
                ev_npz = gap_root / "cache" / f"event_{ev}.npz"
                z, w, ipix, hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)

                if kind == "uniform_glade_index_variant":
                    ra_u, dec_u, z_u, w_u = gather_galaxies_from_pixels(cat_uni, hpix_sel, z_max=z_max)
                    if z_u.size == 0:
                        continue
                    theta = np.deg2rad(90.0 - np.asarray(dec_u, dtype=float))
                    phi = np.deg2rad(np.asarray(ra_u, dtype=float))
                    ipix_u = hp.ang2pix(int(pe.nside), theta, phi, nest=True).astype(np.int64, copy=False)
                    z_use = np.asarray(z_u, dtype=float)
                    w_use = np.asarray(w_u, dtype=float)
                    ipix_use = np.asarray(ipix_u, dtype=np.int64)
                else:
                    z_use = np.asarray(z, dtype=float)
                    w_use = np.asarray(w, dtype=float)
                    ipix_use = np.asarray(ipix, dtype=np.int64)

                # Map pixels to rows.
                npix = int(hp.nside2npix(int(pe.nside)))
                pix_to_row = np.full((npix,), -1, dtype=np.int32)
                pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
                row = pix_to_row[ipix_use]
                good = (row >= 0) & np.isfinite(z_use) & (z_use > 0.0) & np.isfinite(w_use) & (w_use > 0.0)
                if not np.any(good):
                    continue
                z_g = z_use[good]
                w_g = w_use[good]
                prob_g = np.asarray(pe.prob_pix, dtype=float)[row[good]]

                # Variant-specific effective weights.
                if kind == "baseline":
                    w_eff = w_g
                elif kind == "uniform":
                    w_eff = np.ones_like(w_g)
                elif kind == "uniform_glade_index_variant":
                    # Variant selects galaxies from the uniform GLADE+ index, but keeps its
                    # native weighting (typically uniform / MB-missing-inclusive).
                    w_eff = w_g
                elif kind == "cap":
                    p = float(v.get("cap_percentile", 0.99))
                    capv = float(np.quantile(w_g, p))
                    w_eff = np.minimum(w_g, capv)
                elif kind == "bright":
                    p = float(v.get("bright_percentile", 0.9))
                    cut = float(np.quantile(w_g, p))
                    w_eff = np.where(w_g >= cut, w_g, 0.0)
                    if bool(v.get("renormalize", True)):
                        s0 = float(np.sum(w_g * prob_g))
                        s1 = float(np.sum(w_eff * prob_g))
                        if s1 > 0.0 and np.isfinite(s0) and np.isfinite(s1):
                            w_eff = w_eff * (s0 / s1)
                elif kind == "z_flatten":
                    # Flatten the baseline weighted z histogram by reweighting galaxies by 1 / hist(zbin).
                    wprob0 = w_g * prob_g
                    hist0, _ = np.histogram(np.clip(z_g, 0.0, z_max), bins=z_edges, weights=wprob0)
                    hist0 = np.asarray(hist0, dtype=float)
                    mean_h = float(np.mean(hist0[np.isfinite(hist0) & (hist0 > 0.0)])) if np.any(hist0 > 0) else 1.0
                    bin_idx = np.searchsorted(z_edges, np.clip(z_g, 0.0, z_max), side="right") - 1
                    bin_idx = np.clip(bin_idx, 0, hist0.size - 1)
                    denom = hist0[bin_idx]
                    denom = np.where(denom > 0.0, denom, mean_h)
                    f = mean_h / denom
                    f = np.clip(f, float(v.get("clip_factor_min", 0.1)), float(v.get("clip_factor_max", 10.0)))
                    w_eff = w_g * f
                elif kind == "thin":
                    keep = float(v.get("keep_fraction", 0.5))
                    seed = int(v.get("seed", 1))
                    rng = np.random.default_rng(seed)
                    m = rng.random(size=w_g.size) < keep
                    w_eff = np.where(m, w_g / max(keep, 1e-6), 0.0)
                else:
                    raise ValueError(f"Unknown catalog variant kind: {kind}")

                wprob = np.asarray(w_eff, dtype=float) * np.asarray(prob_g, dtype=float)
                ok = np.isfinite(wprob) & (wprob > 0.0)
                hist, _ = np.histogram(np.clip(z_g[ok], 0.0, z_max), bins=z_edges, weights=wprob[ok])
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
                catvar_event_rows.append({"variant": vid, "event": ev, "delta_lpd_event_data": float(_logmeanexp(logL_mu) - _logmeanexp(logL_gr))})

            res = _score_with_cat_replacements(repl)
            dlp_total = float(res.lpd_mu_total - res.lpd_gr_total)
            dlp_data = float(res.lpd_mu_total_data - res.lpd_gr_total_data)
            catvar_score_rows.append(
                {
                    "variant": vid,
                    "kind": kind,
                    "delta_lpd_total": float(dlp_total),
                    "delta_lpd_data": float(dlp_data),
                    "delta_lpd_sel": float(dlp_total - dlp_data),
                    "delta_total_minus_baseline": float(dlp_total - base_delta),
                }
            )

        if catvar_score_rows:
            _write_csv(tab_dir / "catalog_variant_score_rows.csv", catvar_score_rows, fieldnames=list(catvar_score_rows[0].keys()))
        if catvar_event_rows:
            _write_csv(tab_dir / "catalog_variant_event_deltas.csv", catvar_event_rows, fieldnames=list(catvar_event_rows[0].keys()))

        # Bar plot for ΔLPD by variant.
        try:
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            labels = [r["variant"] for r in catvar_score_rows]
            y = np.asarray([float(r["delta_lpd_total"]) for r in catvar_score_rows], dtype=float)
            ax.bar(np.arange(len(labels)), y, color="C0", alpha=0.85)
            ax.axhline(base_delta, color="C3", lw=2.0, label="baseline")
            ax.axhline(1.0, color="k", lw=1.0, linestyle="--", alpha=0.8, label="ΔLPD=1")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set(ylabel="ΔLPD_total (spectral-only + selection)", title="Catalog-weight variant family")
            ax.grid(axis="y", alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "catalog_variant_deltaLPD_bars.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        # Waterfall plot: data vs selection components (relative to baseline).
        try:
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            labels = [r["variant"] for r in catvar_score_rows]
            d_data = np.asarray([float(r["delta_lpd_data"]) - base_delta_data for r in catvar_score_rows], dtype=float)
            d_sel = np.asarray([float(r["delta_lpd_sel"]) - base_delta_sel for r in catvar_score_rows], dtype=float)
            x = np.arange(len(labels))
            ax.bar(x, d_data, color="C0", alpha=0.85, label="Δ(data) vs baseline")
            ax.bar(x, d_sel, bottom=d_data, color="C2", alpha=0.75, label="Δ(selection) vs baseline")
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set(ylabel="Component shift (ΔLPD)", title="Component shifts vs baseline")
            ax.grid(axis="y", alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "catalog_variant_term_waterfall.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        catvar_summary.update({"n_variants": int(len(catvar_score_rows))})

    # ==========================
    # REPORT + SUMMARY
    # ==========================
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(cfg_path),
        "offline": bool(args.offline),
        "baseline": {
            "delta_lpd_total_spectral_only": float(base_delta),
            "delta_lpd_data_spectral_only": float(base_delta_data),
            "delta_lpd_sel_spectral_only": float(base_delta_sel),
            "gap_run_root": str(gap_root),
            "recon_run_dir": str(recon_run_dir),
            "run_label": str(run_label),
            "spectral_only_terms_npz": str(spec_terms_npz),
        },
        "task1_specz_override": {
            "enabled": bool(specz_override_summary.get("enabled", False)),
            "n_conditions": int(len(specz_score_rows)),
            "score_rows_csv": str(tab_dir / "specz_override_score_rows.csv") if specz_score_rows else None,
            "coverage_csv": str(tab_dir / "specz_coverage_summary.csv") if specz_cov_rows else None,
            "notes": "Spec-z overrides applied only to top leverage-proxy events; scoring uses binned-z spectral-only approximation for catalog term.",
        },
        "task2_photoz_prior": photoz_summary,
        "task3_catalog_variants": {
            "enabled": bool(catvar_summary.get("enabled", False)),
            "score_rows_csv": str(tab_dir / "catalog_variant_score_rows.csv") if catvar_score_rows else None,
            "event_deltas_csv": str(tab_dir / "catalog_variant_event_deltas.csv") if catvar_event_rows else None,
        },
    }
    # Include Task-1 runtime metadata (catalog manifests, radii scored, false-match control notes).
    summary["task1_specz_override"].update({k: v for k, v in specz_override_summary.items() if k != "enabled"})

    # Minimum-informative coverage gate (Task 1): mark events as uninformative if the
    # matched host-weight proxy fraction is <5% at the largest (radius, K) condition.
    if specz_cov_rows:
        t1_cfg = cfg.get("task1_specz_override", {})
        try:
            radii_scored = summary.get("task1_specz_override", {}).get("radii_scored_arcsec") or []
            radii_for_gate = radii_scored if radii_scored else [float(r) for r in t1_cfg.get("radii_arcsec", [])]
            r_max = float(max([float(r) for r in radii_for_gate]))
            k_max = int(max([int(k) for k in t1_cfg.get("k_tiers", [])]))
        except Exception:
            r_max, k_max = None, None
        if r_max is not None and k_max is not None:
            uninf = []
            cov_at_max = []
            for ev in top_events:
                rows = [
                    r
                    for r in specz_cov_rows
                    if r.get("event") == ev
                    and r.get("control") == "true"
                    and float(r.get("radius_arcsec", -1.0)) == float(r_max)
                    and int(r.get("k", -1)) == int(k_max)
                ]
                if not rows:
                    continue
                rr = rows[0]
                fwt = float(rr.get("frac_weight_matched_total", float("nan")))
                cov_at_max.append({"event": ev, "radius_arcsec": float(r_max), "k": int(k_max), "frac_weight_matched_total": fwt})
                if np.isfinite(fwt) and fwt < 0.05:
                    uninf.append(ev)
            summary["task1_specz_override"].update(
                {
                    "informative_gate": {"radius_arcsec": float(r_max), "k": int(k_max), "min_frac_weight_total": 0.05},
                    "uninformative_events": uninf,
                    "coverage_at_gate": cov_at_max,
                }
            )
    _write_json(out_root / "summary.json", summary)

    # Markdown report with executive synthesis (<= 1 page, ~ a few hundred words).
    lines: list[str] = []
    lines.append("# Dark-Siren Smoking-Gun Sprint (Non-Strain)\n")
    lines.append(f"Run directory: `{out_root}`\n")
    lines.append("\n## Executive Summary\n")
    lines.append(f"- Baseline (spectral-only + selection): ΔLPD_total = **{base_delta:.3f}** (data-only **{base_delta_data:.3f}**, selection **{base_delta_sel:.3f}**).\n")
    if specz_score_rows:
        # Best/worst conditions for quick scan.
        best = max(specz_score_rows, key=lambda r: float(r["delta_lpd_total"]))
        worst = min(specz_score_rows, key=lambda r: float(r["delta_lpd_total"]))
        lines.append(
            f"- Task 1 (bulk spec-z crossmatch + override): ran {len(specz_score_rows)} conditions over radii/K tiers.\n"
            f"  - Best ΔLPD_total = {float(best['delta_lpd_total']):.3f} at radius={best['radius_arcsec']}\" K={best['k']}.\n"
            f"  - Worst ΔLPD_total = {float(worst['delta_lpd_total']):.3f} at radius={worst['radius_arcsec']}\" K={worst['k']}.\n"
        )
        lines.append(f"  - Coverage summary: `{tab_dir / 'specz_coverage_summary.csv'}`.\n")
        # Minimum informative coverage gate summary.
        gate = summary.get("task1_specz_override", {}).get("informative_gate")
        cov_gate = summary.get("task1_specz_override", {}).get("coverage_at_gate") or []
        uninf = summary.get("task1_specz_override", {}).get("uninformative_events") or []
        if gate and cov_gate:
            lines.append(f"  - Informative-coverage gate: radius={gate['radius_arcsec']}\" K={gate['k']} (min frac-weight-total={gate['min_frac_weight_total']}).\n")
            for r in cov_gate:
                lines.append(f"    - {r['event']}: frac_weight_matched_total={float(r['frac_weight_matched_total']):.4f}\n")
            if uninf:
                lines.append(f"  - Uninformative at gate (<5% weight matched): {', '.join([str(x) for x in uninf])}\n")
    else:
        lines.append("- Task 1 (bulk spec-z crossmatch + override): not executed or no spec-z catalogs loaded.\n")

    if photoz_summary.get("enabled", False):
        ests = photoz_summary.get("estimators", [])
        if ests:
            a = ests[0]
            b = ests[1] if len(ests) > 1 else None
            lines.append(
                "- Task 2 (photo-z prior marginalisation): replaced interpolation+clipping with two no-mirage estimators.\n"
                f"  - Grid quadrature mean/median: {a['mean']:.3f} / {a['median']:.3f}, clip_fraction={a['clip_fraction']:.3f}.\n"
                + (f"  - NN MC mean/median: {b['mean']:.3f} / {b['median']:.3f}, clip_fraction={b['clip_fraction']:.3f}.\n" if b else "")
            )
            lines.append(f"  - Surface plot: `{fig_dir / 'photoz_surface_with_prior.png'}`.\n")
    else:
        lines.append("- Task 2 (photo-z prior marginalisation): not executed.\n")

    if catvar_score_rows:
        best = max(catvar_score_rows, key=lambda r: float(r["delta_lpd_total"]))
        worst = min(catvar_score_rows, key=lambda r: float(r["delta_lpd_total"]))
        lines.append(
            "- Task 3 (catalog-weight family): evaluated controlled variants; sign stability and sensitivity are summarised in:\n"
            f"  - `{tab_dir / 'catalog_variant_score_rows.csv'}`\n"
            f"  - Best ΔLPD_total={float(best['delta_lpd_total']):.3f} ({best['variant']}); worst ΔLPD_total={float(worst['delta_lpd_total']):.3f} ({worst['variant']}).\n"
        )
        lines.append(f"  - Figures: `{fig_dir / 'catalog_variant_deltaLPD_bars.png'}`, `{fig_dir / 'catalog_variant_term_waterfall.png'}`.\n")
    else:
        lines.append("- Task 3 (catalog-weight family): not executed.\n")

    lines.append("\n## Outputs\n")
    lines.append(f"- `summary.json`: `{out_root / 'summary.json'}`\n")
    lines.append(f"- Tables: `{tab_dir}`\n")
    lines.append(f"- Figures: `{fig_dir}`\n")
    lines.append(f"- Raw audit artifacts: `{raw_dir}`\n")

    (out_root / "report.md").write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
