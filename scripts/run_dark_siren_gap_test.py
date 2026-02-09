from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from scipy.special import betaln

from entropy_horizon_recon.dark_sirens import (
    DarkSirenEventScore,
    compute_dark_siren_logL_draws,
    credible_region_pixels,
    gather_galaxies_from_pixels,
    load_gladeplus_index,
    read_skymap_3d,
)
from entropy_horizon_recon.dark_sirens_pe import (
    PePixelDistanceHistogram,
    build_pe_pixel_distance_histogram,
    compute_dark_siren_logL_draws_from_pe_hist,
    load_gwtc_pe_sky_samples,
)
from entropy_horizon_recon.dark_sirens_incompleteness import (
    compute_missing_host_logL_draws_from_histogram,
    compute_missing_host_logL_draws_from_pixels,
    precompute_missing_host_prior,
    select_missing_pixels,
)
from entropy_horizon_recon.dark_sirens_selection import (
    compute_selection_alpha_from_injections,
    load_o3_injections,
    resolve_selection_injection_file,
)
from entropy_horizon_recon.dark_sirens_hierarchical_pe import (
    GWTCPeHierarchicalSamples,
    compute_hierarchical_pe_logL_draws,
    load_gwtc_pe_hierarchical_samples,
)
from entropy_horizon_recon.dark_siren_h0 import (
    compute_gr_h0_posterior_grid,
    compute_gr_h0_posterior_grid_hierarchical_pe,
)
from entropy_horizon_recon.gwtc_pe_index import build_gwtc_pe_index
from entropy_horizon_recon.gwtc_pe_priors import (
    PeAnalyticDistancePrior,
    load_gwtc_pe_analytic_priors,
    select_gwtc_pe_analysis_with_analytic_priors,
)
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior
from entropy_horizon_recon.sirens import load_mu_forward_posterior


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _default_nproc() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return int(os.cpu_count() or 1)


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_logmeanexp expects a 1D array")
    if not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _stable_int_seed(s: str) -> int:
    # Deterministic seed derived from a string.
    import hashlib

    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _resolve_nproc(nproc: int) -> int:
    nproc = int(nproc)
    if nproc < 0:
        raise ValueError("--n-proc must be >=0.")
    if nproc == 0:
        # Respect taskset / scheduler affinity when possible.
        try:
            return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
        except Exception:
            return max(1, int(os.cpu_count() or 1))
    return max(1, nproc)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3D rotation matrix (approximately uniform on SO(3))."""
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _rotate_radec(ra_rad: np.ndarray, dec_rad: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (ra, dec) samples by a 3x3 rotation matrix, returning (ra', dec') in radians."""
    ra = np.asarray(ra_rad, dtype=float)
    dec = np.asarray(dec_rad, dtype=float)
    if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
        raise ValueError("ra_rad/dec_rad must be 1D arrays with matching shapes.")
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3.")

    c = np.cos(dec)
    x = c * np.cos(ra)
    y = c * np.sin(ra)
    z = np.sin(dec)
    v = np.stack([x, y, z], axis=0)  # (3, n)
    v2 = R @ v
    x2, y2, z2 = v2[0], v2[1], v2[2]
    dec2 = np.arcsin(np.clip(z2, -1.0, 1.0))
    ra2 = np.mod(np.arctan2(y2, x2), 2.0 * np.pi)
    return np.asarray(ra2, dtype=float), np.asarray(dec2, dtype=float)


def _downsample_posterior(post, *, max_draws: int, seed: int) -> tuple[object, list[int]]:
    """Downsample a MuForwardPosterior to at most max_draws draws, deterministically."""
    n = int(post.H_samples.shape[0])
    k = int(max_draws)
    if k <= 0:
        raise ValueError("max_draws must be positive")
    if k >= n:
        return post, list(range(n))

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=k, replace=False).astype(int))

    def _sel(a):
        a = np.asarray(a)
        return a[idx]

    from entropy_horizon_recon.sirens import MuForwardPosterior

    post2 = MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=_sel(post.logmu_x_samples),
        z_grid=post.z_grid,
        H_samples=_sel(post.H_samples),
        H0=_sel(post.H0),
        omega_m0=_sel(post.omega_m0),
        omega_k0=_sel(post.omega_k0),
        sigma8_0=_sel(post.sigma8_0) if post.sigma8_0 is not None else None,
    )
    return post2, [int(i) for i in idx.tolist()]


def _completeness_cache_path(index_dir: Path, *, z_max: float, zref_max: float, nbins: int) -> Path:
    # Tag includes "_w" since we use weighted histograms when cat weights are available.
    # "v2" uses a low-z multi-bin least-squares fit for the z^3 normalization (more stable than
    # anchoring on a single reference bin).
    tag = f"z3_cum_v2_zmax{z_max:.3f}_zref{zref_max:.3f}_nb{int(nbins)}_w"
    return index_dir / f"completeness_{tag}.json"


def _estimate_gladeplus_completeness_z3_cumulative(
    cat_z: np.ndarray,
    cat_w: np.ndarray | None = None,
    *,
    z_max: float,
    zref_max: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a crude cumulative completeness curve C(z) using ~z^3 scaling at low z.

    If cat_w is provided, we compute weighted cumulative sums instead of raw counts. This makes the
    estimator interpretable as a luminosity-completeness proxy when w encodes (e.g.) L_B.
    """
    z_max = float(z_max)
    zref_max = float(zref_max)
    nbins = int(nbins)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive")
    if zref_max <= 0.0 or zref_max >= z_max:
        raise ValueError("zref_max must satisfy 0 < zref_max < z_max")
    if nbins < 10:
        raise ValueError("nbins too small")

    z = np.asarray(cat_z, dtype=float)
    w = None if cat_w is None else np.asarray(cat_w, dtype=float)
    m = np.isfinite(z) & (z > 0.0) & (z <= z_max)
    if w is not None:
        m &= np.isfinite(w) & (w > 0.0)
        w = w[m]
    z = z[m]
    if z.size == 0 or (w is not None and w.size == 0):
        raise ValueError("No galaxies in the requested z range for completeness estimation.")

    edges = np.linspace(0.0, z_max, nbins + 1)
    hist, _ = np.histogram(z, bins=edges, weights=w)
    cum = np.cumsum(hist.astype(float))

    z_lo = edges[:-1]
    z_hi = edges[1:]  # cumulative evaluated at upper edge

    # Fit the low-z normalization in the differential space:
    #   hist_i ≈ n0 * (z_hi^3 - z_lo^3),
    # using all bins with z_hi <= zref_max. This is more stable than using a single anchor bin.
    m_ref = (z_hi > 0.0) & (z_hi <= float(zref_max))
    if int(np.count_nonzero(m_ref)) < 5:
        raise ValueError("Too few low-z bins for completeness fit; increase zref_max or nbins.")

    dz3 = (z_hi**3) - (z_lo**3)
    A = dz3[m_ref].astype(float, copy=False)
    y = hist[m_ref].astype(float, copy=False)
    denom = float(np.sum(A * A))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Degenerate low-z completeness fit (denom <= 0).")
    n0 = float(np.sum(y * A) / denom)
    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError("Non-positive low-z completeness fit normalization; increase zref_max.")

    # Expected cumulative totals under constant comoving density at low z: X(<z) ~ n0 * z^3,
    # where X is either counts or a weighted sum (e.g. luminosity).
    exp_cum = n0 * (z_hi**3)
    C = np.clip(cum / np.clip(exp_cum, 1e-30, np.inf), 0.0, 1.0)

    # Enforce non-increasing completeness with z (monotone): completeness shouldn't rise at high z.
    C = np.minimum.accumulate(C)
    return z_hi, C


def _load_or_build_completeness(
    index_dir: Path,
    cat_z: np.ndarray,
    cat_w: np.ndarray | None,
    *,
    z_max: float,
    zref_max: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = _completeness_cache_path(index_dir, z_max=z_max, zref_max=zref_max, nbins=nbins)
    if path.exists():
        obj = json.loads(path.read_text())
        return np.asarray(obj["z"], dtype=float), np.asarray(obj["C"], dtype=float)

    z_grid, C_grid = _estimate_gladeplus_completeness_z3_cumulative(cat_z, cat_w, z_max=z_max, zref_max=zref_max, nbins=nbins)
    path.write_text(
        json.dumps(
            {
                "method": "z3_cumulative_v2",
                "z_max": float(z_max),
                "zref_max": float(zref_max),
                "nbins": int(nbins),
                "weighted": bool(cat_w is not None),
                "z": [float(x) for x in z_grid.tolist()],
                "C": [float(x) for x in C_grid.tolist()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return z_grid, C_grid


def _apply_completeness_weights(*, z: np.ndarray, w: np.ndarray, z_grid: np.ndarray, C_grid: np.ndarray, c_floor: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    C = np.interp(z, z_grid, C_grid, left=float(C_grid[0]), right=float(C_grid[-1]))
    C = np.clip(C, float(c_floor), 1.0)
    return w / C


def _estimate_f_miss_from_completeness(
    *,
    z_grid: np.ndarray,
    C_grid: np.ndarray,
    z_max: float,
    host_prior_z_mode: str,
    host_prior_z_k: float,
) -> float:
    """Estimate a single f_miss from a completeness curve.

    This is intentionally a *simple* scalar summary:
      f_miss = 1 - <C(z)>_{p_host(z)}

    We use a low-z comoving proxy p_host(z) ∝ z^2 * rho_host(z), where:
      rho_host(z) = 1            (comoving_uniform/none)
      rho_host(z) = (1+z)^k      (comoving_powerlaw)
    """
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")
    z_grid = np.asarray(z_grid, dtype=float)
    C_grid = np.asarray(C_grid, dtype=float)
    if z_grid.ndim != 1 or C_grid.ndim != 1 or z_grid.shape != C_grid.shape:
        raise ValueError("z_grid and C_grid must be 1D arrays with matching shape.")
    if np.any(np.diff(z_grid) <= 0.0):
        raise ValueError("z_grid must be strictly increasing.")

    # Dense grid for stable quadrature.
    z = np.linspace(0.0, z_max, 2001)
    C = np.interp(z, z_grid, C_grid, left=float(C_grid[0]), right=float(C_grid[-1]))
    C = np.clip(C, 0.0, 1.0)

    w = np.clip(z, 0.0, None) ** 2
    if host_prior_z_mode == "comoving_powerlaw":
        w = w * np.clip(1.0 + z, 1e-12, np.inf) ** float(host_prior_z_k)
    elif host_prior_z_mode in ("none", "comoving_uniform"):
        pass
    else:
        raise ValueError("Unknown host_prior_z_mode.")

    denom = float(np.trapezoid(w, x=z))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Non-positive host-prior normalization in f_miss estimate.")

    f_cat = float(np.trapezoid(w * C, x=z) / denom)
    f_miss = float(np.clip(1.0 - f_cat, 0.0, 1.0))
    return f_miss


def _maybe_parse_skymap_filelist(skymap_dir: Path) -> dict[str, Path] | None:
    """Try to parse a skyLocalizationFileList.csv-style manifest if present."""
    for cand in [
        skymap_dir / "skyLocalizationFileList.csv",
        skymap_dir / "skymaps" / "skyLocalizationFileList.csv",
    ]:
        if not cand.exists():
            continue
        mapping: dict[str, Path] = {}
        with cand.open("r", newline="") as f:
            reader = csv.DictReader(f)
            cols = [c.lower() for c in (reader.fieldnames or [])]
            # Heuristics: look for event and file columns.
            event_col = None
            file_col = None
            for c in reader.fieldnames or []:
                cl = c.lower()
                if event_col is None and ("event" in cl or cl in ("name", "id", "gid")):
                    event_col = c
                if file_col is None and ("file" in cl or "path" in cl or "skymap" in cl):
                    file_col = c
            if event_col is None or file_col is None:
                raise ValueError(f"{cand}: could not infer event/file columns from {cols}")
            for row in reader:
                ev = str(row[event_col]).strip()
                fp = str(row[file_col]).strip()
                if not ev or not fp:
                    continue
                p = (cand.parent / fp).resolve()
                if p.exists():
                    mapping[ev] = p
        if mapping:
            return mapping
    return None


def _find_skymaps(skymap_dir: Path) -> dict[str, Path]:
    """Find sky maps under a directory, returning {event_name: path}."""
    parsed = _maybe_parse_skymap_filelist(skymap_dir)
    if parsed:
        return parsed

    by_event: dict[str, list[Path]] = {}
    for p in sorted(skymap_dir.rglob("*.fits*")):
        m = re.search(r"(GW\d{6}_\d{6})", p.name)
        if not m:
            continue
        name = m.group(1)
        by_event.setdefault(name, []).append(p)
    if not by_event:
        raise FileNotFoundError(f"No skymaps found under {skymap_dir}")

    def pick(paths: list[Path]) -> Path:
        # Deterministic preference ordering.
        prefs = [
            ":Mixed",
            "IMRPhenomXPHM",
            "SEOBNRv4PHM",
            "IMRPhenom",
            "SEOBNR",
        ]
        for key in prefs:
            cand = [p for p in paths if key in p.name]
            if cand:
                return sorted(cand)[0]
        return sorted(paths)[0]

    out: dict[str, Path] = {ev: pick(paths) for ev, paths in by_event.items()}
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _resolve_gw_distance_prior_mode(*, requested: str, skymap_path: Path) -> str:
    if requested != "auto":
        return requested
    name = skymap_path.name
    # Heuristics for public products:
    # - GWTC-3 sky maps: cosmo_reweight -> comoving prior; otherwise default dL^2-ish.
    # - GWTC PEDataRelease files: *_mixed_cosmo* -> comoving prior; *_mixed_nocosmo* -> dL^2-ish.
    if "nocosmo" in name:
        return "dL_powerlaw"
    if "cosmo_reweight" in name or "mixed_cosmo" in name or name.endswith("_cosmo.h5") or name.endswith("_cosmo.hdf5"):
        return "comoving_lcdm_sourceframe"
    return "dL_powerlaw"


def _default_pe_analysis_prefer_for_analytic_priors(path: Path) -> list[str]:
    # Match the common GWTC-3/GWTC-4 analysis naming conventions (see `dark_sirens_pe.py`).
    if path.suffix == ".h5":
        return [
            "C01:Mixed",
            "C01:IMRPhenomXPHM",
            "C01:SEOBNRv4PHM",
        ]
    if path.suffix == ".hdf5":
        return [
            "C00:Mixed",
            "C00:Mixed+XO4a",
            "C00:IMRPhenomXPHM-SpinTaylor",
            "C00:SEOBNRv5PHM",
            "C00:NRSur7dq4",
            "C00:IMRPhenomXO4a",
        ]
    return []


_GW_PRIOR_CACHE: dict[tuple[Any, ...], Any] = {}


def _gw_prior_for_args(args: argparse.Namespace, gw_data_path: Path, *, pe_analysis: str | None = None) -> Any:
    """Resolve and cache the GW distance-prior correction for a given GW data product path."""
    mode = _resolve_gw_distance_prior_mode(requested=str(args.gw_distance_prior_mode), skymap_path=gw_data_path)
    if mode == "pe_analytic":
        # PEDataRelease-only: divide out the actual analytic PE sampling prior π_PE(dL).
        if gw_data_path.suffix not in (".h5", ".hdf5"):
            raise ValueError("--gw-distance-prior-mode=pe_analytic requires --gw-data-mode=pe (PEDataRelease files).")
        if pe_analysis is None:
            raise ValueError("Internal error: pe_analysis is required for pe_analytic distance-prior mode.")
        key = ("pe_analytic", str(gw_data_path), str(pe_analysis))
        if key not in _GW_PRIOR_CACHE:
            pri = load_gwtc_pe_analytic_priors(
                path=gw_data_path,
                analysis=str(pe_analysis),
                parameters=["luminosity_distance"],
            )
            spec, prior = pri["luminosity_distance"]
            _GW_PRIOR_CACHE[key] = PeAnalyticDistancePrior(
                file=str(gw_data_path),
                analysis=str(pe_analysis),
                spec=spec,
                prior=prior,
            )
        return _GW_PRIOR_CACHE[key]

    key = (
        mode,
        float(args.gw_distance_prior_power),
        float(args.gw_distance_prior_h0_ref),
        float(args.gw_distance_prior_omega_m0),
        float(args.gw_distance_prior_omega_k0),
        float(args.gw_distance_prior_zmax),
    )
    if key not in _GW_PRIOR_CACHE:
        _GW_PRIOR_CACHE[key] = GWDistancePrior(
            mode=str(mode),  # type: ignore[arg-type]
            powerlaw_k=float(args.gw_distance_prior_power),
            h0_ref=float(args.gw_distance_prior_h0_ref),
            omega_m0=float(args.gw_distance_prior_omega_m0),
            omega_k0=float(args.gw_distance_prior_omega_k0),
            z_max=float(args.gw_distance_prior_zmax),
        )
    return _GW_PRIOR_CACHE[key]


# Global worker state for parallel per-event scoring.
_EVENT_WORKER_STATE: dict[str, Any] = {}


def _set_event_worker_state(state: dict[str, Any]) -> None:
    global _EVENT_WORKER_STATE
    _EVENT_WORKER_STATE = dict(state)


def _score_one_event_for_run(ev: str) -> dict[str, Any]:
    """Worker: compute per-event (catalog + optional missing) logL vectors for one run_label.

    This function uses `_EVENT_WORKER_STATE` populated by the parent process. It is designed to be
    run under a fork-based multiprocessing context to avoid pickling large arrays.
    """
    st = _EVENT_WORKER_STATE
    args: argparse.Namespace = st["args"]
    run_label: str = st["run_label"]
    post = st["post"]
    draw_idx = st["draw_idx"]
    gw_data_mode: str = st["gw_data_mode"]
    pe_base_dir: Path = st["pe_base_dir"]
    event_cache: dict[str, Any] = st["event_cache"]
    cache_terms_dir: Path = st["cache_terms_dir"]
    cache_missing_dir: Path = st["cache_missing_dir"]
    missing_pre = st["missing_pre"]

    try:
        debug = str(os.environ.get("EH_DARK_SIREN_PROGRESS", "")).strip().lower() not in ("", "0", "false", "no")
        t0_event = time.perf_counter()

        meta = event_cache[ev]["meta"]
        sky_path = Path(str(meta["skymap_path"]))
        if debug:
            pid = os.getpid()
            n_gal_meta = int(meta.get("n_gal", -1))
            n_gal = n_gal_meta if n_gal_meta >= 0 else int(np.asarray(event_cache[ev]["z"], dtype=float).size)
            area = float(meta.get("sky_area_deg2", float("nan")))
            print(
                f"[dark_sirens] {run_label} {ev}: start pid={pid} n_gal={n_gal:,} area={area:.1f} deg^2 "
                f"pe_distance_mode={str(getattr(args, 'pe_distance_mode', 'full'))} "
                f"gnull={str(getattr(args, 'galaxy_null_mode', 'none'))}",
                flush=True,
            )

        pe_distance_mode = str(getattr(args, "pe_distance_mode", "full"))
        galaxy_null_mode = str(getattr(args, "galaxy_null_mode", "none"))
        galaxy_null_seed = int(getattr(args, "galaxy_null_seed", 0))

        gw_prior = None
        sky = None
        pe_hist = None
        pe_path = None
        if gw_data_mode == "skymap":
            sky = read_skymap_3d(sky_path, nest=True)
            gw_prior = _gw_prior_for_args(args, sky_path)
        else:
            pe_file = meta.get("pe_file")
            if pe_file is None:
                raise FileNotFoundError("Missing pe_file in event cache meta.")
            pe_path = Path(str(pe_file))
            pe_meta = meta.get("pe_meta") or {}
            a = pe_meta.get("analysis")
            pe_analysis = str(a) if a is not None else None
            gw_prior = _gw_prior_for_args(args, pe_path, pe_analysis=pe_analysis)
            pe_hist = PePixelDistanceHistogram(
                nside=int(meta.get("pe_nside", int(args.pe_nside))),
                nest=True,
                p_credible=float(meta.get("p_credible", float(args.p_credible))),
                pix_sel=np.asarray(event_cache[ev]["pe_pix_sel"], dtype=np.int64),
                prob_pix=np.asarray(event_cache[ev]["pe_prob_pix"], dtype=float),
                dL_edges=np.asarray(event_cache[ev]["pe_dL_edges"], dtype=float),
                pdf_bins=np.asarray(event_cache[ev]["pe_pdf_bins"], dtype=float),
            )

        assert gw_prior is not None
        # Keep separate per-event caches for spectral-only vs full PE distance handling to allow
        # clean comparisons within the same output dir (and to avoid accidental cache reuse).
        base_suffix = f"__{pe_distance_mode}" if (gw_data_mode == "pe" and pe_distance_mode != "full") else ""
        cat_suffix = str(base_suffix)
        if galaxy_null_mode != "none":
            cat_suffix += f"__gnull_{galaxy_null_mode}_seed{galaxy_null_seed}"
        miss_suffix = str(base_suffix)
        cat_path = cache_terms_dir / f"cat_{ev}__{run_label}{cat_suffix}.npz"
        want_cat_meta = {
            "event": str(ev),
            "run": str(run_label),
            "gw_data_mode": str(gw_data_mode),
            "pe_distance_mode": str(pe_distance_mode),
            "null_mode": str(getattr(args, "null_mode", "none")),
            "null_seed": int(getattr(args, "null_seed", 0)),
            "skymap_path": str(sky_path),
            "pe_file": str(meta.get("pe_file")),
            "pe_hist": pe_hist.to_jsonable() if pe_hist is not None else None,
            "convention": str(args.convention),
            "gw_distance_prior": gw_prior.to_jsonable(),
            "galaxy_null_mode": str(galaxy_null_mode),
            "galaxy_null_seed": int(galaxy_null_seed),
            "draw_idx": draw_idx,
        }

        logL_cat_mu: np.ndarray | None = None
        logL_cat_gr: np.ndarray | None = None
        cat_cache_hit = False
        if cat_path.exists():
            try:
                with np.load(cat_path, allow_pickle=True) as d:
                    meta_c = json.loads(str(d["meta"].tolist()))
                    if meta_c == want_cat_meta:
                        logL_cat_mu = np.asarray(d["logL_cat_mu"], dtype=float)
                        logL_cat_gr = np.asarray(d["logL_cat_gr"], dtype=float)
                        cat_cache_hit = True
            except Exception:
                logL_cat_mu = None
                logL_cat_gr = None

        if logL_cat_mu is None or logL_cat_gr is None:
            z_gal = np.asarray(event_cache[ev]["z"], dtype=float)
            w_gal = np.asarray(event_cache[ev]["w"], dtype=float)
            ipix_gal = np.asarray(event_cache[ev]["ipix"], dtype=np.int64)
            if galaxy_null_mode != "none":
                rng = np.random.default_rng(_stable_int_seed(f"galnull:{galaxy_null_seed}:{galaxy_null_mode}:{ev}"))
                p = rng.permutation(int(z_gal.size))
                if galaxy_null_mode == "shuffle_zw":
                    z_gal = z_gal[p]
                    w_gal = w_gal[p]
                elif galaxy_null_mode == "shuffle_z":
                    z_gal = z_gal[p]
                else:
                    raise ValueError(f"Unknown galaxy_null_mode: {galaxy_null_mode}")
            if gw_data_mode == "skymap":
                assert sky is not None
                draws = compute_dark_siren_logL_draws(
                    event=ev,
                    sky=sky,
                    sky_area_deg2=float(meta["sky_area_deg2"]),
                    post=post,
                    z=z_gal,
                    w=w_gal,
                    ipix=ipix_gal,
                    convention=args.convention,  # type: ignore[arg-type]
                    max_draws=None,  # already downsampled
                    gw_distance_prior=gw_prior,
                    gal_chunk_size=int(args.galaxy_chunk_size),
                )
                logL_cat_mu = draws.logL_mu
                logL_cat_gr = draws.logL_gr
            else:
                assert pe_hist is not None
                logL_cat_mu, logL_cat_gr = compute_dark_siren_logL_draws_from_pe_hist(
                    event=ev,
                    pe=pe_hist,
                    post=post,
                    z_gal=z_gal,
                    w_gal=w_gal,
                    ipix_gal=ipix_gal,
                    convention=args.convention,  # type: ignore[arg-type]
                    gw_distance_prior=gw_prior,
                    distance_mode=pe_distance_mode,  # type: ignore[arg-type]
                    gal_chunk_size=int(args.galaxy_chunk_size),
                )

        if debug:
            dt_cat = time.perf_counter() - t0_event
            print(
                f"[dark_sirens] {run_label} {ev}: catalog term done dt={dt_cat:.1f}s cache_hit={bool(cat_cache_hit)}",
                flush=True,
            )

        if not cat_cache_hit:
            try:
                np.savez_compressed(
                    cat_path,
                    meta=json.dumps(want_cat_meta),
                    logL_cat_mu=np.asarray(logL_cat_mu, dtype=np.float64),
                    logL_cat_gr=np.asarray(logL_cat_gr, dtype=np.float64),
                )
            except Exception:
                pass

        out: dict[str, Any] = {
            "ok": True,
            "event": str(ev),
            "n_gal": int(meta.get("n_gal", int(np.asarray(event_cache[ev]["z"]).size))),
            "sky_area_deg2": float(meta.get("sky_area_deg2", float("nan"))),
            "cat_cache_hit": bool(cat_cache_hit),
            "logL_cat_mu": np.asarray(logL_cat_mu, dtype=float),
            "logL_cat_gr": np.asarray(logL_cat_gr, dtype=float),
        }

        if str(args.mixture_mode) == "simple":
            if missing_pre is None:
                raise RuntimeError("Internal error: missing_pre is None with mixture_mode=simple.")

            miss_path = cache_missing_dir / f"missing_{ev}__{run_label}{miss_suffix}.npz"
            want_meta = {
                "event": str(ev),
                "gw_data_mode": str(gw_data_mode),
                "pe_distance_mode": str(pe_distance_mode),
                "null_mode": str(getattr(args, "null_mode", "none")),
                "null_seed": int(getattr(args, "null_seed", 0)),
                "skymap_path": str(sky_path),
                "pe_file": str(meta.get("pe_file")),
                "pe_hist": pe_hist.to_jsonable() if pe_hist is not None else None,
                "p_credible": float(args.p_credible),
                "cat_nside": int(st["cat_nside"]),
                "gw_distance_prior": gw_prior.to_jsonable(),
                "missing_z_max": float(args.missing_z_max),
                "host_prior_z_mode": str(args.host_prior_z_mode),
                "host_prior_z_k": float(args.host_prior_z_k),
                "missing_dl_grid_n": int(args.missing_dl_grid_n),
                "missing_dl_min_mpc": float(args.missing_dl_min_mpc),
                "missing_dl_max_mpc": float(args.missing_dl_max_mpc) if args.missing_dl_max_mpc is not None else None,
                "missing_dl_nsigma": float(args.missing_dl_nsigma),
                "missing_pixel_chunk_size": int(args.missing_pixel_chunk_size),
                "draw_idx": draw_idx,
            }

            logL_missing_mu: np.ndarray | None = None
            logL_missing_gr: np.ndarray | None = None
            miss_cache_hit = False
            if miss_path.exists():
                try:
                    with np.load(miss_path, allow_pickle=True) as d:
                        meta_m = json.loads(str(d["meta"].tolist()))
                        if meta_m == want_meta:
                            logL_missing_mu = np.asarray(d["logL_missing_mu"], dtype=float)
                            logL_missing_gr = np.asarray(d["logL_missing_gr"], dtype=float)
                            miss_cache_hit = True
                except Exception:
                    logL_missing_mu = None
                    logL_missing_gr = None

            if logL_missing_mu is None or logL_missing_gr is None:
                if gw_data_mode == "skymap":
                    assert sky is not None
                    hpix_sel = event_cache[ev].get("hpix_sel")
                    if hpix_sel is None:
                        hpix_sel, _ = credible_region_pixels(sky, nside_out=int(st["cat_nside"]), p_credible=float(args.p_credible))

                    prob_pix, distmu, distsigma, distnorm = select_missing_pixels(
                        sky,
                        p_credible=float(args.p_credible),
                        nside_coarse=int(st["cat_nside"]),
                        hpix_coarse=np.asarray(hpix_sel, dtype=np.int64),
                    )

                    dl_min = float(args.missing_dl_min_mpc)
                    if dl_min <= 0.0:
                        raise ValueError("missing_dl_min_mpc must be positive.")

                    if args.missing_dl_max_mpc is not None:
                        dl_max = float(args.missing_dl_max_mpc)
                    else:
                        dl_max = float(np.max(distmu + float(args.missing_dl_nsigma) * distsigma))

                    if not np.isfinite(dl_max) or dl_max <= dl_min:
                        raise ValueError(f"Non-finite/invalid missing dL grid bounds: [{dl_min},{dl_max}].")

                    dL_grid = np.linspace(dl_min, dl_max, int(args.missing_dl_grid_n))
                    logL_missing_mu, logL_missing_gr = compute_missing_host_logL_draws_from_pixels(
                        prob_pix=prob_pix,
                        distmu=distmu,
                        distsigma=distsigma,
                        distnorm=distnorm,
                        pre=missing_pre,
                        dL_grid=dL_grid,
                        gw_distance_prior=gw_prior,
                        pixel_chunk_size=int(args.missing_pixel_chunk_size),
                    )
                else:
                    assert pe_hist is not None
                    logL_missing_mu, logL_missing_gr = compute_missing_host_logL_draws_from_histogram(
                        prob_pix=np.asarray(pe_hist.prob_pix, dtype=float),
                        pdf_bins=np.asarray(pe_hist.pdf_bins, dtype=float),
                        dL_edges=np.asarray(pe_hist.dL_edges, dtype=float),
                        pre=missing_pre,
                        gw_distance_prior=gw_prior,
                        distance_mode=pe_distance_mode,  # type: ignore[arg-type]
                        pixel_chunk_size=int(args.missing_pixel_chunk_size),
                    )

                np.savez(
                    miss_path,
                    meta=json.dumps(want_meta),
                    logL_missing_mu=np.asarray(logL_missing_mu, dtype=np.float64),
                    logL_missing_gr=np.asarray(logL_missing_gr, dtype=np.float64),
                )

            if debug:
                dt_miss = time.perf_counter() - t0_event
                print(
                    f"[dark_sirens] {run_label} {ev}: missing-host term done dt={dt_miss:.1f}s cache_hit={bool(miss_cache_hit)}",
                    flush=True,
                )

            out.update(
                {
                    "miss_cache_hit": bool(miss_cache_hit),
                    "logL_missing_mu": np.asarray(logL_missing_mu, dtype=float),
                    "logL_missing_gr": np.asarray(logL_missing_gr, dtype=float),
                }
            )

        if debug:
            dt_total = time.perf_counter() - t0_event
            print(f"[dark_sirens] {run_label} {ev}: done dt={dt_total:.1f}s", flush=True)
        return out
    except Exception as e:
        return {"ok": False, "event": str(ev), "error": str(e)}


# Global worker state for parallel hierarchical-PE scoring (event-level).
_HIER_WORKER_STATE: dict[str, Any] = {}


def _set_hier_worker_state(state: dict[str, Any]) -> None:
    global _HIER_WORKER_STATE
    _HIER_WORKER_STATE = dict(state)


def _apply_hier_null_worker(pe0: GWTCPeHierarchicalSamples, *, ev: str, mode: str, seed: int) -> GWTCPeHierarchicalSamples:
    mode = str(mode)
    if mode == "none":
        return pe0

    known = {"shuffle_dl", "shuffle_mc", "shuffle_dl_mc"}
    if mode not in known:
        raise ValueError(f"Unknown hier-null mode: {mode}")

    n = int(pe0.dL_mpc.size)
    rng = np.random.default_rng(_stable_int_seed(f"hier_null:{int(seed)}:{mode}:{ev}"))

    dL = np.asarray(pe0.dL_mpc, dtype=float)
    mc = np.asarray(pe0.chirp_mass_det, dtype=float)
    q = np.asarray(pe0.mass_ratio, dtype=float)
    log_pi_dL = np.asarray(pe0.log_pi_dL, dtype=float)
    log_pi_mc = np.asarray(pe0.log_pi_chirp_mass, dtype=float)
    log_pi_q = np.asarray(pe0.log_pi_mass_ratio, dtype=float)

    if mode in ("shuffle_dl", "shuffle_dl_mc"):
        p = rng.permutation(n)
        dL = dL[p]
        log_pi_dL = log_pi_dL[p]
    if mode in ("shuffle_mc", "shuffle_dl_mc"):
        p = rng.permutation(n)
        mc = mc[p]
        log_pi_mc = log_pi_mc[p]

    return GWTCPeHierarchicalSamples(
        file=str(pe0.file),
        analysis=str(pe0.analysis),
        n_total=int(pe0.n_total),
        n_used=int(pe0.n_used),
        dL_mpc=dL,
        chirp_mass_det=mc,
        mass_ratio=q,
        log_pi_dL=log_pi_dL,
        log_pi_chirp_mass=log_pi_mc,
        log_pi_mass_ratio=log_pi_q,
        prior_spec=dict(pe0.prior_spec),
    )


def _score_one_hier_event(ev: str) -> dict[str, Any]:
    """Worker: compute per-event hierarchical PE logL vectors + diagnostics.

    Uses `_HIER_WORKER_STATE` populated by the parent process. Intended for fork-based multiprocessing
    to avoid pickling large arrays (posterior + PE sample payloads).
    """
    st = _HIER_WORKER_STATE
    args: argparse.Namespace = st["args"]
    run_label: str = st["run_label"]
    mode_label: str = st["mode_label"]
    mode: str = st["mode"]
    z_max: float = float(st["z_max"])
    post = st["post"]
    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = st["pe_by_event"]
    log_alpha_mu: np.ndarray | None = st.get("log_alpha_mu")
    log_alpha_gr: np.ndarray | None = st.get("log_alpha_gr")

    try:
        pe0 = pe_by_event[str(ev)]
    except Exception:
        return {"ok": False, "event": str(ev), "error": "missing PE cache entry"}

    try:
        pe = _apply_hier_null_worker(pe0, ev=str(ev), mode=str(mode), seed=int(getattr(args, "hier_null_seed", 0)))
        res = compute_hierarchical_pe_logL_draws(
            pe=pe,
            post=post,
            convention=str(args.convention),  # type: ignore[arg-type]
            z_max=float(z_max),
            pop_z_mode=str(args.selection_pop_z_mode),  # type: ignore[arg-type]
            pop_z_k=float(args.selection_pop_z_k),
            pop_mass_mode=str(args.selection_pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(args.selection_pop_m1_alpha),
            pop_m_min=float(args.selection_pop_m_min),
            pop_m_max=float(args.selection_pop_m_max),
            pop_q_beta=float(args.selection_pop_q_beta),
            pop_m_taper_delta=float(args.selection_pop_m_taper_delta),
            pop_m_peak=float(args.selection_pop_m_peak),
            pop_m_peak_sigma=float(args.selection_pop_m_peak_sigma),
            pop_m_peak_frac=float(args.selection_pop_m_peak_frac),
            return_diagnostics=True,
        )
    except Exception as e:
        return {"ok": False, "event": str(ev), "error": str(e)}

    logL_mu = np.asarray(getattr(res, "logL_mu"), dtype=float)
    logL_gr = np.asarray(getattr(res, "logL_gr"), dtype=float)
    if logL_mu.size == 0 or logL_gr.size == 0 or not np.any(np.isfinite(logL_mu)) or not np.any(np.isfinite(logL_gr)):
        return {"ok": False, "event": str(ev), "error": f"hierarchical logL is all -inf (z_max={float(z_max):.3f})"}

    ess_mu_min = float(np.nanmin(np.asarray(getattr(res, "ess_mu"), dtype=float)))
    ess_gr_min = float(np.nanmin(np.asarray(getattr(res, "ess_gr"), dtype=float)))
    good_mu = float(np.nanmedian(np.asarray(getattr(res, "n_good_mu"), dtype=float) / float(getattr(res, "n_samples", 1))))
    good_gr = float(np.nanmedian(np.asarray(getattr(res, "n_good_gr"), dtype=float) / float(getattr(res, "n_samples", 1))))

    min_good = float(getattr(args, "hier_min_good_frac", 0.0))
    min_ess = float(getattr(args, "hier_min_ess", 0.0))
    bad_reasons: list[str] = []
    if min_good > 0.0 and (good_mu < min_good or good_gr < min_good):
        bad_reasons.append(f"good_frac(mu,gr)=({good_mu:.4g},{good_gr:.4g}) < {min_good:.4g}")
    if min_ess > 0.0 and (ess_mu_min < min_ess or ess_gr_min < min_ess):
        bad_reasons.append(f"ess_min(mu,gr)=({ess_mu_min:.3g},{ess_gr_min:.3g}) < {min_ess:.3g}")
    if bad_reasons and str(getattr(args, "hier_bad_sample_mode", "warn")) == "skip":
        return {
            "ok": False,
            "event": str(ev),
            "error": f"hierarchical diagnostics below threshold: {'; '.join(bad_reasons)} -> skip",
        }

    lpd_mu_data = float(_logmeanexp(logL_mu))
    lpd_gr_data = float(_logmeanexp(logL_gr))
    if log_alpha_mu is not None and log_alpha_gr is not None:
        lpd_mu = float(_logmeanexp(logL_mu - log_alpha_mu))
        lpd_gr = float(_logmeanexp(logL_gr - log_alpha_gr))
    else:
        lpd_mu = float(lpd_mu_data)
        lpd_gr = float(lpd_gr_data)

    def _safe_diff(a: float, b: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)):
            return float("nan")
        return float(a - b)

    row = {
        "mode": str(mode_label),
        "event": str(ev),
        "pe_file": str(getattr(pe0, "file", "")),
        "pe_analysis": str(getattr(pe0, "analysis", "")),
        "n_pe_samples": int(getattr(pe0, "n_used", 0)),
        "pe_weight_ess_mu_p50": float(np.nanmedian(getattr(res, "ess_mu"))),
        "pe_weight_ess_gr_p50": float(np.nanmedian(getattr(res, "ess_gr"))),
        "pe_weight_ess_mu_min": float(ess_mu_min),
        "pe_weight_ess_gr_min": float(ess_gr_min),
        "pe_good_frac_mu_p50": float(good_mu),
        "pe_good_frac_gr_p50": float(good_gr),
        "z_max": float(z_max),
        "lpd_mu_data": float(lpd_mu_data),
        "lpd_gr_data": float(lpd_gr_data),
        "delta_lpd_data": _safe_diff(float(lpd_mu_data), float(lpd_gr_data)),
        "lpd_mu": float(lpd_mu),
        "lpd_gr": float(lpd_gr),
        "delta_lpd": _safe_diff(float(lpd_mu), float(lpd_gr)),
        "lpd_mu_sel": _safe_diff(float(lpd_mu), float(lpd_mu_data)),
        "lpd_gr_sel": _safe_diff(float(lpd_gr), float(lpd_gr_data)),
        "delta_lpd_sel": _safe_diff(_safe_diff(float(lpd_mu), float(lpd_gr)), _safe_diff(float(lpd_mu_data), float(lpd_gr_data))),
    }
    # Return the per-draw stacks for total aggregation + selection-sensitivity caching.
    return {"ok": True, "event": str(ev), "row": row, "logL_mu": logL_mu, "logL_gr": logL_gr, "bad_reasons": bad_reasons}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dark/statistical siren GW-vs-EM propagation scoring using 3D skymaps + galaxy catalog.")
    ap.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Finished EM-only reconstruction run dir (contains samples/mu_forward_posterior.npz). Repeatable.",
    )
    ap.add_argument("--skymap-dir", required=True, help="Directory containing extracted GW skymaps (FITS[.gz]) and optionally skyLocalizationFileList.csv.")
    ap.add_argument(
        "--glade-index",
        default=None,
        help=(
            "Prepared GLADE+ HEALPix index dir (run scripts/build_gladeplus_index.py).\n"
            "Required unless --gw-data-mode=pe --pe-like-mode=hierarchical."
        ),
    )
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/dark_siren_gap_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="R_GW/EM convention.")
    ap.add_argument("--p-credible", type=float, default=0.9, help="Sky credible region to include (default: 0.9).")
    ap.add_argument("--gal-z-max", type=float, default=0.3, help="Max galaxy redshift to include (default: 0.3).")
    ap.add_argument("--max-area-deg2", type=float, default=2_000.0, help="Skip events with sky area above this (deg^2).")
    ap.add_argument("--max-gal", type=int, default=250_000, help="Skip events with >max galaxies in selection (default: 250k).")
    ap.add_argument("--max-events", type=int, default=25, help="Max number of events to score (default: 25).")
    ap.add_argument("--events", default=None, help="Comma-separated list of event names to score (default: auto-select).")
    ap.add_argument("--max-draws", type=int, default=256, help="Max mu-posterior draws to use (default: 256).")
    ap.add_argument(
        "--n-proc",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for per-event scoring within each run_dir (default: 1). "
            "Use 0 for all available cores (respects taskset/CPU affinity when possible)."
        ),
    )
    ap.add_argument(
        "--scale-up",
        action="store_true",
        help="Convenience preset to relax cuts (if defaults are still in place): max_area_deg2=5000, max_gal=1e6, max_events=200.",
    )
    ap.add_argument(
        "--galaxy-chunk-size",
        type=int,
        default=50_000,
        help="Galaxy chunk size used in per-event likelihood evaluation (default: 50000). Lower to reduce peak memory.",
    )

    # GW sky-map distance prior (used to convert p(Ω,dL|data) into a proxy likelihood by dividing by π(dL)).
    ap.add_argument(
        "--gw-distance-prior-mode",
        choices=["auto", "none", "dL_powerlaw", "comoving_lcdm", "comoving_lcdm_sourceframe", "pe_analytic"],
        default="auto",
        help=(
            "Distance prior model used to convert the public 3D sky-map posterior density into a proxy likelihood.\n"
            "  - auto: comoving_lcdm_sourceframe if skymap filename contains 'cosmo_reweight', else dL_powerlaw(k=2)\n"
            "  - dL_powerlaw: π(dL) ∝ dL^k (use --gw-distance-prior-power)\n"
            "  - comoving_lcdm: π(dL) induced by LCDM comoving-volume prior (fixed H0_ref/Ωm/Ωk)\n"
            "  - comoving_lcdm_sourceframe: same but includes 1/(1+z) time-dilation factor\n"
            "  - pe_analytic: divide out the event's analytic PE prior π_PE(dL) from the PEDataRelease file (PE modes only)\n"
            "  - none: do not divide out any distance prior (not recommended)\n"
        ),
    )
    ap.add_argument("--gw-distance-prior-power", type=float, default=2.0, help="k for --gw-distance-prior-mode=dL_powerlaw (default 2).")
    ap.add_argument("--gw-distance-prior-h0-ref", type=float, default=67.7, help="H0_ref for comoving LCDM distance prior (default 67.7).")
    ap.add_argument("--gw-distance-prior-omega-m0", type=float, default=0.31, help="Ωm0 for comoving LCDM distance prior (default 0.31).")
    ap.add_argument("--gw-distance-prior-omega-k0", type=float, default=0.0, help="Ωk0 for comoving LCDM distance prior (default 0).")
    ap.add_argument("--gw-distance-prior-zmax", type=float, default=10.0, help="zmax for comoving LCDM distance-prior cache (default 10).")

    # Use real PE posterior samples (GWTC PEDataRelease) instead of 3D sky-map FITS products.
    ap.add_argument(
        "--gw-data-mode",
        choices=["skymap", "pe"],
        default="skymap",
        help="GW data source: 'skymap' (FITS 3D skymaps) or 'pe' (PE posterior samples; recommended for production).",
    )
    ap.add_argument("--pe-base-dir", default="data/cache/gw/zenodo", help="Base dir containing Zenodo record subdirs for PE files.")
    ap.add_argument(
        "--pe-record-id",
        action="append",
        default=["5546663"],
        help="Zenodo record id(s) to search for PE files (repeatable). Default: 5546663 (GWTC-3 PEDataRelease).",
    )
    ap.add_argument(
        "--pe-prefer-variant",
        action="append",
        default=["mixed_nocosmo", "combined", "mixed_cosmo"],
        help="PE file variant preference order (repeatable). Default: mixed_nocosmo, combined, mixed_cosmo.",
    )
    ap.add_argument("--pe-analysis", default=None, help="Analysis group label inside PE file (default: auto preference).")
    ap.add_argument("--pe-max-samples", type=int, default=None, help="Optional cap on PE samples per event (random, deterministic).")
    ap.add_argument("--pe-seed", type=int, default=0, help="Seed for PE downsampling (default 0).")
    ap.add_argument("--pe-nside", type=int, default=64, help="HEALPix nside for PE-based sky/distance binning (default 64).")
    ap.add_argument("--pe-dl-nbins", type=int, default=64, help="Number of log-distance bins for PE-based p(dL|pix) (default 64).")
    ap.add_argument("--pe-dl-qmin", type=float, default=0.001, help="Lower quantile used to set auto dL bin min (default 0.001).")
    ap.add_argument("--pe-dl-qmax", type=float, default=0.999, help="Upper quantile used to set auto dL bin max (default 0.999).")
    ap.add_argument("--pe-dl-pad-factor", type=float, default=1.2, help="Padding factor applied to auto dL bin min/max (default 1.2).")
    ap.add_argument("--pe-dl-pseudocount", type=float, default=0.05, help="Pseudocount added to each per-pixel distance bin (default 0.05).")
    ap.add_argument("--pe-dl-smooth-iters", type=int, default=2, help="Smoothing iterations for per-pixel distance histograms (default 2).")
    ap.add_argument(
        "--pe-like-mode",
        choices=["hist", "hierarchical"],
        default="hist",
        help=(
            "How to use PE posterior samples when --gw-data-mode=pe.\n"
            "  - hist: build a (pix, dL) histogram proxy and divide out an assumed distance prior (legacy; fast).\n"
            "  - hierarchical: hierarchical PE-sample reweighting using the *event's* analytic PE priors\n"
            "    (dL + chirp_mass + q) and a population model with mass–redshift coupling (recommended).\n"
            "    This mode does not use the galaxy catalog (it targets the spectral/selection mechanism).\n"
        ),
    )
    ap.add_argument(
        "--pe-distance-mode",
        choices=["full", "spectral_only", "prior_only", "sky_only"],
        default="full",
        help=(
            "For --gw-data-mode=pe, how to use the PE (Ω, dL) posterior approximation.\n"
            "  - full: use per-pixel conditional distance histograms p(dL | pix, data) (default).\n"
            "  - spectral_only: replace p(dL | pix, data) with the sky-marginal p(dL | data) (independent of pix),\n"
            "    while keeping the sky weights prob_pix. This removes sky–distance correlation as a control.\n"
            "  - prior_only: replace p(dL | pix, data) with the assumed distance prior π(dL) normalized on the histogram support.\n"
            "    After dividing by π(dL), this becomes a distance-killing null for the catalog term (forensic control).\n"
            "  - sky_only: ignore distance entirely and use only sky weights (w_gal * prob_pix). This is a strict\n"
            "    distance-destruction null that also avoids histogram-support edge artifacts (forensic control).\n"
        ),
    )

    # Null tests / controls.
    ap.add_argument(
        "--null-mode",
        choices=["none", "rotate_pe_sky"],
        default="none",
        help=(
            "Null-test mode.\n"
            "  - none: normal scoring.\n"
            "  - rotate_pe_sky: for --gw-data-mode=pe only, apply a deterministic random sky rotation to each event's (ra,dec) PE samples before binning.\n"
        ),
    )
    ap.add_argument("--null-seed", type=int, default=0, help="Base seed for null tests (default 0). Each event uses a derived deterministic seed.")

    # Galaxy/catalog null tests (for the catalog-based likelihood only).
    ap.add_argument(
        "--galaxy-null-mode",
        choices=["none", "shuffle_zw", "shuffle_z"],
        default="none",
        help=(
            "Galaxy/catalog null-test mode.\n"
            "  - none: normal catalog likelihood.\n"
            "  - shuffle_zw: jointly permute (z,w) across galaxies while keeping sky pixels fixed.\n"
            "  - shuffle_z: permute z only (keeps w fixed).\n"
            "This is intended to break distance–redshift information in the catalog term while preserving sky footprint."
        ),
    )
    ap.add_argument("--galaxy-null-seed", type=int, default=0, help="Seed for --galaxy-null-mode permutations (default 0).")

    # Hierarchical-PE controls (only for --gw-data-mode=pe --pe-like-mode=hierarchical).
    ap.add_argument(
        "--hier-null-mode",
        choices=["none", "shuffle_dl", "shuffle_mc", "shuffle_dl_mc"],
        default="none",
        help=(
            "Hierarchical-PE null/control mode.\n"
            "  - none: normal hierarchical scoring.\n"
            "  - shuffle_dl: permute dL samples (and matching dL prior values) to break dL–mass correlations.\n"
            "  - shuffle_mc: permute chirp-mass samples (and matching mass prior values) to break mass–distance correlations.\n"
            "  - shuffle_dl_mc: apply both shuffles (independent permutations).\n"
        ),
    )
    ap.add_argument("--hier-null-seed", type=int, default=0, help="Seed for --hier-null-mode permutations (default 0).")
    ap.add_argument(
        "--hier-control-battery",
        action="store_true",
        help=(
            "Run a tiny hierarchical control battery (seconds-minutes): modes {real, shuffle_dl, shuffle_mc} on a small subset.\n"
            "Uses --hier-control-battery-max-events and --hier-control-battery-max-draws.\n"
        ),
    )
    ap.add_argument("--hier-control-battery-max-events", type=int, default=1, help="Max events for --hier-control-battery (default 1).")
    ap.add_argument("--hier-control-battery-max-draws", type=int, default=8, help="Max posterior draws for --hier-control-battery (default 8).")
    ap.add_argument(
        "--hier-control-battery-max-pe-samples",
        type=int,
        default=5000,
        help="Default PE sample cap used by --hier-control-battery when --pe-max-samples is not set (default 5000).",
    )
    ap.add_argument(
        "--hier-min-good-frac",
        type=float,
        default=0.0,
        help="Hierarchical mode: minimum median usable-sample fraction required per event (default 0; disables).",
    )
    ap.add_argument(
        "--hier-min-ess",
        type=float,
        default=0.0,
        help="Hierarchical mode: minimum ESS(min over draws) required per event (default 0; disables).",
    )
    ap.add_argument(
        "--hier-bad-sample-mode",
        choices=["warn", "skip"],
        default="warn",
        help="Hierarchical mode: action when an event violates --hier-min-* thresholds (default warn).",
    )

    # Selection normalization alpha(model) (optional).
    ap.add_argument(
        "--selection-injections-hdf",
        default="auto",
        help=(
            "GW injection summary HDF for selection normalization alpha(model).\n"
            "  - auto: choose an era-aware injection file (O3/O4) based on the event set.\n"
            "          O4 requires either --selection-o4-injections-hdf or --selection-auto-record-id-o4.\n"
            "  - none: disable selection normalization.\n"
            "  - path: explicit .hdf/.hdf5 injection summary file.\n"
            "Examples:\n"
            "  - auto\n"
            "  - none\n"
            "  - data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/injections/o3a_bbhpop_inj_info.hdf"
        ),
    )
    ap.add_argument(
        "--selection-o4-injections-hdf",
        default=None,
        help=(
            "Optional explicit O4 injections file used when --selection-injections-hdf=auto "
            "and events are O4-era."
        ),
    )
    ap.add_argument(
        "--selection-auto-record-id-o3",
        type=int,
        default=7890437,
        help="Zenodo record id used for O3 auto-resolution (default 7890437).",
    )
    ap.add_argument(
        "--selection-auto-record-id-o4",
        type=int,
        default=None,
        help="Zenodo record id used for O4 auto-resolution (default none; required for O4 auto unless --selection-o4-injections-hdf is set).",
    )
    ap.add_argument(
        "--selection-auto-strict",
        dest="selection_auto_strict",
        action="store_true",
        help="Fail when auto-resolution cannot map event eras cleanly (default).",
    )
    ap.add_argument(
        "--no-selection-auto-strict",
        dest="selection_auto_strict",
        action="store_false",
        help="Allow legacy fallback behavior for ambiguous/other-era event sets.",
    )
    ap.set_defaults(selection_auto_strict=True)
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold in years for 'found' injections (default 1).")
    ap.add_argument("--selection-z-max", type=float, default=None, help="Max injection redshift used in alpha(model) (default: min(gal-z-max, post.z_grid.max)).")
    ap.add_argument(
        "--selection-det-model",
        choices=["threshold", "snr_binned", "injection_logit"],
        default="snr_binned",
        help=(
            "Detection model used in alpha(model).\n"
            "  - threshold: hard SNR threshold (if --selection-snr-thresh omitted, calibrates to match found_ifar)\n"
            "  - snr_binned: empirical monotone p_det(SNR) curve from injections\n"
            "  - injection_logit: injection-trained logistic p_det(logSNR, mass, z) (recommended)\n"
        ),
    )
    ap.add_argument(
        "--selection-snr-thresh",
        type=float,
        default=None,
        help="Manual network SNR threshold (only used if --selection-det-model=threshold).",
    )
    ap.add_argument(
        "--selection-snr-binned-nbins",
        type=int,
        default=200,
        help="Number of SNR quantile bins for --selection-det-model=snr_binned (default 200).",
    )
    ap.add_argument(
        "--selection-injection-logit-l2",
        type=float,
        default=1e-2,
        help="L2 regularization strength for --selection-det-model=injection_logit.",
    )
    ap.add_argument(
        "--selection-injection-logit-max-iter",
        type=int,
        default=64,
        help="Maximum IRLS iterations for --selection-det-model=injection_logit.",
    )
    ap.add_argument(
        "--selection-weight-mode",
        choices=["none", "inv_sampling_pdf"],
        default="none",
        help=(
            "How to weight injections when estimating alpha(model).\n"
            "  - none: unweighted mean over injections (assumes injection draw distribution is the target)\n"
            "  - inv_sampling_pdf: importance-weight by 1/sampling_pdf (partially corrects for injection sampling)\n"
        ),
    )
    ap.add_argument(
        "--selection-pop-z-mode",
        choices=["none", "comoving_uniform", "comoving_powerlaw"],
        default="none",
        help=(
            "Optional population redshift prior factor for alpha(model), applied *in addition* to selection-weight-mode.\n"
            "  - comoving_uniform: p(z) ∝ dV_c/dz / (1+z)\n"
            "  - comoving_powerlaw: p(z) ∝ dV_c/dz / (1+z) * (1+z)^k\n"
        ),
    )
    ap.add_argument("--selection-pop-z-k", type=float, default=0.0, help="k for comoving_powerlaw (default 0).")
    ap.add_argument(
        "--selection-pop-mass-mode",
        choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="none",
        help=(
            "Optional population mass prior factor for alpha(model), applied *in addition* to selection-weight-mode.\n"
            "  - powerlaw_q: p(m1,m2) ∝ m1^{-alpha} q^{beta_q} on [m_min,m_max] (hard cut)\n"
            "  - powerlaw_q_smooth: same but with smooth tapers at (m_min,m_max) to avoid hard-support truncation\n"
            "  - powerlaw_peak_q_smooth: powerlaw_q_smooth + Gaussian peak in m1 with mixture fraction\n"
        ),
    )
    ap.add_argument("--selection-pop-m1-alpha", type=float, default=2.3, help="Primary-mass power-law slope alpha (default 2.3).")
    ap.add_argument("--selection-pop-m-min", type=float, default=5.0, help="Mass lower bound in Msun (default 5).")
    ap.add_argument("--selection-pop-m-max", type=float, default=80.0, help="Mass upper bound in Msun (default 80).")
    ap.add_argument("--selection-pop-q-beta", type=float, default=0.0, help="Mass-ratio power beta_q (default 0).")
    ap.add_argument(
        "--selection-pop-m-taper-delta",
        type=float,
        default=0.0,
        help="Smooth taper width (Msun) for --selection-pop-mass-mode=powerlaw_q_smooth (default 0).",
    )
    ap.add_argument("--selection-pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for --selection-pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--selection-pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for --selection-pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--selection-pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for --selection-pop-mass-mode=powerlaw_peak_q_smooth.")

    # Catalog incompleteness handling (optional, weight-based).
    ap.add_argument(
        "--completeness-mode",
        choices=["none", "z3_cumulative"],
        default="none",
        help=(
            "Catalog incompleteness handling.\n"
            "  - none: no correction\n"
            "  - z3_cumulative: estimate a cumulative completeness C(z) ~ X(<z)/z^3 from the catalog and upweight w by 1/C(z).\n"
            "    If the catalog index stores weights (e.g. luminosity), uses weighted X; otherwise counts."
        ),
    )
    ap.add_argument("--completeness-zref-max", type=float, default=0.02, help="Reference redshift for completeness normalization (default 0.02).")
    ap.add_argument("--completeness-nbins", type=int, default=300, help="Number of redshift bins for completeness curve (default 300).")
    ap.add_argument("--completeness-c-floor", type=float, default=0.05, help="Floor for completeness C(z) to avoid infinite weights (default 0.05).")

    # Out-of-catalog incompleteness mixture model (recommended).
    ap.add_argument(
        "--mixture-mode",
        choices=["none", "simple"],
        default="none",
        help=(
            "Catalog incompleteness handling via an explicit mixture likelihood.\n"
            "  - none: use in-catalog likelihood only (optionally with legacy --completeness-mode reweighting)\n"
            "  - simple: L = (1-f_miss) L_cat + f_miss L_missing (recommended)\n"
        ),
    )
    ap.add_argument(
        "--mixture-f-miss-mode",
        choices=["fixed", "from_completeness", "marginalize"],
        default="from_completeness",
        help="How to choose f_miss (missing-host mixture fraction).",
    )
    ap.add_argument(
        "--mixture-f-miss",
        type=float,
        default=None,
        help="Missing-host fraction f_miss in [0,1] (required if --mixture-f-miss-mode=fixed).",
    )
    ap.add_argument(
        "--mixture-f-miss-prior",
        choices=["uniform", "beta"],
        default="beta",
        help="Prior on f_miss when --mixture-f-miss-mode=marginalize (default: beta).",
    )
    ap.add_argument(
        "--mixture-f-miss-beta-mean",
        type=float,
        default=None,
        help="Mean for beta prior on f_miss (default: use completeness-derived f_miss estimate).",
    )
    ap.add_argument(
        "--mixture-f-miss-beta-kappa",
        type=float,
        default=8.0,
        help="Concentration for beta prior on f_miss (default: 8). Larger = tighter around the mean.",
    )
    ap.add_argument(
        "--mixture-f-miss-marginalize-n",
        type=int,
        default=401,
        help="Number of f_miss grid points for marginalization (default: 401).",
    )
    ap.add_argument(
        "--mixture-f-miss-marginalize-eps",
        type=float,
        default=1e-6,
        help="Epsilon to avoid evaluating priors at f=0 or f=1 (default: 1e-6).",
    )
    ap.add_argument(
        "--host-prior-z-mode",
        choices=["none", "comoving_uniform", "comoving_powerlaw"],
        default="comoving_uniform",
        help=(
            "Host density prior for the missing-host term (per comoving volume).\n"
            "  - comoving_uniform: rho_host(z) ∝ 1\n"
            "  - comoving_powerlaw: rho_host(z) ∝ (1+z)^k\n"
        ),
    )
    ap.add_argument("--host-prior-z-k", type=float, default=0.0, help="k exponent for --host-prior-z-mode=comoving_powerlaw (default 0).")
    ap.add_argument(
        "--missing-z-max",
        type=float,
        default=None,
        help="Max redshift support for the missing-host integral (default: gal-z-max).",
    )
    ap.add_argument("--missing-dl-grid-n", type=int, default=200, help="Number of dL grid points for missing-host integral (default 200).")
    ap.add_argument("--missing-dl-min-mpc", type=float, default=1.0, help="Min dL (Mpc) for missing-host integral grid (default 1).")
    ap.add_argument(
        "--missing-dl-max-mpc",
        type=float,
        default=None,
        help="Max dL (Mpc) for missing-host integral grid (default: derived from skymap distmu+nsigma*distsigma within the CR).",
    )
    ap.add_argument("--missing-dl-nsigma", type=float, default=5.0, help="Nsigma used to set default missing dL max from skymap distance layers (default 5).")
    ap.add_argument("--missing-pixel-chunk-size", type=int, default=5000, help="Chunk size (pixels) for missing-host integral (default 5000).")

    # Published-control GR baseline (H0 inference plumbing).
    ap.add_argument(
        "--gr-h0-mode",
        choices=["none", "grid"],
        default="none",
        help="Optional GR baseline mode: compute an H0 posterior on a grid using cached events.",
    )
    ap.add_argument("--gr-h0-grid-min", type=float, default=50.0, help="Min H0 for GR H0 grid (default 50).")
    ap.add_argument("--gr-h0-grid-max", type=float, default=90.0, help="Max H0 for GR H0 grid (default 90).")
    ap.add_argument("--gr-h0-grid-n", type=int, default=81, help="Number of H0 grid points for GR baseline (default 81).")
    ap.add_argument("--gr-h0-omega-m0", type=float, default=0.31, help="Omega_m0 used for GR H0 baseline distances (default 0.31).")
    ap.add_argument("--gr-h0-omega-k0", type=float, default=0.0, help="Omega_k0 used for GR H0 baseline distances (default 0).")
    ap.add_argument("--gr-h0-prior", choices=["uniform"], default="uniform", help="Prior on H0 for the GR baseline grid (default uniform).")
    ap.add_argument("--gr-h0-smoke", action="store_true", help="Tiny GR H0 smoke mode (shrinks grid and event count for a seconds-scale check).")
    args = ap.parse_args()

    # Convenience presets / resource knobs.
    if int(args.n_proc) <= 0:
        args.n_proc = _default_nproc()
    if bool(args.scale_up):
        # Only override if the user left defaults in place.
        if float(args.max_area_deg2) == 2_000.0:
            args.max_area_deg2 = 5_000.0
        if int(args.max_gal) == 250_000:
            args.max_gal = 1_000_000
        if int(args.max_events) == 25:
            args.max_events = 200

    if str(args.gw_data_mode) != "pe":
        if str(args.pe_distance_mode) != "full":
            raise ValueError("--pe-distance-mode is only valid with --gw-data-mode=pe.")
        if str(args.pe_like_mode) != "hist":
            raise ValueError("--pe-like-mode is only valid with --gw-data-mode=pe.")
        if str(args.hier_null_mode) != "none" or bool(args.hier_control_battery):
            raise ValueError("--hier-* options require --gw-data-mode=pe --pe-like-mode=hierarchical.")
    else:
        # Hierarchical mode does not use pixel-conditioned distance histograms or sky rotations.
        if str(args.pe_like_mode) == "hierarchical":
            if str(args.pe_distance_mode) != "full":
                raise ValueError("--pe-distance-mode is not applicable with --pe-like-mode=hierarchical.")
            if str(args.null_mode) != "none":
                raise ValueError("--null-mode is not applicable with --pe-like-mode=hierarchical.")
            if str(args.galaxy_null_mode) != "none":
                raise ValueError("--galaxy-null-mode is not applicable with --pe-like-mode=hierarchical.")
        else:
            if str(args.hier_null_mode) != "none" or bool(args.hier_control_battery):
                raise ValueError("--hier-* options require --pe-like-mode=hierarchical.")

    if bool(args.hier_control_battery) and str(args.hier_null_mode) != "none":
        raise ValueError("Cannot combine --hier-control-battery with --hier-null-mode.")

    run_dirs = args.run_dir or []
    if not run_dirs and str(args.gr_h0_mode) == "none":
        raise ValueError("Must provide at least one --run-dir unless --gr-h0-mode is enabled.")

    if args.missing_z_max is None:
        args.missing_z_max = float(args.gal_z_max)

    hierarchical_pe_mode = str(args.gw_data_mode) == "pe" and str(args.pe_like_mode) == "hierarchical"
    if not hierarchical_pe_mode and args.glade_index is None:
        raise ValueError("--glade-index is required unless --gw-data-mode=pe --pe-like-mode=hierarchical.")

    out_dir = Path(args.out) if args.out else Path("outputs") / f"dark_siren_gap_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    cache_dir = out_dir / "cache"
    cache_missing_dir = out_dir / "cache_missing"
    cache_gr_h0_dir = out_dir / "cache_gr_h0"
    cache_terms_dir = out_dir / "cache_terms"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_missing_dir.mkdir(parents=True, exist_ok=True)
    cache_gr_h0_dir.mkdir(parents=True, exist_ok=True)
    cache_terms_dir.mkdir(parents=True, exist_ok=True)

    skymaps = _find_skymaps(Path(args.skymap_dir))

    want_events: list[str] | None = None
    if args.events:
        want_events = [s.strip() for s in str(args.events).split(",") if s.strip()]

    if str(args.mixture_mode) != "none" and str(args.completeness_mode) != "none":
        raise ValueError("Cannot use --completeness-mode with --mixture-mode. The mixture model replaces reweighting.")

    if str(args.mixture_f_miss_mode) == "marginalize" and str(args.gr_h0_mode) != "none":
        raise ValueError("Cannot use --mixture-f-miss-mode=marginalize with --gr-h0-mode. Use a fixed f_miss for GR H0.")

    if hierarchical_pe_mode:
        # Hierarchical PE-sample reweighting is a population/selection ("spectral") test and does not
        # use the galaxy catalog or the missing-host mixture model.
        if str(args.completeness_mode) != "none":
            raise ValueError("--completeness-mode is not applicable with --pe-like-mode=hierarchical.")
        if str(args.mixture_mode) != "none":
            raise ValueError("--mixture-mode is not applicable with --pe-like-mode=hierarchical.")

        pe_base_dir = Path(str(args.pe_base_dir)).expanduser().resolve()
        pe_record_ids = [int(x) for x in (args.pe_record_id or [])]
        pe_prefer_variants = [str(v) for v in (args.pe_prefer_variant or [])]
        pe_index = build_gwtc_pe_index(base_dir=pe_base_dir, record_ids=pe_record_ids or None)

        # Build + cache per-event hierarchical PE sample payloads (shared across run_dir seeds).
        pe_by_event: dict[str, Any] = {}
        cached_events: list[str] = []
        events_meta: list[dict[str, Any]] = []

        candidate_events: list[str]
        if want_events is not None:
            candidate_events = list(want_events)
            max_cache_events = int(len(candidate_events))
        else:
            candidate_events = sorted(skymaps.keys())
            max_cache_events = int(args.max_events)

        if bool(args.hier_control_battery):
            max_cache_events = int(args.hier_control_battery_max_events)
            candidate_events = candidate_events[:max_cache_events]
        if max_cache_events <= 0:
            raise ValueError("max_cache_events must be positive.")

        pe_max_samples_eff: int | None = int(args.pe_max_samples) if args.pe_max_samples is not None else None
        if bool(args.hier_control_battery) and pe_max_samples_eff is None:
            pe_max_samples_eff = int(args.hier_control_battery_max_pe_samples)
        if pe_max_samples_eff is not None and pe_max_samples_eff <= 0:
            raise ValueError("pe_max_samples must be positive when provided.")
        for ev in candidate_events:
            if len(cached_events) >= max_cache_events:
                break

            if ev not in pe_index:
                print(f"[dark_sirens] skip {ev}: no PEDataRelease file under {pe_base_dir}", flush=True)
                continue

            cache_path = cache_dir / f"pe_hier_{ev}.npz"
            want_meta = {
                "event": str(ev),
                "gw_data_mode": "pe",
                "pe_like_mode": "hierarchical",
                "pe_base_dir": str(pe_base_dir),
                "pe_record_ids": pe_record_ids,
                "pe_prefer_variants": pe_prefer_variants,
                "pe_analysis": str(args.pe_analysis) if args.pe_analysis is not None else None,
                "pe_max_samples": int(pe_max_samples_eff) if pe_max_samples_eff is not None else None,
                "pe_seed": int(args.pe_seed),
            }

            if cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=True) as d:
                        meta = json.loads(str(d["meta"].tolist()))
                        ok = all(meta.get(k) == want_meta.get(k) for k in want_meta.keys())
                        if ok:
                            pe_obj = GWTCPeHierarchicalSamples(
                                file=str(meta.get("pe_file")),
                                analysis=str(meta.get("pe_analysis_chosen")),
                                n_total=int(meta.get("n_total", -1)),
                                n_used=int(meta.get("n_used", -1)),
                                dL_mpc=np.asarray(d["dL_mpc"], dtype=float),
                                chirp_mass_det=np.asarray(d["chirp_mass_det"], dtype=float),
                                mass_ratio=np.asarray(d["mass_ratio"], dtype=float),
                                log_pi_dL=np.asarray(d["log_pi_dL"], dtype=float),
                                log_pi_chirp_mass=np.asarray(d["log_pi_chirp_mass"], dtype=float),
                                log_pi_mass_ratio=np.asarray(d["log_pi_mass_ratio"], dtype=float),
                                prior_spec=json.loads(str(meta.get("prior_spec_json", "{}"))),
                            )
                            pe_by_event[ev] = pe_obj
                            cached_events.append(ev)
                            events_meta.append(
                                {
                                    "event": str(ev),
                                    "pe_file": str(meta.get("pe_file")),
                                    "analysis": str(meta.get("pe_analysis_chosen")),
                                    "n_total": int(meta.get("n_total", -1)),
                                    "n_used": int(meta.get("n_used", -1)),
                                }
                            )
                            continue
                except Exception:
                    pass

            files = pe_index[ev]
            ordered: list[Any] = []
            seen: set[str] = set()
            for v in pe_prefer_variants:
                for f in files:
                    if getattr(f, "variant", None) == v:
                        p = str(getattr(f, "path"))
                        if p not in seen:
                            ordered.append(f)
                            seen.add(p)
            for f in files:
                p = str(getattr(f, "path"))
                if p not in seen:
                    ordered.append(f)
                    seen.add(p)

            last_err: Exception | None = None
            pe_obj = None
            pe_file = None
            for cand in ordered:
                pe_try = (pe_base_dir / str(getattr(cand, "path"))).resolve()
                try:
                    pe_obj = load_gwtc_pe_hierarchical_samples(
                        path=pe_try,
                        analysis=str(args.pe_analysis) if args.pe_analysis is not None else None,
                        max_samples=int(pe_max_samples_eff) if pe_max_samples_eff is not None else None,
                        seed=_stable_int_seed(f"pe_hier:{int(args.pe_seed)}:{ev}"),
                    )
                    pe_file = pe_try
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if pe_obj is None or pe_file is None:
                print(f"[dark_sirens] skip {ev}: failed to read hierarchical PE samples ({last_err})", flush=True)
                continue

            meta2 = dict(want_meta)
            meta2.update(
                {
                    "pe_file": str(pe_file),
                    "pe_analysis_chosen": str(pe_obj.analysis),
                    "n_total": int(pe_obj.n_total),
                    "n_used": int(pe_obj.n_used),
                    "prior_spec_json": json.dumps(pe_obj.prior_spec),
                }
            )
            np.savez(
                cache_path,
                meta=json.dumps(meta2),
                dL_mpc=np.asarray(pe_obj.dL_mpc, dtype=np.float64),
                chirp_mass_det=np.asarray(pe_obj.chirp_mass_det, dtype=np.float64),
                mass_ratio=np.asarray(pe_obj.mass_ratio, dtype=np.float64),
                log_pi_dL=np.asarray(pe_obj.log_pi_dL, dtype=np.float64),
                log_pi_chirp_mass=np.asarray(pe_obj.log_pi_chirp_mass, dtype=np.float64),
                log_pi_mass_ratio=np.asarray(pe_obj.log_pi_mass_ratio, dtype=np.float64),
            )

            pe_by_event[ev] = pe_obj
            cached_events.append(ev)
            events_meta.append(
                {
                    "event": str(ev),
                    "pe_file": str(pe_file),
                    "analysis": str(pe_obj.analysis),
                    "n_total": int(pe_obj.n_total),
                    "n_used": int(pe_obj.n_used),
                }
            )
            print(
                f"[dark_sirens] cached {ev}: pe_like=hierarchical n_samples={int(pe_obj.n_used):,} analysis={str(pe_obj.analysis)}",
                flush=True,
            )

        if not cached_events:
            raise RuntimeError("No events passed selection; nothing to score.")
        if want_events is not None:
            missing = [str(ev) for ev in want_events if str(ev) not in set(cached_events)]
            if missing:
                print(
                    f"[dark_sirens] NOTE: requested {int(len(want_events))} events but cached {int(len(cached_events))}; "
                    f"missing: {', '.join(missing)}",
                    flush=True,
                )

        _write_json(
            cache_dir / "cache_manifest.json",
            {
                "timestamp_utc": _utc_stamp(),
                "gw_data_mode": "pe",
                "pe_like_mode": "hierarchical",
                "pe": {
                    "base_dir": str(pe_base_dir),
                    "record_ids": pe_record_ids,
	                "prefer_variants": pe_prefer_variants,
	                "analysis": str(args.pe_analysis) if args.pe_analysis is not None else None,
	                "max_samples": int(pe_max_samples_eff) if pe_max_samples_eff is not None else None,
	                "seed": int(args.pe_seed),
	            },
                "events_filter": want_events,
                "events_cached": cached_events,
                "events_meta": events_meta,
            },
        )

        # Resolve + load selection injections (used for selection normalization alpha(model)).
        injections = None
        resolved_injections_path: Path | None = None
        inj_spec = str(args.selection_injections_hdf).strip()
        if inj_spec.lower() in ("none", "0", "false"):
            injections = None
            resolved_injections_path = None
            print("[dark_sirens] selection: disabled", flush=True)
        else:
            want_auto = inj_spec.lower() == "auto"
            if want_auto:
                resolved_injections_path = resolve_selection_injection_file(
                    events=[str(x) for x in cached_events],
                    base_dir="data/cache/gw/zenodo",
                    population="mixture",
                    auto_download=True,
                    record_id_o3=int(args.selection_auto_record_id_o3),
                    record_id_o4=(int(args.selection_auto_record_id_o4) if args.selection_auto_record_id_o4 is not None else None),
                    o4_injections_hdf=args.selection_o4_injections_hdf,
                    strict=bool(args.selection_auto_strict),
                )
            else:
                resolved_injections_path = Path(inj_spec).expanduser().resolve()
                if not resolved_injections_path.exists():
                    raise FileNotFoundError(f"Selection injections file not found: {resolved_injections_path}")

            assert resolved_injections_path is not None
            injections = load_o3_injections(
                resolved_injections_path,
                ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
            )
            print(
                f"[dark_sirens] selection: injections={resolved_injections_path.name} "
                f"ifar>{float(args.selection_ifar_thresh_yr):g}yr",
                flush=True,
            )

        # Optional published-control baseline: GR H0 posterior on a grid (hierarchical PE mode).
        if str(args.gr_h0_mode) == "grid":
            H0_grid = np.linspace(float(args.gr_h0_grid_min), float(args.gr_h0_grid_max), int(args.gr_h0_grid_n))
            use_events = list(cached_events)
            if bool(args.gr_h0_smoke):
                use_events = use_events[:1]
                H0_grid = np.linspace(65.0, 75.0, 11)

            pe_subset = {str(ev): pe_by_event[str(ev)] for ev in use_events}
            z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(args.gal_z_max)
            gr_res = compute_gr_h0_posterior_grid_hierarchical_pe(
                pe_by_event=pe_subset,
                H0_grid=H0_grid,
                omega_m0=float(args.gr_h0_omega_m0),
                omega_k0=float(args.gr_h0_omega_k0),
                z_max=float(z_sel),
                cache_dir=cache_gr_h0_dir,
                injections=injections,
                ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
                det_model=str(args.selection_det_model),  # type: ignore[arg-type]
                snr_threshold=float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
                snr_binned_nbins=int(args.selection_snr_binned_nbins),
                weight_mode=str(args.selection_weight_mode),  # type: ignore[arg-type]
                pop_z_mode=str(args.selection_pop_z_mode),  # type: ignore[arg-type]
                pop_z_powerlaw_k=float(args.selection_pop_z_k),
                pop_mass_mode=str(args.selection_pop_mass_mode),  # type: ignore[arg-type]
                pop_m1_alpha=float(args.selection_pop_m1_alpha),
                pop_m_min=float(args.selection_pop_m_min),
                pop_m_max=float(args.selection_pop_m_max),
                pop_q_beta=float(args.selection_pop_q_beta),
                pop_m_taper_delta=float(args.selection_pop_m_taper_delta),
                pop_m_peak=float(args.selection_pop_m_peak),
                pop_m_peak_sigma=float(args.selection_pop_m_peak_sigma),
                pop_m_peak_frac=float(args.selection_pop_m_peak_frac),
                prior=str(args.gr_h0_prior),  # type: ignore[arg-type]
            )

            out_path = out_dir / "gr_h0_grid_posterior_hierarchical_pe.json"
            _write_json(out_path, gr_res)
            print(f"[dark_sirens] GR H0 grid posterior (hier PE): wrote {out_path}", flush=True)

            try:
                H0g = np.asarray(gr_res.get("H0_grid", []), dtype=float)
                p = np.asarray(gr_res.get("posterior", []), dtype=float)
                if H0g.size >= 2 and p.size == H0g.size:
                    plt.figure(figsize=(7, 4))
                    plt.plot(H0g, p, color="C0", linewidth=2.0, label="GR posterior (hier PE, with selection α if enabled)")
                    plt.xlabel(r"$H_0$ [km/s/Mpc]")
                    plt.ylabel("posterior (arb. norm.)")
                    plt.title("GR $H_0$ grid posterior (hierarchical PE control)")
                    plt.legend(fontsize=8)
                    plt.tight_layout()
                    plt.savefig(fig_dir / "gr_h0_posterior_hierarchical_pe.png", dpi=160)
                    plt.close()
            except Exception:
                pass

        # Score each EM posterior seed against the same cached per-event PE samples.
        for rd in run_dirs:
            post_full = load_mu_forward_posterior(rd)
            run_label = Path(rd).name
            max_draws_eff = int(args.max_draws)
            if bool(args.hier_control_battery):
                max_draws_eff = min(max_draws_eff, int(args.hier_control_battery_max_draws))
            post, draw_idx = _downsample_posterior(post_full, max_draws=max_draws_eff, seed=_stable_int_seed(run_label))

            # Compute selection normalization alpha(draw) if requested.
            log_alpha_mu: np.ndarray | None = None
            log_alpha_gr: np.ndarray | None = None
            alpha_meta: dict[str, Any] | None = None
            if injections is not None:
                z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(post.z_grid[-1])
                alpha = compute_selection_alpha_from_injections(
                    post=post,
                    injections=injections,
                    convention=args.convention,  # type: ignore[arg-type]
                    z_max=z_sel,
                    snr_threshold=float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
                    det_model=str(args.selection_det_model),  # type: ignore[arg-type]
                    snr_binned_nbins=int(args.selection_snr_binned_nbins),
                    injection_logit_l2=float(args.selection_injection_logit_l2),
                    injection_logit_max_iter=int(args.selection_injection_logit_max_iter),
                    weight_mode=str(args.selection_weight_mode),  # type: ignore[arg-type]
                    pop_z_mode=str(args.selection_pop_z_mode),  # type: ignore[arg-type]
                    pop_z_powerlaw_k=float(args.selection_pop_z_k),
                    pop_mass_mode=str(args.selection_pop_mass_mode),  # type: ignore[arg-type]
                    pop_m1_alpha=float(args.selection_pop_m1_alpha),
                    pop_m_min=float(args.selection_pop_m_min),
                    pop_m_max=float(args.selection_pop_m_max),
                    pop_q_beta=float(args.selection_pop_q_beta),
                    pop_m_taper_delta=float(args.selection_pop_m_taper_delta),
                    pop_m_peak=float(args.selection_pop_m_peak),
                    pop_m_peak_sigma=float(args.selection_pop_m_peak_sigma),
                    pop_m_peak_frac=float(args.selection_pop_m_peak_frac),
                )
                log_alpha_mu = np.log(np.clip(alpha.alpha_mu, 1e-300, np.inf))
                log_alpha_gr = np.log(np.clip(alpha.alpha_gr, 1e-300, np.inf))
                alpha_meta = {
                    "method": alpha.method,
                    "z_max": alpha.z_max,
                    "snr_threshold": alpha.snr_threshold,
                    "n_injections_used": alpha.n_injections_used,
                    "det_model": getattr(alpha, "det_model", "unknown"),
                    "injection_logit_l2": float(args.selection_injection_logit_l2),
                    "injection_logit_max_iter": int(args.selection_injection_logit_max_iter),
                    "weight_mode": getattr(alpha, "weight_mode", "unknown"),
                    "pop_z_mode": str(args.selection_pop_z_mode),
                    "pop_z_k": float(args.selection_pop_z_k),
                    "pop_mass_mode": str(args.selection_pop_mass_mode),
                    "pop_m1_alpha": float(args.selection_pop_m1_alpha),
                    "pop_m_min": float(args.selection_pop_m_min),
                    "pop_m_max": float(args.selection_pop_m_max),
                    "pop_q_beta": float(args.selection_pop_q_beta),
                    "pop_m_taper_delta": float(args.selection_pop_m_taper_delta),
                    "pop_m_peak": float(args.selection_pop_m_peak),
                    "pop_m_peak_sigma": float(args.selection_pop_m_peak_sigma),
                    "pop_m_peak_frac": float(args.selection_pop_m_peak_frac),
                }
                np.savez(
                    tab_dir / f"selection_alpha_{run_label}.npz",
                    alpha_mu=np.asarray(alpha.alpha_mu, dtype=float),
                    alpha_gr=np.asarray(alpha.alpha_gr, dtype=float),
                    log_alpha_mu=np.asarray(log_alpha_mu, dtype=float),
                    log_alpha_gr=np.asarray(log_alpha_gr, dtype=float),
                    meta=json.dumps(alpha_meta, sort_keys=True),
                )

            def _safe_diff(a: float, b: float) -> float:
                if not (np.isfinite(a) and np.isfinite(b)):
                    return float("nan")
                return float(a - b)

            def _mode_label(mode: str) -> str:
                return "real" if str(mode) == "none" else str(mode)

            def _apply_hier_null(
                pe0: GWTCPeHierarchicalSamples,
                *,
                ev: str,
                mode: str,
            ) -> GWTCPeHierarchicalSamples:
                mode = str(mode)
                if mode == "none":
                    return pe0

                known = {"shuffle_dl", "shuffle_mc", "shuffle_dl_mc"}
                if mode not in known:
                    raise ValueError(f"Unknown hier-null mode: {mode}")

                n = int(pe0.dL_mpc.size)
                rng = np.random.default_rng(_stable_int_seed(f"hier_null:{int(args.hier_null_seed)}:{mode}:{ev}"))

                dL = np.asarray(pe0.dL_mpc, dtype=float)
                mc = np.asarray(pe0.chirp_mass_det, dtype=float)
                q = np.asarray(pe0.mass_ratio, dtype=float)
                log_pi_dL = np.asarray(pe0.log_pi_dL, dtype=float)
                log_pi_mc = np.asarray(pe0.log_pi_chirp_mass, dtype=float)
                log_pi_q = np.asarray(pe0.log_pi_mass_ratio, dtype=float)

                if mode in ("shuffle_dl", "shuffle_dl_mc"):
                    p = rng.permutation(n)
                    dL = dL[p]
                    log_pi_dL = log_pi_dL[p]
                if mode in ("shuffle_mc", "shuffle_dl_mc"):
                    p = rng.permutation(n)
                    mc = mc[p]
                    log_pi_mc = log_pi_mc[p]

                return GWTCPeHierarchicalSamples(
                    file=str(pe0.file),
                    analysis=str(pe0.analysis),
                    n_total=int(pe0.n_total),
                    n_used=int(pe0.n_used),
                    dL_mpc=dL,
                    chirp_mass_det=mc,
                    mass_ratio=q,
                    log_pi_dL=log_pi_dL,
                    log_pi_chirp_mass=log_pi_mc,
                    log_pi_mass_ratio=log_pi_q,
                    prior_spec=dict(pe0.prior_spec),
                )

            def _score_hier_mode(*, mode: str, events: list[str], z_max: float) -> dict[str, Any]:
                mode_label = _mode_label(mode)
                n_draws = int(post.H_samples.shape[0])
                logL_mu_total_data = np.zeros((n_draws,), dtype=float)
                logL_gr_total_data = np.zeros((n_draws,), dtype=float)
                event_rows: list[dict[str, Any]] = []
                skipped: list[dict[str, Any]] = []
                # For jackknife/LOO influence diagnostics (discovery gate).
                events_scored: list[str] = []
                logL_mu_events: list[np.ndarray] = []
                logL_gr_events: list[np.ndarray] = []

                _set_hier_worker_state(
                    {
                        "args": args,
                        "run_label": str(run_label),
                        "mode_label": str(mode_label),
                        "mode": str(mode),
                        "z_max": float(z_max),
                        "post": post,
                        "pe_by_event": pe_by_event,
                        "log_alpha_mu": log_alpha_mu,
                        "log_alpha_gr": log_alpha_gr,
                    }
                )

                def _iter_hier_results():
                    nproc = _resolve_nproc(int(args.n_proc))
                    if nproc > 1 and len(events) > 1:
                        try:
                            ctx = mp.get_context("fork")
                        except Exception:
                            ctx = mp.get_context()
                        with ProcessPoolExecutor(max_workers=min(nproc, len(events)), mp_context=ctx) as ex:
                            yield from ex.map(_score_one_hier_event, events)
                    else:
                        for ev in events:
                            yield _score_one_hier_event(ev)

                for res in _iter_hier_results():
                    ev = str(res.get("event"))
                    if not bool(res.get("ok", False)):
                        err = str(res.get("error", "unknown error"))
                        if err:
                            print(f"[dark_sirens] {run_label} {mode_label} {ev}: {err}", flush=True)
                        skipped.append({"event": str(ev), "reason": "worker_fail", "details": err})
                        continue

                    bad_reasons = list(res.get("bad_reasons") or [])
                    if bad_reasons:
                        msg = f"[dark_sirens] {run_label} {mode_label} {ev}: hierarchical diagnostics below threshold: " + "; ".join(bad_reasons)
                        print("[dark_sirens] WARNING: " + msg, flush=True)

                    row = dict(res["row"])
                    logL_mu = np.asarray(res["logL_mu"], dtype=float)
                    logL_gr = np.asarray(res["logL_gr"], dtype=float)
                    event_rows.append(row)
                    events_scored.append(str(ev))
                    logL_mu_events.append(np.asarray(logL_mu, dtype=float))
                    logL_gr_events.append(np.asarray(logL_gr, dtype=float))
                    logL_mu_total_data += logL_mu
                    logL_gr_total_data += logL_gr
                    print(
                        f"[dark_sirens] {run_label} {mode_label} {ev}: ΔLPD_data={row['delta_lpd_data']:+.4f} ΔLPD_total={row['delta_lpd']:+.4f}",
                        flush=True,
                    )

                n_ev = int(len(event_rows))
                # Save tiny per-draw logL stacks for selection-sensitivity sweeps without re-scoring events.
                try:
                    if n_ev > 0 and len(logL_mu_events) == n_ev and len(logL_gr_events) == n_ev:
                        mu_stack = np.stack(logL_mu_events, axis=0).astype(np.float64)  # (n_ev, n_draws)
                        gr_stack = np.stack(logL_gr_events, axis=0).astype(np.float64)  # (n_ev, n_draws)
                        stack_path = tab_dir / f"hier_logL_stack_{run_label}_{mode_label}.npz"
                        np.savez(
                            stack_path,
                            logL_mu_events=mu_stack,
                            logL_gr_events=gr_stack,
                            meta=json.dumps(
                                {
                                    "run": str(run_label),
                                    "mode": str(mode_label),
                                    "hier_null_mode": str(mode),
                                    "n_events": int(n_ev),
                                    "events_scored": [str(e) for e in events_scored],
                                    "posterior_draw_idx": [int(x) for x in draw_idx],
                                    "z_max": float(z_max),
                                },
                                sort_keys=True,
                            ),
                        )
                except Exception as e:
                    print(
                        f"[dark_sirens] WARNING: failed to write hier logL stack for {run_label} {mode_label}: {e}",
                        flush=True,
                    )
                if n_ev <= 0:
                    lpd_mu_total_data = float("-inf")
                    lpd_gr_total_data = float("-inf")
                    lpd_mu_total = float("-inf")
                    lpd_gr_total = float("-inf")
                else:
                    lpd_mu_total_data = float(_logmeanexp(logL_mu_total_data))
                    lpd_gr_total_data = float(_logmeanexp(logL_gr_total_data))
                    if log_alpha_mu is not None and log_alpha_gr is not None:
                        logL_mu_total = logL_mu_total_data - float(n_ev) * log_alpha_mu
                        logL_gr_total = logL_gr_total_data - float(n_ev) * log_alpha_gr
                        lpd_mu_total = float(_logmeanexp(logL_mu_total))
                        lpd_gr_total = float(_logmeanexp(logL_gr_total))
                    else:
                        lpd_mu_total = float(lpd_mu_total_data)
                        lpd_gr_total = float(lpd_gr_total_data)

                delta_lpd_total_data = _safe_diff(lpd_mu_total_data, lpd_gr_total_data)
                delta_lpd_total = _safe_diff(lpd_mu_total, lpd_gr_total)
                delta_lpd_total_sel = _safe_diff(delta_lpd_total, delta_lpd_total_data)

                summary = {
                    "run": str(run_label),
                    "gw_data_mode": "pe",
                    "pe_like_mode": "hierarchical",
                    "mode": str(mode_label),
                    "hier_null_mode": str(mode),
                    "hier_null_seed": int(args.hier_null_seed),
                    "hier_control_battery": bool(args.hier_control_battery),
                    "hier_min_good_frac": float(args.hier_min_good_frac),
                    "hier_min_ess": float(args.hier_min_ess),
                    "hier_bad_sample_mode": str(args.hier_bad_sample_mode),
                    "events_requested": [str(e) for e in events],
                    "events_scored": [r["event"] for r in event_rows],
                    "events_skipped": skipped,
                    "n_events": int(n_ev),
                    "n_events_requested": int(len(events)),
                    "n_events_skipped": int(len(skipped)),
                    "z_max": float(z_max),
                    "max_draws": int(post.H_samples.shape[0]),
                    "posterior_draw_idx": draw_idx,
                    "selection": alpha_meta,
                    "pe_good_frac_mu_min": float(np.nanmin([float(r.get("pe_good_frac_mu_p50", np.nan)) for r in event_rows])) if event_rows else float("nan"),
                    "pe_good_frac_gr_min": float(np.nanmin([float(r.get("pe_good_frac_gr_p50", np.nan)) for r in event_rows])) if event_rows else float("nan"),
                    "pe_weight_ess_mu_min": float(np.nanmin([float(r.get("pe_weight_ess_mu_min", np.nan)) for r in event_rows])) if event_rows else float("nan"),
                    "pe_weight_ess_gr_min": float(np.nanmin([float(r.get("pe_weight_ess_gr_min", np.nan)) for r in event_rows])) if event_rows else float("nan"),
                    "lpd_mu_total_data": float(lpd_mu_total_data),
                    "lpd_gr_total_data": float(lpd_gr_total_data),
                    "delta_lpd_total_data": float(delta_lpd_total_data),
                    "lpd_mu_total": float(lpd_mu_total),
                    "lpd_gr_total": float(lpd_gr_total),
                    "delta_lpd_total": float(delta_lpd_total),
                    "lpd_mu_total_sel": _safe_diff(float(lpd_mu_total), float(lpd_mu_total_data)),
                    "lpd_gr_total_sel": _safe_diff(float(lpd_gr_total), float(lpd_gr_total_data)),
                    "delta_lpd_total_sel": float(delta_lpd_total_sel),
                }
                # Leave-one-out influence diagnostics (like the catalog jackknife).
                jk: list[dict[str, Any]] = []
                if n_ev >= 2 and len(logL_mu_events) == n_ev and len(logL_gr_events) == n_ev:
                    mu_stack = np.stack(logL_mu_events, axis=0)  # (n_ev, n_draws)
                    gr_stack = np.stack(logL_gr_events, axis=0)  # (n_ev, n_draws)
                    tot_mu_data = np.sum(mu_stack, axis=0)
                    tot_gr_data = np.sum(gr_stack, axis=0)
                    for i, ev in enumerate(events_scored):
                        loo_mu_data = tot_mu_data - mu_stack[i]
                        loo_gr_data = tot_gr_data - gr_stack[i]
                        lpd_mu_loo_data = float(_logmeanexp(loo_mu_data))
                        lpd_gr_loo_data = float(_logmeanexp(loo_gr_data))
                        delta_loo_data = _safe_diff(lpd_mu_loo_data, lpd_gr_loo_data)

                        if log_alpha_mu is not None and log_alpha_gr is not None:
                            loo_mu = loo_mu_data - float(n_ev - 1) * log_alpha_mu
                            loo_gr = loo_gr_data - float(n_ev - 1) * log_alpha_gr
                            lpd_mu_loo = float(_logmeanexp(loo_mu))
                            lpd_gr_loo = float(_logmeanexp(loo_gr))
                            delta_loo = _safe_diff(lpd_mu_loo, lpd_gr_loo)
                        else:
                            delta_loo = float(delta_loo_data)

                        jk.append(
                            {
                                "event": str(ev),
                                "delta_lpd_total_leave_one_out": float(delta_loo),
                                "delta_lpd_total_full": float(delta_lpd_total),
                                "influence": float(_safe_diff(float(delta_lpd_total), float(delta_loo))),
                            }
                        )
                return {"summary": summary, "events": event_rows, "jackknife": jk}

            z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(post.z_grid[-1])
            events_to_score = list(cached_events)
            if bool(args.hier_control_battery):
                events_to_score = events_to_score[: int(args.hier_control_battery_max_events)]

            modes = [str(args.hier_null_mode)]
            if bool(args.hier_control_battery):
                modes = ["none", "shuffle_dl", "shuffle_mc", "shuffle_dl_mc"]

            mode_results = [_score_hier_mode(mode=m, events=events_to_score, z_max=z_sel) for m in modes]
            primary = mode_results[0]

            _write_json(out_dir / f"summary_{run_label}.json", primary["summary"])
            _write_json(tab_dir / f"event_scores_{run_label}.json", primary["events"])
            if primary.get("jackknife"):
                _write_json(tab_dir / f"jackknife_{run_label}.json", primary["jackknife"])

            if bool(args.hier_control_battery):
                _write_json(
                    tab_dir / f"hier_control_battery_{run_label}.json",
                    {
                        "run": str(run_label),
                        "events_requested": [str(e) for e in events_to_score],
                        "modes": [r["summary"] for r in mode_results],
                        "details": mode_results,
                    },
                )

                # Quick figure: total ΔLPD by mode.
                try:
                    labels = [str(r["summary"].get("mode", "")) for r in mode_results]
                    deltas = np.array([float(r["summary"].get("delta_lpd_total", float("nan"))) for r in mode_results], dtype=float)
                    plt.figure(figsize=(6, 3.5))
                    xs = np.arange(len(labels))
                    plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
                    plt.bar(xs, deltas)
                    plt.xticks(xs, labels, rotation=30, ha="right")
                    plt.ylabel("ΔLPD_total (model − GR)")
                    plt.title(f"Hierarchical control battery ({run_label})")
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"hier_control_battery_{run_label}.png", dpi=160)
                    plt.close()
                except Exception:
                    pass

            # Quick figure: per-event ΔLPD (total) with data-only markers.
            try:
                rows = primary["events"]
                if rows:
                    plt.figure(figsize=(7, 4))
                    xs = np.arange(len(rows))
                    y_tot = np.array([float(r.get("delta_lpd", float("nan"))) for r in rows], dtype=float)
                    y_dat = np.array([float(r.get("delta_lpd_data", float("nan"))) for r in rows], dtype=float)
                    plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
                    plt.bar(xs, y_tot, alpha=0.6, label="total (with selection if enabled)")
                    plt.plot(xs, y_dat, "ko", markersize=4, label="data-only")
                    plt.xticks(xs, [str(r.get("event", "")) for r in rows], rotation=45, ha="right", fontsize=7)
                    plt.ylabel("ΔLPD (model − GR)")
                    plt.title(f"Hierarchical PE ΔLPD by event ({run_label}; mode={primary['summary'].get('mode')})")
                    plt.legend(fontsize=7)
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"delta_lpd_by_event_{run_label}.png", dpi=160)
                    plt.close()
            except Exception:
                pass

        manifest = {
            "timestamp_utc": _utc_stamp(),
            "run_dir": run_dirs,
            "skymap_dir": str(args.skymap_dir),
            "gw_data_mode": "pe",
            "pe_like_mode": "hierarchical",
            "hier_null_mode": str(args.hier_null_mode),
            "hier_null_seed": int(args.hier_null_seed),
            "hier_control_battery": bool(args.hier_control_battery),
            "hier_control_battery_max_events": int(args.hier_control_battery_max_events),
            "hier_control_battery_max_draws": int(args.hier_control_battery_max_draws),
            "hier_control_battery_max_pe_samples": int(args.hier_control_battery_max_pe_samples),
            "hier_min_good_frac": float(args.hier_min_good_frac),
            "hier_min_ess": float(args.hier_min_ess),
            "hier_bad_sample_mode": str(args.hier_bad_sample_mode),
            "pe_base_dir": str(pe_base_dir),
            "pe_record_id": pe_record_ids,
            "pe_prefer_variant": pe_prefer_variants,
            "pe_analysis": str(args.pe_analysis) if args.pe_analysis is not None else None,
            "pe_max_samples": int(pe_max_samples_eff) if pe_max_samples_eff is not None else None,
            "pe_seed": int(args.pe_seed),
            "max_events": int(max_cache_events),
            "events_filter": want_events,
            "events_cached": cached_events,
            "events_meta": events_meta,
            "max_draws": int(args.max_draws),
            "selection_injections_hdf": str(resolved_injections_path) if resolved_injections_path is not None else None,
            "selection_injections_spec": str(args.selection_injections_hdf),
            "selection_o4_injections_hdf": str(args.selection_o4_injections_hdf) if args.selection_o4_injections_hdf is not None else None,
            "selection_auto_record_id_o3": int(args.selection_auto_record_id_o3),
            "selection_auto_record_id_o4": int(args.selection_auto_record_id_o4) if args.selection_auto_record_id_o4 is not None else None,
            "selection_auto_strict": bool(args.selection_auto_strict),
            "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
            "selection_z_max": float(args.selection_z_max) if args.selection_z_max is not None else None,
            "selection_det_model": str(args.selection_det_model),
            "selection_snr_thresh": float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
            "selection_snr_binned_nbins": int(args.selection_snr_binned_nbins),
            "selection_injection_logit_l2": float(args.selection_injection_logit_l2),
            "selection_injection_logit_max_iter": int(args.selection_injection_logit_max_iter),
            "selection_weight_mode": str(args.selection_weight_mode),
            "selection_pop_z_mode": str(args.selection_pop_z_mode),
            "selection_pop_z_k": float(args.selection_pop_z_k),
            "selection_pop_mass_mode": str(args.selection_pop_mass_mode),
            "selection_pop_m1_alpha": float(args.selection_pop_m1_alpha),
            "selection_pop_m_min": float(args.selection_pop_m_min),
            "selection_pop_m_max": float(args.selection_pop_m_max),
            "selection_pop_q_beta": float(args.selection_pop_q_beta),
        }
        _write_json(out_dir / "manifest.json", manifest)
        print(f"[dark_sirens] done: {out_dir}", flush=True)
        return 0

    cat = load_gladeplus_index(args.glade_index)

    def _trapz_weights(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be 1D with >=2 points")
        if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be finite and strictly increasing")
        dx = np.diff(x)
        w = np.empty_like(x)
        w[0] = 0.5 * dx[0]
        w[-1] = 0.5 * dx[-1]
        if x.size > 2:
            w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        return w

    def _logsumexp_1d(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("_logsumexp_1d expects 1D array")
        m = float(np.max(x))
        return float(m + np.log(np.sum(np.exp(x - m))))

    def _logmeanexp_axis(x: np.ndarray, *, axis: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        m = np.max(x, axis=axis, keepdims=True)
        return np.squeeze(m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)

    # Optional (legacy) incompleteness correction via crude completeness curve.
    comp_z_grid: np.ndarray | None = None
    comp_C_grid: np.ndarray | None = None
    if str(args.completeness_mode) == "z3_cumulative":
        comp_z_grid, comp_C_grid = _load_or_build_completeness(
            Path(args.glade_index),
            cat.z,
            cat.w,
            z_max=float(args.gal_z_max),
            zref_max=float(args.completeness_zref_max),
            nbins=int(args.completeness_nbins),
        )
        print(
            f"[dark_sirens] completeness: mode=z3_cumulative z_max={float(args.gal_z_max):.3f} "
            f"zref_max={float(args.completeness_zref_max):.3f} nbins={int(args.completeness_nbins)}",
            flush=True,
        )

    # Mixture fraction f_miss for the out-of-catalog term.
    f_miss: float = 0.0
    f_miss_meta: dict[str, Any] | None = None
    f_miss_ref: float | None = None  # for per-event reporting when marginalizing
    f_miss_prior: dict[str, Any] | None = None
    if str(args.mixture_mode) == "simple":
        if str(args.mixture_f_miss_mode) == "fixed":
            if args.mixture_f_miss is None:
                raise ValueError("--mixture-f-miss is required when --mixture-f-miss-mode=fixed.")
            f_miss = float(args.mixture_f_miss)
            f_miss_meta = {"mode": "fixed", "f_miss": f_miss}
            f_miss_ref = float(f_miss)
        elif str(args.mixture_f_miss_mode) in ("from_completeness", "marginalize"):
            # Build a completeness curve solely to derive a single f_miss scalar.
            z_grid_c, C_grid_c = _load_or_build_completeness(
                Path(args.glade_index),
                cat.z,
                cat.w,
                z_max=float(args.missing_z_max),
                zref_max=float(args.completeness_zref_max),
                nbins=int(args.completeness_nbins),
            )
            f_miss_est = _estimate_f_miss_from_completeness(
                z_grid=z_grid_c,
                C_grid=C_grid_c,
                z_max=float(args.missing_z_max),
                host_prior_z_mode=str(args.host_prior_z_mode),
                host_prior_z_k=float(args.host_prior_z_k),
            )
            if str(args.mixture_f_miss_mode) == "from_completeness":
                f_miss = float(f_miss_est)
                f_miss_ref = float(f_miss)
                f_miss_meta = {
                    "mode": "from_completeness",
                    "z_max": float(args.missing_z_max),
                    "zref_max": float(args.completeness_zref_max),
                    "nbins": int(args.completeness_nbins),
                    "host_prior_z_mode": str(args.host_prior_z_mode),
                    "host_prior_z_k": float(args.host_prior_z_k),
                }
            else:
                # Marginalize over f_miss with an explicit prior; keep a reference f for per-event reporting.
                prior = str(args.mixture_f_miss_prior)
                n_f = int(args.mixture_f_miss_marginalize_n)
                eps = float(args.mixture_f_miss_marginalize_eps)
                if n_f < 21:
                    raise ValueError("--mixture-f-miss-marginalize-n too small (use >=21).")
                if not (0.0 < eps < 0.1):
                    raise ValueError("--mixture-f-miss-marginalize-eps must be in (0,0.1).")

                if prior == "uniform":
                    f_mean = 0.5
                    f_miss_ref = float(f_mean)
                    f_miss_prior = {"type": "uniform", "mean": float(f_mean)}
                elif prior == "beta":
                    if args.mixture_f_miss_beta_mean is not None:
                        f_mean = float(args.mixture_f_miss_beta_mean)
                        mean_mode = "fixed"
                    else:
                        f_mean = float(f_miss_est)
                        mean_mode = "from_completeness"
                    if not (0.0 < f_mean < 1.0):
                        raise ValueError("Beta prior mean for f_miss must be in (0,1).")
                    kappa = float(args.mixture_f_miss_beta_kappa)
                    if not (np.isfinite(kappa) and kappa > 0.0):
                        raise ValueError("--mixture-f-miss-beta-kappa must be finite and positive.")
                    a = float(f_mean * kappa)
                    b = float((1.0 - f_mean) * kappa)
                    # Keep away from pathological <~0 alpha/beta.
                    a = max(a, 1e-3)
                    b = max(b, 1e-3)
                    f_miss_ref = float(a / (a + b))
                    f_miss_prior = {
                        "type": "beta",
                        "alpha": float(a),
                        "beta": float(b),
                        "mean": float(a / (a + b)),
                        "kappa": float(kappa),
                        "mean_mode": str(mean_mode),
                    }
                else:
                    raise ValueError("Unknown mixture_f_miss_prior.")

                f_miss_meta = {
                    "mode": "marginalize",
                    "estimate_from_completeness": float(f_miss_est),
                    "z_max": float(args.missing_z_max),
                    "zref_max": float(args.completeness_zref_max),
                    "nbins": int(args.completeness_nbins),
                    "host_prior_z_mode": str(args.host_prior_z_mode),
                    "host_prior_z_k": float(args.host_prior_z_k),
                    "prior": f_miss_prior,
                    "grid": {"n": int(n_f), "eps": float(eps)},
                }
                # For code paths that expect a scalar f_miss (e.g. per-event reporting), use f_ref.
                f_miss = float(f_miss_ref)
        else:
            raise ValueError("Unknown mixture_f_miss_mode.")

        if not (0.0 <= f_miss <= 1.0) or not np.isfinite(f_miss):
            raise ValueError("f_miss must be finite and in [0,1].")
        if str(args.mixture_f_miss_mode) == "marginalize":
            print(
                f"[dark_sirens] mixture: mode=simple f_miss=marginalized prior={str(args.mixture_f_miss_prior)} "
                f"f_ref={float(f_miss_ref):.4f}",
                flush=True,
            )
        else:
            print(f"[dark_sirens] mixture: mode=simple f_miss={f_miss:.4f}", flush=True)

    # Selection normalization alpha(model) requires an injection set.
    # We resolve + load injections after event selection so `--selection-injections-hdf auto`
    # can choose an O3a/O3b/O3 file appropriate for the scored event set.
    injections = None
    resolved_injections_path: Path | None = None

    # Build per-event cache once and reuse across run_dir seeds.
    # Cache includes: (z,w,ipix) arrays after all selection cuts + pixel-distance validity checks.
    event_cache: dict[str, dict[str, Any]] = {}
    cached_events: list[str] = []

    gw_data_mode = str(args.gw_data_mode)
    pe_base_dir = Path(str(args.pe_base_dir)).expanduser().resolve()
    pe_record_ids = [int(x) for x in (args.pe_record_id or [])]
    pe_prefer_variants = [str(v) for v in (args.pe_prefer_variant or [])]
    pe_index = None
    if gw_data_mode == "pe":
        pe_index = build_gwtc_pe_index(base_dir=pe_base_dir, record_ids=pe_record_ids or None)
    if str(args.null_mode) != "none" and gw_data_mode != "pe":
        raise ValueError("--null-mode currently supports only --gw-data-mode=pe.")

    candidate_events = want_events if want_events is not None else sorted(skymaps.keys())
    max_events_eff = int(len(candidate_events)) if want_events is not None else int(args.max_events)
    for ev in candidate_events:
        if len(cached_events) >= max_events_eff:
            break
        if ev not in skymaps:
            print(f"[dark_sirens] skip {ev}: no skymap under {str(args.skymap_dir)}", flush=True)
            continue

        cache_path = cache_dir / f"event_{ev}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as d:
                meta = json.loads(str(d["meta"].tolist()))
                # Rebuild cache if the key parameters changed.
                ok = (
                    str(meta.get("gw_data_mode", "skymap")) == gw_data_mode
                    and meta.get("skymap_path") == str(skymaps[ev])
                    and float(meta.get("p_credible", -1.0)) == float(args.p_credible)
                    and float(meta.get("gal_z_max", -1.0)) == float(args.gal_z_max)
                    and int(meta.get("cat_nside", -1)) == int(cat.nside)
                    and str(meta.get("completeness_mode", "none")) == str(args.completeness_mode)
                )
                if ok and gw_data_mode == "pe":
                    want_pe_analysis_requested: str | None
                    if args.pe_analysis is not None:
                        want_pe_analysis_requested = str(args.pe_analysis)
                    elif str(args.gw_distance_prior_mode) == "pe_analytic":
                        # pe_analytic implicitly changes how we pick the analysis group (must have analytic priors).
                        want_pe_analysis_requested = "__auto_analytic__"
                    else:
                        want_pe_analysis_requested = None
                    ok = (
                        ok
                        and meta.get("pe_file") is not None
                        and str(meta.get("null_mode", "none")) == str(args.null_mode)
                        and int(meta.get("null_seed", -999)) == int(args.null_seed)
                        and meta.get("pe_analysis_requested") == want_pe_analysis_requested
                        and int(meta.get("pe_nside", -1)) == int(args.pe_nside)
                        and int(meta.get("pe_dl_nbins", -1)) == int(args.pe_dl_nbins)
                        and float(meta.get("pe_dl_qmin", -1.0)) == float(args.pe_dl_qmin)
                        and float(meta.get("pe_dl_qmax", -1.0)) == float(args.pe_dl_qmax)
                        and float(meta.get("pe_dl_pad_factor", -1.0)) == float(args.pe_dl_pad_factor)
                        and float(meta.get("pe_dl_pseudocount", -1.0)) == float(args.pe_dl_pseudocount)
                        and int(meta.get("pe_dl_smooth_iters", -999)) == int(args.pe_dl_smooth_iters)
                        and ("pe_pix_sel" in d.files)
                        and ("pe_prob_pix" in d.files)
                        and ("pe_dL_edges" in d.files)
                        and ("pe_pdf_bins" in d.files)
                    )
                if ok:
                    event_cache[ev] = {
                        "meta": meta,
                        "z": np.asarray(d["z"], dtype=float),
                        "w": np.asarray(d["w"], dtype=float),
                        "ipix": np.asarray(d["ipix"], dtype=np.int64),
                        "hpix_sel": np.asarray(d["hpix_sel"], dtype=np.int64) if "hpix_sel" in d.files else None,
                        "pe_pix_sel": np.asarray(d["pe_pix_sel"], dtype=np.int64) if "pe_pix_sel" in d.files else None,
                        "pe_prob_pix": np.asarray(d["pe_prob_pix"], dtype=float) if "pe_prob_pix" in d.files else None,
                        "pe_dL_edges": np.asarray(d["pe_dL_edges"], dtype=float) if "pe_dL_edges" in d.files else None,
                        "pe_pdf_bins": np.asarray(d["pe_pdf_bins"], dtype=float) if "pe_pdf_bins" in d.files else None,
                    }
                    cached_events.append(ev)
                    continue

        # (Re)build cache.
        path = skymaps[ev]
        sky = None
        pe = None
        pe_file = None
        pe_meta = None
        if gw_data_mode == "skymap":
            try:
                sky = read_skymap_3d(path, nest=True)
            except Exception as e:
                print(f"[dark_sirens] skip {ev}: failed to read skymap ({e})", flush=True)
                continue

            try:
                hpix_sel, area_deg2 = credible_region_pixels(sky, nside_out=cat.nside, p_credible=float(args.p_credible))
            except Exception as e:
                print(f"[dark_sirens] skip {ev}: credible-region failure ({e})", flush=True)
                continue
        else:
            assert pe_index is not None
            if ev not in pe_index:
                print(f"[dark_sirens] skip {ev}: no PEDataRelease file under {pe_base_dir}", flush=True)
                continue

            files = pe_index[ev]
            # Try multiple PE variants in preference order until one loads successfully.
            ordered: list[Any] = []
            seen: set[str] = set()
            for v in pe_prefer_variants:
                for f in files:
                    if getattr(f, "variant", None) == v:
                        p = str(getattr(f, "path"))
                        if p not in seen:
                            ordered.append(f)
                            seen.add(p)
            for f in files:
                p = str(getattr(f, "path"))
                if p not in seen:
                    ordered.append(f)
                    seen.add(p)

            last_err: Exception | None = None
            pe_file = None
            for cand in ordered:
                pe_try = (pe_base_dir / str(getattr(cand, "path"))).resolve()
                try:
                    analysis = str(args.pe_analysis) if args.pe_analysis is not None else None
                    if analysis is None and str(args.gw_distance_prior_mode) == "pe_analytic":
                        analysis = select_gwtc_pe_analysis_with_analytic_priors(
                            path=pe_try,
                            prefer=_default_pe_analysis_prefer_for_analytic_priors(pe_try),
                            require_parameters=["luminosity_distance"],
                        )
                    ra_s, dec_s, dL_s, pe_meta = load_gwtc_pe_sky_samples(
                        path=pe_try,
                        analysis=analysis,
                        max_samples=int(args.pe_max_samples) if args.pe_max_samples is not None else None,
                        seed=int(args.pe_seed),
                    )
                    if str(args.null_mode) == "rotate_pe_sky":
                        rng = np.random.default_rng(_stable_int_seed(f"{int(args.null_seed)}:{ev}"))
                        R = _random_rotation_matrix(rng)
                        ra_s, dec_s = _rotate_radec(ra_s, dec_s, R)
                    pe_file = pe_try
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if pe_file is None:
                print(f"[dark_sirens] skip {ev}: failed to read PE samples ({last_err})", flush=True)
                continue

            try:
                pe = build_pe_pixel_distance_histogram(
                    ra_rad=ra_s,
                    dec_rad=dec_s,
                    dL_mpc=dL_s,
                    nside=int(args.pe_nside),
                    p_credible=float(args.p_credible),
                    dl_nbins=int(args.pe_dl_nbins),
                    dl_qmin=float(args.pe_dl_qmin),
                    dl_qmax=float(args.pe_dl_qmax),
                    dl_pad_factor=float(args.pe_dl_pad_factor),
                    dl_pseudocount=float(args.pe_dl_pseudocount),
                    dl_smooth_iters=int(args.pe_dl_smooth_iters),
                    nest=True,
                )
            except Exception as e:
                print(f"[dark_sirens] skip {ev}: PE histogram build failed ({e})", flush=True)
                continue

            # Galaxy selection pixels: upsample the PE credible pixels to the catalog nside (NESTED).
            if int(cat.nside) < int(pe.nside) or int(cat.nside) % int(pe.nside) != 0:
                raise ValueError("cat nside must be an integer multiple of pe nside (NESTED).")
            ratio = int(cat.nside) // int(pe.nside)
            if ratio & (ratio - 1) != 0:
                raise ValueError("cat.nside / pe_nside must be a power of two (NESTED upsampling).")
            group = ratio * ratio
            offs = np.arange(group, dtype=np.int64)
            hpix_sel = (np.asarray(pe.pix_sel, dtype=np.int64).reshape((-1, 1)) * group + offs.reshape((1, -1))).reshape(-1)
            area_deg2 = float(np.asarray(pe.pix_sel).size) * float(hp.nside2pixarea(int(pe.nside), degrees=True))

        if area_deg2 > float(args.max_area_deg2):
            print(f"[dark_sirens] skip {ev}: area {area_deg2:.1f} deg^2 > {args.max_area_deg2}", flush=True)
            continue

        # Fast precheck: skip events with too many galaxies before allocating large arrays.
        try:
            hpix_sel_u = np.unique(np.asarray(hpix_sel, dtype=np.int64))
            offs = np.asarray(cat.hpix_offsets, dtype=np.int64)
            n_est = int(np.sum(offs[hpix_sel_u + 1] - offs[hpix_sel_u]))
        except Exception:
            n_est = -1
        if n_est > int(args.max_gal):
            print(f"[dark_sirens] skip {ev}: too many galaxies (est {n_est:,} > {args.max_gal:,})", flush=True)
            continue

        ra, dec, z, w = gather_galaxies_from_pixels(cat, hpix_sel, z_max=float(args.gal_z_max))
        if ra.size == 0:
            print(f"[dark_sirens] skip {ev}: no galaxies after cuts", flush=True)
            continue
        if ra.size > int(args.max_gal):
            print(f"[dark_sirens] skip {ev}: too many galaxies ({ra.size:,} > {args.max_gal:,})", flush=True)
            continue

        if comp_z_grid is not None and comp_C_grid is not None:
            w = _apply_completeness_weights(
                z=z,
                w=w,
                z_grid=comp_z_grid,
                C_grid=comp_C_grid,
                c_floor=float(args.completeness_c_floor),
            )

        # Precompute galaxy -> sky pixel indices once (critical for multi-seed speed).
        theta = np.deg2rad(90.0 - dec.astype(float))
        phi = np.deg2rad(ra.astype(float))
        if sky is not None:
            ipix = np.asarray(hp.ang2pix(sky.nside, theta, phi, nest=True), dtype=np.int64)

            # Prefilter by sky pixel distance-layer validity.
            prob = sky.prob[ipix]
            distmu = sky.distmu[ipix]
            distsigma = sky.distsigma[ipix]
            distnorm = sky.distnorm[ipix]
            good = (
                np.isfinite(prob)
                & (prob > 0.0)
                & np.isfinite(distmu)
                & np.isfinite(distsigma)
                & (distsigma > 0.0)
                & np.isfinite(distnorm)
                & (distnorm > 0.0)
                & np.isfinite(z)
                & (z > 0.0)
                & np.isfinite(w)
                & (w > 0.0)
            )
            if not np.any(good):
                print(f"[dark_sirens] skip {ev}: all galaxies map to invalid distance-layer pixels", flush=True)
                continue

            z = z[good]
            w = w[good]
            ipix = ipix[good]
        else:
            assert pe is not None
            ipix = np.asarray(hp.ang2pix(int(pe.nside), theta, phi, nest=True), dtype=np.int64)
            good = np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
            if not np.any(good):
                print(f"[dark_sirens] skip {ev}: all galaxies have invalid z/w", flush=True)
                continue
            z = z[good]
            w = w[good]
            ipix = ipix[good]

        pe_analysis_requested: str | None
        if args.pe_analysis is not None:
            pe_analysis_requested = str(args.pe_analysis)
        elif str(args.gw_distance_prior_mode) == "pe_analytic":
            pe_analysis_requested = "__auto_analytic__"
        else:
            pe_analysis_requested = None

        meta = {
            "event": ev,
            "skymap_path": str(path),
            "gw_data_mode": str(gw_data_mode),
            "null_mode": str(args.null_mode),
            "null_seed": int(args.null_seed),
            "sky_nside": int(sky.nside) if sky is not None else None,
            "cat_nside": int(cat.nside),
            "ipix_nside": int(sky.nside) if sky is not None else int(pe.nside),
            "p_credible": float(args.p_credible),
            "gal_z_max": float(args.gal_z_max),
            "sky_area_deg2": float(area_deg2),
            "n_gal": int(z.size),
            "completeness_mode": str(args.completeness_mode),
            "pe_file": str(pe_file) if pe_file is not None else None,
            "pe_meta": pe_meta.to_jsonable() if pe_meta is not None else None,
            "pe_analysis_requested": pe_analysis_requested,
            "pe_nside": int(pe.nside) if pe is not None else None,
            "pe_dl_nbins": int(args.pe_dl_nbins) if pe is not None else None,
            "pe_dl_qmin": float(args.pe_dl_qmin) if pe is not None else None,
            "pe_dl_qmax": float(args.pe_dl_qmax) if pe is not None else None,
            "pe_dl_pad_factor": float(args.pe_dl_pad_factor) if pe is not None else None,
            "pe_dl_pseudocount": float(args.pe_dl_pseudocount) if pe is not None else None,
            "pe_dl_smooth_iters": int(args.pe_dl_smooth_iters) if pe is not None else None,
        }
        save_kwargs: dict[str, Any] = {
            "meta": json.dumps(meta),
            "z": z.astype(np.float32),
            "w": w.astype(np.float32),
            "ipix": ipix.astype(np.int64),
            "hpix_sel": np.asarray(hpix_sel, dtype=np.int64),
        }
        if pe is not None:
            save_kwargs.update(
                {
                    "pe_pix_sel": np.asarray(pe.pix_sel, dtype=np.int64),
                    "pe_prob_pix": np.asarray(pe.prob_pix, dtype=np.float32),
                    "pe_dL_edges": np.asarray(pe.dL_edges, dtype=np.float64),
                    "pe_pdf_bins": np.asarray(pe.pdf_bins, dtype=np.float32),
                }
            )
        np.savez(cache_path, **save_kwargs)
        event_cache[ev] = {
            "meta": meta,
            "z": z.astype(float),
            "w": w.astype(float),
            "ipix": ipix.astype(np.int64),
            "hpix_sel": np.asarray(hpix_sel, dtype=np.int64),
            "pe_pix_sel": np.asarray(pe.pix_sel, dtype=np.int64) if pe is not None else None,
            "pe_prob_pix": np.asarray(pe.prob_pix, dtype=float) if pe is not None else None,
            "pe_dL_edges": np.asarray(pe.dL_edges, dtype=float) if pe is not None else None,
            "pe_pdf_bins": np.asarray(pe.pdf_bins, dtype=float) if pe is not None else None,
        }
        cached_events.append(ev)
        print(f"[dark_sirens] cached {ev}: n_gal={int(z.size):,} area={area_deg2:.1f} deg^2", flush=True)

    if not cached_events:
        raise RuntimeError("No events passed selection; nothing to score.")
    if want_events is not None:
        missing = [str(ev) for ev in want_events if str(ev) not in set(cached_events)]
        if missing:
            print(
                f"[dark_sirens] NOTE: requested {int(len(want_events))} events but cached {int(len(cached_events))}; "
                f"missing: {', '.join(missing)}",
                flush=True,
            )

    _write_json(
        cache_dir / "cache_manifest.json",
        {
            "timestamp_utc": _utc_stamp(),
            "gw_data_mode": str(args.gw_data_mode),
            "pe_like_mode": str(args.pe_like_mode),
            "pe": {
                "base_dir": str(pe_base_dir),
                "record_ids": pe_record_ids,
                "prefer_variants": pe_prefer_variants,
                "analysis": str(args.pe_analysis) if args.pe_analysis is not None else None,
                "analysis_requested": (
                    str(args.pe_analysis) if args.pe_analysis is not None else ("__auto_analytic__" if str(args.gw_distance_prior_mode) == "pe_analytic" else None)
                ),
                "max_samples": int(args.pe_max_samples) if args.pe_max_samples is not None else None,
                "seed": int(args.pe_seed),
                "nside": int(args.pe_nside),
                "dl_nbins": int(args.pe_dl_nbins),
            }
            if gw_data_mode == "pe"
            else None,
            "p_credible": float(args.p_credible),
            "gal_z_max": float(args.gal_z_max),
            "cat_nside": int(cat.nside),
            "max_area_deg2": float(args.max_area_deg2),
            "max_gal": int(args.max_gal),
            "max_events": int(args.max_events),
            "events_filter": want_events,
            "events_cached": cached_events,
        },
    )

    # Resolve + load selection injections (used for both GR H0 control and μ-vs-GR scoring).
    inj_spec = str(args.selection_injections_hdf).strip()
    if inj_spec.lower() in ("none", "0", "false"):
        injections = None
        resolved_injections_path = None
        print("[dark_sirens] selection: disabled", flush=True)
    else:
        want_auto = inj_spec.lower() == "auto"

        if want_auto:
            resolved_injections_path = resolve_selection_injection_file(
                events=[str(x) for x in cached_events],
                base_dir="data/cache/gw/zenodo",
                population="mixture",
                auto_download=True,
                record_id_o3=int(args.selection_auto_record_id_o3),
                record_id_o4=(int(args.selection_auto_record_id_o4) if args.selection_auto_record_id_o4 is not None else None),
                o4_injections_hdf=args.selection_o4_injections_hdf,
                strict=bool(args.selection_auto_strict),
            )
        else:
            resolved_injections_path = Path(inj_spec).expanduser().resolve()
            if not resolved_injections_path.exists():
                raise FileNotFoundError(f"Selection injections file not found: {resolved_injections_path}")

        assert resolved_injections_path is not None
        injections = load_o3_injections(
            resolved_injections_path,
            ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
        )
        print(
            f"[dark_sirens] selection: injections={resolved_injections_path.name} "
            f"ifar>{float(args.selection_ifar_thresh_yr):g}yr",
            flush=True,
        )

    # Optional published-control baseline: GR H0 posterior on a grid.
    if str(args.gr_h0_mode) == "grid":
        if str(args.gw_distance_prior_mode) == "pe_analytic":
            raise ValueError("--gr-h0-mode=grid does not currently support --gw-distance-prior-mode=pe_analytic.")
        H0_grid = np.linspace(float(args.gr_h0_grid_min), float(args.gr_h0_grid_max), int(args.gr_h0_grid_n))
        use_events = list(cached_events)
        if bool(args.gr_h0_smoke):
            # Seconds-scale sanity mode: 1 event, tiny grid.
            use_events = use_events[:1]
            H0_grid = np.linspace(65.0, 75.0, 11)

        gr_events = []
        for ev in use_events:
            meta = event_cache[ev]["meta"]
            gw_mode = str(meta.get("gw_data_mode", "skymap"))
            row: dict[str, Any] = {
                "event": str(ev),
                "gw_data_mode": gw_mode,
                "skymap_path": str(meta["skymap_path"]),
                "z": np.asarray(event_cache[ev]["z"], dtype=float),
                "w": np.asarray(event_cache[ev]["w"], dtype=float),
                "ipix": np.asarray(event_cache[ev]["ipix"], dtype=np.int64),
                "hpix_sel": np.asarray(event_cache[ev]["hpix_sel"], dtype=np.int64) if event_cache[ev].get("hpix_sel") is not None else None,
            }
            if gw_mode == "pe":
                row.update(
                    {
                        "pe_file": str(meta.get("pe_file")),
                        "pe_analysis": str((meta.get("pe_meta") or {}).get("analysis")) if meta.get("pe_meta") is not None else None,
                        "pe_nside": int(meta.get("pe_nside")),
                        "pe_pix_sel": np.asarray(event_cache[ev]["pe_pix_sel"], dtype=np.int64),
                        "pe_prob_pix": np.asarray(event_cache[ev]["pe_prob_pix"], dtype=float),
                        "pe_dL_edges": np.asarray(event_cache[ev]["pe_dL_edges"], dtype=float),
                        "pe_pdf_bins": np.asarray(event_cache[ev]["pe_pdf_bins"], dtype=float),
                    }
                )
            gr_events.append(row)

        gw_path0 = Path(gr_events[0]["pe_file"] if str(gr_events[0].get("gw_data_mode")) == "pe" else gr_events[0]["skymap_path"])
        gr_res = compute_gr_h0_posterior_grid(
            events=gr_events,
            H0_grid=H0_grid,
            omega_m0=float(args.gr_h0_omega_m0),
            omega_k0=float(args.gr_h0_omega_k0),
            cache_dir=cache_gr_h0_dir,
            mixture_mode=str(args.mixture_mode),  # type: ignore[arg-type]
            f_miss=float(f_miss),
            p_credible=float(args.p_credible),
            nside_coarse=int(cat.nside),
            host_prior_z_mode=str(args.host_prior_z_mode),  # type: ignore[arg-type]
            host_prior_z_k=float(args.host_prior_z_k),
            missing_z_max=float(args.missing_z_max),
            missing_dl_grid_n=int(args.missing_dl_grid_n),
            missing_dl_min_mpc=float(args.missing_dl_min_mpc),
            missing_dl_max_mpc=float(args.missing_dl_max_mpc) if args.missing_dl_max_mpc is not None else None,
            missing_dl_nsigma=float(args.missing_dl_nsigma),
            missing_pixel_chunk_size=int(args.missing_pixel_chunk_size),
            injections=injections,
            ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
            z_max=float(args.selection_z_max) if args.selection_z_max is not None else float(args.gal_z_max),
            det_model=str(args.selection_det_model),
            snr_threshold=float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
            snr_binned_nbins=int(args.selection_snr_binned_nbins),
            weight_mode=str(args.selection_weight_mode),
            pop_z_mode=str(args.selection_pop_z_mode),
            pop_z_powerlaw_k=float(args.selection_pop_z_k),
            pop_mass_mode=str(args.selection_pop_mass_mode),
            pop_m1_alpha=float(args.selection_pop_m1_alpha),
            pop_m_min=float(args.selection_pop_m_min),
            pop_m_max=float(args.selection_pop_m_max),
            pop_q_beta=float(args.selection_pop_q_beta),
            prior=str(args.gr_h0_prior),
            gw_distance_prior=_gw_prior_for_args(args, gw_path0),
        )
        _write_json(out_dir / "gr_h0_grid_posterior.json", gr_res)
        print(f"[dark_sirens] GR H0 grid posterior: wrote {out_dir/'gr_h0_grid_posterior.json'}", flush=True)

        # Quick figure: GR H0 posterior (optionally compare to no-selection curve).
        try:
            H0g = np.asarray(gr_res.get("H0_grid", []), dtype=float)
            p = np.asarray(gr_res.get("posterior", []), dtype=float)
            if H0g.size >= 2 and p.size == H0g.size:
                plt.figure(figsize=(7, 4))
                plt.plot(H0g, p, color="C0", linewidth=2.0, label="GR posterior (with selection α if enabled)")

                # Also show the uncorrected (no-selection) posterior implied by the event logL terms.
                logL_events = np.zeros_like(H0g)
                for row in gr_res.get("events", []):
                    logL_events += np.asarray(row.get("logL_H0", []), dtype=float)
                if logL_events.size == H0g.size and np.all(np.isfinite(logL_events)):
                    logL0 = logL_events - float(np.max(logL_events))
                    p0 = np.exp(logL0)
                    p0 = p0 / float(np.sum(p0))
                    plt.plot(H0g, p0, color="k", linewidth=1.5, alpha=0.7, label="no-selection (diagnostic)")

                plt.xlabel(r"$H_0$ [km/s/Mpc]")
                plt.ylabel("posterior (arb. norm.)")
                plt.title("GR dark-siren $H_0$ grid posterior (control)")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(fig_dir / "gr_h0_posterior.png", dpi=160)
                plt.close()
        except Exception:
            pass

    # Score each EM posterior seed against the same cached per-event sky/galaxy selections.
    for rd in run_dirs:
        post_full = load_mu_forward_posterior(rd)
        run_label = Path(rd).name
        post, draw_idx = _downsample_posterior(post_full, max_draws=int(args.max_draws), seed=_stable_int_seed(run_label))

        # Compute selection normalization alpha(draw) if requested.
        log_alpha_mu: np.ndarray | None = None
        log_alpha_gr: np.ndarray | None = None
        alpha_meta: dict[str, Any] | None = None
        if injections is not None:
            z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(args.gal_z_max)
            alpha = compute_selection_alpha_from_injections(
                post=post,
                injections=injections,
                convention=args.convention,  # type: ignore[arg-type]
                z_max=z_sel,
                snr_threshold=float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
                det_model=str(args.selection_det_model),  # type: ignore[arg-type]
                snr_binned_nbins=int(args.selection_snr_binned_nbins),
                injection_logit_l2=float(args.selection_injection_logit_l2),
                injection_logit_max_iter=int(args.selection_injection_logit_max_iter),
                weight_mode=str(args.selection_weight_mode),  # type: ignore[arg-type]
                pop_z_mode=str(args.selection_pop_z_mode),  # type: ignore[arg-type]
                pop_z_powerlaw_k=float(args.selection_pop_z_k),
                pop_mass_mode=str(args.selection_pop_mass_mode),  # type: ignore[arg-type]
                pop_m1_alpha=float(args.selection_pop_m1_alpha),
                pop_m_min=float(args.selection_pop_m_min),
                pop_m_max=float(args.selection_pop_m_max),
                pop_q_beta=float(args.selection_pop_q_beta),
                pop_m_taper_delta=float(args.selection_pop_m_taper_delta),
                pop_m_peak=float(args.selection_pop_m_peak),
                pop_m_peak_sigma=float(args.selection_pop_m_peak_sigma),
                pop_m_peak_frac=float(args.selection_pop_m_peak_frac),
            )
            log_alpha_mu = np.log(np.clip(alpha.alpha_mu, 1e-300, np.inf))
            log_alpha_gr = np.log(np.clip(alpha.alpha_gr, 1e-300, np.inf))
            alpha_meta = {
                "method": alpha.method,
                "z_max": alpha.z_max,
                "snr_threshold": alpha.snr_threshold,
                "n_injections_used": alpha.n_injections_used,
                "det_model": getattr(alpha, "det_model", "unknown"),
                "injection_logit_l2": float(args.selection_injection_logit_l2),
                "injection_logit_max_iter": int(args.selection_injection_logit_max_iter),
                "weight_mode": getattr(alpha, "weight_mode", "unknown"),
                "pop_z_mode": str(args.selection_pop_z_mode),
                "pop_z_k": float(args.selection_pop_z_k),
                "pop_mass_mode": str(args.selection_pop_mass_mode),
                "pop_m1_alpha": float(args.selection_pop_m1_alpha),
                "pop_m_min": float(args.selection_pop_m_min),
                "pop_m_max": float(args.selection_pop_m_max),
                "pop_q_beta": float(args.selection_pop_q_beta),
                "pop_m_taper_delta": float(args.selection_pop_m_taper_delta),
                "pop_m_peak": float(args.selection_pop_m_peak),
                "pop_m_peak_sigma": float(args.selection_pop_m_peak_sigma),
                "pop_m_peak_frac": float(args.selection_pop_m_peak_frac),
            }
            np.savez(
                tab_dir / f"selection_alpha_{run_label}.npz",
                alpha_mu=np.asarray(alpha.alpha_mu, dtype=float),
                alpha_gr=np.asarray(alpha.alpha_gr, dtype=float),
                log_alpha_mu=np.asarray(log_alpha_mu, dtype=float),
                log_alpha_gr=np.asarray(log_alpha_gr, dtype=float),
                meta=json.dumps(alpha_meta, sort_keys=True),
            )

        missing_pre = None
        missing_pre_meta: dict[str, Any] | None = None
        if str(args.mixture_mode) == "simple":
            missing_pre = precompute_missing_host_prior(
                post,
                convention=args.convention,  # type: ignore[arg-type]
                z_max=float(args.missing_z_max),
                host_prior_z_mode=str(args.host_prior_z_mode),  # type: ignore[arg-type]
                host_prior_z_k=float(args.host_prior_z_k),
            )
            missing_pre_meta = {
                "z_max": float(args.missing_z_max),
                "host_prior_z_mode": str(args.host_prior_z_mode),
                "host_prior_z_k": float(args.host_prior_z_k),
                **missing_pre.to_jsonable(),
            }

        scores: list[DarkSirenEventScore] = []
        n_draws = int(post.H_samples.shape[0])
        logL_mu_total = np.zeros((n_draws,), dtype=float)
        logL_gr_total = np.zeros((n_draws,), dtype=float)
        do_f_miss_marginalize = str(args.mixture_mode) == "simple" and str(args.mixture_f_miss_mode) == "marginalize"
        logL_cat_mu_list: list[np.ndarray] = []
        logL_cat_gr_list: list[np.ndarray] = []
        logL_missing_mu_list: list[np.ndarray] = []
        logL_missing_gr_list: list[np.ndarray] = []

        # Score events for this run (optionally parallelized over events).
        events_to_score = list(cached_events)
        _set_event_worker_state(
            {
                "args": args,
                "run_label": str(run_label),
                "post": post,
                "draw_idx": draw_idx,
                "gw_data_mode": str(gw_data_mode),
                "pe_base_dir": pe_base_dir,
                "event_cache": event_cache,
                "cache_terms_dir": cache_terms_dir,
                "cache_missing_dir": cache_missing_dir,
                "missing_pre": missing_pre,
                "cat_nside": int(cat.nside),
            }
        )

        def _iter_results():
            nproc = _resolve_nproc(int(args.n_proc))
            if nproc > 1 and len(events_to_score) > 1:
                # Use fork when available to avoid pickling large arrays.
                try:
                    ctx = mp.get_context("fork")
                except Exception:
                    ctx = mp.get_context()
                with ProcessPoolExecutor(max_workers=min(nproc, len(events_to_score)), mp_context=ctx) as ex:
                    yield from ex.map(_score_one_event_for_run, events_to_score)
            else:
                for ev in events_to_score:
                    yield _score_one_event_for_run(ev)

        for res in _iter_results():
            ev = str(res.get("event"))
            if not bool(res.get("ok", False)):
                print(f"[dark_sirens] {run_label} skip {ev}: {str(res.get('error', 'unknown error'))}", flush=True)
                continue

            meta = event_cache[ev]["meta"]
            sky_path = Path(str(meta["skymap_path"]))

            logL_cat_mu = np.asarray(res["logL_cat_mu"], dtype=float)
            logL_cat_gr = np.asarray(res["logL_cat_gr"], dtype=float)
            logL_mu = logL_cat_mu
            logL_gr = logL_cat_gr

            if str(args.mixture_mode) == "simple":
                logL_missing_mu = np.asarray(res["logL_missing_mu"], dtype=float)
                logL_missing_gr = np.asarray(res["logL_missing_gr"], dtype=float)

                log_a = np.log1p(-f_miss) if f_miss < 1.0 else -np.inf
                log_b = np.log(f_miss) if f_miss > 0.0 else -np.inf
                logL_mu = np.logaddexp(log_a + logL_cat_mu, log_b + logL_missing_mu)
                logL_gr = np.logaddexp(log_a + logL_cat_gr, log_b + logL_missing_gr)

                if do_f_miss_marginalize:
                    logL_cat_mu_list.append(logL_cat_mu)
                    logL_cat_gr_list.append(logL_cat_gr)
                    logL_missing_mu_list.append(logL_missing_mu)
                    logL_missing_gr_list.append(logL_missing_gr)

            # Per-event decomposition: data-only vs selection-corrected.
            lpd_mu_data = _logmeanexp(logL_mu)
            lpd_gr_data = _logmeanexp(logL_gr)
            if log_alpha_mu is not None and log_alpha_gr is not None:
                lpd_mu = _logmeanexp(logL_mu - log_alpha_mu)
                lpd_gr = _logmeanexp(logL_gr - log_alpha_gr)
            else:
                lpd_mu = float(lpd_mu_data)
                lpd_gr = float(lpd_gr_data)
            delta_data = float(lpd_mu_data - lpd_gr_data)
            delta_total = float(lpd_mu - lpd_gr)

            sc = DarkSirenEventScore(
                event=str(ev),
                skymap_path=str(sky_path),
                n_gal=int(res.get("n_gal", int(event_cache[ev]["z"].size))),
                sky_area_deg2=float(res.get("sky_area_deg2", float(meta.get("sky_area_deg2", float("nan"))))),
                lpd_mu=float(lpd_mu),
                lpd_gr=float(lpd_gr),
                delta_lpd=float(delta_total),
                lpd_mu_data=float(lpd_mu_data),
                lpd_gr_data=float(lpd_gr_data),
                delta_lpd_data=float(delta_data),
                lpd_mu_sel=float(lpd_mu - lpd_mu_data),
                lpd_gr_sel=float(lpd_gr - lpd_gr_data),
                delta_lpd_sel=float(delta_total - delta_data),
            )
            scores.append(sc)
            logL_mu_total += logL_mu
            logL_gr_total += logL_gr
            if log_alpha_mu is not None and log_alpha_gr is not None:
                print(
                    f"[dark_sirens] {run_label} {ev}: n_gal={sc.n_gal:,} area={sc.sky_area_deg2:.1f} deg^2 "
                    f"ΔLPD_data={sc.delta_lpd_data:+.4f} ΔLPD_total={sc.delta_lpd:+.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[dark_sirens] {run_label} {ev}: n_gal={sc.n_gal:,} area={sc.sky_area_deg2:.1f} deg^2 ΔLPD={sc.delta_lpd:+.4f}",
                    flush=True,
                )

        # Total-score decomposition at f_ref: capture data-only totals before selection correction.
        logL_mu_total_data_ref = np.asarray(logL_mu_total, dtype=float)
        logL_gr_total_data_ref = np.asarray(logL_gr_total, dtype=float)
        lpd_mu_total_data_ref = float(_logmeanexp(logL_mu_total_data_ref))
        lpd_gr_total_data_ref = float(_logmeanexp(logL_gr_total_data_ref))

        # Apply selection normalization at the correct (combined draw) level.
        if log_alpha_mu is not None and log_alpha_gr is not None:
            n_ev = int(len(scores))
            logL_mu_total = logL_mu_total - float(n_ev) * log_alpha_mu
            logL_gr_total = logL_gr_total - float(n_ev) * log_alpha_gr

        lpd_mu_total_ref = _logmeanexp(logL_mu_total)
        lpd_gr_total_ref = _logmeanexp(logL_gr_total)

        # Optionally marginalize over f_miss as a *global* nuisance parameter shared across events.
        if do_f_miss_marginalize:
            if f_miss_meta is None or f_miss_prior is None:
                raise RuntimeError("Internal error: missing f_miss_meta/prior for marginalization.")
            if int(len(logL_cat_mu_list)) != int(len(scores)) or int(len(logL_missing_mu_list)) != int(len(scores)):
                raise RuntimeError("Internal error: mismatch between stored logL vectors and scored events.")

            grid = dict(f_miss_meta.get("grid", {}))
            n_f = int(grid.get("n", int(args.mixture_f_miss_marginalize_n)))
            eps = float(grid.get("eps", float(args.mixture_f_miss_marginalize_eps)))
            f_grid = np.linspace(eps, 1.0 - eps, n_f)
            w_f = _trapz_weights(f_grid)
            logw_f = np.log(np.clip(w_f, 1e-300, np.inf))
            logf = np.log(f_grid)
            log1mf = np.log1p(-f_grid)

            prior_type = str(f_miss_prior.get("type", "uniform"))
            if prior_type == "uniform":
                log_prior_f = np.zeros_like(f_grid, dtype=float)
            elif prior_type == "beta":
                a = float(f_miss_prior["alpha"])
                b = float(f_miss_prior["beta"])
                log_prior_f = (a - 1.0) * logf + (b - 1.0) * log1mf - float(betaln(a, b))
            else:
                raise ValueError("Unknown f_miss prior type for marginalization.")

            # Build per-event f-grid logL matrices so we can (optionally) jackknife event influence
            # without recomputing any galaxy integrals.
            logL_mu_fd = np.zeros((n_f, n_draws), dtype=float)
            logL_gr_fd = np.zeros((n_f, n_draws), dtype=float)
            logL_mu_event_fd_list: list[np.ndarray] = []
            logL_gr_event_fd_list: list[np.ndarray] = []
            for i in range(int(len(scores))):
                ev_mu = np.logaddexp(log1mf[:, None] + logL_cat_mu_list[i][None, :], logf[:, None] + logL_missing_mu_list[i][None, :])
                ev_gr = np.logaddexp(log1mf[:, None] + logL_cat_gr_list[i][None, :], logf[:, None] + logL_missing_gr_list[i][None, :])
                logL_mu_event_fd_list.append(np.asarray(ev_mu, dtype=float))
                logL_gr_event_fd_list.append(np.asarray(ev_gr, dtype=float))
                logL_mu_fd += ev_mu
                logL_gr_fd += ev_gr

            # Apply selection normalization at the correct (combined draw) level.
            have_alpha = log_alpha_mu is not None and log_alpha_gr is not None
            if have_alpha:
                n_ev = int(len(scores))
                logL_mu_fd -= float(n_ev) * log_alpha_mu.reshape((1, -1))
                logL_gr_fd -= float(n_ev) * log_alpha_gr.reshape((1, -1))

            lpd_mu_f = _logmeanexp_axis(logL_mu_fd, axis=1)
            lpd_gr_f = _logmeanexp_axis(logL_gr_fd, axis=1)

            log_int_mu = log_prior_f + lpd_mu_f + logw_f
            log_int_gr = log_prior_f + lpd_gr_f + logw_f
            lpd_mu_total = _logsumexp_1d(log_int_mu)
            lpd_gr_total = _logsumexp_1d(log_int_gr)

            # Decompose the *marginalized* total into a data-only piece and a selection-correction piece.
            # Here, selection enters only through a draw-dependent subtraction -n_ev*log_alpha(draw).
            if have_alpha:
                assert log_alpha_mu is not None and log_alpha_gr is not None
                n_ev = int(len(scores))
                logL_mu_fd_data = logL_mu_fd + float(n_ev) * log_alpha_mu.reshape((1, -1))
                logL_gr_fd_data = logL_gr_fd + float(n_ev) * log_alpha_gr.reshape((1, -1))
                lpd_mu_f_data = _logmeanexp_axis(logL_mu_fd_data, axis=1)
                lpd_gr_f_data = _logmeanexp_axis(logL_gr_fd_data, axis=1)
                lpd_mu_total_data = _logsumexp_1d(log_prior_f + lpd_mu_f_data + logw_f)
                lpd_gr_total_data = _logsumexp_1d(log_prior_f + lpd_gr_f_data + logw_f)
            else:
                lpd_mu_total_data = float(lpd_mu_total)
                lpd_gr_total_data = float(lpd_gr_total)

            def _posterior_mean_f(log_int: np.ndarray) -> float:
                m = float(np.max(log_int))
                w = np.exp(log_int - m)
                denom = float(np.sum(w))
                return float(np.sum(w * f_grid) / denom) if denom > 0 else float("nan")

            f_post_mu = _posterior_mean_f(log_int_mu)
            f_post_gr = _posterior_mean_f(log_int_gr)

            # Save the (discretized) f_miss posterior for transparency/debugging.
            try:
                # Posterior mass on the quadrature grid (sums to 1 over grid points).
                def _norm_logpost(log_int: np.ndarray) -> np.ndarray:
                    m = float(np.max(log_int))
                    w = np.exp(log_int - m)
                    s = float(np.sum(w))
                    return (w / s) if s > 0 else np.full_like(w, np.nan)

                post_mu = _norm_logpost(log_int_mu)
                post_gr = _norm_logpost(log_int_gr)
                np.savez_compressed(
                    tab_dir / f"f_miss_posterior_{run_label}.npz",
                    f_grid=np.asarray(f_grid, dtype=np.float64),
                    posterior_mu=np.asarray(post_mu, dtype=np.float64),
                    posterior_gr=np.asarray(post_gr, dtype=np.float64),
                    log_integrand_mu=np.asarray(log_int_mu, dtype=np.float64),
                    log_integrand_gr=np.asarray(log_int_gr, dtype=np.float64),
                )
            except Exception:
                pass

            # Jackknife event influence on ΔLPD_total under the marginalized-f model.
            # This directly addresses “hero event” dependence at the *total score* level.
            try:
                jk: list[dict[str, Any]] = []
                n_ev = int(len(scores))
                for i, sc in enumerate(scores):
                    logL_mu_loo = logL_mu_fd - logL_mu_event_fd_list[i]
                    logL_gr_loo = logL_gr_fd - logL_gr_event_fd_list[i]
                    if have_alpha:
                        # logL_*_fd already includes -n_ev*log_alpha; LOO needs -(n_ev-1)*log_alpha.
                        assert log_alpha_mu is not None and log_alpha_gr is not None
                        logL_mu_loo = logL_mu_loo + log_alpha_mu.reshape((1, -1))
                        logL_gr_loo = logL_gr_loo + log_alpha_gr.reshape((1, -1))

                    lpd_mu_f_loo = _logmeanexp_axis(logL_mu_loo, axis=1)
                    lpd_gr_f_loo = _logmeanexp_axis(logL_gr_loo, axis=1)
                    lpd_mu_total_loo = _logsumexp_1d(log_prior_f + lpd_mu_f_loo + logw_f)
                    lpd_gr_total_loo = _logsumexp_1d(log_prior_f + lpd_gr_f_loo + logw_f)
                    delta_loo = float(lpd_mu_total_loo - lpd_gr_total_loo)
                    jk.append(
                        {
                            "event": str(sc.event),
                            "delta_lpd_total_leave_one_out": delta_loo,
                            "delta_lpd_total_full": float(float(lpd_mu_total) - float(lpd_gr_total)),
                            "influence": float(float(float(lpd_mu_total) - float(lpd_gr_total)) - delta_loo),
                        }
                    )
                _write_json(tab_dir / f"jackknife_{run_label}.json", jk)
            except Exception:
                pass
        else:
            lpd_mu_total = float(lpd_mu_total_ref)
            lpd_gr_total = float(lpd_gr_total_ref)
            lpd_mu_total_data = float(lpd_mu_total_data_ref)
            lpd_gr_total_data = float(lpd_gr_total_data_ref)
            f_post_mu = None
            f_post_gr = None

        summary = {
            "run": str(run_label),
            "convention": str(args.convention),
            "n_events": int(len(scores)),
            "n_draws": int(n_draws),
            "draw_idx": draw_idx,
            "lpd_mu_total": float(lpd_mu_total),
            "lpd_gr_total": float(lpd_gr_total),
            "delta_lpd_total": float(float(lpd_mu_total) - float(lpd_gr_total)),
            "lpd_mu_total_data": float(lpd_mu_total_data),
            "lpd_gr_total_data": float(lpd_gr_total_data),
            "delta_lpd_total_data": float(float(lpd_mu_total_data) - float(lpd_gr_total_data)),
            "lpd_mu_total_sel": float(float(lpd_mu_total) - float(lpd_mu_total_data)),
            "lpd_gr_total_sel": float(float(lpd_gr_total) - float(lpd_gr_total_data)),
            "delta_lpd_total_sel": float(float(float(lpd_mu_total) - float(lpd_gr_total)) - float(float(lpd_mu_total_data) - float(lpd_gr_total_data))),
            "lpd_mu_total_at_f_ref": float(lpd_mu_total_ref),
            "lpd_gr_total_at_f_ref": float(lpd_gr_total_ref),
            "delta_lpd_total_at_f_ref": float(float(lpd_mu_total_ref) - float(lpd_gr_total_ref)),
            "lpd_mu_total_data_at_f_ref": float(lpd_mu_total_data_ref),
            "lpd_gr_total_data_at_f_ref": float(lpd_gr_total_data_ref),
            "delta_lpd_total_data_at_f_ref": float(float(lpd_mu_total_data_ref) - float(lpd_gr_total_data_ref)),
            "lpd_mu_total_sel_at_f_ref": float(float(lpd_mu_total_ref) - float(lpd_mu_total_data_ref)),
            "lpd_gr_total_sel_at_f_ref": float(float(lpd_gr_total_ref) - float(lpd_gr_total_data_ref)),
            "delta_lpd_total_sel_at_f_ref": float(float(float(lpd_mu_total_ref) - float(lpd_gr_total_ref)) - float(float(lpd_mu_total_data_ref) - float(lpd_gr_total_data_ref))),
            "selection_alpha": alpha_meta,
            "mixture": {
                "mode": str(args.mixture_mode),
                "f_miss_ref": float(f_miss_ref) if f_miss_ref is not None else float(f_miss),
                "f_miss": float(f_miss_ref) if f_miss_ref is not None else float(f_miss),
                "f_miss_meta": f_miss_meta,
                "f_miss_posterior_mean_mu": float(f_post_mu) if f_post_mu is not None else None,
                "f_miss_posterior_mean_gr": float(f_post_gr) if f_post_gr is not None else None,
                "missing_pre": missing_pre_meta,
            }
            if str(args.mixture_mode) != "none"
            else {"mode": "none"},
        }
        _write_json(out_dir / f"summary_{run_label}.json", summary)
        _write_json(tab_dir / f"event_scores_{run_label}.json", [asdict(s) for s in scores])

        # Quick figure: per-event ΔLPD.
        if scores:
            plt.figure(figsize=(7, 4))
            xs = np.arange(len(scores))
            ys = np.array([s.delta_lpd for s in scores], dtype=float)
            plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
            plt.bar(xs, ys)
            plt.xticks(xs, [s.event for s in scores], rotation=45, ha="right", fontsize=7)
            plt.ylabel("ΔLPD (model − GR)")
            if do_f_miss_marginalize:
                plt.title(f"Dark siren ΔLPD by event ({run_label}; f_ref={float(f_miss_ref):.3f})")
            else:
                plt.title(f"Dark siren ΔLPD by event ({run_label})")
            plt.tight_layout()
            plt.savefig(fig_dir / f"delta_lpd_by_event_{run_label}.png", dpi=160)
            plt.close()

    manifest = {
        "timestamp_utc": _utc_stamp(),
        "run_dir": run_dirs,
        "skymap_dir": str(args.skymap_dir),
        "glade_index": str(args.glade_index),
        "gw_data_mode": str(args.gw_data_mode),
        "convention": str(args.convention),
        "galaxy_null_mode": str(args.galaxy_null_mode),
        "galaxy_null_seed": int(args.galaxy_null_seed),
        "gw_distance_prior_mode": str(args.gw_distance_prior_mode),
        "gw_distance_prior_power": float(args.gw_distance_prior_power),
        "gw_distance_prior_h0_ref": float(args.gw_distance_prior_h0_ref),
        "gw_distance_prior_omega_m0": float(args.gw_distance_prior_omega_m0),
        "gw_distance_prior_omega_k0": float(args.gw_distance_prior_omega_k0),
        "gw_distance_prior_zmax": float(args.gw_distance_prior_zmax),
        "pe_base_dir": str(Path(str(args.pe_base_dir)).expanduser().resolve()),
        "pe_record_id": [int(x) for x in (args.pe_record_id or [])],
        "pe_prefer_variant": [str(v) for v in (args.pe_prefer_variant or [])],
        "pe_analysis": str(args.pe_analysis) if args.pe_analysis is not None else None,
        "pe_max_samples": int(args.pe_max_samples) if args.pe_max_samples is not None else None,
        "pe_seed": int(args.pe_seed),
        "pe_nside": int(args.pe_nside),
        "pe_dl_nbins": int(args.pe_dl_nbins),
        "pe_dl_qmin": float(args.pe_dl_qmin),
        "pe_dl_qmax": float(args.pe_dl_qmax),
        "pe_dl_pad_factor": float(args.pe_dl_pad_factor),
        "pe_dl_pseudocount": float(args.pe_dl_pseudocount),
        "pe_dl_smooth_iters": int(args.pe_dl_smooth_iters),
        "pe_like_mode": str(args.pe_like_mode),
        "pe_distance_mode": str(args.pe_distance_mode),
        "p_credible": float(args.p_credible),
        "gal_z_max": float(args.gal_z_max),
        "max_area_deg2": float(args.max_area_deg2),
        "max_gal": int(args.max_gal),
        "max_events": int(args.max_events),
        "events_filter": want_events,
        "max_draws": int(args.max_draws),
        "galaxy_chunk_size": int(args.galaxy_chunk_size),
        "selection_injections_hdf": str(resolved_injections_path) if resolved_injections_path is not None else None,
        "selection_injections_spec": str(args.selection_injections_hdf),
        "selection_o4_injections_hdf": str(args.selection_o4_injections_hdf) if args.selection_o4_injections_hdf is not None else None,
        "selection_auto_record_id_o3": int(args.selection_auto_record_id_o3),
        "selection_auto_record_id_o4": int(args.selection_auto_record_id_o4) if args.selection_auto_record_id_o4 is not None else None,
        "selection_auto_strict": bool(args.selection_auto_strict),
        "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
        "selection_z_max": float(args.selection_z_max) if args.selection_z_max is not None else None,
        "selection_det_model": str(args.selection_det_model),
        "selection_snr_thresh": float(args.selection_snr_thresh) if args.selection_snr_thresh is not None else None,
        "selection_snr_binned_nbins": int(args.selection_snr_binned_nbins),
        "selection_injection_logit_l2": float(args.selection_injection_logit_l2),
        "selection_injection_logit_max_iter": int(args.selection_injection_logit_max_iter),
        "selection_weight_mode": str(args.selection_weight_mode),
        "selection_pop_z_mode": str(args.selection_pop_z_mode),
        "selection_pop_z_k": float(args.selection_pop_z_k),
        "selection_pop_mass_mode": str(args.selection_pop_mass_mode),
        "selection_pop_m1_alpha": float(args.selection_pop_m1_alpha),
        "selection_pop_m_min": float(args.selection_pop_m_min),
        "selection_pop_m_max": float(args.selection_pop_m_max),
        "selection_pop_q_beta": float(args.selection_pop_q_beta),
        "completeness_mode": str(args.completeness_mode),
        "completeness_zref_max": float(args.completeness_zref_max),
        "completeness_nbins": int(args.completeness_nbins),
        "completeness_c_floor": float(args.completeness_c_floor),
        "mixture_mode": str(args.mixture_mode),
        "mixture_f_miss_mode": str(args.mixture_f_miss_mode),
        "mixture_f_miss": float(args.mixture_f_miss) if args.mixture_f_miss is not None else None,
        "mixture_f_miss_prior": str(args.mixture_f_miss_prior),
        "mixture_f_miss_beta_mean": float(args.mixture_f_miss_beta_mean) if args.mixture_f_miss_beta_mean is not None else None,
        "mixture_f_miss_beta_kappa": float(args.mixture_f_miss_beta_kappa),
        "mixture_f_miss_marginalize_n": int(args.mixture_f_miss_marginalize_n),
        "mixture_f_miss_marginalize_eps": float(args.mixture_f_miss_marginalize_eps),
        "host_prior_z_mode": str(args.host_prior_z_mode),
        "host_prior_z_k": float(args.host_prior_z_k),
        "missing_z_max": float(args.missing_z_max),
        "missing_dl_grid_n": int(args.missing_dl_grid_n),
        "missing_dl_min_mpc": float(args.missing_dl_min_mpc),
        "missing_dl_max_mpc": float(args.missing_dl_max_mpc) if args.missing_dl_max_mpc is not None else None,
        "missing_dl_nsigma": float(args.missing_dl_nsigma),
        "missing_pixel_chunk_size": int(args.missing_pixel_chunk_size),
        "gr_h0_mode": str(args.gr_h0_mode),
        "gr_h0_grid_min": float(args.gr_h0_grid_min),
        "gr_h0_grid_max": float(args.gr_h0_grid_max),
        "gr_h0_grid_n": int(args.gr_h0_grid_n),
        "gr_h0_omega_m0": float(args.gr_h0_omega_m0),
        "gr_h0_omega_k0": float(args.gr_h0_omega_k0),
        "gr_h0_prior": str(args.gr_h0_prior),
        "gr_h0_smoke": bool(args.gr_h0_smoke),
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(f"[dark_sirens] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
