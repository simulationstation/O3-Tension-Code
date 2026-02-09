from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix
from entropy_horizon_recon.void_prism_maps import (
    eg_void_from_spectra,
    jackknife_covariance,
    jackknife_region_index,
    load_void_catalog_csv,
    measure_void_prism_spectra,
    select_voids_by_z,
)

# Avoid fork-related deadlocks and OpenMP oversubscription:
# - we use a ThreadPool for jackknife parallelism
# - keep OpenMP at 1 thread per call (healpy/libsharp may use OpenMP internally)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str | None) -> list[str]:
    out = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not out:
        raise ValueError("Expected a comma-separated list.")
    return out


def _none_or_str(s: str | None) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    if t.lower() in ("none", "null"):
        return None
    return t


def _parse_edges(s: str | None) -> np.ndarray:
    if not s:
        raise ValueError("Provide --bin-edges (comma-separated).")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([int(p) for p in parts], dtype=int)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin-edges must be strictly increasing with length>=2.")
    return edges


def _parse_float_edges(s: str | None) -> np.ndarray:
    if not s:
        raise ValueError("Provide comma-separated edges.")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([float(p) for p in parts], dtype=float)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Edges must be finite, strictly increasing, length>=2.")
    return edges


def _region_pix_index(region_id: np.ndarray, good_regions: np.ndarray) -> dict[int, np.ndarray]:
    """Return mapping rid -> indices of high-res pixels in that jackknife region.

    Uses one global argsort to avoid scanning the full region_id array per region.
    """
    region_id = np.asarray(region_id, dtype=np.int32)
    order = np.argsort(region_id, kind="mergesort")
    r_sorted = region_id[order]
    # Starts of new rids in sorted order.
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    ends = np.r_[starts[1:], r_sorted.size]
    good = set(int(r) for r in np.asarray(good_regions, dtype=int))
    out: dict[int, np.ndarray] = {}
    for s, e in zip(starts, ends, strict=False):
        rid = int(r_sorted[s])
        if rid in good:
            out[rid] = order[s:e]
    if len(out) != len(good):
        missing = sorted(good.difference(out.keys()))
        raise RuntimeError(f"Failed to build region index for rids={missing[:10]} (and more)." if missing else "Region index mismatch.")
    return out


@dataclass(frozen=True)
class Block:
    name: str
    zbin_idx: int
    z_min: float
    z_max: float
    z_eff: float
    n_voids: int
    rv_min: float | None
    rv_max: float | None
    rv_split: float | None
    # Pre-binned tracer counts for this block. This avoids re-binning the (often huge) catalog
    # for each leave-one-region-out jackknife resample; the jackknife changes the sky mask, not
    # the tracer locations.
    counts_map: np.ndarray
    ell: np.ndarray


def _overdensity_from_counts(counts: np.ndarray, *, mask: np.ndarray, allow_empty: bool = False) -> np.ndarray:
    """Convert a weighted counts map into a mean-subtracted overdensity map under a given mask."""
    counts = np.asarray(counts, dtype=float)
    m = np.asarray(mask, dtype=float)
    if counts.shape != m.shape:
        raise ValueError("counts/mask shape mismatch")
    good = m > 0
    mean = float(np.mean(counts[good])) if np.any(good) else float(np.mean(counts))
    if not np.isfinite(mean) or mean <= 0:
        if allow_empty:
            out = np.zeros_like(counts)
            out[~good] = 0.0
            return out
        raise ValueError("Non-positive counts mean; check mask/catalog.")
    delta = counts / mean - 1.0
    delta[~good] = 0.0
    return delta


@dataclass(frozen=True)
class SuiteMeta:
    suite_name: str
    void_csv: str
    theta_fits: list[str]
    theta_mask_fits: list[str] | None
    kappa_source: str
    extra_mask_fits: str | None
    frame: str
    nside: int
    lmax: int | None
    bin_edges: list[int]
    prefactor: float
    eg_sign: float
    z_edges: list[float]
    rv_split_mode: str
    jackknife_nside: int | None
    n_regions: int | None
    n_proc: int
    ra_col: str
    dec_col: str
    z_col: str
    Rv_col: str | None
    weight_col: str | None


# -----------------------------------------------------------------------------
# Multiprocessing (fork) globals.
# -----------------------------------------------------------------------------

_JK_KAPPA: np.ndarray | None = None
_JK_THETA_LIST: list[np.ndarray] | None = None
_JK_THETA_MASK_LIST: list[np.ndarray] | None = None
_JK_MASK: np.ndarray | None = None
_JK_REGION_PIX: dict[int, np.ndarray] | None = None
_JK_BLOCKS: list[Block] | None = None
_JK_NSIDE: int | None = None
_JK_LMAX: int | None = None
_JK_BIN_EDGES: np.ndarray | None = None
_JK_PREFACTOR: float | None = None


def _jk_one_region(rid: int) -> np.ndarray:
    kappa = _JK_KAPPA
    theta_list = _JK_THETA_LIST
    theta_mask_list = _JK_THETA_MASK_LIST
    mask0 = _JK_MASK
    region_pix = _JK_REGION_PIX
    blocks = _JK_BLOCKS
    nside = _JK_NSIDE
    lmax = _JK_LMAX
    bin_edges = _JK_BIN_EDGES
    pref = _JK_PREFACTOR
    if any(v is None for v in (kappa, theta_list, theta_mask_list, mask0, region_pix, blocks, nside, bin_edges, pref)):
        raise RuntimeError("Jackknife globals not initialized.")

    m = np.asarray(mask0, dtype=float).copy()
    m[region_pix[int(rid)]] = 0.0

    pieces: list[np.ndarray] = []
    for b in blocks:
        theta = np.asarray(theta_list[int(b.zbin_idx)], dtype=float)
        tmask = np.asarray(theta_mask_list[int(b.zbin_idx)], dtype=float)
        m_block = m * tmask
        vmap = _overdensity_from_counts(b.counts_map, mask=m_block, allow_empty=True)
        res = measure_void_prism_spectra(
            kappa_map=np.asarray(kappa, dtype=float),
            theta_map=np.asarray(theta, dtype=float),
            void_delta_map=vmap,
            mask=m_block,
            lmax=int(lmax) if lmax is not None else None,
            bin_edges=bin_edges,
        )
        eg = eg_void_from_spectra(cl_kappa_void=res["cl_kappa_void"], cl_theta_void=res["cl_theta_void"], prefactor=float(pref))
        pieces.append(eg)
    return np.concatenate(pieces, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure a z-binned + Rv-binned void-prism E_G suite with optional joint jackknife covariance.")
    ap.add_argument("--void-csv", required=True, help="Void catalog CSV with ra/dec/z and optionally Rv_mpc_h and weight_ngal.")
    ap.add_argument(
        "--theta-fits",
        required=True,
        help=(
            "HEALPix theta/velocity-proxy map(s). Provide either a single FITS path, or a comma-separated list "
            "matching the number of z bins (len(z_edges)-1) for tomographic theta."
        ),
    )
    ap.add_argument(
        "--theta-mask-fits",
        default=None,
        help=(
            "Optional HEALPix mask(s) for the theta map(s). Provide either a single FITS path, or a comma-separated list "
            "matching the number of z bins (len(z_edges)-1) for tomographic theta. If omitted, theta is assumed defined "
            "wherever the global analysis mask is nonzero (not recommended for sparse tracer-defined theta)."
        ),
    )
    ap.add_argument("--ra-col", default="ra", help="Catalog RA column (degrees, default 'ra').")
    ap.add_argument("--dec-col", default="dec", help="Catalog Dec column (degrees, default 'dec').")
    ap.add_argument("--z-col", default="z", help="Catalog redshift column (default 'z').")
    ap.add_argument("--Rv-col", default="Rv_mpc_h", help="Catalog void-radius column (default 'Rv_mpc_h'; use 'none' for no radius).")
    ap.add_argument("--weight-col", default="weight_ngal", help="Catalog weight column (default 'weight_ngal'; use 'none' for uniform weights).")
    ap.add_argument("--planck", action="store_true", help="Use Planck 2018 lensing kappa from PLA (download/cache).")
    ap.add_argument("--kappa-fits", default=None, help="If not --planck, provide kappa map fits.")
    ap.add_argument("--mask-fits", default=None, help="Optional mask fits (if not --planck).")
    ap.add_argument("--extra-mask-fits", default=None, help="Optional additional mask to multiply into the analysis mask.")
    ap.add_argument("--frame", choices=["icrs", "galactic"], default="galactic", help="Frame of the input HEALPix maps (default galactic).")

    ap.add_argument("--nside", type=int, default=256, help="Target nside for all maps (default 256).")
    ap.add_argument("--lmax", type=int, default=None, help="Optional lmax for anafast.")
    ap.add_argument("--bin-edges", required=True, help="Comma-separated ell bin edges, e.g. '0,20,50,100,200,400,700'.")
    ap.add_argument("--prefactor", type=float, default=1.0, help="Overall normalization constant (definition-dependent).")
    ap.add_argument("--eg-sign", type=float, default=1.0, help="Multiply EG by this sign (+1 or -1). Default +1.")
    ap.add_argument(
        "--auto-eg-sign",
        action="store_true",
        help="If enabled, flip the overall EG sign so that the median of y_obs is positive. "
        "This is a convention/alignment fix; covariance is unchanged by an overall sign flip.",
    )

    ap.add_argument("--z-edges", required=True, help="Comma-separated z edges, e.g. '0.00,0.03,0.05,0.07'.")
    ap.add_argument(
        "--rv-split",
        choices=["none", "median"],
        default="median",
        help="How to split each z-bin by void radius Rv (default median; 'none' disables Rv split).",
    )
    ap.add_argument("--rv-min", type=float, default=None, help="Optional global min Rv (Mpc/h).")
    ap.add_argument("--rv-max", type=float, default=None, help="Optional global max Rv (Mpc/h).")
    ap.add_argument("--min-voids", type=int, default=30, help="Minimum voids per (z,Rv) bin (default 30).")

    ap.add_argument("--jackknife-nside", type=int, default=None, help="If set, compute a joint JK covariance using these regions (e.g. 2 or 4).")
    ap.add_argument("--n-proc", type=int, default=1, help="Parallel processes for jackknife (default 1).")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print jackknife progress every N regions (default 10; always prints for <=200 regions).",
    )

    ap.add_argument("--suite-name", default=None, help="Human-friendly suite name.")
    ap.add_argument("--out-base", default=None, help="Output directory (default: outputs/void_prism_eg_suite_<stamp>/).")
    args = ap.parse_args()

    global _JK_KAPPA, _JK_THETA_LIST, _JK_THETA_MASK_LIST, _JK_MASK, _JK_REGION_PIX, _JK_BLOCKS, _JK_NSIDE, _JK_LMAX, _JK_BIN_EDGES, _JK_PREFACTOR

    hp = _require_healpy()

    out_dir = Path(args.out_base) if args.out_base else Path("outputs") / f"void_prism_eg_suite_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    nside = int(args.nside)
    bin_edges = _parse_edges(args.bin_edges)
    z_edges = _parse_float_edges(args.z_edges)

    # Load lensing kappa + mask.
    if args.planck:
        planck = load_planck_kappa(paths=DataPaths(Path.cwd()), nside_out=nside)
        kappa = planck.kappa_map
        mask = planck.mask
        kappa_source = "planck"
    else:
        if args.kappa_fits is None:
            raise ValueError("Provide --kappa-fits or set --planck.")
        kappa = hp.read_map(str(args.kappa_fits), verbose=False)
        mask = hp.read_map(str(args.mask_fits), verbose=False) if args.mask_fits else None
        kappa_source = str(args.kappa_fits)

    theta_paths = _parse_csv_list(str(args.theta_fits))
    theta_list = [hp.read_map(str(p), verbose=False) for p in theta_paths]
    theta_mask_paths = _parse_csv_list(args.theta_mask_fits) if args.theta_mask_fits else None
    theta_mask_list = [hp.read_map(str(p), verbose=False) for p in theta_mask_paths] if theta_mask_paths else None

    def _ud(m: np.ndarray) -> np.ndarray:
        return hp.ud_grade(m, nside) if hp.get_nside(m) != nside else m

    kappa = _ud(kappa)
    theta_list = [_ud(np.asarray(t, dtype=float)) for t in theta_list]
    if theta_mask_list is None:
        theta_mask_list = [np.ones_like(theta_list[0], dtype=float) for _ in range(len(theta_list))]
    else:
        theta_mask_list = [_ud(np.asarray(t, dtype=float)) for t in theta_mask_list]
    if mask is None:
        mask = np.ones_like(kappa)
    else:
        mask = _ud(mask)
    if args.extra_mask_fits:
        extra = hp.read_map(str(args.extra_mask_fits), verbose=False)
        extra = _ud(extra)
        mask = mask * np.asarray(extra, dtype=float)

    # Load catalog + global z selection.
    Rv_col = _none_or_str(args.Rv_col)
    weight_col = _none_or_str(args.weight_col)
    cat_all = load_void_catalog_csv(
        args.void_csv,
        ra_col=str(args.ra_col),
        dec_col=str(args.dec_col),
        z_col=str(args.z_col),
        Rv_col=Rv_col,
        weight_col=weight_col,
    )
    cat_all = select_voids_by_z(cat_all, z_min=float(z_edges[0]), z_max=float(z_edges[-1]))
    if str(args.rv_split) != "none" and cat_all.Rv is None:
        raise ValueError("Rv split requested but catalog has no Rv column. Pass --Rv-col or use --rv-split none.")

    # Precompute HEALPix pixels once for the full catalog.
    pix_all = radec_to_healpix(cat_all.ra_deg, cat_all.dec_deg, nside=nside, frame=str(args.frame), nest=False)
    w_all = np.ones_like(pix_all, dtype=float) if cat_all.weight is None else np.asarray(cat_all.weight, dtype=float)

    # Choose theta map per z bin.
    # - If one theta map is provided, we reuse it for all z bins.
    # - If N maps are provided, require N == (len(z_edges)-1).
    if len(theta_list) == 1:
        theta_list = [theta_list[0] for _ in range(int(z_edges.size - 1))]
    if len(theta_mask_list) == 1:
        theta_mask_list = [theta_mask_list[0] for _ in range(int(z_edges.size - 1))]
    if len(theta_list) != int(z_edges.size - 1):
        raise ValueError(
            f"theta-fits count mismatch: got {len(theta_list)} maps but need 1 or {int(z_edges.size - 1)} "
            f"(one per z bin)."
        )
    if len(theta_mask_list) != int(z_edges.size - 1):
        raise ValueError(
            f"theta-mask-fits count mismatch: got {len(theta_mask_list)} masks but need 1 or {int(z_edges.size - 1)} "
            f"(one per z bin)."
        )

    blocks: list[Block] = []
    for iz, (z0, z1) in enumerate(zip(z_edges[:-1], z_edges[1:], strict=False)):
        m_z = (cat_all.z >= float(z0)) & (cat_all.z < float(z1))
        if not np.any(m_z):
            continue
        z_sel = cat_all.z[m_z]
        if cat_all.Rv is None:
            Rv_sel = None
        else:
            Rv_sel = np.asarray(cat_all.Rv, dtype=float)[m_z]
        pix_sel = pix_all[m_z]
        w_sel = w_all[m_z]

        # Global Rv cuts (if available).
        if Rv_sel is not None:
            if args.rv_min is not None:
                m_r = Rv_sel >= float(args.rv_min)
                z_sel, Rv_sel, pix_sel, w_sel = z_sel[m_r], Rv_sel[m_r], pix_sel[m_r], w_sel[m_r]
            if args.rv_max is not None:
                m_r = Rv_sel <= float(args.rv_max)
                z_sel, Rv_sel, pix_sel, w_sel = z_sel[m_r], Rv_sel[m_r], pix_sel[m_r], w_sel[m_r]

        if z_sel.size < int(args.min_voids):
            continue

        z_eff = float(np.mean(z_sel))
        ell_ref: np.ndarray | None = None

        if args.rv_split == "none":
            name = f"zbin{iz}_z{z0:.3f}-{z1:.3f}"
            blocks.append(
                Block(
                    name=name,
                    zbin_idx=int(iz),
                    z_min=float(z0),
                    z_max=float(z1),
                    z_eff=z_eff,
                    n_voids=int(z_sel.size),
                    rv_min=None,
                    rv_max=None,
                    rv_split=None,
                    pix=np.asarray(pix_sel, dtype=int),
                    weights=np.asarray(w_sel, dtype=float),
                    ell=np.array([], dtype=int),
                )
            )
            continue

        # Split by median within the z bin.
        if Rv_sel is None:
            raise ValueError("rv_split requested but Rv is missing in this selection.")
        rv_split = float(np.median(Rv_sel))
        m_small = Rv_sel <= rv_split
        m_large = Rv_sel > rv_split
        for tag, m_rv, rv_lo, rv_hi in (
            ("small", m_small, None, rv_split),
            ("large", m_large, rv_split, None),
        ):
            if not np.any(m_rv):
                continue
            if int(np.sum(m_rv)) < int(args.min_voids):
                continue
            name = f"zbin{iz}_{tag}_z{z0:.3f}-{z1:.3f}"
            blocks.append(
                Block(
                    name=name,
                    zbin_idx=int(iz),
                    z_min=float(z0),
                    z_max=float(z1),
                    z_eff=z_eff,
                    n_voids=int(np.sum(m_rv)),
                    rv_min=float(rv_lo) if rv_lo is not None else None,
                    rv_max=float(rv_hi) if rv_hi is not None else None,
                    rv_split=rv_split,
                    counts_map=np.bincount(
                        np.asarray(pix_sel[m_rv], dtype=int),
                        weights=np.asarray(w_sel[m_rv], dtype=float),
                        minlength=int(hp.nside2npix(int(nside))),
                    ).astype(float),
                    ell=np.array([], dtype=int),
                )
            )

    if not blocks:
        raise RuntimeError("No (z,Rv) blocks created; check --z-edges/--min-voids and catalog.")

    # Base measurement (full mask): compute EG per block and concatenate.
    eg_blocks: list[np.ndarray] = []
    ell_common: np.ndarray | None = None
    eg_sign = float(args.eg_sign)
    if eg_sign not in (-1.0, 1.0):
        raise ValueError("--eg-sign must be +1 or -1.")
    for b in blocks:
        theta = np.asarray(theta_list[int(b.zbin_idx)], dtype=float)
        tmask = np.asarray(theta_mask_list[int(b.zbin_idx)], dtype=float)
        m_block = np.asarray(mask, dtype=float) * tmask
        vmap = _overdensity_from_counts(b.counts_map, mask=m_block, allow_empty=True)
        res = measure_void_prism_spectra(
            kappa_map=kappa,
            theta_map=theta,
            void_delta_map=vmap,
            mask=m_block,
            lmax=int(args.lmax) if args.lmax is not None else None,
            bin_edges=bin_edges,
        )
        ell = res["ell"].astype(int)
        if ell_common is None:
            ell_common = ell
        elif not np.allclose(ell_common, ell):
            raise RuntimeError("ell bins differ across blocks; ensure common bin_edges.")
        eg = eg_void_from_spectra(cl_kappa_void=res["cl_kappa_void"], cl_theta_void=res["cl_theta_void"], prefactor=float(args.prefactor))
        eg_blocks.append(eg_sign * eg)

    if ell_common is None:
        raise RuntimeError("Failed to compute any block spectra.")

    y_obs = np.concatenate(eg_blocks, axis=0)
    if bool(args.auto_eg_sign) and np.isfinite(np.nanmedian(y_obs)) and float(np.nanmedian(y_obs)) < 0.0:
        # Flip the *overall* sign so the observed EG is positive by convention.
        # This avoids an unphysical negative fitted amplitude in the scorer when the only mismatch
        # is a sign convention in the theta/void field definitions.
        eg_sign *= -1.0
        eg_blocks = [(-1.0) * eg for eg in eg_blocks]
        y_obs = -1.0 * y_obs

    # Write per-block measurements (with block-diagonal cov extracted later).
    measurements: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    off = 0
    for b, eg in zip(blocks, eg_blocks, strict=False):
        sl = (int(off), int(off + eg.size))
        off = sl[1]
        env: dict[str, Any] = {
            "z_min": b.z_min,
            "z_max": b.z_max,
            "rv_min": b.rv_min,
            "rv_max": b.rv_max,
            "rv_split": b.rv_split,
            "n_voids": b.n_voids,
        }
        slices.append({"name": b.name, "z_eff": b.z_eff, "ell": ell_common.tolist(), "slice": [sl[0], sl[1]], "env": env})
        measurements.append(
            {
                "name": b.name,
                "z_eff": b.z_eff,
                "ell": ell_common.astype(int).tolist(),
                "eg_obs": eg.tolist(),
                "cov": np.diag(np.full(eg.size, 1.0)).tolist(),  # replaced below if JK cov computed
                "env": env,
            "notes": "Suite block; covariance is block-only unless joint scoring is used.",
            "source": {
                "kappa": kappa_source,
                "theta": theta_paths if len(theta_paths) > 1 else theta_paths[0],
                "void_csv": str(args.void_csv),
                "prefactor": float(args.prefactor),
                "eg_sign": float(eg_sign),
                "z_edges": z_edges.tolist(),
                "rv_split": str(args.rv_split),
                },
            }
        )

    cov_joint = None
    if args.jackknife_nside is not None:
        nside_jk = int(args.jackknife_nside)
        region_id = jackknife_region_index(nside=nside, nside_jk=nside_jk)
        good_regions = np.unique(region_id[np.asarray(mask) > 0])
        region_pix = _region_pix_index(region_id, good_regions)

        n_proc = int(args.n_proc)
        progress_every = int(args.progress_every)
        if good_regions.size <= 200:
            progress_every = 1
        print(
            f"[jk] nside_jk={nside_jk} regions={good_regions.size} blocks={len(blocks)} vec_dim={y_obs.size} n_proc={n_proc}",
            flush=True,
        )

        # Set globals for worker function (threads share memory, so no pickling overhead).
        _JK_KAPPA = kappa
        _JK_THETA_LIST = theta_list
        _JK_THETA_MASK_LIST = theta_mask_list
        _JK_MASK = mask
        _JK_REGION_PIX = region_pix
        _JK_BLOCKS = blocks
        _JK_NSIDE = nside
        _JK_LMAX = int(args.lmax) if args.lmax is not None else None
        _JK_BIN_EDGES = bin_edges
        _JK_PREFACTOR = float(args.prefactor) * float(eg_sign)

        jk = np.empty((good_regions.size, y_obs.size), dtype=float)
        if n_proc <= 1:
            for i, rid in enumerate(good_regions):
                jk[i] = _jk_one_region(int(rid))
                if (i + 1) % progress_every == 0 or (i + 1) == good_regions.size:
                    pct = 100.0 * float(i + 1) / float(good_regions.size)
                    print(f"[jk] {i+1}/{good_regions.size} ({pct:.1f}%)", flush=True)
        else:
            # Multiprocessing with fork: much faster for HEALPix ops because we avoid pickling
            # large maps into each worker. We keep OpenMP threads at 1 to reduce fork risks.
            rids = [int(r) for r in good_regions.tolist()]
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_proc) as pool:
                for i, vec in enumerate(pool.imap(_jk_one_region, rids, chunksize=1), start=1):
                    jk[i - 1] = vec
                    if i % progress_every == 0 or i == len(rids):
                        pct = 100.0 * float(i) / float(len(rids))
                        print(f"[jk] {i}/{len(rids)} ({pct:.1f}%)", flush=True)

        cov_joint = jackknife_covariance(jk)

        # Populate per-block covariances as the block-diagonal slices (for convenience plots).
        for mi, s in enumerate(slices):
            lo, hi = s["slice"]
            blk = np.asarray(cov_joint, dtype=float)[lo:hi, lo:hi]
            measurements[mi]["cov"] = blk.tolist()

    suite_name = args.suite_name or f"void_prism_suite_{Path(args.void_csv).stem}"
    meta = SuiteMeta(
        suite_name=str(suite_name),
        void_csv=str(args.void_csv),
        theta_fits=[str(p) for p in theta_paths],
        theta_mask_fits=[str(p) for p in theta_mask_paths] if theta_mask_paths else None,
        kappa_source=str(kappa_source),
        extra_mask_fits=str(args.extra_mask_fits) if args.extra_mask_fits else None,
        frame=str(args.frame),
        nside=nside,
        lmax=int(args.lmax) if args.lmax is not None else None,
        bin_edges=bin_edges.astype(int).tolist(),
        prefactor=float(args.prefactor),
        eg_sign=float(eg_sign),
        z_edges=z_edges.tolist(),
        rv_split_mode=str(args.rv_split),
        jackknife_nside=int(args.jackknife_nside) if args.jackknife_nside is not None else None,
        n_regions=int(good_regions.size) if args.jackknife_nside is not None else None,
        n_proc=int(args.n_proc),
        ra_col=str(args.ra_col),
        dec_col=str(args.dec_col),
        z_col=str(args.z_col),
        Rv_col=Rv_col,
        weight_col=weight_col,
    )

    (tab_dir / "measurements.json").write_text(json.dumps({"measurements": measurements}, indent=2))
    (tab_dir / "suite_joint.json").write_text(
        json.dumps(
            {
                "meta": asdict(meta),
                "blocks": slices,
                "y_obs": y_obs.tolist(),
                "cov": cov_joint.tolist() if cov_joint is not None else None,
            },
            indent=2,
        )
    )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
