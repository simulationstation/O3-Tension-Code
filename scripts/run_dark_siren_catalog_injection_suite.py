#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import betaln, logsumexp

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.dark_sirens_incompleteness import (
    compute_missing_host_logL_draws_from_histogram,
    precompute_missing_host_prior,
)
from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram, compute_dark_siren_logL_draws_from_pe_hist
from entropy_horizon_recon.dark_sirens_selection import (
    compute_selection_alpha_from_injections,
    load_o3_injections,
    resolve_o3_sensitivity_injection_file,
)
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior
from entropy_horizon_recon.sirens import (
    MuForwardPosterior,
    load_mu_forward_posterior,
    predict_dL_em,
    predict_r_gw_em,
    x_of_z_from_H,
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _default_nproc() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return int(os.cpu_count() or 1)


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be 1D with >=2 points.")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be finite and strictly increasing.")
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
        raise ValueError("_logsumexp_1d expects 1D array.")
    m = float(np.max(x))
    return float(m + np.log(np.sum(np.exp(x - m))))


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _logmeanexp_axis(x: np.ndarray, *, axis: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def _downsample_posterior(post_full: MuForwardPosterior, *, max_draws: int, seed: int) -> tuple[MuForwardPosterior, np.ndarray]:
    n = int(post_full.H_samples.shape[0])
    if int(max_draws) <= 0:
        raise ValueError("max_draws must be positive.")
    if n <= int(max_draws):
        idx = np.arange(n, dtype=np.int64)
        return post_full, idx
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_draws), replace=False).astype(np.int64)
    idx.sort()
    sigma8 = post_full.sigma8_0[idx] if post_full.sigma8_0 is not None else None
    post = MuForwardPosterior(
        x_grid=post_full.x_grid,
        logmu_x_samples=post_full.logmu_x_samples[idx],
        z_grid=post_full.z_grid,
        H_samples=post_full.H_samples[idx],
        H0=post_full.H0[idx],
        omega_m0=post_full.omega_m0[idx],
        omega_k0=post_full.omega_k0[idx],
        sigma8_0=sigma8,
    )
    return post, idx


@dataclass(frozen=True)
class TemplateEvent:
    event: str
    z_gal: np.ndarray
    w_gal: np.ndarray
    ipix_gal: np.ndarray
    pe_nside: int
    p_credible: float
    pix_sel: np.ndarray
    prob_pix: np.ndarray
    sigma_ln_like: float
    z_bins: np.ndarray
    z_cdf: np.ndarray
    spectral_chunks: tuple["SpectralChunk", ...]


@dataclass
class SpectralChunk:
    # Unique redshifts in this chunk (sorted), and summed log-weights log(sum w*prob) per unique z.
    z_u: np.ndarray
    logweight_u: np.ndarray
    # Optional precomputed distance tables across all scoring draws; shape (n_draws, n_uniq).
    dL_em_u: np.ndarray | None = None
    dL_gw_u: np.ndarray | None = None


_G: dict[str, Any] = {}


def _rep_partial_path(reps_dir: Path, rep: int) -> Path:
    return reps_dir / f"rep{int(rep):04d}_partial.json"


def _heartbeat_status(
    reps_dir: Path,
    *,
    n_rep_total: int,
    n_events_per_rep: int,
    recent_sec: float = 180.0,
) -> dict[str, Any]:
    rep_done = sorted(reps_dir.glob("rep[0-9][0-9][0-9][0-9].json"))
    partials = sorted(reps_dir.glob("rep[0-9][0-9][0-9][0-9]_partial.json"))

    events_done_active = 0.0
    latest_partial: dict[str, Any] | None = None
    latest_mtime = -1.0
    n_recent = 0
    now_unix = float(time.time())
    for p in partials:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        try:
            v = d.get("events_done_float", None)
            if v is None:
                v = d.get("events_done", 0)
            events_done_active += float(v)
        except Exception:
            pass
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = -1.0
        if mtime > 0.0 and (now_unix - mtime) <= float(recent_sec):
            n_recent += 1
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_partial = d

    n_done = int(len(rep_done))
    n_active = int(len(partials))
    n_events_total = int(n_rep_total) * int(n_events_per_rep)
    n_events_done_est = float(int(n_done) * int(n_events_per_rep)) + float(events_done_active)
    if n_events_total > 0 and n_events_done_est > float(n_events_total):
        n_events_done_est = float(n_events_total)

    return {
        "n_rep_done": n_done,
        "n_rep_total": int(n_rep_total),
        "n_rep_active": n_active,
        "n_events_done_est": float(n_events_done_est),
        "n_events_total": int(n_events_total),
        "latest_partial": latest_partial,
        "latest_partial_mtime_unix": float(latest_mtime),
        "n_partials_recent": int(n_recent),
        "partials_recent_window_sec": float(recent_sec),
    }


def _run_one_rep(rep: int) -> dict[str, Any]:
    cfg = dict(_G.get("cfg", {}))
    reps_dir = Path(str(cfg["reps_dir"]))
    rep_path = reps_dir / f"rep{int(rep):04d}.json"
    partial_path = _rep_partial_path(reps_dir, int(rep))

    # Re-entrancy: if a previous run completed this rep, just read+return.
    if rep_path.exists():
        try:
            return json.loads(rep_path.read_text())
        except Exception:
            pass

    templates: list[TemplateEvent] = list(_G["templates"])
    post: MuForwardPosterior = _G["post"]
    gw_prior: GWDistancePrior = _G["gw_prior"]
    constants: PhysicalConstants = _G["constants"]
    missing_pre = _G["missing_pre"]
    log_alpha_mu = _G.get("log_alpha_mu")
    log_alpha_gr = _G.get("log_alpha_gr")

    n_draws = int(cfg["n_draws"])
    seed_base = int(cfg["seed_base"])
    truth_model = str(cfg["truth_model"])
    convention = str(cfg["convention"])
    dl_nbins = int(cfg["dl_nbins"])
    dl_nsigma = float(cfg["dl_nsigma"])
    prior_k = float(cfg["prior_k"])
    galaxy_chunk_size = int(cfg["galaxy_chunk_size"])
    missing_pixel_chunk_size = int(cfg["missing_pixel_chunk_size"])
    partial_write_min_sec = float(cfg.get("partial_write_min_sec", 10.0))

    f_miss_mode = str(cfg["f_miss_mode"])
    f_fixed = float(cfg.get("f_miss_fixed", 0.0))

    inj_host_mode = str(cfg.get("inj_host_mode", "catalog_only"))
    inj_f_mode = str(cfg.get("inj_f_miss_mode", "fixed"))
    inj_f_fixed = float(cfg.get("inj_f_miss_fixed", f_fixed))
    inj_beta_a = float(cfg.get("inj_f_miss_beta_a", cfg.get("f_miss_beta_a", 1.0)))
    inj_beta_b = float(cfg.get("inj_f_miss_beta_b", cfg.get("f_miss_beta_b", 1.0)))
    inj_logR_scale = float(cfg.get("inj_logR_scale", 1.0))

    if inj_host_mode not in ("catalog_only", "mixture"):
        raise ValueError("inj_host_mode must be 'catalog_only' or 'mixture'.")
    if inj_f_mode not in ("fixed", "beta", "match_scoring"):
        raise ValueError("inj_f_miss_mode must be 'fixed', 'beta', or 'match_scoring'.")

    rep_seed = int(seed_base + int(rep))
    t0 = float(time.time())
    last_partial_write = 0.0

    def _write_partial(
        *,
        stage: str,
        events_done: int | None,
        events_done_float: float | None,
        event: str | None,
        event_idx: int | None = None,
        force: bool = False,
        **extra: Any,
    ) -> None:
        nonlocal last_partial_write
        now_mono = time.monotonic()
        if not bool(force) and float(partial_write_min_sec) > 0.0 and last_partial_write > 0.0:
            if (now_mono - last_partial_write) < float(partial_write_min_sec):
                return
        last_partial_write = now_mono
        payload: dict[str, Any] = {
            "rep": int(rep),
            "pid": int(os.getpid()),
            "truth_model": truth_model,
            "seed": int(rep_seed),
            "stage": str(stage),
            "events_total": int(len(templates)),
            "events_done": events_done,
            "events_done_float": events_done_float,
            "event": event,
            "event_idx": event_idx,
            "elapsed_sec": float(time.time() - t0),
            "t_start_unix": float(t0),
        }
        payload.update(extra)
        _write_json(partial_path, payload)

    _write_partial(
        stage="start",
        events_done=0,
        events_done_float=0.0,
        event=None,
        event_idx=None,
        force=True,
    )

    try:
        rng = np.random.default_rng(rep_seed)

        j_truth = int(rng.integers(0, n_draws))

        f_true = 0.0
        if inj_host_mode == "mixture":
            if inj_f_mode == "fixed":
                f_true = float(inj_f_fixed)
            elif inj_f_mode == "beta":
                f_true = float(rng.beta(float(inj_beta_a), float(inj_beta_b)))
            else:
                if f_miss_mode == "marginalize":
                    f_true = float(rng.beta(float(inj_beta_a), float(inj_beta_b)))
                else:
                    f_true = float(inj_f_fixed)
            if not (np.isfinite(f_true) and 0.0 <= f_true <= 1.0):
                raise ValueError("Invalid injected f_true.")

        missing_flags: list[bool] = []
        z_true_list: list[float] = []
        for t in templates:
            is_miss = bool(inj_host_mode == "mixture" and float(rng.random()) < float(f_true))
            missing_flags.append(is_miss)
            if is_miss:
                zt = _sample_z_from_missing_prior(
                    z_grid=np.asarray(missing_pre.z_grid, dtype=float),
                    base_z=np.asarray(missing_pre.base_z[j_truth], dtype=float),
                    rng=rng,
                )
            else:
                zt = _sample_z_from_hist(z_bins=t.z_bins, z_cdf=t.z_cdf, rng=rng)
            z_true_list.append(float(zt))

        z_true = np.asarray(z_true_list, dtype=float)
        dL_em_true = _dL_em_for_draw(post=post, draw=j_truth, z_eval=z_true, constants=constants)
        if truth_model == "mu":
            R_true = _R_mu_for_draw(post=post, draw=j_truth, z_eval=z_true, convention=convention)  # type: ignore[arg-type]
            if not (np.isfinite(inj_logR_scale) and inj_logR_scale >= 0.0):
                raise ValueError("inj_logR_scale must be finite and >= 0.")
            if inj_logR_scale != 1.0:
                R_true = np.exp(float(inj_logR_scale) * np.log(np.clip(R_true, 1e-12, np.inf)))
        else:
            R_true = np.ones_like(dL_em_true)
        dL_true = np.clip(np.asarray(dL_em_true * R_true, dtype=float), 1e-6, np.inf)

        logL_cat_mu_list: list[np.ndarray] = []
        logL_cat_gr_list: list[np.ndarray] = []
        logL_miss_mu_list: list[np.ndarray] = []
        logL_miss_gr_list: list[np.ndarray] = []
        event_force_milestones = {
            1,
            max(1, int(len(templates) // 2)),
            int(len(templates)),
        }

        for i, t in enumerate(templates):
            _write_partial(
                stage="event_start",
                events_done=int(i),
                events_done_float=float(i),
                event=str(t.event),
                event_idx=int(i + 1),
                force=False,
            )

            pe = _build_synth_pe_hist(
                template=t,
                dL_true_mpc=float(dL_true[i]),
                sigma_ln_like=float(t.sigma_ln_like),
                prior_k=prior_k,
                dl_nbins=dl_nbins,
                dl_nsigma=dl_nsigma,
            )

            def _cb(info: dict[str, Any]) -> None:
                try:
                    galaxies_done = int(info.get("galaxies_done", 0))
                    galaxies_total = int(info.get("galaxies_total", 0))
                    frac = float(galaxies_done) / float(galaxies_total) if galaxies_total > 0 else 0.0
                except Exception:
                    galaxies_done = None
                    galaxies_total = None
                    frac = 0.0

                _write_partial(
                    stage=str(info.get("stage", "catalog_chunk")),
                    events_done=int(i),
                    events_done_float=float(i) + float(frac),
                    event=str(t.event),
                    event_idx=int(i + 1),
                    force=False,
                    event_chunk_idx=info.get("chunk_idx", None),
                    event_n_chunks=info.get("n_chunks", None),
                    galaxies_done=galaxies_done,
                    galaxies_total=galaxies_total,
                    uniq_z=info.get("uniq_z", None),
                )

            # Synthetic PE posterior is sky-independent, so spectral_only == full.
            # Prefer precomputed spectral chunks/tables to avoid recomputing identical setup.
            if t.spectral_chunks:
                logL_cat_mu, logL_cat_gr = _compute_dark_siren_logL_draws_from_precomp_spectral(
                    template=t,
                    pe=pe,
                    post=post,
                    convention=convention,  # type: ignore[arg-type]
                    gw_distance_prior=gw_prior,
                )
            else:
                logL_cat_mu, logL_cat_gr = compute_dark_siren_logL_draws_from_pe_hist(
                    event=t.event,
                    pe=pe,
                    post=post,
                    z_gal=t.z_gal,
                    w_gal=t.w_gal,
                    ipix_gal=t.ipix_gal,
                    convention=convention,  # type: ignore[arg-type]
                    gw_distance_prior=gw_prior,
                    distance_mode="spectral_only",
                    gal_chunk_size=galaxy_chunk_size,
                    progress_cb=_cb,
                )

            # Synthetic PE posterior is sky-independent, so spectral_only avoids the pixel loop.
            logL_miss_mu, logL_miss_gr = compute_missing_host_logL_draws_from_histogram(
                prob_pix=np.asarray(pe.prob_pix, dtype=float),
                pdf_bins=np.asarray(pe.pdf_bins, dtype=float),
                dL_edges=np.asarray(pe.dL_edges, dtype=float),
                pre=missing_pre,
                gw_distance_prior=gw_prior,
                distance_mode="spectral_only",
                pixel_chunk_size=missing_pixel_chunk_size,
            )

            logL_cat_mu_list.append(np.asarray(logL_cat_mu, dtype=float))
            logL_cat_gr_list.append(np.asarray(logL_cat_gr, dtype=float))
            logL_miss_mu_list.append(np.asarray(logL_miss_mu, dtype=float))
            logL_miss_gr_list.append(np.asarray(logL_miss_gr, dtype=float))

            _write_partial(
                stage="event_done",
                events_done=int(i + 1),
                events_done_float=float(i + 1),
                event=str(t.event),
                event_idx=int(i + 1),
                force=bool((i + 1) in event_force_milestones),
            )

        n_ev = int(len(templates))

        if f_miss_mode == "fixed":
            if not (0.0 < f_fixed < 1.0):
                raise ValueError("f_miss_fixed must be in (0,1).")
            loga = math.log1p(-f_fixed) if f_fixed < 1.0 else -math.inf
            logb = math.log(f_fixed) if f_fixed > 0.0 else -math.inf
            logL_mu = np.zeros((n_draws,), dtype=float)
            logL_gr = np.zeros((n_draws,), dtype=float)
            for i in range(n_ev):
                logL_mu += np.logaddexp(loga + logL_cat_mu_list[i], logb + logL_miss_mu_list[i])
                logL_gr += np.logaddexp(loga + logL_cat_gr_list[i], logb + logL_miss_gr_list[i])

            logL_mu_data = np.asarray(logL_mu, dtype=float)
            logL_gr_data = np.asarray(logL_gr, dtype=float)
            if log_alpha_mu is not None and log_alpha_gr is not None:
                logL_mu = logL_mu - float(n_ev) * np.asarray(log_alpha_mu, dtype=float)
                logL_gr = logL_gr - float(n_ev) * np.asarray(log_alpha_gr, dtype=float)

            lpd_mu = _logmeanexp(logL_mu)
            lpd_gr = _logmeanexp(logL_gr)
            lpd_mu_data = _logmeanexp(logL_mu_data)
            lpd_gr_data = _logmeanexp(logL_gr_data)
        elif f_miss_mode == "marginalize":
            f_grid = np.asarray(cfg["f_grid"], dtype=float)
            logw_f = np.asarray(cfg["logw_f"], dtype=float)
            logf = np.asarray(cfg["logf"], dtype=float)
            log1mf = np.asarray(cfg["log1mf"], dtype=float)
            log_prior_f = np.asarray(cfg["log_prior_f"], dtype=float)

            logL_mu_fd = np.zeros((f_grid.size, n_draws), dtype=float)
            logL_gr_fd = np.zeros((f_grid.size, n_draws), dtype=float)
            for i in range(n_ev):
                ev_mu = np.logaddexp(log1mf[:, None] + logL_cat_mu_list[i][None, :], logf[:, None] + logL_miss_mu_list[i][None, :])
                ev_gr = np.logaddexp(log1mf[:, None] + logL_cat_gr_list[i][None, :], logf[:, None] + logL_miss_gr_list[i][None, :])
                logL_mu_fd += ev_mu
                logL_gr_fd += ev_gr

            have_alpha = log_alpha_mu is not None and log_alpha_gr is not None
            if have_alpha:
                assert log_alpha_mu is not None and log_alpha_gr is not None
                logL_mu_fd -= float(n_ev) * np.asarray(log_alpha_mu, dtype=float).reshape((1, -1))
                logL_gr_fd -= float(n_ev) * np.asarray(log_alpha_gr, dtype=float).reshape((1, -1))

            lpd_mu_f = _logmeanexp_axis(logL_mu_fd, axis=1)
            lpd_gr_f = _logmeanexp_axis(logL_gr_fd, axis=1)
            lpd_mu = _logsumexp_1d(log_prior_f + lpd_mu_f + logw_f)
            lpd_gr = _logsumexp_1d(log_prior_f + lpd_gr_f + logw_f)

            if have_alpha:
                assert log_alpha_mu is not None and log_alpha_gr is not None
                logL_mu_fd_data = logL_mu_fd + float(n_ev) * np.asarray(log_alpha_mu, dtype=float).reshape((1, -1))
                logL_gr_fd_data = logL_gr_fd + float(n_ev) * np.asarray(log_alpha_gr, dtype=float).reshape((1, -1))
                lpd_mu_f_data = _logmeanexp_axis(logL_mu_fd_data, axis=1)
                lpd_gr_f_data = _logmeanexp_axis(logL_gr_fd_data, axis=1)
                lpd_mu_data = _logsumexp_1d(log_prior_f + lpd_mu_f_data + logw_f)
                lpd_gr_data = _logsumexp_1d(log_prior_f + lpd_gr_f_data + logw_f)
            else:
                lpd_mu_data = float(lpd_mu)
                lpd_gr_data = float(lpd_gr)
        else:
            raise ValueError("f_miss_mode must be 'fixed' or 'marginalize'.")

        d_tot = float(lpd_mu - lpd_gr)
        d_dat = float(lpd_mu_data - lpd_gr_data)
        d_sel = float(d_tot - d_dat)

        row = {
            "rep": int(rep),
            "truth_model": truth_model,
            "truth_draw": int(j_truth),
            "inj_host_mode": str(inj_host_mode),
            "inj_f_true": float(f_true),
            "inj_n_missing": int(sum(1 for m in missing_flags if m)),
            "inj_logR_scale": float(inj_logR_scale),
            "lpd_mu_total": float(lpd_mu),
            "lpd_gr_total": float(lpd_gr),
            "lpd_mu_total_data": float(lpd_mu_data),
            "lpd_gr_total_data": float(lpd_gr_data),
            "lpd_mu_total_sel": float(lpd_mu - lpd_mu_data),
            "lpd_gr_total_sel": float(lpd_gr - lpd_gr_data),
            "delta_lpd_total": d_tot,
            "delta_lpd_total_data": d_dat,
            "delta_lpd_total_sel": d_sel,
        }
        _write_json(rep_path, row)
        try:
            partial_path.unlink()
        except Exception:
            pass
        return row
    except Exception:
        tb = traceback.format_exc()
        err = {
            "rep": int(rep),
            "truth_model": truth_model,
            "error": "rep_failed",
            "traceback": tb,
            "delta_lpd_total": float("nan"),
            "delta_lpd_total_data": float("nan"),
            "delta_lpd_total_sel": float("nan"),
        }
        _write_json(rep_path, err)
        _write_json(
            partial_path,
            {
                "rep": int(rep),
                "pid": int(os.getpid()),
                "truth_model": truth_model,
                "seed": int(rep_seed),
                "stage": "error",
                "events_total": int(len(templates)),
                "events_done": None,
                "events_done_float": None,
                "event": None,
                "elapsed_sec": float(time.time() - t0),
                "t_start_unix": float(t0),
            },
        )
        return err


def _estimate_sigma_ln_like_from_template_hist(
    *,
    prob_pix: np.ndarray,
    pdf_bins: np.ndarray,
    dL_edges: np.ndarray,
    prior_k: float,
) -> float:
    prob_pix = np.asarray(prob_pix, dtype=float)
    pdf_bins = np.asarray(pdf_bins, dtype=float)
    dL_edges = np.asarray(dL_edges, dtype=float)
    widths = np.diff(dL_edges)
    dL_mid = 0.5 * (dL_edges[:-1] + dL_edges[1:])

    p_sum = float(np.sum(prob_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError("Invalid prob_pix sum while estimating sigma_ln_like.")

    # Sky-marginal posterior density p(dL|data) on the same bins.
    pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)

    # Proxy likelihood shape: L(dL) ∝ posterior / π(dL).
    prior = np.clip(dL_mid, 1e-12, np.inf) ** float(prior_k)
    like = pdf_1d / prior
    like = np.clip(like, 0.0, np.inf)
    norm = float(np.sum(like * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Invalid normalization while estimating sigma_ln_like.")
    like = like / norm

    ln_dL = np.log(np.clip(dL_mid, 1e-12, np.inf))
    mean = float(np.sum(like * widths * ln_dL))
    var = float(np.sum(like * widths * (ln_dL - mean) ** 2))
    if not (np.isfinite(var) and var > 0.0):
        raise ValueError("Non-finite/invalid variance while estimating sigma_ln_like.")
    return float(math.sqrt(var))


def _build_weighted_z_hist_sampler(
    z: np.ndarray,
    w: np.ndarray,
    *,
    z_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape:
        raise ValueError("z/w must be 1D with matching shapes.")
    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")
    n_bins = int(n_bins)
    if n_bins < 10:
        raise ValueError("n_bins too small.")

    z_bins = np.linspace(0.0, z_max, n_bins + 1)
    hist, _ = np.histogram(z, bins=z_bins, weights=w)
    hist = np.clip(np.asarray(hist, dtype=float), 0.0, np.inf)
    s = float(np.sum(hist))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Invalid weighted z histogram (sum<=0).")
    cdf = np.cumsum(hist) / s
    cdf[-1] = 1.0
    return z_bins, cdf


def _sample_z_from_hist(*, z_bins: np.ndarray, z_cdf: np.ndarray, rng: np.random.Generator) -> float:
    u = float(rng.random())
    i = int(np.searchsorted(z_cdf, u, side="left"))
    i = max(0, min(i, int(z_bins.size - 2)))
    z0 = float(z_bins[i])
    z1 = float(z_bins[i + 1])
    return float(z0 + (z1 - z0) * float(rng.random()))


def _sample_z_from_linear_segment(
    *,
    z0: float,
    z1: float,
    w0: float,
    w1: float,
    rng: np.random.Generator,
) -> float:
    """Sample z in [z0,z1] with density proportional to a linear weight w(z)."""
    z0 = float(z0)
    z1 = float(z1)
    if not (np.isfinite(z0) and np.isfinite(z1) and z1 > z0):
        raise ValueError("Invalid z segment bounds.")
    w0 = float(w0)
    w1 = float(w1)
    if not (np.isfinite(w0) and np.isfinite(w1)):
        raise ValueError("Invalid segment weights.")

    w0 = max(w0, 0.0)
    w1 = max(w1, 0.0)
    if w0 == 0.0 and w1 == 0.0:
        return float(z0 + (z1 - z0) * float(rng.random()))

    u = float(rng.random())

    # Let t in [0,1] and z=z0+t*(z1-z0). With w(t)=w0+(w1-w0)*t,
    #   F(t) = (w0 t + 0.5 (w1-w0) t^2) / ((w0+w1)/2).
    a = 0.5 * (w1 - w0)
    b = w0
    denom = 0.5 * (w0 + w1)
    if denom <= 0.0 or not np.isfinite(denom):
        return float(z0 + (z1 - z0) * float(u))

    if abs(a) < 1e-15:
        t = u
    else:
        c = -u * denom
        disc = b * b - 4.0 * a * c
        disc = max(disc, 0.0)
        t = (-b + math.sqrt(disc)) / (2.0 * a)
        if not np.isfinite(t):
            t = u
        t = min(1.0, max(0.0, float(t)))
    return float(z0 + (z1 - z0) * float(t))


def _sample_z_from_missing_prior(
    *,
    z_grid: np.ndarray,
    base_z: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Sample z from a missing-host prior defined on z_grid with weights base_z(z)."""
    z_grid = np.asarray(z_grid, dtype=float)
    base_z = np.asarray(base_z, dtype=float)
    if z_grid.ndim != 1 or base_z.ndim != 1 or z_grid.shape != base_z.shape:
        raise ValueError("z_grid/base_z must be 1D arrays with matching shapes.")
    if z_grid.size < 2 or float(z_grid[0]) < 0.0 or np.any(np.diff(z_grid) <= 0.0):
        raise ValueError("Invalid z_grid for missing-prior sampler.")

    w = np.clip(np.asarray(base_z, dtype=float), 0.0, np.inf)
    dz = np.diff(z_grid)
    seg = 0.5 * (w[:-1] + w[1:]) * dz
    seg = np.clip(seg, 0.0, np.inf)
    tot = float(np.sum(seg))
    if not (np.isfinite(tot) and tot > 0.0):
        raise ValueError("Invalid missing-host prior (zero total weight).")

    u = float(rng.random()) * tot
    csum = np.cumsum(seg)
    i = int(np.searchsorted(csum, u, side="left"))
    i = max(0, min(i, int(z_grid.size - 2)))
    return _sample_z_from_linear_segment(z0=float(z_grid[i]), z1=float(z_grid[i + 1]), w0=float(w[i]), w1=float(w[i + 1]), rng=rng)


def _dL_em_for_draw(
    *,
    post: MuForwardPosterior,
    draw: int,
    z_eval: np.ndarray,
    constants: PhysicalConstants,
) -> np.ndarray:
    z_eval = np.asarray(z_eval, dtype=float)
    bg = build_background_from_H_grid(post.z_grid, post.H_samples[int(draw)], constants=constants)
    Dc = bg.Dc(z_eval)
    ok = float(post.omega_k0[int(draw)])
    H0 = float(post.H0[int(draw)])
    if ok == 0.0:
        Dm = Dc
    elif ok > 0.0:
        sk = math.sqrt(ok) * (H0 * Dc / constants.c_km_s)
        Dm = (constants.c_km_s / (H0 * math.sqrt(ok))) * np.sinh(sk)
    else:
        sk = math.sqrt(abs(ok)) * (H0 * Dc / constants.c_km_s)
        Dm = (constants.c_km_s / (H0 * math.sqrt(abs(ok)))) * np.sin(sk)
    return (1.0 + z_eval) * Dm


def _R_mu_for_draw(
    *,
    post: MuForwardPosterior,
    draw: int,
    z_eval: np.ndarray,
    convention: Literal["A", "B"],
) -> np.ndarray:
    z_eval = np.asarray(z_eval, dtype=float)
    draw = int(draw)

    xz_grid = x_of_z_from_H(
        post.z_grid,
        post.H_samples[draw],
        H0=float(post.H0[draw]),
        omega_k0=float(post.omega_k0[draw]),
    )
    x_eval = np.interp(z_eval, post.z_grid, xz_grid)
    logmu = np.interp(x_eval, post.x_grid, post.logmu_x_samples[draw])
    mu = np.exp(np.asarray(logmu, dtype=float))
    mu0 = float(np.exp(float(np.interp(0.0, post.x_grid, post.logmu_x_samples[draw]))))
    if not (np.isfinite(mu0) and mu0 > 0.0):
        raise ValueError("Non-finite/invalid mu0 while building truth R(z).")
    if convention == "A":
        return np.sqrt(mu / mu0)
    return np.sqrt(mu0 / mu)


def _build_synth_pe_hist(
    *,
    template: TemplateEvent,
    dL_true_mpc: float,
    sigma_ln_like: float,
    prior_k: float,
    dl_nbins: int,
    dl_nsigma: float,
) -> PePixelDistanceHistogram:
    dL0 = float(dL_true_mpc)
    if not (np.isfinite(dL0) and dL0 > 0.0):
        raise ValueError("dL_true_mpc must be finite and positive.")
    sig = float(sigma_ln_like)
    if not (np.isfinite(sig) and sig > 0.0):
        raise ValueError("sigma_ln_like must be finite and positive.")

    lo = dL0 * math.exp(-float(dl_nsigma) * sig)
    hi = dL0 * math.exp(+float(dl_nsigma) * sig)
    lo = max(lo, 1e-3)
    if not (np.isfinite(hi) and hi > lo):
        raise ValueError("Invalid dL grid bounds for synthetic PE histogram.")

    edges = np.exp(np.linspace(math.log(lo), math.log(hi), int(dl_nbins) + 1))
    widths = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])

    # Lognormal likelihood with median dL_true.
    ln_mid = np.log(np.clip(mid, 1e-12, np.inf))
    mu = math.log(dL0)
    logL = -0.5 * ((ln_mid - mu) / sig) ** 2 - ln_mid - math.log(sig) - 0.5 * math.log(2.0 * math.pi)
    like = np.exp(np.asarray(logL, dtype=float))

    # Posterior density ∝ L(dL) * π(dL) with π(dL) ∝ dL^k.
    prior = np.clip(mid, 1e-12, np.inf) ** float(prior_k)
    pdf = like * prior
    pdf = np.clip(pdf, 0.0, np.inf)
    norm = float(np.sum(pdf * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Invalid normalization for synthetic PE distance posterior.")
    pdf = pdf / norm
    pdf = np.clip(pdf, 1e-300, np.inf)

    # Synthetic PE posterior is sky-independent in this suite, so the per-pixel
    # conditional distance PDF is identical for all selected pixels.
    # Store a single shared row (shape 1 x n_bins) and let downstream
    # spectral-only code handle this compact representation.
    pdf_bins = np.asarray(pdf.reshape((1, -1)), dtype=np.float32)

    return PePixelDistanceHistogram(
        nside=int(template.pe_nside),
        nest=True,
        p_credible=float(template.p_credible),
        pix_sel=np.asarray(template.pix_sel, dtype=np.int64),
        prob_pix=np.asarray(template.prob_pix, dtype=np.float32),
        dL_edges=np.asarray(edges, dtype=np.float64),
        pdf_bins=np.asarray(pdf_bins, dtype=np.float32),
    )


def _compute_dark_siren_logL_draws_from_precomp_spectral(
    *,
    template: TemplateEvent,
    pe: PePixelDistanceHistogram,
    post: MuForwardPosterior,
    convention: Literal["A", "B"],
    gw_distance_prior: GWDistancePrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-only catalog logL using precomputed per-event chunk structures.

    This is algebraically equivalent to the grouped-z spectral-only branch in
    `compute_dark_siren_logL_draws_from_pe_hist`, but avoids per-replicate
    row-mapping/sorting/grouping work.
    """
    if not template.spectral_chunks:
        raise ValueError(f"{template.event}: missing spectral precompute chunks.")

    edges = np.asarray(pe.dL_edges, dtype=float)
    widths = np.diff(edges)
    nb = int(edges.size - 1)
    if nb <= 0:
        raise ValueError(f"{template.event}: invalid PE histogram edges.")

    p_pix = np.asarray(pe.prob_pix, dtype=float)
    pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
    p_sum = float(np.sum(p_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError(f"{template.event}: invalid PE prob_pix sum.")
    if int(pdf_bins.shape[0]) == int(p_pix.size):
        pdf_1d = np.sum(p_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    elif int(pdf_bins.shape[0]) == 1:
        # Compact sky-independent synthetic PE representation.
        pdf_1d = np.asarray(pdf_bins[0], dtype=float)
    else:
        raise ValueError(
            f"{template.event}: incompatible pdf_bins shape {pdf_bins.shape} for prob_pix size {p_pix.size}."
        )
    pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
    norm = float(np.sum(pdf_1d * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError(f"{template.event}: invalid sky-marginal PE distance normalization.")
    pdf_1d = pdf_1d / norm

    n_draws = int(post.H_samples.shape[0])
    logL_mu = np.full((n_draws,), -np.inf, dtype=float)
    logL_gr = np.full((n_draws,), -np.inf, dtype=float)

    for ch in template.spectral_chunks:
        logweight_u = np.asarray(ch.logweight_u, dtype=float).reshape((1, -1))
        if ch.dL_em_u is not None and ch.dL_gw_u is not None:
            dL_em_u = np.asarray(ch.dL_em_u, dtype=float)
            dL_gw_u = np.asarray(ch.dL_gw_u, dtype=float)
        else:
            dL_em_u = predict_dL_em(post, z_eval=np.asarray(ch.z_u, dtype=float))
            _, R_u = predict_r_gw_em(post, z_eval=np.asarray(ch.z_u, dtype=float), convention=convention, allow_extrapolation=False)
            dL_gw_u = dL_em_u * np.asarray(R_u, dtype=float)

        def _chunk_logL_grouped(dL_u: np.ndarray) -> np.ndarray:
            dL_u = np.asarray(dL_u, dtype=float)
            bin_idx = np.searchsorted(edges, dL_u, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL_u) & (dL_u > 0.0)
            pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
            pdf = np.where(valid, pdf, 0.0)
            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = gw_distance_prior.log_pi_dL(np.clip(dL_u, 1e-6, np.inf))
            logterm = logweight_u + logpdf - logprior
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        logL_mu = np.logaddexp(logL_mu, _chunk_logL_grouped(dL_gw_u))
        logL_gr = np.logaddexp(logL_gr, _chunk_logL_grouped(dL_em_u))

    return np.asarray(logL_mu, dtype=float), np.asarray(logL_gr, dtype=float)


def _summarize(vals: list[float]) -> dict[str, float]:
    xs = [float(v) for v in vals if np.isfinite(float(v))]
    if not xs:
        return {"n": 0.0}
    xs.sort()
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(var)
    q = lambda p: xs[max(0, min(n - 1, int(round(p * (n - 1)))))]
    return {
        "n": float(n),
        "mean": float(mean),
        "sd": float(sd),
        "min": float(xs[0]),
        "p16": float(q(0.16)),
        "p50": float(q(0.50)),
        "p84": float(q(0.84)),
        "max": float(xs[-1]),
        "p_ge_0": float(sum(1 for x in xs if x >= 0.0) / n),
        "p_ge_3": float(sum(1 for x in xs if x >= 3.0) / n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Synthetic GR-truth (or mu-truth) injection suite for the catalog-based dark-siren score, using real-data cached galaxy+sky selections as templates."
    )
    ap.add_argument(
        "--base-event-cache",
        type=str,
        default="outputs/dark_siren_verification_suite_20260205_223923UTC/catalog_peAnalytic_auto/cache",
        help="Directory containing event_*.npz caches from a real-data run (default: catalog_peAnalytic_auto cache).",
    )
    ap.add_argument("--run-dir", type=str, default="outputs/finalization/highpower_multistart_v2/M0_start101")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--truth-model", choices=["gr", "mu"], default="gr", help="Truth used for generating dL_true (default gr).")
    ap.add_argument(
        "--inj-host-mode",
        choices=["catalog_only", "mixture"],
        default="catalog_only",
        help="How to sample injected true redshifts: from the in-catalog host histogram only, or from a catalog/missing-host mixture (default catalog_only).",
    )
    ap.add_argument(
        "--inj-f-miss-mode",
        choices=["fixed", "beta", "match_scoring"],
        default="match_scoring",
        help="How to choose the injected missing-host fraction f_true when --inj-host-mode=mixture (default match_scoring).",
    )
    ap.add_argument("--inj-f-miss-fixed", type=float, default=0.6807953774124085, help="Injected f_true for --inj-f-miss-mode=fixed.")
    ap.add_argument("--inj-f-miss-beta-mean", type=float, default=0.6807953774124085, help="Beta prior mean for injected f_true.")
    ap.add_argument("--inj-f-miss-beta-kappa", type=float, default=8.0, help="Beta prior kappa for injected f_true.")
    ap.add_argument(
        "--inj-logR-scale",
        type=float,
        default=1.0,
        help="Scale factor λ applied to log R_true(z) when --truth-model=mu (R_true -> exp(λ log R_true)); λ=1 is nominal, λ=0 forces GR truth.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--n-proc", type=int, default=0, help="Processes for parallel replicates (default: use all cores).")
    ap.add_argument(
        "--parallel-backend",
        choices=["auto", "process", "thread", "sequential"],
        default="auto",
        help=(
            "Replicate parallel backend.\n"
            "  - auto: try process pool (fork), fallback to thread pool on environment semaphore errors.\n"
            "  - process: require process pool.\n"
            "  - thread: force thread pool.\n"
            "  - sequential: disable parallel pool."
        ),
    )
    ap.add_argument("--heartbeat-sec", type=float, default=60.0, help="Top-level progress heartbeat interval in seconds (default 60).")
    ap.add_argument(
        "--partial-write-min-sec",
        type=float,
        default=10.0,
        help="Minimum seconds between per-rep partial JSON updates inside heavy loops (default 10).",
    )
    ap.add_argument("--max-events", type=int, default=36)
    ap.add_argument("--max-draws", type=int, default=128, help="Posterior draws used for scoring (default 128).")
    ap.add_argument("--convention", choices=["A", "B"], default="A")

    ap.add_argument("--prior-k", type=float, default=2.0, help="Assumed PE distance prior power k in π(dL)∝dL^k (default 2).")
    ap.add_argument("--sigma-ln-floor", type=float, default=0.08)
    ap.add_argument("--sigma-ln-ceil", type=float, default=1.0)
    ap.add_argument("--z-hist-bins", type=int, default=400)

    ap.add_argument("--dl-nbins", type=int, default=64)
    ap.add_argument("--dl-nsigma", type=float, default=6.0)

    ap.add_argument("--galaxy-chunk-size", type=int, default=50_000)
    ap.add_argument("--missing-pixel-chunk-size", type=int, default=5000)
    ap.add_argument(
        "--spectral-precompute-dl",
        choices=["auto", "on", "off"],
        default="auto",
        help="Precompute per-event spectral dL tables across all draws for reuse (default auto).",
    )
    ap.add_argument(
        "--spectral-precompute-dl-max-gb",
        type=float,
        default=24.0,
        help="When --spectral-precompute-dl=auto, enable only if estimated table size <= this many GB.",
    )

    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0)
    ap.add_argument("--selection-det-model", choices=["threshold", "snr_binned"], default="snr_binned")
    ap.add_argument("--selection-snr-thresh", type=float, default=None, help="Manual network SNR threshold (only used if --selection-det-model=threshold).")
    ap.add_argument("--selection-snr-binned-nbins", type=int, default=200)
    ap.add_argument("--selection-weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf")
    ap.add_argument("--selection-z-max", type=float, default=0.3)
    ap.add_argument(
        "--selection-injections-hdf",
        type=str,
        default="auto",
        help=(
            "Selection injection file path. Use 'auto' (default) to resolve from event segment, "
            "or provide an explicit HDF5 path for mismatch/sabotage tests."
        ),
    )
    ap.add_argument(
        "--selection-alpha-cache",
        type=str,
        default="",
        help="Optional shared alpha cache (.npz). If present, load alpha_mu/alpha_gr from it; otherwise compute and write.",
    )
    ap.add_argument(
        "--selection-injections-population",
        choices=["mixture", "bbhpop"],
        default="mixture",
        help="Population label used only when --selection-injections-hdf=auto.",
    )
    ap.add_argument(
        "--selection-injections-auto-download",
        action="store_true",
        help="Allow auto-download when --selection-injections-hdf=auto and file is missing locally.",
    )
    ap.add_argument(
        "--selection-snr-offset",
        type=float,
        default=0.0,
        help="Additive SNR offset used by selection alpha proxy (positive => harder detection).",
    )
    ap.add_argument(
        "--selection-mu-det-distance",
        choices=["gw", "em"],
        default="gw",
        help="Distance channel used for mu-model detectability in selection alpha proxy.",
    )
    ap.add_argument(
        "--selection-pop-z-mode",
        choices=["none", "comoving_uniform", "comoving_powerlaw"],
        default="none",
        help=(
            "Optional population redshift prior factor for alpha(model), applied in addition to selection-weight-mode.\n"
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
            "Optional population mass prior factor for alpha(model), applied in addition to selection-weight-mode.\n"
            "  - powerlaw_q: p(m1,m2) ∝ m1^{-alpha} q^{beta_q} on [m_min,m_max]\n"
            "  - powerlaw_q_smooth: same with smooth tapers at mass bounds\n"
            "  - powerlaw_peak_q_smooth: smooth power law plus Gaussian peak in m1\n"
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
        help="Smooth taper width (Msun) for selection-pop-mass-mode=powerlaw_q_smooth/powerlaw_peak_q_smooth.",
    )
    ap.add_argument("--selection-pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 for selection-pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--selection-pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 for selection-pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--selection-pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for selection-pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--no-selection", action="store_true")

    ap.add_argument("--missing-z-max", type=float, default=0.3)
    ap.add_argument("--host-prior-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform")
    ap.add_argument("--host-prior-z-k", type=float, default=0.0)

    ap.add_argument("--f-miss-mode", choices=["fixed", "marginalize"], default="marginalize")
    ap.add_argument("--f-miss-fixed", type=float, default=0.6807953774124085)
    ap.add_argument("--f-miss-beta-mean", type=float, default=0.6807953774124085)
    ap.add_argument("--f-miss-beta-kappa", type=float, default=8.0)
    ap.add_argument("--f-miss-grid-n", type=int, default=401)
    ap.add_argument("--f-miss-grid-eps", type=float, default=1e-6)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = out_dir / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _write_json(
        out_dir / "manifest.json",
        {
            "timestamp_utc": time.strftime("%Y%m%d_%H%M%SUTC", time.gmtime()),
            "base_event_cache": str(Path(args.base_event_cache).expanduser().resolve()),
            "run_dir": str(Path(args.run_dir).expanduser().resolve()),
            "truth_model": str(args.truth_model),
            "injection": {
                "host_mode": str(args.inj_host_mode),
                "f_miss_mode": str(args.inj_f_miss_mode),
                "f_miss_fixed": float(args.inj_f_miss_fixed),
                "f_miss_beta_mean": float(args.inj_f_miss_beta_mean),
                "f_miss_beta_kappa": float(args.inj_f_miss_beta_kappa),
                "logR_scale": float(args.inj_logR_scale),
            },
            "seed": int(args.seed),
            "n_rep": int(args.n_rep),
            "n_proc": int(args.n_proc),
            "parallel_backend": str(args.parallel_backend),
            "max_events": int(args.max_events),
            "max_draws": int(args.max_draws),
            "convention": str(args.convention),
            "prior_k": float(args.prior_k),
            "sigma_ln_floor": float(args.sigma_ln_floor),
            "sigma_ln_ceil": float(args.sigma_ln_ceil),
            "z_hist_bins": int(args.z_hist_bins),
            "dl_nbins": int(args.dl_nbins),
            "dl_nsigma": float(args.dl_nsigma),
            "galaxy_chunk_size": int(args.galaxy_chunk_size),
            "missing_pixel_chunk_size": int(args.missing_pixel_chunk_size),
            "spectral_precompute_dl": str(args.spectral_precompute_dl),
            "spectral_precompute_dl_max_gb": float(args.spectral_precompute_dl_max_gb),
            "selection": {
                "enabled": not bool(args.no_selection),
                "ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
                "det_model": str(args.selection_det_model),
                "snr_thresh": (None if args.selection_snr_thresh is None else float(args.selection_snr_thresh)),
                "snr_binned_nbins": int(args.selection_snr_binned_nbins),
                "weight_mode": str(args.selection_weight_mode),
                "z_max": float(args.selection_z_max),
                "injections_hdf": str(args.selection_injections_hdf),
                "injections_population": str(args.selection_injections_population),
                "injections_auto_download": bool(args.selection_injections_auto_download),
                "snr_offset": float(args.selection_snr_offset),
                "mu_det_distance": str(args.selection_mu_det_distance),
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
                "alpha_cache": str(args.selection_alpha_cache),
            },
            "missing": {
                "z_max": float(args.missing_z_max),
                "host_prior_z_mode": str(args.host_prior_z_mode),
                "host_prior_z_k": float(args.host_prior_z_k),
            },
            "f_miss": {
                "mode": str(args.f_miss_mode),
                "fixed": float(args.f_miss_fixed),
                "beta_mean": float(args.f_miss_beta_mean),
                "beta_kappa": float(args.f_miss_beta_kappa),
                "grid_n": int(args.f_miss_grid_n),
                "grid_eps": float(args.f_miss_grid_eps),
            },
        },
    )

    base_cache = Path(args.base_event_cache).expanduser().resolve()
    files = sorted(base_cache.glob("event_*.npz"))
    if not files:
        raise FileNotFoundError(f"No event_*.npz files found under {base_cache}")
    if int(args.max_events) < len(files):
        files = files[: int(args.max_events)]

    templates: list[TemplateEvent] = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            meta = json.loads(str(d["meta"].tolist()))
            event = str(meta["event"])
            z_gal = np.asarray(d["z"], dtype=float)
            w_gal = np.asarray(d["w"], dtype=float)
            ipix_gal = np.asarray(d["ipix"], dtype=np.int64)
            pix_sel = np.asarray(d["pe_pix_sel"], dtype=np.int64)
            prob_pix = np.asarray(d["pe_prob_pix"], dtype=float)
            dL_edges = np.asarray(d["pe_dL_edges"], dtype=float)
            pdf_bins = np.asarray(d["pe_pdf_bins"], dtype=float)
            pe_nside = int(meta["pe_nside"])
            p_credible = float(meta["p_credible"])

        sigma = _estimate_sigma_ln_like_from_template_hist(prob_pix=prob_pix, pdf_bins=pdf_bins, dL_edges=dL_edges, prior_k=float(args.prior_k))
        sigma = float(np.clip(sigma, float(args.sigma_ln_floor), float(args.sigma_ln_ceil)))

        # Host-redshift sampling weights: include sky probability weights per galaxy to match the
        # actual in-catalog term's (w_gal * prob_pix) structure.
        npix = 12 * int(pe_nside) * int(pe_nside)
        pix_to_row = np.full((npix,), -1, dtype=np.int32)
        pix_to_row[np.asarray(pix_sel, dtype=np.int64)] = np.arange(int(pix_sel.size), dtype=np.int32)
        row = pix_to_row[np.asarray(ipix_gal, dtype=np.int64)]
        good = row >= 0
        if not np.all(good):
            z_gal = np.asarray(z_gal[good], dtype=float)
            w_gal = np.asarray(w_gal[good], dtype=float)
            ipix_gal = np.asarray(ipix_gal[good], dtype=np.int64)
            row = row[good]
        if z_gal.size > 1 and not np.all(z_gal[1:] >= z_gal[:-1]):
            order = np.argsort(z_gal, kind="mergesort")
            z_gal = np.asarray(z_gal[order], dtype=float)
            w_gal = np.asarray(w_gal[order], dtype=float)
            ipix_gal = np.asarray(ipix_gal[order], dtype=np.int64)
            row = row[order]
        prob_gal = np.asarray(prob_pix, dtype=float)[np.asarray(row, dtype=np.int64)]
        w_host = np.asarray(w_gal, dtype=float) * np.asarray(prob_gal, dtype=float)
        z_bins, z_cdf = _build_weighted_z_hist_sampler(z_gal, w_host, z_max=float(args.selection_z_max), n_bins=int(args.z_hist_bins))

        # Reusable spectral-only chunk structures: unique-z grouped log-weights per chunk.
        # This avoids redoing pixel mapping/sorting/grouping work every replicate.
        spectral_chunks: list[SpectralChunk] = []
        chunk = int(args.galaxy_chunk_size)
        if chunk <= 0:
            raise ValueError("--galaxy-chunk-size must be positive.")
        for a in range(0, int(z_gal.size), int(chunk)):
            b = min(int(z_gal.size), int(a + chunk))
            z_c = np.asarray(z_gal[a:b], dtype=float)
            w_c = np.asarray(w_host[a:b], dtype=float)
            if z_c.size == 0:
                continue
            is_new = np.empty(z_c.shape, dtype=bool)
            is_new[0] = True
            is_new[1:] = z_c[1:] != z_c[:-1]
            starts = np.flatnonzero(is_new)
            z_u = z_c[starts]
            w_u = np.add.reduceat(w_c, starts)
            logw_u = np.log(np.clip(np.asarray(w_u, dtype=float), 1e-300, np.inf))
            spectral_chunks.append(
                SpectralChunk(
                    z_u=np.asarray(z_u, dtype=np.float64),
                    logweight_u=np.asarray(logw_u, dtype=np.float64),
                )
            )

        templates.append(
            TemplateEvent(
                event=event,
                z_gal=z_gal,
                w_gal=w_gal,
                ipix_gal=ipix_gal,
                pe_nside=pe_nside,
                p_credible=p_credible,
                pix_sel=pix_sel,
                prob_pix=prob_pix,
                sigma_ln_like=sigma,
                z_bins=z_bins,
                z_cdf=z_cdf,
                spectral_chunks=tuple(spectral_chunks),
            )
        )

    events = [t.event for t in templates]
    print(f"[inj] loaded {len(templates)} template events", flush=True)
    n_chunks_total = int(sum(len(t.spectral_chunks) for t in templates))
    n_unique_total = int(sum(sum(int(ch.z_u.size) for ch in t.spectral_chunks) for t in templates))
    print(
        f"[inj] spectral chunks prepared: chunks={n_chunks_total} total_unique_z={n_unique_total}",
        flush=True,
    )

    post_full = load_mu_forward_posterior(args.run_dir)
    post, draw_idx = _downsample_posterior(post_full, max_draws=int(args.max_draws), seed=int(args.seed) + 12345)
    n_draws = int(post.H_samples.shape[0])
    print(f"[inj] scoring draws: {n_draws} (downsample idx range {int(draw_idx.min())}-{int(draw_idx.max())})", flush=True)

    precompute_mode = str(args.spectral_precompute_dl)
    max_gb = float(args.spectral_precompute_dl_max_gb)
    est_bytes = float(n_unique_total) * float(n_draws) * 2.0 * 8.0
    est_gb = est_bytes / 1.0e9
    do_precompute_dl = False
    if precompute_mode == "on":
        do_precompute_dl = True
    elif precompute_mode == "off":
        do_precompute_dl = False
    else:
        do_precompute_dl = bool(est_gb <= max_gb)

    if do_precompute_dl:
        print(
            f"[inj] precomputing spectral dL tables: est={est_gb:.2f} GB mode={precompute_mode}",
            flush=True,
        )
        # Build tables in larger per-event vectorized batches to reduce repeated
        # predict_* setup overhead across many small chunks. This preserves math.
        target_batch_bytes = int(1_000_000_000)  # ~1 GB target working set
        # Working arrays per z element are roughly: dL_em, R, dL_gw (float64).
        batch_n = max(1, int(target_batch_bytes // max(1, 3 * 8 * n_draws)))
        for ti, t in enumerate(templates, start=1):
            chunk_sizes = [int(ch.z_u.size) for ch in t.spectral_chunks]
            n_u_event = int(sum(chunk_sizes))
            if n_u_event <= 0:
                continue

            z_all = np.empty((n_u_event,), dtype=np.float64)
            pos = 0
            for ch in t.spectral_chunks:
                n_u = int(ch.z_u.size)
                z_all[pos : pos + n_u] = np.asarray(ch.z_u, dtype=np.float64)
                pos += n_u

            # Ensure strictly increasing z for predict_r_gw_em, while preserving
            # per-chunk ordering by mapping back through inverse indices.
            z_unique, inv = np.unique(z_all, return_inverse=True)
            n_u_unique = int(z_unique.size)
            dL_em_unique = np.empty((n_draws, n_u_unique), dtype=np.float64)
            dL_gw_unique = np.empty((n_draws, n_u_unique), dtype=np.float64)
            for a in range(0, n_u_unique, batch_n):
                b = min(n_u_unique, a + batch_n)
                z_batch = z_unique[a:b]
                dL_em_b = predict_dL_em(post, z_eval=z_batch)
                _, R_b = predict_r_gw_em(
                    post,
                    z_eval=z_batch,
                    convention=str(args.convention),  # type: ignore[arg-type]
                    allow_extrapolation=False,
                )
                dL_em_b = np.asarray(dL_em_b, dtype=np.float64)
                dL_em_unique[:, a:b] = dL_em_b
                dL_gw_unique[:, a:b] = dL_em_b * np.asarray(R_b, dtype=np.float64)

            pos = 0
            for ch in t.spectral_chunks:
                n_u = int(ch.z_u.size)
                idx = np.asarray(inv[pos : pos + n_u], dtype=np.int64)
                ch.dL_em_u = dL_em_unique[:, idx]
                ch.dL_gw_u = dL_gw_unique[:, idx]
                pos += n_u
            if ti == 1 or ti == len(templates) or (ti % 8) == 0:
                print(f"[inj] precompute progress: event {ti}/{len(templates)}", flush=True)
    else:
        print(
            f"[inj] spectral dL precompute disabled: est={est_gb:.2f} GB mode={precompute_mode} max_gb={max_gb:.2f}",
            flush=True,
        )

    constants = PhysicalConstants()
    gw_prior = GWDistancePrior(mode="dL_powerlaw", powerlaw_k=float(args.prior_k))

    log_alpha_mu = None
    log_alpha_gr = None
    if not bool(args.no_selection):
        cache_spec = str(args.selection_alpha_cache).strip()
        cache_path = Path(cache_spec).expanduser().resolve() if cache_spec else None
        loaded_from_cache = False
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=True) as d:
                    alpha_mu_cached = np.asarray(d["alpha_mu"], dtype=float)
                    alpha_gr_cached = np.asarray(d["alpha_gr"], dtype=float)
                if alpha_mu_cached.shape == (n_draws,) and alpha_gr_cached.shape == (n_draws,):
                    log_alpha_mu = np.log(np.clip(alpha_mu_cached, 1e-300, np.inf))
                    log_alpha_gr = np.log(np.clip(alpha_gr_cached, 1e-300, np.inf))
                    loaded_from_cache = True
                    print(f"[inj] selection alpha: loaded cache {cache_path}", flush=True)
                else:
                    print(
                        f"[inj] selection alpha cache shape mismatch {cache_path} "
                        f"(got {alpha_mu_cached.shape}/{alpha_gr_cached.shape}, expected {(n_draws,)}) ; recomputing",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[inj] selection alpha cache load failed ({cache_path}): {exc}; recomputing", flush=True)

        if not loaded_from_cache:
            inj_spec = str(args.selection_injections_hdf).strip()
            if inj_spec.lower() in ("auto", ""):
                inj_path = resolve_o3_sensitivity_injection_file(
                    events=events,
                    base_dir="data/cache/gw/zenodo",
                    record_id=7890437,
                    population=str(args.selection_injections_population),  # type: ignore[arg-type]
                    auto_download=bool(args.selection_injections_auto_download),
                )
            else:
                inj_path = Path(inj_spec).expanduser().resolve()
                if not inj_path.exists():
                    raise FileNotFoundError(
                        f"Selection injections file not found: {inj_path} "
                        "(pass --selection-injections-hdf auto to use segment-resolved defaults)"
                    )
            injections = load_o3_injections(inj_path, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
            alpha = compute_selection_alpha_from_injections(
                post=post,
                injections=injections,
                convention=str(args.convention),  # type: ignore[arg-type]
                z_max=float(args.selection_z_max),
                det_model=str(args.selection_det_model),  # type: ignore[arg-type]
                snr_threshold=(None if args.selection_snr_thresh is None else float(args.selection_snr_thresh)),
                snr_binned_nbins=int(args.selection_snr_binned_nbins),
                weight_mode=str(args.selection_weight_mode),  # type: ignore[arg-type]
                snr_offset=float(args.selection_snr_offset),
                mu_det_distance=str(args.selection_mu_det_distance),  # type: ignore[arg-type]
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
            np.savez_compressed(
                out_dir / "selection_alpha.npz",
                alpha_mu=np.asarray(alpha.alpha_mu, dtype=float),
                alpha_gr=np.asarray(alpha.alpha_gr, dtype=float),
                log_alpha_mu=np.asarray(log_alpha_mu, dtype=float),
                log_alpha_gr=np.asarray(log_alpha_gr, dtype=float),
                meta=alpha.to_json(),
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Use a real .npz temp name so numpy does not auto-append another suffix.
                tmp = Path(str(cache_path) + ".tmp.npz")
                with tmp.open("wb") as fh:
                    np.savez_compressed(
                        fh,
                        alpha_mu=np.asarray(alpha.alpha_mu, dtype=float),
                        alpha_gr=np.asarray(alpha.alpha_gr, dtype=float),
                        log_alpha_mu=np.asarray(log_alpha_mu, dtype=float),
                        log_alpha_gr=np.asarray(log_alpha_gr, dtype=float),
                        meta=alpha.to_json(),
                    )
                tmp.replace(cache_path)
                print(f"[inj] selection alpha: cached {cache_path}", flush=True)
            print(f"[inj] selection alpha: {inj_path.name} n_inj_used={alpha.n_injections_used}", flush=True)

    missing_pre = precompute_missing_host_prior(
        post,
        convention=str(args.convention),  # type: ignore[arg-type]
        z_max=float(args.missing_z_max),
        host_prior_z_mode=str(args.host_prior_z_mode),  # type: ignore[arg-type]
        host_prior_z_k=float(args.host_prior_z_k),
        constants=constants,
    )
    _write_json(out_dir / "missing_pre_meta.json", missing_pre.to_jsonable())

    do_f_miss_marg = str(args.f_miss_mode) == "marginalize"
    f_fixed = float(args.f_miss_fixed)
    if not (0.0 < f_fixed < 1.0):
        raise ValueError("--f-miss-fixed must be in (0,1).")
    f_mean = float(args.f_miss_beta_mean)
    if not (0.0 < f_mean < 1.0):
        raise ValueError("--f-miss-beta-mean must be in (0,1).")
    kappa = float(args.f_miss_beta_kappa)
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError("--f-miss-beta-kappa must be finite and positive.")
    a = max(float(f_mean * kappa), 1e-3)
    b = max(float((1.0 - f_mean) * kappa), 1e-3)

    inj_f_fixed = float(args.inj_f_miss_fixed)
    if not (0.0 < inj_f_fixed < 1.0):
        raise ValueError("--inj-f-miss-fixed must be in (0,1).")
    inj_f_mean = float(args.inj_f_miss_beta_mean)
    if not (0.0 < inj_f_mean < 1.0):
        raise ValueError("--inj-f-miss-beta-mean must be in (0,1).")
    inj_kappa = float(args.inj_f_miss_beta_kappa)
    if not (np.isfinite(inj_kappa) and inj_kappa > 0.0):
        raise ValueError("--inj-f-miss-beta-kappa must be finite and positive.")
    inj_a = max(float(inj_f_mean * inj_kappa), 1e-3)
    inj_b = max(float((1.0 - inj_f_mean) * inj_kappa), 1e-3)

    inj_logR_scale = float(args.inj_logR_scale)
    if not (np.isfinite(inj_logR_scale) and inj_logR_scale >= 0.0):
        raise ValueError("--inj-logR-scale must be finite and >= 0.")

    rng = np.random.default_rng(int(args.seed))
    # Precompute f_miss marginalization grid+prior once and share across workers.
    cfg: dict[str, Any] = {
        "reps_dir": str(reps_dir),
        "n_draws": int(n_draws),
        "seed_base": int(args.seed),
        "truth_model": str(args.truth_model),
        "convention": str(args.convention),
        "dl_nbins": int(args.dl_nbins),
        "dl_nsigma": float(args.dl_nsigma),
        "prior_k": float(args.prior_k),
        "galaxy_chunk_size": int(args.galaxy_chunk_size),
        "missing_pixel_chunk_size": int(args.missing_pixel_chunk_size),
        "partial_write_min_sec": float(args.partial_write_min_sec),
        "f_miss_mode": str(args.f_miss_mode),
        "f_miss_fixed": float(f_fixed),
        "f_miss_beta_a": float(a),
        "f_miss_beta_b": float(b),
        "inj_host_mode": str(args.inj_host_mode),
        "inj_f_miss_mode": str(args.inj_f_miss_mode),
        "inj_f_miss_fixed": float(inj_f_fixed),
        "inj_f_miss_beta_a": float(inj_a),
        "inj_f_miss_beta_b": float(inj_b),
        "inj_logR_scale": float(inj_logR_scale),
    }

    if do_f_miss_marg:
        n_f = int(args.f_miss_grid_n)
        eps = float(args.f_miss_grid_eps)
        if n_f < 21:
            raise ValueError("--f-miss-grid-n too small (use >=21).")
        if not (0.0 < eps < 0.1):
            raise ValueError("--f-miss-grid-eps must be in (0,0.1).")
        f_grid = np.linspace(eps, 1.0 - eps, n_f)
        w_f = _trapz_weights(f_grid)
        logw_f = np.log(np.clip(w_f, 1e-300, np.inf))
        logf = np.log(f_grid)
        log1mf = np.log1p(-f_grid)
        log_prior_f = (a - 1.0) * logf + (b - 1.0) * log1mf - float(betaln(a, b))
        cfg.update(
            {
                "f_grid": np.asarray(f_grid, dtype=np.float64),
                "logw_f": np.asarray(logw_f, dtype=np.float64),
                "logf": np.asarray(logf, dtype=np.float64),
                "log1mf": np.asarray(log1mf, dtype=np.float64),
                "log_prior_f": np.asarray(log_prior_f, dtype=np.float64),
            }
        )

    # Share heavy objects via forked workers.
    _G.clear()
    _G["cfg"] = cfg
    _G["templates"] = templates
    _G["post"] = post
    _G["gw_prior"] = gw_prior
    _G["constants"] = constants
    _G["missing_pre"] = missing_pre
    _G["log_alpha_mu"] = log_alpha_mu
    _G["log_alpha_gr"] = log_alpha_gr

    # Determine which reps still need work.
    todo: list[int] = []
    for rep in range(int(args.n_rep)):
        rep_path = reps_dir / f"rep{rep:04d}.json"
        if not rep_path.exists():
            todo.append(int(rep))

    nproc = int(args.n_proc)
    if nproc <= 0:
        nproc = _default_nproc()

    def _run_parallel_pool(
        *,
        executor_cls: Any,
        max_workers: int,
        label: str,
        executor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        done = 0
        hb_sec = float(args.heartbeat_sec)
        if not (np.isfinite(hb_sec) and hb_sec > 0.0):
            raise ValueError("--heartbeat-sec must be finite and positive.")
        kwargs = dict(executor_kwargs or {})
        with executor_cls(max_workers=max_workers, **kwargs) as ex:
            fut_to_rep = {ex.submit(_run_one_rep, rep): int(rep) for rep in todo}
            pending = set(fut_to_rep.keys())
            last_hb = float(time.time())
            hb_prev_time = float(last_hb)
            hb_prev_events = 0.0

            while pending:
                done_futs, pending = wait(pending, timeout=hb_sec, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    row = fut.result()
                    done += 1
                    if done == 1 or (done % 5) == 0:
                        dt = time.time() - t0
                        d = float(row.get("delta_lpd_total", float("nan")))
                        print(
                            f"[inj] rep {done}/{len(todo)} ({label}): ΔLPD_total={d:+.3f} elapsed={dt/60:.1f} min",
                            flush=True,
                        )

                now = float(time.time())
                if (now - last_hb) >= hb_sec:
                    dt = time.time() - t0
                    hb = _heartbeat_status(reps_dir, n_rep_total=int(args.n_rep), n_events_per_rep=int(len(templates)))
                    pct_rep = 100.0 * float(hb["n_rep_done"]) / float(hb["n_rep_total"]) if hb["n_rep_total"] else 0.0
                    pct_evt = 100.0 * float(hb["n_events_done_est"]) / float(hb["n_events_total"]) if hb["n_events_total"] else 0.0
                    ev_now = float(hb.get("n_events_done_est", 0.0))
                    ev_total = float(hb.get("n_events_total", 0.0))
                    dt_hb = max(1e-6, float(now - hb_prev_time))
                    ev_rate_sec = max(0.0, (ev_now - hb_prev_events) / dt_hb)
                    ev_rate_min = 60.0 * ev_rate_sec
                    if ev_rate_sec > 0.0 and ev_total > ev_now:
                        eta_min_txt = f"{(ev_total - ev_now) / ev_rate_sec / 60.0:.1f}"
                    elif ev_total <= ev_now:
                        eta_min_txt = "0.0"
                    else:
                        eta_min_txt = "na"
                    hb_prev_time = now
                    hb_prev_events = ev_now

                    latest = hb.get("latest_partial") or {}
                    latest_rep = latest.get("rep", None)
                    latest_stage = latest.get("stage", None)
                    latest_event = latest.get("event", None)
                    latest_txt = ""
                    if latest_rep is not None:
                        try:
                            latest_txt = f" latest=rep{int(latest_rep):04d}:{latest_stage}:{latest_event}"
                        except Exception:
                            latest_txt = f" latest={latest_rep}:{latest_stage}:{latest_event}"

                    print(
                        f"[inj] heartbeat ({label}): reps_done={hb['n_rep_done']}/{hb['n_rep_total']} ({pct_rep:.2f}%) "
                        f"active={hb['n_rep_active']} partials_recent={hb['n_partials_recent']}/{hb['n_rep_active']} "
                        f"events_done≈{hb['n_events_done_est']:.1f}/{hb['n_events_total']} ({pct_evt:.2f}%)"
                        f" rate≈{ev_rate_min:.1f} ev/min eta≈{eta_min_txt} min"
                        f"{latest_txt} elapsed={dt/60:.1f} min",
                        flush=True,
                    )
                    last_hb = now

    backend = str(args.parallel_backend)
    if todo and nproc > 1 and len(todo) > 1 and backend != "sequential":
        max_workers = min(int(nproc), int(len(todo)))
        ran_pool = False
        if backend in ("auto", "process"):
            try:
                ctx = get_context("fork")
            except Exception:
                ctx = None
            if ctx is not None:
                try:
                    _run_parallel_pool(
                        executor_cls=ProcessPoolExecutor,
                        max_workers=max_workers,
                        label="process",
                        executor_kwargs={"mp_context": ctx},
                    )
                    ran_pool = True
                except PermissionError as exc:
                    if backend == "process":
                        raise
                    print(f"[inj] process pool unavailable ({exc}); falling back to thread pool", flush=True)
                except OSError as exc:
                    if backend == "process":
                        raise
                    print(f"[inj] process pool OS error ({exc}); falling back to thread pool", flush=True)
            elif backend == "process":
                raise RuntimeError("Requested process backend, but fork context is unavailable.")

        if not ran_pool and backend in ("auto", "thread"):
            _run_parallel_pool(
                executor_cls=ThreadPoolExecutor,
                max_workers=max_workers,
                label="thread",
            )
            ran_pool = True

        if not ran_pool:
            for k, rep in enumerate(todo, start=1):
                row = _run_one_rep(rep)
                if k == 1 or (k % 5) == 0:
                    dt = time.time() - t0
                    print(
                        f"[inj] rep {k}/{len(todo)} (sequential): ΔLPD_total={float(row['delta_lpd_total']):+.3f} elapsed={dt/60:.1f} min",
                        flush=True,
                    )
    else:
        for k, rep in enumerate(todo, start=1):
            row = _run_one_rep(rep)
            if k == 1 or (k % 5) == 0:
                dt = time.time() - t0
                print(
                    f"[inj] rep {k}/{len(todo)} (sequential): ΔLPD_total={float(row['delta_lpd_total']):+.3f} elapsed={dt/60:.1f} min",
                    flush=True,
                )

    # Summarize from on-disk reps (includes any preexisting).
    delta_total: list[float] = []
    delta_data: list[float] = []
    delta_sel: list[float] = []
    n_done = 0
    for rep in range(int(args.n_rep)):
        rep_path = reps_dir / f"rep{rep:04d}.json"
        if not rep_path.exists():
            continue
        try:
            d = json.loads(rep_path.read_text())
        except Exception:
            continue
        if str(d.get("truth_model")) != str(args.truth_model):
            continue
        n_done += 1
        delta_total.append(float(d["delta_lpd_total"]))
        delta_data.append(float(d.get("delta_lpd_total_data", float("nan"))))
        delta_sel.append(float(d.get("delta_lpd_total_sel", float("nan"))))

    summary = {
        "truth_model": str(args.truth_model),
        "n_rep": int(args.n_rep),
        "n_rep_done": int(n_done),
        "n_events": int(len(templates)),
        "n_draws": int(n_draws),
        "delta_lpd_total": _summarize(delta_total),
        "delta_lpd_total_data": _summarize(delta_data),
        "delta_lpd_total_sel": _summarize(delta_sel),
        "elapsed_sec": float(time.time() - t0),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
