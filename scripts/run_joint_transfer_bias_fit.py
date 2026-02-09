#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
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
from scipy.linalg import cho_solve

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike


_STATE: dict[str, Any] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _logmeanexp(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def _softmax_cols(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    n_row, n_col = logw.shape
    out = np.zeros((n_row, n_col), dtype=float)
    m = np.max(logw, axis=0)
    valid = np.isfinite(m)
    if np.any(valid):
        lw = logw[:, valid] - m[valid].reshape((1, -1))
        w = np.exp(lw)
        w[~np.isfinite(w)] = 0.0
        s = np.sum(w, axis=0, keepdims=True)
        out[:, valid] = w / np.clip(s, 1e-300, np.inf)
    return out


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    qs = np.asarray(qs, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = np.clip(weights[order], 0.0, np.inf)
    sw = np.sum(w)
    if not np.isfinite(sw) or sw <= 0.0:
        return np.array([float(np.nan) for _ in qs], dtype=float)
    cdf = np.cumsum(w) / sw
    return np.interp(qs, cdf, v, left=v[0], right=v[-1])


def _parse_list_str(text: str) -> list[str]:
    vals = [str(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Expected non-empty string list.")
    return vals


def _infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_run_draws(run_dir: Path) -> dict[str, np.ndarray]:
    npz_path = run_dir / "samples" / "mu_forward_posterior.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing posterior artifact: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as d:
        required = [
            "x_grid",
            "logmu_x_samples",
            "z_grid",
            "H_samples",
            "H0",
            "omega_m0",
            "omega_k0",
            "r_d_Mpc",
        ]
        for key in required:
            if key not in d.files:
                raise ValueError(f"{npz_path}: missing key '{key}'")
        return {
            "x_grid": np.asarray(d["x_grid"], dtype=float),
            "logmu_x_samples": np.asarray(d["logmu_x_samples"], dtype=float),
            "z_grid": np.asarray(d["z_grid"], dtype=float),
            "H_samples": np.asarray(d["H_samples"], dtype=float),
            "H0": np.asarray(d["H0"], dtype=float),
            "omega_m0": np.asarray(d["omega_m0"], dtype=float),
            "omega_k0": np.asarray(d["omega_k0"], dtype=float),
            "r_d_Mpc": np.asarray(d["r_d_Mpc"], dtype=float),
        }


@dataclass(frozen=True)
class DrawFeatures:
    H0: np.ndarray
    sn_mu_base: np.ndarray
    sn_logmu_ratio: np.ndarray
    cc_H_base: np.ndarray
    cc_logmu_ratio: np.ndarray
    bao_y_base_by_dataset: list[np.ndarray]
    bao_logmu_ratio_by_dataset: list[np.ndarray]


def _compute_logmu_ratio(
    *,
    z: np.ndarray,
    H_at_z: np.ndarray,
    H0: float,
    omega_k0: float,
    x_grid: np.ndarray,
    logmu_x: np.ndarray,
) -> np.ndarray:
    denom0 = H0**2 * (1.0 - omega_k0)
    denom = H_at_z**2 - omega_k0 * H0**2 * (1.0 + z) ** 2
    if denom0 <= 0.0 or np.any(denom <= 0.0):
        raise ValueError("Non-physical horizon area mapping in logmu ratio computation.")
    x_eval = np.log(denom0 / denom)
    xmin = float(x_grid[0])
    xmax = float(x_grid[-1])
    x_eval = np.clip(x_eval, xmin, xmax)
    logmu_eval = np.interp(x_eval, x_grid, logmu_x)
    logmu0 = float(np.interp(0.0, x_grid, logmu_x))
    return logmu_eval - logmu0


def _build_data_likes(
    *,
    repo_root: Path,
    z_min: float,
    z_max: float,
    sn_subset: str,
    sn_cov_kind: str,
    sn_z_column: str,
    sn_bin_count: int,
    cc_variant: str,
    bao_datasets: list[str],
) -> tuple[SNLogLike, ChronometerLogLike, list[BaoLogLike]]:
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()

    sn_raw = load_pantheon_plus(
        paths=paths,
        cov_kind=sn_cov_kind,
        subset=sn_subset,
        z_column=sn_z_column,
    )
    sn_like = SNLogLike.from_pantheon(sn_raw, z_min=z_min, z_max=z_max)
    z_edges = np.quantile(sn_like.z, np.linspace(0.0, 1.0, int(sn_bin_count) + 1))
    z_edges = np.maximum.accumulate(z_edges)
    for i in range(1, z_edges.size):
        if z_edges[i] <= z_edges[i - 1]:
            z_edges[i] = z_edges[i - 1] + 1e-6
    sn_like = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=2)

    cc = load_chronometers(paths=paths, variant=cc_variant)
    cc_like = ChronometerLogLike.from_data(cc, z_min=z_min, z_max=z_max)

    bao_likes: list[BaoLogLike] = []
    for ds in bao_datasets:
        bao = load_bao(paths=paths, dataset=ds)
        try:
            like = BaoLogLike.from_data(
                bao,
                dataset=ds,
                constants=constants,
                z_min=z_min,
                z_max=z_max,
                diag_cov=False,
            )
        except ValueError as e:
            if "No BAO points in selected z-range" in str(e):
                continue
            raise
        bao_likes.append(like)
    if not bao_likes:
        raise ValueError("No BAO likelihoods available after z-range filtering.")
    return sn_like, cc_like, bao_likes


def _build_draw_features(
    *,
    draws_by_run: list[dict[str, np.ndarray]],
    draws_total: int,
    seed: int,
    sn_like: SNLogLike,
    cc_like: ChronometerLogLike,
    bao_likes: list[BaoLogLike],
) -> DrawFeatures:
    rng = np.random.default_rng(int(seed))
    n_runs = len(draws_by_run)
    if n_runs <= 0:
        raise ValueError("No run dirs supplied.")
    per_run = int(math.ceil(float(draws_total) / float(n_runs)))

    H0_all: list[np.ndarray] = []
    sn_mu_all: list[np.ndarray] = []
    sn_t_all: list[np.ndarray] = []
    cc_h_all: list[np.ndarray] = []
    cc_t_all: list[np.ndarray] = []
    bao_y_all: list[list[np.ndarray]] = [[] for _ in bao_likes]
    bao_t_all: list[list[np.ndarray]] = [[] for _ in bao_likes]

    constants = PhysicalConstants()
    z_sn = np.asarray(sn_like.z, dtype=float)
    z_cc = np.asarray(cc_like.z, dtype=float)

    for run in draws_by_run:
        z_grid = np.asarray(run["z_grid"], dtype=float)
        x_grid = np.asarray(run["x_grid"], dtype=float)
        H_samples = np.asarray(run["H_samples"], dtype=float)
        H0 = np.asarray(run["H0"], dtype=float)
        omega_k0 = np.asarray(run["omega_k0"], dtype=float)
        logmu = np.asarray(run["logmu_x_samples"], dtype=float)
        r_d = np.asarray(run["r_d_Mpc"], dtype=float)
        n = int(H_samples.shape[0])
        if per_run < n:
            idx = np.sort(rng.choice(n, size=per_run, replace=False))
        else:
            idx = np.arange(n, dtype=int)

        for j in idx:
            bg = build_background_from_H_grid(z_grid, H_samples[j], constants=constants)
            H0_j = float(H0[j])
            ok_j = float(omega_k0[j])
            logmu_j = logmu[j]
            rd_j = float(r_d[j])

            dl_sn = bg.Dl(z_sn)
            sn_mu = 5.0 * np.log10(np.clip(dl_sn, 1e-9, np.inf)) + 25.0
            H_sn = bg.H(z_sn)
            t_sn = _compute_logmu_ratio(
                z=z_sn,
                H_at_z=H_sn,
                H0=H0_j,
                omega_k0=ok_j,
                x_grid=x_grid,
                logmu_x=logmu_j,
            )

            H_cc = bg.H(z_cc)
            t_cc = _compute_logmu_ratio(
                z=z_cc,
                H_at_z=H_cc,
                H0=H0_j,
                omega_k0=ok_j,
                x_grid=x_grid,
                logmu_x=logmu_j,
            )

            y_bao_rows: list[np.ndarray] = []
            t_bao_rows: list[np.ndarray] = []
            for like in bao_likes:
                y = like.predict(bg, r_d_Mpc=rd_j)
                H_b = bg.H(like.z)
                t_b = _compute_logmu_ratio(
                    z=like.z,
                    H_at_z=H_b,
                    H0=H0_j,
                    omega_k0=ok_j,
                    x_grid=x_grid,
                    logmu_x=logmu_j,
                )
                y_bao_rows.append(y)
                t_bao_rows.append(t_b)

            H0_all.append(np.array([H0_j], dtype=float))
            sn_mu_all.append(sn_mu[None, :])
            sn_t_all.append(t_sn[None, :])
            cc_h_all.append(H_cc[None, :])
            cc_t_all.append(t_cc[None, :])
            for k in range(len(bao_likes)):
                bao_y_all[k].append(y_bao_rows[k][None, :])
                bao_t_all[k].append(t_bao_rows[k][None, :])

    H0_arr = np.concatenate(H0_all, axis=0).reshape(-1)
    sn_mu_arr = np.concatenate(sn_mu_all, axis=0)
    sn_t_arr = np.concatenate(sn_t_all, axis=0)
    cc_h_arr = np.concatenate(cc_h_all, axis=0)
    cc_t_arr = np.concatenate(cc_t_all, axis=0)
    bao_y_arr = [np.concatenate(rows, axis=0) for rows in bao_y_all]
    bao_t_arr = [np.concatenate(rows, axis=0) for rows in bao_t_all]

    if H0_arr.size > draws_total:
        keep = np.sort(rng.choice(H0_arr.size, size=int(draws_total), replace=False))
        H0_arr = H0_arr[keep]
        sn_mu_arr = sn_mu_arr[keep]
        sn_t_arr = sn_t_arr[keep]
        cc_h_arr = cc_h_arr[keep]
        cc_t_arr = cc_t_arr[keep]
        bao_y_arr = [x[keep] for x in bao_y_arr]
        bao_t_arr = [x[keep] for x in bao_t_arr]

    return DrawFeatures(
        H0=H0_arr,
        sn_mu_base=sn_mu_arr,
        sn_logmu_ratio=sn_t_arr,
        cc_H_base=cc_h_arr,
        cc_logmu_ratio=cc_t_arr,
        bao_y_base_by_dataset=bao_y_arr,
        bao_logmu_ratio_by_dataset=bao_t_arr,
    )


def _make_synthetic_observations(
    *,
    rng: np.random.Generator,
    features: DrawFeatures,
    sn_like: SNLogLike,
    cc_like: ChronometerLogLike,
    bao_likes: list[BaoLogLike],
    draw_index: int,
    beta_ia: float,
    beta_cc: float,
    beta_bao: float,
    delta_h0_ladder: float,
    h0_local_sigma: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    j = int(draw_index)
    if j < 0 or j >= features.H0.size:
        raise ValueError("synthetic draw_index out of range.")
    sn_mean = features.sn_mu_base[j] + float(beta_ia) * features.sn_logmu_ratio[j]
    sn_obs = rng.multivariate_normal(mean=sn_mean, cov=sn_like.cov)

    cc_mean = features.cc_H_base[j] * (1.0 + float(beta_cc) * features.cc_logmu_ratio[j])
    cc_obs = cc_mean + cc_like.sigma_H * rng.normal(size=cc_like.sigma_H.size)

    bao_obs: list[np.ndarray] = []
    for k, like in enumerate(bao_likes):
        y_mean = features.bao_y_base_by_dataset[k][j] * (
            1.0 + float(beta_bao) * features.bao_logmu_ratio_by_dataset[k][j]
        )
        bao_obs.append(rng.multivariate_normal(mean=y_mean, cov=like.cov))

    h0_local_obs = float(
        features.H0[j]
        + float(delta_h0_ladder)
        + rng.normal(scale=float(h0_local_sigma))
    )
    return sn_obs, cc_obs, bao_obs, h0_local_obs


def _prepare_quadratics(
    *,
    features: DrawFeatures,
    sn_like: SNLogLike,
    cc_like: ChronometerLogLike,
    bao_likes: list[BaoLogLike],
    sn_obs: np.ndarray,
    cc_obs: np.ndarray,
    bao_obs: list[np.ndarray],
) -> dict[str, Any]:
    n_draw = int(features.H0.size)
    K = float(sn_like.ones_cinv_ones)
    if K <= 0.0:
        raise ValueError("Invalid SN normalization (ones^T C^-1 ones <= 0).")
    sn_const = float(sn_like.logdet + np.log(K))
    sn_A = np.empty(n_draw, dtype=float)
    sn_B = np.empty(n_draw, dtype=float)
    sn_C = np.empty(n_draw, dtype=float)
    sn_b0 = np.empty(n_draw, dtype=float)
    sn_bt = np.empty(n_draw, dtype=float)

    cc_U = np.empty(n_draw, dtype=float)
    cc_V = np.empty(n_draw, dtype=float)
    cc_W = np.empty(n_draw, dtype=float)
    cc_tmin = np.empty(n_draw, dtype=float)
    cc_tmax = np.empty(n_draw, dtype=float)
    w_cc = 1.0 / np.clip(cc_like.sigma_H, 1e-12, np.inf) ** 2

    bao_U = np.zeros(n_draw, dtype=float)
    bao_V = np.zeros(n_draw, dtype=float)
    bao_W = np.zeros(n_draw, dtype=float)
    bao_const = float(np.sum([like.logdet for like in bao_likes]))
    bao_tmin = np.full(n_draw, np.inf, dtype=float)
    bao_tmax = np.full(n_draw, -np.inf, dtype=float)

    for j in range(n_draw):
        a0 = sn_obs - features.sn_mu_base[j]
        t_sn = features.sn_logmu_ratio[j]
        cinv_a0 = cho_solve(sn_like.cho, a0, check_finite=False)
        cinv_t = cho_solve(sn_like.cho, t_sn, check_finite=False)
        sn_A[j] = float(a0 @ cinv_a0)
        sn_B[j] = float(a0 @ cinv_t)
        sn_C[j] = float(t_sn @ cinv_t)
        sn_b0[j] = float(np.sum(cinv_a0))
        sn_bt[j] = float(np.sum(cinv_t))

        u_cc = cc_obs - features.cc_H_base[j]
        v_cc = features.cc_H_base[j] * features.cc_logmu_ratio[j]
        cc_U[j] = float(np.sum((u_cc * u_cc) * w_cc))
        cc_V[j] = float(np.sum((u_cc * v_cc) * w_cc))
        cc_W[j] = float(np.sum((v_cc * v_cc) * w_cc))
        cc_tmin[j] = float(np.min(features.cc_logmu_ratio[j]))
        cc_tmax[j] = float(np.max(features.cc_logmu_ratio[j]))

        tmin = np.inf
        tmax = -np.inf
        Uj = 0.0
        Vj = 0.0
        Wj = 0.0
        for k, like in enumerate(bao_likes):
            u = bao_obs[k] - features.bao_y_base_by_dataset[k][j]
            t = features.bao_logmu_ratio_by_dataset[k][j]
            v = features.bao_y_base_by_dataset[k][j] * t
            cinv_u = cho_solve(like.cov_cho, u, check_finite=False)
            cinv_v = cho_solve(like.cov_cho, v, check_finite=False)
            Uj += float(u @ cinv_u)
            Vj += float(u @ cinv_v)
            Wj += float(v @ cinv_v)
            tmin = min(tmin, float(np.min(t)))
            tmax = max(tmax, float(np.max(t)))
        bao_U[j] = Uj
        bao_V[j] = Vj
        bao_W[j] = Wj
        bao_tmin[j] = tmin
        bao_tmax[j] = tmax

    return {
        "H0": features.H0,
        "sn_A": sn_A,
        "sn_B": sn_B,
        "sn_C": sn_C,
        "sn_b0": sn_b0,
        "sn_bt": sn_bt,
        "sn_K": float(K),
        "sn_const": float(sn_const),
        "cc_U": cc_U,
        "cc_V": cc_V,
        "cc_W": cc_W,
        "cc_tmin": cc_tmin,
        "cc_tmax": cc_tmax,
        "bao_U": bao_U,
        "bao_V": bao_V,
        "bao_W": bao_W,
        "bao_const": float(bao_const),
        "bao_tmin": bao_tmin,
        "bao_tmax": bao_tmax,
    }


def _set_state(state: dict[str, Any]) -> None:
    global _STATE
    _STATE = state


def _eval_chunk(
    chunk_id: int,
    beta_ia: np.ndarray,
    beta_cc: np.ndarray,
    delta_h0: np.ndarray,
    beta_bao: np.ndarray,
) -> dict[str, Any]:
    s = _STATE
    H0 = np.asarray(s["H0"], dtype=float)
    n_draw = H0.size
    n_t = int(beta_ia.size)

    b_ia = beta_ia.reshape((1, n_t))
    b_cc = beta_cc.reshape((1, n_t))
    d_h0 = delta_h0.reshape((1, n_t))
    b_bao = beta_bao.reshape((1, n_t))

    sn_A = s["sn_A"].reshape((n_draw, 1))
    sn_B = s["sn_B"].reshape((n_draw, 1))
    sn_C = s["sn_C"].reshape((n_draw, 1))
    sn_b0 = s["sn_b0"].reshape((n_draw, 1))
    sn_bt = s["sn_bt"].reshape((n_draw, 1))
    K = float(s["sn_K"])
    sn_const = float(s["sn_const"])
    sn = -0.5 * (
        sn_A
        - 2.0 * sn_B * b_ia
        + sn_C * (b_ia**2)
        - ((sn_b0 - sn_bt * b_ia) ** 2) / K
        + sn_const
    )

    cc_U = s["cc_U"].reshape((n_draw, 1))
    cc_V = s["cc_V"].reshape((n_draw, 1))
    cc_W = s["cc_W"].reshape((n_draw, 1))
    cc = -0.5 * (cc_U - 2.0 * cc_V * b_cc + cc_W * (b_cc**2))
    cc_tmin = s["cc_tmin"].reshape((n_draw, 1))
    cc_tmax = s["cc_tmax"].reshape((n_draw, 1))
    invalid_cc = ((b_cc >= 0.0) & (1.0 + b_cc * cc_tmin <= 0.0)) | (
        (b_cc < 0.0) & (1.0 + b_cc * cc_tmax <= 0.0)
    )
    cc = np.where(invalid_cc, -np.inf, cc)

    bao_U = s["bao_U"].reshape((n_draw, 1))
    bao_V = s["bao_V"].reshape((n_draw, 1))
    bao_W = s["bao_W"].reshape((n_draw, 1))
    bao_const = float(s["bao_const"])
    bao = -0.5 * (bao_U - 2.0 * bao_V * b_bao + bao_W * (b_bao**2) + bao_const)
    bao_tmin = s["bao_tmin"].reshape((n_draw, 1))
    bao_tmax = s["bao_tmax"].reshape((n_draw, 1))
    invalid_bao = ((b_bao >= 0.0) & (1.0 + b_bao * bao_tmin <= 0.0)) | (
        (b_bao < 0.0) & (1.0 + b_bao * bao_tmax <= 0.0)
    )
    bao = np.where(invalid_bao, -np.inf, bao)

    local_sigma = float(s["local_sigma"])
    local_ref = float(s["local_ref"])
    local_const = np.log(local_sigma) + 0.5 * np.log(2.0 * np.pi)
    local = -0.5 * ((local_ref - (H0.reshape((n_draw, 1)) + d_h0)) / local_sigma) ** 2 - local_const

    logL = sn + cc + bao + local
    logZ = np.full((n_t,), -np.inf, dtype=float)
    valid_col = np.any(np.isfinite(logL), axis=0)
    if np.any(valid_col):
        logZ[valid_col] = _logmeanexp(logL[:, valid_col], axis=0)
    wj = _softmax_cols(logL)

    baseline_gap = float(s["baseline_gap"])
    relief = 1.0 - np.abs((local_ref - d_h0) - H0.reshape((n_draw, 1))) / max(1e-12, baseline_gap)
    relief_mean = np.sum(wj * relief, axis=0)
    relief_m2 = np.sum(wj * (relief**2), axis=0)
    relief_sd = np.sqrt(np.clip(relief_m2 - relief_mean**2, 0.0, np.inf))

    sn0 = s["sn0"].reshape((n_draw, 1))
    cc0 = s["cc0"].reshape((n_draw, 1))
    bao0 = s["bao0"].reshape((n_draw, 1))
    local0 = s["local0"].reshape((n_draw, 1))

    dsn = np.abs(sn - sn0)
    dcc = np.abs(cc - cc0)
    dbo = np.abs(bao - bao0)
    dlo = np.abs(local - local0)
    dsn[~np.isfinite(dsn)] = 0.0
    dcc[~np.isfinite(dcc)] = 0.0
    dbo[~np.isfinite(dbo)] = 0.0
    dlo[~np.isfinite(dlo)] = 0.0

    dsn_abs = np.sum(wj * dsn, axis=0)
    dcc_abs = np.sum(wj * dcc, axis=0)
    dbo_abs = np.sum(wj * dbo, axis=0)
    dlo_abs = np.sum(wj * dlo, axis=0)

    return {
        "chunk_id": int(chunk_id),
        "beta_ia": beta_ia,
        "beta_cc": beta_cc,
        "delta_h0_ladder": delta_h0,
        "beta_bao": beta_bao,
        "logZ_theta": logZ,
        "relief_mean": relief_mean,
        "relief_sd": relief_sd,
        "mean_abs_delta_sn": dsn_abs,
        "mean_abs_delta_cc": dcc_abs,
        "mean_abs_delta_bao": dbo_abs,
        "mean_abs_delta_local": dlo_abs,
    }


def _save_chunk(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint SN+BAO+CC+O3 transfer-bias marginal fit with resumable chunked evaluation.")
    ap.add_argument(
        "--run-dirs",
        default="outputs/finalization/highpower_multistart_v2/M0_start101,outputs/finalization/highpower_multistart_v2/M0_start202,outputs/finalization/highpower_multistart_v2/M0_start303,outputs/finalization/highpower_multistart_v2/M0_start404,outputs/finalization/highpower_multistart_v2/M0_start505",
    )
    ap.add_argument("--out", default=None, help="Output directory (default outputs/joint_transfer_bias_fit_<UTC>).")
    ap.add_argument("--draws-total", type=int, default=4096)
    ap.add_argument("--theta-samples", type=int, default=8192)
    ap.add_argument("--theta-chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--max-chunks", type=int, default=0, help="If >0, process only first N chunks (smoke/debug).")

    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--sn-subset", default="cosmology")
    ap.add_argument("--sn-cov-kind", default="stat+sys")
    ap.add_argument("--sn-z-column", default="zHD")
    ap.add_argument("--sn-bin-count", type=int, default=12)
    ap.add_argument("--cc-variant", default="BC03_all")
    ap.add_argument("--bao-datasets", default="sdss_dr12_consensus_bao,sdss_dr16_lrg_bao_dmdh,desi_2024_bao_all")

    ap.add_argument("--h0-local-ref", type=float, default=73.0)
    ap.add_argument("--h0-local-sigma", type=float, default=1.0)
    ap.add_argument("--h0-planck-ref", type=float, default=67.4)

    ap.add_argument("--beta-ia-sigma", type=float, default=0.05)
    ap.add_argument("--beta-cc-sigma", type=float, default=0.20)
    ap.add_argument("--delta-h0-ladder-sigma", type=float, default=1.0)
    ap.add_argument("--beta-bao-sigma", type=float, default=0.03)

    ap.add_argument("--o3-delta-lpd", type=float, default=3.669945265, help="Reported O3 MG-vs-GR support (added as metadata; cancels in transfer BF).")
    ap.add_argument(
        "--synthetic-theta-json",
        default=None,
        help="Optional JSON with synthetic truth keys: beta_ia,beta_cc,beta_bao,delta_h0_ladder,draw_index,seed.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"joint_transfer_bias_fit_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    progress_path = tab_dir / "progress.json"
    run_log = out_dir / "run.log"

    repo_root = _infer_repo_root()
    run_dirs = [Path(p).resolve() for p in _parse_list_str(args.run_dirs)]
    bao_datasets = _parse_list_str(args.bao_datasets)
    t0 = time.time()

    with run_log.open("a", encoding="utf-8") as lg:
        lg.write(f"[start] utc={_utc_now()} out={out_dir}\n")
        lg.flush()

    sn_like, cc_like, bao_likes = _build_data_likes(
        repo_root=repo_root,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        sn_subset=str(args.sn_subset),
        sn_cov_kind=str(args.sn_cov_kind),
        sn_z_column=str(args.sn_z_column),
        sn_bin_count=int(args.sn_bin_count),
        cc_variant=str(args.cc_variant),
        bao_datasets=bao_datasets,
    )

    draws_by_run = [_load_run_draws(p) for p in run_dirs]
    features = _build_draw_features(
        draws_by_run=draws_by_run,
        draws_total=int(args.draws_total),
        seed=int(args.seed),
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
    )
    n_draw = int(features.H0.size)
    if n_draw <= 20:
        raise ValueError("Too few valid posterior draws after feature construction.")

    sn_obs = np.asarray(sn_like.m, dtype=float)
    cc_obs = np.asarray(cc_like.H, dtype=float)
    bao_obs = [np.asarray(like.y, dtype=float) for like in bao_likes]
    local_ref_obs = float(args.h0_local_ref)

    synthetic_meta: dict[str, Any] | None = None
    if args.synthetic_theta_json:
        cfg = json.loads(Path(args.synthetic_theta_json).read_text(encoding="utf-8"))
        syn_rng = np.random.default_rng(int(cfg.get("seed", int(args.seed) + 9999)))
        sn_obs, cc_obs, bao_obs, local_ref_obs = _make_synthetic_observations(
            rng=syn_rng,
            features=features,
            sn_like=sn_like,
            cc_like=cc_like,
            bao_likes=bao_likes,
            draw_index=int(cfg.get("draw_index", 0)),
            beta_ia=float(cfg.get("beta_ia", 0.0)),
            beta_cc=float(cfg.get("beta_cc", 0.0)),
            beta_bao=float(cfg.get("beta_bao", 0.0)),
            delta_h0_ladder=float(cfg.get("delta_h0_ladder", 0.0)),
            h0_local_sigma=float(args.h0_local_sigma),
        )
        synthetic_meta = {
            "enabled": True,
            "draw_index": int(cfg.get("draw_index", 0)),
            "beta_ia": float(cfg.get("beta_ia", 0.0)),
            "beta_cc": float(cfg.get("beta_cc", 0.0)),
            "beta_bao": float(cfg.get("beta_bao", 0.0)),
            "delta_h0_ladder": float(cfg.get("delta_h0_ladder", 0.0)),
            "seed": int(cfg.get("seed", int(args.seed) + 9999)),
        }

    q = _prepare_quadratics(
        features=features,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        sn_obs=sn_obs,
        cc_obs=cc_obs,
        bao_obs=bao_obs,
    )
    q["local_ref"] = float(local_ref_obs)
    q["local_sigma"] = float(args.h0_local_sigma)
    q["baseline_gap"] = abs(float(local_ref_obs) - float(args.h0_planck_ref))

    H0 = q["H0"]
    n = H0.size
    sn0 = -0.5 * (
        q["sn_A"]
        - (q["sn_b0"] ** 2) / q["sn_K"]
        + q["sn_const"]
    )
    cc0 = -0.5 * q["cc_U"]
    bao0 = -0.5 * (q["bao_U"] + q["bao_const"])
    local_sigma = float(q["local_sigma"])
    local_const = np.log(local_sigma) + 0.5 * np.log(2.0 * np.pi)
    local0 = -0.5 * ((float(local_ref_obs) - H0) / local_sigma) ** 2 - local_const
    logL0 = sn0 + cc0 + bao0 + local0
    logZ0 = float(_logmeanexp(logL0, axis=0))
    q["sn0"] = sn0
    q["cc0"] = cc0
    q["bao0"] = bao0
    q["local0"] = local0

    _set_state(q)

    rng = np.random.default_rng(int(args.seed) + 12345)
    n_theta = int(args.theta_samples)
    beta_ia_all = float(args.beta_ia_sigma) * rng.normal(size=n_theta)
    beta_cc_all = float(args.beta_cc_sigma) * rng.normal(size=n_theta)
    delta_all = float(args.delta_h0_ladder_sigma) * rng.normal(size=n_theta)
    if float(args.beta_bao_sigma) > 0:
        beta_bao_all = float(args.beta_bao_sigma) * rng.normal(size=n_theta)
    else:
        beta_bao_all = np.zeros(n_theta, dtype=float)

    n_chunk = int(math.ceil(n_theta / max(1, int(args.theta_chunk))))
    if int(args.max_chunks) > 0:
        n_chunk = min(n_chunk, int(args.max_chunks))
    chunk_paths = [chunks_dir / f"chunk_{i:05d}.npz" for i in range(n_chunk)]

    _write_json_atomic(
        tab_dir / "manifest.json",
        {
            "created_utc": _utc_now(),
            "argv": os.sys.argv,
            "run_dirs": [str(p) for p in run_dirs],
            "draws_total_requested": int(args.draws_total),
            "draws_effective": int(n_draw),
            "theta_samples": int(n_theta),
            "theta_chunk": int(args.theta_chunk),
            "workers": int(args.workers),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "sn_bin_count": int(args.sn_bin_count),
            "cc_variant": str(args.cc_variant),
            "bao_datasets": bao_datasets,
            "h0_local_ref_input": float(args.h0_local_ref),
            "h0_local_ref_used": float(local_ref_obs),
            "h0_local_sigma": float(args.h0_local_sigma),
            "h0_planck_ref": float(args.h0_planck_ref),
            "priors": {
                "beta_ia_sigma": float(args.beta_ia_sigma),
                "beta_cc_sigma": float(args.beta_cc_sigma),
                "delta_h0_ladder_sigma": float(args.delta_h0_ladder_sigma),
                "beta_bao_sigma": float(args.beta_bao_sigma),
            },
            "o3_delta_lpd_metadata": float(args.o3_delta_lpd),
            "synthetic_mode": synthetic_meta if synthetic_meta is not None else {"enabled": False},
        },
    )

    done = 0
    failed = 0
    last_hb = 0.0
    with run_log.open("a", encoding="utf-8") as lg:
        lg.write(f"[eval] start chunks={n_chunk}\n")
        lg.flush()
        with cf.ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            mp_context=mp.get_context("fork"),
        ) as ex:
            fut_map: dict[cf.Future[dict[str, Any]], tuple[int, int, int]] = {}
            for i in range(n_chunk):
                if bool(args.resume) and chunk_paths[i].exists():
                    done += 1
                    continue
                s0 = i * int(args.theta_chunk)
                s1 = min((i + 1) * int(args.theta_chunk), n_theta)
                fut = ex.submit(
                    _eval_chunk,
                    i,
                    beta_ia_all[s0:s1],
                    beta_cc_all[s0:s1],
                    delta_all[s0:s1],
                    beta_bao_all[s0:s1],
                )
                fut_map[fut] = (i, s0, s1)

            while fut_map:
                try:
                    completed_iter = cf.as_completed(list(fut_map.keys()), timeout=1.0)
                    for fut in completed_iter:
                        i, s0, s1 = fut_map.pop(fut)
                        try:
                            payload = fut.result()
                            _save_chunk(chunk_paths[i], payload)
                            done += 1
                            lg.write(f"[chunk] done i={i} span={s0}:{s1}\n")
                        except Exception as e:
                            failed += 1
                            lg.write(f"[chunk] failed i={i} err={e}\n")
                        lg.flush()
                except cf.TimeoutError:
                    pass
                now = time.time()
                if (now - last_hb) >= float(args.heartbeat_sec):
                    _write_json_atomic(
                        progress_path,
                        {
                            "updated_utc": _utc_now(),
                            "elapsed_sec": float(now - t0),
                            "chunks_total": int(n_chunk),
                            "chunks_done": int(done),
                            "chunks_failed": int(failed),
                            "pct_done": float(100.0 * done / max(1, n_chunk)),
                        },
                    )
                    print(
                        f"[heartbeat] done={done}/{n_chunk} failed={failed} pct={100.0*done/max(1,n_chunk):.1f}% elapsed={now-t0:.1f}s",
                        flush=True,
                    )
                    last_hb = now

    if failed > 0:
        raise RuntimeError(f"Chunk evaluation failed for {failed} chunk(s).")

    chunks = []
    for i, p in enumerate(chunk_paths):
        if not p.exists():
            raise FileNotFoundError(f"Missing chunk output: {p}")
        with np.load(p, allow_pickle=False) as d:
            chunks.append({k: np.asarray(d[k]) for k in d.files})
    beta_ia = np.concatenate([c["beta_ia"] for c in chunks], axis=0)
    beta_cc = np.concatenate([c["beta_cc"] for c in chunks], axis=0)
    delta_h0 = np.concatenate([c["delta_h0_ladder"] for c in chunks], axis=0)
    beta_bao = np.concatenate([c["beta_bao"] for c in chunks], axis=0)
    logZ = np.concatenate([c["logZ_theta"] for c in chunks], axis=0)
    relief_mean = np.concatenate([c["relief_mean"] for c in chunks], axis=0)
    relief_sd = np.concatenate([c["relief_sd"] for c in chunks], axis=0)
    dsn = np.concatenate([c["mean_abs_delta_sn"] for c in chunks], axis=0)
    dcc = np.concatenate([c["mean_abs_delta_cc"] for c in chunks], axis=0)
    dbo = np.concatenate([c["mean_abs_delta_bao"] for c in chunks], axis=0)
    dlo = np.concatenate([c["mean_abs_delta_local"] for c in chunks], axis=0)

    logZ_transfer = float(_logmeanexp(logZ, axis=0))
    logbf_transfer_vs_no = float(logZ_transfer - logZ0)

    lw = logZ - np.max(logZ)
    w_theta = np.exp(lw)
    w_theta /= np.clip(np.sum(w_theta), 1e-300, np.inf)

    q16, q50, q84 = _weighted_quantiles(relief_mean, w_theta, np.array([0.16, 0.5, 0.84], dtype=float))
    relief_post_mean = float(np.sum(w_theta * relief_mean))
    relief_post_var = float(np.sum(w_theta * (relief_sd**2 + relief_mean**2)) - relief_post_mean**2)
    relief_post_sd = float(np.sqrt(max(0.0, relief_post_var)))

    term_means = {
        "beta_ia_sn": float(np.sum(w_theta * dsn)),
        "beta_cc_cc": float(np.sum(w_theta * dcc)),
        "beta_bao_bao": float(np.sum(w_theta * dbo)),
        "delta_h0_ladder_local": float(np.sum(w_theta * dlo)),
    }
    dominant_term = max(term_means.items(), key=lambda kv: kv[1])[0]

    params = {
        "beta_ia": beta_ia,
        "beta_cc": beta_cc,
        "delta_h0_ladder": delta_h0,
        "beta_bao": beta_bao,
    }
    param_post: dict[str, dict[str, float]] = {}
    for name, arr in params.items():
        q = _weighted_quantiles(arr, w_theta, np.array([0.16, 0.5, 0.84], dtype=float))
        param_post[name] = {
            "mean": float(np.sum(w_theta * arr)),
            "p16": float(q[0]),
            "p50": float(q[1]),
            "p84": float(q[2]),
            "sd": float(np.sqrt(max(0.0, np.sum(w_theta * (arr**2)) - (np.sum(w_theta * arr)) ** 2))),
        }

    relief_corr: dict[str, dict[str, float]] = {}
    r_mean = relief_post_mean
    r_var = max(0.0, float(np.sum(w_theta * (relief_mean**2)) - r_mean**2))
    r_sd = float(np.sqrt(r_var)) if r_var > 0.0 else float("nan")
    for name, arr in params.items():
        x_mean = float(np.sum(w_theta * arr))
        x_var = float(np.sum(w_theta * (arr**2)) - x_mean**2)
        x_sd = float(np.sqrt(max(0.0, x_var))) if x_var > 0.0 else float("nan")
        cov = float(np.sum(w_theta * (arr - x_mean) * (relief_mean - r_mean)))
        corr = cov / (x_sd * r_sd) if np.isfinite(x_sd) and np.isfinite(r_sd) and x_sd > 0 and r_sd > 0 else float("nan")
        slope = cov / x_var if np.isfinite(x_var) and x_var > 0 else float("nan")
        relief_corr[name] = {
            "weighted_corr": float(corr),
            "weighted_slope": float(slope),
        }

    summary = {
        "created_utc": _utc_now(),
        "elapsed_sec": float(time.time() - t0),
        "draws_effective": int(n_draw),
        "theta_samples_evaluated": int(logZ.size),
        "logZ_no_transfer": float(logZ0),
        "logZ_transfer_model": float(logZ_transfer),
        "log_bayes_factor_transfer_vs_no_transfer": float(logbf_transfer_vs_no),
        "o3_delta_lpd_metadata": float(args.o3_delta_lpd),
        "relief_posterior": {
            "mean": float(relief_post_mean),
            "sd": float(relief_post_sd),
            "p16": float(q16),
            "p50": float(q50),
            "p84": float(q84),
        },
        "dominant_transfer_term": dominant_term,
        "term_dominance_mean_abs_loglike_delta": term_means,
        "relief_sensitivity_weighted": relief_corr,
        "parameter_posteriors": param_post,
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    rows_path = tab_dir / "theta_posterior_rows.csv"
    with rows_path.open("w", encoding="utf-8") as f:
        f.write("beta_ia,beta_cc,delta_h0_ladder,beta_bao,logZ,posterior_weight,relief_mean,relief_sd,mean_abs_delta_sn,mean_abs_delta_cc,mean_abs_delta_bao,mean_abs_delta_local\n")
        for i in range(logZ.size):
            f.write(
                f"{beta_ia[i]:.9g},{beta_cc[i]:.9g},{delta_h0[i]:.9g},{beta_bao[i]:.9g},"
                f"{logZ[i]:.9g},{w_theta[i]:.9g},{relief_mean[i]:.9g},{relief_sd[i]:.9g},"
                f"{dsn[i]:.9g},{dcc[i]:.9g},{dbo[i]:.9g},{dlo[i]:.9g}\n"
            )

    plt.figure(figsize=(8.8, 5.2))
    plt.hist(relief_mean, bins=60, weights=w_theta, alpha=0.8, color="C0", density=True)
    plt.axvline(float(q50), color="k", linewidth=1.2, label=f"p50={q50:.3f}")
    plt.axvline(float(q16), color="k", linewidth=0.8, linestyle="--")
    plt.axvline(float(q84), color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Relief fraction (posterior mean per theta)")
    plt.ylabel("Posterior density")
    plt.title("Joint transfer-bias relief posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "relief_posterior.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10.0, 7.0))
    for i, (name, arr) in enumerate(params.items(), start=1):
        ax = plt.subplot(2, 2, i)
        ax.hist(arr, bins=60, weights=w_theta, alpha=0.8, density=True)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "transfer_param_posteriors.png", dpi=180)
    plt.close()

    report_lines = [
        "# Joint Transfer-Bias Fit Summary",
        "",
        f"- Created UTC: `{summary['created_utc']}`",
        f"- Effective posterior draws: `{summary['draws_effective']}`",
        f"- Theta samples evaluated: `{summary['theta_samples_evaluated']}`",
        f"- log Bayes factor (transfer vs no-transfer): `{summary['log_bayes_factor_transfer_vs_no_transfer']:.4f}`",
        f"- Relief posterior (mean / p16 / p50 / p84): "
        f"`{summary['relief_posterior']['mean']:.4f} / {summary['relief_posterior']['p16']:.4f} / "
        f"{summary['relief_posterior']['p50']:.4f} / {summary['relief_posterior']['p84']:.4f}`",
        f"- Dominant transfer term: `{summary['dominant_transfer_term']}`",
        "",
        "## Transfer Term Dominance (posterior-weighted mean abs loglike shift)",
        "",
    ]
    for k, v in term_means.items():
        report_lines.append(f"- `{k}`: `{v:.6f}`")
    report_lines.extend(
        [
            "",
            "## Parameter Posteriors",
            "",
        ]
    )
    for name, st in param_post.items():
        report_lines.append(
            f"- `{name}` mean/p16/p50/p84/sd: "
            f"`{st['mean']:.6g}` / `{st['p16']:.6g}` / `{st['p50']:.6g}` / `{st['p84']:.6g}` / `{st['sd']:.6g}`"
        )
    report_lines.extend(
        [
            "",
            "## Relief Sensitivity (weighted corr / slope)",
            "",
        ]
    )
    for name, st in relief_corr.items():
        report_lines.append(
            f"- `{name}` corr/slope: `{st['weighted_corr']:.6g}` / `{st['weighted_slope']:.6g}`"
        )
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/theta_posterior_rows.csv`",
            "- `figures/relief_posterior.png`",
            "- `figures/transfer_param_posteriors.png`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    _write_json_atomic(
        progress_path,
        {
            "updated_utc": _utc_now(),
            "elapsed_sec": float(time.time() - t0),
            "chunks_total": int(n_chunk),
            "chunks_done": int(n_chunk),
            "chunks_failed": 0,
            "pct_done": 100.0,
            "status": "completed",
        },
    )
    with run_log.open("a", encoding="utf-8") as lg:
        lg.write(f"[done] utc={_utc_now()} summary={tab_dir / 'summary.json'}\n")
    print(f"[done] wrote {tab_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
