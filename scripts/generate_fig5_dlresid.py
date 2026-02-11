#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C_KM_S = 299792.458


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    if cdf[-1] <= 0:
        return float(np.nan)
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, xs))


def _pdf_quantile(edges: np.ndarray, pdf: np.ndarray, q: float) -> float:
    widths = np.diff(edges)
    p = np.clip(pdf, 0.0, None) * widths
    s = float(np.sum(p))
    if s <= 0:
        return float(np.nan)
    p = p / s
    cdf = np.cumsum(p)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return float(np.interp(q, cdf, mids))


def _comoving_distance_grid(z: np.ndarray, h: np.ndarray) -> np.ndarray:
    invh = C_KM_S / np.clip(h, 1e-12, None)
    dz = np.diff(z)
    trap = 0.5 * (invh[1:] + invh[:-1]) * dz
    dc = np.concatenate(([0.0], np.cumsum(trap)))
    return dc


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent

    run_root = project_root / "outputs" / "dark_siren_gap_pe_scaleup36max_20260201_155611UTC"
    event_scores_path = run_root / "tables" / "event_scores_M0_start101.json"
    cache_dir = run_root / "cache"
    posterior_path = project_root / "outputs" / "finalization" / "highpower_multistart_v2" / "M0_start101" / "samples" / "mu_forward_posterior.npz"

    out_fig = repo_root / "papers" / "dark_siren" / "figures" / "fig_dlresid_o3.png"
    out_json = repo_root / "artifacts" / "o3" / "fig5_dlresid_data.json"

    if not event_scores_path.exists():
        raise FileNotFoundError(f"Missing event scores: {event_scores_path}")
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior file: {posterior_path}")

    scores = json.loads(event_scores_path.read_text())
    post = np.load(posterior_path)

    x_grid = np.asarray(post["x_grid"], dtype=float)
    logmu_x = np.asarray(post["logmu_x_samples"], dtype=float)
    z_grid = np.asarray(post["z_grid"], dtype=float)
    h_samps = np.asarray(post["H_samples"], dtype=float)

    # Model band: Delta mu = 5 log10(dL_GW / dL_EM)
    n_draw = h_samps.shape[0]
    n_z = z_grid.size
    dmu_draw = np.empty((n_draw, n_z), dtype=float)

    for j in range(n_draw):
        h = h_samps[j]
        denom = np.clip(h**2, 1e-24, None)
        area = 4.0 * np.pi * (C_KM_S**2) / denom
        area0 = float(area[0])
        xz = np.log(np.clip(area / area0, 1e-24, None))

        logmu_z = np.interp(xz, x_grid, logmu_x[j], left=logmu_x[j, 0], right=logmu_x[j, -1])
        logmu0 = float(np.interp(0.0, x_grid, logmu_x[j], left=logmu_x[j, 0], right=logmu_x[j, -1]))
        ratio = np.exp(0.5 * (logmu_z - logmu0))
        dmu_draw[j] = 5.0 * np.log10(np.clip(ratio, 1e-12, None))

    dmu_med = np.percentile(dmu_draw, 50, axis=0)
    dmu_lo = np.percentile(dmu_draw, 16, axis=0)
    dmu_hi = np.percentile(dmu_draw, 84, axis=0)

    # EM distance baseline from same posterior background
    dlem_draw = np.empty((n_draw, n_z), dtype=float)
    for j in range(n_draw):
        dc = _comoving_distance_grid(z_grid, h_samps[j])
        dlem_draw[j] = (1.0 + z_grid) * dc

    # Event-level points from cached per-event PE distance PDFs + host-z weights
    ev_z = []
    ev_dmu = []
    ev_dmu_err = []
    ev_names = []

    for row in scores:
        ev = str(row["event"])
        p = cache_dir / f"event_{ev}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)

        z = np.asarray(d["z"], dtype=float)
        w = np.asarray(d["w"], dtype=float)
        if z.size == 0 or w.size == 0:
            continue
        w = np.clip(w, 0.0, None)
        if float(np.sum(w)) <= 0:
            continue

        z_eff = _weighted_quantile(z, w, 0.5)

        pe_prob = np.asarray(d["pe_prob_pix"], dtype=float)
        pe_bins = np.asarray(d["pe_pdf_bins"], dtype=float)
        dl_edges = np.asarray(d["pe_dL_edges"], dtype=float)
        if pe_prob.size == 0 or pe_bins.size == 0:
            continue

        pw = np.clip(pe_prob, 0.0, None)
        if float(np.sum(pw)) <= 0:
            continue
        pw = pw / float(np.sum(pw))
        pdf = np.sum(pe_bins * pw[:, None], axis=0)

        dl16 = _pdf_quantile(dl_edges, pdf, 0.16)
        dl50 = _pdf_quantile(dl_edges, pdf, 0.50)
        dl84 = _pdf_quantile(dl_edges, pdf, 0.84)
        if not (np.isfinite(dl16) and np.isfinite(dl50) and np.isfinite(dl84)):
            continue

        mu_obs = 5.0 * math.log10(max(dl50, 1e-12)) + 25.0
        mu_obs_lo = mu_obs - (5.0 * math.log10(max(dl16, 1e-12)) + 25.0)
        mu_obs_hi = (5.0 * math.log10(max(dl84, 1e-12)) + 25.0) - mu_obs
        sig_obs = 0.5 * (abs(mu_obs_lo) + abs(mu_obs_hi))

        mu_em_draw = np.empty(n_draw, dtype=float)
        for j in range(n_draw):
            dlm = float(np.interp(z_eff, z_grid, dlem_draw[j]))
            mu_em_draw[j] = 5.0 * math.log10(max(dlm, 1e-12)) + 25.0
        mu_em = float(np.median(mu_em_draw))

        ev_names.append(ev)
        ev_z.append(float(z_eff))
        ev_dmu.append(float(mu_obs - mu_em))
        ev_dmu_err.append(float(max(sig_obs, 1e-6)))

    ev_z = np.asarray(ev_z, dtype=float)
    ev_dmu = np.asarray(ev_dmu, dtype=float)
    ev_dmu_err = np.asarray(ev_dmu_err, dtype=float)

    # Plot
    plt.figure(figsize=(8.0, 4.8))
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="GR baseline")
    plt.fill_between(z_grid, dmu_lo, dmu_hi, color="#9ecae1", alpha=0.45, label="Modified-propagation 68% band")
    plt.plot(z_grid, dmu_med, color="#1f77b4", linewidth=2.0, label="Modified-propagation median")
    if ev_z.size > 0:
        plt.errorbar(ev_z, ev_dmu, yerr=ev_dmu_err, fmt="o", ms=4.0, color="#d62728", ecolor="#d62728", alpha=0.85, capsize=2, label="Dark siren effective points")
    plt.xlim(0.0, float(np.max(z_grid)))
    ypad = max(0.03, float(np.nanpercentile(np.abs(dmu_hi), 95)) * 1.35)
    plt.ylim(float(np.nanmin([np.min(dmu_lo), np.min(ev_dmu - ev_dmu_err) if ev_z.size else 0.0])) - 0.02, ypad)
    plt.xlabel("Redshift z")
    plt.ylabel(r"Distance-modulus residual $\Delta\mu$ (mag)")
    plt.title("GWTC-3 O3: reconstructed propagation residual")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=180)
    plt.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {
            "event_scores": str(event_scores_path),
            "event_cache_dir": str(cache_dir),
            "posterior": str(posterior_path),
        },
        "model_band": {
            "z": z_grid.tolist(),
            "dmu_median": dmu_med.tolist(),
            "dmu_p16": dmu_lo.tolist(),
            "dmu_p84": dmu_hi.tolist(),
        },
        "event_points": [
            {"event": e, "z_eff": float(zv), "dmu_obs_minus_em": float(y), "sigma_mu": float(s)}
            for e, zv, y, s in zip(ev_names, ev_z, ev_dmu, ev_dmu_err, strict=False)
        ],
        "notes": "Event points use effective z from weighted host catalog z and PE distance PDF medians from cached event products.",
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"wrote figure: {out_fig}")
    print(f"wrote data:   {out_json}")
    print(f"n_events_plotted: {len(ev_names)}")


if __name__ == "__main__":
    main()
