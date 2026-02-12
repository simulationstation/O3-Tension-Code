#!/usr/bin/env python3
"""
Regenerate the Dark-Siren paper figures that annotate the observed Delta LPD_tot.

Motivation:
  The MNRAS letter text uses the updated observed value Delta LPD_tot = +3.670, but
  some legacy plots still marked +3.03. This script regenerates those plots
  from stored pipeline outputs, ensuring the observed-line position/label is
  consistent everywhere.

This script is intentionally lightweight (json + numpy + matplotlib only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_injection_deltas(*, inj_dir: Path | None, inj_deltas_json: Path | None) -> np.ndarray:
    if inj_deltas_json is not None:
        d = _read_json(inj_deltas_json)
        rows = d.get("rows", [])
        if not rows:
            raise ValueError(f"No rows in injection deltas json: {inj_deltas_json}")
        vals = [float(r["delta_lpd_total"]) for r in rows]
        return np.asarray(vals, dtype=float)

    if inj_dir is not None:
        reps = sorted((inj_dir / "reps").glob("rep*.json"))
        if not reps:
            raise FileNotFoundError(f"No rep json files found under: {inj_dir/'reps'}")
        vals = []
        for p in reps:
            d = _read_json(p)
            vals.append(float(d["delta_lpd_total"]))
        return np.asarray(vals, dtype=float)

    raise ValueError("Provide either inj_dir or inj_deltas_json.")


def make_fig_delta_lpd_total_hist(
    *,
    inj_dir: Path | None,
    inj_deltas_json: Path | None,
    observed: float,
    out_png: Path,
) -> None:
    deltas = _load_injection_deltas(inj_dir=inj_dir, inj_deltas_json=inj_deltas_json)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        raise ValueError("No finite injection deltas loaded.")

    # Match existing raster size ~1120x640 at dpi=160.
    plt.figure(figsize=(7.0, 4.0))
    bins = 40
    plt.hist(deltas, bins=bins, color="C0", alpha=0.85, edgecolor="none", label="GR-truth injections (N=512)")
    plt.axvline(0.0, color="k", lw=1.0, alpha=0.7, label=r"$\Delta\mathrm{LPD}=0$")
    plt.axvline(
        observed,
        color="tab:orange",
        lw=1.6,
        ls="--",
        alpha=0.95,
        label=rf"observed $\Delta\mathrm{{LPD}}_{{\rm tot}}=+{observed:.3f}$",
    )

    mu = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1)) if deltas.size >= 2 else 0.0
    mx = float(np.max(deltas))
    plt.title(f"GR-consistent injection calibration (N={deltas.size})\nmean={mu:+.3f}, sd={sd:.3f}, max={mx:+.3f}")
    plt.xlabel(r"$\Delta\mathrm{LPD}_{\rm tot}$")
    plt.ylabel("count")

    # Reviewer-friendly framing: leave room for the observed line on the right.
    plt.xlim(-1.5, 4.5)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_fig_fixed_power_grid(
    *,
    grid_summary_json: Path,
    observed: float,
    out_png: Path,
) -> None:
    d = _read_json(grid_summary_json)
    rows = d.get("rows", [])
    if not rows:
        raise ValueError(f"No rows in {grid_summary_json}")

    rows = sorted(rows, key=lambda r: float(r.get("scale", 0.0)))
    scale = np.asarray([float(r["scale"]) for r in rows], dtype=float)
    mean_total = np.asarray([float(r["mean_total"]) for r in rows], dtype=float)
    sd_total = np.asarray([float(r["sd_total"]) for r in rows], dtype=float)
    max_total = np.asarray([float(r["max_total"]) for r in rows], dtype=float)

    # Match existing raster size ~1480x919 at dpi=160.
    plt.figure(figsize=(9.25, 5.75))
    plt.errorbar(scale, mean_total, yerr=sd_total, fmt="o-", color="C0", lw=1.6, ms=5, label="mean +/- 1 sigma")
    plt.plot(scale, max_total, "s", color="k", ms=5, label="max (per scale)")
    plt.axhline(
        observed,
        color="tab:red",
        lw=1.6,
        ls="--",
        alpha=0.9,
        label=rf"observed $\Delta\mathrm{{LPD}}_{{\rm tot}}=+{observed:.3f}$",
    )
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.4)
    plt.xlabel("injected log-$R$ scale")
    plt.ylabel(r"$\Delta\mathrm{LPD}_{\rm tot}$")
    plt.title("Fixed-power injection grid (GR-consistent null; 256 reps/scale)")
    plt.grid(alpha=0.2)
    plt.ylim(-1.0, 4.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_fig_systematics_matrix(
    *,
    matrix_summary_json: Path,
    observed: float,
    out_png: Path,
) -> None:
    d = _read_json(matrix_summary_json)
    rows = d.get("rows", [])
    if not rows:
        raise ValueError(f"No rows in {matrix_summary_json}")

    # Preserve a sensible ordering: baseline first, then the remainder alphabetical.
    def _key(r: dict) -> tuple[int, str]:
        name = str(r.get("name", ""))
        return (0 if name == "baseline" else 1, name)

    rows = sorted(rows, key=_key)
    names = [str(r["name"]) for r in rows]
    mean_total = np.asarray([float(r["mean_total"]) for r in rows], dtype=float)
    sd_total = np.asarray([float(r["sd_total"]) for r in rows], dtype=float)
    max_total = np.asarray([float(r["max_total"]) for r in rows], dtype=float)

    x = np.arange(len(names))

    # Match existing raster size ~1839x960 at dpi=160.
    plt.figure(figsize=(11.5, 6.0))
    plt.bar(x, mean_total, color="C0", alpha=0.85, label="mean")
    plt.errorbar(x, mean_total, yerr=sd_total, fmt="none", ecolor="k", elinewidth=1.0, capsize=3, alpha=0.7, label="+/- 1 sigma")
    plt.plot(x, max_total, "D", color="k", ms=4, label="max")
    plt.axhline(
        observed,
        color="tab:red",
        lw=1.6,
        ls="--",
        alpha=0.9,
        label=rf"observed $\Delta\mathrm{{LPD}}_{{\rm tot}}=+{observed:.3f}$",
    )
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.4)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel(r"$\Delta\mathrm{LPD}_{\rm tot}$")
    plt.title("GR-consistent systematics matrix (128 reps/variant)")
    plt.grid(axis="y", alpha=0.2)
    plt.ylim(-1.0, 4.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate MNRAS dark-siren figures with consistent observed Delta LPD_tot annotation.")
    ap.add_argument(
        "--observed",
        type=float,
        default=None,
        help="Observed real-data Delta LPD_tot to annotate (default: read from artifacts/o3/summary_M0_start101.json).",
    )
    ap.add_argument(
        "--inj-deltas-json",
        type=Path,
        default=None,
        help="Path to a compact injection delta dump json (preferred over scanning rep*.json).",
    )
    ap.add_argument(
        "--inj-dir",
        type=Path,
        default=None,
        help="Injection calibration directory containing reps/rep*.json (optional if --inj-deltas-json is provided).",
    )
    ap.add_argument(
        "--power-grid-summary",
        type=Path,
        default=None,
        help="Power-grid grid_summary.json path.",
    )
    ap.add_argument(
        "--systematics-summary",
        type=Path,
        default=None,
        help="Systematics matrix_summary.json path.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory for regenerated PNG figures (default: paper_gen/mrnas/figures).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    out_dir: Path = args.out_dir
    observed: float
    if args.observed is None:
        observed = float(_read_json(repo_root / "artifacts/o3/summary_M0_start101.json")["delta_lpd_total"])
    else:
        observed = float(args.observed)

    inj_deltas_json: Path | None = args.inj_deltas_json
    if inj_deltas_json is None:
        cand = repo_root / "artifacts/o3/catalog_injection_deltas_n512.json"
        inj_deltas_json = cand if cand.exists() else None

    power_grid_summary = (repo_root / "artifacts/o3/fixed_power_grid_summary.json") if args.power_grid_summary is None else args.power_grid_summary
    systematics_summary = (repo_root / "artifacts/o3/systematics_matrix_summary.json") if args.systematics_summary is None else args.systematics_summary

    make_fig_delta_lpd_total_hist(
        inj_dir=args.inj_dir,
        inj_deltas_json=inj_deltas_json,
        observed=observed,
        out_png=out_dir / "fig_delta_lpd_total_hist.png",
    )
    make_fig_fixed_power_grid(grid_summary_json=power_grid_summary, observed=observed, out_png=out_dir / "fig_fixed_power_grid.png")
    make_fig_systematics_matrix(matrix_summary_json=systematics_summary, observed=observed, out_png=out_dir / "fig_systematics_matrix.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
