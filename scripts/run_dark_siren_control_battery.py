from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _drop_flag(argv: list[str], flag: str, *, takes_value: bool) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    argv = _drop_flag(argv, flag, takes_value=True)
    return argv + [flag, str(value)]


def _set_bool(argv: list[str], flag: str, enabled: bool) -> list[str]:
    argv = _drop_flag(argv, flag, takes_value=False)
    if enabled:
        argv = argv + [flag]
    return argv


@dataclass(frozen=True)
class BatteryMode:
    name: str
    kind: str  # "catalog" | "hierarchical"
    # Overrides applied to the forwarded runner args.
    null_mode: str | None = None
    pe_like_mode: str | None = None
    pe_distance_mode: str | None = None
    galaxy_null_mode: str | None = None
    hier_null_mode: str | None = None
    selection_injections_hdf: str | None = None
    # For hierarchical runs, ensure incompatible knobs are disabled.
    force_no_mixture: bool = False
    force_no_completeness: bool = False


def _mode_args(base: list[str], mode: BatteryMode) -> list[str]:
    argv = list(base)

    # Always strip any existing --out; this script controls subdirs.
    argv = _drop_flag(argv, "--out", takes_value=True)

    # Battery assumes PE modes (rotate + spectral-only need PE posterior samples).
    argv = _set_flag(argv, "--gw-data-mode", "pe")

    if mode.pe_like_mode is not None:
        argv = _set_flag(argv, "--pe-like-mode", mode.pe_like_mode)
    if mode.pe_distance_mode is not None:
        argv = _set_flag(argv, "--pe-distance-mode", mode.pe_distance_mode)
    if mode.null_mode is not None:
        argv = _set_flag(argv, "--null-mode", mode.null_mode)
    if mode.galaxy_null_mode is not None:
        argv = _set_flag(argv, "--galaxy-null-mode", mode.galaxy_null_mode)
    if mode.hier_null_mode is not None:
        argv = _set_flag(argv, "--hier-null-mode", mode.hier_null_mode)
    if mode.selection_injections_hdf is not None:
        argv = _set_flag(argv, "--selection-injections-hdf", mode.selection_injections_hdf)

    if mode.force_no_mixture:
        argv = _set_flag(argv, "--mixture-mode", "none")
        argv = _set_flag(argv, "--mixture-f-miss-mode", "from_completeness")
        argv = _drop_flag(argv, "--mixture-f-miss", takes_value=True)
    if mode.force_no_completeness:
        argv = _set_flag(argv, "--completeness-mode", "none")

    # Hierarchical mode forbids these; keep them quiet by forcing safe values.
    if mode.kind == "hierarchical":
        argv = _set_flag(argv, "--null-mode", "none")
        argv = _set_flag(argv, "--galaxy-null-mode", "none")
        argv = _set_flag(argv, "--pe-distance-mode", "full")

    return argv


def _load_summaries(mode_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(mode_dir.glob("summary_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        run = str(d.get("run", p.stem.replace("summary_", "")))
        out[run] = d
    return out


def _extract_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    def _f(key: str) -> float:
        v = summary.get(key)
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _i(key: str) -> int:
        v = summary.get(key)
        try:
            return int(v)
        except Exception:
            return -1

    return {
        "n_events": _i("n_events"),
        "delta_lpd_total": _f("delta_lpd_total"),
        "delta_lpd_total_data": _f("delta_lpd_total_data"),
        "delta_lpd_total_sel": _f("delta_lpd_total_sel"),
    }


def _extract_selection_alpha(mode_dir: Path, *, run: str) -> dict[str, float]:
    """Load selection alpha arrays saved by the runner, returning small summary stats."""
    p = mode_dir / "tables" / f"selection_alpha_{run}.npz"
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as d:
            log_mu = np.asarray(d["log_alpha_mu"], dtype=float)
            log_gr = np.asarray(d["log_alpha_gr"], dtype=float)
    except Exception:
        return {}

    if log_mu.shape != log_gr.shape or log_mu.ndim != 1:
        return {}
    lr = log_mu - log_gr
    if lr.size == 0 or not np.any(np.isfinite(lr)):
        return {}
    lr = lr[np.isfinite(lr)]
    return {
        "log_alpha_ratio_p50": float(np.median(lr)),
        "log_alpha_ratio_p05": float(np.quantile(lr, 0.05)),
        "log_alpha_ratio_p95": float(np.quantile(lr, 0.95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small control battery for dark-siren scoring (catalog nulls + hierarchical spectral test).")
    ap.add_argument("--out", required=True, help="Output directory for the battery bundle.")
    ap.add_argument("--max-events", type=int, default=3, help="Max events for the battery (default 3).")
    ap.add_argument("--max-draws", type=int, default=8, help="Max posterior draws for the battery (default 8).")
    ap.add_argument("--n-proc", type=int, default=1, help="Per-run worker processes passed to the runner (default 1).")
    ap.add_argument("--pe-max-samples", type=int, default=20000, help="Cap PE samples per event (default 20000).")
    ap.add_argument(
        "--hier-null-battery",
        action="store_true",
        help="If hierarchical mode is enabled, also run hierarchical nulls (shuffle_dl, shuffle_mc, shuffle_dl_mc).",
    )
    ap.add_argument(
        "--skip-catalog",
        action="store_true",
        help="Skip catalog-based modes entirely (run only hierarchical PE modes).",
    )
    ap.add_argument(
        "--skip-hierarchical",
        action="store_true",
        help="Skip the hierarchical PE mode (only run catalog-mode nulls).",
    )
    ap.add_argument(
        "--hier-no-selection",
        action="store_true",
        help="Also run a hierarchical data-only mode (forces --selection-injections-hdf none).",
    )
    ap.add_argument(
        "--skip-rotate",
        action="store_true",
        help="Skip rotate_pe_sky null (useful if you want to avoid event-set changes from rotation).",
    )
    ap.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/run_dark_siren_gap_test.py (prefix with --).",
    )
    args = ap.parse_args()

    if not args.runner_args or args.runner_args[0] != "--":
        raise SystemExit("Pass runner args after a '--' separator, e.g. ...control_battery.py --out X -- --run-dir ...")
    forwarded = list(args.runner_args[1:])

    # Enforce small-battery limits (override any existing values).
    forwarded = _set_flag(forwarded, "--max-events", str(int(args.max_events)))
    forwarded = _set_flag(forwarded, "--max-draws", str(int(args.max_draws)))
    forwarded = _set_flag(forwarded, "--n-proc", str(int(args.n_proc)))
    forwarded = _set_flag(forwarded, "--pe-max-samples", str(int(args.pe_max_samples)))

    out_base = Path(str(args.out)).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    modes: list[BatteryMode] = []
    if not bool(args.skip_catalog):
        modes = [
            BatteryMode(
                name="catalog_full",
                kind="catalog",
                pe_like_mode="hist",
                pe_distance_mode="full",
                null_mode="none",
                galaxy_null_mode="none",
            ),
            BatteryMode(
                name="catalog_spectral_only",
                kind="catalog",
                pe_like_mode="hist",
                pe_distance_mode="spectral_only",
                null_mode="none",
                galaxy_null_mode="none",
            ),
            BatteryMode(
                name="catalog_shuffle_z",
                kind="catalog",
                pe_like_mode="hist",
                pe_distance_mode="full",
                null_mode="none",
                galaxy_null_mode="shuffle_z",
            ),
            BatteryMode(
                name="catalog_shuffle_zw",
                kind="catalog",
                pe_like_mode="hist",
                pe_distance_mode="full",
                null_mode="none",
                galaxy_null_mode="shuffle_zw",
            ),
        ]
        if not bool(args.skip_rotate):
            modes.insert(
                1,
                BatteryMode(
                    name="catalog_rotate",
                    kind="catalog",
                    pe_like_mode="hist",
                    pe_distance_mode="full",
                    null_mode="rotate_pe_sky",
                    galaxy_null_mode="none",
                ),
            )
    if not bool(args.skip_hierarchical):
        hier_modes = ["none"]
        if bool(args.hier_null_battery):
            hier_modes = ["none", "shuffle_dl", "shuffle_mc", "shuffle_dl_mc"]
        for hm in hier_modes:
            suffix = "hierarchical" if hm == "none" else f"hier_{hm}"
            modes.append(
                BatteryMode(
                    name=suffix,
                    kind="hierarchical",
                    pe_like_mode="hierarchical",
                    null_mode="none",
                    galaxy_null_mode="none",
                    pe_distance_mode="full",
                    hier_null_mode=str(hm),
                    force_no_mixture=True,
                    force_no_completeness=True,
                )
            )
        if bool(args.hier_no_selection):
            modes.append(
                BatteryMode(
                    name="hierarchical_noselection",
                    kind="hierarchical",
                    pe_like_mode="hierarchical",
                    null_mode="none",
                    galaxy_null_mode="none",
                    pe_distance_mode="full",
                    hier_null_mode="none",
                    selection_injections_hdf="none",
                    force_no_mixture=True,
                    force_no_completeness=True,
                )
            )

    env = dict(os.environ)
    # Ensure the runner can import src/ without requiring the user to export PYTHONPATH.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(Path("src").resolve()) + (os.pathsep + pp if pp else "")

    runner = [sys.executable, str(Path("scripts/run_dark_siren_gap_test.py").resolve())]

    commands: dict[str, list[str]] = {}
    summaries_by_mode: dict[str, dict[str, dict[str, Any]]] = {}

    for m in modes:
        mode_dir = out_base / m.name
        mode_dir.mkdir(parents=True, exist_ok=True)
        argv = _mode_args(forwarded, m) + ["--out", str(mode_dir)]
        cmd = runner + argv
        commands[m.name] = cmd
        print(f"[battery] running {m.name} -> {mode_dir}", flush=True)
        res = subprocess.run(cmd, check=False, env=env)
        if res.returncode != 0:
            print(f"[battery] WARNING: mode {m.name} failed with code {res.returncode}", flush=True)
        summaries_by_mode[m.name] = _load_summaries(mode_dir)

    # Combine metrics.
    runs: set[str] = set()
    for d in summaries_by_mode.values():
        runs |= set(d.keys())
    runs = set(sorted(runs))

    combined: dict[str, Any] = {
        "timestamp_utc": _utc_stamp(),
        "out": str(out_base),
        "runner": str(runner[1]),
        "modes": [m.name for m in modes],
        "skip_catalog": bool(args.skip_catalog),
        "max_events": int(args.max_events),
        "max_draws": int(args.max_draws),
        "pe_max_samples": int(args.pe_max_samples),
        "commands": {k: v for k, v in commands.items()},
        "runs": {},
    }

    for run in sorted(runs):
        combined["runs"][run] = {}
        for m in modes:
            sm = summaries_by_mode.get(m.name, {}).get(run)
            if sm is None:
                continue
            mode_dir = out_base / m.name
            combined["runs"][run][m.name] = {
                "summary_path": str((mode_dir / f"summary_{run}.json").resolve()),
                "metrics": _extract_metrics(sm),
                "selection_alpha": _extract_selection_alpha(mode_dir, run=run),
            }

    _write_json(out_base / "battery_summary.json", combined)

    # Plot: stacked bars (data + selection) per mode, averaged across runs.
    mode_names = [m.name for m in modes]
    data_means: list[float] = []
    sel_means: list[float] = []
    tot_means: list[float] = []
    tot_stds: list[float] = []

    for mn in mode_names:
        vals_data = []
        vals_sel = []
        vals_tot = []
        for run in sorted(runs):
            mm = combined["runs"].get(run, {}).get(mn, {}).get("metrics")
            if not mm:
                continue
            vals_data.append(float(mm.get("delta_lpd_total_data", float("nan"))))
            vals_sel.append(float(mm.get("delta_lpd_total_sel", float("nan"))))
            vals_tot.append(float(mm.get("delta_lpd_total", float("nan"))))
        vals_data = [v for v in vals_data if np.isfinite(v)]
        vals_sel = [v for v in vals_sel if np.isfinite(v)]
        vals_tot = [v for v in vals_tot if np.isfinite(v)]

        data_means.append(float(np.mean(vals_data)) if vals_data else float("nan"))
        sel_means.append(float(np.mean(vals_sel)) if vals_sel else float("nan"))
        tot_means.append(float(np.mean(vals_tot)) if vals_tot else float("nan"))
        tot_stds.append(float(np.std(vals_tot, ddof=1)) if len(vals_tot) >= 2 else 0.0)

    x = np.arange(len(mode_names))
    plt.figure(figsize=(10, 4.5))
    plt.bar(x, data_means, color="C0", label="ΔLPD_data")
    plt.bar(x, sel_means, bottom=data_means, color="C1", label="ΔLPD_sel")
    plt.errorbar(x, tot_means, yerr=tot_stds, fmt="none", ecolor="k", elinewidth=1.0, capsize=3, alpha=0.7)
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    plt.xticks(x, mode_names, rotation=30, ha="right")
    plt.ylabel("ΔLPD (mean across run-dir seeds)")
    plt.title("Dark-siren control battery (totals; stacked data/selection)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_base / "battery_totals.png", dpi=160)
    plt.close()

    print(f"[battery] wrote {out_base/'battery_summary.json'}", flush=True)
    print(f"[battery] wrote {out_base/'battery_totals.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
