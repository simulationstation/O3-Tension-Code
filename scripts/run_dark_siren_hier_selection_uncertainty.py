#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _summary_complete(path: Path, n_rep: int) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return int(d.get("n_rep_done", 0)) >= int(n_rep)


def _tail_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return str(lines[-1])[:500] if lines else ""


def _logsumexp(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    m = float(np.max(v))
    return float(m + np.log(np.sum(np.exp(v - m))))


def _logsumexp_rows(m: np.ndarray) -> np.ndarray:
    x = np.asarray(m, dtype=float)
    mx = np.max(x, axis=1, keepdims=True)
    return np.squeeze(mx + np.log(np.sum(np.exp(x - mx), axis=1, keepdims=True)), axis=1)


def _summ(values: list[float]) -> dict[str, float]:
    xs = np.asarray(values, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {"n": 0.0}
    return {
        "n": float(xs.size),
        "mean": float(np.mean(xs)),
        "sd": float(np.std(xs, ddof=0)),
        "min": float(np.min(xs)),
        "p16": float(np.percentile(xs, 16)),
        "p50": float(np.percentile(xs, 50)),
        "p84": float(np.percentile(xs, 84)),
        "max": float(np.max(xs)),
        "p_ge_0": float(np.mean(xs >= 0.0)),
        "p_ge_3": float(np.mean(xs >= 3.0)),
    }


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float
    description: str
    extra_args: list[str]


def _default_variants() -> list[Variant]:
    return [
        Variant("baseline", 0.40, "Matched selection baseline", []),
        Variant("selection_threshold", 0.20, "Threshold det model", ["--selection-det-model", "threshold"]),
        Variant("score_fmiss_fixed_low", 0.15, "Fixed low f_miss in scoring", ["--f-miss-mode", "fixed", "--f-miss-fixed", "0.580795"]),
        Variant("score_fmiss_fixed_high", 0.15, "Fixed high f_miss in scoring", ["--f-miss-mode", "fixed", "--f-miss-fixed", "0.780795"]),
        Variant("selection_weight_none", 0.10, "Selection weights disabled", ["--selection-weight-mode", "none"]),
    ]


def _load_variants(path: str | None) -> list[Variant]:
    if path is None:
        return _default_variants()
    p = Path(path).expanduser().resolve()
    raw = json.loads(p.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("--variants-json must contain a non-empty list.")
    out: list[Variant] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ValueError(f"variant entry {i} must be object")
        name = str(r.get("name", "")).strip()
        if not name:
            raise ValueError(f"variant entry {i} missing name")
        w = float(r.get("weight", 0.0))
        if not (np.isfinite(w) and w > 0.0):
            raise ValueError(f"variant {name} has invalid positive weight")
        desc = str(r.get("description", ""))
        extra = [str(x) for x in r.get("extra_args", [])]
        out.append(Variant(name=name, weight=w, description=desc, extra_args=extra))
    return out


def _read_rep_row(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    need = ("lpd_mu_total", "lpd_gr_total", "lpd_mu_total_data", "lpd_gr_total_data")
    if any(k not in d for k in need):
        return None
    return d


def _read_real_summary(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    need = ("lpd_mu_total", "lpd_gr_total", "lpd_mu_total_data", "lpd_gr_total_data")
    if any(k not in d for k in need):
        return None
    return d


def _find_real_summary(out_dir: Path, run_tag: str) -> Path | None:
    cand = out_dir / f"summary_{run_tag}.json"
    if cand.exists():
        return cand
    paths = sorted(out_dir.glob("summary_*.json"))
    for p in paths:
        if _read_real_summary(p) is not None:
            return p
    return None


def _reuse_realdata_caches(src_real_out: Path, dst_real_out: Path) -> None:
    """
    Reuse heavy per-event caches across variant real-data runs.
    This is safe because cache payloads are event/run dependent, not variant-name dependent.
    """
    for name in ("cache", "cache_terms", "cache_missing", "cache_gr_h0"):
        src = src_real_out / name
        dst = dst_real_out / name
        if not src.exists() or dst.exists():
            continue
        try:
            dst.symlink_to(src, target_is_directory=True)
        except Exception:
            shutil.copytree(src, dst)


def _delta_from_weights(
    *,
    logw: np.ndarray,
    lpd_mu: np.ndarray,
    lpd_gr: np.ndarray,
) -> float:
    return float(_logsumexp(logw + lpd_mu) - _logsumexp(logw + lpd_gr))


def _translate_variant_args_for_real(extra_args: list[str]) -> list[str]:
    mapped: list[str] = []
    map_key = {
        "--f-miss-mode": "--mixture-f-miss-mode",
        "--f-miss-fixed": "--mixture-f-miss",
    }
    i = 0
    while i < len(extra_args):
        tok = str(extra_args[i])
        if tok in map_key:
            if i + 1 >= len(extra_args):
                raise ValueError(f"Variant arg {tok} requires a value.")
            mapped.extend([map_key[tok], str(extra_args[i + 1])])
            i += 2
            continue
        if i + 1 < len(extra_args) and not str(extra_args[i + 1]).startswith("--"):
            mapped.extend([tok, str(extra_args[i + 1])])
            i += 2
        else:
            mapped.append(tok)
            i += 1
    return mapped


def _selection_overrides_from_extra_args(extra_args: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(extra_args):
        tok = str(extra_args[i])
        nxt = str(extra_args[i + 1]) if (i + 1) < len(extra_args) else None
        if tok.startswith("--selection-"):
            if nxt is not None and not nxt.startswith("--"):
                out[tok] = nxt
                i += 2
            else:
                out[tok] = True
                i += 1
            continue
        if nxt is not None and not nxt.startswith("--"):
            i += 2
        else:
            i += 1
    return out


def _selection_alpha_cache_path(
    *,
    out_root: Path,
    run_dir: str,
    base_event_cache: str,
    max_events: int,
    max_draws: int,
    seed: int,
    variant_extra_args: list[str],
) -> Path:
    payload = {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "base_event_cache": str(Path(base_event_cache).expanduser().resolve()),
        "max_events": int(max_events),
        "max_draws": int(max_draws),
        "seed": int(seed),
        # Defaults from run_dark_siren_catalog_injection_suite.py selection args:
        "selection_defaults": {
            "--selection-ifar-thresh-yr": "1.0",
            "--selection-det-model": "snr_binned",
            "--selection-snr-binned-nbins": "200",
            "--selection-weight-mode": "inv_sampling_pdf",
            "--selection-z-max": "0.3",
            "--selection-injections-hdf": "auto",
            "--selection-injections-population": "mixture",
            "--selection-snr-offset": "0.0",
            "--selection-mu-det-distance": "gw",
            "--selection-pop-z-mode": "none",
            "--selection-pop-z-k": "0.0",
            "--selection-pop-mass-mode": "none",
            "--selection-pop-m1-alpha": "2.3",
            "--selection-pop-m-min": "5.0",
            "--selection-pop-m-max": "80.0",
            "--selection-pop-q-beta": "0.0",
            "--selection-pop-m-taper-delta": "0.0",
            "--selection-pop-m-peak": "35.0",
            "--selection-pop-m-peak-sigma": "5.0",
            "--selection-pop-m-peak-frac": "0.1",
        },
        "selection_overrides": _selection_overrides_from_extra_args(variant_extra_args),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return out_root / "_shared_cache" / "selection_alpha" / f"alpha_{digest}.npz"


def _synth_stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0}
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=0)),
        "p16": float(np.percentile(x, 16)),
        "p50": float(np.percentile(x, 50)),
        "p84": float(np.percentile(x, 84)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hierarchical selection-uncertainty integration for GR-truth injections.")
    ap.add_argument("--out-root", required=True, type=str)
    ap.add_argument("--run-dir", default="outputs/finalization/highpower_multistart_v2/M0_start101", type=str)
    ap.add_argument(
        "--base-event-cache",
        default="outputs/dark_siren_gap_pe_scaleup36max_20260201_155611UTC/cache",
        type=str,
        help="Base cache of event_*.npz files used by injection stage.",
    )
    ap.add_argument("--variants-json", default=None, type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--n-rep", default=24, type=int)
    ap.add_argument("--n-proc", default=0, type=int)
    ap.add_argument("--max-events", default=36, type=int)
    ap.add_argument("--max-draws", default=256, type=int)
    ap.add_argument(
        "--inj-spectral-precompute-dl",
        default="on",
        choices=["auto", "on", "off"],
        help="Forwarded to injection suite for spectral dL precompute (default: on).",
    )
    ap.add_argument(
        "--inj-spectral-precompute-dl-max-gb",
        default=96.0,
        type=float,
        help="Forwarded to injection suite precompute memory cap (default: 96 GB).",
    )
    ap.add_argument(
        "--inj-parallel-backend",
        default="auto",
        choices=["auto", "process", "thread", "sequential"],
        help="Forwarded to injection suite --parallel-backend (default: auto).",
    )
    ap.add_argument("--heartbeat-sec", default=60.0, type=float)
    ap.add_argument("--partial-write-min-sec", default=20.0, type=float)
    ap.add_argument("--skip-realdata", action="store_true", help="Skip real-data variant scoring stage.")
    ap.add_argument(
        "--real-skymap-dir",
        default="data/cache/gw/zenodo/5546663/extracted/skymaps",
        type=str,
        help="Skymap directory for real-data variant scoring.",
    )
    ap.add_argument(
        "--real-glade-index",
        default="data/processed/galaxies/gladeplus/index_nside128_wlumB_zmax0.3",
        type=str,
        help="GLADE index for real-data variant scoring.",
    )
    ap.add_argument("--real-gw-data-mode", default="pe", choices=["skymap", "pe"], type=str)
    ap.add_argument("--real-pe-base-dir", default="data/cache/gw/zenodo", type=str)
    ap.add_argument(
        "--real-pe-record-ids",
        default="5546663",
        type=str,
        help="Comma list of PE record ids for real-data stage.",
    )
    ap.add_argument("--real-max-area-deg2", default=20000.0, type=float)
    ap.add_argument("--real-max-gal", default=8000000, type=int)
    ap.add_argument(
        "--real-n-proc",
        default=1,
        type=int,
        help="Worker processes for real-data gap scoring stage (default 1 for robust semaphore-safe operation).",
    )
    ap.add_argument("--real-selection-injections-hdf", default="auto", type=str)
    ap.add_argument("--real-selection-det-model", default="snr_binned", choices=["threshold", "snr_binned"], type=str)
    ap.add_argument("--real-selection-weight-mode", default="inv_sampling_pdf", choices=["none", "inv_sampling_pdf"], type=str)
    ap.add_argument("--weight-kappa", default=200.0, type=float, help="Dirichlet concentration for variant-weight uncertainty.")
    ap.add_argument("--weight-samples", default=2000, type=int, help="Weight posterior samples for robust-interval integration.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_log = out_root / "run.log"
    status_path = out_root / "status.json"
    manifest_path = out_root / "manifest.json"
    results_path = out_root / "hierarchical_summary.json"
    table_path = out_root / "hierarchical_reps.csv"

    variants = _load_variants(args.variants_json)
    w = np.asarray([v.weight for v in variants], dtype=float)
    w = w / np.sum(w)
    logw = np.log(np.clip(w, 1e-300, np.inf))
    n_proc = int(args.n_proc) if int(args.n_proc) > 0 else int(os.cpu_count() or 1)

    _write_json(
        manifest_path,
        {
            "created_utc": _now_utc(),
            "argv": sys.argv,
            "run_dir": str(args.run_dir),
            "base_event_cache": str(args.base_event_cache),
            "n_rep": int(args.n_rep),
            "n_proc": int(n_proc),
            "max_events": int(args.max_events),
            "max_draws": int(args.max_draws),
            "inj_parallel_backend": str(args.inj_parallel_backend),
            "variants": [asdict(v) for v in variants],
            "weights_normalized": [float(x) for x in w],
            "skip_realdata": bool(args.skip_realdata),
            "real_skymap_dir": str(args.real_skymap_dir),
            "real_glade_index": str(args.real_glade_index),
            "real_gw_data_mode": str(args.real_gw_data_mode),
            "real_pe_base_dir": str(args.real_pe_base_dir),
            "real_pe_record_ids": [x.strip() for x in str(args.real_pe_record_ids).split(",") if x.strip()],
            "real_selection_injections_hdf": str(args.real_selection_injections_hdf),
            "real_selection_det_model": str(args.real_selection_det_model),
            "real_selection_weight_mode": str(args.real_selection_weight_mode),
            "real_n_proc": int(args.real_n_proc),
            "weight_kappa": float(args.weight_kappa),
            "weight_samples": int(args.weight_samples),
        },
    )
    with run_log.open("a", encoding="utf-8") as lf:
        lf.write(f"[hier] start {_now_utc()} variants={len(variants)} n_rep={int(args.n_rep)} n_proc={n_proc}\n")

    variant_dirs: dict[str, Path] = {}
    real_rows: dict[str, dict[str, Any]] = {}
    run_tag = Path(args.run_dir).name
    for vi, v in enumerate(variants):
        vdir = out_root / "variants" / v.name
        vdir.mkdir(parents=True, exist_ok=True)
        variant_dirs[v.name] = vdir
        summary_path = vdir / "summary.json"
        alpha_cache_path = _selection_alpha_cache_path(
            out_root=out_root,
            run_dir=str(args.run_dir),
            base_event_cache=str(args.base_event_cache),
            max_events=int(args.max_events),
            max_draws=int(args.max_draws),
            seed=int(args.seed),
            variant_extra_args=list(v.extra_args),
        )
        cmd = [
            sys.executable,
            "scripts/run_dark_siren_catalog_injection_suite.py",
            "--out",
            str(vdir),
            "--run-dir",
            str(args.run_dir),
            "--base-event-cache",
            str(args.base_event_cache),
            "--truth-model",
            "gr",
            "--inj-host-mode",
            "mixture",
            "--inj-f-miss-mode",
            "fixed",
            "--inj-f-miss-fixed",
            "0.6807953774124085",
            "--inj-logR-scale",
            "0.0",
            "--f-miss-mode",
            "marginalize",
            "--f-miss-beta-mean",
            "0.6807953774124085",
            "--f-miss-beta-kappa",
            "8.0",
            "--seed",
            str(int(args.seed)),
            "--n-rep",
            str(int(args.n_rep)),
            "--n-proc",
            str(int(n_proc)),
            "--parallel-backend",
            str(args.inj_parallel_backend),
            "--max-events",
            str(int(args.max_events)),
            "--max-draws",
            str(int(args.max_draws)),
            "--selection-alpha-cache",
            str(alpha_cache_path),
            "--spectral-precompute-dl",
            str(args.inj_spectral_precompute_dl),
            "--spectral-precompute-dl-max-gb",
            str(float(args.inj_spectral_precompute_dl_max_gb)),
            "--heartbeat-sec",
            str(float(args.heartbeat_sec)),
            "--partial-write-min-sec",
            str(float(args.partial_write_min_sec)),
            *v.extra_args,
        ]
        inj_done = _summary_complete(summary_path, int(args.n_rep))
        if inj_done:
            with run_log.open("a", encoding="utf-8") as lf:
                lf.write(f"[hier] skip variant={v.name}: summary complete\n")
        else:
            with run_log.open("a", encoding="utf-8") as lf:
                lf.write(f"[hier] launch variant={v.name} cmd={shlex.join(cmd)}\n")
            variant_log = (vdir / "run.log").open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=variant_log, stderr=subprocess.STDOUT)
            hb = float(args.heartbeat_sec)
            poll_sec = min(5.0, max(0.5, hb / 12.0))
            last_hb = 0.0
            while True:
                rc = proc.poll()
                now = time.time()
                if last_hb <= 0.0 or (now - last_hb) >= hb:
                    st = {
                        "updated_utc": _now_utc(),
                        "state": "running",
                        "current_variant": v.name,
                        "variant_index": int(vi),
                        "variant_total": int(len(variants)),
                        "pid": int(proc.pid),
                        "last_line": _tail_last_line(vdir / "run.log"),
                        "variant_out": str(vdir),
                    }
                    _write_json(status_path, st)
                    with run_log.open("a", encoding="utf-8") as lf:
                        lf.write(f"[hier] heartbeat variant={v.name} pid={proc.pid} last='{st['last_line']}'\n")
                    last_hb = now
                if rc is not None:
                    break
                time.sleep(poll_sec)
            variant_log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"Variant {v.name} failed with exit {proc.returncode}.")

        if not bool(args.skip_realdata):
            real_out = vdir / "realdata"
            real_out.mkdir(parents=True, exist_ok=True)
            if v.name != variants[0].name:
                base_real_out = variant_dirs[variants[0].name] / "realdata"
                if base_real_out.exists():
                    _reuse_realdata_caches(base_real_out, real_out)
            real_summary = _find_real_summary(real_out, run_tag)
            extra_real = _translate_variant_args_for_real(v.extra_args)
            cmd_real = [
                sys.executable,
                "scripts/run_dark_siren_gap_test.py",
                "--run-dir",
                str(args.run_dir),
                "--skymap-dir",
                str(args.real_skymap_dir),
                "--glade-index",
                str(args.real_glade_index),
                "--out",
                str(real_out),
                "--gw-data-mode",
                str(args.real_gw_data_mode),
                "--pe-base-dir",
                str(args.real_pe_base_dir),
                "--max-events",
                str(int(args.max_events)),
                "--max-draws",
                str(int(args.max_draws)),
                "--n-proc",
                str(max(1, int(args.real_n_proc))),
                "--max-area-deg2",
                str(float(args.real_max_area_deg2)),
                "--max-gal",
                str(int(args.real_max_gal)),
                "--selection-injections-hdf",
                str(args.real_selection_injections_hdf),
                "--selection-det-model",
                str(args.real_selection_det_model),
                "--selection-weight-mode",
                str(args.real_selection_weight_mode),
                "--mixture-mode",
                "simple",
                "--mixture-f-miss-mode",
                "marginalize",
                "--mixture-f-miss-beta-mean",
                "0.6807953774124085",
                "--mixture-f-miss-beta-kappa",
                "8.0",
                *extra_real,
            ]
            for rec in [x.strip() for x in str(args.real_pe_record_ids).split(",") if x.strip()]:
                cmd_real.extend(["--pe-record-id", rec])
            if real_summary is None:
                with run_log.open("a", encoding="utf-8") as lf:
                    lf.write(f"[hier] launch real variant={v.name} cmd={shlex.join(cmd_real)}\n")
                real_log_path = real_out / "run.log"
                real_log = real_log_path.open("a", encoding="utf-8")
                proc = subprocess.Popen(cmd_real, cwd=str(Path.cwd()), stdout=real_log, stderr=subprocess.STDOUT)
                hb = float(args.heartbeat_sec)
                poll_sec = min(5.0, max(0.5, hb / 12.0))
                last_hb = 0.0
                while True:
                    rc = proc.poll()
                    now = time.time()
                    if last_hb <= 0.0 or (now - last_hb) >= hb:
                        st = {
                            "updated_utc": _now_utc(),
                            "state": "running",
                            "current_variant": v.name,
                            "stage": "realdata",
                            "variant_index": int(vi),
                            "variant_total": int(len(variants)),
                            "pid": int(proc.pid),
                            "last_line": _tail_last_line(real_log_path),
                            "variant_out": str(vdir),
                        }
                        _write_json(status_path, st)
                        with run_log.open("a", encoding="utf-8") as lf:
                            lf.write(f"[hier] heartbeat real variant={v.name} pid={proc.pid} last='{st['last_line']}'\n")
                        last_hb = now
                    if rc is not None:
                        break
                    time.sleep(poll_sec)
                real_log.close()
                if proc.returncode != 0:
                    raise RuntimeError(f"Real-data variant {v.name} failed with exit {proc.returncode}.")
                real_summary = _find_real_summary(real_out, run_tag)
            if real_summary is None:
                raise RuntimeError(f"Missing real-data summary for variant={v.name} in {real_out}")
            row = _read_real_summary(real_summary)
            if row is None:
                raise RuntimeError(f"Invalid real-data summary for variant={v.name}: {real_summary}")
            real_rows[v.name] = row

    # Integrate variant uncertainty at per-rep level.
    rep_ids: set[int] | None = None
    rep_data: dict[str, dict[int, dict[str, Any]]] = {}
    for v in variants:
        vrows: dict[int, dict[str, Any]] = {}
        for p in sorted((variant_dirs[v.name] / "reps").glob("rep[0-9][0-9][0-9][0-9].json")):
            rid = int(p.stem.replace("rep", ""))
            row = _read_rep_row(p)
            if row is None:
                continue
            vrows[rid] = row
        rep_data[v.name] = vrows
        ids = set(vrows.keys())
        rep_ids = ids if rep_ids is None else (rep_ids & ids)
    rep_ids = set() if rep_ids is None else rep_ids
    if not rep_ids:
        raise RuntimeError("No aligned replicate ids with required lpd_* fields across all variants.")

    rep_order = sorted(rep_ids)
    rep_lmu = np.asarray(
        [[float(rep_data[v.name][rid]["lpd_mu_total"]) for v in variants] for rid in rep_order],
        dtype=float,
    )
    rep_lgr = np.asarray(
        [[float(rep_data[v.name][rid]["lpd_gr_total"]) for v in variants] for rid in rep_order],
        dtype=float,
    )
    rep_lmu_d = np.asarray(
        [[float(rep_data[v.name][rid]["lpd_mu_total_data"]) for v in variants] for rid in rep_order],
        dtype=float,
    )
    rep_lgr_d = np.asarray(
        [[float(rep_data[v.name][rid]["lpd_gr_total_data"]) for v in variants] for rid in rep_order],
        dtype=float,
    )

    dt_arr = _logsumexp_rows(rep_lmu + logw.reshape((1, -1))) - _logsumexp_rows(rep_lgr + logw.reshape((1, -1)))
    dd_arr = _logsumexp_rows(rep_lmu_d + logw.reshape((1, -1))) - _logsumexp_rows(rep_lgr_d + logw.reshape((1, -1)))
    ds_arr = dt_arr - dd_arr

    rep_rows: list[dict[str, Any]] = []
    d_total: list[float] = []
    d_data: list[float] = []
    d_sel: list[float] = []
    for i, rid in enumerate(rep_order):
        dt = float(dt_arr[i])
        dd = float(dd_arr[i])
        ds = float(ds_arr[i])
        d_total.append(dt)
        d_data.append(dd)
        d_sel.append(ds)
        rep_rows.append(
            {
                "rep": int(rid),
                "delta_lpd_total_hier": dt,
                "delta_lpd_data_hier": dd,
                "delta_lpd_sel_hier": ds,
            }
        )

    rep_rows_path = out_root / "hierarchical_reps.json"
    _write_json(rep_rows_path, {"rows": rep_rows})

    real_section: dict[str, Any] = {"enabled": not bool(args.skip_realdata)}
    if not bool(args.skip_realdata):
        real_lmu = np.asarray([float(real_rows[v.name]["lpd_mu_total"]) for v in variants], dtype=float)
        real_lgr = np.asarray([float(real_rows[v.name]["lpd_gr_total"]) for v in variants], dtype=float)
        real_lmu_d = np.asarray([float(real_rows[v.name]["lpd_mu_total_data"]) for v in variants], dtype=float)
        real_lgr_d = np.asarray([float(real_rows[v.name]["lpd_gr_total_data"]) for v in variants], dtype=float)

        real_dt = _delta_from_weights(logw=logw, lpd_mu=real_lmu, lpd_gr=real_lgr)
        real_dd = _delta_from_weights(logw=logw, lpd_mu=real_lmu_d, lpd_gr=real_lgr_d)
        real_ds = float(real_dt - real_dd)
        p_ge_real_fixed = float(np.mean(dt_arr >= real_dt))

        real_section.update(
            {
                "delta_lpd_total_hier_real_fixed_weights": float(real_dt),
                "delta_lpd_data_hier_real_fixed_weights": float(real_dd),
                "delta_lpd_sel_hier_real_fixed_weights": float(real_ds),
                "calibrated_p_ge_real_fixed_weights": float(p_ge_real_fixed),
                "variant_rows": {
                    v.name: {
                        "delta_lpd_total": float(real_rows[v.name]["lpd_mu_total"] - real_rows[v.name]["lpd_gr_total"]),
                        "delta_lpd_data": float(real_rows[v.name]["lpd_mu_total_data"] - real_rows[v.name]["lpd_gr_total_data"]),
                        "summary_file": str(_find_real_summary(variant_dirs[v.name] / "realdata", run_tag)),
                    }
                    for v in variants
                },
            }
        )

        if int(args.weight_samples) > 0 and float(args.weight_kappa) > 0.0:
            rng = np.random.default_rng(int(args.seed) + 11357)
            alpha = np.clip(w * float(args.weight_kappa), 1e-9, np.inf)
            ws = rng.dirichlet(alpha, size=int(args.weight_samples))
            real_dt_samples = np.empty((ws.shape[0],), dtype=float)
            pge_samples = np.empty((ws.shape[0],), dtype=float)
            for i in range(ws.shape[0]):
                lw_i = np.log(np.clip(ws[i], 1e-300, np.inf))
                rep_dt_i = _logsumexp_rows(rep_lmu + lw_i.reshape((1, -1))) - _logsumexp_rows(rep_lgr + lw_i.reshape((1, -1)))
                real_dt_i = _delta_from_weights(logw=lw_i, lpd_mu=real_lmu, lpd_gr=real_lgr)
                real_dt_samples[i] = real_dt_i
                pge_samples[i] = float(np.mean(rep_dt_i >= real_dt_i))
            real_section["weight_uncertainty"] = {
                "weight_kappa": float(args.weight_kappa),
                "weight_samples": int(args.weight_samples),
                "real_delta_lpd_total_hier_stats": _synth_stats(real_dt_samples),
                "calibrated_p_ge_real_stats": _synth_stats(pge_samples),
            }

    summary = {
        "updated_utc": _now_utc(),
        "n_rep_aligned": int(len(rep_rows)),
        "variants": [asdict(v) for v in variants],
        "weights_normalized": [float(x) for x in w],
        "delta_lpd_total_hier": _summ(d_total),
        "delta_lpd_data_hier": _summ(d_data),
        "delta_lpd_sel_hier": _summ(d_sel),
        "real_data_joint": real_section,
        "variants_out_dirs": {k: str(v) for k, v in variant_dirs.items()},
    }
    _write_json(results_path, summary)

    lines = ["rep,delta_lpd_total_hier,delta_lpd_data_hier,delta_lpd_sel_hier"]
    for r in rep_rows:
        lines.append(
            f"{int(r['rep'])},{float(r['delta_lpd_total_hier']):.9f},{float(r['delta_lpd_data_hier']):.9f},{float(r['delta_lpd_sel_hier']):.9f}"
        )
    table_path.write_text("\n".join(lines) + "\n")

    _write_json(
        status_path,
        {
            "updated_utc": _now_utc(),
            "state": "complete",
            "n_rep_aligned": int(len(rep_rows)),
            "summary": str(results_path),
        },
    )
    print(f"[hier] complete summary={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
