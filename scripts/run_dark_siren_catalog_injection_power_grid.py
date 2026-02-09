#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _parse_scales(text: str) -> list[float]:
    out: list[float] = []
    for tok in [t.strip() for t in str(text).split(",") if t.strip()]:
        val = float(tok)
        if val < 0.0:
            raise ValueError(f"Scale must be >= 0, got {val}")
        out.append(val)
    if not out:
        raise ValueError("No scales parsed from --scales")
    return out


def _scale_label(scale: float) -> str:
    s = f"{scale:.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _is_complete_summary(path: Path, n_rep: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return int(data.get("n_rep_done", 0)) >= int(n_rep)


@dataclass
class ScaleResult:
    scale: float
    out_dir: str
    n_rep_done: int
    mean_total: float
    sd_total: float
    max_total: float
    p_ge_0: float
    p_ge_3: float
    mean_data: float
    mean_sel: float


def _load_scale_result(scale: float, out_dir: Path) -> ScaleResult:
    summary = json.loads((out_dir / "summary.json").read_text())
    total = summary.get("delta_lpd_total", {})
    data = summary.get("delta_lpd_total_data", {})
    sel = summary.get("delta_lpd_total_sel", {})
    return ScaleResult(
        scale=float(scale),
        out_dir=str(out_dir),
        n_rep_done=int(summary.get("n_rep_done", 0)),
        mean_total=float(total.get("mean", float("nan"))),
        sd_total=float(total.get("sd", float("nan"))),
        max_total=float(total.get("max", float("nan"))),
        p_ge_0=float(total.get("p_ge_0", float("nan"))),
        p_ge_3=float(total.get("p_ge_3", float("nan"))),
        mean_data=float(data.get("mean", float("nan"))),
        mean_sel=float(sel.get("mean", float("nan"))),
    )


def _build_rep_cmd(
    *,
    out_dir: Path,
    run_dir: str,
    truth_model: str,
    inj_host_mode: str,
    inj_f_miss_mode: str,
    inj_f_miss_fixed: float,
    inj_logR_scale: float,
    seed: int,
    n_rep: int,
    n_proc: int,
    max_events: int,
    max_draws: int,
    galaxy_chunk_size: int,
    missing_pixel_chunk_size: int,
    spectral_precompute_dl: str,
    spectral_precompute_dl_max_gb: float,
    heartbeat_sec: float,
    partial_write_min_sec: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_dark_siren_catalog_injection_suite.py",
        "--out",
        str(out_dir),
        "--run-dir",
        str(run_dir),
        "--truth-model",
        str(truth_model),
        "--inj-host-mode",
        str(inj_host_mode),
        "--inj-f-miss-mode",
        str(inj_f_miss_mode),
        "--inj-f-miss-fixed",
        str(inj_f_miss_fixed),
        "--inj-logR-scale",
        str(inj_logR_scale),
        "--seed",
        str(seed),
        "--n-rep",
        str(n_rep),
        "--n-proc",
        str(n_proc),
        "--max-events",
        str(max_events),
        "--max-draws",
        str(max_draws),
        "--galaxy-chunk-size",
        str(galaxy_chunk_size),
        "--missing-pixel-chunk-size",
        str(missing_pixel_chunk_size),
        "--spectral-precompute-dl",
        str(spectral_precompute_dl),
        "--spectral-precompute-dl-max-gb",
        str(float(spectral_precompute_dl_max_gb)),
        "--heartbeat-sec",
        str(heartbeat_sec),
        "--partial-write-min-sec",
        str(partial_write_min_sec),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a resumable dose-response grid over --inj-logR-scale for dark-siren catalog injections."
    )
    ap.add_argument("--out-root", required=True, type=str, help="Top-level output directory for this grid run.")
    ap.add_argument("--run-dir", type=str, default="outputs/finalization/highpower_multistart_v2/M0_start101")
    ap.add_argument("--truth-model", choices=["gr", "mu"], default="mu")
    ap.add_argument("--scales", type=str, default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--inj-host-mode", choices=["catalog_only", "mixture"], default="mixture")
    ap.add_argument("--inj-f-miss-mode", choices=["fixed", "beta", "match_scoring"], default="fixed")
    ap.add_argument("--inj-f-miss-fixed", type=float, default=0.6807953774124085)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-step", type=int, default=0, help="Optional seed increment per scale index.")
    ap.add_argument("--n-rep", type=int, default=128)
    ap.add_argument("--n-proc", type=int, default=248)
    ap.add_argument("--max-events", type=int, default=36)
    ap.add_argument("--max-draws", type=int, default=256)
    ap.add_argument("--galaxy-chunk-size", type=int, default=200000)
    ap.add_argument("--missing-pixel-chunk-size", type=int, default=20000)
    ap.add_argument("--spectral-precompute-dl", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--spectral-precompute-dl-max-gb", type=float, default=24.0)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--partial-write-min-sec", type=float, default=30.0)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    scales = _parse_scales(args.scales)
    n_proc = int(args.n_proc) if int(args.n_proc) > 0 else int(os.cpu_count() or 1)

    progress_path = out_root / "grid_progress.json"
    summary_path = out_root / "grid_summary.json"
    manifest_path = out_root / "grid_manifest.json"
    top_log_path = out_root / "grid_run.log"

    _write_json_atomic(
        manifest_path,
        {
            "created_utc": _now_utc(),
            "out_root": str(out_root),
            "argv": sys.argv,
            "config": {
                "run_dir": str(args.run_dir),
                "truth_model": str(args.truth_model),
                "scales": list(scales),
                "inj_host_mode": str(args.inj_host_mode),
                "inj_f_miss_mode": str(args.inj_f_miss_mode),
                "inj_f_miss_fixed": float(args.inj_f_miss_fixed),
                "seed": int(args.seed),
                "seed_step": int(args.seed_step),
                "n_rep": int(args.n_rep),
                "n_proc": int(n_proc),
                "max_events": int(args.max_events),
                "max_draws": int(args.max_draws),
                "galaxy_chunk_size": int(args.galaxy_chunk_size),
                "missing_pixel_chunk_size": int(args.missing_pixel_chunk_size),
                "spectral_precompute_dl": str(args.spectral_precompute_dl),
                "spectral_precompute_dl_max_gb": float(args.spectral_precompute_dl_max_gb),
                "heartbeat_sec": float(args.heartbeat_sec),
                "partial_write_min_sec": float(args.partial_write_min_sec),
            },
        },
    )

    t0 = time.time()
    statuses: list[dict[str, Any]] = [
        {
            "scale": float(scale),
            "label": _scale_label(float(scale)),
            "status": "pending",
            "out_dir": str(out_root / f"scale_{_scale_label(float(scale))}"),
        }
        for scale in scales
    ]

    def write_progress(current_scale: float | None, current_index: int | None) -> None:
        done_count = sum(1 for s in statuses if s.get("status") == "done")
        _write_json_atomic(
            progress_path,
            {
                "updated_utc": _now_utc(),
                "elapsed_sec": float(time.time() - t0),
                "scales_total": int(len(scales)),
                "scales_done": int(done_count),
                "current_scale": current_scale,
                "current_index": current_index,
                "statuses": statuses,
            },
        )

    hb_re = re.compile(
        r"reps_done=(?P<rd>\d+)/(?P<rt>\d+).+events_done≈(?P<ed>[0-9.]+)/(?P<et>[0-9.]+)\s+\((?P<ep>[0-9.]+)%\)"
    )
    rep_re = re.compile(r"rep\s+(?P<rk>\d+)/(?P<rt>\d+):\s+ΔLPD_total=(?P<dl>[^\s]+)")

    write_progress(current_scale=None, current_index=None)

    with top_log_path.open("a", encoding="utf-8") as top_log:
        top_log.write(f"[grid] start utc={_now_utc()} out_root={out_root}\n")
        top_log.flush()
        print(f"[grid] start out_root={out_root}", flush=True)

        for idx, scale in enumerate(scales):
            status = statuses[idx]
            scale_dir = Path(status["out_dir"])
            scale_dir.mkdir(parents=True, exist_ok=True)
            summary = scale_dir / "summary.json"

            if _is_complete_summary(summary, int(args.n_rep)):
                status["status"] = "done"
                status["note"] = "preexisting_complete"
                status["updated_utc"] = _now_utc()
                write_progress(current_scale=float(scale), current_index=int(idx))
                msg = f"[grid] skip scale={scale} (already complete: {summary})"
                print(msg, flush=True)
                top_log.write(msg + "\n")
                top_log.flush()
                continue

            status["status"] = "running"
            status["updated_utc"] = _now_utc()
            write_progress(current_scale=float(scale), current_index=int(idx))

            seed = int(args.seed) + int(args.seed_step) * int(idx)
            cmd = _build_rep_cmd(
                out_dir=scale_dir,
                run_dir=str(args.run_dir),
                truth_model=str(args.truth_model),
                inj_host_mode=str(args.inj_host_mode),
                inj_f_miss_mode=str(args.inj_f_miss_mode),
                inj_f_miss_fixed=float(args.inj_f_miss_fixed),
                inj_logR_scale=float(scale),
                seed=int(seed),
                n_rep=int(args.n_rep),
                n_proc=int(n_proc),
                max_events=int(args.max_events),
                max_draws=int(args.max_draws),
                galaxy_chunk_size=int(args.galaxy_chunk_size),
                missing_pixel_chunk_size=int(args.missing_pixel_chunk_size),
                spectral_precompute_dl=str(args.spectral_precompute_dl),
                spectral_precompute_dl_max_gb=float(args.spectral_precompute_dl_max_gb),
                heartbeat_sec=float(args.heartbeat_sec),
                partial_write_min_sec=float(args.partial_write_min_sec),
            )
            cmd_txt = " ".join(shlex.quote(x) for x in cmd)
            start_msg = f"[grid] run scale={scale} seed={seed} cmd={cmd_txt}"
            print(start_msg, flush=True)
            top_log.write(start_msg + "\n")
            top_log.flush()

            scale_run_log = scale_dir / "run.log"
            with scale_run_log.open("a", encoding="utf-8") as per_scale_log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    prefixed = f"[scale {scale}] {line}"
                    print(prefixed, flush=True)
                    top_log.write(prefixed + "\n")
                    top_log.flush()
                    per_scale_log.write(line + "\n")
                    per_scale_log.flush()

                    if "heartbeat:" in line:
                        m = hb_re.search(line)
                        if m is not None:
                            status["reps_done"] = int(m.group("rd"))
                            status["reps_total"] = int(m.group("rt"))
                            status["events_done_est"] = float(m.group("ed"))
                            status["events_total"] = float(m.group("et"))
                            status["events_pct"] = float(m.group("ep"))
                        status["last_line"] = line[-500:]
                        status["updated_utc"] = _now_utc()
                        write_progress(current_scale=float(scale), current_index=int(idx))
                    elif "ΔLPD_total=" in line and "rep " in line:
                        m = rep_re.search(line)
                        if m is not None:
                            status["reps_done"] = int(m.group("rk"))
                            status["reps_total"] = int(m.group("rt"))
                            status["last_delta_lpd_total"] = str(m.group("dl"))
                        status["last_line"] = line[-500:]
                        status["updated_utc"] = _now_utc()
                        write_progress(current_scale=float(scale), current_index=int(idx))

                rc = proc.wait()
            if rc != 0:
                status["status"] = "failed"
                status["returncode"] = int(rc)
                status["updated_utc"] = _now_utc()
                write_progress(current_scale=float(scale), current_index=int(idx))
                raise RuntimeError(f"Scale {scale} failed with return code {rc}. See {top_log_path}.")

            status["status"] = "done"
            status["updated_utc"] = _now_utc()
            write_progress(current_scale=float(scale), current_index=int(idx))
            done_msg = f"[grid] done scale={scale} summary={summary}"
            print(done_msg, flush=True)
            top_log.write(done_msg + "\n")
            top_log.flush()

    rows = [_load_scale_result(float(scale), out_root / f"scale_{_scale_label(float(scale))}") for scale in scales]
    rows_dicts = [asdict(r) for r in rows]
    _write_json_atomic(
        summary_path,
        {
            "updated_utc": _now_utc(),
            "elapsed_sec": float(time.time() - t0),
            "n_scales": int(len(rows)),
            "rows": rows_dicts,
        },
    )
    print(json.dumps({"grid_summary": rows_dicts}, indent=2, sort_keys=True), flush=True)
    print(f"[grid] complete summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
