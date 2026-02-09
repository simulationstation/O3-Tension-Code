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


def _is_complete_summary(path: Path, n_rep: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return int(data.get("n_rep_done", 0)) >= int(n_rep)


@dataclass
class Variant:
    name: str
    description: str
    extra_args: list[str]


@dataclass
class VariantResult:
    name: str
    out_dir: str
    n_rep_done: int
    mean_total: float
    sd_total: float
    max_total: float
    p_ge_0: float
    p_ge_3: float
    mean_data: float
    mean_sel: float


def _variant_specs(*, f_ref: float, f_kappa: float) -> list[Variant]:
    f_lo = max(1e-3, float(f_ref) - 0.10)
    f_hi = min(0.999, float(f_ref) + 0.10)
    return [
        Variant(
            name="baseline",
            description="Matched injection/scoring baseline",
            extra_args=[],
        ),
        Variant(
            name="score_fmiss_fixed_low",
            description="Score with fixed low f_miss",
            extra_args=["--f-miss-mode", "fixed", "--f-miss-fixed", f"{f_lo:.6f}"],
        ),
        Variant(
            name="score_fmiss_fixed_high",
            description="Score with fixed high f_miss",
            extra_args=["--f-miss-mode", "fixed", "--f-miss-fixed", f"{f_hi:.6f}"],
        ),
        Variant(
            name="score_fmiss_beta_low",
            description="Score with marginalized low-mean f_miss prior",
            extra_args=[
                "--f-miss-mode",
                "marginalize",
                "--f-miss-beta-mean",
                f"{f_lo:.6f}",
                "--f-miss-beta-kappa",
                f"{float(f_kappa):.6f}",
            ],
        ),
        Variant(
            name="score_fmiss_beta_high",
            description="Score with marginalized high-mean f_miss prior",
            extra_args=[
                "--f-miss-mode",
                "marginalize",
                "--f-miss-beta-mean",
                f"{f_hi:.6f}",
                "--f-miss-beta-kappa",
                f"{float(f_kappa):.6f}",
            ],
        ),
        Variant(
            name="inj_f_low",
            description="Inject low f_miss, score baseline",
            extra_args=["--inj-f-miss-fixed", f"{f_lo:.6f}"],
        ),
        Variant(
            name="inj_f_high",
            description="Inject high f_miss, score baseline",
            extra_args=["--inj-f-miss-fixed", f"{f_hi:.6f}"],
        ),
        Variant(
            name="hostz_k_minus2",
            description="Host prior with comoving_powerlaw k=-2",
            extra_args=["--host-prior-z-mode", "comoving_powerlaw", "--host-prior-z-k", "-2.0"],
        ),
        Variant(
            name="hostz_k_plus2",
            description="Host prior with comoving_powerlaw k=+2",
            extra_args=["--host-prior-z-mode", "comoving_powerlaw", "--host-prior-z-k", "2.0"],
        ),
        Variant(
            name="selection_threshold",
            description="Selection model threshold instead of snr_binned",
            extra_args=["--selection-det-model", "threshold"],
        ),
        Variant(
            name="selection_weight_none",
            description="Selection weights disabled",
            extra_args=["--selection-weight-mode", "none"],
        ),
        Variant(
            name="no_selection_extreme",
            description="Disable selection normalization (extreme control)",
            extra_args=["--no-selection"],
        ),
    ]


def _build_suite_cmd(
    *,
    out_dir: Path,
    run_dir: str,
    seed: int,
    n_rep: int,
    n_proc: int,
    max_events: int,
    max_draws: int,
    heartbeat_sec: float,
    partial_write_min_sec: float,
    f_ref: float,
    f_kappa: float,
    extra_args: list[str],
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_dark_siren_catalog_injection_suite.py",
        "--out",
        str(out_dir),
        "--run-dir",
        str(run_dir),
        "--truth-model",
        "gr",
        "--inj-host-mode",
        "mixture",
        "--inj-f-miss-mode",
        "fixed",
        "--inj-f-miss-fixed",
        str(float(f_ref)),
        "--inj-logR-scale",
        "0.0",
        "--f-miss-mode",
        "marginalize",
        "--f-miss-beta-mean",
        str(float(f_ref)),
        "--f-miss-beta-kappa",
        str(float(f_kappa)),
        "--seed",
        str(int(seed)),
        "--n-rep",
        str(int(n_rep)),
        "--n-proc",
        str(int(n_proc)),
        "--max-events",
        str(int(max_events)),
        "--max-draws",
        str(int(max_draws)),
        "--heartbeat-sec",
        str(float(heartbeat_sec)),
        "--partial-write-min-sec",
        str(float(partial_write_min_sec)),
        *extra_args,
    ]


def _load_variant_result(name: str, out_dir: Path) -> VariantResult:
    summary = json.loads((out_dir / "summary.json").read_text())
    total = summary.get("delta_lpd_total", {})
    data = summary.get("delta_lpd_total_data", {})
    sel = summary.get("delta_lpd_total_sel", {})
    return VariantResult(
        name=str(name),
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


def _tail_lines(path: Path, max_lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _refresh_status_from_runlog(
    *,
    status: dict[str, Any],
    run_log: Path,
    hb_re: re.Pattern[str],
    rep_re: re.Pattern[str],
) -> None:
    lines = _tail_lines(run_log, max_lines=200)
    if not lines:
        return
    for line in reversed(lines):
        text = line.strip()
        if text:
            status["last_line"] = text[-500:]
            break
    for line in reversed(lines):
        if "heartbeat:" in line:
            m = hb_re.search(line)
            if m is not None:
                status["reps_done"] = int(m.group("rd"))
                status["reps_total"] = int(m.group("rt"))
                status["events_done_est"] = float(m.group("ed"))
                status["events_total"] = float(m.group("et"))
                status["events_pct"] = float(m.group("ep"))
            status["updated_utc"] = _now_utc()
            return
        if "ΔLPD_total=" in line and "rep " in line:
            m = rep_re.search(line)
            if m is not None:
                status["reps_done"] = int(m.group("rk"))
                status["reps_total"] = int(m.group("rt"))
                status["last_delta_lpd_total"] = str(m.group("dl"))
                status["updated_utc"] = _now_utc()
            return


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a resumable GR-systematics matrix using dark-siren catalog injection suite."
    )
    ap.add_argument("--out-root", required=True, type=str)
    ap.add_argument("--run-dir", type=str, default="outputs/finalization/highpower_multistart_v2/M0_start101")
    ap.add_argument("--variants", type=str, default="", help="Comma list of variant names; empty => full built-in matrix.")
    ap.add_argument("--f-miss-ref", type=float, default=0.6807953774124085)
    ap.add_argument("--f-miss-kappa", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-step", type=int, default=0, help="Optional seed increment per variant index.")
    ap.add_argument("--n-rep", type=int, default=128)
    ap.add_argument("--n-proc", type=int, default=0)
    ap.add_argument("--parallel-variants", type=int, default=1, help="Number of variants to run concurrently.")
    ap.add_argument("--max-events", type=int, default=36)
    ap.add_argument("--max-draws", type=int, default=256)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--partial-write-min-sec", type=float, default=10.0)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    n_proc = int(args.n_proc) if int(args.n_proc) > 0 else int(os.cpu_count() or 1)

    variants = _variant_specs(f_ref=float(args.f_miss_ref), f_kappa=float(args.f_miss_kappa))
    vmap = {v.name: v for v in variants}
    if str(args.variants).strip():
        wanted = [x.strip() for x in str(args.variants).split(",") if x.strip()]
        missing = [x for x in wanted if x not in vmap]
        if missing:
            raise ValueError(f"Unknown variant(s): {missing}. Available: {sorted(vmap.keys())}")
        variants = [vmap[x] for x in wanted]
    parallel_variants = max(1, int(args.parallel_variants))
    parallel_variants = min(parallel_variants, max(1, len(variants)))
    n_proc_per_variant = max(1, int(n_proc // parallel_variants))

    progress_path = out_root / "matrix_progress.json"
    manifest_path = out_root / "matrix_manifest.json"
    summary_path = out_root / "matrix_summary.json"
    top_log_path = out_root / "run.log"

    _write_json_atomic(
        manifest_path,
        {
            "created_utc": _now_utc(),
            "out_root": str(out_root),
            "argv": sys.argv,
            "config": {
                "run_dir": str(args.run_dir),
                "f_miss_ref": float(args.f_miss_ref),
                "f_miss_kappa": float(args.f_miss_kappa),
                "seed": int(args.seed),
                "seed_step": int(args.seed_step),
                "n_rep": int(args.n_rep),
                "n_proc": int(n_proc),
                "parallel_variants": int(parallel_variants),
                "n_proc_per_variant": int(n_proc_per_variant),
                "max_events": int(args.max_events),
                "max_draws": int(args.max_draws),
                "heartbeat_sec": float(args.heartbeat_sec),
                "partial_write_min_sec": float(args.partial_write_min_sec),
            },
            "variants": [
                {"name": v.name, "description": v.description, "extra_args": list(v.extra_args)}
                for v in variants
            ],
        },
    )

    t0 = time.time()
    statuses: list[dict[str, Any]] = [
        {
            "name": v.name,
            "description": v.description,
            "status": "pending",
            "out_dir": str(out_root / v.name),
        }
        for v in variants
    ]

    def write_progress(current_name: str | None, current_index: int | None) -> None:
        done_count = sum(1 for s in statuses if s.get("status") == "done")
        _write_json_atomic(
            progress_path,
            {
                "updated_utc": _now_utc(),
                "elapsed_sec": float(time.time() - t0),
                "variants_total": int(len(statuses)),
                "variants_done": int(done_count),
                "current_variant": current_name,
                "current_index": current_index,
                "statuses": statuses,
            },
        )

    hb_re = re.compile(
        r"reps_done=(?P<rd>\d+)/(?P<rt>\d+).+events_done≈(?P<ed>[0-9.]+)/(?P<et>[0-9.]+)\s+\((?P<ep>[0-9.]+)%\)"
    )
    rep_re = re.compile(r"rep\s+(?P<rk>\d+)/(?P<rt>\d+):\s+ΔLPD_total=(?P<dl>[^\s]+)")

    write_progress(current_name=None, current_index=None)
    with top_log_path.open("a", encoding="utf-8") as top_log:
        print(f"[sysgrid] start out_root={out_root}", flush=True)
        top_log.write(f"[sysgrid] start utc={_now_utc()} out_root={out_root}\n")
        top_log.flush()

        pending: list[int] = []
        for idx, var in enumerate(variants):
            status = statuses[idx]
            var_dir = Path(status["out_dir"])
            var_dir.mkdir(parents=True, exist_ok=True)
            summary_file = var_dir / "summary.json"
            if _is_complete_summary(summary_file, int(args.n_rep)):
                status["status"] = "done"
                status["note"] = "preexisting_complete"
                status["updated_utc"] = _now_utc()
                msg = f"[sysgrid] skip variant={var.name} (already complete)"
                print(msg, flush=True)
                top_log.write(msg + "\n")
                top_log.flush()
            else:
                pending.append(int(idx))
        write_progress(current_name=None, current_index=None)

        running: dict[int, dict[str, Any]] = {}
        hb_last = 0.0
        while pending or running:
            while pending and len(running) < parallel_variants:
                idx = pending.pop(0)
                var = variants[idx]
                status = statuses[idx]
                var_dir = Path(status["out_dir"])
                seed = int(args.seed) + int(args.seed_step) * int(idx)
                cmd = _build_suite_cmd(
                    out_dir=var_dir,
                    run_dir=str(args.run_dir),
                    seed=int(seed),
                    n_rep=int(args.n_rep),
                    n_proc=int(n_proc_per_variant),
                    max_events=int(args.max_events),
                    max_draws=int(args.max_draws),
                    heartbeat_sec=float(args.heartbeat_sec),
                    partial_write_min_sec=float(args.partial_write_min_sec),
                    f_ref=float(args.f_miss_ref),
                    f_kappa=float(args.f_miss_kappa),
                    extra_args=list(var.extra_args),
                )
                cmd_txt = " ".join(shlex.quote(x) for x in cmd)
                start_msg = f"[sysgrid] run variant={var.name} seed={seed} n_proc={n_proc_per_variant} cmd={cmd_txt}"
                print(start_msg, flush=True)
                top_log.write(start_msg + "\n")
                top_log.flush()

                var_log = (var_dir / "run.log").open("a", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=var_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                status["status"] = "running"
                status["pid"] = int(proc.pid)
                status["updated_utc"] = _now_utc()
                running[idx] = {"proc": proc, "var_log": var_log, "var_dir": var_dir, "name": var.name}

            current_name: str | None = None
            current_index: int | None = None
            for idx in list(running.keys()):
                info = running[idx]
                proc: subprocess.Popen[str] = info["proc"]
                status = statuses[idx]
                _refresh_status_from_runlog(
                    status=status,
                    run_log=Path(info["var_dir"]) / "run.log",
                    hb_re=hb_re,
                    rep_re=rep_re,
                )
                rc = proc.poll()
                if rc is None:
                    if current_name is None:
                        current_name = str(status["name"])
                        current_index = int(idx)
                    continue

                info["var_log"].close()
                if rc != 0:
                    status["status"] = "failed"
                    status["returncode"] = int(rc)
                    status["updated_utc"] = _now_utc()
                    write_progress(current_name=current_name, current_index=current_index)
                    raise RuntimeError(
                        f"Variant {status['name']} failed with return code {rc}. See {Path(info['var_dir']) / 'run.log'}."
                    )

                summary_file = Path(info["var_dir"]) / "summary.json"
                if not _is_complete_summary(summary_file, int(args.n_rep)):
                    status["status"] = "failed_missing_summary"
                    status["updated_utc"] = _now_utc()
                    write_progress(current_name=current_name, current_index=current_index)
                    raise RuntimeError(f"Variant {status['name']} finished but summary incomplete: {summary_file}")

                status["status"] = "done"
                status["updated_utc"] = _now_utc()
                done_msg = f"[sysgrid] done variant={status['name']} summary={summary_file}"
                print(done_msg, flush=True)
                top_log.write(done_msg + "\n")
                top_log.flush()
                del running[idx]

            now = time.time()
            if now - hb_last >= float(args.heartbeat_sec):
                done_count = sum(1 for s in statuses if s.get("status") == "done")
                running_names = ",".join(str(statuses[idx]["name"]) for idx in sorted(running.keys())) or "none"
                msg = (
                    f"[sysgrid] heartbeat done={done_count}/{len(statuses)} "
                    f"running={len(running)} [{running_names}] pending={len(pending)}"
                )
                print(msg, flush=True)
                top_log.write(msg + "\n")
                top_log.flush()
                hb_last = now

            write_progress(current_name=current_name, current_index=current_index)
            if running:
                time.sleep(2.0)

    rows = [_load_variant_result(v.name, out_root / v.name) for v in variants]
    rows_dicts = [asdict(r) for r in rows]
    _write_json_atomic(
        summary_path,
        {
            "updated_utc": _now_utc(),
            "elapsed_sec": float(time.time() - t0),
            "n_variants": int(len(rows_dicts)),
            "rows": rows_dicts,
        },
    )
    print(json.dumps({"matrix_summary": rows_dicts}, indent=2, sort_keys=True), flush=True)
    print(f"[sysgrid] complete summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
