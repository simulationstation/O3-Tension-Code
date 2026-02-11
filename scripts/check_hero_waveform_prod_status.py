#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Status summary for hero waveform production matrix.")
    ap.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root (e.g. outputs/forward_tests/hero_waveform_consistency_prod_latest).",
    )
    ap.add_argument(
        "--analysis-cmd-pattern",
        type=str,
        default="bilby_pipe_analysis",
        help="Substring used to identify active bilby analysis processes in ps output.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    return ap.parse_args()


def proc_config_set(run_root: Path, analysis_cmd_pattern: str) -> set[str]:
    patt = re.escape(analysis_cmd_pattern)
    cmd = (
        "ps -eo args | "
        f"rg -i '{patt} .*"
        + str(run_root).replace("/", "\\/")
        + "' | rg -v 'rg -i' | "
        "awk '{for(i=1;i<=NF;i++){if($i ~ /config_complete\\.ini$/){print $i}}}'"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


_PROG_RE = re.compile(
    r"(?P<it>\d+)it\s+\[(?P<elapsed>[0-9:]+)\s+bound:(?P<bound>\d+)\s+nc:\s*(?P<nc>\d+)\s+"
    r"ncall:(?P<ncall>[0-9.e+\-]+)\s+eff:(?P<eff>[0-9.]+)%.*?dlogz:(?P<dlogz>[0-9.]+)>",
    re.DOTALL,
)


def parse_progress(run_log: Path) -> dict | None:
    if not run_log.exists():
        return None
    try:
        blob = run_log.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    blob = blob.replace("\r", "\n")
    matches = list(_PROG_RE.finditer(blob))
    if not matches:
        return None
    m = matches[-1]
    return {
        "it": int(m.group("it")),
        "elapsed": m.group("elapsed"),
        "bound": int(m.group("bound")),
        "nc": int(m.group("nc")),
        "ncall": m.group("ncall"),
        "eff_pct": float(m.group("eff")),
        "dlogz": float(m.group("dlogz")),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    job_dirs = sorted((run_root / "jobs").glob("*"))
    active_cfg = proc_config_set(run_root, args.analysis_cmd_pattern)

    rows = []
    counts = {"running": 0, "finished_ok": 0, "finished_fail": 0, "unknown": 0}
    for jd in job_dirs:
        cfg = jd / "config_complete.ini"
        exit_path = jd / "exit.code"
        run_log = jd / "run.log"
        result_files = sorted((jd / "run_out" / "result").glob("*result.hdf5"))
        progress = parse_progress(run_log)
        status = "unknown"
        exit_code = None
        if str(cfg) in active_cfg:
            status = "running"
        elif exit_path.exists():
            try:
                exit_code = int(exit_path.read_text(encoding="utf-8").strip())
            except Exception:
                exit_code = None
            status = "finished_ok" if exit_code == 0 else "finished_fail"
        elif result_files:
            status = "finished_ok"
        counts[status] += 1
        rows.append(
            {
                "job": jd.name,
                "status": status,
                "exit_code": exit_code,
                "result_files": len(result_files),
                "progress": progress,
            }
        )

    summary = {
        "run_root": str(run_root),
        "jobs_total": len(job_dirs),
        "counts": counts,
        "active_config_count": len(active_cfg),
    }

    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, indent=2))
        return

    print(f"run_root: {summary['run_root']}")
    print(f"jobs_total: {summary['jobs_total']}")
    print(
        "counts: "
        f"running={counts['running']} "
        f"finished_ok={counts['finished_ok']} "
        f"finished_fail={counts['finished_fail']} "
        f"unknown={counts['unknown']}"
    )
    print(f"active_config_count: {summary['active_config_count']}")
    for row in rows:
        prog = row["progress"]
        if prog is None:
            print(f"{row['job']}: {row['status']} (no progress line yet)")
            continue
        print(
            f"{row['job']}: {row['status']} "
            f"it={prog['it']} ncall={prog['ncall']} eff={prog['eff_pct']:.1f}% "
            f"dlogz={prog['dlogz']:.3f} elapsed={prog['elapsed']}"
        )


if __name__ == "__main__":
    main()
