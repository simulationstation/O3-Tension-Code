#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("expected non-empty float list")
    return vals


def _parse_strs(text: str) -> list[str]:
    vals = [str(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("expected non-empty string list")
    return vals


def _case_cmd(
    *,
    run_dir: str,
    out_dir: Path,
    draws: int,
    n_rep: int,
    seed: int,
    z_max: float,
    z_n: int,
    z_anchors: str,
    sigma_highz_frac: float,
    highz_bias_frac: float,
    local_mode: str,
    h0_local_ref: float,
    h0_local_sigma: float,
    local_bias: float,
    h0_planck_ref: float,
    h0_planck_sigma: float,
    omega_m_planck: float,
    gr_omega_mode: str,
    gr_omega_fixed: float,
    heartbeat_sec: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_hubble_tension_mg_forecast.py",
        "--run-dir",
        str(run_dir),
        "--out",
        str(out_dir),
        "--draws",
        str(int(draws)),
        "--seed",
        str(int(seed)),
        "--z-max",
        str(float(z_max)),
        "--z-n",
        str(int(z_n)),
        "--z-anchors",
        str(z_anchors),
        "--n-rep",
        str(int(n_rep)),
        "--sigma-highz-frac",
        str(float(sigma_highz_frac)),
        "--highz-bias-frac",
        str(float(highz_bias_frac)),
        "--local-mode",
        str(local_mode),
        "--h0-local-ref",
        str(float(h0_local_ref)),
        "--h0-local-sigma",
        str(float(h0_local_sigma)),
        "--local-bias",
        str(float(local_bias)),
        "--h0-planck-ref",
        str(float(h0_planck_ref)),
        "--h0-planck-sigma",
        str(float(h0_planck_sigma)),
        "--omega-m-planck",
        str(float(omega_m_planck)),
        "--gr-omega-mode",
        str(gr_omega_mode),
        "--gr-omega-fixed",
        str(float(gr_omega_fixed)),
        "--heartbeat-sec",
        str(float(heartbeat_sec)),
        "--resume",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bias-transfer sweep for Hubble forecast: quantify apparent relief from high-z/local calibration offsets."
    )
    ap.add_argument("--out-root", required=True)
    ap.add_argument(
        "--run-dirs",
        default="outputs/finalization/highpower_multistart_v2/M0_start101,outputs/finalization/highpower_multistart_v2/M0_start202,outputs/finalization/highpower_multistart_v2/M0_start303,outputs/finalization/highpower_multistart_v2/M0_start404,outputs/finalization/highpower_multistart_v2/M0_start505",
    )
    ap.add_argument("--highz-bias-fracs", default="-0.01,-0.005,0.0,0.005,0.01")
    ap.add_argument("--local-biases", default="-0.5,0.0,0.5")
    ap.add_argument("--draws", type=int, default=8192)
    ap.add_argument("--n-rep", type=int, default=40000)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--z-n", type=int, default=320)
    ap.add_argument("--z-anchors", type=str, default="0.2,0.35,0.5,0.62")
    ap.add_argument("--sigma-highz-frac", type=float, default=0.01)
    ap.add_argument("--local-mode", choices=["external", "truth"], default="external")
    ap.add_argument("--h0-local-ref", type=float, default=73.0)
    ap.add_argument("--h0-local-sigma", type=float, default=1.0)
    ap.add_argument("--h0-planck-ref", type=float, default=67.4)
    ap.add_argument("--h0-planck-sigma", type=float, default=0.5)
    ap.add_argument("--omega-m-planck", type=float, default=0.315)
    ap.add_argument("--gr-omega-mode", choices=["sample", "fixed"], default="sample")
    ap.add_argument("--gr-omega-fixed", type=float, default=0.315)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases_dir = out_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "sweep_progress.json"
    summary_path = out_root / "sweep_summary.json"
    csv_path = out_root / "sweep_results.csv"
    run_log = out_root / "sweep_run.log"

    run_dirs = _parse_strs(args.run_dirs)
    highz_biases = _parse_floats(args.highz_bias_fracs)
    local_biases = _parse_floats(args.local_biases)

    cases: list[dict[str, Any]] = []
    for rd in run_dirs:
        rd_name = Path(rd).name
        for hb in highz_biases:
            for lb in local_biases:
                lbl = f"{rd_name}__hb{hb:+.4f}__lb{lb:+.3f}".replace(".", "p")
                cases.append(
                    {
                        "label": lbl,
                        "run_dir": rd,
                        "highz_bias_frac": float(hb),
                        "local_bias": float(lb),
                        "out_dir": str(cases_dir / lbl),
                        "status": "pending",
                    }
                )

    _write_json_atomic(
        out_root / "sweep_manifest.json",
        {
            "created_utc": _utc_now(),
            "argv": sys.argv,
            "n_cases_total": int(len(cases)),
            "run_dirs": run_dirs,
            "highz_bias_fracs": highz_biases,
            "local_biases": local_biases,
            "draws": int(args.draws),
            "n_rep": int(args.n_rep),
            "sigma_highz_frac": float(args.sigma_highz_frac),
            "local_mode": str(args.local_mode),
            "h0_local_ref": float(args.h0_local_ref),
            "h0_local_sigma": float(args.h0_local_sigma),
            "gr_omega_mode": str(args.gr_omega_mode),
            "gr_omega_fixed": float(args.gr_omega_fixed),
        },
    )

    t0 = time.time()

    def _save_progress(current: str | None) -> None:
        n_done = sum(1 for c in cases if c["status"] == "done")
        n_failed = sum(1 for c in cases if c["status"] == "failed")
        _write_json_atomic(
            progress_path,
            {
                "updated_utc": _utc_now(),
                "elapsed_sec": float(time.time() - t0),
                "n_cases_total": int(len(cases)),
                "n_done": int(n_done),
                "n_failed": int(n_failed),
                "pct_done": float(100.0 * n_done / max(1, len(cases))),
                "current_case": current,
                "cases": cases,
            },
        )

    _save_progress(None)

    with run_log.open("a", encoding="utf-8") as lg:
        lg.write(f"[sweep] start utc={_utc_now()} n_cases={len(cases)}\n")
        lg.flush()
        for i, case in enumerate(cases):
            case_out = Path(case["out_dir"])
            case_out.mkdir(parents=True, exist_ok=True)
            summary_file = case_out / "tables" / "summary.json"
            if summary_file.exists():
                case["status"] = "done"
                case["note"] = "preexisting_complete"
                case["updated_utc"] = _utc_now()
                _save_progress(case["label"])
                continue

            cmd = _case_cmd(
                run_dir=str(case["run_dir"]),
                out_dir=case_out,
                draws=int(args.draws),
                n_rep=int(args.n_rep),
                seed=int(args.seed0 + i),
                z_max=float(args.z_max),
                z_n=int(args.z_n),
                z_anchors=str(args.z_anchors),
                sigma_highz_frac=float(args.sigma_highz_frac),
                highz_bias_frac=float(case["highz_bias_frac"]),
                local_mode=str(args.local_mode),
                h0_local_ref=float(args.h0_local_ref),
                h0_local_sigma=float(args.h0_local_sigma),
                local_bias=float(case["local_bias"]),
                h0_planck_ref=float(args.h0_planck_ref),
                h0_planck_sigma=float(args.h0_planck_sigma),
                omega_m_planck=float(args.omega_m_planck),
                gr_omega_mode=str(args.gr_omega_mode),
                gr_omega_fixed=float(args.gr_omega_fixed),
                heartbeat_sec=float(args.heartbeat_sec),
            )
            case["status"] = "running"
            case["updated_utc"] = _utc_now()
            case["cmd"] = " ".join(shlex.quote(x) for x in cmd)
            _save_progress(case["label"])
            lg.write(f"[sweep] run {case['label']}\n")
            lg.flush()
            with (case_out / "run.log").open("a", encoding="utf-8") as clog:
                proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=clog, stderr=subprocess.STDOUT)
                rc = proc.wait()
            case["returncode"] = int(rc)
            case["updated_utc"] = _utc_now()
            if rc == 0 and summary_file.exists():
                case["status"] = "done"
                lg.write(f"[sweep] done {case['label']}\n")
            else:
                case["status"] = "failed"
                lg.write(f"[sweep] failed {case['label']} rc={rc}\n")
            lg.flush()
            _save_progress(case["label"])

    rows: list[dict[str, Any]] = []
    for case in cases:
        if case["status"] != "done":
            continue
        summary_file = Path(case["out_dir"]) / "tables" / "summary.json"
        s = json.loads(summary_file.read_text(encoding="utf-8"))
        proj = dict(s.get("h0_tension_projection", {}))
        rows.append(
            {
                "label": case["label"],
                "run_dir": case["run_dir"],
                "highz_bias_frac": case["highz_bias_frac"],
                "local_bias": case["local_bias"],
                "anchor_gr_relief": float(proj.get("tension_relief_fraction_anchor_gr", float("nan"))),
                "anchor_gr_gap_sigma": float(proj.get("anchor_gap_local_minus_gr_sigma", float("nan"))),
                "posterior_h0_relief": float(proj.get("tension_relief_fraction_vs_planck_local_baseline", float("nan"))),
                "out_dir": case["out_dir"],
            }
        )

    fieldnames = [
        "label",
        "run_dir",
        "highz_bias_frac",
        "local_bias",
        "anchor_gr_relief",
        "anchor_gr_gap_sigma",
        "posterior_h0_relief",
        "out_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    def _agg(col: str) -> dict[str, float]:
        vals = [float(r[col]) for r in rows]
        return {
            "mean": float(sum(vals) / max(1, len(vals))),
            "min": float(min(vals)) if vals else float("nan"),
            "max": float(max(vals)) if vals else float("nan"),
        }

    by_highz: dict[str, dict[str, float]] = {}
    for hb in sorted(set(float(r["highz_bias_frac"]) for r in rows)):
        subset = [r for r in rows if float(r["highz_bias_frac"]) == hb]
        vals = [float(r["anchor_gr_relief"]) for r in subset]
        by_highz[str(hb)] = {
            "mean_anchor_gr_relief": float(sum(vals) / max(1, len(vals))),
            "min_anchor_gr_relief": float(min(vals)) if vals else float("nan"),
            "max_anchor_gr_relief": float(max(vals)) if vals else float("nan"),
        }
    by_local: dict[str, dict[str, float]] = {}
    for lb in sorted(set(float(r["local_bias"]) for r in rows)):
        subset = [r for r in rows if float(r["local_bias"]) == lb]
        vals = [float(r["anchor_gr_relief"]) for r in subset]
        by_local[str(lb)] = {
            "mean_anchor_gr_relief": float(sum(vals) / max(1, len(vals))),
            "min_anchor_gr_relief": float(min(vals)) if vals else float("nan"),
            "max_anchor_gr_relief": float(max(vals)) if vals else float("nan"),
        }

    summary = {
        "created_utc": _utc_now(),
        "elapsed_sec": float(time.time() - t0),
        "n_cases_total": int(len(cases)),
        "n_done": int(sum(1 for c in cases if c["status"] == "done")),
        "n_failed": int(sum(1 for c in cases if c["status"] == "failed")),
        "overall": {
            "anchor_gr_relief": _agg("anchor_gr_relief"),
            "anchor_gr_gap_sigma": _agg("anchor_gr_gap_sigma"),
            "posterior_h0_relief": _agg("posterior_h0_relief"),
        },
        "by_highz_bias_frac": by_highz,
        "by_local_bias": by_local,
    }
    _write_json_atomic(summary_path, summary)
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
