#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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


def _parse_list_f(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Empty float list.")
    return vals


def _parse_list_s(text: str) -> list[str]:
    vals = [str(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Empty string list.")
    return vals


@dataclass(frozen=True)
class Case:
    run_dir: str
    sigma_highz_frac: float
    h0_local_ref: float
    local_mode: str
    gr_omega_mode: str
    gr_omega_fixed: float

    @property
    def label(self) -> str:
        rd = Path(self.run_dir).name
        s = f"s{self.sigma_highz_frac:.4f}".replace(".", "p")
        l = f"l{self.h0_local_ref:.2f}".replace(".", "p")
        om = self.gr_omega_mode
        loc = self.local_mode
        return f"{rd}__{s}__{l}__{loc}__{om}"


def _case_cmd(
    *,
    case: Case,
    out_dir: Path,
    draws: int,
    n_rep: int,
    seed: int,
    z_max: float,
    z_n: int,
    z_anchors: str,
    h0_local_sigma: float,
    h0_planck_ref: float,
    h0_planck_sigma: float,
    omega_m_planck: float,
    heartbeat_sec: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_hubble_tension_mg_forecast.py",
        "--run-dir",
        str(case.run_dir),
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
        str(float(case.sigma_highz_frac)),
        "--local-mode",
        str(case.local_mode),
        "--h0-local-ref",
        str(float(case.h0_local_ref)),
        "--h0-local-sigma",
        str(float(h0_local_sigma)),
        "--h0-planck-ref",
        str(float(h0_planck_ref)),
        "--h0-planck-sigma",
        str(float(h0_planck_sigma)),
        "--omega-m-planck",
        str(float(omega_m_planck)),
        "--gr-omega-mode",
        str(case.gr_omega_mode),
        "--gr-omega-fixed",
        str(float(case.gr_omega_fixed)),
        "--heartbeat-sec",
        str(float(heartbeat_sec)),
        "--resume",
    ]


def _summary_complete(case_out: Path) -> bool:
    return (case_out / "tables" / "summary.json").exists()


def _load_case_metrics(case_out: Path) -> dict[str, Any]:
    s = json.loads((case_out / "tables" / "summary.json").read_text(encoding="utf-8"))
    proj = dict(s.get("h0_tension_projection", {}))
    refs = dict(s.get("references", {}))
    post = dict(s.get("h0_posterior_mg_truth", {}))
    return {
        "tension_relief_fraction_vs_planck_local_baseline": float(
            proj.get("tension_relief_fraction_vs_planck_local_baseline", float("nan"))
        ),
        "mg_p50_minus_planck": float(proj.get("mg_p50_minus_planck", float("nan"))),
        "mg_p50_minus_local": float(proj.get("mg_p50_minus_local", float("nan"))),
        "mg_sigma_vs_planck": float(proj.get("mg_sigma_vs_planck", float("nan"))),
        "mg_sigma_vs_local": float(proj.get("mg_sigma_vs_local", float("nan"))),
        "anchor_h0_gr_mean": float(proj.get("anchor_h0_gr_mean", float("nan"))),
        "anchor_h0_gr_internal_sd": float(proj.get("anchor_h0_gr_internal_sd", float("nan"))),
        "anchor_h0_gr_between_z_sd": float(proj.get("anchor_h0_gr_between_z_sd", float("nan"))),
        "anchor_gap_local_minus_gr": float(proj.get("anchor_gap_local_minus_gr", float("nan"))),
        "anchor_gap_local_minus_gr_sigma": float(proj.get("anchor_gap_local_minus_gr_sigma", float("nan"))),
        "tension_relief_fraction_anchor_gr": float(proj.get("tension_relief_fraction_anchor_gr", float("nan"))),
        "anchor_h0_mg_mean": float(proj.get("anchor_h0_mg_mean", float("nan"))),
        "anchor_gap_local_minus_mg": float(proj.get("anchor_gap_local_minus_mg", float("nan"))),
        "anchor_gap_local_minus_mg_sigma": float(proj.get("anchor_gap_local_minus_mg_sigma", float("nan"))),
        "tension_relief_fraction_anchor_mg": float(proj.get("tension_relief_fraction_anchor_mg", float("nan"))),
        "h0_p50": float(post.get("p50", float("nan"))),
        "h0_sd": float(post.get("sd", float("nan"))),
        "h0_local_ref": float(refs.get("h0_local_ref", float("nan"))),
        "sigma_highz_frac": float(refs.get("sigma_highz_frac", float("nan"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resumable robustness grid for Hubble-tension MG forecast across seeds/assumptions."
    )
    ap.add_argument("--out-root", required=True, help="Top-level output for the grid run.")
    ap.add_argument(
        "--run-dirs",
        default="outputs/finalization/highpower_multistart_v2/M0_start101",
        help="Comma list of run dirs (e.g., all O3 seeds).",
    )
    ap.add_argument("--sigma-highz-fracs", default="0.005,0.01,0.02")
    ap.add_argument("--h0-local-refs", default="72,73,74")
    ap.add_argument("--local-modes", default="external,truth")
    ap.add_argument("--gr-omega-modes", default="sample,fixed")
    ap.add_argument("--gr-omega-fixed", type=float, default=0.315)
    ap.add_argument("--draws", type=int, default=4096)
    ap.add_argument("--n-rep", type=int, default=5000)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--z-n", type=int, default=240)
    ap.add_argument("--z-anchors", type=str, default="0.2,0.35,0.5,0.62")
    ap.add_argument("--h0-local-sigma", type=float, default=1.0)
    ap.add_argument("--h0-planck-ref", type=float, default=67.4)
    ap.add_argument("--h0-planck-sigma", type=float, default=0.5)
    ap.add_argument("--omega-m-planck", type=float, default=0.315)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases_dir = out_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "grid_progress.json"
    summary_path = out_root / "grid_summary.json"
    csv_path = out_root / "grid_results.csv"
    log_path = out_root / "grid_run.log"

    run_dirs = _parse_list_s(args.run_dirs)
    sigmas = _parse_list_f(args.sigma_highz_fracs)
    locals_ref = _parse_list_f(args.h0_local_refs)
    local_modes = _parse_list_s(args.local_modes)
    omega_modes = _parse_list_s(args.gr_omega_modes)

    for m in local_modes:
        if m not in {"external", "truth"}:
            raise ValueError(f"Unsupported local mode: {m}")
    for m in omega_modes:
        if m not in {"sample", "fixed"}:
            raise ValueError(f"Unsupported gr omega mode: {m}")

    all_cases: list[Case] = []
    for rd in run_dirs:
        for s in sigmas:
            for hl in locals_ref:
                for lm in local_modes:
                    for om in omega_modes:
                        all_cases.append(
                            Case(
                                run_dir=str(rd),
                                sigma_highz_frac=float(s),
                                h0_local_ref=float(hl),
                                local_mode=str(lm),
                                gr_omega_mode=str(om),
                                gr_omega_fixed=float(args.gr_omega_fixed),
                            )
                        )
    if not all_cases:
        raise ValueError("No cases built.")

    _write_json_atomic(
        out_root / "grid_manifest.json",
        {
            "created_utc": _utc_now(),
            "argv": sys.argv,
            "n_cases_total": int(len(all_cases)),
            "run_dirs": run_dirs,
            "sigma_highz_fracs": sigmas,
            "h0_local_refs": locals_ref,
            "local_modes": local_modes,
            "gr_omega_modes": omega_modes,
            "draws": int(args.draws),
            "n_rep": int(args.n_rep),
            "z_max": float(args.z_max),
            "z_n": int(args.z_n),
            "z_anchors": str(args.z_anchors),
            "h0_local_sigma": float(args.h0_local_sigma),
            "h0_planck_ref": float(args.h0_planck_ref),
            "h0_planck_sigma": float(args.h0_planck_sigma),
            "omega_m_planck": float(args.omega_m_planck),
        },
    )

    t0 = time.time()
    statuses: list[dict[str, Any]] = []
    for c in all_cases:
        statuses.append(
            {
                "label": c.label,
                "status": "pending",
                "out_dir": str(cases_dir / c.label),
                "case": asdict(c),
            }
        )

    def _write_progress(current_label: str | None) -> None:
        n_done = sum(1 for st in statuses if st["status"] == "done")
        n_fail = sum(1 for st in statuses if st["status"] == "failed")
        _write_json_atomic(
            progress_path,
            {
                "updated_utc": _utc_now(),
                "elapsed_sec": float(time.time() - t0),
                "n_cases_total": int(len(statuses)),
                "n_done": int(n_done),
                "n_failed": int(n_fail),
                "pct_done": float(100.0 * n_done / max(1, len(statuses))),
                "current_case": current_label,
                "statuses": statuses,
            },
        )

    _write_progress(None)
    with log_path.open("a", encoding="utf-8") as lg:
        lg.write(f"[grid] start utc={_utc_now()} n_cases={len(all_cases)}\n")
        lg.flush()
        for i, case in enumerate(all_cases):
            st = statuses[i]
            case_out = Path(st["out_dir"])
            case_out.mkdir(parents=True, exist_ok=True)
            if _summary_complete(case_out):
                st["status"] = "done"
                st["note"] = "preexisting_complete"
                st["updated_utc"] = _utc_now()
                _write_progress(case.label)
                msg = f"[grid] skip {case.label} (complete)"
                print(msg, flush=True)
                lg.write(msg + "\n")
                lg.flush()
                continue

            cmd = _case_cmd(
                case=case,
                out_dir=case_out,
                draws=int(args.draws),
                n_rep=int(args.n_rep),
                seed=int(args.seed0 + i),
                z_max=float(args.z_max),
                z_n=int(args.z_n),
                z_anchors=str(args.z_anchors),
                h0_local_sigma=float(args.h0_local_sigma),
                h0_planck_ref=float(args.h0_planck_ref),
                h0_planck_sigma=float(args.h0_planck_sigma),
                omega_m_planck=float(args.omega_m_planck),
                heartbeat_sec=float(args.heartbeat_sec),
            )
            st["status"] = "running"
            st["updated_utc"] = _utc_now()
            st["cmd"] = " ".join(shlex.quote(x) for x in cmd)
            _write_progress(case.label)
            msg = f"[grid] run {case.label}"
            print(msg, flush=True)
            lg.write(msg + "\n")
            lg.flush()

            case_log = case_out / "run.log"
            with case_log.open("a", encoding="utf-8") as clog:
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
                    clog.write(line + "\n")
                    clog.flush()
                    lg.write(f"[{case.label}] {line}\n")
                    lg.flush()
                    if "heartbeat" in line.lower():
                        st["last_line"] = line[-500:]
                        st["updated_utc"] = _utc_now()
                        _write_progress(case.label)
                rc = proc.wait()

            if rc == 0 and _summary_complete(case_out):
                st["status"] = "done"
                st["updated_utc"] = _utc_now()
                _write_progress(case.label)
                msg = f"[grid] done {case.label}"
                print(msg, flush=True)
                lg.write(msg + "\n")
                lg.flush()
            else:
                st["status"] = "failed"
                st["returncode"] = int(rc)
                st["updated_utc"] = _utc_now()
                _write_progress(case.label)
                msg = f"[grid] FAILED {case.label} rc={rc}"
                print(msg, flush=True)
                lg.write(msg + "\n")
                lg.flush()

    rows: list[dict[str, Any]] = []
    for st in statuses:
        if st["status"] != "done":
            continue
        case = dict(st["case"])
        metrics = _load_case_metrics(Path(st["out_dir"]))
        row = {**case, **metrics, "label": st["label"], "out_dir": st["out_dir"]}
        rows.append(row)

    if rows:
        # CSV
        cols = [
            "label",
            "run_dir",
            "sigma_highz_frac",
            "h0_local_ref",
            "local_mode",
            "gr_omega_mode",
            "gr_omega_fixed",
            "tension_relief_fraction_vs_planck_local_baseline",
            "mg_p50_minus_planck",
            "mg_p50_minus_local",
            "mg_sigma_vs_planck",
            "mg_sigma_vs_local",
            "anchor_h0_gr_mean",
            "anchor_h0_gr_internal_sd",
            "anchor_h0_gr_between_z_sd",
            "anchor_gap_local_minus_gr",
            "anchor_gap_local_minus_gr_sigma",
            "tension_relief_fraction_anchor_gr",
            "anchor_h0_mg_mean",
            "anchor_gap_local_minus_mg",
            "anchor_gap_local_minus_mg_sigma",
            "tension_relief_fraction_anchor_mg",
            "h0_p50",
            "h0_sd",
            "out_dir",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    def _dist_stats(vals: list[float]) -> dict[str, float]:
        vals = [x for x in vals if x == x]  # NaN filter
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "mean": float(sum(s) / n),
            "min": float(s[0]),
            "max": float(s[-1]),
            "p16": float(s[max(0, int(0.16 * (n - 1)))]),
            "p50": float(s[max(0, int(0.50 * (n - 1)))]),
            "p84": float(s[max(0, int(0.84 * (n - 1)))]),
        }

    rel_stats = {
        "posterior_h0_relief": _dist_stats(
            [float(r.get("tension_relief_fraction_vs_planck_local_baseline", float("nan"))) for r in rows]
        ),
        "anchor_gr_relief": _dist_stats([float(r.get("tension_relief_fraction_anchor_gr", float("nan"))) for r in rows]),
        "anchor_gr_gap_sigma": _dist_stats([float(r.get("anchor_gap_local_minus_gr_sigma", float("nan"))) for r in rows]),
        "anchor_mg_gap_sigma": _dist_stats([float(r.get("anchor_gap_local_minus_mg_sigma", float("nan"))) for r in rows]),
    }

    summary = {
        "updated_utc": _utc_now(),
        "n_cases_total": int(len(statuses)),
        "n_done": int(sum(1 for st in statuses if st["status"] == "done")),
        "n_failed": int(sum(1 for st in statuses if st["status"] == "failed")),
        "elapsed_sec": float(time.time() - t0),
        "relief_stats": rel_stats,
        "rows": rows,
    }
    _write_json_atomic(summary_path, summary)
    _write_progress(None)
    print(f"[grid] wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
