#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EventSpec:
    name: str
    base_config: Path
    data_dump: Path


@dataclass
class JobSpec:
    job_id: str
    event: str
    waveform: str
    seed: int
    slot: int
    core_range: str
    job_dir: Path
    config_path: Path
    run_sh: Path
    result_glob: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build and optionally launch a production hero-event waveform matrix."
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "forward_tests" / f"hero_waveform_consistency_prod_{utc_stamp()}",
        help="Run output root directory.",
    )
    ap.add_argument(
        "--cores-total",
        type=int,
        default=256,
        help="Total cores to consume.",
    )
    ap.add_argument(
        "--cores-per-job",
        type=int,
        default=8,
        help="Cores pinned per job.",
    )
    ap.add_argument(
        "--waveforms",
        type=str,
        default="IMRPhenomXPHM,IMRPhenomPv3HM,SEOBNRv4PHM,IMRPhenomXHM",
        help="Comma-separated waveform approximants.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="1101,2202,3303,4404",
        help="Comma-separated sampling seeds.",
    )
    ap.add_argument("--nlive", type=int, default=512)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--maxcall", type=int, default=600000)
    ap.add_argument("--naccept", type=int, default=20)
    ap.add_argument("--dlogz", type=float, default=0.2)
    ap.add_argument("--checkpoint-delta-t", type=int, default=600)
    ap.add_argument(
        "--event-root",
        type=Path,
        default=ROOT / "artifacts" / "waveform" / "hero_waveform_inputs_20260210" / "events",
        help="Path containing GW200220_061928/ and GW200308_173609/ event input folders.",
    )
    ap.add_argument(
        "--bilby-pipe-analysis",
        type=str,
        default="bilby_pipe_analysis",
        help="Path or command name for bilby_pipe_analysis.",
    )
    ap.add_argument("--launch", action="store_true", help="Launch detached immediately.")
    return ap.parse_args()


def resolve_bilby_pipe_analysis(cmd: str) -> str:
    p = Path(cmd)
    if p.is_absolute() or "/" in cmd:
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Missing bilby_pipe_analysis: {cmd}")
    found = shutil.which(cmd)
    if found:
        return found
    raise FileNotFoundError(f"Could not resolve bilby_pipe_analysis command: {cmd}")


def event_specs(base: Path) -> Dict[str, EventSpec]:
    return {
        "GW200220_061928": EventSpec(
            name="GW200220_061928",
            base_config=base / "GW200220_061928" / "run_out" / "GW200220_061928_strain_reanalysis_config_complete.ini",
            data_dump=base / "GW200220_061928" / "run_out" / "data" / "GW200220_061928_strain_reanalysis_data0_1266214786-0_generation_data_dump.pickle",
        ),
        "GW200308_173609": EventSpec(
            name="GW200308_173609",
            base_config=base / "GW200308_173609" / "run_out" / "GW200308_173609_strain_reanalysis_config_complete.ini",
            data_dump=base / "GW200308_173609" / "run_out" / "data" / "GW200308_173609_strain_reanalysis_data0_1267724187-0_generation_data_dump.pickle",
        ),
    }


def patch_config(text: str, updates: Dict[str, str]) -> str:
    for key, value in updates.items():
        pattern = re.compile(rf"(?m)^({re.escape(key)})=.*$")
        replacement = rf"\1={value}"
        new_text, n = pattern.subn(replacement, text, count=1)
        if n == 0:
            raise ValueError(f"Could not find key '{key}' in base config.")
        text = new_text
    return text


def build_jobs(args: argparse.Namespace, out_root: Path, bilby_pipe_analysis: str) -> List[JobSpec]:
    waveforms = [w.strip() for w in args.waveforms.split(",") if w.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    events = list(event_specs(args.event_root.resolve()).values())

    for ev in events:
        if not ev.base_config.exists():
            raise FileNotFoundError(f"Missing base config: {ev.base_config}")
        if not ev.data_dump.exists():
            raise FileNotFoundError(f"Missing data dump: {ev.data_dump}")

    max_parallel = args.cores_total // args.cores_per_job
    job_count = len(events) * len(waveforms) * len(seeds)
    if job_count > max_parallel:
        raise ValueError(
            f"Job count {job_count} exceeds max_parallel {max_parallel}. "
            "Increase cores or reduce matrix."
        )

    jobs: List[JobSpec] = []
    slot = 0
    for ev in events:
        base_text = ev.base_config.read_text(encoding="utf-8")
        for wf in waveforms:
            for seed in seeds:
                core_start = slot * args.cores_per_job
                core_end = core_start + args.cores_per_job - 1
                core_range = f"{core_start}-{core_end}"
                job_id = f"{ev.name}__{wf}__s{seed}"
                job_dir = out_root / "jobs" / job_id
                cfg = job_dir / "config_complete.ini"
                run_sh = job_dir / "run.sh"
                outdir = job_dir / "run_out"

                sampler_kwargs = (
                    "{"
                    f"'nlive': {args.nlive}, "
                    f"'maxiter': {args.maxiter}, "
                    f"'maxcall': {args.maxcall}, "
                    f"'naccept': {args.naccept}, "
                    f"'dlogz': {args.dlogz}, "
                    "'check_point_plot': False, "
                    f"'check_point_delta_t': {args.checkpoint_delta_t}, "
                    "'print_method': 'interval-300', "
                    "'sample': 'acceptance-walk', "
                    f"'npool': {args.cores_per_job}, "
                    "'resume': True"
                    "}"
                )

                patched = patch_config(
                    base_text,
                    {
                        "ignore-gwpy-data-quality-check": "False",
                        "label": job_id,
                        "outdir": str(outdir),
                        "request-cpus": str(args.cores_per_job),
                        "distance-marginalization": "True",
                        "phase-marginalization": "True",
                        "time-marginalization": "True",
                        "sampling-seed": str(seed),
                        "n-parallel": str(args.cores_per_job),
                        "sampler-kwargs": sampler_kwargs,
                        "waveform-approximant": wf,
                    },
                )

                job_dir.mkdir(parents=True, exist_ok=True)
                cfg.write_text(patched, encoding="utf-8")
                run_sh.write_text(
                    "\n".join(
                        [
                            "#!/usr/bin/env bash",
                            "set -euo pipefail",
                            f'cd "{job_dir}"',
                            "export OMP_NUM_THREADS=1",
                            "export MKL_NUM_THREADS=1",
                            "export OPENBLAS_NUM_THREADS=1",
                            "export NUMEXPR_NUM_THREADS=1",
                            'echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)" > status.log',
                            "set +e",
                            (
                                f'taskset -c {core_range} "{bilby_pipe_analysis}" '
                                f'"{cfg}" --data-dump-file "{ev.data_dump}" > run.log 2>&1'
                            ),
                            "ec=$?",
                            'echo "[end] $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$ec" >> status.log',
                            'echo "$ec" > exit.code',
                            "exit $ec",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                os.chmod(run_sh, 0o775)

                jobs.append(
                    JobSpec(
                        job_id=job_id,
                        event=ev.name,
                        waveform=wf,
                        seed=seed,
                        slot=slot,
                        core_range=core_range,
                        job_dir=job_dir,
                        config_path=cfg,
                        run_sh=run_sh,
                        result_glob=str(outdir / "result" / "*result.hdf5"),
                    )
                )
                slot += 1
    return jobs


def write_manifest(
    args: argparse.Namespace,
    out_root: Path,
    jobs: List[JobSpec],
) -> Path:
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "hero_waveform_consistency_prod_matrix",
        "out_root": str(out_root),
        "settings": {
            "cores_total": args.cores_total,
            "cores_per_job": args.cores_per_job,
            "nlive": args.nlive,
            "maxiter": args.maxiter,
            "maxcall": args.maxcall,
            "naccept": args.naccept,
            "dlogz": args.dlogz,
            "checkpoint_delta_t": args.checkpoint_delta_t,
            "waveforms": [w.strip() for w in args.waveforms.split(",") if w.strip()],
            "seeds": [int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        },
        "jobs": [asdict(j) | {"job_dir": str(j.job_dir), "config_path": str(j.config_path), "run_sh": str(j.run_sh)} for j in jobs],
    }
    path = out_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_launch_script(args: argparse.Namespace, out_root: Path, jobs: List[JobSpec]) -> Path:
    job_list = out_root / "job_list.txt"
    job_list.write_text("\n".join(str(j.run_sh) for j in jobs) + "\n", encoding="utf-8")
    max_parallel = args.cores_total // args.cores_per_job
    launch = out_root / "launch_all.sh"
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'cd "{out_root}"',
                f'echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting {len(jobs)} jobs with parallel={max_parallel}"',
                f'xargs -P {max_parallel} -I{{}} bash "{{}}" < "{job_list}"',
                'echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) all jobs exited"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(launch, 0o775)
    return launch


def launch_detached_per_job(out_root: Path, jobs: List[JobSpec]) -> int:
    pid_path = out_root / "launcher_setsid.pids"
    pid_path.write_text("", encoding="utf-8")
    launched = 0
    for job in jobs:
        cmd = f'setsid "{job.run_sh}" > /dev/null 2>&1 < /dev/null & echo $!'
        out = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
        if out:
            with pid_path.open("a", encoding="utf-8") as pf:
                pf.write(out + "\n")
            launched += 1
    return launched


def main() -> None:
    args = parse_args()
    bilby_pipe_analysis = resolve_bilby_pipe_analysis(args.bilby_pipe_analysis)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, out_root, bilby_pipe_analysis)
    manifest = write_manifest(args, out_root, jobs)
    launch_script = write_launch_script(args, out_root, jobs)

    print(f"[ok] matrix prepared: {out_root}")
    print(f"[ok] manifest: {manifest}")
    print(f"[ok] launch script: {launch_script}")
    print(f"[ok] jobs: {len(jobs)}")

    if args.launch:
        launched = launch_detached_per_job(out_root, jobs)
        print(f"[ok] launched detached jobs={launched}")
        print(f"[ok] pid list: {out_root / 'launcher_setsid.pids'}")


if __name__ == "__main__":
    main()
