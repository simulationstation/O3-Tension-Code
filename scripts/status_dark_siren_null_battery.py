#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _ps_status(pid: int) -> str | None:
    try:
        cp = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "pid,etime,%cpu,%mem,cmd"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    lines = [ln.rstrip() for ln in cp.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    return lines[-1]


def _tail_text(path: Path, *, n_lines: int = 1, max_bytes: int = 64_000) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            n = int(min(size, int(max_bytes)))
            f.seek(-n, 2)
            data = f.read(n)
        txt = data.decode("utf-8", errors="replace")
        lines = txt.splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-int(n_lines) :])
    except Exception:
        return None


def _head_text(path: Path, *, max_bytes: int = 64_000) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            data = f.read(int(max_bytes))
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _count_done(rep_dirs: Iterable[Path]) -> int:
    n = 0
    for d in rep_dirs:
        if any(d.glob("summary_*.json")):
            n += 1
    return n


def _split_done(rep_dirs: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    done: list[Path] = []
    incomplete: list[Path] = []
    for d in rep_dirs:
        if any(d.glob("summary_*.json")):
            done.append(d)
        else:
            incomplete.append(d)
    return done, incomplete


def _rep_dirs(mode_dir: Path) -> list[Path]:
    if not mode_dir.exists():
        return []
    reps = [p for p in mode_dir.glob("rep[0-9][0-9][0-9]") if p.is_dir()]
    return sorted(reps, key=lambda p: p.name)


def _find_latest_incomplete(rep_dirs: list[Path]) -> Path | None:
    best: tuple[float, Path] | None = None
    for d in rep_dirs:
        if any(d.glob("summary_*.json")):
            continue
        log = d / "run.log"
        if not log.exists():
            continue
        try:
            mtime = float(log.stat().st_mtime)
        except Exception:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, d)
    return best[1] if best is not None else None


def _parse_seed_ranges(run_log: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    head = _head_text(run_log, max_bytes=200_000) or ""
    tail = _tail_text(run_log, n_lines=500, max_bytes=200_000) or ""
    txt = head + "\n" + tail
    if not txt.strip():
        return out
    # Examples:
    #   [job] rotate: seeds=0..99 nproc=4 max_concurrent=30
    #   [job] galnull: seeds=0..49 nproc=4 max_concurrent=30
    for key in ("rotate", "galnull"):
        m = re.search(rf"\[job\] {re.escape(key)}: seeds=(\d+)\.\.(\d+)", txt)
        if not m:
            continue
        try:
            a = int(m.group(1))
            b = int(m.group(2))
        except Exception:
            continue
        if a <= b:
            out[key] = (a, b)
    return out


def _parse_events_per_rep(run_log: Path) -> int | None:
    head = _head_text(run_log, max_bytes=200_000) or ""
    tail = _tail_text(run_log, n_lines=500, max_bytes=200_000) or ""
    txt = head + "\n" + tail
    m = re.search(r"\[job\] events:\s*(\d+)", txt)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _count_event_done(rep_dirs: Iterable[Path]) -> int:
    n = 0
    needle = ": done dt="
    for d in rep_dirs:
        log = d / "run.log"
        if not log.exists():
            continue
        try:
            with log.open("r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    # Event-level completion lines look like:
                    #   [dark_sirens] <run> <EVENT>: done dt=...
                    if needle in ln and "[dark_sirens]" in ln:
                        n += 1
        except Exception:
            continue
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Status helper for dark_siren_null_battery_* outputs.")
    ap.add_argument("out_dir", type=Path, help="Null-battery output directory.")
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if not out_dir.exists():
        print(f"missing: {out_dir}")
        return 2

    pid = _read_int(out_dir / "pid.txt")
    ps_line = _ps_status(pid) if pid is not None else None
    running = ps_line is not None

    run_log = out_dir / "run.log"
    seed_ranges = _parse_seed_ranges(run_log)
    events_per_rep = _parse_events_per_rep(run_log) if run_log.exists() else None

    print(f"out={out_dir}")
    print(f"pid={pid if pid is not None else '<missing>'} running={running}")
    if ps_line is not None:
        print(f"ps={ps_line}")

    if run_log.exists():
        tail = _tail_text(run_log, n_lines=3)
        if tail is not None:
            print("run.log tail:")
            print(tail)

    def _mode_status(label: str, mode_dir: Path, *, expected: int | None) -> None:
        rep_dirs = _rep_dirs(mode_dir)
        done_dirs, incomplete_dirs = _split_done(rep_dirs)
        n_done = int(len(done_dirs))
        n_rep = len(rep_dirs)
        target = expected if expected is not None else n_rep
        print(f"{label}: done={n_done}/{target} reps (dirs={n_rep})")
        if events_per_rep is not None and expected is not None:
            total_events = int(events_per_rep) * int(expected)
            if total_events > 0:
                # Count event-level completion only for incomplete reps (done reps might have truncated
                # logs but are guaranteed complete by summary_*.json presence).
                n_event_done_incomplete = _count_event_done(incomplete_dirs)
                n_event_done_est = int(events_per_rep) * int(n_done) + int(n_event_done_incomplete)
                if n_event_done_est > total_events:
                    n_event_done_est = total_events
                frac = float(n_event_done_est) / float(total_events)
                print(f"  events_done={n_event_done_est}/{total_events} ({100.0 * frac:.2f}%)")
        latest = _find_latest_incomplete(rep_dirs)
        if latest is None:
            return
        last_line = _tail_text(latest / "run.log", n_lines=1)
        print(f"  latest_incomplete={latest.name} last_line={last_line.strip() if last_line else '<none>'}")

    rot_expected = None
    if "rotate" in seed_ranges:
        a, b = seed_ranges["rotate"]
        rot_expected = b - a + 1
    gal_expected = None
    if "galnull" in seed_ranges:
        a, b = seed_ranges["galnull"]
        gal_expected = b - a + 1

    _mode_status("rotate_pe_sky", out_dir / "rotate_pe_sky", expected=rot_expected)
    _mode_status("galnull_shuffle_z", out_dir / "galnull_shuffle_z", expected=gal_expected)
    _mode_status("galnull_shuffle_zw", out_dir / "galnull_shuffle_zw", expected=gal_expected)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
