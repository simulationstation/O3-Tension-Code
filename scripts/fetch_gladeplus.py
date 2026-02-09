from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess


GLADEPLUS_URL_DEFAULT = "http://elysium.elte.hu/~dalyag/GLADE+.txt"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _run(cmd: list[str]) -> None:
    print(f"[fetch_gladeplus] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


@contextlib.contextmanager
def _file_lock(path: Path):
    """POSIX lock to prevent concurrent runs corrupting shared .part files."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, *, max_conns: int = 4, limit_rate: str | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = Path(str(dst) + ".part")
    lock = Path(str(tmp) + ".lock")
    with _file_lock(lock):
        if dst.exists() and dst.stat().st_size > 0:
            return

        use_aria2 = shutil.which("aria2c") is not None
        if use_aria2:
            conns = int(max_conns)
            cmd = [
                "aria2c",
                "--continue=true",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--file-allocation=none",
                "--max-connection-per-server",
                str(conns),
                "--split",
                str(conns),
                "--min-split-size",
                "50M",
                "-d",
                str(tmp.parent),
                "-o",
                tmp.name,
                url,
            ]
            if limit_rate:
                cmd.extend(["--max-overall-download-limit", str(limit_rate)])
            _run(cmd)
        else:
            cmd = ["wget", "-c", "-O", str(tmp), url]
            if limit_rate:
                cmd.insert(2, f"--limit-rate={limit_rate}")
            _run(cmd)

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {url} -> {tmp}")
        tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the GLADE+ galaxy catalog (large download).")
    ap.add_argument("--url", default=GLADEPLUS_URL_DEFAULT, help=f"GLADE+ URL (default: {GLADEPLUS_URL_DEFAULT}).")
    ap.add_argument("--out", default=None, help="Output directory (default: data/cache/galaxies/gladeplus/).")
    ap.add_argument("--max-conns", type=int, default=4, help="aria2 connections (be polite; default: 4).")
    ap.add_argument("--limit-rate", default=None, help="Optional overall download limit, e.g. '50M'.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("data") / "cache" / "galaxies" / "gladeplus"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = str(args.url)
    dst = out_dir / "GLADE+.txt"

    _download(url, dst, max_conns=int(args.max_conns), limit_rate=args.limit_rate)

    manifest = {
        "url": url,
        "downloaded_path": str(dst),
        "size_bytes": int(dst.stat().st_size),
        "sha256": _sha256(dst),
        "timestamp_utc": _utc_stamp(),
        "note": "GLADE+ is an external dataset; keep it under data/ (gitignored).",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fetch_gladeplus] done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

