from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import urllib.request


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _run(cmd: list[str]) -> None:
    print(f"[fetch_skymaps] $ {' '.join(cmd)}", flush=True)
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


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_zenodo_record(record_id: int) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _download(url: str, dst: Path, *, expected_md5: str | None, max_conns: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = Path(str(dst) + ".part")
    lock = Path(str(tmp) + ".lock")
    with _file_lock(lock):
        if dst.exists() and dst.stat().st_size > 0:
            return

        use_aria2 = (shutil.which("aria2c") is not None) and ("zenodo.org/" not in url)
        if use_aria2:
            # Keep it modest to avoid hammering Zenodo; tune up only if needed.
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
                "20M",
                "-d",
                str(tmp.parent),
                "-o",
                tmp.name,
                url,
            ]
            try:
                _run(cmd)
            except subprocess.CalledProcessError as e:
                # Some hosts block aria2's default UA; fall back to wget (still resumable).
                print(f"[fetch_skymaps] aria2 failed (exit {e.returncode}); falling back to wget", flush=True)
                _run(["wget", "-c", "-O", str(tmp), url])
        else:
            _run(["wget", "-c", "-O", str(tmp), url])

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {url} -> {tmp}")
        tmp.replace(dst)

    if expected_md5:
        got = _md5(dst)
        if got.lower() != expected_md5.lower():
            raise RuntimeError(f"MD5 mismatch for {dst}: expected {expected_md5}, got {got}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch GW sky maps tarballs from Zenodo (GWTC-3/GWTC-4, etc).")
    ap.add_argument("--record-id", type=int, default=5546663, help="Zenodo record ID (default: 5546663, GWTC-3 skymaps).")
    ap.add_argument("--file", default=None, help="Filename inside the Zenodo record (default: auto-detect skymaps tarball).")
    ap.add_argument("--out", default=None, help="Output directory (default: data/cache/gw/zenodo/<record-id>/).")
    ap.add_argument("--extract", action="store_true", help="Extract the tarball into <out>/extracted/ (idempotent).")
    ap.add_argument("--max-conns", type=int, default=4, help="aria2 connections (be polite; default: 4).")
    args = ap.parse_args()

    record_id = int(args.record_id)
    out_dir = Path(args.out) if args.out else Path("data") / "cache" / "gw" / "zenodo" / str(record_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = _fetch_zenodo_record(record_id)
    files = rec.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {record_id} has no files field.")

    want = args.file
    if want is None:
        # Prefer the largest "skymap(s)" tarball.
        cand = [f for f in files if "skymap" in f["key"].lower() and f["key"].endswith(".tar.gz")]
        if not cand:
            raise RuntimeError(f"Zenodo record {record_id}: could not auto-detect a skymaps tarball.")
        want = sorted(cand, key=lambda f: int(f.get("size", 0)), reverse=True)[0]["key"]

    meta = None
    for f in files:
        if f["key"] == want:
            meta = f
            break
    if meta is None:
        raise RuntimeError(f"Zenodo record {record_id}: file '{want}' not found.")

    url = meta["links"]["self"]
    checksum = str(meta.get("checksum", ""))
    expected_md5 = None
    if checksum.startswith("md5:"):
        expected_md5 = checksum.split(":", 1)[1]

    dst = out_dir / want
    _download(url, dst, expected_md5=expected_md5, max_conns=int(args.max_conns))

    # Save a small manifest for reproducibility.
    manifest = {
        "record_id": record_id,
        "title": rec.get("metadata", {}).get("title"),
        "file": want,
        "url": url,
        "size_bytes": int(meta.get("size", 0)),
        "checksum": checksum,
        "downloaded_path": str(dst),
        "sha256": _sha256(dst),
        "timestamp_utc": _utc_stamp(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.extract:
        extract_dir = out_dir / "extracted"
        marker = extract_dir / ".extracted.ok"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            _run(["tar", "-xzf", str(dst), "-C", str(extract_dir)])
            marker.write_text(_utc_stamp() + "\n")

    print(f"[fetch_skymaps] done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
