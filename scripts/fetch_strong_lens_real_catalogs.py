#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.strong_lens_time_delay import (
    fetch_h0licow_distance_catalog,
    fetch_tdcosmo2025_chain_release,
    fetch_tdcosmo_sample_posteriors,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize(paths: dict[str, Path]) -> dict[str, object]:
    out: dict[str, object] = {
        "n_files": int(len(paths)),
        "total_bytes": int(sum(p.stat().st_size for p in paths.values())),
        "files": {},
    }
    files = out["files"]
    assert isinstance(files, dict)
    for name, path in sorted(paths.items()):
        files[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Download real public strong-lens datasets (H0LiCOW + TDCOSMO2025).")
    ap.add_argument("--repo-root", type=str, default=None, help="Repo root (default: inferred from script path).")
    ap.add_argument("--skip-h0licow", action="store_true")
    ap.add_argument("--skip-tdcosmo2025", action="store_true")
    ap.add_argument("--skip-tdcosmo-sample", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    out: dict[str, object] = {"repo_root": str(repo_root)}

    if not args.skip_h0licow:
        h0licow = fetch_h0licow_distance_catalog(paths)
        out["h0licow_distance_files"] = _summarize(h0licow)
    if not args.skip_tdcosmo2025:
        td = fetch_tdcosmo2025_chain_release(paths)
        out["tdcosmo2025_chain_files"] = _summarize(td)
    if not args.skip_tdcosmo_sample:
        td_sample = fetch_tdcosmo_sample_posteriors(paths)
        out["tdcosmo_sample_posteriors"] = _summarize(td_sample)

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
