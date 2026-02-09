from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from entropy_horizon_recon.gwtc_pe_index import build_gwtc_pe_index, summarize_gwtc_pe_index


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a lightweight index of locally downloaded GWTC PE posterior files.")
    ap.add_argument(
        "--base-dir",
        default="data/cache/gw/zenodo",
        help="Base directory containing Zenodo record subdirectories (default: data/cache/gw/zenodo).",
    )
    ap.add_argument(
        "--record-id",
        action="append",
        default=None,
        help="Zenodo record id to scan (repeatable). If omitted, scans all numeric subdirs under base-dir.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: outputs/gwtc_pe_index_<UTCSTAMP>/gwtc_pe_index.json).",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    record_ids = [int(x) for x in (args.record_id or [])]
    index = build_gwtc_pe_index(base_dir=base_dir, record_ids=record_ids or None)
    summary = summarize_gwtc_pe_index(index)

    out = Path(args.out) if args.out else Path("outputs") / f"gwtc_pe_index_{_utc_stamp()}" / "gwtc_pe_index.json"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(
            {
                "timestamp_utc": _utc_stamp(),
                "base_dir": str(base_dir),
                "record_ids": record_ids or None,
                "summary": summary,
                "index": {k: [x.to_jsonable() for x in v] for k, v in index.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

