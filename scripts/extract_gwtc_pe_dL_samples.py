from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.gwtc_pe_index import build_gwtc_pe_index
from entropy_horizon_recon.gwtc_pe_samples import extract_parameter_samples, list_analyses


def _pick_file_for_event(
    *,
    base_dir: Path,
    event: str,
    record_ids: list[int] | None,
    prefer_variants: list[str],
) -> Path:
    idx = build_gwtc_pe_index(base_dir=base_dir, record_ids=record_ids)
    if event not in idx:
        raise KeyError(f"Event '{event}' not found under {base_dir}.")
    files = idx[event]

    # Prefer variants in order (e.g. mixed_nocosmo first).
    for v in prefer_variants:
        for f in files:
            if f.variant == v:
                return (base_dir / f.path).resolve()

    # Otherwise take the first deterministically (index is already sorted).
    return (base_dir / files[0].path).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract luminosity-distance samples from GWTC PEDataRelease HDF5 files.")
    ap.add_argument("--file", dest="pe_file", default=None, help="Direct path to a PEDataRelease .h5/.hdf5 file.")
    ap.add_argument("--event", default=None, help="GW event name like GW190412_053044 (resolved via local Zenodo cache).")
    ap.add_argument("--base-dir", default="data/cache/gw/zenodo", help="Base dir containing Zenodo record subdirs.")
    ap.add_argument("--record-id", action="append", default=None, help="Restrict file search to these record IDs (repeatable).")
    ap.add_argument(
        "--prefer-variant",
        action="append",
        default=["mixed_nocosmo", "combined", "mixed_cosmo"],
        help="Variant preference order (repeatable). Default: mixed_nocosmo, combined, mixed_cosmo.",
    )

    ap.add_argument("--analysis", default=None, help="Analysis group label (e.g. C01:Mixed or C00:Mixed). Default: auto.")
    ap.add_argument("--list-analyses", action="store_true", help="List available analyses in the selected file and exit.")
    ap.add_argument("--parameter", default="luminosity_distance", help="Parameter field to extract (default luminosity_distance).")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional downsample cap (random, deterministic).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for downsampling (default 0).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output .npz path (default: data/cache/gw/pe_samples/<event>_<analysis>_<param>.npz).",
    )
    ap.add_argument(
        "--out-key",
        default=None,
        help="Array key used in the output npz (default: dL_samples_Mpc for luminosity_distance, else '<parameter>').",
    )
    args = ap.parse_args()

    if (args.pe_file is None) == (args.event is None):
        raise ValueError("Provide exactly one of --file or --event.")

    base_dir = Path(args.base_dir).expanduser().resolve()
    record_ids = [int(x) for x in (args.record_id or [])] or None

    if args.pe_file is not None:
        pe_path = Path(args.pe_file).expanduser().resolve()
    else:
        pe_path = _pick_file_for_event(
            base_dir=base_dir,
            event=str(args.event),
            record_ids=record_ids,
            prefer_variants=[str(v) for v in (args.prefer_variant or [])],
        )

    if bool(args.list_analyses):
        print(json.dumps({"file": str(pe_path), "analyses": list_analyses(pe_path)}, indent=2))
        return 0

    samples, meta = extract_parameter_samples(
        path=pe_path,
        analysis=str(args.analysis) if args.analysis is not None else None,
        parameter=str(args.parameter),
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
        seed=int(args.seed),
    )

    out_key = args.out_key
    if out_key is None:
        out_key = "dL_samples_Mpc" if str(args.parameter) == "luminosity_distance" else str(args.parameter)

    if args.out is not None:
        out = Path(args.out).expanduser().resolve()
    else:
        event = str(args.event) if args.event is not None else pe_path.name
        safe_analysis = meta.analysis.replace(":", "_")
        out = Path("data/cache/gw/pe_samples") / f"{event}__{safe_analysis}__{meta.parameter}.npz"
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out, **{out_key: samples}, meta=json.dumps(meta.to_jsonable(), sort_keys=True))
    print(f"Wrote {out} ({out_key}, n={samples.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

