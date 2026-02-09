from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_horizon_recon.void_prism_maps import eg_void_from_spectra


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert measured void-prism cross spectra to an E_G measurement JSON.")
    ap.add_argument("--spectra-npz", required=True, help="Input .npz with ell, cl_kappa_void, cl_theta_void.")
    ap.add_argument("--name", default=None, help="Measurement name (default: derived from npz filename).")
    ap.add_argument("--z-eff", type=float, required=True, help="Effective redshift for this measurement bin.")
    ap.add_argument("--prefactor", type=float, default=1.0, help="Overall normalization constant (definition-dependent).")
    ap.add_argument(
        "--eg-sigma",
        type=float,
        default=None,
        help="Constant 1-sigma error to attach to each ell bin (placeholder; real use should supply a covariance).",
    )
    ap.add_argument(
        "--eg-sigma-frac",
        type=float,
        default=None,
        help="If set, uses eg_sigma = eg_sigma_frac * |eg_obs| per bin.",
    )
    ap.add_argument("--out", default=None, help="Output JSON path (default: outputs/void_prism_eg_meas_<UTCSTAMP>.json).")
    args = ap.parse_args()

    npz = Path(args.spectra_npz)
    with np.load(npz, allow_pickle=True) as d:
        ell = np.asarray(d["ell"], dtype=int)
        cl_kv = np.asarray(d["cl_kappa_void"], dtype=float)
        cl_tv = np.asarray(d["cl_theta_void"], dtype=float)

    eg = eg_void_from_spectra(cl_kappa_void=cl_kv, cl_theta_void=cl_tv, prefactor=float(args.prefactor))

    if args.eg_sigma is not None:
        sig = np.full_like(eg, float(args.eg_sigma), dtype=float)
    elif args.eg_sigma_frac is not None:
        sig = float(args.eg_sigma_frac) * np.abs(eg)
    else:
        raise ValueError("Provide --eg-sigma or --eg-sigma-frac (this script does not estimate covariance).")

    name = args.name or npz.stem
    out = Path(args.out) if args.out else Path("outputs") / f"void_prism_eg_meas_{_utc_stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "measurements": [
            {
                "name": str(name),
                "z_eff": float(args.z_eff),
                "ell": ell.tolist(),
                "eg_obs": eg.tolist(),
                "eg_sigma": sig.tolist(),
                "notes": "Derived from spectra with placeholder errors. For a publication-grade test, provide a real covariance.",
                "source": {"kind": "derived", "spectra_npz": str(npz)},
            }
        ]
    }
    out.write_text(json.dumps(payload, indent=2))
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

