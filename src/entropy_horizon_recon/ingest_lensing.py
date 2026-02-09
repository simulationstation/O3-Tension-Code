from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tarfile

import pandas as pd
import pooch

from .cache import DataPaths, make_pooch


@dataclass(frozen=True)
class PlanckLensingProxy:
    """A lightweight compressed Planck CMB lensing constraint."""

    # Constraint on sigma8 * Omega_m^0.25 (Planck lensing reconstruction).
    sigma8_om025_mean: float
    sigma8_om025_sigma: float
    meta: dict


# Match the LaTeX source line like: "\sigma_8 \Omm^{0.25} = 0.589\pm 0.020"
_RE_OM025 = re.compile(r"\\sigma_8\s+\\Omm\^\{0\.25\}\s*=\s*([0-9]*\.?[0-9]+)\\pm\s*([0-9]*\.?[0-9]+)")


def load_planck_lensing_proxy(*, paths: DataPaths, source: str = "planck2018_params_paper") -> PlanckLensingProxy:
    """Load a compressed Planck lensing constraint.

    This defaults to parsing the Planck 2018 cosmological-parameters paper source
    (arXiv:1807.06209) for the lensing-reconstruction constraint:

        sigma8 * Omega_m^0.25 = 0.589 +/- 0.020

    We use this as a late-time clustering/potential constraint proxy without
    introducing Planck TT/TE/EE distance priors.
    """
    if source != "planck2018_params_paper":
        raise ValueError("Only source='planck2018_params_paper' is currently supported.")

    processed_dir = paths.processed_dir / "lensing"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_json = processed_dir / "planck2018_lensing_proxy.json"

    if processed_json.exists():
        obj = json.loads(processed_json.read_text(encoding="utf-8"))
        return PlanckLensingProxy(
            sigma8_om025_mean=float(obj["sigma8_om025_mean"]),
            sigma8_om025_sigma=float(obj["sigma8_om025_sigma"]),
            meta=dict(obj.get("meta", {})),
        )

    base_url = "https://arxiv.org/e-print/"
    registry = {
        "1807.06209": "sha256:868a49562c9550e510d7446739aa36f9ece5c72a9d5e2f7248eb4772dd11e18b",
    }
    p = make_pooch(cache_dir=paths.pooch_cache_dir, base_url=base_url, registry=registry)
    downloader = pooch.HTTPDownloader(timeout=600)
    src_path = Path(p.fetch("1807.06209", downloader=downloader))

    # Parse ms.tex from the arXiv source tarball.
    ms_tex = None
    with tarfile.open(src_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if name.endswith("ms.tex"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                ms_tex = f.read().decode("utf-8", errors="replace")
                break
    if ms_tex is None:
        raise RuntimeError("Failed to find ms.tex in the Planck arXiv source tarball.")

    m = _RE_OM025.search(ms_tex)
    if not m:
        raise RuntimeError("Failed to locate the sigma8*Omega_m^0.25 lensing proxy line in ms.tex.")
    mean = float(m.group(1))
    sigma = float(m.group(2))

    meta = {
        "source": "arXiv:1807.06209 (Planck 2018 cosmological parameters paper source tarball)",
        "file": "1807.06209:ms.tex",
        "parsed_regex": _RE_OM025.pattern,
        "quantity": "sigma8 * Omega_m^0.25",
        "note": "Used as a minimal Gaussian proxy for Planck CMB lensing reconstruction.",
    }
    out = {"sigma8_om025_mean": mean, "sigma8_om025_sigma": sigma, "meta": meta}
    processed_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Also mirror into data/cache/ for inspection.
    cache_json = paths.pooch_cache_dir / "planck_lensing_proxy.json"
    cache_meta = paths.pooch_cache_dir / "planck_lensing_proxy.meta.json"
    cache_json.write_text(json.dumps({"sigma8_om025_mean": mean, "sigma8_om025_sigma": sigma}, indent=2), encoding="utf-8")
    pd.Series(meta).to_json(cache_meta, indent=2)

    return PlanckLensingProxy(sigma8_om025_mean=mean, sigma8_om025_sigma=sigma, meta=meta)
