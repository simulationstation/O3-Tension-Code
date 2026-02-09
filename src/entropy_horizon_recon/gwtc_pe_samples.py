from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GWTCPeParameterSamples:
    """A 1D posterior-sample vector extracted from a GWTC PEDataRelease HDF5 file."""

    file: str
    analysis: str
    parameter: str
    n_total: int
    n_used: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _default_analysis_preference(path: Path) -> list[str]:
    # These are the most common group labels across GWTC releases.
    if path.suffix == ".h5":
        return [
            "C01:Mixed",
            "C01:IMRPhenomXPHM",
            "C01:SEOBNRv4PHM",
        ]
    if path.suffix == ".hdf5":
        return [
            "C00:Mixed",
            "C00:Mixed+XO4a",
            "C00:IMRPhenomXPHM-SpinTaylor",
            "C00:SEOBNRv5PHM",
            "C00:NRSur7dq4",
            "C00:IMRPhenomXO4a",
        ]
    return []


def list_analyses(path: str | Path) -> list[str]:
    """List analysis group labels present at the top level of a GWTC PE file."""
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    with h5py.File(path, "r") as f:
        return [str(k) for k in f.keys() if str(k) not in ("history", "version")]


def extract_parameter_samples(
    *,
    path: str | Path,
    parameter: str,
    analysis: str | None = None,
    analysis_prefer: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, GWTCPeParameterSamples]:
    """Extract a single parameter's posterior samples from a GWTC PEDataRelease HDF5 file.

    The GWTC PE releases store samples in a structured HDF5 dataset named `posterior_samples`
    under an analysis group (e.g. `C01:Mixed` or `C00:Mixed`).
    """
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")

    prefer = list(analysis_prefer) if analysis_prefer is not None else _default_analysis_preference(path)

    with h5py.File(path, "r") as f:
        if analysis is None:
            # Pick first available from preference list.
            for cand in prefer:
                if cand in f:
                    analysis = cand
                    break
        if analysis is None:
            # Fall back to the first non-meta group.
            keys = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
            if not keys:
                raise ValueError(f"{path}: no analysis groups found.")
            analysis = keys[0]

        if analysis not in f:
            raise KeyError(f"{path}: analysis group '{analysis}' not found. Available: {list_analyses(path)}")

        dset = f[analysis]["posterior_samples"]
        # Structured dataset fields are accessible by name without loading all columns.
        try:
            x = np.asarray(dset[parameter], dtype=float)
        except Exception as e:
            fields = getattr(getattr(dset, "dtype", None), "names", None)
            raise KeyError(f"{path}: parameter '{parameter}' not found in {analysis}/posterior_samples (fields={fields}).") from e

    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError(f"{path}: too few finite samples for '{parameter}' in '{analysis}' ({x.size}).")

    n_total = int(x.size)
    if max_samples is not None and x.size > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(x.size, size=int(max_samples), replace=False)
        x = x[idx]

    meta = GWTCPeParameterSamples(
        file=str(path),
        analysis=str(analysis),
        parameter=str(parameter),
        n_total=n_total,
        n_used=int(x.size),
    )
    return x, meta

