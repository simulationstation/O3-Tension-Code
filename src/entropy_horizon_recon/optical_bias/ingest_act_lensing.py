from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActLensing:
    kappa_map: np.ndarray
    mask: np.ndarray | None
    nside: int
    meta: dict[str, str]


def load_act_kappa(*args, **kwargs) -> ActLensing:  # pragma: no cover - optional dataset
    raise RuntimeError(
        "ACT lensing ingest not configured. Provide a public ACT kappa map URL and SHA256."
    )
