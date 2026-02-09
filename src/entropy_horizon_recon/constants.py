from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used throughout the pipeline.

    Units:
      - c_km_s: km / s
      - G_Mpc_km2_s2_Msun: Mpc (km/s)^2 / Msun
    """

    c_km_s: float = 299_792.458

