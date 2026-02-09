from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TsallisParams:
    delta: float
    log_mu0: float
    logA0: float


def log_mu_tsallis(logA: np.ndarray, params: TsallisParams) -> np.ndarray:
    return params.log_mu0 - (params.delta - 1.0) * (logA - params.logA0)


@dataclass(frozen=True)
class BarrowParams:
    Delta: float
    log_mu0: float
    logA0: float


def log_mu_barrow(logA: np.ndarray, params: BarrowParams) -> np.ndarray:
    delta_eff = 1.0 + params.Delta / 2.0
    return params.log_mu0 - (delta_eff - 1.0) * (logA - params.logA0)


@dataclass(frozen=True)
class KaniadakisParams:
    beta_tilde: float
    log_mu0: float
    A_ref: float


def mu_kaniadakis(A: np.ndarray, params: KaniadakisParams) -> np.ndarray:
    """Kaniadakis-inspired μ(A) model: μ ∝ 1/cosh(β A/A_ref).

    This corresponds to the commonly used horizon entropy ansatz:

      S_κ = (1/κ) sinh(κ S_BH),  with S_BH ∝ A

    implying dS/dA ∝ cosh(κ S_BH) and therefore μ(A) = (dS/dA)_BH/(dS/dA) ∝ 1/cosh(κ S_BH).

    We absorb constants into a dimensionless deformation parameter `beta_tilde` by scaling A with A_ref.
    """
    A = np.asarray(A, dtype=float)
    x = params.beta_tilde * (A / params.A_ref)
    # logcosh(x) = log((e^x + e^{-x})/2) = logaddexp(x,-x) - log(2)
    log_cosh = np.logaddexp(x, -x) - np.log(2.0)
    return np.exp(params.log_mu0 - log_cosh)


def log_mu_kaniadakis(logA: np.ndarray, params: KaniadakisParams) -> np.ndarray:
    A = np.exp(np.asarray(logA, dtype=float))
    x = params.beta_tilde * (A / params.A_ref)
    log_cosh = np.logaddexp(x, -x) - np.log(2.0)
    return params.log_mu0 - log_cosh
