from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GWTCPeAnalyticPriorSpec:
    """Parsed representation of a GWTC PEDataRelease bilby-style prior string."""

    expr: str
    class_name: str
    kwargs: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticPrior:
    """Analytic 1D prior interface (log-PDF)."""

    def logpdf(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover (interface)
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:  # pragma: no cover (interface)
        raise NotImplementedError


@dataclass(frozen=True)
class PeAnalyticDistancePrior:
    """Distance-prior wrapper built from a GWTC PEDataRelease analytic prior.

    This is used to divide out the *actual* PE sampling prior π_PE(dL) when converting a PE
    posterior density into a proxy likelihood.
    """

    file: str
    analysis: str
    spec: GWTCPeAnalyticPriorSpec
    prior: AnalyticPrior

    def log_pi_dL(self, dL_mpc: np.ndarray) -> np.ndarray:
        return self.prior.logpdf(dL_mpc)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "mode": "pe_analytic",
            "file": str(self.file),
            "analysis": str(self.analysis),
            "spec": self.spec.to_jsonable(),
        }


def _as_1d_float(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x


class UniformPrior(AnalyticPrior):
    def __init__(self, *, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum):
            raise ValueError("UniformPrior requires finite maximum > minimum.")
        self._log_norm = -float(np.log(self.maximum - self.minimum))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        out[m] = self._log_norm
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        return rng.uniform(self.minimum, self.maximum, size=n).astype(float)


class PowerLawPrior(AnalyticPrior):
    def __init__(self, *, alpha: float, minimum: float, maximum: float) -> None:
        self.alpha = float(alpha)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.alpha) and np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum > 0.0):
            raise ValueError("PowerLawPrior requires finite alpha and 0 < minimum < maximum.")

        a = self.alpha
        if np.isclose(a, -1.0):
            norm = float(np.log(self.maximum) - np.log(self.minimum))
            self._log_norm = -float(np.log(norm))
        else:
            p = a + 1.0
            norm = float((self.maximum**p - self.minimum**p) / p)
            if not (np.isfinite(norm) and norm > 0.0):
                raise ValueError("PowerLawPrior got invalid normalization.")
            self._log_norm = -float(np.log(norm))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        if np.any(m):
            xm = np.clip(x[m], 1e-300, np.inf)
            out[m] = self.alpha * np.log(xm) + self._log_norm
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        u = rng.uniform(0.0, 1.0, size=n).astype(float)
        a = float(self.alpha)
        if np.isclose(a, -1.0):
            lo = float(np.log(self.minimum))
            hi = float(np.log(self.maximum))
            return np.exp(lo + u * (hi - lo)).astype(float)
        p = a + 1.0
        lo = float(self.minimum**p)
        hi = float(self.maximum**p)
        return np.power(lo + u * (hi - lo), 1.0 / p).astype(float)


class SinePrior(AnalyticPrior):
    def __init__(self, *, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum):
            raise ValueError("SinePrior requires finite maximum > minimum.")
        norm = float(np.cos(self.minimum) - np.cos(self.maximum))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("SinePrior got invalid normalization.")
        self._log_norm = -float(np.log(norm))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        if np.any(m):
            s = np.sin(x[m])
            s = np.clip(s, 0.0, np.inf)
            out[m] = np.log(np.clip(s, 1e-300, np.inf)) + self._log_norm
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        # Assumes sin(x) >= 0 on [min,max], as in the common [0,pi] use case.
        if self.minimum < 0.0 or self.maximum > np.pi:
            raise ValueError("SinePrior sampling only supported for 0 <= min <= max <= pi.")
        cmin = float(np.cos(self.minimum))
        cmax = float(np.cos(self.maximum))
        u = rng.uniform(0.0, 1.0, size=n).astype(float)
        c = cmin - u * (cmin - cmax)
        c = np.clip(c, -1.0, 1.0)
        return np.arccos(c).astype(float)


class GaussianPrior(AnalyticPrior):
    def __init__(self, *, mu: float, sigma: float) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        if not (np.isfinite(self.mu) and np.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("GaussianPrior requires finite mu and sigma>0.")
        self._log_norm = -0.5 * float(np.log(2.0 * np.pi)) - float(np.log(self.sigma))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x)
        if np.any(m):
            t = (x[m] - self.mu) / self.sigma
            out[m] = self._log_norm - 0.5 * t**2
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        return rng.normal(loc=self.mu, scale=self.sigma, size=n).astype(float)


class DeltaFunctionPrior(AnalyticPrior):
    def __init__(self, *, peak: float, atol: float = 0.0) -> None:
        self.peak = float(peak)
        self.atol = float(atol)
        if not np.isfinite(self.peak):
            raise ValueError("DeltaFunctionPrior peak must be finite.")
        if self.atol < 0.0:
            raise ValueError("DeltaFunctionPrior atol must be non-negative.")

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        if self.atol > 0.0:
            m = np.isfinite(x) & (np.abs(x - self.peak) <= self.atol)
        else:
            m = np.isfinite(x) & (x == self.peak)
        out[m] = 0.0
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:  # noqa: ARG002
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        return np.full((n,), float(self.peak), dtype=float)


class ConstraintPrior(AnalyticPrior):
    """A hard support constraint used by bilby.

    This is not a normalized PDF; it is an indicator function:
      - logpdf = 0 inside [min,max]
      - logpdf = -inf outside
    """

    def __init__(self, *, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum):
            raise ValueError("ConstraintPrior requires finite maximum > minimum.")

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        out[m] = 0.0
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError("ConstraintPrior is not a normalized PDF; sampling is undefined.")


class UniformInComponentsChirpMassPrior(AnalyticPrior):
    """bilby `UniformInComponentsChirpMass`: p(Mc) ∝ Mc on [min,max]."""

    def __init__(self, *, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum > 0.0):
            raise ValueError("UniformInComponentsChirpMassPrior requires 0<min<max.")
        denom = float(self.maximum**2 - self.minimum**2)
        if not (np.isfinite(denom) and denom > 0.0):
            raise ValueError("UniformInComponentsChirpMassPrior got invalid normalization.")
        self._log_norm = float(np.log(2.0) - np.log(denom))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        if np.any(m):
            out[m] = np.log(np.clip(x[m], 1e-300, np.inf)) + self._log_norm
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        u = rng.uniform(0.0, 1.0, size=n).astype(float)
        lo2 = float(self.minimum**2)
        hi2 = float(self.maximum**2)
        return np.sqrt(lo2 + u * (hi2 - lo2)).astype(float)


class UniformInComponentsMassRatioPrior(AnalyticPrior):
    """bilby `UniformInComponentsMassRatio`: p(q) ∝ (1+q)^(2/5) / q^(6/5) on [min,max]."""

    def __init__(self, *, minimum: float, maximum: float, n_grid: int = 20001) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if not (np.isfinite(self.minimum) and np.isfinite(self.maximum) and self.maximum > self.minimum > 0.0):
            raise ValueError("UniformInComponentsMassRatioPrior requires 0<min<max.")
        n = int(n_grid)
        if n < 1001:
            raise ValueError("UniformInComponentsMassRatioPrior requires n_grid >= 1001.")
        q = np.linspace(self.minimum, self.maximum, n, dtype=float)
        f = (1.0 + q) ** (2.0 / 5.0) * np.clip(q, 1e-300, np.inf) ** (-6.0 / 5.0)
        norm = float(np.trapezoid(f, q))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("UniformInComponentsMassRatioPrior got invalid normalization.")
        self._log_norm = -float(np.log(norm))
        # Cache a CDF for inverse sampling (monotone by construction).
        cdf = np.cumsum(np.concatenate([[0.0], 0.5 * (f[1:] + f[:-1]) * np.diff(q)]))
        cdf = cdf / float(cdf[-1])
        self._q_grid = q
        self._cdf_grid = cdf

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_float(x)
        out = np.full_like(x, -np.inf, dtype=float)
        m = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
        if np.any(m):
            q = np.clip(x[m], 1e-300, np.inf)
            out[m] = (2.0 / 5.0) * np.log1p(q) - (6.0 / 5.0) * np.log(q) + self._log_norm
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        if n < 0:
            raise ValueError("size must be non-negative.")
        u = rng.uniform(0.0, 1.0, size=n).astype(float)
        return np.interp(u, self._cdf_grid, self._q_grid).astype(float)


def parse_gwtc_analytic_prior(expr: str) -> tuple[GWTCPeAnalyticPriorSpec, AnalyticPrior]:
    """Parse a GWTC bilby prior string into a log-PDF evaluator for the supported subset."""
    expr = str(expr).strip()
    node = ast.parse(expr, mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Unsupported prior expression (expected call): {expr!r}")
    if not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported prior expression function: {expr!r}")
    class_name = str(node.func.id)
    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError(f"Unsupported **kwargs prior expression: {expr!r}")
        kwargs[str(kw.arg)] = ast.literal_eval(kw.value)

    spec = GWTCPeAnalyticPriorSpec(expr=expr, class_name=class_name, kwargs=kwargs)

    # Only a small subset is needed for reweighting in hierarchical PE likelihoods.
    if class_name == "Uniform":
        prior = UniformPrior(minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    elif class_name == "PowerLaw":
        prior = PowerLawPrior(alpha=float(kwargs["alpha"]), minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    elif class_name == "Sine":
        prior = SinePrior(minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    elif class_name == "Gaussian":
        prior = GaussianPrior(mu=float(kwargs["mu"]), sigma=float(kwargs["sigma"]))
    elif class_name == "DeltaFunction":
        # Use an absolute tolerance to handle minor float serialization differences.
        prior = DeltaFunctionPrior(peak=float(kwargs["peak"]), atol=1e-12 * max(1.0, abs(float(kwargs["peak"]))))
    elif class_name == "Constraint":
        prior = ConstraintPrior(minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    elif class_name == "UniformInComponentsChirpMass":
        prior = UniformInComponentsChirpMassPrior(minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    elif class_name == "UniformInComponentsMassRatio":
        prior = UniformInComponentsMassRatioPrior(minimum=float(kwargs["minimum"]), maximum=float(kwargs["maximum"]))
    else:
        raise ValueError(f"Unsupported analytic prior class '{class_name}' in expr: {expr!r}")

    return spec, prior


def load_gwtc_pe_analytic_priors(
    *,
    path: str | Path,
    analysis: str,
    parameters: list[str],
) -> dict[str, tuple[GWTCPeAnalyticPriorSpec, AnalyticPrior]]:
    """Load and parse `analysis/priors/analytic/<param>` bilby-style prior strings for given params."""
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    parameters = [str(p) for p in parameters]
    if not parameters:
        raise ValueError("parameters must be non-empty.")

    out: dict[str, tuple[GWTCPeAnalyticPriorSpec, AnalyticPrior]] = {}
    with h5py.File(path, "r") as f:
        if analysis not in f:
            keys = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
            raise KeyError(f"{path}: analysis group '{analysis}' not found. Available: {keys}")
        g = f[analysis]
        if "priors" not in g or "analytic" not in g["priors"]:
            raise KeyError(f"{path}: missing '{analysis}/priors/analytic' group.")
        pri = g["priors/analytic"]
        for p in parameters:
            if p not in pri:
                raise KeyError(f"{path}: missing analytic prior for '{p}' under '{analysis}/priors/analytic'.")
            raw = pri[p][()]
            if not (isinstance(raw, np.ndarray) and raw.size >= 1):
                raise ValueError(f"{path}: unexpected analytic prior dataset format for '{p}'.")
            s = raw[0]
            if isinstance(s, (bytes, np.bytes_)):
                s = s.decode("utf-8", "ignore")
            spec, prior = parse_gwtc_analytic_prior(str(s))
            out[p] = (spec, prior)
    return out


def select_gwtc_pe_analysis_with_analytic_priors(
    *,
    path: str | Path,
    prefer: list[str],
    require_parameters: list[str],
) -> str:
    """Pick the first analysis group that has non-empty `priors/analytic` entries for the required params."""
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    prefer = [str(a) for a in prefer]
    require_parameters = [str(p) for p in require_parameters]

    with h5py.File(path, "r") as f:
        analyses = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
        if not analyses:
            raise ValueError(f"{path}: no analysis groups found.")

        candidates = [a for a in prefer if a in analyses] + [a for a in analyses if a not in prefer]
        for a in candidates:
            try:
                g = f[a]
                if "priors" not in g or "analytic" not in g["priors"]:
                    continue
                pri = g["priors/analytic"]
                ok = True
                for p in require_parameters:
                    if p not in pri:
                        ok = False
                        break
                    raw = pri[p][()]
                    if not (isinstance(raw, np.ndarray) and raw.size >= 1):
                        ok = False
                        break
                    s = raw[0]
                    if isinstance(s, (bytes, np.bytes_)):
                        s = s.decode("utf-8", "ignore")
                    if not str(s).strip():
                        ok = False
                        break
                if ok:
                    return a
            except Exception:
                continue

    raise ValueError(f"{path}: no analysis group found with analytic priors for {require_parameters}.")
