from __future__ import annotations

import json
import math
import pickle
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from sklearn.neighbors import KernelDensity

from .cache import DataPaths

# Real public strong-lens distance products from H0LiCOW-public.
H0LICOW_DISTANCE_URLS: dict[str, str] = {
    "HE0435_Ddt_AO+HST.dat": "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/h0licow_distance_chains/HE0435_Ddt_AO+HST.dat",
    "J1206_final.csv": "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/h0licow_distance_chains/J1206_final.csv",
    "PG1115_AO+HST_Dd_Ddt.dat": "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/h0licow_distance_chains/PG1115_AO+HST_Dd_Ddt.dat",
    "RXJ1131_AO+HST_Dd_Ddt.dat": "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/h0licow_distance_chains/RXJ1131_AO+HST_Dd_Ddt.dat",
    "wfi2033_dt_bic.dat": "https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/master/h0licow_distance_chains/wfi2033_dt_bic.dat",
}

# TDCOSMO 2025 chain release (real public HDF5 chains).
TDCOSMO_2025_CHAIN_NAMES: tuple[str, ...] = (
    "LambdaCDM1a.h5",
    "LambdaCDM1b.h5",
    "LambdaCDM1c.h5",
    "LambdaCDM1d.h5",
    "LambdaCDM2a.h5",
    "LambdaCDM2b.h5",
    "LambdaCDM2c.h5",
    "LambdaCDM2d.h5",
    "LambdaCDM3a.h5",
    "LambdaCDM3b.h5",
    "ULambdaCDM1.h5",
    "ULambdaCDM2.h5",
    "ULambdaCDM3.h5",
    "ULambdaCDM4.h5",
    "UoLambdaCDM.h5",
    "UwCDM.h5",
    "Uw_0w_aCDM.h5",
    "Uw_phiCDM.h5",
    "oLambdaCDM.h5",
    "wCDM1.h5",
    "wCDM2.h5",
    "wCDM3.h5",
    "w_0w_aCDM1.h5",
    "w_0w_aCDM2.h5",
    "w_0w_aCDM3.h5",
    "w_0w_aCDM4.h5",
    "w_phiCDM1.h5",
    "w_phiCDM2.h5",
)

# TDCOSMO sample per-lens processed posteriors (includes STRIDES-era DES0408 and newer WGD2038).
TDCOSMO_SAMPLE_PKL_NAMES: tuple[str, ...] = (
    "B1608+656_const_processed.pkl",
    "HE0435-1223_const_processed.pkl",
    "PG1115+080_const_processed.pkl",
    "RXJ1131-1231_const_processed.pkl",
    "SDSS1206+4332_const_processed.pkl",
    "WFI2033-4723_const_processed.pkl",
    "DES0408-5354_const_processed.pkl",
    "WGD2038-4008_const_processed.pkl",
)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _download_if_missing(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded empty file from {url}")
    tmp.replace(dst)


def _logmeanexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 0:
        return float("-inf")
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("-inf")
    m = float(np.max(finite))
    return float(m + np.log(np.mean(np.exp(finite - m))))


def fetch_h0licow_distance_catalog(paths: DataPaths) -> dict[str, Path]:
    cache_dir = paths.pooch_cache_dir / "strong_lens" / "h0licow_distance_chains"
    out: dict[str, Path] = {}
    for fname, url in H0LICOW_DISTANCE_URLS.items():
        dst = cache_dir / fname
        _download_if_missing(url, dst)
        out[fname] = dst
    _write_json_atomic(
        cache_dir / "manifest.json",
        {
            "source": "shsuyu/H0LiCOW-public",
            "files": {k: str(v) for k, v in out.items()},
        },
    )
    return out


def fetch_tdcosmo2025_chain_release(paths: DataPaths) -> dict[str, Path]:
    base = "https://raw.githubusercontent.com/TDCOSMO/TDCOSMO2025_public/main/chains_export/"
    cache_dir = paths.pooch_cache_dir / "strong_lens" / "tdcosmo2025" / "chains_export"
    out: dict[str, Path] = {}
    for fname in TDCOSMO_2025_CHAIN_NAMES:
        dst = cache_dir / fname
        _download_if_missing(base + fname, dst)
        out[fname] = dst
    _write_json_atomic(
        cache_dir.parent / "manifest.json",
        {
            "source": "TDCOSMO/TDCOSMO2025_public",
            "files": {k: str(v) for k, v in out.items()},
        },
    )
    return out


def fetch_tdcosmo_sample_posteriors(paths: DataPaths) -> dict[str, Path]:
    base = "https://raw.githubusercontent.com/TDCOSMO/TDCOSMO2025_public/main/TDCOSMO_sample/"
    cache_dir = paths.pooch_cache_dir / "strong_lens" / "tdcosmo_sample"
    out: dict[str, Path] = {}
    for fname in TDCOSMO_SAMPLE_PKL_NAMES:
        dst = cache_dir / fname
        _download_if_missing(base + fname, dst)
        out[fname] = dst
    _write_json_atomic(
        cache_dir / "manifest.json",
        {
            "source": "TDCOSMO/TDCOSMO2025_public/TDCOSMO_sample",
            "files": {k: str(v) for k, v in out.items()},
        },
    )
    return out


def _loglike_sklogn(x: float, *, mu: float, sigma: float, lam: float, explim: float = 100.0) -> float:
    if not np.isfinite(x) or x <= lam:
        return float("-inf")
    arg = ((-mu + math.log(x - lam)) ** 2) / (2.0 * sigma * sigma)
    if not np.isfinite(arg) or arg > explim:
        return float("-inf")
    val = math.exp(-arg) / (math.sqrt(2.0 * math.pi) * (x - lam) * sigma)
    if not np.isfinite(val) or val <= 0.0:
        return float("-inf")
    return float(math.log(val))


class _KDE1D:
    def __init__(
        self,
        values: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        bandwidth: float = 20.0,
        hist_bins: int | None = None,
    ) -> None:
        values = np.asarray(values, dtype=float).reshape(-1)
        good = np.isfinite(values)
        values = values[good]
        w = None
        if weights is not None:
            ww = np.asarray(weights, dtype=float).reshape(-1)
            ww = ww[good]
            ww = np.clip(ww, 0.0, np.inf)
            if np.any(ww > 0):
                w = ww
        if values.size == 0:
            raise ValueError("No valid values for KDE1D.")

        if hist_bins is not None:
            hist, edges = np.histogram(values, bins=int(hist_bins), weights=w)
            centers = 0.5 * (edges[:-1] + edges[1:])
            keep = hist > 0
            x = centers[keep].reshape(-1, 1)
            sample_weight = hist[keep].astype(float)
        else:
            x = values.reshape(-1, 1)
            sample_weight = w

        self._kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
        self._kde.fit(x, sample_weight=sample_weight)

    def logpdf(self, value: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            arr = np.array([[float(values)]], dtype=float)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
                return float(self._kde.score_samples(arr)[0])
        arr = values.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
            out = self._kde.score_samples(arr)
        return out.reshape(values.shape)


class _KDE2D:
    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        bandwidth: float = 20.0,
        hist_bins: int | None = None,
    ) -> None:
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        if x_values.size != y_values.size:
            raise ValueError("x and y samples must have same length.")
        good = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[good]
        y_values = y_values[good]
        w = None
        if weights is not None:
            ww = np.asarray(weights, dtype=float).reshape(-1)
            ww = ww[good]
            ww = np.clip(ww, 0.0, np.inf)
            if np.any(ww > 0):
                w = ww
        if x_values.size == 0:
            raise ValueError("No valid values for KDE2D.")

        if hist_bins is not None:
            hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=int(hist_bins), weights=w)
            x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
            xc, yc = np.meshgrid(x_cent, y_cent, indexing="ij")
            ww = hist.reshape(-1)
            keep = ww > 0
            pts = np.column_stack([xc.reshape(-1)[keep], yc.reshape(-1)[keep]])
            sample_weight = ww[keep].astype(float)
        else:
            pts = np.column_stack([x_values, y_values])
            sample_weight = w

        self._kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
        self._kde.fit(pts, sample_weight=sample_weight)

    def logpdf(self, x_value: float, y_value: float) -> float:
        arr = np.array([[float(x_value), float(y_value)]], dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
            return float(self._kde.score_samples(arr)[0])


@dataclass(frozen=True)
class LensPrediction:
    dd_mpc: float
    ds_mpc: float
    dds_mpc: float
    ddt_mpc: float


class H0LiCOW6Likelihood:
    """Real 6-lens time-delay likelihood based on public H0LiCOW distance products."""

    def __init__(
        self,
        files: dict[str, Path],
        *,
        bandwidth_1d: float = 20.0,
        bandwidth_2d: float = 20.0,
        bins_1d: int | None = 400,
        bins_2d: int | None = 80,
        wfi_dt_max: float = 8000.0,
        j1206_bins_2d: int | None = None,
    ) -> None:
        self.bandwidth_1d = float(bandwidth_1d)
        self.bandwidth_2d = float(bandwidth_2d)
        self.bins_1d = None if bins_1d is None else int(bins_1d)
        self.bins_2d = None if bins_2d is None else int(bins_2d)
        self.wfi_dt_max = float(wfi_dt_max)
        self.j1206_bins_2d = None if j1206_bins_2d is None else int(j1206_bins_2d)

        he0435 = pd.read_csv(files["HE0435_Ddt_AO+HST.dat"], delimiter=" ", skiprows=1, names=("ddt",))
        self._he0435 = _KDE1D(
            he0435["ddt"].to_numpy(),
            bandwidth=self.bandwidth_1d,
            hist_bins=self.bins_1d,
        )

        j1206 = pd.read_csv(files["J1206_final.csv"])
        if "ddt" not in j1206 or "dd" not in j1206:
            raise ValueError("J1206_final.csv missing required columns ddt, dd.")
        self._j1206 = _KDE2D(
            j1206["dd"].to_numpy(),
            j1206["ddt"].to_numpy(),
            bandwidth=self.bandwidth_2d,
            hist_bins=self.j1206_bins_2d,  # default None = use full released samples
        )

        pg1115 = pd.read_csv(files["PG1115_AO+HST_Dd_Ddt.dat"], delimiter=" ", skiprows=1, names=("dd", "ddt"))
        self._pg1115 = _KDE2D(
            pg1115["dd"].to_numpy(),
            pg1115["ddt"].to_numpy(),
            bandwidth=self.bandwidth_2d,
            hist_bins=self.bins_2d,
        )

        rxj1131 = pd.read_csv(files["RXJ1131_AO+HST_Dd_Ddt.dat"], delimiter=" ", skiprows=1, names=("dd", "ddt"))
        self._rxj1131 = _KDE2D(
            rxj1131["dd"].to_numpy(),
            rxj1131["ddt"].to_numpy(),
            bandwidth=self.bandwidth_2d,
            hist_bins=self.bins_2d,
        )

        wfi2033 = pd.read_csv(files["wfi2033_dt_bic.dat"])
        if "Dt" not in wfi2033 or "weight" not in wfi2033:
            raise ValueError("wfi2033_dt_bic.dat missing required columns Dt, weight.")
        keep = (wfi2033["Dt"].to_numpy() > 0.0) & (wfi2033["Dt"].to_numpy() < self.wfi_dt_max)
        self._wfi2033 = _KDE1D(
            wfi2033["Dt"].to_numpy()[keep],
            weights=wfi2033["weight"].to_numpy()[keep],
            bandwidth=self.bandwidth_1d,
            hist_bins=self.bins_1d,
        )

    @staticmethod
    def _build_cosmology(H0: float, omega_m0: float, omega_k0: float) -> FlatLambdaCDM | LambdaCDM:
        if not (np.isfinite(H0) and np.isfinite(omega_m0) and np.isfinite(omega_k0)):
            raise ValueError("Non-finite cosmology parameter.")
        if H0 <= 0.0 or omega_m0 <= 0.0:
            raise ValueError("Non-physical H0/Omega_m0.")
        ode0 = 1.0 - float(omega_m0) - float(omega_k0)
        if ode0 <= 0.0:
            raise ValueError("Non-physical Omega_de <= 0.")
        if abs(float(omega_k0)) < 1e-12:
            return FlatLambdaCDM(H0=float(H0), Om0=float(omega_m0))
        return LambdaCDM(H0=float(H0), Om0=float(omega_m0), Ode0=float(ode0))

    @staticmethod
    def _predict_distances(cosmo: FlatLambdaCDM | LambdaCDM, zlens: float, zsource: float) -> LensPrediction:
        dd = float(cosmo.angular_diameter_distance(float(zlens)).value)
        ds = float(cosmo.angular_diameter_distance(float(zsource)).value)
        dds = float(cosmo.angular_diameter_distance_z1z2(float(zlens), float(zsource)).value)
        if not (np.isfinite(dd) and np.isfinite(ds) and np.isfinite(dds) and dds > 0.0):
            raise ValueError("Non-physical predicted distances.")
        ddt = (1.0 + float(zlens)) * dd * ds / dds
        return LensPrediction(dd_mpc=dd, ds_mpc=ds, dds_mpc=dds, ddt_mpc=float(ddt))

    def loglike(self, H0: float, omega_m0: float, omega_k0: float) -> float:
        cosmo = self._build_cosmology(H0, omega_m0, omega_k0)
        # B1608: analytic fit to Ddt + Dd
        b1608 = self._predict_distances(cosmo, zlens=0.6304, zsource=1.394)
        ll = _loglike_sklogn(b1608.ddt_mpc, mu=7.0531390, sigma=0.2282395, lam=4000.0)
        ll += _loglike_sklogn(b1608.dd_mpc, mu=6.79671, sigma=0.1836, lam=334.2)

        # J1206: 2D KDE on (Dd, Ddt)
        j1206 = self._predict_distances(cosmo, zlens=0.745, zsource=1.789)
        ll += self._j1206.logpdf(j1206.dd_mpc, j1206.ddt_mpc)

        # WFI2033: weighted 1D KDE on Ddt
        wfi2033 = self._predict_distances(cosmo, zlens=0.6575, zsource=1.662)
        ll += self._wfi2033.logpdf(wfi2033.ddt_mpc)

        # HE0435: 1D KDE on Ddt
        he0435 = self._predict_distances(cosmo, zlens=0.4546, zsource=1.693)
        ll += self._he0435.logpdf(he0435.ddt_mpc)

        # RXJ1131: 2D KDE on (Dd, Ddt)
        rxj1131 = self._predict_distances(cosmo, zlens=0.295, zsource=0.654)
        ll += self._rxj1131.logpdf(rxj1131.dd_mpc, rxj1131.ddt_mpc)

        # PG1115: 2D KDE on (Dd, Ddt)
        pg1115 = self._predict_distances(cosmo, zlens=0.311, zsource=1.722)
        ll += self._pg1115.logpdf(pg1115.dd_mpc, pg1115.ddt_mpc)
        return float(ll)

    def config(self) -> dict[str, Any]:
        return {
            "bandwidth_1d": float(self.bandwidth_1d),
            "bandwidth_2d": float(self.bandwidth_2d),
            "bins_1d": None if self.bins_1d is None else int(self.bins_1d),
            "bins_2d": None if self.bins_2d is None else int(self.bins_2d),
            "wfi_dt_max": float(self.wfi_dt_max),
            "j1206_bins_2d": None if self.j1206_bins_2d is None else int(self.j1206_bins_2d),
        }


@dataclass(frozen=True)
class LensChainPosterior:
    name: str
    z_lens: float
    z_source: float
    ddt_samples: np.ndarray
    ddt_weights: np.ndarray | None
    kappa_values: np.ndarray
    kappa_probs: np.ndarray
    source_file: str


def load_tdcosmo_sample_posteriors(
    files: dict[str, Path],
    *,
    max_samples_per_lens: int = 120_000,
    seed: int = 0,
) -> dict[str, LensChainPosterior]:
    rng = np.random.default_rng(int(seed))
    out: dict[str, LensChainPosterior] = {}
    for fname, path in files.items():
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected payload in {path}: expected dict.")
        if "ddt_samples" not in payload or "z_lens" not in payload or "z_source" not in payload:
            raise ValueError(f"Missing required keys in {path}.")
        ddt = np.asarray(payload["ddt_samples"], dtype=float).reshape(-1)
        good = np.isfinite(ddt) & (ddt > 0.0)
        ddt = ddt[good]
        if ddt.size == 0:
            raise ValueError(f"No valid ddt samples in {path}.")

        w_raw = payload.get("ddt_weights", None)
        weights: np.ndarray | None = None
        if w_raw is not None:
            ww = np.asarray(w_raw, dtype=float).reshape(-1)
            ww = ww[good]
            ww = np.clip(ww, 0.0, np.inf)
            if np.any(ww > 0.0):
                weights = ww

        if int(max_samples_per_lens) > 0 and ddt.size > int(max_samples_per_lens):
            if weights is None:
                idx = rng.choice(ddt.size, size=int(max_samples_per_lens), replace=False)
            else:
                p = weights / np.sum(weights)
                idx = rng.choice(ddt.size, size=int(max_samples_per_lens), replace=False, p=p)
                weights = weights[idx]
            ddt = ddt[idx]

        if weights is not None:
            ws = float(np.sum(weights))
            if ws > 0.0:
                weights = weights / ws
            else:
                weights = None

        kpdf = payload.get("kappa_pdf", None)
        kbins = payload.get("kappa_bin_edges", None)
        if kpdf is not None and kbins is not None:
            kp = np.asarray(kpdf, dtype=float).reshape(-1)
            ke = np.asarray(kbins, dtype=float).reshape(-1)
            if ke.size != kp.size + 1:
                raise ValueError(f"kappa bin mismatch in {path}.")
            kval = 0.5 * (ke[:-1] + ke[1:])
            kp = np.clip(kp, 0.0, np.inf)
            s = float(np.sum(kp))
            if s <= 0.0:
                raise ValueError(f"kappa_pdf has non-positive norm in {path}.")
            kprob = kp / s
        else:
            kval = np.array([0.0], dtype=float)
            kprob = np.array([1.0], dtype=float)

        name = str(payload.get("name", Path(fname).stem))
        out[name] = LensChainPosterior(
            name=name,
            z_lens=float(payload["z_lens"]),
            z_source=float(payload["z_source"]),
            ddt_samples=ddt,
            ddt_weights=weights,
            kappa_values=kval,
            kappa_probs=kprob,
            source_file=str(path),
        )
    return out


class UnifiedStrongLensChainLikelihood:
    """Unified strong-lens likelihood using full per-lens posterior chains (no histogram compression)."""

    def __init__(
        self,
        lens_posteriors: dict[str, LensChainPosterior],
        *,
        include_lenses: list[str] | None = None,
        bandwidth_1d: float = 20.0,
        nuisance_draws: int = 64,
        sigma_kappa_shift: float = 0.01,
        sigma_mst_lambda: float = 0.05,
        sigma_mst_slope: float = 0.10,
        seed: int = 0,
    ) -> None:
        if include_lenses is None:
            selected = sorted(lens_posteriors.keys())
        else:
            selected = [x for x in include_lenses if x in lens_posteriors]
        if not selected:
            raise ValueError("No lenses selected for unified likelihood.")
        self._lenses = [lens_posteriors[k] for k in selected]
        self._bandwidth_1d = float(bandwidth_1d)
        self._nuisance_draws = max(1, int(nuisance_draws))
        self._sigma_kappa_shift = max(0.0, float(sigma_kappa_shift))
        self._sigma_mst_lambda = max(0.0, float(sigma_mst_lambda))
        self._sigma_mst_slope = max(0.0, float(sigma_mst_slope))
        self._z_pivot = float(np.mean([x.z_lens for x in self._lenses]))

        self._ddt_dens: dict[str, _KDE1D] = {}
        self._kappa_draws_seq: list[np.ndarray] = []
        rng = np.random.default_rng(int(seed))
        for lens in self._lenses:
            self._ddt_dens[lens.name] = _KDE1D(
                lens.ddt_samples,
                weights=lens.ddt_weights,
                bandwidth=self._bandwidth_1d,
                hist_bins=None,
            )
            # Keep kappa draws per lens instance (not per name) so bootstrap
            # samples with repeated lenses remain statistically well-defined.
            self._kappa_draws_seq.append(
                rng.choice(
                lens.kappa_values,
                size=self._nuisance_draws,
                replace=True,
                p=lens.kappa_probs,
                )
            )

        # Draw global nuisance hyperparameters once; these are integrated in loglike via MC average.
        self._kappa_shift_draws = np.clip(
            rng.normal(0.0, self._sigma_kappa_shift, size=self._nuisance_draws),
            -0.2,
            0.2,
        )
        self._mst_lambda_draws = np.clip(
            rng.normal(1.0, self._sigma_mst_lambda, size=self._nuisance_draws),
            0.6,
            1.5,
        )
        self._mst_slope_draws = np.clip(
            rng.normal(0.0, self._sigma_mst_slope, size=self._nuisance_draws),
            -1.0,
            1.0,
        )

    @property
    def lens_names(self) -> list[str]:
        return [x.name for x in self._lenses]

    @staticmethod
    def _build_cosmology(H0: float, omega_m0: float, omega_k0: float) -> FlatLambdaCDM | LambdaCDM:
        if not (np.isfinite(H0) and np.isfinite(omega_m0) and np.isfinite(omega_k0)):
            raise ValueError("Non-finite cosmology parameter.")
        if H0 <= 0.0 or omega_m0 <= 0.0:
            raise ValueError("Non-physical H0/Omega_m0.")
        ode0 = 1.0 - float(omega_m0) - float(omega_k0)
        if ode0 <= 0.0:
            raise ValueError("Non-physical Omega_de <= 0.")
        if abs(float(omega_k0)) < 1e-12:
            return FlatLambdaCDM(H0=float(H0), Om0=float(omega_m0))
        return LambdaCDM(H0=float(H0), Om0=float(omega_m0), Ode0=float(ode0))

    @staticmethod
    def _predict_ddt(cosmo: FlatLambdaCDM | LambdaCDM, zlens: float, zsource: float) -> float:
        dd = float(cosmo.angular_diameter_distance(float(zlens)).value)
        ds = float(cosmo.angular_diameter_distance(float(zsource)).value)
        dds = float(cosmo.angular_diameter_distance_z1z2(float(zlens), float(zsource)).value)
        if not (np.isfinite(dd) and np.isfinite(ds) and np.isfinite(dds) and dds > 0.0):
            raise ValueError("Non-physical predicted distances.")
        return float((1.0 + float(zlens)) * dd * ds / dds)

    def _lens_loglike_marginalized(self, *, lens: LensChainPosterior, lens_idx: int, ddt_pred: float) -> float:
        dens = self._ddt_dens[lens.name]
        if self._nuisance_draws <= 1:
            return float(dens.logpdf(ddt_pred))

        z_term = float(lens.z_lens - self._z_pivot)
        lambda_eff = self._mst_lambda_draws * (1.0 + self._mst_slope_draws * z_term)
        kappa_eff = self._kappa_draws_seq[lens_idx] + self._kappa_shift_draws
        den = 1.0 - kappa_eff
        good = den > 1e-4
        if not np.any(good):
            return float("-inf")
        ddt_eff = ddt_pred * lambda_eff[good] / den[good]
        logp = np.asarray(dens.logpdf(ddt_eff), dtype=float)
        return _logmeanexp(logp)

    def loglike(self, H0: float, omega_m0: float, omega_k0: float) -> float:
        cosmo = self._build_cosmology(H0, omega_m0, omega_k0)
        ll = 0.0
        for i, lens in enumerate(self._lenses):
            ddt_pred = self._predict_ddt(cosmo, lens.z_lens, lens.z_source)
            lli = self._lens_loglike_marginalized(lens=lens, lens_idx=i, ddt_pred=ddt_pred)
            if not np.isfinite(lli):
                return float("-inf")
            ll += float(lli)
        return float(ll)

    def config(self) -> dict[str, Any]:
        return {
            "lens_names": self.lens_names,
            "n_lens": int(len(self._lenses)),
            "bandwidth_1d": float(self._bandwidth_1d),
            "nuisance_draws": int(self._nuisance_draws),
            "sigma_kappa_shift": float(self._sigma_kappa_shift),
            "sigma_mst_lambda": float(self._sigma_mst_lambda),
            "sigma_mst_slope": float(self._sigma_mst_slope),
            "z_pivot": float(self._z_pivot),
        }
