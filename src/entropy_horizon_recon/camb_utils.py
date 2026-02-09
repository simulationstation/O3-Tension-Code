from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CambFiducial:
    ombh2: float = 0.02237
    ns: float = 0.9649
    tau: float = 0.0544
    mnu: float = 0.06
    As: float = 2.1e-9


def _as_for_sigma8(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_target: float,
    fid: CambFiducial,
    max_iter: int = 30,
) -> float:
    import camb

    h = H0 / 100.0
    ombh2 = fid.ombh2
    omch2 = omega_m0 * h * h - ombh2
    if omch2 <= 0:
        raise ValueError("Computed omch2 <= 0; invalid parameters for CAMB.")

    # bracket in log10 As
    lo, hi = -10.0, -8.0
    As_lo, As_hi = 10**lo, 10**hi

    def sigma8_for_As(As: float) -> float:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k0, mnu=fid.mnu, tau=fid.tau)
        pars.InitPower.set_params(As=As, ns=fid.ns)
        pars.set_for_lmax(lmax=1500, lens_potential_accuracy=1)
        pars.WantCls = False
        pars.WantTransfer = True
        results = camb.get_results(pars)
        return float(results.get_sigma8_0())

    try:
        sigma8_fid = sigma8_for_As(fid.As)
        if sigma8_fid > 0:
            As_scaled = fid.As * (sigma8_target / sigma8_fid) ** 2
            if np.isfinite(As_scaled) and As_scaled > 0:
                return float(As_scaled)
    except Exception:
        pass

    s_lo = sigma8_for_As(As_lo)
    s_hi = sigma8_for_As(As_hi)
    if not (s_lo <= sigma8_target <= s_hi):
        # Expand bracket if needed
        if sigma8_target < s_lo:
            hi = lo
            lo = -11.0
        else:
            lo = hi
            hi = -7.0
        As_lo, As_hi = 10**lo, 10**hi
        s_lo = sigma8_for_As(As_lo)
        s_hi = sigma8_for_As(As_hi)
        if not (s_lo <= sigma8_target <= s_hi):
            raise ValueError("Unable to bracket sigma8 target with As range.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        As_mid = 10**mid
        s_mid = sigma8_for_As(As_mid)
        if s_mid < sigma8_target:
            lo = mid
        else:
            hi = mid
        if abs(s_mid - sigma8_target) / sigma8_target < 1e-3:
            return As_mid
    return 10**(0.5 * (lo + hi))


def _camb_cache_dir() -> Path | None:
    env_dir = os.environ.get("EHR_CAMB_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    # Try to locate repo root by finding a "data" directory above this file.
    for parent in Path(__file__).resolve().parents:
        data_dir = parent / "data"
        if data_dir.is_dir():
            return data_dir / "cache" / "camb"
    return Path.home() / ".cache" / "entropy_horizon_recon" / "camb"


def _cache_key(kind: str, params: dict) -> str:
    payload = {"kind": kind, **params}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_load(kind: str, key: str) -> dict | None:
    cache_dir = _camb_cache_dir()
    if cache_dir is None:
        return None
    path = cache_dir / f"{kind}_{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception:
        return None


def _cache_save(kind: str, key: str, arrays: dict) -> None:
    cache_dir = _camb_cache_dir()
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{kind}_{key}.npz"
    # np.savez_compressed appends ".npz" when given a filename that doesn't end with ".npz".
    # Use a file handle so we can write to an explicit temp path without extension surprises.
    tmp_path = cache_dir / f".{kind}_{key}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@lru_cache(maxsize=256)
def _cached_camb_clpp(
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    lmax: int,
    fid: CambFiducial,
) -> np.ndarray:
    import camb

    params = {
        "H0": float(H0),
        "omega_m0": float(omega_m0),
        "omega_k0": float(omega_k0),
        "sigma8_0": float(sigma8_0),
        "lmax": int(lmax),
        "fid": {
            "ombh2": float(fid.ombh2),
            "ns": float(fid.ns),
            "tau": float(fid.tau),
            "mnu": float(fid.mnu),
            "As": float(fid.As),
        },
        "camb_version": getattr(camb, "__version__", "unknown"),
    }
    key = _cache_key("clpp", params)
    cached = _cache_load("clpp", key)
    if cached is not None and "clpp" in cached:
        return cached["clpp"]

    As = _as_for_sigma8(
        H0=H0,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        sigma8_target=sigma8_0,
        fid=fid,
    )
    if not np.isfinite(As) or As <= 0:
        raise ValueError("Computed As is non-finite or non-positive.")
    # CAMB lensing requires a realistic normalization; P(k=0.05/Mpc) ~ As.
    if As > 2e-8:
        raise ValueError("Computed As exceeds CAMB lensing normalization limit.")

    h = H0 / 100.0
    ombh2 = fid.ombh2
    omch2 = omega_m0 * h * h - ombh2
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k0, mnu=fid.mnu, tau=fid.tau)
    pars.InitPower.set_params(As=As, ns=fid.ns)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    pars.WantCls = True
    results = camb.get_results(pars)
    clpp = results.get_lens_potential_cls(lmax=lmax)[:, 0]
    _cache_save("clpp", key, {"clpp": np.asarray(clpp, dtype=float)})
    return clpp


@lru_cache(maxsize=256)
def _cached_camb_pk(
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    z_eff: float,
    kmin: float,
    kmax: float,
    nk: int,
    fid: CambFiducial,
) -> Tuple[np.ndarray, np.ndarray]:
    import camb

    params = {
        "H0": float(H0),
        "omega_m0": float(omega_m0),
        "omega_k0": float(omega_k0),
        "sigma8_0": float(sigma8_0),
        "z_eff": float(z_eff),
        "kmin": float(kmin),
        "kmax": float(kmax),
        "nk": int(nk),
        "fid": {
            "ombh2": float(fid.ombh2),
            "ns": float(fid.ns),
            "tau": float(fid.tau),
            "mnu": float(fid.mnu),
            "As": float(fid.As),
        },
        "camb_version": getattr(camb, "__version__", "unknown"),
    }
    key = _cache_key("pk", params)
    cached = _cache_load("pk", key)
    if cached is not None and "kh" in cached and "pk" in cached:
        return cached["kh"], cached["pk"]

    As = _as_for_sigma8(
        H0=H0,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        sigma8_target=sigma8_0,
        fid=fid,
    )
    if not np.isfinite(As) or As <= 0:
        raise ValueError("Computed As is non-finite or non-positive.")
    # Keep the same conservative normalization guard as the lensing paths.
    if As > 2e-8:
        raise ValueError("Computed As exceeds CAMB normalization limit.")

    h = H0 / 100.0
    ombh2 = fid.ombh2
    omch2 = omega_m0 * h * h - ombh2
    if omch2 <= 0:
        raise ValueError("Computed omch2 <= 0; invalid parameters for CAMB.")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k0, mnu=fid.mnu, tau=fid.tau)
    pars.InitPower.set_params(As=As, ns=fid.ns)
    pars.set_matter_power(redshifts=[z_eff], kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh = np.logspace(math.log10(kmin), math.log10(kmax), nk)
    interp = results.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        extrap_kmax=kmax,
    )
    pk = interp.P(z_eff, kh)
    _cache_save("pk", key, {"kh": np.asarray(kh, dtype=float), "pk": np.asarray(pk, dtype=float)})
    return kh, pk


@lru_cache(maxsize=256)
def _cached_camb_clpp_pk(
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    lmax: int,
    z_eff: float,
    kmin: float,
    kmax: float,
    nk: int,
    fid: CambFiducial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import camb

    params = {
        "H0": float(H0),
        "omega_m0": float(omega_m0),
        "omega_k0": float(omega_k0),
        "sigma8_0": float(sigma8_0),
        "lmax": int(lmax),
        "z_eff": float(z_eff),
        "kmin": float(kmin),
        "kmax": float(kmax),
        "nk": int(nk),
        "fid": {
            "ombh2": float(fid.ombh2),
            "ns": float(fid.ns),
            "tau": float(fid.tau),
            "mnu": float(fid.mnu),
            "As": float(fid.As),
        },
        "camb_version": getattr(camb, "__version__", "unknown"),
    }
    key = _cache_key("clpp_pk", params)
    cached = _cache_load("clpp_pk", key)
    if cached is not None and "clpp" in cached and "kh" in cached and "pk" in cached:
        return cached["clpp"], cached["kh"], cached["pk"]

    As = _as_for_sigma8(
        H0=H0,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        sigma8_target=sigma8_0,
        fid=fid,
    )
    if not np.isfinite(As) or As <= 0:
        raise ValueError("Computed As is non-finite or non-positive.")
    # CAMB lensing requires a realistic normalization; P(k=0.05/Mpc) ~ As.
    if As > 2e-8:
        raise ValueError("Computed As exceeds CAMB lensing normalization limit.")

    h = H0 / 100.0
    ombh2 = fid.ombh2
    omch2 = omega_m0 * h * h - ombh2
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omega_k0, mnu=fid.mnu, tau=fid.tau)
    pars.InitPower.set_params(As=As, ns=fid.ns)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    pars.set_matter_power(redshifts=[z_eff], kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none
    pars.WantCls = True

    results = camb.get_results(pars)
    clpp = results.get_lens_potential_cls(lmax=lmax)[:, 0]
    kh = np.logspace(math.log10(kmin), math.log10(kmax), nk)
    interp = results.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        extrap_kmax=kmax,
    )
    pk = interp.P(z_eff, kh)
    _cache_save(
        "clpp_pk",
        key,
        {"clpp": np.asarray(clpp, dtype=float), "kh": np.asarray(kh, dtype=float), "pk": np.asarray(pk, dtype=float)},
    )
    return clpp, kh, pk


def camb_clpp(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    ell: np.ndarray,
    fid: CambFiducial | None = None,
) -> np.ndarray:
    fid = fid or CambFiducial()
    ell = np.asarray(ell, dtype=int)
    lmax = int(np.max(ell))
    clpp = _cached_camb_clpp(H0, omega_m0, omega_k0, sigma8_0, lmax, fid)
    return clpp[ell]


def camb_pk_linear(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    z_eff: float,
    k: np.ndarray,
    fid: CambFiducial | None = None,
) -> np.ndarray:
    fid = fid or CambFiducial()
    k = np.asarray(k, dtype=float)
    kmin = float(np.min(k))
    kmax = float(np.max(k))
    nk = max(200, int(len(k) * 5))
    kh, pk = _cached_camb_pk(H0, omega_m0, omega_k0, sigma8_0, z_eff, kmin, kmax, nk, fid)
    return np.interp(k, kh, pk)


def camb_clpp_and_pk(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    sigma8_0: float,
    ell: np.ndarray,
    z_eff: float,
    k: np.ndarray,
    fid: CambFiducial | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    fid = fid or CambFiducial()
    ell = np.asarray(ell, dtype=int)
    k = np.asarray(k, dtype=float)
    lmax = int(np.max(ell))
    kmin = float(np.min(k))
    kmax = float(np.max(k))
    nk = max(200, int(len(k) * 5))
    clpp, kh, pk = _cached_camb_clpp_pk(H0, omega_m0, omega_k0, sigma8_0, lmax, z_eff, kmin, kmax, nk, fid)
    return clpp[ell], np.interp(k, kh, pk)
