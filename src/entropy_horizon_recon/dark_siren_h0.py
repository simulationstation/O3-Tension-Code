from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from ligo.skymap import distance as ligo_distance

from .constants import PhysicalConstants
from .dark_sirens import SkyMap3D, read_skymap_3d
from .dark_sirens_incompleteness import select_missing_pixels
from .dark_sirens_selection import (
    O3aBbhInjectionSet,
    calibrate_snr_threshold_match_count,
    calibrate_snr_threshold_match_found_fraction,
)
from .dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from .gw_distance_priors import GWDistancePrior
from .importance_sampling import smooth_logweights


@dataclass(frozen=True)
class LCDMDistanceCache:
    """Dimensionless distance factors for FRW LCDM with fixed (Omega_m0, Omega_k0).

    For a given H0, the luminosity distance is:

      dL(z; H0) = (c/H0) * f(z),

    where f(z) is dimensionless and depends only on (Omega_m0, Omega_k0).
    """

    omega_m0: float
    omega_k0: float
    z_grid: np.ndarray  # (n_z,)
    f_grid: np.ndarray  # (n_z,)

    def f(self, z: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(z, dtype=float), self.z_grid, self.f_grid)


@dataclass(frozen=True)
class CalibratedDetectionModel:
    """Cheap calibrated detectability proxy p_det(snr).

    This is intentionally simple: either a hard SNR threshold or a monotone binned curve.
    It is used for selection-alpha estimation and (optionally) for selection-aware per-event
    numerator terms in hierarchical sanity tests.
    """

    det_model: Literal["threshold", "snr_binned"]
    snr_threshold: float | None = None
    snr_binned_edges: np.ndarray | None = None  # (n_edges,)
    snr_binned_pdet: np.ndarray | None = None  # (n_edges-1,)

    def pdet(self, snr: np.ndarray) -> np.ndarray:
        snr = np.asarray(snr, dtype=float)
        if self.det_model == "threshold":
            if self.snr_threshold is None:
                raise ValueError("CalibratedDetectionModel missing snr_threshold for det_model='threshold'.")
            return (snr > float(self.snr_threshold)).astype(float)
        if self.det_model == "snr_binned":
            if self.snr_binned_edges is None or self.snr_binned_pdet is None:
                raise ValueError("CalibratedDetectionModel missing binned curve for det_model='snr_binned'.")
            idx = np.clip(np.digitize(snr, self.snr_binned_edges) - 1, 0, int(self.snr_binned_pdet.size) - 1)
            return np.asarray(self.snr_binned_pdet[idx], dtype=float)
        raise ValueError("Unknown det_model.")


def _calibrate_detection_model_from_snr_and_found(
    *,
    snr_net_opt: np.ndarray,
    found_ifar: np.ndarray,
    det_model: Literal["threshold", "snr_binned"],
    snr_threshold: float | None,
    snr_binned_nbins: int,
) -> CalibratedDetectionModel:
    """Calibrate a p_det(snr) proxy curve from injections (unweighted)."""
    snr = np.asarray(snr_net_opt, dtype=float)
    found = np.asarray(found_ifar, dtype=bool)
    if snr.shape != found.shape:
        raise ValueError("snr_net_opt and found_ifar must have matching shapes for detection-model calibration.")
    if snr.size < 100:
        raise ValueError("Too few injections for detection-model calibration.")

    if det_model == "threshold":
        thresh = float(snr_threshold) if snr_threshold is not None else float(calibrate_snr_threshold_match_count(snr_net_opt=snr, found_ifar=found))
        return CalibratedDetectionModel(det_model="threshold", snr_threshold=float(thresh))

    if det_model == "snr_binned":
        nb = int(snr_binned_nbins)
        if nb < 20:
            raise ValueError("snr_binned_nbins too small (need >= 20).")
        edges = np.quantile(snr, np.linspace(0.0, 1.0, nb + 1))
        edges = np.unique(edges)
        if edges.size < 10:
            raise ValueError("Too few unique SNR edges for snr_binned.")
        bin_idx = np.clip(np.digitize(snr, edges) - 1, 0, edges.size - 2)
        p = np.zeros(edges.size - 1, dtype=float)
        for i in range(p.size):
            m_i = bin_idx == i
            if not np.any(m_i):
                p[i] = p[i - 1] if i > 0 else 0.0
                continue
            p[i] = float(np.mean(found[m_i].astype(float)))
        p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
        return CalibratedDetectionModel(det_model="snr_binned", snr_binned_edges=np.asarray(edges, dtype=float), snr_binned_pdet=np.asarray(p, dtype=float))

    raise ValueError("Unknown det_model.")


def _build_lcdm_distance_cache(
    *,
    z_max: float,
    omega_m0: float,
    omega_k0: float,
    n_grid: int = 10_001,
) -> LCDMDistanceCache:
    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")
    if int(n_grid) < 200:
        raise ValueError("n_grid too small for stable distances.")

    om = float(omega_m0)
    ok = float(omega_k0)
    ol = 1.0 - om - ok
    if not np.isfinite(om) or not np.isfinite(ok) or not np.isfinite(ol):
        raise ValueError("Non-finite omega parameters.")

    z = np.linspace(0.0, z_max, int(n_grid))
    Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
    Ez2 = np.clip(Ez2, 1e-15, np.inf)
    invE = 1.0 / np.sqrt(Ez2)

    dz = np.diff(z)
    chi = np.empty_like(z)
    chi[0] = 0.0
    chi[1:] = np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))

    if ok == 0.0:
        Sk = chi
    elif ok > 0.0:
        rk = np.sqrt(ok)
        Sk = np.sinh(rk * chi) / rk
    else:
        rk = np.sqrt(abs(ok))
        Sk = np.sin(rk * chi) / rk

    f = (1.0 + z) * Sk
    return LCDMDistanceCache(omega_m0=om, omega_k0=ok, z_grid=z, f_grid=f)


def _log_pop_weight_z_fixed_lcdm(
    z: np.ndarray,
    *,
    mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    k: float,
) -> np.ndarray:
    """Unnormalized log population weight for redshift (fixed LCDM; matches selection proxy).

    Note: this helper is intended for *fixed-cosmology* proxy uses. For GR H0 inference where H0 is
    scanned, the comoving-volume element carries an H0^{-3} scale factor that should be included
    consistently; see `_event_logL_h0_grid_from_hierarchical_pe_samples`.
    """
    z = np.asarray(z, dtype=float)
    if mode == "none":
        return np.zeros_like(z, dtype=float)

    # Match `dark_sirens_selection.compute_selection_alpha_from_injections` (Planck-ish).
    H0 = 67.7  # km/s/Mpc
    om0 = 0.31
    c = 299792.458  # km/s

    zmax = float(np.max(z)) if z.size else 0.0
    zmax = max(zmax, 1e-6)
    z_grid = np.linspace(0.0, zmax, 5001)
    Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
    invEz = 1.0 / Ez
    dc = (c / H0) * np.cumsum(np.concatenate([[0.0], 0.5 * (invEz[1:] + invEz[:-1]) * np.diff(z_grid)]))
    Dc = np.interp(z, z_grid, dc)
    Ez_z = np.interp(z, z_grid, Ez)

    dVdz = (c / (H0 * Ez_z)) * (Dc**2)
    base = dVdz / np.clip(1.0 + z, 1e-12, np.inf)
    if mode == "comoving_uniform":
        w = base
    elif mode == "comoving_powerlaw":
        w = base * np.clip(1.0 + z, 1e-12, np.inf) ** float(k)
    else:  # pragma: no cover
        raise ValueError("Unknown z population mode.")
    return np.log(np.clip(w, 1e-300, np.inf))


def _log_pop_weight_mass_source_powerlaw_q(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
) -> np.ndarray:
    """Unnormalized log population weight for source-frame masses under a simple powerlaw-q model."""
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & (m2 > 0.0)
    good &= (m1 >= float(m_min)) & (m1 <= float(m_max)) & (m2 >= float(m_min)) & (m2 <= m1)
    out = np.full_like(m1, -np.inf, dtype=float)
    if np.any(good):
        out[good] = -float(alpha) * np.log(np.clip(m1[good], 1e-300, np.inf)) + float(q_beta) * np.log(np.clip(q[good], 1e-300, np.inf))
    return out


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # Stable logistic using tanh to avoid overflow in exp for large |x|.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _log_pop_weight_mass_source_powerlaw_q_smooth(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
    m_taper_delta: float,
) -> np.ndarray:
    """Powerlaw-q mass model with smooth low/high-mass tapers (no hard zero-support edges).

    This replaces hard constraints m>=m_min and m<=m_max with sigmoid tapers of width m_taper_delta.
    It is intended to avoid artificial "support truncation" in H0 scans from a single event crossing
    a hard population bound.
    """
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    delta = float(m_taper_delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("m_taper_delta must be finite and > 0 for powerlaw_q_smooth.")

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & np.isfinite(m2) & (m2 > 0.0)
    out = np.full_like(m1, -np.inf, dtype=float)
    if not np.any(good):
        return out

    m1g = m1[good]
    qg = q[good]
    m2g = m2[good]

    # Smooth tapers: sigmoid((m-m_min)/d) * sigmoid((m_max-m)/d)
    t1 = _sigmoid_stable((m1g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m1g) / delta)
    t2 = _sigmoid_stable((m2g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m2g) / delta)
    taper = np.clip(t1 * t2, 1e-300, 1.0)

    out[good] = (
        -float(alpha) * np.log(np.clip(m1g, 1e-300, np.inf))
        + float(q_beta) * np.log(np.clip(qg, 1e-300, np.inf))
        + np.log(taper)
    )
    return out


def _log_pop_weight_mass_source_powerlaw_peak_q_smooth(
    m1_source: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    m_min: float,
    m_max: float,
    q_beta: float,
    m_taper_delta: float,
    m_peak: float,
    m_peak_sigma: float,
    m_peak_frac: float,
) -> np.ndarray:
    """Powerlaw+Gaussian-peak model with smooth mass tapers for both component masses.

    This is a minimal "PowerLaw+Peak" style primary-mass model intended to avoid the
    monotonic-in-H0 behavior that can arise when the population strictly prefers lower
    source-frame masses. The peak is applied to m1 (primary) while q is modeled separately.
    """
    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = q * m1

    delta = float(m_taper_delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("m_taper_delta must be finite and > 0 for powerlaw_peak_q_smooth.")

    f_peak = float(m_peak_frac)
    if not (np.isfinite(f_peak) and 0.0 <= f_peak <= 1.0):
        raise ValueError("m_peak_frac must be finite and in [0,1].")

    mp = float(m_peak)
    sig = float(m_peak_sigma)
    if not (np.isfinite(mp) and mp > 0.0 and np.isfinite(sig) and sig > 0.0):
        raise ValueError("m_peak and m_peak_sigma must be finite and positive for powerlaw_peak_q_smooth.")

    good = np.isfinite(m1) & np.isfinite(q) & (m1 > 0.0) & (q > 0.0) & (q <= 1.0) & np.isfinite(m2) & (m2 > 0.0)
    out = np.full_like(m1, -np.inf, dtype=float)
    if not np.any(good):
        return out

    m1g = m1[good]
    qg = q[good]
    m2g = m2[good]

    t1 = _sigmoid_stable((m1g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m1g) / delta)
    t2 = _sigmoid_stable((m2g - float(m_min)) / delta) * _sigmoid_stable((float(m_max) - m2g) / delta)
    taper = np.clip(t1 * t2, 1e-300, 1.0)

    log_q = float(q_beta) * np.log(np.clip(qg, 1e-300, np.inf))
    log_taper = np.log(taper)

    log_pl = -float(alpha) * np.log(np.clip(m1g, 1e-300, np.inf)) + log_q + log_taper
    # Gaussian peak in m1. Normalization constant is omitted (drops out in Bayes factors on fixed models).
    log_peak = -0.5 * ((m1g - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper

    if f_peak <= 0.0:
        out[good] = log_pl
        return out
    if f_peak >= 1.0:
        out[good] = log_peak
        return out

    log_mix = logsumexp(
        np.stack([np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak], axis=0),
        axis=0,
    )
    out[good] = log_mix
    return out


def _m1_source_from_chirp_mass_and_q(
    Mc_source: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute primary source-frame mass m1 from (chirp_mass_source, q=m2/m1)."""
    Mc = np.asarray(Mc_source, dtype=float)
    q = np.asarray(q, dtype=float)
    return Mc * np.clip(1.0 + q, 1e-12, np.inf) ** (1.0 / 5.0) / np.clip(q, 1e-12, np.inf) ** (3.0 / 5.0)



def _event_logL_h0_grid_from_histogram(
    *,
    pix_sel: np.ndarray,
    prob_pix: np.ndarray,
    dL_edges: np.ndarray,
    pdf_bins: np.ndarray,
    nside: int,
    nest: bool,
    z_gal: np.ndarray,
    w_gal: np.ndarray,
    ipix_gal: np.ndarray,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    constants: PhysicalConstants,
    gw_distance_prior: GWDistancePrior,
    gal_chunk_size: int = 50_000,
) -> np.ndarray:
    """Compute event log-likelihood on an H0 grid from a binned p(Ω,dL|data) histogram.

    This is the PE-posterior analogue of `_event_logL_h0_grid`, with:

      p(Ω, dL | data) ≈ prob_pix(Ω) * pdf_bins(dL | pix, data)

    where prob_pix is probability *mass* per pixel and pdf_bins is a conditional distance density
    (per Mpc) in bins given by dL_edges.
    """
    pix_sel = np.asarray(pix_sel, dtype=np.int64)
    prob_pix = np.asarray(prob_pix, dtype=float)
    dL_edges = np.asarray(dL_edges, dtype=float)
    pdf_bins = np.asarray(pdf_bins, dtype=float)

    if pix_sel.ndim != 1 or prob_pix.ndim != 1:
        raise ValueError("pix_sel/prob_pix must be 1D arrays.")
    if pix_sel.size == 0:
        raise ValueError("No pixels provided.")
    if pix_sel.shape != prob_pix.shape:
        raise ValueError("pix_sel and prob_pix must have matching shapes.")
    if pdf_bins.ndim != 2:
        raise ValueError("pdf_bins must be 2D (n_pix, n_bins).")
    if pdf_bins.shape[0] != pix_sel.size:
        raise ValueError("pdf_bins.shape[0] must match pix_sel.size.")
    if dL_edges.ndim != 1 or dL_edges.size < 3:
        raise ValueError("dL_edges must be 1D with >=3 entries (>=2 bins).")
    if np.any(~np.isfinite(dL_edges)) or np.any(np.diff(dL_edges) <= 0.0):
        raise ValueError("dL_edges must be finite and strictly increasing.")
    if pdf_bins.shape[1] != dL_edges.size - 1:
        raise ValueError("pdf_bins.shape[1] must equal len(dL_edges)-1.")

    if not bool(nest):
        raise ValueError("_event_logL_h0_grid_from_histogram assumes NESTED pixel indexing.")
    nside = int(nside)
    if nside <= 0:
        raise ValueError("nside must be positive.")
    npix = 12 * nside * nside

    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or not (z.shape == w.shape == ipix.shape):
        raise ValueError("z_gal/w_gal/ipix_gal must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided.")
    if np.any(ipix < 0) or np.any(ipix >= int(npix)):
        raise ValueError("ipix_gal contains out-of-range pixels for this histogram nside.")

    # Map pixels to row indices.
    pix_to_row = np.full((int(npix),), -1, dtype=np.int32)
    pix_to_row[pix_sel.astype(np.int64, copy=False)] = np.arange(pix_sel.size, dtype=np.int32)
    row = pix_to_row[ipix]

    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        raise ValueError("All galaxies map outside the provided credible region (or have invalid z/w).")
    z = z[good]
    w = w[good]
    row = row[good].astype(np.int64, copy=False)
    prob = prob_pix[row]

    good2 = np.isfinite(prob) & (prob > 0.0)
    if not np.any(good2):
        raise ValueError("All galaxies map to pixels with non-positive probability mass.")
    z = z[good2]
    w = w[good2]
    row = row[good2]
    prob = prob[good2]

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    # Dimensionless distance factor f(z).
    fz = dist_cache.f(z)  # (n_gal,)
    if not np.all(np.isfinite(fz)) or np.any(fz <= 0.0):
        raise ValueError("Non-finite/non-positive f(z) in LCDM cache (check z_max / omega parameters).")

    nb = int(dL_edges.size - 1)
    pdf_flat = np.asarray(pdf_bins, dtype=float).reshape(-1)

    logw = np.log(np.clip(w, 1e-30, np.inf))
    logprob = np.log(np.clip(prob, 1e-300, np.inf))

    logL = np.full((int(H0_grid.size),), -np.inf, dtype=float)
    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    # Chunk over galaxies to keep peak memory bounded.
    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        fz_c = fz[a:b]
        row_c = row[a:b]
        logw_c = logw[a:b]
        logprob_c = logprob[a:b]

        dL = (constants.c_km_s / H0_grid.reshape((-1, 1))) * fz_c.reshape((1, -1))
        dL = np.asarray(dL, dtype=float)
        valid = np.isfinite(dL) & (dL > 0.0)

        bin_idx = np.searchsorted(dL_edges, dL, side="right") - 1
        valid = valid & (bin_idx >= 0) & (bin_idx < nb)

        # Flattened [row, bin] lookup.
        lin = row_c.reshape((1, -1)) * nb + np.clip(bin_idx, 0, nb - 1)
        pdf = pdf_flat[lin]
        logpdf = np.where(valid, np.log(np.clip(pdf, 1e-300, np.inf)), -np.inf)

        logprior = gw_distance_prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
        logterm = logw_c.reshape((1, -1)) + logprob_c.reshape((1, -1)) + logpdf - logprior
        logL_chunk = logsumexp(logterm, axis=1)
        logL = np.logaddexp(logL, logL_chunk)

    return logL


def _event_logL_h0_grid(
    *,
    sky: SkyMap3D,
    z_gal: np.ndarray,
    w_gal: np.ndarray,
    ipix_gal: np.ndarray,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    constants: PhysicalConstants,
    gw_distance_prior: GWDistancePrior,
    gal_chunk_size: int = 20_000,
) -> np.ndarray:
    """Compute event log-likelihood on an H0 grid (proxy likelihood)."""
    if not sky.nest:
        raise ValueError("_event_logL_h0_grid assumes NESTED sky maps.")
    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or not (z.shape == w.shape == ipix.shape):
        raise ValueError("z_gal/w_gal/ipix_gal must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided.")

    if np.any(ipix < 0) or np.any(ipix >= int(sky.npix)):
        raise ValueError("ipix_gal contains out-of-range pixels for this sky map.")

    # Sky layers per galaxy.
    prob = np.asarray(sky.prob[ipix], dtype=float)
    distmu = np.asarray(sky.distmu[ipix], dtype=float)
    distsigma = np.asarray(sky.distsigma[ipix], dtype=float)
    distnorm = np.asarray(sky.distnorm[ipix], dtype=float)

    good = (
        np.isfinite(prob)
        & (prob > 0.0)
        & np.isfinite(distmu)
        & np.isfinite(distsigma)
        & (distsigma > 0.0)
        & np.isfinite(distnorm)
        & (distnorm > 0.0)
        & np.isfinite(z)
        & (z > 0.0)
        & np.isfinite(w)
        & (w > 0.0)
    )
    if not np.any(good):
        raise ValueError("All galaxies map to invalid distance-layer pixels (or have invalid z/w).")
    z = z[good]
    w = w[good]
    prob = prob[good]
    distmu = distmu[good]
    distsigma = distsigma[good]
    distnorm = distnorm[good]

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    # Precompute the dimensionless distance factor f(z) once.
    fz = dist_cache.f(z)  # (n_gal,)
    if not np.all(np.isfinite(fz)) or np.any(fz <= 0.0):
        raise ValueError("Non-finite/non-positive f(z) in LCDM cache (check z_max / omega parameters).")

    logw = np.log(np.clip(w, 1e-30, np.inf))
    logprob = np.log(np.clip(prob, 1e-300, np.inf))

    n_h0 = int(H0_grid.size)
    logL = np.full((n_h0,), -np.inf, dtype=float)

    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    # Chunk over galaxies to keep peak memory bounded.
    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        fz_c = fz[a:b]
        distmu_c = distmu[a:b]
        distsigma_c = distsigma[a:b]
        distnorm_c = distnorm[a:b]
        logw_c = logw[a:b]
        logprob_c = logprob[a:b]

        # dL(H0, gal) = (c/H0) * f(z)
        dL = (constants.c_km_s / H0_grid.reshape((-1, 1))) * fz_c.reshape((1, -1))
        dL = np.clip(dL, 1e-6, np.inf)

        pdf = ligo_distance.conditional_pdf(dL, distmu_c.reshape((1, -1)), distsigma_c.reshape((1, -1)), distnorm_c.reshape((1, -1)))
        pdf = np.clip(np.asarray(pdf, dtype=float), 1e-300, np.inf)
        logpdf = np.log(pdf)

        logprior = gw_distance_prior.log_pi_dL(dL)
        logterm = logw_c.reshape((1, -1)) + logprob_c.reshape((1, -1)) + logpdf - logprior
        logL_chunk = logsumexp(logterm, axis=1)
        logL = np.logaddexp(logL, logL_chunk)

    return logL


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least 2 points.")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be finite and strictly increasing.")
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def compute_missing_host_logL_h0_grid_from_pixels(
    *,
    prob_pix: np.ndarray,
    distmu: np.ndarray,
    distsigma: np.ndarray,
    distnorm: np.ndarray,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    dL_grid: np.ndarray,
    z_max: float,
    host_prior_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform",
    host_prior_z_k: float = 0.0,
    gw_distance_prior: GWDistancePrior,
    pixel_chunk_size: int = 5_000,
) -> np.ndarray:
    """Compute missing-host log-likelihood on an H0 grid from preselected sky pixel arrays.

    This is the H0-grid analogue of `dark_sirens_incompleteness.compute_missing_host_logL_draws_from_pixels`,
    but for a GR LCDM distance-redshift relation with H0 as a grid parameter.

      L_missing(H0) = sum_{pix in CR} prob_pix * ∫ ddL p(dL | pix, skymap) * host(H0; dL),

    where:

      host(H0; dL) = rho_host(z(dL)) * (dV/dz/dOmega) * (dz/ddL) * (1/pi(dL)).

    The 1/pi(dL) factor corrects for the implicit distance prior in the public sky maps, matching
    the convention used for the in-catalog term.
    """
    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")

    prob_pix = np.asarray(prob_pix, dtype=float)
    distmu = np.asarray(distmu, dtype=float)
    distsigma = np.asarray(distsigma, dtype=float)
    distnorm = np.asarray(distnorm, dtype=float)
    if prob_pix.ndim != 1 or distmu.ndim != 1 or distsigma.ndim != 1 or distnorm.ndim != 1:
        raise ValueError("Pixel arrays must be 1D.")
    if not (prob_pix.shape == distmu.shape == distsigma.shape == distnorm.shape):
        raise ValueError("Pixel arrays must have matching shapes.")
    if prob_pix.size == 0:
        raise ValueError("No valid sky pixels provided for missing-host term.")

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    dL_grid = np.asarray(dL_grid, dtype=float)
    if dL_grid.ndim != 1 or dL_grid.size < 2:
        raise ValueError("dL_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(dL_grid)) or np.any(np.diff(dL_grid) <= 0.0):
        raise ValueError("dL_grid must be finite and strictly increasing.")

    if int(pixel_chunk_size) <= 0:
        raise ValueError("pixel_chunk_size must be positive.")

    constants = PhysicalConstants()

    # Precompute trapezoid weights and 1/pi(dL) prior correction.
    w_dl = _trapz_weights(dL_grid)
    inv_pi = np.exp(-gw_distance_prior.log_pi_dL(dL_grid))

    # Build a dimensionless q(z) factor on the cache z-grid (truncated at z_max):
    #   host(H0; dL) = (c/H0)^2 * q(z(dL;H0)) * 1/pi(dL)
    z = np.asarray(dist_cache.z_grid, dtype=float)
    f = np.asarray(dist_cache.f_grid, dtype=float)
    m_z = z <= z_max
    z = z[m_z]
    f = f[m_z]
    if z.size < 5:
        raise ValueError("z_max too small relative to LCDM distance cache.")

    om = float(dist_cache.omega_m0)
    ok = float(dist_cache.omega_k0)
    ol = 1.0 - om - ok
    Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
    Ez2 = np.clip(Ez2, 1e-15, np.inf)
    E = np.sqrt(Ez2)

    Sk = f / np.clip(1.0 + z, 1e-12, np.inf)
    dfdz = np.gradient(f, z)
    dfdz = np.clip(dfdz, 1e-12, np.inf)

    if host_prior_z_mode in ("none", "comoving_uniform"):
        rho = np.ones_like(z, dtype=float)
    elif host_prior_z_mode == "comoving_powerlaw":
        rho = np.clip(1.0 + z, 1e-12, np.inf) ** float(host_prior_z_k)
    else:
        raise ValueError("Unknown host_prior_z_mode.")

    q = (Sk**2) * rho / np.clip(E * dfdz, 1e-30, np.inf)

    # host_w matrix: (n_dL, n_H0)
    host_w = np.zeros((dL_grid.size, H0_grid.size), dtype=float)
    for j, H0 in enumerate(H0_grid.tolist()):
        # dL(z) = (c/H0) * f(z)
        dL_of_z = (constants.c_km_s / float(H0)) * f
        if not np.all(np.isfinite(dL_of_z)) or np.any(np.diff(dL_of_z) <= 0.0):
            raise ValueError("Non-monotone/invalid dL(z) encountered; cannot invert for missing-host term.")

        dL_min = float(dL_of_z[0])
        dL_max = float(dL_of_z[-1])
        m = (dL_grid >= dL_min) & (dL_grid <= dL_max) & (dL_grid > 0.0)
        if not np.any(m):
            continue

        z_of_dL = np.interp(dL_grid[m], dL_of_z, z)
        q_of_dL = np.interp(z_of_dL, z, q)
        scale = (constants.c_km_s / float(H0)) ** 2
        host = scale * q_of_dL * inv_pi[m]
        host_w[m, j] = host * w_dl[m]

    # Integrate per-pixel distance PDFs against host_w, then sum over pixels weighted by prob_pix.
    L = np.zeros((H0_grid.size,), dtype=float)
    n_pix = int(prob_pix.size)
    for a in range(0, n_pix, int(pixel_chunk_size)):
        b = min(n_pix, a + int(pixel_chunk_size))
        p = prob_pix[a:b]
        mu = distmu[a:b]
        sig = distsigma[a:b]
        norm = distnorm[a:b]

        pdf = ligo_distance.conditional_pdf(dL_grid.reshape((1, -1)), mu.reshape((-1, 1)), sig.reshape((-1, 1)), norm.reshape((-1, 1)))
        pdf = np.clip(np.asarray(pdf, dtype=float), 1e-300, np.inf)  # (chunk, n_dL)

        proj = pdf @ host_w  # (chunk, n_H0)
        L += p @ proj

    L = np.clip(L, 1e-300, np.inf)
    return np.log(L)


def compute_missing_host_logL_h0_grid_from_histogram(
    *,
    prob_pix: np.ndarray,
    pdf_bins: np.ndarray,
    dL_edges: np.ndarray,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    z_max: float,
    host_prior_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform",
    host_prior_z_k: float = 0.0,
    gw_distance_prior: GWDistancePrior,
    pixel_chunk_size: int = 5_000,
) -> np.ndarray:
    """Compute missing-host log-likelihood on an H0 grid from a (pix, dL) posterior histogram.

    This is the H0-grid analogue of `dark_sirens_incompleteness.compute_missing_host_logL_draws_from_histogram`,
    but for a GR LCDM distance-redshift relation with H0 as a grid parameter.
    """
    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")

    prob_pix = np.asarray(prob_pix, dtype=float)
    pdf_bins = np.asarray(pdf_bins, dtype=float)
    dL_edges = np.asarray(dL_edges, dtype=float)
    if prob_pix.ndim != 1:
        raise ValueError("prob_pix must be 1D.")
    if pdf_bins.ndim != 2:
        raise ValueError("pdf_bins must be 2D (n_pix, n_bins).")
    if dL_edges.ndim != 1 or dL_edges.size < 3:
        raise ValueError("dL_edges must be 1D with >=3 entries (>=2 bins).")
    if np.any(~np.isfinite(dL_edges)) or np.any(np.diff(dL_edges) <= 0.0):
        raise ValueError("dL_edges must be finite and strictly increasing.")
    if pdf_bins.shape[0] != prob_pix.size:
        raise ValueError("pdf_bins.shape[0] must match prob_pix.size.")
    if pdf_bins.shape[1] != dL_edges.size - 1:
        raise ValueError("pdf_bins.shape[1] must equal len(dL_edges)-1.")
    if prob_pix.size == 0:
        raise ValueError("No valid sky pixels provided for missing-host term.")

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    if int(pixel_chunk_size) <= 0:
        raise ValueError("pixel_chunk_size must be positive.")

    constants = PhysicalConstants()

    # Midpoints + widths for ddL integral.
    widths = np.diff(dL_edges)
    dL_mid = 0.5 * (dL_edges[:-1] + dL_edges[1:])
    inv_pi = np.exp(-gw_distance_prior.log_pi_dL(np.clip(dL_mid, 1e-6, np.inf)))

    # Build dimensionless q(z) factor on the cache z-grid (truncated at z_max).
    z = np.asarray(dist_cache.z_grid, dtype=float)
    f = np.asarray(dist_cache.f_grid, dtype=float)
    m_z = z <= z_max
    z = z[m_z]
    f = f[m_z]
    if z.size < 5:
        raise ValueError("z_max too small relative to LCDM distance cache.")

    om = float(dist_cache.omega_m0)
    ok = float(dist_cache.omega_k0)
    ol = 1.0 - om - ok
    Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
    Ez2 = np.clip(Ez2, 1e-15, np.inf)
    E = np.sqrt(Ez2)

    Sk = f / np.clip(1.0 + z, 1e-12, np.inf)
    dfdz = np.gradient(f, z)
    dfdz = np.clip(dfdz, 1e-12, np.inf)

    if host_prior_z_mode in ("none", "comoving_uniform"):
        rho = np.ones_like(z, dtype=float)
    elif host_prior_z_mode == "comoving_powerlaw":
        rho = np.clip(1.0 + z, 1e-12, np.inf) ** float(host_prior_z_k)
    else:
        raise ValueError("Unknown host_prior_z_mode.")

    q = (Sk**2) * rho / np.clip(E * dfdz, 1e-30, np.inf)

    # host_w: (n_bins, n_H0), includes ddL weights.
    host_w = np.zeros((dL_mid.size, H0_grid.size), dtype=float)
    for j, H0 in enumerate(H0_grid.tolist()):
        dL_of_z = (constants.c_km_s / float(H0)) * f
        if not np.all(np.isfinite(dL_of_z)) or np.any(np.diff(dL_of_z) <= 0.0):
            raise ValueError("Non-monotone/invalid dL(z) encountered; cannot invert for missing-host term.")

        dL_min = float(dL_of_z[0])
        dL_max = float(dL_of_z[-1])
        m = (dL_mid >= dL_min) & (dL_mid <= dL_max) & (dL_mid > 0.0)
        if not np.any(m):
            continue

        z_of_dL = np.interp(dL_mid[m], dL_of_z, z)
        q_of_dL = np.interp(z_of_dL, z, q)
        scale = (constants.c_km_s / float(H0)) ** 2
        host = scale * q_of_dL * inv_pi[m]
        host_w[m, j] = host * widths[m]

    # Integrate per-pixel histogram PDFs against host_w, then sum over pixels weighted by prob_pix.
    L = np.zeros((H0_grid.size,), dtype=float)
    n_pix = int(prob_pix.size)
    for a in range(0, n_pix, int(pixel_chunk_size)):
        b = min(n_pix, a + int(pixel_chunk_size))
        p = prob_pix[a:b]
        pdf = np.clip(np.asarray(pdf_bins[a:b, :], dtype=float), 0.0, np.inf)
        proj = pdf @ host_w
        L += p @ proj

    L = np.clip(L, 1e-300, np.inf)
    return np.log(L)


def _injection_weights(
    injections: O3aBbhInjectionSet,
    *,
    weight_mode: Literal["none", "inv_sampling_pdf"],
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_powerlaw_k: float,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_m_taper_delta: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    z_max: float,
    inj_mass_pdf_coords: Literal["m1m2", "m1q"] = "m1m2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered (z, dL_fid, snr, found_ifar, w) arrays for alpha(H0)."""
    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr = np.asarray(injections.snr_net_opt, dtype=float)
    found = np.asarray(injections.found_ifar, dtype=bool)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)

    z_hi = float(z_max)
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr)
        & (snr > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError("No injections remain after z/dL/SNR cuts.")

    z = z[m]
    dL_fid = dL_fid[m]
    snr = snr[m]
    found = found[m]
    m1 = m1[m]
    m2 = m2[m]

    w = np.ones_like(z, dtype=float)
    # Mixture-model injections provide an additional mixture weight (close to unity).
    if hasattr(injections, "mixture_weight"):
        mw = np.asarray(getattr(injections, "mixture_weight"), dtype=float)
        if mw.shape == m.shape:
            mw = mw[m]
        if mw.shape != z.shape:
            raise ValueError("injections.mixture_weight must match injections.z shape.")
        w = w * mw

    if weight_mode == "none":
        pass
    elif weight_mode == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite or non-positive values.")
        w = w / pdf
    else:
        raise ValueError("Unknown weight_mode.")

    if pop_z_mode != "none":
        # Same simple LCDM approximation used in dark_sirens_selection.py (population factor only).
        H0 = 67.7  # km/s/Mpc
        om0 = 0.31
        c = 299792.458  # km/s
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        dc = (c / H0) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        dVdz = (c / (H0 * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        base = dVdz / (1.0 + z)
        if pop_z_mode == "comoving_uniform":
            w = w * base
        elif pop_z_mode == "comoving_powerlaw":
            w = w * base * (1.0 + z) ** float(pop_z_powerlaw_k)
        else:
            raise ValueError("Unknown pop_z_mode.")

    if pop_mass_mode != "none":
        if weight_mode == "inv_sampling_pdf":
            # Population mass models are parameterized in (m1_source, q=m2/m1). If the injection
            # `sampling_pdf` is provided in (m1_source, m2_source) coordinates, we must include
            # the Jacobian dm2 = m1 dq to convert to the (m1,q) density used by the population.
            if inj_mass_pdf_coords == "m1m2":
                w = w / np.clip(m1, 1e-300, np.inf)
            elif inj_mass_pdf_coords == "m1q":
                pass
            else:
                raise ValueError("Unknown inj_mass_pdf_coords (expected 'm1m2' or 'm1q').")

        alpha = float(pop_m1_alpha)
        mmin = float(pop_m_min)
        mmax = float(pop_m_max)
        beta_q = float(pop_q_beta)
        q = np.clip(m2 / m1, 1e-6, 1.0)
        if pop_mass_mode == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        elif pop_mass_mode == "powerlaw_q_smooth":
            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for pop_mass_mode=powerlaw_q_smooth.")
            t1 = _sigmoid_stable((m1 - mmin) / delta) * _sigmoid_stable((mmax - m1) / delta)
            t2 = _sigmoid_stable((m2 - mmin) / delta) * _sigmoid_stable((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 0.0, 1.0)
            w = w * np.clip(taper, 0.0, 1.0) * (m1 ** (-alpha)) * (q ** beta_q)
        elif pop_mass_mode == "powerlaw_peak_q_smooth":
            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for pop_mass_mode=powerlaw_peak_q_smooth.")
            mp = float(pop_m_peak)
            sig = float(pop_m_peak_sigma)
            f_peak = float(pop_m_peak_frac)
            if not (np.isfinite(mp) and mp > 0.0 and np.isfinite(sig) and sig > 0.0):
                raise ValueError("pop_m_peak and pop_m_peak_sigma must be finite and positive for pop_mass_mode=powerlaw_peak_q_smooth.")
            if not (np.isfinite(f_peak) and 0.0 <= f_peak <= 1.0):
                raise ValueError("pop_m_peak_frac must be finite and in [0,1] for pop_mass_mode=powerlaw_peak_q_smooth.")

            t1 = _sigmoid_stable((m1 - mmin) / delta) * _sigmoid_stable((mmax - m1) / delta)
            t2 = _sigmoid_stable((m2 - mmin) / delta) * _sigmoid_stable((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 1e-300, 1.0)

            log_q = beta_q * np.log(np.clip(q, 1e-300, np.inf))
            log_taper = np.log(taper)
            log_pl = -alpha * np.log(np.clip(m1, 1e-300, np.inf)) + log_q + log_taper
            log_peak = -0.5 * ((m1 - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper

            if f_peak <= 0.0:
                log_mass = log_pl
            elif f_peak >= 1.0:
                log_mass = log_peak
            else:
                log_mass = logsumexp(
                    np.stack([np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak], axis=0),
                    axis=0,
                )
            # Only relative weights matter for alpha; rescale for numerical stability.
            m_ok = np.isfinite(log_mass)
            if not np.any(m_ok):
                raise ValueError("Mass weights are non-finite under pop_mass_mode=powerlaw_peak_q_smooth.")
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)
        else:
            raise ValueError("Unknown pop_mass_mode.")

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr = snr[good_w]
        found = found[good_w]
        w = w[good_w]

    return z, dL_fid, snr, found, w


def _alpha_h0_grid_from_injections(
    *,
    injections: O3aBbhInjectionSet,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    z_max: float,
    det_model: Literal["threshold", "snr_binned"],
    snr_threshold: float | None,
    snr_binned_nbins: int,
    weight_mode: Literal["none", "inv_sampling_pdf"],
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_powerlaw_k: float,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_m_taper_delta: float = 0.0,
    pop_m_peak: float = 35.0,
    pop_m_peak_sigma: float = 5.0,
    pop_m_peak_frac: float = 0.1,
    inj_mass_pdf_coords: Literal["m1m2", "m1q"] = "m1m2",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute alpha(H0) on a grid using the same injection-rescaling proxy approach."""
    z, dL_fid, snr, found, w = _injection_weights(
        injections,
        weight_mode=weight_mode,
        pop_z_mode=pop_z_mode,
        pop_z_powerlaw_k=pop_z_powerlaw_k,
        pop_mass_mode=pop_mass_mode,
        pop_m1_alpha=pop_m1_alpha,
        pop_m_min=pop_m_min,
        pop_m_max=pop_m_max,
        pop_q_beta=pop_q_beta,
        pop_m_taper_delta=pop_m_taper_delta,
        pop_m_peak=pop_m_peak,
        pop_m_peak_sigma=pop_m_peak_sigma,
        pop_m_peak_frac=pop_m_peak_frac,
        z_max=z_max,
        inj_mass_pdf_coords=inj_mass_pdf_coords,
    )

    det = _calibrate_detection_model_from_snr_and_found(
        snr_net_opt=snr,
        found_ifar=found,
        det_model=det_model,
        snr_threshold=snr_threshold,
        snr_binned_nbins=int(snr_binned_nbins),
    )
    thresh = det.snr_threshold

    # Precompute the dimensionless distance factor once for injection redshifts.
    fz = dist_cache.f(z)
    if not np.all(np.isfinite(fz)) or np.any(fz <= 0.0):
        raise ValueError("Invalid f(z) for injections; check z_max and omega parameters.")

    constants = PhysicalConstants()
    alpha = np.empty((H0_grid.size,), dtype=float)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Sum of injection weights is non-positive.")

    # Loop over H0 grid values to keep memory bounded.
    for j, H0 in enumerate(np.asarray(H0_grid, dtype=float).tolist()):
        dL_model = (constants.c_km_s / float(H0)) * fz
        dL_model = np.clip(dL_model, 1e-6, np.inf)
        snr_rescaled = snr * (dL_fid / dL_model)
        pdet = det.pdet(snr_rescaled)
        alpha[j] = float(np.sum(w * pdet) / wsum)

    meta = {
        "method": "o3a_injections_snr_rescale_gr_h0_grid",
        "det_model": str(det_model),
        "snr_threshold": float(thresh) if thresh is not None else None,
        "snr_binned_nbins": int(snr_binned_nbins),
        "weight_mode": str(weight_mode),
        "pop_z_mode": str(pop_z_mode),
        "pop_z_k": float(pop_z_powerlaw_k),
        "pop_mass_mode": str(pop_mass_mode),
        "inj_mass_pdf_coords": str(inj_mass_pdf_coords),
        "pop_m1_alpha": float(pop_m1_alpha),
        "pop_m_min": float(pop_m_min),
        "pop_m_max": float(pop_m_max),
        "pop_q_beta": float(pop_q_beta),
        "pop_m_taper_delta": float(pop_m_taper_delta),
        "pop_m_peak": float(pop_m_peak),
        "pop_m_peak_sigma": float(pop_m_peak_sigma),
        "pop_m_peak_frac": float(pop_m_peak_frac),
        "n_injections_used": int(z.size),
        "z_max": float(z_max),
    }
    return alpha, meta


def compute_gr_h0_posterior_grid(
    *,
    events: list[dict[str, Any]],
    H0_grid: np.ndarray,
    omega_m0: float,
    omega_k0: float,
    cache_dir: str | Path | None = None,
    mixture_mode: Literal["none", "simple"] = "none",
    f_miss: float = 0.0,
    p_credible: float = 0.9,
    nside_coarse: int | None = None,
    host_prior_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform",
    host_prior_z_k: float = 0.0,
    missing_z_max: float | None = None,
    missing_dl_grid_n: int = 200,
    missing_dl_min_mpc: float = 1.0,
    missing_dl_max_mpc: float | None = None,
    missing_dl_nsigma: float = 5.0,
    missing_pixel_chunk_size: int = 5_000,
    injections: O3aBbhInjectionSet | None,
    ifar_threshold_yr: float,
    z_max: float,
    det_model: str,
    snr_threshold: float | None,
    snr_binned_nbins: int,
    weight_mode: str,
    pop_z_mode: str,
    pop_z_powerlaw_k: float,
    pop_mass_mode: str,
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    prior: str = "uniform",
    gw_distance_prior: GWDistancePrior | None = None,
) -> dict[str, Any]:
    """Compute a GR H0 posterior on a grid using cached per-event selections.

    This is a "published-control" plumbing path: it reuses the same sky-map proxy likelihood and
    optional selection normalization machinery, but treats GR LCDM distances with H0 as the sole
    cosmological degree of freedom on a grid.

    Events may be supplied in either:
      - 3D skymap mode: provide `skymap_path` and galaxy `ipix` at the skymap nside, or
      - PE-histogram mode: provide a binned posterior histogram (`pe_*` arrays) and galaxy `ipix`
        at the PE histogram nside.
    """
    if prior != "uniform":
        raise ValueError("Only prior='uniform' is implemented for now.")

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    if not events:
        raise ValueError("No events provided.")

    constants = PhysicalConstants()
    gw_distance_prior = gw_distance_prior or GWDistancePrior()
    if mixture_mode not in ("none", "simple"):
        raise ValueError("Unknown mixture_mode.")
    if mixture_mode == "simple":
        f_miss = float(f_miss)
        if not (0.0 <= f_miss <= 1.0) or not np.isfinite(f_miss):
            raise ValueError("f_miss must be finite and in [0,1].")
        if not (0.0 < float(p_credible) <= 1.0):
            raise ValueError("p_credible must be in (0,1].")
        if missing_z_max is None:
            missing_z_max = float(z_max)
        if not (np.isfinite(float(missing_z_max)) and float(missing_z_max) > 0.0):
            raise ValueError("missing_z_max must be finite and positive.")
        if int(missing_dl_grid_n) < 50:
            raise ValueError("missing_dl_grid_n too small (need >=50).")
        if not (np.isfinite(float(missing_dl_min_mpc)) and float(missing_dl_min_mpc) > 0.0):
            raise ValueError("missing_dl_min_mpc must be finite and positive.")
        if int(missing_pixel_chunk_size) <= 0:
            raise ValueError("missing_pixel_chunk_size must be positive.")

    # Build a single distance cache up to the maximum galaxy redshift across all cached events.
    z_max_events = 0.0
    for e in events:
        z = np.asarray(e["z"], dtype=float)
        if z.size:
            z_max_events = max(z_max_events, float(np.nanmax(z)))
    z_max_cache = max(z_max_events, float(z_max))
    dist_cache = _build_lcdm_distance_cache(z_max=z_max_cache, omega_m0=float(omega_m0), omega_k0=float(omega_k0))

    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir).expanduser().resolve()
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Event likelihoods on the H0 grid.
    per_event = []
    logL_total = np.zeros_like(H0_grid, dtype=float)
    for e in events:
        ev = str(e["event"])
        gw_data_mode = str(e.get("gw_data_mode", "skymap"))
        if gw_data_mode not in ("skymap", "pe"):
            raise ValueError(f"{ev}: unknown gw_data_mode '{gw_data_mode}'.")

        sky = None
        sky_path = None
        if gw_data_mode == "skymap":
            sky_path = str(e["skymap_path"])
            sky = read_skymap_3d(sky_path, nest=True)

        want_meta = {
            "event": str(ev),
            "gw_data_mode": str(gw_data_mode),
            "skymap_path": str(sky_path) if sky_path is not None else None,
            "pe_file": str(e.get("pe_file")) if gw_data_mode == "pe" else None,
            "pe_analysis": str(e.get("pe_analysis")) if gw_data_mode == "pe" and e.get("pe_analysis") is not None else None,
            "pe_nside": int(e.get("pe_nside")) if gw_data_mode == "pe" and e.get("pe_nside") is not None else None,
            "pe_dl_nbins": int(np.asarray(e.get("pe_dL_edges")).size - 1) if gw_data_mode == "pe" and e.get("pe_dL_edges") is not None else None,
            "omega_m0": float(omega_m0),
            "omega_k0": float(omega_k0),
            "H0_grid": [float(x) for x in H0_grid.tolist()],
            "gw_distance_prior": gw_distance_prior.to_jsonable(),
            "mixture_mode": str(mixture_mode),
            "f_miss": float(f_miss),
            "p_credible": float(p_credible) if mixture_mode == "simple" else None,
            "nside_coarse": int(nside_coarse) if nside_coarse is not None else None,
            "host_prior_z_mode": str(host_prior_z_mode) if mixture_mode == "simple" else None,
            "host_prior_z_k": float(host_prior_z_k) if mixture_mode == "simple" else None,
            "missing_z_max": float(missing_z_max) if (mixture_mode == "simple" and missing_z_max is not None) else None,
            "missing_dl_grid_n": int(missing_dl_grid_n) if mixture_mode == "simple" else None,
            "missing_dl_min_mpc": float(missing_dl_min_mpc) if mixture_mode == "simple" else None,
            "missing_dl_max_mpc": float(missing_dl_max_mpc) if (mixture_mode == "simple" and missing_dl_max_mpc is not None) else None,
            "missing_dl_nsigma": float(missing_dl_nsigma) if mixture_mode == "simple" else None,
            "missing_pixel_chunk_size": int(missing_pixel_chunk_size) if mixture_mode == "simple" else None,
        }

        cache_path: Path | None = None
        if cache_dir_path is not None:
            cache_path = cache_dir_path / f"{ev}__gr_h0_terms.npz"

        logL_cat: np.ndarray | None = None
        logL_missing: np.ndarray | None = None
        cache_hit = False
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=True) as d:
                    meta_m = json.loads(str(d["meta"].tolist()))
                    if meta_m == want_meta:
                        logL_cat = np.asarray(d["logL_cat"], dtype=float)
                        if mixture_mode == "simple":
                            logL_missing = np.asarray(d["logL_missing"], dtype=float)
                        cache_hit = True
            except Exception:
                logL_cat = None
                logL_missing = None
                cache_hit = False

        if logL_cat is None:
            if gw_data_mode == "skymap":
                assert sky is not None
                logL_cat = _event_logL_h0_grid(
                    sky=sky,
                    z_gal=np.asarray(e["z"], dtype=float),
                    w_gal=np.asarray(e["w"], dtype=float),
                    ipix_gal=np.asarray(e["ipix"], dtype=np.int64),
                    H0_grid=H0_grid,
                    dist_cache=dist_cache,
                    constants=constants,
                    gw_distance_prior=gw_distance_prior,
                )
            else:
                logL_cat = _event_logL_h0_grid_from_histogram(
                    pix_sel=np.asarray(e["pe_pix_sel"], dtype=np.int64),
                    prob_pix=np.asarray(e["pe_prob_pix"], dtype=float),
                    dL_edges=np.asarray(e["pe_dL_edges"], dtype=float),
                    pdf_bins=np.asarray(e["pe_pdf_bins"], dtype=float),
                    nside=int(e["pe_nside"]),
                    nest=True,
                    z_gal=np.asarray(e["z"], dtype=float),
                    w_gal=np.asarray(e["w"], dtype=float),
                    ipix_gal=np.asarray(e["ipix"], dtype=np.int64),
                    H0_grid=H0_grid,
                    dist_cache=dist_cache,
                    constants=constants,
                    gw_distance_prior=gw_distance_prior,
                )

        logL_ev = logL_cat
        if mixture_mode == "simple":
            if logL_missing is None:
                if gw_data_mode == "skymap":
                    assert sky is not None
                    hpix_sel = e.get("hpix_sel")
                    hpix_sel = None if hpix_sel is None else np.asarray(hpix_sel, dtype=np.int64)

                    prob_pix, distmu, distsigma, distnorm = select_missing_pixels(
                        sky,
                        p_credible=float(p_credible),
                        nside_coarse=int(nside_coarse) if nside_coarse is not None else None,
                        hpix_coarse=hpix_sel,
                    )

                    dl_min = float(missing_dl_min_mpc)
                    if missing_dl_max_mpc is not None:
                        dl_max = float(missing_dl_max_mpc)
                    else:
                        dl_max = float(np.max(distmu + float(missing_dl_nsigma) * distsigma))
                    if not np.isfinite(dl_max) or dl_max <= dl_min:
                        raise ValueError(f"{ev}: invalid missing dL grid bounds [{dl_min},{dl_max}].")
                    dL_grid = np.linspace(dl_min, dl_max, int(missing_dl_grid_n))

                    logL_missing = compute_missing_host_logL_h0_grid_from_pixels(
                        prob_pix=prob_pix,
                        distmu=distmu,
                        distsigma=distsigma,
                        distnorm=distnorm,
                        H0_grid=H0_grid,
                        dist_cache=dist_cache,
                        dL_grid=dL_grid,
                        z_max=float(missing_z_max),
                        host_prior_z_mode=host_prior_z_mode,
                        host_prior_z_k=float(host_prior_z_k),
                        gw_distance_prior=gw_distance_prior,
                        pixel_chunk_size=int(missing_pixel_chunk_size),
                    )
                else:
                    logL_missing = compute_missing_host_logL_h0_grid_from_histogram(
                        prob_pix=np.asarray(e["pe_prob_pix"], dtype=float),
                        pdf_bins=np.asarray(e["pe_pdf_bins"], dtype=float),
                        dL_edges=np.asarray(e["pe_dL_edges"], dtype=float),
                        H0_grid=H0_grid,
                        dist_cache=dist_cache,
                        z_max=float(missing_z_max),
                        host_prior_z_mode=host_prior_z_mode,
                        host_prior_z_k=float(host_prior_z_k),
                        gw_distance_prior=gw_distance_prior,
                        pixel_chunk_size=int(missing_pixel_chunk_size),
                    )

            log_a = np.log1p(-float(f_miss)) if float(f_miss) < 1.0 else -np.inf
            log_b = np.log(float(f_miss)) if float(f_miss) > 0.0 else -np.inf
            logL_ev = np.logaddexp(log_a + logL_cat, log_b + logL_missing)

        if cache_path is not None and not cache_hit:
            # Only write after successful computation; ignore failures silently (cache is optional).
            try:
                if mixture_mode == "simple":
                    np.savez_compressed(
                        cache_path,
                        meta=json.dumps(want_meta, sort_keys=True),
                        logL_cat=np.asarray(logL_cat, dtype=np.float64),
                        logL_missing=np.asarray(logL_missing, dtype=np.float64),
                    )
                else:
                    np.savez_compressed(
                        cache_path,
                        meta=json.dumps(want_meta, sort_keys=True),
                        logL_cat=np.asarray(logL_cat, dtype=np.float64),
                    )
            except Exception:
                pass

        logL_total += logL_ev
        row: dict[str, Any] = {"event": ev, "logL_H0": [float(x) for x in logL_ev.tolist()]}
        if mixture_mode == "simple":
            row.update(
                {
                    "logL_H0_cat": [float(x) for x in logL_cat.tolist()],
                    "logL_H0_missing": [float(x) for x in (np.asarray(logL_missing) if logL_missing is not None else np.full_like(logL_cat, np.nan)).tolist()],
                    "f_miss": float(f_miss),
                }
            )
        per_event.append(row)

    alpha_meta = None
    alpha_grid = None
    if injections is not None:
        alpha, alpha_meta = _alpha_h0_grid_from_injections(
            injections=injections,
            H0_grid=H0_grid,
            dist_cache=dist_cache,
            z_max=float(z_max),
            det_model=det_model,  # type: ignore[arg-type]
            snr_threshold=snr_threshold,
            snr_binned_nbins=int(snr_binned_nbins),
            weight_mode=weight_mode,  # type: ignore[arg-type]
            pop_z_mode=pop_z_mode,  # type: ignore[arg-type]
            pop_z_powerlaw_k=float(pop_z_powerlaw_k),
            pop_mass_mode=pop_mass_mode,  # type: ignore[arg-type]
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
        )
        alpha_grid = np.asarray(alpha, dtype=float)
        logL_total = logL_total - float(len(events)) * np.log(np.clip(alpha, 1e-300, np.inf))

    # Uniform prior in H0 => constant in log space (drops out).
    log_post = logL_total - float(np.max(logL_total))
    p = np.exp(log_post)
    p = p / float(np.sum(p))

    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))

    mean = float(np.sum(p * H0_grid))
    sd = float(np.sqrt(np.sum(p * (H0_grid - mean) ** 2)))
    H0_map = float(H0_grid[int(np.argmax(p))])

    return {
        "method": "gr_h0_grid",
        "prior": str(prior),
        "omega_m0": float(omega_m0),
        "omega_k0": float(omega_k0),
        "n_events": int(len(events)),
        "events": per_event,
        "mixture": {
            "mode": str(mixture_mode),
            "f_miss": float(f_miss),
            "p_credible": float(p_credible),
            "nside_coarse": int(nside_coarse) if nside_coarse is not None else None,
            "host_prior_z_mode": str(host_prior_z_mode),
            "host_prior_z_k": float(host_prior_z_k),
            "missing_z_max": float(missing_z_max) if missing_z_max is not None else None,
            "missing_dl_grid_n": int(missing_dl_grid_n),
            "missing_dl_min_mpc": float(missing_dl_min_mpc),
            "missing_dl_max_mpc": float(missing_dl_max_mpc) if missing_dl_max_mpc is not None else None,
            "missing_dl_nsigma": float(missing_dl_nsigma),
            "missing_pixel_chunk_size": int(missing_pixel_chunk_size),
        }
        if mixture_mode != "none"
        else {"mode": "none"},
        "H0_map": float(H0_map),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "logL_H0_rel": [float(x) for x in log_post.tolist()],
        "posterior": [float(x) for x in p.tolist()],
        "summary": {"mean": mean, "sd": sd, "p50": q50, "p16": q16, "p84": q84},
        "selection_alpha": alpha_meta,
        "selection_alpha_grid": [float(x) for x in alpha_grid.tolist()] if alpha_grid is not None else None,
        "selection_ifar_threshold_yr": float(ifar_threshold_yr),
        "gw_distance_prior": gw_distance_prior.to_jsonable(),
    }


def _event_logL_h0_grid_from_hierarchical_pe_samples(
    *,
    pe: GWTCPeHierarchicalSamples,
    H0_grid: np.ndarray,
    dist_cache: LCDMDistanceCache,
    constants: PhysicalConstants,
    z_max: float,
    det_model: CalibratedDetectionModel | None = None,
    include_pdet_in_event_term: bool = False,
    pop_z_include_h0_volume_scaling: bool = False,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_m_taper_delta: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    importance_smoothing: Literal["none", "truncate", "psis"] = "none",
    importance_truncate_tau: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Compute per-event hierarchical PE log-likelihood on an H0 grid (GR distances).

    Uses the standard posterior-sample reweighting identity:

      p(d | H0) ∝ ⟨ p_pop(theta | H0) / π_PE(theta) ⟩_{theta ~ p(theta|d)}
    """
    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")

    dL_s = np.asarray(pe.dL_mpc, dtype=float)
    mc_det = np.asarray(pe.chirp_mass_det, dtype=float)
    q = np.asarray(pe.mass_ratio, dtype=float)
    log_pi_dL = np.asarray(pe.log_pi_dL, dtype=float)
    log_pi_mc = np.asarray(pe.log_pi_chirp_mass, dtype=float)
    log_pi_q = np.asarray(pe.log_pi_mass_ratio, dtype=float)
    if not (
        dL_s.ndim == mc_det.ndim == q.ndim == log_pi_dL.ndim == log_pi_mc.ndim == log_pi_q.ndim == 1
        and dL_s.size == mc_det.size == q.size == log_pi_dL.size == log_pi_mc.size == log_pi_q.size
    ):
        raise ValueError("PE samples must be 1D arrays with matching sizes.")
    n_samp = int(dL_s.size)
    if n_samp < 1000:
        raise ValueError("Too few PE samples for hierarchical reweighting (need >=1000).")
    if not np.all(np.isfinite(dL_s)) or np.any(dL_s <= 0.0):
        raise ValueError("Non-finite/non-positive dL samples.")

    log_pdet_samp: np.ndarray | None = None
    if bool(include_pdet_in_event_term):
        if det_model is None:
            raise ValueError("include_pdet_in_event_term=True requires det_model to be provided.")
        snr_ref = pe.snr_net_opt_ref
        dL_ref = pe.dL_mpc_ref
        if snr_ref is None or dL_ref is None:
            raise ValueError("PE samples missing (snr_net_opt_ref, dL_mpc_ref) required for include_pdet_in_event_term=True.")
        snr_norm = float(snr_ref) * float(dL_ref)
        if not (np.isfinite(snr_norm) and snr_norm > 0.0):
            raise ValueError("Invalid snr_norm from (snr_net_opt_ref, dL_mpc_ref).")
        snr_samp = snr_norm / np.clip(dL_s, 1e-12, np.inf)
        pdet = det_model.pdet(snr_samp)
        log_pdet_samp = np.log(np.clip(pdet, 1e-300, 1.0))

    z_grid = np.asarray(dist_cache.z_grid, dtype=float)
    f_grid = np.asarray(dist_cache.f_grid, dtype=float)
    if z_grid.ndim != 1 or f_grid.ndim != 1 or z_grid.shape != f_grid.shape:
        raise ValueError("Invalid dist_cache grids.")
    if z_grid.size < 200 or np.any(np.diff(z_grid) <= 0.0):
        raise ValueError("dist_cache.z_grid must be strictly increasing with sufficient resolution.")
    # f(z) is allowed to be 0 at z=0, but must be strictly increasing for inversion.
    if np.any(~np.isfinite(f_grid)) or np.any(f_grid < 0.0) or np.any(np.diff(f_grid) <= 0.0):
        raise ValueError("dist_cache.f_grid must be finite, non-negative, and strictly increasing for inversion.")

    df_dz_grid = np.gradient(f_grid, z_grid)
    if np.any(~np.isfinite(df_dz_grid)) or np.any(df_dz_grid <= 0.0):
        raise ValueError("Non-positive df/dz encountered in distance cache; cannot invert.")

    logL = np.full((H0_grid.size,), -np.inf, dtype=float)
    ess = np.zeros((H0_grid.size,), dtype=float)
    n_good = np.zeros((H0_grid.size,), dtype=float)

    for j, H0 in enumerate(np.asarray(H0_grid, dtype=float).tolist()):
        # Invert dL(z;H0) = (c/H0) * f(z) => f = dL*H0/c.
        f_samp = dL_s * float(H0) / float(constants.c_km_s)
        z_samp = np.interp(f_samp, f_grid, z_grid, left=np.nan, right=np.nan)

        df_dz_s = np.interp(z_samp, z_grid, df_dz_grid, left=np.nan, right=np.nan)
        ddLdz_s = (float(constants.c_km_s) / float(H0)) * df_dz_s

        good = np.isfinite(z_samp) & (z_samp > 0.0) & (z_samp <= z_max) & np.isfinite(ddLdz_s) & (ddLdz_s > 0.0)
        n_good[j] = float(np.count_nonzero(good))
        if not np.any(good):
            logL[j] = -np.inf
            ess[j] = 0.0
            continue

        log_jac = -np.log(ddLdz_s[good])
        if log_pdet_samp is None:
            log_pdet = 0.0
        else:
            log_pdet = log_pdet_samp[good]
        if pop_z_mode == "none":
            log_z = np.zeros_like(log_jac)
        else:
            # Comoving-uniform source-frame rate density in z (up to an overall constant).
            #
            # With dist_cache.f(z) ≡ (1+z) D_M * (H0/c), we have:
            #   base(z) ∝ (c/H0)^3 * f(z)^2 / [(1+z)^3 E(z)].
            #
            # By default we drop the explicit (c/H0)^3 factor (shape-only). For selection/coverage
            # audits it can be useful to include it so that numerator/selection normalizations share
            # the same H0 volume scaling.
            z_g = z_samp[good]
            f_g = np.asarray(f_samp[good], dtype=float)
            f_g = np.clip(f_g, 1e-300, np.inf)

            om = float(dist_cache.omega_m0)
            ok = float(dist_cache.omega_k0)
            ol = 1.0 - om - ok
            Ez2 = om * (1.0 + z_g) ** 3 + ok * (1.0 + z_g) ** 2 + ol
            Ez = np.sqrt(np.clip(Ez2, 1e-30, np.inf))

            log_base = 2.0 * np.log(f_g) - 3.0 * np.log1p(z_g) - np.log(Ez)
            if bool(pop_z_include_h0_volume_scaling):
                log_base = log_base + 3.0 * np.log(float(constants.c_km_s)) - 3.0 * np.log(float(H0))
            if pop_z_mode == "comoving_uniform":
                log_z = log_base
            elif pop_z_mode == "comoving_powerlaw":
                log_z = log_base + float(pop_z_k) * np.log1p(z_g)
            else:  # pragma: no cover
                raise ValueError("Unknown pop_z_mode.")

        log_m = np.zeros_like(log_z)
        log_mass_coord_jac = np.zeros_like(log_z)
        log_pi_denom = log_pi_dL[good]
        if pop_mass_mode != "none":
            log_pi_denom = log_pi_denom + log_pi_mc[good] + log_pi_q[good]
            z_g = z_samp[good]
            q_g = q[good]
            Mc_src = mc_det[good] / (1.0 + z_g)
            m1_src = _m1_source_from_chirp_mass_and_q(Mc_source=Mc_src, q=q_g)
            if pop_mass_mode == "powerlaw_q":
                log_m = _log_pop_weight_mass_source_powerlaw_q(
                    m1_src,
                    q_g,
                    alpha=float(pop_m1_alpha),
                    m_min=float(pop_m_min),
                    m_max=float(pop_m_max),
                    q_beta=float(pop_q_beta),
                )
            elif pop_mass_mode == "powerlaw_q_smooth":
                log_m = _log_pop_weight_mass_source_powerlaw_q_smooth(
                    m1_src,
                    q_g,
                    alpha=float(pop_m1_alpha),
                    m_min=float(pop_m_min),
                    m_max=float(pop_m_max),
                    q_beta=float(pop_q_beta),
                    m_taper_delta=float(pop_m_taper_delta),
                )
            elif pop_mass_mode == "powerlaw_peak_q_smooth":
                log_m = _log_pop_weight_mass_source_powerlaw_peak_q_smooth(
                    m1_src,
                    q_g,
                    alpha=float(pop_m1_alpha),
                    m_min=float(pop_m_min),
                    m_max=float(pop_m_max),
                    q_beta=float(pop_q_beta),
                    m_taper_delta=float(pop_m_taper_delta),
                    m_peak=float(pop_m_peak),
                    m_peak_sigma=float(pop_m_peak_sigma),
                    m_peak_frac=float(pop_m_peak_frac),
                )
            else:
                raise ValueError("Unknown pop_mass_mode.")
            log_mass_coord_jac = (1.0 / 5.0) * np.log1p(q_g) - (3.0 / 5.0) * np.log(np.clip(q_g, 1e-300, np.inf)) - np.log1p(z_g)

        logw = log_z + log_jac + log_pdet + log_m + log_mass_coord_jac - log_pi_denom
        sm = smooth_logweights(
            logw,
            method=importance_smoothing,
            truncate_tau=importance_truncate_tau,
        )
        log_sum = float(logsumexp(sm.logw))
        logL[j] = float(log_sum - np.log(float(n_samp)))
        ess[j] = float(sm.ess_smooth)

    if return_diagnostics:
        return logL, ess, n_good
    return logL


def compute_gr_h0_posterior_grid_hierarchical_pe(
    *,
    pe_by_event: dict[str, GWTCPeHierarchicalSamples],
    H0_grid: np.ndarray,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    cache_dir: str | Path | None = None,
    include_pdet_in_event_term: bool = False,
    pdet_model: CalibratedDetectionModel | None = None,
    pop_z_include_h0_volume_scaling: bool = False,
    injections: O3aBbhInjectionSet | None,
    ifar_threshold_yr: float,
    det_model: Literal["threshold", "snr_binned"],
    snr_threshold: float | None,
    snr_binned_nbins: int,
    weight_mode: Literal["none", "inv_sampling_pdf"],
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_powerlaw_k: float,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_m_taper_delta: float = 0.0,
    pop_m_peak: float = 35.0,
    pop_m_peak_sigma: float = 5.0,
    pop_m_peak_frac: float = 0.1,
    importance_smoothing: Literal["none", "truncate", "psis"] = "none",
    importance_truncate_tau: float | None = None,
    event_qc_mode: Literal["fail", "skip"] = "skip",
    event_min_finite_frac: float = 0.0,
    prior: Literal["uniform"] = "uniform",
) -> dict[str, Any]:
    """Compute a GR H0 posterior on a grid using hierarchical PE-sample reweighting + selection alpha(H0)."""
    if prior != "uniform":
        raise ValueError("Only prior='uniform' is implemented for now.")

    if not pe_by_event:
        raise ValueError("No events provided (pe_by_event empty).")

    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and strictly positive.")

    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")

    dist_cache = _build_lcdm_distance_cache(z_max=float(z_max), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    constants = PhysicalConstants()

    det_model_obj: CalibratedDetectionModel | None = pdet_model
    if bool(include_pdet_in_event_term) and det_model_obj is None:
        if injections is None:
            raise ValueError("include_pdet_in_event_term=True requires either pdet_model or injections for calibration.")
        # Match the filtering used by alpha(H0): use the same injection cuts and population support (via _injection_weights).
        _z, _dL_fid, _snr, _found, _w = _injection_weights(
            injections,
            weight_mode=weight_mode,
            pop_z_mode=pop_z_mode,
            pop_z_powerlaw_k=float(pop_z_powerlaw_k),
            pop_mass_mode=pop_mass_mode,
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
            pop_m_taper_delta=float(pop_m_taper_delta),
            pop_m_peak=float(pop_m_peak),
            pop_m_peak_sigma=float(pop_m_peak_sigma),
            pop_m_peak_frac=float(pop_m_peak_frac),
            z_max=float(z_max),
        )
        det_model_obj = _calibrate_detection_model_from_snr_and_found(
            snr_net_opt=_snr,
            found_ifar=_found,
            det_model=det_model,
            snr_threshold=snr_threshold,
            snr_binned_nbins=int(snr_binned_nbins),
        )

    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir).expanduser().resolve()
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    if str(event_qc_mode) not in ("fail", "skip"):
        raise ValueError("event_qc_mode must be one of {'fail','skip'}.")
    event_min_finite_frac = float(event_min_finite_frac)
    if not (np.isfinite(event_min_finite_frac) and 0.0 <= event_min_finite_frac <= 1.0):
        raise ValueError("event_min_finite_frac must be finite and in [0,1].")

    per_event: list[dict[str, Any]] = []
    skipped_events: list[dict[str, Any]] = []
    logL_sum_events = np.zeros_like(H0_grid, dtype=float)

    for ev in sorted(pe_by_event.keys()):
        pe = pe_by_event[ev]
        want_meta = {
            "event": str(ev),
            "pe_file": str(pe.file),
            "pe_analysis": str(pe.analysis),
            "n_samples": int(pe.n_used),
            "omega_m0": float(omega_m0),
            "omega_k0": float(omega_k0),
            "z_max": float(z_max),
            "H0_grid": [float(x) for x in H0_grid.tolist()],
            "pop_z_mode": str(pop_z_mode),
            "pop_z_k": float(pop_z_powerlaw_k),
            "pop_mass_mode": str(pop_mass_mode),
            "pop_m1_alpha": float(pop_m1_alpha),
            "pop_m_min": float(pop_m_min),
            "pop_m_max": float(pop_m_max),
            "pop_q_beta": float(pop_q_beta),
            "pop_m_taper_delta": float(pop_m_taper_delta),
            "pop_m_peak": float(pop_m_peak),
            "pop_m_peak_sigma": float(pop_m_peak_sigma),
            "pop_m_peak_frac": float(pop_m_peak_frac),
            "include_pdet_in_event_term": bool(include_pdet_in_event_term),
            "pop_z_include_h0_volume_scaling": bool(pop_z_include_h0_volume_scaling),
            "importance_smoothing": str(importance_smoothing),
            "importance_truncate_tau": float(importance_truncate_tau) if importance_truncate_tau is not None else None,
            "pdet_model": {
                "det_model": str(det_model_obj.det_model) if det_model_obj is not None else None,
                "snr_threshold": float(det_model_obj.snr_threshold) if det_model_obj is not None and det_model_obj.snr_threshold is not None else None,
                "snr_binned_nbins": int(snr_binned_nbins),
            }
            if bool(include_pdet_in_event_term)
            else None,
        }

        cache_path: Path | None = None
        if cache_dir_path is not None:
            cache_path = cache_dir_path / f"{str(ev)}__hier_gr_h0_terms.npz"

        logL_ev: np.ndarray | None = None
        ess_ev: np.ndarray | None = None
        n_good_ev: np.ndarray | None = None
        cache_hit = False
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=True) as d:
                    meta_m = json.loads(str(d["meta"].tolist()))
                    if meta_m == want_meta:
                        logL_ev = np.asarray(d["logL_H0"], dtype=float)
                        if "ess" in d and "n_good" in d:
                            ess_ev = np.asarray(d["ess"], dtype=float)
                            n_good_ev = np.asarray(d["n_good"], dtype=float)
                        cache_hit = True
            except Exception:
                logL_ev = None
                ess_ev = None
                n_good_ev = None
                cache_hit = False

        if logL_ev is None:
            out = _event_logL_h0_grid_from_hierarchical_pe_samples(
                pe=pe,
                H0_grid=H0_grid,
                dist_cache=dist_cache,
                constants=constants,
                z_max=float(z_max),
                det_model=det_model_obj,
                include_pdet_in_event_term=bool(include_pdet_in_event_term),
                pop_z_include_h0_volume_scaling=bool(pop_z_include_h0_volume_scaling),
                pop_z_mode=pop_z_mode,
                pop_z_k=float(pop_z_powerlaw_k),
                pop_mass_mode=pop_mass_mode,
                pop_m1_alpha=float(pop_m1_alpha),
                pop_m_min=float(pop_m_min),
                pop_m_max=float(pop_m_max),
                pop_q_beta=float(pop_q_beta),
                pop_m_taper_delta=float(pop_m_taper_delta),
                pop_m_peak=float(pop_m_peak),
                pop_m_peak_sigma=float(pop_m_peak_sigma),
                pop_m_peak_frac=float(pop_m_peak_frac),
                importance_smoothing=importance_smoothing,
                importance_truncate_tau=importance_truncate_tau,
                return_diagnostics=True,
            )
            assert isinstance(out, tuple)
            logL_ev, ess_ev, n_good_ev = out

            if cache_path is not None:
                np.savez(
                    cache_path,
                    meta=json.dumps(want_meta, sort_keys=True),
                    logL_H0=np.asarray(logL_ev, dtype=np.float64),
                    ess=np.asarray(ess_ev, dtype=np.float64),
                    n_good=np.asarray(n_good_ev, dtype=np.float64),
                )

        if not np.any(np.isfinite(np.asarray(logL_ev, dtype=float))):
            # This event has zero support under the specified (z_max, population) assumptions across the full H0 grid.
            # This most commonly occurs when the chosen population is a BBH-only model but the event is likely NSBH/BNS,
            # or when z_max is too small for the event's distance support.
            rec = {
                "event": str(ev),
                "reason": "no_finite_logL_across_H0_grid",
                "pe_file": str(pe.file),
                "pe_analysis": str(pe.analysis),
                "pe_n_samples": int(pe.n_used),
                "cache_hit": bool(cache_hit),
            }
            if event_qc_mode == "skip":
                skipped_events.append(rec)
                continue
            raise ValueError(
                f"{ev}: no finite logL across the provided H0 grid under the current population/z_max assumptions. "
                f"Either widen z_max/H0_grid, loosen population bounds (e.g. mass limits), or run with event_qc_mode='skip'."
            )

        fin = np.isfinite(np.asarray(logL_ev, dtype=float))
        fin_frac = float(np.mean(fin.astype(float))) if fin.size else 0.0
        if fin_frac < event_min_finite_frac:
            rec = {
                "event": str(ev),
                "reason": "insufficient_finite_support_across_H0_grid",
                "finite_frac": fin_frac,
                "pe_file": str(pe.file),
                "pe_analysis": str(pe.analysis),
                "pe_n_samples": int(pe.n_used),
                "cache_hit": bool(cache_hit),
            }
            if event_qc_mode == "skip":
                skipped_events.append(rec)
                continue
            raise ValueError(
                f"{ev}: finite support fraction across H0 grid is {fin_frac:.3f}, below event_min_finite_frac={event_min_finite_frac:.3f}. "
                "This usually indicates a population mismatch (hard support cutoffs) or an H0 grid/z_max that exceeds the event's mapped support."
            )

        per_event.append(
            {
                "event": str(ev),
                "pe_file": str(pe.file),
                "pe_analysis": str(pe.analysis),
                "pe_n_samples": int(pe.n_used),
                "cache_hit": bool(cache_hit),
                "include_pdet_in_event_term": bool(include_pdet_in_event_term),
                "pop_z_include_h0_volume_scaling": bool(pop_z_include_h0_volume_scaling),
                "importance_smoothing": str(importance_smoothing),
                "importance_truncate_tau": float(importance_truncate_tau) if importance_truncate_tau is not None else None,
                "logL_H0": [float(x) for x in np.asarray(logL_ev, dtype=float).tolist()],
                "ess_min": float(np.nanmin(np.asarray(ess_ev, dtype=float))) if ess_ev is not None and ess_ev.size else float("nan"),
                "n_good_min": float(np.nanmin(np.asarray(n_good_ev, dtype=float))) if n_good_ev is not None and n_good_ev.size else float("nan"),
                "finite_frac": fin_frac,
            }
        )
        logL_sum_events = logL_sum_events + np.asarray(logL_ev, dtype=float)

    logL_total = np.asarray(logL_sum_events, dtype=float)
    alpha_grid: np.ndarray | None = None
    alpha_meta: dict[str, Any] | None = None
    if injections is not None:
        alpha, meta = _alpha_h0_grid_from_injections(
            injections=injections,
            H0_grid=H0_grid,
            dist_cache=dist_cache,
            z_max=float(z_max),
            det_model=det_model,
            snr_threshold=snr_threshold,
            snr_binned_nbins=int(snr_binned_nbins),
            weight_mode=weight_mode,
            pop_z_mode=pop_z_mode,
            pop_z_powerlaw_k=float(pop_z_powerlaw_k),
            pop_mass_mode=pop_mass_mode,
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
            pop_m_taper_delta=float(pop_m_taper_delta),
            pop_m_peak=float(pop_m_peak),
            pop_m_peak_sigma=float(pop_m_peak_sigma),
            pop_m_peak_frac=float(pop_m_peak_frac),
        )
        alpha_grid = np.asarray(alpha, dtype=float)
        alpha_meta = meta
        log_alpha_grid = np.log(np.clip(alpha_grid, 1e-300, np.inf))
        logL_total = logL_total - float(len(per_event)) * log_alpha_grid
    else:
        log_alpha_grid = None

    if not np.any(np.isfinite(logL_total)):
        raise ValueError(
            "GR H0 grid posterior is undefined: log-likelihood is non-finite at all grid points. "
            "This usually means the population prior has zero support for the PE samples across the entire H0 grid "
            "(e.g. too-tight mass bounds) or prior-division mismatches. "
            "Try widening population support (e.g. pop_m_max), disabling restrictive population weights, "
            "or expanding z_max/H0_grid."
        )

    log_post = logL_total - float(np.max(logL_total))
    p = np.exp(log_post)
    p = p / float(np.sum(p))

    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))

    mean = float(np.sum(p * H0_grid))
    sd = float(np.sqrt(np.sum(p * (H0_grid - mean) ** 2)))
    i_map = int(np.argmax(p))
    H0_map = float(H0_grid[i_map])
    peak_at_edge = bool(i_map == 0 or i_map == (H0_grid.size - 1))
    gate2_pass = bool((not peak_at_edge) and (len(skipped_events) == 0))

    return {
        "method": "gr_h0_grid_hierarchical_pe",
        "prior": str(prior),
        "omega_m0": float(omega_m0),
        "omega_k0": float(omega_k0),
        "z_max": float(z_max),
        "n_events": int(len(per_event)),
        "n_events_skipped": int(len(skipped_events)),
        "events": per_event,
        "events_skipped": skipped_events,
        "H0_map": float(H0_map),
        "H0_map_index": int(i_map),
        "H0_map_at_edge": bool(peak_at_edge),
        "gate2_pass": bool(gate2_pass),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "logL_sum_events_rel": [float(x) for x in (logL_sum_events - float(np.max(logL_sum_events))).tolist()],
        "log_alpha_grid": [float(x) for x in log_alpha_grid.tolist()] if log_alpha_grid is not None else None,
        "logL_H0_rel": [float(x) for x in log_post.tolist()],
        "posterior": [float(x) for x in p.tolist()],
        "summary": {"mean": mean, "sd": sd, "p50": q50, "p16": q16, "p84": q84},
        "selection_alpha": alpha_meta,
        "selection_alpha_grid": [float(x) for x in alpha_grid.tolist()] if alpha_grid is not None else None,
        "selection_ifar_threshold_yr": float(ifar_threshold_yr),
        "include_pdet_in_event_term": bool(include_pdet_in_event_term),
        "pop_z_include_h0_volume_scaling": bool(pop_z_include_h0_volume_scaling),
        "importance_smoothing": str(importance_smoothing),
        "importance_truncate_tau": float(importance_truncate_tau) if importance_truncate_tau is not None else None,
        "pdet_model": {
            "det_model": str(det_model_obj.det_model) if det_model_obj is not None else None,
            "snr_threshold": float(det_model_obj.snr_threshold) if det_model_obj is not None and det_model_obj.snr_threshold is not None else None,
            "snr_binned_nbins": int(snr_binned_nbins),
        }
        if bool(include_pdet_in_event_term)
        else None,
        "population": {
            "pop_z_mode": str(pop_z_mode),
            "pop_z_k": float(pop_z_powerlaw_k),
            "pop_mass_mode": str(pop_mass_mode),
            "pop_m1_alpha": float(pop_m1_alpha),
            "pop_m_min": float(pop_m_min),
            "pop_m_max": float(pop_m_max),
            "pop_q_beta": float(pop_q_beta),
            "pop_m_taper_delta": float(pop_m_taper_delta),
            "pop_m_peak": float(pop_m_peak),
            "pop_m_peak_sigma": float(pop_m_peak_sigma),
            "pop_m_peak_frac": float(pop_m_peak_frac),
        },
        "event_qc_mode": str(event_qc_mode),
        "event_min_finite_frac": float(event_min_finite_frac),
    }
