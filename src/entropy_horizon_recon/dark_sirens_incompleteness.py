from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ligo.skymap import distance as ligo_distance

from .constants import PhysicalConstants
from .dark_sirens import SkyMap3D, credible_region_pixels
from .gw_distance_priors import GWDistancePrior
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class MissingHostPriorPrecompute:
    """Precomputed draw-wise background quantities for the missing-host (out-of-catalog) term.

    This object is designed to be computed once per posterior `post` and reused across many events.
    """

    z_grid: np.ndarray  # (n_z,)
    dL_em: np.ndarray  # (n_draws, n_z)
    dL_gw: np.ndarray  # (n_draws, n_z)
    base_z: np.ndarray  # (n_draws, n_z) proportional to rho_host(z) * dV/dz/dOmega
    ddLdz_em: np.ndarray  # (n_draws, n_z)
    ddLdz_gw: np.ndarray  # (n_draws, n_z)

    def to_jsonable(self) -> dict[str, Any]:
        # Avoid huge arrays; callers can save those separately if needed.
        return {
            "z_grid_min": float(self.z_grid[0]),
            "z_grid_max": float(self.z_grid[-1]),
            "n_draws": int(self.dL_em.shape[0]),
            "n_z": int(self.z_grid.size),
        }


def precompute_missing_host_prior(
    post: MuForwardPosterior,
    *,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    host_prior_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform",
    host_prior_z_k: float = 0.0,
    constants: PhysicalConstants | None = None,
) -> MissingHostPriorPrecompute:
    """Precompute draw-wise arrays needed to evaluate the missing-host integral.

    The missing-host likelihood uses a simple host density prior in comoving coordinates:
      rho_host(z) ∝ 1                      (comoving_uniform / none)
      rho_host(z) ∝ (1+z)^k               (comoving_powerlaw)

    and a geometric Jacobian:
      dV/dz/dOmega = (c/H(z)) * D_M(z)^2.

    For the sky-map proxy likelihood, we apply an explicit distance-prior correction by dividing
    by an assumed π(dL), consistent with `dark_sirens.compute_dark_siren_logL_draws`.
    """
    constants = constants or PhysicalConstants()
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")

    z_grid_full = np.asarray(post.z_grid, dtype=float)
    if z_grid_full.ndim != 1 or z_grid_full.size < 2 or float(z_grid_full[0]) != 0.0:
        raise ValueError("post.z_grid must be a 1D array starting at z=0 with at least two points.")

    # Limit to within the posterior grid (no extrapolation by default).
    z_hi = float(min(z_max, float(z_grid_full[-1])))
    m = z_grid_full <= z_hi
    z_grid = z_grid_full[m]
    if z_grid.size < 2:
        raise ValueError("z_max too small relative to posterior grid.")

    # Background distances for each draw.
    dL_em = predict_dL_em(post, z_eval=z_grid, constants=constants)  # (n_draws, n_z)
    _, R = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)
    dL_gw = dL_em * np.asarray(R, dtype=float)

    # Geometric piece dV/dz/dOmega = (c/H) * D_M^2.
    z = z_grid.reshape((1, -1))
    H = np.asarray(post.H_samples, dtype=float)[:, m]  # (n_draws, n_z)
    if H.shape != dL_em.shape:
        raise ValueError("Unexpected shape mismatch between H_samples and dL_em.")
    Dm = dL_em / np.clip(1.0 + z, 1e-12, np.inf)
    dVdz = (constants.c_km_s / np.clip(H, 1e-12, np.inf)) * (Dm**2)

    # Host density evolution factor (comoving).
    if host_prior_z_mode in ("none", "comoving_uniform"):
        rho = np.ones_like(z, dtype=float)
    elif host_prior_z_mode == "comoving_powerlaw":
        rho = np.clip(1.0 + z, 1e-12, np.inf) ** float(host_prior_z_k)
    else:
        raise ValueError("Unknown host_prior_z_mode.")

    base_z = dVdz * rho  # (n_draws, n_z)

    # Derivatives needed for dz/ddL = 1 / (ddL/dz).
    ddLdz_em = np.gradient(dL_em, z_grid, axis=1)
    ddLdz_gw = np.gradient(dL_gw, z_grid, axis=1)

    return MissingHostPriorPrecompute(
        z_grid=z_grid,
        dL_em=dL_em,
        dL_gw=dL_gw,
        base_z=base_z,
        ddLdz_em=ddLdz_em,
        ddLdz_gw=ddLdz_gw,
    )


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


def _host_prior_matrix_from_precompute(
    pre: MissingHostPriorPrecompute,
    *,
    dL_grid: np.ndarray,
    model: Literal["mu", "gr"],
    gw_distance_prior: GWDistancePrior,
) -> np.ndarray:
    """Build per-draw host prior factors on a dL grid.

    Returns an array of shape (n_draws, n_dL) with:

      host(dL) = [rho(z) * dV/dz/dOmega] * (dz/ddL) * (1/π(dL)),

    where z=z(dL) is defined by the chosen distance-redshift relation:
      - model='gr': dL(dz) = dL_EM(z)
      - model='mu': dL(dz) = dL_GW(z) (propagation-modified)

    This is intended for use inside:
      L_missing(draw) = sum_pix prob_pix * ∫ p(dL | pix, skymap) * host(draw; dL) ddL.
    """
    dL_grid = np.asarray(dL_grid, dtype=float)
    if dL_grid.ndim != 1 or dL_grid.size < 2:
        raise ValueError("dL_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(dL_grid)) or np.any(np.diff(dL_grid) <= 0.0):
        raise ValueError("dL_grid must be finite and strictly increasing.")

    if model == "gr":
        dL_of_z = pre.dL_em
        ddLdz = pre.ddLdz_em
    elif model == "mu":
        dL_of_z = pre.dL_gw
        ddLdz = pre.ddLdz_gw
    else:
        raise ValueError("model must be 'mu' or 'gr'.")

    z_grid = pre.z_grid
    n_draws = int(dL_of_z.shape[0])
    out = np.zeros((n_draws, dL_grid.size), dtype=float)

    log_pi = gw_distance_prior.log_pi_dL(dL_grid)
    inv_pi = np.zeros_like(log_pi, dtype=float)
    ok_pi = np.isfinite(log_pi)
    # If π(dL)=0 (log_pi=-inf) outside its support, treat 1/π as 0 rather than +∞; the posterior
    # provides no information there and dividing by π is undefined.
    inv_pi[ok_pi] = np.exp(-log_pi[ok_pi])

    for j in range(n_draws):
        dL_j = np.asarray(dL_of_z[j], dtype=float)
        if not np.all(np.isfinite(dL_j)) or np.any(np.diff(dL_j) <= 0.0):
            raise ValueError("Non-monotone or invalid dL(z) encountered; cannot invert for missing-host term.")

        # Valid only within the supported z-range.
        dL_min = float(dL_j[0])
        dL_max = float(dL_j[-1])
        m = (dL_grid >= dL_min) & (dL_grid <= dL_max) & (dL_grid > 0.0)
        if not np.any(m):
            continue

        z_of_dL = np.interp(dL_grid[m], dL_j, z_grid)
        base = np.interp(z_of_dL, z_grid, pre.base_z[j])
        dd = np.interp(z_of_dL, z_grid, ddLdz[j])
        dd = np.clip(dd, 1e-12, np.inf)
        out[j, m] = base / dd * inv_pi[m]

    return out


def select_missing_pixels(
    sky: SkyMap3D,
    *,
    p_credible: float,
    nside_coarse: int | None = None,
    hpix_coarse: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select sky pixels for the missing-host integral and return their distance-layer arrays.

    Preferred mode (fast, no global sorting):
    - Pass `hpix_coarse` at resolution `nside_coarse` (NESTED), e.g. the same pixels used to gather
      galaxies from the HEALPix-indexed catalog credible region.
    - If `sky.nside` is an integer multiple of `nside_coarse`, we include all child pixels in those
      coarse pixels.

    Fallback mode:
    - If no coarse pixels are provided (or the nsides are incompatible), compute the sky credible
      region directly at the native sky-map nside by sorting `sky.prob`.
    """
    if not sky.nest:
        raise ValueError("select_missing_pixels currently assumes sky arrays are NESTED ordered.")
    if not (0.0 < float(p_credible) <= 1.0):
        raise ValueError("p_credible must be in (0,1].")

    if hpix_coarse is not None:
        if nside_coarse is None:
            raise ValueError("nside_coarse must be provided when hpix_coarse is provided.")
        hpix_coarse = np.asarray(hpix_coarse, dtype=np.int64)
        if hpix_coarse.ndim != 1:
            raise ValueError("hpix_coarse must be 1D.")
        nside_coarse = int(nside_coarse)

        if sky.nside >= nside_coarse and sky.nside % nside_coarse == 0:
            ratio = sky.nside // nside_coarse
            if ratio & (ratio - 1) == 0:
                group = ratio * ratio
                offs = np.arange(group, dtype=np.int64)
                fine = (hpix_coarse.reshape((-1, 1)) * group + offs.reshape((1, -1))).reshape(-1)
                prob = sky.prob[fine]
                distmu = sky.distmu[fine]
                distsigma = sky.distsigma[fine]
                distnorm = sky.distnorm[fine]

                good = (
                    np.isfinite(prob)
                    & (prob > 0.0)
                    & np.isfinite(distmu)
                    & np.isfinite(distsigma)
                    & (distsigma > 0.0)
                    & np.isfinite(distnorm)
                    & (distnorm > 0.0)
                )
                return (
                    np.asarray(prob[good], dtype=float),
                    np.asarray(distmu[good], dtype=float),
                    np.asarray(distsigma[good], dtype=float),
                    np.asarray(distnorm[good], dtype=float),
                )

    # Fallback: native credible region by sorting sky.prob.
    idx = np.argsort(np.asarray(sky.prob, dtype=float))[::-1]
    csum = np.cumsum(np.asarray(sky.prob, dtype=float)[idx])
    k = int(np.searchsorted(csum, float(p_credible), side="left")) + 1
    sel = np.sort(idx[:k].astype(np.int64, copy=False))

    prob = sky.prob[sel]
    distmu = sky.distmu[sel]
    distsigma = sky.distsigma[sel]
    distnorm = sky.distnorm[sel]
    good = (
        np.isfinite(prob)
        & (prob > 0.0)
        & np.isfinite(distmu)
        & np.isfinite(distsigma)
        & (distsigma > 0.0)
        & np.isfinite(distnorm)
        & (distnorm > 0.0)
    )
    return (
        np.asarray(prob[good], dtype=float),
        np.asarray(distmu[good], dtype=float),
        np.asarray(distsigma[good], dtype=float),
        np.asarray(distnorm[good], dtype=float),
    )


def compute_missing_host_logL_draws(
    *,
    sky: SkyMap3D,
    hpix_coarse: np.ndarray | None,
    nside_coarse: int | None,
    p_credible: float,
    pre: MissingHostPriorPrecompute,
    dL_grid: np.ndarray,
    gw_distance_prior: GWDistancePrior,
    pixel_chunk_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-draw log-likelihood vectors for the missing-host mixture component.

    The missing-host term is:

      L_missing(draw) = sum_{pix in CR} prob_pix * ∫ ddL p(dL | pix, skymap) * host(draw; dL),

    where:
      host(draw; dL) = rho_host(z(dL)) * (dV/dz/dOmega) * (dz/ddL) * (1/π(dL)),

    and the 1/π(dL) factor (implemented via `gw_distance_prior`) corrects for the implicit distance
    prior in the public 3D sky maps (same correction used in the in-catalog term).

    Returns:
      (logL_missing_mu, logL_missing_gr), each with shape (n_draws,).
    """
    prob_pix, distmu, distsigma, distnorm = select_missing_pixels(
        sky,
        p_credible=p_credible,
        nside_coarse=nside_coarse,
        hpix_coarse=hpix_coarse,
    )
    return compute_missing_host_logL_draws_from_pixels(
        prob_pix=prob_pix,
        distmu=distmu,
        distsigma=distsigma,
        distnorm=distnorm,
        pre=pre,
        dL_grid=dL_grid,
        gw_distance_prior=gw_distance_prior,
        pixel_chunk_size=pixel_chunk_size,
    )


def compute_missing_host_logL_draws_from_pixels(
    *,
    prob_pix: np.ndarray,
    distmu: np.ndarray,
    distsigma: np.ndarray,
    distnorm: np.ndarray,
    pre: MissingHostPriorPrecompute,
    dL_grid: np.ndarray,
    gw_distance_prior: GWDistancePrior,
    pixel_chunk_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as compute_missing_host_logL_draws, but uses preselected sky pixel arrays."""
    prob_pix = np.asarray(prob_pix, dtype=float)
    distmu = np.asarray(distmu, dtype=float)
    distsigma = np.asarray(distsigma, dtype=float)
    distnorm = np.asarray(distnorm, dtype=float)
    if prob_pix.ndim != 1 or distmu.ndim != 1 or distsigma.ndim != 1 or distnorm.ndim != 1:
        raise ValueError("Pixel arrays must be 1D.")
    if not (prob_pix.shape == distmu.shape == distsigma.shape == distnorm.shape):
        raise ValueError("Pixel arrays must have matching shapes.")
    if prob_pix.size == 0:
        raise ValueError("No valid sky pixels selected for missing-host term.")

    dL_grid = np.asarray(dL_grid, dtype=float)
    if dL_grid.ndim != 1 or dL_grid.size < 2:
        raise ValueError("dL_grid must be a 1D array with >=2 points.")
    if np.any(np.diff(dL_grid) <= 0.0):
        raise ValueError("dL_grid must be strictly increasing.")

    # Draw-wise host factors on this dL grid.
    host_mu = _host_prior_matrix_from_precompute(pre, dL_grid=dL_grid, model="mu", gw_distance_prior=gw_distance_prior)
    host_gr = _host_prior_matrix_from_precompute(pre, dL_grid=dL_grid, model="gr", gw_distance_prior=gw_distance_prior)

    # Trapezoidal weights for the ddL integral.
    w_dl = _trapz_weights(dL_grid)
    host_mu_w = host_mu * w_dl.reshape((1, -1))
    host_gr_w = host_gr * w_dl.reshape((1, -1))

    n_draws = int(host_mu.shape[0])
    L_mu = np.zeros((n_draws,), dtype=float)
    L_gr = np.zeros((n_draws,), dtype=float)

    n_pix = int(prob_pix.size)
    chunk = int(pixel_chunk_size)
    if chunk <= 0:
        raise ValueError("pixel_chunk_size must be positive.")

    for a in range(0, n_pix, chunk):
        b = min(n_pix, a + chunk)
        p = prob_pix[a:b]
        mu = distmu[a:b]
        sig = distsigma[a:b]
        norm = distnorm[a:b]

        pdf = ligo_distance.conditional_pdf(dL_grid.reshape((1, -1)), mu.reshape((-1, 1)), sig.reshape((-1, 1)), norm.reshape((-1, 1)))
        pdf = np.clip(np.asarray(pdf, dtype=float), 1e-300, np.inf)

        # (n_draws, chunk) = (n_draws, n_dL) @ (n_dL, chunk)
        # Do mu/gr sequentially to keep peak memory lower.
        proj_mu = host_mu_w @ pdf.T
        L_mu += proj_mu @ p
        proj_gr = host_gr_w @ pdf.T
        L_gr += proj_gr @ p

    L_mu = np.clip(L_mu, 1e-300, np.inf)
    L_gr = np.clip(L_gr, 1e-300, np.inf)
    return np.log(L_mu), np.log(L_gr)


def compute_missing_host_logL_draws_from_histogram(
    *,
    prob_pix: np.ndarray,
    pdf_bins: np.ndarray,
    dL_edges: np.ndarray,
    pre: MissingHostPriorPrecompute,
    gw_distance_prior: GWDistancePrior,
    distance_mode: Literal["full", "spectral_only", "prior_only", "sky_only"] = "full",
    pixel_chunk_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Missing-host logL vectors using a binned per-pixel distance posterior.

    This is the PE-sample analogue of `compute_missing_host_logL_draws_from_pixels`, where the
    distance posterior is provided as per-pixel histograms:

      p(dL | pix, data) ≈ pdf_bins[pix, k]  for dL in bin k (density per Mpc).

    Inputs:
      prob_pix: (n_pix,) pixel probability masses (sum to ~1 on the selected CR)
      pdf_bins: (n_pix, n_bins) conditional distance densities per pixel
      dL_edges: (n_bins+1,) bin edges (Mpc), strictly increasing

    distance_mode:
      - full: use per-pixel conditional distance densities p(dL | pix, data).
      - spectral_only: replace p(dL | pix, data) with the sky-marginal p(dL | data) (independent of pix),
        while keeping prob_pix sky weights. This removes sky–distance correlation as a control.
      - prior_only: replace p(dL | pix, data) with the assumed GW distance prior π(dL), normalized over
        the histogram support. This is an audit/null option; it is not intended for science use.
      - sky_only: ignore distance entirely and return a sky-only constant based on prob_pix. This is a
        strict distance-destruction null (logL_mu == logL_gr by construction), intended only for forensic
        tests (not for science use).
    """
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
    if pdf_bins.shape[1] != dL_edges.size - 1:
        raise ValueError("pdf_bins.shape[1] must equal len(dL_edges)-1.")
    if distance_mode == "full":
        if pdf_bins.shape[0] != prob_pix.size:
            raise ValueError("pdf_bins.shape[0] must match prob_pix.size in full mode.")
    else:
        # For spectral/prior/sky controls we also allow a compact single-row
        # representation for sky-independent distance PDFs.
        if int(pdf_bins.shape[0]) not in (1, int(prob_pix.size)):
            raise ValueError(
                "pdf_bins.shape[0] must be 1 or prob_pix.size in non-full modes."
            )

    if prob_pix.size == 0:
        raise ValueError("No valid sky pixels provided for missing-host term.")

    if distance_mode == "sky_only":
        n_draws = int(pre.dL_em.shape[0])
        p_sum = float(np.sum(prob_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum in sky_only missing-host term.")
        logL0 = float(np.log(p_sum))
        out = np.full((n_draws,), logL0, dtype=float)
        return out, out.copy()

    # Midpoints + widths for ddL integral.
    widths = np.diff(dL_edges)
    dL_mid = 0.5 * (dL_edges[:-1] + dL_edges[1:])

    # Draw-wise host factors on this dL grid (includes 1/pi(dL) correction).
    host_mu = _host_prior_matrix_from_precompute(pre, dL_grid=dL_mid, model="mu", gw_distance_prior=gw_distance_prior)
    host_gr = _host_prior_matrix_from_precompute(pre, dL_grid=dL_mid, model="gr", gw_distance_prior=gw_distance_prior)

    host_mu_w = host_mu * widths.reshape((1, -1))
    host_gr_w = host_gr * widths.reshape((1, -1))

    n_draws = int(host_mu_w.shape[0])
    L_mu = np.zeros((n_draws,), dtype=float)
    L_gr = np.zeros((n_draws,), dtype=float)

    if distance_mode == "spectral_only":
        # Sky-marginal distance density on the same bins:
        #   p(dL | data) ∝ Σ_pix prob_pix(pix) * p(dL | pix, data),
        # then renormalize to integrate to 1 over the selected pixels' dL support.
        p_sum = float(np.sum(prob_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum while building spectral_only missing-host term.")

        if int(pdf_bins.shape[0]) == int(prob_pix.size):
            pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum  # (n_bins,)
        elif int(pdf_bins.shape[0]) == 1:
            # Compact sky-independent representation.
            pdf_1d = np.asarray(pdf_bins[0], dtype=float)
        else:
            raise ValueError(
                "Incompatible pdf_bins shape for spectral_only missing-host term: "
                f"{pdf_bins.shape} vs prob_pix.size={prob_pix.size}"
            )
        pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
        norm = float(np.sum(pdf_1d * widths))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("Invalid sky-marginal distance density normalization in spectral_only missing-host term.")
        pdf_1d = pdf_1d / norm

        # L(draw) = ∫ host(draw; dL) * p(dL|data) ddL  (since Σ prob_pix = 1 after conditioning)
        L_mu = np.clip(host_mu_w @ pdf_1d, 1e-300, np.inf)
        L_gr = np.clip(host_gr_w @ pdf_1d, 1e-300, np.inf)
        return np.log(L_mu), np.log(L_gr)
    if distance_mode == "prior_only":
        log_pi = np.asarray(gw_distance_prior.log_pi_dL(np.clip(dL_mid, 1e-6, np.inf)), dtype=float)
        t = log_pi + np.log(np.clip(widths, 1e-300, np.inf))
        m = np.isfinite(t)
        if not np.any(m):
            raise ValueError("Invalid π(dL) normalization in prior_only missing-host term.")
        t0 = float(np.max(t[m]))
        logZ = float(t0 + np.log(float(np.sum(np.exp(t[m] - t0)))))
        pdf_1d = np.zeros_like(log_pi, dtype=float)
        pdf_1d[m] = np.exp(log_pi[m] - logZ)
        pdf_1d = np.clip(pdf_1d, 0.0, np.inf)
        L_mu = np.clip(host_mu_w @ pdf_1d, 1e-300, np.inf)
        L_gr = np.clip(host_gr_w @ pdf_1d, 1e-300, np.inf)
        return np.log(L_mu), np.log(L_gr)
    if distance_mode != "full":
        raise ValueError("distance_mode must be 'full', 'spectral_only', 'prior_only', or 'sky_only'.")

    n_pix = int(prob_pix.size)
    chunk = int(pixel_chunk_size)
    if chunk <= 0:
        raise ValueError("pixel_chunk_size must be positive.")

    for a in range(0, n_pix, chunk):
        b = min(n_pix, a + chunk)
        p = prob_pix[a:b]
        pdf = pdf_bins[a:b, :]  # (chunk, n_bins)
        pdf = np.clip(np.asarray(pdf, dtype=float), 0.0, np.inf)

        # (n_draws, chunk) = (n_draws, n_bins) @ (n_bins, chunk)
        proj_mu = host_mu_w @ pdf.T
        L_mu += proj_mu @ p
        proj_gr = host_gr_w @ pdf.T
        L_gr += proj_gr @ p

    L_mu = np.clip(L_mu, 1e-300, np.inf)
    L_gr = np.clip(L_gr, 1e-300, np.inf)
    return np.log(L_mu), np.log(L_gr)


def compute_coarse_credible_pixels_for_sky(
    sky: SkyMap3D,
    *,
    nside_coarse: int,
    p_credible: float,
) -> np.ndarray:
    """Convenience wrapper to match selection+missing-term coarse pixels."""
    hpix_sel, _area_deg2 = credible_region_pixels(sky, nside_out=int(nside_coarse), p_credible=float(p_credible))
    return hpix_sel
