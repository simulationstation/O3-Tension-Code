from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Literal

import healpy as hp
import numpy as np
from scipy.special import logsumexp

from ligo.skymap import distance as ligo_distance
from ligo.skymap.io import fits as ligo_fits

from .gw_distance_priors import GWDistancePrior
from .sirens import MuForwardPosterior, predict_dL_em, predict_dL_gw, predict_r_gw_em


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


@dataclass(frozen=True)
class SkyMap3D:
    path: str
    nside: int
    nest: bool
    prob: np.ndarray  # (npix,) probability per pixel (sum ~ 1)
    distmu: np.ndarray  # (npix,) Mpc
    distsigma: np.ndarray  # (npix,) Mpc
    distnorm: np.ndarray  # (npix,) Mpc^-2

    @property
    def npix(self) -> int:
        return int(self.prob.size)

    @property
    def pixarea_sr(self) -> float:
        return 4.0 * np.pi / float(self.npix)

    @property
    def pixarea_deg2(self) -> float:
        return self.pixarea_sr * (180.0 / np.pi) ** 2


def read_skymap_3d(path: str | Path, *, nest: bool = True) -> SkyMap3D:
    path = Path(path)
    m, meta = ligo_fits.read_sky_map(str(path), nest=nest, distances=True, moc=False)

    # read_sky_map(distances=True) may return either:
    # - a structured array with PROB, DISTMU, DISTSIGMA, DISTNORM fields, or
    # - a tuple: (prob, distmu, distsigma, distnorm)
    if isinstance(m, tuple):
        if len(m) != 4:
            raise ValueError(f"{path}: expected 4-tuple sky map (prob, distmu, distsigma, distnorm); got len={len(m)}")
        prob, distmu, distsigma, distnorm = (np.asarray(a, dtype=float) for a in m)
    else:
        if getattr(m, "dtype", None) is None or m.dtype.fields is None:
            raise ValueError(f"{path}: expected distance layers; got dtype={getattr(m, 'dtype', None)}")
        fields = {k.upper(): k for k in m.dtype.fields.keys()}
        need = ["PROB", "DISTMU", "DISTSIGMA", "DISTNORM"]
        for k in need:
            if k not in fields:
                raise ValueError(f"{path}: missing '{k}' layer; found fields={list(m.dtype.fields.keys())}")
        prob = np.asarray(m[fields["PROB"]], dtype=float)
        distmu = np.asarray(m[fields["DISTMU"]], dtype=float)
        distsigma = np.asarray(m[fields["DISTSIGMA"]], dtype=float)
        distnorm = np.asarray(m[fields["DISTNORM"]], dtype=float)

    nside_meta = meta.get("nside")
    if nside_meta is None:
        # Infer from npix; some releases omit nside in the header.
        nside_meta = hp.npix2nside(prob.size)
    nside_meta = int(nside_meta)

    # meta['nest'] indicates the ordering in the FITS file. We asked read_sky_map to return in a
    # particular ordering via the nest= argument; therefore we trust that here.
    return SkyMap3D(
        path=str(path),
        nside=nside_meta,
        nest=bool(nest),
        prob=prob,
        distmu=distmu,
        distsigma=distsigma,
        distnorm=distnorm,
    )


@dataclass(frozen=True)
class GalaxyIndex:
    """Galaxy catalog stored as HEALPix buckets with contiguous per-pixel ranges."""

    nside: int
    nest: bool
    hpix_offsets: np.ndarray  # (npix+1,) int64 offsets into the flat arrays
    ra_deg: np.ndarray  # (N,) float32
    dec_deg: np.ndarray  # (N,) float32
    z: np.ndarray  # (N,) float32
    w: np.ndarray  # (N,) float32

    @property
    def npix(self) -> int:
        return hp.nside2npix(int(self.nside))

    @property
    def n_gal(self) -> int:
        return int(self.z.size)


def load_gladeplus_index(index_dir: str | Path) -> GalaxyIndex:
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "meta.json").read_text())
    nside = int(meta["nside"])
    nest = bool(meta["nest"])
    hpix_offsets = np.load(index_dir / "hpix_offsets.npy")
    ra_deg = np.load(index_dir / "ra_deg.npy", mmap_mode="r")
    dec_deg = np.load(index_dir / "dec_deg.npy", mmap_mode="r")
    z = np.load(index_dir / "z.npy", mmap_mode="r")
    w = np.load(index_dir / "w.npy", mmap_mode="r")
    return GalaxyIndex(
        nside=nside,
        nest=nest,
        hpix_offsets=np.asarray(hpix_offsets, dtype=np.int64),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        z=z,
        w=w,
    )


def _resample_prob_nest(prob_nest: np.ndarray, *, nside_in: int, nside_out: int) -> np.ndarray:
    """Resample a NESTED probability-per-pixel map.

    - If nside_out < nside_in: downgrade by summing child pixels.
    - If nside_out > nside_in: upsample by splitting parent pixel prob equally among children.

    This keeps total probability conserved (sum unchanged).
    """
    prob_nest = np.asarray(prob_nest, dtype=float)
    if nside_in == nside_out:
        return prob_nest
    if nside_in % nside_out == 0:
        # Downsample.
        ratio = nside_in // nside_out
        if ratio & (ratio - 1) != 0:
            raise ValueError("nside ratio must be a power of two for nested resampling.")
        group = ratio * ratio
        npix_out = hp.nside2npix(nside_out)
        if prob_nest.size != npix_out * group:
            raise ValueError("Unexpected prob map size for the given nsides.")
        return prob_nest.reshape(npix_out, group).sum(axis=1)
    if nside_out % nside_in == 0:
        # Upsample.
        ratio = nside_out // nside_in
        if ratio & (ratio - 1) != 0:
            raise ValueError("nside ratio must be a power of two for nested resampling.")
        group = ratio * ratio
        # Split each parent pixel prob evenly among children.
        return np.repeat(prob_nest / float(group), group)
    raise ValueError("nside_in and nside_out must be integer multiples of each other for nested resampling.")


def credible_region_pixels(
    sky: SkyMap3D,
    *,
    nside_out: int,
    p_credible: float,
) -> tuple[np.ndarray, float]:
    """Return coarse pixels covering p_credible sky probability and the resulting sky area (deg^2)."""
    if not (0.0 < p_credible <= 1.0):
        raise ValueError("p_credible must be in (0,1].")
    if not sky.nest:
        raise ValueError("credible_region_pixels currently assumes sky.prob is NESTED ordered.")

    prob_coarse = _resample_prob_nest(sky.prob, nside_in=sky.nside, nside_out=int(nside_out))
    idx = np.argsort(prob_coarse)[::-1]
    csum = np.cumsum(prob_coarse[idx])
    k = int(np.searchsorted(csum, p_credible, side="left")) + 1
    sel = np.sort(idx[:k].astype(np.int64, copy=False))

    pixarea_deg2 = (4.0 * np.pi / float(hp.nside2npix(int(nside_out)))) * (180.0 / np.pi) ** 2
    area_deg2 = float(sel.size) * float(pixarea_deg2)
    return sel, area_deg2


def gather_galaxies_from_pixels(
    cat: GalaxyIndex,
    hpix: np.ndarray,
    *,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather a subset of galaxies (ra, dec, z, w) for a set of HEALPix pixels."""
    hpix = np.asarray(hpix, dtype=np.int64)
    hpix = np.unique(hpix)
    if hpix.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Total length for preallocation.
    offs = cat.hpix_offsets
    n = int(np.sum(offs[hpix + 1] - offs[hpix]))
    ra = np.empty(n, dtype=np.float32)
    dec = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    w = np.empty(n, dtype=np.float32)

    pos = 0
    for p in hpix.tolist():
        a = int(offs[p])
        b = int(offs[p + 1])
        if b <= a:
            continue
        m = slice(pos, pos + (b - a))
        ra[m] = cat.ra_deg[a:b]
        dec[m] = cat.dec_deg[a:b]
        z[m] = cat.z[a:b]
        w[m] = cat.w[a:b]
        pos += b - a

    ra = ra[:pos]
    dec = dec[:pos]
    z = z[:pos]
    w = w[:pos]

    if z_max is not None:
        m = np.isfinite(z) & (z > 0.0) & (z <= float(z_max))
        ra, dec, z, w = ra[m], dec[m], z[m], w[m]
    return ra, dec, z, w


@dataclass(frozen=True)
class DarkSirenEventScore:
    event: str
    skymap_path: str
    n_gal: int
    sky_area_deg2: float
    lpd_mu: float
    lpd_gr: float
    delta_lpd: float
    # Optional decomposition (when selection normalization is applied at scoring time).
    lpd_mu_data: float = float("nan")
    lpd_gr_data: float = float("nan")
    delta_lpd_data: float = float("nan")
    lpd_mu_sel: float = float("nan")
    lpd_gr_sel: float = float("nan")
    delta_lpd_sel: float = float("nan")


@dataclass(frozen=True)
class DarkSirenEventLogLDraws:
    """Per-draw log-likelihood vectors for a single event.

    This enables applying global/per-run corrections (e.g. selection normalization) at the draw level,
    and computing combined multi-event LPD in the mathematically correct way:
      LPD_total = logmeanexp(sum_events logL_draw).
    """

    event: str
    skymap_path: str
    n_gal: int
    sky_area_deg2: float
    logL_mu: np.ndarray  # (n_draws,)
    logL_gr: np.ndarray  # (n_draws,)


def compute_dark_siren_logL_draws(
    *,
    event: str,
    sky: SkyMap3D,
    sky_area_deg2: float,
    post: MuForwardPosterior,
    z: np.ndarray,
    w: np.ndarray,
    ra_deg: np.ndarray | None = None,
    dec_deg: np.ndarray | None = None,
    ipix: np.ndarray | None = None,
    convention: Literal["A", "B"] = "A",
    max_draws: int | None = None,
    gw_distance_prior: GWDistancePrior | None = None,
    gal_chunk_size: int = 50_000,
) -> DarkSirenEventLogLDraws:
    """Compute per-draw log-likelihood vectors for one event.

    Notes:
    - Uses the public sky map posterior as a proxy likelihood via a distance-prior correction (divide by π(dL)).
    - For speed, callers can pass precomputed ipix (galaxy -> sky pixel) to avoid repeated ang2pix().
    """
    if not sky.nest:
        raise ValueError("compute_dark_siren_logL_draws currently assumes sky arrays are NESTED ordered.")

    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape:
        raise ValueError("z and w must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided for this event.")

    if ipix is None:
        if ra_deg is None or dec_deg is None:
            raise ValueError("Either (ra_deg, dec_deg) or ipix must be provided.")
        ra_deg = np.asarray(ra_deg, dtype=float)
        dec_deg = np.asarray(dec_deg, dtype=float)
        if ra_deg.shape != z.shape or dec_deg.shape != z.shape:
            raise ValueError("ra_deg/dec_deg must match z shape.")
        theta = np.deg2rad(90.0 - dec_deg)
        phi = np.deg2rad(ra_deg)
        ipix = hp.ang2pix(sky.nside, theta, phi, nest=True).astype(np.int64, copy=False)
    else:
        ipix = np.asarray(ipix, dtype=np.int64)
        if ipix.shape != z.shape:
            raise ValueError("ipix must match z shape.")
        if np.any(ipix < 0) or np.any(ipix >= int(sky.npix)):
            raise ValueError("ipix contains out-of-range pixel indices for this sky map.")

    # Downsample posterior draws if requested.
    if max_draws is not None:
        keep = int(max_draws)
        if keep <= 0:
            raise ValueError("max_draws must be positive.")
        post = MuForwardPosterior(
            x_grid=post.x_grid,
            logmu_x_samples=post.logmu_x_samples[:keep],
            z_grid=post.z_grid,
            H_samples=post.H_samples[:keep],
            H0=post.H0[:keep],
            omega_m0=post.omega_m0[:keep],
            omega_k0=post.omega_k0[:keep],
            sigma8_0=post.sigma8_0[:keep] if post.sigma8_0 is not None else None,
        )

    # Map sky posterior layers onto each galaxy pixel.
    prob = sky.prob[ipix]
    distmu = sky.distmu[ipix]
    distsigma = sky.distsigma[ipix]
    distnorm = sky.distnorm[ipix]

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
        raise ValueError("All candidate galaxies map to invalid sky-map pixels (or have invalid z/w).")

    z = z[good]
    w = w[good]
    prob = prob[good]
    distmu = distmu[good]
    distsigma = distsigma[good]
    distnorm = distnorm[good]

    prior = gw_distance_prior or GWDistancePrior()
    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    # Performance: as in the PE-hist implementation, chunking relies on repeated redshifts within
    # a chunk so that distance prediction needs to be evaluated on far fewer points than galaxies.
    # Sorting by redshift before chunking improves redshift locality and can reduce runtime by
    # large factors for multi-million-galaxy events without changing the likelihood math.
    if z.size > chunk and not np.all(z[1:] >= z[:-1]):
        order = np.argsort(z, kind="mergesort")
        z = z[order]
        w = w[order]
        prob = prob[order]
        distmu = distmu[order]
        distsigma = distsigma[order]
        distnorm = distnorm[order]

    # Chunk over galaxies to keep peak memory bounded (critical for large sky areas / large max_gal).
    n_draws = int(post.H_samples.shape[0])
    logL_mu = np.full((n_draws,), -np.inf, dtype=float)
    logL_gr = np.full((n_draws,), -np.inf, dtype=float)

    debug = str(os.environ.get("EH_DARK_SIREN_PROGRESS", "")).strip().lower() not in ("", "0", "false", "no")
    try:
        min_ngal = int(os.environ.get("EH_DARK_SIREN_PROGRESS_MIN_NGAL", "1000000"))
    except Exception:
        min_ngal = 1_000_000
    try:
        every_chunks = int(os.environ.get("EH_DARK_SIREN_PROGRESS_EVERY_CHUNKS", "10"))
    except Exception:
        every_chunks = 10
    do_progress = bool(debug) and (int(z.size) >= int(min_ngal)) and int(every_chunks) > 0
    n_chunks = int((int(z.size) + int(chunk) - 1) // int(chunk)) if do_progress else 0
    t0 = time.perf_counter()
    t_last = t0

    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        if do_progress:
            i = int(a // chunk)
        z_c = np.asarray(z[a:b], dtype=float)
        w_c = np.asarray(w[a:b], dtype=float)
        prob_c = np.asarray(prob[a:b], dtype=float)
        distmu_c = np.asarray(distmu[a:b], dtype=float)
        distsigma_c = np.asarray(distsigma[a:b], dtype=float)
        distnorm_c = np.asarray(distnorm[a:b], dtype=float)

        # predict_r_gw_em requires strictly increasing z, so evaluate on unique z values and map back.
        # If the full `z` array was sorted above, each chunk is sorted too; exploit that to compute
        # a fast unique+inverse map without re-sorting.
        if z_c.size > 1 and np.all(z_c[1:] >= z_c[:-1]):
            is_new = np.empty(z_c.shape, dtype=bool)
            is_new[0] = True
            is_new[1:] = z_c[1:] != z_c[:-1]
            z_u = z_c[is_new]
            inv = np.cumsum(is_new, dtype=np.int64) - 1
        else:
            z_u, inv = np.unique(z_c, return_inverse=True)

        if do_progress and (i == 0 or i == n_chunks - 1 or (i % every_chunks == 0)):
            now = time.perf_counter()
            dt = now - t_last
            elapsed = now - t0
            t_last = now
            pct = 100.0 * (float(b) / float(z.size))
            print(
                f"[dark_sirens] {event}: galaxies {b:,}/{z.size:,} ({pct:.1f}%) "
                f"chunk {i + 1}/{n_chunks} uniq_z={int(z_u.size):,} dt={dt:.1f}s elapsed={elapsed:.1f}s",
                flush=True,
            )
        dL_em_u = predict_dL_em(post, z_eval=z_u)  # (n_draws, n_zuniq)
        _, R_u = predict_r_gw_em(post, z_eval=z_u, convention=convention, allow_extrapolation=False)
        dL_gw_u = dL_em_u * np.asarray(R_u, dtype=float)

        dL_em = dL_em_u[:, inv]  # (n_draws, n_chunk)
        dL_gw = dL_gw_u[:, inv]

        # Evaluate the distance posterior ansatz per pixel, then divide out the assumed distance prior π(dL).
        w_c = np.clip(w_c.astype(float), 1e-30, np.inf)
        logw = np.log(w_c)[None, :]
        logprob = np.log(np.clip(prob_c.astype(float), 1e-300, np.inf))[None, :]

        def _chunk_logL(dL: np.ndarray) -> np.ndarray:
            dL = np.asarray(dL, dtype=float)
            pdf = ligo_distance.conditional_pdf(dL, distmu_c[None, :], distsigma_c[None, :], distnorm_c[None, :])
            pdf = np.clip(pdf, 1e-300, np.inf)
            logpdf = np.log(pdf)
            logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
            logterm = logw + logprob + logpdf - logprior
            return logsumexp(logterm, axis=1)

        logL_mu = np.logaddexp(logL_mu, _chunk_logL(dL_gw))
        logL_gr = np.logaddexp(logL_gr, _chunk_logL(dL_em))

    return DarkSirenEventLogLDraws(
        event=str(event),
        skymap_path=str(sky.path),
        n_gal=int(z.size),
        sky_area_deg2=float(sky_area_deg2),
        logL_mu=np.asarray(logL_mu, dtype=float),
        logL_gr=np.asarray(logL_gr, dtype=float),
    )


def score_dark_siren_event(
    *,
    event: str,
    sky: SkyMap3D,
    sky_area_deg2: float,
    post: MuForwardPosterior,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    ipix: np.ndarray | None = None,
    convention: Literal["A", "B"] = "A",
    max_draws: int | None = None,
) -> DarkSirenEventScore:
    """Score one dark siren (GW 3D skymap) against the mu(A) propagation model vs GR baseline.

    This is a *development-stage* score:
    - Uses the public sky map posterior as a proxy likelihood via a distance-prior correction (divide by π(dL)).
    - Does not include selection normalization alpha(model) yet.
    """
    draws = compute_dark_siren_logL_draws(
        event=event,
        sky=sky,
        sky_area_deg2=sky_area_deg2,
        post=post,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ipix=ipix,
        z=z,
        w=w,
        convention=convention,
        max_draws=max_draws,
    )
    lpd_mu = float(_logmeanexp(draws.logL_mu))
    lpd_gr = float(_logmeanexp(draws.logL_gr))

    # sky area is an input-level selection property; the caller should compute it.
    return DarkSirenEventScore(
        event=str(event),
        skymap_path=str(sky.path),
        n_gal=int(draws.n_gal),
        sky_area_deg2=float(sky_area_deg2),
        lpd_mu=lpd_mu,
        lpd_gr=lpd_gr,
        delta_lpd=float(lpd_mu - lpd_gr),
        lpd_mu_data=float(lpd_mu),
        lpd_gr_data=float(lpd_gr),
        delta_lpd_data=float(lpd_mu - lpd_gr),
        lpd_mu_sel=0.0,
        lpd_gr_sel=0.0,
        delta_lpd_sel=0.0,
    )


@dataclass(frozen=True)
class DarkSirenTestSummary:
    run: str
    convention: str
    n_events: int
    n_draws: int
    lpd_mu_total: float
    lpd_gr_total: float
    delta_lpd_total: float


def summarize_scores(*, run_label: str, convention: str, post: MuForwardPosterior, scores: list[DarkSirenEventScore]) -> DarkSirenTestSummary:
    lpd_mu_total = float(sum(s.lpd_mu for s in scores))
    lpd_gr_total = float(sum(s.lpd_gr for s in scores))
    return DarkSirenTestSummary(
        run=str(run_label),
        convention=str(convention),
        n_events=int(len(scores)),
        n_draws=int(post.H_samples.shape[0]),
        lpd_mu_total=lpd_mu_total,
        lpd_gr_total=lpd_gr_total,
        delta_lpd_total=float(lpd_mu_total - lpd_gr_total),
    )


def to_json(obj) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), indent=2)
    raise TypeError("Object is not a dataclass.")
