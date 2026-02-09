from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.stats

from .sirens import MuForwardPosterior, predict_dL_em, predict_dL_gw


PriorKind = Literal["none", "dL2"]
DistanceKind = Literal["normal", "two_piece_normal", "samples"]


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


def _logpdf_normal(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sig <= 0:
        raise ValueError("sig must be positive.")
    return -np.log(sig) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((x - mu) / sig) ** 2


def _logpdf_two_piece_normal(x: np.ndarray, mu: float, sig_lo: float, sig_hi: float) -> np.ndarray:
    """Two-piece (split) normal log-PDF evaluated at x.

    Normalization uses: f(x) = 2/(sig_lo+sig_hi) * phi((x-mu)/sig)/sig, sig=lo/hi by side.
    """
    x = np.asarray(x, dtype=float)
    if sig_lo <= 0 or sig_hi <= 0:
        raise ValueError("sig_lo/sig_hi must be positive.")
    sig = np.where(x <= mu, sig_lo, sig_hi)
    return (
        np.log(2.0)
        - np.log(sig_lo + sig_hi)
        - np.log(sig)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * ((x - mu) / sig) ** 2
    )


@dataclass
class SirenEvent:
    name: str
    z: float
    dist: DistanceKind
    z_sigma: float | None = None

    # Normal
    dL_Mpc: float | None = None
    dL_sigma_Mpc: float | None = None

    # Two-piece normal
    dL_sigma_lo_Mpc: float | None = None
    dL_sigma_hi_Mpc: float | None = None

    # Posterior samples (preferred). Units: Mpc.
    dL_samples_Mpc: np.ndarray | None = None
    # Approximate prior correction to turn posterior density into a likelihood proxy:
    #   like(dL) ∝ post(dL) / prior(dL).
    # This is an approximation; if a publication-grade per-event prior is available, use it.
    prior: PriorKind = "none"
    kde_bw: str | float = "scott"

    _kde: scipy.stats.gaussian_kde | None = None

    def _ensure_kde(self) -> scipy.stats.gaussian_kde:
        if self._kde is not None:
            return self._kde
        if self.dL_samples_Mpc is None or self.dL_samples_Mpc.size < 20:
            raise ValueError(f"Event '{self.name}': need >=20 distance samples for KDE.")
        samp = np.asarray(self.dL_samples_Mpc, dtype=float)
        samp = samp[np.isfinite(samp)]
        if samp.size < 20:
            raise ValueError(f"Event '{self.name}': too few finite distance samples for KDE.")
        self._kde = scipy.stats.gaussian_kde(samp, bw_method=self.kde_bw)
        return self._kde

    def loglike_proxy(self, dL_Mpc: np.ndarray) -> np.ndarray:
        """Approximate log-likelihood proxy for the GW measurement, evaluated at dL.

        - If dist is parametric, we treat it as a measurement PDF in dL.
        - If dist is 'samples', we KDE the GW posterior and optionally divide by an approximate
          distance prior (e.g. dL^2 for a volume prior).

        This yields a *proxy* for log p(data | dL), up to an additive constant. The constant cancels
        in Delta_LPD comparisons between models using the same events.
        """
        dL_Mpc = np.asarray(dL_Mpc, dtype=float)
        if self.dist == "normal":
            if self.dL_Mpc is None or self.dL_sigma_Mpc is None:
                raise ValueError(f"Event '{self.name}': missing dL_Mpc/dL_sigma_Mpc.")
            return _logpdf_normal(dL_Mpc, mu=float(self.dL_Mpc), sig=float(self.dL_sigma_Mpc))
        if self.dist == "two_piece_normal":
            if self.dL_Mpc is None or self.dL_sigma_lo_Mpc is None or self.dL_sigma_hi_Mpc is None:
                raise ValueError(f"Event '{self.name}': missing two-piece normal parameters.")
            return _logpdf_two_piece_normal(
                dL_Mpc,
                mu=float(self.dL_Mpc),
                sig_lo=float(self.dL_sigma_lo_Mpc),
                sig_hi=float(self.dL_sigma_hi_Mpc),
            )
        if self.dist == "samples":
            kde = self._ensure_kde()
            pdf = kde(dL_Mpc)
            pdf = np.clip(pdf, 1e-300, np.inf)
            logp = np.log(pdf)
            if self.prior == "dL2":
                # prior(dL) ∝ dL^2
                d = np.clip(dL_Mpc, 1e-6, np.inf)
                logp = logp - 2.0 * np.log(d)
            elif self.prior != "none":
                raise ValueError(f"Unknown prior='{self.prior}' for event '{self.name}'.")
            return logp
        raise ValueError(f"Unsupported dist='{self.dist}' for event '{self.name}'.")


def load_siren_events(path: str | Path) -> list[SirenEvent]:
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError("Unsupported siren JSON format. Expected {'events': [...]} or a list.")
    if not isinstance(events, list):
        raise ValueError("Expected 'events' to be a list.")

    out: list[SirenEvent] = []
    for ev in events:
        if not isinstance(ev, dict):
            raise ValueError("Each event must be a JSON object.")
        dist = str(ev.get("dist", "two_piece_normal"))
        if dist not in ("normal", "two_piece_normal", "samples"):
            raise ValueError(f"Unsupported dist='{dist}' in siren JSON.")

        samples = None
        if dist == "samples":
            if "dL_samples_file" in ev:
                p = Path(str(ev["dL_samples_file"]))
                if not p.is_absolute():
                    p = path.parent / p
                if p.suffix == ".npy":
                    samples = np.load(p)
                else:
                    with np.load(p) as d:
                        # accept common key names
                        for k in ("dL_samples_Mpc", "dL_samples", "dL"):
                            if k in d.files:
                                samples = d[k]
                                break
                        if samples is None:
                            raise ValueError(f"{p}: missing dL samples array key.")
            elif "dL_samples_Mpc" in ev:
                samples = np.asarray(ev["dL_samples_Mpc"], dtype=float)
            else:
                raise ValueError("dist='samples' requires dL_samples_Mpc or dL_samples_file.")
            samples = np.asarray(samples, dtype=float)

        out.append(
            SirenEvent(
                name=str(ev.get("name", "event")),
                z=float(ev["z"]),
                z_sigma=float(ev["z_sigma"]) if "z_sigma" in ev else None,
                dist=dist,  # type: ignore[arg-type]
                dL_Mpc=float(ev["dL_Mpc"]) if "dL_Mpc" in ev else None,
                dL_sigma_Mpc=float(ev["dL_sigma_Mpc"]) if "dL_sigma_Mpc" in ev else None,
                dL_sigma_lo_Mpc=float(ev["dL_sigma_lo_Mpc"]) if "dL_sigma_lo_Mpc" in ev else None,
                dL_sigma_hi_Mpc=float(ev["dL_sigma_hi_Mpc"]) if "dL_sigma_hi_Mpc" in ev else None,
                dL_samples_Mpc=samples,
                prior=str(ev.get("prior", "none")),  # type: ignore[arg-type]
                kde_bw=ev.get("kde_bw", "scott"),
            )
        )
    return out


@dataclass(frozen=True)
class SirenEventScore:
    run: str
    event: str
    z: float
    lpd_mu: float
    lpd_gr: float
    delta_lpd: float


@dataclass(frozen=True)
class SirenTestSummary:
    run: str
    convention: str
    n_events: int
    n_draws: int
    lpd_mu_total: float
    lpd_gr_total: float
    delta_lpd_total: float


def score_siren_events(
    *,
    run_label: str,
    post: MuForwardPosterior,
    events: list[SirenEvent],
    convention: Literal["A", "B"] = "A",
    allow_extrapolation: bool = False,
    max_draws: int | None = None,
) -> tuple[SirenTestSummary, list[SirenEventScore]]:
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

    lpd_mu_total = 0.0
    lpd_gr_total = 0.0
    scores: list[SirenEventScore] = []
    for ev in events:
        z0 = float(ev.z)
        z_sigma = float(ev.z_sigma) if ev.z_sigma is not None else 0.0

        # If z uncertainty is provided, marginalize by sampling z.
        if z_sigma > 0:
            # Keep modest: per-event marginalization is cheap and stable at O(50-200) samples.
            n_z = 96
            # Use deterministic equal-probability quantiles (more stable than RNG draws).
            qs = (np.arange(n_z, dtype=float) + 0.5) / float(n_z)
            z_samp = z0 + z_sigma * scipy.stats.norm.ppf(qs)
            # Avoid non-physical/negative z draws.
            z_samp = np.clip(z_samp, 1e-6, None)
            z_samp = np.sort(z_samp)
        else:
            z_samp = np.array([z0], dtype=float)

        # Model implied GW distance samples: shape (n_draws, n_z)
        dL_gw, _R = predict_dL_gw(
            post,
            z_eval=z_samp,
            convention=convention,
            allow_extrapolation=allow_extrapolation,
        )
        # GR baseline implied EM distance samples (same EM posterior draws, but no GW damping).
        dL_em = predict_dL_em(post, z_eval=z_samp)

        dL_gw_flat = dL_gw.reshape(-1)
        dL_em_flat = dL_em.reshape(-1)

        logp_mu = ev.loglike_proxy(dL_gw_flat)
        logp_gr = ev.loglike_proxy(dL_em_flat)

        lpd_mu = float(_logmeanexp(logp_mu))
        lpd_gr = float(_logmeanexp(logp_gr))
        delta = lpd_mu - lpd_gr

        lpd_mu_total += lpd_mu
        lpd_gr_total += lpd_gr

        scores.append(SirenEventScore(run=run_label, event=ev.name, z=z0, lpd_mu=lpd_mu, lpd_gr=lpd_gr, delta_lpd=delta))

    summ = SirenTestSummary(
        run=run_label,
        convention=str(convention),
        n_events=int(len(events)),
        n_draws=int(post.H_samples.shape[0]),
        lpd_mu_total=float(lpd_mu_total),
        lpd_gr_total=float(lpd_gr_total),
        delta_lpd_total=float(lpd_mu_total - lpd_gr_total),
    )
    return summ, scores


def to_json(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), indent=2)
    raise TypeError("Object is not a dataclass.")
