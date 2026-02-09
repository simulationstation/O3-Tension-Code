from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str) -> list[str]:
    out = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not out:
        raise ValueError("Expected a comma-separated list.")
    return out


def _parse_float_edges(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([float(p) for p in parts], dtype=float)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Edges must be finite, strictly increasing, length>=2.")
    return edges


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def _run(cmd: list[str]) -> None:
    print(f"[theta_kszx] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _ensure_wget(url: str, dst: Path) -> None:
    """Resumable download helper.

    We avoid relying on the python 'wget' module here since it restarts from 0 on transient
    failures. This helper is intentionally serial: do not parallelize downloads.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return
    except Exception:
        pass

    # Use system wget with resume.
    _run(["wget", "-c", "-O", str(dst), url])
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed: url={url} -> {dst}")


def _sdss_dr_str(dr: int) -> str:
    if dr == 11:
        return "DR11v1"
    if dr == 12:
        return "DR12v5"
    raise ValueError(f"Unsupported SDSS dr={dr} (supported: 11, 12).")


def _ensure_gunzip(src_gz: Path, dst: Path) -> None:
    if dst.exists():
        return
    import gzip

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp")
    try:
        with gzip.open(src_gz, "rb") as f_in, open(tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _ensure_sdss_galaxy_catalog(*, kszx_data_dir: Path, survey: str, dr: int) -> None:
    """Fetch SDSS/BOSS LSS catalogs in the exact location kszx expects."""
    dr_str = _sdss_dr_str(int(dr))
    base = kszx_data_dir / "sdss" / dr_str
    fits = base / f"galaxy_{dr_str}_{survey}.fits"
    gz = Path(str(fits) + ".gz")  # -> .fits.gz
    url = f"https://data.sdss.org/sas/dr{int(dr)}/boss/lss/{fits.name}.gz"
    _ensure_wget(url, gz)
    _ensure_gunzip(gz, fits)


def _ensure_act_map_and_ivar(*, kszx_data_dir: Path, act_dr: int, freq: int, act_night: bool) -> None:
    """Ensure the ACT DR6 srcfree coadd + ivar exist under KSZX_DATA_DIR.

    Note: ACT DR6 maps are available on LAMBDA. If LAMBDA is flaky in this environment,
    prefer downloading through kszx (which may use alternate paths) or the NERSC portal.
    """
    act_time = "night" if bool(act_night) else "daynight"
    if int(act_dr) == 5:
        url_base = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/"
        cmb_rel = f"act_planck_dr5.01_s08s18_AA_f{int(freq):03d}_{act_time}_map_srcfree.fits"
        ivar_rel = f"act_planck_dr5.01_s08s18_AA_f{int(freq):03d}_{act_time}_ivar.fits"
        dst_cmb = kszx_data_dir / "act" / "dr5.01" / cmb_rel
        dst_ivar = kszx_data_dir / "act" / "dr5.01" / ivar_rel
        _ensure_wget(url_base + cmb_rel, dst_cmb)
        _ensure_wget(url_base + ivar_rel, dst_ivar)
        return

    if int(act_dr) == 6:
        url_base = "https://lambda.gsfc.nasa.gov/data/act/"
        cmb_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(freq):03d}_map_srcfree.fits"
        ivar_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(freq):03d}_ivar.fits"
        dst_cmb = kszx_data_dir / "act" / "dr6.02" / cmb_rel
        dst_ivar = kszx_data_dir / "act" / "dr6.02" / ivar_rel
        _ensure_wget(url_base + cmb_rel, dst_cmb)
        _ensure_wget(url_base + ivar_rel, dst_ivar)
        return

    raise ValueError(f"Unsupported ACT dr={act_dr} (supported: 5, 6).")


@dataclass(frozen=True)
class ThetaBinMeta:
    z_min: float
    z_max: float
    n_gal_used: int
    theta_fits: str
    mask_fits: str


@dataclass(frozen=True)
class Manifest:
    created_utc: str
    act_dr: int
    act_freq_ghz: int
    act_night: bool
    sdss_surveys: list[str]
    sdss_dr: int
    # kSZ weighting / filter
    planck_galmask_sky_pct: int
    planck_galmask_apod_deg: int
    act_rms_threshold_ukarcmin: float
    ksz_lmin: int
    ksz_lmax: int
    halo_zeff: float
    halo_ngal_mpc3: float
    halo_profile: str
    # output grids
    frame: str
    nside: int
    remove_dipole: bool
    kszx_data_dir: str
    z_edges: list[float]
    bins: list[ThetaBinMeta]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build tomographic HEALPix theta maps using the kszx kSZ filter scaffold:\n"
            "  - Planck HFI galactic mask (projected into ACT pixell geometry)\n"
            "  - ACT DR6 srcfree temperature filtered with F_l ~ (b_l * X_l^{ge}/X_l^{gg}) / C_l^{tot}\n"
            "  - sample filtered temperature at SDSS/BOSS galaxies, mean-subtract per z bin, bin to HEALPix\n"
            "\n"
            "This is an accuracy-oriented upgrade over a simple (lmin,lmax) bandpass.\n"
            "It is still a *projected* velocity-proxy map (Tier 1.5), but uses the standard kSZ weighting logic.\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--act-dr", type=int, default=6, help="ACT data release (default 6).")
    ap.add_argument("--act-freq", type=int, default=150, help="ACT frequency in GHz (default 150).")
    ap.add_argument("--act-night", action="store_true", help="Use ACT night coadd (default daynight).")
    ap.add_argument("--act-rms-threshold", type=float, default=70.0, help="Noise threshold in uK-arcmin for wcmb mask (default 70).")
    ap.add_argument("--planck-galmask-sky-pct", type=int, default=70, help="Planck HFI GAL0xx sky fraction (default 70).")
    ap.add_argument("--planck-galmask-apod-deg", type=int, default=0, help="Planck HFI galmask apodization in deg (default 0).")

    ap.add_argument("--sdss-surveys", default="CMASSLOWZTOT_North,CMASSLOWZTOT_South", help="Comma-separated SDSS surveys.")
    ap.add_argument("--sdss-dr", type=int, default=12, help="SDSS data release (default 12).")
    ap.add_argument("--zmin", type=float, default=0.20, help="Min redshift (default 0.20).")
    ap.add_argument("--zmax", type=float, default=0.70, help="Max redshift (default 0.70).")
    ap.add_argument(
        "--z-edges",
        default="0.2,0.36,0.48,0.56,0.67",
        help="Comma-separated z edges (tomographic bins). Default matches the current void prism suite.",
    )
    ap.add_argument("--weights", choices=["wfkp", "fkp_sys"], default="fkp_sys", help="Galaxy weights (default fkp_sys).")
    ap.add_argument("--max-galaxies", type=int, default=None, help="Optional global subsample size for a smoke run.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed used when --max-galaxies is set.")

    # kSZ halo-model filter config (matches kszx notebook defaults by default).
    ap.add_argument("--halo-zeff", type=float, default=0.55, help="Effective redshift for halo-model filter (default 0.55).")
    ap.add_argument("--halo-ngal", type=float, default=1e-4, help="Galaxy number density for shot-noise term (Mpc^-3). Default 1e-4.")
    ap.add_argument("--halo-profile", default="AGN", help="hmvec Battaglia electron profile family (default AGN).")

    ap.add_argument("--ksz-lmin", type=int, default=1500, help="kSZ filter lmin (default 1500).")
    ap.add_argument("--ksz-lmax", type=int, default=8000, help="kSZ filter lmax (default 8000).")

    ap.add_argument("--nside", type=int, default=256, help="Output HEALPix nside (default 256).")
    ap.add_argument("--frame", choices=["galactic", "icrs"], default="galactic", help="Output HEALPix frame (default galactic).")
    ap.add_argument("--remove-dipole", action="store_true", help="Remove monopole/dipole from each theta map (after binning).")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: data/processed/void_prism/theta_tomo_kszx_<stamp>/).")
    args = ap.parse_args()

    hp = _require_healpy()

    kszx_data_dir = Path(args.kszx_data_dir).resolve()
    kszx_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KSZX_DATA_DIR"] = str(kszx_data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else Path("data/processed/void_prism") / f"theta_tomo_kszx_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    # Ensure critical public inputs (downloads are serial and resumable).
    _ensure_act_map_and_ivar(
        kszx_data_dir=kszx_data_dir,
        act_dr=int(args.act_dr),
        freq=int(args.act_freq),
        act_night=bool(args.act_night),
    )
    surveys = _parse_csv_list(str(args.sdss_surveys))
    for s in surveys:
        _ensure_sdss_galaxy_catalog(kszx_data_dir=kszx_data_dir, survey=s, dr=int(args.sdss_dr))

    # Defer heavyweight imports until after we set KSZX_DATA_DIR.
    import hmvec as hm  # type: ignore
    import pixell  # type: ignore
    import pixell.enmap as enmap  # type: ignore
    import pixell.reproject as reproject  # type: ignore
    import kszx  # type: ignore

    from kszx import act as kact  # type: ignore
    from kszx import planck as kplanck  # type: ignore
    from kszx import pixell_utils as pxu  # type: ignore
    from kszx import sdss as ksdss  # type: ignore
    from kszx.Catalog import Catalog  # type: ignore
    from kszx.Cosmology import Cosmology  # type: ignore

    # Load ACT map + ivar.
    cmb = kact.read_cmb(int(args.act_freq), int(args.act_dr), night=bool(args.act_night), download=False)
    ivar = kact.read_ivar(int(args.act_freq), int(args.act_dr), night=bool(args.act_night), download=False)
    if cmb.shape != ivar.shape:
        raise RuntimeError(f"ACT cmb/ivar shapes differ: {cmb.shape} vs {ivar.shape}")

    # Project Planck galmask into pixell geometry matching the ACT map.
    healpix_galmask = kplanck.read_hfi_galmask(
        sky_percentage=int(args.planck_galmask_sky_pct),
        apodization=int(args.planck_galmask_apod_deg),
        download=True,
    )
    pixell_galmask = reproject.healpix2map(
        healpix_galmask,
        cmb.shape,
        cmb.wcs,
        rot="gal,equ",
        method="spline",
        order=0,
    )

    # ACT DR6 NILC masks (from NERSC portal; already proven to download reliably here).
    # These are in the same pixell geometry already.
    nilc_wide = kact.read_nilc_wide_mask(download=True)
    cluster_mask = kact.read_cluster_mask(download=True)

    # CMB weight map W_CMB(theta): galmask * wide * cluster * (noise<thresh).
    wcmb = pixell_galmask.astype(np.float32, copy=True)
    wcmb *= nilc_wide
    wcmb *= cluster_mask
    wcmb *= (pxu.uK_arcmin_from_ivar(ivar) < float(args.act_rms_threshold))

    # Halo-model-based galaxy/electron filter: compute X_l^{ge} and X_l^{gg}.
    # This follows the kszx "prepare_gal_filter" notebook scaffold.
    lmin = int(args.ksz_lmin)
    lmax = int(args.ksz_lmax)
    if lmin < 2 or lmax <= lmin:
        raise ValueError("--ksz-lmin must be >=2 and < --ksz-lmax.")

    ks = np.geomspace(1e-5, 100.0, 1000)
    ms = np.geomspace(2e10, 1e17, 40)
    zeff = float(args.halo_zeff)
    ngal = float(args.halo_ngal)

    hcos = hm.HaloModel([zeff], ks, ms=ms)
    hcos.add_battaglia_profile("electron", family=str(args.halo_profile))
    hcos.add_hod(name="g", ngal=np.asarray([ngal]))
    hpge = hcos.get_power_1halo("g", "electron") + hcos.get_power_2halo("g", "electron")
    hpgg = hcos.get_power_1halo("g", "g") + hcos.get_power_2halo("g", "g") + 1.0 / ngal
    hpge = np.asarray(hpge[0], dtype=float)
    hpgg = np.asarray(hpgg[0], dtype=float)
    chieff = float(hcos.comoving_radial_distance(zeff))

    # X_l^{ij} = P_ij(k=l/chi_eff)
    import scipy.interpolate  # type: ignore

    interp_logk_pge = scipy.interpolate.InterpolatedUnivariateSpline(np.log(ks), hpge)
    interp_logk_pgg = scipy.interpolate.InterpolatedUnivariateSpline(np.log(ks), hpgg)
    ell = np.arange(lmax + 1, dtype=float)
    ell[0] = 0.1  # avoid log(0)
    logk = np.log(ell / chieff)
    xl_ge = interp_logk_pge(logk)
    xl_gg = interp_logk_pgg(logk)
    xl_ge = np.asarray(xl_ge, dtype=float)
    xl_gg = np.asarray(xl_gg, dtype=float)

    xl_path = out_dir / "intermediate" / "xl_ge_gg.txt"
    np.savetxt(
        xl_path,
        np.transpose([np.arange(lmax + 1), xl_ge, xl_gg]),
        header=(
            "Col 0: l\n"
            "Col 1: X_l^{ge} (Mpc^3)\n"
            "Col 2: X_l^{gg} (Mpc^3)\n"
            "We define X_l^{ij} = P_{ij}(k)_{k=l/chi_eff}\n"
        ),
    )

    # Beam and CMB power fit.
    # Note: kszx.CmbClFitter internally pads its harmonic transforms by lpad (default 1000),
    # so it requires both:
    #   - a beam defined out to (lmax+lpad)
    #   - a Cosmology object with cosmo.lmax >= (lmax+lpad)
    # Therefore, we read the full beam file here (no truncation) and bump cosmo.lmax accordingly.
    bl_full = kact.read_beam(
        int(args.act_freq),
        dr=int(args.act_dr),
        lmax=None,
        night=bool(args.act_night),
        download=True,
    )
    cosmo = Cosmology("planck18+bao", lmax=int(lmax + 1000))
    cl_fitter = kszx.CmbClFitter(
        cosmo=cosmo,
        cmb_map=cmb,
        weight_map=wcmb,
        bl=bl_full,
        lmin=lmin,
        lmax=lmax,
        ivar=ivar,
    )

    cl_tot = np.asarray(cl_fitter.cl_tot, dtype=float)
    if cl_tot.shape[0] < lmax + 1:
        raise RuntimeError("CmbClFitter returned cl_tot shorter than expected.")

    # Filter F_l.
    if bl_full.shape[0] < lmax + 1:
        raise RuntimeError(f"Beam file too short: len(bl)={bl_full.shape[0]} but need >= {lmax+1}")
    bl = bl_full[: lmax + 1]
    fl = bl * xl_ge / xl_gg / cl_tot[: lmax + 1]
    fl[:lmin] = 0.0
    fl_path = out_dir / "intermediate" / f"fl_f{int(args.act_freq):03d}.txt"
    np.savetxt(
        fl_path,
        np.transpose([np.arange(lmax + 1), fl, cl_tot[: lmax + 1]]),
        header=(
            "Col 0: l\n"
            "Col 1: F_l (CMB l-weighting in kSZ quadratic estimator)\n"
            "Col 2: C_l (fit for CMB power spectrum, beam-convolved, includes noise)\n"
        ),
    )

    # Filtered temperature map t_cmb(theta) = Alm^{-1}[ F_l * Alm[ Wcmb * T(theta) ] ].
    tmask = wcmb.copy()
    tcmb = cmb.copy()
    tcmb *= tmask
    alm = pxu.map2alm(tcmb, lmax)
    alm = pixell.curvedsky.almxfl(alm, fl)
    tcmb_f = pxu.alm2map(alm, tcmb.shape, tcmb.wcs)

    # Load SDSS galaxy catalogs and merge.
    cats: list[Catalog] = []
    for s in surveys:
        cat = ksdss.read_galaxies(s, dr=int(args.sdss_dr), download=False)
        cat.apply_redshift_cut(float(args.zmin), float(args.zmax))
        cats.append(cat)
    gcat = Catalog.concatenate(cats, name=" + ".join(surveys), destructive=True) if len(cats) > 1 else cats[0]

    # Base per-galaxy weights (matching SDSS conventions).
    if args.weights == "wfkp":
        w_base = np.asarray(gcat.wfkp, dtype=float)
    else:
        w_base = np.asarray((gcat.wzf + gcat.wcp - 1.0) * gcat.wsys * gcat.wfkp, dtype=float)

    z = np.asarray(gcat.z, dtype=float)
    ra = np.asarray(gcat.ra_deg, dtype=float)
    dec = np.asarray(gcat.dec_deg, dtype=float)

    m0 = np.isfinite(z) & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(w_base) & (w_base > 0)
    if not np.any(m0):
        raise RuntimeError("No valid galaxies after weights/finite cuts.")
    ra = ra[m0]
    dec = dec[m0]
    z = z[m0]
    w_base = w_base[m0]

    if args.max_galaxies is not None and 0 < int(args.max_galaxies) < ra.size:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(ra.size, size=int(args.max_galaxies), replace=False)
        ra, dec, z, w_base = ra[idx], dec[idx], z[idx], w_base[idx]

    z_edges = _parse_float_edges(str(args.z_edges))
    assert z_edges is not None

    bins: list[ThetaBinMeta] = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:], strict=False):
        m_z = (z >= float(z0)) & (z < float(z1)) & np.isfinite(z)
        if not np.any(m_z):
            continue

        ra_sel = ra[m_z]
        dec_sel = dec[m_z]
        w_sel = w_base[m_z]

        # Minimal Catalog for eval_map_on_catalog.
        cat_bin = Catalog({"ra_deg": ra_sel, "dec_deg": dec_sel, "z": z[m_z], "wfkp": np.ones_like(w_sel)})
        tvals, inb = pxu.eval_map_on_catalog(tcmb_f, cat_bin, pad=0.0, return_mask=True)
        tvals = np.asarray(tvals, dtype=float)
        inb = np.asarray(inb, dtype=bool)

        ok = inb & np.isfinite(tvals) & np.isfinite(w_sel) & (w_sel > 0)
        if not np.any(ok):
            continue

        # Foreground mitigation: subtract weighted mean within the bin.
        mu = _weighted_mean(tvals[ok], w_sel[ok])
        if np.isfinite(mu):
            tvals = np.array(tvals, copy=True)
            tvals[ok] -= mu

        pix = radec_to_healpix(
            ra_sel[ok],
            dec_sel[ok],
            nside=int(args.nside),
            frame=str(args.frame),
            nest=False,
        )
        npix = int(hp.nside2npix(int(args.nside)))
        wsum = np.bincount(pix, weights=w_sel[ok] * tvals[ok], minlength=npix).astype(float)
        wcnt = np.bincount(pix, weights=w_sel[ok], minlength=npix).astype(float)
        theta_map = np.zeros(npix, dtype=float)
        good = wcnt > 0
        theta_map[good] = wsum[good] / wcnt[good]
        mask = np.zeros(npix, dtype=float)
        mask[good] = 1.0

        if bool(args.remove_dipole):
            theta_map = hp.remove_dipole(theta_map, fitval=False, verbose=False)

        tag = f"z{float(z0):.3f}-{float(z1):.3f}"
        theta_path = out_dir / f"theta_kszx_act_dr{int(args.act_dr)}_f{int(args.act_freq):03d}_sdss_{tag}.fits"
        mask_path = out_dir / f"mask_kszx_act_dr{int(args.act_dr)}_f{int(args.act_freq):03d}_sdss_{tag}.fits"
        hp.write_map(str(theta_path), theta_map, overwrite=True, dtype=np.float64)
        hp.write_map(str(mask_path), mask, overwrite=True, dtype=np.float64)

        bins.append(
            ThetaBinMeta(
                z_min=float(z0),
                z_max=float(z1),
                n_gal_used=int(np.sum(ok)),
                theta_fits=str(theta_path),
                mask_fits=str(mask_path),
            )
        )

    man = Manifest(
        created_utc=_utc_stamp(),
        act_dr=int(args.act_dr),
        act_freq_ghz=int(args.act_freq),
        act_night=bool(args.act_night),
        sdss_surveys=surveys,
        sdss_dr=int(args.sdss_dr),
        planck_galmask_sky_pct=int(args.planck_galmask_sky_pct),
        planck_galmask_apod_deg=int(args.planck_galmask_apod_deg),
        act_rms_threshold_ukarcmin=float(args.act_rms_threshold),
        ksz_lmin=lmin,
        ksz_lmax=lmax,
        halo_zeff=float(args.halo_zeff),
        halo_ngal_mpc3=float(args.halo_ngal),
        halo_profile=str(args.halo_profile),
        frame=str(args.frame),
        nside=int(args.nside),
        remove_dipole=bool(args.remove_dipole),
        kszx_data_dir=str(kszx_data_dir),
        z_edges=[float(x) for x in z_edges.tolist()],
        bins=bins,
    )
    (out_dir / "manifest.json").write_text(
        __import__("json").dumps(asdict(man), indent=2, sort_keys=True) + "\n"
    )

    print(f"[theta_kszx] Wrote {len(bins)} theta bins under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
