# Seed Replication (one-command)

Goal: reproduce the **headline CQG numbers** (and rebuild the CQG PDF if `pdflatex` is available) in a single command, without hunting through scripts.

## Quickstart

```bash
make reproduce
```

If you don’t already have a virtualenv, the minimal setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
make reproduce
```

This creates a timestamped folder:

```text
outputs/seed_reproduce_YYYYMMDD_HHMMSSUTC/
  report.md
  summary.json
  figures/
  papers/        # only if pdflatex is found and build succeeds
```

If you don’t have a TeX install:

```bash
make reproduce-fast
```

## What it reproduces (from in-repo artifacts)

- **Baseline (full) ΔLPD** from `artifacts/o3/summary_M0_start101.json`
- **GR-truth injection calibration** (tail probability estimate + Z-equivalent) from `artifacts/o3/catalog_injection_deltas_n512.json`
- **Hardening-suite summaries** (fixed-power grid + GR-systematics matrix) from:
  - `artifacts/o3/fixed_power_grid_summary.json`
  - `artifacts/o3/systematics_matrix_summary.json`
- **Jackknife influences** from `artifacts/o3/jackknife_M0_start101.json`
- **Spectral-only baseline ΔLPD (cached)** computed from:
  - `artifacts/o3/smoking_gun_inputs/spectral_only_cached_terms.npz`
  - `artifacts/o3/selection_alpha_M0_start101.npz`

The report also prints SHA256 checksums of the key inputs to make offline verification straightforward.

## Expected headline numbers

From the current artifact bundle (see `outputs/.../report.md`):

- Full baseline: `ΔLPD_total ≈ +3.670` (with `exp(ΔLPD) ≈ 39`)
- GR-truth injection calibration (N=512): max `ΔLPD_total ≈ +0.076` (so `P(Δ>=obs)` is effectively zero at this ensemble size)
- Spectral-only baseline (cached): `ΔLPD_total ≈ +3.634` (computed in `scripts/reproduce_seed_pack.py`)

## Full rerun (optional; heavy)

This repository contains the *code* to rerun the full pipeline, but the full “from-scratch” rerun requires large external inputs (GW PE samples, GLADE+ indices, injection products, etc.). Those are expected to be obtained from the project archive (Zenodo DOI referenced in the CQG paper Declarations) and/or generated locally.

If you want the repo to support a “true” full rerun in one command (not just artifact verification), track that work in `missing_still.md`.
