# O3 Tension Code (Reviewer Build)

This repository is a clean, review-focused export of the code and artifacts used for:

1. GWTC-3/O3 dark-siren posterior-predictive model comparison (propagation model vs internal GR baseline), and
2. Follow-on Hubble-tension transfer-bias analysis driven by the same inferred propagation deformation.

## What this repository is testing

The modified-propagation model here is **not** a generic free MG ansatz. It is a fixed propagation history derived from the inverse nonparametric reconstruction pipeline (`scripts/run_realdata_recon.py`, `src/entropy_horizon_recon/inversion.py`) and then evaluated out-of-sample on dark sirens.

In short:
- Stage A: reconstruct `mu(A)` from late-time cosmology (inverse pipeline).
- Stage B: map that to GW propagation (`dL_GW/dL_EM`) and score GWTC-3 dark sirens vs GR.
- Stage C: calibrate false-positive behavior under GR-truth injections.
- Stage D: propagate to a transfer-bias Hubble-tension relief analysis.

## Included contents

- `src/entropy_horizon_recon/` core package (dark sirens + inverse reconstruction + transfer-bias utilities).
- `scripts/` runnable entrypoints for the O3 tests and Hubble follow-up.
- `tests/` focused smoke/regression tests for dark-siren and mapping components.
- `papers/dark_siren/` current dark-siren letter source and PDF.
- `papers/hubble_tension/` Hubble-tension follow-up source and PDF.
- `papers/inverse_recon/main_updated.tex` inverse reconstruction manuscript source.
- `paper_gen/mrnas/` submission-ready MNRAS Letter build (TeX + embedded figures + PDF).
- `artifacts/o3/` key O3 result summaries and plots.
- `artifacts/hubble/` key transfer-bias fit summaries and plots.
- `artifacts/ancillary/` secondary cross-probe checks (void prism + strong-lens holdouts).

## Key result artifacts (already produced)

- O3 real-data baseline summary:
  - `artifacts/o3/summary_M0_start101.json`
- Event leverage / hero events:
  - `artifacts/o3/jackknife_M0_start101.json`
- Per-event contribution figure:
  - `artifacts/o3/delta_lpd_by_event_M0_start101.png`
- GR-truth catalog-injection calibration summary + figures:
  - `artifacts/o3/catalog_injection_summary.json`
  - `artifacts/o3/catalog_injection_deltas_n512.json`
  - `artifacts/o3/fig_delta_lpd_total_hist.png`
  - `artifacts/o3/fig_delta_lpd_components_hist.png`
- Fixed-power injection response grid:
  - `artifacts/o3/fixed_power_grid_summary.json`
  - `artifacts/o3/fig_fixed_power_grid.png`
- GR-consistent systematics truth matrix:
  - `artifacts/o3/systematics_matrix_summary.json`
  - `artifacts/o3/fig_systematics_matrix.png`
- Hubble transfer-bias posterior summary:
  - `artifacts/hubble/summary.json`
  - `artifacts/hubble/final_relief_posterior_summary.json`
- Ancillary strong-lens holdout summaries:
  - `artifacts/ancillary/strong_lens/strong_lens_real_catalog_results.json`
  - `artifacts/ancillary/strong_lens/strong_lens_calibration_suite_results.json`
- Ancillary void-prism joint summaries:
  - `artifacts/ancillary/void/void_prism_three_source_results.json`
  - `artifacts/ancillary/void/void_prism_three_source_suite_joint.json`

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install healpy astropy h5py ligo.skymap
```

## Minimal reproducibility commands

### 1) Build / verify local data indexes

```bash
PYTHONPATH=src .venv/bin/python scripts/build_gwtc_pe_index.py
```

### 2) O3 dark-siren real-data scoring

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_gap_test.py \
  --out outputs/dark_siren_o3_repro \
  --selection-injections-hdf auto
```

### 3) O3 GR-truth catalog injection calibration

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_catalog_injection_suite.py \
  --out outputs/dark_siren_catalog_injection_repro \
  --n-reps 128 --max-draws 256
```

### 4) Hubble transfer-bias fit

```bash
PYTHONPATH=src .venv/bin/python scripts/run_joint_transfer_bias_fit.py \
  --out outputs/joint_transfer_bias_fit_repro
```

### 5) Ancillary strong-lens checks (real-catalog + nuisance calibration)

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_strong_lens_real_catalogs.py
PYTHONPATH=src .venv/bin/python scripts/run_strong_lens_real_catalog_test.py \
  --out outputs/strong_lens_real_catalog_repro
PYTHONPATH=src .venv/bin/python scripts/run_strong_lens_calibration_suite.py \
  --out outputs/strong_lens_calibration_suite_repro
```

### 6) Ancillary void-prism three-source joint check

```bash
PYTHONPATH=src .venv/bin/python scripts/build_theta_maps_tomo_from_act_dr6_sdss_kszx.py \
  --out outputs/theta_tomo_kszx_repro
PYTHONPATH=src .venv/bin/python scripts/build_void_prism_eg_measurement.py \
  --out outputs/void_prism_measurement_repro
PYTHONPATH=src .venv/bin/python scripts/measure_void_prism_eg_suite_jackknife.py \
  --out outputs/void_prism_suite_repro
PYTHONPATH=src .venv/bin/python scripts/run_void_prism_eg_joint_test.py \
  --embedding minimal \
  --out outputs/void_prism_joint_repro
```

`run_void_prism_eg_joint_test.py` now requires explicit `--embedding` selection
(unless `--allow-implicit-minimal` is set for legacy behavior) and records both
fixed-amplitude and fitted-amplitude score channels by default.

### 7) Hero-event waveform-family consistency matrix

Inputs for this run are bundled in:
- `artifacts/waveform/hero_waveform_inputs_20260210/events/`

Prepare and launch a 256-core matrix:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_hero_waveform_consistency_prod_matrix.py \
  --event-root artifacts/waveform/hero_waveform_inputs_20260210/events \
  --out-root outputs/forward_tests/hero_waveform_consistency_prod_repro \
  --cores-total 256 --cores-per-job 8 \
  --bilby-pipe-analysis .venv/bin/bilby_pipe_analysis \
  --launch
```

Check live status:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_hero_waveform_prod_status.py \
  --run-root outputs/forward_tests/hero_waveform_consistency_prod_repro
```

## Detached long-run launchers

Detachable launch scripts are included and follow the required robust pattern (`setsid`, `taskset`, `pid.txt`, `run.log`):

- `scripts/launch_dark_siren_high_value_suite_single_nohup.sh`
- `scripts/launch_dark_siren_hier_selection_uncertainty_single_nohup.sh`
- `scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh`
- `scripts/launch_joint_transfer_bias_fit_single_nohup.sh`

## Data source DOI manifest

Core O3 dark-siren inputs:
- GWTC-3 PE products: DOI `10.1103/PhysRevX.13.041039`
- GLADE+ galaxy catalog: DOI `10.1093/mnras/stac1443`
- Code/archive bundle for this analysis: DOI `10.5281/zenodo.18510138`

Ancillary cross-probe inputs:
- Planck 2018 lensing products: DOI `10.1051/0004-6361/201833886`
- BOSS DR12 cosmology release (void environment context): DOI `10.1093/mnras/stx721`
- H0LiCOW lensing-program reference: DOI `10.1093/mnras/stz3094`
- TDCOSMO 2025 public chain payloads: public release repo (no DOI assigned in release metadata)

If a source release has no DOI, the exact upstream URL and file manifest are captured by the corresponding ingest/fetch script metadata and run outputs.

## Notes for reviewers

- This repo intentionally excludes large exploratory clutter and unrelated pipelines.
- It keeps the exact model path that generated the reported O3 tension and the follow-on transfer-bias analysis.
- The strongest constraints in this codebase remain O3 dark-siren based; several other tested families were retained only as secondary stress tests and are not required here.
