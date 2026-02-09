# Ancillary Cross-Probe Tests

This document describes the two secondary checks included with the reviewer build. They are context tests, not the primary evidence channel.

## 1) Strong-lens time-delay holdout

Purpose:
- Check whether an independent lens-distance likelihood prefers the same propagation direction as the O3 dark-siren signal.

How it was run:
- Fetch public lens products:
  - `scripts/fetch_strong_lens_real_catalogs.py`
- Run real-catalog comparison:
  - `scripts/run_strong_lens_real_catalog_test.py`
- Run KDE nuisance calibration suite:
  - `scripts/run_strong_lens_calibration_suite.py`

Stored artifacts:
- `artifacts/ancillary/strong_lens/strong_lens_real_catalog_results.json`
- `artifacts/ancillary/strong_lens/strong_lens_calibration_suite_results.json`
- `artifacts/ancillary/strong_lens/strong_lens_lpd_by_run.png`
- `artifacts/ancillary/strong_lens/kde_nuisance_delta_lpd_ranked.png`

Headline outcome:
- Mild BH/GR-favoring tendency in this implementation (negative `DeltaLPD(MG-BH)` across the nuisance sweep).

## 2) Void-prism three-source joint check

Purpose:
- Test whether a void-conditioned lensing/velocity prism observable carries a directional MG-like shift.

How it was run:
- Build tomographic `theta` maps from ACT DR6 + SDSS kSZx inputs:
  - `scripts/build_theta_maps_tomo_from_act_dr6_sdss_kszx.py`
- Build void-prism observable vector:
  - `scripts/build_void_prism_eg_measurement.py`
- Estimate covariance/jackknife suite:
  - `scripts/measure_void_prism_eg_suite_jackknife.py`
- Joint model-vs-GR score:
  - `scripts/run_void_prism_eg_joint_test.py`

Stored artifacts:
- `artifacts/ancillary/void/void_prism_three_source_results.json`
- `artifacts/ancillary/void/void_prism_three_source_suite_joint.json`

Headline outcome:
- Tiny same-sign `DeltaLPD` shifts in all five seeds (positive but near-tie scale); treated as non-decisive at current S/N.

## Interpretation guardrail

These ancillary probes are retained to show cross-check behavior and stress-test directionality. The primary discriminator in this project remains the calibrated O3 dark-siren pipeline.
