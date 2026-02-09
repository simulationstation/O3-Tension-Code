# Reviewer Checklist

## Scope

This package is scoped to reproducibility of:
- O3/GWTC-3 dark-siren propagation-vs-GR scoring,
- GR-truth catalog-injection calibration,
- transfer-bias Hubble relief follow-up.

## Core checks

1. Install environment from `pyproject.toml`.
2. Run dark-siren smoke tests in `tests/`.
3. Recompute O3 score via `scripts/run_dark_siren_gap_test.py`.
4. Recompute GR-truth calibration via `scripts/run_dark_siren_catalog_injection_suite.py`.
5. Recompute transfer-bias fit via `scripts/run_joint_transfer_bias_fit.py`.

## Expected outputs

- O3 summary JSON (`summary_M0_start101.json`-style fields): total score, data term, selection term, Bayes proxy.
- Injection-suite summary JSON: mean/sd/max and tail probabilities under GR-truth.
- Transfer-bias fit summary JSON: relief fraction posterior and nuisance posteriors.

## Provenance pointers

- Inverse model construction: `scripts/run_realdata_recon.py`, `src/entropy_horizon_recon/inversion.py`.
- Dark-siren scoring path: `scripts/run_dark_siren_gap_test.py`, `src/entropy_horizon_recon/dark_sirens*.py`.
- Selection alpha path: `src/entropy_horizon_recon/dark_sirens_selection.py`.

