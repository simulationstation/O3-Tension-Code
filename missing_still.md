# Missing / TODO for “full” seed replication

The repo now supports **one-command headline verification from the committed artifact bundle** (`make reproduce`). A true *from-scratch* rerun (no precomputed artifacts) still needs a few large / external pieces that are not committed here.

## Large inputs not committed

- **Full O3 gap-test event cache** (per-event `event_*.npz`), required to rerun `scripts/run_dark_siren_gap_test.py` end-to-end.
  - Example location on the dev machine: `/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC/cache` (~573 MB).
  - Action: package/download via the project archive (Zenodo) into a stable in-repo path (e.g. `data/`), and update configs/scripts to use relative paths.

- **GLADE+ pixel indices** used for host-catalog queries (needed for spec-z override and other hardening tasks).
  - Example dev machine paths (not portable): `/home/primary/PROJECT/data/processed/galaxies/gladeplus/index_nside128*`
  - Action: document how to obtain/build these, and/or provide a lightweight “top-host candidate list” artifact so downstream audits don’t require the full index.

## Spec-z override offline caches

The “spec-z override” audit uses public spec-z catalogs and (optionally) CDS XMatch.

- Expected cache dirs (configurable):
  - `data/cache/specz_catalogs/`
  - `data/cache/specz_xmatch/`
- Status: this repo now includes an **offline replay pack** (tables + match manifests + override maps) at `artifacts/o3/specz_override_pack_20260214_064541UTC/` so the reported coverage/ΔLPD figures can be inspected without re-querying external services.
- Remaining: the full upstream catalog caches (e.g. complete DESI/SDSS downloads) are still intentionally not committed; they should live in the project archive (Zenodo) and/or be re-downloaded by the user.

## One-command *full rerun* entrypoint

Add a single driver (e.g. `make reproduce-full`) that:

1. downloads/locates large inputs,
2. runs the baseline scoring, injection null suite, and hardening suite with fixed seeds,
3. writes a timestamped `outputs/.../report.md` + `summary.json`,
4. verifies expected hashes/ΔLPD outputs.

## External repo note

If any missing CQG-paper-relevant piece is only present in `/home/O2-Modified-Gravity-Hubble resolution` (active work-in-progress), do **not** vendor it yet; instead, list it here with an exact path and the minimal file(s) needed.
