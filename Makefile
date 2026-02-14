.PHONY: reproduce reproduce-fast

# Seed replication entrypoints.
# - `reproduce`: generates outputs/seed_reproduce_<timestamp>/ with report.md + summary.json,
#   and attempts to rebuild CQG_PAPER PDFs if pdflatex is available.
# - `reproduce-fast`: same, but skips the LaTeX build step.

# Prefer the local venv if present.
PY ?= python3
ifneq ("$(wildcard .venv/bin/python)","")
PY := .venv/bin/python
endif

reproduce:
	$(PY) scripts/reproduce_seed_pack.py --build-paper

reproduce-fast:
	$(PY) scripts/reproduce_seed_pack.py
