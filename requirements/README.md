# Requirements (pinned)

These files are intended to make reproduction less “dependency-archaeology”-heavy.

## Lite (seed report only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements/seed_reproduce_lite.lock.txt
make reproduce-fast
```

## Full (pipeline + sirens/optical extras)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements/seed_reproduce_full.lock.txt
```

Notes:
- The “full” lock includes an editable install of this repo (`-e .[...]`).
- A true from-scratch rerun still requires large external datasets; see `missing_still.md`.

