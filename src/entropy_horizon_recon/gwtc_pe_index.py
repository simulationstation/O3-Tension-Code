from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_EVENT_RE = re.compile(r"(GW\d{6}_\d{6})")


@dataclass(frozen=True)
class GWTCPeFile:
    """Reference to a GWTC posterior-sample release file on disk."""

    event: str
    record_id: int
    release: str
    variant: str
    path: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _infer_release(record_id: int) -> str:
    # These are the Zenodo records we currently mirror locally.
    m = {
        6513631: "GWTC-2.1",
        5546663: "GWTC-3",
        17014085: "GWTC-4.0",
    }
    return m.get(int(record_id), f"zenodo:{int(record_id)}")


def _infer_variant(name: str) -> str:
    # GWTC-2.1 / GWTC-3 convention.
    m = re.search(r"_PEDataRelease_(?P<variant>[^.]+)\.(h5|hdf5)$", name)
    if m:
        return str(m.group("variant"))
    # GWTC-4.0 convention.
    if name.endswith("-combined_PEDataRelease.hdf5"):
        return "combined"
    return "unknown"


def build_gwtc_pe_index(
    *,
    base_dir: str | Path,
    record_ids: list[int] | None = None,
) -> dict[str, list[GWTCPeFile]]:
    """Scan downloaded GWTC posterior-sample releases and return {event: [files...]}.

    This index is intended to be lightweight and filename-driven (no HDF5 reads) so it can be
    generated quickly even on headless systems.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    if record_ids is None:
        record_ids = []
        for p in sorted(base_dir.iterdir()) if base_dir.exists() else []:
            if p.is_dir() and p.name.isdigit():
                record_ids.append(int(p.name))
    record_ids = [int(x) for x in record_ids]
    if not record_ids:
        raise FileNotFoundError(f"No Zenodo record directories found under {base_dir}")

    out: dict[str, list[GWTCPeFile]] = {}
    for rec in record_ids:
        rec_dir = base_dir / str(rec)
        if not rec_dir.exists():
            continue
        release = _infer_release(rec)
        for p in sorted(rec_dir.glob("*")):
            if not p.is_file():
                continue
            if p.suffix not in (".h5", ".hdf5"):
                continue
            if "PEDataRelease" not in p.name:
                continue
            m = _EVENT_RE.search(p.name)
            if not m:
                continue
            event = str(m.group(1))
            variant = _infer_variant(p.name)
            out.setdefault(event, []).append(
                GWTCPeFile(
                    event=event,
                    record_id=int(rec),
                    release=release,
                    variant=variant,
                    path=str(p.relative_to(base_dir)),
                )
            )

    # Deterministic ordering.
    for ev in list(out.keys()):
        out[ev] = sorted(out[ev], key=lambda r: (r.release, r.variant, r.path))
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def summarize_gwtc_pe_index(index: dict[str, list[GWTCPeFile]]) -> dict[str, Any]:
    """Return a small summary dict (counts by release and variant)."""
    by_release: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    for files in index.values():
        for f in files:
            by_release[f.release] = by_release.get(f.release, 0) + 1
            by_variant[f.variant] = by_variant.get(f.variant, 0) + 1
    return {
        "n_events": int(len(index)),
        "n_files": int(sum(len(v) for v in index.values())),
        "files_by_release": dict(sorted(by_release.items())),
        "files_by_variant": dict(sorted(by_variant.items())),
    }
