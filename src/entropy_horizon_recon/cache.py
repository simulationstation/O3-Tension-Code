from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pooch


@dataclass(frozen=True)
class DataPaths:
    repo_root: Path | str

    def __post_init__(self) -> None:
        # Allow callers/tests to pass a string repo_root.
        object.__setattr__(self, "repo_root", Path(self.repo_root))

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def pooch_cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def make_pooch(
    *,
    cache_dir: Path,
    base_url: str,
    registry: dict[str, str],
    retry_if_failed: int = 3,
) -> pooch.Pooch:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return pooch.create(
        path=cache_dir,
        base_url=base_url,
        registry=registry,
        retry_if_failed=retry_if_failed,
    )
