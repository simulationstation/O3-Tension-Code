from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def command_str(argv: Sequence[str] | None = None) -> str:
    """Return a shell-escaped command line string."""
    if argv is None:
        argv = sys.argv
    return " ".join(shlex.quote(str(a)) for a in argv)


def git_head_sha(*, repo_root: Path) -> str | None:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode("utf-8").strip()
        return sha if sha else None
    except Exception:
        return None


def git_is_dirty(*, repo_root: Path) -> bool | None:
    """Return whether the git worktree has uncommitted changes, or None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode("utf-8").strip())
    except Exception:
        return None

