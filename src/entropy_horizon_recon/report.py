from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportPaths:
    out_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.out_dir / "tables"

    @property
    def samples_dir(self) -> Path:
        return self.out_dir / "samples"

    @property
    def report_md(self) -> Path:
        return self.out_dir / "report.md"


def write_markdown_report(*, paths: ReportPaths, markdown: str) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text(markdown, encoding="utf-8")


def format_table(rows: list[list], headers: list[str]) -> str:
    """Minimal markdown table formatter."""
    if len(headers) == 0:
        raise ValueError("headers must be non-empty")
    widths = [len(h) for h in headers]
    for r in rows:
        if len(r) != len(headers):
            raise ValueError("Row length mismatch.")
        widths = [max(w, len(str(v))) for w, v in zip(widths, r, strict=True)]
    def fmt_row(r):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths, strict=True)) + " |"
    out = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)
