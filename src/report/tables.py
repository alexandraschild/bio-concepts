"""Render row dicts as CSV."""

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class Column:
    """One table column. ``precision`` is carried for the plots that share these
    specs; the CSV always writes full precision."""

    key: str
    header: str
    precision: int = 3
    align: str = "r"


def columns(*specs: str | tuple | Column) -> list[Column]:
    """Terse column construction: ``columns("layer", ("auroc", "AUROC"))``."""
    out: list[Column] = []
    for spec in specs:
        if isinstance(spec, Column):
            out.append(spec)
        elif isinstance(spec, str):
            out.append(Column(spec, spec.replace("_", " "), align="l"))
        else:
            out.append(Column(*spec))
    return out


def render(rows: Sequence[dict[str, Any]], cols: Sequence[Column]) -> str:

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([col.key for col in cols])
    for row in rows:
        writer.writerow([_value(row.get(col.key)) for col in cols])
    return buf.getvalue()


def _value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write(
    rows: Sequence[dict[str, Any]],
    cols: Sequence[Column],
    out_stem: Path | str,
    *,
    title: str | None = None,
) -> list[Path]:
    """Write the table as one CSV. *title* names what it holds for the caller's log
    line; it does not appear in the file."""
    path = Path(out_stem).with_name(Path(out_stem).name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(rows, cols), encoding="utf-8")
    return [path]
