"""The on-disk probe-evaluation cache: one schema for holdout and for ID-CV.

    <evals>/<name>/{config.json, metrics.csv, per_class.csv, identities.csv, directions.npz}

Long-format across layers; ``split`` is test/train (holdout) or id_cv.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np

from src.probes.discovery import sort_layers

SCHEMA_VERSION = 1

#: Every table the cache carries, and the file it lives in.
TABLES = {
    "metrics": "metrics.csv",
    "per_class": "per_class.csv",
    "identities": "identities.csv",
}

#: Columns that are always numbers when present, so the reader can restore types
#: a CSV round-trip would otherwise flatten to strings.
_INT_COLUMNS = frozenset({"n", "n_pos", "n_neg", "n_images", "n_identities", "n_encounters",
                          "label"})
_BOOL_COLUMNS = frozenset({"seen"})
_STR_COLUMNS = frozenset({"layer", "split", "unit", "threshold_kind", "threshold_objective",
                          "group_kind", "group", "identity", "age_sex", "sex"})


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


class EvalWriter:
    """Accumulates rows across layers, then writes one cache directory. Tables
    are long-format so the report picks a layer at read time."""

    def __init__(self, outdir: Path | str, config: dict[str, Any]):
        self.outdir = Path(outdir)
        self.config = dict(config)
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
        self.directions: dict[str, np.ndarray] = {}

    def add(self, table: str, layer: str, split: str, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.tables[table].append({"layer": layer, "split": split, **row})

    def add_direction(self, layer: str, direction: np.ndarray) -> None:
        self.directions[layer] = np.asarray(direction, dtype=np.float64).reshape(-1)

    def save(self) -> Path:
        self.outdir.mkdir(parents=True, exist_ok=True)
        for name, filename in TABLES.items():
            _write_csv(self.outdir / filename, self.tables[name])

        if self.directions:
            np.savez_compressed(self.outdir / "directions.npz", **self.directions)

        config = dict(self.config)
        config |= {
            "schema_version": SCHEMA_VERSION,
            "layers": sorted({r["layer"] for r in self.tables["metrics"]}),
            "splits": sorted({r["split"] for r in self.tables["metrics"]}),
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
        }
        with (self.outdir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        return self.outdir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        # An absent table is meaningful (a mode that produces no pairs, say), and
        # an empty file with no header would break the reader's DictReader.
        path.unlink(missing_ok=True)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return f"{float(value):.6g}"
    if value is None:
        return ""
    return value


@dataclass
class ProbeEval:
    """One evaluation cache, read back."""

    path: Path
    config: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return str(self.config["label"])

    @property
    def mode(self) -> str:
        return str(self.config["mode"])

    @property
    def layers(self) -> list[str]:

        return sort_layers({r["layer"] for r in self.tables.get("metrics", [])})

    @property
    def splits(self) -> list[str]:
        return sorted({r["split"] for r in self.tables.get("metrics", [])})

    @property
    def class_names(self) -> tuple[str, str]:
        names = self.config["class_names"]
        return (str(names[0]), str(names[-1]))

    @property
    def class_order(self) -> list[str]:
        return [str(c) for c in self.config["class_order"]]

    def default_split(self) -> str:
        """The split a report means when it does not say: eval before train."""
        for candidate in ("id_cv", "test", "train"):
            if candidate in self.splits:
                return candidate
        return self.splits[0] if self.splits else ""

    def rows(self, table: str, **filters: Any) -> list[dict[str, Any]]:
        """Rows of *table* matching every ``column=value`` filter (None ignores)."""
        out = self.tables.get(table, [])
        for column, wanted in filters.items():
            if wanted is not None:
                out = [r for r in out if r.get(column) == wanted]
        return out

    def value(
        self,
        metric: str,
        layer: str,
        *,
        split: str | None = None,
        unit: str = "image",
        threshold_kind: str = "probe",
    ) -> float | None:
        """One global metric, or None when this cache does not carry it."""
        matches = self.rows(
            "metrics",
            layer=layer,
            split=split or self.default_split(),
            unit=unit,
            threshold_kind=threshold_kind,
        )
        if not matches:
            return None
        return matches[0].get(metric)

    def curve(
        self,
        metric: str,
        *,
        split: str | None = None,
        unit: str = "image",
        threshold_kind: str = "probe",
    ) -> tuple[list[str], list[float | None]]:
        """``(layers, values)`` over depth."""
        layers = self.layers
        return layers, [
            self.value(metric, layer, split=split, unit=unit, threshold_kind=threshold_kind)
            for layer in layers
        ]

    def best_layer(
        self,
        metric: str = "auroc",
        *,
        split: str | None = None,
        unit: str = "image",
        threshold_kind: str = "probe",
    ) -> tuple[str | None, float | None]:
        layers, values = self.curve(
            metric, split=split, unit=unit, threshold_kind=threshold_kind
        )
        scored = [(l, v) for l, v in zip(layers, values) if v is not None and v == v]
        if not scored:
            return None, None
        return max(scored, key=lambda item: item[1])

    def direction(self, layer: str) -> np.ndarray | None:
        """The probe direction in raw activation space, or None if not stored."""
        npz = self.path / "directions.npz"
        if not npz.exists():
            return None
        with np.load(npz) as z:
            return z[layer].copy() if layer in z.files else None


    @classmethod
    def load(cls, path: Path | str) -> "ProbeEval":
        path = Path(path)
        config_path = path / "config.json"
        if not config_path.exists():
            raise SystemExit(f"{path} is not a probe-evaluation cache (no config.json)")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(
                f"{path}: cache schema v{config.get('schema_version')} but this code reads "
                f"v{SCHEMA_VERSION}. Re-run src.probes.evaluate for this cache."
            )
        tables = {name: _read_csv(path / filename) for name, filename in TABLES.items()}
        return cls(path=path, config=config, tables=tables)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: _parse(k, v) for k, v in row.items()} for row in rows]


def _parse(column: str, value: str) -> Any:
    if value == "" or value is None:
        return None
    if column in _STR_COLUMNS:
        return value
    if column in _BOOL_COLUMNS:
        return value.lower() in ("true", "1", "yes")
    if column in _INT_COLUMNS:
        try:
            return int(value)
        except ValueError:
            return value
    try:
        return float(value)
    except ValueError:
        return value


def find_evals(root: Path | str) -> list[Path]:
    root = Path(root)
    if (root / "config.json").exists():
        return [root]
    return sorted(
        cfg.parent for cfg in root.rglob("config.json") if (cfg.parent / "metrics.csv").exists()
    )


def load_evals(paths: Iterable[Path | str]) -> list[ProbeEval]:
    out: list[ProbeEval] = []
    for entry in paths:
        found = find_evals(entry)
        if not found:
            raise SystemExit(f"No probe-evaluation cache at or under {entry}")
        out.extend(ProbeEval.load(p) for p in found)
    return out
