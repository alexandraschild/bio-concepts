"""The on-disk attribution cache: one schema for every method.

    <outdir>/<layer>/{config.json, attribution.npz, sources.csv, targets.csv}

Positive attribution means the unit pushes toward the positive class.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.data.metadata import GroupIndex, SampleMeta
from src.probes.cache import git_sha

SCHEMA_VERSION = 1


@dataclass
class AttributionCache:
    """Attributions from one method on one layer, plus their provenance."""

    method: str
    layer: str
    source_meta: SampleMeta
    target_meta: SampleMeta
    target_logit: np.ndarray
    target_pred: np.ndarray
    attribution: np.ndarray | None = None  # (n_test, n_src) toward the positive class
    alpha: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        self._group_index: dict[str, GroupIndex] = {}

    @property
    def n_src(self) -> int:
        return len(self.source_meta)

    @property
    def n_test(self) -> int:
        return len(self.target_meta)

    @property
    def class_names(self) -> tuple[str, str]:
        names = self.config["probe"]["class_names"]
        return (str(names[0]), str(names[-1]))

    @property
    def matrix(self) -> np.ndarray:
        """The ``(n_test, n_src)`` attribution matrix, decompressed on first use."""
        if self.attribution is None:
            with np.load(self.path / "attribution.npz", allow_pickle=False) as z:
                self.attribution = z["attribution"]
        return self.attribution

    def release(self) -> None:
        """Drop the cached matrix. Call between layers to keep peak memory flat."""
        if self.path is not None:
            self.attribution = None
        self._group_index.clear()

    def group_index(self, level: str) -> GroupIndex:
        if level not in self._group_index:
            self._group_index[level] = GroupIndex.build(self.source_meta.group_keys(level))
        return self._group_index[level]

    def aggregate(self, level: str) -> tuple[np.ndarray, np.ndarray]:
        """(keys, (n_test, n_groups)) attribution summed per group."""
        index = self.group_index(level)
        return index.keys, index.sum(self.matrix)

    def iter_aggregated(
        self,
        level: str,
        chunk_rows: int = 512,
        rows: np.ndarray | None = None,
    ) -> "Iterator[tuple[np.ndarray, np.ndarray]]":
        """aggregate() streamed in row blocks, for row-wise consumers."""
        index = self.group_index(level)
        matrix = self.matrix
        selected = np.arange(matrix.shape[0]) if rows is None else np.asarray(rows)
        for start in range(0, selected.shape[0], chunk_rows):
            block = matrix[selected[start : start + chunk_rows]]
            yield index.keys, index.sum(block)

    def target_sample(self, max_targets: int | None) -> np.ndarray | None:
        if max_targets is None or max_targets >= self.n_test:
            return None
        picked = np.random.default_rng(0).choice(self.n_test, size=max_targets, replace=False)
        return np.sort(picked)

    def total_per_source(self, absolute: bool = False) -> np.ndarray:
        matrix = self.matrix
        out = np.zeros(matrix.shape[1], dtype=np.float64)
        # Row blocks: np.abs on a gigabyte matrix would otherwise copy it whole.
        block = max(1, (1 << 24) // max(matrix.shape[1], 1))
        for start in range(0, matrix.shape[0], block):
            chunk = matrix[start : start + block]
            out += (np.abs(chunk) if absolute else chunk).sum(axis=0, dtype=np.float64)
        return out

    def source_weight(self, weight: str = "alpha") -> np.ndarray:
        """Per-source weight: count (membership), |alpha|, or |attribution| (the only
    option for methods without alpha)."""
        if weight == "count":
            return np.ones(self.n_src, dtype=np.float64)
        if weight == "alpha":
            if self.alpha is None:
                raise ValueError(f"{self.method} has no alpha; use weight='attribution'.")
            return np.abs(np.asarray(self.alpha, dtype=np.float64))
        if weight == "attribution":
            return self.total_per_source(absolute=True)
        raise ValueError(f"weight must be count/alpha/attribution, got {weight!r}")

    def save(self, outdir: Path | str) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "source_index": self.source_meta.index,
            "source_label": self.source_meta.label,
            "attribution": np.asarray(self.matrix, dtype=np.float32),
            "target_index": self.target_meta.index,
            "target_label": self.target_meta.label,
            "target_logit": np.asarray(self.target_logit, dtype=np.float32),
            "target_pred": np.asarray(self.target_pred, dtype=np.int64),
        }
        if self.alpha is not None:
            arrays["alpha"] = np.asarray(self.alpha, dtype=np.float64)
        np.savez_compressed(outdir / "attribution.npz", **arrays)

        _write_csv(outdir / "sources.csv", self.source_meta, extra={"alpha": self.alpha})
        _write_csv(
            outdir / "targets.csv",
            self.target_meta,
            extra={"logit": self.target_logit, "pred": self.target_pred},
        )

        config = dict(self.config)
        config |= {
            "schema_version": SCHEMA_VERSION,
            "method": self.method,
            "layer": self.layer,
            "n_src": self.n_src,
            "n_test": self.n_test,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
        }
        with (outdir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        self.path = outdir
        return outdir

    @classmethod
    def load(cls, path: Path | str, *, matrix: bool = False) -> "AttributionCache":
        """Read a cache; matrix=False (default) defers the dense array to first access."""
        path = Path(path)
        with (path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: cache schema v{config.get('schema_version')} but this code "
                f"reads v{SCHEMA_VERSION}. Re-run src.attribution.attribute for this cache."
            )
        with np.load(path / "attribution.npz", allow_pickle=False) as z:
            source_meta = _read_csv(path / "sources.csv", z["source_index"], z["source_label"])
            target_meta = _read_csv(path / "targets.csv", z["target_index"], z["target_label"])
            return cls(
                method=config["method"],
                layer=config["layer"],
                attribution=z["attribution"] if matrix else None,
                source_meta=source_meta,
                target_meta=target_meta,
                target_logit=z["target_logit"],
                target_pred=z["target_pred"],
                alpha=z["alpha"] if "alpha" in z.files else None,
                config=config,
                path=path,
            )


def _write_csv(path: Path, meta: SampleMeta, extra: dict[str, np.ndarray | None]) -> None:
    cols = meta.as_columns()
    extra_cols = {k: v for k, v in extra.items() if v is not None}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(cols) + list(extra_cols))
        for i in range(len(meta)):
            writer.writerow(
                [cols[k][i] for k in cols] + [extra_cols[k][i] for k in extra_cols]
            )


def _read_csv(path: Path, index: np.ndarray, label: np.ndarray) -> SampleMeta:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return SampleMeta(
        index=np.asarray(index, dtype=np.int64),
        path=np.asarray([r["path"] for r in rows], dtype=object),
        label=np.asarray(label, dtype=np.int64),
        identity=np.asarray([r["identity"] for r in rows], dtype=object),
        encounter=np.asarray([r["encounter"] for r in rows], dtype=object),
        age_sex=np.asarray([r["age_sex"] for r in rows], dtype=object),
    )


def find_caches(root: Path | str) -> list[Path]:
    return [
        cfg.parent
        for cfg in sorted(Path(root).rglob("config.json"))
        if (cfg.parent / "attribution.npz").exists()
    ]
