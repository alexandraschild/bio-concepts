"""Per-sample metadata parsed from image paths.

AP00_R066_20221118_110_0_1172946.png -> identity, encounter (identity+video+
date), image. Age-sex classes come from data/gorilla_mappings_new.csv. The jaguar
and polar-bear caches store their identities instead, plus any coarser grouping
(the polar bears' zoo).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

# Identity is the leading two-letter + two-digit tag; encounter adds video+date.
IDENTITY_RE = re.compile(r"^([A-Za-z]{2}\d{2})_")
_ENCOUNTER_RE = re.compile(r"^([A-Za-z]{2}\d{2})_([A-Za-z0-9]+)_(\d{8})_")

DEFAULT_CLASS_MAP = Path(__file__).resolve().parents[2] / "data" / "gorilla_mappings_new.csv"

#: Canonical order for the age-sex classes, used for every table and figure axis.
AGE_SEX_ORDER = ("Silverback", "Blackback", "Adult Female")

GROUP_LEVELS = ("image", "encounter", "identity")


def identity_of(path: str) -> str:
    """Identity tag for one image path, e.g. ``AP00``."""
    match = IDENTITY_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"no gorilla ID in filename: {path}")
    return match.group(1).upper()


def encounter_of(path: str) -> str:
    """Encounter key, e.g. AP00_R066_20221118. Falls back to the identity so two
    animals are never silently merged."""
    match = _ENCOUNTER_RE.match(Path(path).name)
    if match is None:
        return identity_of(path)
    return f"{match.group(1).upper()}_{match.group(2)}_{match.group(3)}"


def load_age_sex_map(path: Path | str | None = None) -> dict[str, str]:
    """identity tag -> age-sex class, from the gorilla mappings CSV."""
    csv_path = Path(path) if path is not None else DEFAULT_CLASS_MAP
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "gorilla_id" not in rows[0] or "age_sex_class" not in rows[0]:
        raise ValueError(f"Expected columns gorilla_id,age_sex_class in {csv_path}")
    return {r["gorilla_id"].upper(): r["age_sex_class"] for r in rows}


def stored_columns(payload: Mapping[str, object], rows: Sequence[int]) -> dict[str, object]:
    """`build_sample_meta` keywords for these cache rows: the identities and groups a
    jaguar or polar-bear cache stores. Empty for a gorilla cache."""
    if "columns" not in payload:
        return {}
    return {
        "identity": [payload["columns"]["identity"][i] for i in rows],
        "groups": {name: [col[i] for i in rows] for name, col in payload.get("groups", {}).items()},
    }


@dataclass(frozen=True)
class SampleMeta:
    """Metadata for one aligned block of samples (train rows, test rows, ...)."""

    index: np.ndarray  # (n,) global index into the activation payload
    path: np.ndarray  # (n,) str
    label: np.ndarray  # (n,) int, 0/1 in probe orientation (1 = positive class)
    identity: np.ndarray  # (n,) str
    encounter: np.ndarray  # (n,) str
    age_sex: np.ndarray  # (n,) str, "" when the identity is not in the mapping
    #: Levels coarser than identity, e.g. the polar bears' zoo.
    groups: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def group_keys(self, level: str) -> np.ndarray:
        if level == "image":
            return self.index.astype(str)
        if level == "encounter":
            return self.encounter
        if level == "identity":
            return self.identity
        return self.groups[level]

    def select(self, rows: Sequence[int] | np.ndarray) -> "SampleMeta":
        rows = np.asarray(rows, dtype=np.int64)
        return SampleMeta(
            index=self.index[rows],
            path=self.path[rows],
            label=self.label[rows],
            identity=self.identity[rows],
            encounter=self.encounter[rows],
            age_sex=self.age_sex[rows],
            groups={name: col[rows] for name, col in self.groups.items()},
        )

    def as_columns(self) -> dict[str, np.ndarray]:
        return {
            "index": self.index,
            "path": self.path,
            "label": self.label,
            "identity": self.identity,
            "encounter": self.encounter,
            "age_sex": self.age_sex,
        }


def build_sample_meta(
    indices: Iterable[int],
    paths: Sequence[str],
    labels: Iterable[int],
    *,
    age_sex_map: dict[str, str] | None = None,
    identity: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> SampleMeta:
    """Identity, encounter and age-sex class follow the gorilla filenames unless
    `identity` is given; then the encounter is the identity and the class empty."""
    idx = np.asarray(list(indices), dtype=np.int64)
    path_arr = np.asarray(list(paths), dtype=object)
    label_arr = np.asarray(list(labels), dtype=np.int64)
    if identity is None:
        class_map = load_age_sex_map() if age_sex_map is None else age_sex_map
        identity_arr = np.asarray([identity_of(p) for p in path_arr], dtype=object)
        encounter_arr = np.asarray([encounter_of(p) for p in path_arr], dtype=object)
        age_sex = np.asarray([class_map.get(g, "") for g in identity_arr], dtype=object)
    else:
        identity_arr = encounter_arr = np.asarray(list(identity), dtype=object)
        age_sex = np.full(idx.shape[0], "", dtype=object)
    return SampleMeta(
        index=idx,
        path=path_arr,
        label=label_arr,
        identity=identity_arr,
        encounter=encounter_arr,
        age_sex=age_sex,
        groups={name: np.asarray(list(col), dtype=object) for name, col in (groups or {}).items()},
    )


#: Elements per temporary block when reducing a wide matrix (~130 MB in float64).
_REDUCE_BLOCK = 1 << 24


@dataclass(frozen=True)
class GroupIndex:
    """Reusable keys -> columns grouping: sort once, then each aggregation is one
    np.add.reduceat over contiguous runs."""

    keys: np.ndarray  # (n_groups,) unique keys, sorted
    order: np.ndarray  # (n,) column permutation grouping equal keys together
    starts: np.ndarray  # (n_groups,) where each run begins in the permuted order

    @classmethod
    def build(cls, keys: np.ndarray) -> "GroupIndex":
        uniq, inverse = np.unique(keys, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        order = np.argsort(inverse, kind="stable")
        # Every unique key came from `keys`, so every run is non-empty and
        # searchsorted gives exactly one start per group.
        starts = np.searchsorted(inverse[order], np.arange(uniq.shape[0]))
        return cls(keys=uniq, order=order, starts=starts)

    @property
    def n_groups(self) -> int:
        return int(self.keys.shape[0])

    def sum(self, values: np.ndarray) -> np.ndarray:
        """Sum values (..., n) over the last axis within each group, one column
        per unique key in sorted order."""
        values = np.asarray(values)
        flat = values.reshape(-1, values.shape[-1]) if values.ndim else values.reshape(1, 1)
        n_rows = flat.shape[0]
        out = np.empty((n_rows, self.n_groups), dtype=np.float64)
        # Chunk over rows so the permuted float64 temporary stays bounded; the
        # caller's matrix may be a gigabyte in float32 and would double on cast.
        block = max(1, _REDUCE_BLOCK // max(flat.shape[-1], 1))
        for start in range(0, n_rows, block):
            stop = min(start + block, n_rows)
            permuted = flat[start:stop][:, self.order].astype(np.float64, copy=False)
            out[start:stop] = np.add.reduceat(permuted, self.starts, axis=1)

        return out.reshape(values.shape[:-1] + (self.n_groups,))


def group_counts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(keys, return_counts=True)
