"""The gorilla image dataset and the identity-class -> task-label presets."""

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.metadata import DEFAULT_CLASS_MAP, IDENTITY_RE


# All known identity classes, in maturity order. The order fixes the label
# indices a run gets when it keeps every class (ALL_CLASSES_LABEL_MAP).
IDENTITY_CLASS_ORDER: tuple[str, ...] = (
    "Silverback",
    "Blackback",
    "Adult Female",
    "Adolescent & Juvenile",
    "Infant",
)
IDENTITY_CLASSES = set(IDENTITY_CLASS_ORDER)

#: Male/Female are the jaguar and polar-bear caches, whose identity class is the sex.
LABEL_PRESETS: dict[str, dict[str, list[str]]] = {
    "sex": {
        "Male": ["Silverback", "Blackback", "Male"],
        "Female": ["Adult Female", "Female"],
    },
    "age": {
        "Adult": ["Silverback", "Blackback", "Adult Female"],
        "Non-adult": ["Infant", "Adolescent & Juvenile"],
    },
}

# Every CSV-mapped image under its own identity class, so one activation run
# serves every preset.
ALL_CLASSES_LABEL_MAP: dict[str, list[str]] = {c: [c] for c in IDENTITY_CLASS_ORDER}

def _gorilla_id_from_filename(filename: str) -> str | None:
    """Gorilla ID from a filename, or None -- the image scan skips what it cannot
    place, where src.data.metadata.identity_of raises instead."""
    m = IDENTITY_RE.match(filename)
    return m.group(1) if m else None


def load_id_to_identity(csv_path: str | Path) -> dict[str, str]:
    """gorilla_id -> identity class, keyed by uppercase ID."""
    df = pd.read_csv(csv_path, dtype=str)
    ids = df["gorilla_id"].str.strip().str.upper()
    classes = df["age_sex_class"].str.strip()

    unknown = sorted(set(classes.dropna()) - IDENTITY_CLASSES)
    if unknown:
        raise ValueError(
            f"{csv_path}: unrecognised identity classes {unknown}; "
            f"expected a subset of {sorted(IDENTITY_CLASSES)}"
        )

    return dict(zip(ids, classes))


def concept_of_identity(preset: str) -> dict[str, str | None]:
    """gorilla_id (upper) -> its class under `preset`, None where the preset has none."""
    concept_of_class = {
        cls: concept for concept, classes in LABEL_PRESETS[preset].items() for cls in classes
    }
    return {
        gid: concept_of_class.get(cls)
        for gid, cls in load_id_to_identity(DEFAULT_CLASS_MAP).items()
    }


def labels_for_preset(
    identity_classes: Sequence[str], preset: str
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(rows, y, label_names) for one preset. A cache holds every identity class,
    so `rows` indexes the subset the preset covers."""
    label_map = LABEL_PRESETS[preset]
    class_to_label = {
        ic: label_idx
        for label_idx, classes in enumerate(label_map.values())
        for ic in classes
    }
    kept = [
        (row, class_to_label[ic])
        for row, ic in enumerate(identity_classes)
        if ic in class_to_label
    ]
    rows = torch.tensor([row for row, _ in kept], dtype=torch.long)
    y = torch.tensor([label for _, label in kept], dtype=torch.long)
    return rows, y, list(label_map)


def cache_labels(payload: dict, preset: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(rows, y, label_names) from a cache payload."""
    return labels_for_preset(payload["identity_classes"], preset)


class GorillaDataset(Dataset):
    """Gorilla images labelled by mapping .csv"""

    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        label_map: dict[str, list[str]],
        transform=None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.label_names: list[str] = list(label_map.keys())
        self._class_to_label: dict[str, int] = {}
        self._class_to_label = {
            ic: label_idx
            for label_idx, classes in enumerate(label_map.values())
            for ic in classes
        }

        self._id_to_identity: dict[str, str] = load_id_to_identity(csv_path)

        self._samples: list[tuple[Path, int]] = []  # (image_path, label_index)
        self._skipped: list[str] = []

        for img_path in sorted(self.image_dir.glob("*.png")):
            gorilla_id = _gorilla_id_from_filename(img_path.name)
            if gorilla_id is None:
                self._skipped.append(img_path.name)
                continue

            gorilla_id_upper = gorilla_id.upper()
            identity_class = self._id_to_identity.get(gorilla_id_upper)
            if identity_class is None:
                self._skipped.append(img_path.name)
                continue

            label_idx = self._class_to_label.get(identity_class)
            if label_idx is None:
                continue  # class outside this label_map

            self._samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label_idx = self._samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx, str(img_path)

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.label_names}
        for _, label_idx in self._samples:
            counts[self.label_names[label_idx]] += 1
        return counts
