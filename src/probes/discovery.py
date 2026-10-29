"""Finding probes, naming them, and reading their directions out."""

from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import torch

_PROBE_RE = re.compile(r"^probe_(.+)\.pt$")


def layer_index(layer: str) -> int:
    """Numeric sort key: 'blocks.7' -> 7. Unnumbered layers sort last."""
    digits = re.findall(r"\d+", str(layer))
    return int(digits[-1]) if digits else 1 << 30


def sort_layers(layers: Iterable[str]) -> list[str]:
    """Depth order -- 'blocks.2' before 'blocks.10', which is not lexicographic."""
    return sorted(set(layers), key=lambda l: (layer_index(l), l))


def normalize_layer(layer: str) -> str:
    """Accept '11', 'blocks.11' or 'probe_blocks.11' for the same layer."""
    text = str(layer).strip()
    if re.fullmatch(r"\d+", text):
        return f"blocks.{text}"
    return text.removeprefix("probe_").removesuffix(".pt")


def probes_in_dir(probe_dir: Path | str, layers: Sequence[str] | None = None) -> list[tuple[str, Path]]:
    """``(layer, probe_path)`` for every probe in a directory, depth-sorted."""
    probe_dir = Path(probe_dir)
    if not probe_dir.is_dir():
        raise SystemExit(f"not a probe directory: {probe_dir}")

    found: list[tuple[str, Path]] = []
    for path in probe_dir.glob("probe_*.pt"):
        match = _PROBE_RE.match(path.name)
        if match is None or path.stem.endswith("_split"):
            continue
        found.append((match.group(1), path))

    if layers:
        wanted = {normalize_layer(l) for l in layers}
        available = {layer for layer, _ in found}
        missing = wanted - available
        if missing:
            raise SystemExit(
                f"No probe for layer(s) {sorted(missing)} in {probe_dir}. "
                f"Available: {', '.join(sort_layers(available))}"
            )
        found = [(layer, p) for layer, p in found if layer in wanted]

    if not found:
        raise SystemExit(f"No probe_<layer>.pt files in {probe_dir}")
    return sorted(found, key=lambda item: (layer_index(item[0]), item[0]))


def fmt_c(C: float) -> str:
    """0.001 not 1e-03, 10 not 10.0 — C values are read, not computed on."""
    text = f"{C:.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


def load_probe_direction(probe_path: Path | str) -> np.ndarray:
    """A probe's weight vector, pulled back to raw activation space so a standardized
    and an unstandardized probe stay comparable."""
    probe_path = Path(probe_path)
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)

    def vec(obj) -> np.ndarray:
        return obj.detach().cpu().numpy().astype(np.float64).reshape(-1)

    w = vec(payload["probe_tensor"])
    # logistic_regression packs [direction..., bias]; scaler_scale has one entry per
    # feature, so it says where the direction ends.
    scale = vec(payload["scaler_scale"])
    return w[: scale.shape[0]] / np.where(scale == 0.0, 1.0, scale)
