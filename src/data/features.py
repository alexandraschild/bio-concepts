"""Pooled feature matrices off an activation cache, plus the shared log helper."""

from datetime import datetime
import hashlib
import os
from pathlib import Path

import numpy as np
import torch


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _cache_key(activations_path: Path, layer: str, reduction: str) -> str:
    stat = activations_path.stat()
    raw = f"{activations_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{layer}|{reduction}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_pooled_features(
    activations_path: Path,
    layer: str,
    reduction: str = "mean",
    *,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """(N, dim) pooled features for one layer, memoised as .npy -- pooling the chunks
    takes minutes and every stage wants the same matrix."""
    activations_path = Path(activations_path)
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(activations_path, layer, reduction)
        cache_path = cache_dir / f"{layer}_{reduction}_{key}.npy"
        if cache_path.exists():
            return np.load(cache_path, mmap_mode="r")

    payload = torch.load(activations_path, map_location="cpu", weights_only=False)
    layer_acts = payload["activations"][layer]
    source_dir = activations_path.parent

    log(f"pooling {layer} from {len(layer_acts['chunk_paths'])} chunks of {activations_path}")
    parts = []
    for chunk_path in layer_acts["chunk_paths"]:
        p = Path(chunk_path)
        if not p.is_absolute():
            p = (source_dir / p).resolve()
        # Token-pool each (B, tokens, dim) chunk down to (B, dim).
        chunk = torch.load(p, map_location="cpu", weights_only=False)
        parts.append(chunk if chunk.ndim == 2 else chunk.mean(dim=1))
    feats = torch.cat(parts, dim=0)

    out = feats.to(torch.float32).numpy()
    if cache_path is not None:
        # Via a handle: np.save(path) would append a second ".npy" and the rename
        # would miss. The pid keeps concurrent jobs pooling the same layer apart.
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.save(f, out)
        tmp.replace(cache_path)
        log(f"cached pooled features -> {cache_path}")
    return out


