"""Steer a reid backbone along a probe direction and save the 256-d embeddings."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.features import log
from src.data.collect import _load_local_model, load_state_dict
from src.data.dataset import concept_of_identity
from src.data.metadata import DEFAULT_CLASS_MAP, identity_of
from src.paths import add_run_arg, run_of, split_path_for
from src.probes.discovery import probes_in_dir


#: The balanced subset every sweep steers: up to PER_ID images of each held-out
#: identity, NUM_SAMPLES in all.
PER_ID = 40
NUM_SAMPLES = 2000
SUBSET_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steer a reid backbone along probe directions; save 256-d embeddings.")
    add_run_arg(parser)
    parser.add_argument("--probes", default=None,
                        help="Directory of probe_<layer>.pt files (default: <run>/probes). "
                             "Each probe is steered at its own layer.")
    parser.add_argument("--out", default=None,
                        help="One steered_layer<L>.pt per probe "
                             "(default: <run>/steering/embeddings).")
    parser.add_argument("--alpha", type=float, nargs="+", required=True,
                        help="Steering scales; act += alpha * direction at every token. "
                             "The alpha=0 baseline is always added.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.run and not (args.probes and args.out):
        parser.error("--run, or --probes together with --out, is required")
    return args


# ---------------------------------------------------------------------------
# Model + head loading
# ---------------------------------------------------------------------------


def _install_head(model: nn.Module, model_path: str, device: str) -> nn.Linear:
    """Install the reid head (768 -> 256) that the feature-extraction loader drops."""
    sd = load_state_dict(model_path)

    w_key = next((k for k in sd if k.endswith("embedding_layer.weight")), None)
    if w_key is None:
        raise ValueError(
            f"No 'embedding_layer.weight' found in checkpoint {model_path}; "
            "cannot build the 256-d head."
        )
    b_key = w_key[: -len("weight")] + "bias"
    b_key = b_key if b_key in sd else None

    weight = sd[w_key].float()
    out_dim, in_dim = weight.shape
    has_bias = b_key is not None
    head = nn.Linear(in_dim, out_dim, bias=has_bias)
    with torch.no_grad():
        head.weight.copy_(weight)
        if has_bias:
            head.bias.copy_(sd[b_key].float())

    model.head = head.to(device)
    log(f"Installed head from '{w_key}': Linear({in_dim} -> {out_dim}, bias={has_bias})")
    return head


def load_model_with_head(model_path: str, timm_model: str | None, device: str) -> nn.Module:
    model = _load_local_model(model_path, device, timm_model)
    _install_head(model, model_path, device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Probe direction
# ---------------------------------------------------------------------------


def extract_direction(payload: dict, d_model: int) -> torch.Tensor:
    """Unit-normalized probe direction (points toward the probe's positive class)."""
    direction = payload["probe_tensor"].reshape(-1).float()
    if direction.numel() == d_model + 1:
        # logistic_regression stores [direction..., bias]; drop the trailing bias.
        direction = direction[:d_model]
    norm = float(direction.norm(p=2))
    log(f"Probe direction: dim={direction.numel()}, L2={norm:.4f} (unit-normalized)")
    if norm == 0.0:
        raise ValueError("Probe direction has zero norm; cannot normalize.")
    return direction / norm


# ---------------------------------------------------------------------------
# Activation cache access
# ---------------------------------------------------------------------------


def _resolve_chunk_path(path_str: str, source_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (source_dir / path).resolve()
    return path


def load_layer_activations(
    layer_entry: Any, indices: list[int], source_dir: Path
) -> torch.Tensor:
    """Cached block outputs for `indices`, shape (len(indices), T, d), float32."""
    chunk_paths = layer_entry["chunk_paths"]
    starts = np.concatenate([[0], np.cumsum(layer_entry["chunk_sizes"])])

    idx_arr = np.asarray(indices)
    chunk_of = np.searchsorted(starts, idx_arr, side="right") - 1

    out: torch.Tensor | None = None
    # Load each needed chunk once; scatter its rows into the output.
    for chunk_i in np.unique(chunk_of):
        chunk_path = _resolve_chunk_path(chunk_paths[chunk_i], source_dir)
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False).float()

        sel = np.where(chunk_of == chunk_i)[0]  # positions in `indices`
        rows = chunk[torch.as_tensor(idx_arr[sel] - starts[chunk_i])]
        if out is None:
            out = torch.empty((len(indices), *rows.shape[1:]), dtype=torch.float32)
        out[torch.as_tensor(sel)] = rows
    return out


def load_eval_split_ids(split_file: str, eval_splits: list[str]) -> set[str]:
    """Union of gorilla-ID tags for the given split-file keys (uppercased)."""
    with Path(split_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {t.upper() for key in eval_splits for t in data[key]}


def select_balanced_subset(
    paths: list[str], allowed_ids: set[str]
) -> tuple[list[int], list[str]]:
    """Balanced-per-identity subset. Returns (global indices, gorilla-ID labels)."""
    rng = random.Random(SUBSET_SEED)

    id_to_indices: dict[str, list[int]] = {}
    for idx, path in enumerate(paths):
        gid = identity_of(path)
        if gid in allowed_ids:
            id_to_indices.setdefault(gid, []).append(idx)

    kept: list[int] = []
    for gid in sorted(id_to_indices):
        pool = id_to_indices[gid][:]
        rng.shuffle(pool)
        kept.extend(pool[:PER_ID])

    rng.shuffle(kept)
    kept = sorted(kept[:NUM_SAMPLES])  # cache order, for efficient chunk reads

    labels = [identity_of(paths[i]) for i in kept]
    log(f"Subset: {len(kept)} samples across {len(set(labels))} gorilla IDs (per_id={PER_ID}).")
    return kept, labels


# ---------------------------------------------------------------------------
# Steered forward
# ---------------------------------------------------------------------------


def compute_rope(model: nn.Module, num_tokens: int) -> torch.Tensor | None:
    """RoPE embedding for the patch grid inferred from the cached token count."""
    if getattr(model, "rope", None) is None:
        return None
    n_patch = num_tokens - model.num_prefix_tokens
    side = int(round(math.sqrt(n_patch)))
    if side * side != n_patch:
        raise ValueError(f"Non-square patch grid: {n_patch} patch tokens ({num_tokens} total).")
    return model.rope.get_embed(shape=(side, side))


@torch.no_grad()
def steered_embeddings(
    model: nn.Module,
    acts: torch.Tensor,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
    signs: torch.Tensor,
    rope: torch.Tensor | None,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Steer block `layer_idx`, resume from layer_idx+1, return the 256-d embeddings.
    `signs` is -1 for the probe's positive class and +1 for the other, so every
    sample is pushed toward its opposite concept class."""
    n = acts.shape[0]
    dir_dev = direction.to(device)
    signs_dev = signs.to(device)
    rope_dev = rope.to(device) if rope is not None else None
    blocks = model.blocks[layer_idx + 1:]

    outputs: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        x = acts[start:start + batch_size].to(device)
        if alpha != 0.0:
            coeff = (alpha * signs_dev[start:start + x.shape[0]]).view(-1, 1, 1)
            x = x + coeff * dir_dev  # per-sample signed add over all token positions
        for blk in blocks:
            x = blk(x, rope=rope_dev)
        x = model.norm(x)
        emb = model.forward_head(x)  # pool (patch tokens) -> fc_norm -> head
        outputs.append(emb.float().cpu())
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Resolving what to steer, from the probes themselves
# ---------------------------------------------------------------------------


@dataclass
class SteerJob:
    """One probe, and everything its own split manifest says about how to steer it."""

    layer_name: str
    layer_idx: int
    probe_path: Path
    activations_path: Path
    split_file: str
    eval_splits: list[str]
    out_path: Path


def resolve_jobs(args: argparse.Namespace) -> list[SteerJob]:
    run = run_of(args)
    probe_dir = Path(args.probes) if args.probes else run.probes
    out_dir = Path(args.out) if args.out else run.steering_embeddings
    jobs: list[SteerJob] = []
    for layer_name, probe_path in probes_in_dir(probe_dir):
        with split_path_for(probe_path).open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        layer_idx = int(layer_name.removeprefix("blocks."))
        jobs.append(SteerJob(
            layer_name=layer_name,
            layer_idx=layer_idx,
            probe_path=probe_path,
            activations_path=Path(manifest["source_activation_file"]),
            split_file=manifest["split"]["split_file"],
            eval_splits=[str(s) for s in manifest["split"]["eval_splits"]],
            out_path=out_dir / f"steered_layer{layer_idx}.pt",
        ))
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class _Shared:
    """Work identical across layers, done once. The subset especially: depth curves are
    only comparable if every layer steers the same images."""

    caches: dict[Path, dict[str, Any]]
    models: dict[tuple[str, str | None], nn.Module]
    subsets: dict[tuple, tuple[list[int], list[str]]]


def _activation_cache(shared: _Shared, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in shared.caches:
        log(f"Loading activation cache: {resolved}")
        shared.caches[resolved] = torch.load(resolved, map_location="cpu", weights_only=False)
    return shared.caches[resolved]


def _backbone(shared: _Shared, model_path: str, timm_model: str | None, device: str) -> nn.Module:
    key = (str(model_path), timm_model)
    if key not in shared.models:
        log(f"Loading backbone: {model_path} ({timm_model})")
        shared.models[key] = load_model_with_head(model_path, timm_model, device)
    return shared.models[key]


def _subset(shared: _Shared, job: SteerJob, paths: list[str]) -> tuple[list[int], list[str]]:
    key = (job.activations_path.resolve(), job.split_file, tuple(job.eval_splits))
    if key not in shared.subsets:
        allowed_ids = load_eval_split_ids(job.split_file, job.eval_splits)
        log(f"Restricting subset to {len(allowed_ids)} identities from splits "
            f"{job.eval_splits} of {Path(job.split_file).resolve()}")
        shared.subsets[key] = select_balanced_subset(paths, allowed_ids)
    return shared.subsets[key]


def run_job(job: SteerJob, args: argparse.Namespace, shared: _Shared) -> None:
    """Steer one probe's layer over every alpha and write its embeddings file."""
    cache = _activation_cache(shared, job.activations_path)
    meta = cache["meta"]
    paths = cache["paths"]
    source_dir = job.activations_path.resolve().parent

    cache_layer = f"blocks.{job.layer_idx}"
    indices, labels = _subset(shared, job, paths)
    subset_paths = [paths[i] for i in indices]

    # Cached block-L activations for the subset.
    log(f"Reading cached '{cache_layer}' activations for {len(indices)} samples")
    acts = load_layer_activations(cache["activations"][cache_layer], indices, source_dir)
    n_tokens, d_model = acts.shape[1], acts.shape[2]
    log(f"Activation subset: {tuple(acts.shape)} (N, tokens, d_model)")

    # Model + head.
    model_path = meta["model_path"]
    timm_model = meta["timm_model"]
    model = _backbone(shared, model_path, timm_model, args.device)

    # Probe direction + positive (concept) class + rope.
    probe_payload = torch.load(job.probe_path, map_location="cpu", weights_only=False)
    direction = extract_direction(probe_payload, d_model)
    # The preset decides which class each identity carries, hence the steering sign.
    positive_class = str(probe_payload["positive_class_name"])
    preset = str(probe_payload["preset"])
    rope = compute_rope(model, n_tokens)

    # Every sample is steered TOWARD its opposite concept class, so the class the
    # direction points at gets -1 and the other +1; a class outside the preset gets 0.
    id_to_concept = concept_of_identity(preset)
    sample_concept = [id_to_concept.get(gid) for gid in labels]
    signs = torch.tensor(
        [(-1.0 if c == positive_class else (1.0 if c is not None else 0.0))
         for c in sample_concept],
        dtype=torch.float32,
    )
    n_pos = int((signs == -1.0).sum())
    n_neg = int((signs == 1.0).sum())
    n_unknown = int((signs == 0.0).sum())
    log(
        f"Preset '{preset}', probe positive class '{positive_class}': "
        f"{n_pos} sample(s) subtract (are '{positive_class}'), {n_neg} add"
        + (f", {n_unknown} outside the preset not steered" if n_unknown else "")
    )

    alphas = list(args.alpha) if 0.0 in args.alpha else [0.0, *args.alpha]

    out_dim = model.head.out_features
    embeddings = torch.empty((len(alphas), len(indices), out_dim), dtype=torch.float32)
    for a_i, alpha in enumerate(alphas):
        log(f"Steering alpha={alpha} ({a_i + 1}/{len(alphas)})")
        embeddings[a_i] = steered_embeddings(
            model, acts, job.layer_idx, direction, alpha,
            signs, rope, args.batch_size, args.device,
        )

    # Integer label encoding for downstream convenience.
    label_names = sorted(set(labels))
    label_to_int = {name: i for i, name in enumerate(label_names)}
    label_ids = torch.tensor([label_to_int[l] for l in labels], dtype=torch.long)

    result = {
        "meta": {
            "source_activations": str(job.activations_path.resolve()),
            "probe": str(job.probe_path.resolve()),
            "probe_dir": str(job.probe_path.resolve().parent),
            "probe_layer": job.layer_name,
            "probe_type": probe_payload["probe_type"],
            "model_path": str(model_path),
            "timm_model": timm_model,
            "steering_layer": job.layer_idx,
            "injection": "all_tokens",
            "normalize_direction": True,
            "preset": preset,
            "positive_class": positive_class,
            "sign_convention": "toward_opposite_concept",
            "csv": str(DEFAULT_CLASS_MAP),
            "alphas": alphas,
            "per_id": PER_ID,
            "num_samples": NUM_SAMPLES,
            "seed": SUBSET_SEED,
            "split_file": str(Path(job.split_file).resolve()),
            "eval_splits": job.eval_splits,
            "label_type": "gorilla_id",
            "embedding_dim": out_dim,
            "n_samples": len(indices),
        },
        "alphas": alphas,
        "embeddings": embeddings,       # (n_alphas, n_samples, 256)
        "labels": labels,               # list[str] gorilla IDs
        "label_ids": label_ids,         # (n_samples,) int-encoded
        "label_names": label_names,     # index -> gorilla ID
        "paths": subset_paths,
        "sample_indices": indices,      # global indices into the cache
        "sample_concept": sample_concept,  # per-sample concept class or None
        "signs": signs,                 # per-sample steering sign (-1/+1/0)
    }

    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, job.out_path)

    log(f"Saved steered embeddings to: {job.out_path.resolve()}")
    print(f"Embeddings: {tuple(embeddings.shape)} (alphas, samples, dim)")
    print(f"Alphas: {alphas}")
    print(f"Samples: {len(indices)}  |  Gorilla IDs: {len(label_names)}")


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)

    log(
        f"{len(jobs)} probe(s) to steer from {jobs[0].probe_path.parent.resolve()}: "
        + ", ".join(f"{j.layer_name}->block {j.layer_idx}" for j in jobs)
    )

    shared = _Shared(caches={}, models={}, subsets={})
    for i, job in enumerate(jobs, start=1):
        print()
        log(f"===== [{i}/{len(jobs)}] {job.probe_path.name} @ block {job.layer_idx} =====")
        run_job(job, args, shared)

    print()
    log(f"All {len(jobs)} layer(s) written.")


if __name__ == "__main__":
    main()
