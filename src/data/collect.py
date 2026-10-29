"""Run a DINOv3 backbone over a dataset's images and cache each block's output."""

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any
import uuid

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.features import log
from src.data.dataset import ALL_CLASSES_LABEL_MAP, LABEL_PRESETS, GorillaDataset, labels_for_preset


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    """The flags every collector shares: backbone, layers, batching and output."""
    parser.add_argument("--model-path", required=True,
                        help="Backbone checkpoint (.pth), a raw state dict.")
    parser.add_argument("--timm-model", required=True,
                        help="timm architecture to load the checkpoint into, "
                             "e.g. vit_base_patch16_dinov3.")
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, nargs="+", metavar="LAYER_IDX", default=None,
                             help="Block indices to hook as 'blocks.<index>'.")
    layer_group.add_argument("--all-layers", action="store_true",
                             help="Capture every block output.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=25,
                        help="Print progress every N batches (default: 25)")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"],
                        default="float32",
                        help="Storage dtype for the activation chunks (default: float32).")
    parser.add_argument("--activation-chunk-dir", default=None,
                        help="Where the activation chunks go. "
                             "Default: <output_stem>_chunks beside --output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect layer activations and labels from GorillaDataset."
    )
    parser.add_argument("--image-dir", required=True, help="Directory with .png image files")
    parser.add_argument("--mapping", default="data/gorilla_mappings_new.csv",
                        help="Gorilla identity CSV (default: data/gorilla_mappings_new.csv)")
    add_collect_args(parser)
    return parser.parse_args()


# The reid export wraps the backbone; unstripped, the checkpoint loads non-strict
# and the model silently runs on random init.
_WRAPPER_PREFIX = "model."

# Feature extraction needs no classifier head, and the reid head is installed
# separately by src.steering.steer.
_HEAD_KEYS = {"head.weight", "head.bias", "classifier.weight", "classifier.bias"}


def load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """The backbone weights from a raw state-dict checkpoint, unwrapped and headless."""
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if any(k.startswith(_WRAPPER_PREFIX) for k in state_dict):
        state_dict = {k.removeprefix(_WRAPPER_PREFIX): v for k, v in state_dict.items()}
    return {k: v for k, v in state_dict.items() if k not in _HEAD_KEYS}


def _load_local_model(model_path: str, device: str, timm_model: str) -> nn.Module:

    log(f"Loading {timm_model} from checkpoint: {model_path}")
    model = timm.create_model(timm_model, pretrained=False)
    missing, unexpected = model.load_state_dict(load_state_dict(model_path), strict=False)
    if missing:
        raise ValueError(f"{model_path} leaves {len(missing)} weights unset: {missing[:10]}")
    if unexpected:
        log(f"Ignored {len(unexpected)} non-backbone keys, e.g. {unexpected[:5]}")

    model.to(device)
    model.eval()
    log(f"Model ready on device: {device}")
    return model


def _collate_batch(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(list(images), dim=0), torch.tensor(labels, dtype=torch.long), list(paths)


def _select_layer_names(
    model: nn.Module,
    layer_indices: list[int] | None,
    all_layers: bool,
) -> list[str]:
    available = sorted(
        int(m.group(1))
        for name, _ in model.named_modules()
        if (m := re.fullmatch(r"blocks\.(\d+)", name))
    )
    if not available:
        raise ValueError("This model has no 'blocks.<index>' modules to hook.")
    if all_layers:
        return [f"blocks.{idx}" for idx in available]

    requested = list(dict.fromkeys(layer_indices))
    missing = [idx for idx in requested if idx not in available]
    if missing:
        raise ValueError(f"No blocks {missing}. Available block indices: {available}")
    return [f"blocks.{idx}" for idx in requested]


def _prepare_chunk_dir(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).resolve()
    if args.activation_chunk_dir is None:
        base_dir = output_path.parent / f"{output_path.stem}_chunks"
    else:
        raw_dir = Path(args.activation_chunk_dir)
        base_dir = raw_dir.resolve() if raw_dir.is_absolute() else (Path.cwd() / raw_dir).resolve()

    run_dir = base_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _make_hooks(
    model: nn.Module,
    layer_names: list[str],
    *,
    activation_dtype: torch.dtype,
    layer_chunk_dirs: dict[str, Path],
    chunk_paths: dict[str, list[str]],
    chunk_sizes: dict[str, list[int]],
):
    name_to_module = dict(model.named_modules())
    handles = []
    chunk_counters: dict[str, int] = {name: 0 for name in layer_names}

    for layer_name in layer_names:
        module = name_to_module[layer_name]

        def hook_fn(_module, _inputs, output, ln=layer_name):
            chunk = output.detach().to(device="cpu", dtype=activation_dtype)

            chunk_idx = chunk_counters[ln]
            chunk_counters[ln] += 1
            chunk_path = layer_chunk_dirs[ln] / f"chunk_{chunk_idx:06d}.pt"
            torch.save(chunk, chunk_path)
            chunk_paths[ln].append(str(chunk_path))
            chunk_sizes[ln].append(int(chunk.shape[0]) if chunk.ndim >= 1 else 1)

        handles.append(module.register_forward_hook(hook_fn))

    return handles


def collect_activations(args: argparse.Namespace, dataset: Dataset, source: str) -> dict[str, Any]:
    """Hook every requested layer over `dataset` and return the chunk manifest. The
    dataset yields (image, identity-class index, path) and has label_names."""
    log("Starting activation collection")
    activation_dtype = getattr(torch, args.activation_dtype)
    model = _load_local_model(args.model_path, args.device, args.timm_model)
    distribution = ", ".join(
        f"{name}={count}" for name, count in dataset.class_distribution().items()
    )
    log(f"Dataset loaded with {len(dataset)} samples across all identity classes")
    log(f"Identity classes: {distribution}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering. Check image dir, CSV, and preset.")

    layer_names = _select_layer_names(model, args.layer, args.all_layers)
    preview_layers = layer_names[:5]
    preview = ", ".join(preview_layers)
    suffix = "" if len(layer_names) <= 5 else f", ... ({len(layer_names)} total)"
    log(f"Hooking layers: {preview}{suffix}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )
    log(
        f"Dataloader ready: {len(loader)} batches (batch_size={args.batch_size}, num_workers={args.num_workers})"
    )
    log(f"Activation chunks stored as {args.activation_dtype}")
    chunk_dir = _prepare_chunk_dir(args)
    log(f"Writing activation chunks to: {chunk_dir}")

    chunk_paths: dict[str, list[str]] = {name: [] for name in layer_names}
    chunk_sizes: dict[str, list[int]] = {name: [] for name in layer_names}
    layer_chunk_dirs: dict[str, Path] = {}
    for layer_name in layer_names:
        layer_dir = chunk_dir / re.sub(r"[^0-9A-Za-z_.-]", "_", layer_name)
        layer_dir.mkdir(parents=True, exist_ok=False)
        layer_chunk_dirs[layer_name] = layer_dir

    class_indices: list[torch.Tensor] = []
    image_paths: list[str] = []

    handles = _make_hooks(
        model,
        layer_names,
        activation_dtype=activation_dtype,
        layer_chunk_dirs=layer_chunk_dirs,
        chunk_paths=chunk_paths,
        chunk_sizes=chunk_sizes,
    )
    log("Forward pass started")

    try:
        for batch_idx, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
            with torch.no_grad():
                model(images.to(args.device))

            class_indices.append(batch_labels.cpu())
            image_paths.extend(batch_paths)

            if batch_idx == 1 or batch_idx % max(1, args.log_every) == 0 or batch_idx == len(loader):
                seen = len(image_paths)
                log(f"Processed batch {batch_idx}/{len(loader)} (samples seen: {seen})")
    finally:
        for h in handles:
            h.remove()
        log("Hooks removed")

    output_dict: dict[str, Any] = {
        "meta": {
            # model_path/timm_model so the steering stage can rebuild this backbone
            # without being told which one it was.
            "model_path": args.model_path,
            "timm_model": args.timm_model,
            "layer_names": layer_names,
            "num_samples": len(dataset),
            "image_dir": source,
            "image_size": args.image_size,
            "activation_dtype": args.activation_dtype,
            "activation_chunk_dir": str(chunk_dir),
        },
        # Identity class per row, not a task label: the preset is applied downstream.
        "identity_classes": [
            dataset.label_names[idx]
            for idx in torch.cat(class_indices, dim=0).tolist()
        ],
        "paths": image_paths,
        "activations": {},
    }

    output_parent = Path(args.output).resolve().parent
    for layer_name in layer_names:
        layer_paths: list[str] = []
        for path_str in chunk_paths[layer_name]:
            path_obj = Path(path_str)
            try:
                rel = path_obj.relative_to(output_parent)
                layer_paths.append(str(rel))
            except ValueError:
                layer_paths.append(str(path_obj))

        output_dict["activations"][layer_name] = {
            "storage": "disk_chunks",
            "chunk_paths": layer_paths,
            "chunk_sizes": chunk_sizes[layer_name],
            "dtype": args.activation_dtype,
        }

    log("Activation collection finished")

    return output_dict


def square_resize(image_size: int) -> transforms.Compose:
    """The preprocessing of the paper: squash to a square, no normalization."""
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def main() -> None:
    args = parse_args()
    dataset = GorillaDataset(
        image_dir=args.image_dir,
        csv_path=args.mapping,
        label_map=ALL_CLASSES_LABEL_MAP,
        transform=square_resize(args.image_size),
    )
    result = collect_activations(args, dataset, str(Path(args.image_dir)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)

    print(f"Saved activations to: {out_path.resolve()}")
    print(f"Samples: {result['meta']['num_samples']}")
    print(f"Layers: {len(result['meta']['layer_names'])}")
    for preset in LABEL_PRESETS:
        rows, _, label_names = labels_for_preset(result["identity_classes"], preset)
        print(f"Preset '{preset}': {rows.numel()} rows, labels {label_names}")


if __name__ == "__main__":
    main()
