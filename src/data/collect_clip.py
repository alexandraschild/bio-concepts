"""CLIP ablation, not part of the main DINOv3 pipeline: the token mean of every block of
OpenAI's CLIP ViT-B/16 over the gorillas, in the cache format src.probes reads."""

import argparse
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.collect import _collate_batch, square_resize
from src.data.dataset import ALL_CLASSES_LABEL_MAP, GorillaDataset
from src.data.features import log

#: The quickgelu name matters: vit_base_patch16_clip_224.openai runs these weights through GELU.
CLIP_MODEL = "vit_base_patch16_clip_quickgelu_224.openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, help="Directory with .png image files")
    parser.add_argument("--mapping", default="data/gorilla_mappings_new.csv")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # num_classes=0 drops CLIP's 512-d projection head.
    model = timm.create_model(CLIP_MODEL, pretrained=True, num_classes=0).to(args.device).eval()
    # Unlike the DINOv3 caches, normalised with CLIP's own mean/std, which it was trained on.
    cfg = model.pretrained_cfg
    transform = transforms.Compose([square_resize(224), transforms.Normalize(cfg["mean"], cfg["std"])])
    dataset = GorillaDataset(args.image_dir, args.mapping, ALL_CLASSES_LABEL_MAP, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate_batch)

    # Pooled in the hook: the probes only read the token mean, and full tokens are ~150 GB.
    layer_names = [f"blocks.{i}" for i in range(len(model.blocks))]
    pooled: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
    for name, block in zip(layer_names, model.blocks):
        block.register_forward_hook(lambda _m, _i, out, name=name: pooled[name].append(out.mean(dim=1).cpu()))

    log(f"{CLIP_MODEL} over {len(dataset)} images on {args.device}")
    identity_classes: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for i, (images, labels, batch_paths) in enumerate(loader, start=1):
            model.forward_features(images.to(args.device))
            identity_classes.extend(dataset.label_names[l] for l in labels.tolist())
            paths.extend(batch_paths)
            if i % 25 == 0 or i == len(loader):
                log(f"batch {i}/{len(loader)}")

    # One (N, 768) chunk per block; load_pooled_features passes 2-d chunks through as they are.
    out = Path(args.output)
    chunk_dir = out.parent / f"{out.stem}_pooled"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    activations = {}
    for name in layer_names:
        torch.save(torch.cat(pooled[name]), chunk_dir / f"{name}.pt")
        activations[name] = {
            "storage": "disk_chunks",
            "chunk_paths": [f"{chunk_dir.name}/{name}.pt"],
            "chunk_sizes": [len(paths)],
            "dtype": "float32",
        }
    torch.save({
        "meta": {
            "model": CLIP_MODEL,
            "layer_names": layer_names,
            "num_samples": len(paths),
            "image_dir": args.image_dir,
            "image_size": 224,
            "normalize": {"mean": list(cfg["mean"]), "std": list(cfg["std"])},
            "token_pooling": "mean",
        },
        "identity_classes": identity_classes,
        "paths": paths,
        "activations": activations,
    }, out)
    log(f"saved {len(paths)} images x {len(layer_names)} blocks -> {out.resolve()}")


if __name__ == "__main__":
    main()
