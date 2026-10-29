"""Random-direction controls for a probe's steering sweep: same subset, alphas and
per-sample signs, only the direction vector changes.

Every direction is unit-normalized, so the injected vector alpha*sign*d has L2
norm exactly alpha at every token position -- the same magnitude the probe sweep
injects at that alpha.
"""

import argparse
from pathlib import Path

import torch

from src.data.features import log
from src.paths import Run
from src.steering.steer import (
    compute_rope,
    load_layer_activations,
    load_model_with_head,
    steered_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", required=True,
        help="<run>/steering/embeddings/steered_layer<L>.pt the controls are matched to. "
             "They are written to <run>/steering/controls/random_s<seed>/.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    ref_meta = reference["meta"]
    layer_idx = int(ref_meta["steering_layer"])
    alphas = [float(a) for a in reference["alphas"]]
    signs = reference["signs"]
    indices = reference["sample_indices"]
    log(
        f"Reference {Path(args.reference).name}: layer {layer_idx}, alphas {alphas}, "
        f"{len(indices)} samples, {int((signs != 0).sum())} of them steered"
    )

    cache_path = Path(ref_meta["source_activations"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    acts = load_layer_activations(
        cache["activations"][f"blocks.{layer_idx}"], indices, cache_path.parent
    )
    d_model = acts.shape[2]
    log(f"Activation subset: {tuple(acts.shape)} (N, tokens, d_model)")

    model = load_model_with_head(ref_meta["model_path"], ref_meta["timm_model"], args.device)
    rope = compute_rope(model, acts.shape[1])

    # <run>/steering/embeddings/steered_layer<L>.pt -> <run>/steering/controls
    out_dir = Run(Path(args.reference).resolve().parents[2]).steering_controls
    for seed in args.seeds:
        tag = f"random_s{seed}"
        log(f"===== {tag} =====")
        direction = torch.randn(d_model, generator=torch.Generator().manual_seed(seed))
        direction = direction / direction.norm(p=2)

        embeddings = torch.empty_like(reference["embeddings"])
        for a_i, alpha in enumerate(alphas):
            if alpha == 0.0:
                # No perturbation is added, so this is bit-identical to the reference.
                embeddings[a_i] = reference["embeddings"][a_i]
                continue
            log(f"Steering alpha={alpha} ({a_i + 1}/{len(alphas)})")
            embeddings[a_i] = steered_embeddings(
                model, acts, layer_idx, direction, alpha,
                signs, rope, args.batch_size, args.device,
            )

        meta = dict(ref_meta)
        # 'preset' stays the reference's: a control is scored with that concept's metric.
        # The signs stay the reference's too, so the perturbation has the same structure.
        meta.update({
            "control_direction": tag,
            "control_of": str(Path(args.reference).resolve()),
            "probe": None,
            "probe_layer": None,
            "probe_type": "random_unit_vector",
            "sign_convention": "signs_from_reference",
        })
        # A run-shaped directory, so evaluate_reid reads a control with --run like any sweep.
        out_path = out_dir / tag / "steering" / "embeddings" / f"steered_layer{layer_idx}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**reference, "meta": meta, "embeddings": embeddings}, out_path)
        log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
