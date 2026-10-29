"""KNN re-identification on steered embeddings: how many cross-identity predictions
flip to the opposite concept class, and how many queries leave the manifold."""

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any

import torch

from src.data.dataset import concept_of_identity
from src.data.features import log
from src.paths import add_run_arg, run_of

GALLERY_FRAC = 0.8
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arg(parser)
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Steered-embeddings .pt produced by src.steering.steer. "
             "Default: <run>/steering/embeddings/steered_layer<--layer>.pt",
    )
    parser.add_argument("--layer", default=None, help="Layer index to read under --run")
    parser.add_argument("--k", type=int, default=5, help="KNN neighbors (default: 5).")
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <run>/steering/eval/eval_layer<L>_k<K>.json",
    )
    args = parser.parse_args()
    if not args.embeddings and not (args.run and args.layer):
        parser.error("--embeddings, or --run together with --layer, is required")
    return args


def _embeddings_path(args: argparse.Namespace) -> Path:
    if args.embeddings:
        return Path(args.embeddings)
    return run_of(args).steering_embeddings / f"steered_layer{args.layer}.pt"


def _output_path(args: argparse.Namespace) -> Path:
    """One JSON per (layer, K) -- what src.report.steering reads back."""
    if args.output:
        return Path(args.output)
    embeddings = _embeddings_path(args)
    layer = re.search(r"steered_layer(\d+)", embeddings.stem).group(1)
    stem = f"eval_layer{layer}_k{args.k}.json"
    run = run_of(args)
    if run is not None:
        return run.steering_eval / stem
    return embeddings.parent.parent / "eval" / stem


def split_gallery_query(label_ids: torch.Tensor) -> tuple[list[int], list[int]]:
    """Per-ID split into (gallery, query); >=2 samples always give at least one of each."""
    rng = random.Random(SPLIT_SEED)
    by_id: dict[int, list[int]] = {}
    for pos, lab in enumerate(label_ids.tolist()):
        by_id.setdefault(lab, []).append(pos)

    gallery: list[int] = []
    query: list[int] = []
    for lab in sorted(by_id):
        positions = by_id[lab][:]
        rng.shuffle(positions)
        n = len(positions)
        if n < 2:
            gallery.extend(positions)
            continue
        n_gallery = max(1, min(int(round(GALLERY_FRAC * n)), n - 1))
        gallery.extend(positions[:n_gallery])
        query.extend(positions[n_gallery:])
    return sorted(gallery), sorted(query)


def cross_identity_prediction(
    distances: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_labels: torch.Tensor,
    k: int,
) -> list[int]:
    """Majority vote over the K nearest gallery items of a DIFFERENT identity, ties to
    the closest. With the query's own identity in the gallery nothing could flip."""
    own = gallery_labels.unsqueeze(0) == query_labels.unsqueeze(1)
    distances = distances.masked_fill(own, float("inf"))
    neigh_dist, neigh_idx = torch.topk(distances, k=k, dim=1, largest=False)
    preds: list[int] = []
    for labels_row, dist_row in zip(gallery_labels[neigh_idx].tolist(), neigh_dist.tolist()):
        votes: dict[int, int] = {}
        nearest: dict[int, float] = {}
        for lab, dist in zip(labels_row, dist_row):
            votes[lab] = votes.get(lab, 0) + 1
            nearest[lab] = min(nearest.get(lab, float("inf")), dist)
        preds.append(min(votes, key=lambda lab: (-votes[lab], nearest[lab])))
    return preds


def gallery_spacing_p99(gallery: torch.Tensor) -> float:
    """99th percentile of the gallery's own NN distances -- the manifold edge."""
    distances = torch.cdist(gallery, gallery, p=2)
    distances.fill_diagonal_(float("inf"))
    own_nn = distances.min(dim=1).values.sort().values
    return float(own_nn[int(0.99 * (len(own_nn) - 1))])


def main() -> None:
    args = parse_args()

    log(f"Loading embeddings: {_embeddings_path(args)}")
    data = torch.load(_embeddings_path(args), map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = data["embeddings"].float()  # (A, N, D)
    alphas: list[float] = [float(a) for a in data["alphas"]]
    label_names: list[str] = data["label_names"]
    label_ids: torch.Tensor = data["label_ids"].long()  # (N,)
    baseline_a = alphas.index(0.0)

    preset = data["meta"]["preset"]
    id_to_concept = concept_of_identity(preset)

    # Identities the preset gives no class are dropped after the split, so every
    # neighbour and every prediction carries one.
    gallery_idx, query_idx = split_gallery_query(label_ids)
    labelled = {pos for pos, lab in enumerate(label_ids.tolist())
                if id_to_concept.get(label_names[lab]) is not None}
    gallery_idx = torch.tensor([i for i in gallery_idx if i in labelled])
    query_idx = torch.tensor([i for i in query_idx if i in labelled])
    gallery_labels = label_ids[gallery_idx]
    query_labels = label_ids[query_idx]
    n_query = len(query_idx)
    own_concept = [id_to_concept[label_names[lab]] for lab in query_labels.tolist()]
    log(f"preset '{preset}': gallery {len(gallery_idx)}, query {n_query} "
        f"({len(set(query_labels.tolist()))} ids)")

    # Only the query is steered, so the gallery and its manifold edge are fixed.
    gallery = embeddings[baseline_a][gallery_idx]
    manifold_edge = gallery_spacing_p99(gallery)

    predicted: list[list[str]] = []
    off_manifold: list[float] = []
    for a_i in range(len(alphas)):
        distances = torch.cdist(embeddings[a_i][query_idx], gallery, p=2)
        preds = cross_identity_prediction(distances, gallery_labels, query_labels, args.k)
        predicted.append([id_to_concept[label_names[p]] for p in preds])
        off = distances.min(dim=1).values > manifold_edge
        off_manifold.append(100.0 * float(off.sum()) / n_query)

    # A flip needs the baseline prediction to carry the query's OWN class, so a query
    # already predicted as the opposite class cannot flip into it.
    per_alpha: list[dict[str, Any]] = []
    for a_i, alpha in enumerate(alphas):
        flips = sum(
            1 for own, before, after in zip(own_concept, predicted[baseline_a], predicted[a_i])
            if before == own and after != own
        )
        per_alpha.append({
            "alpha": alpha,
            "flip_toward_concept_pct": 100.0 * flips / n_query,
            "off_manifold_pct": off_manifold[a_i],
        })
        log(f"alpha={alpha:g}: flipped {per_alpha[-1]['flip_toward_concept_pct']:6.2f}%  "
            f"off-manifold {off_manifold[a_i]:5.1f}%")

    report = {
        "config": {
            "embeddings": str(_embeddings_path(args).resolve()),
            "k": args.k,
            "preset": preset,
            "n_gallery": len(gallery_idx),
            "n_query": n_query,
            # Carried verbatim so a directory of eval JSONs says which probes it
            # steered without keeping the .pt files.
            "steering": data["meta"],
        },
        "alphas": alphas,
        "per_alpha": per_alpha,
    }
    out_path = _output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Saved report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
