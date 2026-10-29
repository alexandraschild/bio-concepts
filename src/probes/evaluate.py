"""Evaluate probes into a uniform cache (holdout / id-cv / transfer).

A holdout run scores the probes as trained; id-cv refits per held-out identity;
transfer is a holdout run with --activations pointing at another backbone.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.features import load_pooled_features, log
from src.paths import add_run_arg, run_of, split_path_for
from src.probes.bundle import load_bundle
from src.data.metadata import SampleMeta, build_sample_meta, load_age_sex_map, stored_columns
from src.data.dataset import LABEL_PRESETS, cache_labels
from src.probes import metrics as M
from src.probes.cache import EvalWriter
from src.probes.discovery import load_probe_direction, normalize_layer, probes_in_dir, sort_layers
from src.probes.id_cv import IdCvConfig, id_cross_validate


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _seen_identities(split_file: str | None) -> set[str] | None:
    """Identities the backbone was fine-tuned on (the split's train key), so every
    identity row can be tagged. None = unknown, rendered blank rather than False."""
    if not split_file:
        return None
    with Path(split_file).open("r", encoding="utf-8") as f:
        return {str(tag).upper() for tag in json.load(f)["train"]}


def _filter(
    y: np.ndarray,
    scores: np.ndarray,
    meta: SampleMeta,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, SampleMeta]:
    """Drop identities absent from the mapping CSV -- they carry no class statistic.
    A cache with no age-sex classes at all (jaguars, polar bears) keeps every row."""
    keep = np.ones(y.shape[0], dtype=bool)
    unclassified = meta.age_sex == ""
    if unclassified.all():
        return y, scores, meta
    if unclassified.any():
        names = sorted(set(meta.identity[unclassified].tolist()))
        log(
            f"  [{context}] dropping {len(names)} identity/identities absent from the "
            f"mapping CSV ({int(unclassified.sum())} images): {', '.join(names[:8])}"
            f"{' ...' if len(names) > 8 else ''}"
        )
        keep &= ~unclassified

    if not keep.any():
        raise SystemExit(f"[{context}] every sample was filtered out")
    rows = np.where(keep)[0]
    return y[rows], scores[rows], meta.select(rows)


def _emit(
    writer: EvalWriter,
    *,
    layer: str,
    split: str,
    y: np.ndarray,
    scores: np.ndarray,
    meta: SampleMeta,
    probe_threshold: float,
    class_names: tuple[str, str],
    seen_ids: set[str] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compute and stage every table for one (layer, split). Returns the headline."""
    global_rows = M.global_rows(
        y, scores, meta,
        probe_threshold=probe_threshold,
    )
    writer.add("metrics", layer, split, global_rows)

    # Class accuracies are image-level, so they use the image-level thresholds.
    best_image = next(
        r["threshold"] for r in global_rows if r["unit"] == "image" and r["threshold_kind"] == "best"
    )
    writer.add("per_class", layer, split, M.class_rows(
        y, scores, meta,
        thresholds={"probe": probe_threshold, "best": best_image},
        class_names=class_names,
    ))

    identity_table = M.identity_rows(
        y, scores, meta,
        probe_threshold=probe_threshold,
        class_names=class_names,
        seen_ids=seen_ids,
    )
    writer.add("identities", layer, split, identity_table)

    headline = next(r for r in global_rows if r["unit"] == "image" and r["threshold_kind"] == "probe")
    identity_row = next(
        r for r in global_rows if r["unit"] == "identity" and r["threshold_kind"] == "probe"
    )
    log(
        f"  [{layer}/{split}] AUROC image={headline['auroc']:.4f} "
        f"identity={identity_row['auroc']:.4f} | acc={headline['accuracy']:.4f} "
        f"bal_acc={headline['balanced_accuracy']:.4f} "
        f"(n={headline['n']} images, {len(identity_table)} identities)"
    )
    return headline


# --------------------------------------------------------------------------- #
# holdout
# --------------------------------------------------------------------------- #
def run_holdout(args: argparse.Namespace) -> None:
    class_map = load_age_sex_map()

    run = run_of(args)
    probe_dirs = [Path(p) for p in args.probes] if args.probes else [run.probes]
    # None unless --activations names a cache other than the one the probes were
    # fitted on, in which case this run is a transfer and says so everywhere.
    targets = [_transfer_target(args, d) for d in probe_dirs]
    # The run names the probe, so step up past probes/; a transfer run also names
    # the backbone it was scored on.
    labels = [
        (d.parent.name if d.name == "probes" else d.name)
        + ("" if target is None else f" on {_backbone_tag(target)}")
        for d, target in zip(probe_dirs, targets)
    ]

    for probe_dir, label, target in zip(probe_dirs, labels, targets):
        outdir = _holdout_outdir(args, probe_dir, target)
        log(f"=== {probe_dir} -> {outdir}  [{label}] ===")
        _run_holdout_dir(args, probe_dir, label, outdir, class_map, target)


def _backbone_tag(activations: Path) -> str:
    return Path(activations).parent.name


def _fit_activations(probe_dir: Path) -> Path:
    _, probe_path = probes_in_dir(probe_dir)[0]
    with split_path_for(probe_path).open("r", encoding="utf-8") as f:
        return Path(json.load(f)["source_activation_file"])


def _transfer_target(args: argparse.Namespace, probe_dir: Path) -> Path | None:
    if not args.activations:
        return None
    target = Path(args.activations)
    # An override that names the probe's own cache is a plain holdout.
    return None if _fit_activations(probe_dir).resolve() == target.resolve() else target


def _holdout_outdir(
    args: argparse.Namespace, probe_dir: Path, transfer_target: Path | None = None
) -> Path:
    """Its own place in the run, so the numbers never land among the .pt files they
    describe. A transfer run is named after the backbone it was scored on."""
    if args.out is not None:
        root = Path(args.out)
        return root / probe_dir.name if len(args.probes or []) > 1 else root
    run = run_of(args)
    if run is not None:
        return run.eval_holdout if transfer_target is None else run.eval_transfer(
            _backbone_tag(transfer_target)
        )
    root = probe_dir.parent / "eval"
    if transfer_target is None:
        return root / "holdout"
    return root / f"transfer_{_backbone_tag(transfer_target)}"


def _run_holdout_dir(
    args: argparse.Namespace,
    probe_dir: Path,
    label: str,
    outdir: Path,
    class_map: dict[str, str],
    transfer_target: Path | None = None,
) -> None:
    probes = probes_in_dir(probe_dir)
    writer: EvalWriter | None = None
    config: dict[str, Any] = {}

    seen_ids = _seen_identities(args.split_file)

    if transfer_target is not None:
        _check_row_alignment(probe_dir, transfer_target)

    for layer, probe_path in probes:
        bundle = load_bundle(
            probe_path,
            activations_path=args.activations,
            feature_cache=args.feature_cache,
        )

        if writer is None:
            with Path(bundle.split_path).open("r", encoding="utf-8") as f:
                split_doc = json.load(f)
            config = {
                "mode": "holdout" if transfer_target is None else "transfer",
                "label": label,
                "probes_dir": str(probe_dir.resolve()),
                "activations": str(bundle.activations_path),
                # On a transfer run "activations" is the backbone that was read, not
                # the one the probe was fitted on.
                "fit_activations": str(_fit_activations(probe_dir)),
                "class_names": list(bundle.class_names),
                "split_file": split_doc["split"].get("split_file"),
                "probe": {
                    k: v for k, v in bundle.describe().items()
                    if k not in ("layer", "probe_path", "split_path", "n_train", "n_test")
                },
                "probe_paths": {},
            }
            if transfer_target is not None:
                config["transfer"] = {
                    "from": probe_dir.parent.name,
                    "to": _backbone_tag(transfer_target),
                }
            if seen_ids is None:
                seen_ids = _seen_identities(config["split_file"])
            writer = EvalWriter(outdir, config)

        config["probe_paths"][layer] = str(probe_path.resolve())

        direction = load_probe_direction(probe_path)
        writer.add_direction(layer, direction)

        for split in ("test", "train"):
            X, y, meta = (
                (bundle.X_test, bundle.y_test, bundle.test_meta) if split == "test"
                else (bundle.X_train, bundle.y_train, bundle.train_meta)
            )
            meta = dataclasses.replace(
                meta,
                age_sex=np.asarray(
                    [class_map.get(g, "") for g in meta.identity], dtype=object
                ),
            )
            y, scores, meta = _filter(
                y, bundle.decision(X), meta,
                context=f"{layer}/{split}",
            )
            _emit(
                writer, layer=layer, split=split, y=y, scores=scores, meta=meta,
                probe_threshold=bundle.decision_threshold,
                class_names=bundle.class_names, seen_ids=seen_ids, args=args,
            )

    writer.config.update(config)
    log(f"wrote {writer.save()}")


def _check_row_alignment(probe_dir: Path, activations: Path) -> None:
    """Verify row i is the same image in both caches before a transfer run."""
    _, probe_path = probes_in_dir(probe_dir)[0]
    manifest = split_path_for(probe_path)
    with manifest.open("r", encoding="utf-8") as f:
        split_doc = json.load(f)


    payload = torch.load(activations, map_location="cpu", weights_only=False)
    paths = np.asarray(payload["paths"], dtype=object)

    for side, records in split_doc["samples"].items():
        index = np.asarray([int(r["index"]) for r in records], dtype=np.int64)
        want = np.asarray([r["path"] for r in records], dtype=object)
        if int(index.max(initial=-1)) >= paths.shape[0]:
            raise SystemExit(
                f"{activations} holds {paths.shape[0]} rows, but the probe's {side} split "
                f"indexes up to row {int(index.max())} — the two caches were collected over "
                "different image sets, so the probe cannot be transferred onto this one."
            )
        differ = np.flatnonzero(paths[index] != want)
        if differ.size:
            i = int(differ[0])
            raise SystemExit(
                f"{activations} is not row-aligned with the probe's activations: row "
                f"{int(index[i])} is {paths[index[i]]} there and {want[i]} in "
                f"{manifest.name} ({differ.size} of {index.shape[0]} {side} rows differ)."
            )


# --------------------------------------------------------------------------- #
# id-cv
# --------------------------------------------------------------------------- #
def run_id_cv(args: argparse.Namespace) -> None:

    activations = Path(args.activations).resolve()
    payload = torch.load(activations, map_location="cpu", weights_only=False)

    paths = payload["paths"]
    # The preset picks the rows it can label; `rows` indexes the cached feature
    # matrix, which is what SampleMeta.index holds.
    rows, y_preset, names = cache_labels(payload, args.preset)
    rows = rows.numpy().astype(np.int64)
    y_all = y_preset.numpy().astype(np.int64)
    class_names = (str(names[0]), str(names[1]))
    log(f"preset '{args.preset}': {rows.shape[0]}/{len(paths)} cached rows labelled "
        f"{class_names[0]}/{class_names[1]}")

    row_paths = [paths[i] for i in rows]
    meta_all = build_sample_meta(rows, row_paths, y_all, **stored_columns(payload, rows))

    layers = _id_cv_layers(payload, args)
    cfg = IdCvConfig(standardize=args.standardize, max_iter=args.max_iter)
    outdir = Path(args.out) if args.out else run_of(args).eval_id_cv
    label = outdir.name
    seen_ids = _seen_identities(args.split_file)

    log(f"=== ID-CV, one fold per {args.fold_level} -> {outdir}  [{label}] ===")
    log(f"    {len(layers)} layer(s): {', '.join(layers)}")

    writer = EvalWriter(outdir, {
        "mode": "id_cv",
        "label": label,
        "activations": str(activations),
        "preset": args.preset,
        "class_names": list(class_names),
        "split_file": args.split_file,
        "fold_level": args.fold_level,
        "probe": cfg.describe(),
        "n_jobs": args.n_jobs,
    })

    fold_agreement: dict[str, float] = {}
    for layer in layers:
        log(f"--- {layer} ---")
        features = load_pooled_features(activations, layer, cache_dir=args.feature_cache)
        y, _unused, meta = _filter(y_all, np.zeros(y_all.shape[0]), meta_all, context=layer)
        X = np.asarray(features, dtype=np.float64)[meta.index]

        result = id_cross_validate(
            X, y, meta.group_keys(args.fold_level), cfg, n_jobs=args.n_jobs, log=log
        )
        writer.add_direction(layer, result.mean_direction)
        fold_agreement[layer] = M.mean_pairwise_cosine(result.directions)
        log(f"  mean pairwise cosine between the {len(result.directions)} fold "
            f"directions: {fold_agreement[layer]:.4f}")

        # An ID-CV score vector has no fitted threshold -- each identity's logits come
        # from a different fold -- so 0 is the only defensible one.
        _emit(
            writer, layer=layer, split="id_cv", y=y, scores=result.scores, meta=meta,
            probe_threshold=0.0, class_names=class_names, seen_ids=seen_ids, args=args,
        )

    writer.config["fold_direction_agreement"] = fold_agreement
    log(f"wrote {writer.save()}")


def _id_cv_layers(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    available = list(payload["activations"].keys())
    if args.all_layers:
        return sort_layers(available)
    wanted = [normalize_layer(l) for l in args.layers]
    missing = [l for l in wanted if l not in available]
    if missing:
        raise SystemExit(
            f"layer(s) {missing} not in {args.activations}. "
            f"Available: {', '.join(sort_layers(available))}"
        )
    return sort_layers(wanted)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    add_run_arg(parser)
    parser.add_argument("--out", default=None,
                        help="Cache directory. Default: <run>/eval/holdout or "
                             "<run>/eval/id_cv.")
    parser.add_argument("--split-file", default=None,
                        help="Identity split JSON; its train identities are marked as seen "
                             "by the backbone. holdout defaults to the probe's own.")
    parser.add_argument("--feature-cache", default=".cache/features",
                        help="Directory for memoised pooled features (empty to disable).")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    hold = sub.add_parser(
        "holdout",
        help="Score saved probes on their held-out identities",
        description="Evaluate probes exactly as src.probes.train wrote them.",
    )
    hold.add_argument("--probes", nargs="+", default=None,
                      help="Probe director(ies) of probe_<layer>.pt files. One cache each.")
    hold.add_argument("--activations", default=None,
                      help="Override the manifest's activation file. Another backbone's "
                           "cache makes this a transfer run.")
    _add_common(hold)
    hold.set_defaults(func=run_holdout)

    id_cv = sub.add_parser(
        "id-cv",
        help="Identity-wise cross-validation: one refit per held-out identity, so "
             "every logit is out-of-sample",
        description="Fit one probe per held-out identity and score that identity with it.",
    )
    id_cv.add_argument("--activations", required=True,
                       help="Activation file from src.data.collect")
    id_cv.add_argument("--preset", default="sex", choices=list(LABEL_PRESETS),
                       help="Task to label the cached rows for; also decides which rows take "
                            "part at all.")
    layer_group = id_cv.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layers", nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")
    id_cv.add_argument("--fold-level", default="identity",
                       help="What one fold holds out: an identity, or a coarser group the "
                            "cache stores (the polar bears' zoo).")
    id_cv.add_argument("--standardize", action="store_true",
                       help="Standardize inside each fold. Off by default, matching the "
                            "probes these are compared against.")
    id_cv.add_argument("--max-iter", type=int, default=2000)
    id_cv.add_argument("--n-jobs", type=int, default=1, help="Parallel folds (joblib)")
    _add_common(id_cv)
    id_cv.set_defaults(func=run_id_cv)

    args = ap.parse_args()
    args.feature_cache = args.feature_cache or None
    if not args.run and not (args.probes if args.command == "holdout" else args.out):
        ap.error("--run is required, or --probes (holdout) / --out (id-cv)")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
