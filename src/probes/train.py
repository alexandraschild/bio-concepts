"""Train a linear probe on cached activations; writes probe_<layer>.pt + a split manifest."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import re
from typing import Any, NamedTuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.dataset import LABEL_PRESETS, cache_labels
from src.data.features import load_pooled_features, log
from src.data.metadata import identity_of
from src.paths import add_run_arg, run_of
from src.probes.discovery import fmt_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a probe on activation tensors from activations.pt"
    )
    parser.add_argument(
        "--input",
        default="activations.pt",
        help="Path to activation file produced by src.data.collect",
    )
    parser.add_argument(
        "--probe",
        choices=["logistic_regression", "difference_in_means", "dualxda_svm"],
        default="logistic_regression",
        help="Probe type; dualxda_svm fits through the DualXDA package, so the probe and "
             "its attributions come from one fit.",
    )
    parser.add_argument(
        "--preset",
        default="sex",
        choices=list(LABEL_PRESETS.keys()),
        help="Task to label the cached rows for; also decides which rows take part at all.",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer",
        nargs="+",
        default=None,
        metavar="LAYER_NAME",
        help="Layer name(s) to train a probe for.",
    )
    layer_group.add_argument(
        "--all-layers",
        action="store_true",
        help="Train one probe per layer for every layer in the activation file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction used for test split (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--subsample",
        choices=("none", "identity"),
        default="none",
        help="Keep one image per identity, drawn with --seed, before anything else runs.",
    )
    parser.add_argument(
        "--subsample-draws",
        type=int,
        default=1,
        metavar="N",
        help="Independent --subsample draws to fit, each into its own draw<NN>/ dir.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON split file mapping split keys to gorilla identity tags.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Split-file key(s) whose identities form the probe TRAIN set (default: train).",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Split-file key(s) forming the held-out EVAL set; membership is by identity, "
             "so no individual appears on both sides.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Solver tolerance (default: 1e-8, far tighter than sklearn's 1e-4). The "
             "representer identity w = sum_i alpha_i x_i only holds at the exact optimum, "
             "and the slack sklearn permits in w is C * n_train * tol.",
    )
    parser.add_argument(
        "--dualxda-tol",
        type=float,
        default=1e-4,
        help="Solver tolerance for dualxda_svm, which must stay loose (~1e-4): below ~1e-5 "
             "the Crammer-Singer solver decouples w from its dual variables.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Max solver iterations (default: 20000). Below ~15000 the cap binds before "
             "--tol does on unstandardized features, making the tolerance inert.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength C (default: 1.0)",
    )
    parser.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Reweight classes inversely to their train frequency (difference_in_means "
             "ignores it). Default: off, letting the train prior stand.",
    )
    parser.add_argument(
        "--feature-reduction",
        choices=["mean"],
        default="mean",
        help="Token pooling for rank>=3 activations.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="StandardScaler the features, fit on train. Default: off (raw features).",
    )
    add_run_arg(parser)
    parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Directory the run writes to. Default: a name derived from the fit, beside the "
             "activation file.",
    )
    return parser.parse_args()


def _load_split_file_ids(path: Path, key: str) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise ValueError(f"{path} has no '{key}' key; it holds {sorted(data)}")
    return {tag.upper() for tag in data[key]}


def _load_split_file_ids_union(path: Path, keys: list[str]) -> set[str]:
    ids: set[str] = set()
    for key in keys:
        ids |= _load_split_file_ids(path, key)
    return ids


def _filter_by_train_ids(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    train_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    keep_mask = np.array([identity_of(p) in train_ids for p in sample_paths], dtype=bool)
    n_kept = int(keep_mask.sum())
    log(
        f"Split-file filter: keeping {n_kept}/{X.shape[0]} samples "
        f"from {len(train_ids)} train identities"
    )
    if n_kept == 0:
        raise ValueError(
            "Split-file filter removed all samples — no path matched any train identity. "
            "Check that the identity tags in the split file match path name prefixes."
        )
    kept_rows = np.where(keep_mask)[0]
    return (
        X[kept_rows],
        y[kept_rows],
        sample_indices[kept_rows],
        [sample_paths[i] for i in kept_rows],
    )


def _filter_one_image_per_group(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    rows_by_group: dict[str, list[int]] = {}
    for row, path in enumerate(sample_paths):
        rows_by_group.setdefault(identity_of(path), []).append(row)

    keep_rows: list[int] = []
    for key in sorted(rows_by_group):
        candidate_rows = sorted(rows_by_group[key], key=lambda r: sample_paths[r])
        keep_rows.append(int(rng.choice(candidate_rows)))
    keep_rows.sort()

    log(
        f"--subsample identity: keeping {len(keep_rows)}/{X.shape[0]} samples "
        f"(one per identity across {len(rows_by_group)} identities, seed={seed})"
    )
    rows = np.asarray(keep_rows, dtype=np.int64)
    return (
        X[rows],
        y[rows],
        sample_indices[rows],
        [sample_paths[i] for i in keep_rows],
    )


SPLIT_TRAIN, SPLIT_TEST = 0, 1


def _filter_and_assign_splits(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    train_ids: set[str],
    eval_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    keep_rows: list[int] = []
    split_id_list: list[int] = []
    for i, p in enumerate(sample_paths):
        gid = identity_of(p)
        if gid in eval_ids:
            split = SPLIT_TEST
        elif gid in train_ids:
            split = SPLIT_TRAIN
        else:
            continue
        keep_rows.append(i)
        split_id_list.append(split)

    if not keep_rows:
        raise ValueError(
            "Open-set filter removed all samples — no path matched any train, val or eval "
            "identity. Check that the identity tags match path name prefixes."
        )

    split_id = np.array(split_id_list, dtype=np.int64)
    counts = {name: int((split_id == sid).sum()) for name, sid in
              (("train", SPLIT_TRAIN), ("test", SPLIT_TEST))}
    log(
        f"Open-set filter: kept {counts['train']} train-identity + {counts['test']} "
        f"eval-identity samples from {X.shape[0]} total"
    )
    rows = np.asarray(keep_rows, dtype=np.int64)
    return (
        X[rows],
        y[rows],
        sample_indices[rows],
        [sample_paths[i] for i in keep_rows],
        split_id,
    )


def _select_layers(
    payload: dict[str, Any],
    requested_layers: list[str] | None,
    all_layers: bool,
) -> list[str]:
    """The layer names to train a probe for, in collection order."""
    available = payload["meta"]["layer_names"]
    if all_layers:
        return list(available)
    if not requested_layers:
        raise ValueError(f"Pass --layer or --all-layers; the cache holds {available}")
    missing = [name for name in requested_layers if name not in payload["activations"]]
    if missing:
        raise ValueError(f"Layers not found: {missing}. Available: {available}")
    return list(dict.fromkeys(requested_layers))


def _auto_probe_dirname(args: argparse.Namespace, probe_type: str) -> str:
    """A directory name naming the fit: method, C, and whichever knobs are off default."""

    parts = [probe_type]
    if probe_type != "difference_in_means":
        parts.append(f"c{fmt_c(args.c)}")
    if args.class_weight_balanced:
        parts.insert(0, "balanced")
    if args.standardize:
        parts.append("std")
    if args.subsample != "none":
        parts.append(args.subsample)
    return "_".join(parts)


def _resolve_outdir(args: argparse.Namespace, probe_type: str) -> Path:
    """--out, else <run>/probes, else a name derived from the fit."""
    # Not resolve(): run directories are symlinks, and following one puts the output
    # beside the real location instead of the name it was reached by.
    if args.out is not None:
        return Path(args.out)
    run = run_of(args)
    if run is not None:
        return run.probes
    return Path(args.input).parent / _auto_probe_dirname(args, probe_type)


def _ensure_binary_labels(
    labels: torch.Tensor,
    preset: str,
    preset_names: list[str] | None = None,
) -> np.ndarray:
    """Both classes have to be present; name the missing one, since a cache holds only
    the identity classes it was collected over."""
    y = labels.detach().cpu().numpy().astype(np.int64)
    unique = np.unique(y)
    if unique.shape[0] == 2:
        return y

    names = preset_names or [str(i) for i in range(2)]
    present = ", ".join(f"{names[i]} ({int((y == i).sum())} images)"
                        for i in unique.tolist() if i < len(names))
    missing = ", ".join(names[i] for i in range(len(names)) if i not in unique.tolist())
    raise ValueError(
        f"Preset '{preset}' needs both classes; this cache has only {present} -- no {missing}."
    )


def _standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    standardize: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler the features, fit on train. When off, report an identity scaler so
    consumers that de-standardize with (mean, scale) become a no-op."""

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler.mean_, scaler.scale_

    log(f"[{label}] standardization DISABLED — fitting on raw features")
    n_features = X_train.shape[1]
    scaler_mean = np.zeros(n_features, dtype=np.float64)
    scaler_scale = np.ones(n_features, dtype=np.float64)
    return X_train, X_test, scaler_mean, scaler_scale


def _resolve_class_weight(
    class_weight_balanced: bool,
    y_train: np.ndarray,
    *,
    label: str,
) -> str | None:
    counts = Counter(int(v) for v in y_train.tolist())
    summary = ", ".join(f"{cls}: {n}" for cls, n in sorted(counts.items()))
    if class_weight_balanced:
        log(f"[{label}] class_weight='balanced' (train counts — {summary})")
        return "balanced"
    log(f"[{label}] class weighting DISABLED — fitting on the raw train prior ({summary})")
    return None


def _build_split_sample_records(
    indices: list[int],
    labels: np.ndarray,
    *,
    sample_paths: list[str],
    label_names: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx in indices:
        label_id = int(labels[idx])
        record: dict[str, Any] = {
            "index": int(idx),
            "label": label_id,
        }
        if 0 <= label_id < len(label_names):
            record["label_name"] = label_names[label_id]
        record["path"] = sample_paths[idx]
        records.append(record)
    return records


class _Splits(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    idx_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray


def _train_test_split_features(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    *,
    test_size: float,
    seed: int,
    sample_paths: list[str],
    predefined_split: tuple[np.ndarray, np.ndarray] | None = None,
) -> _Splits:
    if predefined_split is not None:
        # Split is fully determined by the caller (open-set identity split): just
        # slice. test_size / seed are ignored on this path.
        train_rows, test_rows = predefined_split
        return _Splits(
            X_train=X[train_rows],
            y_train=y[train_rows],
            idx_train=sample_indices[train_rows],
            X_test=X[test_rows],
            y_test=y[test_rows],
            idx_test=sample_indices[test_rows],
        )


    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        sample_indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return _Splits(
        X_train=X_train,
        y_train=y_train,
        idx_train=idx_train,
        X_test=X_test,
        y_test=y_test,
        idx_test=idx_test,
    )


def _prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    *,
    test_size: float,
    seed: int,
    predefined_split: tuple[np.ndarray, np.ndarray] | None,
    standardize: bool,
    label: str,
) -> tuple[_Splits, np.ndarray, np.ndarray]:
    splits = _train_test_split_features(
        X,
        y,
        sample_indices,
        sample_paths=sample_paths,
        test_size=test_size,
        seed=seed,
        predefined_split=predefined_split,
    )
    X_train, X_test, scaler_mean, scaler_scale = _standardize_features(
        splits.X_train, splits.X_test, standardize=standardize, label=label,
    )
    return splits._replace(X_train=X_train, X_test=X_test), scaler_mean, scaler_scale


def train_logistic_regression_probe(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    *,
    test_size: float,
    seed: int,
    max_iter: int,
    tol: float,
    c: float,
    predefined_split: tuple[np.ndarray, np.ndarray] | None = None,
    standardize: bool = False,
    class_weight_balanced: bool = False,
) -> tuple[tuple[float, float], _Splits, dict[str, Any], torch.Tensor]:

    splits, scaler_mean_arr, scaler_scale_arr = _prepare_splits(
        X, y, sample_indices, sample_paths, test_size=test_size, seed=seed,
        predefined_split=predefined_split, standardize=standardize,
        label="logistic_regression",
    )
    X_train, X_test = splits.X_train, splits.X_test

    class_weight = _resolve_class_weight(
        class_weight_balanced, splits.y_train, label="logistic_regression"
    )
    clf = LogisticRegression(
        C=c,
        max_iter=max_iter,
        tol=tol,
        solver="lbfgs",
        class_weight=class_weight,
    )
    clf.fit(X_train, splits.y_train)

    negative_class, positive_class = (int(c) for c in clf.classes_)

    # decision_function, not predict_proba: these logits saturate the sigmoid to
    # exactly 0.0/1.0 and the resulting ties depress AUROC.
    train_scores = clf.decision_function(X_train)
    test_scores = clf.decision_function(X_test)

    auroc = (float(roc_auc_score(splits.y_train, train_scores)),
             float(roc_auc_score(splits.y_test, test_scores)))

    direction = clf.coef_.reshape(-1).astype(np.float32)
    bias = clf.intercept_.reshape(-1).astype(np.float32)
    scaler_mean = scaler_mean_arr.reshape(-1).astype(np.float32)
    scaler_scale = scaler_scale_arr.reshape(-1).astype(np.float32)
    # Save one contiguous probe tensor: [direction..., bias...]
    probe_tensor = torch.from_numpy(np.concatenate([direction, bias], axis=0))
    probe_payload: dict[str, Any] = {
        "probe_type": "logistic_regression",
        "probe_tensor": probe_tensor,
        "bias": torch.from_numpy(bias),
        "class_labels": torch.tensor([negative_class, positive_class], dtype=torch.int64),
        "scaler_mean": torch.from_numpy(scaler_mean),
        "scaler_scale": torch.from_numpy(scaler_scale),
        "standardize": bool(standardize),
        "class_weight": class_weight,
        "decision_threshold": 0.0,
        # Attribution needs the objective this probe minimised: representer
        # weights scale with C and the Newton ridge is 1/C.
        "C": float(c),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
    }

    return auroc, splits, probe_payload, probe_tensor


def train_difference_in_means_probe(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    *,
    test_size: float,
    seed: int,
    predefined_split: tuple[np.ndarray, np.ndarray] | None = None,
    standardize: bool = False,
) -> tuple[tuple[float, float], _Splits, dict[str, Any], torch.Tensor]:
    splits, scaler_mean_arr, scaler_scale_arr = _prepare_splits(
        X, y, sample_indices, sample_paths, test_size=test_size, seed=seed,
        predefined_split=predefined_split, standardize=standardize,
        label="difference_in_means",
    )
    X_train, X_test, y_train = splits.X_train, splits.X_test, splits.y_train
    negative_class, positive_class = (int(c) for c in np.unique(y_train))

    mean_neg = X_train[y_train == negative_class].mean(axis=0)
    mean_pos = X_train[y_train == positive_class].mean(axis=0)

    direction = (mean_pos - mean_neg).astype(np.float32)
    midpoint = 0.5 * (mean_pos + mean_neg)
    bias = np.array([-float(np.dot(direction, midpoint))], dtype=np.float32)

    train_logits = (X_train @ direction) + bias[0]
    test_logits = (X_test @ direction) + bias[0]

    auroc = (float(roc_auc_score(splits.y_train, train_logits)),
             float(roc_auc_score(splits.y_test, test_logits)))

    probe_tensor = torch.from_numpy(direction)
    probe_payload: dict[str, Any] = {
        "probe_type": "difference_in_means",
        "probe_tensor": probe_tensor,
        "bias": torch.from_numpy(bias),
        "class_labels": torch.tensor([negative_class, positive_class], dtype=torch.int64),
        "scaler_mean": torch.from_numpy(scaler_mean_arr.reshape(-1).astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler_scale_arr.reshape(-1).astype(np.float32)),
        "standardize": bool(standardize),
        "decision_threshold": 0.0,
    }

    return auroc, splits, probe_payload, probe_tensor


def train_dualxda_svm_probe(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    sample_paths: list[str],
    *,
    test_size: float,
    seed: int,
    max_iter: int,
    tol: float,
    c: float,
    predefined_split: tuple[np.ndarray, np.ndarray] | None = None,
    standardize: bool = False,
    class_weight_balanced: bool = False,
) -> tuple[tuple[float, float], _Splits, dict[str, Any], torch.Tensor]:
    """Fit through DualDA's patched LIBLINEAR so the dual variables come back in the
    svm_* payload keys. See src/attribution/dualxda.py for the caveats."""
    from src.attribution.dualxda import fit_dualxda_svm

    splits, scaler_mean_arr, scaler_scale_arr = _prepare_splits(
        X, y, sample_indices, sample_paths, test_size=test_size, seed=seed,
        predefined_split=predefined_split, standardize=standardize, label="dualxda_svm",
    )
    X_train, X_test = splits.X_train, splits.X_test
    y_train, idx_train = splits.y_train, splits.idx_train

    class_weight = _resolve_class_weight(class_weight_balanced, y_train, label="dualxda_svm")
    fit = fit_dualxda_svm(
        X_train,
        y_train,
        C=c,
        class_weight=class_weight,
        tol=tol,
        max_iter=max_iter,
    )

    negative_class, positive_class = (int(c) for c in fit.classes)

    # Decision values from the fitted direction; the dual expansion reproduces
    # these to the precision reported in fit.decision_max_error.
    train_scores = X_train @ fit.w + fit.b
    test_scores = X_test @ fit.w + fit.b

    auroc = (float(roc_auc_score(splits.y_train, train_scores)),
             float(roc_auc_score(splits.y_test, test_scores)))

    direction = fit.w.astype(np.float32)
    bias = np.asarray([fit.b], dtype=np.float32)
    support_local_indices = fit.active_positions.astype(np.int64)
    support_global_indices = idx_train[support_local_indices].astype(np.int64)
    dual_coef_signed = fit.dual_coef.astype(np.float32)
    support_labels = y_train[support_local_indices].astype(np.int64)

    log(
        f"dualxda_svm: {support_local_indices.shape[0]} active "
        f"({support_local_indices.shape[0] / X_train.shape[0]:.1%} of train), "
        f"w recon rel.err={fit.reconstruction_rel_error:.2e}, "
        f"attributions sum to prediction within {fit.decision_max_error:.2e} "
        f"(cancellation {fit.cancellation_amplification:.1f}x)"
    )

    probe_tensor = torch.from_numpy(direction)
    probe_payload: dict[str, Any] = {
        "probe_type": "dualxda_svm",
        "probe_tensor": probe_tensor,
        "bias": torch.from_numpy(bias),
        "class_labels": torch.tensor([negative_class, positive_class], dtype=torch.int64),
        "scaler_mean": torch.from_numpy(scaler_mean_arr.reshape(-1).astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler_scale_arr.reshape(-1).astype(np.float32)),
        "svm_support_train_indices": torch.from_numpy(support_local_indices),
        "svm_support_sample_indices": torch.from_numpy(support_global_indices),
        # Crammer-Singer margin coefficients alpha[:,1] - alpha[:,0], playing the role
        # sklearn's dual_coef_ does, so nothing downstream branches.
        "svm_dual_coef": torch.from_numpy(dual_coef_signed),
        "svm_alphas": torch.from_numpy(np.abs(dual_coef_signed)),
        "svm_support_labels": torch.from_numpy(support_labels),
        # DualDA.attribute()'s own per-class coefficients (this repo's attribution / 2).
        "dualxda_alpha_matrix": torch.from_numpy(fit.alpha_matrix.astype(np.float32)),
        "dualxda_reconstruction_rel_error": fit.reconstruction_rel_error,
        "dualxda_decision_rel_error": fit.decision_max_error,
        "dualxda_cancellation_amplification": fit.cancellation_amplification,
        "standardize": bool(standardize),
        "class_weight": class_weight,
        "decision_threshold": 0.0,
        "C": float(c),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
    }

    return auroc, splits, probe_payload, probe_tensor


def _train_probe_for_layer(
    payload: dict[str, Any],
    layer_name: str,
    *,
    probe_type: str,
    args: argparse.Namespace,
    label_names: list[str],
    train_ids: set[str] | None,
    eval_ids: set[str] | None,
    train_split_names: list[str] | None,
    eval_split_names: list[str] | None,
    split_file_path: Path | None,
    outdir: Path,
    seed: int,
    draw: int | None,
    draws_total: int,
) -> None:
    log(f"Selected layer: {layer_name}")

    X = load_pooled_features(Path(args.input), layer_name, args.feature_reduction)
    log(
        f"Feature matrix tensor shape after reduction='{args.feature_reduction}': {tuple(X.shape)}"
    )

    # Row numbers stay in cache space, so indices, split records and paths keep agreeing.
    n_cache_rows = X.shape[0]
    preset_rows, y_preset, preset_names = cache_labels(payload, args.preset)
    sample_indices = preset_rows.numpy().astype(np.int64)
    y = _ensure_binary_labels(y_preset, args.preset, preset_names)
    X = X[sample_indices]
    log(
        f"Preset '{args.preset}': {sample_indices.shape[0]}/{n_cache_rows} cached rows labelled"
    )

    # float16 storage overflowing a massive-activation channel; otherwise this blows
    # up opaquely inside StandardScaler.
    n_nonfinite = int((~np.isfinite(X)).sum())
    if n_nonfinite:
        bad_cols = np.where(~np.isfinite(X).all(axis=0))[0]
        raise ValueError(
            f"Layer '{layer_name}': {n_nonfinite} non-finite feature value(s) across "
            f"{bad_cols.size} column(s), e.g. {bad_cols[:10].tolist()}. Re-collect with "
            "--activation-dtype bfloat16 (same size as float16, float32 range)."
        )

    class_counts = Counter(y.tolist())
    sample_paths_full = payload["paths"]
    sample_paths = [sample_paths_full[i] for i in sample_indices.tolist()]

    n_total_original = X.shape[0]
    # Lookups below address cache rows, so these span the whole cache; rows the
    # preset dropped hold -1 and are never read.
    y_full = np.full(n_cache_rows, -1, dtype=np.int64)
    y_full[sample_indices] = y
    if args.subsample != "none":
        # Before the split filter, so the draw is made over the whole payload.
        X, y, sample_indices, sample_paths = _filter_one_image_per_group(
            X, y, sample_indices, sample_paths, seed=seed
        )
        class_counts = Counter(y.tolist())
    predefined_split: tuple[np.ndarray, np.ndarray] | None = None
    if split_file_path is not None:
        if eval_ids is not None:
            # Open-set: membership is by identity, so no individual appears on
            # more than one side.
            X, y, sample_indices, sample_paths, split_id = _filter_and_assign_splits(
                X, y, sample_indices, sample_paths, train_ids or set(), eval_ids
            )
            train_rows = np.where(split_id == SPLIT_TRAIN)[0]
            test_rows = np.where(split_id == SPLIT_TEST)[0]
            if train_rows.size == 0 or test_rows.size == 0:
                raise ValueError(
                    "Open-set split produced an empty side "
                    f"(train={train_rows.size}, eval={test_rows.size}). "
                    "Check identity tags and dataset coverage for this split."
                )
            predefined_split = (train_rows, test_rows)
        else:
            X, y, sample_indices, sample_paths = _filter_by_train_ids(
                X, y, sample_indices, sample_paths, train_ids or set()
            )
        class_counts = Counter(y.tolist())

    # probe_<layer>.pt even for a one-layer run: it is the only layout
    # src.probes.evaluate reads. The split manifest is shared across layers.
    output_path = outdir / f"probe_{re.sub(r'[^0-9A-Za-z_.-]', '_', layer_name)}.pt"
    split_output_path = outdir / "split.json"

    if probe_type == "logistic_regression":
        auroc, splits, probe_payload, probe_tensor = train_logistic_regression_probe(
            X,
            y,
            sample_indices=sample_indices,
            sample_paths=sample_paths,
            test_size=args.test_size,
            seed=seed,
            max_iter=args.max_iter,
            tol=args.tol,
            c=args.c,
            predefined_split=predefined_split,
            standardize=args.standardize,
            class_weight_balanced=args.class_weight_balanced,
        )
    elif probe_type == "difference_in_means":
        auroc, splits, probe_payload, probe_tensor = train_difference_in_means_probe(
            X,
            y,
            sample_indices=sample_indices,
            sample_paths=sample_paths,
            test_size=args.test_size,
            seed=seed,
            predefined_split=predefined_split,
            standardize=args.standardize,
        )
    else:
        auroc, splits, probe_payload, probe_tensor = train_dualxda_svm_probe(
            X,
            y,
            sample_indices=sample_indices,
            sample_paths=sample_paths,
            test_size=args.test_size,
            seed=seed,
            max_iter=args.max_iter,
            tol=args.dualxda_tol,
            c=args.c,
            predefined_split=predefined_split,
            standardize=args.standardize,
            class_weight_balanced=args.class_weight_balanced,
        )

    split_info: dict[str, Any] = {
        "seed": int(seed),
        "test_size": float(args.test_size),
        "n_total": int(X.shape[0]),
        "n_train": int(splits.X_train.shape[0]),
        "n_test": int(splits.X_test.shape[0]),
        "train_indices": splits.idx_train.astype(np.int64).tolist(),
        "test_indices": splits.idx_test.astype(np.int64).tolist(),
    }

    # The shared split manifest names no layer, so the probe carries its own.
    probe_payload["layer"] = layer_name
    probe_payload["feature_reduction"] = args.feature_reduction
    probe_payload["preset"] = args.preset
    probe_payload["subsample"] = args.subsample
    if predefined_split is not None:
        probe_payload["n_train_images"] = int(predefined_split[0].size)
    if draw is not None:
        # The draw index ties every layer of one draw together for the spread.
        probe_payload["subsample_draw"] = int(draw)
        probe_payload["subsample_draws_total"] = int(draws_total)

    # Which class the direction points TOWARD, so steering can flip its sign per query.
    neg_idx, pos_idx = (int(c) for c in probe_payload["class_labels"])
    probe_payload["negative_class_index"] = neg_idx
    probe_payload["positive_class_index"] = pos_idx
    probe_payload["negative_class_name"] = label_names[neg_idx]
    probe_payload["positive_class_name"] = label_names[pos_idx]
    probe_payload["label_names"] = list(label_names)

    if split_file_path is not None:
        split_info["split_file"] = str(split_file_path.resolve())
        split_info["n_total_original"] = n_total_original
        if eval_ids is not None:
            split_info["split_mode"] = "open_set_identity"
            split_info["train_splits"] = train_split_names
            split_info["eval_splits"] = eval_split_names

    split_info["subsample"] = args.subsample
    if draw is not None:
        split_info["subsample_draw"] = int(draw)
        split_info["subsample_draws_total"] = int(draws_total)
    if args.subsample != "none":
        split_info["n_total_after_subsample"] = int(X.shape[0])
    split_info["decision_threshold"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_payload, output_path)
    log(f"Saved probe payload to: {output_path.resolve()}")

    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    split_payload: dict[str, Any] = {
        "source_activation_file": str(Path(args.input).resolve()),
        "preset": args.preset,
        "feature_reduction": args.feature_reduction,
        "split": split_info,
        "samples": {
            "train": _build_split_sample_records(
                split_info["train_indices"],
                y_full,
                sample_paths=sample_paths_full,
                label_names=label_names,
            ),
            "test": _build_split_sample_records(
                split_info["test_indices"],
                y_full,
                sample_paths=sample_paths_full,
                label_names=label_names,
            ),
        },
    }
    for side in ("train", "test"):
        split_payload["split"][f"{side}_paths"] = [
            sample_paths_full[idx] for idx in split_payload["split"][f"{side}_indices"]
        ]

    with split_output_path.open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
    log(f"Saved split manifest to: {split_output_path.resolve()}")

    log(
        f"{layer_name} {probe_type}: {X.shape[0]} samples x {X.shape[1]} features, "
        f"classes {dict(class_counts)} -> AUROC train {auroc[0]:.4f} / test {auroc[1]:.4f}"
    )
    log(f"Saved probe to: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    probe_type = args.probe

    if args.subsample_draws < 1:
        raise ValueError(f"--subsample-draws must be >= 1, got {args.subsample_draws}")
    if args.subsample_draws > 1 and args.subsample == "none":
        raise ValueError(
            "--subsample-draws > 1 requires --subsample (there is no draw to vary "
            "without one). Use e.g. --subsample identity --subsample-draws 8."
        )
    payload = torch.load(args.input, map_location="cpu", mmap=True)

    # The cache stores raw identity classes; the preset names the two the probe separates.
    label_names = list(LABEL_PRESETS[args.preset])

    if (args.train_splits is not None or args.eval_splits is not None) and args.split_file is None:
        raise ValueError("--train-splits/--eval-splits require --split-file.")

    # Layer-independent, so load them once.
    split_file_path: Path | None = None
    train_ids: set[str] | None = None
    eval_ids: set[str] | None = None
    train_split_names: list[str] | None = None
    eval_split_names: list[str] | None = args.eval_splits
    if args.split_file is not None:
        split_file_path = Path(args.split_file)
        train_split_names = args.train_splits if args.train_splits is not None else ["train"]
        train_ids = _load_split_file_ids_union(split_file_path, train_split_names)
        log(
            f"Loaded split file: {split_file_path} — train identities from "
            f"{train_split_names}: {len(train_ids)}"
        )
        if eval_split_names is not None:
            named_sets = {"--train-splits": train_split_names, "--eval-splits": eval_split_names}
            for (name_a, keys_a), (name_b, keys_b) in itertools.combinations(named_sets.items(), 2):
                key_overlap = set(keys_a) & set(keys_b)
                if key_overlap:
                    raise ValueError(
                        f"Split keys appear in both {name_a} and {name_b}: "
                        f"{sorted(key_overlap)}. They must be disjoint."
                    )

            eval_ids = _load_split_file_ids_union(split_file_path, eval_split_names)
            id_sets = {"train": train_ids, "eval": eval_ids}
            for (name_a, ids_a), (name_b, ids_b) in itertools.combinations(id_sets.items(), 2):
                id_overlap = ids_a & ids_b
                if id_overlap:
                    raise ValueError(
                        f"The {name_a} and {name_b} identity sets share individuals "
                        f"({len(id_overlap)} shared, e.g. {sorted(id_overlap)[:10]}). "
                        "Open-set evaluation requires disjoint identity sets."
                    )
            log(
                f"Open-set eval: train on {train_split_names} ({len(train_ids)} ids), "
                f"test on {eval_split_names} ({len(eval_ids)} ids)"
            )

    layers = _select_layers(payload, args.layer, args.all_layers)
    multi_layer = len(layers) > 1
    if multi_layer:
        log(f"Training one probe per layer for {len(layers)} layers: {layers}")

    draws_total = args.subsample_draws
    base_outdir = _resolve_outdir(args, probe_type)
    log(f"Probe directory: {base_outdir}")
    if draws_total > 1:
        log(
            f"--subsample-draws {draws_total}: fitting {draws_total} independent "
            f"{args.subsample} draws (seeds {args.seed}..{args.seed + draws_total - 1}) into "
            f"draw<NN>/ subdirectories of {base_outdir}"
        )

    for draw in range(draws_total):
        if draws_total > 1:
            outdir = base_outdir / f"draw{draw:02d}"
            seed = args.seed + draw
            draw_index: int | None = draw
            print()
            print("#" * 60)
            print(f"Draw {draw + 1}/{draws_total} (seed {seed}) -> {outdir}")
            print("#" * 60)
        else:
            outdir = base_outdir
            seed = args.seed
            draw_index = None

        for layer_name in layers:
            if multi_layer:
                print()
                print("=" * 60)
                print(f"Layer {layer_name}")
                print("=" * 60)
            _train_probe_for_layer(
                payload,
                layer_name,
                probe_type=probe_type,
                args=args,
                label_names=label_names,
                train_ids=train_ids,
                eval_ids=eval_ids,
                train_split_names=train_split_names,
                eval_split_names=eval_split_names,
                split_file_path=split_file_path,
                outdir=outdir,
                seed=seed,
                draw=draw_index,
                draws_total=draws_total,
            )

    if multi_layer or draws_total > 1:
        print()
        print(f"Trained {len(layers)} layer(s) x {draws_total} draw(s)")


if __name__ == "__main__":
    main()
