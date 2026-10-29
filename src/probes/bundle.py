"""Load a probe, its split and its pooled features into one object.

y is always 0/1 with 1 = the probe's positive class.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.features import load_pooled_features
from src.data.metadata import SampleMeta, build_sample_meta
from src.paths import split_path_for

#: Probe types that carry a linear direction we can attribute through.
LINEAR_PROBE_TYPES = (
    "logistic_regression",
    "dualxda_svm",
    "difference_in_means",
)


@dataclass
class SVMDual:
    """The support-vector expansion the SVM probe already saved."""

    support_global_indices: np.ndarray  # (n_sv,) index into the activation payload
    support_train_positions: np.ndarray  # (n_sv,) row within the train split
    dual_coef: np.ndarray  # (n_sv,) signed y_i * alpha_i
    alphas: np.ndarray  # (n_sv,) |dual_coef|
    support_labels: np.ndarray  # (n_sv,) original label ids


@dataclass
class ProbeBundle:
    """A probe plus everything needed to attribute its predictions."""

    layer: str
    probe_type: str
    w: np.ndarray  # (d,) direction in the probe's fitting space
    b: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    standardize: bool
    class_labels: tuple[int, int]  # (negative, positive) original label ids
    class_names: tuple[str, str]
    decision_threshold: float
    class_weight: Any
    C: float | None

    X_train: np.ndarray  # (n_train, d) float64, already scaled
    y_train: np.ndarray  # (n_train,) 0/1, 1 = positive class
    train_meta: SampleMeta
    X_test: np.ndarray
    y_test: np.ndarray
    test_meta: SampleMeta

    probe_path: Path
    split_path: Path
    activations_path: Path
    feature_reduction: str
    subsample: str
    svm: SVMDual | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def dim(self) -> int:
        return int(self.w.shape[0])

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def scale(self, X_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(X_raw, dtype=np.float64) - self.scaler_mean) / self.scaler_scale

    def describe(self) -> dict[str, Any]:
        """Provenance block embedded into every attribution cache."""
        return {
            "layer": self.layer,
            "probe_type": self.probe_type,
            "probe_path": str(self.probe_path),
            "split_path": str(self.split_path),
            "activations_path": str(self.activations_path),
            "feature_reduction": self.feature_reduction,
            "standardize": self.standardize,
            "class_weight": self.class_weight,
            "C": self.C,
            "class_labels": list(self.class_labels),
            "class_names": list(self.class_names),
            "decision_threshold": self.decision_threshold,
            "subsample": self.subsample,
            "n_train_images": self.raw_payload.get("n_train_images", self.n_train),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "dim": self.dim,
            "n_support_vectors": None if self.svm is None else int(self.svm.dual_coef.shape[0]),
        }


def load_bundle(
    probe_path: Path | str,
    *,
    split_path: Path | str | None = None,
    activations_path: Path | str | None = None,
    feature_cache: Path | str | None = None,
) -> ProbeBundle:
    """Load probe + split + pooled features into a ProbeBundle. split_path and
    activations_path default to what the probe's manifest records."""
    probe_path = Path(probe_path)
    split_path = Path(split_path) if split_path is not None else split_path_for(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"No probe at {probe_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"No split manifest at {split_path}")

    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    with split_path.open("r", encoding="utf-8") as f:
        split_doc = json.load(f)
    split = split_doc["split"]

    probe_type = payload["probe_type"]
    if probe_type not in LINEAR_PROBE_TYPES:
        raise ValueError(
            f"{probe_path.name}: attribution needs a linear probe "
            f"({', '.join(LINEAR_PROBE_TYPES)}); this payload is {probe_type!r}."
        )

    # probe_tensor is [direction] for svm/dim but [direction, bias] for LR; the scaler
    # statistics are one per feature either way, so they say where the direction ends.
    tensor = payload["probe_tensor"].to(torch.float64).numpy().reshape(-1)
    bias = payload["bias"].to(torch.float64).numpy().reshape(-1)
    d = int(payload["scaler_mean"].reshape(-1).shape[0])
    w = tensor[:d]

    class_labels = payload["class_labels"].numpy().astype(np.int64)
    neg_label, pos_label = int(class_labels[0]), int(class_labels[-1])
    class_names = (str(payload["negative_class_name"]), str(payload["positive_class_name"]))

    activations_path = Path(
        activations_path
        if activations_path is not None
        else split_doc["source_activation_file"]
    )
    layer = payload["layer"]
    reduction = payload["feature_reduction"]

    feats = load_pooled_features(
        activations_path, layer, reduction, cache_dir=None if feature_cache is None else Path(feature_cache)
    )
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)

    scaler_mean = payload["scaler_mean"].to(torch.float64).numpy().reshape(-1)
    scaler_scale = payload["scaler_scale"].to(torch.float64).numpy().reshape(-1)
    X_train = (np.asarray(feats[train_idx], dtype=np.float64) - scaler_mean) / scaler_scale
    X_test = (np.asarray(feats[test_idx], dtype=np.float64) - scaler_mean) / scaler_scale

    # The manifest records carry the original label ids; re-orient so 1 == positive class.
    def _labels_from(records: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([int(r["label"]) for r in records], dtype=np.int64)

    train_records = split_doc["samples"]["train"]
    test_records = split_doc["samples"]["test"]
    y_train = (_labels_from(train_records) == pos_label).astype(np.int64)
    y_test = (_labels_from(test_records) == pos_label).astype(np.int64)

    train_meta = build_sample_meta(
        train_idx, [r["path"] for r in train_records], _labels_from(train_records)
    )
    test_meta = build_sample_meta(
        test_idx, [r["path"] for r in test_records], _labels_from(test_records)
    )

    svm = None
    if probe_type == "dualxda_svm":
        svm = SVMDual(
            support_global_indices=payload["svm_support_sample_indices"].numpy().astype(np.int64),
            support_train_positions=payload["svm_support_train_indices"].numpy().astype(np.int64),
            dual_coef=payload["svm_dual_coef"].to(torch.float64).numpy().reshape(-1),
            alphas=payload["svm_alphas"].to(torch.float64).numpy().reshape(-1),
            support_labels=payload["svm_support_labels"].numpy().astype(np.int64),
        )

    bundle = ProbeBundle(
        layer=layer,
        probe_type=probe_type,
        w=w,
        b=float(bias[0]),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        standardize=bool(payload["standardize"]),
        class_labels=(neg_label, pos_label),
        class_names=class_names,
        decision_threshold=float(payload["decision_threshold"]),
        class_weight=payload.get("class_weight"),
        C=payload.get("C"),
        X_train=X_train,
        y_train=y_train,
        train_meta=train_meta,
        X_test=X_test,
        y_test=y_test,
        test_meta=test_meta,
        probe_path=probe_path.resolve(),
        split_path=split_path.resolve(),
        activations_path=activations_path.resolve(),
        feature_reduction=reduction,
        subsample=str(payload["subsample"]),
        svm=svm,
        raw_payload=payload,
    )
    _sanity_check(bundle)
    return bundle


def _sanity_check(bundle: ProbeBundle) -> None:
    """Verify the loaded pieces reproduce the saved probe -- if not, every attribution
    computed from them is wrong."""
    if bundle.X_train.shape[1] != bundle.w.shape[0]:
        raise ValueError(
            f"Feature dim {bundle.X_train.shape[1]} != probe dim {bundle.w.shape[0]} "
            f"for {bundle.probe_path.name}"
        )
    if bundle.svm is None:
        return
    sv_rows = bundle.svm.support_train_positions
    if sv_rows.max(initial=-1) >= bundle.n_train:
        raise ValueError("Support-vector train positions fall outside the train split")
    w_dual = bundle.svm.dual_coef @ bundle.X_train[sv_rows]
    rel = float(np.linalg.norm(w_dual - bundle.w) / max(np.linalg.norm(bundle.w), 1e-12))
    if rel > 1e-3:
        raise ValueError(
            f"{bundle.probe_path.name}: the saved support vectors do not reproduce the "
            f"probe direction (relative error {rel:.3g}); the features loaded here are not "
            "the ones it was fit on."
        )
