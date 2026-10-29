"""Fit an SVM probe through the authors' DualXDA package, so the probe and its
attributions come from one fit.

Its patched LIBLINEAR (sklearn_dual) is the only thing that exposes the dual
variables. DualDA.train() runs verbatim; the one intervention is a monkeypatch
of the LinearSVC name it constructs, so the estimator can take class_weight/tol
and be kept afterwards (DualDA discards it, and with it the intercept).
"""

from dataclasses import dataclass
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def _require_package() -> tuple[Any, Any]:
    """The DualXDA package and the patched LIBLINEAR fork (`uv sync` installs both)."""
    import dualxda.dualda as dd
    from sklearn_dual.svm import LinearSVC

    return dd, LinearSVC


@dataclass
class DualXDAFit:
    """The result of one DualXDA SVM fit, in this repo's binary conventions."""

    w: np.ndarray  # (d,) margin direction, = sum_i dual_coef_i x_i
    b: float  # intercept
    active_positions: np.ndarray  # (n_active,) rows of the training matrix
    dual_coef: np.ndarray  # (n_active,) alpha[:,1] - alpha[:,0]
    alpha_matrix: np.ndarray  # (n_active, n_classes) the package's own coefficients
    classes: np.ndarray  # sklearn classes_, [negative, positive]
    reconstruction_rel_error: float
    cancellation_amplification: float
    decision_max_error: float


def fit_dualxda_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    class_weight: Any = None,
    tol: float = 1e-4,
    max_iter: int = 1_000_000,
) -> DualXDAFit:
    """Fit DualDA on pre-pooled features; return the binary dual expansion.

    tol must STAY at 1e-4 -- below ~1e-5 the CS solver decouples w from alpha,
    the opposite of the LR probes."""

    dd, base_linear_svc = _require_package()

    class IdentityFeatureModel(nn.Module):
        """Model whose hooked ``classifier`` layer receives the pooled feature."""

        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(x)

    class FeatureDataset(Dataset):
        """Returns (feature_vector, label) for DualDA's feature-collection loop."""

        def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
            self.features = features.to(torch.float32)
            self.labels = labels.to(torch.long)

        def __len__(self) -> int:
            return self.features.shape[0]

        def __getitem__(self, idx: int):
            return self.features[idx], int(self.labels[idx])

    captured: dict[str, Any] = {}

    class ParameterisedLinearSVC(base_linear_svc):
        """The package's own solver with class_weight/tol injected and the fitted
        estimator kept -- DualDA.train() runs unmodified otherwise."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("class_weight", class_weight)
            kwargs.setdefault("tol", tol)
            super().__init__(**kwargs)

        def fit(self, X, y):
            out = super().fit(X, y)
            captured["model"] = self
            return out

    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)

    # Fresh every fit: DualDA silently reuses a cached surrogate if one is there.
    cache_dir = tempfile.mkdtemp(prefix="dualda_")

    original_svc = dd.LinearSVC
    try:
        dd.LinearSVC = ParameterisedLinearSVC
        dataset = FeatureDataset(
            torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train)
        )
        da = dd.DualDA(
            model=IdentityFeatureModel(),
            dataset=dataset,
            classifier_layer="classifier",
            device="cpu",
            cache_dir=cache_dir,
            C=float(C),
            max_iter=int(max_iter),
        )
        try:
            estimator = captured["model"]
            active = da.active_indices.cpu().numpy().astype(np.int64)
            alpha_matrix = da.coefficients.cpu().numpy().astype(np.float64)
        finally:
            da.remove_hook()
    finally:
        dd.LinearSVC = original_svc
        shutil.rmtree(cache_dir, ignore_errors=True)

    classes = np.asarray(estimator.classes_)
    if classes.shape[0] != 2:
        raise ValueError(
            f"The DualXDA probe is defined for two classes; got {classes.tolist()}."
        )
    if alpha_matrix.shape[1] != 2:
        raise ValueError(
            f"Expected two coefficient columns from Crammer-Singer, got "
            f"{alpha_matrix.shape[1]}."
        )

    # Binary CS: sum_c alpha_ic = 0, so the margin coefficient is the difference.
    dual_coef = alpha_matrix[:, 1] - alpha_matrix[:, 0]
    b = float(np.asarray(estimator.intercept_).reshape(-1)[0])

    # coef_ is the source; the dual expansion is a heavily cancelling sum, so
    # recomputing it is a check, not a source.
    w = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)

    # The cancellation means the check must use exactly the values the solver
    # saw -- DualDA feeds it float32, and reconstructing from the unrounded
    # float64 features inflates the error by orders of magnitude.
    X_seen = X_train.astype(np.float32).astype(np.float64)
    reconstruction = dual_coef @ X_seen[active]
    rel_error = float(
        np.linalg.norm(reconstruction - w) / max(np.linalg.norm(w), 1e-30)
    )
    # How much cancellation; large values mean the attributions are soft.
    amplification = float(
        np.abs(dual_coef) @ np.linalg.norm(X_seen[active], axis=1)
        / max(np.linalg.norm(w), 1e-30)
    )
    # The check that matters: do the per-point contributions sum back to the
    # prediction? Sampled, since the full cross-correlation is (n, n_active).
    probe_rows = np.linspace(0, X_seen.shape[0] - 1, min(512, X_seen.shape[0])).astype(int)
    dual_decision = (X_seen[probe_rows] @ X_seen[active].T) @ dual_coef + b
    reference = estimator.decision_function(X_train[probe_rows]).reshape(-1)
    decision_error = float(
        np.max(np.abs(dual_decision - reference)) / max(np.abs(reference).max(), 1e-30)
    )

    # float32 round-off amplified by the cancellation is the error floor, so
    # judge the residual against that rather than a fixed threshold.
    eps32 = float(np.finfo(np.float32).eps)
    explained_by_roundoff = amplification * eps32
    if rel_error > max(1e-6, 50.0 * explained_by_roundoff):
        raise ValueError(
            f"The Crammer-Singer dual expansion does not reproduce the fitted direction "
            f"(relative error {rel_error:.3g}), so its dual variables cannot be used for "
            f"attribution. Cancellation is {amplification:.1f}x, which at float32 explains "
            f"only {explained_by_roundoff:.2g} of it. The usual cause is too tight a "
            f"tolerance: this was fit at tol={tol:g}, and MCSVM_CS keeps w and alpha in "
            f"sync only above ~1e-5. Use --dualxda-tol 1e-4."
        )

    return DualXDAFit(
        w=w,
        b=b,
        active_positions=active,
        dual_coef=dual_coef,
        alpha_matrix=alpha_matrix,
        classes=classes,
        reconstruction_rel_error=rel_error,
        cancellation_amplification=amplification,
        decision_max_error=decision_error,
    )


