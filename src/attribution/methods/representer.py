"""Representer-point attribution: the probe's decision value, decomposed.

w = sum_i alpha_i x_i, so attribution_i(x) = alpha_i <x_i, x>. SVM: the saved
dual coefficient, sparse. LR: alpha_i = C c_i y_i sigma(-y_i f_i), dense.
"""

from typing import Any

import numpy as np
from scipy.special import expit

from src.data.features import log
from src.probes.bundle import ProbeBundle


def _sample_weights(bundle: ProbeBundle) -> np.ndarray:
    n = bundle.n_train
    if bundle.class_weight is None:
        return np.ones(n, dtype=np.float64)
    if bundle.class_weight != "balanced":
        raise ValueError(
            f"Unsupported class_weight {bundle.class_weight!r}; expected None or 'balanced'."
        )
    weights = np.ones(n, dtype=np.float64)
    classes, counts = np.unique(bundle.y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        weights[bundle.y_train == cls] = n / (len(classes) * count)
    return weights


def svm_alphas(bundle: ProbeBundle) -> tuple[np.ndarray, np.ndarray]:
    """``(train_positions, alpha)`` for the support vectors of an SVM probe."""
    return bundle.svm.support_train_positions, bundle.svm.dual_coef


def logistic_alphas(bundle: ProbeBundle) -> tuple[np.ndarray, np.ndarray]:
    """``(train_positions, alpha)`` for every training point of an LR probe."""
    y_pm = np.where(bundle.y_train == 1, 1.0, -1.0)
    margin = y_pm * bundle.decision(bundle.X_train)
    # sigma(-y*f) is the residual (1 - p_correct); expit(-m) keeps it finite where
    # 1 - expit(m) would cancel to exactly zero on confidently-fit points.
    alpha = bundle.C * _sample_weights(bundle) * y_pm * expit(-margin)
    return np.arange(bundle.n_train, dtype=np.int64), alpha


def representer_attribution(bundle: ProbeBundle) -> dict[str, Any]:
    if bundle.probe_type == "dualxda_svm":
        positions, alpha = svm_alphas(bundle)
        family = "dualxda_svm"
    elif bundle.probe_type == "logistic_regression":
        positions, alpha = logistic_alphas(bundle)
        family = "representer_lr"
    else:
        raise ValueError(
            f"Representer attribution is defined for dualxda_svm and "
            f"logistic_regression probes; got {bundle.probe_type!r}."
        )

    X_src = bundle.X_train[positions]

    # Exact for the SVM; for LR it holds only at a converged optimum with the
    # right C, so a large error means one of those is wrong.
    w_recon = alpha @ X_src
    w_norm = float(np.linalg.norm(bundle.w))
    rel_error = float(np.linalg.norm(w_recon - bundle.w) / max(w_norm, 1e-12))
    if rel_error > 1e-2:
        # sklearn's gtol is on the mean-normalised gradient and the residual here is
        # exactly -C*n times it, so C * n_train * tol is the slack the fit was allowed.
        tol = float(bundle.raw_payload["tol"])
        log(
            f"[{bundle.layer}] WARNING: the representer expansion reproduces w only to "
            f"relative error {rel_error:.3g}, so attribution magnitudes are off by about "
            f"that much. Fit at tol={tol:g}, C={bundle.C}, which permits "
            f"{bundle.C * bundle.n_train * tol:.3g} in |w| against ||w||={w_norm:.3g} -- "
            f"the fit is short of the optimum the identity needs."
        )

    # attribution[t, i] = alpha_i * <x_i, x_t>, oriented toward the positive class.
    attribution = (bundle.X_test @ X_src.T) * alpha[None, :]

    logit = bundle.decision(bundle.X_test)
    reconstructed_logit = attribution.sum(axis=1) + bundle.b
    logit_error = float(np.max(np.abs(reconstructed_logit - logit)))

    diagnostics = {
        "family": family,
        "n_sources": int(positions.shape[0]),
        "source_fraction": float(positions.shape[0] / bundle.n_train),
        "direction_reconstruction_rel_error": rel_error,
        "max_logit_reconstruction_error": logit_error,
        "alpha_abs_mean": float(np.abs(alpha).mean()),
        "alpha_abs_max": float(np.abs(alpha).max()),
    }
    log(
        f"[{bundle.layer}] {family}: {positions.shape[0]} sources "
        f"({diagnostics['source_fraction']:.1%} of train), "
        f"w recon rel.err={rel_error:.2e}, logit recon max.err={logit_error:.2e}"
    )
    return {
        "source_positions": positions,
        "alpha": alpha,
        "attribution": attribution,
        "diagnostics": diagnostics,
        "method": family,
    }
