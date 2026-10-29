"""Exact leave-one-identity-out attribution: drop an identity, refit, measure the logit shift."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.features import log
from src.probes.bundle import ProbeBundle

#: Tight by design: an LOO delta is a difference of two fits.
REFIT_TOL = 1e-10


def _fit_probe(bundle: ProbeBundle, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Refit the logistic-regression probe on (X, y) with its own hyperparameters."""
    clf = LogisticRegression(
        C=bundle.C,
        max_iter=int(bundle.raw_payload["max_iter"]),
        tol=REFIT_TOL,
        solver="lbfgs",
        class_weight=bundle.class_weight,
    )
    clf.fit(X, y)
    # y is already 0/1 with 1 = positive class, so classes_ == [0, 1] and
    # decision_function points toward the positive class, as the bundle's does.
    return clf.coef_.reshape(-1).astype(np.float64), float(clf.intercept_.reshape(-1)[0])


def exact_loo_attribution(bundle: ProbeBundle) -> dict[str, Any]:
    """Refit once per held-out identity and record the test-logit shift."""
    keys = bundle.train_meta.group_keys("identity")
    units = np.unique(keys)
    n_units = int(units.shape[0])

    # From a refit, not the saved probe: a tolerance difference would leak into
    # every LOO delta as a constant offset.
    w_full, b_full = _fit_probe(bundle, bundle.X_train, bundle.y_train)
    logit_full = bundle.X_test @ w_full + b_full

    refit_gap = float(np.max(np.abs(logit_full - bundle.decision(bundle.X_test))))
    log(
        f"[{bundle.layer}] exact LOO: {n_units} refits on {bundle.n_train} rows at "
        f"tol={REFIT_TOL:g} (refit vs saved probe max logit gap {refit_gap:.3g})"
    )

    attribution = np.zeros((bundle.n_test, n_units), dtype=np.float64)
    delta_w_norm = np.zeros(n_units, dtype=np.float64)
    unit_size = np.zeros(n_units, dtype=np.int64)
    kept_positions: list[int] = []

    for u, unit in enumerate(units):
        mask = keys != unit
        unit_size[u] = int((~mask).sum())
        w_i, b_i = _fit_probe(bundle, bundle.X_train[mask], bundle.y_train[mask])
        attribution[:, u] = logit_full - (bundle.X_test @ w_i + b_i)
        delta_w_norm[u] = float(np.linalg.norm(w_i - w_full))
        # Representative row for this unit, so the cache's source metadata lines up.
        kept_positions.append(int(np.flatnonzero(~mask)[0]))
        if (u + 1) % 20 == 0:
            log(f"[{bundle.layer}]   {u + 1}/{n_units} refits")

    diagnostics = {
        "family": "exact_loo",
        "level": "identity",
        "tol": REFIT_TOL,
        "n_units": n_units,
        "refit_vs_saved_max_logit_gap": refit_gap,
        "unit_size_mean": float(unit_size.mean()),
        "delta_w_norm_mean": float(delta_w_norm.mean()),
        "delta_w_norm_max": float(delta_w_norm.max()),
        "abs_attribution_mean": float(np.abs(attribution).mean()),
    }
    return {
        "source_positions": np.asarray(kept_positions, dtype=np.int64),
        "source_keys": units,
        "alpha": None,
        "attribution": attribution,
        "delta_w_norm": delta_w_norm,
        "unit_size": unit_size,
        "diagnostics": diagnostics,
        "method": "exact_loo",
    }
