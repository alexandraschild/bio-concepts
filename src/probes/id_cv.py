"""Identity-wise cross-validated probe fitting: one fold per identity, so every
logit is out-of-sample."""

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed


@dataclass
class IdCvConfig:
    """A class-balanced C=1 logistic regression per fold."""

    standardize: bool
    max_iter: int

    def describe(self) -> dict[str, object]:
        return {"method": "logistic_regression", "C": 1.0, "class_weight_balanced": True,
                **asdict(self), "tol": 1e-4}


@dataclass
class IdCvResult:
    scores: np.ndarray  # (n_images,) out-of-sample decision value per image
    directions: np.ndarray  # (n_folds, d) fold directions, raw activation space
    fold_identities: np.ndarray  # (n_folds,) which identity each fold held out

    @property
    def mean_direction(self) -> np.ndarray:
        mean = self.directions.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean


def fit_direction(
    X_fit: np.ndarray, y_fit: np.ndarray, cfg: IdCvConfig
) -> tuple[np.ndarray, float]:
    """Fit one probe, returning (w, b) in RAW feature space: standardization is
    folded into the coefficients so a fold's scaler never travels with its direction."""
    X_fit = np.asarray(X_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.int64)

    if cfg.standardize:
        mean = X_fit.mean(axis=0)
        scale = X_fit.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
    else:
        mean = np.zeros(X_fit.shape[1], dtype=np.float64)
        scale = np.ones(X_fit.shape[1], dtype=np.float64)
    Z = (X_fit - mean) / scale

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=cfg.max_iter, tol=1e-4)
    model.fit(Z, y_fit)
    # sklearn orients coef_ toward classes_[1]; y is already 0/1 with 1 =
    # positive, so classes_ is [0, 1] and no reorientation is needed.
    w_z = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    b_z = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])

    w = w_z / scale
    return w, float(b_z - (mean / scale) @ w_z)


def id_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: IdCvConfig,
    *,
    n_jobs: int = 1,
    log: Callable[[str], None] = print,
) -> IdCvResult:
    """One out-of-sample score for every image, plus every fold's direction. A fold
    holds out one group: an identity, or a coarser level such as a zoo."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    unique_ids = np.unique(groups)

    def run_fold(held_out) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fit_mask = groups != held_out
        score_mask = groups == held_out
        w, b = fit_direction(X[fit_mask], y[fit_mask], cfg)
        return score_mask, X[score_mask] @ w + b, w

    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_fold)(identity) for identity in unique_ids
        )
    else:
        results = []
        for n, identity in enumerate(unique_ids, start=1):
            results.append(run_fold(identity))
            if n % 10 == 0 or n == unique_ids.shape[0]:
                log(f"  ID-CV fold {n}/{unique_ids.shape[0]}")

    scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    for score_mask, fold_scores, _ in results:
        scores[score_mask] = fold_scores
    if np.isnan(scores).any():
        raise RuntimeError("some images were never scored — identity bookkeeping is wrong")

    return IdCvResult(
        scores=scores,
        directions=np.stack([w for _, _, w in results]),
        fold_identities=np.asarray(unique_ids, dtype=object),
    )
