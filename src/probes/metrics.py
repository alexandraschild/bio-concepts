"""Every number the probe evaluation reports, from ``(y, scores, meta)``.

Reported at image and identity units, at the probe's own threshold and at the
split-optimal one (a ceiling: it is chosen on the data it is scored on).
"""

from typing import Any, Iterable, Sequence

import numpy as np

from src.data.metadata import AGE_SEX_ORDER, SampleMeta
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)

#: Global metrics, in the order tables present them.
METRIC_KEYS = ("auroc", "accuracy", "balanced_accuracy")

#: Pretty names for table headers and axis labels.
METRIC_LABELS = {
    "auroc": "AUROC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced acc.",
    "cos": "Cosine similarity",
}

#: The value a metric takes when the probe carries no signal at all.
METRIC_CHANCE = {"auroc": 0.5, "balanced_accuracy": 0.5}

UNITS = ("image", "identity")

#: Operating points every evaluation reports. See the module docstring.
THRESHOLD_KINDS = ("probe", "best")


# --------------------------------------------------------------------------- #
# Global metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics for one score set at one operating point.

    scores must be the raw decision value: a sigmoid saturates to exactly 0/1
    on these logits and the ties drag AUROC down."""

    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    preds = (scores >= threshold).astype(np.int64)
    both_classes = np.unique(y).shape[0] == 2

    return {
        "auroc": float(roc_auc_score(y, scores, labels=[0, 1])) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, preds)) if both_classes else float("nan")
        ),
        "n": int(y.shape[0]),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """The threshold maximising balanced accuracy (Youden's J) on these scores."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(y).shape[0] < 2:
        return 0.0

    fpr, tpr, thresholds = roc_curve(y, scores)
    best = thresholds[int(np.argmax(tpr - fpr))]

    # roc_curve prepends an infinite threshold; clamp it so a degenerate split
    # cannot emit one that rejects every sample.
    if not np.isfinite(best):
        best = float(np.nextafter(float(np.max(scores)), np.inf))
    return float(best)


def aggregate(
    scores: np.ndarray, y: np.ndarray, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse to one score per unit. Mean, not sum, so a unit's size does not move it."""
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros(uniq.shape[0], dtype=np.float64)
    counts = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(sums, inverse, np.asarray(scores, dtype=np.float64))
    np.add.at(counts, inverse, 1.0)

    labels = np.full(uniq.shape[0], -1, dtype=np.int64)
    for pos, key in enumerate(uniq):
        member_labels = np.unique(np.asarray(y)[inverse == pos])
        if member_labels.shape[0] != 1:
            raise ValueError(
                f"unit {key!r} carries more than one label ({member_labels.tolist()}); "
                "the path-derived grouping and the labels disagree"
            )
        labels[pos] = int(member_labels[0])
    return uniq, sums / counts, labels


def global_rows(
    y: np.ndarray,
    scores: np.ndarray,
    meta: SampleMeta,
    *,
    probe_threshold: float,
) -> list[dict[str, Any]]:
    """One row per (unit, threshold kind). The threshold is fitted per unit --
    averaging to identity means changes the score distribution."""
    rows: list[dict[str, Any]] = []
    for unit in UNITS:
        if unit == "image":
            unit_scores, unit_y = np.asarray(scores, dtype=np.float64), np.asarray(y)
        else:
            _, unit_scores, unit_y = aggregate(scores, y, meta.group_keys(unit))

        for kind in THRESHOLD_KINDS:
            threshold = (
                probe_threshold
                if kind == "probe"
                else best_threshold(unit_y, unit_scores)
            )
            rows.append(
                {
                    "unit": unit,
                    "threshold_kind": kind,
                    "threshold": float(threshold),
                    "threshold_objective": "balanced_accuracy" if kind == "best" else "",
                    **binary_metrics(unit_y, unit_scores, threshold),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Age-sex breakdown
# --------------------------------------------------------------------------- #
def class_rows(
    y: np.ndarray,
    scores: np.ndarray,
    meta: SampleMeta,
    *,
    thresholds: dict[str, float],
    class_names: tuple[str, str],
    class_order: Sequence[str] = AGE_SEX_ORDER,
) -> list[dict[str, Any]]:
    """Accuracy per age-sex class, per binary label, and overall."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    def emit(kind: str, group: str, mask: np.ndarray) -> None:
        if not mask.any():
            return
        row: dict[str, Any] = {
            "group_kind": kind,
            "group": group,
            "n_identities": int(np.unique(meta.identity[mask]).shape[0]),
            "n_images": int(mask.sum()),
            "mean_logit": float(scores[mask].mean()),
            "median_logit": float(np.median(scores[mask])),
        }
        for kind_name, threshold in thresholds.items():
            preds = (scores[mask] >= threshold).astype(np.int64)
            row[f"accuracy_at_{kind_name}"] = float((preds == y[mask]).mean())
        rows.append(row)

    present = [c for c in class_order if (meta.age_sex == c).any()]
    present += sorted(set(meta.age_sex.tolist()) - set(class_order) - {""})
    for age_sex in present:
        emit("age_sex", age_sex, meta.age_sex == age_sex)

    for label_id, name in ((0, class_names[0]), (1, class_names[1])):
        emit("label", name, y == label_id)
    emit("all", "ALL", np.ones(y.shape[0], dtype=bool))
    return rows


# --------------------------------------------------------------------------- #
# Per-identity table (the graded axis itself)
# --------------------------------------------------------------------------- #
def identity_rows(
    y: np.ndarray,
    scores: np.ndarray,
    meta: SampleMeta,
    *,
    probe_threshold: float,
    class_names: tuple[str, str],
    seen_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for identity in sorted(set(meta.identity.tolist())):
        mask = meta.identity == identity
        member = scores[mask]
        labels = np.unique(y[mask])
        if labels.shape[0] != 1:
            raise ValueError(f"identity {identity} carries labels {labels.tolist()}")
        rows.append(
            {
                "identity": identity,
                "age_sex": str(meta.age_sex[mask][0]),
                "sex": class_names[int(labels[0])],
                "label": int(labels[0]),
                "seen": None if seen_ids is None else identity in seen_ids,
                "n_images": int(mask.sum()),
                "n_encounters": int(np.unique(meta.encounter[mask]).shape[0]),
                "mean_logit": float(member.mean()),
                # ddof=0: the spread of the images that exist, not an estimate for
                # a wider population of images of this individual.
                "sd_logit": float(member.std()),
                "median_logit": float(np.median(member)),
                "min_logit": float(member.min()),
                "max_logit": float(member.max()),
                "frac_above_threshold": float((member >= probe_threshold).mean()),
                "correct_at_threshold": float(
                    ((member >= probe_threshold).astype(np.int64) == labels[0]).mean()
                ),
            }
        )
    rows.sort(key=lambda r: r["mean_logit"])
    return rows


# --------------------------------------------------------------------------- #
# Direction comparison
# --------------------------------------------------------------------------- #
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, NaN if either vector is degenerate."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"directions have different dimensions: {a.shape} vs {b.shape}")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def mean_pairwise_cosine(directions: Iterable[np.ndarray]) -> float:
    mat = np.asarray(list(directions), dtype=np.float64)
    if mat.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.where(norms == 0.0, 1.0, norms)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(gram.shape[0], k=1)]
    return float(upper.mean())
