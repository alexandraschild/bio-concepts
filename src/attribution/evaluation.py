"""Metrics computed on attribution caches."""

from typing import Any

import numpy as np

from .cache import AttributionCache
from src.data.metadata import AGE_SEX_ORDER


def selection_mask(
    cache: AttributionCache,
    *,
    top_frac: float = 100.0,
    top_n: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Source mask by |alpha|. Only reads alpha when a selection is asked for, since
    the counterfactual methods have none and top-frac 100 must still work."""
    if top_n is None and top_frac >= 100.0:
        return np.ones(cache.n_src, dtype=bool)
    return select_top_sources(
        cache.source_weight("alpha"), top_frac=top_frac, top_n=top_n, seed=seed
    )


def select_top_sources(
    values: np.ndarray,
    *,
    top_frac: float = 100.0,
    top_n: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Mask of the largest-|values| sources, by mass fraction or fixed count.
    Ties break at random, seeded: bounded SVs all sit at exactly |alpha| = C."""
    a = np.abs(np.asarray(values, dtype=np.float64))
    if a.size == 0:
        return np.ones(0, dtype=bool)

    if top_n is not None:
        k = min(int(top_n), a.shape[0])
    elif top_frac >= 100.0 or a.sum() <= 0:
        return np.ones(a.shape[0], dtype=bool)
    else:
        cumulative = np.cumsum(np.sort(a)[::-1]) / a.sum()
        k = min(int(np.searchsorted(cumulative, top_frac / 100.0) + 1), a.shape[0])

    shuffled = np.random.default_rng(seed).permutation(a.shape[0])
    order = shuffled[np.argsort(-a[shuffled], kind="stable")]
    mask = np.zeros(a.shape[0], dtype=bool)
    mask[order[:k]] = True
    return mask


def composition(
    cache: AttributionCache,
    *,
    baseline_counts: dict[str, int] | None = None,
    weight: str = "count",
    top_frac: float = 100.0,
    top_n: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-class share, baseline share and enrichment of the attribution sources."""
    if weight == "count":
        w = np.ones(cache.n_src, dtype=np.float64)
    elif weight in ("alpha", "attribution"):
        w = cache.source_weight(weight)
    else:
        raise ValueError(f"weight must be count/alpha/attribution, got {weight!r}")

    keep = selection_mask(cache, top_frac=top_frac, top_n=top_n, seed=seed)
    classes = cache.source_meta.age_sex[keep]
    labels = np.asarray(cache.source_meta.label)[keep]
    identities = cache.source_meta.identity[keep]
    w = w[keep]

    total = float(w.sum())
    order = [c for c in AGE_SEX_ORDER if c in set(classes)] + sorted(
        {c for c in set(classes) if c not in AGE_SEX_ORDER}
    )
    baseline_total = float(sum(baseline_counts.values())) if baseline_counts else None

    # Each age-sex class sits entirely inside one label, so a class's label is
    # well defined and the within-label denominators are just per-label totals.
    class_label = {cls: int(labels[classes == cls][0]) for cls in order if (classes == cls).any()}
    label_total = {lab: float(w[labels == lab].sum()) for lab in np.unique(labels)}
    baseline_label_total: dict[int, float] = {}
    if baseline_counts is not None:
        for cls, count in baseline_counts.items():
            lab = class_label.get(cls)
            if lab is not None:
                baseline_label_total[lab] = baseline_label_total.get(lab, 0.0) + float(count)

    rows = []
    for cls in order:
        mask = classes == cls
        share = float(w[mask].sum() / total) if total > 0 else 0.0
        lab = class_label.get(cls)
        denom = label_total.get(lab, 0.0)
        row: dict[str, Any] = {
            "age_sex": cls,
            "label": lab,
            "n_sources": int(mask.sum()),
            "share": share,
            "share_within_label": float(w[mask].sum() / denom) if denom > 0 else float("nan"),
            "n_identities": int(np.unique(identities[mask]).shape[0]),
        }
        if baseline_counts is not None and baseline_total:
            base_share = baseline_counts.get(cls, 0) / baseline_total
            row["baseline_share"] = float(base_share)
            row["enrichment"] = float(share / base_share) if base_share > 0 else float("nan")
            base_denom = baseline_label_total.get(lab, 0.0)
            base_within = baseline_counts.get(cls, 0) / base_denom if base_denom > 0 else 0.0
            row["baseline_share_within_label"] = float(base_within)
            row["enrichment_within_label"] = (
                float(row["share_within_label"] / base_within) if base_within > 0 else float("nan")
            )
        rows.append(row)

    label_shares = {
        f"label_{int(lab)}_share": (float(tot / total) if total > 0 else 0.0)
        for lab, tot in label_total.items()
    }
    return {
        "weight": weight,
        "top_frac": float(top_frac) if top_n is None else "",
        "top_n": int(top_n) if top_n is not None else "",
        "n_selected": int(keep.sum()),
        "classes": rows,
        "label_split": label_shares,
    }


def per_identity_totals(
    cache: AttributionCache,
    level: str = "identity",
    weight: str = "attribution",
    *,
    top_frac: float = 100.0,
    top_n: int | None = None,
    seed: int = 0,
    pool_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Total signed/absolute weight per identity or encounter. Pass pool_counts, or
    the shares just measure who was photographed most."""
    index = cache.group_index(level)
    keep = selection_mask(cache, top_frac=top_frac, top_n=top_n, seed=seed)
    if weight == "count":
        per_source = keep.astype(np.float64)
    elif weight == "alpha":
        per_source = np.asarray(cache.alpha, dtype=np.float64) * keep
    else:
        per_source = cache.total_per_source(absolute=False) * keep
    uniq = index.keys
    signed = index.sum(per_source)
    absolute = index.sum(np.abs(per_source))
    counts = index.sum(keep.astype(np.float64))
    keys = cache.source_meta.group_keys(level)

    # Carry the class label along; it is constant within an identity by construction.
    first_class = {}
    for key, cls in zip(keys, cache.source_meta.age_sex):
        first_class.setdefault(key, cls)

    total_abs = float(np.abs(absolute).sum())
    rows = [
        {
            "key": str(k),
            "age_sex": first_class.get(k, ""),
            "weight": weight,
            "n_sources": int(c),
            "attribution_sum": float(s),
            "attribution_abs_sum": float(a),
            "share_of_total": float(s / total_abs) if total_abs > 0 else 0.0,
            "total_abs": total_abs,
        }
        for k, s, a, c in zip(uniq, signed, absolute, counts)
        if c > 0  # groups with nothing selected are not part of the set
    ]

    if pool_counts:
        # Over the whole pool, so a group absent from the expansion has a true
        # zero rate rather than a missing value.
        pool_total = float(sum(pool_counts.values()))
        per_image = {
            r["key"]: r["attribution_abs_sum"] / pool_counts[r["key"]]
            for r in rows
            if pool_counts.get(r["key"])
        }
        per_image_total = float(sum(per_image.values()))
        for r in rows:
            n_pool = pool_counts.get(r["key"])
            if not n_pool:
                continue
            pool_share = n_pool / pool_total
            r["n_pool"] = int(n_pool)
            r["pool_share"] = pool_share
            r["selection_rate"] = r["n_sources"] / n_pool
            r["mass_per_image"] = per_image[r["key"]]
            r["mass_per_image_share"] = (
                per_image[r["key"]] / per_image_total if per_image_total > 0 else 0.0
            )
            r["enrichment_vs_size"] = (
                abs(r["share_of_total"]) / pool_share if pool_share > 0 else 0.0
            )

    rows.sort(key=lambda r: -r["attribution_abs_sum"])
    return rows


#: Sample points along the cumulative curve, log-spaced: a sparse method's
#: curve is finished within the first 1% of the pool.
CURVE_POINTS = 60


def cumulative_curve(
    cache: AttributionCache,
    *,
    level: str = "image",
    max_targets: int | None = None,
) -> list[dict[str, Any]]:
    """Mean cumulative |attribution| mass vs. number of top sources per test point.
    The x-axis is an absolute count: a sparse cache stores only its support."""
    n_groups = cache.group_index(level).n_groups
    rows = cache.target_sample(max_targets)

    # The pool is every training unit the probe saw, not just the ones this
    # method kept -- that is what makes two methods' curves commensurable.
    n_train = cache.config["probe"]["n_train"]
    pool = int(n_train) if (level == "image" and n_train) else n_groups

    ks = np.unique(np.round(np.geomspace(1, max(pool, 2), CURVE_POINTS)).astype(np.int64))
    ks = ks[ks <= pool]
    inside = np.clip(ks, 1, n_groups)  # columns that exist in this cache

    n_rows = 0
    total = np.zeros(inside.shape[0], dtype=np.float64)
    total_sq = np.zeros(inside.shape[0], dtype=np.float64)
    for _, block in cache.iter_aggregated(level, rows=rows):
        a = np.abs(block)
        mass = a.sum(axis=1, keepdims=True)
        share = a / np.where(mass > 0, mass, 1.0)
        cumulative = np.cumsum(-np.sort(-share, axis=1), axis=1)
        picked = cumulative[:, inside - 1]
        total += picked.sum(axis=0)
        total_sq += (picked**2).sum(axis=0)
        n_rows += a.shape[0]

    mean = total / n_rows
    std = np.sqrt(np.maximum(total_sq / n_rows - mean**2, 0.0))
    return [
        {
            "layer": cache.layer,
            "method": cache.method,
            "level": level,
            "n_groups": int(n_groups),
            "pool": int(pool),
            "n_targets": int(n_rows),
            "k": int(k),
            "frac": float(k / pool),
            "mass_mean": float(m),
            "mass_std": float(s),
        }
        for k, m, s in zip(ks, mean, std)
    ]
