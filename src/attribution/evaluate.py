"""Turn src.attribution.attribute's caches into the CSVs src.report.attribution draws from."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.data.features import log
from src.paths import EVAL_DIRNAME, add_run_arg, run_of
from src.attribution.cache import AttributionCache, find_caches
from src.attribution.evaluation import (
    composition,
    cumulative_curve,
    per_identity_totals,
)
from src.data.metadata import GROUP_LEVELS, encounter_of, identity_of, load_age_sex_map


def _train_pool_paths(cache: AttributionCache) -> list[str]:
    with Path(cache.config["probe"]["split_path"]).open("r", encoding="utf-8") as f:
        return [r["path"] for r in json.load(f)["samples"]["train"]]


def _baseline_counts(cache: AttributionCache) -> dict[str, int] | None:
    """Age-sex counts of the full pool -- the base rate enrichment is measured against."""
    class_map = load_age_sex_map()
    counts: dict[str, int] = {}
    for path in _train_pool_paths(cache):
        cls = class_map.get(identity_of(path), "")
        counts[cls] = counts.get(cls, 0) + 1
    counts.pop("", None)
    return counts or None


def _pool_group_counts(cache: AttributionCache, level: str) -> dict[str, int]:
    """Training images per group. The cache cannot supply this: it holds only the
    support vectors, so selection_rate would be 1.0 by construction."""
    counts: dict[str, int] = {}
    for path in _train_pool_paths(cache):
        key = path if level == "image" else (
            encounter_of(path) if level == "encounter" else identity_of(path)
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _layer_sort_key(layer: str) -> tuple[int, str]:

    digits = re.findall(r"\d+", layer)
    return (int(digits[-1]) if digits else 1 << 30, layer)


def _caches_dir(args: argparse.Namespace) -> Path:
    return Path(args.caches) if args.caches else run_of(args).attributions(args.method)


def _eval_outdir(args: argparse.Namespace) -> Path:
    """One per method, never per layer: the CSVs carry a layer column."""
    if args.out:
        return Path(args.out)
    run = run_of(args)
    if run is not None:
        return run.attribution_eval(args.method)
    return _caches_dir(args) / EVAL_DIRNAME


def _load_all(root: Path) -> list[AttributionCache]:
    paths = find_caches(root)
    if not paths:
        raise SystemExit(f"No attribution caches under {root} (looking for config.json + attribution.npz)")
    caches = [AttributionCache.load(p) for p in paths]
    caches.sort(key=lambda c: _layer_sort_key(c.layer))
    return caches


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {path}")


def _identity_concentration(
    rows: list[dict[str, Any]], pool_counts: dict[str, int], ks: list[int]
) -> list[dict[str, Any]]:
    """Top-K mass share with its baselines on the row, since a raw share has to be
    read against the photo share those K animals own."""
    pool_total = float(sum(pool_counts.values())) or 1.0
    n_groups = len(pool_counts) or 1
    by_mass = sorted(rows, key=lambda r: -abs(r["share_of_total"]))
    by_rate = sorted(rows, key=lambda r: -r.get("mass_per_image_share", 0.0))
    photo_shares = sorted((n / pool_total for n in pool_counts.values()), reverse=True)

    out = []
    for k in ks:
        out.append(
            {
                "top_k": k,
                "n_pool_groups": n_groups,
                "n_present": len(rows),
                "mass_share": sum(abs(r["share_of_total"]) for r in by_mass[:k]),
                "mass_share_baseline": sum(photo_shares[:k]),
                "per_image_share": sum(r.get("mass_per_image_share", 0.0) for r in by_rate[:k]),
                "per_image_share_baseline": min(k, n_groups) / n_groups,
            }
        )
    return out


def cmd_composition(args: argparse.Namespace) -> None:
    caches = _load_all(_caches_dir(args))
    rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    conc_rows: list[dict[str, Any]] = []

    for cache in caches:
        baseline = _baseline_counts(cache)
        for weight in args.weights:
            if weight == "alpha" and cache.alpha is None:
                continue
            comp = composition(
                cache,
                baseline_counts=baseline,
                weight=weight,
                top_frac=args.top_frac,
                top_n=args.top_n,
                seed=args.seed,
            )
            shared = {
                "layer": cache.layer,
                "method": cache.method,
                "weight": weight,
                "top_frac": comp["top_frac"],
                "top_n": comp["top_n"],
                "n_selected": comp["n_selected"],
            }
            for entry in comp["classes"]:
                rows.append({**shared, **entry})
            label_rows.append({**shared, **comp["label_split"]})
        top_weight = args.top_weight
        if top_weight == "alpha" and cache.alpha is None:
            top_weight = "attribution"
        pool_counts = _pool_group_counts(cache, args.level)
        all_groups = per_identity_totals(
            cache,
            level=args.level,
            weight=top_weight,
            top_frac=args.top_frac,
            top_n=args.top_n,
            seed=args.seed,
            pool_counts=pool_counts,
        )
        for entry in all_groups[: args.top_identities]:
            identity_rows.append({"layer": cache.layer, "method": cache.method, **entry})
        # Only this stage sees every group; the CSV above is truncated to the top few.
        for entry in _identity_concentration(all_groups, pool_counts, args.concentration_k):
            conc_rows.append(
                {"layer": cache.layer, "method": cache.method, "weight": top_weight, **entry}
            )
        cache.release()

    outdir = _eval_outdir(args)
    _write_csv(outdir / "composition.csv", rows)
    _write_csv(outdir / "label_split.csv", label_rows)
    _write_csv(outdir / f"top_{args.level}.csv", identity_rows)
    _write_csv(outdir / f"{args.level}_concentration.csv", conc_rows)
    _print_composition(rows)
    _print_identity_concentration(conc_rows, args.level)


def _print_identity_concentration(rows: list[dict[str, Any]], level: str) -> None:
    if not rows:
        return
    print(f"\n=== {level} concentration, against the baseline that matters ===")
    print(
        f"{'layer':<12} {'K':>3} {'mass':>7} {'photo base':>11} {'ratio':>7} "
        f"| {'per-image':>10} {'base':>7} {'ratio':>7}"
    )
    for r in rows:
        mb, pb = r["mass_share_baseline"], r["per_image_share_baseline"]
        print(
            f"{r['layer']:<12} {r['top_k']:>3} {r['mass_share']:>7.3f} {mb:>11.3f} "
            f"{(r['mass_share'] / mb if mb else float('nan')):>7.2f} "
            f"| {r['per_image_share']:>10.3f} {pb:>7.3f} "
            f"{(r['per_image_share'] / pb if pb else float('nan')):>7.2f}"
        )


def _print_composition(rows: list[dict[str, Any]]) -> None:
    by_weight: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_weight.setdefault(r["weight"], []).append(r)
    for weight, entries in by_weight.items():
        print(f"\n=== composition (weight={weight}) ===")
        print(
            f"{'layer':<12} {'class':<14} {'n':>7} {'share':>8} {'base':>8} {'enrich':>8}"
            f" | {'in-label':>9} {'base':>8} {'enrich':>8}"
        )
        for r in entries:
            base, enrich = r.get("baseline_share"), r.get("enrichment")
            wbase, wenrich = r.get("baseline_share_within_label"), r.get("enrichment_within_label")
            print(
                f"{r['layer']:<12} {r['age_sex']:<14} {r['n_sources']:>7} "
                f"{r['share']:>8.3f} "
                f"{'—' if base is None else f'{base:>8.3f}'} "
                f"{'—' if enrich is None else f'{enrich:>8.2f}'}"
                f" | {r['share_within_label']:>9.3f} "
                f"{'—' if wbase is None else f'{wbase:>8.3f}'} "
                f"{'—' if wenrich is None else f'{wenrich:>8.2f}'}"
            )


def cmd_curve(args: argparse.Namespace) -> None:
    caches = _load_all(_caches_dir(args))
    rows: list[dict[str, Any]] = []
    for cache in caches:
        rows.extend(
            cumulative_curve(cache, level=args.level, max_targets=args.max_targets)
        )
        cache.release()

    outdir = _eval_outdir(args)
    _write_csv(outdir / f"cumulative_{args.level}.csv", rows)

    # A handful of round counts, as a table; the full grid is in the CSV.
    marks = [1, 10, 100, 1000]
    print(f"\n=== mean cumulative attribution mass in the top-k sources "
          f"(level={args.level}) ===")
    print(f"{'layer':<12}" + "".join(f"{'k=' + str(k):>10}" for k in marks) + f"{'n_src':>9}")
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    for layer, lrows in by_layer.items():
        cells = "".join(
            f"{min(lrows, key=lambda r: abs(r['k'] - k))['mass_mean']:>10.3f}" for k in marks
        )
        print(f"{layer:<12}{cells}{lrows[0]['n_groups']:>9}")


def cmd_loo(args: argparse.Namespace) -> None:
    caches = _load_all(_caches_dir(args))
    # A representer cache would run, but its "LOO" would not be a refit.
    if {c.method for c in caches} != {"exact_loo"}:
        raise SystemExit("loo needs an exact-loo cache")

    rows: list[dict[str, Any]] = []
    for cache in caches:
        keys, grouped = cache.aggregate("identity")
        logit = np.asarray(cache.target_logit, dtype=np.float64)
        y = np.asarray(cache.target_meta.label)
        full = {
            "auroc": float(roc_auc_score(y, logit, labels=[0, 1])),
            "accuracy": float(accuracy_score(y, (logit > 0).astype(int))),
            "balanced_accuracy": float(balanced_accuracy_score(y, (logit > 0).astype(int))),
        }
        first_class = {}
        for key, cls in zip(cache.source_meta.group_keys("identity"), cache.source_meta.age_sex):
            first_class.setdefault(key, cls)
        pool = _pool_group_counts(cache, "identity")

        for j, key in enumerate(keys):
            loo_logit = logit - grouped[:, j].astype(np.float64)
            pred = (loo_logit > 0).astype(int)
            auroc = float(roc_auc_score(y, loo_logit, labels=[0, 1]))
            rows.append(
                {
                    "layer": cache.layer,
                    "method": cache.method,
                    "level": "identity",
                    "key": str(key),
                    "age_sex": first_class.get(key, ""),
                    "n_pool": pool.get(str(key), ""),
                    "full_auroc": full["auroc"],
                    "loo_auroc": auroc,
                    "delta_auroc": full["auroc"] - auroc,
                    "delta_accuracy": full["accuracy"] - float(accuracy_score(y, pred)),
                    "delta_balanced_accuracy": full["balanced_accuracy"]
                    - float(balanced_accuracy_score(y, pred)),
                    "mean_abs_delta_logit": float(np.abs(grouped[:, j]).mean()),
                    "std_delta_logit": float(np.std(grouped[:, j])),
                }
            )
        cache.release()

    outdir = _eval_outdir(args)
    _write_csv(outdir / "loo_metrics_identity.csv", rows)

    print("\n=== leave-one-identity-out damage ===")
    print(
        f"{'layer':<12} {'full':>7} {'worst dAUROC':>13} {'unit':>10} "
        f"{'mean|d|':>9} {'n>0.005':>8} {'top10 share':>12}"
    )
    for layer in sorted({r["layer"] for r in rows}, key=_layer_sort_key):
        g = [r for r in rows if r["layer"] == layer]
        worst = max(g, key=lambda r: r["delta_auroc"])
        mags = sorted((abs(r["delta_auroc"]) for r in g), reverse=True)
        total = sum(mags)
        print(
            f"{layer:<12} {g[0]['full_auroc']:>7.4f} {worst['delta_auroc']:>+13.4f} "
            f"{worst['key']:>10} {total / len(g):>9.4f} "
            f"{sum(1 for r in g if r['delta_auroc'] > 0.005):>8d} "
            f"{(sum(mags[:10]) / total if total else 0.0):>12.3f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        add_run_arg(p)
        p.add_argument("--caches", default=None,
                       help="Attribution method directory. Default: <run>/attributions/<method>")
        p.add_argument("--method", default=None, choices=("representer", "exact-loo"),
                       help="Which method under --run to evaluate (only needed with --run)")
        p.add_argument("--out", default=None, help="Default: <caches>/eval")

    comp = sub.add_parser("composition", help="Age-sex composition of attribution sources")
    _common(comp)
    comp.add_argument(
        "--weights",
        nargs="+",
        default=["count", "attribution"],
        choices=["count", "alpha", "attribution"],
        help="How to weight each source (default: count and attribution)",
    )
    select = comp.add_mutually_exclusive_group()
    select.add_argument(
        "--top-frac",
        type=float,
        default=100.0,
        help="Percent of total |alpha| the described sources carry: 90 = the k90 "
        "set, 100 (default) = the whole expansion.",
    )
    select.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Exactly N sources by |alpha| instead of a mass fraction.",
    )
    comp.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Tie-break seed (bounded SVs all sit at |alpha| = C exactly).",
    )
    comp.add_argument("--level", default="identity", choices=GROUP_LEVELS)
    comp.add_argument("--top-identities", type=int, default=20)
    comp.add_argument(
        "--concentration-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="Top-K sizes for <level>_concentration.csv.",
    )
    comp.add_argument(
        "--top-weight",
        default="alpha",
        choices=["count", "alpha", "attribution"],
        help="Weight for the top-<level> table; attribution loads the dense matrix (~900 "
             "MB/layer).",
    )
    comp.set_defaults(func=cmd_composition)

    curve = sub.add_parser(
        "curve",
        help="Cumulative attribution mass vs. fraction of top sources (DualXDA Fig. 5)",
    )
    _common(curve)
    curve.add_argument("--level", default="image", choices=GROUP_LEVELS)
    curve.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Subsample test points; the curve is a mean over them (default: all)",
    )
    curve.set_defaults(func=cmd_curve)

    loo = sub.add_parser("loo", help="Eval-metric damage from removing each identity")
    _common(loo)
    loo.set_defaults(func=cmd_loo)

    args = ap.parse_args()
    if not args.caches and not (args.run and args.method):
        ap.error("--caches, or --run together with --method, is required")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
