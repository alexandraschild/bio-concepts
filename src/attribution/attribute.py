"""Compute data attributions (representer / exact-loo) for a trained probe."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.data.features import log
from src.paths import add_run_arg, run_of
from src.probes.bundle import load_bundle
from src.probes.discovery import probes_in_dir
from src.attribution.cache import AttributionCache
from src.attribution.methods import exact_loo_attribution, representer_attribution

METHODS = ("representer", "exact-loo")


def run_one(
    layer: str,
    probe_path: Path,
    args: argparse.Namespace,
    outroot: Path,
) -> dict[str, Any]:
    bundle = load_bundle(probe_path, feature_cache=args.feature_cache)

    if args.method == "representer":
        result = representer_attribution(bundle)
    else:
        result = exact_loo_attribution(bundle)

    source_meta = bundle.train_meta.select(result["source_positions"])
    logit = bundle.decision(bundle.X_test)
    pred = (logit >= bundle.decision_threshold).astype(np.int64)

    config: dict[str, Any] = {
        "probe": bundle.describe(),
        "method_params": result["diagnostics"],
        "source_level": "identity" if args.method == "exact-loo" else "image",
        "eval": {
            "eval_AUROC": float(roc_auc_score(bundle.y_test, logit, labels=[0, 1])),
            "eval_Accuracy": float(accuracy_score(bundle.y_test, pred)),
            "eval_BalancedAccuracy": float(balanced_accuracy_score(bundle.y_test, pred)),
        },
    }
    if args.method == "exact-loo":
        config["source_note"] = (
            "Each source is one identity; index/path name a representative image of it."
        )

    cache = AttributionCache(
        method=result["method"],
        layer=layer,
        attribution=result["attribution"],
        source_meta=source_meta,
        target_meta=bundle.test_meta,
        target_logit=logit,
        target_pred=pred,
        alpha=result.get("alpha"),
        config=config,
    )
    outdir = cache.save(outroot / layer)

    # Method-specific extras that do not fit the shared schema.
    extras = {
        k: result[k]
        for k in ("delta_w_norm", "unit_size")
        if k in result
    }
    if extras:
        np.savez_compressed(outdir / "extras.npz", **extras)

    log(f"[{layer}] wrote {result['method']} cache -> {outdir}")
    return {"layer": layer, "path": str(outdir), **result["diagnostics"], **config["eval"]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arg(ap)
    ap.add_argument("--probes", default=None,
                    help="Directory of probe_<layer>.pt files. Default: <run>/probes")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--layers", nargs="+", default=None, help="Default: every probe found")
    ap.add_argument("--out", default=None,
                    help="Default: <run>/attributions/<method>, else <probes>/../attributions/<method>")
    ap.add_argument(
        "--feature-cache",
        default=".cache/features",
        help="Directory for memoised pooled features (set empty to disable)",
    )
    args = ap.parse_args()
    if not args.probes and not args.run:
        ap.error("--probes or --run is required")
    return args


def main() -> None:
    args = parse_args()
    args.feature_cache = args.feature_cache or None

    run = run_of(args)
    probes_dir = Path(args.probes) if args.probes else run.probes
    if not probes_dir.is_dir():
        raise SystemExit(f"--probes must be a directory of probe_<layer>.pt files: {probes_dir}")

    probes = probes_in_dir(probes_dir, args.layers)

    if args.out:
        outroot = Path(args.out)
    elif run is not None:
        outroot = run.attributions(args.method)
    else:
        outroot = probes_dir.parent / "attributions" / args.method
    outroot.mkdir(parents=True, exist_ok=True)
    log(f"{args.method}: {len(probes)} layer(s) from {probes_dir} -> {outroot}")

    summary = [run_one(layer, probe_path, args, outroot) for layer, probe_path in probes]
    with (outroot / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"method": args.method, "layers": summary}, f, indent=2, default=str)
    log(f"wrote {outroot / 'summary.json'}")


if __name__ == "__main__":
    main()
