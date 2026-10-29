"""Steering figures from the JSONs src.steering.evaluate_reid wrote."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from src.report import tables as T
from src.report.style import (
    add_style_args,
    finish_layer_axes,
    line_kw,
    list_palettes,
    place_legend,
    ref_kw,
    report_written,
    save_figure,
    style_from_args,
    zoom_ylim,
)

# Below src.report.style: it sets the Agg backend before pyplot is imported.
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

_FNAME_RE = re.compile(r"eval_layer(\d+)_k(\d+)\.json$")

FLIP_LABEL = "Predictions flipped (%)"
OFF_MANIFOLD_LABEL = "Queries outside manifold (%)"
#: The cost axis is drawn inverted, so "up" is good on both sides of a twin figure.
_COST_DASH = (0, (3.5, 1.8))
_COST_REF_DASH = (0, (5.0, 2.5))


@dataclass
class SteerEval:
    """One sweep's eval/ directory: a record per steered (layer, K, alpha), and the
    alpha=0 row per K."""

    path: Path
    records: list[dict[str, Any]]
    baselines: dict[int, dict[str, Any]]

    def at(self, k: int, layer: int) -> list[dict[str, Any]]:
        if k not in self.baselines:
            raise SystemExit(f"{self.path}: no K={k} (has: {sorted(self.baselines)})")
        return sorted((r for r in self.records if r["K"] == k and r["layer"] == layer),
                      key=lambda r: r["alpha"])


def load_steer_eval(entry: str) -> SteerEval:
    """Accepts the eval/ dir, its sweep root, or the run root above that."""
    p = Path(entry)
    eval_dir = next((p / sub for sub in (".", "eval", "steering/eval")
                     if next((p / sub).glob("eval_layer*_k*.json"), None) is not None), None)
    if eval_dir is None:
        raise SystemExit(f"no eval_layer*_k*.json in {p}, {p}/eval or {p}/steering/eval")

    records: list[dict[str, Any]] = []
    baselines: dict[int, dict[str, Any]] = {}
    for path in sorted(eval_dir.glob("eval_layer*_k*.json")):
        layer, k = (int(g) for g in _FNAME_RE.search(path.name).groups())
        for r in json.loads(path.read_text())["per_alpha"]:
            row = {
                "layer": layer, "K": k, "alpha": float(r["alpha"]),
                "flip_toward": r["flip_toward_concept_pct"],
                "off_manifold": r["off_manifold_pct"],
            }
            if row["alpha"] == 0.0:
                baselines[k] = row
            else:
                records.append(row)
    return SteerEval(eval_dir, records, baselines)


# --------------------------------------------------------------------------- #
# layers -- flip rate against its cost over depth, at one dose
# --------------------------------------------------------------------------- #
def _percent_ylim(series: Sequence[Sequence[float]], full_range: bool) -> tuple[float, float]:
    if full_range:
        return 0.0, 100.0
    lo, hi = zoom_ylim([[v / 100 for v in s] for s in series], None, False, (0.0, 1.0))
    return lo * 100, hi * 100


def cmd_layers(args: argparse.Namespace) -> None:
    ev = load_steer_eval(args.evals)
    style = style_from_args(args)
    points = [r for layer in sorted({r["layer"] for r in ev.records})
              for r in ev.at(args.k, layer) if r["alpha"] == args.alpha]
    if not points:
        raise SystemExit(f"{ev.path}: no records at alpha={args.alpha:g}, K={args.k}")
    xs = [p["layer"] for p in points]
    flip = [p["flip_toward"] for p in points]
    off = [p["off_manifold"] for p in points]
    off_base = ev.baselines[args.k]["off_manifold"]

    colors = style.series(2)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.axhline(off_base, ls=_COST_REF_DASH, lw=0.7, color=style.c["muted"], zorder=2)
    ax.plot(xs, flip, **line_kw(colors[0], style), label="flips toward")
    ax2.plot(xs, off, **{**line_kw(colors[1], style), "ls": _COST_DASH, "marker": "s"},
             label="off-manifold")
    ax.set_ylim(*_percent_ylim([flip, []], args.full_range))
    lo, hi = _percent_ylim([off, [off_base] * len(off)], args.full_range)
    ax2.set_ylim(hi, lo)

    ax.set_xlabel("Layer")
    ax.set_ylabel(FLIP_LABEL, color=colors[0])
    ax2.set_ylabel(OFF_MANIFOLD_LABEL, color=colors[1])
    ax.tick_params(axis="y", labelcolor=colors[0])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    finish_layer_axes(ax, xs, style, clamp=(0.0, 100.0))
    ax2.grid(False)
    lo, hi = sorted(ax2.get_ylim())
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax2.set_yticks([t for t in ax2.get_yticks() if max(lo, 0.0) <= t <= min(hi, 100.0)])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(style.c["axis"])
    ax2.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h2 + [plt.Line2D([], [], ls=_COST_REF_DASH, lw=0.7, color=style.c["muted"])]
    labels += l2 + [f"off-manifold: baseline ({off_base:.1f})"]
    place_legend(ax, handles, labels, style, y=-0.30, max_cols=2)
    report_written(save_figure(fig, Path(args.out) / args.name, args.format))


# --------------------------------------------------------------------------- #
# dose -- one direction against its random control, over alpha at one layer
# --------------------------------------------------------------------------- #
def _series(evals: Sequence[SteerEval], k: int, layer: int, metric: str
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(alphas, mean, sd) over one or more sweeps -- several means seeds of one series.
    Steered alphas only: a log axis has no place for 0."""
    rows = [ev.at(k, layer) for ev in evals]
    alphas = np.array([r["alpha"] for r in rows[0]])
    values = np.array([[r[metric] for r in one] for one in rows], dtype=float)
    return alphas, values.mean(axis=0), values.std(axis=0)


def cmd_dose(args: argparse.Namespace) -> None:
    ev = load_steer_eval(args.evals)
    control = [load_steer_eval(c) for c in args.control]
    style = style_from_args(args)
    k, layer = args.k, args.layer

    alphas, flip, _ = _series([ev], k, layer, "flip_toward")
    _, ctl, ctl_sd = _series(control, k, layer, "flip_toward")
    _, off, _ = _series([ev], k, layer, "off_manifold")
    off_base = ev.baselines[k]["off_manifold"]

    colors = style.series(2)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(alphas, flip, **line_kw(colors[0], style), label=args.label)
    ax.plot(alphas, ctl, **{**line_kw(colors[0], style), "ls": (0, (1, 1.6)), "marker": "^"},
            label="random control")
    if ctl_sd.any():
        ax.fill_between(alphas, ctl - ctl_sd, ctl + ctl_sd, color=colors[0], alpha=0.15, lw=0)
    ax2.plot(alphas, off, **{**line_kw(colors[1], style), "ls": _COST_DASH, "marker": "s"},
             label=f"off-manifold ({args.label})")
    ax2.axhline(off_base, **ref_kw(style))

    ax.set_xscale("log")
    ax.set_xlim(alphas.min() / 1.15, alphas.max() * 1.15)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator(list(alphas)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel("Steering scale (alpha)")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(style.c["axis"])
    ax.set_ylim(0, None)
    ax.set_ylabel(FLIP_LABEL, color=colors[0])
    ax.tick_params(axis="y", labelcolor=colors[0])

    ax2.set_ylabel(OFF_MANIFOLD_LABEL, color=colors[1])
    ax2.set_ylim(100, 0)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(style.c["axis"])
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", labelcolor=colors[1])

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h2 + [plt.Line2D([], [], **ref_kw(style))]
    labels += l2 + [f"off-manifold baseline ({off_base:.1f}%)"]
    place_legend(ax, handles, labels, style, y=-0.34, max_cols=2)

    stem = Path(args.out) / args.name
    written = save_figure(fig, stem, args.format)
    rows = [{"alpha": a, "flip_toward": f, "control_flip_toward": c,
             "control_sd": sd, "off_manifold": o}
            for a, f, c, sd, o in zip(alphas, flip, ctl, ctl_sd, off)]
    cols = [T.Column("alpha", "alpha", 3, "r")] + [
        T.Column(key, key.replace("_", " "), 2) for key in
        ("flip_toward", "control_flip_toward", "control_sd", "off_manifold")
    ]
    written += T.write(rows, cols, stem,
                       title=f"{args.label} against random control, layer {layer}, K={k}.")
    report_written(written)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    lay = sub.add_parser("layers", help="Flip rate against off-manifold, over depth")
    lay.add_argument("--alpha", required=True, type=float, help="Dose each layer is drawn at.")
    lay.set_defaults(func=cmd_layers)

    dose = sub.add_parser("dose", help="Flip rate against a random control, over dose")
    dose.add_argument("--layer", type=int, required=True)
    dose.add_argument("--control", nargs="+", required=True,
                      help="Control sweeps from src.steering.steer_controls, averaged as "
                           "seeds of one series.")
    dose.add_argument("--label", required=True, help="Legend name of the direction.")
    dose.set_defaults(func=cmd_dose)

    for p, name in ((lay, "steering_by_layer"), (dose, "steering_by_alpha")):
        p.add_argument("--evals", required=True,
                       help="Steering sweep: its eval/ dir, the sweep root or the run root.")
        p.add_argument("--k", type=int, default=5, help="Neighbour count (default: 5).")
        p.add_argument("--out", required=True)
        p.add_argument("--name", default=name,
                       help=f"Output filename minus the extension (default: {name})")
        add_style_args(p)

    args = ap.parse_args()
    if args.list_palettes:
        list_palettes()
        raise SystemExit(0)
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
