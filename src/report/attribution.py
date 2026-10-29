"""Attribution figures from the CSVs src.attribution.evaluate wrote."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.metadata import AGE_SEX_ORDER, GROUP_LEVELS
from src.paths import EVAL_DIRNAME
from src.probes.discovery import layer_index, normalize_layer, sort_layers
from src.report.style import (
    add_style_args,
    DASHES,
    finish_layer_axes,
    line_kw,
    list_palettes,
    place_legend,
    ref_kw,
    report_written,
    save_figure,
    slug,
    Style,
    style_from_args,
    zoom_ylim,
)

# Below src.report.style: it sets the Agg backend before pyplot is imported.
import matplotlib.pyplot as plt


INF = float("inf")

_STRING_COLS = frozenset(
    {"layer", "method", "weight", "age_sex", "key", "level"}
)


@dataclass
class AttribEval:
    """One ``eval/`` directory of CSVs, plus a label for its method dir."""

    path: Path
    label: str

    def read(self, name: str) -> list[dict[str, Any]]:
        return _read_csv(self.path / name)

    def has(self, name: str) -> bool:
        return (self.path / name).exists()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"missing {path.name} under {path.parent} — run the matching "
            f"src.attribution.evaluate command first."
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            if k in _STRING_COLS:
                continue
            row[k] = None if v in ("", None) else float(v)
    return rows


def _resolve_eval_dir(entry: str) -> Path:
    """The eval/ dir, or the method dir holding it."""
    p = Path(entry)
    if not p.is_dir():
        raise SystemExit(f"no such directory: {p}")
    return p / EVAL_DIRNAME if (p / EVAL_DIRNAME).is_dir() else p


def _label_fields(d: Path) -> tuple[str | None, str, str | None]:
    """(probe, method, layer) read off the <probe>/attributions/<method>/<layer> layout."""
    parts = [p for p in d.parts if p not in ("/", EVAL_DIRNAME)]
    if "attributions" in parts[:-1]:
        i = len(parts) - 1 - parts[::-1].index("attributions")
        probe = parts[i - 1] if i else None
        method, *rest = parts[i + 1:]
        return probe, method, "/".join(rest) or None
    return None, parts[-1], None


def load_attrib_evals(entries: Sequence[str],
                      labels: Sequence[str] | None = None) -> list[AttribEval]:
    """The --evals directories, each carrying the name its curves will use."""
    dirs = [_resolve_eval_dir(e) for e in entries]
    if labels:
        if len(labels) != len(dirs):
            raise SystemExit(
                f"--label needs one name per --evals cache ({len(dirs)}), "
                f"got {len(labels)}"
            )
        return [AttribEval(d, lab) for d, lab in zip(dirs, labels)]
    # Name each cache by the fields that differ on this figure; a shared
    # component is legend noise. One cache falls back to the method.
    fields = [_label_fields(d) for d in dirs]
    varying = [i for i in range(3) if len({f[i] for f in fields}) > 1]
    raw = [
        "/".join(str(f[i]) for i in varying if f[i] is not None) or f[1]
        for f in fields
    ]
    return [AttribEval(d, lab) for d, lab in zip(dirs, raw)]


def _outdir(args: argparse.Namespace, evals: Sequence[AttribEval]) -> Path:
    if args.out:
        return Path(args.out)
    return evals[0].path.parent / "reports"


def _ordered_classes(present: set[str]) -> list[str]:
    known = [c for c in AGE_SEX_ORDER if c in present]
    return known + sorted(present - set(AGE_SEX_ORDER))


def _class_colors(present: Sequence[str], style: Style) -> tuple[dict[str, str], list[str]]:
    classes = _ordered_classes(set(present))
    palette = style.series(len(classes))
    return {c: palette[i] for i, c in enumerate(classes)}, classes


def _ylimits(series, ref, full_range: bool, limits) -> tuple[float, float]:
    """Zoom to the data; an unbounded metric has no full range to force."""
    if full_range and np.isfinite(limits[1]):
        return limits
    return zoom_ylim(series, ref, full_range=False, limits=limits)


def _pool_rule(style: Style) -> dict:
    """Dashed rule for the pool ceiling -- heavier than ref_kw's dotted null."""
    return dict(ref_kw(style), ls=DASHES[1])


def _finish_curve(ax, xs, series, style: Style, *, ref, full_range: bool, limits) -> None:
    ax.set_ylim(*_ylimits(series, ref, full_range, limits))
    clamp = limits if np.isfinite(limits[1]) else None
    finish_layer_axes(ax, xs, style, clamp=clamp)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def cmd_composition(args: argparse.Namespace) -> None:
    """Count-weighted enrichment of each --age-sex class over depth, one curve per
    (class, --evals): colour = class, dash = method."""
    evals = load_attrib_evals(args.evals, args.label)
    style = style_from_args(args)
    classes = _ordered_classes(set(args.age_sex))
    # One class => methods take the palette hues; several => colour is the class
    # and dash the method, so the eye pairs a class's two curves.
    per_class_hue = len(classes) > 1
    colors = _class_colors(classes, style)[0] if per_class_hue else {}
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    marks: list[dict] = []
    drawn: list[tuple[str, int]] = []  # (class, method index) behind the key
    for cls in classes:
        for i, ev in enumerate(evals):
            rows = [r for r in ev.read("composition.csv")
                    if r["weight"] == "count" and r["age_sex"] == cls]
            if not rows:
                continue
            sources.append((f"{ev.label} · {cls}" if per_class_hue else ev.label, rows))
            if per_class_hue:
                marks.append({**line_kw(colors[cls], style),
                              "ls": DASHES[i % len(DASHES)] if i else "-"})
                drawn.append((cls, i))
    if not sources:
        raise SystemExit(f"no count-weighted rows for {args.age_sex} in any --evals")

    fig, ax = plt.subplots()
    ax.axhline(1.0, **ref_kw(style))
    series: list[list[float | None]] = []
    all_x: list[int] = []
    for i, (label, rows) in enumerate(sources):
        srows = sorted(rows, key=lambda r: layer_index(r["layer"]))
        ys = [r.get("enrichment") for r in srows]
        series.append(ys)
        xs = [layer_index(r["layer"]) for r in srows]
        all_x += xs
        kw = marks[i] if marks else style.series_kw(i, len(sources))
        ax.plot(xs, ys, **kw, label=label)
    ax.set_ylabel("Enrichment")
    _finish_curve(ax, all_x, series, style, ref=1.0, full_range=args.full_range,
                  limits=(0.0, INF))
    ax.set_xlabel("Layer")
    handles, labels = (
        _class_method_key(drawn, classes, colors, evals, style)
        if per_class_hue else ax.get_legend_handles_labels()
    )
    place_legend(ax, handles, labels, style)
    report_written(save_figure(fig, _outdir(args, evals) / args.name, args.format))


def _class_method_key(drawn: Sequence[tuple[str, int]], classes: Sequence[str],
                      colors: dict[str, str], evals, style: Style):
    """Legend keys for the class x method cross product: colour = class, dash = method."""
    handles = [plt.Line2D([], [], color=colors[c], lw=1.8)
               for c in classes if any(c == d for d, _ in drawn)]
    labels = [c for c in classes if any(c == d for d, _ in drawn)]
    for i, ev in enumerate(evals):
        if not any(i == m for _, m in drawn):
            continue
        handles.append(plt.Line2D([], [], color=style.c["ink_soft"], lw=1.3,
                                  ls=DASHES[i % len(DASHES)] if i else "-"))
        labels.append(ev.label)
    return handles, labels


#: What each --mode selects on the signed ``attribution_sum``.
_TOP_MODES: dict[str, str] = {
    "abs": "top |attribution|",
    "positive": "top proponents",
    "negative": "top opponents",
}


def cmd_top(args: argparse.Namespace) -> None:
    evals = load_attrib_evals(args.evals, args.label)
    style = style_from_args(args)
    outdir = _outdir(args, evals)
    written: list[Path] = []

    fname = f"top_{args.level}.csv"
    for ev in evals:
        esuffix = f"_{slug(ev.label)}" if len(evals) > 1 else ""
        rows = ev.read(fname)
        layers = _pick_layers(rows, args.layer)
        for layer in layers:
            lsuffix = f"_{layer}" if len(layers) > 1 else ""
            lrows = [r for r in rows if r["layer"] == layer]
            shown = _select_top(lrows, args.mode, args.top_n, _top_column(args))
            if not shown:
                print(f"{ev.label} [{layer}]: no {args.mode} sources to show")
                continue
            if len(shown) < args.top_n:
                print(f"{ev.label} [{layer}]: only {len(shown)} {args.mode} source(s) "
                      f"available (top_{args.level} keeps the top {len(lrows)} by "
                      f"absolute mass, so a one-sided view can be shorter than --top-n).")
            fig = _draw_top(shown, style, args)
            stem = outdir / f"{args.name}{esuffix}{lsuffix}"
            written += save_figure(fig, stem, args.format)

    report_written(written)


def _select_top(rows, mode: str, k: int, column: str = "attribution_sum") -> list[dict[str, Any]]:
    """The K sources *mode* wants, most extreme first, ranked on the drawn column."""
    rows = [r for r in rows if r.get(column) is not None]
    if mode == "abs":
        ranked = sorted(rows, key=lambda r: -abs(r[column]))
    elif mode == "positive":
        ranked = sorted((r for r in rows if r[column] > 0), key=lambda r: -r[column])
    else:  # negative
        ranked = sorted((r for r in rows if r[column] < 0), key=lambda r: r[column])
    return ranked[:k]


#: Columns ``--value`` can draw, with axis label and percent flag. Prefer the
#: size-free ones: share_of_total mostly measures who was photographed most.
_VALUE_COLUMNS: dict[str, tuple[str, bool]] = {
    "attribution_sum": ("Attribution sum", False),
    "share_of_total": ("Share of total attribution mass", True),
    "selection_rate": ("Share of ID's images in set", True),
    "mass_per_image": ("Attribution mass per photo", False),
    "mass_per_image_share": ("Share of total per-photo mass", True),
    "enrichment_vs_size": ("Mass share ÷ photo share", False),
}


def _top_column(args: argparse.Namespace) -> str:
    if args.value:
        return args.value
    return "share_of_total" if args.share else "attribution_sum"


def _pick_layers(rows, spec: str) -> list[str]:
    layers = sort_layers({r["layer"] for r in rows})
    if not layers:
        raise SystemExit("the top-<level> CSV has no rows")
    if spec == "all":
        return layers
    if spec == "deepest":
        return [layers[-1]]
    layer = normalize_layer(spec)
    if layer not in layers:
        raise SystemExit(f"no layer {layer} in the CSV (has: {', '.join(layers)})")
    return [layer]


def _draw_top(rows, style: Style, args):
    """Bars of the selected sources; heights relative to the largest unless --raw."""
    column = _top_column(args)
    values = np.array([r[column] for r in rows], dtype=float)
    # --absolute loses nothing (a group's sign is its label's, and colour
    # carries that) and buys a y-axis from zero.
    if args.absolute:
        values = np.abs(values)
    # --share and --raw are already on their own scale; only the default
    # normalises against the largest bar.
    if args.raw or args.share or args.value:
        heights = values
    else:
        denom = float(np.max(np.abs(values))) if values.size else 1.0
        heights = values / (denom or 1.0)

    colors, classes = _class_colors({r["age_sex"] for r in rows}, style)
    bar_colors = [colors.get(r["age_sex"], style.c["muted"]) for r in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(3.4, 0.30 * len(rows) + 1.1), 2.7))
    if not args.absolute:
        ax.axhline(0.0, color=style.c["axis"], lw=0.6, zorder=2)
    ax.bar(x, heights, color=bar_colors, width=0.78, zorder=3,
           edgecolor=style.c["surface"], linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r["key"]) for r in rows], rotation=90)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    # The column holds whatever src.attribution.evaluate --top-weight summed, so name
    # the axis after that rather than always calling it attribution.
    weight = str(rows[0].get("weight") or "attribution")
    if weight == "count":
        quantity = "Images in the selected set"          # never signed
    elif weight == "alpha":
        quantity = "Summed |α|" if args.absolute else "Summed α"
    else:
        quantity = "|Attribution| sum" if args.absolute else "Attribution sum"

    if args.value:
        label_text, is_pct = _VALUE_COLUMNS[args.value]
        ax.set_ylabel(label_text)
        if is_pct:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        # A ratio's null is 1, and the figure is unreadable without it drawn.
        if args.value == "enrichment_vs_size" and getattr(args, "baselines", True):
            ax.axhline(1.0, **_pool_rule(style))
    elif args.share:
        noun = {"count": "images", "alpha": "|α|"}.get(weight, "attribution mass")
        ax.set_ylabel(f"Share of total {noun}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    else:
        ax.set_ylabel(quantity if args.raw else f"{quantity} (relative to largest)")
    # Vertical bars keep the default horizontal (y) grid the probe curves use.
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(style.c["axis"])

    if style.legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classes]
        ax.legend(handles, classes, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# curve -- cumulative attribution mass vs. number of sources (DualXDA Fig. 5)
# --------------------------------------------------------------------------- #
def cmd_curve(args: argparse.Namespace) -> None:
    """Cumulative mass over top-k, one curve per method."""
    evals = load_attrib_evals(args.evals, args.label)
    style = style_from_args(args)
    outdir = _outdir(args, evals)
    written: list[Path] = []
    fname = f"cumulative_{args.level}.csv"

    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for ev in evals:
        if not ev.has(fname):
            print(f"{ev.label}: no {fname}; run src.attribution.evaluate curve; skipped")
            continue
        sources.append((ev.label, ev.read(fname)))
    if not sources:
        raise SystemExit(f"no {fname} matched")

    layers = _pick_layers([r for _, rows in sources for r in rows], args.layer)

    # Colour = method, dash = whatever varies within it (usually a C sweep), so
    # the grouping is visible before the legend is read.
    methods: list[str] = []
    for _, rows in sources:
        m = rows[0].get("method", "")
        if m not in methods:
            methods.append(m)
    palette = style.series(max(len(methods), 2))
    seen: dict[str, int] = {}
    marks = []
    for _, rows in sources:
        m = rows[0].get("method", "")
        within = seen.get(m, 0)
        seen[m] = within + 1
        kw = line_kw(palette[methods.index(m)], style)
        marks.append({**kw, "marker": "", "clip_on": True,
                      "ls": DASHES[within % len(DASHES)] if within else "-"})

    for layer in layers:
        fig, ax = plt.subplots()
        pool = 0
        for i, (label, rows) in enumerate(sources):
            lrows = sorted((r for r in rows if r["layer"] == layer), key=lambda r: r["k"])
            if not lrows:
                continue
            pool = max(pool, int(lrows[-1]["pool"]))
            # No markers: densely-spaced samples, not measured points. clip_on
            # because --xmax windows the axis and the tight bbox would grow.
            ax.plot([r[args.x] for r in lrows], [r["mass_mean"] for r in lrows],
                    **marks[i], label=label)
        if args.baselines and pool:
            # Perfectly even attribution: the top k of n carry k/n of the mass.
            ks = np.unique(np.round(np.geomspace(1, pool, 200)).astype(int))
            ax.plot(ks if args.x == "k" else ks / pool, ks / pool, **ref_kw(style))

        # A window (the published figure's 0-5%) reads linearly; the full sweep
        # spans four decades and only reads on a log axis.
        if args.xmax is not None:
            ax.set_xlim(0, args.xmax)
        else:
            ax.set_xscale("log")
            ax.set_xlim(1 / pool if args.x == "frac" else 1, pool if args.x == "k" else 1.0)
        ax.set_ylim(0.0, 1.02)
        if args.x == "frac":
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.2%}")
            ax.set_xlabel(f"Top % of training {args.level}s")
        else:
            ax.set_xlabel(f"Top-k training {args.level}s")
        ax.set_ylabel("Cumulative mass share")
        handles, labels = ax.get_legend_handles_labels()
        place_legend(ax, handles, labels, style)
        suffix = f"_{layer}" if len(layers) > 1 else ""
        written += save_figure(fig, outdir / f"{args.name}{suffix}", args.format)

    report_written(written)


# --------------------------------------------------------------------------- #
# idshare -- is identity concentration anything but identity size
# --------------------------------------------------------------------------- #
def cmd_idshare(args: argparse.Namespace) -> None:
    """Top-K concentration as a ratio to its size baseline."""
    evals = load_attrib_evals(args.evals, args.label)
    style = style_from_args(args)
    outdir = _outdir(args, evals)
    written: list[Path] = []
    fname = f"{args.level}_concentration.csv"

    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for ev in evals:
        if not ev.has(fname):
            print(f"{ev.label}: no {fname}; run src.attribution.evaluate composition; skipped")
            continue
        rows = [r for r in ev.read(fname) if int(r["top_k"]) == args.k]
        if rows:
            sources.append((ev.label, sorted(rows, key=lambda r: layer_index(r["layer"]))))
    if not sources:
        raise SystemExit(f"no {fname} with top_k={args.k}; --concentration-k sets which K exist")

    curves = [
        ("mass_share", "mass_share_baseline", f"Top-{args.k} mass share"),
        ("per_image_share", "per_image_share_baseline", f"Top-{args.k} per-image share"),
    ]
    if args.metric == "ratio":
        curves = curves[1:]  # keep the size-free curve alone
    fig, ax = plt.subplots()
    series: list[list[float | None]] = []
    i = 0
    n_series = len(sources) * len(curves)
    for label, rows in sources:
        for col, base_col, cname in curves:
            xs = [layer_index(r["layer"]) for r in rows]
            if args.metric == "ratio":
                ys = [
                    (r[col] / r[base_col]) if r[base_col] else None for r in rows
                ]
            else:
                ys = [r[col] for r in rows]
            series.append(ys)
            name = cname if len(sources) == 1 else f"{label} · {cname}"
            ax.plot(xs, ys, **style.series_kw(i, n_series), label=name)
            i += 1

    if args.baselines:
        if args.metric == "ratio":
            ax.axhline(1.0, **_pool_rule(style))
            rule_name = "size-matched baseline"
        else:
            # Both baselines are properties of the pool, so they are flat in depth.
            for _, base_col, cname in curves:
                ax.axhline(sources[0][1][0][base_col], **_pool_rule(style))
            rule_name = "baselines (image share / equal rate)"
    ax.set_xlabel("Layer")
    ax.set_ylabel(
        f"Share ÷ baseline" if args.metric == "ratio" else f"Top-{args.k} share"
    )
    all_x = [layer_index(r["layer"]) for _, rows in sources for r in rows]
    _finish_curve(ax, all_x, series, style, ref=1.0 if args.metric == "ratio" else 0.6,
                  full_range=args.full_range, limits=(0.0, INF))
    handles, leg = ax.get_legend_handles_labels()
    if args.baselines:
        handles.append(plt.Line2D([], [], **_pool_rule(style)))
        leg.append(rule_name)
    place_legend(ax, handles, leg, style)
    written += save_figure(fig, outdir / args.name, args.format)
    report_written(written)


# --------------------------------------------------------------------------- #
# loo -- what a unit is actually worth, in eval metrics
# --------------------------------------------------------------------------- #
def cmd_loo(args: argparse.Namespace) -> None:
    """ΔAUROC bars for the units whose removal moves the probe most."""
    evals = load_attrib_evals(args.evals, args.label)
    style = style_from_args(args)
    outdir = _outdir(args, evals)
    written: list[Path] = []
    fname = "loo_metrics_identity.csv"

    for ev in evals:
        if not ev.has(fname):
            raise SystemExit(f"{ev.label}: no {fname}; run src.attribution.evaluate loo first")
        rows = ev.read(fname)
        ssuffix = f"_{slug(ev.label)}" if len(evals) > 1 else ""
        layers = _pick_layers(rows, args.layer)
        for layer in layers:
            lsuffix = f"_{layer}" if len(layers) > 1 else ""
            lrows = sorted(
                (r for r in rows if r["layer"] == layer),
                key=lambda r: -abs(r["delta_auroc"]),
            )[: args.top_n]
            fig = _draw_loo_bars(lrows, style)
            written += save_figure(fig, outdir / f"{args.name}{ssuffix}{lsuffix}", args.format)

    report_written(written)


def _draw_loo_bars(rows, style: Style):
    # The CSV holds full - loo; a bar shows loo - full, so up = better off without the unit.
    values = np.array([-float(r["delta_auroc"]) for r in rows], dtype=float)
    colors, classes = _class_colors({r["age_sex"] for r in rows}, style)
    bar_colors = [colors.get(r["age_sex"], style.c["muted"]) for r in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(3.4, 0.30 * len(rows) + 1.1), 2.7))
    ax.axhline(0.0, color=style.c["axis"], lw=0.6, zorder=2)
    ax.bar(x, values, color=bar_colors, width=0.78, zorder=3,
           edgecolor=style.c["surface"], linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["key"]) for r in rows], rotation=90)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("ΔAUROC")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(style.c["axis"])
    if style.legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classes]
        ax.legend(handles, classes, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser, *, name: str) -> None:
    parser.add_argument("--evals", nargs="+", required=True,
                        help="Attribution-evaluation director(ies): "
                             "<run>/attributions/<method> or its eval/.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: <cache dir>/reports")
    parser.add_argument("--label", nargs="+", default=None, metavar="NAME",
                        help="Legend name per --evals cache, in order. Also retitles the "
                             "figure, and tells the files apart when several caches "
                             "are drawn in one run.")
    parser.add_argument("--name", default=name,
                        help=f"The whole output filename, minus the extension "
                             f"(default: {name})")
    add_style_args(parser)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("composition", help="Class share / enrichment over depth")
    _add_common(comp, name="composition")
    comp.add_argument("--age-sex", nargs="+", default=["Blackback"],
                      help="Class(es) to draw (default: Blackback).")
    comp.set_defaults(func=cmd_composition)

    crv = sub.add_parser(
        "curve", help="Cumulative mass vs. top-k sources, one curve per method (DualXDA Fig. 5)"
    )
    _add_common(crv, name="curve")
    crv.add_argument("--level", default="image", choices=GROUP_LEVELS,
                     help="Which cumulative_<level>.csv to read (default: image)")
    crv.add_argument("--layer", default="deepest",
                     help="'deepest' (default), 'all', or a layer name")
    crv.add_argument("--x", default="frac", choices=["frac", "k"],
                     help="x-axis: 'frac' (default) is the share of the training pool; 'k' "
                          "the absolute count.")
    crv.add_argument("--xmax", type=float, default=None,
                     help="Upper x limit in the units of --x; switches the axis to linear.")
    crv.add_argument("--no-baselines", dest="baselines", action="store_false",
                     help="Hide the uniform-attribution reference curve.")
    crv.set_defaults(func=cmd_curve)

    top = sub.add_parser("top", help="Bar chart of the most influential sources at one layer")
    _add_common(top, name="top")
    top.add_argument("--mode", default="abs", choices=list(_TOP_MODES),
                     help="Which K sources to show: largest |attribution|, strongest "
                          "proponents, or strongest opponents.")
    top.add_argument("--level", default="identity", choices=GROUP_LEVELS,
                     help="Which top_<level>.csv to read (default: identity)")
    top.add_argument("--layer", default="deepest",
                     help="'deepest' (default), 'all', or a layer name")
    top.add_argument("--top-n", type=int, default=10,
                     help="How many sources to show")
    scale = top.add_mutually_exclusive_group()
    scale.add_argument("--value", default=None, choices=list(_VALUE_COLUMNS),
                       help="Draw and rank by this column instead of the normalised "
                            "attribution sum; prefer the size-free ones.")
    scale.add_argument("--raw", action="store_true",
                       help="Plot raw attribution sums instead of values relative to the "
                            "largest source. The absolute scale is layer-dependent.")
    scale.add_argument("--share", action="store_true",
                       help="Each group's share of the total, as a percentage; the total is "
                            "over every group, not just those drawn.")
    top.add_argument("--absolute", action="store_true",
                     help="Draw |value|, so bars go up from zero; the class colour still "
                          "carries the sign. Pairs with --mode abs.")
    top.set_defaults(func=cmd_top)

    ids = sub.add_parser(
        "idshare", help="Top-K identity concentration against its size baseline, over depth"
    )
    _add_common(ids, name="idshare")
    ids.add_argument("--level", default="identity", choices=GROUP_LEVELS)
    ids.add_argument("--k", type=int, default=10,
                     help="Which top-K row to draw; must be one the composition run wrote.")
    ids.add_argument("--metric", default="ratio", choices=["ratio", "share"],
                     help="'ratio' (default) draws the per-image share over its baseline, "
                          "so the null is 1; 'share' plots the raw mass and per-image "
                          "shares.")
    ids.add_argument("--no-baselines", dest="baselines", action="store_false",
                     help="Hide the baseline rule. Rarely what you want here: the "
                          "baseline is the entire point of this figure.")
    ids.set_defaults(func=cmd_idshare)

    loo = sub.add_parser("loo", help="What removing a unit costs, in eval metrics")
    _add_common(loo, name="loo")
    loo.add_argument("--layer", default="deepest",
                     help="'deepest' (default), 'all', or a layer name")
    loo.add_argument("--top-n", type=int, default=20,
                     help="How many units to show (default: 20)")
    loo.set_defaults(func=cmd_loo)

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
