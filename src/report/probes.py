"""Probe tables and figures from the caches src.probes.evaluate wrote."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.metadata import AGE_SEX_ORDER
from src.report import tables as T
from src.probes.cache import ProbeEval, load_evals
from src.probes.discovery import (
    layer_index,
    load_probe_direction,
    normalize_layer,
    probes_in_dir,
    sort_layers,
)
from src.probes.metrics import (
    METRIC_CHANCE,
    METRIC_KEYS,
    METRIC_LABELS,
    UNITS,
    cosine,
)
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
    set_pub_style,
    slug,
    Style,
    style_from_args,
    zoom_ylim,
)

# Below src.report.style: it sets the Agg backend before pyplot is imported.
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _outdir(args: argparse.Namespace, evals: Sequence[ProbeEval] | None = None) -> Path:
    """A reports/ beside the cache, never inside it; the first cache anchors it."""
    if args.out:
        return Path(args.out)
    if evals:
        return evals[0].path.parent / "reports"
    return Path("reports")


def _labels(evals: Sequence[ProbeEval], args: argparse.Namespace) -> list[str]:
    if not args.label:
        # Disambiguated when two caches share a run name (a probe on two backbones).
        names = [_cache_tag(e.path) for e in evals]
        return [
            name if names.count(name) == 1 else e.path.parent.name
            for e, name in zip(evals, names)
        ]
    if len(args.label) != len(evals):
        raise SystemExit(
            f"--label needs one name per --evals cache ({len(evals)}), "
            f"got {len(args.label)}"
        )
    return list(args.label)


#: Stage directories, which name a kind of cache and not the probe it describes.
_STAGE_DIRNAMES = frozenset({"eval", "holdout", "id_cv", "probes"})


def _cache_tag(path: Path) -> str:
    """The run a cache belongs to; transfer_<backbone> names itself."""
    p = path
    while p.name in _STAGE_DIRNAMES and p.parent != p:
        p = p.parent
    return p.name


def _layers_for(
    ev: ProbeEval,
    spec: str,
    *,
    select_by: str,
    split: str,
    unit: str,
    threshold_kind: str,
) -> list[str]:
    """Resolve --layer per cache. 'best' is per-cache: a comparison asks what each
    probe achieves at its own best depth."""
    if spec == "all":
        return ev.layers
    if spec == "best":
        layer, _ = ev.best_layer(
            select_by, split=split, unit=unit, threshold_kind=threshold_kind
        )
        if layer is None:
            raise SystemExit(f"{ev.path.name}: no layer carries {select_by} on split {split}")
        return [layer]
    layer = normalize_layer(spec)
    if layer not in ev.layers:
        raise SystemExit(
            f"{ev.path.name} has no layer {layer} (has: {', '.join(ev.layers)})"
        )
    return [layer]


def _place_sectioned_legend(target, entries, style: Style) -> None:
    """Held-out curves in one titled block, their train twins in another."""

    if not style.legend:  # --no-legend
        return
    main, train = entries["main"], entries["train"]
    if not train:
        place_legend(target, [h for h, _ in main], [l for _, l in main], style)
        return

    sections = [("held-out", main), ("train", train)]
    for k, (title, section) in enumerate(sections):
        center = 0.5 + ((2 * k + 1) / (2 * len(sections)) - 0.5)
        leg = target.legend(
            [h for h, _ in section], [l for _, l in section], title=title,
            loc="upper center", bbox_to_anchor=(center, -0.28),
            ncol=1, borderaxespad=0.0, alignment="left",
            title_fontproperties={"weight": "bold", "size": plt.rcParams["legend.fontsize"]},
        )
        leg.get_title().set_color(style.c["ink_soft"])
        # Axes.legend keeps only the last, and add_artist would clip it to the
        # axes patch and crop it out of the tight bbox.
        if isinstance(target, Axes) and k < len(sections) - 1:
            target.add_artist(leg)
            leg.set_clip_on(False)


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def cmd_table(args: argparse.Namespace) -> None:
    evals = load_evals(args.evals)
    labels = _labels(evals, args)
    rows: list[dict[str, Any]] = []

    for ev, label in zip(evals, labels):
        split = ev.default_split()
        with_train = args.with_train and split != "train" and "train" in ev.splits
        for unit in args.unit:
            for kind in ("probe",):
                for layer in _layers_for(
                    ev, args.layer, select_by="auroc", split=split,
                    unit=unit, threshold_kind=kind,
                ):
                    matches = ev.rows(
                        "metrics", layer=layer, split=split, unit=unit, threshold_kind=kind
                    )
                    if not matches:
                        continue
                    source = matches[0]
                    row = {
                        "probe": label,
                        "mode": ev.mode,
                        "split": split,
                        "layer": layer,
                        "unit": unit,
                        "threshold_kind": kind,
                        "threshold": source["threshold"],
                        "n": source["n"],
                        **{m: source.get(m) for m in args.metrics},
                    }
                    # The train twin at the same layer -- the layer stays the one
                    # the eval split chose, so the two columns are comparable.
                    if with_train:
                        twin = ev.rows("metrics", layer=layer, split="train",
                                       unit=unit, threshold_kind=kind)
                        source_train = twin[0] if twin else {}
                        row["n_train"] = source_train.get("n")
                        row |= {f"{m}_train": source_train.get(m) for m in args.metrics}
                    rows.append(row)

    if not rows:
        raise SystemExit("no rows matched — check --split / --unit / --layer")

    cols = T.columns(("probe", "Probe", 3, "l"), ("layer", "Layer", 3, "l"))
    if len({r["mode"] for r in rows}) > 1:
        cols += T.columns(("mode", "Mode", 3, "l"))
    if len(args.unit) > 1:
        cols += T.columns(("unit", "Unit", 3, "l"))
    has_train = any(f"{args.metrics[0]}_train" in r for r in rows)
    for m in args.metrics:
        label = METRIC_LABELS.get(m, m)
        cols.append(T.Column(m, label, 3, "r"))
        if has_train:
            cols.append(T.Column(f"{m}_train", f"{label} (train)", 3, "r"))

    where = {"best": "best layer by auroc", "all": "every layer"}.get(
        args.layer, f"layer {args.layer}"
    )
    report_written(T.write(
        rows, cols, _outdir(args, evals) / args.name,
        title=f"Probe metrics at the {where} "
              f"({', '.join(sorted({e.mode for e in evals}))})",
    ))


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def cmd_layers(args: argparse.Namespace) -> None:
    evals = load_evals(args.evals) if args.evals else []
    labels = _labels(evals, args)
    bands = _load_bands(args)
    if not evals and not bands:
        raise SystemExit("layers needs --evals and/or --band")
    style = style_from_args(args)
    # Bands continue the palette after the solid --evals curves, so a full-probe
    # line and its single-image band never collide in hue.
    band_colors = style.series(len(evals) + len(bands))[len(evals):] if bands else []
    outdir = _outdir(args, evals or bands[0]["evals"])
    written: list[Path] = []

    for metric in args.metrics:
        fig, ax = plt.subplots()
        entries = _draw_layer_metric(ax, evals, labels, metric, style, args,
                                     bands, band_colors)
        ax.set_xlabel("Layer")
        _place_sectioned_legend(ax, entries, style)
        suffix = f"_{metric}" if len(args.metrics) > 1 else ""
        written += save_figure(fig, outdir / f"{args.name}{suffix}", args.format)

    report_written(written)


#: ``--dash`` vocabulary: words, not matplotlib's "-"/"--", since argparse reads
#: a leading dash as the next option name. Patterns are the shared ones.
DASH_STYLES: dict[str, Any] = {
    "solid": DASHES[0], "dashed": DASHES[1], "dotted": DASHES[2], "dashdot": DASHES[3],
}


def _per_curve(args, flag: str, n: int, what: str) -> list | None:
    values = getattr(args, flag)
    if not values:
        return None
    if len(values) != n:
        raise SystemExit(
            f"--{flag} needs one {what} per --evals cache ({n}), got {len(values)}"
        )
    return list(values)


def _draw_layer_metric(ax, evals, labels, metric, style: Style, args,
                       bands: Sequence[dict[str, Any]] = (),
                       band_colors: Sequence[str] = ()) -> dict[str, list[tuple[Any, str]]]:
    """Draw the curves; return legend entries with held-out and train kept apart."""
    n = len(evals)
    series: list[list[float | None]] = []
    entries: dict[str, list[tuple[Any, str]]] = {"main": [], "train": []}

    # Hand-set colour/dash per curve, for groupings the one-hue-per-cache ramp
    # cannot express (a transfer pair shares a hue and splits on the dash).
    hues = _per_curve(args, "hue", n, "slot")
    dashes = _per_curve(args, "dash", n, "style")
    units = _per_curve(args, "unit", n, "unit") if len(args.unit) > 1 else args.unit * n
    palette = style.series(max(n, max(hues) + 1) if hues else n)

    baseline = _baseline(evals, metric, args)
    if baseline is not None:
        ax.axhline(baseline[0], **ref_kw(style))

    # With --with-train, dash encodes the split (solid eval / dashed train), so
    # it cannot also encode the probe -- hue carries the probe alone there.
    dashed_series = not args.with_train

    for i, (ev, label) in enumerate(zip(evals, labels)):
        split = ev.default_split()
        layers, values = ev.curve(metric, split=split, unit=units[i], threshold_kind="probe")
        xs = [layer_index(l) for l in layers]
        series.append(values)
        kw = style.series_kw(i, n, dashed=dashed_series)
        if hues:
            kw["color"] = kw["mfc"] = palette[hues[i]]
        if dashes:
            kw["ls"] = DASH_STYLES[dashes[i]]
        line, = ax.plot(xs, values, **kw, label=label)
        entries["main"].append((line, label))

        # Train curves stay recessive: same hue, dashed, no markers.
        if args.with_train and "train" in ev.splits and split != "train":
            _, train_values = ev.curve(metric, split="train", unit=units[i],
                                       threshold_kind="probe")
            series.append(train_values)
            train_line, = ax.plot(xs, train_values, ls=(0, (3.5, 1.8)), lw=1.2,
                                  color=kw["color"], zorder=3, clip_on=False,
                                  label=f"{label} (train)")
            entries["train"].append((train_line, label))

    # A band is one series from many caches: mean as the line, spread shaded.
    for j, band in enumerate(bands):
        bs = _band_series(band["evals"], metric, args)
        if not bs["xs"]:
            continue
        color = band_colors[j] if j < len(band_colors) else style.series(n + len(bands))[n + j]
        ax.fill_between(bs["xs"], bs["lo"], bs["hi"], color=color, alpha=0.18, lw=0, zorder=2)
        band_label = band["label"]
        band_line, = ax.plot(bs["xs"], bs["mean"], **line_kw(color, style), label=band_label)
        entries["main"].append((band_line, band_label))
        series += [bs["mean"], bs["lo"], bs["hi"]]

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylim(*zoom_ylim(series, baseline[0] if baseline else None, args.full_range))
    all_x = [layer_index(l) for ev in evals for l in ev.layers]
    for band in bands:
        all_x += [layer_index(l) for ev in band["evals"] for l in ev.layers]
    finish_layer_axes(ax, all_x, style)
    return entries


def _load_bands(args) -> list[dict[str, Any]]:
    """Resolve each --band group to {evals, label}."""
    groups = args.band or []
    provided = args.band_label or []
    bands: list[dict[str, Any]] = []
    for i, entry in enumerate(groups):
        group_evals = load_evals(entry)
        if not group_evals:
            continue
        label = provided[i] if i < len(provided) else group_evals[0].label
        bands.append({"evals": group_evals, "label": label})
    return bands


def _band_series(group: Sequence[ProbeEval], metric: str, args) -> dict[str, Any]:

    by_layer: dict[str, list[float]] = defaultdict(list)
    for ev in group:
        split = ev.default_split()
        layers, values = ev.curve(metric, split=split, unit=args.unit[0],
                                   threshold_kind="probe")
        for layer, value in zip(layers, values):
            if value is not None and value == value:  # drop None and NaN
                by_layer[layer].append(float(value))

    names, xs, mean, lo, hi = [], [], [], [], []
    for layer in sort_layers(by_layer.keys()):
        arr = np.asarray(by_layer[layer], dtype=float)
        m = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        lo_, hi_ = m - sd, m + sd
        names.append(layer)
        xs.append(layer_index(layer))
        mean.append(m)
        lo.append(lo_)
        hi.append(hi_)
    return {"layers": names, "xs": xs, "mean": mean, "lo": lo, "hi": hi, "k": len(group)}


def _baseline(evals, metric: str, args) -> tuple[float, str] | None:
    """The metric's no-signal value. Accuracy's is the majority rate, which depends on
    class balance, so it is only drawn when every cache agrees."""
    if not args.baselines:
        return None
    if metric in METRIC_CHANCE:
        return METRIC_CHANCE[metric], "chance"
    if metric != "accuracy":
        return None

    rates = set()
    for ev in evals:
        rows = ev.rows("metrics", split=ev.default_split(), unit=args.unit[0],
                       threshold_kind="probe")
        for row in rows:
            total = (row.get("n_pos") or 0) + (row.get("n_neg") or 0)
            if total:
                rates.add(round(max(row["n_pos"], row["n_neg"]) / total, 4))
    return (rates.pop(), "majority") if len(rates) == 1 else None


# --------------------------------------------------------------------------- #
# cosine
# --------------------------------------------------------------------------- #
def cmd_cosine(args: argparse.Namespace) -> None:
    sources = _cosine_sources(args)
    if len(sources) < 2:
        raise SystemExit("cosine needs at least two sources (--probes and/or --evals)")

    shared = sort_layers(set.intersection(*(set(s["layers"]) for s in sources)))
    if not shared:
        raise SystemExit(
            "the sources share no layer: "
            + "; ".join(f"{s['label']}: {', '.join(sort_layers(s['layers']))}" for s in sources)
        )

    rows: list[dict[str, Any]] = []
    for layer in shared:
        vectors = [s["direction"](layer) for s in sources]
        for i, a in enumerate(sources):
            for j in range(i + 1, len(sources)):
                b = sources[j]
                wa, wb = vectors[i], vectors[j]
                if wa is None or wb is None:
                    continue
                if wa.shape != wb.shape:
                    raise SystemExit(
                        f"layer {layer}: {a['label']} has {wa.shape[0]} dimensions and "
                        f"{b['label']} has {wb.shape[0]} — they were fit on different "
                        "activations, so their directions are not comparable."
                    )
                value = cosine(wa, wb)
                rows.append({
                    "layer": layer,
                    "index": layer_index(layer),
                    "a": a["label"],
                    "b": b["label"],
                    "pair": f"{a['label']} vs {b['label']}",
                    "cos": value,
                })

    if not rows:
        raise SystemExit("no comparable direction pairs were found")

    # cosine can run off probe dirs alone, so its sources carry their own
    # directory and the first takes the output.
    outdir = Path(args.out) if args.out else sources[0]["path"].parent / "reports"
    written = T.write(
        rows,
        T.columns("layer", "a", "b") + [T.Column("cos", "cos", 4)],
        outdir / args.name,
        title="Direction alignment (cos, raw activation space)",
    )

    written += _plot_cosine(rows, outdir, args)
    report_written(written)


def _plot_cosine(rows, outdir: Path, args) -> list[Path]:
    style = style_from_args(args)
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    fig, ax = plt.subplots()
    # 0 = orthogonal: the "these probes share no direction" reference. Cropping
    # it would let a curve hovering near 0 read as if it were near 1.
    ax.axhline(0.0, **ref_kw(style))
    series = []
    for i, pair in enumerate(pairs):
        pair_rows = sorted((r for r in rows if r["pair"] == pair), key=lambda r: r["index"])
        xs = [r["index"] for r in pair_rows]
        ys = [r["cos"] for r in pair_rows]
        series.append(ys)
        ax.plot(xs, ys, **style.series_kw(i, len(pairs)), label=pair)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    limits = (-1.0, 1.0)
    ax.set_ylim(*zoom_ylim(series, 0.0, args.full_range, limits))
    finish_layer_axes(ax, [r["index"] for r in rows], style, clamp=limits)
    handles, labels = ax.get_legend_handles_labels()
    place_legend(ax, handles, labels, style)
    return save_figure(fig, outdir / args.name, args.format)


def _cosine_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Directions from a probe dir or an eval cache (an ID-CV axis has no single probe)."""
    sources: list[dict[str, Any]] = []
    raw_labels: list[str] = []

    for entry in args.probes or []:
        probe_dir = Path(entry) / "probes"
        found = dict(probes_in_dir(probe_dir))
        raw_labels.append(_cache_tag(probe_dir))
        sources.append({
            "path": probe_dir,
            "layers": list(found),
            "direction": lambda layer, found=found: (
                load_probe_direction(found[layer]) if layer in found else None
            ),
        })

    for ev in load_evals(args.evals or []):
        raw_labels.append(_cache_tag(ev.path))
        sources.append({
            "path": ev.path,
            "layers": ev.layers,
            "direction": lambda layer, ev=ev: ev.direction(layer),
        })

    # Here --label names every source, --probes and --evals together in that order,
    # since a cosine curve is a pair of them and both halves need a name.
    if args.label:
        if len(args.label) != len(sources):
            raise SystemExit(
                f"--label needs one name per source ({len(sources)}: "
                f"{len(args.probes or [])} --probes + {len(args.evals or [])} --evals), "
                f"got {len(args.label)}"
            )
        names = list(args.label)
    else:
        # Qualify by backbone when two sources share a run name (fine-tuned vs pre-trained).
        names = [
            name if raw_labels.count(name) == 1
            else f"{_cache_tag(src['path'].parent.parent)}/{name}"
            for name, src in zip(raw_labels, sources)
        ]
    for source, label in zip(sources, names):
        source["label"] = label
    return sources


# --------------------------------------------------------------------------- #
# logits
# --------------------------------------------------------------------------- #
# Plotted top-to-bottom in this order. Silverback and Blackback are the two
# halves of the "Male" label and belong next to each other.
LOGIT_CLASS_ORDER = ["Silverback", "Blackback", "Adult Female"]


def _logit_rows(ev: ProbeEval, layer: str, split: str, classes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in ev.rows("identities", layer=layer, split=split):
        if record["age_sex"] not in classes:
            continue
        rows.append({
            "identity": record["identity"],
            "age_sex_class": record["age_sex"],
            "n_images": record["n_images"],
            "mean_logit": record["mean_logit"],
            "sd_logit": record["sd_logit"],
            "median_logit": record["median_logit"],
            "frac_above_threshold": record["frac_above_threshold"],
            "seen": record["seen"],
        })
    return rows


def _render_logit_layer(
    ev: ProbeEval, layer: str, split: str, args: argparse.Namespace, style: Style,
    *, multi: bool = False
) -> list[Path] | None:
    rows = _logit_rows(ev, layer, split, args.classes)
    if not rows:
        print(f"WARNING: {layer}: no identity matched {args.classes} — nothing to plot")
        return None

    metric_rows = ev.rows("metrics", layer=layer, split=split, unit="image", threshold_kind="probe")
    threshold = float(metric_rows[0]["threshold"]) if metric_rows else 0.0

    if all(r["seen"] is None for r in rows):
        print("WARNING: this cache was written without a split file — every marker will be "
             "filled, and the figure will not say which identities the backbone saw")

    fig = _draw_logit_figure(
        rows,
        classes=args.classes,
        style=style,
        threshold=threshold,
        negative_name=ev.class_names[0],
        positive_name=ev.class_names[1],
        figsize=tuple(args.figsize) if args.figsize else None,
    )

    # Figures land next to the numbers behind them.
    outdir = _outdir(args, [ev])
    outdir.mkdir(parents=True, exist_ok=True)
    # A run that draws one figure names it --name; several need telling apart.
    stem = f"{args.name}_{slug(ev.label)}_{layer}_{split}" if multi else args.name
    written: list[Path] = []
    for fmt in args.format:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    n_seen = sum(1 for r in rows if r["seen"])
    print(f"{layer}/{split}: {len(rows)} identities plotted ({n_seen} in-sample, "
          f"{len(rows) - n_seen} held out), threshold {threshold:.3f}")
    for path in written:
        print(f"  {path}")
    return written


def _logit_class_colors(classes: list[str], style: Style) -> dict[str, str]:
    """Colour per age-sex class in CANONICAL order, not --classes order (which sets
    rows only), so a class keeps one hue across the document."""

    known = [c for c in AGE_SEX_ORDER if c in classes]
    ordered = known + sorted(set(classes) - set(AGE_SEX_ORDER))
    palette = style.series(len(ordered))
    return {c: palette[i] for i, c in enumerate(ordered)}


def _draw_logit_figure(
    rows: list[dict[str, Any]],
    *,
    classes: list[str],
    style: Style,
    threshold: float,
    negative_name: str,
    positive_name: str,
    figsize: tuple[float, float] | None,
) -> plt.Figure:
    colors = _logit_class_colors(classes, style)
    c = style.c
    has_seen_marks = any(r["seen"] is not None for r in rows)

    n_rows = len(rows)
    n_groups = sum(1 for cls in classes if any(r["age_sex_class"] == cls for r in rows))
    if figsize is None:
        # Height follows the row count, capped at 9.5" so the full identity set
        # fits one page; past the cap the rows compress instead.
        units = n_rows + 2.2 * max(n_groups - 1, 0)
        figsize = (5.4, min(1.55 + 0.105 * units, 9.5))
    fig, ax = plt.subplots(figsize=figsize)

    ypos: list[float] = []
    ylab: list[str] = []
    ycol: list[str] = []
    y = 0.0
    for cls in classes:
        group = sorted(
            (r for r in rows if r["age_sex_class"] == cls), key=lambda r: r["mean_logit"]
        )
        if not group:
            continue
        color = colors[cls]
        for r in group:
            # The SD bar is a wash behind the point, never a competing mark.
            ax.plot(
                [r["mean_logit"] - r["sd_logit"], r["mean_logit"] + r["sd_logit"]],
                [y, y],
                lw=1.6,
                color=color,
                alpha=0.35,
                solid_capstyle="round",
                zorder=2,
            )
            # Knocked out = in-sample, filled = held out (the heavier mark).
            ax.plot(
                r["mean_logit"],
                y,
                marker="o",
                ls="",
                ms=4.2,
                mew=1.0,
                mfc=c["surface"] if r["seen"] else color,
                mec=color,
                zorder=3,
            )
            ypos.append(y)
            ylab.append(f"{r['identity']}  ({r['n_images']})")
            ycol.append(color)
            y += 1

        # The group median: the class's one-number position on the axis. Its
        # label goes in the gap below the group, never over the last row's bar.
        median = float(np.median([r["mean_logit"] for r in group]))
        ax.plot([median, median], [y - len(group) - 0.4, y - 0.6], lw=1.2,
                color=color, zorder=4)
        ax.text(median, y + 0.2, f"median {median:+.1f}", color=color, fontsize=6.5,
                ha="center", va="center")
        y += 2.2

    # By hand: autoscale would silently drop a threshold that sits outside the
    # identity means.
    lo = min(r["mean_logit"] - r["sd_logit"] for r in rows)
    hi = max(r["mean_logit"] + r["sd_logit"] for r in rows)
    lo, hi = min(lo, threshold), max(hi, threshold)
    pad = 0.06 * (hi - lo) or 1.0
    ax.set_xlim(lo - pad, hi + pad)

    ax.axvline(threshold, ls=(0, (4, 3)), lw=0.8, color=c["ink_soft"], zorder=1)
    # The label goes on whichever side of the line has room.
    on_right = threshold > (lo + hi) / 2
    ax.text(threshold + (-1 if on_right else 1) * 0.012 * (hi - lo), -1.5,
            "decision boundary", fontsize=6.5, color=c["muted"],
            ha="right" if on_right else "left", va="center")

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=6)
    for tick, color in zip(ax.get_yticklabels(), ycol):
        tick.set_color(color)
    ax.set_ylim(-1.8, y - 1.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"Probe logit")

    # Rows are the categories here, so the rules run the other way than in the
    # layerwise figures: vertical, along the value axis.
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=c["grid"], lw=0.6, ls="-")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.2, mew=1.0, mfc=colors[cls],
               mec=colors[cls], label=cls)
        for cls in classes
        if any(r["age_sex_class"] == cls for r in rows)
    ]
    if has_seen_marks:
        # In ink, not a class colour: these entries are about the fill, which
        # every class shares.
        handles += [
            Line2D([], [], marker="o", ls="", ms=4.2, mew=1.0, mfc=c["surface"],
                   mec=c["ink"], label="seen in training"),
            Line2D([], [], marker="o", ls="", ms=4.2, mew=1.0, mfc=c["ink"],
                   mec=c["ink"], label="held out"),
        ]
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    # One row: a multi-column legend fills column-first and would interleave
    # the classes with the fill encoding.
    if style.legend:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.005), fontsize=7, columnspacing=1.2,
                   handletextpad=0.4)
    return fig


def cmd_logits(args: argparse.Namespace) -> None:
    evals = load_evals(args.evals)
    style = style_from_args(args)
    set_pub_style(style, usetex=args.usetex)

    drawn = total = 0
    for ev in evals:
        split = ev.default_split()
        layers = ev.layers
        if args.layer is not None:
            wanted = normalize_layer(args.layer)
            if wanted not in layers:
                raise SystemExit(f"{ev.path.name} has no layer {wanted} "
                                 f"(has: {', '.join(layers)})")
            layers = [wanted]

        print(f"{ev.label} ({ev.path}): {len(layers)} layer(s), split '{split}'")
        multi = len(evals) > 1 or len(layers) > 1
        for layer in layers:
            total += 1
            if _render_logit_layer(ev, layer, split, args, style, multi=multi):
                drawn += 1

    if drawn == 0:
        raise SystemExit("no identity matched the requested classes — nothing plotted")
    print(f"{drawn}/{total} figure(s) drawn  [palette: {style.eval_color} / "
          f"{style.train_color}, {style.theme}]")


def _add_common(parser: argparse.ArgumentParser, *, name: str) -> None:
    parser.add_argument("--evals", nargs="+", default=None,
                        help="Evaluation cache director(ies) from src.probes.evaluate")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: the first --evals cache itself")
    parser.add_argument("--label", nargs="+", default=None, metavar="NAME",
                        help="Legend name per --evals cache, in the order given. Default: "
                             "derived from each cache's fit (e.g. 'LR C=1').")
    parser.add_argument("--name", default=name,
                        help=f"The whole output filename, minus the extension "
                             f"(default: {name}). Output lands in the "
                             "cache directory, so a stem naming one of the cache's own "
                             "tables (metrics, per_class, pairs, identities, trend, "
                             "variance) would overwrite it.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    tab = sub.add_parser("table", help="Global metrics per probe, at one layer or the best one")
    # "table", not "metrics": output now lands in the cache directory, where
    # metrics.csv is the cache's own global-metrics table.
    _add_common(tab, name="table")
    tab.add_argument("--layer", default="best",
                     help="'best' (per probe, by AUROC), 'all', or a layer name. Default: best")
    tab.add_argument("--metrics", nargs="+", default=["auroc", "accuracy", "balanced_accuracy"],
                     choices=list(METRIC_KEYS))
    tab.add_argument("--unit", nargs="+", default=["image"], choices=list(UNITS),
                     help="Evaluation unit(s). identity is the honest one on this data; "
                          "image is the conventional one. Default: image")
    tab.add_argument("--with-train", action="store_true",
                     help="Add a train column beside each metric, read at the layer the "
                          "eval split selected. Ignored for a cache with no train split "
                          "(ID-CV) or when --split train already asked for it.")
    tab.set_defaults(func=cmd_table)

    lay = sub.add_parser("layers", help="One metric over depth, one curve per probe")
    _add_common(lay, name="layers")
    lay.add_argument("--metrics", nargs="+", default=["auroc"], choices=list(METRIC_KEYS))
    lay.add_argument("--unit", nargs="+", default=["image"], choices=list(UNITS),
                     help="Evaluation unit, or one per --evals cache to draw the same "
                          "cache at several units. Default: image")
    lay.add_argument("--band", action="append", nargs="+", default=None, metavar="EVAL",
                     help="A group of caches drawn as one mean curve with a spread band; "
                          "repeat for several bands.")
    lay.add_argument("--band-label", action="append", default=None, metavar="LABEL",
                     help="Legend label for each --band group, in the order given. "
                          "Default: the group's shared probe label.")
    lay.add_argument("--with-train", action="store_true",
                     help="Also draw each probe's train curve, dashed in the same hue.")
    lay.add_argument("--hue", nargs="+", type=int, default=None, metavar="SLOT",
                     help="Palette slot per --evals cache; repeat a slot to give two curves "
                          "one colour.")
    lay.add_argument("--dash", nargs="+", default=None, metavar="STYLE",
                     choices=list(DASH_STYLES),
                     help=f"Line style per --evals cache, in the order given: "
                          f"{', '.join(DASH_STYLES)}. Together with --hue this is how a "
                          f"transfer figure pairs each direction with its in-domain "
                          f"reference: --hue 0 1 0 1 --dash solid solid dashed dashed.")
    lay.add_argument("--no-baselines", dest="baselines", action="store_false",
                     help="Hide the chance / majority reference lines.")
    add_style_args(lay)
    lay.set_defaults(func=cmd_layers)

    cos = sub.add_parser("cosine", help="Cosine similarity between probe directions")
    _add_common(cos, name="cosine")
    cos.add_argument("--probes", nargs="+", default=None, metavar="RUN",
                     help="Run director(ies); the probes are read from <run>/probes. "
                          "Needs no evaluation to have been run.")
    add_style_args(cos)
    cos.set_defaults(func=cmd_cosine)

    lg = sub.add_parser("logits", help="Per-identity logit strip, one row per gorilla")
    _add_common(lg, name="identity_logits")
    lg.add_argument("--layer", default=None,
                    help="Layer as 'blocks.8' or '8'. Default: every layer in the cache.")
    lg.add_argument("--classes", nargs="+", default=LOGIT_CLASS_ORDER, metavar="CLASS",
                    help=f"Age-sex classes to plot, in row order "
                         f"(default: {' '.join(LOGIT_CLASS_ORDER)}).")
    lg.add_argument("--figsize", nargs=2, type=float, default=None, metavar=("W", "H"),
                    help="Inches. Default: 5.4 wide, height scaled to the row count.")
    add_style_args(lg)
    lg.set_defaults(func=cmd_logits)

    args = ap.parse_args()
    # `table` takes no style flags, so this one is optional.
    if getattr(args, "list_palettes", False):
        list_palettes()
        raise SystemExit(0)
    if args.command != "cosine" and not args.evals and not getattr(args, "band", None):
        ap.error("--evals is required")
    if args.command == "cosine" and not (args.probes or args.evals):
        ap.error("cosine needs --probes and/or --evals")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
