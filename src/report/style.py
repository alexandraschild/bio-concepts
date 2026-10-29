"""Print chrome and colour for every figure in the reports."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: cluster / no display
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

#: Slot 1 = headline series, slot 2 = supporting. A new pair needs checking for
#: colour-blind separation and page contrast, not just on-screen distinctness.
PALETTES: dict[str, dict[str, tuple[str, str]]] = {
    #                 light surface (paper)       dark surface (slides)
    "blue-magenta": {"light": ("#2a78d6", "#c43e93"), "dark": ("#3987e5", "#cf55a0")},
    "blue-green":   {"light": ("#2a78d6", "#0e9e6e"), "dark": ("#3987e5", "#199e70")},
    "blue-orange":  {"light": ("#2a78d6", "#d97706"), "dark": ("#3987e5", "#f08a24")},
    "okabe-ito":    {"light": ("#0072B2", "#D55E00"), "dark": ("#3987e5", "#d95926")},
}

# Overflow hues, used from the third series on (a C sweep); the palette's own
# two slots always come first.
EXTRA_SERIES: dict[str, tuple[str, ...]] = {
    "light": ("#0e9e6e", "#E69F00", "#7d3ac1", "#56B4E9"),
    "dark":  ("#199e70", "#e9a93a", "#9d5fd6", "#7cc4ef"),
}

# Dash patterns paired with the overflow hues. Past ~4 curves hue alone stops
# being separable in grayscale print.
DASHES: tuple[tuple[int, tuple[float, ...]] | str, ...] = (
    "-",
    (0, (3.5, 1.8)),
    (0, (1.2, 1.4)),
    (0, (5.5, 1.6, 1.2, 1.6)),
    (0, (2.5, 1.4, 1.2, 1.4)),
    (0, (7.0, 2.0)),
)

# Chart chrome. Marks carry the series color; text and rules never do.
CHROME: dict[str, dict[str, str]] = {
    "light": {"surface": "#ffffff", "ink": "#1a1a19", "ink_soft": "#52514e",
              "muted": "#898781", "grid": "#e6e5e0", "axis": "#c3c2b7"},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink_soft": "#c3c2b7",
              "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}


class Style:
    """Resolved colors + mark specs for one render."""

    def __init__(self, palette: str = "okabe-ito", theme: str = "light",
                 colors: str | None = None, legend: bool = True):
        self.theme = theme
        self.c = CHROME[theme]
        # The legend switch rides along with the colours.
        self.legend = legend
        if colors:
            parts = [p.strip() for p in colors.split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit("error: --colors needs exactly two hexes, e.g. '#2a78d6,#c43e93'")
            self.eval_color, self.train_color = parts
        else:
            self.eval_color, self.train_color = PALETTES[palette][theme]

    def series(self, n: int) -> list[str]:
        """*n* series colours: the two palette slots, then overflow hues (they
        repeat past six, where dashes take over)."""
        out = [self.eval_color, self.train_color]
        for hex_ in EXTRA_SERIES[self.theme]:
            if len(out) >= n:
                break
            if hex_.lower() not in {c.lower() for c in out}:
                out.append(hex_)
        return (out * (n // len(out) + 1))[:n]

    def series_kw(self, i: int, n: int, *, dashed: bool = True) -> dict:
        """Mark spec for series *i* of *n*. Two stay solid (only the hue should
        differ); beyond that the dash carries identity where colour cannot."""
        color = self.series(n)[i]
        kw = line_kw(color, self)
        if dashed and n > 2:
            kw["ls"] = DASHES[i % len(DASHES)]
        return kw


def line_kw(color: str, style: Style) -> dict:
    """Solid line, markers knocked out of the surface so overlapping points stay countable.
    clip_on=False keeps the end markers whole (the x-limits are the data range exactly)."""
    return dict(ls="-", marker="o", color=color, mfc=color,
                mec=style.c["surface"], mew=0.7, zorder=4, clip_on=False)


def tight_extras(fig) -> list:
    """Artists the tight bbox must be told about or it trims them."""
    return fig.get_default_bbox_extra_artists() + [ax.yaxis.label for ax in fig.axes]


def ref_kw(style: Style) -> dict:
    """Quiet dotted rule for a reference value: chance, majority, orthogonality."""
    return dict(ls=(0, (1, 2.2)), lw=0.7, color=style.c["muted"], zorder=2)


def save_figure(fig, out_stem: Path, formats: Sequence[str]) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_stem.with_name(f"{out_stem.name}.{fmt}")
        fig.savefig(path, bbox_extra_artists=tight_extras(fig))
        written.append(path)
    plt.close(fig)
    return written


def place_legend(ax, handles, labels, style: Style, *,
                 y: float = -0.28, max_cols: int = 3) -> None:
    """Below the axes: no corner is reliably free on a layer sweep."""
    if not style.legend:  # --no-legend
        return
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=min(len(labels), max_cols), borderaxespad=0.0)


def report_written(written: Sequence[Path]) -> None:
    if not written:
        return
    print(f"\nWrote {len(written)} file(s):")
    for path in written:
        print(f"  {path}")


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def finish_layer_axes(ax, xs, style: Style,
                      clamp: tuple[float, float] | None = (0.0, 1.0)) -> None:
    """Shared x-over-layers chrome; call once the y-limits are set. Every layer gets a tick
    (labels shrink past 16). *clamp* is the metric's real range, so the top-spine headroom
    cannot invent a tick outside it."""
    lo, hi = ax.get_ylim()
    ax.set_xlim(min(xs), max(xs))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="x", labelsize=6 if len(set(xs)) > 16 else 7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    if clamp is not None:
        ax.set_yticks([t for t in ax.get_yticks()
                       if max(lo, clamp[0]) <= t <= min(hi, clamp[1])])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(style.c["axis"])


def zoom_ylim(series, reference: float | None, full_range: bool,
              limits: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    """Zoom to the data, keeping *reference* in frame and headroom on top, so a
    saturating curve does not sit on the spine."""
    if full_range:
        return limits
    vals = [v for s in series for v in s if v is not None]
    if not vals:
        return limits
    lo, hi = min(vals), max(vals)
    if reference is not None:
        lo, hi = min(lo, reference), max(hi, reference)
    span = hi - lo
    lo = max(limits[0] - 0.04, lo - max(0.02, 0.08 * span))
    hi = min(limits[1] + 0.04, hi + max(0.03, 0.08 * span))
    if hi - lo < 0.12:  # never a hairline band
        mid = (lo + hi) / 2
        lo, hi = max(limits[0] - 0.04, mid - 0.06), min(limits[1] + 0.04, mid + 0.06)
    return lo, hi


def set_pub_style(style: Style, usetex: bool = False) -> None:
    """rcParams tuned for single-column print figures: quiet chrome, loud data."""
    c = style.c
    plt.rcParams.update({
        "figure.figsize": (3.4, 2.5),       # ~1 journal column
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        # Serif matches the LaTeX body text of the paper.
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "text.color": c["ink"],
        "axes.labelcolor": c["ink_soft"],
        "axes.edgecolor": c["axis"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["ink_soft"],
        "ytick.labelcolor": c["ink_soft"],
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.7,
        "lines.markersize": 3.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # Solid hairline grid, one step off the surface — never dashed.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": c["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "pdf.fonttype": 42,                 # embed editable TrueType, not bitmaps
        "ps.fonttype": 42,
    })
    if usetex:
        # Needs a working LaTeX install; drop --usetex if rendering fails.
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def add_style_args(parser: argparse.ArgumentParser) -> None:
    """The style flags every figure-producing command accepts."""
    group = parser.add_argument_group("figure style")
    group.add_argument("--palette", choices=sorted(PALETTES), default="okabe-ito",
                       help="Named color palette (default: okabe-ito).")
    group.add_argument("--colors", default=None,
                       help="Ad-hoc override: 'hex_a,hex_b'. Beats --palette.")
    group.add_argument("--theme", choices=["light", "dark"], default="light",
                       help="Surface the figure is drawn on (default: light, i.e. paper).")
    group.add_argument("--format", nargs="+", choices=["pdf", "png", "svg"], default=["svg"],
                       help="Figure format(s) (default: svg).")
    group.add_argument("--no-legend", dest="legend", action="store_false",
                       help="Draw no legend at all — for a figure whose series are "
                            "named in the caption or by a shared legend elsewhere.")
    group.add_argument("--usetex", action="store_true",
                       help="Render text with LaTeX (requires a LaTeX install).")
    group.add_argument("--full-range", action="store_true",
                       help="Force the y-axis to the metric's full range instead of "
                            "zooming to the data.")
    group.add_argument("--list-palettes", action="store_true",
                       help="Print the palette registry and exit.")


def style_from_args(args: argparse.Namespace) -> Style:
    style = Style(args.palette, args.theme, args.colors, legend=args.legend)
    set_pub_style(style, usetex=args.usetex)
    return style


def list_palettes() -> None:
    print("Palettes (two slots each):\n")
    for name, modes in PALETTES.items():
        light = "  ".join(modes["light"])
        dark = "  ".join(modes["dark"])
        print(f"  {name:<14} light: {light}    dark: {dark}")
    print("\nAd-hoc:  --colors '#2a78d6,#eb6834'")
