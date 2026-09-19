"""Figure style for the Alpamayo-on-Xavier analysis.

Conference-paper conventions, not slide conventions: serif type, no chartjunk,
vector output, and a column width that survives being dropped into LaTeX at 100%
scale. NeurIPS single-column text is 5.5 in wide, so FULL is 5.5 and anything
wider will be silently downscaled by the template and lose its point sizes.

Everything here is deliberate about one thing: a figure in a paper is read at
print size, so nothing may rely on colour alone, and nothing may be smaller than
7 pt after scaling.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL = 5.5          # NeurIPS text width, inches
WIDE = 7.1          # a figure* spanning both columns
HALF = 2.65         # two panels side by side with a gutter
ROW = 1.9           # a comfortable panel height

# Stage colours, carried over from the project's other artefacts so the same
# stage is the same colour everywhere. Checked for colour-blind separability and
# for staying distinguishable when printed greyscale (luminance is monotone).
C = {
    "vision": "#0B6B63",
    "prefill": "#27548A",
    "decode": "#B4530A",
    "expert": "#5E4F8D",
    "ink": "#15191C",
    "muted": "#6B7780",
    "line": "#C3CBD1",
    "ok": "#256B3C",
    "fault": "#A32318",
    "accent": "#B4530A",
    "band": "#ECEFF1",      # header fill for tables, as in the project's other artefacts
}

# Markers, so the figure survives greyscale printing and photocopying.
M = {"vision": "o", "prefill": "s", "decode": "^", "expert": "D"}


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": C["line"],
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.6,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "pdf.fonttype": 42,          # editable text in the PDF, not outlines
        "ps.fonttype": 42,
    })


def panel(ax, letter, title=None, dx=-0.155, dy=1.16):
    """(a), (b), ... in the corner, with an optional short panel title."""
    ax.text(dx, dy, "(%s)" % letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")
    if title:
        ax.set_title(title, loc="left", pad=4)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=C["line"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def stamp_synthetic(fig):
    """Make fabricated data impossible to mistake for a measurement.

    A preview run with no golden file still produces a figure, because iterating
    on layout should not require the cluster -- but it must never be possible to
    paste that preview into a talk by accident.
    """
    fig.text(0.5, 0.5, "SYNTHETIC PREVIEW\nnot measured data", fontsize=26,
             color="#C0392B", alpha=0.20, ha="center", va="center",
             rotation=24, fontweight="bold", zorder=1000)


def table(ax, rows, top=0.96, bot=0.03, fontsize=6.5, pad=0.022):
    """A flat two-column table: [(left, right)] or [(left, right, colour)].

    One row per entry, no grouping and no header -- the panel around it carries
    the meaning. Rules above and below, a hairline between rows, left column
    ranged left and right column ranged right, which is what makes a column of
    mixed shapes and magnitudes scannable.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    step = (top - bot) / max(len(rows), 1)

    def rule(y, lw, colour):
        ax.plot([0, 1], [y, y], color=colour, lw=lw, solid_capstyle="butt",
                clip_on=False, zorder=4)

    rule(top, 0.9, C["ink"])
    y = top
    for row in rows:
        left, right = row[0], row[1]
        colour = row[2] if len(row) > 2 else None
        y -= step
        ax.text(pad, y + step * 0.42, left, fontsize=fontsize, va="center", ha="left",
                color=colour or C["ink"], zorder=5)
        ax.text(1 - pad, y + step * 0.42, right, fontsize=fontsize, va="center",
                ha="right", color=colour or C["ink"], zorder=5)
        rule(y, 0.5, C["line"])
    rule(y, 0.9, C["ink"])


def save(fig, name, outdir, also_png=True):
    os.makedirs(outdir, exist_ok=True)
    written = []
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(pdf)
    written.append(pdf)
    if also_png:
        png = os.path.join(outdir, name + ".png")
        fig.savefig(png)
        written.append(png)
    for path in written:
        print("wrote %s  (%.0f kB)" % (path, os.path.getsize(path) / 1024))
    return written


def close(fig):
    plt.close(fig)
