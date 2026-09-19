#!/usr/bin/env python3
"""Figure 2 -- what the export costs, per component.

One panel. For each of the four exported graphs, two bars: the size the
component occupies in the original checkpoint, and the size it occupies on disk
after torch.onnx.export. The parameter count sits above each pair.

The figure exists to make one thing visible: decode has no bar on the left. In
the checkpoint the language model is a single module, called two different ways
-- over the whole prompt, and over one token against a cache. It is the export
that turns those two call patterns into two files, because a traced graph is
frozen at one input shape. Nothing is shared afterwards: the 15.17 GB is written
out twice, byte for byte (see analysis/check_export_weights.py).

Sizes are read off the files; parameter counts come from the weight map where it
exists and from bytes / 2 otherwise, which agrees with the map to 0.004%.

    python analysis/fig02_export_sizes.py

Output: analysis/figures/fig02_export_sizes.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_figs import data, style                             # noqa: E402
from alpamayo_figs.style import C                                 # noqa: E402

GRAPHS = ["vision", "prefill", "decode", "expert"]
# decode adds no parameters to the checkpoint: it is the same module as prefill.
SHARES_WITH = {"decode": "prefill"}

CHECKPOINT = "#1F4E79"      # dark blue: what the checkpoint holds
EXPORTED = "#9E2A2B"        # dark red: what the export writes


def measure(root):
    """Per graph: parameters, bytes in the checkpoint, bytes after export."""
    onnx_dir = os.path.join(root, "onnx")
    wmap = {}
    path = os.path.join(onnx_dir, "weight_map.json")
    if os.path.exists(path):
        for name, spec in json.load(open(path))["graphs"].items():
            n = 0
            for layer in spec["layers"] + [spec["head"]]:
                for w in layer.values():
                    n += int(np.prod(w["dims"]))
            wmap[name] = n

    out = []
    for g in GRAPHS:
        proto = os.path.join(onnx_dir, g + ".onnx")
        blob = os.path.join(onnx_dir, g + ".onnx.data")
        if not os.path.exists(blob):
            continue
        data_bytes = os.path.getsize(blob)
        proto_bytes = os.path.getsize(proto) if os.path.exists(proto) else 0
        params = wmap.get(g, data_bytes // 2)
        out.append(dict(name=g, params=params, weights=data_bytes,
                        graph=proto_bytes, exported=data_bytes + proto_bytes,
                        checkpoint=0 if g in SHARES_WITH else data_bytes,
                        shared=SHARES_WITH.get(g),
                        counted=g not in wmap))
    return out


def panel_bars(ax, rows):
    x = np.arange(len(rows))
    w = 0.22
    ck = [r["checkpoint"] / 1e9 for r in rows]
    ex = [r["exported"] / 1e9 for r in rows]
    top = max(ex) * 1.34

    wt = [r["weights"] / 1e9 for r in rows]
    gr = [r["graph"] / 1e9 for r in rows]
    ax.bar(x - w / 2, ck, w, color=CHECKPOINT, label="in the checkpoint", zorder=3)
    ax.bar(x + w / 2, wt, w, color=EXPORTED, label="exported to ONNX", zorder=3)
    ax.bar(x + w / 2, gr, w, bottom=wt, facecolor=EXPORTED, hatch="////",
           edgecolor="white", linewidth=0.0, zorder=4,
           label="graph constants: 36 causal masks")

    for xi, r, a, b in zip(x, rows, ck, ex):
        if a > 0:
            ax.annotate("%.2f" % a, xy=(xi - w / 2, a), xytext=(-2, 2), fontsize=5.6,
                        textcoords="offset points", ha="right", va="bottom",
                        color=CHECKPOINT)
        else:
            ax.annotate("0", xy=(xi - w / 2, 0), xytext=(-2, 2), fontsize=5.6,
                        textcoords="offset points", ha="right", va="bottom",
                        color=CHECKPOINT)
        ax.annotate("%.2f" % b, xy=(xi + w / 2, b), xytext=(2, 2), fontsize=5.6,
                    textcoords="offset points", ha="left", va="bottom", color=EXPORTED)
        label = "%.3f B" % (r["params"] / 1e9)
        if r["shared"]:
            label += "\nsame as %s" % r["shared"]
        ax.annotate(label, xy=(xi, top * 0.90), fontsize=6.2, ha="center", va="top",
                    color=C["ink"])

    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], fontsize=7.4)
    ax.set_ylabel("weights on disk (GB)")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              handlelength=1.0, handletextpad=0.5, columnspacing=1.2,
              labelspacing=0.35, borderaxespad=0.0, fontsize=6.2)
    style.grid(ax)


def panel_why(ax):
    """Three parameter blocks on the left, four traced graphs on the right.

    The whole answer to "why four, and why these four" is the arrow count: one
    arrow per (parameter block, input shape) pair that a frame actually needs.
    The language model is the only block used at two shapes, so it is the only
    one traced twice -- and both traces write the same weights.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    LW, RW, H = 0.30, 0.32, 0.105
    LX, RX = 0.02, 0.66

    def box(x, y, w, name, sub, colour, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y - H / 2), w, H,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=1.0, edgecolor=colour,
                                    facecolor=colour if not dashed else "none",
                                    alpha=1.0 if not dashed else 1.0,
                                    linestyle="--" if dashed else "-", zorder=3))
        ax.text(x + w / 2, y + 0.018, name, fontsize=6.4, ha="center", va="center",
                color="white" if not dashed else C["muted"], zorder=4)
        ax.text(x + w / 2, y - 0.024, sub, fontsize=5.6, ha="center", va="center",
                color="white" if not dashed else C["muted"], zorder=4)

    def arrow(y0, y1, style="-", colour=None, label=None, dy=0.0):
        ax.annotate("", xy=(RX - 0.008, y1), xytext=(LX + LW + 0.008, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=colour or C["muted"], linestyle=style,
                                    connectionstyle="arc3,rad=0.0"), zorder=2)
        if label:
            ax.text((LX + LW + RX) / 2, (y0 + y1) / 2 + dy, label, fontsize=5.4,
                    ha="center", va="bottom" if dy > 0 else "top",
                    color=C["muted"], zorder=5)

    ax.text(LX + LW / 2, 0.975, "checkpoint", fontsize=6.4, ha="center",
            color=CHECKPOINT)
    ax.text(RX + RW / 2, 0.975, "traced graphs", fontsize=6.4, ha="center",
            color=EXPORTED)

    yv, ylm, yex, yem = 0.845, 0.545, 0.235, 0.055
    box(LX, yv, LW, "vision tower", "0.576 B", CHECKPOINT)
    box(LX, ylm, LW, "language model", "7.584 B", CHECKPOINT)
    box(LX, yex, LW, "action expert", "2.282 B", CHECKPOINT)
    box(LX, yem, LW, "token embeddings", "0.638 B", C["line"], dashed=True)

    box(RX, yv, RW, "vision", "1.16 GB", EXPORTED)
    box(RX, 0.655, RW, "prefill", "15.82 GB", EXPORTED)
    box(RX, 0.435, RW, "decode", "15.17 GB", EXPORTED)
    box(RX, yex, RW, "expert", "4.56 GB", EXPORTED)
    box(RX, yem, RW, "fixture, not a graph", "1.28 GB", C["line"], dashed=True)

    arrow(yv, yv)
    arrow(ylm, 0.655, colour=EXPORTED, label="3\u2009006 positions", dy=0.048)
    arrow(ylm, 0.435, colour=EXPORTED, label="1 position + cache", dy=-0.048)
    arrow(yex, yex)
    arrow(yem, yem, style="--")


def build(rows, outdir):
    style.use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(style.FULL, 2.45), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.06))
    panel_bars(ax_a, rows)
    panel_why(ax_b)
    style.save(fig, "fig02_export_sizes", outdir)
    style.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default=os.path.join(here, "figures"))
    args = ap.parse_args()

    root = data.find_root(args.root)
    if root is None:
        raise SystemExit("no alpamayo root with an onnx/ directory; pass --root")
    rows = measure(root)
    if not rows:
        raise SystemExit("no .onnx.data files under %s/onnx" % root)

    print("source        : %s/onnx\n" % root)
    print("%-9s %14s %12s %12s %12s   %s"
          % ("graph", "parameters", "checkpoint", "weights", "graph file", "note"))
    print("-" * 88)
    for r in rows:
        note = "shared with %s" % r["shared"] if r["shared"] else ""
        if r["counted"]:
            note = (note + "; " if note else "") + "params from bytes/2"
        print("%-9s %14s %9.3f GB %9.3f GB %9.3f GB   %s"
              % (r["name"], format(r["params"], ",d"), r["checkpoint"] / 1e9,
                 r["weights"] / 1e9, r["graph"] / 1e9, note))
    print("-" * 88)
    ck = sum(r["checkpoint"] for r in rows)
    ex = sum(r["exported"] for r in rows)
    print("%-9s %14s %11.3f GB %11.3f GB"
          % ("total", format(sum(r["params"] for r in rows if not r["shared"]), ",d"),
             ck / 1e9, ex / 1e9))
    print("\n  the export adds %.2f GB: the language model written a second time"
          % ((ex - ck) / 1e9))
    onnx_dir = os.path.join(root, "onnx")
    proto = os.path.getsize(os.path.join(onnx_dir, "prefill.onnx"))
    mask = 3006 * 3006 * 2
    print("  of which %.2f GB is inline constants in prefill.onnx --"
          % (proto / 1e9))
    print("  36 copies of the 3006x3006 fp16 causal mask (%s bytes) plus the graph itself"
          % format(36 * mask, ",d"))

    build(rows, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
