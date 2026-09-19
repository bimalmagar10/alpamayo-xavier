#!/usr/bin/env python3
"""Figure 3 -- cutting the graphs up, and reading the weights back out.

Left: why there are twenty-eight pieces. TensorRT's Myelin compiler fuses a whole
transformer into one kernel graph and then asks for a single allocation larger
than all its weights, which a 32 GB board cannot give it for a 15 GB graph. So
each graph is cut at layer boundaries into pieces of roughly 1.2-1.5 GB. Every
block in the bars is one piece, drawn to its real weight size from pieces.json.

Right: how decode reads the language model without TensorRT at all. Every weight
already sits in prefill.onnx.data; h100/a7_weight_map.py records where. On the
board the file is memory-mapped and the layer stack is built straight out of it,
so nothing is exported, copied or duplicated -- the strip at the top is the file,
the bar below it is one layer of it, and both are drawn from the recorded byte
ranges.

    python analysis/fig03_pieces_and_weightmap.py

Output: analysis/figures/fig03_pieces_and_weightmap.{pdf,png}
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

ORDER = ["prefill", "decode", "expert"]
# Darkened stage colours: the same hues the rest of the figures use, at print weight.
DARK = {"prefill": "#1D3F6B", "decode": "#8A3F08", "expert": "#473C6B"}
# The MLP dominates a layer, so the ramp runs dark for the big tensors.
RAMP = ["#12304F", "#1D4A73", "#2A6491", "#3E7EAC"]


def load_pieces(root):
    path = os.path.join(root, "onnx", "pieces.json")
    if not os.path.exists(path):
        raise SystemExit("no pieces.json under %s/onnx" % root)
    return json.load(open(path))["graphs"]


def load_layer(root):
    path = os.path.join(root, "onnx", "weight_map.json")
    if not os.path.exists(path):
        raise SystemExit("no weight_map.json under %s/onnx" % root)
    spec = json.load(open(path))["graphs"]["prefill"]
    return spec["layers"][0], spec["head"], len(spec["layers"])


# ---------------------------------------------------------------------------
def panel_pieces(ax, graphs):
    y = np.arange(len(ORDER))[::-1]
    for yi, name in zip(y, ORDER):
        pieces = graphs[name]["pieces"]
        left = 0.0
        for p in pieces:
            w = p["weight_bytes"] / 1e9
            ax.barh(yi, w, left=left, height=0.52, color=DARK[name],
                    edgecolor="white", linewidth=0.7, zorder=3)
            left += w
        ax.annotate("%d pieces" % len(pieces), xy=(left, yi), xytext=(5, 3),
                    textcoords="offset points", fontsize=6.4, va="bottom",
                    ha="left", color=DARK[name])
        # A head piece that carries only a norm is invisible at this scale, so
        # the block count would not match the label unless it is spelled out.
        tiny = [q for q in pieces if q["weight_bytes"] < 1e7]
        if tiny:
            ax.annotate("head is a %.0f kB norm" % (tiny[0]["weight_bytes"] / 1e3),
                        xy=(left, yi), xytext=(5, -3), textcoords="offset points",
                        fontsize=5.6, va="top", ha="left", color=C["muted"])

    ax.set_yticks(y)
    ax.set_yticklabels(ORDER, fontsize=7.4)
    ax.set_xlabel("weights (GB)")
    ax.set_xlim(0, 18.6)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.tick_params(axis="y", length=0)
    style.grid(ax, axis="x")


# ---------------------------------------------------------------------------
def panel_weightmap(ax, layer, head, n_layers):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, x1 = 0.02, 0.98
    span = x1 - x0
    layer_bytes = sum(w["length"] for w in layer.values())
    head_bytes = sum(w["length"] for w in head.values())
    total = n_layers * layer_bytes + head_bytes

    # ---- the file, one slice per layer ---------------------------------
    ytop, h = 0.88, 0.085
    lw_frac = layer_bytes / total * span
    for i in range(n_layers):
        ax.add_patch(plt.Rectangle((x0 + i * lw_frac, ytop - h), lw_frac, h,
                                   facecolor=C["line"] if i else RAMP[1],
                                   edgecolor="white", linewidth=0.35, zorder=3))
    ax.add_patch(plt.Rectangle((x0 + n_layers * lw_frac, ytop - h),
                               head_bytes / total * span, h,
                               facecolor=C["muted"], edgecolor="white",
                               linewidth=0.35, zorder=3))
    ax.text(x0, ytop + 0.035, "prefill.onnx.data", fontsize=6.6, va="bottom")
    ax.text(x1, ytop + 0.035, "%.2f GB" % (total / 1e9), fontsize=6.4,
            va="bottom", ha="right", color=C["muted"])
    ax.text(x0 + n_layers * lw_frac + head_bytes / total * span / 2, ytop - h - 0.035,
            "lm_head", fontsize=5.6, ha="center", va="top", color=C["muted"])
    ax.text(x0 + lw_frac / 2, ytop - h - 0.035, "layer 0", fontsize=5.6,
            ha="center", va="top", color=RAMP[1])

    # ---- that slice, expanded ------------------------------------------
    ymid, h2 = 0.50, 0.105
    ax.add_patch(plt.Polygon([(x0, ytop - h - 0.005), (x0 + lw_frac, ytop - h - 0.005),
                              (x1, ymid + h2 / 2 + 0.005), (x0, ymid + h2 / 2 + 0.005)],
                             closed=True, facecolor=C["line"], alpha=0.30,
                             edgecolor="none", zorder=1))
    order = ["input_ln", "q", "k", "v", "q_norm", "k_norm", "o", "post_ln",
             "gate", "up", "down"]
    shade = {"q": RAMP[2], "o": RAMP[2], "k": RAMP[3], "v": RAMP[3],
             "gate": RAMP[0], "up": RAMP[1], "down": RAMP[0]}
    left = x0
    for role in order:
        w = layer[role]["length"] / layer_bytes * span
        ax.add_patch(plt.Rectangle((left, ymid - h2 / 2), w, h2,
                                   facecolor=shade.get(role, C["muted"]),
                                   edgecolor="white", linewidth=0.5, zorder=3))
        if w > 0.055:
            ax.text(left + w / 2, ymid, role, fontsize=5.8, ha="center", va="center",
                    color="white", zorder=4)
        left += w
    ax.text(x0, ymid + h2 / 2 + 0.035, "one layer, 11 tensors", fontsize=6.4, va="bottom")
    ax.text(x1, ymid + h2 / 2 + 0.035, "%.1f MB" % (layer_bytes / 1e6), fontsize=6.4,
            va="bottom", ha="right", color=C["muted"])
    ax.text(x0, ymid - h2 / 2 - 0.035, "norms and the 128-element q/k norms are "
            "too small to see here", fontsize=5.5, va="top", color=C["muted"])

    # ---- what the board does with it ------------------------------------
    ax.annotate("", xy=(0.5, 0.185), xytext=(0.5, 0.325),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C["ink"]))
    ax.text(0.53, 0.255, "np.memmap, no copy", fontsize=6.2, va="center",
            ha="left", color=C["ink"])
    ax.add_patch(plt.Rectangle((x0, 0.035), span, 0.135, facecolor="none",
                               edgecolor=C["ink"], linewidth=0.9, zorder=3))
    ax.text(0.5, 0.125, "torch decode on the Xavier", fontsize=6.6,
            ha="center", va="center")
    ax.text(0.5, 0.072, "%d layers + head built straight from the file · %.2f GB"
            % (n_layers, total / 1e9), fontsize=5.9, ha="center", va="center",
            color=C["muted"])


# ---------------------------------------------------------------------------
def build(graphs, layer, head, n_layers, outdir):
    style.use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(style.FULL, 2.5), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.12], wspace=0.06))
    panel_pieces(ax_a, graphs)
    panel_weightmap(ax_b, layer, head, n_layers)
    style.save(fig, "fig03_pieces_and_weightmap", outdir)
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
    graphs = load_pieces(root)
    layer, head, n_layers = load_layer(root)

    print("source   : %s/onnx\n" % root)
    total_pieces = 0
    for name in ORDER:
        pieces = graphs[name]["pieces"]
        wb = [p["weight_bytes"] for p in pieces]
        total_pieces += len(pieces)
        print("%-8s %2d pieces  %2d layers each  total %6.2f GB  "
              "smallest %.2f  largest %.2f"
              % (name, len(pieces), graphs[name]["layers_per_piece"],
                 sum(wb) / 1e9, min(wb) / 1e9, max(wb) / 1e9))
    print("%-8s %2d pieces in total" % ("", total_pieces))

    layer_bytes = sum(w["length"] for w in layer.values())
    head_bytes = sum(w["length"] for w in head.values())
    print("\none layer of the language model: %s bytes (%.1f MB) in %d tensors"
          % (format(layer_bytes, ",d"), layer_bytes / 1e6, len(layer)))
    for role, w in sorted(layer.items(), key=lambda kv: -kv[1]["length"]):
        print("  %-9s %-17s %13s bytes  %5.1f%%"
              % (role, str(w["dims"]), format(w["length"], ",d"),
                 100 * w["length"] / layer_bytes))
    print("  head     %-17s %13s bytes" % ("lm_head + norm", format(head_bytes, ",d")))
    print("  %d layers + head = %s bytes (%.2f GB)"
          % (n_layers, format(n_layers * layer_bytes + head_bytes, ",d"),
             (n_layers * layer_bytes + head_bytes) / 1e9))

    build(graphs, layer, head, n_layers, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
