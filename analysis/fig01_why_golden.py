#!/usr/bin/env python3
"""Figure 1 -- why the golden run is recorded, and what it stores.

Two panels, one claim each.

  (a) Agreement with the golden run, stage by stage. The translated pipeline
      matches everywhere. The same pipeline carrying the fp16 RMSNorm overflow
      that this project shipped is identical through preprocessing and vision,
      then collapses at prefill. Both versions emit a trajectory; only the
      per-stage comparison says which stage is wrong.

  (b) What the golden run actually stores, with the dimension or the value that
      matters for each component. Read out of golden/*.npz at run time, so the
      table cannot drift away from the file it describes.

    python analysis/fig01_why_golden.py
    python analysis/fig01_why_golden.py --root /path/to/alpamayo-work

Output: analysis/figures/fig01_why_golden.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_figs import data, facts, style                      # noqa: E402
from alpamayo_figs.style import C                                 # noqa: E402

# The five stages verify.py compares, in pipeline order.
STAGES = ["pixels", "vision", "prefill", "decode", "expert"]
HEALTHY = [facts.VERIFY[0]["cos"], facts.VERIFY[1]["cos"], facts.VERIFY[2]["cos"],
           facts.VERIFY[3]["cos"], facts.VERIFY[5]["cos"]]
# The defect was measured through prefill; the run was abandoned there.
DEFECT = [facts.VERIFY[0]["cos"], facts.VERIFY[1]["cos"], facts.DEFECTS[0]["cos"]]


# ---------------------------------------------------------------------------
# (a) agreement per stage
# ---------------------------------------------------------------------------
def panel_agreement(ax):
    x = np.arange(len(STAGES))
    ax.axhspan(facts.ACCEPT_THRESHOLD, 1.03, color=C["ok"], alpha=0.10, lw=0)
    ax.axhline(facts.ACCEPT_THRESHOLD, color=C["ok"], lw=0.8, ls=(0, (4, 2)))

    ax.plot(x, HEALTHY, "-o", color=C["ok"], mfc="white", mew=1.1, ms=4.6, zorder=6,
            label="translated pipeline")
    ax.plot(x[:len(DEFECT)], DEFECT, "-X", color=C["fault"], ms=5.6, lw=1.2, zorder=7,
            label="fp16 RMSNorm defect")

    # Direct labels: a legend box here would sit on top of the defect line.
    ax.annotate("every stage $\\geq$ 0.9996", xy=(2.25, 1.045), fontsize=6.4,
                color=C["ok"], ha="center", va="center")
    ax.annotate("identical through\npixels and vision", xy=(-0.40, 0.79), fontsize=6.0,
                color=C["muted"], ha="left", va="center")
    ax.annotate("prefill is wrong,\nand only this\ncomparison says so",
                xy=(2, 0.02), xytext=(2.45, 0.30), fontsize=6.6, color=C["fault"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C["fault"]))

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=7)
    ax.set_xlim(-0.45, len(STAGES) - 0.55)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("cosine similarity")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 0.90),
              handletextpad=0.5, labelspacing=0.45)
    style.grid(ax)


# ---------------------------------------------------------------------------
# (b) what is stored
# ---------------------------------------------------------------------------
def _peak(golden, *keys):
    vals = [float(np.abs(golden[k].astype(np.float64)).max()) for k in keys if k in golden]
    return max(vals) if vals else None


def _shape(golden, key):
    if key not in golden:
        return "—"
    return " × ".join(format(d, ",d").replace(",", " ")
                           for d in golden[key].shape if d != 1) or "scalar"


def table_rows(golden):
    """(component, the dimension or value that matters) -- read from the file."""
    thin = lambda n: format(int(n), ",d").replace(",", "\u2009")   # noqa: E731
    dot = "  \u00b7  "
    return [
        ("camera pixels, patched", _shape(golden, "in_pixel_values")),
        ("prompt token ids",
         _shape(golden, "in_input_ids") + dot + thin(facts.MODEL["vocab"]) + " vocabulary"),
        ("ego history", _shape(golden, "in_ego_history_xyz")),
        ("visual embeddings", _shape(golden, "visual")),
        ("DeepStack features $\\times$ 3",
         _shape(golden, "deepstack0") + dot + "injected at layers 0\u20132"),
        ("planned trajectory",
         _shape(golden, "pred_xyz") + dot + "%.1f m over 6.4 s" % (_peak(golden, "pred_xyz") or 0)),
        ("ego heading", _shape(golden, "pred_rot") + dot + "rotation matrices"),
    ]


def panel_table(ax, golden):
    """Two columns, one row per component, styled like the project's other tables."""
    rows = table_rows(golden)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bot = 0.96, 0.03
    step = (top - bot) / len(rows)
    pad = 0.022

    def rule(y, lw, colour):
        ax.plot([0, 1], [y, y], color=colour, lw=lw, solid_capstyle="butt",
                clip_on=False, zorder=4)

    rule(top, 0.9, C["ink"])
    y = top
    for name, value in rows:
        y -= step
        missing = value == "not recorded"
        ax.text(pad, y + step * 0.42, name, fontsize=6.5, va="center", ha="left",
                color=C["muted"] if missing else C["ink"], zorder=5)
        ax.text(1 - pad, y + step * 0.42, value, fontsize=6.5, va="center", ha="right",
                color=C["fault"] if missing else C["ink"], zorder=5)
        rule(y, 0.5, C["line"])
    rule(y, 0.9, C["ink"])


# ---------------------------------------------------------------------------
def build(golden, synthetic, outdir):
    style.use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(style.FULL, 2.55), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.35], wspace=0.04))

    panel_agreement(ax_a)
    panel_table(ax_b, golden)

    if synthetic:
        style.stamp_synthetic(fig)
    style.save(fig, "fig01_why_golden", outdir)
    style.close(fig)


def inventory(golden):
    print("%-22s %-22s %-9s %10s %12s" % ("key", "shape", "dtype", "size", "max |x|"))
    print("-" * 80)
    total = 0
    for k in sorted(golden):
        a = golden[k]
        total += a.nbytes
        try:
            peak = "%12.4g" % np.abs(a.astype(np.float64)).max()
        except (TypeError, ValueError):
            peak = "%12s" % "n/a"
        print("%-22s %-22s %-9s %8.2f MB %s"
              % (k, str(a.shape), a.dtype, a.nbytes / 1e6, peak))
    print("-" * 80)
    print("%-22s %47.2f MB" % ("total recorded", total / 1e6))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="the alpamayo work root holding golden/")
    ap.add_argument("--outdir", default=os.path.join(here, "figures"))
    ap.add_argument("--synthetic", action="store_true",
                    help="run without golden data, for layout only; output is stamped")
    args = ap.parse_args()

    if args.synthetic:
        golden, where = data.synthetic_golden(), "SYNTHETIC -- no measurement in this figure"
    else:
        golden, where = data.load_golden(args.root)
        if golden is None:
            print("no golden data: %s" % where)
            print("run this where golden/ lives, pass --root, or use --synthetic")
            return 2

    print("golden source : %s\n" % where)
    inventory(golden)
    print()
    build(golden, args.synthetic, args.outdir)

    print("\nprovenance")
    print("  panel (a) cosines      : %s" % facts.VERIFY_SOURCE)
    print("  panel (a) defect trace : %s" % facts.DEFECTS[0]["source"])
    print("  panel (b)              : read from golden/*.npz at run time")
    print("  sampled reasoning      : absent -- a1_golden.py prints the trace, "
          "never saves it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
