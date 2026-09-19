#!/usr/bin/env python3
"""Figure 5 -- end-to-end latency of one frame, averaged over the repeats.

A single horizontal bar: the mean wall time of a frame, split into the
telemetry's exclusive scopes. Exclusive means the scopes do not overlap, so they
stack to the measured frame wall with nothing double counted and nothing hidden.

The run is `--repeat 3` over one clip, so the mean mixes a process-cold first
repeat with two warm ones. The thin range bar underneath shows that spread
rather than letting the average bury it.

    python analysis/fig05_latency.py
    python analysis/fig05_latency.py --json path/to/other.json

Output: analysis/figures/fig05_latency.{pdf,png}
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
from alpamayo_figs import style                                   # noqa: E402
from alpamayo_figs.style import C                                 # noqa: E402

DEFAULT = "results/xavier-results/fp16_telemetry.json"

# Dark, print-safe, and separable in greyscale by luminance.
SCOPE = [("engine load", "#7A2E23"), ("compute", "#1F4E79"), ("release", "#52606E"),
         ("preprocess", "#3F6B52"), ("other", "#9AA1A8")]
# The model stages, in the colours the rest of the figures give them.
STAGE = [("vision", "#0B6B63"), ("prefill", "#1D3F6B"),
         ("decode", "#8A3F08"), ("expert", "#473C6B")]


def classify(key):
    """Group an exclusive wall scope into something a reader can act on."""
    if key == "stage:engine load" or key.startswith("engine_load:"):
        return "engine load"
    if (key.startswith("engine_execute:") or key.startswith("flow_step")
            or key in ("stage:decode", "stage:embed assembly")):
        return "compute"
    if key.startswith("engine_release:") or key.startswith("stage_release:"):
        return "release"
    if key == "preprocess":
        return "preprocess"
    return "other"


def scopes(run):
    timing = run["telemetry"]["timing"]
    out = {name: 0.0 for name, _ in SCOPE}
    for key, ms in timing["exclusive_wall_ms"].items():
        out[classify(key)] += ms
    out["other"] += timing.get("uninstrumented_wall_ms", 0.0)
    return out


def panel_total(ax, runs):
    """The mean frame, split into the telemetry's non-overlapping scopes."""
    per = [scopes(r) for r in runs]
    mean = {name: float(np.mean([p[name] for p in per])) / 1000.0 for name, _ in SCOPE}
    totals = np.array([r["frame_wall_ms"] / 1000.0 for r in runs])

    left = 0.0
    for name, colour in SCOPE:
        v = mean[name]
        ax.barh(0, v, left=left, height=0.42, color=colour, edgecolor="white",
                linewidth=0.6, zorder=3, label="%s  %.1f s" % (name, v))
        if v / totals.mean() > 0.06:
            ax.annotate("%.1f s" % v, xy=(left + v / 2, 0), fontsize=5.8,
                        ha="center", va="center", color="white", zorder=5)
        left += v
    ax.annotate("%.1f s" % left, xy=(left, 0), xytext=(4, 0), fontsize=6.8,
                textcoords="offset points", va="center", ha="left", color=C["ink"])

    y = -0.42
    ax.plot([totals.min(), totals.max()], [y, y], "-", color=C["muted"], lw=0.9, zorder=4)
    for t in (totals.min(), totals.max()):
        ax.plot([t, t], [y - 0.05, y + 0.05], "-", color=C["muted"], lw=0.9, zorder=4)
    ax.annotate("%.0f\u2013%.0f s" % (totals.min(), totals.max()),
                xy=(totals.max(), y), xytext=(4, 0), textcoords="offset points",
                fontsize=5.6, va="center", ha="left", color=C["muted"])

    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.34)
    ax.set_xlim(0, totals.max() * 1.16)
    ax.set_xlabel("wall time per frame (s)", fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.50), ncol=3, fontsize=5.2,
              handlelength=0.8, handletextpad=0.35, columnspacing=0.8,
              labelspacing=0.28, borderaxespad=0.0)
    ax.spines["left"].set_visible(False)
    style.grid(ax, axis="x")
    return mean, totals


def panel_stages(ax, runs):
    """Mean time in each model stage, with the spread across repeats."""
    x = np.arange(len(STAGE))
    means, lows, highs = [], [], []
    for name, _ in STAGE:
        v = np.array([r["stages"][name] / 1000.0 for r in runs])
        means.append(v.mean())
        lows.append(v.min())
        highs.append(v.max())
    ax.bar(x, means, 0.56, color=[c for _, c in STAGE], zorder=3)
    for xi, m, lo, hi in zip(x, means, lows, highs):
        if hi - lo > 0.05:
            ax.plot([xi, xi], [lo, hi], "-", color=C["ink"], lw=0.8, zorder=5)
            for t in (lo, hi):
                ax.plot([xi - 0.08, xi + 0.08], [t, t], "-", color=C["ink"], lw=0.8,
                        zorder=5)
        ax.annotate("%.2f" % m, xy=(xi, max(m, hi)), xytext=(0, 2), fontsize=5.8,
                    textcoords="offset points", ha="center", va="bottom",
                    color=C["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in STAGE], fontsize=6.4)
    ax.set_ylabel("seconds", fontsize=7)
    ax.set_ylim(0, max(highs) * 1.22)
    ax.tick_params(axis="x", length=0)
    style.grid(ax)
    return means


def build(runs, outdir):
    style.use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 1.62), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.05))
    mean, totals = panel_total(ax_a, runs)
    stage_means = panel_stages(ax_b, runs)
    style.save(fig, "fig05_latency", outdir)
    style.close(fig)
    return mean, totals, stage_means


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=os.path.join(repo, DEFAULT))
    ap.add_argument("--outdir", default=os.path.join(here, "figures"))
    args = ap.parse_args()

    if not os.path.exists(args.json):
        raise SystemExit("no telemetry at %s" % args.json)
    doc = json.load(open(args.json))
    runs = doc["runs"]

    print("telemetry : %s" % args.json)
    print("run       : %s, %s, %s, residency %s, %d repeats of one clip\n"
          % (doc["started_at_utc"], doc["precision"], doc["device"],
             doc["configuration"]["effective_residency"], len(runs)))

    mean, totals, stage_means = build(runs, args.outdir)
    print("mean of %d repeats" % len(runs))
    for name, _ in SCOPE:
        print("  %-13s %8.1f s   %5.1f%%"
              % (name, mean[name], 100 * mean[name] / sum(mean.values())))
    print("  %-13s %8.1f s" % ("end to end", sum(mean.values())))
    print("\nmean per stage")
    for (name, _), v in zip(STAGE, stage_means):
        spread = [r["stages"][name] / 1000.0 for r in runs]
        print("  %-9s %6.2f s   (%.2f\u2013%.2f across repeats)"
              % (name, v, min(spread), max(spread)))
    print("\nper repeat  : %s" % "   ".join("%.1f s" % t for t in totals))
    print("note        : the mean mixes one process-cold repeat with two warm ones; "
          "the range bar shows that spread")
    print("\ncaveats carried from the telemetry record")
    for n in doc["telemetry_notes"]:
        print("  - %s" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
