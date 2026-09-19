#!/usr/bin/env python3
"""The largest activations in every layer of all three stacks, against the fp16 wall.

This is the picture behind the RMSNorm defect. RMSNorm squares its input before
it reduces, so fp16 needs |x| <= sqrt(65504) = 255.94 for x**2 to exist at all.
The question this answers is: which layers of which stack cross that line, by how
much, and -- because the fix depends on it -- whether the values that cross sit
in the same few channels every time.

Two figures, each three panels -- vision tower (27 blocks), language model (36
layers, the weights prefill and decode share) and action expert (36 layers):

  fig_activations          the three largest |activations| per layer, against
                           both fp16 ceilings
  fig_activation_surface   |x| over tokens x channels at each stack's peak
                           layer, which is where the outlier columns show

Nothing in this model comes close to fp16's 65 504 -- the largest value measured
is 26 240, 40% of it. That is the point: RMSNorm squares before it reduces, so
26 240^2 = 6.9e8 is what overflows, and the ceiling that binds is the square
root of the real one.

    bash study/run_study.sh act              # on the H100
    python study/activation_study.py --stage plot     # redraw from the saved files

Diagnostic borrowed from PrefixQuant (Chen et al., github.com/ChenMnZ/PrefixQuant,
plot_activation.py), which plots `activation.abs()` per layer against an
`outlier_threshold` to find the channels that break post-training quantization.
Their drawing code lives in utils/plot_utils.py and is not reproduced here; what
is taken is the diagnostic -- look at the largest magnitudes per layer, and at
which channel they land in, before trusting any low-precision format. Here the
threshold is not a tuning knob, it is fp16's own 255.94.

Captured in bfloat16 on purpose. fp16 is the precision under test; measuring in
it would let the very overflow we are looking for corrupt the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "h100"))

TOPK = 3
SURF_TOKENS = 256        # token rows kept in the npz for the 3D view
SURF_PLOT_ROWS = 64      # token rows actually drawn; every channel is kept
FP16_MAX = 65504.0
FP16_SQUARE_SAFE = FP16_MAX ** 0.5          # 255.94
STACKS = ["vision", "language model", "expert"]


def stats(x, k=TOPK):
    """Top-k |values| of one residual stream, with the context to read them honestly.

    `chan` is what turns this from a magnitude plot into a quantization verdict:
    outliers that keep landing in the same column are survivable per-channel and
    fatal per-tensor.
    """
    import torch
    a = x.detach().float().abs()
    width = a.shape[-1]
    flat = a.reshape(-1)
    top = torch.topk(flat, min(k, flat.numel()))
    rows = a.reshape(-1, width).amax(dim=1)
    return dict(top=[float(v) for v in top.values],
                chan=[int(i) % width for i in top.indices],
                width=width,
                median=float(flat.median()),
                frac_over=float((flat > FP16_SQUARE_SAFE).float().mean()),
                rows_over=float((rows > FP16_SQUARE_SAFE).float().mean()))


def surface(x, rows=SURF_TOKENS):
    """|x| as a [tokens, channels] grid for the 3D view, tokens strided down.

    Every channel is kept -- striding the channel axis could step straight over
    the outlier column, which is the one thing the plot exists to show. Tokens
    are sampled across the whole sequence rather than truncated to the first N,
    so the grid is representative of the run and not just its opening.
    """
    import torch
    a = x.detach().float().abs().reshape(-1, x.shape[-1])
    if a.shape[0] > rows:
        idx = torch.linspace(0, a.shape[0] - 1, rows).long().to(a.device)
        a = a.index_select(0, idx)
    return a.to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------------------
def capture(args):
    import gc

    import torch
    import arch
    import graphs
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    gold = os.path.join(args.root, "golden", "inputs.npz")
    fx = os.path.join(args.root, "fixtures")
    for p in (gold, os.path.join(fx, "embed_tokens.fp16.npy"), os.path.join(fx, "meta.json")):
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run h100/a1_golden.py and h100/a3b_fixtures.py first" % p)
    if not os.path.isdir(args.model):
        raise SystemExit("checkpoint not found: %s" % args.model)

    g = np.load(gold, allow_pickle=True)
    meta = json.load(open(os.path.join(fx, "meta.json")))
    prefill = int(meta["prefill"])
    max_seq = int(meta["max_seq"])
    rope_delta = int(meta.get("rope_deltas", 0))
    dtype = torch.bfloat16                      # see the module docstring

    print("loading %s in %s" % (args.model, dtype))
    model = AlpamayoR1.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
    lm = model.vlm.model.language_model

    rec = {k: [] for k in STACKS}
    inp, best = {}, {}
    bucket = {"name": None}

    def note(name, x, st):
        """Keep the grid of whichever layer holds the largest value so far."""
        rec[name].append(st)
        if st["top"][0] > best.get(name, (0.0,))[0]:
            best[name] = (st["top"][0], len(rec[name]) - 1, surface(x))

    # decoder_layer is a module-level function that PrefillGraph and ExpertGraph
    # look up at call time, so wrapping it measures exactly what the export
    # traces -- no duplicated forward loop to drift out of sync. Module hooks
    # would not fire here: those two graphs call the free function, not the
    # layer modules.
    real_layer = graphs.decoder_layer

    def wrapped(layer, x, *a, **kw):
        name = bucket["name"]
        if name and not rec[name]:
            inp[name] = stats(x)                      # the stream as it enters layer 0
        out = real_layer(layer, x, *a, **kw)
        if name:
            note(name, out[0], stats(out[0]))
        return out

    graphs.decoder_layer = wrapped

    # ---- 1. vision tower -------------------------------------------------
    # The ViT blocks are called as modules, so ordinary hooks do fire.
    px = torch.tensor(g["pixel_values"], device="cuda", dtype=dtype)
    grid = torch.tensor(g["image_grid_thw"], dtype=torch.long, device="cuda")
    unwrap = lambda o: o[0] if isinstance(o, (tuple, list)) else o
    blocks = model.vlm.model.visual.blocks
    hooks = [blocks[0].register_forward_pre_hook(
        lambda m, i: inp.__setitem__("vision", stats(unwrap(i))))]
    hooks += [b.register_forward_hook(
        lambda m, i, o: note("vision", unwrap(o), stats(unwrap(o)))) for b in blocks]
    with torch.no_grad():
        vout = graphs.VisionGraph(model.vlm.model.visual, grid).eval()(px)
    for h in hooks:
        h.remove()
    if not isinstance(vout, tuple):
        raise SystemExit("vision tower returned no DeepStack maps -- cannot drive prefill")
    visual, ds = vout[0], vout[1:]
    print("vision         : %2d blocks, embeds %s" % (len(rec["vision"]), tuple(visual.shape)))

    # ---- 2. language model, over the real prompt -------------------------
    embed = np.load(os.path.join(fx, "embed_tokens.fp16.npy"), mmap_mode="r")
    ids = np.load(os.path.join(fx, "input_ids.npy")).reshape(-1)[:prefill]
    vmask = torch.from_numpy(np.load(os.path.join(fx, "visual_mask.npy"))[:prefill]).cuda()
    pos = torch.from_numpy(np.load(os.path.join(fx, "position_ids.npy"))[:, :prefill])
    cos, sin = (t.cuda() for t in graphs.rope_tables(pos.reshape(3, 1, -1), dtype=dtype))

    embeds = torch.from_numpy(np.ascontiguousarray(embed[ids])).to("cuda", dtype)[None]
    n_vis = int(vmask.sum())
    embeds[0, vmask] = visual[:n_vis].to(dtype)
    ds_full = []
    for d in ds:
        z = torch.zeros_like(embeds)
        z[0, vmask] = d[:n_vis].to(dtype)
        ds_full.append(z)
    del px, vout, visual, ds
    gc.collect()
    torch.cuda.empty_cache()

    bucket["name"] = "language model"
    pf = graphs.PrefillGraph(lm, model.vlm.lm_head, prefill, dtype).cuda().eval()
    with torch.no_grad():
        out = pf(embeds, cos, sin, *ds_full)
    bucket["name"] = None
    k_cache, v_cache = out[2], out[3]
    print("language model : %2d layers, %d prompt positions"
          % (len(rec["language model"]), prefill))

    # ---- 3. action expert, over the cache prefill just wrote -------------
    # No reasoning tokens are generated here, so the expert sits at pos=prefill.
    # Its inputs are otherwise assembled exactly as xavier/run_alpamayo.py does.
    L, KV, HD = arch.EXPERT["layers"], arch.EXPERT["kv_heads"], arch.EXPERT["head_dim"]
    W, STEPS = arch.N_WAYPOINTS, arch.FLOW_STEPS
    past_k = torch.zeros(L, 1, KV, max_seq, HD, device="cuda", dtype=dtype)
    past_v = torch.zeros_like(past_k)
    past_k[:, :, :, :prefill] = k_cache.to(dtype)
    past_v[:, :, :, :prefill] = v_cache.to(dtype)
    del out, k_cache, v_cache, embeds, ds_full
    gc.collect()
    torch.cuda.empty_cache()

    wpos = np.broadcast_to(np.arange(W) + prefill + rope_delta, (3, W))
    wcos, wsin = (t.cuda() for t in graphs.rope_tables(
        torch.from_numpy(wpos.copy()).reshape(3, 1, -1), dtype=dtype))
    emask = torch.full((1, 1, W, max_seq + W), torch.finfo(dtype).min,
                       device="cuda", dtype=dtype)
    emask[..., :prefill] = 0.0
    emask[..., max_seq:] = 0.0                        # expert tokens see each other
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn(1, W, 2, device="cuda", dtype=dtype, generator=gen)
    ts = torch.linspace(0.0, 1.0, STEPS + 1)

    bucket["name"] = "expert"
    ex = graphs.ExpertGraph(model.expert, model.action_in_proj, model.action_out_proj,
                            max_seq, dtype=dtype).cuda().eval()
    with torch.no_grad():
        for i in range(STEPS):                        # the real Euler schedule
            t = torch.full((1, 1, 1), float(ts[i]), device="cuda", dtype=dtype)
            x = x + float(ts[i + 1] - ts[i]) * ex(x, t, wcos, wsin, past_k, past_v, emask)
    bucket["name"] = None
    graphs.decoder_layer = real_layer

    # Fold the 10 flow steps down to the worst step per layer, so the expert
    # subplot shows the largest value that layer ever has to represent.
    flat = rec["expert"]
    nsteps = len(flat) // L
    rec["expert"] = [max((flat[s * L + l] for s in range(nsteps)),
                         key=lambda w: w["top"][0]) for l in range(L)]
    if "expert" in best:            # re-index the kept grid onto the folded layers
        best["expert"] = (best["expert"][0], best["expert"][1] % L, best["expert"][2])
    print("expert         : %2d layers, %d waypoints, worst of %d flow steps"
          % (len(rec["expert"]), W, STEPS))

    os.makedirs(args.out, exist_ok=True)
    doc = dict(dtype=str(dtype), prefill=prefill, max_seq=max_seq, flow_steps=STEPS,
               fp16_max=FP16_MAX, fp16_square_safe=FP16_SQUARE_SAFE, topk=TOPK,
               gpu=torch.cuda.get_device_name(0), stacks=rec, inputs=inp,
               note="bf16 capture: fp16 is the precision under test, measuring in "
                    "it would hide the overflow. Expert folded to the worst of "
                    "%d flow steps. 'language model' is the weight set prefill "
                    "and decode share." % STEPS)
    doc["surfaces"] = {n: dict(layer=v[1], peak=v[0], tokens=int(v[2].shape[0]),
                               channels=int(v[2].shape[1]))
                       for n, v in best.items()}
    path = os.path.join(args.out, "activation_study.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    npz = os.path.join(args.out, "activation_study.npz")
    np.savez_compressed(npz, **{"surf_%s" % n.replace(" ", "_"): v[2]
                                for n, v in best.items()})
    print("\nwrote %s" % path)
    print("wrote %s  (%.1f MB)" % (npz, os.path.getsize(npz) / 1e6))
    return doc


# ---------------------------------------------------------------------------
def panel(ax, letter, title):
    """(a) + a neutral description, as in vision_study.py. Findings go in
    data-driven annotations, never in the title, so the figure cannot assert
    something the run did not show."""
    ax.set_title("(%s) %s" % (letter, title), loc="left", pad=3, fontsize=7.2)


def save(fig, name, outdir):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, "%s.%s" % (name, ext))
        fig.savefig(path)
        print("wrote %s" % path)
    plt.close(fig)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:                                           # reuse the project's figure style
        sys.path.insert(0, os.path.join(REPO, "analysis"))
        from alpamayo_figs import style
        style.use_style()
        MUTED = style.C["muted"]
    except Exception:
        MUTED = "#6B7780"
        matplotlib.rcParams.update({"font.size": 8, "figure.dpi": 150,
                                    "font.family": "serif", "savefig.dpi": 600,
                                    "savefig.bbox": "tight",
                                    "axes.spines.top": False, "axes.spines.right": False})
    BLUE, RUST, TEAL, PLUM, FAULT = "#1F4E79", "#7A2E23", "#0B6B63", "#473C6B", "#A32318"

    path = os.path.join(args.out, "activation_study.json")
    if not os.path.exists(path):
        raise SystemExit("no %s -- run the capture stage on the H100 first" % path)
    doc = json.load(open(path))
    lim, ceil, k = doc["fp16_square_safe"], doc["fp16_max"], doc["topk"]
    COLOUR = {"vision": TEAL, "language model": BLUE, "expert": PLUM}
    TITLE = {"vision": "vision tower, %d blocks",
             "language model": "language model, %d layers",
             "expert": "action expert, %d layers"}
    RANK = [("largest $|x|$", "-o", 2.2, 0.95, 1.00), ("2nd", "--s", 1.9, 0.85, 0.72),
            ("3rd", ":^", 1.7, 0.80, 0.50)]

    def grid(ax, axis="y"):
        ax.grid(True, axis=axis, lw=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    # ---- figure A: magnitude with depth ----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.05), constrained_layout=True,
                           sharey=True)
    summary = []
    for i, name in enumerate(STACKS):
        a = ax[i]
        rows = doc["stacks"].get(name) or []
        if not rows:
            a.text(0.5, 0.5, "not captured", ha="center", va="center", fontsize=6.4,
                   color=MUTED, transform=a.transAxes)
            a.set_yticks([])
            panel(a, "abc"[i], TITLE[name].split(",")[0])
            continue
        c = COLOUR[name]
        x = np.arange(len(rows))
        entry = doc.get("inputs", {}).get(name)
        xin = -max(2.4, 0.10 * len(rows))         # far enough left of tick 0 to read
        for r, (lab, mk, ms, lw, al) in enumerate(RANK[:k]):
            y = [w["top"][r] if len(w["top"]) > r else np.nan for w in rows]
            a.semilogy(x, y, mk, ms=ms, lw=lw, color=c, alpha=al, label=lab,
                       zorder=6 - r)
            if entry:                                 # the stream entering layer 0
                a.semilogy([xin], [entry["top"][r]], mk[-1], ms=ms, color=c,
                           alpha=al, zorder=6 - r)
        a.semilogy(x, [w["median"] for w in rows], "-", color=MUTED, lw=0.9,
                   label="median $|x|$", zorder=3)
        if entry:
            a.axvline(xin / 2.0, color=MUTED, lw=0.6, ls=":", zorder=1)

        # Both fp16 ceilings, because the gap between them IS the defect: every
        # value clears the upper one and almost none clear the lower.
        a.axhline(ceil, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2,
                  label="fp16 max, $65\\,504$ \u2014 every $|x|$ fits here")
        a.axhline(lim, color=FAULT, lw=1.0, zorder=4,
                  label="$\\sqrt{65\\,504}=256$ \u2014 above this $x^2$ overflows")

        a.set_xlabel("layer")
        a.set_xlim(xin - 1.3, len(rows) - 0.4)
        ticks = [t for t in a.get_xticks() if 0 <= t <= len(rows) - 1]
        a.set_xticks([xin] + list(ticks))
        a.set_xticklabels(["in"] + ["%d" % t for t in ticks])
        panel(a, "abc"[i], TITLE[name] % len(rows))
        grid(a)
        top1 = np.array([w["top"][0] for w in rows])
        summary.append((name, rows, entry, top1, np.nonzero(top1 > lim)[0]))

    ax[0].set_ylabel("$|x|$ in the residual stream")
    ax[0].set_ylim(top=ceil * 3.0)
    ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=15))
    # Legend and caption stack below the panels. Both are anchored explicitly
    # against measured positions: an "outside" legend is owned by the layout
    # engine, which re-runs inside savefig and slides it back over the caption.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    ybot = min(a.get_tightbbox(rend).transformed(inv).y0 for a in ax)
    fig.set_layout_engine("none")                 # freeze the panels where they are
    h, l = ax[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, ybot - 0.05),
                     ncol=3, fontsize=5.8, handlelength=1.6, handletextpad=0.4,
                     columnspacing=1.6, labelspacing=0.32, borderaxespad=0.0,
                     frameon=False)
    fig.canvas.draw()
    y0 = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.text(0.5, y0 - 0.035, caption(summary, lim, ceil), ha="center", va="top",
             fontsize=5.9, color=MUTED, linespacing=1.55)
    save(fig, "fig_activations", args.out)

    # ---- figure B: the token x channel landscape -------------------------
    surfaces(doc, args.out, COLOUR, MUTED, TITLE)
    report(summary, lim, ceil, doc.get("flow_steps", 10))


def caption(summary, lim, ceil):
    """One sentence under the figure, built from the run so it cannot go stale."""
    if not summary:
        return ""
    name, rows, _, top1, _ = max(summary, key=lambda t: t[3].max())
    peak = float(top1.max())
    return ("Every value fits in fp16: the largest, %s in the %s, is %.0f%% of the "
            "%s ceiling.\nRMSNorm squares its input before it reduces, so the limit "
            "that binds is $\\sqrt{65\\,504}=256$ \u2014 and %s$^2$ = %s does not fit."
            % (_fmt(peak), name, 100 * peak / ceil, _fmt(ceil), _fmt(peak),
               _sci(peak ** 2)))


def surfaces(doc, outdir, COLOUR, MUTED, TITLE):
    """|x| over tokens x channels at each stack's peak layer.

    The 3D view is PrefixQuant's plot_3D_tensor (Chen et al.,
    github.com/ChenMnZ/PrefixQuant, utils/plot_utils.py): plot_surface with the
    coolwarm map, Channel on x, Token on y, viewed from elev=20, azim=-45.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d)

    npz = os.path.join(outdir, "activation_study.npz")
    if not os.path.exists(npz):
        print("\nno %s -- re-run the capture stage for the 3D view" % npz)
        return
    z = np.load(npz)
    got = [n for n in STACKS if "surf_%s" % n.replace(" ", "_") in z.files]
    if not got:
        return

    fig = plt.figure(figsize=(7.1, 2.35), constrained_layout=True)
    for i, name in enumerate(STACKS):
        a = fig.add_subplot(1, 3, i + 1, projection="3d")
        key = "surf_%s" % name.replace(" ", "_")
        if key not in z.files:
            a.set_axis_off()
            continue
        g = z[key].astype(np.float32)
        if g.shape[0] > SURF_PLOT_ROWS:               # keep every channel; thin tokens
            g = g[np.linspace(0, g.shape[0] - 1, SURF_PLOT_ROWS).astype(int)]
        X, Y = np.meshgrid(np.arange(g.shape[1]), np.arange(g.shape[0]))
        a.plot_surface(X, Y, g, cmap="coolwarm", antialiased=False, shade=True,
                       linewidth=0, rstride=1, cstride=1, rasterized=True)
        a.view_init(elev=20.0, azim=-45)
        try:                      # fill the panel; 3D axes default to tiny
            a.set_box_aspect((4, 4, 2.4), zoom=1.22)
        except TypeError:         # matplotlib < 3.6 has no zoom
            a.set_box_aspect((4, 4, 2.4))
        meta = doc.get("surfaces", {}).get(name, {})
        a.set_title("(%s) %s, layer %s"
                    % ("abc"[i], name, meta.get("layer", "?")), loc="left",
                    pad=-8, fontsize=7.2)
        a.set_xlabel("Channel", fontsize=6.2, labelpad=-5)
        a.set_ylabel("Token", fontsize=6.2, labelpad=-5)
        a.tick_params(labelsize=5.2, pad=-2.5)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        a.set_zlim(0, float(g.max()) * 1.05)
        for pane in (a.xaxis, a.yaxis, a.zaxis):
            pane.pane.set_alpha(0.0)
            pane._axinfo["grid"]["linewidth"] = 0.25
    fig.text(0.5, -0.02,
             "The large values stand in a handful of fixed columns, the same ones at "
             "every layer. A per-channel scale keeps them;\na per-tensor scale spends "
             "its whole range on them \u2014 which is also why rescaling each row by its "
             "own max makes RMSNorm fp16-safe.",
             ha="center", va="top", fontsize=5.9, color=MUTED, linespacing=1.5)
    save(fig, "fig_activation_surface", outdir)


def _sci(v):
    """1.7e8 -> $1.7\\times10^{8}$, so the caption reads like the paper it sits in."""
    e = int(np.floor(np.log10(abs(v)))) if v else 0
    return "$%.1f\\times10^{%d}$" % (v / 10.0 ** e, e)


def _fmt(v):
    return format(int(round(v)), ",d").replace(",", "\u2009") if v >= 100 else "%.1f" % v


def report(summary, lim, ceil, steps):
    """The audit trail, in the same shape vision_study.py prints."""
    print("\nwhat the capture recorded")
    print("  %-15s %10s %6s %10s   %s"
          % ("stack", "peak |x|", "layer", "median", "verdict"))
    for name, rows, entry, top1, over in summary:
        i = int(np.argmax(top1))
        print("  %-15s %10.1f %6d %10.3f   %s"
              % (name, top1[i], i, rows[i]["median"],
                 "%d of %d layers over %.0f" % (len(over), len(rows), lim)
                 if over.size else "stays under %.0f" % lim))
    for name, rows, entry, top1, over in summary:
        print("\n%s" % name)
        if entry:
            print("  entering layer 0   max %.1f, median %.3f"
                  % (entry["top"][0], entry["median"]))
        if over.size:
            f = rows[over[0]]
            print("  first crossing     layer %d, |x| = %.1f in channel %d of %d"
                  % (over[0], f["top"][0], f["chan"][0], f["width"]))
            print("  at that layer      %.4f%% of values and %.2f%% of tokens are over"
                  % (100 * f["frac_over"], 100 * f["rows_over"]))
        chans = [w["chan"][0] for w in rows]
        common = max(set(chans), key=chans.count)
        print("  top channel        %d holds the largest value in %d of %d layers"
              % (common, chans.count(common), len(chans)))
        seen = sorted({c for w in rows for c in w["chan"]})
        print("  top-3 ever land in %d distinct channels of %d: %s"
              % (len(seen), rows[0]["width"],
                 ", ".join(str(c) for c in seen[:8]) + (" ..." if len(seen) > 8 else "")))
    print("\n  expert folded to the worst of %d flow steps; 'language model' is the"
          % steps)
    print("  weight set prefill and decode share, so it covers both.")
    print("\nwhy the line is at %.0f and not at %.0f" % (lim, ceil))
    for name, rows, entry, top1, over in summary:
        pk = float(top1.max())
        print("  %-15s peak %9s = %4.1f%% of fp16 max, but x^2 = %8.2e, %6.0fx over"
              % (name, _fmt(pk), 100 * pk / ceil, pk ** 2, pk ** 2 / ceil))
    print("  RMSNorm computes mean(x^2) before it reduces, so Pow(2) is the node")
    print("  that overflows -- the values themselves were never the problem.")
    print("\n  A handful of channels carrying every large value is the per-channel")
    print("  case: one scale per column keeps them, one scale per tensor spends its")
    print("  whole range on them. It is also why the fp16 RMSNorm fix works --")
    print("  rescaling by the row max moves those columns back under %.0f without" % lim)
    print("  touching the ratio the norm actually depends on.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", choices=["all", "capture", "plot"])
    ap.add_argument("--model", default=os.environ.get("ALPAMAYO_MODEL", ""))
    ap.add_argument("--root", default=os.environ.get("ALPAMAYO_ROOT", ""))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage in ("all", "capture"):
        if not args.model or not args.root:
            raise SystemExit("set ALPAMAYO_MODEL and ALPAMAYO_ROOT (source env.sh) "
                             "or pass --model/--root")
        capture(args)
    if args.stage in ("all", "plot"):
        plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
