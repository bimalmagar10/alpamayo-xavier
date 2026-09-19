#!/usr/bin/env python3
"""The largest activations in every layer of all three stacks, against the fp16 wall.

This is the picture behind the RMSNorm defect. RMSNorm squares its input before
it reduces, so fp16 needs |x| <= sqrt(65504) = 255.94 for x**2 to exist at all.
The question this answers is: which layers of which stack cross that line, by how
much, and -- because the fix depends on it -- whether the values that cross sit
in the same few channels every time.

Three subplots, one per stack: vision tower (27 blocks), language model (36
layers, the weights prefill and decode share) and action expert (36 layers).
Each plots the three largest |activations| of that layer's residual stream.

    bash study/run_study.sh act              # on the H100
    python study/activation_study.py --stage plot     # redraw, no GPU

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
    inp = {}
    bucket = {"name": None}

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
            rec[name].append(stats(out[0]))
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
        lambda m, i, o: rec["vision"].append(stats(unwrap(o)))) for b in blocks]
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
    rec["expert"] = [max((flat[s * L + l] for s in range(len(flat) // L)),
                         key=lambda w: w["top"][0]) for l in range(L)]
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
    path = os.path.join(args.out, "activation_study.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print("\nwrote %s" % path)
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
    lim, k = doc["fp16_square_safe"], doc["topk"]
    steps = doc.get("flow_steps", 10)
    COLOUR = {"vision": TEAL, "language model": BLUE, "expert": PLUM}
    TITLE = {"vision": "vision tower, %d blocks",
             "language model": "language model, %d layers",
             "expert": "action expert, %d layers"}
    RANK = [("largest", "-o", 2.2, 0.95, 1.00), ("2nd", "--s", 1.9, 0.85, 0.72),
            ("3rd", ":^", 1.7, 0.80, 0.50)]

    def grid(ax, axis="y"):
        ax.grid(True, axis=axis, lw=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.05), constrained_layout=True,
                           sharey=True)
    summary, peaks = [], {}
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

        a.axhline(lim, color=FAULT, lw=1.0, zorder=4)
        a.annotate("$\\sqrt{65\\,504}$", xy=(xin - 0.4, lim), xytext=(2, 3),
                   textcoords="offset points", fontsize=5.7, color=FAULT,
                   ha="left", va="bottom")

        # One annotation per panel, at the peak: the magnitude, the channel it
        # sits in -- the reason per-channel scaling survives where per-tensor
        # does not -- and whether the stack ever crosses the line.
        top1 = np.array([w["top"][0] for w in rows])
        over = np.nonzero(top1 > lim)[0]
        j = int(np.argmax(top1))
        a.annotate("peak %s at layer %d,\nin channel %d of %d\n%s"
                   % (_fmt(top1[j]), j, rows[j]["chan"][0], rows[j]["width"],
                      "crossed at layer %d, %d of %d over"
                      % (over[0], len(over), len(rows)) if over.size
                      else "never crossed"),
                   xy=(0.03, 0.97), xycoords="axes fraction", fontsize=5.6,
                   color=c, ha="left", va="top")

        a.set_xlabel("layer")
        a.set_xlim(xin - 1.3, len(rows) - 0.4)
        ticks = [t for t in a.get_xticks() if 0 <= t <= len(rows) - 1]
        a.set_xticks([xin] + list(ticks))
        a.set_xticklabels(["in"] + ["%d" % t for t in ticks])
        panel(a, "abc"[i], TITLE[name] % len(rows))
        grid(a)
        peaks[i] = float(top1.max())
        summary.append((name, rows, entry, top1, over))

    ax[0].set_ylabel("$|x|$ in the residual stream")
    if peaks:
        ax[0].set_ylim(top=max(peaks.values()) * 3.5)      # room for the corner notes
    if peaks:                       # legend goes wherever there is most headroom
        ax[min(peaks, key=peaks.get)].legend(
            fontsize=5.5, handlelength=0.9, handletextpad=0.35, loc="upper right",
            labelspacing=0.28, borderaxespad=0.2)
    save(fig, "fig_activations", args.out)
    report(summary, lim, steps)


def _fmt(v):
    return format(int(round(v)), ",d").replace(",", "\u2009") if v >= 100 else "%.1f" % v


def report(summary, lim, steps):
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
