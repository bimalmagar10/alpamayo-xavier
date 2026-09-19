#!/usr/bin/env python3
"""The largest activations in every layer of all three stacks, against the fp16 wall.

This is the picture behind the RMSNorm defect. RMSNorm squares its input before
it reduces, so fp16 needs |x| <= sqrt(65504) = 255.94 for x**2 to exist at all.
The question this answers is: which layers of which stack cross that line, by how
much, and -- because the fix depends on it -- whether the values that cross sit
in the same few channels every time.

Columns are the three stacks -- vision tower (27 blocks), language model (36
layers, the weights prefill and decode share) and action expert (36 layers):

  fig_activations          the three largest |activations| per layer against
                           both fp16 ceilings, one row per pass
  fig_activation_surface   |x| over tokens x channels at each stack's peak
                           layer, which is where the outlier columns show

Two passes, so the rows can be compared directly:

  bf16   the reference, where every value is representable
  fp16   every weight and activation cast down, and RMSNorm's square kept in
         fp16 the way the exported graph computes it. The reference
         implementation (h100/graphs.py:109) upcasts to fp32 first and so never
         shows the defect; --upcast-norm reproduces that if you want to see the
         difference the upcast alone makes. --no-fp16 captures bf16 only.

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
    # An fp16 pass can produce inf or nan. Rank the finite values and report the
    # rest as a fraction, rather than letting one nan swallow the whole topk.
    ok = torch.isfinite(flat)
    bad = float((~ok).float().mean())
    flat = torch.where(ok, flat, torch.zeros_like(flat))
    top = torch.topk(flat, min(k, flat.numel()))
    rows = flat.reshape(-1, width).amax(dim=1)
    return dict(top=[float(v) for v in top.values],
                chan=[int(i) % width for i in top.indices],
                width=width, bad=bad,
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
def one_pass(torch, arch, graphs, model, dtype, fx, args, fp16_norm):
    """Run all three stacks once at `dtype` and return (rec, inputs, grids).

    fp16_norm keeps RMSNorm's square in `dtype` instead of upcasting to fp32.
    The reference implementation (h100/graphs.py:109) upcasts, and so hides the
    defect; the exported graph decomposes the norm into Pow/ReduceMean/Sqrt at
    the engine's precision and does not. Matching the export is the point of
    the fp16 pass.
    """
    import gc

    rec = {k: [] for k in STACKS}
    inp, best = {}, {}
    bucket = {"name": None}

    def note(name, x, st):
        """Keep the grid of whichever layer holds the largest value so far."""
        rec[name].append(st)
        if st["top"][0] > best.get(name, (0.0,))[0]:
            best[name] = (st["top"][0], len(rec[name]) - 1, surface(x))

    # Probe the norm itself, not just the stream it reads. The residual stream
    # is representable in both formats -- that is why the two passes agree on
    # it. What differs is one operation applied to it: RMSNorm squares before
    # it reduces, and in fp16 that square is what overflows.
    real_norm = graphs.rms_norm
    norms = {k: [] for k in STACKS}

    def probe(x, weight, eps):
        ref = real_norm(x, weight, eps)               # fp32 upcast: the reference
        if fp16_norm:
            v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            out = (v * weight).to(x.dtype)
        else:
            out = ref
        name = bucket["name"]
        if name and x.shape[-1] > 256:                # skip the per-head QK norms
            width = x.shape[-1]
            rows = x.detach().float().abs().reshape(-1, width).amax(dim=1)
            o = out.detach().float()
            r = ref.detach().float()
            den = float(r.norm())
            norms[name].append(dict(
                max_in=float(rows.max()),
                rows_over=float((rows > FP16_SQUARE_SAFE).float().mean()),
                rel=float((o - r).norm()) / (den if den else 1.0),
                zeroed=float((o.reshape(-1, width).amax(dim=1) == 0).float().mean()),
                nonfinite=float((~torch.isfinite(o)).float().mean())))
        return out

    graphs.rms_norm = probe

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
    try:
        _stacks(torch, arch, graphs, model, dtype, fx, args, rec, inp, note, bucket)
    finally:
        graphs.decoder_layer = real_layer
        graphs.rms_norm = real_norm

    # Fold the flow steps down to the worst step per layer, so the expert
    # subplot shows the largest value that layer ever has to represent.
    L = arch.EXPERT["layers"]
    flat = rec["expert"]
    if flat:
        nsteps = max(1, len(flat) // L)
        rec["expert"] = [max((flat[s * L + l] for s in range(nsteps)),
                             key=lambda w: w["top"][0]) for l in range(L)]
        if "expert" in best:        # re-index the kept grid onto the folded layers
            best["expert"] = (best["expert"][0], best["expert"][1] % L, best["expert"][2])
    # Two residual-stream norms per decoder layer; keep the worse of each pair.
    for name in STACKS:
        v = norms[name]
        n = len(rec[name])
        if v and n and len(v) % n == 0:
            per = len(v) // n
            norms[name] = [max(v[i * per:(i + 1) * per], key=lambda w: w["rel"])
                           for i in range(n)]
    gc.collect()
    torch.cuda.empty_cache()
    return rec, inp, best, norms


def _stacks(torch, arch, graphs, model, dtype, fx, args, rec, inp, note, bucket):
    """Drive vision, then prefill, then the expert, on the inputs in `fx`."""
    import gc

    prefill, max_seq = fx["prefill"], fx["max_seq"]

    # ---- 1. vision tower -------------------------------------------------
    # The ViT blocks are called as modules, so ordinary hooks do fire.
    px = fx["pixel_values"].to(dtype)
    unwrap = lambda o: o[0] if isinstance(o, (tuple, list)) else o
    blocks = model.vlm.model.visual.blocks
    hooks = [blocks[0].register_forward_pre_hook(
        lambda m, i: inp.__setitem__("vision", stats(unwrap(i))))]
    hooks += [b.register_forward_hook(
        lambda m, i, o: note("vision", unwrap(o), stats(unwrap(o)))) for b in blocks]
    with torch.no_grad():
        vout = graphs.VisionGraph(model.vlm.model.visual, fx["grid"]).eval()(px)
    for h in hooks:
        h.remove()
    if not isinstance(vout, tuple):
        raise SystemExit("vision tower returned no DeepStack maps -- cannot drive prefill")
    visual, ds = vout[0], vout[1:]
    print("  vision         : %2d blocks, embeds %s"
          % (len(rec["vision"]), tuple(visual.shape)))

    # ---- 2. language model, over the real prompt -------------------------
    cos, sin = (t.cuda() for t in graphs.rope_tables(fx["pos"], dtype=dtype))
    vmask = fx["vmask"]
    embeds = fx["embeds"].to(dtype).clone()
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
    pf = graphs.PrefillGraph(model.vlm.model.language_model, model.vlm.lm_head,
                             prefill, dtype).cuda().eval()
    with torch.no_grad():
        out = pf(embeds, cos, sin, *ds_full)
    bucket["name"] = None
    k_cache, v_cache = out[2], out[3]
    print("  language model : %2d layers, %d prompt positions"
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
    del out, k_cache, v_cache, embeds, ds_full, pf
    gc.collect()
    torch.cuda.empty_cache()

    wcos, wsin = (t.cuda() for t in graphs.rope_tables(fx["wpos"], dtype=dtype))
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
    print("  expert         : %2d layers, %d waypoints, worst of %d flow steps"
          % (len(rec["expert"]) // max(1, len(rec["expert"]) // L), W, STEPS))


def capture(args):
    import torch
    import arch
    import graphs
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    gold = os.path.join(args.root, "golden", "inputs.npz")
    fxdir = os.path.join(args.root, "fixtures")
    for p in (gold, os.path.join(fxdir, "embed_tokens.fp16.npy"),
              os.path.join(fxdir, "meta.json")):
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run h100/a1_golden.py and "
                             "h100/a3b_fixtures.py first" % p)
    if not os.path.isdir(args.model):
        raise SystemExit("checkpoint not found: %s" % args.model)

    g = np.load(gold, allow_pickle=True)
    meta = json.load(open(os.path.join(fxdir, "meta.json")))
    prefill, max_seq = int(meta["prefill"]), int(meta["max_seq"])
    rope_delta = int(meta.get("rope_deltas", 0))
    W = arch.N_WAYPOINTS

    load = lambda n: np.load(os.path.join(fxdir, n))
    embed = np.load(os.path.join(fxdir, "embed_tokens.fp16.npy"), mmap_mode="r")
    ids = load("input_ids.npy").reshape(-1)[:prefill]
    wpos = np.broadcast_to(np.arange(W) + prefill + rope_delta, (3, W)).copy()
    fx = dict(prefill=prefill, max_seq=max_seq,
              pixel_values=torch.tensor(g["pixel_values"], device="cuda"),
              grid=torch.tensor(g["image_grid_thw"], dtype=torch.long, device="cuda"),
              vmask=torch.from_numpy(load("visual_mask.npy")[:prefill]).cuda(),
              pos=torch.from_numpy(load("position_ids.npy")[:, :prefill]).reshape(3, 1, -1),
              wpos=torch.from_numpy(wpos).reshape(3, 1, -1),
              embeds=torch.from_numpy(np.ascontiguousarray(embed[ids])).cuda()[None])

    print("loading %s" % args.model)
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    passes, grids = {}, {}
    plan = [("bf16", torch.bfloat16, False)]
    if not args.no_fp16:
        plan.append(("fp16", torch.float16, not args.upcast_norm))
    for tag, dtype, fp16_norm in plan:
        print("\n%s pass%s" % (tag, "  (RMSNorm squares in %s, as the export does)" % tag
                                if fp16_norm else ""))
        if dtype != torch.bfloat16:
            model = model.to(dtype)
        rec, inp, best, norms = one_pass(torch, arch, graphs, model, dtype,
                                         fx, args, fp16_norm)
        passes[tag] = dict(dtype=str(dtype), narrow_norm=fp16_norm, stacks=rec,
                           inputs=inp, norms=norms,
                           surfaces={n: dict(layer=v[1], peak=v[0],
                                             tokens=int(v[2].shape[0]),
                                             channels=int(v[2].shape[1]))
                                     for n, v in best.items()})
        grids.update({"surf_%s_%s" % (tag, n.replace(" ", "_")): v[2]
                      for n, v in best.items()})

    os.makedirs(args.out, exist_ok=True)
    doc = dict(prefill=prefill, max_seq=max_seq, flow_steps=arch.FLOW_STEPS,
               fp16_max=FP16_MAX, fp16_square_safe=FP16_SQUARE_SAFE, topk=TOPK,
               gpu=torch.cuda.get_device_name(0), passes=passes,
               note="bf16 is the reference. The fp16 pass casts every weight and "
                    "activation and, unless --upcast-norm, keeps RMSNorm's square "
                    "in fp16 the way the exported graph does -- graphs.rms_norm "
                    "upcasts to fp32 and would hide the defect. Expert folded to "
                    "the worst of %d flow steps; 'language model' is the weight "
                    "set prefill and decode share." % arch.FLOW_STEPS)
    path = os.path.join(args.out, "activation_study.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    npz = os.path.join(args.out, "activation_study.npz")
    np.savez_compressed(npz, **grids)
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
    except Exception:
        matplotlib.rcParams.update({"font.size": 8, "figure.dpi": 150,
                                    "font.family": "serif", "savefig.dpi": 600,
                                    "savefig.bbox": "tight",
                                    "axes.spines.top": False, "axes.spines.right": False})
    # Okabe-Ito for the data, neutral ink for the two fp16 rules: distinguishable
    # in greyscale and under every common form of colour blindness.
    SERIES = ("#0072B2", "#D55E00", "#009E73")
    INK, GREY, BAD = "#1A1A1A", "#9099A1", "#CC3311"

    path = os.path.join(args.out, "activation_study.json")
    if not os.path.exists(path):
        raise SystemExit("no %s -- run the capture stage on the H100 first" % path)
    doc = json.load(open(path))
    lim, ceil, k = doc["fp16_square_safe"], doc["fp16_max"], doc["topk"]
    passes = [t for t in ("bf16", "fp16") if t in doc.get("passes", {})]
    if not passes:
        raise SystemExit("%s holds no passes -- re-run the capture stage" % path)
    COLOUR = dict(zip(STACKS, SERIES))
    TITLE = {"vision": "vision tower, %d blocks",
             "language model": "language model, %d layers",
             "expert": "action expert, %d layers"}
    RANK = [("largest", "-o", 1.9, 0.85, 1.00), ("2nd", "--s", 1.7, 0.75, 0.70),
            ("3rd", ":^", 1.5, 0.70, 0.48)]

    nrow = len(passes)
    fig, ax = plt.subplots(nrow, 3, figsize=(6.9, 1.55 * nrow + 0.30), sharey=True,
                           sharex="col", constrained_layout=True, squeeze=False)
    summary, floor, any_bad = [], [], False
    for r, tag in enumerate(passes):
        pas = doc["passes"][tag]
        for i, name in enumerate(STACKS):
            a_ = ax[r][i]
            rows = pas["stacks"].get(name) or []
            if not rows:
                a_.text(0.5, 0.5, "not captured", ha="center", va="center",
                        fontsize=6.4, color=GREY, transform=a_.transAxes)
                continue
            c = COLOUR[name]
            x = np.arange(len(rows))
            entry = pas.get("inputs", {}).get(name)
            xin = -max(2.4, 0.10 * len(rows))     # far enough left of tick 0 to read
            for j, (lab, mk, ms, lw, al) in enumerate(RANK[:k]):
                y = [w["top"][j] if len(w["top"]) > j else np.nan for w in rows]
                a_.semilogy(x, y, mk, ms=ms, lw=lw, color=c, alpha=al, label=lab,
                            zorder=6 - j)
                if entry:                          # the stream entering layer 0
                    a_.semilogy([xin], [entry["top"][j]], mk[-1], ms=ms, color=c,
                                alpha=al, zorder=6 - j)
            a_.semilogy(x, [w["median"] for w in rows], "-", color=GREY, lw=0.8,
                        label="median", zorder=3)
            if entry:
                a_.axvline(xin / 2.0, color=GREY, lw=0.5, ls=":", zorder=1)

            # Layers the pass could not represent at all.
            nf = [j for j, w in enumerate(rows) if w.get("bad", 0.0) > 0]
            if nf:
                any_bad = True
                a_.semilogy(nf, [rows[j]["top"][0] or lim for j in nf], "x",
                            ms=3.0, mew=0.9, color=BAD, label="inf or nan", zorder=8)

            # Both fp16 ceilings: the gap between them is the defect.
            a_.axhline(ceil, color=GREY, lw=0.7, ls=(0, (4, 2)), zorder=2,
                       label="$65\\,504$")
            a_.axhline(lim, color=INK, lw=0.9, zorder=4, label="$\\sqrt{65\\,504}$")

            a_.set_xlim(xin - 1.3, len(rows) - 0.4)
            if r == nrow - 1:
                a_.set_xlabel("layer", labelpad=1)
                ticks = [t for t in a_.get_xticks() if 0 <= t <= len(rows) - 1]
                a_.set_xticks([xin] + list(ticks))
                a_.set_xticklabels(["in"] + ["%d" % t for t in ticks])
            if r == 0:
                panel(a_, "abc"[i], TITLE[name] % len(rows))
            a_.grid(True, axis="y", lw=0.4, alpha=0.55)
            a_.set_axisbelow(True)
            floor += [w["median"] for w in rows if w["median"] > 0]
            summary.append((tag, name, rows, entry,
                            np.array([w["top"][0] for w in rows])))
        ax[r][0].set_ylabel("%s      $|x|$" % tag, labelpad=2)

    ax[0][0].set_ylim(top=ceil * 3.0, bottom=min(floor) * 0.25 if floor else None)
    ax[0][0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=15))

    # One row of bare labels under the panels, anchored against a measured
    # position: an "outside" legend is owned by the layout engine, which re-runs
    # inside savefig and moves it afterwards.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ybot = min(a_.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0
               for a_ in ax[-1])
    fig.set_layout_engine("none")                 # freeze the panels where they are
    h, l = ax[-1][0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for handle, label in zip(h, l):               # one entry each, order preserved
        if label not in seen:
            seen.add(label)
            hh.append(handle)
            ll.append(label)
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, ybot - 0.02),
               ncol=len(hh), fontsize=5.6, handlelength=1.4, handletextpad=0.35,
               columnspacing=1.1, borderaxespad=0.0, frameon=False)
    save(fig, "fig_activations", args.out)

    # ---- figure B: the token x channel landscape -------------------------
    rmsnorm_figure(doc, args.out, COLOUR, INK, GREY)
    surfaces(doc, args.out, passes[0])
    report(doc, summary, lim, ceil)


def rmsnorm_figure(doc, outdir, COLOUR, INK, GREY):
    """What the fp16 square actually costs, layer by layer.

    Two fractions on one 0-100% axis:
      * how many tokens carry a value whose square fp16 cannot hold, measured on
        the bf16 activations -- the reach of the defect;
      * how far the fp16 norm's output then lands from the fp32 reference on
        exactly the same input -- the damage.

    The second is only measurable where the stack goes through graphs.rms_norm,
    which is the language model and the expert: the code the ONNX export traces.
    The vision tower runs the HF module's own norms, so only the first curve
    appears there -- but a3c_decompose_layernorm.py rewrites both LayerNorm and
    RMSNorm in the exported graph, and both square their input, so the same
    criterion applies to it.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    base = doc["passes"].get("bf16")
    fp = doc["passes"].get("fp16")
    if not base:
        return
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 1.72), sharey=True,
                           constrained_layout=True)
    for i, name in enumerate(STACKS):
        a = ax[i]
        rows = base["stacks"].get(name) or []
        if not rows:
            a.text(0.5, 0.5, "not captured", ha="center", va="center", fontsize=6.4,
                   color=GREY, transform=a.transAxes)
            continue
        c = COLOUR[name]
        x = np.arange(len(rows))
        a.plot(x, [100 * w["rows_over"] for w in rows], "-o", ms=1.9, lw=0.85,
               color=c, label="tokens fp16 cannot square", zorder=5)
        nm = (fp or {}).get("norms", {}).get(name) or []
        if len(nm) == len(rows):
            a.plot(x, [100 * min(w["rel"], 1.0) for w in nm], "--s", ms=1.7, lw=0.75,
                   color=INK, alpha=0.85, label="fp16 norm output error", zorder=6)
        a.set_xlabel("layer", labelpad=1)
        a.set_xlim(-0.6, len(rows) - 0.4)
        a.set_ylim(-4, 104)
        panel(a, "abc"[i], name)
        a.grid(True, axis="y", lw=0.4, alpha=0.55)
        a.set_axisbelow(True)
    ax[0].set_ylabel("% of tokens", labelpad=2)

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ybot = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0
               for a in ax)
    fig.set_layout_engine("none")
    # Proxy handles: the first series is drawn in each panel's own colour, so a
    # handle lifted from one panel would claim that colour for all three.
    from matplotlib.lines import Line2D
    proxies = [Line2D([], [], color=GREY, marker="o", ms=1.9, lw=0.85),
               Line2D([], [], color=INK, marker="s", ms=1.7, lw=0.75, ls="--")]
    labels = ["tokens fp16 cannot square", "fp16 norm output error"]
    fig.legend(proxies, labels, loc="upper center", bbox_to_anchor=(0.5, ybot - 0.02),
               ncol=2, fontsize=5.6, handlelength=1.4, handletextpad=0.35,
               columnspacing=1.2, borderaxespad=0.0, frameon=False)
    save(fig, "fig_rmsnorm", outdir)


def surfaces(doc, outdir, tag):
    """|x| over tokens x channels at each stack's peak layer.

    The 3D view is PrefixQuant's plot_3D_tensor (Chen et al.,
    github.com/ChenMnZ/PrefixQuant, utils/plot_utils.py): plot_surface, Channel
    on x, Token on y, viewed from elev=20, azim=-45. Their coolwarm is swapped
    for viridis, which keeps its ordering in greyscale.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d)

    npz = os.path.join(outdir, "activation_study.npz")
    if not os.path.exists(npz):
        print("\nno %s -- re-run the capture stage for the 3D view" % npz)
        return
    z = np.load(npz)
    key = lambda n: "surf_%s_%s" % (tag, n.replace(" ", "_"))
    if not any(key(n) in z.files for n in STACKS):
        print("\nno %s grids in %s -- re-run the capture stage" % (tag, npz))
        return

    fig = plt.figure(figsize=(6.9, 1.95), constrained_layout=True)
    for i, name in enumerate(STACKS):
        a = fig.add_subplot(1, 3, i + 1, projection="3d")
        if key(name) not in z.files:
            a.set_axis_off()
            continue
        g = np.nan_to_num(z[key(name)].astype(np.float32), posinf=0.0, neginf=0.0)
        if g.shape[0] > SURF_PLOT_ROWS:               # keep every channel; thin tokens
            g = g[np.linspace(0, g.shape[0] - 1, SURF_PLOT_ROWS).astype(int)]
        X, Y = np.meshgrid(np.arange(g.shape[1]), np.arange(g.shape[0]))
        a.plot_surface(X, Y, g, cmap="viridis", antialiased=False, shade=True,
                       linewidth=0, rstride=1, cstride=1, rasterized=True)
        a.view_init(elev=20.0, azim=-45)
        try:                      # fill the panel; 3D axes default to tiny
            a.set_box_aspect((4, 4, 2.6), zoom=1.38)
        except TypeError:         # matplotlib < 3.6 has no zoom
            a.set_box_aspect((4, 4, 2.6))
        meta = doc["passes"][tag].get("surfaces", {}).get(name, {})
        a.set_title("(%s) %s, layer %s"
                    % ("abc"[i], name, meta.get("layer", "?")), loc="left",
                    pad=-16, fontsize=7.2)
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
    save(fig, "fig_activation_surface", outdir)


def _fmt(v):
    return format(int(round(v)), ",d").replace(",", "\u2009") if v >= 100 else "%.1f" % v


def report(doc, summary, lim, ceil):
    """The audit trail, in the same shape vision_study.py prints."""
    steps = doc.get("flow_steps", 10)
    print("\nwhat the capture recorded")
    print("  %-6s %-15s %10s %6s %10s %9s   %s"
          % ("pass", "stack", "peak |x|", "layer", "median", "non-finite", "verdict"))
    for tag, name, rows, entry, top1 in summary:
        i = int(np.argmax(top1))
        over = np.nonzero(top1 > lim)[0]
        bad = max(w.get("bad", 0.0) for w in rows)
        print("  %-6s %-15s %10.1f %6d %10.3f %8.2f%%   %s"
              % (tag, name, top1[i], i, rows[i]["median"], 100 * bad,
                 "%d of %d layers over %.0f" % (len(over), len(rows), lim)
                 if over.size else "stays under %.0f" % lim))

    base = doc["passes"].get("bf16")
    if base:
        print("\nwhere the large values sit  (bf16)")
        for name in STACKS:
            rows = base["stacks"].get(name) or []
            if not rows:
                continue
            chans = [w["chan"][0] for w in rows]
            common = max(set(chans), key=chans.count)
            seen = sorted({c for w in rows for c in w["chan"]})
            print("  %-15s channel %-5d holds the largest value in %2d of %2d layers; "
                  "top-3 ever land in %d of %d channels"
                  % (name, common, chans.count(common), len(chans), len(seen),
                     rows[0]["width"]))

    print("\nwhy the line is at %.0f and not at %.0f" % (lim, ceil))
    for tag, name, rows, entry, top1 in summary:
        if tag != "bf16":
            continue
        pk = float(top1.max())
        print("  %-15s peak %9s = %4.1f%% of fp16 max, but x^2 = %8.2e, %7.0fx over"
              % (name, _fmt(pk), 100 * pk / ceil, pk ** 2, pk ** 2 / ceil))
    print("  RMSNorm computes mean(x^2) before it reduces, so Pow(2) is the node")
    print("  that overflows -- the values themselves were never the problem.")

    fp = doc["passes"].get("fp16")
    if fp:
        print("\nthe residual stream itself")
        for name in STACKS:
            r16 = fp["stacks"].get(name) or []
            r32 = (base or {}).get("stacks", {}).get(name) or []
            if not r16 or len(r16) != len(r32):
                continue
            rel = [abs(b["top"][0] - a["top"][0]) / max(a["top"][0], 1e-9)
                   for a, b in zip(r32, r16)]
            print("  %-15s fp16 peak differs from bf16 by at most %.2f%% -- both "
                  "formats hold these values" % (name, 100 * max(rel)))

        print("\nthe RMSNorm applied to it  (squares in %s)"
              % ("fp16, as the exported graph does" if fp.get("narrow_norm")
                 else "fp32, as graphs.rms_norm does -- the defect stays hidden"))
        for name in STACKS:
            nm = fp.get("norms", {}).get(name) or []
            if not nm:
                print("  %-15s not routed through graphs.rms_norm (HF module's own "
                      "norms); see the %% column above" % name)
                continue
            hit = [j for j, w in enumerate(nm) if w["rows_over"] > 0]
            worst = max(nm, key=lambda w: w["rel"])
            print("  %-15s first layer whose input cannot be squared: %s"
                  % (name, hit[0] if hit else "none"))
            print("  %-15s worst output error %.1f%% of the reference norm; "
                  "%.1f%% of tokens came out all-zero"
                  % ("", 100 * worst["rel"], 100 * worst["zeroed"]))
    print("\n  expert folded to the worst of %d flow steps; 'language model' is the"
          % steps)
    print("  weight set prefill and decode share, so it covers both.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", choices=["all", "capture", "plot"])
    ap.add_argument("--model", default=os.environ.get("ALPAMAYO_MODEL", ""))
    ap.add_argument("--root", default=os.environ.get("ALPAMAYO_ROOT", ""))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", action="store_true",
                    help="capture the bf16 reference only")
    ap.add_argument("--upcast-norm", action="store_true",
                    help="let the fp16 pass keep graphs.rms_norm's fp32 upcast; "
                         "the reference does, the exported graph does not")
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
