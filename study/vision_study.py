#!/usr/bin/env python3
"""Dissect the Alpamayo-R1 vision tower: what each of the 27 ViT blocks costs,
how much each one changes, and how much of the image the tower actually needs.

Runs on the H100 against the SAVED golden pixel tensor, so it needs no dataset,
no network and no tokenizer -- only the checkpoint and $ALPAMAYO_ROOT/golden.

    python study/vision_study.py                  # capture, then plot
    python study/vision_study.py --stage capture  # H100 only
    python study/vision_study.py --stage plot     # anywhere, from the .npz

What is measured, and where the idea comes from
-----------------------------------------------
* per-block latency, and EfficientVLA's layer-importance score, Eq (1) of
  arXiv:2506.10100 (Yang et al., NeurIPS 2025), implemented as written:
      I(l) = 1 - mean_j cos( x_in[j] , x_out[j] )
  over every position j, where a high cosine means the block barely changed its
  input and therefore scores as redundant. Their Section 3.2 then sorts I
  ascending and drops the first n blocks; panel (b) shows that order. Applied
  here to the VISION tower rather than the language module. That paper's code
  URL 404s (checked 2026-09-18) and no mirror exists, so Eq (1)-(3) were read
  out of the PDF instead.

* token importance from ATTENTION RECEIVED, and a keep-fraction ablation against
  a norm proxy and a random control. The criterion is SpecPrune-VLA's, read from
  openvla-oft/experiments/robot/spec_prune_vla.py in
  github.com/alexwhz-sjtu/SpecPrune-VLA: vlm_layer_attn() averages the attention
  map over heads (`attn_map = ... .mean(dim=0)`) and then over query rows
  (`attn_dict[layer_idx] = relation.mean(dim=0)`), and get_layer_attn_indices()
  ranks patches by that score and unions the top-k over several goal_layers.
  Two deviations, both forced by where we are looking: theirs scores VISION keys
  against TEXT queries inside the LLM, and there are no text tokens inside a ViT,
  so queries here are the patches themselves; and we rank once on the summed
  score over the tap blocks instead of unioning per-layer top-k, because the
  ablation needs one ranking at an exact budget. The "compact, diverse set of
  visual tokens" framing is EfficientVLA Section 3.3, whose Eq (2) scores a token
  by attention averaged over heads and summed over the task-context tokens --
  the same ranking as SpecPrune's, since summing and averaging over the context
  differ only by a constant. Their Eq (3) anchors the selection with the top
  K_key tokens (K_key = 4 in their reported setup) and then augments it by
  trading relevance against feature diversity; `--mmr` reimplements that as a
  greedy maximal-marginal-relevance pass, because Section 3.3.3's exact
  augmentation equation did not survive text extraction from the PDF.

* per-patch similarity between the four timesteps of each camera, and between
  whole images. This is SpecPrune-VLA's get_similarity_indices(), which keeps
  patches whose frame-to-frame `similarity >= sim_threshold` as low-change
  regions. Alpamayo feeds 4 cameras x 4 timesteps in ONE forward, so their
  spatial-temporal consistency claim is testable inside a single frame rather
  than across action steps.

No lines are copied: SpecPrune-VLA's scoring is reimplemented for a ViT, with
the two deviations named above. Nothing here is a speedup claim -- the ablation
ZEROES tokens rather than removing them, so it measures information loss only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "h100"))

DEPTH = 27                     # arch.VISION["depth"]
TAPS = (8, 16, 24)             # arch.VISION["deepstack_indexes"]
KEEP = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)
CAMERAS, STEPS = 4, 4


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def find_blocks(visual, depth):
    """The ModuleList of transformer blocks, whatever the release calls it."""
    import torch.nn as nn
    for name, mod in visual.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == depth:
            return name, mod
    raise SystemExit("no ModuleList of length %d under the vision tower; "
                     "inspect visual.named_modules() and set DEPTH" % depth)


def as_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class SDPAEntropy:
    """Normalised attention entropy per block, without touching the model.

    SpecPrune-VLA Fig 4(I) reports attention entropy across layers as its main
    layer-level diagnostic. Qwen3-VL's ViT runs fused SDPA and never returns
    attention weights, so this wraps torch's SDPA, recomputes the probabilities
    for a random subset of query rows, and takes their entropy. Entropy is
    divided by log(kv_len) so 1.0 means "attends uniformly" and 0 means "attends
    to one token", which makes blocks with different sequence lengths
    comparable. Queries are subsampled because the full 11 520 x 11 520 matrix
    does not fit; the real SDPA still computes the actual output.
    """

    def __init__(self, rows=256, seed=0):
        self.rows, self.seed, self.values, self.ok = rows, seed, [], True
        self.recv = []          # attention received per key, SpecPrune's score

    def __enter__(self):
        import torch
        import torch.nn.functional as F
        self._real = F.scaled_dot_product_attention
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        def wrapper(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=None, **kw):
            try:
                if q.dim() == 4 and q.shape[-2] >= 8 and not is_causal:
                    n = min(self.rows, q.shape[-2])
                    idx = torch.randperm(q.shape[-2], generator=g)[:n].to(q.device)
                    qs = q.index_select(-2, idx).float()
                    logits = (qs @ k.float().transpose(-1, -2)) * (
                        scale if scale is not None else q.shape[-1] ** -0.5)
                    if isinstance(attn_mask, torch.Tensor):
                        m = attn_mask
                        if m.dtype == torch.bool:
                            m = torch.zeros_like(m, dtype=logits.dtype).masked_fill(
                                ~m, float("-inf"))
                        if m.shape[-2] == q.shape[-2]:
                            m = m.index_select(-2, idx)
                        logits = logits + m
                    p = logits.softmax(dim=-1)
                    ent = -(p * (p.clamp_min(1e-12)).log()).sum(-1)
                    self.values.append(float(ent.mean()) / float(np.log(k.shape[-2])))
                    # mean over heads then over query rows == SpecPrune's
                    # vlm_layer_attn(): attn_map.mean(dim=0) then relation.mean(dim=0)
                    self.recv.append(p.mean(dim=tuple(range(p.dim() - 1))).cpu().numpy())
            except Exception:
                self.ok = False
            return self._real(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, **kw)

        F.scaled_dot_product_attention = wrapper
        return self

    def __exit__(self, *exc):
        import torch.nn.functional as F
        F.scaled_dot_product_attention = self._real
        return False


def capture(args):
    import torch
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    root = os.environ.get("ALPAMAYO_ROOT", args.root)
    inputs = os.path.join(root, "golden", "inputs.npz")
    if not os.path.exists(inputs):
        raise SystemExit("no %s -- run h100/a1_golden.py first" % inputs)
    z = np.load(inputs, allow_pickle=True)
    px = torch.from_numpy(z["pixel_values"])
    grid = torch.from_numpy(z["image_grid_thw"])
    n_img = int(grid.shape[0])
    print("pixels %s  grid %s  (%d images)" % (tuple(px.shape), grid.tolist()[0], n_img))

    if not os.path.isdir(args.model):
        raise SystemExit("checkpoint not found: %s\n"
                         "Set ALPAMAYO_MODEL or pass --model." % args.model)
    shards = len([f for f in os.listdir(args.model) if f.endswith(".safetensors")])
    print("loading %s  (%d safetensors shards, local, no download)"
          % (args.model, shards))
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    visual = model.vlm.model.visual.to("cuda")
    del model                                     # only the tower is needed
    torch.cuda.empty_cache()
    name, blocks = find_blocks(visual, args.depth)
    dtype = next(visual.parameters()).dtype
    px = px.to("cuda", dtype)
    grid = grid.to("cuda")
    print("vision tower: %s, %d blocks, %s" % (name, len(blocks), dtype))

    rec = {}

    # ---- pass 1: what each block does to the hidden state ----------------
    state = {"in": None, "stats": [], "taps": {}, "first_in": None}

    def in_hook(i):
        def fn(mod, inp):
            x = as_tensor(inp[0]).detach().float()
            state["in"] = x.reshape(-1, x.shape[-1])
            if i == 0:
                state["first_in"] = state["in"]
        return fn

    def stats_hook(i):
        def fn(mod, inp, out):
            h = as_tensor(out).detach().float()
            h2 = h.reshape(-1, h.shape[-1])
            a = h2.abs()
            row = dict(block=i,
                       absmax=float(a.max()),
                       p999=float(torch.quantile(a[:: max(1, a.numel() // 2_000_000)].flatten(), 0.999)),
                       median=float(a.median()),
                       token_norm=float(h2.norm(dim=-1).mean()))
            x_in = state["in"]
            if x_in is not None and x_in.shape == h2.shape:
                # EfficientVLA Eq (1): mean over positions of cos(input, output)
                row["cos_io"] = float(torch.nn.functional.cosine_similarity(
                    x_in, h2, dim=-1).mean())
                row["importance"] = 1.0 - row["cos_io"]
            state["stats"].append(row)
            if i in TAPS or i == len(blocks) - 1:
                state["taps"][i] = h2.norm(dim=-1).cpu().numpy()
        return fn

    handles = [b.register_forward_pre_hook(in_hook(i)) for i, b in enumerate(blocks)]
    handles += [b.register_forward_hook(stats_hook(i)) for i, b in enumerate(blocks)]
    ent = SDPAEntropy(rows=args.attn_rows)
    with torch.no_grad(), ent:
        base = visual(px, grid)
    base_embeds = as_tensor(base).detach().float()
    for h in handles:
        h.remove()
    rec["stats"] = state["stats"]
    rec["taps"] = {str(k): v for k, v in state["taps"].items()}
    # One SDPA call per block, in order -- if that does not hold, say so rather
    # than plotting a curve whose x axis means nothing.
    nb, nc = len(blocks), len(ent.values)
    per_block = nc // nb if nb and nc % nb == 0 else 0
    rec["calls_per_block"] = per_block
    if per_block:
        e = np.array(ent.values, dtype=np.float64).reshape(nb, per_block)
        rec["entropy"] = e.mean(axis=1)
        # one call per block covers all patches; several calls means one per
        # image, so their key vectors concatenate back into the full grid.
        rows = [np.concatenate(ent.recv[i * per_block:(i + 1) * per_block])
                for i in range(nb)]
        rec["recv"] = (np.stack(rows) if all(r.size == rows[0].size for r in rows)
                       else np.zeros((nb, 0)))
    else:
        rec["entropy"], rec["recv"] = np.zeros(nb), np.zeros((nb, 0))
    n_patch = int(state["first_in"].shape[0])
    rec["entropy_ok"] = bool(ent.ok and per_block)
    rec["recv_ok"] = bool(rec["entropy_ok"] and rec["recv"].shape[1] == n_patch)
    print("attention probe: %d SDPA calls / %d blocks = %s per block; entropy %s, "
          "per-patch scores %s"
          % (nc, nb, per_block or "?", "ok" if rec["entropy_ok"] else "UNUSABLE",
             "ok" if rec["recv_ok"] else "unusable"))
    print("hidden-state pass done: %d blocks, embeds %s"
          % (len(state["stats"]), tuple(base_embeds.shape)))

    # ---- pass 2: per-block latency ---------------------------------------
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in blocks]
    times = np.zeros((args.repeat, len(blocks)), dtype=np.float64)

    def timing_hooks():
        out = []
        for i, b in enumerate(blocks):
            out.append(b.register_forward_pre_hook(lambda m, inp, i=i: ev[i][0].record()))
            out.append(b.register_forward_hook(lambda m, inp, o, i=i: ev[i][1].record()))
        return out

    handles = timing_hooks()
    with torch.no_grad():
        for _ in range(2):                         # warm up clocks and caches
            visual(px, grid)
        torch.cuda.synchronize()
        for r in range(args.repeat):
            t0 = time.perf_counter()
            visual(px, grid)
            torch.cuda.synchronize()
            whole = (time.perf_counter() - t0) * 1e3
            for i in range(len(blocks)):
                times[r, i] = ev[i][0].elapsed_time(ev[i][1])
            print("  repeat %d: tower %.1f ms, blocks sum %.1f ms"
                  % (r + 1, whole, times[r].sum()))
    for h in handles:
        h.remove()
    rec["block_ms"] = times
    rec["tower_ms"] = float(times.sum(axis=1).mean())

    # ---- pass 3: how much of the image is needed -------------------------
    # Importance = L2 norm of each patch at the tower input, the cheapest proxy
    # that needs no attention weights. Tokens are ZEROED, never removed, so the
    # shapes stay valid and this measures information loss, not speedup.
    norm_score = state["first_in"].norm(dim=-1)
    n_tok = norm_score.numel()
    ranks = {"norm": torch.argsort(norm_score, descending=True)}
    if rec["recv_ok"]:
        tap_rows = [t for t in TAPS if t < len(blocks)]
        attn_score = torch.from_numpy(rec["recv"][tap_rows].sum(axis=0)).to(norm_score.device)
        ranks["attention"] = torch.argsort(attn_score, descending=True)
    if args.mmr and "attention" in ranks:
        # EfficientVLA Sec 3.3: anchor on the top K_key most task-relevant
        # tokens, then add tokens that stay relevant while being unlike what is
        # already held. Greedy maximal marginal relevance; lambda is ours, since
        # Sec 3.3.3's exact form did not survive extraction from the PDF.
        feat = torch.nn.functional.normalize(state["first_in"], dim=-1)
        rel = attn_score.float()
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-12)   # Eq (2) min-max
        budget = int(round(max(KEEP) * n_tok))
        chosen = ranks["attention"][: args.k_key].tolist()          # Eq (3), V_key
        taken = torch.zeros(n_tok, dtype=torch.bool, device=feat.device)
        taken[torch.tensor(chosen, device=feat.device)] = True
        msim = (feat @ feat[chosen].T).max(dim=1).values
        lam = args.mmr_lambda
        while len(chosen) < budget:
            score = lam * rel - (1.0 - lam) * msim
            score[taken] = -1e9
            j = int(torch.argmax(score))
            chosen.append(j)
            taken[j] = True
            msim = torch.maximum(msim, feat @ feat[j])
        ranks["relevance+diversity"] = torch.tensor(chosen, device=feat.device)
        print("  MMR order built: %d tokens, K_key=%d, lambda=%.2f"
              % (budget, args.k_key, lam))

    rng = np.random.default_rng(0)
    modes = ([m for m in ("attention", "relevance+diversity") if m in ranks]
             + ["norm", "random"])
    curves = {"keep": list(KEEP)}
    curves.update({m: [] for m in modes})
    for frac in KEEP:
        k = max(1, int(round(frac * n_tok)))
        for mode in modes:
            if frac >= 1.0:
                curves[mode].append(1.0)
                continue
            mask = torch.zeros_like(norm_score)
            idx = (ranks[mode][:k] if mode in ranks
                   else torch.from_numpy(rng.choice(n_tok, k, replace=False)).to(norm_score.device))
            mask[idx] = 1.0
            hook = blocks[0].register_forward_pre_hook(
                lambda m, inp, mk=mask: (as_tensor(inp[0]) * mk.to(inp[0].dtype)[:, None],))
            with torch.no_grad():
                out = as_tensor(visual(px, grid)).detach().float()
            hook.remove()
            curves[mode].append(float(torch.nn.functional.cosine_similarity(
                base_embeds.reshape(1, -1), out.reshape(1, -1)).item()))
        print("  keep %4.0f%%: %s" % (frac * 100, "  ".join(
            "%s %.4f" % (m, curves[m][-1]) for m in modes)))
    rec["prune"], rec["modes"] = curves, modes

    # ---- pass 4: is one camera's four timesteps redundant? ---------------
    e = base_embeds.reshape(n_img, -1, base_embeds.shape[-1]).mean(dim=1)
    e = torch.nn.functional.normalize(e, dim=-1)
    rec["image_sim"] = (e @ e.T).cpu().numpy()

    # Patch-level version of the same question, which is what SpecPrune's
    # get_similarity_indices() actually thresholds: does THIS patch change
    # between consecutive timesteps of the SAME camera?
    per_img = state["first_in"].shape[0] // n_img
    g = state["first_in"].reshape(n_img, per_img, -1)
    pairs = [torch.nn.functional.cosine_similarity(
                 g[c * STEPS + t], g[c * STEPS + t + 1], dim=-1)
             for c in range(n_img // STEPS) for t in range(STEPS - 1)]
    rec["patch_sim"] = torch.cat(pairs).cpu().numpy() if pairs else np.zeros(0)

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "vision_study.npz")
    np.savez_compressed(
        npz, block_ms=rec["block_ms"], image_sim=rec["image_sim"],
        keep=np.array(curves["keep"]), prune_importance=np.array(curves["importance"]),
        entropy=rec["entropy"], patch_sim=rec["patch_sim"],
        **{"prune_%s" % m: np.array(curves[m]) for m in modes},
        **{"tap%s" % k: v for k, v in rec["taps"].items()})
    meta = dict(depth=len(blocks), module=name, dtype=str(dtype), images=n_img,
                patches=int(px.shape[0]), merged=int(base_embeds.shape[0]),
                tower_ms=rec["tower_ms"], repeats=args.repeat,
                gpu=torch.cuda.get_device_name(0), stats=rec["stats"],
                taps=list(TAPS), entropy_ok=rec["entropy_ok"],
                recv_ok=rec["recv_ok"], calls_per_block=rec["calls_per_block"],
                prune_modes=modes, entropy_rows=args.attn_rows,
                k_key=args.k_key, mmr_lambda=args.mmr_lambda,
                note="tokens are zeroed, not removed: information loss, not speedup")
    with open(os.path.join(args.out, "vision_study.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nwrote %s and vision_study.json" % npz)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def panel(ax, letter, title):
    """(a) + a neutral description. Findings go in data-driven annotations,
    never in the title, so the figure cannot assert something the run did not
    show."""
    ax.set_title("(%s) %s" % (letter, title), loc="left", pad=3, fontsize=7.2)


def longest_run(values, threshold):
    """The longest stretch of consecutive blocks at or above `threshold`."""
    best = cur = (0, -1)
    for i, v in enumerate(values):
        if v is not None and not np.isnan(v) and v >= threshold:
            cur = (cur[0] if cur[1] == i - 1 else i, i)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
    return best if best[1] > best[0] else None


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

    z = np.load(os.path.join(args.out, "vision_study.npz"))
    meta = json.load(open(os.path.join(args.out, "vision_study.json")))
    stats, depth = meta["stats"], meta["depth"]
    x = np.arange(depth)
    ms = z["block_ms"].mean(axis=0)
    taps = meta["taps"]
    imp_l = np.array([s.get("importance", np.nan) for s in stats], dtype=float)
    fin = imp_l[~np.isnan(imp_l)]

    def grid(ax, axis="y"):
        ax.grid(True, axis=axis, lw=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    # ---- figure A: the 27 blocks ----------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.05), constrained_layout=True)

    ax[0].bar(x, ms, 0.74, color=BLUE, zorder=3, label="block")
    ax[0].bar(taps, ms[taps], 0.74, color=TEAL, zorder=4, label="DeepStack tap")
    ax[0].set_ylim(0, ms.max() * 1.34)
    ax[0].set_xlabel("ViT block")
    ax[0].set_ylabel("ms")
    ax[0].annotate("%.0f ms total\nspread %.0f%% of the mean"
                   % (meta["tower_ms"], 100 * (ms.max() - ms.min()) / ms.mean()),
                   xy=(0.03, 0.97), xycoords="axes fraction", fontsize=5.7,
                   va="top", color=MUTED)
    ax[0].legend(fontsize=5.5, handlelength=0.8, handletextpad=0.35,
                 loc="upper right", borderaxespad=0.2)
    panel(ax[0], "a", "cost per block")
    grid(ax[0])

    ax[1].bar(x, np.nan_to_num(imp_l), 0.74, color=RUST, zorder=3)
    if fin.size:
        n_drop = max(1, int(round(0.25 * depth)))        # illustrative budget
        drop = np.argsort(np.where(np.isnan(imp_l), np.inf, imp_l))[:n_drop]
        ax[1].bar(drop, np.nan_to_num(imp_l)[drop], 0.74, color=MUTED, zorder=4)
        ax[1].annotate("lowest $I$, dropped first:\n%s" % ", ".join(
            str(int(d)) for d in np.sort(drop)),
            xy=(0.03, 0.97), xycoords="axes fraction", fontsize=5.6, va="top",
            color=MUTED)
        ax[1].set_ylim(0, float(np.nanmax(fin)) * 1.38)
    ax[1].set_xlabel("ViT block")
    ax[1].set_ylabel("$I(\\ell) = 1 - \\cos(x_{in}, x_{out})$")
    panel(ax[1], "b", "layer importance, EfficientVLA Eq (1)")
    grid(ax[1])

    ent = z["entropy"] if "entropy" in z.files else np.array([])
    if meta.get("entropy_ok") and ent.size == depth:
        ax[2].plot(x, ent, "-o", ms=2.6, lw=1.0, color=PLUM, zorder=4)
        ax[2].axhline(1.0, color=MUTED, lw=0.7, ls=(0, (4, 2)))
        ax[2].annotate("1.0 = attends everywhere", xy=(depth - 1, 1.0), xytext=(-2, -6),
                       textcoords="offset points", fontsize=5.6, ha="right", va="top",
                       color=MUTED)
        lo = int(np.argmin(ent))
        ax[2].annotate("most selective:\nblock %d, %.2f" % (lo, ent[lo]),
                       xy=(lo, float(ent[lo])), xytext=(0, -10),
                       textcoords="offset points", fontsize=5.7, color=PLUM,
                       ha="center", va="top",
                       arrowprops=dict(arrowstyle="-", lw=0.5, color=PLUM))
        ax[2].set_ylim(0, 1.14)
        ax[2].set_ylabel("normalised attention entropy")
    else:
        ax[2].text(0.5, 0.5, "attention not captured\n(fused kernel)", ha="center",
                   va="center", fontsize=6.4, color=MUTED, transform=ax[2].transAxes)
        ax[2].set_yticks([])
    ax[2].set_xlabel("ViT block")
    panel(ax[2], "c", "how focused attention is")
    grid(ax[2])

    amax = np.array([s["absmax"] for s in stats])
    ax[3].semilogy(x, amax, "-o", ms=2.6, lw=1.0, color=BLUE, label="max $|x|$", zorder=4)
    ax[3].semilogy(x, [s["p999"] for s in stats], "--s", ms=2.2, lw=0.9, color=MUTED,
                   label="99.9th pct", zorder=3)
    lim = 65504.0 ** 0.5
    ax[3].axhline(lim, color=FAULT, lw=1.0)
    over = np.nonzero(amax > lim)[0]
    tail = ("\nfirst crossed at block %d" % over[0]) if over.size else "\nnever crossed"
    ax[3].annotate("$\\sqrt{65\\,504}$" + tail, xy=(0, lim), xytext=(2, 3),
                   textcoords="offset points", fontsize=5.7, color=FAULT,
                   ha="left", va="bottom")
    ax[3].set_xlabel("ViT block")
    ax[3].set_ylabel("activation")
    ax[3].legend(fontsize=5.5, handlelength=0.9, handletextpad=0.35, loc="lower right",
                 borderaxespad=0.2)
    panel(ax[3], "d", "activation magnitude")
    grid(ax[3])
    save(fig, "fig_vision_layers", args.out)

    # ---- figure B: the 11 520 patches ------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.05), constrained_layout=True)

    half = None
    for t in sorted(taps) + [depth - 1]:
        key = "tap%d" % t
        if key not in z.files:
            continue
        v = np.sort(z[key])[::-1] ** 2
        frac = np.arange(1, v.size + 1) / v.size
        cum = np.cumsum(v) / v.sum()
        ax[0].plot(frac, cum, lw=1.0, label="block %d" % t)
        if t == depth - 1:
            half = float(frac[int(np.searchsorted(cum, 0.5))])
    ax[0].plot([0, 1], [0, 1], ":", color=MUTED, lw=0.8)
    if half:
        ax[0].annotate("half the norm sits\nin %.0f%% of patches" % (100 * half),
                       xy=(half, 0.5), xytext=(12, -16), textcoords="offset points",
                       fontsize=5.7, color=MUTED,
                       arrowprops=dict(arrowstyle="->", lw=0.5, color=MUTED))
    ax[0].set_xlabel("patches kept, most important first")
    ax[0].set_ylabel("share of squared norm")
    ax[0].legend(fontsize=5.5, handlelength=0.9, handletextpad=0.35, loc="lower right",
                 borderaxespad=0.2)
    panel(ax[0], "a", "where the information sits")
    grid(ax[0], axis="both")

    keep = z["keep"] * 100
    STYLE = {"attention": (BLUE, "-o", "attention received"),
             "relevance+diversity": (TEAL, "-D", "relevance + diversity"),
             "norm": (PLUM, "-.^", "patch norm"),
             "random": (RUST, "--s", "random")}
    modes = [m for m in meta.get("prune_modes", ["norm", "random"])
             if "prune_%s" % m in z.files]
    ax[1].axhspan(0.99, 1.005, color=TEAL, alpha=0.10, lw=0)
    best = None
    for m in modes:
        c, mk, lab = STYLE.get(m, (MUTED, "-o", m))
        y = z["prune_%s" % m]
        ax[1].plot(keep, y, mk, ms=2.8, lw=1.05, color=c, label=lab,
                   zorder=5 if m == "attention" else 3)
        ok = keep[y >= 0.99]
        if ok.size and (best is None or ok.min() < best[0]):
            best = (float(ok.min()), float(y[keep == ok.min()][0]), c, lab)
    if best:
        ax[1].annotate("%.0f%% holds\ncosine $\\geq$ 0.99" % best[0],
                       xy=(best[0], best[1]), xytext=(-8, -22), ha="right",
                       textcoords="offset points", fontsize=5.6, color=best[2],
                       arrowprops=dict(arrowstyle="->", lw=0.5, color=best[2]))
    ax[1].set_xlabel("patches kept (%)")
    ax[1].set_ylabel("cosine with the full tower")
    ax[1].legend(fontsize=5.4, handlelength=1.0, handletextpad=0.35, loc="lower right",
                 borderaxespad=0.2)
    panel(ax[1], "b", "keeping top patches (zeroed, not removed)")
    grid(ax[1], axis="both")

    ps = z["patch_sim"] if "patch_sim" in z.files else np.zeros(0)
    if ps.size:
        v = np.sort(ps)
        cdf = np.arange(1, v.size + 1) / v.size
        ax[2].plot(v, cdf, lw=1.2, color=TEAL, zorder=4)
        for thr in (0.90, 0.99):
            frac_above = float((ps >= thr).mean())
            ax[2].axvline(thr, color=MUTED, lw=0.7, ls=(0, (4, 2)))
            ax[2].annotate("%.0f%% of patches\n$\\geq$ %.2f" % (100 * frac_above, thr),
                           xy=(thr, 1.0 - frac_above), xytext=(-4, 6),
                           textcoords="offset points", fontsize=5.5, ha="right",
                           va="bottom", color=MUTED)
        ax[2].set_xlim(max(-1.0, float(v.min()) - 0.02), 1.005)
    else:
        ax[2].text(0.5, 0.5, "not captured", ha="center", va="center",
                   fontsize=6.4, color=MUTED, transform=ax[2].transAxes)
    ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("cosine, same patch, next timestep")
    ax[2].set_ylabel("cumulative fraction of patches")
    panel(ax[2], "c", "does a patch change between steps?")
    grid(ax[2], axis="both")

    sim = z["image_sim"]
    n = sim.shape[0]
    im = ax[3].imshow(sim, cmap="magma", vmin=float(sim.min()), vmax=1.0)
    for c in range(1, CAMERAS):
        ax[3].axhline(c * STEPS - 0.5, color="white", lw=0.6)
        ax[3].axvline(c * STEPS - 0.5, color="white", lw=0.6)
    ticks = [c * STEPS + (STEPS - 1) / 2.0 for c in range(CAMERAS)]
    labels = ["cam%d" % c for c in range(CAMERAS)]
    ax[3].set_xticks(ticks); ax[3].set_xticklabels(labels, fontsize=6)
    ax[3].set_yticks(ticks); ax[3].set_yticklabels(labels, fontsize=6)
    same = float(np.mean([sim[c * STEPS + i, c * STEPS + j] for c in range(CAMERAS)
                          for i in range(STEPS) for j in range(STEPS) if i != j]))
    other = float(np.mean([sim[a, b] for a in range(n) for b in range(n)
                           if a // STEPS != b // STEPS]))
    ax[3].set_xlabel("same camera %.3f    different %.3f" % (same, other), fontsize=6.2)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    panel(ax[3], "d", "4 cameras $\\times$ 4 timesteps")
    save(fig, "fig_vision_tokens", args.out)

    # ---- audit trail -----------------------------------------------------
    print("\nwhat the capture recorded")
    print("  tower            %.1f ms over %d blocks on %s"
          % (meta["tower_ms"], depth, meta["gpu"]))
    print("  slowest/fastest  block %d (%.2f ms) / block %d (%.2f ms)"
          % (int(np.argmax(ms)), ms.max(), int(np.argmin(ms)), ms.min()))
    if fin.size:
        i = int(np.nanargmin(np.where(np.isnan(imp_l), np.inf, imp_l)))
        print("  lowest I(l)      block %d, I = %.4f (cosine %.4f with its input)"
              % (i, imp_l[i], 1.0 - imp_l[i]))
    if meta.get("entropy_ok") and ent.size == depth:
        print("  attention entropy %.3f (block %d) to %.3f (block %d)"
              % (ent.min(), int(np.argmin(ent)), ent.max(), int(np.argmax(ent))))
    else:
        print("  attention entropy not captured")
    for i, f in enumerate(z["keep"]):
        print("  keep %4.0f%%       %s" % (f * 100, "   ".join(
            "%s %.4f" % (m, z["prune_%s" % m][i]) for m in modes)))
    if ps.size:
        print("  patch similarity  %.0f%% of patches >= 0.90 between timesteps"
              % (100 * float((ps >= 0.90).mean())))
    print("  same camera / different camera: %.4f / %.4f mean cosine" % (same, other))


def save(fig, name, outdir):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "%s.%s" % (name, ext))
        fig.savefig(p)
        print("wrote %s" % p)
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", choices=["all", "capture", "plot"])
    ap.add_argument("--model", default=os.environ.get(
        "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B"))
    ap.add_argument("--root", default=os.environ.get(
        "ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--attn-rows", type=int, default=256,
                    help="query rows sampled per block for the entropy estimate")
    ap.add_argument("--mmr", action="store_true", default=True,
                    help="also rank by EfficientVLA's relevance+diversity selection")
    ap.add_argument("--no-mmr", dest="mmr", action="store_false")
    ap.add_argument("--k-key", type=int, default=4,
                    help="EfficientVLA K_key, the unconditionally kept anchor set")
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    args = ap.parse_args()

    if args.stage in ("all", "capture"):
        capture(args)
    if args.stage in ("all", "plot"):
        plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
