#!/usr/bin/env python3
"""Run Alpamayo-1 end to end on a Jetson AGX Xavier and time every stage.

The model runs entirely inside four TensorRT engines. This driver owns only the
parts TensorRT cannot: image preprocessing, the embedding gather, token sampling,
the persistent KV cache, and the flow-matching Euler loop.

    python run_alpamayo.py --work /mnt/ssdhome/models/alpamayo \
        --images frames/*.jpg --precision int8 --max-new-tokens 256

Fixtures the H100 must have produced (see h100/a1_golden.py):
    fixtures/embed_tokens.fp16.npy   [155697, 4096] token embedding table
    fixtures/input_ids.npy           [S] prompt ids with visual placeholders
    fixtures/position_ids.npy        [3, S] Qwen3-VL 3D mRoPE positions
    fixtures/visual_mask.npy         [S] bool, True at visual-token positions
    fixtures/meta.json               prefill length, rope_deltas, max_seq
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_xavier import postprocess, preprocess, rope           # noqa: E402
from alpamayo_xavier.trt_runner import (Engine, engine_path,        # noqa: E402
                                        load_engines, plan_bytes)

LAYERS, KV_HEADS, HEAD_DIM = 36, 8, 128
N_WAYPOINTS, FLOW_STEPS = 64, 10
TRAJ_TOKEN_START, TRAJ_VOCAB = 151_669, 4_000
TRAJ_FUTURE_START = 155_681
NEG_INF = -65504.0                       # float16 minimum, not -inf: TRT dislikes inf


class Timer(object):
    """CUDA-event stage timing. Wall-clock around async kernels measures nothing."""

    def __init__(self):
        self.stages, self._open = {}, None

    def __call__(self, name):
        self._open = name
        return self

    def __enter__(self):
        self.s, self.e = torch.cuda.Event(True), torch.cuda.Event(True)
        self.s.record()
        return self

    def __exit__(self, *_):
        self.e.record()
        torch.cuda.synchronize()
        self.stages.setdefault(self._open, []).append(self.s.elapsed_time(self.e))
        return False

    def total(self):
        return sum(sum(v) for v in self.stages.values())

    def report(self):
        total = self.total()
        print("\n%-26s %10s %8s %9s" % ("stage", "ms", "share", "calls"))
        print("-" * 56)
        for name, vals in self.stages.items():
            ms = sum(vals)
            print("%-26s %10.1f %7.1f%% %9d" % (name, ms, 100.0 * ms / total, len(vals)))
        print("-" * 56)
        print("%-26s %10.1f %7s   %6.3f Hz" % ("TOTAL", total, "", 1000.0 / total))
        return total


def sample(logits, temperature, top_p, generator):
    """Nucleus sampling with Alpamayo's trajectory-token mask applied.

    The reference masks the 4,000 discrete trajectory tokens out of the reasoning
    rollout entirely -- they exist for training, and at inference the flow-matching
    expert produces the trajectory instead. Leaving them unmasked lets the model
    emit trajectory ids mid-sentence and derails the trace.
    """
    logits = logits[0, -1].float()
    logits[TRAJ_TOKEN_START:TRAJ_TOKEN_START + TRAJ_VOCAB] = float("-inf")
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, descending=True)
    cdf = torch.cumsum(probs[order], dim=-1)
    keep = cdf <= top_p
    keep[0] = True                                   # never empty the nucleus
    idx = order[keep]
    pick = torch.multinomial(probs[idx], 1, generator=generator)
    return int(idx[pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("ALPAMAYO_WORK",
                                                     "/mnt/ssdhome/models/alpamayo"))
    ap.add_argument("--images", nargs="+", required=True,
                    help="16 frames: 4 cameras x 4 timesteps, in the training order")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--flow-steps", type=int, default=FLOW_STEPS)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="skip the CoC rollout entirely and go straight to the expert. "
                         "This is the single largest latency lever on Xavier; it costs "
                         "planning accuracy on long-tail cases.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--residency", default="auto", choices=["auto", "lazy", "resident"],
                    help="auto: keep all engines resident if they fit, else load and "
                         "release one stage at a time. The four FP16 engines total "
                         "~35 GB against ~25 GiB free, so FP16 needs lazy; INT8 fits.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    work = args.work
    fx = os.path.join(work, "fixtures")
    meta = json.load(open(os.path.join(fx, "meta.json")))
    max_seq, prefill_len = meta["max_seq"], meta["prefill"]

    files = sorted(sum([glob.glob(p) for p in args.images], []))
    if len(files) != 16:
        print("warning: expected 16 frames (4 cameras x 4 timesteps), got %d" % len(files))

    print("device      : %s sm_%d%d" % ((torch.cuda.get_device_name(0),) +
                                        torch.cuda.get_device_capability()))
    print("precision   : %s   prefill %d   cache %d" % (args.precision, prefill_len, max_seq))

    eng_dir = os.path.join(work, "engines")
    need = plan_bytes(eng_dir, args.precision)
    free, _ = torch.cuda.mem_get_info()
    headroom = 2.5 * 2**30                       # KV cache, embed table, activations
    mode = args.residency
    if mode == "auto":
        mode = "resident" if need + headroom < free else "lazy"
    print("engines     : %.1f GB of plans, %.1f GiB free -> %s"
          % (need / 1e9, free / 2**30, mode))
    if mode == "lazy":
        print("              (loading one stage at a time; add --residency resident "
              "to override)")

    eng = load_engines(eng_dir, args.precision) if mode == "resident" else {}

    def get(name):
        """Return the engine for a stage, loading it on demand in lazy mode."""
        if name not in eng:
            eng[name] = Engine(engine_path(eng_dir, name, args.precision))
        return eng[name]

    def release(name):
        if mode == "lazy" and name in eng:
            eng.pop(name).close()

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    timer = Timer()

    # ---- persistent state, allocated once ---------------------------------
    past_k = torch.zeros(LAYERS, 1, KV_HEADS, max_seq, HEAD_DIM, dtype=torch.float16, device="cuda")
    past_v = torch.zeros_like(past_k)
    if mode == "resident":
        for name in ("decode", "expert"):
            eng[name].bind("past_k", past_k)
            eng[name].bind("past_v", past_v)
    print("kv cache    : %.0f MiB resident\n" % (2 * past_k.numel() * 2 / 2**20))

    embed = torch.from_numpy(np.load(os.path.join(fx, "embed_tokens.fp16.npy"))).cuda()
    input_ids = torch.from_numpy(np.load(os.path.join(fx, "input_ids.npy"))).long().cuda()
    position_ids = np.load(os.path.join(fx, "position_ids.npy"))
    visual_mask = torch.from_numpy(np.load(os.path.join(fx, "visual_mask.npy"))).cuda()
    cos_p, sin_p = rope.tables(position_ids[:, :prefill_len])
    cos_p = torch.from_numpy(cos_p).cuda()
    sin_p = torch.from_numpy(sin_p).cuda()

    results = []
    for run in range(args.repeat):
        timer.stages.clear()

        # ---- 1. preprocessing (CPU) ---------------------------------------
        t0 = time.perf_counter()
        pixel_values, grid_thw = preprocess.preprocess_images(
            [np.asarray(__import__("PIL.Image", fromlist=["Image"]).Image.open(f)) for f in files])
        cpu_ms = (time.perf_counter() - t0) * 1e3
        vit_tok, llm_tok = preprocess.token_counts(grid_thw)

        # ---- 2. vision tower ----------------------------------------------
        with timer("vision"):
            vout = get("vision")({"pixel_values": pixel_values})
        visual = vout["visual_embeds"]
        deepstack = [vout["deepstack%d" % i] for i in range(3)]

        # ---- 3. assemble prefill embeddings -------------------------------
        with timer("embed assembly"):
            embeds = embed[input_ids[:prefill_len]].unsqueeze(0).clone()
            embeds[0, visual_mask[:prefill_len]] = visual[:llm_tok].to(embeds.dtype)
            ds_full = []
            for d in deepstack:
                z = torch.zeros_like(embeds)
                z[0, visual_mask[:prefill_len]] = d[:llm_tok].to(z.dtype)
                ds_full.append(z)

        # ---- 4. prefill ----------------------------------------------------
        with timer("prefill"):
            pout = get("prefill")({"inputs_embeds": embeds, "cos": cos_p, "sin": sin_p,
                                   "deepstack0": ds_full[0], "deepstack1": ds_full[1],
                                   "deepstack2": ds_full[2]})
            past_k[:, :, :, :prefill_len].copy_(pout["k_cache"])
            past_v[:, :, :, :prefill_len].copy_(pout["v_cache"])
            hidden = pout["last_hidden"].clone()
        release("prefill")

        # ---- 5. chain-of-causation rollout ---------------------------------
        pos = prefill_len
        tokens = []
        mask = torch.full((1, 1, 1, max_seq + 1), NEG_INF, dtype=torch.float16, device="cuda")
        if not args.no_reasoning:
            for _ in range(args.max_new_tokens):
                mask[..., :pos] = 0.0
                mask[..., max_seq] = 0.0                  # the token being generated
                c, s = rope.tables(position_ids[:, pos:pos + 1])
                with timer("decode"):
                    d_eng = get("decode")
                    if mode == "lazy" and d_eng.inputs.get("past_k") is not past_k:
                        d_eng.bind("past_k", past_k); d_eng.bind("past_v", past_v)
                    dout = d_eng({
                        "hidden": hidden, "cos": c, "sin": s, "mask": mask})
                    past_k[:, :, :, pos:pos + 1].copy_(dout["new_k"])
                    past_v[:, :, :, pos:pos + 1].copy_(dout["new_v"])
                    tok = sample(dout["logits"], args.temperature, args.top_p, gen)
                pos += 1
                tokens.append(tok)
                if tok == TRAJ_FUTURE_START:
                    break
                hidden = embed[tok].view(1, 1, -1).clone()

        # ---- 6. flow-matching action expert --------------------------------
        emask = torch.full((1, 1, N_WAYPOINTS, max_seq + N_WAYPOINTS), NEG_INF,
                           dtype=torch.float16, device="cuda")
        emask[..., :pos] = 0.0
        emask[..., max_seq:] = 0.0                        # expert tokens see each other
        wpos = np.arange(N_WAYPOINTS) + pos + meta.get("rope_deltas", 0)
        wcos, wsin = rope.tables(np.broadcast_to(wpos, (3, N_WAYPOINTS)))
        wcos, wsin = torch.from_numpy(wcos).cuda(), torch.from_numpy(wsin).cuda()

        release("decode")
        x = torch.randn(1, N_WAYPOINTS, 2, dtype=torch.float16, device="cuda", generator=gen)
        ts = torch.linspace(0.0, 1.0, args.flow_steps + 1)
        with timer("expert"):
            e_eng = get("expert")
            if mode == "lazy" and e_eng.inputs.get("past_k") is not past_k:
                e_eng.bind("past_k", past_k); e_eng.bind("past_v", past_v)
            for i in range(args.flow_steps):
                dt = float(ts[i + 1] - ts[i])
                t = torch.full((1, 1, 1), float(ts[i]), dtype=torch.float16, device="cuda")
                out = e_eng({"noisy_action": x, "timestep": t,
                                     "cos": wcos, "sin": wsin, "mask": emask})
                x = x + dt * out["velocity"]

        release("expert")
        xy, heading = postprocess.action_to_waypoints(x.float().cpu().numpy()[0])

        total = timer.report()
        print("\npreprocess (CPU)          %10.1f ms  [not in TOTAL]" % cpu_ms)
        print("reasoning tokens          %10d" % len(tokens))
        print("visual tokens             %10d  (%d ViT patches)" % (llm_tok, vit_tok))
        print("final waypoint            %10s m" % ("%.1f, %.1f" % (xy[-1, 0], xy[-1, 1])))
        print("peak CUDA allocation      %10.0f MiB" %
              (torch.cuda.max_memory_allocated() / 2**20))
        results.append(dict(run=run, total_ms=total, cpu_preprocess_ms=cpu_ms,
                            reasoning_tokens=len(tokens), visual_tokens=llm_tok,
                            stages={k: sum(v) for k, v in timer.stages.items()},
                            waypoints=xy.tolist()))

    if args.repeat > 1:
        totals = [r["total_ms"] for r in results]
        print("\nacross %d runs: p50 %.0f ms   p95 %.0f ms" %
              (len(totals), statistics.median(totals), sorted(totals)[-1]))
    if args.json:
        json.dump(dict(precision=args.precision, device=torch.cuda.get_device_name(0),
                       runs=results), open(args.json, "w"), indent=2)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
