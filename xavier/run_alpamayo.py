#!/usr/bin/env python3
"""Run Alpamayo-1 end to end on a Jetson AGX Xavier and time every stage.

The stages use TensorRT engines with an optional PyTorch decoder. This driver owns the
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
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_xavier import pieces as piecelib                      # noqa: E402
from alpamayo_xavier import envinfo, sysmon, telemetry, sample_inputs
from alpamayo_xavier.detok import Detokenizer
from alpamayo_xavier.torch_decode import TorchDecode                # noqa: E402
from alpamayo_xavier import postprocess, preprocess, rope           # noqa: E402
from alpamayo_xavier.trt_runner import Engine, engine_path, plan_bytes

LAYERS, KV_HEADS, HEAD_DIM = 36, 8, 128
N_WAYPOINTS, FLOW_STEPS = 64, 10
TRAJ_TOKEN_START, TRAJ_VOCAB = 151_669, 4_000
TRAJ_FUTURE_START = 155_681
NEG_INF = -65504.0                       # float16 minimum, not -inf: TRT dislikes inf


Timer = telemetry.Trace


def sysmem():
    """What the system really has. On a Xavier the CPU and GPU share one pool, and
    cudaMemGetInfo counts page cache as free although NvMap cannot allocate from it."""
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) * 1024
        return "sys free %.1f, avail %.1f, cached %.1f GB" % (
            d.get("MemFree", 0) / 1e9, d.get("MemAvailable", 0) / 1e9, d.get("Cached", 0) / 1e9)
    except (OSError, ValueError, IndexError):
        return "sys ?"


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
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+",
                        help="16 images matching WORK/fixtures; sorted by filename")
    source.add_argument("--sample", help="prepared sample directory with frames/ and fixtures/")
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
    ap.add_argument("--decode", default="auto", choices=["auto", "trt", "torch"],
                    help="auto: PyTorch when onnx/weight_map.json is there. Decode reads the "
                         "whole language model per token, so it is bandwidth-bound; TensorRT "
                         "measured slower (350 vs 318 ms/token) and its engines do not fit.")
    ap.add_argument("--residency", default="auto", choices=["auto", "lazy", "resident"],
                    help="auto: keep decode's weights loaded between frames (44 s of the "
                         "~85 s per-frame reload) and stream vision, prefill and the expert. "
                         "lazy: release every stage after use (slowest, smallest). resident: "
                         "keep everything, which fp16 does not fit into.")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vocab", default=None, help="exported vocab.json; defaults to WORK/fixtures/vocab.json")
    ap.add_argument("--telemetry-interval", type=float, default=0.5,
                    help="seconds between sysfs samples; 0 disables sensor sampling")
    args = ap.parse_args()
    if args.repeat < 1 or args.flow_steps < 1 or args.max_new_tokens < 0:
        ap.error("repeat and flow-steps must be positive; max-new-tokens must be nonnegative")
    if args.telemetry_interval < 0:
        ap.error("telemetry-interval must be nonnegative")
    setup_start = time.perf_counter()

    work = args.work
    shared_fx = os.path.join(work, "fixtures")
    fx = shared_fx
    if args.sample:
        try:
            fx, meta, files = sample_inputs.load_sample(args.sample, shared_fx)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
    else:
        meta = sample_inputs.read_meta(fx)
        files = sorted(sum([glob.glob(p) for p in args.images], []))
    if len(files) != 16 or len(set(files)) != 16:
        ap.error("expected exactly 16 distinct images (4 cameras x 4 timesteps)")
    reference_grid = sample_inputs.expected_grid(shared_fx, work)
    sample_grid = sample_inputs.expected_grid(fx, work)
    if args.sample:
        if reference_grid is None:
            ap.error("reference image grid is missing; copy fixtures/image_grid_thw.npy or golden/inputs.npz")
        try:
            sample_inputs.validate_contract(meta, sample_inputs.read_meta(shared_fx), sample_grid, reference_grid)
        except ValueError as exc:
            ap.error(str(exc))
    max_seq, prefill_len = meta["max_seq"], meta["prefill"]
    if not args.no_reasoning and prefill_len + args.max_new_tokens > max_seq:
        ap.error("max-new-tokens exceeds the available KV cache slots")
    detok = Detokenizer.find(args.vocab) if args.vocab else Detokenizer.find(fx, shared_fx)
    if not args.no_reasoning and not detok.available:
        print("reasoning text: unavailable -- export fixtures/vocab.json with h100/a8_token_strings.py; IDs are still saved")
    rope_delta = int(meta.get("rope_deltas", 0))   # generated tokens sit at slot + rope_delta

    # The expert predicts acceleration and curvature, so turning them into positions
    # needs the speed the car is doing now. With 0 a correct model "travels" 1.4 m
    # instead of 57. a3b writes the reference's own estimate into meta.json.
    v0 = meta.get("v0")
    if v0 is None:
        gold = os.path.join(work, "golden", "inputs.npz")
        if os.path.exists(gold):
            v0 = postprocess.estimate_v0(np.load(gold, allow_pickle=True)["ego_history_xyz"])
            print("ego speed   : %.2f m/s (estimated from the golden ego history)" % v0)
        else:
            v0 = 0.0
            print("ego speed   : 0 m/s -- no v0 in meta.json and no golden/inputs.npz; "
                  "the trajectory will start from standstill")
    else:
        print("ego speed   : %.2f m/s (fixtures/meta.json)" % v0)

    print("device      : %s sm_%d%d" % ((torch.cuda.get_device_name(0),) +
                                        torch.cuda.get_device_capability()))
    print("precision   : %s   prefill %d   cache %d" % (args.precision, prefill_len, max_seq))

    eng_dir = os.path.join(work, "engines")
    # A stage is one engine, or the pieces h100/a3d_split_graphs.py cut it into
    # (listed in engines/pieces.json, which build_engines.sh copies there).
    graphs = piecelib.load(eng_dir)
    use_torch_decode = args.decode == "torch" or (
        args.decode == "auto" and (os.path.exists(os.path.join(eng_dir, "weight_map.json"))
                                   or os.path.exists(os.path.join(work, "onnx", "weight_map.json"))))
    timer = Timer(torch)
    stages = {}
    for name in ("vision", "prefill", "decode", "expert"):
        if name == "decode" and use_torch_decode:
            stages[name] = TorchDecode(work)
            print("%-12s: PyTorch (weights from %s.onnx.data)" % (name, stages[name].graph))
            continue
        # Prefill runs each of its engines exactly once, so it loads them one at a
        # time: peak 1.5 GB instead of 15.2 GB, and no 13-block fragmentation left
        # behind for decode, which does need all of its engines at once (16 tokens).
        stages[name] = piecelib.Stage(
            name, piecelib.specs(graphs, name),
            lambda piece, skip=(): Engine(engine_path(eng_dir, piece, args.precision), skip=skip),
            sequential=(name == "prefill"), observer=timer.observer)
        if len(stages[name].pieces) > 1:
            print("%-12s: %d engines" % (name, len(stages[name].pieces)))
    need = plan_bytes(eng_dir, args.precision)
    free, _ = torch.cuda.mem_get_info()
    mode = args.residency
    if mode == "resident" and need + 2.5 * 2**30 > free:
        print("engines     : %.1f GB of plans against %.1f GiB free -- too much to keep "
              "resident, falling back to auto" % (need / 1e9, free / 2**30))
        mode = "auto"
    print("engines     : %.1f GB of plans, %.1f GiB free -> %s"
          % (need / 1e9, free / 2**30, mode))

    def preload(name):
        """Lazy mode reads each stage's plans from disk every frame (prefill and
        decode are 15 GB each). Time that as its own stage so it never inflates a
        compute number; in resident mode everything is loaded already."""
        st = stages[name]
        if st.sequential:
            return                     # loads itself, one engine at a time, inside run()
        if len(st.runners) < len(st.pieces):
            with timer.scope("load_preparation", name):
                gc.collect()
                torch.cuda.empty_cache()
            before = torch.cuda.mem_get_info()[0]
            with timer("engine load") as loading:
                loading["detail"]["target_stage"] = name
                try:
                    st.load()
                except (RuntimeError, MemoryError):
                    # Something we are holding between frames is in the way. Give it
                    # up and reload from disk: slower, but the run finishes.
                    held = [n for n in sorted(keep)
                            if n != name and stages[n].runners]
                    if not held:
                        raise
                    print("  %s did not fit -- releasing %s, which stops being resident"
                          % (name, ", ".join(held)))
                    residency_events.append(dict(stage=name, evicted=held,
                                                 reason="load exception; retry after eviction"))
                    for n in held:
                        keep.discard(n)
                        stages[n].close()
                    gc.collect()
                    torch.cuda.empty_cache()
                    st.load()
            print("  loaded   %-8s gpu free %.1f GB (was %.1f) · %s"
                  % (name, torch.cuda.mem_get_info()[0] / 1e9, before / 1e9, sysmem()))

    # Keep decode between frames in auto mode; other stages are streamed. Runtime
    # memory snapshots record actual usage instead of inferring it from plan size.
    keep = {"lazy": set(), "auto": {"decode"},
            "resident": {"vision", "prefill", "decode", "expert"}}[mode]

    def release(name):
        if name in keep:
            return
        if mode in ("lazy", "auto", "resident"):
            with timer.scope("stage_release", name):
                stages[name].close()
                gc.collect()
            print("  released %-8s gpu free %.1f GB · %s"
                  % (name, torch.cuda.mem_get_info()[0] / 1e9, sysmem()))

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    residency_events = []

    # ---- persistent state, allocated once ---------------------------------
    past_k = torch.zeros(LAYERS, 1, KV_HEADS, max_seq, HEAD_DIM, dtype=torch.float16, device="cuda")
    past_v = torch.zeros_like(past_k)
    stages["decode"].bind_kv(past_k, past_v)     # every piece reads the one shared cache
    stages["expert"].bind_kv(past_k, past_v)
    if keep:
        print("resident    : %s (kept between frames)" % ", ".join(sorted(keep)))
    print("kv cache    : %.0f MiB resident   %s" % (2 * past_k.numel() * 2 / 2**20, sysmem()))

    # 155,697 x 4,096 fp16 = 1.28 GB. Prefill reads 3,006 rows once and each decode step
    # reads one, so map it from disk instead of holding it in the shared CPU/GPU memory.
    embed = np.load(os.path.join(shared_fx, "embed_tokens.fp16.npy"), mmap_mode="r")
    input_ids = np.load(os.path.join(fx, "input_ids.npy"))
    position_ids = np.load(os.path.join(fx, "position_ids.npy"))
    visual_mask_np = np.load(os.path.join(fx, "visual_mask.npy"))
    sample_inputs.validate_arrays(meta, input_ids, position_ids, visual_mask_np, embed.shape[0])
    visual_mask = torch.from_numpy(visual_mask_np).cuda()
    cos_p, sin_p = rope.tables(position_ids[:, :prefill_len])
    cos_p = torch.from_numpy(cos_p).cuda()
    sin_p = torch.from_numpy(sin_p).cuda()

    environment = envinfo.collect(work, eng_dir, args.precision,
                                  repo_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selected = {name: dict(backend="torch" if name == "decode" and use_torch_decode else "tensorrt",
                           precision="fp16" if name == "decode" and use_torch_decode else args.precision,
                           enabled=name != "decode" or (not args.no_reasoning and args.max_new_tokens > 0),
                           pieces=[p["name"] for p in st.pieces], sequential=st.sequential)
                for name, st in stages.items()}
    if use_torch_decode:
        selected["decode"]["weights"] = stages["decode"].doc["graphs"][stages["decode"].graph]["source"]
    document = dict(schema_version=2, precision=args.precision, device=torch.cuda.get_device_name(0),
                    started_at_utc=telemetry.utc_now(), environment=environment,
                    configuration=dict(arguments=vars(args), images=files, fixtures=meta,
                                       sample_directory=os.path.abspath(args.sample) if args.sample else None,
                                       fixtures_directory=os.path.abspath(fx), shared_fixtures_directory=os.path.abspath(shared_fx),
                                       effective_residency=mode, selected_stages=selected,
                                       tokenizer=detok.metadata(),
                                       seed_policy="One CUDA generator seeded once; state advances across repeats."),
                    telemetry_notes=["PyTorch allocation counters exclude TensorRT's direct CUDA allocations.",
                                     "System RAM and GPU memory share physical memory on Xavier; do not add them.",
                                     "File size / load time is not measured disk bandwidth.",
                                     "Detailed kernel bottlenecks require a separate Nsight profile.",
                                     "stages and total_ms are synchronized wall scopes, with streamed prefill "
                                     "loading separated. They are not pure GPU compute. Use frame_wall_ms "
                                     "for end-to-end latency, including preprocessing and cleanup."],
                    setup_wall_ms=(time.perf_counter() - setup_start) * 1000, runs=[], status="running")
    results = document["runs"]
    sampler = sysmon.Sampler(args.telemetry_interval or 0.5)
    if args.telemetry_interval:
        sampler.start()
    try:
        if args.json:
            telemetry.write_json(args.json, document)
        for run in range(args.repeat):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timer.reset()
            residency_events.clear()
            resident_before = sorted(n for n, st in stages.items() if st.runners)
            before_counters = telemetry.snapshot(torch)
            memory_checkpoints = [dict(point="frame_start", values=before_counters)]
            frame_start = time.perf_counter()

            # ---- 1. preprocessing (CPU) ---------------------------------------
            with timer.scope("preprocess") as prep:
                pixel_values, grid_thw = preprocess.preprocess_images(
                    [np.asarray(Image.open(f).convert("RGB")) for f in files])
            cpu_ms = prep["wall_ms"]
            vit_tok, llm_tok = preprocess.token_counts(grid_thw)
            if sample_grid is not None and not np.array_equal(grid_thw, sample_grid):
                raise ValueError("preprocessed images do not match the sample/engine image grid")
            if meta.get("visual_tokens") is not None and llm_tok != meta["visual_tokens"]:
                raise ValueError("preprocessed images produce a different visual-token count")

            # ---- 2. vision tower ----------------------------------------------
            preload("vision")
            with timer("vision"):
                vout, _ = stages["vision"].run({"pixel_values": pixel_values})
            visual = vout["visual_embeds"]
            deepstack = [vout["deepstack%d" % i] for i in range(3)]

            # ---- 3. assemble prefill embeddings -------------------------------
            with timer("embed assembly"):
                embeds = torch.from_numpy(
                    np.ascontiguousarray(embed[input_ids[:prefill_len]])).unsqueeze(0).cuda()
                embeds[0, visual_mask[:prefill_len]] = visual[:llm_tok].to(embeds.dtype)
                ds_full = []
                for d in deepstack:
                    z = torch.zeros_like(embeds)
                    z[0, visual_mask[:prefill_len]] = d[:llm_tok].to(z.dtype)
                    ds_full.append(z)
            release("vision")

            # ---- 4. prefill ----------------------------------------------------
            preload("prefill")
            with timer("prefill"):
                pout, kv = stages["prefill"].run({"inputs_embeds": embeds, "cos": cos_p, "sin": sin_p,
                                                  "deepstack0": ds_full[0], "deepstack1": ds_full[1],
                                                  "deepstack2": ds_full[2]})
                for a, b, k, v in kv:                     # each piece's own layers
                    past_k[a:b + 1, :, :, :prefill_len].copy_(k)
                    past_v[a:b + 1, :, :, :prefill_len].copy_(v)
                # The first generated token is predicted at the LAST PROMPT position, so it
                # comes from prefill's logits. Decode only ever receives token embeddings.
                first_logits = pout["logits"].clone()
            release("prefill")
            memory_checkpoints.append(dict(point="after_prefill", values=telemetry.snapshot(torch)))

            # ---- 5. chain-of-causation rollout ---------------------------------
            # Each decode step feeds ONE token, writes that token's K,V into slot `pos`,
            # and returns logits for the next. The stop token <traj_future_start> is fed
            # through once too, so its K,V is in the cache -- the reference does the same
            # (StopAfterEOS) and the expert attends to it. Generated tokens have no entry
            # in the prompt's 3D position table; their position is slot + rope_delta on
            # all three axes, as in Qwen3-VL's own decode path.
            pos = prefill_len
            tokens = []
            mask = torch.full((1, 1, 1, max_seq + 1), NEG_INF, dtype=torch.float16, device="cuda")
            stop_reason = "disabled" if args.no_reasoning else "max_new_tokens"
            if not args.no_reasoning and args.max_new_tokens:
                preload("decode")
                memory_checkpoints.append(dict(point="decode_loaded", values=telemetry.snapshot(torch)))
                logits = first_logits
                for token_index in range(args.max_new_tokens):
                    with timer.scope("token_sampling", "decode", token_index=token_index):
                        tok = sample(logits, args.temperature, args.top_p, gen)
                    tokens.append(tok)
                    with timer.scope("token_preparation", "decode", token_index=token_index):
                        mask[..., :pos] = 0.0
                        mask[..., max_seq] = 0.0                  # the token being fed in
                        c, s = rope.tables(np.full((3, 1), pos + rope_delta, dtype=np.int64))
                        hidden = torch.from_numpy(embed[tok].copy()).view(1, 1, -1).cuda()
                    with timer("decode"):
                        dout, dkv = stages["decode"].run({"hidden": hidden, "cos": c, "sin": s,
                                                          "mask": mask})
                        for a, b, k, v in dkv:
                            past_k[a:b + 1, :, :, pos:pos + 1].copy_(k)
                            past_v[a:b + 1, :, :, pos:pos + 1].copy_(v)
                    pos += 1
                    if tok == TRAJ_FUTURE_START:              # its K,V is now cached
                        stop_reason = "traj_future_start"
                        break
                    logits = dout["logits"]

            # ---- 6. flow-matching action expert --------------------------------
            emask = torch.full((1, 1, N_WAYPOINTS, max_seq + N_WAYPOINTS), NEG_INF,
                               dtype=torch.float16, device="cuda")
            emask[..., :pos] = 0.0
            emask[..., max_seq:] = 0.0                        # expert tokens see each other
            wpos = np.arange(N_WAYPOINTS) + pos + rope_delta   # reference: arange(64) + offset + rope_deltas
            wcos, wsin = rope.tables(np.broadcast_to(wpos, (3, N_WAYPOINTS)))
            wcos, wsin = torch.from_numpy(wcos).cuda(), torch.from_numpy(wsin).cuda()

            memory_checkpoints.append(dict(point="after_decode", values=telemetry.snapshot(torch)))
            release("decode")
            x = torch.randn(1, N_WAYPOINTS, 2, dtype=torch.float16, device="cuda", generator=gen)
            ts = torch.linspace(0.0, 1.0, args.flow_steps + 1)
            preload("expert")
            memory_checkpoints.append(dict(point="expert_loaded", values=telemetry.snapshot(torch)))
            with timer("expert"):
                for i in range(args.flow_steps):
                    with timer.scope("flow_step", "expert", cuda=True, step=i):
                        dt = float(ts[i + 1] - ts[i])
                        t = torch.full((1, 1, 1), float(ts[i]), dtype=torch.float16, device="cuda")
                        out, _ = stages["expert"].run({"noisy_action": x, "timestep": t,
                                                       "cos": wcos, "sin": wsin, "mask": emask})
                        x = x + dt * out["velocity"]
            release("expert")
            with timer.scope("postprocess", sync=True):
                xy, heading = postprocess.action_to_waypoints(x.float().cpu().numpy()[0], v0=v0)

            torch.cuda.synchronize()
            frame_end = time.perf_counter()
            frame_ms = (frame_end - frame_start) * 1000
            after_counters = telemetry.snapshot(torch)
            memory_checkpoints.append(dict(point="frame_end", values=after_counters))
            timing = timer.export(frame_ms)
            sensors = sampler.report(frame_start, frame_end)
            for rec in timing["records"]:
                if rec["kind"] == "stage":
                    lo = timer.origin + rec["start_ms"] / 1000
                    rec["sensors"] = sampler.report(lo, lo + rec["wall_ms"] / 1000)["channels"]
            counters = telemetry.counter_delta(before_counters, after_counters)
            text = detok.decode(tokens) if detok.available else ("" if not tokens else None)

            total = timer.report()
            load_ms = sum(timer.stages.get("engine load", []))
            if load_ms:
                print("stages excluding load     %10.1f ms  (includes host work; load %.0f ms)"
                      % (total - load_ms, load_ms))
            print("\npreprocess (CPU)          %10.1f ms  [not in TOTAL]" % cpu_ms)
            print("reasoning tokens          %10d" % len(tokens))
            print("visual tokens             %10d  (%d ViT patches)" % (llm_tok, vit_tok))
            print("final waypoint            %10s m  (%.1f m of path over %.1f s)"
                  % ("%.1f, %.1f" % (xy[-1, 0], xy[-1, 1]),
                     float(np.linalg.norm(np.diff(np.vstack([[0, 0], xy]), axis=0), axis=-1).sum()),
                     len(xy) * postprocess.DT))
            print("peak CUDA allocation      %10.0f MiB" %
                  (torch.cuda.max_memory_allocated() / 2**20))
            print("frame wall time           %10.1f ms" % frame_ms)
            if text is not None:
                print("reasoning text            %s" % text)
            results.append(dict(run=run, total_ms=total, frame_wall_ms=frame_ms,
                                cpu_preprocess_ms=cpu_ms, reasoning_tokens=len(tokens),
                                reasoning_token_ids=tokens, reasoning_text=text,
                                reasoning_token_pieces=detok.pieces(tokens) if detok.available else None,
                                reasoning_text_status="available" if detok.available else "vocabulary_missing",
                                stop_reason=stop_reason, decode_steps=pos - prefill_len,
                                visual_tokens=llm_tok, vit_patches=vit_tok,
                                stages={k: sum(v) for k, v in timer.stages.items()},
                                stage_samples_ms=timer.stages.copy(),
                                decode_latency=telemetry.distribution(timer.stages.get("decode", [])),
                                telemetry=dict(timing=timing, sensors=sensors, counters=counters,
                                               memory_checkpoints=memory_checkpoints,
                                               residency=dict(before=resident_before,
                                                              after=sorted(n for n, st in stages.items() if st.runners),
                                                              keep=sorted(keep), events=list(residency_events)),
                                               observations=telemetry.observations(timing, counters)),
                                waypoints=xy.tolist()))
            document["summary"] = telemetry.summary(results)
            document["updated_at_utc"] = telemetry.utc_now()
            if args.json:
                telemetry.write_json(args.json, document)

        document["status"] = "complete"
        document["summary"] = telemetry.summary(results)
        print("\nfirst frame %.0f ms; subsequent-frame p50 %s ms" %
              (results[0]["frame_wall_ms"], document["summary"]["subsequent_frames"].get("p50_ms", "n/a")))
        if args.json:
            telemetry.write_json(args.json, document)
            print("wrote %s" % args.json)
    except BaseException as exc:
        document["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        document["error"] = dict(type=type(exc).__name__, message=str(exc),
                                  completed_frames=len(results), timestamp=telemetry.utc_now())
        if args.json:
            telemetry.write_json(args.json, document)
        raise
    finally:
        sampler.stop()
        for name, st in stages.items():
            try:
                st.close()
            except Exception as exc:
                print("cleanup %s: %s" % (name, exc), file=sys.stderr)


if __name__ == "__main__":
    main()
