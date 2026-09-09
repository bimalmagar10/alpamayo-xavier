#!/usr/bin/env python3
"""Stage A1b (H100) -- per-stage timing on the reference machine.

a1_golden.py timed only the vision tower, and the measured trace length turned out
to be ~16 tokens rather than the assumed 256. That moves the bottleneck from
autoregressive decode to **prefill**, which is the one term in the Xavier roofline
that has never been checked against a real measurement.

This times all four stages using only the public API, and prints the achieved
TFLOP/s against the analytic FLOP estimates. Five minutes, and it says whether the
model that drives every Xavier prediction is sound before you spend two hours on
the ONNX export.

    python h100/a1b_h100_stages.py
"""
import argparse
import json
import os
import statistics
import time

import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

MODEL_DIR = os.environ.get(
    "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"

# Analytic FLOP estimates the Xavier roofline is built on. If the achieved
# TFLOP/s below come out wildly different between stages, the estimates are wrong,
# not the hardware.
FLOPS = dict(vision=14.3e12, prefill=47.0e12)


def timeit(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--decode-tokens", type=int, default=16,
                    help="matches the measured p50 trace length")
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "golden", "h100_stages.json"))
    args = ap.parse_args()

    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    tok = helper.to_device(dict(tok), "cuda")

    px, grid = tok["pixel_values"], tok["image_grid_thw"]
    n_text = int(tok["input_ids"].shape[1])
    n_vit = int(px.shape[0])
    n_visual = n_vit // 4                       # after the 2x2 spatial merge
    res = dict(device=torch.cuda.get_device_name(0),
               vit_tokens=n_vit, visual_tokens=n_visual,
               prompt_tokens=n_text, prefill_tokens=n_text,
               grid_thw=grid.tolist(), pixel_values=list(px.shape))

    print("=== token accounting (validates every static ONNX shape) ===")
    print("  pixel_values      :", tuple(px.shape))
    print("  images            :", grid.shape[0], " grid[0] =", grid[0].tolist())
    print("  ViT tokens        : %d   (predicted 11520)" % n_vit)
    print("  visual tokens     : %d   (predicted 2880)" % n_visual)
    print("  prompt tokens     : %d" % n_text)
    print("  prefill tokens    : %d   (traj placeholders already included)\n" % n_text)

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        print("=== per-stage timing (median of 5) ===")
        vis_ms, vis_best = timeit(lambda: model.vlm.model.visual(px, grid))
        res["vision_ms"] = vis_ms

        def full_prefill():
            return model.vlm(**{k: v for k, v in tok.items()}, use_cache=True)
        both_ms, _ = timeit(full_prefill, warmup=1, iters=3)
        pre_ms = max(both_ms - vis_ms, 0.0)
        res["vision_plus_prefill_ms"] = both_ms
        res["prefill_ms"] = pre_ms

        for name, ms, fl in (("vision", vis_ms, FLOPS["vision"]),
                             ("prefill", pre_ms, FLOPS["prefill"])):
            ach = fl / (ms / 1000) / 1e12
            print("  %-22s %8.1f ms   %6.0f TFLOP/s achieved on %.1f TFLOP"
                  % (name, ms, ach, fl / 1e12))
            res["%s_tflops" % name] = ach

        # decode: difference between generating 1 token and N+1 tokens
        gcfg = model.vlm.generation_config
        gcfg.do_sample, gcfg.top_p, gcfg.temperature = True, 0.98, 0.6
        gcfg.pad_token_id = model.tokenizer.pad_token_id

        def gen(n):
            gcfg.max_new_tokens = n
            return model.vlm.generate(input_ids=tok["input_ids"], generation_config=gcfg,
                                      **{k: v for k, v in tok.items() if k != "input_ids"})
        n = args.decode_tokens
        t1, _ = timeit(lambda: gen(1), warmup=1, iters=3)
        tn, _ = timeit(lambda: gen(n + 1), warmup=1, iters=3)
        per_tok = max(tn - t1, 0.0) / n
        res["decode_ms_per_token"] = per_tok
        res["decode_tokens_measured"] = n
        print("  %-22s %8.2f ms/token  (%.0f tok/s)" % ("decode", per_tok, 1000 / max(per_tok, 1e-9)))

    print("\n=== what this means for the Xavier prediction ===")
    print("  Xavier has 11.3 TFLOP/s peak FP16 vs this GPU's ~835 TFLOP/s BF16 dense.")
    print("  If a stage achieves a low fraction of peak here, it will on Xavier too --")
    print("  the roofline's MFU assumption should be lowered to match, not left at 45%.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
